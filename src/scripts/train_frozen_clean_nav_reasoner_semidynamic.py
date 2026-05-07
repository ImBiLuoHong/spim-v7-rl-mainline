from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.utils import softmax as gnn_softmax
from torch_scatter import scatter_max, scatter_sum

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.semi_dynamic_bank import (
    SemiDynamicBankStats,
    SemiDynamicTrajectoryBankDataset,
    SemiDynamicTrajectoryBankWriter,
    build_bank_sample,
    path_size_bytes,
    semi_dynamic_bank_collate_fn,
    write_json,
)
from src.data.v6.loader import create_dataloaders
from src.modeling.builders.model_builder import ModelBuilder
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.scripts.training_campaign_round1 import compare_delta, evaluate_split, load_yaml, prepare_cfg
from src.shared.artifacts import save_checkpoint


DEFAULT_BRIDGE_PACKAGE = PROJECT_ROOT / "artifacts/clean_navigator_v1/navigator_final_delivery_p_seed0_currentrunner_20260402"
DEFAULT_ALIGNED_INIT_CKPT = PROJECT_ROOT / "runs/clean_aligned_reasoner_mainline_f6549ff3d356/model_best.pt"
DEFAULT_BRIDGED_ONLINE_BASELINE = PROJECT_ROOT / "artifacts/reasoner_frozen_clean_nav_bridge/20260402_mainline_bounded2/eval/frozen_clean_nav_reasoner_test_summary.json"
DEFAULT_ALIGNED_ONLINE_BASELINE = PROJECT_ROOT / "artifacts/reasoner_clean_aligned_mainline/20260403_bounded1/eval/clean_aligned_reasoner_test_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/reasoner_clean_aligned_semidynamic/20260403_line_sd"
DEFAULT_FAST_CACHE_ROOT = PROJECT_ROOT / "data" / "cache_lmdb"


def deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def json_ready(value: Any):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def configure_large_bank_dataloader_runtime() -> None:
    try:
        if mp.get_sharing_strategy() != "file_system":
            mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def check_gpu_exclusive() -> Dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "exclusive_ok": None, "processes": []}
    result = os.popen(
        "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null"
    ).read().strip()
    rows = []
    for line in result.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "process_name": parts[1], "used_gpu_memory_mb": parts[2]})
    return {
        "available": True,
        "exclusive_ok": len(rows) == 0,
        "processes": rows,
    }


def preferred_bank_root(output_dir: Path) -> Path:
    bank_root = PROJECT_ROOT / "data" / "cache_lmdb" / "semidynamic_bank"
    bank_root.mkdir(parents=True, exist_ok=True)
    return bank_root


def load_existing_bank_manifest(output_dir: Path) -> Dict[str, Any] | None:
    manifest_path = output_dir / "trajectory_bank" / "trajectory_bank_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lmdb_path = Path(manifest.get("lmdb_path", ""))
    if not lmdb_path.exists():
        return None
    return manifest


def build_semidynamic_overrides(
    *,
    bridge_package_dir: Path,
    cache_version: str,
    max_samples: int | None,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    run_name: str,
    rebuild_cache: bool,
) -> Dict[str, Any]:
    base = load_yaml("configs/evidence_v1/support_mainline.yaml")
    aligned = load_yaml("configs/evidence_v1/clean_aligned_reasoner_mainline.yaml")
    bridge_control_dir = json.loads((bridge_package_dir / "seed_metadata.json").read_text(encoding="utf-8"))["control_dir"]
    runtime_cache_dir = DEFAULT_FAST_CACHE_ROOT
    runtime_cache_dir.mkdir(parents=True, exist_ok=True)
    overlay = {
        "paths": {
            "cache_dir": str(runtime_cache_dir),
        },
        "system": {
            "enable_audit": False,
        },
        "training": {
            "run_name": run_name,
            "enable_eval": False,
            "train_only": True,
            "enable_wandb": False,
            "log_every_n_steps": 50,
            "collect_detailed_step_metrics": False,
        },
        "data": {
            "cache_version": cache_version,
            "rebuild_cache": rebuild_cache,
            "skip_lmdb": False,
            "max_samples": max_samples,
            "num_workers": int(num_workers),
            "prefetch_factor": int(prefetch_factor),
            "pin_memory": True,
            "persistent_workers": bool(num_workers > 0),
        },
        "efficiency": {
            "batch_size": int(batch_size),
            "use_amp": True,
        },
        "model": {
            "navigator_type": "frozen_clean_v1_bridge",
            "training_mode": "frozen_nav",
            "sampling_policy": "navigator_only",
            "navigator": {
                "bridge_control_dir": bridge_control_dir,
                "bridge_deterministic_train": True,
                "bridge_deterministic_eval": True,
            },
        },
        "life_support": {
            "allow_constraint_label_fallback": False,
        },
        "loss": {
            "weights": {
                "w_hard": 1.0,
                "w_soft": 0.0,
                "w_delta": 0.0,
                "w_hit": 0.0,
                "w_mono": 0.0,
                "w_ent": 0.0,
                "w_surv": 0.0,
            }
        },
    }
    return deep_merge(deep_merge(base, aligned), overlay)


def dump_resolved_config(cfg: Any, path: Path) -> None:
    snapshot = cfg.save_snapshot(save_dir=str(path.parent), filename=path.name.replace(".yaml", ".json"))
    snapshot = dict(snapshot)
    snapshot.pop("path", None)
    data = yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def extract_scenario_ids(batch) -> List[int]:
    scenario_id = getattr(batch, "scenario_id")
    if isinstance(scenario_id, torch.Tensor):
        return [int(v) for v in scenario_id.view(-1).tolist()]
    if isinstance(scenario_id, (list, tuple)):
        return [int(v) for v in scenario_id]
    return [int(scenario_id)]


def localize_edge_index(edge_index: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    node_indices = torch.nonzero(node_mask, as_tuple=True)[0]
    mapping = torch.full((node_mask.numel(),), -1, dtype=torch.long, device=node_mask.device)
    mapping[node_indices] = torch.arange(node_indices.numel(), device=node_mask.device)
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    local_edge_index = mapping[edge_index[:, edge_mask]]
    valid_edges = (local_edge_index >= 0).all(dim=0)
    return node_indices, local_edge_index[:, valid_edges]


def selected_globals_for_graph(
    step: Dict[str, Any],
    fused_batch: torch.Tensor,
    graph_idx: int,
) -> List[int]:
    selected_indices = step.get("selected_indices")
    selected_global_ids = step.get("selected_global_ids")
    if selected_indices is None or selected_global_ids is None or selected_indices.numel() == 0:
        return []
    selected_indices = selected_indices.view(-1).long()
    selected_global_ids = selected_global_ids.view(-1).long()
    graph_mask = fused_batch[selected_indices] == int(graph_idx)
    return [int(v) for v in selected_global_ids[graph_mask].tolist()]


def build_rollout_model(cfg: Any, checkpoint_path: Path, device: torch.device):
    model = ModelBuilder.build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    nav = model.navigator_module
    nav.force_deterministic_eval = False
    if hasattr(nav, "clean_navigator"):
        nav.clean_navigator.greedy_eval = False
    return model


def generate_trajectory_bank(
    *,
    output_dir: Path,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    trajectories_per_case: int,
    generation_batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    max_episodes: int,
    bank_root: Path,
    device: torch.device,
) -> Dict[str, Any]:
    cache_version = f"semidynamic_source_full_b{generation_batch_size}"
    overrides = build_semidynamic_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=cache_version,
        max_samples=None,
        batch_size=generation_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        run_name="clean_aligned_semidynamic_source",
        rebuild_cache=False,
    )
    cfg = prepare_cfg(overrides, run_name="clean_aligned_semidynamic_source", max_epochs=1, seed=45)
    train_loader, _, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=generation_batch_size,
        train_only=True,
    )
    lmdb_path = bank_root / "frozen_nav_semidynamic_train_bank.lmdb"
    writer = SemiDynamicTrajectoryBankWriter(lmdb_path)
    bank_start = time.perf_counter()
    model = build_rollout_model(cfg, init_checkpoint, device)

    trajectory_rows: List[Dict[str, Any]] = []
    signatures_by_case: Dict[int, set[str]] = defaultdict(set)
    seen_cases: set[int] = set()
    total_steps = 0
    total_selected = 0

    for traj_id in range(int(trajectories_per_case)):
        torch.manual_seed(45 + traj_id * 100003)
        random.seed(45 + traj_id * 100003)
        for batch in train_loader:
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=True)
            scenario_ids = extract_scenario_ids(batch)
            with torch.no_grad():
                rollout = model(
                    batch,
                    inference_mode=True,
                    max_episodes=int(max_episodes),
                    tau=1.0,
                    skip_reasoner_forward=True,
                )
            trajectory = rollout["trajectory"]
            per_case = {
                int(case_id): {
                    "steps": 0,
                    "selected": [],
                    "budget_final": 0.0,
                    "t_final_minutes": 0.0,
                }
                for case_id in scenario_ids
            }
            for step_id, step in enumerate(trajectory):
                fused_batch = step["fused_batch"].view(-1).long()
                payload = build_clean_aligned_feature_payload(
                    step["reasoner_input_state"],
                    batch_index=fused_batch,
                    edge_index=step["curr_edge_index"].view(2, -1).long(),
                    physics_ctx=step.get("physics_ctx"),
                    frontier_mode="unresolved_without_pair",
                )
                node_features_all = payload["node_features"]
                graph_features_all = payload["graph_features_by_graph"]
                valid_mask_all = payload["valid_mask"]
                source_mask_all = step["fused_source_label"].view(-1).float()
                constraint_state = step["dynamic_state"]["constraint_state"]
                sampled_mask = constraint_state.sampled_mask.view(-1).float()
                t_sim_tensor = step["dynamic_state"]["t_sim"].view(-1).float()

                for graph_idx, case_id in enumerate(scenario_ids):
                    node_mask = fused_batch == int(graph_idx)
                    if not bool(node_mask.any()):
                        continue
                    node_indices, local_edge_index = localize_edge_index(step["curr_edge_index"].view(2, -1).long(), node_mask)
                    budget_used = float(sampled_mask[node_mask].sum().item())
                    t_sim_minutes = float(t_sim_tensor[int(graph_idx)].item())
                    sample = build_bank_sample(
                        node_features=node_features_all[node_mask],
                        edge_index=local_edge_index,
                        graph_features=graph_features_all[int(graph_idx)],
                        valid_mask=valid_mask_all[node_mask],
                        source_mask=source_mask_all[node_mask],
                        case_id=int(case_id),
                        trajectory_id=int(traj_id),
                        step_id=int(step_id),
                        budget_used=budget_used,
                        t_sim_minutes=t_sim_minutes,
                    )
                    writer.add(sample)
                    selected_globals = selected_globals_for_graph(step, fused_batch, int(graph_idx))
                    per_case[int(case_id)]["steps"] += 1
                    per_case[int(case_id)]["selected"].append(selected_globals)
                    per_case[int(case_id)]["budget_final"] = budget_used
                    per_case[int(case_id)]["t_final_minutes"] = t_sim_minutes
                    total_steps += 1
                    total_selected += len(selected_globals)

            for case_id, meta in per_case.items():
                signature = json.dumps(meta["selected"])
                signatures_by_case[int(case_id)].add(signature)
                seen_cases.add(int(case_id))
                trajectory_rows.append(
                    {
                        "case_id": int(case_id),
                        "trajectory_id": int(traj_id),
                        "step_count": int(meta["steps"]),
                        "budget_final": float(meta["budget_final"]),
                        "t_final_minutes": float(meta["t_final_minutes"]),
                        "trajectory_signature": signature,
                    }
                )

    sample_count = writer.close()
    wall_s = time.perf_counter() - bank_start
    stats = SemiDynamicBankStats(
        case_count=len(seen_cases),
        trajectory_count=len(trajectory_rows),
        sample_count=sample_count,
        total_steps=total_steps,
        total_selected=total_selected,
        unique_signature_count=sum(len(v) for v in signatures_by_case.values()),
    )
    trajectory_rows_jsonl = output_dir / "trajectory_bank" / "trajectory_rows.jsonl"
    trajectory_rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_rows_jsonl.open("w", encoding="utf-8") as handle:
        for row in trajectory_rows:
            handle.write(json.dumps(row) + "\n")
    write_csv(
        output_dir / "trajectory_bank" / "trajectory_rows.csv",
        trajectory_rows,
        ["case_id", "trajectory_id", "step_count", "budget_final", "t_final_minutes", "trajectory_signature"],
    )
    lmdb_size_bytes = path_size_bytes(lmdb_path)
    manifest = {
        "authoritative_frozen_navigator_package": str(bridge_package_dir),
        "authoritative_frozen_navigator_checkpoint": str(bridge_package_dir / "navigator_final_selected_best.pt"),
        "source_reasoner_init_checkpoint": str(init_checkpoint),
        "trajectories_per_case": int(trajectories_per_case),
        "generation_batch_size": int(generation_batch_size),
        "generation_loader_num_workers": int(num_workers),
        "generation_loader_prefetch_factor": int(prefetch_factor),
        "max_episodes": int(max_episodes),
        "lmdb_path": str(lmdb_path),
        "lmdb_size_bytes": int(lmdb_size_bytes),
        "generation_wall_seconds": float(wall_s),
        "stats": stats.to_dict(),
        "trajectory_rows_jsonl": str(trajectory_rows_jsonl),
        "trajectory_rows_csv": str(output_dir / "trajectory_bank" / "trajectory_rows.csv"),
        "unique_signature_case_fraction": float(sum(1 for v in signatures_by_case.values() if len(v) > 1) / max(1, len(signatures_by_case))),
        "train_split_raw_size": len(seen_cases),
    }
    write_json(output_dir / "trajectory_bank" / "trajectory_bank_manifest.json", manifest)
    return manifest


def bank_loss(
    logits: torch.Tensor,
    batch_index: torch.Tensor,
    source_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.view(-1)
    valid_mask = valid_mask.view(-1).bool()
    masked_logits = flat_logits.clone()
    masked_logits[~valid_mask] = -float("inf")
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    if num_graphs > 0:
        graph_has_any_finite = scatter_max(
            torch.isfinite(masked_logits).float(),
            batch_index,
            dim=0,
            dim_size=num_graphs,
        )[0] > 0.5
        fully_masked = ~graph_has_any_finite
        if bool(fully_masked.any()):
            masked_logits[fully_masked[batch_index]] = flat_logits[fully_masked[batch_index]]
    probs = gnn_softmax(masked_logits.view(-1, 1), batch_index).view(-1)
    log_probs = torch.log(probs + 1e-9)
    target_mass = source_mask.view(-1).float()
    lp_star = scatter_sum(log_probs * target_mass, batch_index, dim=0)
    return -lp_star.mean()


def build_reasoner_state_from_bank(batch) -> Dict[str, Any]:
    return {
        "clean_aligned_node_features": batch.x,
        "clean_aligned_graph_features": batch.clean_aligned_graph_features,
        "valid_mask": batch.valid_mask,
    }


def build_offline_model(cfg: Any, init_checkpoint: Path, device: torch.device):
    model = ModelBuilder.build_model(cfg).to(device)
    state = torch.load(init_checkpoint, map_location=device)
    load_result = model.load_state_dict(state, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    allowed_missing = {
        "reasoner_module.evidence_core_contrast_adapter.weight",
        "reasoner_module.evidence_core_contrast_adapter.bias",
    }
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"Offline model init checkpoint load mismatch. missing={disallowed_missing}, unexpected={unexpected}"
        )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reasoner_module.parameters():
        param.requires_grad = True
    model.navigator_module.eval()
    if getattr(model, "evidence_refiner", None) is not None:
        model.evidence_refiner.eval()
    return model


def run_batch_gate(
    *,
    bank_lmdb_path: Path,
    cfg: Any,
    init_checkpoint: Path,
    output_dir: Path,
    candidate_batch_sizes: List[int],
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    max_probe_batches: int = 8,
) -> Dict[str, Any]:
    configure_large_bank_dataloader_runtime()
    dataset = SemiDynamicTrajectoryBankDataset(str(bank_lmdb_path))
    subset_cap = min(len(dataset), max(4096, int(max(candidate_batch_sizes)) * 2))
    subset = Subset(dataset, list(range(subset_cap)))
    results = []
    chosen = None
    for batch_size in candidate_batch_sizes:
        result = {
            "batch_size": int(batch_size),
            "ok": False,
            "oom_like": False,
        }
        model = None
        optimizer = None
        loader = None
        torch.cuda.empty_cache() if device.type == "cuda" else None
        model = build_offline_model(cfg, init_checkpoint, device)
        optimizer = torch.optim.AdamW(model.reasoner_module.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(getattr(cfg.efficiency, "use_amp", True))))
        loader = DataLoader(
            subset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=True,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=(int(prefetch_factor) if num_workers > 0 else None),
            collate_fn=semi_dynamic_bank_collate_fn,
        )
        step_times = []
        start = time.perf_counter()
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.reasoner_module.train()
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                step_start = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and scaler.is_enabled())):
                    state = build_reasoner_state_from_bank(batch)
                    out = model.reasoner_module(state, batch, physics_ctx=None)
                    loss = bank_loss(out["logits"], batch.batch.view(-1).long(), batch.source_mask, batch.valid_mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_times.append(time.perf_counter() - step_start)
                if batch_idx + 1 >= int(max_probe_batches):
                    break
            peak_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
            result.update(
                {
                    "ok": True,
                    "avg_step_time_s": sum(step_times[1:]) / max(1, len(step_times[1:])),
                    "peak_allocated_mb": peak_allocated_mb,
                    "amp_enabled": bool(device.type == "cuda" and scaler.is_enabled()),
                    "probe_batches": len(step_times),
                }
            )
            results.append(result)
            chosen = result
            break
        except RuntimeError as exc:
            message = str(exc)
            result["error"] = message
            result["oom_like"] = ("out of memory" in message.lower()) or ("cuda error" in message.lower())
            results.append(result)
        finally:
            result["wall_s"] = time.perf_counter() - start
            del loader
            del optimizer
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError(f"No stable batch size found: {results}")
    summary = {
        "candidate_batch_sizes": candidate_batch_sizes,
        "results": results,
        "chosen_batch_size": chosen["batch_size"],
        "probe_subset_size": int(subset_cap),
        "loader_num_workers": int(num_workers),
        "loader_prefetch_factor": int(prefetch_factor),
    }
    write_json(output_dir / "batch_gate" / "batch_gate_summary.json", summary)
    return summary


def write_epoch_history(output_dir: Path, history_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    history_path = output_dir / "train" / "epoch_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        for row in history_rows:
            handle.write(json.dumps(json_ready(row)) + "\n")
    curve_json = output_dir / "train" / "train_loss_curve.json"
    curve_csv = output_dir / "train" / "train_loss_curve.csv"
    write_json(curve_json, history_rows)
    write_csv(
        curve_csv,
        history_rows,
        [
            "epoch",
            "train_loss",
            "epoch_train_loop_wall_s",
            "samples_per_sec",
            "steps_per_sec",
            "avg_batch_time_s",
            "learning_rate",
        ],
    )
    return {
        "history_path": str(history_path),
        "curve_json": str(curve_json),
        "curve_csv": str(curve_csv),
    }


def load_epoch_history(output_dir: Path) -> List[Dict[str, Any]]:
    history_path = output_dir / "train" / "epoch_history.jsonl"
    if not history_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def train_offline_reasoner(
    *,
    bank_lmdb_path: Path,
    cfg: Any,
    init_checkpoint: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    use_amp: bool,
    periodic_every: int,
    device: torch.device,
) -> Dict[str, Any]:
    configure_large_bank_dataloader_runtime()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    run_name = f"clean_aligned_semidynamic_{hashlib.md5(str(output_dir).encode()).hexdigest()[:12]}"
    run_dir = PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = run_dir / "resolved_config.yaml"
    dump_resolved_config(cfg, resolved_cfg_path)

    dataset = SemiDynamicTrajectoryBankDataset(str(bank_lmdb_path))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=(int(prefetch_factor) if num_workers > 0 else None),
        collate_fn=semi_dynamic_bank_collate_fn,
    )
    model = build_offline_model(cfg, init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.reasoner_module.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    history_rows: List[Dict[str, Any]] = load_epoch_history(output_dir)
    latest_checkpoint_path = run_dir / "checkpoints" / "checkpoint_latest.pt"
    best_checkpoint_path = run_dir / "checkpoints" / "model_best_train.pt"
    start_epoch = 0
    if latest_checkpoint_path.exists():
        latest_ckpt = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(latest_ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest_ckpt["optimizer_state_dict"])
        start_epoch = int(latest_ckpt.get("epoch", -1)) + 1
        if not history_rows and isinstance(latest_ckpt.get("metrics"), dict):
            metrics = latest_ckpt["metrics"]
            history_rows.append(
                {
                    "epoch": int(latest_ckpt.get("epoch", -1)) + 1,
                    "train_loss": float(metrics.get("train_loss")) if metrics.get("train_loss") is not None else None,
                    "epoch_train_loop_wall_s": None,
                    "samples_per_sec": None,
                    "steps_per_sec": None,
                    "avg_batch_time_s": None,
                    "learning_rate": float(metrics.get("learning_rate")) if metrics.get("learning_rate") is not None else None,
                }
            )
    best_train = float("inf")
    if best_checkpoint_path.exists():
        best_ckpt = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        metrics = best_ckpt.get("metrics") or {}
        if metrics.get("train_loss") is not None:
            best_train = float(metrics["train_loss"])
    elif history_rows:
        known_losses = [float(row["train_loss"]) for row in history_rows if row.get("train_loss") is not None]
        if known_losses:
            best_train = min(known_losses)
    total_train_loop_wall = 0.0
    peak_gpu_memory_mb = 0.0

    for epoch in range(int(start_epoch), int(epochs)):
        model.reasoner_module.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        batch_count = 0
        step_times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            step_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and scaler.is_enabled())):
                state = build_reasoner_state_from_bank(batch)
                out = model.reasoner_module(state, batch, physics_ctx=None)
                loss = bank_loss(out["logits"], batch.batch.view(-1).long(), batch.source_mask, batch.valid_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            batch_count += 1
            step_times.append(time.perf_counter() - step_start)
        epoch_wall = time.perf_counter() - epoch_start
        total_train_loop_wall += epoch_wall
        train_loss = epoch_loss / max(1, batch_count)
        current_lr = float(optimizer.param_groups[0]["lr"])
        peak_gpu_memory_mb = max(
            peak_gpu_memory_mb,
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        )
        epoch_row = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "epoch_train_loop_wall_s": float(epoch_wall),
            "samples_per_sec": float(len(dataset) / epoch_wall) if epoch_wall > 0 else None,
            "steps_per_sec": float(batch_count / epoch_wall) if epoch_wall > 0 else None,
            "avg_batch_time_s": float(sum(step_times) / max(1, len(step_times))),
            "learning_rate": current_lr,
        }
        history_rows.append(epoch_row)
        write_epoch_history(output_dir, history_rows)
        is_best_train = train_loss < best_train
        if is_best_train:
            best_train = train_loss
        save_checkpoint(
            str(run_dir),
            model,
            optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss, "learning_rate": current_lr},
            is_best_val=False,
            is_best_train=is_best_train,
            periodic_every=int(periodic_every),
            extra_state={"amp_enabled": bool(device.type == "cuda" and scaler.is_enabled())},
        )
        if is_best_train:
            torch.save(model.state_dict(), run_dir / "model_best.pt")

    torch.save(model.state_dict(), run_dir / "model_final.pt")
    loss_artifacts = write_epoch_history(output_dir, history_rows)
    completed_epochs = len([row for row in history_rows if row.get("epoch") is not None])
    timed_rows = [row for row in history_rows if isinstance(row.get("epoch_train_loop_wall_s"), (int, float))]
    total_recorded_wall = sum(float(row["epoch_train_loop_wall_s"]) for row in timed_rows)
    throughput_summary = {
        "train_dataset_size": int(len(dataset)),
        "train_batches_per_epoch": int(len(loader)),
        "num_epochs": int(completed_epochs),
        "target_num_epochs": int(epochs),
        "total_train_samples_seen": int(len(dataset) * completed_epochs),
        "total_train_steps": int(len(loader) * completed_epochs),
        "train_loop_wall_s": float(total_recorded_wall),
        "samples_per_sec": float((len(dataset) * completed_epochs) / total_recorded_wall) if total_recorded_wall > 0 else None,
        "steps_per_sec": float((len(loader) * completed_epochs) / total_recorded_wall) if total_recorded_wall > 0 else None,
        "avg_batch_time_s": float(total_recorded_wall / max(1, len(loader) * completed_epochs)),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
    }
    write_json(output_dir / "throughput_summary.json", throughput_summary)
    return {
        "run_dir": str(run_dir),
        "resolved_config_path": str(resolved_cfg_path),
        "latest_checkpoint_path": str(run_dir / "checkpoints" / "checkpoint_latest.pt"),
        "best_train_checkpoint_path": str(run_dir / "checkpoints" / "model_best_train.pt"),
        "final_model_state_path": str(run_dir / "model_final.pt"),
        "best_model_state_path": str(run_dir / "model_best.pt"),
        "train_dataset_size": int(len(dataset)),
        "train_batches_per_epoch": int(len(loader)),
        "num_epochs": int(epochs),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "loss_artifacts": loss_artifacts,
        "throughput_summary_path": str(output_dir / "throughput_summary.json"),
    }


def build_test_loader(cfg: Any, cache_version: str) -> DataLoader:
    from src.data.v6.dataset import NpzDatasetV6
    from src.data.v6.lmdb_dataset import LmdbDatasetV6
    from src.tools.convert_to_lmdb import convert_to_lmdb
    from src.data.v6.collate import v6_collate_fn

    raw_test = NpzDatasetV6(
        samples_dir=cfg.paths.samples_path,
        foundation_dir=cfg.paths.foundation_path,
        mode="test",
        window_size=cfg.data.window_size,
        split_dir=cfg.paths.split_dir,
        preload=False,
        keep_raw=True,
        task_mode=getattr(cfg.data, "task_mode", "forensics"),
        online_config=vars(cfg.data.online) if hasattr(cfg.data, "online") else {},
        use_edge_attr=bool(getattr(cfg.data, "use_edge_attr", False)),
        use_virtual_edges=bool(getattr(cfg.data, "use_virtual_edges", False)),
        filter_no_source=bool(getattr(cfg.data, "filter_no_source", False)),
        num_workers=0,
        audit_mode=None,
        log_normalize=True,
        edge_config={"dim": getattr(cfg.data, "edge_dim", 8), "channels": getattr(cfg.data, "edge_channels", {})},
        feature_mode=getattr(cfg.data, "feature_mode", "baseline"),
        max_samples=None,
    )
    cache_dir = PROJECT_ROOT / "data" / "cache_lmdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = cache_dir / f"v6_dataset_test_{cache_version}.lmdb"
    if not lmdb_path.exists():
        ok = convert_to_lmdb(raw_test, mode="test", output_dir=str(cache_dir), cache_version=cache_version)
        if not ok:
            raise RuntimeError("Test-only LMDB conversion failed")
    test_ds = LmdbDatasetV6(str(lmdb_path))
    return DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=v6_collate_fn,
    )


def run_final_eval_and_compare(
    *,
    cfg: Any,
    checkpoint_path: Path,
    output_dir: Path,
    bridged_online_summary: Path,
    aligned_online_summary: Path,
    device: torch.device,
) -> Dict[str, Any]:
    test_loader = build_test_loader(cfg, cache_version="semidynamic_final_test_eval")
    model = ModelBuilder.build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    start = time.perf_counter()
    summary, _ = evaluate_split(model, test_loader, cfg, "test", device, eval_policy="formal")
    eval_seconds = time.perf_counter() - start
    write_json(output_dir / "eval" / "semidynamic_reasoner_test_summary.json", summary)

    bridged = json.loads(bridged_online_summary.read_text(encoding="utf-8"))
    aligned = json.loads(aligned_online_summary.read_text(encoding="utf-8"))
    compare = {
        "candidate_summary": summary,
        "baselines": {
            "bridged_online": {
                "summary_path": str(bridged_online_summary),
                "summary": bridged,
                "delta": compare_delta(summary, bridged),
            },
            "clean_aligned_online": {
                "summary_path": str(aligned_online_summary),
                "summary": aligned,
                "delta": compare_delta(summary, aligned),
            },
        },
    }
    write_json(output_dir / "compare" / "final_compare.json", compare)
    compare_rows = []
    keys = ["top1_hit", "top3_hit", "top5_hit", "mrr_valid", "success_rate", "true_source_rank_mean", "avg_budget_used"]
    for baseline_name, payload in compare["baselines"].items():
        for key in keys:
            compare_rows.append(
                {
                    "baseline": baseline_name,
                    "metric": key,
                    "baseline_value": payload["summary"].get(key),
                    "candidate_value": summary.get(key),
                    "delta": payload["delta"].get(key),
                }
            )
    write_csv(output_dir / "compare" / "final_compare.csv", compare_rows, ["baseline", "metric", "baseline_value", "candidate_value", "delta"])
    return {"summary": summary, "eval_seconds": eval_seconds, "compare_path": str(output_dir / "compare" / "final_compare.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Navigator semi-dynamic bank + clean-aligned reasoner formal training")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bridge-package-dir", type=str, default=str(DEFAULT_BRIDGE_PACKAGE))
    parser.add_argument("--aligned-init-ckpt", type=str, default=str(DEFAULT_ALIGNED_INIT_CKPT))
    parser.add_argument("--bridged-online-baseline", type=str, default=str(DEFAULT_BRIDGED_ONLINE_BASELINE))
    parser.add_argument("--aligned-online-baseline", type=str, default=str(DEFAULT_ALIGNED_ONLINE_BASELINE))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trajectories-per-case", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--generation-workers", type=int, default=8)
    parser.add_argument("--generation-prefetch-factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--offline-workers", type=int, default=16)
    parser.add_argument("--offline-prefetch-factor", type=int, default=4)
    parser.add_argument("--batch-candidates", type=str, default="8192,6144,4096,3072,2048,1024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpu_snapshot = check_gpu_exclusive()
    write_json(output_dir / "gpu_snapshot_before.json", gpu_snapshot)
    if gpu_snapshot.get("available") and gpu_snapshot.get("exclusive_ok") is False:
        raise RuntimeError(f"GPU already occupied: {gpu_snapshot['processes']}")

    bridge_package_dir = Path(args.bridge_package_dir)
    init_checkpoint = Path(args.aligned_init_ckpt)
    bank_root = preferred_bank_root(output_dir)
    stage_timing: Dict[str, Any] = {}

    bank_manifest = load_existing_bank_manifest(output_dir)
    if bank_manifest is None:
        bank_manifest = generate_trajectory_bank(
            output_dir=output_dir,
            bridge_package_dir=bridge_package_dir,
            init_checkpoint=init_checkpoint,
            trajectories_per_case=int(args.trajectories_per_case),
            generation_batch_size=int(args.generation_batch_size),
            num_workers=int(args.generation_workers),
            prefetch_factor=int(args.generation_prefetch_factor),
            max_episodes=10,
            bank_root=bank_root,
            device=device,
        )
        stage_timing["trajectory_bank_generation_seconds"] = bank_manifest["generation_wall_seconds"]
    else:
        stage_timing["trajectory_bank_generation_seconds"] = 0.0

    train_cache_version = f"semidynamic_offline_train_{hashlib.md5(str(output_dir).encode()).hexdigest()[:8]}"
    overrides = build_semidynamic_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=train_cache_version,
        max_samples=None,
        batch_size=8,
        num_workers=int(args.offline_workers),
        prefetch_factor=int(args.offline_prefetch_factor),
        run_name="clean_aligned_semidynamic_offline",
        rebuild_cache=False,
    )
    cfg = prepare_cfg(overrides, run_name="clean_aligned_semidynamic_offline", max_epochs=int(args.epochs), seed=45)
    candidate_batch_sizes = [int(v) for v in args.batch_candidates.split(",") if v.strip()]
    batch_gate = run_batch_gate(
        bank_lmdb_path=Path(bank_manifest["lmdb_path"]),
        cfg=cfg,
        init_checkpoint=init_checkpoint,
        output_dir=output_dir,
        candidate_batch_sizes=candidate_batch_sizes,
        num_workers=int(args.offline_workers),
        prefetch_factor=int(args.offline_prefetch_factor),
        device=device,
    )
    chosen_batch_size = int(batch_gate["chosen_batch_size"])

    train_result = train_offline_reasoner(
        bank_lmdb_path=Path(bank_manifest["lmdb_path"]),
        cfg=cfg,
        init_checkpoint=init_checkpoint,
        output_dir=output_dir,
        epochs=int(args.epochs),
        batch_size=chosen_batch_size,
        num_workers=int(args.offline_workers),
        prefetch_factor=int(args.offline_prefetch_factor),
        use_amp=True,
        periodic_every=int(args.checkpoint_every),
        device=device,
    )
    stage_timing["offline_training_seconds"] = json.loads((output_dir / "throughput_summary.json").read_text())["train_loop_wall_s"]

    eval_result = run_final_eval_and_compare(
        cfg=cfg,
        checkpoint_path=Path(train_result["best_model_state_path"]),
        output_dir=output_dir,
        bridged_online_summary=Path(args.bridged_online_baseline),
        aligned_online_summary=Path(args.aligned_online_baseline),
        device=device,
    )
    stage_timing["final_formal_eval_seconds"] = eval_result["eval_seconds"]
    stage_timing["total_wall_seconds"] = sum(v for v in stage_timing.values() if isinstance(v, (int, float)))
    write_json(output_dir / "stage_timing_summary.json", stage_timing)

    run_manifest = {
        "entrypoint": str(Path(__file__).resolve()),
        "authoritative_frozen_navigator_package": str(bridge_package_dir),
        "authoritative_frozen_navigator_checkpoint": str(bridge_package_dir / "navigator_final_selected_best.pt"),
        "authoritative_clean_aligned_init_checkpoint": str(init_checkpoint),
        "run_dir": train_result["run_dir"],
        "resolved_config_path": train_result["resolved_config_path"],
        "trajectory_bank_manifest_path": str(output_dir / "trajectory_bank" / "trajectory_bank_manifest.json"),
        "batch_gate_summary_path": str(output_dir / "batch_gate" / "batch_gate_summary.json"),
        "throughput_summary_path": str(output_dir / "throughput_summary.json"),
        "stage_timing_summary_path": str(output_dir / "stage_timing_summary.json"),
        "final_eval_path": str(output_dir / "eval" / "semidynamic_reasoner_test_summary.json"),
        "final_compare_path": str(output_dir / "compare" / "final_compare.json"),
        "latest_checkpoint_path": train_result["latest_checkpoint_path"],
        "best_train_checkpoint_path": train_result["best_train_checkpoint_path"],
        "best_model_state_path": train_result["best_model_state_path"],
        "final_model_state_path": train_result["final_model_state_path"],
        "epochs": int(args.epochs),
        "batch_size": chosen_batch_size,
        "offline_workers": int(args.offline_workers),
        "offline_prefetch_factor": int(args.offline_prefetch_factor),
        "amp_enabled": True,
        "loss": "graphwise_ce_only",
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    summary = {
        "run_manifest_path": str(output_dir / "run_manifest.json"),
        "trajectory_bank_manifest_path": str(output_dir / "trajectory_bank" / "trajectory_bank_manifest.json"),
        "batch_gate_summary_path": str(output_dir / "batch_gate" / "batch_gate_summary.json"),
        "throughput_summary_path": str(output_dir / "throughput_summary.json"),
        "stage_timing_summary_path": str(output_dir / "stage_timing_summary.json"),
        "train_curve_json_path": str(output_dir / "train" / "train_loss_curve.json"),
        "train_curve_csv_path": str(output_dir / "train" / "train_loss_curve.csv"),
        "epoch_history_path": str(output_dir / "train" / "epoch_history.jsonl"),
        "final_eval_path": str(output_dir / "eval" / "semidynamic_reasoner_test_summary.json"),
        "final_compare_json_path": str(output_dir / "compare" / "final_compare.json"),
        "final_compare_csv_path": str(output_dir / "compare" / "final_compare.csv"),
        "final_checkpoint_path": train_result["best_model_state_path"],
        "run_dir": train_result["run_dir"],
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
