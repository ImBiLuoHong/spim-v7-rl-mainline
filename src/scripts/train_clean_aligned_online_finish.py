from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.dataset import NpzDatasetV6
from src.data.v6.loader import create_dataloaders
from src.modeling.builders.model_builder import ModelBuilder
from src.modeling.losses import ModularLossEngine
from src.scripts.diagnostics.run_clean_aligned_reasoner_mainline import (
    DEFAULT_BASELINE_CKPT as DEFAULT_BRIDGED_BASELINE_CKPT,
    DEFAULT_BASELINE_SUMMARY as DEFAULT_BRIDGED_BASELINE_SUMMARY,
    DEFAULT_BRIDGE_PACKAGE,
    build_aligned_overrides,
    deep_merge,
    prepare_cfg,
)
from src.scripts.train_phase4_end2end import run_training
from src.scripts.training_campaign_round1 import build_eval_loaders_with_batch, evaluate_split
from src.utils.hardware_optim import DevicePrefetcher, apply_hardware_optimizations, get_device_and_scaler


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reasoner_clean_aligned_online_finish" / "20260404_line_final"
DEFAULT_BOUNDED_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_clean_aligned_mainline" / "20260403_bounded1"
DEFAULT_BOUNDED_TEST_SUMMARY = DEFAULT_BOUNDED_ROOT / "eval" / "clean_aligned_reasoner_test_summary.json"
DEFAULT_BOUNDED_BEST_CKPT = PROJECT_ROOT / "runs" / "clean_aligned_reasoner_mainline_f6549ff3d356" / "model_best.pt"
DEFAULT_BOUNDED_LATEST_CKPT = PROJECT_ROOT / "runs" / "clean_aligned_reasoner_mainline_f6549ff3d356" / "checkpoints" / "checkpoint_latest.pt"
DEFAULT_SHM_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
BEST_CRITERION_ORDER = ["mrr_valid", "top1_hit", "top3_hit", "top5_hit"]


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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
    return {"available": True, "exclusive_ok": len(rows) == 0, "processes": rows}


def make_cache_version(*, stage: str, output_dir: Path) -> str:
    digest = hashlib.md5(f"{output_dir.resolve()}::{stage}".encode("utf-8")).hexdigest()[:8]
    return f"clean_aligned_online_finish_{stage}_{digest}"


def build_finish_overrides(
    *,
    bridge_package_dir: Path,
    cache_version: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    epochs: int,
    periodic_checkpoint_every: int,
    run_name: str,
    init_checkpoint: Path,
    train_only: bool,
    enable_eval: bool,
    cache_dir: Path,
    sampling_budget_anneal: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides = build_aligned_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=cache_version,
        max_samples=None,
        batch_size=batch_size,
        epochs=epochs,
        run_name=run_name,
        rebuild_cache=False,
    )
    overlay = {
        "training": {
            "run_name": run_name,
            "num_epochs": int(epochs),
            "init_checkpoint": str(init_checkpoint),
            "init_checkpoint_strict": True,
            "resume_checkpoint": None,
            "resume": False,
            "enable_eval": bool(enable_eval),
            "train_only": bool(train_only),
            "formal_eval_batch_size": 1,
            "val_every_n_epochs": max(1, int(periodic_checkpoint_every)),
            "periodic_checkpoint_every_n_epochs": int(periodic_checkpoint_every),
            "ofb_save_every": int(periodic_checkpoint_every),
            "collect_detailed_step_metrics": False,
            "log_every_n_steps": 20,
            "sampling_budget_anneal": sampling_budget_anneal or {"enabled": False},
        },
        "data": {
            "cache_version": cache_version,
            "rebuild_cache": False,
            "skip_lmdb": False,
            "max_samples": None,
            "num_workers": int(num_workers),
            "prefetch_factor": int(prefetch_factor),
            "pin_memory": True,
            "persistent_workers": bool(num_workers > 0),
        },
        "paths": {
            "cache_dir": str(cache_dir),
        },
        "efficiency": {
            "batch_size": int(batch_size),
            "use_amp": True,
            "num_workers": int(num_workers),
            "prefetch_factor": int(prefetch_factor),
            "pin_memory": True,
            "persistent_workers": bool(num_workers > 0),
            "performance": {
                "tf32": True,
            },
        },
    }
    return deep_merge(overrides, overlay)


def build_raw_train_size(cfg_overrides: Dict[str, Any]) -> int:
    cfg = Config()
    cfg.apply_overrides(cfg_overrides)
    dataset = NpzDatasetV6(
        samples_dir=cfg.paths.samples_path,
        foundation_dir=cfg.paths.foundation_path,
        mode="train",
        window_size=cfg.data.window_size,
        split_dir=cfg.paths.split_dir,
        preload=False,
        keep_raw=True,
        task_mode=cfg.data.task_mode,
        online_config=vars(cfg.data.online),
        use_edge_attr=bool(cfg.data.use_edge_attr),
        use_virtual_edges=bool(cfg.data.use_virtual_edges),
        filter_no_source=bool(cfg.data.filter_no_source),
        num_workers=0,
        audit_mode=None,
        log_normalize=bool(cfg.data.normalize),
        edge_config={"dim": getattr(cfg.data, "edge_dim", 8), "channels": getattr(cfg.data, "edge_channels", {})},
        feature_mode=cfg.data.feature_mode,
        max_samples=getattr(cfg.data, "max_samples", None),
    )
    return len(dataset)


def estimate_full_graph_budget(cfg_overrides: Dict[str, Any]) -> int:
    cfg = Config()
    cfg.apply_overrides(cfg_overrides)
    dataset = NpzDatasetV6(
        samples_dir=cfg.paths.samples_path,
        foundation_dir=cfg.paths.foundation_path,
        mode="train",
        window_size=cfg.data.window_size,
        split_dir=cfg.paths.split_dir,
        preload=False,
        keep_raw=False,
        task_mode=cfg.data.task_mode,
        online_config=vars(cfg.data.online),
        use_edge_attr=bool(cfg.data.use_edge_attr),
        use_virtual_edges=bool(cfg.data.use_virtual_edges),
        filter_no_source=bool(cfg.data.filter_no_source),
        num_workers=0,
        audit_mode=None,
        log_normalize=bool(cfg.data.normalize),
        edge_config={"dim": getattr(cfg.data, "edge_dim", 8), "channels": getattr(cfg.data, "edge_channels", {})},
        feature_mode=cfg.data.feature_mode,
        max_samples=1,
    )
    sample = dataset[0]
    sample_num_nodes = getattr(sample, "num_nodes", None)
    if sample_num_nodes is None and isinstance(sample, dict):
        sample_num_nodes = sample.get("num_nodes")
    if sample_num_nodes is None and hasattr(sample, "x") and sample.x is not None:
        sample_num_nodes = int(sample.x.size(0))
    if sample_num_nodes is None:
        raise RuntimeError("Unable to infer per-sample full-graph budget from train dataset sample.")
    return int(sample_num_nodes)


def build_probe_components(overrides: Dict[str, Any], device: torch.device):
    cfg = prepare_cfg(overrides, run_name=overrides["training"]["run_name"], max_epochs=overrides["training"]["num_epochs"], seed=45)
    train_loader, _, _, imbalance_info = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=cfg.efficiency.batch_size,
        eval_batch_size=1,
        skip_lmdb=bool(cfg.data.skip_lmdb),
        train_only=True,
    )
    model = ModelBuilder.build_model(cfg).to(device)
    model = apply_hardware_optimizations(model, cfg)
    state_dict = torch.load(str(overrides["training"]["init_checkpoint"]), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    loss_cfg = _to_dict_recursive(cfg.loss)
    if imbalance_info and "class_weights_vec" in imbalance_info:
        loss_cfg.setdefault("params", {})["class_weight_pos"] = imbalance_info["class_weights_vec"][1]
    loss_engine = ModularLossEngine(loss_cfg).to(device)
    scaler = get_device_and_scaler(cfg)[1]
    return cfg, train_loader, model, optimizer, loss_engine, scaler


def _to_dict_recursive(obj):
    if isinstance(obj, dict):
        return {k: _to_dict_recursive(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _to_dict_recursive(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warm_train_cache(
    *,
    output_dir: Path,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_version: str,
    cache_dir: Path,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> Dict[str, Any]:
    start = time.perf_counter()
    overrides = build_finish_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=cache_version,
        batch_size=256,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        epochs=50,
        periodic_checkpoint_every=5,
        run_name="clean_aligned_online_finish_cache_prewarm",
        init_checkpoint=init_checkpoint,
        train_only=True,
        enable_eval=False,
        cache_dir=cache_dir,
    )
    cfg = prepare_cfg(overrides, run_name=overrides["training"]["run_name"], max_epochs=1, seed=45)
    loader, _, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        eval_batch_size=1,
        skip_lmdb=bool(cfg.data.skip_lmdb),
        train_only=True,
    )
    it = iter(loader)
    _ = next(it)
    summary = {
        "cache_version": cache_version,
        "cache_dir": str(cache_dir),
        "wall_seconds": time.perf_counter() - start,
        "train_dataset_size": len(loader.dataset),
        "train_batches": len(loader),
    }
    write_json(output_dir / "cache_prewarm" / "cache_prewarm_summary.json", summary)
    return summary


def probe_batch_sizes(
    *,
    output_dir: Path,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_version: str,
    cache_dir: Path,
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
    candidate_batch_sizes: Sequence[int],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for batch_size in candidate_batch_sizes:
        overrides = build_finish_overrides(
            bridge_package_dir=bridge_package_dir,
            cache_version=cache_version,
            batch_size=int(batch_size),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            epochs=50,
            periodic_checkpoint_every=5,
            run_name=f"clean_aligned_online_finish_probe_bs{batch_size}",
            init_checkpoint=init_checkpoint,
            train_only=True,
            enable_eval=False,
            cache_dir=cache_dir,
        )
        result = {"batch_size": int(batch_size), "ok": False}
        start = time.perf_counter()
        try:
            cfg, train_loader, model, optimizer, loss_engine, scaler = build_probe_components(overrides, device)
            prefetcher = DevicePrefetcher(train_loader, device) if device.type == "cuda" else train_loader
            iterator = iter(prefetcher)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            step_times: List[float] = []
            for _ in range(2):
                step_start = time.perf_counter()
                batch = next(iterator)
                use_amp = scaler is not None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(batch, inference_mode=False, max_episodes=cfg.training.max_train_episodes, tau=1.0)
                    graph_structure = {"dist_to_source": out["step_metrics"].get("fused_dist")}
                    loss, _ = loss_engine(out["trajectory"], cfg=cfg, graph_structure=graph_structure)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                sync_cuda(device)
                step_times.append(time.perf_counter() - step_start)
            avg_step = sum(step_times[1:]) / max(1, len(step_times[1:]))
            result.update(
                {
                    "ok": True,
                    "train_dataset_size": len(train_loader.dataset),
                    "train_batches_per_epoch": len(train_loader),
                    "avg_step_time_s": avg_step,
                    "throughput_samples_per_s": (batch_size / avg_step) if avg_step > 0 else None,
                    "peak_gpu_memory_mb": (
                        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
                    ),
                    "amp_enabled": scaler is not None,
                }
            )
        except RuntimeError as exc:
            message = str(exc)
            result["error"] = message
            result["oom_like"] = ("out of memory" in message.lower()) or ("cuda error" in message.lower())
            if device.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            result["wall_s"] = time.perf_counter() - start
            results.append(result)
    stable = [row for row in results if row.get("ok")]
    if not stable:
        raise RuntimeError(f"No stable batch size found: {results}")
    chosen = max(
        stable,
        key=lambda row: (
            safe_float(row.get("throughput_samples_per_s")),
            row.get("batch_size", 0),
        ),
    )
    summary = {
        "candidate_batch_sizes": list(candidate_batch_sizes),
        "loader_num_workers": int(num_workers),
        "loader_prefetch_factor": int(prefetch_factor),
        "cache_version": cache_version,
        "results": results,
        "chosen_batch_size": int(chosen["batch_size"]),
        "chosen_throughput_samples_per_s": chosen.get("throughput_samples_per_s"),
    }
    write_json(output_dir / "batch_gate" / "batch_gate_summary.json", summary)
    return summary


def run_smoke_batch_gate(
    *,
    output_dir: Path,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_version: str,
    cache_dir: Path,
    num_workers: int,
    prefetch_factor: int,
    candidate_batch_sizes: Sequence[int],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for batch_size in candidate_batch_sizes:
        overrides = build_finish_overrides(
            bridge_package_dir=bridge_package_dir,
            cache_version=cache_version,
            batch_size=int(batch_size),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            epochs=1,
            periodic_checkpoint_every=5,
            run_name=f"clean_aligned_online_finish_gate_bs{batch_size}",
            init_checkpoint=init_checkpoint,
            train_only=True,
            enable_eval=False,
            cache_dir=cache_dir,
        )
        start = time.perf_counter()
        row: Dict[str, Any] = {"batch_size": int(batch_size), "ok": False}
        try:
            metrics = run_training(
                loss_config_override=overrides,
                run_name=f"clean_aligned_online_finish_gate_bs{batch_size}",
                max_epochs=1,
                seed=45,
                skip_audit=True,
                force_rebuild=False,
            )
            train_loop_wall = safe_float(metrics.get("run_timing", {}).get("train_loop_wall_s"))
            dataset_size = int(metrics.get("train_dataset_size") or 0)
            steps = int(metrics.get("train_batches_per_epoch") or 0)
            row.update(
                {
                    "ok": True,
                    "run_dir": metrics.get("run_dir"),
                    "train_dataset_size": dataset_size,
                    "train_batches_per_epoch": steps,
                    "train_loop_wall_s": train_loop_wall,
                    "samples_per_sec": (dataset_size / train_loop_wall) if train_loop_wall > 0 else None,
                    "steps_per_sec": (steps / train_loop_wall) if train_loop_wall > 0 else None,
                    "avg_batch_time_s": (train_loop_wall / steps) if steps > 0 else None,
                    "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
                }
            )
        except RuntimeError as exc:
            row["error"] = str(exc)
            row["oom_like"] = "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            row["wall_s"] = time.perf_counter() - start
            results.append(row)
    stable = [row for row in results if row.get("ok")]
    if not stable:
        raise RuntimeError(f"No stable batch gate candidate found: {results}")
    chosen = max(
        stable,
        key=lambda row: (
            safe_float(row.get("samples_per_sec")),
            safe_float(row.get("steps_per_sec")),
            row.get("batch_size", 0),
        ),
    )
    summary = {
        "gate_mode": "one_epoch_campaign_like_smoke",
        "candidate_batch_sizes": list(candidate_batch_sizes),
        "loader_num_workers": int(num_workers),
        "loader_prefetch_factor": int(prefetch_factor),
        "cache_version": cache_version,
        "results": results,
        "chosen_batch_size": int(chosen["batch_size"]),
        "chosen_samples_per_sec": chosen.get("samples_per_sec"),
        "chosen_steps_per_sec": chosen.get("steps_per_sec"),
    }
    write_json(output_dir / "batch_gate" / "batch_gate_summary.json", summary)
    return summary


def build_eval_overrides(
    *,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_version: str,
    cache_dir: Path,
) -> Dict[str, Any]:
    return build_finish_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=cache_version,
        batch_size=256,
        num_workers=16,
        prefetch_factor=4,
        epochs=50,
        periodic_checkpoint_every=5,
        run_name="clean_aligned_online_finish_eval",
        init_checkpoint=init_checkpoint,
        train_only=False,
        enable_eval=False,
        cache_dir=cache_dir,
    )


def load_plain_state_dict(checkpoint_path: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    overrides: Dict[str, Any],
    split_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = prepare_cfg(overrides, run_name=overrides["training"]["run_name"], max_epochs=1, seed=45)
    _, eval_loaders = build_eval_loaders_with_batch(eval_batch_size=1)
    loader = eval_loaders[split_name]
    model = ModelBuilder.build_model(cfg).to(device)
    state_dict = load_plain_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    summary, df = evaluate_split(model, loader, cfg, split_name, device, eval_policy="formal")
    return {"summary": summary, "per_case_rows": len(df)}


def checkpoint_epoch(checkpoint_path: Path, default_epoch: int) -> int:
    if checkpoint_path.name.startswith("checkpoint_epoch_"):
        try:
            return int(checkpoint_path.stem.split("_")[-1])
        except Exception:
            return default_epoch
    if checkpoint_path.name == "model_final.pt":
        return default_epoch
    if checkpoint_path.name == "model_best.pt":
        return default_epoch
    return default_epoch


def selection_tuple(summary: Dict[str, Any]) -> tuple:
    return (
        safe_float(summary.get("mrr_valid")),
        safe_float(summary.get("top1_hit")),
        safe_float(summary.get("top3_hit")),
        safe_float(summary.get("top5_hit")),
    )


def better_than(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> bool:
    return selection_tuple(lhs) > selection_tuple(rhs)


def compare_delta(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    delta = {}
    for key in ["top1_hit", "top3_hit", "top5_hit", "mrr_valid", "success_rate", "true_source_rank_mean", "avg_budget_used"]:
        if key in candidate and key in baseline:
            delta[key] = safe_float(candidate[key]) - safe_float(baseline[key])
    return delta


def collect_candidate_checkpoints(run_dir: Path, total_epochs: int, periodic_every: int) -> List[Path]:
    paths: List[Path] = []
    checkpoints_dir = run_dir / "checkpoints"
    for epoch in range(periodic_every, total_epochs + 1, periodic_every):
        candidate = checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        if candidate.exists():
            paths.append(candidate)
    final_model = run_dir / "model_final.pt"
    if final_model.exists():
        paths.append(final_model)
    dedup = []
    seen = set()
    for path in paths:
        if path not in seen:
            dedup.append(path)
            seen.add(path)
    return dedup


def write_train_history(run_dir: Path, output_dir: Path) -> Dict[str, str]:
    history_path = run_dir / "epoch_history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    out_json = output_dir / "train" / "train_history.json"
    out_csv = output_dir / "train" / "train_history.csv"
    write_json(out_json, rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "tau",
                "sample_budget",
                "sampling_policy",
                "epoch_train_loop_wall_s",
                "epoch_run_training_wall_s",
                "epoch_has_validation",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "train_loss": row.get("train_loss"),
                    "tau": row.get("tau"),
                    "sample_budget": row.get("sample_budget"),
                    "sampling_policy": row.get("sampling_policy"),
                    "epoch_train_loop_wall_s": row.get("epoch_train_loop_wall_s"),
                    "epoch_run_training_wall_s": row.get("epoch_run_training_wall_s"),
                    "epoch_has_validation": row.get("epoch_has_validation"),
                }
            )
    return {"json": str(out_json), "csv": str(out_csv), "source_history": str(history_path)}


def build_compare_rows(
    *,
    best_summary: Dict[str, Any],
    final_summary: Dict[str, Any],
    bridged_summary: Dict[str, Any],
    bounded_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    baselines = {
        "bridged_online": bridged_summary,
        "clean_aligned_bounded_online": bounded_summary,
    }
    candidates = {
        "best_checkpoint": best_summary,
        "final_checkpoint": final_summary,
    }
    metrics = ["top1_hit", "top3_hit", "top5_hit", "mrr_valid", "success_rate", "true_source_rank_mean", "avg_budget_used"]
    for baseline_name, baseline in baselines.items():
        for candidate_name, candidate in candidates.items():
            for metric in metrics:
                rows.append(
                    {
                        "baseline": baseline_name,
                        "candidate": candidate_name,
                        "metric": metric,
                        "baseline_value": baseline.get(metric),
                        "candidate_value": candidate.get(metric),
                        "delta": safe_float(candidate.get(metric)) - safe_float(baseline.get(metric)),
                    }
                )
    return rows


def write_compare_pack(
    *,
    output_dir: Path,
    best_summary: Dict[str, Any],
    final_summary: Dict[str, Any],
    bridged_summary: Dict[str, Any],
    bounded_summary: Dict[str, Any],
) -> Dict[str, str]:
    compare_json = {
        "best_checkpoint": best_summary,
        "final_checkpoint": final_summary,
        "baselines": {
            "bridged_online": bridged_summary,
            "clean_aligned_bounded_online": bounded_summary,
        },
        "delta": {
            "best_vs_bridged_online": compare_delta(best_summary, bridged_summary),
            "best_vs_clean_aligned_bounded_online": compare_delta(best_summary, bounded_summary),
            "final_vs_bridged_online": compare_delta(final_summary, bridged_summary),
            "final_vs_clean_aligned_bounded_online": compare_delta(final_summary, bounded_summary),
        },
    }
    rows = build_compare_rows(
        best_summary=best_summary,
        final_summary=final_summary,
        bridged_summary=bridged_summary,
        bounded_summary=bounded_summary,
    )
    compare_dir = output_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    write_json(compare_dir / "final_compare.json", compare_json)
    with (compare_dir / "final_compare.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "json": str(compare_dir / "final_compare.json"),
        "csv": str(compare_dir / "final_compare.csv"),
    }


def copy_delivery_artifacts(
    *,
    output_dir: Path,
    best_checkpoint_path: Path,
    final_checkpoint_path: Path,
    resolved_config_path: Path,
    run_dir: Path,
) -> Dict[str, str]:
    delivery_dir = output_dir / "delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    best_dst = delivery_dir / "best_checkpoint.pt"
    final_dst = delivery_dir / "final_checkpoint.pt"
    resolved_dst = delivery_dir / "resolved_config.yaml"
    copy_file(best_checkpoint_path, best_dst)
    copy_file(final_checkpoint_path, final_dst)
    copy_file(resolved_config_path, resolved_dst)
    snapshot = {
        "run_dir": str(run_dir),
        "entrypoint": str(Path(__file__).resolve()),
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "source_files": [
            str(Path(__file__).resolve()),
            str(PROJECT_ROOT / "src" / "scripts" / "diagnostics" / "run_clean_aligned_reasoner_mainline.py"),
            str(PROJECT_ROOT / "src" / "scripts" / "train_phase4_end2end.py"),
            str(PROJECT_ROOT / "src" / "modeling" / "reasoners" / "clean_aligned.py"),
            str(PROJECT_ROOT / "src" / "modeling" / "clean_aligned_features.py"),
        ],
    }
    write_json(delivery_dir / "runner_snapshot.json", snapshot)
    return {
        "best_checkpoint": str(best_dst),
        "final_checkpoint": str(final_dst),
        "resolved_config": str(resolved_dst),
        "runner_snapshot": str(delivery_dir / "runner_snapshot.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal finish-training for frozen P-best Navigator + clean-aligned online reasoner.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bridge-package-dir", type=str, default=str(DEFAULT_BRIDGE_PACKAGE))
    parser.add_argument("--bridged-baseline-ckpt", type=str, default=str(DEFAULT_BRIDGED_BASELINE_CKPT))
    parser.add_argument("--bridged-baseline-summary", type=str, default=str(DEFAULT_BRIDGED_BASELINE_SUMMARY))
    parser.add_argument("--bounded-root", type=str, default=str(DEFAULT_BOUNDED_ROOT))
    parser.add_argument("--bounded-best-ckpt", type=str, default=str(DEFAULT_BOUNDED_BEST_CKPT))
    parser.add_argument("--bounded-latest-ckpt", type=str, default=str(DEFAULT_BOUNDED_LATEST_CKPT))
    parser.add_argument("--bounded-summary", type=str, default=str(DEFAULT_BOUNDED_TEST_SUMMARY))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--periodic-checkpoint-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=[640, 512, 384, 320])
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_SHM_CACHE_DIR))
    parser.add_argument("--sampling-budget-anneal", action="store_true")
    parser.add_argument("--anneal-full-budget", type=int, default=None)
    parser.add_argument("--anneal-floor-budget", type=int, default=3)
    parser.add_argument("--anneal-warm-epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_package_dir = Path(args.bridge_package_dir)
    bounded_best_ckpt = Path(args.bounded_best_ckpt)
    bounded_latest_ckpt = Path(args.bounded_latest_ckpt)
    bridged_baseline_summary_path = Path(args.bridged_baseline_summary)
    bounded_summary_path = Path(args.bounded_summary)
    cache_dir = Path(args.cache_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gpu_snapshot = check_gpu_exclusive()
    write_json(output_dir / "gpu_snapshot_before.json", gpu_snapshot)
    if gpu_snapshot.get("available") and gpu_snapshot.get("exclusive_ok") is False:
        raise RuntimeError(f"GPU is already occupied by other compute processes: {gpu_snapshot['processes']}")

    stage_timing: Dict[str, float] = {}
    overall_start = time.perf_counter()
    train_cache_version = make_cache_version(stage="train_full", output_dir=output_dir)

    init_source_manifest = {
        "authoritative_training_entry": str(Path(__file__).resolve()),
        "authoritative_bridge_package": str(bridge_package_dir),
        "authoritative_frozen_navigator_checkpoint": str(bridge_package_dir / "navigator_final_selected_best.pt"),
        "authoritative_clean_aligned_bounded_root": str(Path(args.bounded_root)),
        "authoritative_clean_aligned_bounded_best_checkpoint": str(bounded_best_ckpt),
        "resume_candidate_checkpoint": str(bounded_latest_ckpt),
        "chosen_start_mode": "continue_from_clean_aligned_bounded_best_weights",
        "chosen_init_checkpoint": str(bounded_best_ckpt),
        "resume_checkpoint_used": None,
        "reason": (
            "bounded clean-aligned mainline already proved contract compatibility and was reused as the semidynamic source init; "
            "continuing from its best weights is faster and cleaner than restarting from bridged online, while avoiding stale subset optimizer state."
        ),
        "best_checkpoint_selection_order": BEST_CRITERION_ORDER,
        "seed": 45,
    }
    write_json(output_dir / "init_source_manifest.json", init_source_manifest)

    base_probe_overrides = build_finish_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=train_cache_version,
        batch_size=256,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        epochs=int(args.epochs),
        periodic_checkpoint_every=int(args.periodic_checkpoint_every),
        run_name="clean_aligned_online_finish_probe_cfg",
        init_checkpoint=bounded_best_ckpt,
        train_only=True,
        enable_eval=False,
        cache_dir=cache_dir,
    )
    raw_train_size = build_raw_train_size(base_probe_overrides)
    anneal_full_budget = int(args.anneal_full_budget) if args.anneal_full_budget is not None else estimate_full_graph_budget(base_probe_overrides)
    sampling_budget_anneal = {
        "enabled": bool(args.sampling_budget_anneal),
        "full_budget": int(anneal_full_budget),
        "floor_budget": int(args.anneal_floor_budget),
        "warm_epoch_count": int(args.anneal_warm_epochs),
        "pre_floor_policy": "random_valid",
        "floor_policy": "navigator_only",
    }
    write_json(output_dir / "sampling_budget_anneal.json", sampling_budget_anneal)

    stage_start = time.perf_counter()
    cache_prewarm_summary = warm_train_cache(
        output_dir=output_dir,
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=bounded_best_ckpt,
        cache_version=train_cache_version,
        cache_dir=cache_dir,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        device=device,
    )
    stage_timing["cache_prewarm_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    batch_gate_summary = run_smoke_batch_gate(
        output_dir=output_dir,
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=bounded_best_ckpt,
        cache_version=train_cache_version,
        cache_dir=cache_dir,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        candidate_batch_sizes=[int(x) for x in args.batch_candidates],
    )
    stage_timing["batch_gate_seconds"] = time.perf_counter() - stage_start
    chosen_batch_size = int(batch_gate_summary["chosen_batch_size"])

    train_overrides = build_finish_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=train_cache_version,
        batch_size=chosen_batch_size,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        epochs=int(args.epochs),
        periodic_checkpoint_every=int(args.periodic_checkpoint_every),
        run_name="clean_aligned_online_finish",
        init_checkpoint=bounded_best_ckpt,
        train_only=True,
        enable_eval=False,
        cache_dir=cache_dir,
        sampling_budget_anneal=sampling_budget_anneal,
    )

    stage_start = time.perf_counter()
    train_metrics = run_training(
        loss_config_override=train_overrides,
        run_name="clean_aligned_online_finish",
        max_epochs=int(args.epochs),
        seed=45,
        skip_audit=True,
        force_rebuild=False,
    )
    stage_timing["formal_training_seconds"] = time.perf_counter() - stage_start
    run_dir = Path(train_metrics["run_dir"])
    write_json(output_dir / "train" / "run_metrics.json", train_metrics)

    history_paths = write_train_history(run_dir, output_dir)
    throughput_summary = {
        "train_dataset_size": int(train_metrics.get("train_dataset_size") or 0),
        "train_batches_per_epoch": int(train_metrics.get("train_batches_per_epoch") or 0),
        "num_epochs": int(args.epochs),
        "batch_size": chosen_batch_size,
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "cache_version": train_cache_version,
        "cache_dir": str(cache_dir),
        "sampling_budget_anneal": sampling_budget_anneal,
        "train_loop_wall_s": safe_float(train_metrics.get("run_timing", {}).get("train_loop_wall_s")),
        "run_training_wall_s": safe_float(train_metrics.get("run_timing", {}).get("run_training_wall_s")),
        "samples_per_sec": (
            (int(train_metrics.get("train_dataset_size") or 0) * int(args.epochs))
            / max(safe_float(train_metrics.get("run_timing", {}).get("train_loop_wall_s")), 1e-9)
        ),
        "steps_per_sec": (
            (int(train_metrics.get("train_batches_per_epoch") or 0) * int(args.epochs))
            / max(safe_float(train_metrics.get("run_timing", {}).get("train_loop_wall_s")), 1e-9)
        ),
        "peak_gpu_memory_mb": train_metrics.get("peak_gpu_memory_mb"),
        "batch_gate_summary_path": str(output_dir / "batch_gate" / "batch_gate_summary.json"),
    }
    write_json(output_dir / "throughput_summary.json", throughput_summary)

    eval_overrides = build_eval_overrides(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=bounded_best_ckpt,
        cache_version=train_cache_version,
        cache_dir=cache_dir,
    )

    checkpoint_paths = collect_candidate_checkpoints(
        run_dir=run_dir,
        total_epochs=int(args.epochs),
        periodic_every=int(args.periodic_checkpoint_every),
    )
    val_rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    stage_start = time.perf_counter()
    for checkpoint_path in checkpoint_paths:
        eval_result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            overrides=eval_overrides,
            split_name="val",
            device=device,
        )
        row = {
            "checkpoint_path": str(checkpoint_path),
            "epoch": checkpoint_epoch(checkpoint_path, int(args.epochs)),
            "metrics": eval_result["summary"],
            "selection_tuple": selection_tuple(eval_result["summary"]),
        }
        val_rows.append(row)
        if best_row is None or better_than(eval_result["summary"], best_row["metrics"]):
            best_row = row
    stage_timing["checkpoint_val_sweep_seconds"] = time.perf_counter() - stage_start
    if best_row is None:
        raise RuntimeError("No candidate checkpoint was available for validation sweep.")

    best_test = evaluate_checkpoint(
        checkpoint_path=Path(best_row["checkpoint_path"]),
        overrides=eval_overrides,
        split_name="test",
        device=device,
    )["summary"]
    final_checkpoint_path = run_dir / "model_final.pt"
    final_test = evaluate_checkpoint(
        checkpoint_path=final_checkpoint_path,
        overrides=eval_overrides,
        split_name="test",
        device=device,
    )["summary"]
    stage_timing["test_eval_seconds"] = time.perf_counter() - stage_start - stage_timing["checkpoint_val_sweep_seconds"]

    bridged_summary = json.loads(bridged_baseline_summary_path.read_text(encoding="utf-8"))
    bounded_summary = json.loads(bounded_summary_path.read_text(encoding="utf-8"))
    compare_paths = write_compare_pack(
        output_dir=output_dir,
        best_summary=best_test,
        final_summary=final_test,
        bridged_summary=bridged_summary,
        bounded_summary=bounded_summary,
    )

    delivery_paths = copy_delivery_artifacts(
        output_dir=output_dir,
        best_checkpoint_path=Path(best_row["checkpoint_path"]),
        final_checkpoint_path=final_checkpoint_path,
        resolved_config_path=Path(train_metrics["resolved_config_path"]),
        run_dir=run_dir,
    )

    selection_manifest = {
        "criterion_order": BEST_CRITERION_ORDER,
        "selection_scope": "formal_val_batch1_sparse_checkpoint_sweep",
        "candidate_checkpoints": val_rows,
        "selected_best_checkpoint": {
            "path": best_row["checkpoint_path"],
            "epoch": best_row["epoch"],
            "val_summary": best_row["metrics"],
            "test_summary": best_test,
        },
        "final_checkpoint": {
            "path": str(final_checkpoint_path),
            "epoch": int(args.epochs),
            "test_summary": final_test,
        },
        "baselines": {
            "bridged_online_summary_path": str(bridged_baseline_summary_path),
            "clean_aligned_bounded_online_summary_path": str(bounded_summary_path),
        },
        "delta": {
            "best_vs_bridged_online": compare_delta(best_test, bridged_summary),
            "best_vs_clean_aligned_bounded_online": compare_delta(best_test, bounded_summary),
            "final_vs_bridged_online": compare_delta(final_test, bridged_summary),
            "final_vs_clean_aligned_bounded_online": compare_delta(final_test, bounded_summary),
        },
    }
    write_json(output_dir / "selection_manifest.json", selection_manifest)

    stage_timing["total_wall_seconds"] = time.perf_counter() - overall_start
    write_json(output_dir / "stage_timing_summary.json", stage_timing)

    run_manifest = {
        "entrypoint": str(Path(__file__).resolve()),
        "authoritative_frozen_navigator_package": str(bridge_package_dir),
        "authoritative_frozen_navigator_checkpoint": str(bridge_package_dir / "navigator_final_selected_best.pt"),
        "authoritative_clean_aligned_bounded_best": str(bounded_best_ckpt),
        "authoritative_clean_aligned_bounded_latest_wrapper": str(bounded_latest_ckpt),
        "train_split_raw_size": raw_train_size,
        "train_loader_visible_size": int(train_metrics.get("train_dataset_size") or 0),
        "train_batches_per_epoch": int(train_metrics.get("train_batches_per_epoch") or 0),
        "run_dir": str(run_dir),
        "resolved_config_path": str(train_metrics["resolved_config_path"]),
        "history_paths": history_paths,
        "throughput_summary_path": str(output_dir / "throughput_summary.json"),
        "stage_timing_summary_path": str(output_dir / "stage_timing_summary.json"),
        "compare_paths": compare_paths,
        "delivery_paths": delivery_paths,
        "selection_manifest_path": str(output_dir / "selection_manifest.json"),
        "cache_prewarm_summary_path": str(output_dir / "cache_prewarm" / "cache_prewarm_summary.json"),
        "batch_gate_summary_path": str(output_dir / "batch_gate" / "batch_gate_summary.json"),
        "seed": 45,
        "panel_version": "formal_val_batch1_and_formal_test_batch1",
    }
    write_json(output_dir / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
