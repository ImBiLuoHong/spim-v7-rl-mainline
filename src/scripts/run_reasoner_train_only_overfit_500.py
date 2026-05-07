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
from typing import Any, Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.dataset import NpzDatasetV6
from src.data.v6.loader import create_dataloaders
from src.modeling.builders.model_builder import ModelBuilder
from src.scripts.audit.build_reasoner_metric_contract_repair_bundle import (
    standardize_case_df,
    standardize_step_df,
)
from src.scripts.diagnostics.run_clean_aligned_reasoner_mainline import prepare_cfg
from src.scripts.diagnostics.run_reasoner_metric_semantics_audit import extract_case_and_step_rows
from src.scripts.train_clean_aligned_online_finish import build_finish_overrides, load_plain_state_dict
from src.scripts.train_phase4_end2end import run_training


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reasoner_train_only_overfit" / "20260407_train500_frozen_nav"
DEFAULT_BRIDGE_PACKAGE = PROJECT_ROOT / "artifacts" / "clean_navigator_v1" / "navigator_final_delivery_p_seed0_newdataset_currentrunner_20260406"
DEFAULT_INIT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "reasoner_clean_aligned_mainline" / "20260403_bounded1" / "delivery" / "best_checkpoint.pt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
DEFAULT_BATCH_CANDIDATES = [500, 250, 125]


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
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
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


def make_cache_version(*, output_dir: Path, max_samples: int) -> str:
    digest = hashlib.md5(f"{output_dir.resolve()}::n{max_samples}".encode("utf-8")).hexdigest()[:8]
    return f"reasoner_train_only_overfit_train_n{max_samples}_{digest}"


def build_overfit_overrides(
    *,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_version: str,
    cache_dir: Path,
    batch_size: int,
    epochs: int,
    periodic_checkpoint_every: int,
    num_workers: int,
    prefetch_factor: int,
    max_samples: int,
    run_name: str,
    split_dir: Path | None = None,
) -> Dict[str, Any]:
    overrides = build_finish_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=cache_version,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        epochs=epochs,
        periodic_checkpoint_every=periodic_checkpoint_every,
        run_name=run_name,
        init_checkpoint=init_checkpoint,
        train_only=True,
        enable_eval=False,
        cache_dir=cache_dir,
    )
    overrides["training"]["enable_eval"] = False
    overrides["training"]["train_only"] = True
    overrides["training"]["val_every_n_epochs"] = int(epochs) + 1
    overrides["training"]["periodic_checkpoint_every_n_epochs"] = int(periodic_checkpoint_every)
    overrides["training"]["ofb_save_every"] = int(periodic_checkpoint_every)
    overrides["training"]["formal_eval_batch_size"] = 1
    overrides["training"]["collect_detailed_step_metrics"] = False
    overrides["training"]["resume"] = False
    overrides["training"]["resume_checkpoint"] = None
    overrides["data"]["max_samples"] = int(max_samples)
    overrides["data"]["rebuild_cache"] = False
    overrides["data"]["skip_lmdb"] = False
    overrides["data"]["num_workers"] = int(num_workers)
    overrides["data"]["prefetch_factor"] = int(prefetch_factor)
    overrides["data"]["pin_memory"] = True
    overrides["data"]["persistent_workers"] = bool(num_workers > 0)
    overrides["paths"]["cache_dir"] = str(cache_dir)
    if split_dir is not None:
        overrides.setdefault("paths", {})["split_dir"] = str(split_dir)
    overrides["efficiency"]["batch_size"] = int(batch_size)
    overrides["efficiency"]["use_amp"] = True
    overrides["efficiency"]["num_workers"] = int(num_workers)
    overrides["efficiency"]["prefetch_factor"] = int(prefetch_factor)
    overrides["efficiency"]["pin_memory"] = True
    overrides["efficiency"]["persistent_workers"] = bool(num_workers > 0)
    overrides.setdefault("efficiency", {}).setdefault("performance", {})["tf32"] = True
    return overrides


def build_subset_manifest(*, max_samples: int, split_dir: Path | None = None) -> Dict[str, Any]:
    cfg = Config()
    dataset = NpzDatasetV6(
        samples_dir=cfg.paths.samples_path,
        foundation_dir=cfg.paths.foundation_path,
        mode="train",
        window_size=cfg.data.window_size,
        split_dir=str(split_dir) if split_dir is not None else cfg.paths.split_dir,
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
        max_samples=max_samples,
    )
    return {
        "selection_rule": f"first {max_samples} grouped train scenarios in current split order (`data/train.txt` order after grouping/filtering)",
        "max_samples": int(max_samples),
        "scenario_count": int(len(dataset)),
        "split_dir": str(split_dir) if split_dir is not None else str(cfg.paths.split_dir),
        "first_groups": dataset.groups[:5],
        "last_groups": dataset.groups[-5:],
    }


def warm_train_cache(overrides: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    start = time.perf_counter()
    cfg = prepare_cfg(overrides, run_name=overrides["training"]["run_name"], max_epochs=1, seed=45)
    loader, _, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=max(1, min(int(cfg.efficiency.batch_size), 64)),
        eval_batch_size=1,
        skip_lmdb=bool(cfg.data.skip_lmdb),
        train_only=True,
    )
    batch = next(iter(loader))
    batch_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 0
    return {
        "cache_version": cfg.data.cache_version,
        "cache_dir": cfg.paths.cache_dir,
        "dataset_size": int(len(loader.dataset)),
        "train_batches_per_epoch": int(len(loader)),
        "first_batch_graphs": batch_graphs,
        "wall_s": time.perf_counter() - start,
        "device": str(device),
    }


def probe_batch_sizes(
    *,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_dir: Path,
    cache_version: str,
    output_dir: Path,
    candidate_batch_sizes: List[int],
    max_samples: int,
    num_workers: int,
    prefetch_factor: int,
    split_dir: Path | None = None,
) -> Dict[str, Any]:
    rows = []
    best_row = None
    for batch_size in candidate_batch_sizes:
        gate_output = output_dir / "batch_gate" / f"bs{batch_size}"
        overrides = build_overfit_overrides(
            bridge_package_dir=bridge_package_dir,
            init_checkpoint=init_checkpoint,
            cache_version=cache_version,
            cache_dir=cache_dir,
            batch_size=batch_size,
            epochs=1,
            periodic_checkpoint_every=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            max_samples=max_samples,
            run_name=f"{output_dir.name}_gate_bs{batch_size}",
            split_dir=split_dir,
        )
        start = time.perf_counter()
        try:
            metrics = run_training(
                loss_config_override=overrides,
                run_name=f"{output_dir.name}_gate_bs{batch_size}",
                max_epochs=1,
                seed=45,
                skip_audit=True,
                force_rebuild=False,
            )
            row = {
                "batch_size": int(batch_size),
                "ok": True,
                "run_dir": metrics.get("run_dir"),
                "train_dataset_size": int(metrics.get("train_dataset_size") or 0),
                "train_batches_per_epoch": int(metrics.get("train_batches_per_epoch") or 0),
                "train_loop_wall_s": safe_float(metrics.get("run_timing", {}).get("train_loop_wall_s")),
                "samples_per_sec": (
                    int(metrics.get("train_dataset_size") or 0)
                    / max(safe_float(metrics.get("run_timing", {}).get("train_loop_wall_s")), 1e-9)
                ),
                "steps_per_sec": (
                    int(metrics.get("train_batches_per_epoch") or 0)
                    / max(safe_float(metrics.get("run_timing", {}).get("train_loop_wall_s")), 1e-9)
                ),
                "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
            }
        except RuntimeError as exc:
            row = {
                "batch_size": int(batch_size),
                "ok": False,
                "error": str(exc),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row["wall_s"] = time.perf_counter() - start
        rows.append(row)
        write_json(gate_output / "summary.json", row)
        if row.get("ok"):
            if best_row is None or float(row["samples_per_sec"]) > float(best_row["samples_per_sec"]):
                best_row = row
    if best_row is None:
        raise RuntimeError(f"No stable batch candidate found. Rows={rows}")
    summary = {
        "gate_mode": "one_epoch_train_only_overfit_probe",
        "candidate_batch_sizes": [int(x) for x in candidate_batch_sizes],
        "chosen": best_row,
        "rows": rows,
        "cache_version": cache_version,
        "max_samples": int(max_samples),
    }
    write_json(output_dir / "batch_gate" / "batch_gate_summary.json", summary)
    return summary


def write_train_history(run_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    history_path = run_dir / "epoch_history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    write_json(output_dir / "train" / "train_history.json", rows)
    with (output_dir / "train" / "train_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "tau",
                "epoch_train_loop_wall_s",
                "epoch_run_training_wall_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "train_loss": row.get("train_loss"),
                    "tau": row.get("tau"),
                    "epoch_train_loop_wall_s": row.get("epoch_train_loop_wall_s"),
                    "epoch_run_training_wall_s": row.get("epoch_run_training_wall_s"),
                }
            )
    return rows


def collect_candidate_checkpoints(run_dir: Path, total_epochs: int, periodic_every: int) -> List[Dict[str, Any]]:
    checkpoints = [{"epoch": 0, "path": None, "label": "init"}]
    ckpt_dir = run_dir / "checkpoints"
    for epoch in range(periodic_every, total_epochs + 1, periodic_every):
        path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        if path.exists():
            checkpoints.append({"epoch": int(epoch), "path": path, "label": f"epoch_{epoch:03d}"})
    final_path = run_dir / "model_final.pt"
    if final_path.exists():
        checkpoints.append({"epoch": int(total_epochs), "path": final_path, "label": "final"})
    return checkpoints


def evaluate_train_checkpoint(
    *,
    checkpoint_path: Path,
    overrides: Dict[str, Any],
    device: torch.device,
    replay_batch_size: int,
) -> Dict[str, Any]:
    cfg = prepare_cfg(overrides, run_name=overrides["training"]["run_name"], max_epochs=1, seed=45)
    loader, _, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=int(replay_batch_size),
        eval_batch_size=int(replay_batch_size),
        skip_lmdb=bool(cfg.data.skip_lmdb),
        train_only=True,
    )
    model = ModelBuilder.build_model(cfg).to(device)
    model.load_state_dict(load_plain_state_dict(checkpoint_path), strict=True)
    model.eval()
    extracted = extract_case_and_step_rows(
        model=model,
        loader=loader,
        cfg=cfg,
        device=device,
        split_name="train",
        max_cases=None,
    )
    raw_case_df = extracted["cases"].sort_values("case_id").reset_index(drop=True)
    raw_step_df = extracted["steps"].sort_values(["case_id", "episode_index"]).reset_index(drop=True)
    std_step_df = standardize_step_df(raw_step_df, split="train")
    std_case_df = standardize_case_df(raw_case_df, std_step_df, split="train")
    valid = std_case_df[std_case_df["valid_case"] == True].copy()
    summary = {
        "case_count": int(len(std_case_df)),
        "valid_final_ranking_case_count": int(len(valid)),
        "top1_hit": float(valid["final_top1_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "top3_hit": float(valid["final_top3_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "top5_hit": float(valid["final_top5_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "mrr_valid": float(valid["final_mrr"].mean()) if len(valid) else float("nan"),
        "true_source_rank_mean": float(valid["final_true_source_rank"].mean()) if len(valid) else float("nan"),
        "median_true_source_rank": float(valid["final_true_source_rank"].median()) if len(valid) else float("nan"),
    }
    return {
        "summary": summary,
        "raw_case_df": raw_case_df,
        "raw_step_df": raw_step_df,
        "std_case_df": std_case_df,
        "std_step_df": std_step_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only 500-case reasoner overfit verdict under frozen navigator trajectories.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bridge-package-dir", type=str, default=str(DEFAULT_BRIDGE_PACKAGE))
    parser.add_argument("--init-checkpoint", type=str, default=str(DEFAULT_INIT_CHECKPOINT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--periodic-checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=DEFAULT_BATCH_CANDIDATES)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--replay-batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_package_dir = Path(args.bridge_package_dir)
    init_checkpoint = Path(args.init_checkpoint)
    cache_dir = Path(args.cache_dir)
    split_dir = Path(args.split_dir) if args.split_dir else None
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gpu_snapshot = check_gpu_exclusive()
    write_json(output_dir / "gpu_snapshot_before.json", gpu_snapshot)
    if gpu_snapshot.get("available") and gpu_snapshot.get("exclusive_ok") is False:
        raise RuntimeError(f"GPU is already occupied: {gpu_snapshot['processes']}")

    subset_manifest = build_subset_manifest(max_samples=int(args.max_samples), split_dir=split_dir)
    write_json(output_dir / "subset_manifest.json", subset_manifest)

    cache_version = make_cache_version(output_dir=output_dir, max_samples=int(args.max_samples))
    base_overrides = build_overfit_overrides(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_version=cache_version,
        cache_dir=cache_dir,
        batch_size=min(args.batch_candidates),
        epochs=int(args.epochs),
        periodic_checkpoint_every=int(args.periodic_checkpoint_every),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        max_samples=int(args.max_samples),
        run_name=f"{output_dir.name}_probe_cfg",
        split_dir=split_dir,
    )
    warm_summary = warm_train_cache(base_overrides, device)
    write_json(output_dir / "cache_prewarm" / "cache_prewarm_summary.json", warm_summary)

    batch_gate = probe_batch_sizes(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        cache_version=cache_version,
        output_dir=output_dir,
        candidate_batch_sizes=[int(x) for x in args.batch_candidates],
        max_samples=int(args.max_samples),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        split_dir=split_dir,
    )
    chosen_batch_size = int(batch_gate["chosen"]["batch_size"])

    train_overrides = build_overfit_overrides(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_version=cache_version,
        cache_dir=cache_dir,
        batch_size=chosen_batch_size,
        epochs=int(args.epochs),
        periodic_checkpoint_every=int(args.periodic_checkpoint_every),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        max_samples=int(args.max_samples),
        run_name=output_dir.name,
        split_dir=split_dir,
    )
    write_json(output_dir / "overrides_manifest.json", train_overrides)

    train_start = time.perf_counter()
    train_metrics = run_training(
        loss_config_override=train_overrides,
        run_name=output_dir.name,
        max_epochs=int(args.epochs),
        seed=45,
        skip_audit=True,
        force_rebuild=False,
    )
    run_dir = Path(train_metrics["run_dir"])
    train_history = write_train_history(run_dir, output_dir)
    train_wall_s = time.perf_counter() - train_start
    write_json(output_dir / "train" / "run_metrics.json", train_metrics)

    throughput_summary = {
        "train_dataset_size": int(train_metrics.get("train_dataset_size") or 0),
        "train_batches_per_epoch": int(train_metrics.get("train_batches_per_epoch") or 0),
        "num_epochs": int(args.epochs),
        "batch_size": chosen_batch_size,
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "cache_version": cache_version,
        "cache_dir": str(cache_dir),
        "train_loop_wall_s": safe_float(train_metrics.get("run_timing", {}).get("train_loop_wall_s")),
        "run_training_wall_s": safe_float(train_metrics.get("run_timing", {}).get("run_training_wall_s")),
        "external_wall_s": float(train_wall_s),
        "samples_per_sec": (
            (int(train_metrics.get("train_dataset_size") or 0) * int(args.epochs))
            / max(safe_float(train_metrics.get("run_timing", {}).get("train_loop_wall_s")), 1e-9)
        ),
        "steps_per_sec": (
            (int(train_metrics.get("train_batches_per_epoch") or 0) * int(args.epochs))
            / max(safe_float(train_metrics.get("run_timing", {}).get("train_loop_wall_s")), 1e-9)
        ),
        "peak_gpu_memory_mb": train_metrics.get("peak_gpu_memory_mb"),
    }
    write_json(output_dir / "throughput_summary.json", throughput_summary)

    train_loss_by_epoch = {int(row["epoch"]): safe_float(row.get("train_loss")) for row in train_history}
    eval_rows = []
    checkpoints = collect_candidate_checkpoints(run_dir, int(args.epochs), int(args.periodic_checkpoint_every))
    for checkpoint in checkpoints:
        checkpoint_path = init_checkpoint if checkpoint["path"] is None else Path(checkpoint["path"])
        evaluated = evaluate_train_checkpoint(
            checkpoint_path=checkpoint_path,
            overrides=train_overrides,
            device=device,
            replay_batch_size=int(args.replay_batch_size),
        )
        epoch_dir = output_dir / "train_eval" / checkpoint["label"]
        epoch_dir.mkdir(parents=True, exist_ok=True)
        evaluated["raw_case_df"].to_csv(epoch_dir / "raw_case_metrics.csv", index=False)
        evaluated["raw_step_df"].to_csv(epoch_dir / "raw_step_metrics.csv", index=False)
        evaluated["std_case_df"].to_csv(epoch_dir / "standardized_case_metrics.csv", index=False)
        evaluated["std_step_df"].to_csv(epoch_dir / "standardized_step_metrics.csv", index=False)
        row = {
            "epoch": int(checkpoint["epoch"]),
            "label": checkpoint["label"],
            "checkpoint_path": str(checkpoint_path),
            "train_loss": None if int(checkpoint["epoch"]) == 0 else train_loss_by_epoch.get(int(checkpoint["epoch"])),
            **evaluated["summary"],
        }
        eval_rows.append(row)
        write_json(epoch_dir / "summary.json", row)

    eval_df = pd.DataFrame(eval_rows).sort_values(["epoch", "label"]).reset_index(drop=True)
    eval_df.to_csv(output_dir / "train_eval" / "train_checkpoint_metrics.csv", index=False)
    write_json(output_dir / "train_eval" / "train_checkpoint_metrics.json", eval_rows)

    final_manifest = {
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "bridge_package_dir": str(bridge_package_dir),
        "init_checkpoint": str(init_checkpoint),
        "train_only": True,
        "val_used": False,
        "test_used": False,
        "max_samples": int(args.max_samples),
        "epochs": int(args.epochs),
        "periodic_checkpoint_every": int(args.periodic_checkpoint_every),
        "chosen_batch_size": int(chosen_batch_size),
        "replay_batch_size": int(args.replay_batch_size),
        "fixed_trajectory_contract": {
            "navigator_type": "frozen_clean_v1_bridge",
            "sampling_policy": "navigator_only",
            "navigator_updated_during_training": False,
            "rl_used": False,
            "note": "Train distribution stays on the frozen navigator induced trajectory contract; no navigator optimization or action-policy switch is introduced in this overfit run.",
        },
        "split_dir": str(split_dir) if split_dir is not None else None,
    }
    write_json(output_dir / "run_manifest.json", final_manifest)


if __name__ == "__main__":
    main()
