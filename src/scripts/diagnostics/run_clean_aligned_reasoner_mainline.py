from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.builders.model_builder import ModelBuilder
from src.scripts.train_phase4_end2end import run_training
from src.scripts.training_campaign_round1 import (
    build_eval_loaders_with_batch,
    compare_delta,
    evaluate_split,
    load_yaml,
    prepare_cfg,
)


DEFAULT_BRIDGE_PACKAGE = PROJECT_ROOT / "artifacts/clean_navigator_v1/navigator_final_delivery_p_seed0_currentrunner_20260402"
DEFAULT_BASELINE_CKPT = PROJECT_ROOT / "runs/frozen_clean_nav_reasoner_mainline_b03eba49ddf5/model_best.pt"
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "artifacts/reasoner_frozen_clean_nav_bridge/20260402_mainline_bounded2/eval/frozen_clean_nav_reasoner_test_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/reasoner_clean_aligned_mainline/20260403_bounded1"


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def make_cache_version(
    *,
    stage: str,
    output_dir: Path,
    max_samples: int | None,
    batch_size: int,
    epochs: int,
) -> str:
    sample_tag = "full" if max_samples is None else str(max_samples)
    scope = f"{output_dir.resolve()}::{stage}::{sample_tag}::{batch_size}::{epochs}"
    digest = hashlib.md5(scope.encode("utf-8")).hexdigest()[:8]
    return f"clean_aligned_reasoner_{stage}_m{sample_tag}_b{batch_size}_e{epochs}_{digest}"


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


def build_aligned_overrides(
    *,
    bridge_package_dir: Path,
    cache_version: str,
    max_samples: int | None,
    batch_size: int,
    epochs: int,
    run_name: str,
    rebuild_cache: bool = False,
) -> Dict[str, Any]:
    base = load_yaml("configs/evidence_v1/support_mainline.yaml")
    aligned = load_yaml("configs/evidence_v1/clean_aligned_reasoner_mainline.yaml")
    bridge_control_dir = json.loads((bridge_package_dir / "seed_metadata.json").read_text(encoding="utf-8"))["control_dir"]
    overlay = {
        "system": {
            "enable_audit": False,
        },
        "training": {
            "run_name": run_name,
            "num_epochs": epochs,
            "val_every_n_epochs": 1,
            "enable_wandb": False,
            "formal_eval_batch_size": 1,
            "log_every_n_steps": 20,
            "collect_detailed_step_metrics": False,
        },
        "data": {
            "cache_version": cache_version,
            "rebuild_cache": rebuild_cache,
            "skip_lmdb": False,
            "max_samples": max_samples,
            "num_workers": 8,
            "prefetch_factor": 2,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "efficiency": {
            "batch_size": batch_size,
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


def build_aligned_init_checkpoint(
    *,
    output_dir: Path,
    baseline_ckpt_path: Path,
    standalone_nav_ckpt_path: Path,
    bridge_package_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    init_ckpt_path = output_dir / "init" / "phase45_clean_aligned_reasoner_init.pt"
    manifest_path = output_dir / "init" / "init_manifest.json"
    overrides = build_aligned_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version="clean_aligned_build_eval_only",
        max_samples=1,
        batch_size=1,
        epochs=1,
        run_name="clean_aligned_reasoner_build",
        rebuild_cache=False,
    )
    cfg = prepare_cfg(overrides, run_name="clean_aligned_reasoner_build", max_epochs=1, seed=45)
    model = ModelBuilder.build_model(cfg).to(device)

    baseline_state = torch.load(baseline_ckpt_path, map_location=device)
    transferred = {}
    for key, value in baseline_state.items():
        if key.startswith("reasoner_module."):
            continue
        if key.startswith("navigator_module.") and not key.startswith("navigator_module.backbone."):
            continue
        transferred[key] = value
    baseline_load = model.load_state_dict(transferred, strict=False)

    standalone_state = torch.load(standalone_nav_ckpt_path, map_location=device)
    nav_load = model.navigator_module.clean_navigator.load_state_dict(standalone_state, strict=True)
    for param in model.navigator_module.parameters():
        param.requires_grad = False
    model.eval()

    init_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), init_ckpt_path)
    manifest = {
        "init_checkpoint_path": str(init_ckpt_path),
        "baseline_checkpoint_path": str(baseline_ckpt_path),
        "standalone_navigator_checkpoint_path": str(standalone_nav_ckpt_path),
        "bridge_package_dir": str(bridge_package_dir),
        "load_steps": {
            "baseline_non_reasoner_load": {
                "transferred_key_count": len(transferred),
                "missing_key_count": len(baseline_load.missing_keys),
                "unexpected_key_count": len(baseline_load.unexpected_keys),
                "missing_keys": baseline_load.missing_keys,
                "unexpected_keys": baseline_load.unexpected_keys,
            },
            "standalone_clean_navigator_load": {
                "transferred_key_count": len(standalone_state),
                "missing_key_count": len(nav_load.missing_keys),
                "unexpected_key_count": len(nav_load.unexpected_keys),
                "missing_keys": nav_load.missing_keys,
                "unexpected_keys": nav_load.unexpected_keys,
            },
        },
        "default_runtime_contract": {
            "navigator_type": "frozen_clean_v1_bridge",
            "sampling_policy": "navigator_only",
            "training_mode": "frozen_nav",
            "reasoner_type": "clean_aligned_reasoner_mainline",
            "primary_frozen_ckpt": str(standalone_nav_ckpt_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def run_eval_summary(
    *,
    checkpoint_path: Path,
    overrides: Dict[str, Any],
    device: torch.device,
    eval_batch_size: int = 1,
) -> Dict[str, Any]:
    cfg = prepare_cfg(overrides, run_name="clean_aligned_reasoner_eval", max_epochs=1, seed=45)
    _, eval_loaders = build_eval_loaders_with_batch(eval_batch_size=eval_batch_size)
    model = ModelBuilder.build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    summary, df = evaluate_split(model, eval_loaders["test"], cfg, "test", device, eval_policy="formal")
    return {"summary": summary, "per_case_rows": len(df)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded clean-aligned reasoner training on frozen clean navigator trajectories.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bridge-package-dir", type=str, default=str(DEFAULT_BRIDGE_PACKAGE))
    parser.add_argument("--baseline-ckpt", type=str, default=str(DEFAULT_BASELINE_CKPT))
    parser.add_argument("--baseline-summary-json", type=str, default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke-max-samples", type=int, default=32)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-batch-size", type=int, default=4)
    parser.add_argument("--train-max-samples", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_package_dir = Path(args.bridge_package_dir)
    baseline_ckpt_path = Path(args.baseline_ckpt)
    baseline_summary_path = Path(args.baseline_summary_json)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gpu_snapshot = check_gpu_exclusive()
    write_json(output_dir / "gpu_snapshot_before.json", gpu_snapshot)
    if gpu_snapshot.get("available") and gpu_snapshot.get("exclusive_ok") is False:
        raise RuntimeError(f"GPU is already occupied by other compute processes: {gpu_snapshot['processes']}")

    stage_timing: Dict[str, float] = {}
    smoke_cache_version = make_cache_version(
        stage="smoke",
        output_dir=output_dir,
        max_samples=int(args.smoke_max_samples),
        batch_size=int(args.smoke_batch_size),
        epochs=int(args.smoke_epochs),
    )
    train_cache_version = make_cache_version(
        stage="train",
        output_dir=output_dir,
        max_samples=int(args.train_max_samples),
        batch_size=int(args.train_batch_size),
        epochs=int(args.train_epochs),
    )

    stage_start = time.perf_counter()
    init_manifest = build_aligned_init_checkpoint(
        output_dir=output_dir,
        baseline_ckpt_path=baseline_ckpt_path,
        standalone_nav_ckpt_path=bridge_package_dir / "navigator_final_selected_best.pt",
        bridge_package_dir=bridge_package_dir,
        device=device,
    )
    stage_timing["init_build_seconds"] = time.perf_counter() - stage_start

    smoke_overrides = build_aligned_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=smoke_cache_version,
        max_samples=int(args.smoke_max_samples),
        batch_size=int(args.smoke_batch_size),
        epochs=int(args.smoke_epochs),
        run_name="clean_aligned_reasoner_smoke",
        rebuild_cache=bool(args.force_rebuild),
    )
    smoke_overrides["training"]["init_checkpoint"] = init_manifest["init_checkpoint_path"]
    smoke_overrides["training"]["init_checkpoint_strict"] = True

    stage_start = time.perf_counter()
    smoke_metrics = run_training(
        loss_config_override=smoke_overrides,
        run_name="clean_aligned_reasoner_smoke",
        max_epochs=int(args.smoke_epochs),
        seed=45,
        skip_audit=True,
        force_rebuild=bool(args.force_rebuild),
    )
    stage_timing["consumer_smoke_seconds"] = time.perf_counter() - stage_start
    write_json(output_dir / "smoke" / "run_metrics.json", smoke_metrics)

    train_overrides = build_aligned_overrides(
        bridge_package_dir=bridge_package_dir,
        cache_version=train_cache_version,
        max_samples=int(args.train_max_samples),
        batch_size=int(args.train_batch_size),
        epochs=int(args.train_epochs),
        run_name="clean_aligned_reasoner_mainline",
        rebuild_cache=bool(args.force_rebuild),
    )
    train_overrides["training"]["init_checkpoint"] = init_manifest["init_checkpoint_path"]
    train_overrides["training"]["init_checkpoint_strict"] = True

    stage_start = time.perf_counter()
    train_metrics = run_training(
        loss_config_override=train_overrides,
        run_name="clean_aligned_reasoner_mainline",
        max_epochs=int(args.train_epochs),
        seed=45,
        skip_audit=True,
        force_rebuild=bool(args.force_rebuild),
    )
    stage_timing["bounded_training_seconds"] = time.perf_counter() - stage_start
    write_json(output_dir / "train" / "run_metrics.json", train_metrics)

    final_ckpt = Path(train_metrics.get("final_checkpoint_path") or (Path(train_metrics["run_dir"]) / "model_best.pt"))
    stage_start = time.perf_counter()
    final_eval = run_eval_summary(
        checkpoint_path=final_ckpt,
        overrides=train_overrides,
        device=device,
    )
    stage_timing["final_eval_seconds"] = time.perf_counter() - stage_start
    write_json(output_dir / "eval" / "clean_aligned_reasoner_test_summary.json", final_eval["summary"])

    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    compare_summary = {
        "baseline_path": str(baseline_summary_path),
        "baseline_summary": baseline_summary,
        "candidate_summary": final_eval["summary"],
        "delta_vs_baseline": compare_delta(final_eval["summary"], baseline_summary),
    }
    write_json(output_dir / "compare" / "baseline_compare.json", compare_summary)

    stage_timing["total_wall_seconds"] = sum(stage_timing.values())
    write_json(output_dir / "stage_timing_summary.json", stage_timing)
    final_summary = {
        "init_manifest_path": str(output_dir / "init" / "init_manifest.json"),
        "smoke_metrics_path": str(output_dir / "smoke" / "run_metrics.json"),
        "train_metrics_path": str(output_dir / "train" / "run_metrics.json"),
        "final_eval_path": str(output_dir / "eval" / "clean_aligned_reasoner_test_summary.json"),
        "compare_path": str(output_dir / "compare" / "baseline_compare.json"),
        "stage_timing_summary_path": str(output_dir / "stage_timing_summary.json"),
        "final_checkpoint_path": str(final_ckpt),
        "smoke_cache_version": smoke_cache_version,
        "train_cache_version": train_cache_version,
    }
    write_json(output_dir / "summary.json", final_summary)


if __name__ == "__main__":
    main()
