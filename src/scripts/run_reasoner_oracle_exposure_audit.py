from __future__ import annotations

import argparse
import math
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    ActionStep,
    RUNNER_VERSION as SAME_CASE_RUNNER_VERSION,
    PANEL_VERSION,
    build_arm_cfg,
    evaluate_arm_checkpoint,
    load_same_cases,
    read_json,
    summarize_train_metrics,
)
from src.scripts.run_reasoner_train_only_overfit_500 import (
    check_gpu_exclusive,
    collect_candidate_checkpoints,
    safe_float,
    write_json,
)
from src.scripts.train_frozen_clean_nav_reasoner_semidynamic import (
    run_batch_gate,
    train_offline_reasoner,
)
from src.scripts.diagnostics.run_slot1_counterfactual_leverage_audit import (
    build_namespace_from_control_args,
    load_control_bundle,
)


DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact136_h3_formal_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "reasoner_oracle_exposure_audit" / "20260407_exact136_oracle_exposure_v1"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
DEFAULT_BATCH_CANDIDATES = [64, 32, 16]
RUNNER_VERSION = "oracle_exposure_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny-bank oracle-only optimizer exposure audit.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--periodic-checkpoint-every", type=int, default=20)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=DEFAULT_BATCH_CANDIDATES)
    parser.add_argument("--offline-workers", type=int, default=0)
    parser.add_argument("--offline-prefetch-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_action_plan(csv_path: Path) -> Dict[str, List[ActionStep]]:
    df = pd.read_csv(csv_path)
    plan: Dict[str, List[ActionStep]] = {}
    for case_id, group in df.groupby("case_id"):
        plan[str(case_id)] = []
        ordered = group.sort_values("round_index")
        for row in ordered.itertuples(index=False):
            plan[str(case_id)].append(
                ActionStep(
                    case_id=str(row.case_id),
                    scenario_id=int(row.scenario_id),
                    part_id=int(row.part_id),
                    round_index=int(row.round_index),
                    global_ids=[int(v) for v in json.loads(str(row.global_ids))],
                    label=str(row.label),
                )
            )
    return plan


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gpu_snapshot = check_gpu_exclusive()
    write_json(output_dir / "gpu_snapshot_before.json", gpu_snapshot)
    if gpu_snapshot.get("available") and gpu_snapshot.get("exclusive_ok") is False:
        raise RuntimeError(f"GPU is already occupied: {gpu_snapshot['processes']}")

    same_case_manifest = read_json(source_root / "summary.json")
    oracle_arm_manifest = read_json(source_root / "arm_b_task_defined_oracle" / "run_manifest.json")
    oracle_bank_manifest = read_json(source_root / "arm_b_task_defined_oracle" / "trajectory_bank" / "trajectory_bank_manifest.json")
    baseline_metrics = pd.read_csv(source_root / "arm_b_task_defined_oracle" / "train_eval" / "train_checkpoint_metrics.csv")

    bank_lmdb_path = Path(oracle_bank_manifest["lmdb_path"])
    if not bank_lmdb_path.exists():
        raise RuntimeError(f"Oracle bank LMDB does not exist: {bank_lmdb_path}")

    oracle_root = Path(same_case_manifest["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])

    replayable_df = pd.read_csv(source_root / "same_case_replayable_manifest.csv")
    target_case_ids = replayable_df["case_id"].astype(str).tolist()
    cases, dataset_assets = load_same_cases(
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        target_case_ids=target_case_ids,
    )
    bridge_package_dir = Path(oracle_arm_manifest["bridge_package_dir"])
    seed_meta = read_json(bridge_package_dir / "seed_metadata.json")
    control_bundle = load_control_bundle(Path(seed_meta["control_dir"]))
    nav_args = build_namespace_from_control_args(control_bundle["args"], "cpu")
    num_episodes = int(getattr(nav_args, "num_episodes"))
    action_budget = int(getattr(nav_args, "action_budget"))
    episode_duration_min = float(getattr(nav_args, "episode_duration_min"))
    frontier_role_mode = str(getattr(nav_args, "frontier_role_mode"))

    oracle_action_plan = load_action_plan(source_root / "oracle_action_plan.csv")
    init_checkpoint = Path(oracle_arm_manifest["init_checkpoint"])
    sample_count = int(oracle_bank_manifest["sample_count"])

    # For tiny-bank memorization audits, prefer more optimizer exposure over max throughput.
    # Keep only candidates that produce at least 8 batches/epoch when possible.
    candidate_batch_sizes = [int(v) for v in args.batch_candidates]
    exposure_candidates = [
        bs for bs in candidate_batch_sizes if math.ceil(sample_count / max(1, bs)) >= 8 and bs * 2 <= max(sample_count, 1)
    ]
    if not exposure_candidates:
        exposure_candidates = [bs for bs in candidate_batch_sizes if bs * 2 <= max(sample_count, 1)]
    if not exposure_candidates:
        exposure_candidates = [max(1, min(sample_count, candidate_batch_sizes[-1]))]

    cfg = build_arm_cfg(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=int(args.epochs),
        periodic_every=int(args.periodic_checkpoint_every),
        batch_size=min(exposure_candidates),
    )
    batch_gate = run_batch_gate(
        bank_lmdb_path=bank_lmdb_path,
        cfg=cfg,
        init_checkpoint=init_checkpoint,
        output_dir=output_dir,
        candidate_batch_sizes=exposure_candidates,
        num_workers=int(args.offline_workers),
        prefetch_factor=int(args.offline_prefetch_factor),
        device=device,
    )
    chosen_batch_size = int(batch_gate["chosen_batch_size"])
    cfg = build_arm_cfg(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=int(args.epochs),
        periodic_every=int(args.periodic_checkpoint_every),
        batch_size=chosen_batch_size,
    )
    train_result = train_offline_reasoner(
        bank_lmdb_path=bank_lmdb_path,
        cfg=cfg,
        init_checkpoint=init_checkpoint,
        output_dir=output_dir,
        epochs=int(args.epochs),
        batch_size=chosen_batch_size,
        num_workers=int(args.offline_workers),
        prefetch_factor=int(args.offline_prefetch_factor),
        use_amp=True,
        periodic_every=int(args.periodic_checkpoint_every),
        device=device,
    )

    run_dir = Path(train_result["run_dir"])
    history_rows = read_json(output_dir / "train" / "train_loss_curve.json")
    train_loss_by_epoch = {
        int(row["epoch"]): safe_float(row.get("train_loss"))
        for row in history_rows
        if row.get("epoch") is not None
    }
    checkpoints = collect_candidate_checkpoints(run_dir, int(args.epochs), int(args.periodic_checkpoint_every))
    checkpoints.append({"epoch": int(args.epochs), "path": Path(train_result["best_model_state_path"]), "label": "best_train_loss"})
    seen_labels = set()
    metric_rows = []
    for checkpoint in checkpoints:
        if checkpoint["label"] in seen_labels:
            continue
        seen_labels.add(checkpoint["label"])
        checkpoint_path = init_checkpoint if checkpoint["path"] is None else Path(checkpoint["path"])
        evaluated = evaluate_arm_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            cases=cases,
            action_plan=oracle_action_plan,
            dataset_assets=dataset_assets,
            num_episodes=num_episodes,
            action_budget=action_budget,
            episode_duration_min=episode_duration_min,
            frontier_role_mode=frontier_role_mode,
            device=device,
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
            "train_loss": None if checkpoint["label"] == "init" else train_loss_by_epoch.get(int(checkpoint["epoch"])),
            **evaluated["summary"],
        }
        metric_rows.append(row)
        write_json(epoch_dir / "summary.json", row)
    metric_df = pd.DataFrame(metric_rows).sort_values(["epoch", "label"]).reset_index(drop=True)
    metric_df.to_csv(output_dir / "train_eval" / "train_checkpoint_metrics.csv", index=False)
    write_json(output_dir / "train_eval" / "train_checkpoint_metrics.json", metric_rows)
    best_summary = summarize_train_metrics(metric_rows)
    write_json(output_dir / "train_eval" / "best_checkpoint_summary.json", best_summary)

    baseline_best = baseline_metrics.sort_values(
        ["mrr_valid", "top1_hit", "top3_hit", "top5_hit", "true_source_rank_mean"],
        ascending=[False, False, False, False, True],
    ).iloc[0].to_dict()
    baseline_final = baseline_metrics[baseline_metrics["label"] == "final"].iloc[0].to_dict()
    throughput = read_json(output_dir / "throughput_summary.json")
    audit_summary = {
        "runner_version": RUNNER_VERSION,
        "reused_same_case_runner_version": SAME_CASE_RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "source_root": str(source_root),
        "bank_reused": True,
        "bank_lmdb_path": str(bank_lmdb_path),
        "baseline_oracle_manifest_path": str(source_root / "arm_b_task_defined_oracle" / "run_manifest.json"),
        "baseline_sample_count": sample_count,
        "baseline_batch_size": int(oracle_arm_manifest["chosen_batch_size"]),
        "baseline_batches_per_epoch": int(read_json(source_root / "arm_b_task_defined_oracle" / "throughput_summary.json")["train_batches_per_epoch"]),
        "baseline_epochs": int(oracle_arm_manifest["epochs"]),
        "baseline_total_optimizer_steps": int(read_json(source_root / "arm_b_task_defined_oracle" / "throughput_summary.json")["total_train_steps"]),
        "baseline_best": baseline_best,
        "baseline_final": baseline_final,
        "new_candidate_batch_sizes": exposure_candidates,
        "new_chosen_batch_size": chosen_batch_size,
        "new_batches_per_epoch": int(throughput["train_batches_per_epoch"]),
        "new_epochs": int(throughput["num_epochs"]),
        "new_total_optimizer_steps": int(throughput["total_train_steps"]),
        "seed": 45,
    }
    write_json(output_dir / "audit_summary.json", audit_summary)


if __name__ == "__main__":
    main()
