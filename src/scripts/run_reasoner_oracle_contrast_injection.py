from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.builders.model_builder import ModelBuilder
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.build_reasoner_metric_contract_repair_bundle import standardize_case_df, standardize_step_df
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.run_reasoner_oracle_exposure_audit import load_action_plan
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    PANEL_VERSION as SAME_CASE_PANEL_VERSION,
    TempGraph,
    build_arm_cfg,
    build_state_input,
    load_same_cases,
    make_rollout_state,
    move_payload,
    read_json,
    summarize_train_metrics,
    translate_global_ids,
)
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_reasoner_train_only_overfit_500 import (
    check_gpu_exclusive,
    collect_candidate_checkpoints,
    safe_float,
    write_json,
)
from src.scripts.train_clean_aligned_online_finish import load_plain_state_dict
from src.scripts.train_frozen_clean_nav_reasoner_semidynamic import run_batch_gate, train_offline_reasoner
from src.scripts.diagnostics.run_slot1_counterfactual_leverage_audit import build_namespace_from_control_args, load_control_bundle

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact136_h3_formal_v1"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_oracle_exposure_audit" / "20260407_exact136_oracle_exposure_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reasoner_oracle_contrast_injection" / "20260407_exact136_oracle_contrast_v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
DEFAULT_BATCH_CANDIDATES = [64, 32, 16]
RUNNER_VERSION = "oracle_contrast_injection_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal training-side evidence-core contrast injection run.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--baseline-root", type=str, default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--periodic-checkpoint-every", type=int, default=20)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=DEFAULT_BATCH_CANDIDATES)
    parser.add_argument("--offline-workers", type=int, default=0)
    parser.add_argument("--offline-prefetch-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def enable_contrast_adapter(cfg: Any) -> Any:
    reasoner_cfg = getattr(cfg.model, "reasoner", None)
    if isinstance(reasoner_cfg, dict):
        reasoner_cfg["enable_evidence_core_contrast_adapter"] = True
        reasoner_cfg["evidence_core_contrast_mode"] = "residual"
    else:
        setattr(reasoner_cfg, "enable_evidence_core_contrast_adapter", True)
        setattr(reasoner_cfg, "evidence_core_contrast_mode", "residual")
    return cfg


def build_cfg_with_contrast(
    *,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_dir: Path,
    epochs: int,
    periodic_every: int,
    batch_size: int,
) -> Any:
    cfg = build_arm_cfg(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=epochs,
        periodic_every=periodic_every,
        batch_size=batch_size,
    )
    return enable_contrast_adapter(cfg)


def load_model_with_contrast(cfg: Any, checkpoint_path: Path, device: torch.device):
    model = ModelBuilder.build_model(cfg).to(device)
    state_dict = load_plain_state_dict(checkpoint_path)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    allowed_missing = {
        "reasoner_module.evidence_core_contrast_adapter.weight",
        "reasoner_module.evidence_core_contrast_adapter.bias",
    }
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    if disallowed_missing or unexpected:
        raise RuntimeError(f"Contrast eval checkpoint load mismatch. missing={disallowed_missing}, unexpected={unexpected}")
    model.eval()
    return model


def score_reasoner_state(reasoner_module: torch.nn.Module, state: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    source_local_idx = resolve_source_local_idx(state["rollout"])
    valid_mask = state["valid_mask"].view(-1).bool()
    if source_local_idx is None or not bool(valid_mask[int(source_local_idx)].item()):
        return {"valid_case": False, "true_source_rank": None, "mrr": 0.0, "top1_hit": False, "top3_hit": False, "top5_hit": False, "softmax_candidate_count": int(valid_mask.sum().item())}
    graph = TempGraph(state["edge_index"], int(valid_mask.numel()), device)
    state_input = move_payload(build_state_input(state), device)
    physics_ctx = move_payload(state["phys_ctx"].__dict__, device)
    with torch.no_grad():
        out = reasoner_module(state_input, graph, physics_ctx=physics_ctx)
    logits = out["logits"].detach().float().view(-1).cpu()
    safe_logits = logits.clone()
    safe_logits[~valid_mask.cpu()] = -float("inf")
    order = torch.argsort(safe_logits, descending=True)
    valid_order = order[torch.isfinite(safe_logits[order])]
    positions = (valid_order == int(source_local_idx)).nonzero(as_tuple=True)[0]
    if positions.numel() <= 0:
        return {"valid_case": False, "true_source_rank": None, "mrr": 0.0, "top1_hit": False, "top3_hit": False, "top5_hit": False, "softmax_candidate_count": int(valid_mask.sum().item())}
    rank = int(positions.min().item()) + 1
    return {
        "valid_case": True,
        "true_source_rank": int(rank),
        "mrr": 1.0 / float(rank),
        "top1_hit": bool(rank <= 1),
        "top3_hit": bool(rank <= 3),
        "top5_hit": bool(rank <= 5),
        "softmax_candidate_count": int(valid_order.numel()),
    }


def evaluate_checkpoint_with_contrast(
    *,
    checkpoint_path: Path,
    cfg: Any,
    cases: List[Any],
    action_plan: Dict[str, List[Any]],
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    env = CleanTwoChannelEvidenceEnv()
    topology = dataset_assets["topology"]
    model = load_model_with_contrast(cfg, checkpoint_path, device)
    reasoner_module = getattr(model, "reasoner_module", model)
    raw_case_rows = []
    raw_step_rows = []
    for case in cases:
        rollout = PracticalRollout(
            event_data=deepcopy(case.data),
            global_edge_index=dataset_assets["global_edge_index"],
            stt_dynamic_series=dataset_assets["stt_dynamic_series"],
            num_global_nodes=int(dataset_assets["num_global_nodes"]),
            num_episodes=int(num_episodes),
            samples_per_episode=int(action_budget),
            episode_duration_min=float(episode_duration_min),
        )
        history = ObservationWitnessHistory()
        plan_steps = action_plan.get(case.case_id, [])
        first_success_episode = None
        for step in plan_steps:
            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"eval_slot_{idx}" for idx in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            post_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=num_episodes,
                action_budget=action_budget,
                frontier_role_mode=frontier_role_mode,
            )
            post_metrics = score_reasoner_state(reasoner_module, post_state, device)
            if first_success_episode is None and bool(post_metrics.get("top1_hit")):
                first_success_episode = int(step.round_index)
            observed_count = int(rollout.revealed_mask.sum().item())
            raw_step_rows.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "part_id": case.part_id,
                    "episode_index": int(step.round_index) + 1,
                    "true_source_rank": post_metrics.get("true_source_rank"),
                    "top1_hit": post_metrics.get("top1_hit"),
                    "top3_hit": post_metrics.get("top3_hit"),
                    "top5_hit": post_metrics.get("top5_hit"),
                    "mrr": post_metrics.get("mrr"),
                    "logits_candidate_size": post_metrics.get("softmax_candidate_count"),
                    "pre_action_valid_size": int(post_state["valid_mask"].sum().item()),
                    "post_action_valid_size": int(post_state["valid_mask"].sum().item()),
                    "unrevealed_candidate_ratio": float(post_metrics.get("softmax_candidate_count") or 0) / max(int(post_state["valid_mask"].sum().item()), 1),
                    "total_nodes": int(post_state["valid_mask"].numel()),
                    "revealed_ratio": float(observed_count) / max(int(post_state["valid_mask"].numel()), 1),
                    "observed_count": observed_count,
                    "fallback_triggered": False,
                }
            )
        final_state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=num_episodes,
            action_budget=action_budget,
            frontier_role_mode=frontier_role_mode,
        )
        final_metrics = score_reasoner_state(reasoner_module, final_state, device)
        final_observed = int(rollout.revealed_mask.sum().item())
        total_nodes = int(final_state["valid_mask"].numel())
        final_candidate_count = int(final_metrics.get("softmax_candidate_count") or 0)
        raw_case_rows.append(
            {
                "case_id": case.case_id,
                "scenario_id": case.scenario_id,
                "part_id": case.part_id,
                "success": bool(final_metrics.get("top1_hit")),
                "budget_used": float(final_observed),
                "episodes_completed": float(len(plan_steps)),
                "physical_time_mins": float(rollout.current_time_min),
                "first_success_episode": first_success_episode,
                "step_count_observed": int(len(plan_steps)),
                "final_top1_hit": final_metrics.get("top1_hit"),
                "final_top3_hit": final_metrics.get("top3_hit"),
                "final_top5_hit": final_metrics.get("top5_hit"),
                "final_mrr": final_metrics.get("mrr"),
                "final_true_source_rank": final_metrics.get("true_source_rank"),
                "final_entropy": None,
                "final_max_prob": None,
                "final_logits_candidate_size": final_candidate_count,
                "final_pre_action_valid_size": int(final_state["valid_mask"].sum().item()),
                "final_post_action_valid_size": int(final_state["valid_mask"].sum().item()),
                "final_revealed_ratio": float(final_observed) / max(total_nodes, 1),
                "final_revealed_candidate_count": max(total_nodes - final_candidate_count, 0),
                "final_unrevealed_candidate_count": final_candidate_count,
                "final_unrevealed_candidate_ratio": float(final_candidate_count) / max(total_nodes, 1),
                "total_nodes": total_nodes,
                "final_confirmed_source_count": 0,
                "final_confirmed_non_source_count": final_observed,
                "final_no_resample_count": final_observed,
                "valid_case": bool(final_metrics.get("valid_case")),
            }
        )
    raw_case_df = pd.DataFrame(raw_case_rows).sort_values("case_id").reset_index(drop=True)
    raw_step_df = pd.DataFrame(raw_step_rows).sort_values(["case_id", "episode_index"]).reset_index(drop=True)
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
    return {"raw_case_df": raw_case_df, "raw_step_df": raw_step_df, "std_case_df": std_case_df, "std_step_df": std_step_df, "summary": summary}


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    baseline_root = Path(args.baseline_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gpu_snapshot = check_gpu_exclusive()
    write_json(output_dir / "gpu_snapshot_before.json", gpu_snapshot)
    if gpu_snapshot.get("available") and gpu_snapshot.get("exclusive_ok") is False:
        raise RuntimeError(f"GPU is already occupied: {gpu_snapshot['processes']}")

    source_summary = read_json(source_root / "summary.json")
    oracle_arm_manifest = read_json(source_root / "arm_b_task_defined_oracle" / "run_manifest.json")
    oracle_bank_manifest = read_json(source_root / "arm_b_task_defined_oracle" / "trajectory_bank" / "trajectory_bank_manifest.json")
    baseline_metrics = pd.read_csv(baseline_root / "train_eval" / "train_checkpoint_metrics.csv")
    bank_lmdb_path = Path(oracle_bank_manifest["lmdb_path"])
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    replayable_df = pd.read_csv(source_root / "same_case_replayable_manifest.csv")
    target_case_ids = replayable_df["case_id"].astype(str).tolist()
    cases, dataset_assets = load_same_cases(cfg_path=cfg_path, cache_dir=cache_dir, target_case_ids=target_case_ids)
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

    candidate_batch_sizes = [int(v) for v in args.batch_candidates]
    exposure_candidates = [bs for bs in candidate_batch_sizes if math.ceil(sample_count / max(1, bs)) >= 8 and bs * 2 <= max(sample_count, 1)]
    if not exposure_candidates:
        exposure_candidates = [bs for bs in candidate_batch_sizes if bs * 2 <= max(sample_count, 1)]
    if not exposure_candidates:
        exposure_candidates = [max(1, min(sample_count, candidate_batch_sizes[-1]))]

    cfg = build_cfg_with_contrast(
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
    cfg = build_cfg_with_contrast(
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
    train_loss_by_epoch = {int(row["epoch"]): safe_float(row.get("train_loss")) for row in history_rows if row.get("epoch") is not None}
    checkpoints = collect_candidate_checkpoints(run_dir, int(args.epochs), int(args.periodic_checkpoint_every))
    checkpoints.append({"epoch": int(args.epochs), "path": Path(train_result["best_model_state_path"]), "label": "best_train_loss"})
    seen_labels = set()
    metric_rows = []
    for checkpoint in checkpoints:
        if checkpoint["label"] in seen_labels:
            continue
        seen_labels.add(checkpoint["label"])
        checkpoint_path = init_checkpoint if checkpoint["path"] is None else Path(checkpoint["path"])
        evaluated = evaluate_checkpoint_with_contrast(
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

    baseline_best = baseline_metrics.sort_values(["mrr_valid", "top1_hit", "top3_hit", "top5_hit", "true_source_rank_mean"], ascending=[False, False, False, False, True]).iloc[0].to_dict()
    baseline_final = baseline_metrics[baseline_metrics["label"] == "final"].iloc[0].to_dict()
    throughput = read_json(output_dir / "throughput_summary.json")
    write_json(
        output_dir / "summary.json",
        {
            "runner_version": RUNNER_VERSION,
            "panel_version": SAME_CASE_PANEL_VERSION,
            "contrast_family": "evidence_core_candidate_relative_contrast",
            "injection_path": "linear_adapter_residual_into_node_dims_0_6_keep_total_input_dim_21",
            "bank_lmdb_path": str(bank_lmdb_path),
            "baseline_best": baseline_best,
            "baseline_final": baseline_final,
            "chosen_batch_size": int(chosen_batch_size),
            "batches_per_epoch": int(throughput["train_batches_per_epoch"]),
            "epochs": int(throughput["num_epochs"]),
            "total_optimizer_steps": int(throughput["total_train_steps"]),
            "seed": 45,
        },
    )


if __name__ == "__main__":
    main()
