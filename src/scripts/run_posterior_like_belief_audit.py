from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.belief_updaters.evidence_posterior_like import EvidencePosteriorLikeBelief
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.diagnostics.run_slot1_counterfactual_leverage_audit import (
    build_namespace_from_control_args,
    load_control_bundle,
)
from src.scripts.run_reasoner_oracle_contrast_injection import build_cfg_with_contrast, load_model_with_contrast
from src.scripts.run_reasoner_oracle_exposure_audit import load_action_plan
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    TempGraph,
    build_state_input,
    load_same_cases,
    make_rollout_state,
    move_payload,
    read_json,
    translate_global_ids,
)


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact136_h3_formal_v1"
DEFAULT_CONTRAST_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_oracle_contrast_injection" / "20260407_exact136_oracle_contrast_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "posterior_like_belief_audit" / "20260407_exact136_posterior_like_v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
RUNNER_VERSION = "posterior_like_belief_audit_v1"
PANEL_VERSION = "exact136_train_only_posterior_like_belief_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posterior-like belief updater audit on exact136 oracle train cases.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda-q", type=float, default=0.75)
    parser.add_argument("--lambda-reasoner", type=float, default=1.0)
    parser.add_argument("--lambda-contrast", type=float, default=0.5)
    parser.add_argument("--lambda-contradiction", type=float, default=0.25)
    parser.add_argument("--support-plausible-delta", type=float, default=0.25)
    parser.add_argument("--not-ruled-out-threshold", type=float, default=0.5)
    parser.add_argument("--confusion-logit-delta", type=float, default=1.0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_runtime_context(source_root: Path, cache_dir: Path) -> Dict[str, Any]:
    oracle_arm_manifest = read_json(source_root / "arm_b_task_defined_oracle" / "run_manifest.json")
    bridge_package_dir = Path(oracle_arm_manifest["bridge_package_dir"])
    init_checkpoint = Path(oracle_arm_manifest["init_checkpoint"])
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    replayable_df = pd.read_csv(source_root / "same_case_replayable_manifest.csv")
    target_case_ids = replayable_df["case_id"].astype(str).tolist()
    cases, dataset_assets = load_same_cases(cfg_path=cfg_path, cache_dir=cache_dir, target_case_ids=target_case_ids)
    action_plan = load_action_plan(source_root / "oracle_action_plan.csv")
    seed_meta = read_json(bridge_package_dir / "seed_metadata.json")
    control_bundle = load_control_bundle(Path(seed_meta["control_dir"]))
    nav_args = build_namespace_from_control_args(control_bundle["args"], "cpu")
    return {
        "bridge_package_dir": bridge_package_dir,
        "init_checkpoint": init_checkpoint,
        "cases": cases,
        "dataset_assets": dataset_assets,
        "action_plan": action_plan,
        "num_episodes": int(getattr(nav_args, "num_episodes")),
        "action_budget": int(getattr(nav_args, "action_budget")),
        "episode_duration_min": float(getattr(nav_args, "episode_duration_min")),
        "frontier_role_mode": str(getattr(nav_args, "frontier_role_mode")),
    }


def load_frozen_reasoner(runtime: Dict[str, Any], cache_dir: Path, device: torch.device):
    cfg = build_cfg_with_contrast(
        bridge_package_dir=runtime["bridge_package_dir"],
        init_checkpoint=runtime["init_checkpoint"],
        cache_dir=cache_dir,
        epochs=240,
        periodic_every=20,
        batch_size=64,
    )
    checkpoint = PROJECT_ROOT / "runs" / "clean_aligned_semidynamic_37116e45862b" / "checkpoints" / "checkpoint_epoch_240.pt"
    model = load_model_with_contrast(cfg, checkpoint, device)
    reasoner_module = getattr(model, "reasoner_module", model)
    for param in reasoner_module.parameters():
        param.requires_grad = False
    return cfg, checkpoint, reasoner_module


def raw_rank(scores: torch.Tensor, valid_mask: torch.Tensor, source_local: int | None) -> Dict[str, Any]:
    valid_mask = valid_mask.view(-1).bool().cpu()
    scores = scores.view(-1).float().cpu()
    if source_local is None or not bool(valid_mask[int(source_local)].item()):
        return {
            "rank": None,
            "top1_hit": 0.0,
            "top3_hit": 0.0,
            "top5_hit": 0.0,
            "mrr": 0.0,
            "top1_margin": float("nan"),
        }
    safe_scores = scores.clone()
    safe_scores[~valid_mask] = -float("inf")
    order = torch.argsort(safe_scores, descending=True)
    valid_order = order[torch.isfinite(safe_scores[order])]
    positions = (valid_order == int(source_local)).nonzero(as_tuple=True)[0]
    if positions.numel() <= 0:
        return {
            "rank": None,
            "top1_hit": 0.0,
            "top3_hit": 0.0,
            "top5_hit": 0.0,
            "mrr": 0.0,
            "top1_margin": float("nan"),
        }
    rank = int(positions.min().item()) + 1
    top2 = torch.topk(safe_scores[valid_mask], k=min(2, int(valid_mask.sum().item()))).values
    margin = float(top2[0].item() - top2[1].item()) if top2.numel() >= 2 else 0.0
    return {
        "rank": int(rank),
        "top1_hit": float(rank <= 1),
        "top3_hit": float(rank <= 3),
        "top5_hit": float(rank <= 5),
        "mrr": float(1.0 / float(rank)),
        "top1_margin": float(margin),
    }


def summarise_metric_rows(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, float]:
    valid = [row for row in rows if row.get(f"{prefix}_rank") is not None]
    if not valid:
        return {}
    return {
        f"{prefix}_valid_case_count": float(len(valid)),
        f"{prefix}_top1_hit": float(sum(float(row[f"{prefix}_top1_hit"]) for row in valid) / len(valid)),
        f"{prefix}_top3_hit": float(sum(float(row[f"{prefix}_top3_hit"]) for row in valid) / len(valid)),
        f"{prefix}_top5_hit": float(sum(float(row[f"{prefix}_top5_hit"]) for row in valid) / len(valid)),
        f"{prefix}_mrr": float(sum(float(row[f"{prefix}_mrr"]) for row in valid) / len(valid)),
        f"{prefix}_true_rank_mean": float(sum(float(row[f"{prefix}_rank"]) for row in valid) / len(valid)),
        f"{prefix}_median_rank": float(pd.Series([float(row[f"{prefix}_rank"]) for row in valid]).median()),
        f"{prefix}_top1_margin_mean": float(sum(float(row[f"{prefix}_top1_margin"]) for row in valid) / len(valid)),
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    contrast_root = Path(args.contrast_root)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    reasoner_cfg, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)
    updater = EvidencePosteriorLikeBelief(
        temperature=float(args.temperature),
        lambda_q=float(args.lambda_q),
        lambda_reasoner=float(args.lambda_reasoner),
        lambda_contrast=float(args.lambda_contrast),
        lambda_contradiction=float(args.lambda_contradiction),
        support_plausible_delta=float(args.support_plausible_delta),
        not_ruled_out_threshold=float(args.not_ruled_out_threshold),
        confusion_logit_delta=float(args.confusion_logit_delta),
    )

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    step_rows: List[Dict[str, Any]] = []
    case_final_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    for case in runtime["cases"]:
        rollout = PracticalRollout(
            event_data=deepcopy(case.data),
            global_edge_index=runtime["dataset_assets"]["global_edge_index"],
            stt_dynamic_series=runtime["dataset_assets"]["stt_dynamic_series"],
            num_global_nodes=int(runtime["dataset_assets"]["num_global_nodes"]),
            num_episodes=int(runtime["num_episodes"]),
            samples_per_episode=int(runtime["action_budget"]),
            episode_duration_min=float(runtime["episode_duration_min"]),
        )
        history = ObservationWitnessHistory()
        belief_state = updater.init_state(batch_size=1, num_nodes=int(rollout.num_nodes), device=torch.device("cpu"))
        previous_belief_ctx = None
        final_case_row = None

        for step in runtime["action_plan"].get(case.case_id, []):
            state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=runtime["num_episodes"],
                action_budget=runtime["action_budget"],
                frontier_role_mode=runtime["frontier_role_mode"],
            )
            source_local = resolve_source_local_idx(rollout)

            graph = TempGraph(state["edge_index"], int(state["valid_mask"].numel()), device)
            state_input = move_payload(build_state_input(state), device)
            physics_ctx = move_payload(state["phys_ctx"].__dict__, device)
            with torch.no_grad():
                out = reasoner_module(state_input, graph, physics_ctx=physics_ctx)
            reasoner_logits = out["logits"].detach().float().view(-1).cpu()

            payload = build_clean_aligned_feature_payload(
                build_state_input(state),
                batch_index=torch.zeros(int(state["valid_mask"].numel()), dtype=torch.long),
                edge_index=state["edge_index"].view(2, -1).long(),
                physics_ctx=state["phys_ctx"].__dict__,
                frontier_mode="unresolved_without_pair",
            )

            belief_state, belief_ctx = updater.step(
                belief_state,
                {
                    "step_idx": int(step.round_index),
                    "t_sim": torch.tensor([float(rollout.current_time_min)], dtype=torch.float32),
                    "valid_mask": state["valid_mask"].view(-1).bool(),
                    "evidence_state": state["evidence_state"],
                    "constraint_state": state["constraint_state"],
                    "reasoner_logits": reasoner_logits,
                    "node_features": payload["node_features"],
                    "graph_features": payload["graph_features_by_graph"].view(-1),
                    "batch": torch.zeros(int(state["valid_mask"].numel()), dtype=torch.long),
                },
            )

            support_rank = raw_rank(state["support_score"], state["valid_mask"], source_local)
            logits_rank = raw_rank(reasoner_logits, state["valid_mask"], source_local)
            belief_rank = raw_rank(belief_ctx["belief"], state["valid_mask"], source_local)

            hardest_confuser_local = None
            hardest_confuser_mass = None
            if source_local is not None:
                for idx in belief_ctx["ordered_candidates"].tolist():
                    if int(idx) != int(source_local):
                        hardest_confuser_local = int(idx)
                        hardest_confuser_mass = float(belief_ctx["belief"][int(idx)].item())
                        break

            row = {
                "case_id": case.case_id,
                "scenario_id": case.scenario_id,
                "part_id": case.part_id,
                "episode_index": int(step.round_index) + 1,
                "candidate_count": int(belief_ctx["candidate_mask"].float().sum().item()),
                "support_score_rank": support_rank["rank"],
                "support_score_top1_hit": support_rank["top1_hit"],
                "support_score_top3_hit": support_rank["top3_hit"],
                "support_score_top5_hit": support_rank["top5_hit"],
                "support_score_mrr": support_rank["mrr"],
                "support_score_top1_margin": support_rank["top1_margin"],
                "reasoner_logit_rank": logits_rank["rank"],
                "reasoner_logit_top1_hit": logits_rank["top1_hit"],
                "reasoner_logit_top3_hit": logits_rank["top3_hit"],
                "reasoner_logit_top5_hit": logits_rank["top5_hit"],
                "reasoner_logit_mrr": logits_rank["mrr"],
                "reasoner_logit_top1_margin": logits_rank["top1_margin"],
                "belief_rank": belief_rank["rank"],
                "belief_top1_hit": belief_rank["top1_hit"],
                "belief_top3_hit": belief_rank["top3_hit"],
                "belief_top5_hit": belief_rank["top5_hit"],
                "belief_mrr": belief_rank["mrr"],
                "belief_top1_margin": belief_rank["top1_margin"],
                "belief_entropy": float(belief_ctx["entropy"].item()),
                "belief_top1_mass": float(belief_ctx["top1_mass"].item()),
                "belief_top3_mass": float(belief_ctx["top3_mass"].item()),
                "belief_top5_mass": float(belief_ctx["top5_mass"].item()),
                "belief_cluster_mass": float(belief_ctx["cluster_mass"].item()),
                "belief_cluster_count": int(belief_ctx["cluster_count"].item()),
                "hardest_confuser_local": hardest_confuser_local,
                "hardest_confuser_mass": hardest_confuser_mass,
            }

            if source_local is not None and bool(state["valid_mask"][int(source_local)].item()):
                row["belief_true_mass"] = float(belief_ctx["belief"][int(source_local)].item())
                if previous_belief_ctx is not None:
                    row["delta_entropy"] = float(previous_belief_ctx["entropy"].item() - belief_ctx["entropy"].item())
                    row["delta_true_mass"] = float(
                        belief_ctx["belief"][int(source_local)].item() - previous_belief_ctx["belief"][int(source_local)].item()
                    )
                    row["delta_cluster_count"] = float(previous_belief_ctx["cluster_count"].item() - belief_ctx["cluster_count"].item())
                else:
                    row["delta_entropy"] = None
                    row["delta_true_mass"] = None
                    row["delta_cluster_count"] = None
            else:
                row["belief_true_mass"] = None
                row["delta_entropy"] = None
                row["delta_true_mass"] = None
                row["delta_cluster_count"] = None

            step_rows.append(row)
            previous_belief_ctx = {
                "belief": belief_ctx["belief"].detach().cpu(),
                "entropy": belief_ctx["entropy"].detach().cpu(),
                "cluster_count": belief_ctx["cluster_count"].detach().cpu(),
            }
            final_case_row = row

            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"oracle_slot_{i}" for i in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])

        if final_case_row is not None:
            case_final_rows.append(final_case_row)

    step_df = pd.DataFrame(step_rows)
    case_df = pd.DataFrame(case_final_rows)
    step_df.to_csv(output_dir / "belief_step_rows.csv", index=False)
    case_df.to_csv(output_dir / "belief_case_rows.csv", index=False)

    summary: Dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "posterior_like_definition": {
            "candidate_support": "candidate_mask from build_candidate_semantics(valid_mask, not_ruled_out_gate, confirmed_non_source_mask)",
            "energy": "lambda_q * z(q_score) + lambda_reasoner * z(reasoner_logits) + lambda_contrast * z(contrast_signal) - lambda_contradiction * z(contradiction_score)",
            "distribution": "masked_softmax(energy / temperature) over candidate_mask",
            "update_mode": "recompute from current unified state X_t after each new observation, while tracking previous belief only for delta diagnostics",
            "temperature": float(args.temperature),
            "lambda_q": float(args.lambda_q),
            "lambda_reasoner": float(args.lambda_reasoner),
            "lambda_contrast": float(args.lambda_contrast),
            "lambda_contradiction": float(args.lambda_contradiction),
            "note": "Posterior-like decision belief, not a strict analytical Bayesian posterior.",
        },
        "reasoner_asset": {
            "contrast_root": str(contrast_root),
            "frozen_checkpoint": str(frozen_checkpoint),
        },
        "belief_interface_outputs": [
            "belief",
            "entropy",
            "top1_mass",
            "top3_mass",
            "top5_mass",
            "cluster_mass",
            "cluster_count",
            "ordered_candidates",
        ],
        "step_count": int(len(step_df)),
        "case_count": int(len(case_df)),
    }

    for prefix in ["support_score", "reasoner_logit", "belief"]:
        summary.update(summarise_metric_rows(step_rows, prefix))

    summary["belief_entropy_mean"] = float(step_df["belief_entropy"].mean())
    summary["belief_entropy_std"] = float(step_df["belief_entropy"].std())
    summary["candidate_count_mean"] = float(step_df["candidate_count"].mean())
    summary["candidate_count_median"] = float(step_df["candidate_count"].median())
    summary["candidate_count_max"] = float(step_df["candidate_count"].max())
    summary["belief_top1_mass_mean"] = float(step_df["belief_top1_mass"].mean())
    summary["belief_top3_mass_mean"] = float(step_df["belief_top3_mass"].mean())
    summary["belief_top5_mass_mean"] = float(step_df["belief_top5_mass"].mean())
    summary["belief_cluster_mass_mean"] = float(step_df["belief_cluster_mass"].mean())
    summary["belief_cluster_count_mean"] = float(step_df["belief_cluster_count"].mean())
    summary["belief_cluster_count_median"] = float(step_df["belief_cluster_count"].median())
    valid_delta = step_df[step_df["delta_entropy"].notna()].copy()
    summary["delta_entropy_mean"] = float(valid_delta["delta_entropy"].mean()) if len(valid_delta) else None
    summary["delta_true_mass_mean"] = float(valid_delta["delta_true_mass"].mean()) if len(valid_delta) else None
    summary["delta_cluster_count_mean"] = float(valid_delta["delta_cluster_count"].mean()) if len(valid_delta) else None
    summary["delta_entropy_positive_rate"] = float((valid_delta["delta_entropy"] > 0).mean()) if len(valid_delta) else None
    summary["delta_true_mass_positive_rate"] = float((valid_delta["delta_true_mass"] > 0).mean()) if len(valid_delta) else None
    summary["delta_cluster_count_positive_rate"] = float((valid_delta["delta_cluster_count"] > 0).mean()) if len(valid_delta) else None
    if len(valid_delta):
        summary["all_three_update_signals_positive_rate"] = float(
            (
                (valid_delta["delta_entropy"] > 0)
                & (valid_delta["delta_true_mass"] > 0)
                & (valid_delta["delta_cluster_count"] > 0)
            ).mean()
        )
    else:
        summary["all_three_update_signals_positive_rate"] = None
    summary["belief_beats_support_rank_rate"] = float((step_df["belief_rank"] < step_df["support_score_rank"]).mean())
    summary["belief_matches_reasoner_rank_rate"] = float((step_df["belief_rank"] == step_df["reasoner_logit_rank"]).mean())
    summary["belief_better_than_reasoner_rank_rate"] = float((step_df["belief_rank"] < step_df["reasoner_logit_rank"]).mean())
    summary["belief_worse_than_reasoner_rank_rate"] = float((step_df["belief_rank"] > step_df["reasoner_logit_rank"]).mean())

    improved_examples = step_df[step_df["delta_true_mass"].notna()].sort_values("delta_true_mass", ascending=False).head(5)
    contraction_examples = step_df[step_df["delta_entropy"].notna()].sort_values("delta_entropy", ascending=False).head(5)
    improved_examples.to_csv(output_dir / "belief_true_mass_gain_examples.csv", index=False)
    contraction_examples.to_csv(output_dir / "belief_entropy_drop_examples.csv", index=False)
    summary["example_paths"] = {
        "belief_true_mass_gain_examples": str(output_dir / "belief_true_mass_gain_examples.csv"),
        "belief_entropy_drop_examples": str(output_dir / "belief_entropy_drop_examples.csv"),
    }

    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
