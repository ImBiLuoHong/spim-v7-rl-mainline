from __future__ import annotations

import argparse
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.modeling.navigators.clean_v1 import pick_topk_valid
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_fixed_posterior_action_value_audit import (
    DEFAULT_ACCEPTABILITY_ROOT,
    DEFAULT_CACHE_DIR,
    DEFAULT_CONTRAST_ROOT,
    DEFAULT_OUTPUT_DIR as _UNUSED_ACTION_VALUE_OUTPUT_DIR,
    DEFAULT_POLICY_ROOT,
    DEFAULT_SOURCE_ROOT,
    _build_action_pool,
    _compute_posterior_from_state,
    _load_calibrated_params,
    _load_summary_threshold,
)
from src.scripts.run_posterior_like_belief_audit import load_frozen_reasoner, load_runtime_context, write_json
from src.scripts.run_reasoner_same_case_stronger_source_overfit import make_rollout_state, translate_global_ids


DEFAULT_ACTION_VALUE_ROOT = PROJECT_ROOT / "artifacts" / "fixed_posterior_action_value_audit" / "20260407_exact136_action_value_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "fixed_posterior_deterministic_myopic_baseline" / "20260407_exact136_myopic_baseline_v1"
RUNNER_VERSION = "fixed_posterior_deterministic_myopic_baseline_v1"
PANEL_VERSION = "exact136_train_only_fixed_posterior_episode_compare_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic myopic closed-loop baseline under fixed posterior.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--policy-root", type=str, default=str(DEFAULT_POLICY_ROOT))
    parser.add_argument("--action-value-root", type=str, default=str(DEFAULT_ACTION_VALUE_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--action-pool-cap", type=int, default=12)
    parser.add_argument("--topk-each-source", type=int, default=4)
    parser.add_argument("--use-action-audit-case-subset", action="store_true", default=True)
    parser.add_argument("--progress-every-cases", type=int, default=10)
    return parser.parse_args()


def _load_action_target_scales(action_value_root: Path) -> Dict[str, float]:
    df = pd.read_csv(action_value_root / "action_value_rows.csv")
    return {
        "true_mass_std": float(max(df["next_delta_true_mass"].std(), 1e-9)),
        "entropy_std": float(max(df["next_delta_entropy"].std(), 1e-9)),
        "confusion_std": float(max(df["next_confusion_shrink"].std(), 1e-9)),
    }


def _load_case_subset(action_value_root: Path) -> List[str]:
    df = pd.read_csv(action_value_root / "selected_state_manifest.csv")
    return sorted(df["case_id"].astype(str).unique().tolist())


def _myopic_utility(post: Dict[str, Any], pre: Dict[str, Any], scales: Dict[str, float]) -> float:
    delta_true = 0.0
    if pre["source_mass"] is not None and post["source_mass"] is not None:
        delta_true = float(post["source_mass"] - pre["source_mass"])
    delta_entropy = float(pre["entropy"] - post["entropy"])
    delta_confusion = float(pre["mass_cover_size_ratio"] - post["mass_cover_size_ratio"])
    return float(
        delta_true / scales["true_mass_std"]
        + 0.5 * (delta_entropy / scales["entropy_std"])
        + 0.5 * (delta_confusion / scales["confusion_std"])
    )


def _simulate_one_step(
    *,
    case: Any,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    candidate: int,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    runtime: Dict[str, Any],
    reasoner_module,
    posterior_params: Dict[str, float],
    mass_cover_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    sim_rollout = deepcopy(rollout)
    sim_history = deepcopy(history)
    sim_rollout.step_with_actions([int(candidate)], sample_types=["oracle_ref_slot"])
    if sim_rollout.history_steps:
        sim_history.append_from_history_step(sim_rollout.history_steps[-1])
    post_state = make_rollout_state(
        case=case,
        rollout=sim_rollout,
        history=sim_history,
        env=env,
        topology=topology,
        num_episodes=runtime["num_episodes"],
        action_budget=runtime["action_budget"],
        frontier_role_mode=runtime["frontier_role_mode"],
    )
    return _compute_posterior_from_state(
        state=post_state,
        reasoner_module=reasoner_module,
        params=posterior_params,
        mass_cover_threshold=mass_cover_threshold,
        device=device,
    )


def _select_action(
    *,
    policy_name: str,
    pre: Dict[str, Any],
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    case: Any,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    runtime: Dict[str, Any],
    reasoner_module,
    posterior_params: Dict[str, float],
    mass_cover_threshold: float,
    action_pool_cap: int,
    topk_each_source: int,
    scales: Dict[str, float],
    rng: random.Random,
    device: torch.device,
) -> Dict[str, Any]:
    if policy_name == "top_support_legacy":
        action = int(pick_topk_valid(pre["support_score"], pre["candidate_mask"], 1)[0])
        return {"action": action}
    if policy_name == "posterior_greedy":
        action = int(pre["top_order"][0])
        return {"action": action}
    if policy_name == "bounded_one_step_oracle":
        pool = _build_action_pool(
            posterior=pre,
            topk_each=int(topk_each_source),
            cap=int(action_pool_cap),
            rng=rng,
        )
        best = None
        action_rows = []
        for action in pool:
            post = _simulate_one_step(
                case=case,
                rollout=rollout,
                history=history,
                candidate=int(action),
                env=env,
                topology=topology,
                runtime=runtime,
                reasoner_module=reasoner_module,
                posterior_params=posterior_params,
                mass_cover_threshold=mass_cover_threshold,
                device=device,
            )
            utility = _myopic_utility(post, pre, scales)
            row = {
                "action_local": int(action),
                "utility": float(utility),
                "in_mass_cover_summary": float(int(action) in pre["mass_cover_set"]),
                "posterior_mass": float(pre["probs"][int(action)].item()),
            }
            action_rows.append(row)
            if best is None or float(utility) > float(best["utility"]) or (
                float(utility) == float(best["utility"]) and int(action) < int(best["action_local"])
            ):
                best = row
        assert best is not None
        return {
            "action": int(best["action_local"]),
            "pool": pool,
            "pool_rows": action_rows,
            "best_utility": float(best["utility"]),
        }
    raise ValueError(policy_name)


def _run_case_policy(
    *,
    case: Any,
    policy_name: str,
    runtime: Dict[str, Any],
    reasoner_module,
    posterior_params: Dict[str, float],
    mass_cover_threshold: float,
    action_pool_cap: int,
    topk_each_source: int,
    scales: Dict[str, float],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
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
    rng = random.Random((hash(case.case_id) ^ hash(policy_name) ^ int(seed)) & 0xFFFFFFFF)
    step_rows: List[Dict[str, Any]] = []
    oracle_pool_rows: List[Dict[str, Any]] = []

    for episode_idx in range(int(runtime["num_episodes"])):
        pre_state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=runtime["num_episodes"],
            action_budget=runtime["action_budget"],
            frontier_role_mode=runtime["frontier_role_mode"],
        )
        if int(pre_state["valid_mask"].sum().item()) <= 0:
            break
        pre = _compute_posterior_from_state(
            state=pre_state,
            reasoner_module=reasoner_module,
            params=posterior_params,
            mass_cover_threshold=mass_cover_threshold,
            device=device,
        )
        choice = _select_action(
            policy_name=policy_name,
            pre=pre,
            rollout=rollout,
            history=history,
            case=case,
            env=env,
            topology=topology,
            runtime=runtime,
            reasoner_module=reasoner_module,
            posterior_params=posterior_params,
            mass_cover_threshold=mass_cover_threshold,
            action_pool_cap=action_pool_cap,
            topk_each_source=topk_each_source,
            scales=scales,
            rng=rng,
            device=device,
        )
        action = int(choice["action"])
        rollout.step_with_actions([action], sample_types=[f"{policy_name}_slot_0"])
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])
        post_state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=runtime["num_episodes"],
            action_budget=runtime["action_budget"],
            frontier_role_mode=runtime["frontier_role_mode"],
        )
        post = _compute_posterior_from_state(
            state=post_state,
            reasoner_module=reasoner_module,
            params=posterior_params,
            mass_cover_threshold=mass_cover_threshold,
            device=device,
        )
        delta_true = None
        if pre["source_mass"] is not None and post["source_mass"] is not None:
            delta_true = float(post["source_mass"] - pre["source_mass"])
        step_rows.append(
            {
                "case_id": case.case_id,
                "policy_name": policy_name,
                "episode_index": int(episode_idx + 1),
                "selected_action_local": action,
                "pre_entropy": float(pre["entropy"]),
                "post_entropy": float(post["entropy"]),
                "delta_entropy": float(pre["entropy"] - post["entropy"]),
                "pre_true_mass": pre["source_mass"],
                "post_true_mass": post["source_mass"],
                "delta_true_mass": delta_true,
                "pre_margin_true_vs_hard": pre["margin_true_vs_hard"],
                "post_margin_true_vs_hard": post["margin_true_vs_hard"],
                "delta_margin": (
                    float(post["margin_true_vs_hard"] - pre["margin_true_vs_hard"])
                    if pre["margin_true_vs_hard"] is not None and post["margin_true_vs_hard"] is not None
                    else None
                ),
                "pre_mass_cover_size_ratio": float(pre["mass_cover_size_ratio"]),
                "post_mass_cover_size_ratio": float(post["mass_cover_size_ratio"]),
                "delta_confusion_shrink": float(pre["mass_cover_size_ratio"] - post["mass_cover_size_ratio"]),
                "source_hit_after_step": float((post["source_rank"] or 10**9) <= 1),
                "source_rank_after_step": post["source_rank"],
                "budget_used": float(rollout.revealed_mask.sum().item()),
            }
        )
        if policy_name == "bounded_one_step_oracle":
            for row in choice.get("pool_rows", []):
                oracle_pool_rows.append(
                    {
                        "case_id": case.case_id,
                        "episode_index": int(episode_idx + 1),
                        **row,
                        "selected": float(int(row["action_local"]) == int(action)),
                    }
                )

    final_state = make_rollout_state(
        case=case,
        rollout=rollout,
        history=history,
        env=env,
        topology=topology,
        num_episodes=runtime["num_episodes"],
        action_budget=runtime["action_budget"],
        frontier_role_mode=runtime["frontier_role_mode"],
    )
    final = _compute_posterior_from_state(
        state=final_state,
        reasoner_module=reasoner_module,
        params=posterior_params,
        mass_cover_threshold=mass_cover_threshold,
        device=device,
    )
    hit_round = None
    for row in step_rows:
        if float(row["source_hit_after_step"]) > 0.5:
            hit_round = int(row["episode_index"])
            break
    return {
        "case_row": {
            "case_id": case.case_id,
            "policy_name": policy_name,
            "success_rate": float((final["source_rank"] or 10**9) <= 1),
            "final_top1_hit": float((final["source_rank"] or 10**9) <= 1),
            "final_top3_hit": float((final["source_rank"] or 10**9) <= 3),
            "final_top5_hit": float((final["source_rank"] or 10**9) <= 5),
            "final_mrr": float(1.0 / final["source_rank"]) if final["source_rank"] is not None else 0.0,
            "final_source_rank": final["source_rank"],
            "final_true_mass": final["source_mass"],
            "final_entropy": float(final["entropy"]),
            "final_mass_cover_size_ratio": float(final["mass_cover_size_ratio"]),
            "budget_used": float(rollout.revealed_mask.sum().item()),
            "hit_round": hit_round,
        },
        "step_rows": step_rows,
        "oracle_pool_rows": oracle_pool_rows,
    }


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    source_root = Path(args.source_root)
    contrast_root = Path(args.contrast_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)
    policy_root = Path(args.policy_root)
    action_value_root = Path(args.action_value_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    _, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)
    posterior_params = _load_calibrated_params(acceptability_root)
    mass_cover_threshold = _load_summary_threshold(policy_root)
    scales = _load_action_target_scales(action_value_root)
    case_subset = _load_case_subset(action_value_root) if bool(args.use_action_audit_case_subset) else None
    if case_subset is not None:
        keep = set(case_subset)
        runtime["cases"] = [case for case in runtime["cases"] if str(case.case_id) in keep]

    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    oracle_pool_rows: List[Dict[str, Any]] = []

    policies = ["top_support_legacy", "posterior_greedy", "bounded_one_step_oracle"]
    for policy_name in policies:
        for case_idx, case in enumerate(runtime["cases"], start=1):
            out = _run_case_policy(
                case=case,
                policy_name=policy_name,
                runtime=runtime,
                reasoner_module=reasoner_module,
                posterior_params=posterior_params,
                mass_cover_threshold=mass_cover_threshold,
                action_pool_cap=int(args.action_pool_cap),
                topk_each_source=int(args.topk_each_source),
                scales=scales,
                device=device,
                seed=int(args.seed),
            )
            case_rows.append(out["case_row"])
            step_rows.extend(out["step_rows"])
            oracle_pool_rows.extend(out["oracle_pool_rows"])
            if int(case_idx) % int(args.progress_every_cases) == 0 or int(case_idx) == int(len(runtime["cases"])):
                pd.DataFrame(case_rows).to_csv(output_dir / "episode_case_rows.partial.csv", index=False)
                pd.DataFrame(step_rows).to_csv(output_dir / "episode_step_rows.partial.csv", index=False)
                if len(oracle_pool_rows):
                    pd.DataFrame(oracle_pool_rows).to_csv(output_dir / "oracle_pool_rows.partial.csv", index=False)
                print(
                    f"[progress] policy={policy_name} case={case_idx}/{len(runtime['cases'])} "
                    f"partial_cases={len(case_rows)} partial_steps={len(step_rows)}",
                    flush=True,
                )

    case_df = pd.DataFrame(case_rows)
    step_df = pd.DataFrame(step_rows)
    oracle_pool_df = pd.DataFrame(oracle_pool_rows)
    case_df.to_csv(output_dir / "episode_case_rows.csv", index=False)
    step_df.to_csv(output_dir / "episode_step_rows.csv", index=False)
    if len(oracle_pool_df):
        oracle_pool_df.to_csv(output_dir / "oracle_pool_rows.csv", index=False)

    compare_rows = []
    for policy_name, sub in case_df.groupby("policy_name"):
        compare_rows.append(
            {
                "policy_name": policy_name,
                "case_count": int(len(sub)),
                "success_rate": float(sub["success_rate"].mean()),
                "final_top1_hit": float(sub["final_top1_hit"].mean()),
                "final_top3_hit": float(sub["final_top3_hit"].mean()),
                "final_top5_hit": float(sub["final_top5_hit"].mean()),
                "final_mrr": float(sub["final_mrr"].mean()),
                "hit_round_mean": float(sub["hit_round"].dropna().mean()) if sub["hit_round"].notna().any() else None,
                "budget_used_mean": float(sub["budget_used"].mean()),
                "final_true_mass_mean": float(sub["final_true_mass"].dropna().mean()) if sub["final_true_mass"].notna().any() else None,
                "final_entropy_mean": float(sub["final_entropy"].mean()),
                "final_mass_cover_size_ratio_mean": float(sub["final_mass_cover_size_ratio"].mean()),
            }
        )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(output_dir / "policy_compare.csv", index=False)

    round_rows = []
    for (policy_name, episode_index), sub in step_df.groupby(["policy_name", "episode_index"]):
        round_rows.append(
            {
                "policy_name": policy_name,
                "episode_index": int(episode_index),
                "delta_true_mass_mean": float(sub["delta_true_mass"].dropna().mean()) if sub["delta_true_mass"].notna().any() else None,
                "delta_entropy_mean": float(sub["delta_entropy"].mean()),
                "delta_confusion_shrink_mean": float(sub["delta_confusion_shrink"].mean()),
                "source_hit_after_step_rate": float(sub["source_hit_after_step"].mean()),
            }
        )
    round_df = pd.DataFrame(round_rows)
    round_df.to_csv(output_dir / "roundwise_compare.csv", index=False)

    oracle_pool_summary = None
    if len(oracle_pool_df):
        oracle_pool_summary = {
            "pool_size_mean": float(oracle_pool_df.groupby(["case_id", "episode_index"]).size().mean()),
            "selected_best_utility_mean": float(oracle_pool_df[oracle_pool_df["selected"] > 0.5]["utility"].mean()),
            "posterior_greedy_coverage_note": "posterior_greedy action is always in pool because posterior_topk is one source of the bounded pool",
        }

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "fixed_posterior_type": "lightly_calibrated_fused_posterior",
        "fixed_posterior_params": posterior_params,
        "evaluated_case_count": int(len(runtime["cases"])),
        "evaluated_case_subset_source": (
            str(action_value_root / "selected_state_manifest.csv") if case_subset is not None else "full_exact136"
        ),
        "policy_definitions": {
            "top_support_legacy": "select highest support_score valid unsampled node; deterministic tie-break by pick_topk_valid / score order",
            "posterior_greedy": "select highest posterior mass valid unsampled node; deterministic tie-break by posterior order then local id",
            "bounded_one_step_oracle": "within bounded action pool, choose action maximizing exact one-step myopic_utility_target",
        },
        "oracle_pool_definition": {
            "cap": int(args.action_pool_cap),
            "topk_each_source": int(args.topk_each_source),
            "sources": [
                "top_support_legacy top-k",
                "q_score top-k",
                "posterior top-k",
                "posterior mass-cover members",
                "random diversity",
            ],
            "summary": oracle_pool_summary,
        },
        "best_policy": compare_df.sort_values(["final_mrr", "final_top1_hit"], ascending=[False, False]).iloc[0].to_dict(),
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
