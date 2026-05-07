from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_reasoner_same_case_stronger_source_overfit import make_rollout_state
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_PRECHECK_ROOT,
    DEFAULT_SOURCE_ROOT,
    GLOBAL_FEATURE_NAMES,
    LOCAL_FEATURE_NAMES_BASE,
    LOCAL_FEATURE_NAMES_SURROGATE,
    PaperLikeHSRState,
    SpimNativePolicy,
    _belief_metrics,
    _extract_trigger_global,
    _pick_topk_unsampled,
    _posterior_topk_unsampled,
    auto_select_teacher,
    build_spim_native_state,
    compute_teacher_belief,
    get_device,
    load_train_full_cases,
    read_json,
    seed_everything,
)
from src.scripts.run_posterior_like_belief_audit import load_runtime_context, write_json
from src.scripts.run_spim_policy_eval_strict import build_runtime_strict


RUNNER_VERSION = "spim_teacher_regret_reward_alignment_audit_v1"
PANEL_VERSION = "cross_split_b30_teacher_regret_reward_alignment_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-regret and reward-alignment audit on SPIM mainline without new RL training.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--precheck-root", type=str, default=str(DEFAULT_PRECHECK_ROOT))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--teacher-family", type=str, default="auto", choices=["auto", "hsr_soft_scenario_posterior_v3", "hsr_paper_topk_ema_v1"])
    parser.add_argument("--bc-checkpoint", type=str, required=True)
    parser.add_argument("--include-surrogate-features", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--actions-per-round", type=int, default=3)
    parser.add_argument("--eval-b60", action="store_true")
    parser.add_argument("--runtime-split", type=str, default="exact136", choices=["exact136", "train", "val", "test"])
    parser.add_argument("--train-max-cases", type=int, default=0)
    parser.add_argument("--train-cache-version", type=str, default="")
    parser.add_argument("--case-limit", type=int, default=0)

    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)

    parser.add_argument("--top-source-k", type=int, default=8)

    parser.add_argument("--slot3-topk", type=int, default=6)
    parser.add_argument("--state-limit", type=int, default=0, help="Optional cap for smoke runs; 0 means full.")
    parser.add_argument(
        "--audit-max-per-policy-round",
        type=int,
        default=12,
        help="Bounded audit budget: max audited states per (policy_source, round). 0 disables this bound.",
    )
    parser.add_argument("--shuffle-cases", action="store_true", help="Shuffle case order before bounded auditing.")

    parser.add_argument("--bundle-topk", type=int, default=6)
    parser.add_argument("--bundle-max-states", type=int, default=96)
    parser.add_argument("--bundle-max-permutations", type=int, default=120)
    parser.add_argument("--bundle-min-slot3-regret", type=float, default=0.02)

    parser.add_argument("--ambiguity-top1-top2-max", type=float, default=0.08)
    parser.add_argument("--ambiguity-min-candidate-count", type=int, default=8)
    parser.add_argument("--corrective-candidate-topk", type=int, default=12)
    return parser.parse_args()


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    xs = pd.Series(list(x), dtype=float)
    ys = pd.Series(list(y), dtype=float)
    v = xs.corr(ys, method="spearman")
    return None if pd.isna(v) else float(v)


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _state_key(policy_source: str, case_id: str, episode_idx: int) -> str:
    return f"{policy_source}::{case_id}::ep{int(episode_idx)}"


def _load_bc_model(checkpoint_path: Path, hidden_dim: int, include_surrogate_features: bool, device: torch.device) -> SpimNativePolicy:
    local_feature_names = list(LOCAL_FEATURE_NAMES_BASE)
    if bool(include_surrogate_features):
        local_feature_names.extend(LOCAL_FEATURE_NAMES_SURROGATE)
    model = SpimNativePolicy(
        global_dim=len(GLOBAL_FEATURE_NAMES),
        local_dim=len(local_feature_names),
        hidden_dim=int(hidden_dim),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _make_rollout(runtime: Dict[str, Any], case: Any) -> PracticalRollout:
    return PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=runtime["dataset_assets"]["global_edge_index"],
        stt_dynamic_series=runtime["dataset_assets"]["stt_dynamic_series"],
        num_global_nodes=int(runtime["dataset_assets"]["num_global_nodes"]),
        num_episodes=int(runtime["num_episodes"]),
        samples_per_episode=int(runtime["action_budget"]),
        episode_duration_min=float(runtime["episode_duration_min"]),
    )


def _available_local_ids(candidate_mask: torch.Tensor, rollout: PracticalRollout) -> List[int]:
    avail = candidate_mask.view(-1).bool().cpu() & (~rollout.revealed_mask.view(-1).bool().cpu())
    return [int(v) for v in torch.nonzero(avail, as_tuple=True)[0].tolist()]


def _action_proxy_features(spim_state: Dict[str, Any], action_local: int, belief: torch.Tensor) -> Dict[str, float]:
    lf = spim_state["local_features"]
    belief_vec = belief.view(-1).float().cpu()
    top1_mass = float(belief_vec.max().item()) if belief_vec.numel() else 0.0
    action_mass = float(belief_vec[int(action_local)].item())
    gap = float(top1_mass - action_mass)
    # local_features layout in build_spim_native_state:
    # [belief, rank_percentile, expected_positive, disagreement, dist_trigger, dist_pos, dist_neg, available, sampled, ...]
    rank_percentile = float(lf[int(action_local), 1].item())
    expected_positive = float(lf[int(action_local), 2].item())
    disagreement = float(lf[int(action_local), 3].item())
    cover_shrink_proxy = float(expected_positive * max(1.0 - action_mass, 0.0))
    entropy_drop_proxy = float(disagreement * max(1.0 - rank_percentile, 0.0))
    return {
        "proxy_action_mass": action_mass,
        "proxy_action_gap_to_top1": gap,
        "proxy_rank_percentile": rank_percentile,
        "proxy_expected_positive": expected_positive,
        "proxy_disagreement": disagreement,
        "proxy_cover_shrink": cover_shrink_proxy,
        "proxy_entropy_drop": entropy_drop_proxy,
    }


def _compute_step_belief(
    *,
    family: str,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    case: Any,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    onset_grid: Sequence[float],
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
    source_local: Optional[int],
    gate: DynamicReachabilityRuleModule,
    top_source_k: int,
    include_surrogate_features: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    state = make_rollout_state(
        case=case,
        rollout=rollout,
        history=history,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        num_episodes=int(runtime["num_episodes"]),
        action_budget=int(runtime["action_budget"]),
        frontier_role_mode=str(runtime["frontier_role_mode"]),
    )
    belief_ctx = compute_teacher_belief(
        family=family,
        rollout=rollout,
        state=state,
        history=history,
        trigger_global=trigger_global,
        paper_state=paper_state,
        onset_offsets_min=onset_grid,
        paper_like_alpha=float(paper_like_alpha),
        paper_like_topk_fraction=float(paper_like_topk_fraction),
        paper_like_time_tol_min=float(paper_like_time_tol_min),
        soft_scenario_beta=float(soft_scenario_beta),
    )
    spim_state = build_spim_native_state(
        rollout=rollout,
        state=state,
        history=history,
        belief_ctx=belief_ctx,
        trigger_global=trigger_global,
        gate=gate,
        num_rounds=int(runtime["num_episodes"]),
        action_budget=int(runtime["action_budget"]),
        episode_duration_min=float(runtime["episode_duration_min"]),
        top_source_k=int(top_source_k),
        include_surrogate_features=bool(include_surrogate_features),
        source_local=source_local,
    )
    metrics = _belief_metrics(
        belief_ctx["belief"],
        belief_ctx["candidate_mask"],
        source_local,
    )
    return state, belief_ctx, spim_state, metrics


def _reward_r0(step_selected_count: int, round_hit: bool) -> float:
    return float((-1.0 / 30.0) * float(step_selected_count) + (1.0 if bool(round_hit) else 0.0))


def _rollout_with_teacher_continuation(
    *,
    family: str,
    case: Any,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    paper_state: PaperLikeHSRState,
    trigger_global: Optional[int],
    source_local: Optional[int],
    onset_grid: Sequence[float],
    first_episode_idx: int,
    first_actions: Sequence[int],
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> Dict[str, Any]:
    total_reward = 0.0
    hit_round: Optional[int] = None
    hit_sample_index: Optional[int] = None
    budget_used = int(rollout.revealed_mask.sum().item())

    def _apply_actions(ep: int, actions: Sequence[int]) -> bool:
        nonlocal total_reward, hit_round, hit_sample_index, budget_used
        chosen = [int(v) for v in actions]
        if len(chosen) <= 0:
            return False
        round_hit = source_local is not None and int(source_local) in set(chosen)
        if round_hit and hit_round is None:
            hit_round = int(ep)
            source_slot = chosen.index(int(source_local)) + 1
            hit_sample_index = int((int(ep) - 1) * int(runtime["action_budget"]) + int(source_slot))
        total_reward += _reward_r0(step_selected_count=len(chosen), round_hit=bool(round_hit))
        rollout.step_with_actions(chosen, sample_types=[f"audit_slot_{i}" for i in range(len(chosen))])
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])
        budget_used += int(len(chosen))
        return bool(round_hit)

    if _apply_actions(int(first_episode_idx), list(first_actions)):
        return {
            "success": 1.0,
            "hit_round": int(hit_round),
            "hit_sample_index": int(hit_sample_index),
            "budget_used": int(budget_used),
            "return_r0": float(total_reward),
            "termination_reason": "source_hit",
        }

    for ep in range(int(first_episode_idx) + 1, int(runtime["num_episodes"]) + 1):
        state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=runtime["dataset_assets"]["topology"],
            num_episodes=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
            frontier_role_mode=str(runtime["frontier_role_mode"]),
        )
        if int(state["valid_mask"].sum().item()) <= 0:
            return {
                "success": 0.0,
                "hit_round": None,
                "hit_sample_index": None,
                "budget_used": int(budget_used),
                "return_r0": float(total_reward),
                "termination_reason": "no_valid_nodes",
            }
        belief_ctx = compute_teacher_belief(
            family=family,
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=onset_grid,
            paper_like_alpha=float(paper_like_alpha),
            paper_like_topk_fraction=float(paper_like_topk_fraction),
            paper_like_time_tol_min=float(paper_like_time_tol_min),
            soft_scenario_beta=float(soft_scenario_beta),
        )
        teacher_actions = _pick_topk_unsampled(
            belief_ctx["belief"],
            belief_ctx["candidate_mask"],
            rollout,
            int(runtime["action_budget"]),
        )
        teacher_actions = [int(v) for v in teacher_actions]
        if not teacher_actions:
            return {
                "success": 0.0,
                "hit_round": None,
                "hit_sample_index": None,
                "budget_used": int(budget_used),
                "return_r0": float(total_reward),
                "termination_reason": "teacher_no_action",
            }
        if _apply_actions(ep, teacher_actions):
            return {
                "success": 1.0,
                "hit_round": int(hit_round),
                "hit_sample_index": int(hit_sample_index),
                "budget_used": int(budget_used),
                "return_r0": float(total_reward),
                "termination_reason": "source_hit",
            }

    return {
        "success": 0.0,
        "hit_round": None,
        "hit_sample_index": None,
        "budget_used": int(budget_used),
        "return_r0": float(total_reward),
        "termination_reason": "budget_exhausted",
    }


def _evaluate_slot3_candidates(
    *,
    family: str,
    case: Any,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    trigger_global: Optional[int],
    source_local: Optional[int],
    onset_grid: Sequence[float],
    episode_idx: int,
    teacher_actions: Sequence[int],
    belief_ctx: Dict[str, Any],
    spim_state: Dict[str, Any],
    pre_metrics: Dict[str, Any],
    pre_rollout: PracticalRollout,
    pre_history: ObservationWitnessHistory,
    pre_paper_state: PaperLikeHSRState,
    slot3_topk: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(teacher_actions) < 3:
        return [], {}

    blocked = {int(teacher_actions[0]), int(teacher_actions[1])}
    slot3_teacher = int(teacher_actions[2])

    pool = _posterior_topk_unsampled(
        belief=belief_ctx["belief"],
        candidate_mask=belief_ctx["candidate_mask"],
        rollout=pre_rollout,
        topk=max(int(slot3_topk), 3),
    )
    slot3_candidates = [int(v) for v in pool if int(v) not in blocked]
    if slot3_teacher not in slot3_candidates:
        slot3_candidates.append(slot3_teacher)
    # deterministic, keep order by posterior list then teacher fallback
    seen = set()
    ordered_candidates: List[int] = []
    for v in slot3_candidates:
        if int(v) in seen:
            continue
        seen.add(int(v))
        ordered_candidates.append(int(v))
    ordered_candidates = ordered_candidates[: max(int(slot3_topk), 1)]
    if slot3_teacher not in ordered_candidates:
        ordered_candidates = [slot3_teacher] + ordered_candidates[: max(int(slot3_topk) - 1, 0)]

    rows: List[Dict[str, Any]] = []
    baseline_row: Optional[Dict[str, Any]] = None

    for cand_slot3 in ordered_candidates:
        cf_rollout = deepcopy(pre_rollout)
        cf_history = deepcopy(pre_history)
        cf_paper = deepcopy(pre_paper_state)

        action_set = [int(teacher_actions[0]), int(teacher_actions[1]), int(cand_slot3)]

        # one-step post metrics for proxy alignment
        one_step = _rollout_with_teacher_continuation(
            family=family,
            case=case,
            runtime=runtime,
            env=env,
            rollout=cf_rollout,
            history=cf_history,
            paper_state=cf_paper,
            trigger_global=trigger_global,
            source_local=source_local,
            onset_grid=onset_grid,
            first_episode_idx=int(episode_idx),
            first_actions=action_set,
            paper_like_alpha=0.55,
            paper_like_topk_fraction=0.12,
            paper_like_time_tol_min=30.0,
            soft_scenario_beta=2.0,
        )

        # recompute immediate post-step belief for real one-step entropy/cover deltas
        post_state = make_rollout_state(
            case=case,
            rollout=cf_rollout,
            history=cf_history,
            env=env,
            topology=runtime["dataset_assets"]["topology"],
            num_episodes=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
            frontier_role_mode=str(runtime["frontier_role_mode"]),
        )
        post_belief_ctx = compute_teacher_belief(
            family=family,
            rollout=cf_rollout,
            state=post_state,
            history=cf_history,
            trigger_global=trigger_global,
            paper_state=cf_paper,
            onset_offsets_min=onset_grid,
            paper_like_alpha=0.55,
            paper_like_topk_fraction=0.12,
            paper_like_time_tol_min=30.0,
            soft_scenario_beta=2.0,
        )
        post_metrics = _belief_metrics(post_belief_ctx["belief"], post_belief_ctx["candidate_mask"], source_local)

        proxy = _action_proxy_features(spim_state, int(cand_slot3), belief_ctx["belief"])
        row = {
            "candidate_slot3_local": int(cand_slot3),
            "is_teacher_slot3": float(int(cand_slot3) == int(slot3_teacher)),
            "success": float(one_step["success"]),
            "hit_round": one_step["hit_round"],
            "hit_sample_index": one_step["hit_sample_index"],
            "budget_used": int(one_step["budget_used"]),
            "return_r0": float(one_step["return_r0"]),
            "termination_reason": str(one_step["termination_reason"]),
            "real_next_entropy_drop": float(pre_metrics["entropy"] - post_metrics["entropy"]),
            "real_next_cover_shrink": float(pre_metrics["mass_cover_size_ratio"] - post_metrics["mass_cover_size_ratio"]),
            "real_next_top1_mass_gain": float(post_metrics["top1_mass"] - pre_metrics["top1_mass"]),
            **proxy,
        }
        rows.append(row)
        if int(cand_slot3) == int(slot3_teacher):
            baseline_row = row

    if baseline_row is None and rows:
        baseline_row = rows[0]

    if baseline_row is None:
        return rows, {}

    base_ret = float(baseline_row["return_r0"])
    base_sr = float(baseline_row["success"])
    for row in rows:
        row["delta_return_vs_teacher_slot3"] = float(row["return_r0"] - base_ret)
        row["delta_success_vs_teacher_slot3"] = float(row["success"] - base_sr)

    return rows, baseline_row


def _evaluate_bundle_regret(
    *,
    family: str,
    case: Any,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    trigger_global: Optional[int],
    source_local: Optional[int],
    onset_grid: Sequence[float],
    episode_idx: int,
    teacher_actions: Sequence[int],
    belief_ctx: Dict[str, Any],
    spim_state: Dict[str, Any],
    pre_rollout: PracticalRollout,
    pre_history: ObservationWitnessHistory,
    pre_paper_state: PaperLikeHSRState,
    bundle_topk: int,
    bundle_max_permutations: int,
) -> List[Dict[str, Any]]:
    if len(teacher_actions) < 3:
        return []

    pool = _posterior_topk_unsampled(
        belief=belief_ctx["belief"],
        candidate_mask=belief_ctx["candidate_mask"],
        rollout=pre_rollout,
        topk=max(int(bundle_topk), 3),
    )
    pool = [int(v) for v in pool]
    for a in list(teacher_actions[:3]):
        if int(a) not in pool:
            pool.append(int(a))
    pool = pool[: max(int(bundle_topk), 3)]

    perms = list(itertools.permutations(pool, 3))
    if int(bundle_max_permutations) > 0:
        perms = perms[: int(bundle_max_permutations)]

    rows: List[Dict[str, Any]] = []
    for perm in perms:
        cf_rollout = deepcopy(pre_rollout)
        cf_history = deepcopy(pre_history)
        cf_paper = deepcopy(pre_paper_state)

        out = _rollout_with_teacher_continuation(
            family=family,
            case=case,
            runtime=runtime,
            env=env,
            rollout=cf_rollout,
            history=cf_history,
            paper_state=cf_paper,
            trigger_global=trigger_global,
            source_local=source_local,
            onset_grid=onset_grid,
            first_episode_idx=int(episode_idx),
            first_actions=list(perm),
            paper_like_alpha=0.55,
            paper_like_topk_fraction=0.12,
            paper_like_time_tol_min=30.0,
            soft_scenario_beta=2.0,
        )

        p0 = _action_proxy_features(spim_state, int(perm[0]), belief_ctx["belief"])
        p1 = _action_proxy_features(spim_state, int(perm[1]), belief_ctx["belief"])
        p2 = _action_proxy_features(spim_state, int(perm[2]), belief_ctx["belief"])

        rows.append(
            {
                "bundle_actions": json.dumps([int(perm[0]), int(perm[1]), int(perm[2])]),
                "is_teacher_bundle": float(list(perm) == list(teacher_actions[:3])),
                "success": float(out["success"]),
                "return_r0": float(out["return_r0"]),
                "proxy_entropy_drop_sum": float(p0["proxy_entropy_drop"] + p1["proxy_entropy_drop"] + p2["proxy_entropy_drop"]),
                "proxy_cover_shrink_sum": float(p0["proxy_cover_shrink"] + p1["proxy_cover_shrink"] + p2["proxy_cover_shrink"]),
                "proxy_disagreement_sum": float(p0["proxy_disagreement"] + p1["proxy_disagreement"] + p2["proxy_disagreement"]),
                "proxy_mass_sum": float(p0["proxy_action_mass"] + p1["proxy_action_mass"] + p2["proxy_action_mass"]),
                "state_top1_top2_margin": float(spim_state["diagnostics"]["top1_top2_margin"]),
                "state_entropy": float(spim_state["diagnostics"]["posterior_entropy"]),
            }
        )

    if not rows:
        return rows

    teacher_row = next((r for r in rows if float(r["is_teacher_bundle"]) > 0.5), rows[0])
    base_ret = float(teacher_row["return_r0"])
    base_sr = float(teacher_row["success"])
    for row in rows:
        row["delta_return_vs_teacher_bundle"] = float(row["return_r0"] - base_ret)
        row["delta_success_vs_teacher_bundle"] = float(row["success"] - base_sr)
    return rows


def _analyze_proxy_alignment(candidate_df: pd.DataFrame, group_key: str, delta_col: str, proxy_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []

    for proxy in proxy_cols:
        x = candidate_df[proxy].astype(float)
        y = candidate_df[delta_col].astype(float)
        overall_rows.append(
            {
                "proxy": str(proxy),
                "scope": "overall",
                "spearman": _safe_spearman(x.tolist(), y.tolist()),
            }
        )

        per_state_corr: List[float] = []
        top1_hits: List[float] = []
        positive_regret_ranks: List[int] = []

        for _, sub in candidate_df.groupby(group_key):
            if len(sub) < 2:
                continue
            corr = _safe_spearman(sub[proxy].astype(float).tolist(), sub[delta_col].astype(float).tolist())
            if corr is not None:
                per_state_corr.append(float(corr))

            pred_top = sub.sort_values(proxy, ascending=False).iloc[0]
            true_top = sub.sort_values(delta_col, ascending=False).iloc[0]
            top1_hits.append(float(pred_top.get("candidate_slot3_local", pred_top.get("bundle_actions")) == true_top.get("candidate_slot3_local", true_top.get("bundle_actions"))))

            pos = sub[sub[delta_col] > 1e-12].sort_values(proxy, ascending=False)
            if len(pos) > 0:
                sorted_full = sub.sort_values(proxy, ascending=False).reset_index(drop=True)
                pos_idx = sorted_full[sorted_full.index.isin(pos.index)].index.tolist()
                if pos_idx:
                    positive_regret_ranks.append(int(min(pos_idx) + 1))

        state_rows.append(
            {
                "proxy": str(proxy),
                "scope": "per_state",
                "state_spearman_mean": None if not per_state_corr else float(statistics.mean(per_state_corr)),
                "state_spearman_median": None if not per_state_corr else float(statistics.median(per_state_corr)),
                "top1_hit_rate": None if not top1_hits else float(statistics.mean(top1_hits)),
                "positive_regret_rank_median": None if not positive_regret_ranks else float(statistics.median(positive_regret_ranks)),
                "positive_regret_rank_p75": None if not positive_regret_ranks else float(pd.Series(positive_regret_ranks).quantile(0.75)),
            }
        )

    return pd.DataFrame(overall_rows), pd.DataFrame(state_rows)


def _summarize_slot3_regret(state_rows: pd.DataFrame, candidate_rows: pd.DataFrame) -> Dict[str, Any]:
    if len(state_rows) == 0:
        return {"state_count": 0}
    positive = state_rows[state_rows["best_delta_return_vs_teacher_slot3"] > 1e-12]
    return {
        "state_count": int(len(state_rows)),
        "positive_regret_state_count": int(len(positive)),
        "positive_regret_state_fraction": float(len(positive) / max(len(state_rows), 1)),
        "positive_regret_mean_delta_return": float(positive["best_delta_return_vs_teacher_slot3"].mean()) if len(positive) else 0.0,
        "positive_regret_max_delta_return": float(positive["best_delta_return_vs_teacher_slot3"].max()) if len(positive) else 0.0,
        "oracle_slot3_upper_bound_avg_delta_return": float(state_rows["best_delta_return_vs_teacher_slot3"].clip(lower=0.0).mean()),
        "oracle_slot3_upper_bound_avg_delta_success": float(state_rows["best_delta_success_vs_teacher_slot3"].clip(lower=0.0).mean()),
        "teacher_slot3_top1_by_return_fraction": float((state_rows["teacher_slot3_rank_by_return"] == 1).mean()),
        "teacher_slot3_top3_by_return_fraction": float((state_rows["teacher_slot3_rank_by_return"] <= 3).mean()),
        "candidate_pool_size_mean": float(candidate_rows.groupby("state_key").size().mean()) if len(candidate_rows) else 0.0,
    }


def _summarize_bundle_regret(bundle_state_rows: pd.DataFrame) -> Dict[str, Any]:
    if len(bundle_state_rows) == 0:
        return {"audited_state_count": 0}
    pos = bundle_state_rows[bundle_state_rows["best_delta_return_vs_teacher_bundle"] > 1e-12]
    return {
        "audited_state_count": int(len(bundle_state_rows)),
        "positive_regret_state_count": int(len(pos)),
        "positive_regret_state_fraction": float(len(pos) / max(len(bundle_state_rows), 1)),
        "mean_best_delta_return": float(bundle_state_rows["best_delta_return_vs_teacher_bundle"].mean()),
        "mean_best_delta_success": float(bundle_state_rows["best_delta_success_vs_teacher_bundle"].mean()),
        "mean_extra_over_slot3_best": float(bundle_state_rows["bundle_best_minus_slot3_best_delta_return"].mean()),
        "slot3_enough_fraction": float((bundle_state_rows["bundle_best_minus_slot3_best_delta_return"] <= 1e-12).mean()),
    }


def run_panel(
    *,
    panel_name: str,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    device: torch.device,
    bc_model: SpimNativePolicy,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]

    state_manifest_rows: List[Dict[str, Any]] = []
    slot3_candidate_rows: List[Dict[str, Any]] = []
    slot3_state_rows: List[Dict[str, Any]] = []

    bundle_candidate_rows: List[Dict[str, Any]] = []
    bundle_state_rows: List[Dict[str, Any]] = []

    gate = DynamicReachabilityRuleModule()
    seen_states = 0

    # preselect bundle candidates later from slot3 state rows
    provisional_bundle_pool: List[Dict[str, Any]] = []
    audit_bucket_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    target_bucket_keys: List[Tuple[str, int]] = [
        (policy, ep) for policy in ["teacher", "bc"] for ep in range(1, int(runtime["num_episodes"]) + 1)
    ]

    ordered_cases = list(runtime["cases"])
    if bool(args.shuffle_cases):
        rnd = random.Random(int(args.seed))
        rnd.shuffle(ordered_cases)

    for policy_source in ["teacher", "bc"]:
        for case_idx, case in enumerate(ordered_cases):
            if int(args.audit_max_per_policy_round) > 0:
                current_policy_full = all(
                    audit_bucket_counts[(str(policy_source), ep)] >= int(args.audit_max_per_policy_round)
                    for ep in range(1, int(runtime["num_episodes"]) + 1)
                )
                if current_policy_full:
                    break
            if int(args.audit_max_per_policy_round) > 0:
                all_full = all(
                    audit_bucket_counts[(pol, ep)] >= int(args.audit_max_per_policy_round)
                    for (pol, ep) in target_bucket_keys
                )
                if all_full:
                    break
            rollout = _make_rollout(runtime, case)
            history = ObservationWitnessHistory()
            paper_state = PaperLikeHSRState(source_prior=None)

            trigger_global = _extract_trigger_global(case.data)
            source_local = resolve_source_local_idx(rollout)

            for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
                if int(args.audit_max_per_policy_round) > 0:
                    current_policy_full = all(
                        audit_bucket_counts[(str(policy_source), ep)] >= int(args.audit_max_per_policy_round)
                        for ep in range(1, int(runtime["num_episodes"]) + 1)
                    )
                    if current_policy_full:
                        break
                if int(args.audit_max_per_policy_round) > 0:
                    all_full = all(
                        audit_bucket_counts[(pol, ep)] >= int(args.audit_max_per_policy_round)
                        for (pol, ep) in target_bucket_keys
                    )
                    if all_full:
                        break
                if int(args.state_limit) > 0 and seen_states >= int(args.state_limit):
                    break

                pre_rollout = deepcopy(rollout)
                pre_history = deepcopy(history)
                pre_paper_state = deepcopy(paper_state)

                state, belief_ctx, spim_state, pre_metrics = _compute_step_belief(
                    family=family,
                    rollout=rollout,
                    history=history,
                    case=case,
                    runtime=runtime,
                    env=env,
                    trigger_global=trigger_global,
                    paper_state=paper_state,
                    onset_grid=onset_grid,
                    paper_like_alpha=float(args.paper_like_alpha),
                    paper_like_topk_fraction=float(args.paper_like_topk_fraction),
                    paper_like_time_tol_min=float(args.paper_like_time_tol_min),
                    soft_scenario_beta=float(args.soft_scenario_beta),
                    source_local=source_local,
                    gate=gate,
                    top_source_k=int(args.top_source_k),
                    include_surrogate_features=bool(args.include_surrogate_features),
                )

                valid_count = int(state["valid_mask"].sum().item())
                if valid_count <= 0:
                    break

                teacher_actions = _pick_topk_unsampled(
                    belief_ctx["belief"],
                    belief_ctx["candidate_mask"],
                    rollout,
                    int(runtime["action_budget"]),
                )
                teacher_actions = [int(v) for v in teacher_actions]
                if len(teacher_actions) <= 0:
                    break

                bucket_key = (str(policy_source), int(episode_idx))
                should_audit_state = True
                if int(args.audit_max_per_policy_round) > 0 and audit_bucket_counts[bucket_key] >= int(args.audit_max_per_policy_round):
                    should_audit_state = False

                # Conservative BC path: first two teacher slots fixed, slot3 optionally corrected.
                selected_actions = list(teacher_actions)
                gate_triggered = False
                corrective_replaced = False
                corrective_pool: List[int] = []
                if policy_source == "bc":
                    can_attempt = (
                        len(teacher_actions) >= 3
                        and int(runtime["action_budget"]) >= 3
                        and float(spim_state["diagnostics"]["candidate_count"]) >= float(args.ambiguity_min_candidate_count)
                        and float(spim_state["diagnostics"]["top1_top2_margin"]) <= float(args.ambiguity_top1_top2_max)
                    )
                    if can_attempt:
                        gate_triggered = True
                        candidate_pool = _posterior_topk_unsampled(
                            belief=belief_ctx["belief"],
                            candidate_mask=belief_ctx["candidate_mask"],
                            rollout=rollout,
                            topk=int(args.corrective_candidate_topk),
                        )
                        blocked = {int(teacher_actions[0]), int(teacher_actions[1])}
                        corrective_pool = [int(v) for v in candidate_pool if int(v) not in blocked]
                        if int(teacher_actions[2]) not in corrective_pool:
                            corrective_pool.append(int(teacher_actions[2]))
                        if corrective_pool:
                            with torch.no_grad():
                                logits = bc_model.score_actions(
                                    spim_state["global_features"].to(device),
                                    spim_state["local_features"].to(device),
                                )
                                idx = torch.tensor(corrective_pool, dtype=torch.long, device=device)
                                chosen_pos = int(torch.argmax(logits[idx]).item())
                                chosen_action = int(corrective_pool[chosen_pos])
                                selected_actions[2] = chosen_action
                                corrective_replaced = int(chosen_action) != int(teacher_actions[2])

                available_ids = _available_local_ids(belief_ctx["candidate_mask"], rollout)
                state_key = _state_key(policy_source, case.case_id, episode_idx)
                state_row = {
                    "panel": str(panel_name),
                    "policy_source": str(policy_source),
                    "state_key": str(state_key),
                    "case_id": str(case.case_id),
                    "episode_index": int(episode_idx),
                    "remaining_budget": int(spim_state["diagnostics"]["remaining_budget"]),
                    "candidate_count": int(spim_state["diagnostics"]["candidate_count"]),
                    "posterior_entropy": float(spim_state["diagnostics"]["posterior_entropy"]),
                    "mass_cover_0p7": float(spim_state["diagnostics"]["mass_cover_0p7"]),
                    "top1_mass": float(spim_state["diagnostics"]["top1_mass"]),
                    "top3_mass": float(spim_state["diagnostics"]["top3_mass"]),
                    "top1_top2_margin": float(spim_state["diagnostics"]["top1_top2_margin"]),
                    "teacher_actions": json.dumps([int(v) for v in teacher_actions]),
                    "selected_actions": json.dumps([int(v) for v in selected_actions]),
                    "legal_unsampled_count": int(len(available_ids)),
                    "legal_unsampled_local_ids": json.dumps([int(v) for v in available_ids]),
                    "gate_triggered": float(bool(gate_triggered)),
                    "corrective_third_replaced": float(bool(corrective_replaced)),
                    "corrective_pool_size": int(len(corrective_pool)),
                }
                if should_audit_state:
                    state_manifest_rows.append(state_row)
                    seen_states += 1
                    audit_bucket_counts[bucket_key] += 1

                    slot3_rows, baseline = _evaluate_slot3_candidates(
                        family=family,
                        case=case,
                        runtime=runtime,
                        env=env,
                        trigger_global=trigger_global,
                        source_local=source_local,
                        onset_grid=onset_grid,
                        episode_idx=int(episode_idx),
                        teacher_actions=teacher_actions,
                        belief_ctx=belief_ctx,
                        spim_state=spim_state,
                        pre_metrics=pre_metrics,
                        pre_rollout=pre_rollout,
                        pre_history=pre_history,
                        pre_paper_state=pre_paper_state,
                        slot3_topk=int(args.slot3_topk),
                    )
                    if slot3_rows:
                        for row in slot3_rows:
                            slot3_candidate_rows.append({**state_row, **row})

                        df = pd.DataFrame(slot3_rows)
                        teacher_mask = df["is_teacher_slot3"] > 0.5
                        teacher_return = float(df.loc[teacher_mask, "return_r0"].iloc[0]) if bool(teacher_mask.any()) else float(df.iloc[0]["return_r0"])
                        teacher_success = float(df.loc[teacher_mask, "success"].iloc[0]) if bool(teacher_mask.any()) else float(df.iloc[0]["success"])
                        best_idx = int(df["delta_return_vs_teacher_slot3"].astype(float).idxmax())
                        best_row = df.loc[best_idx]
                        order = df.sort_values("return_r0", ascending=False).reset_index(drop=True)
                        teacher_slot = int(df.loc[teacher_mask, "candidate_slot3_local"].iloc[0]) if bool(teacher_mask.any()) else int(df.iloc[0]["candidate_slot3_local"])
                        teacher_rank = int(order.index[order["candidate_slot3_local"] == teacher_slot][0]) + 1

                        slot3_state = {
                            **state_row,
                            "teacher_slot3_local": int(teacher_slot),
                            "teacher_slot3_return": float(teacher_return),
                            "teacher_slot3_success": float(teacher_success),
                            "best_slot3_local": int(best_row["candidate_slot3_local"]),
                            "best_delta_return_vs_teacher_slot3": float(best_row["delta_return_vs_teacher_slot3"]),
                            "best_delta_success_vs_teacher_slot3": float(best_row["delta_success_vs_teacher_slot3"]),
                            "teacher_slot3_rank_by_return": int(teacher_rank),
                        }
                        slot3_state_rows.append(slot3_state)

                        if float(best_row["delta_return_vs_teacher_slot3"]) >= float(args.bundle_min_slot3_regret):
                            provisional_bundle_pool.append(
                                {
                                    "priority": float(best_row["delta_return_vs_teacher_slot3"]),
                                    "state_ref": {
                                        "case": case,
                                        "episode_idx": int(episode_idx),
                                        "teacher_actions": teacher_actions,
                                        "belief_ctx": deepcopy(belief_ctx),
                                        "spim_state": deepcopy(spim_state),
                                        "pre_rollout": deepcopy(pre_rollout),
                                        "pre_history": deepcopy(pre_history),
                                        "pre_paper_state": deepcopy(pre_paper_state),
                                        "trigger_global": trigger_global,
                                        "source_local": source_local,
                                        "state_row": dict(state_row),
                                        "slot3_best_delta": float(best_row["delta_return_vs_teacher_slot3"]),
                                    },
                                }
                            )

                # step forward with current policy source trajectory
                round_hit = source_local is not None and int(source_local) in set(selected_actions)
                rollout.step_with_actions(
                    [int(v) for v in selected_actions],
                    sample_types=[f"{policy_source}_slot_{i}" for i in range(len(selected_actions))],
                )
                if rollout.history_steps:
                    history.append_from_history_step(rollout.history_steps[-1])
                if bool(round_hit):
                    break

            if int(args.state_limit) > 0 and seen_states >= int(args.state_limit):
                break
        if int(args.state_limit) > 0 and seen_states >= int(args.state_limit):
            break

    # Bundle audit on top priority slot3-regret states
    provisional_bundle_pool = sorted(provisional_bundle_pool, key=lambda x: (-float(x["priority"]), x["state_ref"]["state_row"]["state_key"]))
    selected_bundle_states = provisional_bundle_pool[: max(int(args.bundle_max_states), 0)]

    for item in selected_bundle_states:
        ref = item["state_ref"]
        b_rows = _evaluate_bundle_regret(
            family=family,
            case=ref["case"],
            runtime=runtime,
            env=env,
            trigger_global=ref["trigger_global"],
            source_local=ref["source_local"],
            onset_grid=onset_grid,
            episode_idx=int(ref["episode_idx"]),
            teacher_actions=ref["teacher_actions"],
            belief_ctx=ref["belief_ctx"],
            spim_state=ref["spim_state"],
            pre_rollout=ref["pre_rollout"],
            pre_history=ref["pre_history"],
            pre_paper_state=ref["pre_paper_state"],
            bundle_topk=int(args.bundle_topk),
            bundle_max_permutations=int(args.bundle_max_permutations),
        )
        if not b_rows:
            continue
        for row in b_rows:
            bundle_candidate_rows.append({**ref["state_row"], **row})

        bdf = pd.DataFrame(b_rows)
        best = bdf.sort_values("delta_return_vs_teacher_bundle", ascending=False).iloc[0]
        bundle_state_rows.append(
            {
                **ref["state_row"],
                "teacher_bundle": json.dumps([int(v) for v in ref["teacher_actions"][:3]]),
                "best_bundle": str(best["bundle_actions"]),
                "best_delta_return_vs_teacher_bundle": float(best["delta_return_vs_teacher_bundle"]),
                "best_delta_success_vs_teacher_bundle": float(best["delta_success_vs_teacher_bundle"]),
                "slot3_best_delta_return": float(ref["slot3_best_delta"]),
                "bundle_best_minus_slot3_best_delta_return": float(best["delta_return_vs_teacher_bundle"] - float(ref["slot3_best_delta"])),
            }
        )

    state_df = pd.DataFrame(state_manifest_rows)
    slot3_cand_df = pd.DataFrame(slot3_candidate_rows)
    slot3_state_df = pd.DataFrame(slot3_state_rows)
    bundle_cand_df = pd.DataFrame(bundle_candidate_rows)
    bundle_state_df = pd.DataFrame(bundle_state_rows)

    state_df.to_csv(output_dir / f"{panel_name}_state_manifest.csv", index=False)
    slot3_cand_df.to_csv(output_dir / f"{panel_name}_slot3_candidate_rows.csv", index=False)
    slot3_state_df.to_csv(output_dir / f"{panel_name}_slot3_state_summary.csv", index=False)
    bundle_cand_df.to_csv(output_dir / f"{panel_name}_bundle_candidate_rows.csv", index=False)
    bundle_state_df.to_csv(output_dir / f"{panel_name}_bundle_state_summary.csv", index=False)

    slot3_proxy_cols = [
        "proxy_entropy_drop",
        "proxy_cover_shrink",
        "proxy_disagreement",
        "proxy_action_mass",
        "proxy_action_gap_to_top1",
        "proxy_rank_percentile",
        "real_next_entropy_drop",
        "real_next_cover_shrink",
        "real_next_top1_mass_gain",
        "top1_top2_margin",
        "posterior_entropy",
        "mass_cover_0p7",
    ]
    if len(slot3_cand_df) > 0:
        slot3_overall_corr, slot3_state_corr = _analyze_proxy_alignment(
            slot3_cand_df,
            group_key="state_key",
            delta_col="delta_return_vs_teacher_slot3",
            proxy_cols=slot3_proxy_cols,
        )
    else:
        slot3_overall_corr = pd.DataFrame([])
        slot3_state_corr = pd.DataFrame([])
    slot3_overall_corr.to_csv(output_dir / f"{panel_name}_slot3_proxy_overall_corr.csv", index=False)
    slot3_state_corr.to_csv(output_dir / f"{panel_name}_slot3_proxy_state_corr.csv", index=False)

    bundle_proxy_cols = [
        "proxy_entropy_drop_sum",
        "proxy_cover_shrink_sum",
        "proxy_disagreement_sum",
        "proxy_mass_sum",
        "state_top1_top2_margin",
        "state_entropy",
    ]
    if len(bundle_cand_df) > 0:
        bundle_overall_corr, bundle_state_corr = _analyze_proxy_alignment(
            bundle_cand_df,
            group_key="state_key",
            delta_col="delta_return_vs_teacher_bundle",
            proxy_cols=bundle_proxy_cols,
        )
    else:
        bundle_overall_corr = pd.DataFrame([])
        bundle_state_corr = pd.DataFrame([])
    bundle_overall_corr.to_csv(output_dir / f"{panel_name}_bundle_proxy_overall_corr.csv", index=False)
    bundle_state_corr.to_csv(output_dir / f"{panel_name}_bundle_proxy_state_corr.csv", index=False)

    round_bucket = slot3_state_df.groupby(["policy_source", "episode_index"])["best_delta_return_vs_teacher_slot3"].agg(["count", "mean", "max"]).reset_index() if len(slot3_state_df) else pd.DataFrame([])
    round_bucket.to_csv(output_dir / f"{panel_name}_slot3_regret_by_round.csv", index=False)

    candidate_count_bucket = (
        slot3_state_df.assign(
            candidate_bin=pd.cut(
                slot3_state_df["candidate_count"],
                bins=[0, 4, 8, 12, 16, 24, 9999],
                labels=["1-4", "5-8", "9-12", "13-16", "17-24", "25+"],
                include_lowest=True,
            )
        )
        .groupby(["policy_source", "candidate_bin"])["best_delta_return_vs_teacher_slot3"]
        .agg(["count", "mean", "max"])
        .reset_index()
        if len(slot3_state_df)
        else pd.DataFrame([])
    )
    candidate_count_bucket.to_csv(output_dir / f"{panel_name}_slot3_regret_by_candidate_bin.csv", index=False)

    panel_summary = {
        "panel": str(panel_name),
        "state_collection": {
            "state_count": int(len(state_df)),
            "policy_state_counts": state_df["policy_source"].value_counts().to_dict() if len(state_df) else {},
            "case_count": int(state_df["case_id"].nunique()) if len(state_df) else 0,
            "gate_trigger_rate": float(state_df["gate_triggered"].mean()) if len(state_df) else 0.0,
            "corrective_third_replace_rate": float(state_df["corrective_third_replaced"].mean()) if len(state_df) else 0.0,
        },
        "slot3_regret": _summarize_slot3_regret(slot3_state_df, slot3_cand_df),
        "bundle_regret": _summarize_bundle_regret(bundle_state_df),
        "artifacts": {
            "state_manifest": str(output_dir / f"{panel_name}_state_manifest.csv"),
            "slot3_candidate_rows": str(output_dir / f"{panel_name}_slot3_candidate_rows.csv"),
            "slot3_state_summary": str(output_dir / f"{panel_name}_slot3_state_summary.csv"),
            "bundle_candidate_rows": str(output_dir / f"{panel_name}_bundle_candidate_rows.csv"),
            "bundle_state_summary": str(output_dir / f"{panel_name}_bundle_state_summary.csv"),
            "slot3_proxy_overall_corr": str(output_dir / f"{panel_name}_slot3_proxy_overall_corr.csv"),
            "slot3_proxy_state_corr": str(output_dir / f"{panel_name}_slot3_proxy_state_corr.csv"),
            "bundle_proxy_overall_corr": str(output_dir / f"{panel_name}_bundle_proxy_overall_corr.csv"),
            "bundle_proxy_state_corr": str(output_dir / f"{panel_name}_bundle_proxy_state_corr.csv"),
            "slot3_regret_by_round": str(output_dir / f"{panel_name}_slot3_regret_by_round.csv"),
            "slot3_regret_by_candidate_bin": str(output_dir / f"{panel_name}_slot3_regret_by_candidate_bin.csv"),
        },
    }
    return panel_summary


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    precheck_root = Path(args.precheck_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bc_checkpoint = Path(args.bc_checkpoint)
    if not bc_checkpoint.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {bc_checkpoint}")

    if str(args.teacher_family) == "auto":
        teacher_decision = auto_select_teacher(precheck_root)
        teacher_family = str(teacher_decision["selected_family"])
    else:
        teacher_family = str(args.teacher_family)
        teacher_decision = {
            "selected_family": str(teacher_family),
            "selection_rule": "manual_override",
            "precheck_path": str(precheck_root / "family_summary.csv"),
        }

    device = get_device(str(args.device))
    env = CleanTwoChannelEvidenceEnv()
    bc_model = _load_bc_model(
        checkpoint_path=bc_checkpoint,
        hidden_dim=int(args.hidden_dim),
        include_surrogate_features=bool(args.include_surrogate_features),
        device=device,
    )

    runtime_split = str(args.runtime_split)
    runtime_b30, split_meta_b30 = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=runtime_split,
        num_rounds=int(args.num_rounds),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=int(args.train_max_cases),
        train_cache_version=str(args.train_cache_version),
        case_limit=int(args.case_limit),
    )
    panel_b30_name = "exact136_B30" if runtime_split == "exact136" else f"{runtime_split}_B30"

    panel_summaries: Dict[str, Any] = {}
    panel_summaries[panel_b30_name] = run_panel(
        panel_name=panel_b30_name,
        runtime=runtime_b30,
        family=teacher_family,
        env=env,
        device=device,
        bc_model=bc_model,
        args=args,
        output_dir=output_dir,
    )

    runtime_b60 = None
    split_meta_b60 = None
    if bool(args.eval_b60):
        runtime_b60, split_meta_b60 = build_runtime_strict(
            source_root=source_root,
            cache_dir=cache_dir,
            split=runtime_split,
            num_rounds=20,
            actions_per_round=int(args.actions_per_round),
            train_max_cases=int(args.train_max_cases),
            train_cache_version=str(args.train_cache_version),
            case_limit=int(args.case_limit),
        )
        panel_b60_name = "exact136_B60" if runtime_split == "exact136" else f"{runtime_split}_B60"
        panel_summaries[panel_b60_name] = run_panel(
            panel_name=panel_b60_name,
            runtime=runtime_b60,
            family=teacher_family,
            env=env,
            device=device,
            bc_model=bc_model,
            args=args,
            output_dir=output_dir,
        )

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "teacher_family": str(teacher_family),
        "teacher_decision": teacher_decision,
        "fixed_contract": {
            "posterior_family": "hsr_soft_scenario_posterior_v3" if str(teacher_family) == "hsr_soft_scenario_posterior_v3" else str(teacher_family),
            "teacher_policy": "posterior_greedy",
            "bc_checkpoint": str(bc_checkpoint),
            "actions_per_round": int(args.actions_per_round),
            "b30_rounds": int(args.num_rounds),
            "b60_rounds": 20 if bool(args.eval_b60) else None,
            "action_contract": "3 legal unsampled actions each round when available",
        },
        "audit_config": {
            "runtime_split": runtime_split,
            "slot3_topk": int(args.slot3_topk),
            "bundle_topk": int(args.bundle_topk),
            "bundle_max_states": int(args.bundle_max_states),
            "bundle_max_permutations": int(args.bundle_max_permutations),
            "bundle_min_slot3_regret": float(args.bundle_min_slot3_regret),
            "ambiguity_top1_top2_max": float(args.ambiguity_top1_top2_max),
            "ambiguity_min_candidate_count": int(args.ambiguity_min_candidate_count),
            "corrective_candidate_topk": int(args.corrective_candidate_topk),
            "state_limit": int(args.state_limit),
            "eval_b60": bool(args.eval_b60),
            "train_max_cases": int(args.train_max_cases),
            "train_cache_version": str(args.train_cache_version),
            "case_limit": int(args.case_limit),
        },
        "split_meta": {
            "b30": split_meta_b30,
            "b60": split_meta_b60,
        },
        "panel_summaries": panel_summaries,
        "commands_hint": {
            "main": "python -m src.scripts.audit.run_spim_teacher_regret_reward_alignment_audit --output-dir <dir> --bc-checkpoint <pt>",
        },
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
