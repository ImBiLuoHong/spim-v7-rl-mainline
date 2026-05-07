from __future__ import annotations

import argparse
import itertools
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch

from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.run_spim_teacher_regret_reward_alignment_audit import (
    _compute_step_belief,
    _make_rollout,
    _rollout_with_teacher_continuation,
)
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_audit import write_json
from src.scripts.run_spim_family_sweep import PaperLikeHSRState, _extract_trigger_global
from src.scripts.run_spim_policy_eval_strict import build_runtime_strict
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_PRECHECK_ROOT,
    DEFAULT_SOURCE_ROOT,
    _pick_topk_unsampled,
    _posterior_topk_unsampled,
    auto_select_teacher,
    get_device,
    seed_everything,
)


RUNNER_VERSION = "spim_strict_set_bundle_headroom_audit_v1"
PANEL_VERSION = "strict_val_b30_b60_set_bundle_headroom_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict held-out set-level/bundle-level headroom audit under fixed posterior+teacher.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--precheck-root", type=str, default=str(DEFAULT_PRECHECK_ROOT))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--teacher-family", type=str, default="hsr_soft_scenario_posterior_v3", choices=["auto", "hsr_soft_scenario_posterior_v3", "hsr_paper_topk_ema_v1"])

    parser.add_argument("--runtime-split", type=str, default="val", choices=["exact136", "train", "val", "test"])
    parser.add_argument("--train-max-cases", type=int, default=0)
    parser.add_argument("--train-cache-version", type=str, default="")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--shuffle-cases", action="store_true")

    parser.add_argument("--num-rounds-b30", type=int, default=10)
    parser.add_argument("--num-rounds-b60", type=int, default=20)
    parser.add_argument("--actions-per-round", type=int, default=3)

    parser.add_argument("--audit-max-per-episode", type=int, default=8, help="Bounded per-panel quota for each episode index. 0 means no bound.")
    parser.add_argument("--state-limit", type=int, default=0, help="Hard cap per panel; 0 means no cap.")

    parser.add_argument("--candidate-topk", type=int, default=6, help="Candidate pool contract: posterior top-k legal unsampled (teacher actions forced-in).")
    parser.add_argument("--max-two-member-combos", type=int, default=0, help="Optional cap per state; 0 means full.")
    parser.add_argument("--max-bundle-permutations", type=int, default=0, help="Optional cap per state; 0 means full.")
    parser.add_argument(
        "--state-selector-root",
        type=str,
        default="",
        help="Optional directory containing {panel}_slot3_state_summary.csv or {panel}_state_manifest.csv; only these teacher state_key values are audited.",
    )

    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)
    parser.add_argument("--top-source-k", type=int, default=8)
    parser.add_argument("--include-surrogate-features", action="store_true")
    return parser.parse_args()


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _safe_clip_mean(values: Iterable[float]) -> float:
    return _safe_mean(max(float(v), 0.0) for v in values)


def _state_key(case_id: str, episode_idx: int) -> str:
    return f"teacher::{case_id}::ep{int(episode_idx)}"


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and v != v


def _same_hit_round(a: Any, b: Any) -> bool:
    if (a is None or _is_nan(a)) and (b is None or _is_nan(b)):
        return True
    if a is None or _is_nan(a) or b is None or _is_nan(b):
        return False
    return int(a) == int(b)


def _candidate_pool(
    *,
    belief_ctx: Dict[str, Any],
    rollout: Any,
    teacher_actions: Sequence[int],
    topk: int,
) -> List[int]:
    pool = _posterior_topk_unsampled(
        belief=belief_ctx["belief"],
        candidate_mask=belief_ctx["candidate_mask"],
        rollout=rollout,
        topk=max(int(topk), 3),
    )
    ordered = [int(v) for v in pool]
    cap = max(int(topk), 3)
    for a in [int(teacher_actions[0]), int(teacher_actions[1]), int(teacher_actions[2])]:
        if a in ordered:
            continue
        ordered.insert(0, a)
    seen = set()
    dedup: List[int] = []
    for v in ordered:
        if int(v) in seen:
            continue
        seen.add(int(v))
        dedup.append(int(v))
    return dedup[:cap]


def _summarize_panel(df: pd.DataFrame, num_rounds: int, action_budget: int) -> Dict[str, Any]:
    if len(df) <= 0:
        return {"case_count": 0}
    hit_mask = df["success"] > 0.5
    fallback_round = int(num_rounds) + 1
    fallback_sample = int(num_rounds * action_budget + 1)
    return {
        "case_count": int(len(df)),
        "success_rate": float(df["success"].mean()),
        "avg_hit_round_conditional": None if not bool(hit_mask.any()) else float(df.loc[hit_mask, "hit_round"].mean()),
        "avg_hit_round_with_fail_fallback": float(df["hit_round"].fillna(fallback_round).mean()),
        "avg_hit_sample_conditional": None if not bool(hit_mask.any()) else float(df.loc[hit_mask, "hit_sample_index"].mean()),
        "avg_hit_sample_with_fail_fallback": float(df["hit_sample_index"].fillna(fallback_sample).mean()),
        "avg_return_r0": float(df["return_r0"].mean()),
        "avg_budget_used": float(df["budget_used"].mean()),
    }


def _contract_eval_from_state_row(state_row: pd.Series, action_col: str) -> Dict[str, Any]:
    action_list = json.loads(str(state_row[action_col])) if action_col in state_row and str(state_row[action_col]).strip() else []
    return {
        "bundle_actions": [int(v) for v in action_list] if isinstance(action_list, list) else [],
        "success": float(state_row.get(f"{action_col.replace('_action', '')}_success", state_row.get("teacher_success", 0.0))),
        "hit_round": state_row.get(f"{action_col.replace('_action', '')}_hit_round", state_row.get("teacher_hit_round", None)),
        "hit_sample_index": state_row.get(
            f"{action_col.replace('_action', '')}_hit_sample_index",
            state_row.get("teacher_hit_sample_index", None),
        ),
        "budget_used": int(state_row.get(f"{action_col.replace('_action', '')}_budget_used", state_row.get("teacher_budget_used", 0))),
        "return_r0": float(state_row.get(f"{action_col.replace('_action', '')}_return_r0", state_row.get("teacher_return_r0", 0.0))),
        "termination_reason": str(
            state_row.get(f"{action_col.replace('_action', '')}_termination_reason", state_row.get("teacher_termination_reason", "unknown"))
        ),
    }


def _load_panel_state_selector(state_selector_root: Path, panel_name: str) -> Set[str]:
    candidates = [
        state_selector_root / f"{panel_name}_slot3_state_summary.csv",
        state_selector_root / f"{panel_name}_state_manifest.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "policy_source" in df.columns:
            df = df[df["policy_source"].astype(str) == "teacher"].copy()
        if "state_key" not in df.columns:
            continue
        return {str(v) for v in df["state_key"].astype(str).tolist()}
    return set()


def _pick_best(rows: List[Dict[str, Any]], delta_ret_key: str, delta_succ_key: str) -> Dict[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda r: (float(r[delta_ret_key]), float(r[delta_succ_key])),
        reverse=True,
    )[0]


def _evaluate_state_contracts(
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
    pool: Sequence[int],
    pre_rollout: Any,
    pre_history: ObservationWitnessHistory,
    pre_paper_state: PaperLikeHSRState,
    max_two_member_combos: int,
    max_bundle_permutations: int,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> Dict[str, Any]:
    teacher_bundle = [int(teacher_actions[0]), int(teacher_actions[1]), int(teacher_actions[2])]
    eval_cache: Dict[Tuple[int, int, int], Dict[str, float]] = {}

    def eval_bundle(actions3: Sequence[int]) -> Dict[str, float]:
        key = (int(actions3[0]), int(actions3[1]), int(actions3[2]))
        if key in eval_cache:
            return eval_cache[key]
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
            first_actions=[int(v) for v in actions3],
            paper_like_alpha=float(paper_like_alpha),
            paper_like_topk_fraction=float(paper_like_topk_fraction),
            paper_like_time_tol_min=float(paper_like_time_tol_min),
            soft_scenario_beta=float(soft_scenario_beta),
        )
        eval_cache[key] = {
            "success": float(out["success"]),
            "hit_round": out["hit_round"],
            "hit_sample_index": out["hit_sample_index"],
            "budget_used": int(out["budget_used"]),
            "return_r0": float(out["return_r0"]),
            "termination_reason": str(out["termination_reason"]),
        }
        return eval_cache[key]

    teacher_eval = eval_bundle(teacher_bundle)
    base_ret = float(teacher_eval["return_r0"])
    base_succ = float(teacher_eval["success"])

    def add_deltas(rows: List[Dict[str, Any]], action_key: str) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for row in rows:
            metrics = eval_bundle(row[action_key])
            out_rows.append(
                {
                    **row,
                    "success": float(metrics["success"]),
                    "hit_round": metrics["hit_round"],
                    "hit_sample_index": metrics["hit_sample_index"],
                    "budget_used": int(metrics["budget_used"]),
                    "return_r0": float(metrics["return_r0"]),
                    "termination_reason": str(metrics["termination_reason"]),
                    "delta_return_vs_teacher": float(metrics["return_r0"] - base_ret),
                    "delta_success_vs_teacher": float(metrics["success"] - base_succ),
                }
            )
        return out_rows

    single_rows_by_slot: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: []}
    for slot_idx in [0, 1, 2]:
        fixed = [teacher_bundle[j] for j in [0, 1, 2] if j != slot_idx]
        for cand in pool:
            c = int(cand)
            if c in fixed:
                continue
            bundle = list(teacher_bundle)
            bundle[slot_idx] = c
            if len(set(bundle)) < 3:
                continue
            single_rows_by_slot[slot_idx].append(
                {
                    "slot_idx": int(slot_idx + 1),
                    "candidate_local": int(c),
                    "bundle_actions": [int(v) for v in bundle],
                }
            )
        single_rows_by_slot[slot_idx] = add_deltas(single_rows_by_slot[slot_idx], "bundle_actions")

    best_single_slot = {
        1: _pick_best(single_rows_by_slot[0], "delta_return_vs_teacher", "delta_success_vs_teacher"),
        2: _pick_best(single_rows_by_slot[1], "delta_return_vs_teacher", "delta_success_vs_teacher"),
        3: _pick_best(single_rows_by_slot[2], "delta_return_vs_teacher", "delta_success_vs_teacher"),
    }
    all_single = single_rows_by_slot[0] + single_rows_by_slot[1] + single_rows_by_slot[2]
    best_single_any = _pick_best(all_single, "delta_return_vs_teacher", "delta_success_vs_teacher")

    two_rows: List[Dict[str, Any]] = []
    for pair in [(0, 1), (0, 2), (1, 2)]:
        pair_rows: List[Dict[str, Any]] = []
        fixed_idx = [i for i in [0, 1, 2] if i not in pair][0]
        fixed_val = int(teacher_bundle[fixed_idx])
        for c0 in pool:
            for c1 in pool:
                a = int(c0)
                b = int(c1)
                if a == b:
                    continue
                bundle = list(teacher_bundle)
                bundle[pair[0]] = a
                bundle[pair[1]] = b
                if len(set(bundle)) < 3:
                    continue
                if fixed_val in {a, b}:
                    continue
                pair_rows.append(
                    {
                        "pair": f"{pair[0] + 1}{pair[1] + 1}",
                        "bundle_actions": [int(v) for v in bundle],
                    }
                )
        if int(max_two_member_combos) > 0:
            pair_rows = pair_rows[: int(max_two_member_combos)]
        pair_rows = add_deltas(pair_rows, "bundle_actions")
        two_rows.extend(pair_rows)
    best_two_any = _pick_best(two_rows, "delta_return_vs_teacher", "delta_success_vs_teacher")

    bundle_rows: List[Dict[str, Any]] = []
    perms = list(itertools.permutations([int(v) for v in pool], 3))
    if int(max_bundle_permutations) > 0:
        perms = perms[: int(max_bundle_permutations)]
    for perm in perms:
        bundle_rows.append({"bundle_actions": [int(perm[0]), int(perm[1]), int(perm[2])]})
    bundle_rows = add_deltas(bundle_rows, "bundle_actions")
    best_bundle = _pick_best(bundle_rows, "delta_return_vs_teacher", "delta_success_vs_teacher")

    return {
        "teacher_return_r0": float(base_ret),
        "teacher_success": float(base_succ),
        "teacher_hit_round": teacher_eval["hit_round"],
        "teacher_hit_sample_index": teacher_eval["hit_sample_index"],
        "teacher_budget_used": int(teacher_eval["budget_used"]),
        "teacher_termination_reason": str(teacher_eval["termination_reason"]),
        "best_single_slot1": best_single_slot[1],
        "best_single_slot2": best_single_slot[2],
        "best_single_slot3": best_single_slot[3],
        "best_single_any": best_single_any,
        "best_two_any": best_two_any,
        "best_bundle": best_bundle,
        "single_eval_count": int(len(all_single)),
        "two_eval_count": int(len(two_rows)),
        "bundle_eval_count": int(len(bundle_rows)),
        "unique_bundle_eval_count": int(len(eval_cache)),
    }


def _run_policy_rollout(
    *,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    correction_bundle_map: Dict[str, Sequence[int]],
    top_source_k: int,
    include_surrogate_features: bool,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> pd.DataFrame:
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]
    gate = DynamicReachabilityRuleModule()
    rows: List[Dict[str, Any]] = []

    for case in runtime["cases"]:
        rollout = _make_rollout(runtime, case)
        history = ObservationWitnessHistory()
        paper_state = PaperLikeHSRState(source_prior=None)
        trigger_global = _extract_trigger_global(case.data)
        source_local = resolve_source_local_idx(rollout)

        hit_round: Optional[int] = None
        hit_sample_index: Optional[int] = None
        total_reward = 0.0
        budget_used = 0
        termination_reason = "budget_exhausted"

        for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
            state, belief_ctx, _spim_state, _ = _compute_step_belief(
                family=family,
                rollout=rollout,
                history=history,
                case=case,
                runtime=runtime,
                env=env,
                trigger_global=trigger_global,
                paper_state=paper_state,
                onset_grid=onset_grid,
                paper_like_alpha=float(paper_like_alpha),
                paper_like_topk_fraction=float(paper_like_topk_fraction),
                paper_like_time_tol_min=float(paper_like_time_tol_min),
                soft_scenario_beta=float(soft_scenario_beta),
                source_local=source_local,
                gate=gate,
                top_source_k=int(top_source_k),
                include_surrogate_features=bool(include_surrogate_features),
            )
            if int(state["valid_mask"].sum().item()) <= 0:
                termination_reason = "no_valid_nodes"
                break

            teacher_actions = _pick_topk_unsampled(
                belief_ctx["belief"],
                belief_ctx["candidate_mask"],
                rollout,
                int(runtime["action_budget"]),
            )
            teacher_actions = [int(v) for v in teacher_actions]
            if not teacher_actions:
                termination_reason = "teacher_no_action"
                break

            selected_actions = list(teacher_actions)
            key = _state_key(str(case.case_id), int(episode_idx))
            if key in correction_bundle_map:
                corrected = [int(v) for v in correction_bundle_map[key]]
                if len(corrected) >= 3:
                    selected_actions[:3] = corrected[:3]

            round_hit = source_local is not None and int(source_local) in set(selected_actions)
            if round_hit and hit_round is None:
                hit_round = int(episode_idx)
                source_slot = selected_actions.index(int(source_local)) + 1
                hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

            total_reward += float((-1.0 / 30.0) * float(len(selected_actions)) + (1.0 if bool(round_hit) else 0.0))
            rollout.step_with_actions(
                selected_actions,
                sample_types=[f"setbundle_oracle_slot_{i}" for i in range(len(selected_actions))],
            )
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            budget_used += int(len(selected_actions))
            if round_hit:
                termination_reason = "source_hit"
                break

        rows.append(
            {
                "case_id": str(case.case_id),
                "success": float(hit_round is not None),
                "hit_round": None if hit_round is None else int(hit_round),
                "hit_sample_index": None if hit_sample_index is None else int(hit_sample_index),
                "budget_used": int(budget_used),
                "return_r0": float(total_reward),
                "termination_reason": str(termination_reason),
            }
        )
    return pd.DataFrame(rows)


def _panel_contract_summary(state_df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    ret_col = f"{prefix}_delta_return"
    succ_col = f"{prefix}_delta_success"
    if len(state_df) == 0 or ret_col not in state_df.columns:
        return {
            "state_count": 0,
            "positive_state_fraction": 0.0,
            "upper_bound_avg_delta_return": 0.0,
            "upper_bound_avg_delta_success": 0.0,
            "raw_mean_delta_return": 0.0,
            "raw_mean_delta_success": 0.0,
        }
    ret = state_df[ret_col].astype(float)
    succ = state_df[succ_col].astype(float)
    return {
        "state_count": int(len(state_df)),
        "positive_state_fraction": float((ret > 1e-12).mean()),
        "upper_bound_avg_delta_return": _safe_clip_mean(ret.tolist()),
        "upper_bound_avg_delta_success": _safe_clip_mean(succ.tolist()),
        "raw_mean_delta_return": _safe_mean(ret.tolist()),
        "raw_mean_delta_success": _safe_mean(succ.tolist()),
    }


def _build_panel_summary(state_df: pd.DataFrame, panel_name: str) -> Dict[str, Any]:
    contracts = {
        "single_slot1": _panel_contract_summary(state_df, "best_single_slot1"),
        "single_slot2": _panel_contract_summary(state_df, "best_single_slot2"),
        "single_slot3": _panel_contract_summary(state_df, "best_single_slot3"),
        "single_any": _panel_contract_summary(state_df, "best_single_any"),
        "two_member_any": _panel_contract_summary(state_df, "best_two_any"),
        "bundle_full": _panel_contract_summary(state_df, "best_bundle"),
    }
    single = float(contracts["single_any"]["upper_bound_avg_delta_return"])
    two = float(contracts["two_member_any"]["upper_bound_avg_delta_return"])
    full = float(contracts["bundle_full"]["upper_bound_avg_delta_return"])
    return {
        "panel": str(panel_name),
        "state_count": int(len(state_df)),
        "teacher_success_mean": _safe_mean(state_df["teacher_success"].tolist()) if len(state_df) else 0.0,
        "teacher_return_r0_mean": _safe_mean(state_df["teacher_return_r0"].tolist()) if len(state_df) else 0.0,
        "contracts": contracts,
        "incremental_headroom_return": {
            "two_minus_single_any": float(two - single),
            "bundle_minus_single_any": float(full - single),
            "bundle_minus_two_member": float(full - two),
        },
        "workload": {
            "single_eval_count_total": int(state_df["single_eval_count"].sum()) if len(state_df) else 0,
            "two_eval_count_total": int(state_df["two_eval_count"].sum()) if len(state_df) else 0,
            "bundle_eval_count_total": int(state_df["bundle_eval_count"].sum()) if len(state_df) else 0,
            "unique_bundle_eval_count_total": int(state_df["unique_bundle_eval_count"].sum()) if len(state_df) else 0,
        },
    }


def _compute_case_change_metrics(teacher_df: pd.DataFrame, oracle_df: pd.DataFrame) -> Dict[str, float]:
    merged = teacher_df.merge(oracle_df, on="case_id", suffixes=("_teacher", "_oracle"), how="inner")
    if len(merged) <= 0:
        return {
            "result_changed_episode_fraction": 0.0,
            "success_flip_episode_fraction": 0.0,
            "success_flip_down_episode_fraction": 0.0,
        }
    changed = []
    success_flip_up = []
    success_flip_down = []
    for _, r in merged.iterrows():
        success_t = float(r["success_teacher"])
        success_o = float(r["success_oracle"])
        ret_t = float(r["return_r0_teacher"])
        ret_o = float(r["return_r0_oracle"])
        hr_t = r["hit_round_teacher"]
        hr_o = r["hit_round_oracle"]
        row_changed = (abs(ret_o - ret_t) > 1e-12) or (abs(success_o - success_t) > 1e-12) or (not _same_hit_round(hr_t, hr_o))
        changed.append(bool(row_changed))
        success_flip_up.append(bool(success_t < 0.5 and success_o > 0.5))
        success_flip_down.append(bool(success_t > 0.5 and success_o < 0.5))
    return {
        "result_changed_episode_fraction": float(sum(changed) / max(len(changed), 1)),
        "success_flip_episode_fraction": float(sum(success_flip_up) / max(len(success_flip_up), 1)),
        "success_flip_down_episode_fraction": float(sum(success_flip_down) / max(len(success_flip_down), 1)),
    }


def _build_panel_oracle_upper_bound(
    *,
    state_df: pd.DataFrame,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    top_source_k: int,
    include_surrogate_features: bool,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> Dict[str, Any]:
    contract_to_action_col = {
        "single_slot1": "best_single_slot1_action",
        "single_slot2": "best_single_slot2_action",
        "single_slot3": "best_single_slot3_action",
        "single_any": "best_single_any_action",
        "two_member_any": "best_two_any_action",
        "bundle_full": "best_bundle_action",
    }
    baseline_df = _run_policy_rollout(
        runtime=runtime,
        family=family,
        env=env,
        correction_bundle_map={},
        top_source_k=int(top_source_k),
        include_surrogate_features=bool(include_surrogate_features),
        paper_like_alpha=float(paper_like_alpha),
        paper_like_topk_fraction=float(paper_like_topk_fraction),
        paper_like_time_tol_min=float(paper_like_time_tol_min),
        soft_scenario_beta=float(soft_scenario_beta),
    )
    baseline_summary = _summarize_panel(
        baseline_df,
        num_rounds=int(runtime["num_episodes"]),
        action_budget=int(runtime["action_budget"]),
    )

    contract_rows: List[Dict[str, Any]] = []
    case_rows: Dict[str, pd.DataFrame] = {"teacher": baseline_df}
    for contract_name, action_col in contract_to_action_col.items():
        correction_bundle_map: Dict[str, Sequence[int]] = {}
        for _, row in state_df.iterrows():
            parsed = _contract_eval_from_state_row(row, action_col)
            bundle = parsed["bundle_actions"]
            if len(bundle) >= 3:
                correction_bundle_map[str(row["state_key"])] = [int(v) for v in bundle[:3]]

        oracle_df = _run_policy_rollout(
            runtime=runtime,
            family=family,
            env=env,
            correction_bundle_map=correction_bundle_map,
            top_source_k=int(top_source_k),
            include_surrogate_features=bool(include_surrogate_features),
            paper_like_alpha=float(paper_like_alpha),
            paper_like_topk_fraction=float(paper_like_topk_fraction),
            paper_like_time_tol_min=float(paper_like_time_tol_min),
            soft_scenario_beta=float(soft_scenario_beta),
        )
        case_rows[contract_name] = oracle_df
        oracle_summary = _summarize_panel(
            oracle_df,
            num_rounds=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
        )
        case_changes = _compute_case_change_metrics(baseline_df, oracle_df)
        contract_rows.append(
            {
                "contract": str(contract_name),
                "teacher_success_rate": float(baseline_summary["success_rate"]),
                "oracle_success_rate": float(oracle_summary["success_rate"]),
                "delta_success": float(oracle_summary["success_rate"] - baseline_summary["success_rate"]),
                "teacher_avg_hit_round_conditional": baseline_summary["avg_hit_round_conditional"],
                "oracle_avg_hit_round_conditional": oracle_summary["avg_hit_round_conditional"],
                "delta_hit_round": (
                    None
                    if oracle_summary["avg_hit_round_conditional"] is None or baseline_summary["avg_hit_round_conditional"] is None
                    else float(oracle_summary["avg_hit_round_conditional"] - baseline_summary["avg_hit_round_conditional"])
                ),
                "teacher_avg_return_r0": float(baseline_summary["avg_return_r0"]),
                "oracle_avg_return_r0": float(oracle_summary["avg_return_r0"]),
                "delta_return_r0": float(oracle_summary["avg_return_r0"] - baseline_summary["avg_return_r0"]),
                "result_changed_episode_fraction": float(case_changes["result_changed_episode_fraction"]),
                "success_flip_episode_fraction": float(case_changes["success_flip_episode_fraction"]),
                "success_flip_down_episode_fraction": float(case_changes["success_flip_down_episode_fraction"]),
                "oracle_corrected_state_count": int(len(correction_bundle_map)),
            }
        )
    return {
        "teacher_baseline": baseline_summary,
        "contracts": contract_rows,
        "case_rows": case_rows,
    }


def _run_panel(
    *,
    panel_name: str,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    args: argparse.Namespace,
    output_dir: Path,
    state_selector: Optional[Set[str]],
) -> Dict[str, Any]:
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]
    gate = DynamicReachabilityRuleModule()

    rows: List[Dict[str, Any]] = []
    episode_audit_counts: Dict[int, int] = {ep: 0 for ep in range(1, int(runtime["num_episodes"]) + 1)}

    ordered_cases = list(runtime["cases"])
    if bool(args.shuffle_cases):
        rnd = random.Random(int(args.seed))
        rnd.shuffle(ordered_cases)

    for case in ordered_cases:
        if int(args.audit_max_per_episode) > 0:
            done = all(v >= int(args.audit_max_per_episode) for v in episode_audit_counts.values())
            if done:
                break
        if int(args.state_limit) > 0 and len(rows) >= int(args.state_limit):
            break

        rollout = _make_rollout(runtime, case)
        history = ObservationWitnessHistory()
        paper_state = PaperLikeHSRState(source_prior=None)
        trigger_global = _extract_trigger_global(case.data)
        source_local = resolve_source_local_idx(rollout)

        for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
            if int(args.audit_max_per_episode) > 0 and episode_audit_counts[int(episode_idx)] >= int(args.audit_max_per_episode):
                continue
            if int(args.state_limit) > 0 and len(rows) >= int(args.state_limit):
                break

            pre_rollout = deepcopy(rollout)
            pre_history = deepcopy(history)
            pre_paper_state = deepcopy(paper_state)

            state, belief_ctx, spim_state, _pre_metrics = _compute_step_belief(
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
            if len(teacher_actions) < 3:
                break

            pool = _candidate_pool(
                belief_ctx=belief_ctx,
                rollout=rollout,
                teacher_actions=teacher_actions,
                topk=int(args.candidate_topk),
            )
            if len(pool) < 3:
                break

            current_state_key = _state_key(str(case.case_id), int(episode_idx))
            if state_selector is not None and len(state_selector) > 0 and current_state_key not in state_selector:
                round_hit = source_local is not None and int(source_local) in set(teacher_actions[:3])
                rollout.step_with_actions(
                    [int(v) for v in teacher_actions[:3]],
                    sample_types=[f"teacher_slot_{i}" for i in range(3)],
                )
                if rollout.history_steps:
                    history.append_from_history_step(rollout.history_steps[-1])
                if bool(round_hit):
                    break
                continue

            contract = _evaluate_state_contracts(
                family=family,
                case=case,
                runtime=runtime,
                env=env,
                trigger_global=trigger_global,
                source_local=source_local,
                onset_grid=onset_grid,
                episode_idx=int(episode_idx),
                teacher_actions=teacher_actions,
                pool=pool,
                pre_rollout=pre_rollout,
                pre_history=pre_history,
                pre_paper_state=pre_paper_state,
                max_two_member_combos=int(args.max_two_member_combos),
                max_bundle_permutations=int(args.max_bundle_permutations),
                paper_like_alpha=float(args.paper_like_alpha),
                paper_like_topk_fraction=float(args.paper_like_topk_fraction),
                paper_like_time_tol_min=float(args.paper_like_time_tol_min),
                soft_scenario_beta=float(args.soft_scenario_beta),
            )

            row = {
                "panel": str(panel_name),
                "state_key": current_state_key,
                "case_id": str(case.case_id),
                "episode_index": int(episode_idx),
                "remaining_budget": int(spim_state["diagnostics"]["remaining_budget"]),
                "candidate_count": int(spim_state["diagnostics"]["candidate_count"]),
                "posterior_entropy": float(spim_state["diagnostics"]["posterior_entropy"]),
                "top1_top2_margin": float(spim_state["diagnostics"]["top1_top2_margin"]),
                "teacher_actions": json.dumps([int(v) for v in teacher_actions[:3]]),
                "candidate_pool": json.dumps([int(v) for v in pool]),
                "candidate_pool_size": int(len(pool)),
                "teacher_success": float(contract["teacher_success"]),
                "teacher_hit_round": contract["teacher_hit_round"],
                "teacher_hit_sample_index": contract["teacher_hit_sample_index"],
                "teacher_budget_used": int(contract["teacher_budget_used"]),
                "teacher_termination_reason": str(contract["teacher_termination_reason"]),
                "teacher_return_r0": float(contract["teacher_return_r0"]),
                "best_single_slot1_action": json.dumps(contract["best_single_slot1"].get("bundle_actions", [])),
                "best_single_slot1_success": float(contract["best_single_slot1"].get("success", contract["teacher_success"])),
                "best_single_slot1_hit_round": contract["best_single_slot1"].get("hit_round", contract["teacher_hit_round"]),
                "best_single_slot1_hit_sample_index": contract["best_single_slot1"].get("hit_sample_index", contract["teacher_hit_sample_index"]),
                "best_single_slot1_budget_used": int(contract["best_single_slot1"].get("budget_used", contract["teacher_budget_used"])),
                "best_single_slot1_termination_reason": str(contract["best_single_slot1"].get("termination_reason", contract["teacher_termination_reason"])),
                "best_single_slot1_delta_return": float(contract["best_single_slot1"].get("delta_return_vs_teacher", 0.0)),
                "best_single_slot1_delta_success": float(contract["best_single_slot1"].get("delta_success_vs_teacher", 0.0)),
                "best_single_slot1_return_r0": float(contract["best_single_slot1"].get("return_r0", contract["teacher_return_r0"])),
                "best_single_slot2_action": json.dumps(contract["best_single_slot2"].get("bundle_actions", [])),
                "best_single_slot2_success": float(contract["best_single_slot2"].get("success", contract["teacher_success"])),
                "best_single_slot2_hit_round": contract["best_single_slot2"].get("hit_round", contract["teacher_hit_round"]),
                "best_single_slot2_hit_sample_index": contract["best_single_slot2"].get("hit_sample_index", contract["teacher_hit_sample_index"]),
                "best_single_slot2_budget_used": int(contract["best_single_slot2"].get("budget_used", contract["teacher_budget_used"])),
                "best_single_slot2_termination_reason": str(contract["best_single_slot2"].get("termination_reason", contract["teacher_termination_reason"])),
                "best_single_slot2_delta_return": float(contract["best_single_slot2"].get("delta_return_vs_teacher", 0.0)),
                "best_single_slot2_delta_success": float(contract["best_single_slot2"].get("delta_success_vs_teacher", 0.0)),
                "best_single_slot2_return_r0": float(contract["best_single_slot2"].get("return_r0", contract["teacher_return_r0"])),
                "best_single_slot3_action": json.dumps(contract["best_single_slot3"].get("bundle_actions", [])),
                "best_single_slot3_success": float(contract["best_single_slot3"].get("success", contract["teacher_success"])),
                "best_single_slot3_hit_round": contract["best_single_slot3"].get("hit_round", contract["teacher_hit_round"]),
                "best_single_slot3_hit_sample_index": contract["best_single_slot3"].get("hit_sample_index", contract["teacher_hit_sample_index"]),
                "best_single_slot3_budget_used": int(contract["best_single_slot3"].get("budget_used", contract["teacher_budget_used"])),
                "best_single_slot3_termination_reason": str(contract["best_single_slot3"].get("termination_reason", contract["teacher_termination_reason"])),
                "best_single_slot3_delta_return": float(contract["best_single_slot3"].get("delta_return_vs_teacher", 0.0)),
                "best_single_slot3_delta_success": float(contract["best_single_slot3"].get("delta_success_vs_teacher", 0.0)),
                "best_single_slot3_return_r0": float(contract["best_single_slot3"].get("return_r0", contract["teacher_return_r0"])),
                "best_single_any_action": json.dumps(contract["best_single_any"].get("bundle_actions", [])),
                "best_single_any_success": float(contract["best_single_any"].get("success", contract["teacher_success"])),
                "best_single_any_hit_round": contract["best_single_any"].get("hit_round", contract["teacher_hit_round"]),
                "best_single_any_hit_sample_index": contract["best_single_any"].get("hit_sample_index", contract["teacher_hit_sample_index"]),
                "best_single_any_budget_used": int(contract["best_single_any"].get("budget_used", contract["teacher_budget_used"])),
                "best_single_any_termination_reason": str(contract["best_single_any"].get("termination_reason", contract["teacher_termination_reason"])),
                "best_single_any_delta_return": float(contract["best_single_any"].get("delta_return_vs_teacher", 0.0)),
                "best_single_any_delta_success": float(contract["best_single_any"].get("delta_success_vs_teacher", 0.0)),
                "best_single_any_return_r0": float(contract["best_single_any"].get("return_r0", contract["teacher_return_r0"])),
                "best_two_any_pair": str(contract["best_two_any"].get("pair", "")),
                "best_two_any_action": json.dumps(contract["best_two_any"].get("bundle_actions", [])),
                "best_two_any_success": float(contract["best_two_any"].get("success", contract["teacher_success"])),
                "best_two_any_hit_round": contract["best_two_any"].get("hit_round", contract["teacher_hit_round"]),
                "best_two_any_hit_sample_index": contract["best_two_any"].get("hit_sample_index", contract["teacher_hit_sample_index"]),
                "best_two_any_budget_used": int(contract["best_two_any"].get("budget_used", contract["teacher_budget_used"])),
                "best_two_any_termination_reason": str(contract["best_two_any"].get("termination_reason", contract["teacher_termination_reason"])),
                "best_two_any_delta_return": float(contract["best_two_any"].get("delta_return_vs_teacher", 0.0)),
                "best_two_any_delta_success": float(contract["best_two_any"].get("delta_success_vs_teacher", 0.0)),
                "best_two_any_return_r0": float(contract["best_two_any"].get("return_r0", contract["teacher_return_r0"])),
                "best_bundle_action": json.dumps(contract["best_bundle"].get("bundle_actions", [])),
                "best_bundle_success": float(contract["best_bundle"].get("success", contract["teacher_success"])),
                "best_bundle_hit_round": contract["best_bundle"].get("hit_round", contract["teacher_hit_round"]),
                "best_bundle_hit_sample_index": contract["best_bundle"].get("hit_sample_index", contract["teacher_hit_sample_index"]),
                "best_bundle_budget_used": int(contract["best_bundle"].get("budget_used", contract["teacher_budget_used"])),
                "best_bundle_termination_reason": str(contract["best_bundle"].get("termination_reason", contract["teacher_termination_reason"])),
                "best_bundle_delta_return": float(contract["best_bundle"].get("delta_return_vs_teacher", 0.0)),
                "best_bundle_delta_success": float(contract["best_bundle"].get("delta_success_vs_teacher", 0.0)),
                "best_bundle_return_r0": float(contract["best_bundle"].get("return_r0", contract["teacher_return_r0"])),
                "single_eval_count": int(contract["single_eval_count"]),
                "two_eval_count": int(contract["two_eval_count"]),
                "bundle_eval_count": int(contract["bundle_eval_count"]),
                "unique_bundle_eval_count": int(contract["unique_bundle_eval_count"]),
            }
            rows.append(row)
            episode_audit_counts[int(episode_idx)] += 1

            round_hit = source_local is not None and int(source_local) in set(teacher_actions[:3])
            rollout.step_with_actions(
                [int(v) for v in teacher_actions[:3]],
                sample_types=[f"teacher_slot_{i}" for i in range(3)],
            )
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            if bool(round_hit):
                break

    state_df = pd.DataFrame(rows)
    state_path = output_dir / f"{panel_name}_state_level_headroom.csv"
    state_df.to_csv(state_path, index=False)

    panel_summary = _build_panel_summary(state_df, panel_name)
    panel_oracle = _build_panel_oracle_upper_bound(
        state_df=state_df,
        runtime=runtime,
        family=family,
        env=env,
        top_source_k=int(args.top_source_k),
        include_surrogate_features=bool(args.include_surrogate_features),
        paper_like_alpha=float(args.paper_like_alpha),
        paper_like_topk_fraction=float(args.paper_like_topk_fraction),
        paper_like_time_tol_min=float(args.paper_like_time_tol_min),
        soft_scenario_beta=float(args.soft_scenario_beta),
    )
    panel_summary["panel_oracle_upper_bound"] = {
        "teacher_baseline": panel_oracle["teacher_baseline"],
        "contracts": panel_oracle["contracts"],
    }
    panel_summary["episode_audit_counts"] = {str(k): int(v) for k, v in episode_audit_counts.items()}
    panel_summary["artifacts"] = {
        "state_level_headroom": str(state_path),
    }
    pd.DataFrame(
        [
            {"contract": k, **v}
            for k, v in panel_summary["contracts"].items()
        ]
    ).to_csv(output_dir / f"{panel_name}_panel_headroom_summary.csv", index=False)
    pd.DataFrame(panel_oracle["contracts"]).to_csv(output_dir / f"{panel_name}_panel_oracle_upper_bound.csv", index=False)
    panel_oracle["case_rows"]["teacher"].to_csv(output_dir / f"{panel_name}_teacher_baseline_case_rows.csv", index=False)
    for contract_name, df in panel_oracle["case_rows"].items():
        if contract_name == "teacher":
            continue
        df.to_csv(output_dir / f"{panel_name}_{contract_name}_oracle_case_rows.csv", index=False)
    panel_summary["artifacts"]["panel_oracle_upper_bound"] = str(output_dir / f"{panel_name}_panel_oracle_upper_bound.csv")
    panel_summary["artifacts"]["teacher_baseline_case_rows"] = str(output_dir / f"{panel_name}_teacher_baseline_case_rows.csv")
    return panel_summary


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    precheck_root = Path(args.precheck_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    runtime_b30, split_meta_b30 = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.runtime_split),
        num_rounds=int(args.num_rounds_b30),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=int(args.train_max_cases),
        train_cache_version=str(args.train_cache_version),
        case_limit=int(args.case_limit),
    )
    panel_b30_name = "exact136_B30" if str(args.runtime_split) == "exact136" else f"{args.runtime_split}_B30"
    state_selector_root = Path(str(args.state_selector_root)) if str(args.state_selector_root).strip() else None
    state_selector_b30 = None
    state_selector_b60 = None
    if state_selector_root is not None:
        state_selector_b30 = _load_panel_state_selector(state_selector_root, panel_b30_name)
    summary_b30 = _run_panel(
        panel_name=panel_b30_name,
        runtime=runtime_b30,
        family=teacher_family,
        env=env,
        args=args,
        output_dir=output_dir,
        state_selector=state_selector_b30,
    )

    runtime_b60, split_meta_b60 = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.runtime_split),
        num_rounds=int(args.num_rounds_b60),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=int(args.train_max_cases),
        train_cache_version=str(args.train_cache_version),
        case_limit=int(args.case_limit),
    )
    panel_b60_name = "exact136_B60" if str(args.runtime_split) == "exact136" else f"{args.runtime_split}_B60"
    if state_selector_root is not None:
        state_selector_b60 = _load_panel_state_selector(state_selector_root, panel_b60_name)
    summary_b60 = _run_panel(
        panel_name=panel_b60_name,
        runtime=runtime_b60,
        family=teacher_family,
        env=env,
        args=args,
        output_dir=output_dir,
        state_selector=state_selector_b60,
    )

    final_summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "device": str(device),
        "teacher_family": str(teacher_family),
        "teacher_decision": teacher_decision,
        "fixed_contract": {
            "posterior_family": str(teacher_family),
            "teacher_policy": "posterior_greedy",
            "runtime_split": str(args.runtime_split),
            "actions_per_round": int(args.actions_per_round),
            "num_rounds_b30": int(args.num_rounds_b30),
            "num_rounds_b60": int(args.num_rounds_b60),
            "candidate_pool_contract": f"posterior_top{int(args.candidate_topk)}_legal_unsampled_with_teacher_forced_in",
            "continuation_policy": "teacher_continuation",
        },
        "audit_config": {
            "audit_max_per_episode": int(args.audit_max_per_episode),
            "state_limit": int(args.state_limit),
            "candidate_topk": int(args.candidate_topk),
            "max_two_member_combos": int(args.max_two_member_combos),
            "max_bundle_permutations": int(args.max_bundle_permutations),
            "case_limit": int(args.case_limit),
            "shuffle_cases": bool(args.shuffle_cases),
            "paper_like_alpha": float(args.paper_like_alpha),
            "paper_like_topk_fraction": float(args.paper_like_topk_fraction),
            "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
            "soft_scenario_beta": float(args.soft_scenario_beta),
            "top_source_k": int(args.top_source_k),
            "include_surrogate_features": bool(args.include_surrogate_features),
            "state_selector_root": str(state_selector_root) if state_selector_root is not None else "",
            "state_selector_b30_count": int(len(state_selector_b30)) if state_selector_b30 is not None else 0,
            "state_selector_b60_count": int(len(state_selector_b60)) if state_selector_b60 is not None else 0,
        },
        "split_meta": {
            "b30": split_meta_b30,
            "b60": split_meta_b60,
        },
        "panel_summaries": {
            panel_b30_name: summary_b30,
            panel_b60_name: summary_b60,
        },
        "artifacts": {
            "val_b30_state_headroom": str(output_dir / f"{panel_b30_name}_state_level_headroom.csv"),
            "val_b60_state_headroom": str(output_dir / f"{panel_b60_name}_state_level_headroom.csv"),
            "val_b30_panel_summary": str(output_dir / f"{panel_b30_name}_panel_headroom_summary.csv"),
            "val_b60_panel_summary": str(output_dir / f"{panel_b60_name}_panel_headroom_summary.csv"),
        },
        "commands_hint": {
            "main": "python -m src.scripts.audit.run_spim_strict_set_bundle_headroom_audit --runtime-split val --output-dir <dir>",
        },
    }
    write_json(output_dir / "summary.json", final_summary)


if __name__ == "__main__":
    main()
