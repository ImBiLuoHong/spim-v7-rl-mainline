from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.run_spim_regret_upper_bound_gateability_audit import (
    _run_policy_rollout as trusted_slot3_rollout,
    _summarize_panel,
)
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
    DEFAULT_SOURCE_ROOT,
    _pick_topk_unsampled,
    _posterior_topk_unsampled,
    seed_everything,
)


RUNNER_VERSION = "spim_runtime_semantic_panel_oracle_abc_v1"
PANEL_VERSION = "strict_val_runtime_semantic_panel_oracle_abc_v1"


CONTRACTS = [
    "slot1_only",
    "slot2_only",
    "slot3_only",
    "single_any",
    "two_member_any",
    "bundle_full",
]


def _resolve_contracts(raw: str) -> List[str]:
    text = str(raw).strip()
    if not text or text.lower() == "all":
        return list(CONTRACTS)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out: List[str] = []
    seen = set()
    for p in parts:
        if p not in CONTRACTS:
            raise ValueError(f"Unknown contract '{p}'. Supported: {', '.join(CONTRACTS)}")
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    if not out:
        raise ValueError("No valid contracts resolved from --contracts.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict held-out runtime-semantic panel oracle A/B/C with single intervention per case."
    )
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--runtime-split", type=str, default="val", choices=["exact136", "train", "val", "test"])
    parser.add_argument("--teacher-family", type=str, default="hsr_soft_scenario_posterior_v3")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument(
        "--contracts",
        type=str,
        default="all",
        help=f"Comma-separated subset from: {','.join(CONTRACTS)}. Use 'all' for full set.",
    )
    parser.add_argument("--skip-trusted-slot3-reference", action="store_true")

    parser.add_argument("--num-rounds-b30", type=int, default=10)
    parser.add_argument("--num-rounds-b60", type=int, default=20)
    parser.add_argument("--actions-per-round", type=int, default=3)

    parser.add_argument("--candidate-topk", type=int, default=6)
    parser.add_argument("--max-two-member-combos", type=int, default=0, help="0 means full")
    parser.add_argument("--max-bundle-permutations", type=int, default=0, help="0 means full")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--state-selector-root",
        type=str,
        default="",
        help="Optional directory containing {panel}_slot3_state_summary.csv or {panel}_state_manifest.csv; only these state_key values are audited.",
    )

    parser.add_argument("--top-source-k", type=int, default=8)
    parser.add_argument("--include-surrogate-features", action="store_true")
    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)
    return parser.parse_args()


def _state_key(case_id: str, episode_idx: int) -> str:
    return f"teacher::{case_id}::ep{int(episode_idx)}"


def _same_hit_round(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return int(a) == int(b)


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


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


def _candidate_pool(*, belief_ctx: Dict[str, Any], rollout: Any, teacher_actions: Sequence[int], topk: int) -> List[int]:
    pool = _posterior_topk_unsampled(
        belief=belief_ctx["belief"],
        candidate_mask=belief_ctx["candidate_mask"],
        rollout=rollout,
        topk=max(int(topk), 3),
    )
    ordered = [int(v) for v in pool]
    cap = max(int(topk), 3)
    for a in [int(teacher_actions[0]), int(teacher_actions[1]), int(teacher_actions[2])]:
        if a not in ordered:
            ordered.insert(0, a)
    out: List[int] = []
    seen = set()
    for v in ordered:
        if int(v) in seen:
            continue
        seen.add(int(v))
        out.append(int(v))
    return out[:cap]


def _enumerate_contract_bundles(
    *,
    teacher_bundle: Sequence[int],
    pool: Sequence[int],
    contract: str,
    max_two_member_combos: int,
    max_bundle_permutations: int,
) -> List[List[int]]:
    t = [int(teacher_bundle[0]), int(teacher_bundle[1]), int(teacher_bundle[2])]
    p = [int(v) for v in pool]
    rows: List[List[int]] = []

    if contract in {"slot1_only", "slot2_only", "slot3_only"}:
        slot = {"slot1_only": 0, "slot2_only": 1, "slot3_only": 2}[contract]
        fixed = {t[i] for i in [0, 1, 2] if i != slot}
        for cand in p:
            if cand in fixed or cand == t[slot]:
                continue
            bundle = list(t)
            bundle[slot] = int(cand)
            if len(set(bundle)) == 3:
                rows.append(bundle)
    elif contract == "single_any":
        for slot in [0, 1, 2]:
            fixed = {t[i] for i in [0, 1, 2] if i != slot}
            for cand in p:
                if cand in fixed or cand == t[slot]:
                    continue
                bundle = list(t)
                bundle[slot] = int(cand)
                if len(set(bundle)) == 3:
                    rows.append(bundle)
    elif contract == "two_member_any":
        for pair in [(0, 1), (0, 2), (1, 2)]:
            fixed_idx = [i for i in [0, 1, 2] if i not in pair][0]
            fixed_val = t[fixed_idx]
            pair_rows: List[List[int]] = []
            for c0 in p:
                for c1 in p:
                    a = int(c0)
                    b = int(c1)
                    if a == b:
                        continue
                    if fixed_val in {a, b}:
                        continue
                    bundle = list(t)
                    bundle[pair[0]] = a
                    bundle[pair[1]] = b
                    if bundle[pair[0]] == t[pair[0]] or bundle[pair[1]] == t[pair[1]]:
                        continue
                    if len(set(bundle)) == 3:
                        pair_rows.append(bundle)
            if int(max_two_member_combos) > 0:
                pair_rows = pair_rows[: int(max_two_member_combos)]
            rows.extend(pair_rows)
    elif contract == "bundle_full":
        perms = list(itertools.permutations(p, 3))
        if int(max_bundle_permutations) > 0:
            perms = perms[: int(max_bundle_permutations)]
        for perm in perms:
            rows.append([int(perm[0]), int(perm[1]), int(perm[2])])
    else:
        raise ValueError(f"Unsupported contract: {contract}")

    dedup: List[List[int]] = []
    seen = set()
    for b in rows:
        key = (int(b[0]), int(b[1]), int(b[2]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append([int(b[0]), int(b[1]), int(b[2])])
    return dedup


def _collect_candidate_bundles(
    *,
    teacher_bundle: Sequence[int],
    pool: Sequence[int],
    active_contracts: Sequence[str],
    max_two_member_combos: int,
    max_bundle_permutations: int,
) -> Dict[str, List[List[int]]]:
    out: Dict[str, List[List[int]]] = {}
    for contract in active_contracts:
        out[contract] = _enumerate_contract_bundles(
            teacher_bundle=teacher_bundle,
            pool=pool,
            contract=contract,
            max_two_member_combos=int(max_two_member_combos),
            max_bundle_permutations=int(max_bundle_permutations),
        )
    return out


def _pick_best_row(rows: List[Dict[str, Any]], delta_ret_key: str, delta_succ_key: str) -> Dict[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda r: (float(r[delta_ret_key]), float(r[delta_succ_key])),
        reverse=True,
    )[0]


def _load_panel_state_selector(state_selector_root: Path, panel_name: str) -> List[str]:
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
        return [str(v) for v in df["state_key"].astype(str).tolist()]
    return []


def _bundle_matches_contract(bundle: Sequence[int], teacher: Sequence[int], contract: str) -> bool:
    b = [int(bundle[0]), int(bundle[1]), int(bundle[2])]
    t = [int(teacher[0]), int(teacher[1]), int(teacher[2])]
    diffs = [int(b[i] != t[i]) for i in [0, 1, 2]]
    diff_count = int(sum(diffs))

    if contract == "slot1_only":
        return bool(diffs[0] == 1 and diffs[1] == 0 and diffs[2] == 0)
    if contract == "slot2_only":
        return bool(diffs[0] == 0 and diffs[1] == 1 and diffs[2] == 0)
    if contract == "slot3_only":
        return bool(diffs[0] == 0 and diffs[1] == 0 and diffs[2] == 1)
    if contract == "single_any":
        return bool(diff_count == 1)
    if contract == "two_member_any":
        return bool(diff_count == 2)
    if contract == "bundle_full":
        return True
    raise ValueError(f"Unsupported contract: {contract}")


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
    active_contracts: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    teacher_bundle = [int(teacher_actions[0]), int(teacher_actions[1]), int(teacher_actions[2])]
    eval_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    def eval_bundle(actions3: Sequence[int]) -> Dict[str, Any]:
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
            "bundle_actions": [int(actions3[0]), int(actions3[1]), int(actions3[2])],
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

    bundles_by_contract = _collect_candidate_bundles(
        teacher_bundle=teacher_bundle,
        pool=pool,
        active_contracts=active_contracts,
        max_two_member_combos=int(max_two_member_combos),
        max_bundle_permutations=int(max_bundle_permutations),
    )

    results: Dict[str, Dict[str, Any]] = {}
    workload: Dict[str, int] = {}
    for contract in active_contracts:
        cand_rows: List[Dict[str, Any]] = []
        for bundle in bundles_by_contract.get(contract, []):
            metrics = eval_bundle(bundle)
            cand_rows.append(
                {
                    **metrics,
                    "delta_return_vs_teacher": float(metrics["return_r0"] - base_ret),
                    "delta_success_vs_teacher": float(metrics["success"] - base_succ),
                }
            )
        best = _pick_best_row(cand_rows, "delta_return_vs_teacher", "delta_success_vs_teacher")
        if not best:
            best = {
                **teacher_eval,
                "delta_return_vs_teacher": 0.0,
                "delta_success_vs_teacher": 0.0,
            }
        results[contract] = best
        workload[contract] = int(len(cand_rows))

    results["teacher"] = {
        **teacher_eval,
        "delta_return_vs_teacher": 0.0,
        "delta_success_vs_teacher": 0.0,
    }
    results["workload"] = workload
    results["unique_bundle_eval_count"] = int(len(eval_cache))
    return results


def _simulate_case_with_single_intervention_plan(
    *,
    case: Any,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    plan_state_key: Optional[str],
    plan_bundle: Optional[Sequence[int]],
    top_source_k: int,
    include_surrogate_features: bool,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> Dict[str, Any]:
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]
    gate = DynamicReachabilityRuleModule()

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
    intervention_applied = False

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
        if (not intervention_applied) and plan_state_key is not None and key == str(plan_state_key) and plan_bundle is not None:
            b = [int(plan_bundle[0]), int(plan_bundle[1]), int(plan_bundle[2])]
            selected_actions[:3] = b
            intervention_applied = True

        round_hit = source_local is not None and int(source_local) in set(selected_actions)
        if round_hit and hit_round is None:
            hit_round = int(episode_idx)
            source_slot = selected_actions.index(int(source_local)) + 1
            hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

        total_reward += float((-1.0 / 30.0) * float(len(selected_actions)) + (1.0 if bool(round_hit) else 0.0))
        rollout.step_with_actions(
            selected_actions,
            sample_types=[f"runtime_semantic_oracle_slot_{i}" for i in range(len(selected_actions))],
        )
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])
        budget_used += int(len(selected_actions))
        if round_hit:
            termination_reason = "source_hit"
            break

    return {
        "case_id": str(case.case_id),
        "success": float(hit_round is not None),
        "hit_round": None if hit_round is None else int(hit_round),
        "hit_sample_index": None if hit_sample_index is None else int(hit_sample_index),
        "budget_used": int(budget_used),
        "return_r0": float(total_reward),
        "termination_reason": str(termination_reason),
        "planned_intervention": float(plan_state_key is not None),
        "applied_intervention": float(intervention_applied),
    }


def _run_panel(
    *,
    panel_name: str,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    args: argparse.Namespace,
    output_dir: Path,
    state_selector: Optional[set[str]],
    active_contracts: Sequence[str],
) -> Dict[str, Any]:
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]
    gate = DynamicReachabilityRuleModule()

    state_rows: List[Dict[str, Any]] = []
    teacher_case_rows: List[Dict[str, Any]] = []
    contract_case_rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in active_contracts}

    contract_case_plan: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k in active_contracts}
    trusted_slot3_map: Dict[str, int] = {}

    for case in runtime["cases"]:
        rollout = _make_rollout(runtime, case)
        history = ObservationWitnessHistory()
        paper_state = PaperLikeHSRState(source_prior=None)
        trigger_global = _extract_trigger_global(case.data)
        source_local = resolve_source_local_idx(rollout)

        teacher_hit_round: Optional[int] = None
        teacher_hit_sample_index: Optional[int] = None
        teacher_return_r0 = 0.0
        teacher_budget_used = 0
        teacher_termination_reason = "budget_exhausted"

        per_case_best_state: Dict[str, Dict[str, Any]] = {}

        for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
            pre_rollout = deepcopy(rollout)
            pre_history = deepcopy(history)
            pre_paper_state = deepcopy(paper_state)

            state, belief_ctx, spim_state, _ = _compute_step_belief(
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
            if int(state["valid_mask"].sum().item()) <= 0:
                teacher_termination_reason = "no_valid_nodes"
                break

            teacher_actions = _pick_topk_unsampled(
                belief_ctx["belief"],
                belief_ctx["candidate_mask"],
                rollout,
                int(runtime["action_budget"]),
            )
            teacher_actions = [int(v) for v in teacher_actions]
            if len(teacher_actions) <= 0:
                teacher_termination_reason = "teacher_no_action"
                break

            key = _state_key(str(case.case_id), int(episode_idx))
            should_audit_state = True
            if state_selector is not None and len(state_selector) > 0:
                should_audit_state = bool(key in state_selector)

            can_audit_state = False
            pool: List[int] = []
            if should_audit_state and len(teacher_actions) >= 3:
                pool = _candidate_pool(
                    belief_ctx=belief_ctx,
                    rollout=rollout,
                    teacher_actions=teacher_actions,
                    topk=int(args.candidate_topk),
                )
                can_audit_state = len(pool) >= 3

            if not can_audit_state:
                round_hit = source_local is not None and int(source_local) in set(teacher_actions[:3])
                if round_hit and teacher_hit_round is None:
                    teacher_hit_round = int(episode_idx)
                    source_slot = teacher_actions[:3].index(int(source_local)) + 1
                    teacher_hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))
                teacher_return_r0 += float((-1.0 / 30.0) * float(len(teacher_actions)) + (1.0 if bool(round_hit) else 0.0))
                rollout.step_with_actions(
                    [int(v) for v in teacher_actions],
                    sample_types=[f"teacher_slot_{i}" for i in range(len(teacher_actions))],
                )
                if rollout.history_steps:
                    history.append_from_history_step(rollout.history_steps[-1])
                teacher_budget_used += int(len(teacher_actions))
                if round_hit:
                    teacher_termination_reason = "source_hit"
                    break
                continue

            state_eval = _evaluate_state_contracts(
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
                active_contracts=active_contracts,
            )

            row = {
                "panel": str(panel_name),
                "case_id": str(case.case_id),
                "state_key": str(key),
                "episode_index": int(episode_idx),
                "teacher_actions": json.dumps([int(v) for v in teacher_actions[:3]]),
                "candidate_pool": json.dumps([int(v) for v in pool]),
                "candidate_pool_size": int(len(pool)),
                "teacher_success": float(state_eval["teacher"]["success"]),
                "teacher_return_r0": float(state_eval["teacher"]["return_r0"]),
                "teacher_hit_round": state_eval["teacher"]["hit_round"],
                "teacher_hit_sample_index": state_eval["teacher"]["hit_sample_index"],
                "teacher_budget_used": int(state_eval["teacher"]["budget_used"]),
                "teacher_termination_reason": str(state_eval["teacher"]["termination_reason"]),
                "remaining_budget": int(spim_state["diagnostics"]["remaining_budget"]),
                "posterior_entropy": float(spim_state["diagnostics"]["posterior_entropy"]),
                "top1_top2_margin": float(spim_state["diagnostics"]["top1_top2_margin"]),
                "single_eval_count": int(
                    state_eval["workload"].get("slot1_only", 0)
                    + state_eval["workload"].get("slot2_only", 0)
                    + state_eval["workload"].get("slot3_only", 0)
                ),
                "single_any_eval_count": int(state_eval["workload"].get("single_any", 0)),
                "two_eval_count": int(state_eval["workload"].get("two_member_any", 0)),
                "bundle_eval_count": int(state_eval["workload"].get("bundle_full", 0)),
                "unique_bundle_eval_count": int(state_eval["unique_bundle_eval_count"]),
            }

            for contract in active_contracts:
                best = state_eval[contract]
                row[f"{contract}_best_action"] = json.dumps(best.get("bundle_actions", []))
                row[f"{contract}_best_success"] = float(best.get("success", row["teacher_success"]))
                row[f"{contract}_best_hit_round"] = best.get("hit_round", row["teacher_hit_round"])
                row[f"{contract}_best_hit_sample_index"] = best.get("hit_sample_index", row["teacher_hit_sample_index"])
                row[f"{contract}_best_budget_used"] = int(best.get("budget_used", row["teacher_budget_used"]))
                row[f"{contract}_best_termination_reason"] = str(
                    best.get("termination_reason", row["teacher_termination_reason"])
                )
                row[f"{contract}_best_return_r0"] = float(best.get("return_r0", row["teacher_return_r0"]))
                row[f"{contract}_delta_return_vs_teacher"] = float(best.get("delta_return_vs_teacher", 0.0))
                row[f"{contract}_delta_success_vs_teacher"] = float(best.get("delta_success_vs_teacher", 0.0))

                current = per_case_best_state.get(contract)
                candidate = {
                    "state_key": str(key),
                    "episode_index": int(episode_idx),
                    "bundle_actions": best.get("bundle_actions", []),
                    "delta_return_vs_teacher": float(best.get("delta_return_vs_teacher", 0.0)),
                    "delta_success_vs_teacher": float(best.get("delta_success_vs_teacher", 0.0)),
                }
                if current is None:
                    per_case_best_state[contract] = candidate
                else:
                    ck = (
                        float(candidate["delta_return_vs_teacher"]),
                        float(candidate["delta_success_vs_teacher"]),
                        -int(candidate["episode_index"]),
                    )
                    pk = (
                        float(current["delta_return_vs_teacher"]),
                        float(current["delta_success_vs_teacher"]),
                        -int(current["episode_index"]),
                    )
                    if ck > pk:
                        per_case_best_state[contract] = candidate

            if "slot3_only" in active_contracts and float(state_eval["slot3_only"].get("delta_return_vs_teacher", 0.0)) > float(args.eps):
                slot3_best_actions = state_eval["slot3_only"].get("bundle_actions", [])
                if isinstance(slot3_best_actions, list) and len(slot3_best_actions) >= 3:
                    trusted_slot3_map[str(key)] = int(slot3_best_actions[2])

            state_rows.append(row)

            round_hit = source_local is not None and int(source_local) in set(teacher_actions[:3])
            if round_hit and teacher_hit_round is None:
                teacher_hit_round = int(episode_idx)
                source_slot = teacher_actions[:3].index(int(source_local)) + 1
                teacher_hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))
            teacher_return_r0 += float((-1.0 / 30.0) * float(len(teacher_actions)) + (1.0 if bool(round_hit) else 0.0))
            rollout.step_with_actions(
                [int(v) for v in teacher_actions],
                sample_types=[f"teacher_slot_{i}" for i in range(len(teacher_actions))],
            )
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            teacher_budget_used += int(len(teacher_actions))
            if round_hit:
                teacher_termination_reason = "source_hit"
                break

        teacher_case_rows.append(
            {
                "case_id": str(case.case_id),
                "success": float(teacher_hit_round is not None),
                "hit_round": None if teacher_hit_round is None else int(teacher_hit_round),
                "hit_sample_index": None if teacher_hit_sample_index is None else int(teacher_hit_sample_index),
                "budget_used": int(teacher_budget_used),
                "return_r0": float(teacher_return_r0),
                "termination_reason": str(teacher_termination_reason),
            }
        )

        for contract in active_contracts:
            best_state = per_case_best_state.get(contract)
            if best_state is None or float(best_state["delta_return_vs_teacher"]) <= float(args.eps):
                contract_case_plan[contract][str(case.case_id)] = {
                    "state_key": None,
                    "bundle_actions": None,
                    "planned": False,
                    "delta_return_vs_teacher": 0.0,
                    "delta_success_vs_teacher": 0.0,
                }
            else:
                bundle = [int(v) for v in best_state.get("bundle_actions", [])]
                if len(bundle) >= 3:
                    contract_case_plan[contract][str(case.case_id)] = {
                        "state_key": str(best_state["state_key"]),
                        "bundle_actions": bundle[:3],
                        "planned": True,
                        "delta_return_vs_teacher": float(best_state["delta_return_vs_teacher"]),
                        "delta_success_vs_teacher": float(best_state["delta_success_vs_teacher"]),
                    }
                else:
                    contract_case_plan[contract][str(case.case_id)] = {
                        "state_key": None,
                        "bundle_actions": None,
                        "planned": False,
                        "delta_return_vs_teacher": 0.0,
                        "delta_success_vs_teacher": 0.0,
                    }

    teacher_df = pd.DataFrame(teacher_case_rows)

    for case in runtime["cases"]:
        case_id = str(case.case_id)
        for contract in active_contracts:
            plan = contract_case_plan[contract][case_id]
            out = _simulate_case_with_single_intervention_plan(
                case=case,
                runtime=runtime,
                family=str(family),
                env=env,
                plan_state_key=plan.get("state_key"),
                plan_bundle=plan.get("bundle_actions"),
                top_source_k=int(args.top_source_k),
                include_surrogate_features=bool(args.include_surrogate_features),
                paper_like_alpha=float(args.paper_like_alpha),
                paper_like_topk_fraction=float(args.paper_like_topk_fraction),
                paper_like_time_tol_min=float(args.paper_like_time_tol_min),
                soft_scenario_beta=float(args.soft_scenario_beta),
            )
            out["planned_intervention"] = float(bool(plan.get("planned", False)))
            out["plan_state_key"] = "" if plan.get("state_key") is None else str(plan.get("state_key"))
            out["plan_bundle_actions"] = json.dumps(plan.get("bundle_actions") if plan.get("bundle_actions") is not None else [])
            out["plan_delta_return_vs_teacher"] = float(plan.get("delta_return_vs_teacher", 0.0))
            out["plan_delta_success_vs_teacher"] = float(plan.get("delta_success_vs_teacher", 0.0))
            contract_case_rows[contract].append(out)

    contracts_panel_rows: List[Dict[str, Any]] = []
    teacher_summary = _summarize_panel(teacher_df, num_rounds=int(runtime["num_episodes"]), action_budget=int(runtime["action_budget"]))

    contract_df_map: Dict[str, pd.DataFrame] = {}
    for contract in active_contracts:
        oracle_df = pd.DataFrame(contract_case_rows[contract])
        contract_df_map[contract] = oracle_df
        oracle_summary = _summarize_panel(oracle_df, num_rounds=int(runtime["num_episodes"]), action_budget=int(runtime["action_budget"]))
        change = _compute_case_change_metrics(teacher_df, oracle_df)
        contracts_panel_rows.append(
            {
                "contract": str(contract),
                "teacher_success_rate": float(teacher_summary["success_rate"]),
                "oracle_success_rate": float(oracle_summary["success_rate"]),
                "delta_success": float(oracle_summary["success_rate"] - teacher_summary["success_rate"]),
                "teacher_avg_hit_round_conditional": teacher_summary["avg_hit_round_conditional"],
                "oracle_avg_hit_round_conditional": oracle_summary["avg_hit_round_conditional"],
                "delta_hit_round": (
                    None
                    if oracle_summary["avg_hit_round_conditional"] is None or teacher_summary["avg_hit_round_conditional"] is None
                    else float(oracle_summary["avg_hit_round_conditional"] - teacher_summary["avg_hit_round_conditional"])
                ),
                "teacher_avg_return_r0": float(teacher_summary["avg_return_r0"]),
                "oracle_avg_return_r0": float(oracle_summary["avg_return_r0"]),
                "delta_return_r0": float(oracle_summary["avg_return_r0"] - teacher_summary["avg_return_r0"]),
                "result_changed_episode_fraction": float(change["result_changed_episode_fraction"]),
                "success_flip_episode_fraction": float(change["success_flip_episode_fraction"]),
                "success_flip_down_episode_fraction": float(change["success_flip_down_episode_fraction"]),
                "planned_intervention_fraction": float(oracle_df["planned_intervention"].astype(float).mean()) if len(oracle_df) else 0.0,
                "applied_intervention_fraction": float(oracle_df["applied_intervention"].astype(float).mean()) if len(oracle_df) else 0.0,
            }
        )

    trusted_slot3_df = pd.DataFrame()
    trusted_slot3_summary: Dict[str, Any] = {"case_count": int(len(teacher_df))}
    trusted_slot3_change: Dict[str, float] = {
        "result_changed_episode_fraction": 0.0,
        "success_flip_episode_fraction": 0.0,
        "success_flip_down_episode_fraction": 0.0,
    }
    if not bool(args.skip_trusted_slot3_reference):
        trusted_slot3_df = trusted_slot3_rollout(
            runtime=runtime,
            family=str(family),
            env=env,
            action_budget=int(runtime["action_budget"]),
            top_source_k=int(args.top_source_k),
            include_surrogate_features=bool(args.include_surrogate_features),
            correction_slot3_map=trusted_slot3_map,
        )
        trusted_slot3_summary = _summarize_panel(
            trusted_slot3_df,
            num_rounds=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
        )
        trusted_slot3_change = _compute_case_change_metrics(teacher_df, trusted_slot3_df)

    state_df = pd.DataFrame(state_rows)
    contract_df = pd.DataFrame(contracts_panel_rows)

    slot3_row = contract_df.loc[contract_df["contract"].astype(str) == "slot3_only"]
    slot3_delta_success = float(slot3_row.iloc[0]["delta_success"]) if len(slot3_row) else 0.0
    slot3_delta_return = float(slot3_row.iloc[0]["delta_return_r0"]) if len(slot3_row) else 0.0

    extra_rows: List[Dict[str, Any]] = []
    for k in ["single_any", "two_member_any", "bundle_full"]:
        if k not in active_contracts:
            continue
        sub = contract_df.loc[contract_df["contract"].astype(str) == k]
        if len(sub) <= 0:
            continue
        extra_rows.append(
            {
                "contract": str(k),
                "extra_delta_success_vs_slot3_only": float(sub.iloc[0]["delta_success"] - slot3_delta_success),
                "extra_delta_return_r0_vs_slot3_only": float(sub.iloc[0]["delta_return_r0"] - slot3_delta_return),
            }
        )
    extra_df = pd.DataFrame(extra_rows)

    panel_prefix = output_dir / panel_name
    state_df.to_csv(panel_prefix.with_name(f"{panel_name}_state_level_runtime_semantic.csv"), index=False)
    teacher_df.to_csv(panel_prefix.with_name(f"{panel_name}_teacher_case_rows.csv"), index=False)
    contract_df.to_csv(panel_prefix.with_name(f"{panel_name}_panel_oracle_contracts.csv"), index=False)
    extra_df.to_csv(panel_prefix.with_name(f"{panel_name}_extra_headroom_vs_slot3_only.csv"), index=False)
    if not bool(args.skip_trusted_slot3_reference):
        trusted_slot3_df.to_csv(panel_prefix.with_name(f"{panel_name}_slot3_trusted_reference_case_rows.csv"), index=False)

    for contract, cdf in contract_df_map.items():
        cdf.to_csv(panel_prefix.with_name(f"{panel_name}_{contract}_case_rows.csv"), index=False)

    panel_summary = {
        "panel": str(panel_name),
        "teacher_baseline": teacher_summary,
        "contracts": contracts_panel_rows,
        "extra_headroom_vs_slot3_only": extra_rows,
        "slot3_only_trusted_reference": {
            **trusted_slot3_summary,
            "delta_success_vs_teacher": (
                None
                if "success_rate" not in trusted_slot3_summary
                else float(trusted_slot3_summary["success_rate"] - teacher_summary["success_rate"])
            ),
            "delta_return_r0_vs_teacher": (
                None
                if "avg_return_r0" not in trusted_slot3_summary
                else float(trusted_slot3_summary["avg_return_r0"] - teacher_summary["avg_return_r0"])
            ),
            "result_changed_episode_fraction": float(trusted_slot3_change["result_changed_episode_fraction"]),
            "success_flip_episode_fraction": float(trusted_slot3_change["success_flip_episode_fraction"]),
            "success_flip_down_episode_fraction": float(trusted_slot3_change["success_flip_down_episode_fraction"]),
            "trusted_positive_state_count": int(len(trusted_slot3_map)),
            "enabled": bool(not args.skip_trusted_slot3_reference),
        },
        "workload": {
            "state_count": int(len(state_df)),
            "single_eval_count_total": int(state_df["single_eval_count"].sum()) if len(state_df) else 0,
            "single_any_eval_count_total": int(state_df["single_any_eval_count"].sum()) if len(state_df) else 0,
            "two_eval_count_total": int(state_df["two_eval_count"].sum()) if len(state_df) else 0,
            "bundle_eval_count_total": int(state_df["bundle_eval_count"].sum()) if len(state_df) else 0,
            "unique_bundle_eval_count_total": int(state_df["unique_bundle_eval_count"].sum()) if len(state_df) else 0,
        },
        "artifacts": {
            "state_level_runtime_semantic": str(panel_prefix.with_name(f"{panel_name}_state_level_runtime_semantic.csv")),
            "teacher_case_rows": str(panel_prefix.with_name(f"{panel_name}_teacher_case_rows.csv")),
            "panel_oracle_contracts": str(panel_prefix.with_name(f"{panel_name}_panel_oracle_contracts.csv")),
            "extra_headroom_vs_slot3_only": str(panel_prefix.with_name(f"{panel_name}_extra_headroom_vs_slot3_only.csv")),
            "slot3_trusted_reference_case_rows": str(panel_prefix.with_name(f"{panel_name}_slot3_trusted_reference_case_rows.csv")),
        },
    }
    return panel_summary


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    active_contracts = _resolve_contracts(str(args.contracts))

    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = CleanTwoChannelEvidenceEnv()

    runtime_b30, split_meta_b30 = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.runtime_split),
        num_rounds=int(args.num_rounds_b30),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=0,
        train_cache_version="",
        case_limit=int(args.case_limit),
    )
    runtime_b60, split_meta_b60 = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.runtime_split),
        num_rounds=int(args.num_rounds_b60),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=0,
        train_cache_version="",
        case_limit=int(args.case_limit),
    )
    panel_b30 = "exact136_B30" if str(args.runtime_split) == "exact136" else f"{args.runtime_split}_B30"
    panel_b60 = "exact136_B60" if str(args.runtime_split) == "exact136" else f"{args.runtime_split}_B60"
    selector_root = Path(str(args.state_selector_root)) if str(args.state_selector_root).strip() else None
    selector_b30 = set(_load_panel_state_selector(selector_root, panel_b30)) if selector_root is not None else None
    selector_b60 = set(_load_panel_state_selector(selector_root, panel_b60)) if selector_root is not None else None

    summary_b30 = _run_panel(
        panel_name=panel_b30,
        runtime=runtime_b30,
        family=str(args.teacher_family),
        env=env,
        args=args,
        output_dir=output_dir,
        state_selector=selector_b30,
        active_contracts=active_contracts,
    )
    summary_b60 = _run_panel(
        panel_name=panel_b60,
        runtime=runtime_b60,
        family=str(args.teacher_family),
        env=env,
        args=args,
        output_dir=output_dir,
        state_selector=selector_b60,
        active_contracts=active_contracts,
    )

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "fixed_contract": {
            "posterior_family": str(args.teacher_family),
            "teacher_policy": "posterior_greedy",
            "runtime_split": str(args.runtime_split),
            "actions_per_round": int(args.actions_per_round),
            "num_rounds_b30": int(args.num_rounds_b30),
            "num_rounds_b60": int(args.num_rounds_b60),
            "candidate_pool": f"posterior_top{int(args.candidate_topk)}_legal_unsampled_with_teacher_forced_in",
            "continuation_policy": "teacher_continuation",
            "intervention_budget": "single_intervention_per_case",
            "positive_only_rule": "best_delta_return_vs_teacher_current_set > eps",
            "contracts": list(active_contracts),
            "all_supported_contracts": list(CONTRACTS),
            "slot3_trusted_reference": "legacy_slot3_runner_with_positive_state_map",
        },
        "audit_config": {
            "eps": float(args.eps),
            "case_limit": int(args.case_limit),
            "max_two_member_combos": int(args.max_two_member_combos),
            "max_bundle_permutations": int(args.max_bundle_permutations),
            "top_source_k": int(args.top_source_k),
            "include_surrogate_features": bool(args.include_surrogate_features),
            "paper_like_alpha": float(args.paper_like_alpha),
            "paper_like_topk_fraction": float(args.paper_like_topk_fraction),
            "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
            "soft_scenario_beta": float(args.soft_scenario_beta),
            "state_selector_root": str(selector_root) if selector_root is not None else "",
            "state_selector_b30_count": int(len(selector_b30)) if selector_b30 is not None else 0,
            "state_selector_b60_count": int(len(selector_b60)) if selector_b60 is not None else 0,
        },
        "split_meta": {
            "b30": split_meta_b30,
            "b60": split_meta_b60,
        },
        "panel_summaries": {
            panel_b30: summary_b30,
            panel_b60: summary_b60,
        },
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "b30_contracts": str(output_dir / f"{panel_b30}_panel_oracle_contracts.csv"),
            "b60_contracts": str(output_dir / f"{panel_b60}_panel_oracle_contracts.csv"),
        },
        "commands_hint": {
            "main": "python -m src.scripts.audit.run_spim_runtime_semantic_panel_oracle_abc --runtime-split val --output-dir <dir>",
        },
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
