from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

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
    DEFAULT_SOURCE_ROOT,
    _pick_topk_unsampled,
    _posterior_topk_unsampled,
    seed_everything,
)

RUNNER_VERSION = "spim_oracle_headroom_ladder_v1"
PANEL_VERSION = "strict_val_b30_oracle_headroom_ladder_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-level oracle headroom ladder under fixed SPIM v3 + posterior_greedy contract.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--teacher-family", type=str, default="hsr_soft_scenario_posterior_v3")
    parser.add_argument("--runtime-split", type=str, default="val", choices=["exact136", "train", "val", "test"])

    parser.add_argument("--run-b30", action="store_true", default=True)
    parser.add_argument("--run-b60", action="store_true")
    parser.add_argument("--b60-case-limit", type=int, default=0, help="0 means full split")

    parser.add_argument("--actions-per-round", type=int, default=3)

    parser.add_argument("--candidate-topk", type=int, default=5)
    parser.add_argument("--max-bundles-per-state", type=int, default=24)
    parser.add_argument("--level2-case-limit", type=int, default=0, help="0 means use full runtime case set for Level2.")

    parser.add_argument("--l3-lookahead-horizon", type=int, default=2)
    parser.add_argument("--l3-beam-width", type=int, default=8)
    parser.add_argument("--l3-tail-topk", type=int, default=12)

    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)
    parser.add_argument("--top-source-k", type=int, default=8)
    parser.add_argument("--include-surrogate-features", action="store_true")

    parser.add_argument(
        "--trusted-baseline-root",
        type=str,
        default="artifacts/runtime_semantic_panel_oracle_abc/20260413_val_full_slot3only_sel_v2_recon",
    )
    parser.add_argument("--smoke-case-limit", type=int, default=0)
    return parser.parse_args()


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(len(vals), 1))


def _state_key(case_id: str, episode_idx: int) -> str:
    return f"teacher::{case_id}::ep{int(episode_idx)}"


def _same_hit_round(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return int(a) == int(b)


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


def _compute_case_change_metrics(teacher_df: pd.DataFrame, policy_df: pd.DataFrame) -> Dict[str, float]:
    merged = teacher_df.merge(policy_df, on="case_id", suffixes=("_teacher", "_policy"), how="inner")
    if len(merged) <= 0:
        return {
            "result_changed_episode_fraction": 0.0,
            "success_flip_episode_fraction": 0.0,
            "success_flip_down_episode_fraction": 0.0,
            "success_flip_count": 0,
        }
    changed = []
    success_flip_up = []
    success_flip_down = []
    for _, r in merged.iterrows():
        success_t = float(r["success_teacher"])
        success_p = float(r["success_policy"])
        ret_t = float(r["return_r0_teacher"])
        ret_p = float(r["return_r0_policy"])
        hr_t = r["hit_round_teacher"]
        hr_p = r["hit_round_policy"]
        row_changed = (abs(ret_p - ret_t) > 1e-12) or (abs(success_p - success_t) > 1e-12) or (not _same_hit_round(hr_t, hr_p))
        changed.append(bool(row_changed))
        success_flip_up.append(bool(success_t < 0.5 and success_p > 0.5))
        success_flip_down.append(bool(success_t > 0.5 and success_p < 0.5))
    return {
        "result_changed_episode_fraction": float(sum(changed) / max(len(changed), 1)),
        "success_flip_episode_fraction": float(sum(success_flip_up) / max(len(success_flip_up), 1)),
        "success_flip_down_episode_fraction": float(sum(success_flip_down) / max(len(success_flip_down), 1)),
        "success_flip_count": int(sum(success_flip_up)),
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


def _enumerate_bundles(pool: Sequence[int], max_bundles: int) -> List[List[int]]:
    rows: List[List[int]] = []
    for a in pool:
        for b in pool:
            if int(b) == int(a):
                continue
            for c in pool:
                if int(c) in {int(a), int(b)}:
                    continue
                rows.append([int(a), int(b), int(c)])
    # prioritize by descending summed posterior mass is done by caller via ranking if needed.
    if int(max_bundles) > 0:
        rows = rows[: int(max_bundles)]
    return rows


def _policy_value_tail_greedy(probs: List[float], remaining_rounds: int, step_penalty: float, action_budget: int) -> float:
    if remaining_rounds <= 0:
        return 0.0
    p = [max(float(v), 0.0) for v in probs]
    s = sum(p)
    if s <= 1e-12:
        return 0.0
    p = [v / s for v in p]
    available = sorted(p, reverse=True)
    survival = 1.0
    out = 0.0
    idx = 0
    for _round in range(remaining_rounds):
        chunk = available[idx : idx + int(action_budget)]
        if not chunk:
            break
        chunk_mass = sum(chunk)
        out += survival * (float(step_penalty) * float(action_budget) + float(chunk_mass))
        survival *= max(1.0 - float(chunk_mass), 0.0)
        idx += int(action_budget)
        if survival <= 1e-12:
            break
    return float(out)


def _bundle_mass(bundle: Sequence[int], belief_vec: Sequence[float]) -> float:
    return float(sum(float(belief_vec[int(i)]) for i in bundle))


def _l3_best_bundle(
    *,
    belief_vec: List[float],
    candidate_pool: Sequence[int],
    action_budget: int,
    remaining_rounds: int,
    horizon: int,
    beam_width: int,
    max_bundles_per_state: int,
    step_penalty: float,
) -> List[int]:
    bundles = _enumerate_bundles(candidate_pool, max_bundles=max_bundles_per_state)
    if not bundles:
        top = sorted([(float(belief_vec[i]), int(i)) for i in candidate_pool], reverse=True)
        return [int(v[1]) for v in top[: int(action_budget)]]

    scored = []
    for b in bundles:
        mass = _bundle_mass(b, belief_vec)
        scored.append((float(mass), b))
    scored.sort(key=lambda x: x[0], reverse=True)
    trimmed = [b for _, b in scored[: max(int(beam_width), 1)]]

    def recurse(cur_probs: List[float], rem: int, depth_left: int) -> float:
        if rem <= 0:
            return 0.0
        total_mass = sum(cur_probs)
        if total_mass <= 1e-12:
            return 0.0
        norm_probs = [float(v) / float(total_mass) for v in cur_probs]
        if depth_left <= 0:
            return _policy_value_tail_greedy(norm_probs, rem, step_penalty=step_penalty, action_budget=action_budget)

        local = sorted([(norm_probs[int(i)], int(i)) for i in candidate_pool if norm_probs[int(i)] > 1e-12], reverse=True)
        local_pool = [int(v[1]) for v in local[: max(3, len(candidate_pool))]]
        cands = _enumerate_bundles(local_pool, max_bundles=max_bundles_per_state)
        if not cands:
            return _policy_value_tail_greedy(norm_probs, rem, step_penalty=step_penalty, action_budget=action_budget)
        cands = sorted(cands, key=lambda b: _bundle_mass(b, norm_probs), reverse=True)[: max(int(beam_width), 1)]

        best = -1e9
        for b in cands:
            p_hit = max(min(_bundle_mass(b, norm_probs), 1.0), 0.0)
            next_probs = list(norm_probs)
            for idx in b:
                next_probs[int(idx)] = 0.0
            immediate = float(step_penalty) * float(action_budget) + float(p_hit)
            cont = 0.0 if rem <= 1 else (1.0 - float(p_hit)) * recurse(next_probs, rem - 1, depth_left - 1)
            val = float(immediate + cont)
            if val > best:
                best = val
        return float(best)

    best_bundle = trimmed[0]
    best_val = -1e9
    for b in trimmed:
        p_hit = max(min(_bundle_mass(b, belief_vec), 1.0), 0.0)
        next_probs = list(belief_vec)
        for idx in b:
            next_probs[int(idx)] = 0.0
        immediate = float(step_penalty) * float(action_budget) + float(p_hit)
        cont = 0.0 if remaining_rounds <= 1 else (1.0 - float(p_hit)) * recurse(next_probs, remaining_rounds - 1, int(horizon) - 1)
        val = float(immediate + cont)
        if val > best_val:
            best_val = val
            best_bundle = b
    return [int(best_bundle[0]), int(best_bundle[1]), int(best_bundle[2])]


def _choose_level1_source_aware(
    *,
    source_local: Optional[int],
    teacher_actions: Sequence[int],
    belief_ctx: Dict[str, Any],
    rollout: Any,
) -> List[int]:
    selected = [int(v) for v in teacher_actions[:3]]
    if source_local is None:
        return selected
    src = int(source_local)
    candidate_mask = belief_ctx["candidate_mask"].view(-1).bool().cpu()
    legal = bool(src >= 0 and src < candidate_mask.numel() and candidate_mask[src].item()) and (not bool(rollout.revealed_mask[src].item()))
    if not legal:
        return selected
    if src in selected:
        return selected
    # Replace the weakest teacher slot with true source while preserving uniqueness.
    selected[-1] = int(src)
    if len(set(selected)) < 3:
        pool = _posterior_topk_unsampled(
            belief=belief_ctx["belief"],
            candidate_mask=belief_ctx["candidate_mask"],
            rollout=rollout,
            topk=12,
        )
        for cand in pool:
            c = int(cand)
            if c == int(src) or c in set(selected[:-1]):
                continue
            selected[1] = c
            if len(set(selected)) == 3:
                break
    return [int(selected[0]), int(selected[1]), int(selected[2])]


def _choose_level2_hindsight(
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
    max_bundles_per_state: int,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> Tuple[List[int], Dict[str, Any]]:
    eval_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    bundles = _enumerate_bundles(pool, max_bundles=max_bundles_per_state)
    teacher_bundle = [int(v) for v in teacher_actions[:3]]
    if tuple(teacher_bundle) not in {tuple(b) for b in bundles}:
        bundles = [teacher_bundle] + bundles

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
            "return_r0": float(out["return_r0"]),
        }
        return eval_cache[key]

    teacher_eval = eval_bundle(teacher_bundle)
    best = None
    for b in bundles:
        row = eval_bundle(b)
        tup = (float(row["return_r0"]), float(row["success"]))
        if best is None or tup > best[0]:
            best = (tup, row)
    assert best is not None
    chosen = [int(v) for v in best[1]["bundle_actions"]]
    meta = {
        "teacher_return_r0": float(teacher_eval["return_r0"]),
        "chosen_return_r0": float(best[1]["return_r0"]),
        "evaluated_bundle_count": int(len(eval_cache)),
    }
    return chosen, meta


def _run_policy_panel(
    *,
    policy_name: str,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    candidate_topk: int,
    max_bundles_per_state: int,
    l3_lookahead_horizon: int,
    l3_beam_width: int,
    l3_tail_topk: int,
    top_source_k: int,
    include_surrogate_features: bool,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]
    gate = DynamicReachabilityRuleModule()
    rows: List[Dict[str, Any]] = []

    level2_eval_count = 0

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
            pre_rollout = deepcopy(rollout)
            pre_history = deepcopy(history)
            pre_paper_state = deepcopy(paper_state)

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
            if len(teacher_actions) < int(runtime["action_budget"]):
                termination_reason = "teacher_no_action"
                break

            selected_actions = [int(v) for v in teacher_actions[:3]]
            pool = _candidate_pool(
                belief_ctx=belief_ctx,
                rollout=rollout,
                teacher_actions=teacher_actions,
                topk=int(candidate_topk),
            )

            if policy_name == "level1_source_aware":
                selected_actions = _choose_level1_source_aware(
                    source_local=source_local,
                    teacher_actions=teacher_actions,
                    belief_ctx=belief_ctx,
                    rollout=rollout,
                )
            elif policy_name == "level2_hindsight":
                selected_actions, meta = _choose_level2_hindsight(
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
                    max_bundles_per_state=int(max_bundles_per_state),
                    paper_like_alpha=float(paper_like_alpha),
                    paper_like_topk_fraction=float(paper_like_topk_fraction),
                    paper_like_time_tol_min=float(paper_like_time_tol_min),
                    soft_scenario_beta=float(soft_scenario_beta),
                )
                level2_eval_count += int(meta["evaluated_bundle_count"])
            elif policy_name == "level3_belief_nonmyopic":
                belief_vec = belief_ctx["belief"].view(-1).detach().cpu().float().tolist()
                selected_actions = _l3_best_bundle(
                    belief_vec=belief_vec,
                    candidate_pool=pool[: max(int(l3_tail_topk), 3)],
                    action_budget=int(runtime["action_budget"]),
                    remaining_rounds=int(runtime["num_episodes"]) - int(episode_idx) + 1,
                    horizon=int(l3_lookahead_horizon),
                    beam_width=int(l3_beam_width),
                    max_bundles_per_state=int(max_bundles_per_state),
                    step_penalty=-1.0 / 30.0,
                )
            elif policy_name == "teacher_greedy":
                pass
            else:
                raise ValueError(f"Unsupported policy_name: {policy_name}")

            selected_actions = [int(v) for v in selected_actions[:3]]
            if len(set(selected_actions)) < 3:
                # Enforce contract uniqueness by fallback to teacher
                selected_actions = [int(v) for v in teacher_actions[:3]]

            round_hit = source_local is not None and int(source_local) in set(selected_actions)
            if round_hit and hit_round is None:
                hit_round = int(episode_idx)
                source_slot = selected_actions.index(int(source_local)) + 1
                hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

            total_reward += float((-1.0 / 30.0) * float(len(selected_actions)) + (1.0 if bool(round_hit) else 0.0))
            rollout.step_with_actions(
                selected_actions,
                sample_types=[f"{policy_name}_slot_{i}" for i in range(len(selected_actions))],
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

    out_df = pd.DataFrame(rows)
    policy_meta = {
        "level2_total_bundle_evals": int(level2_eval_count),
    }
    return out_df, policy_meta


def _calc_delta(policy_summary: Dict[str, Any], base_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "delta_success": float(policy_summary["success_rate"] - base_summary["success_rate"]),
        "delta_hit_round": (
            None
            if policy_summary["avg_hit_round_conditional"] is None or base_summary["avg_hit_round_conditional"] is None
            else float(policy_summary["avg_hit_round_conditional"] - base_summary["avg_hit_round_conditional"])
        ),
        "delta_return_r0": float(policy_summary["avg_return_r0"] - base_summary["avg_return_r0"]),
    }


def _headroom_capture(numer: float, denom: float) -> Optional[float]:
    if abs(float(denom)) < 1e-12:
        return None
    return float(float(numer) / float(denom))


def _load_trusted_baseline(root: Path, panel_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    case_path = root / f"{panel_name}_teacher_case_rows.csv"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    case_df = pd.read_csv(case_path)
    panel_summary = summary["panel_summaries"][panel_name]["teacher_baseline"]
    return case_df, panel_summary


def run_panel(
    *,
    panel_name: str,
    rounds: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    env = CleanTwoChannelEvidenceEnv()

    case_limit = int(args.smoke_case_limit)
    if case_limit <= 0 and panel_name == "val_B60" and int(args.b60_case_limit) > 0:
        case_limit = int(args.b60_case_limit)

    runtime, split_meta = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.runtime_split),
        num_rounds=int(rounds),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=0,
        train_cache_version="",
        case_limit=int(case_limit),
    )

    trusted_root = Path(args.trusted_baseline_root)
    trusted_case_df, trusted_summary = _load_trusted_baseline(trusted_root, panel_name)
    trusted_map = trusted_case_df.set_index("case_id")

    # Trusted baseline is kept as fixed anchor; runtime teacher baseline is used for all delta comparisons.
    runtime_case_ids = [str(case.case_id) for case in runtime["cases"]]
    trusted_aligned_df = trusted_map.loc[runtime_case_ids].reset_index()
    trusted_aligned_summary = _summarize_panel(trusted_aligned_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))

    base_case_df, _base_meta = _run_policy_panel(
        policy_name="teacher_greedy",
        runtime=runtime,
        family=str(args.teacher_family),
        env=env,
        candidate_topk=int(args.candidate_topk),
        max_bundles_per_state=int(args.max_bundles_per_state),
        l3_lookahead_horizon=int(args.l3_lookahead_horizon),
        l3_beam_width=int(args.l3_beam_width),
        l3_tail_topk=int(args.l3_tail_topk),
        top_source_k=int(args.top_source_k),
        include_surrogate_features=bool(args.include_surrogate_features),
        paper_like_alpha=float(args.paper_like_alpha),
        paper_like_topk_fraction=float(args.paper_like_topk_fraction),
        paper_like_time_tol_min=float(args.paper_like_time_tol_min),
        soft_scenario_beta=float(args.soft_scenario_beta),
    )
    base_summary = _summarize_panel(base_case_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))

    l1_df, _l1_meta = _run_policy_panel(
        policy_name="level1_source_aware",
        runtime=runtime,
        family=str(args.teacher_family),
        env=env,
        candidate_topk=int(args.candidate_topk),
        max_bundles_per_state=int(args.max_bundles_per_state),
        l3_lookahead_horizon=int(args.l3_lookahead_horizon),
        l3_beam_width=int(args.l3_beam_width),
        l3_tail_topk=int(args.l3_tail_topk),
        top_source_k=int(args.top_source_k),
        include_surrogate_features=bool(args.include_surrogate_features),
        paper_like_alpha=float(args.paper_like_alpha),
        paper_like_topk_fraction=float(args.paper_like_topk_fraction),
        paper_like_time_tol_min=float(args.paper_like_time_tol_min),
        soft_scenario_beta=float(args.soft_scenario_beta),
    )
    runtime_l2 = runtime
    if int(args.level2_case_limit) > 0:
        runtime_l2 = dict(runtime)
        runtime_l2["cases"] = list(runtime["cases"][: int(args.level2_case_limit)])

    l2_df, l2_meta = _run_policy_panel(
        policy_name="level2_hindsight",
        runtime=runtime_l2,
        family=str(args.teacher_family),
        env=env,
        candidate_topk=int(args.candidate_topk),
        max_bundles_per_state=int(args.max_bundles_per_state),
        l3_lookahead_horizon=int(args.l3_lookahead_horizon),
        l3_beam_width=int(args.l3_beam_width),
        l3_tail_topk=int(args.l3_tail_topk),
        top_source_k=int(args.top_source_k),
        include_surrogate_features=bool(args.include_surrogate_features),
        paper_like_alpha=float(args.paper_like_alpha),
        paper_like_topk_fraction=float(args.paper_like_topk_fraction),
        paper_like_time_tol_min=float(args.paper_like_time_tol_min),
        soft_scenario_beta=float(args.soft_scenario_beta),
    )
    l3_df, _l3_meta = _run_policy_panel(
        policy_name="level3_belief_nonmyopic",
        runtime=runtime,
        family=str(args.teacher_family),
        env=env,
        candidate_topk=int(args.candidate_topk),
        max_bundles_per_state=int(args.max_bundles_per_state),
        l3_lookahead_horizon=int(args.l3_lookahead_horizon),
        l3_beam_width=int(args.l3_beam_width),
        l3_tail_topk=int(args.l3_tail_topk),
        top_source_k=int(args.top_source_k),
        include_surrogate_features=bool(args.include_surrogate_features),
        paper_like_alpha=float(args.paper_like_alpha),
        paper_like_topk_fraction=float(args.paper_like_topk_fraction),
        paper_like_time_tol_min=float(args.paper_like_time_tol_min),
        soft_scenario_beta=float(args.soft_scenario_beta),
    )

    l1_summary = _summarize_panel(l1_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))
    l2_summary = _summarize_panel(l2_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))
    l3_summary = _summarize_panel(l3_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))

    l1_changes = _compute_case_change_metrics(base_case_df, l1_df)
    l2_base_df = base_case_df
    if int(args.level2_case_limit) > 0:
        l2_case_ids = set(str(v) for v in l2_df["case_id"].tolist())
        l2_base_df = base_case_df[base_case_df["case_id"].astype(str).isin(l2_case_ids)].copy()
    l2_changes = _compute_case_change_metrics(l2_base_df, l2_df)
    l3_changes = _compute_case_change_metrics(base_case_df, l3_df)

    l1_delta = _calc_delta(l1_summary, base_summary)
    l2_base_summary = _summarize_panel(l2_base_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))
    l2_delta = _calc_delta(l2_summary, l2_base_summary)
    l3_delta = _calc_delta(l3_summary, base_summary)

    capture = {
        "success_l3_over_l1": _headroom_capture(float(l3_delta["delta_success"]), float(l1_delta["delta_success"])),
        "success_l3_over_l2": _headroom_capture(float(l3_delta["delta_success"]), float(l2_delta["delta_success"])),
        "return_l3_over_l1": _headroom_capture(float(l3_delta["delta_return_r0"]), float(l1_delta["delta_return_r0"])),
        "return_l3_over_l2": _headroom_capture(float(l3_delta["delta_return_r0"]), float(l2_delta["delta_return_r0"])),
    }

    panel_dir = output_dir / panel_name
    panel_dir.mkdir(parents=True, exist_ok=True)
    base_case_df.to_csv(panel_dir / "level0_teacher_case_rows.csv", index=False)
    l1_df.to_csv(panel_dir / "level1_source_aware_case_rows.csv", index=False)
    l2_df.to_csv(panel_dir / "level2_hindsight_case_rows.csv", index=False)
    l3_df.to_csv(panel_dir / "level3_belief_nonmyopic_case_rows.csv", index=False)

    panel_summary = {
        "panel": panel_name,
        "num_rounds": int(rounds),
        "case_count": int(len(base_case_df)),
        "trusted_baseline_reference": {
            "root": str(trusted_root),
            "panel_teacher_baseline": trusted_summary,
            "trusted_aligned_subset_summary": trusted_aligned_summary,
        },
        "runtime_split_meta": split_meta,
        "level0_teacher_greedy": {
            **base_summary,
            "delta_success": 0.0,
            "delta_hit_round": 0.0,
            "delta_return_r0": 0.0,
            "result_changed_episode_fraction": 0.0,
            "success_flip_episode_fraction": 0.0,
            "success_flip_down_episode_fraction": 0.0,
            "success_flip_count": 0,
        },
        "level1_source_aware_oracle": {
            **l1_summary,
            **l1_delta,
            **l1_changes,
        },
        "level2_hindsight_oracle": {
            **l2_summary,
            **l2_delta,
            **l2_changes,
            "level2_total_bundle_evals": int(l2_meta["level2_total_bundle_evals"]),
            "level2_case_limit": int(args.level2_case_limit),
            "level2_baseline_reference_case_count": int(len(l2_base_df)),
        },
        "level3_belief_feasible_nonmyopic": {
            **l3_summary,
            **l3_delta,
            **l3_changes,
            "lookahead_horizon": int(args.l3_lookahead_horizon),
            "beam_width": int(args.l3_beam_width),
            "candidate_topk": int(args.candidate_topk),
            "max_bundles_per_state": int(args.max_bundles_per_state),
            "tail_topk": int(args.l3_tail_topk),
        },
        "headroom_capture_ratio": capture,
        "artifacts": {
            "level0_teacher_case_rows": str(panel_dir / "level0_teacher_case_rows.csv"),
            "level1_case_rows": str(panel_dir / "level1_source_aware_case_rows.csv"),
            "level2_case_rows": str(panel_dir / "level2_hindsight_case_rows.csv"),
            "level3_case_rows": str(panel_dir / "level3_belief_nonmyopic_case_rows.csv"),
        },
    }
    write_json(panel_dir / "summary.json", panel_summary)
    return panel_summary


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_runs: List[Tuple[str, int]] = []
    if bool(args.run_b30):
        panel_runs.append(("val_B30", 10))
    if bool(args.run_b60):
        panel_runs.append(("val_B60", 20))

    panel_summaries: Dict[str, Any] = {}
    for panel_name, rounds in panel_runs:
        panel_summaries[panel_name] = run_panel(panel_name=panel_name, rounds=int(rounds), args=args, output_dir=output_dir)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "fixed_contract": {
            "posterior_family": "hsr_soft_scenario_posterior_v3",
            "teacher_policy": "posterior_greedy",
            "runtime_split": str(args.runtime_split),
            "actions_per_round": int(args.actions_per_round),
            "main_budget": "B30",
        },
        "planner_contract_level3": {
            "leakage_free": True,
            "belief_only": True,
            "lookahead_horizon": int(args.l3_lookahead_horizon),
            "beam_width": int(args.l3_beam_width),
            "candidate_topk": int(args.candidate_topk),
            "max_bundles_per_state": int(args.max_bundles_per_state),
            "tail_topk": int(args.l3_tail_topk),
            "approximation": "belief-elimination DP with bounded action bundle beam",
        },
        "level2_contract": {
            "type": "hindsight_best_current_intervention_policy",
            "approximation": "bounded bundle pool + teacher continuation terminal evaluation",
            "candidate_topk": int(args.candidate_topk),
            "max_bundles_per_state": int(args.max_bundles_per_state),
        },
        "panel_summaries": panel_summaries,
        "command": " ".join(__import__("sys").argv),
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
