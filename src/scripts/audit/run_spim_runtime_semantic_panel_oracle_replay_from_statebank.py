from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.run_spim_regret_upper_bound_gateability_audit import (
    _run_policy_rollout as trusted_slot3_rollout,
    _summarize_panel,
)
from src.scripts.audit.run_spim_teacher_regret_reward_alignment_audit import (
    _compute_step_belief,
    _extract_trigger_global,
    _make_rollout,
    _pick_topk_unsampled,
)
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_audit import write_json
from src.scripts.run_spim_family_sweep import PaperLikeHSRState
from src.scripts.run_spim_policy_eval_strict import build_runtime_strict
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SOURCE_ROOT,
    seed_everything,
)


RUNNER_VERSION = "spim_runtime_semantic_panel_oracle_replay_from_statebank_v1"
PANEL_VERSION = "strict_val_runtime_semantic_panel_oracle_replay_from_statebank_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Runtime-semantic single-intervention panel replay from strict state-level headroom bank.")
    p.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--runtime-split", type=str, default="val", choices=["exact136", "train", "val", "test"])
    p.add_argument("--teacher-family", type=str, default="hsr_soft_scenario_posterior_v3")
    p.add_argument("--statebank-root", type=str, required=True, help="Directory containing {panel}_state_level_headroom.csv")
    p.add_argument("--slot3-selector-root", type=str, required=True, help="Directory containing {panel}_selector_teacher_slot3_state_summary.csv")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--case-limit", type=int, default=0)
    p.add_argument("--num-rounds-b30", type=int, default=10)
    p.add_argument("--num-rounds-b60", type=int, default=20)
    p.add_argument("--actions-per-round", type=int, default=3)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--top-source-k", type=int, default=8)
    p.add_argument("--include-surrogate-features", action="store_true")
    return p.parse_args()


def _state_key(case_id: str, episode_idx: int) -> str:
    return f"teacher::{case_id}::ep{int(episode_idx)}"


def _same_hit_round(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return int(a) == int(b)


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


def _parse_actions(raw: Any) -> List[int]:
    try:
        x = json.loads(str(raw))
    except Exception:
        x = []
    if not isinstance(x, list):
        return []
    return [int(v) for v in x]


def _load_panel_statebank(root: Path, panel: str) -> pd.DataFrame:
    p = root / f"{panel}_state_level_headroom.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing statebank: {p}")
    return pd.read_csv(p)


def _load_slot3_selector(root: Path, panel: str) -> pd.DataFrame:
    p = root / f"{panel}_selector_teacher_slot3_state_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing slot3 selector summary: {p}")
    df = pd.read_csv(p)
    if "policy_source" in df.columns:
        df = df[df["policy_source"].astype(str) == "teacher"].copy()
    return df


def _build_plan_map(
    *,
    state_df: pd.DataFrame,
    action_col: str,
    delta_ret_col: str,
    delta_succ_col: str,
    eps: float,
) -> Dict[str, Dict[str, Any]]:
    # case_id -> one best positive state plan
    plan: Dict[str, Dict[str, Any]] = {}
    for _, r in state_df.iterrows():
        case_id = str(r["case_id"])
        delta_ret = float(r.get(delta_ret_col, 0.0))
        delta_succ = float(r.get(delta_succ_col, 0.0))
        if delta_ret <= float(eps):
            continue
        actions = _parse_actions(r.get(action_col, "[]"))
        if len(actions) < 3:
            continue
        cand = {
            "state_key": str(r["state_key"]),
            "episode_index": int(r.get("episode_index", 0)),
            "bundle_actions": [int(actions[0]), int(actions[1]), int(actions[2])],
            "delta_return_vs_teacher": float(delta_ret),
            "delta_success_vs_teacher": float(delta_succ),
        }
        old = plan.get(case_id)
        if old is None:
            plan[case_id] = cand
            continue
        ck = (float(cand["delta_return_vs_teacher"]), float(cand["delta_success_vs_teacher"]), -int(cand["episode_index"]))
        ok = (float(old["delta_return_vs_teacher"]), float(old["delta_success_vs_teacher"]), -int(old["episode_index"]))
        if ck > ok:
            plan[case_id] = cand
    return plan


def _runtime_rollout_with_plan(
    *,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    plan_map: Dict[str, Dict[str, Any]],
    top_source_k: int,
    include_surrogate_features: bool,
) -> pd.DataFrame:
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]
    gate = DynamicReachabilityRuleModule()
    rows: List[Dict[str, Any]] = []

    for case in runtime["cases"]:
        case_id = str(case.case_id)
        plan = plan_map.get(case_id)

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
                paper_like_alpha=0.55,
                paper_like_topk_fraction=0.12,
                paper_like_time_tol_min=30.0,
                soft_scenario_beta=2.0,
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
            key = _state_key(case_id, int(episode_idx))
            if (not intervention_applied) and plan is not None and key == str(plan["state_key"]):
                b = [int(v) for v in plan["bundle_actions"]]
                if len(b) >= 3:
                    selected_actions[:3] = b[:3]
                    intervention_applied = True

            round_hit = source_local is not None and int(source_local) in set(selected_actions)
            if round_hit and hit_round is None:
                hit_round = int(episode_idx)
                source_slot = selected_actions.index(int(source_local)) + 1
                hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

            total_reward += float((-1.0 / 30.0) * float(len(selected_actions)) + (1.0 if bool(round_hit) else 0.0))
            rollout.step_with_actions(
                selected_actions,
                sample_types=[f"runtime_semantic_replay_slot_{i}" for i in range(len(selected_actions))],
            )
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            budget_used += int(len(selected_actions))
            if round_hit:
                termination_reason = "source_hit"
                break

        rows.append(
            {
                "case_id": case_id,
                "success": float(hit_round is not None),
                "hit_round": None if hit_round is None else int(hit_round),
                "hit_sample_index": None if hit_sample_index is None else int(hit_sample_index),
                "budget_used": int(budget_used),
                "return_r0": float(total_reward),
                "termination_reason": str(termination_reason),
                "planned_intervention": float(plan is not None),
                "applied_intervention": float(intervention_applied),
            }
        )

    return pd.DataFrame(rows)


def _panel_eval(
    *,
    panel: str,
    rounds: int,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    state_df: pd.DataFrame,
    slot3_selector_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    # Trusted slot3 control map from trusted selector rows.
    trusted_map: Dict[str, int] = {}
    if "best_slot3_local" in slot3_selector_df.columns and "best_delta_return_vs_teacher_slot3" in slot3_selector_df.columns:
        for _, r in slot3_selector_df.iterrows():
            if float(r.get("best_delta_return_vs_teacher_slot3", 0.0)) > float(args.eps):
                trusted_map[str(r["state_key"])] = int(r["best_slot3_local"])

    contract_specs = {
        "slot1_only": ("best_single_slot1_action", "best_single_slot1_delta_return", "best_single_slot1_delta_success"),
        "slot2_only": ("best_single_slot2_action", "best_single_slot2_delta_return", "best_single_slot2_delta_success"),
        "slot3_only": ("best_single_slot3_action", "best_single_slot3_delta_return", "best_single_slot3_delta_success"),
        "single_any": ("best_single_any_action", "best_single_any_delta_return", "best_single_any_delta_success"),
        "two_member_any": ("best_two_any_action", "best_two_any_delta_return", "best_two_any_delta_success"),
        "bundle_full": ("best_bundle_action", "best_bundle_delta_return", "best_bundle_delta_success"),
    }

    teacher_df = _runtime_rollout_with_plan(
        runtime=runtime,
        family=family,
        env=env,
        plan_map={},
        top_source_k=int(args.top_source_k),
        include_surrogate_features=bool(args.include_surrogate_features),
    )
    teacher_summary = _summarize_panel(teacher_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))

    contract_rows: List[Dict[str, Any]] = []
    contract_case_rows: Dict[str, pd.DataFrame] = {}

    for contract, (action_col, dret_col, dsucc_col) in contract_specs.items():
        plan = _build_plan_map(
            state_df=state_df,
            action_col=action_col,
            delta_ret_col=dret_col,
            delta_succ_col=dsucc_col,
            eps=float(args.eps),
        )
        oracle_df = _runtime_rollout_with_plan(
            runtime=runtime,
            family=family,
            env=env,
            plan_map=plan,
            top_source_k=int(args.top_source_k),
            include_surrogate_features=bool(args.include_surrogate_features),
        )
        oracle_summary = _summarize_panel(oracle_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))
        change = _compute_case_change_metrics(teacher_df, oracle_df)
        contract_rows.append(
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
                "planned_case_count": int(len(plan)),
            }
        )
        contract_case_rows[contract] = oracle_df

    trusted_df = trusted_slot3_rollout(
        runtime=runtime,
        family=family,
        env=env,
        action_budget=int(args.actions_per_round),
        top_source_k=int(args.top_source_k),
        include_surrogate_features=bool(args.include_surrogate_features),
        correction_slot3_map=trusted_map,
    )
    trusted_summary = _summarize_panel(trusted_df, num_rounds=int(rounds), action_budget=int(args.actions_per_round))
    trusted_change = _compute_case_change_metrics(teacher_df, trusted_df)

    contract_df = pd.DataFrame(contract_rows)
    slot3 = contract_df.loc[contract_df["contract"].astype(str) == "slot3_only"]
    slot3_ds = float(slot3.iloc[0]["delta_success"]) if len(slot3) else 0.0
    slot3_dr = float(slot3.iloc[0]["delta_return_r0"]) if len(slot3) else 0.0
    extra_rows = []
    for k in ["single_any", "two_member_any", "bundle_full"]:
        sub = contract_df.loc[contract_df["contract"].astype(str) == k]
        if len(sub) <= 0:
            continue
        extra_rows.append(
            {
                "contract": str(k),
                "extra_delta_success_vs_slot3_only": float(sub.iloc[0]["delta_success"] - slot3_ds),
                "extra_delta_return_r0_vs_slot3_only": float(sub.iloc[0]["delta_return_r0"] - slot3_dr),
            }
        )

    teacher_df.to_csv(output_dir / f"{panel}_teacher_case_rows.csv", index=False)
    contract_df.to_csv(output_dir / f"{panel}_panel_oracle_contracts.csv", index=False)
    pd.DataFrame(extra_rows).to_csv(output_dir / f"{panel}_extra_headroom_vs_slot3_only.csv", index=False)
    trusted_df.to_csv(output_dir / f"{panel}_slot3_trusted_reference_case_rows.csv", index=False)
    for k, df in contract_case_rows.items():
        df.to_csv(output_dir / f"{panel}_{k}_case_rows.csv", index=False)

    return {
        "panel": str(panel),
        "teacher_baseline": teacher_summary,
        "contracts": contract_rows,
        "extra_headroom_vs_slot3_only": extra_rows,
        "slot3_only_trusted_reference": {
            **trusted_summary,
            "delta_success_vs_teacher": float(trusted_summary["success_rate"] - teacher_summary["success_rate"]),
            "delta_return_r0_vs_teacher": float(trusted_summary["avg_return_r0"] - teacher_summary["avg_return_r0"]),
            "result_changed_episode_fraction": float(trusted_change["result_changed_episode_fraction"]),
            "success_flip_episode_fraction": float(trusted_change["success_flip_episode_fraction"]),
            "success_flip_down_episode_fraction": float(trusted_change["success_flip_down_episode_fraction"]),
            "trusted_positive_state_count": int(len(trusted_map)),
        },
        "statebank_state_count": int(len(state_df)),
        "slot3_selector_state_count": int(len(slot3_selector_df)),
        "artifacts": {
            "teacher_case_rows": str(output_dir / f"{panel}_teacher_case_rows.csv"),
            "panel_oracle_contracts": str(output_dir / f"{panel}_panel_oracle_contracts.csv"),
            "extra_headroom_vs_slot3_only": str(output_dir / f"{panel}_extra_headroom_vs_slot3_only.csv"),
            "slot3_trusted_reference_case_rows": str(output_dir / f"{panel}_slot3_trusted_reference_case_rows.csv"),
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    statebank_root = Path(args.statebank_root)
    slot3_root = Path(args.slot3_selector_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_b30 = "exact136_B30" if str(args.runtime_split) == "exact136" else f"{args.runtime_split}_B30"
    panel_b60 = "exact136_B60" if str(args.runtime_split) == "exact136" else f"{args.runtime_split}_B60"

    state_b30 = _load_panel_statebank(statebank_root, panel_b30)
    state_b60 = _load_panel_statebank(statebank_root, panel_b60)
    slot3_b30 = _load_slot3_selector(slot3_root, panel_b30)
    slot3_b60 = _load_slot3_selector(slot3_root, panel_b60)

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

    env = CleanTwoChannelEvidenceEnv()
    summary_b30 = _panel_eval(
        panel=panel_b30,
        rounds=int(args.num_rounds_b30),
        runtime=runtime_b30,
        family=str(args.teacher_family),
        env=env,
        state_df=state_b30,
        slot3_selector_df=slot3_b30,
        args=args,
        output_dir=output_dir,
    )
    summary_b60 = _panel_eval(
        panel=panel_b60,
        rounds=int(args.num_rounds_b60),
        runtime=runtime_b60,
        family=str(args.teacher_family),
        env=env,
        state_df=state_b60,
        slot3_selector_df=slot3_b60,
        args=args,
        output_dir=output_dir,
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
            "candidate_pool": "inherited_from_statebank: posterior_top6_legal_unsampled_with_teacher_forced_in",
            "continuation_policy": "teacher_continuation",
            "intervention_budget": "single_intervention_per_case",
            "positive_only_rule": "statebank best_delta_return_vs_teacher_current_set > eps",
        },
        "inputs": {
            "statebank_root": str(statebank_root),
            "slot3_selector_root": str(slot3_root),
        },
        "audit_config": {
            "eps": float(args.eps),
            "case_limit": int(args.case_limit),
            "top_source_k": int(args.top_source_k),
            "include_surrogate_features": bool(args.include_surrogate_features),
        },
        "split_meta": {
            "b30": split_meta_b30,
            "b60": split_meta_b60,
        },
        "panel_summaries": {
            panel_b30: summary_b30,
            panel_b60: summary_b60,
        },
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
