from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.run_spim_teacher_regret_reward_alignment_audit import (
    _compute_step_belief,
    _extract_trigger_global,
    _make_rollout,
    _pick_topk_unsampled,
    _reward_r0,
)
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_audit import load_runtime_context, write_json
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SOURCE_ROOT,
    PaperLikeHSRState,
    get_device,
    seed_everything,
)


RUNNER_VERSION = "spim_regret_upper_bound_gateability_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute panel/policy upper bounds and simple gateability from SPIM regret audit artifacts."
    )
    parser.add_argument("--audit-root", type=str, required=True, help="Path containing exact136_B30/B60_* audit csv files.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--teacher-family", type=str, default="hsr_soft_scenario_posterior_v3")
    parser.add_argument("--actions-per-round", type=int, default=3)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--top-source-k", type=int, default=8)
    parser.add_argument("--include-surrogate-features", action="store_true")
    return parser.parse_args()


def _state_key(case_id: str, episode_idx: int) -> str:
    return f"teacher::{case_id}::ep{int(episode_idx)}"


def _quantile_threshold(series: pd.Series, trigger_rate: float, lower_is_trigger: bool) -> float:
    if len(series) <= 0:
        return 0.0
    q = float(trigger_rate) if lower_is_trigger else float(1.0 - trigger_rate)
    return float(series.astype(float).quantile(q))


def _compute_gate_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, bool]], Dict[str, Any]]:
    assert len(df) > 0
    labels = df["label_positive_regret"].astype(bool)

    margin_thr = _quantile_threshold(df["top1_top2_margin"], trigger_rate=0.30, lower_is_trigger=True)
    entropy_thr = _quantile_threshold(df["posterior_entropy"], trigger_rate=0.30, lower_is_trigger=False)
    top3_mass_thr = _quantile_threshold(df["top3_mass"], trigger_rate=0.30, lower_is_trigger=True)
    cover_thr = _quantile_threshold(df["mass_cover_0p7"], trigger_rate=0.30, lower_is_trigger=False)
    combo_margin_thr = _quantile_threshold(df["top1_top2_margin"], trigger_rate=0.40, lower_is_trigger=True)
    combo_entropy_thr = _quantile_threshold(df["posterior_entropy"], trigger_rate=0.40, lower_is_trigger=False)

    gate_defs: List[Tuple[str, pd.Series, Dict[str, float]]] = [
        (
            "margin_low",
            df["top1_top2_margin"].astype(float) <= float(margin_thr),
            {"top1_top2_margin_max": float(margin_thr)},
        ),
        (
            "entropy_high",
            df["posterior_entropy"].astype(float) >= float(entropy_thr),
            {"posterior_entropy_min": float(entropy_thr)},
        ),
        (
            "top3_mass_low",
            df["top3_mass"].astype(float) <= float(top3_mass_thr),
            {"top3_mass_max": float(top3_mass_thr)},
        ),
        (
            "cover_size_high",
            df["mass_cover_0p7"].astype(float) >= float(cover_thr),
            {"mass_cover_0p7_min": float(cover_thr)},
        ),
        (
            "margin_low_and_entropy_high",
            (df["top1_top2_margin"].astype(float) <= float(combo_margin_thr))
            & (df["posterior_entropy"].astype(float) >= float(combo_entropy_thr)),
            {
                "top1_top2_margin_max": float(combo_margin_thr),
                "posterior_entropy_min": float(combo_entropy_thr),
            },
        ),
    ]

    rows: List[Dict[str, Any]] = []
    gate_map: Dict[str, Dict[str, bool]] = {name: {} for name, _, _ in gate_defs}
    for name, pred, params in gate_defs:
        pred_bool = pred.astype(bool)
        tp = int((pred_bool & labels).sum())
        fp = int((pred_bool & ~labels).sum())
        fn = int((~pred_bool & labels).sum())
        rows.append(
            {
                "gate_family": str(name),
                "trigger_rate": float(pred_bool.mean()),
                "precision_on_positive_regret": float(tp / max(tp + fp, 1)),
                "recall_on_positive_regret": float(tp / max(tp + fn, 1)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                **params,
            }
        )
        for _, r in df[["state_key"]].assign(_pred=pred_bool).iterrows():
            gate_map[str(name)][str(r["state_key"])] = bool(r["_pred"])
    return pd.DataFrame(rows), gate_map, {
        "margin_low_threshold": float(margin_thr),
        "entropy_high_threshold": float(entropy_thr),
        "top3_mass_low_threshold": float(top3_mass_thr),
        "cover_size_high_threshold": float(cover_thr),
        "combo_margin_threshold": float(combo_margin_thr),
        "combo_entropy_threshold": float(combo_entropy_thr),
    }


def _run_policy_rollout(
    *,
    runtime: Dict[str, Any],
    family: str,
    env: CleanTwoChannelEvidenceEnv,
    action_budget: int,
    top_source_k: int,
    include_surrogate_features: bool,
    correction_slot3_map: Dict[str, int],
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
                belief_ctx["belief"], belief_ctx["candidate_mask"], rollout, int(action_budget)
            )
            teacher_actions = [int(v) for v in teacher_actions]
            if not teacher_actions:
                termination_reason = "teacher_no_action"
                break

            selected_actions = list(teacher_actions)
            if len(selected_actions) >= 3:
                key = _state_key(str(case.case_id), int(episode_idx))
                if key in correction_slot3_map:
                    cand = int(correction_slot3_map[key])
                    if cand not in {int(selected_actions[0]), int(selected_actions[1])}:
                        selected_actions[2] = int(cand)

            round_hit = source_local is not None and int(source_local) in set(selected_actions)
            if round_hit and hit_round is None:
                hit_round = int(episode_idx)
                source_slot = selected_actions.index(int(source_local)) + 1
                hit_sample_index = int((int(episode_idx) - 1) * int(action_budget) + int(source_slot))

            total_reward += _reward_r0(step_selected_count=len(selected_actions), round_hit=bool(round_hit))
            rollout.step_with_actions(
                selected_actions,
                sample_types=[f"upper_bound_slot_{i}" for i in range(len(selected_actions))],
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


def _panel_rounds_from_name(panel_name: str) -> int:
    return 20 if str(panel_name).endswith("B60") else 10


def _load_panel_teacher_audit(audit_root: Path, panel_name: str) -> pd.DataFrame:
    state_path = audit_root / f"{panel_name}_state_manifest.csv"
    slot3_path = audit_root / f"{panel_name}_slot3_state_summary.csv"
    if not state_path.exists() or not slot3_path.exists():
        return pd.DataFrame([])
    state_df = pd.read_csv(state_path)
    slot3_df = pd.read_csv(slot3_path)
    if "policy_source" in state_df.columns:
        state_df = state_df[state_df["policy_source"].astype(str) == "teacher"].copy()
    if "policy_source" in slot3_df.columns:
        slot3_df = slot3_df[slot3_df["policy_source"].astype(str) == "teacher"].copy()
    merged = state_df.merge(
        slot3_df[["state_key", "best_slot3_local", "best_delta_return_vs_teacher_slot3", "best_delta_success_vs_teacher_slot3"]],
        on="state_key",
        how="inner",
    )
    merged["label_positive_regret"] = merged["best_delta_return_vs_teacher_slot3"].astype(float) > 1e-12
    return merged


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    _ = get_device(str(args.device))

    audit_root = Path(args.audit_root)
    output_dir = Path(args.output_dir)
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = CleanTwoChannelEvidenceEnv()
    summary_in = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    available_panels = [p for p in ["exact136_B30", "exact136_B60"] if (audit_root / f"{p}_state_manifest.csv").exists()]

    panel_results: Dict[str, Any] = {}
    for panel_name in available_panels:
        teacher_df = _load_panel_teacher_audit(audit_root, panel_name)
        if len(teacher_df) <= 0:
            panel_results[panel_name] = {"status": "missing_or_empty_teacher_audit"}
            continue

        gate_table, gate_map, gate_thresholds = _compute_gate_table(teacher_df)
        gate_table.to_csv(output_dir / f"{panel_name}_gate_family_metrics.csv", index=False)

        oracle_all = {
            str(r["state_key"]): int(r["best_slot3_local"])
            for _, r in teacher_df.iterrows()
            if bool(r["label_positive_regret"])
        }
        oracle_by_gate: Dict[str, Dict[str, int]] = {}
        for gate_name in gate_table["gate_family"].tolist():
            oracle_by_gate[str(gate_name)] = {
                str(r["state_key"]): int(r["best_slot3_local"])
                for _, r in teacher_df.iterrows()
                if bool(r["label_positive_regret"]) and bool(gate_map[str(gate_name)].get(str(r["state_key"]), False))
            }

        rounds = _panel_rounds_from_name(panel_name)
        runtime = load_runtime_context(source_root, cache_dir)
        runtime["num_episodes"] = int(rounds)
        runtime["action_budget"] = int(args.actions_per_round)
        if int(args.case_limit) > 0:
            runtime["cases"] = list(runtime["cases"][: int(args.case_limit)])

        baseline_df = _run_policy_rollout(
            runtime=runtime,
            family=str(args.teacher_family),
            env=env,
            action_budget=int(args.actions_per_round),
            top_source_k=int(args.top_source_k),
            include_surrogate_features=bool(args.include_surrogate_features),
            correction_slot3_map={},
        )
        oracle_df = _run_policy_rollout(
            runtime=runtime,
            family=str(args.teacher_family),
            env=env,
            action_budget=int(args.actions_per_round),
            top_source_k=int(args.top_source_k),
            include_surrogate_features=bool(args.include_surrogate_features),
            correction_slot3_map=oracle_all,
        )
        baseline_df.to_csv(output_dir / f"{panel_name}_teacher_baseline_case_rows.csv", index=False)
        oracle_df.to_csv(output_dir / f"{panel_name}_oracle_slot3_case_rows.csv", index=False)

        baseline_summary = _summarize_panel(baseline_df, num_rounds=rounds, action_budget=int(args.actions_per_round))
        oracle_summary = _summarize_panel(oracle_df, num_rounds=rounds, action_budget=int(args.actions_per_round))

        gate_upper_rows: List[Dict[str, Any]] = []
        for gate_name in gate_table["gate_family"].tolist():
            df_gate = _run_policy_rollout(
                runtime=runtime,
                family=str(args.teacher_family),
                env=env,
                action_budget=int(args.actions_per_round),
                top_source_k=int(args.top_source_k),
                include_surrogate_features=bool(args.include_surrogate_features),
                correction_slot3_map=oracle_by_gate[str(gate_name)],
            )
            df_gate.to_csv(output_dir / f"{panel_name}_gate_{gate_name}_oracle_case_rows.csv", index=False)
            gate_summary = _summarize_panel(df_gate, num_rounds=rounds, action_budget=int(args.actions_per_round))
            gate_upper_rows.append(
                {
                    "gate_family": str(gate_name),
                    "case_count": int(gate_summary["case_count"]),
                    "success_rate": float(gate_summary["success_rate"]),
                    "delta_success_vs_teacher": float(gate_summary["success_rate"] - baseline_summary["success_rate"]),
                    "avg_hit_round_conditional": gate_summary["avg_hit_round_conditional"],
                    "delta_avg_hit_round_conditional_vs_teacher": (
                        None
                        if gate_summary["avg_hit_round_conditional"] is None
                        else float(gate_summary["avg_hit_round_conditional"] - baseline_summary["avg_hit_round_conditional"])
                    ),
                    "avg_return_r0": float(gate_summary["avg_return_r0"]),
                    "delta_return_r0_vs_teacher": float(gate_summary["avg_return_r0"] - baseline_summary["avg_return_r0"]),
                    "num_corrected_states": int(len(oracle_by_gate[str(gate_name)])),
                }
            )
        gate_upper_df = pd.DataFrame(gate_upper_rows)
        gate_upper_df.to_csv(output_dir / f"{panel_name}_gate_constrained_upper_bound.csv", index=False)

        panel_results[panel_name] = {
            "gate_thresholds": gate_thresholds,
            "teacher_audited_state_count": int(len(teacher_df)),
            "teacher_positive_regret_count": int(teacher_df["label_positive_regret"].sum()),
            "teacher_positive_regret_fraction": float(teacher_df["label_positive_regret"].mean()),
            "oracle_slot3_correction_state_count": int(len(oracle_all)),
            "teacher_panel_metrics": baseline_summary,
            "oracle_slot3_panel_upper_bound": {
                **oracle_summary,
                "delta_success_vs_teacher": float(oracle_summary["success_rate"] - baseline_summary["success_rate"]),
                "delta_avg_hit_round_conditional_vs_teacher": (
                    None
                    if oracle_summary["avg_hit_round_conditional"] is None
                    else float(oracle_summary["avg_hit_round_conditional"] - baseline_summary["avg_hit_round_conditional"])
                ),
                "delta_return_r0_vs_teacher": float(oracle_summary["avg_return_r0"] - baseline_summary["avg_return_r0"]),
            },
            "artifacts": {
                "gate_family_metrics": str(output_dir / f"{panel_name}_gate_family_metrics.csv"),
                "teacher_baseline_case_rows": str(output_dir / f"{panel_name}_teacher_baseline_case_rows.csv"),
                "oracle_slot3_case_rows": str(output_dir / f"{panel_name}_oracle_slot3_case_rows.csv"),
                "gate_constrained_upper_bound": str(output_dir / f"{panel_name}_gate_constrained_upper_bound.csv"),
            },
        }

    summary_out = {
        "runner_version": RUNNER_VERSION,
        "source_audit_root": str(audit_root),
        "source_runner_version": summary_in.get("runner_version"),
        "source_panel_version": summary_in.get("panel_version"),
        "seed": int(args.seed),
        "teacher_family": str(args.teacher_family),
        "fixed_contract": {
            "posterior_family": "hsr_soft_scenario_posterior_v3",
            "teacher_policy": "posterior_greedy",
            "actions_per_round": int(args.actions_per_round),
        },
        "panel_results": panel_results,
    }
    write_json(output_dir / "summary.json", summary_out)


if __name__ == "__main__":
    main()
