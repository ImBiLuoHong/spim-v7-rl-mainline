#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


ARTIFACTS: Dict[str, Dict[str, str]] = {
    "old_v3_greedy": {
        "summary": "artifacts/spim_v7_offset_sweep/20260424_strict_val_b30/v3/summary.json",
        "case": "artifacts/spim_v7_offset_sweep/20260424_strict_val_b30/v3/case_rows.csv",
        "step": "artifacts/spim_v7_offset_sweep/20260424_strict_val_b30/v3/step_rows.csv",
    },
    "v7_greedy": {
        "summary": "artifacts/spim_v7_offset_sweep/20260424_strict_val_b30/v7_7offset/summary.json",
        "case": "artifacts/spim_v7_offset_sweep/20260424_strict_val_b30/v7_7offset/case_rows.csv",
        "step": "artifacts/spim_v7_offset_sweep/20260424_strict_val_b30/v7_7offset/step_rows.csv",
    },
    "old_v3_fresh_rl": {
        "summary": "artifacts/spim_set_level_rl_mainline/20260415_causal_exp3C_train4823_seed42_random_valueonly_v1/strict_eval_val_B30/rl_set_seed42_random_valueonly_train4823/summary.json",
        "case": "artifacts/spim_set_level_rl_mainline/20260415_causal_exp3C_train4823_seed42_random_valueonly_v1/strict_eval_val_B30/rl_set_seed42_random_valueonly_train4823/case_rows.csv",
        "step": "artifacts/spim_set_level_rl_mainline/20260415_causal_exp3C_train4823_seed42_random_valueonly_v1/strict_eval_val_B30/rl_set_seed42_random_valueonly_train4823/step_rows.csv",
        "history": "artifacts/spim_set_level_rl_mainline/20260415_causal_exp3C_train4823_seed42_random_valueonly_v1/rl_train_history.csv",
        "run_summary": "artifacts/spim_set_level_rl_mainline/20260415_causal_exp3C_train4823_seed42_random_valueonly_v1/summary.json",
        "checkpoint": "artifacts/spim_set_level_rl_mainline/20260415_causal_exp3C_train4823_seed42_random_valueonly_v1/checkpoints/rl_student_final.pt",
    },
    "old_v3_strongest_rl": {
        "summary": "artifacts/spim_set_level_rl_mainline/20260420_v3_early_set_credit_v1/stage1_c1_r2_seed45/strict_eval_val_B30/rl_set_seed45_v3_early_set_credit_c1_r2_train4823/summary.json",
        "case": "artifacts/spim_set_level_rl_mainline/20260420_v3_early_set_credit_v1/stage1_c1_r2_seed45/strict_eval_val_B30/rl_set_seed45_v3_early_set_credit_c1_r2_train4823/case_rows.csv",
        "step": "artifacts/spim_set_level_rl_mainline/20260420_v3_early_set_credit_v1/stage1_c1_r2_seed45/strict_eval_val_B30/rl_set_seed45_v3_early_set_credit_c1_r2_train4823/step_rows.csv",
        "history": "artifacts/spim_set_level_rl_mainline/20260420_v3_early_set_credit_v1/stage1_c1_r2_seed45/rl_train_history.csv",
        "run_summary": "artifacts/spim_set_level_rl_mainline/20260420_v3_early_set_credit_v1/stage1_c1_r2_seed45/summary.json",
        "checkpoint": "artifacts/spim_set_level_rl_mainline/20260420_v3_early_set_credit_v1/stage1_c1_r2_seed45/checkpoints/rl_student_final.pt",
    },
    "v7_anchor_seed45_rl": {
        "summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed45/strict_eval_val_B30/rl_seed45_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2/summary.json",
        "case": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed45/strict_eval_val_B30/rl_seed45_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2/case_rows.csv",
        "step": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed45/strict_eval_val_B30/rl_seed45_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2/step_rows.csv",
        "history": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed45/rl_train_history.csv",
        "run_summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed45/summary.json",
        "checkpoint": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed45/checkpoints/rl_student_final.pt",
    },
    "v7_anchor_seed46_rl": {
        "summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed46/strict_eval_val_B30/rl_seed46_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2/summary.json",
        "case": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed46/strict_eval_val_B30/rl_seed46_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2/case_rows.csv",
        "step": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed46/strict_eval_val_B30/rl_seed46_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2/step_rows.csv",
        "history": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed46/rl_train_history.csv",
        "run_summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed46/summary.json",
        "checkpoint": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_fresh_randominit_stronger_value_head_v2_v1/seed46/checkpoints/rl_student_final.pt",
    },
    "v7_pure_seed45_rl": {
        "summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed45_v1/seed45/strict_eval_val_B30/rl_seed45_v7_7offset_alpha055_pure_rl_no_anchor/summary.json",
        "case": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed45_v1/seed45/strict_eval_val_B30/rl_seed45_v7_7offset_alpha055_pure_rl_no_anchor/case_rows.csv",
        "step": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed45_v1/seed45/strict_eval_val_B30/rl_seed45_v7_7offset_alpha055_pure_rl_no_anchor/step_rows.csv",
        "history": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed45_v1/seed45/rl_train_history.csv",
        "run_summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed45_v1/seed45/summary.json",
        "checkpoint": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed45_v1/seed45/checkpoints/rl_student_final.pt",
    },
    "v7_pure_seed46_rl": {
        "summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed46_v1/seed46/strict_eval_val_B30/rl_seed46_v7_7offset_alpha055_pure_rl_no_anchor/summary.json",
        "case": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed46_v1/seed46/strict_eval_val_B30/rl_seed46_v7_7offset_alpha055_pure_rl_no_anchor/case_rows.csv",
        "step": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed46_v1/seed46/strict_eval_val_B30/rl_seed46_v7_7offset_alpha055_pure_rl_no_anchor/step_rows.csv",
        "history": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed46_v1/seed46/rl_train_history.csv",
        "run_summary": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed46_v1/seed46/summary.json",
        "checkpoint": "artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed46_v1/seed46/checkpoints/rl_student_final.pt",
    },
}

BASELINES = {
    "old_v3_greedy": 0.8962172647914646,
    "v7_greedy": 0.9059165858389913,
    "old_v3_fresh_rl": 0.9233753637245393,
    "old_v3_strongest_rl": 0.9243452958292919,
}


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: str | Path) -> Dict[str, Any]:
    if str(path).strip() == "":
        return {}
    p = project_path(path)
    if not p.exists() or p.is_dir():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> Optional[pd.DataFrame]:
    if str(path).strip() == "":
        return None
    p = project_path(path)
    if not p.exists() or p.is_dir():
        return None
    return pd.read_csv(p)


def success_rate(path: str | Path) -> Optional[float]:
    data = load_json(path)
    summary = data.get("summary", {})
    if isinstance(summary, dict) and summary.get("success_rate") is not None:
        return float(summary["success_rate"])
    if isinstance(data.get("rl_summary"), dict) and data["rl_summary"].get("success_rate") is not None:
        return float(data["rl_summary"]["success_rate"])
    return None


def nested_get(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def success_map(df: pd.DataFrame) -> pd.Series:
    return df.set_index("case_id")["success_rate"].astype(float).ge(0.5)


def float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def parse_list(value: Any) -> List[int]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        out = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(out, (list, tuple)):
        return []
    return [int(v) for v in out]


def set_jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return float(len(sa & sb) / max(len(union), 1))


def set_intersection_count(a: Iterable[int], b: Iterable[int]) -> int:
    return len(set(a) & set(b))


def case_source_map(case_df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in case_df.itertuples(index=False):
        value = getattr(r, "source_global_id")
        if value is None or pd.isna(value):
            continue
        out[str(r.case_id)] = int(value)
    return out


def rows_for_cases(step_df: pd.DataFrame, cases: Iterable[str]) -> pd.DataFrame:
    case_set = set(map(str, cases))
    return step_df[step_df["case_id"].astype(str).isin(case_set)].copy()


def first_round_with_positive(step_df: pd.DataFrame, case_id: str) -> Optional[int]:
    rows = step_df[step_df["case_id"].astype(str) == str(case_id)]
    if "positive_count" not in rows:
        return None
    pos = rows[pd.to_numeric(rows["positive_count"], errors="coerce").fillna(0) > 0]
    if pos.empty:
        return None
    return int(pos["round_index"].min())


def true_source_oracle(case_df: pd.DataFrame, step_df: pd.DataFrame) -> pd.DataFrame:
    sources = case_source_map(case_df)
    slate_ids_available = "policy_slate_global_ids" in step_df.columns
    rows: List[Dict[str, Any]] = []
    for case_id, source in sources.items():
        steps = step_df[step_df["case_id"].astype(str) == str(case_id)].copy()
        first_slate: Optional[int] = None
        first_selected: Optional[int] = None
        source_in_slate_any = False
        source_selected_any = False
        posterior_top_in_slate_rounds = 0
        nonempty_slate_rounds = 0
        for step in steps.itertuples(index=False):
            round_index = int(getattr(step, "round_index"))
            slate = parse_list(getattr(step, "policy_slate_global_ids", None)) if slate_ids_available else []
            selected = parse_list(getattr(step, "selected_global_ids", None))
            if slate:
                nonempty_slate_rounds += 1
                # The active slate builder emits posterior-ranked candidates first.
                posterior_top_in_slate_rounds += 1
            if int(source) in set(slate):
                source_in_slate_any = True
                first_slate = round_index if first_slate is None else min(first_slate, round_index)
            if int(source) in set(selected):
                source_selected_any = True
                first_selected = round_index if first_selected is None else min(first_selected, round_index)
        case_row = case_df[case_df["case_id"].astype(str) == str(case_id)].iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "source_global_id": int(source),
                "success": bool(float(case_row["success_rate"]) >= 0.5),
                "slate_ids_available": bool(slate_ids_available),
                "hit_round": float_or_none(case_row.get("hit_round")),
                "source_in_any_slate": None if not slate_ids_available else bool(source_in_slate_any),
                "source_selected": bool(source_selected_any),
                "first_round_source_enters_slate": None if not slate_ids_available else first_slate,
                "first_round_selected_hits_source": first_selected,
                "first_positive_round_proxy": first_round_with_positive(step_df, case_id),
                "posterior_top_source_in_slate_round_rate_inferred": (
                    None if nonempty_slate_rounds <= 0 else float(posterior_top_in_slate_rounds / nonempty_slate_rounds)
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_oracle(label: str, case_df: pd.DataFrame, step_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_case = true_source_oracle(case_df, step_df)
    slate_available = bool(per_case["slate_ids_available"].any()) if len(per_case) else False
    summary = {
        "policy": label,
        "case_count": int(len(per_case)),
        "success_rate_from_case_rows": float(per_case["success"].mean()) if len(per_case) else math.nan,
        "slate_ids_available": bool(slate_available),
        "true_source_in_any_slate_rate": float(per_case["source_in_any_slate"].dropna().mean()) if slate_available and len(per_case) else math.nan,
        "true_source_selected_rate": float(per_case["source_selected"].mean()) if len(per_case) else math.nan,
        "first_round_true_source_enters_slate_mean": float(per_case["first_round_source_enters_slate"].dropna().mean())
        if per_case["first_round_source_enters_slate"].notna().any()
        else math.nan,
        "first_round_selected_hits_source_mean": float(per_case["first_round_selected_hits_source"].dropna().mean())
        if per_case["first_round_selected_hits_source"].notna().any()
        else math.nan,
        "posterior_top_source_in_slate_rate_inferred": float(
            per_case["posterior_top_source_in_slate_round_rate_inferred"].dropna().mean()
        )
        if per_case["posterior_top_source_in_slate_round_rate_inferred"].notna().any()
        else math.nan,
    }
    return pd.DataFrame([summary]), per_case


def collect_step_metrics(step_df: pd.DataFrame, cases: Iterable[str], prefix: str) -> Dict[str, Any]:
    sub = rows_for_cases(step_df, cases)
    out: Dict[str, Any] = {}
    for scope, scoped in [
        ("all_steps", sub),
        ("round1", sub[pd.to_numeric(sub["round_index"], errors="coerce") == 1]),
    ]:
        for col in ["posterior_entropy", "top1_top2_margin", "top1_mass", "top3_mass", "top5_mass", "effective_support_size_ratio"]:
            if col in scoped and not scoped.empty:
                out[f"{prefix}_{scope}_{col}_mean"] = float(pd.to_numeric(scoped[col], errors="coerce").mean())
                out[f"{prefix}_{scope}_{col}_median"] = float(pd.to_numeric(scoped[col], errors="coerce").median())
            else:
                out[f"{prefix}_{scope}_{col}_mean"] = math.nan
                out[f"{prefix}_{scope}_{col}_median"] = math.nan
    return out


def state_flags(step_df: pd.DataFrame) -> pd.DataFrame:
    round1 = step_df[pd.to_numeric(step_df["round_index"], errors="coerce") == 1].copy()
    entropy = pd.to_numeric(round1["posterior_entropy"], errors="coerce")
    margin = pd.to_numeric(round1["top1_top2_margin"], errors="coerce")
    ent_med = float(entropy.median())
    margin_med = float(margin.median())
    out = round1[["case_id"]].copy()
    out["high_entropy"] = entropy > ent_med
    out["low_margin"] = margin < margin_med
    out["high_entropy_low_margin"] = out["high_entropy"] & out["low_margin"]
    return out.set_index("case_id")


def selected_or_slate_sets(
    step_df: pd.DataFrame,
    case_id: str,
    rounds: Sequence[int],
    column: str,
) -> List[int]:
    rows = step_df[
        (step_df["case_id"].astype(str) == str(case_id))
        & (pd.to_numeric(step_df["round_index"], errors="coerce").isin(list(rounds)))
    ]
    values: List[int] = []
    for value in rows[column].tolist():
        values.extend(parse_list(value))
    return values


def overlap_summary(
    label: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    rounds: Sequence[int],
    cases: Sequence[str],
) -> Dict[str, Any]:
    jaccards: List[float] = []
    overlaps: List[int] = []
    exact: List[bool] = []
    for case_id in cases:
        lset = selected_or_slate_sets(left, case_id, rounds, column)
        rset = selected_or_slate_sets(right, case_id, rounds, column)
        jaccards.append(set_jaccard(lset, rset))
        overlaps.append(set_intersection_count(lset, rset))
        exact.append(set(lset) == set(rset))
    return {
        "comparison": label,
        "column": column,
        "rounds": ",".join(str(r) for r in rounds),
        "case_count": int(len(cases)),
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else math.nan,
        "mean_overlap_count": float(np.mean(overlaps)) if overlaps else math.nan,
        "exact_set_match_rate": float(np.mean(exact)) if exact else math.nan,
    }


def partition_diagnostics(
    *,
    name: str,
    cases: Sequence[str],
    v7_step: pd.DataFrame,
    v7_rl_case: pd.DataFrame,
    v7_rl_step: pd.DataFrame,
    old_rl_case: pd.DataFrame,
    old_rl_step: pd.DataFrame,
    flags: pd.DataFrame,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "partition": name,
        "case_count": int(len(cases)),
    }
    out.update(collect_step_metrics(v7_step, cases, "v7_posterior"))
    if len(cases) > 0:
        subset_flags = flags.reindex(list(cases))
        out["high_entropy_low_margin_case_fraction"] = float(subset_flags["high_entropy_low_margin"].fillna(False).mean())
        out["high_entropy_case_fraction"] = float(subset_flags["high_entropy"].fillna(False).mean())
        out["low_margin_case_fraction"] = float(subset_flags["low_margin"].fillna(False).mean())
    else:
        out["high_entropy_low_margin_case_fraction"] = math.nan
        out["high_entropy_case_fraction"] = math.nan
        out["low_margin_case_fraction"] = math.nan

    oracle = true_source_oracle(v7_rl_case[v7_rl_case["case_id"].astype(str).isin(cases)], v7_rl_step)
    out["v7_rl_true_source_in_any_slate_rate"] = float(oracle["source_in_any_slate"].mean()) if len(oracle) else math.nan
    out["v7_rl_source_in_slate_but_not_selected_rate"] = (
        float((oracle["source_in_any_slate"] & ~oracle["source_selected"]).mean()) if len(oracle) else math.nan
    )
    out["v7_rl_no_slate_opportunity_rate"] = float((~oracle["source_in_any_slate"]).mean()) if len(oracle) else math.nan
    out["v7_rl_first_positive_round_proxy_mean"] = (
        float(oracle["first_positive_round_proxy"].dropna().mean()) if len(oracle) and oracle["first_positive_round_proxy"].notna().any() else math.nan
    )

    old_hit = old_rl_case.set_index("case_id").reindex(cases)["hit_round"] if len(cases) else pd.Series(dtype=float)
    v7_hit = v7_rl_case.set_index("case_id").reindex(cases)["hit_round"] if len(cases) else pd.Series(dtype=float)
    out["old_v3_rl_hit_round_mean"] = float(pd.to_numeric(old_hit, errors="coerce").dropna().mean()) if old_hit.notna().any() else math.nan
    out["v7_rl_hit_round_mean"] = float(pd.to_numeric(v7_hit, errors="coerce").dropna().mean()) if v7_hit.notna().any() else math.nan

    for rounds in ([1], [1, 2], [1, 2, 3]):
        ov = overlap_summary(
            "old_v3_rl_vs_v7_rl_selected",
            old_rl_step,
            v7_rl_step,
            "selected_global_ids",
            rounds,
            list(cases),
        )
        key = "r" + "".join(map(str, rounds))
        out[f"{key}_old_v3_rl_vs_v7_rl_selected_jaccard"] = ov["mean_jaccard"]
        out[f"{key}_old_v3_rl_vs_v7_rl_selected_overlap_count"] = ov["mean_overlap_count"]
        out[f"{key}_old_v3_rl_vs_v7_rl_selected_exact_rate"] = ov["exact_set_match_rate"]
    return out


def build_anchor_table(loaded: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(
        key: str,
        config: str,
        posterior_core: str,
        policy_type: str,
        supervision: str,
        seed: Any,
        epochs: Any,
        lineage: str = "",
    ) -> None:
        item = ARTIFACTS[key]
        sr = success_rate(item["summary"])
        run_summary = load_json(item.get("run_summary", ""))
        history = read_csv(item.get("history", ""))
        ppo_updates: Any = "not available"
        if history is not None and "ppo_update_steps" in history:
            ppo_updates = int(round(float(pd.to_numeric(history["ppo_update_steps"], errors="coerce").fillna(0).sum())))
        rows.append(
            {
                "config": config,
                "posterior_core": posterior_core,
                "policy_type": policy_type,
                "RL_supervision_signal": supervision,
                "seed": seed,
                "epochs": epochs,
                "PPO_updates": ppo_updates,
                "strict_val_B30_SR": sr if sr is not None else "not available",
                "delta_vs_V3_posterior_greedy": None if sr is None else float(sr - BASELINES["old_v3_greedy"]),
                "delta_vs_V7_posterior_greedy": None if sr is None else float(sr - BASELINES["v7_greedy"]),
                "delta_vs_old_V3_fresh_RL": None if sr is None else float(sr - BASELINES["old_v3_fresh_rl"]),
                "delta_vs_old_V3_strongest_RL": None if sr is None else float(sr - BASELINES["old_v3_strongest_rl"]),
                "train_full_case_count": run_summary.get("train_full_case_count", "not available"),
                "rl_init_mode": nested_get(run_summary, ["policy_spec", "rl_init_mode"], "not available"),
                "rl_anchor_start": nested_get(run_summary, ["policy_spec", "rl_teacher_anchor_start"], "not available"),
                "rl_anchor_end": nested_get(run_summary, ["policy_spec", "rl_teacher_anchor_end"], "not available"),
                "lineage": lineage,
                "summary_path": item["summary"],
            }
        )

    add("old_v3_greedy", "old V3 / 3offset / alpha=0.55", "old V3 / 3offset / alpha=0.55", "posterior-greedy", "none", "not available", "not available")
    add("v7_greedy", "V7 / 7offset / alpha=0.55", "V7 / 7offset / alpha=0.55", "posterior-greedy", "none", "not available", "not available")
    add("old_v3_fresh_rl", "old V3 fresh random_init RL", "old V3 / 3offset / alpha=0.55", "fresh random_init RL with imitation_anchor", "imitation_anchor", 42, 3)
    add(
        "old_v3_strongest_rl",
        "old V3 strongest greedy RL",
        "old V3 / 3offset / alpha=0.55",
        "continuation RL",
        "continuation lineage contains prior RL checkpoint; imitation_anchor in continuation history is 0",
        45,
        "3 parent + 1 continuation",
        "stronger_value_head_v2 parent 3 epochs + 1 continuation epoch",
    )
    add("v7_anchor_seed45_rl", "V7 fresh random_init RL seed45", "V7 / 7offset / alpha=0.55", "fresh random_init RL with imitation_anchor", "imitation_anchor 0.05 -> 0.025 -> 0.0", 45, 3)
    add("v7_anchor_seed46_rl", "V7 fresh random_init RL seed46", "V7 / 7offset / alpha=0.55", "fresh random_init RL with imitation_anchor", "imitation_anchor 0.05 -> 0.025 -> 0.0", 46, 3)
    add("v7_pure_seed45_rl", "V7 pure RL no imitation_anchor seed45", "V7 / 7offset / alpha=0.55", "fresh random_init pure RL no imitation_anchor", "none", 45, 3)
    add("v7_pure_seed46_rl", "V7 pure RL no imitation_anchor seed46", "V7 / 7offset / alpha=0.55", "fresh random_init pure RL no imitation_anchor", "none", 46, 3)
    return pd.DataFrame(rows)


def build_case_partitions(case_dfs: Dict[str, pd.DataFrame], step_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    v3_g = success_map(case_dfs["old_v3_greedy"])
    v7_g = success_map(case_dfs["v7_greedy"])
    old_rl = success_map(case_dfs["old_v3_fresh_rl"])
    flags = state_flags(step_dfs["v7_greedy"])
    all_cases = sorted(set(v3_g.index) & set(v7_g.index) & set(old_rl.index))

    rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    for rl_key in ["v7_anchor_seed45_rl", "v7_pure_seed45_rl", "v7_pure_seed46_rl"]:
        if rl_key not in case_dfs:
            continue
        v7_rl = success_map(case_dfs[rl_key])
        cases = sorted(set(all_cases) & set(v7_rl.index))
        a_cases = [c for c in cases if bool(v7_g[c]) and not bool(v3_g[c])]
        b_cases = [c for c in cases if not bool(v7_g[c]) and bool(v3_g[c])]
        c_cases = [c for c in cases if bool(old_rl[c]) and not bool(v7_rl[c])]
        d_cases = [c for c in cases if bool(v7_rl[c]) and not bool(old_rl[c])]
        partitions = {
            "A_v7_greedy_wins_over_v3_greedy": a_cases,
            "B_v7_greedy_loses_to_v3_greedy": b_cases,
            "C_old_v3_rl_wins_v7_rl_loses": c_cases,
            "D_v7_rl_wins_old_v3_rl_loses": d_cases,
        }
        for name, part_cases in partitions.items():
            row = {
                "v7_rl_policy": rl_key,
                "partition": name,
                "case_count": int(len(part_cases)),
                "v7_rl_success_rate_on_partition": float(v7_rl.reindex(part_cases).mean()) if part_cases else math.nan,
                "old_v3_rl_success_rate_on_partition": float(old_rl.reindex(part_cases).mean()) if part_cases else math.nan,
                "v7_rl_success_count_on_partition": int(v7_rl.reindex(part_cases).fillna(False).sum()) if part_cases else 0,
                "old_v3_rl_success_count_on_partition": int(old_rl.reindex(part_cases).fillna(False).sum()) if part_cases else 0,
            }
            if name.startswith("A_"):
                row["v7_rl_preserved_v7_greedy_win_rate"] = row["v7_rl_success_rate_on_partition"]
            if name.startswith("B_"):
                row["v7_rl_repair_rate"] = row["v7_rl_success_rate_on_partition"]
                row["old_v3_rl_repair_rate"] = row["old_v3_rl_success_rate_on_partition"]
            rows.append(row)
        for name, part_cases in [("C_old_v3_rl_wins_v7_rl_loses", c_cases), ("D_v7_rl_wins_old_v3_rl_loses", d_cases)]:
            diag = partition_diagnostics(
                name=name,
                cases=part_cases,
                v7_step=step_dfs["v7_greedy"],
                v7_rl_case=case_dfs[rl_key],
                v7_rl_step=step_dfs[rl_key],
                old_rl_case=case_dfs["old_v3_fresh_rl"],
                old_rl_step=step_dfs["old_v3_fresh_rl"],
                flags=flags,
            )
            diag["v7_rl_policy"] = rl_key
            diag_rows.append(diag)
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def build_slate_overlap(case_dfs: Dict[str, pd.DataFrame], step_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cases = sorted(set(case_dfs["old_v3_greedy"]["case_id"].astype(str)) & set(case_dfs["v7_greedy"]["case_id"].astype(str)))
    rows: List[Dict[str, Any]] = []
    for rounds in ([1], [1, 2], [1, 2, 3]):
        rows.append(overlap_summary("v3_greedy_vs_v7_greedy_slate", step_dfs["old_v3_greedy"], step_dfs["v7_greedy"], "policy_slate_global_ids", rounds, cases))
        rows.append(overlap_summary("v3_greedy_vs_v7_greedy_selected", step_dfs["old_v3_greedy"], step_dfs["v7_greedy"], "selected_global_ids", rounds, cases))
        for rl_key in ["v7_anchor_seed45_rl", "v7_pure_seed45_rl", "v7_pure_seed46_rl"]:
            if rl_key in step_dfs:
                common = sorted(set(case_dfs["old_v3_fresh_rl"]["case_id"].astype(str)) & set(case_dfs[rl_key]["case_id"].astype(str)))
                rows.append(
                    overlap_summary(
                        f"old_v3_fresh_rl_vs_{rl_key}_selected",
                        step_dfs["old_v3_fresh_rl"],
                        step_dfs[rl_key],
                        "selected_global_ids",
                        rounds,
                        common,
                    )
                )
                if "policy_slate_global_ids" in step_dfs["old_v3_strongest_rl"].columns and "policy_slate_global_ids" in step_dfs[rl_key].columns:
                    common_strong = sorted(
                        set(case_dfs["old_v3_strongest_rl"]["case_id"].astype(str)) & set(case_dfs[rl_key]["case_id"].astype(str))
                    )
                    rows.append(
                        overlap_summary(
                            f"old_v3_strongest_rl_vs_{rl_key}_slate",
                            step_dfs["old_v3_strongest_rl"],
                            step_dfs[rl_key],
                            "policy_slate_global_ids",
                            rounds,
                            common_strong,
                        )
                    )
                    rows.append(
                        overlap_summary(
                            f"old_v3_strongest_rl_vs_{rl_key}_selected",
                            step_dfs["old_v3_strongest_rl"],
                            step_dfs[rl_key],
                            "selected_global_ids",
                            rounds,
                            common_strong,
                        )
                    )
    return pd.DataFrame(rows)


def build_posterior_softness(step_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label in ["old_v3_greedy", "v7_greedy"]:
        df = step_dfs[label]
        for scope, scoped in [
            ("all_steps", df),
            ("round1", df[pd.to_numeric(df["round_index"], errors="coerce") == 1]),
        ]:
            row: Dict[str, Any] = {"posterior_core": label, "scope": scope, "row_count": int(len(scoped))}
            for col in ["posterior_entropy", "top1_top2_margin", "top1_mass", "top3_mass", "top5_mass", "effective_support_size_ratio"]:
                if col in scoped:
                    vals = pd.to_numeric(scoped[col], errors="coerce")
                    row[f"{col}_mean"] = float(vals.mean())
                    row[f"{col}_median"] = float(vals.median())
                    row[f"{col}_p25"] = float(vals.quantile(0.25))
                    row[f"{col}_p75"] = float(vals.quantile(0.75))
            rows.append(row)
    out = pd.DataFrame(rows)
    return out


def build_oracle_tables(case_dfs: Dict[str, pd.DataFrame], step_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[pd.DataFrame] = []
    per_case_rows: List[pd.DataFrame] = []
    for key in [
        "old_v3_greedy",
        "v7_greedy",
        "old_v3_fresh_rl",
        "old_v3_strongest_rl",
        "v7_anchor_seed45_rl",
        "v7_pure_seed45_rl",
        "v7_pure_seed46_rl",
    ]:
        if key not in case_dfs or key not in step_dfs:
            continue
        summary, per_case = aggregate_oracle(key, case_dfs[key], step_dfs[key])
        summary_rows.append(summary)
        per_case["policy"] = key
        per_case_rows.append(per_case)
    return pd.concat(summary_rows, ignore_index=True), pd.concat(per_case_rows, ignore_index=True)


def build_within_slate_regret(case_dfs: Dict[str, pd.DataFrame], step_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    v7_g_success = success_map(case_dfs["v7_greedy"])
    sources = case_source_map(case_dfs["v7_greedy"])
    for rl_key in ["v7_anchor_seed45_rl", "v7_pure_seed45_rl", "v7_pure_seed46_rl"]:
        if rl_key not in case_dfs:
            continue
        rl_success = success_map(case_dfs[rl_key])
        loss_cases = [c for c in rl_success.index if not bool(rl_success[c])]
        oracle = true_source_oracle(case_dfs[rl_key][case_dfs[rl_key]["case_id"].astype(str).isin(loss_cases)], step_dfs[rl_key])
        direct_missed = 0
        for case_id in loss_cases:
            source = sources.get(str(case_id))
            if source is None:
                continue
            greedy_steps = step_dfs["v7_greedy"][step_dfs["v7_greedy"]["case_id"].astype(str) == str(case_id)]
            rl_steps = step_dfs[rl_key][step_dfs[rl_key]["case_id"].astype(str) == str(case_id)]
            for g_step in greedy_steps.itertuples(index=False):
                g_selected = parse_list(getattr(g_step, "selected_global_ids", None))
                if int(source) not in set(g_selected):
                    continue
                round_index = int(getattr(g_step, "round_index"))
                same_round = rl_steps[pd.to_numeric(rl_steps["round_index"], errors="coerce") == round_index]
                if same_round.empty:
                    continue
                r = same_round.iloc[0]
                r_slate = parse_list(r.get("policy_slate_global_ids"))
                r_selected = parse_list(r.get("selected_global_ids"))
                if int(source) in set(r_slate) and int(source) not in set(r_selected):
                    direct_missed += 1
                break
        first_pos_missing = oracle["first_positive_round_proxy"].isna() if len(oracle) else pd.Series(dtype=bool)
        rows.append(
            {
                "v7_rl_policy": rl_key,
                "v7_rl_loss_case_count": int(len(loss_cases)),
                "loss_no_slate_opportunity_count": int((~oracle["source_in_any_slate"]).sum()) if len(oracle) else 0,
                "loss_no_slate_opportunity_rate": float((~oracle["source_in_any_slate"]).mean()) if len(oracle) else math.nan,
                "loss_source_in_slate_but_not_selected_count": int((oracle["source_in_any_slate"] & ~oracle["source_selected"]).sum()) if len(oracle) else 0,
                "loss_source_in_slate_but_not_selected_rate": float((oracle["source_in_any_slate"] & ~oracle["source_selected"]).mean()) if len(oracle) else math.nan,
                "loss_cases_where_v7_greedy_succeeded_count": int(v7_g_success.reindex(loss_cases).fillna(False).sum()) if loss_cases else 0,
                "direct_within_slate_missed_v7_greedy_hit_count": int(direct_missed),
                "loss_without_positive_observation_count_proxy": int(first_pos_missing.sum()) if len(oracle) else 0,
                "loss_with_positive_but_no_hit_count_proxy": int((~first_pos_missing).sum()) if len(oracle) else 0,
            }
        )
    return pd.DataFrame(rows)


def build_training_audit() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key in ["old_v3_fresh_rl", "v7_anchor_seed45_rl", "v7_anchor_seed46_rl", "v7_pure_seed45_rl", "v7_pure_seed46_rl"]:
        item = ARTIFACTS[key]
        hist = read_csv(item.get("history", ""))
        if hist is None:
            rows.append({"policy": key, "available": False})
            continue
        for row in hist.to_dict("records"):
            out = {"policy": key, "available": True}
            for col in [
                "epoch",
                "train_success_rate",
                "heldout_success_rate",
                "entropy_coef",
                "imitation_anchor",
                "ppo_policy_loss",
                "ppo_value_loss",
                "ppo_entropy",
                "ppo_anchor_loss",
                "ppo_update_steps",
                "ppo_set_aux_clip_frac",
            ]:
                out[col] = row.get(col, "not available")
            out["explained_variance"] = "not available"
            out["approx_kl"] = "not available"
            out["clip_fraction"] = row.get("ppo_set_aux_clip_frac", "not available")
            out["rollout_return"] = "not available"
            rows.append(out)
    return pd.DataFrame(rows)


def value_calibration(output_dir: Path, device_arg: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    from src.scripts.run_spim_policy_eval_strict import (
        _load_model_for_mode,
        _resolve_mode_branch,
        build_runtime_strict,
        run_policy_on_cases_strict,
    )
    from src.scripts.run_spim_teacher_imitation_rl_pilot import (
        DEFAULT_CACHE_DIR,
        DEFAULT_SOURCE_ROOT,
        _prepare_ppo_targets,
        get_device,
    )
    from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv

    device = get_device(device_arg)
    source_root = project_path(DEFAULT_SOURCE_ROOT)
    cache_dir = project_path(DEFAULT_CACHE_DIR)
    runtime, _ = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split="val",
        num_rounds=10,
        actions_per_round=3,
        train_max_cases=0,
        train_cache_version="",
        case_limit=0,
    )
    env = CleanTwoChannelEvidenceEnv()
    specs = [
        {
            "policy": "old_v3_fresh_rl",
            "family": "hsr_soft_scenario_posterior_v3",
            "checkpoint": ARTIFACTS["old_v3_fresh_rl"]["checkpoint"],
            "value_mlp_depth": 2,
            "value_head_width_mult": 1.0,
            "legacy_linear_indexing": True,
        },
        {
            "policy": "v7_anchor_seed45_rl",
            "family": "hsr_soft_scenario_posterior_v7_7offset",
            "checkpoint": ARTIFACTS["v7_anchor_seed45_rl"]["checkpoint"],
            "value_mlp_depth": 3,
            "value_head_width_mult": 2.0,
        },
        {
            "policy": "v7_pure_seed45_rl",
            "family": "hsr_soft_scenario_posterior_v7_7offset",
            "checkpoint": ARTIFACTS["v7_pure_seed45_rl"]["checkpoint"],
            "value_mlp_depth": 3,
            "value_head_width_mult": 2.0,
        },
    ]
    transition_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for spec in specs:
        checkpoint = project_path(spec["checkpoint"])
        if not checkpoint.exists():
            summary_rows.append({"policy": spec["policy"], "available": False, "reason": "checkpoint_missing"})
            continue
        if spec.get("legacy_linear_indexing"):
            state_dict = torch.load(checkpoint, map_location="cpu")
            remapped = {}
            for key, value in state_dict.items():
                new_key = key
                for prefix in ("action_mlp", "value_mlp"):
                    new_key = new_key.replace(f"{prefix}.2.", f"{prefix}.3.")
                    new_key = new_key.replace(f"{prefix}.4.", f"{prefix}.6.")
                remapped[new_key] = value
            checkpoint = output_dir / f"{spec['policy']}_legacy_remapped_for_calibration.pt"
            torch.save(remapped, checkpoint)
        branch, ckpt = _resolve_mode_branch("rl", str(checkpoint))
        model, checkpoint_info = _load_model_for_mode(
            policy_mode="rl",
            checkpoint=ckpt,
            device=device,
            hidden_dim=128,
            include_surrogate_features=False,
            include_uncertainty_regime_features=False,
            policy_arch="separate_heads",
            policy_mlp_depth=2,
            value_mlp_depth=int(spec["value_mlp_depth"]),
            value_head_width_mult=float(spec["value_head_width_mult"]),
            critic_trunk_depth=0,
            critic_trunk_hidden_dim=0,
            policy_dropout=0.0,
            policy_norm="none",
            candidate_encoder="none",
            candidate_attn_heads=4,
            enable_early_stage_specialist_head=False,
            early_stage_round_cutoff=0,
            enable_regime_head=False,
            regime_head_classes=3,
            regime_embed_dim=12,
            arch_backbone="baseline_mlp",
            residual_hidden_dim=256,
            residual_depth=4,
            residual_head_dim=128,
            transformer_token_dim=128,
            transformer_layers=2,
            transformer_heads=4,
            transformer_ffn_dim=256,
            graph_hidden_dim=128,
            graph_layers=2,
            graph_heads=4,
            graph_max_subgraph_nodes=512,
            graph_use_onehop=False,
            cnn_channels=128,
            cnn_kernel_size=3,
            cnn_norm="layernorm",
        )
        assert model is not None
        with torch.no_grad():
            rollout = run_policy_on_cases_strict(
                cases=runtime["cases"],
                family=str(spec["family"]),
                runtime=runtime,
                env=env,
                policy_mode="rl",
                requested_policy_name=str(spec["policy"]),
                model=model,
                branch_taken=branch,
                deterministic=True,
                base_seed=45,
                include_surrogate_features=False,
                include_uncertainty_regime_features=False,
                top_source_k=8,
                paper_like_alpha=0.55,
                paper_like_topk_fraction=0.12,
                paper_like_time_tol_min=30.0,
                soft_scenario_beta=2.0,
                hit_reward=1.0,
                step_penalty=-1.0 / 30.0,
                reward_family="reward_r0_terminal_step",
                reward_lambda_cover=0.0,
                reward_lambda_error=0.0,
                reward_cover_delta_clip=0.2,
                reward_error_delta_clip=2.0,
                reward_topk_fraction=0.12,
                reward_time_tol_min=30.0,
                device=device,
                collect_transitions=True,
                trace_case_limit=0,
                trace_step_limit=0,
                checkpoint_info=checkpoint_info,
                slate_size=10,
                slate_top_posterior_k=8,
                slate_high_disagreement_k=1,
                slate_novelty_k=1,
                early_stage_round_cutoff=0,
                early_stage_slate_top_posterior_k=None,
                early_stage_slate_high_disagreement_k=None,
                early_stage_slate_novelty_k=None,
                decode_mode="greedy",
            )
        transitions = list(rollout["transitions"])
        _prepare_ppo_targets(transitions, gamma=0.97, baseline_returns=None)
        for tr in transitions:
            transition_rows.append(
                {
                    "policy": spec["policy"],
                    "case_id": tr["case_id"],
                    "episode_index": int(tr["episode_index"]),
                    "predicted_value": float(tr["old_value"]),
                    "realized_return": float(tr["raw_return"]),
                    "reward": float(tr["reward"]),
                }
            )
    trans_df = pd.DataFrame(transition_rows)
    if trans_df.empty:
        return pd.DataFrame(summary_rows), trans_df
    bin_rows: List[Dict[str, Any]] = []
    for policy, group in trans_df.groupby("policy"):
        pred = group["predicted_value"].astype(float)
        ret = group["realized_return"].astype(float)
        corr = float(pred.corr(ret)) if len(group) > 1 and pred.std() > 0 and ret.std() > 0 else math.nan
        summary_rows.append(
            {
                "policy": policy,
                "available": True,
                "transition_count": int(len(group)),
                "predicted_value_mean": float(pred.mean()),
                "realized_return_mean": float(ret.mean()),
                "bias_pred_minus_return": float((pred - ret).mean()),
                "mae": float((pred - ret).abs().mean()),
                "mse": float(((pred - ret) ** 2).mean()),
                "pearson_corr": corr,
            }
        )
        try:
            bins = pd.qcut(pred.rank(method="first"), q=5, labels=False)
        except ValueError:
            bins = pd.Series(np.zeros(len(group), dtype=int), index=group.index)
        tmp = group.copy()
        tmp["value_bin"] = bins.to_numpy()
        for value_bin, bin_group in tmp.groupby("value_bin"):
            bpred = bin_group["predicted_value"].astype(float)
            bret = bin_group["realized_return"].astype(float)
            bin_rows.append(
                {
                    "policy": policy,
                    "value_bin": int(value_bin),
                    "count": int(len(bin_group)),
                    "predicted_value_mean": float(bpred.mean()),
                    "realized_return_mean": float(bret.mean()),
                    "bias_pred_minus_return": float((bpred - bret).mean()),
                    "mae": float((bpred - bret).abs().mean()),
                }
            )
    bins_df = pd.DataFrame(bin_rows)
    bins_df.to_csv(output_dir / "value_calibration_bins.csv", index=False)
    trans_df.to_csv(output_dir / "value_calibration_transitions.csv", index=False)
    return pd.DataFrame(summary_rows), bins_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/spim_set_level_rl_mainline/20260506_v7_7offset_alpha055_pure_rl_no_anchor_seed45_v1/analysis",
    )
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--calibration-device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, Dict[str, Any]] = {}
    case_dfs: Dict[str, pd.DataFrame] = {}
    step_dfs: Dict[str, pd.DataFrame] = {}
    for key, paths in ARTIFACTS.items():
        summary_path = project_path(paths["summary"])
        case_path = project_path(paths.get("case", ""))
        step_path = project_path(paths.get("step", ""))
        loaded[key] = {
            "summary_exists": summary_path.exists(),
            "case_exists": case_path.exists(),
            "step_exists": step_path.exists(),
        }
        case = read_csv(paths.get("case", ""))
        step = read_csv(paths.get("step", ""))
        if case is not None:
            case["case_id"] = case["case_id"].astype(str)
            case_dfs[key] = case
        if step is not None:
            step["case_id"] = step["case_id"].astype(str)
            step_dfs[key] = step

    anchor_table = build_anchor_table(loaded)
    anchor_table.to_csv(output_dir / "stage0_anchor_table.csv", index=False)

    required = {"old_v3_greedy", "v7_greedy", "old_v3_fresh_rl", "v7_anchor_seed45_rl"}
    if required.issubset(case_dfs) and required.issubset(step_dfs):
        partition_summary, partition_diagnostics_df = build_case_partitions(case_dfs, step_dfs)
        partition_summary.to_csv(output_dir / "case_partition_summary.csv", index=False)
        partition_diagnostics_df.to_csv(output_dir / "case_partition_diagnostics.csv", index=False)
        build_slate_overlap(case_dfs, step_dfs).to_csv(output_dir / "slate_overlap_summary.csv", index=False)
        build_posterior_softness(step_dfs).to_csv(output_dir / "posterior_softness_summary.csv", index=False)
        oracle_summary, oracle_per_case = build_oracle_tables(case_dfs, step_dfs)
        oracle_summary.to_csv(output_dir / "slate_oracle_summary.csv", index=False)
        oracle_per_case.to_csv(output_dir / "slate_oracle_per_case.csv", index=False)
        build_within_slate_regret(case_dfs, step_dfs).to_csv(output_dir / "within_slate_regret_summary.csv", index=False)

    training_audit = build_training_audit()
    training_audit.to_csv(output_dir / "value_training_log_audit.csv", index=False)

    calibration_summary: Optional[pd.DataFrame] = None
    if args.include_calibration:
        calibration_summary, _ = value_calibration(output_dir, args.calibration_device)
        calibration_summary.to_csv(output_dir / "value_calibration_summary.csv", index=False)

    manifest = {
        "runner": "tools/analyze_v7_rl_breakpoint.py",
        "loaded": loaded,
        "outputs": sorted(str(p.relative_to(PROJECT_ROOT)) for p in output_dir.glob("*")),
        "calibration_included": bool(args.include_calibration),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
