from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.scripts.run_posterior_like_belief_audit import write_json


RUNNER_VERSION = "teacher_relative_slot3_capture_gap_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc strict val slot3 upper-bound + capture-gap diagnosis.")
    parser.add_argument("--strict-summary", type=str, required=True, help="summary.json from strict residual run")
    parser.add_argument("--strict-root", type=str, required=True, help="strict residual run output directory")
    parser.add_argument("--eval-audit-root", type=str, required=True, help="strict val audit root with slot3 candidate/state csvs")
    parser.add_argument(
        "--exact-summary",
        type=str,
        default="",
        help="optional summary.json from exact136 residual run for headroom comparison",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    if abs(float(den)) <= 1e-12:
        return 0.0
    return float(num) / float(den)


def _f1(precision: float, recall: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _qcut_labels(series: pd.Series, q: int, prefix: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    uniq = s.dropna().nunique()
    if uniq <= 1:
        return pd.Series([f"{prefix}_flat"] * len(s), index=s.index)
    bins = min(int(q), int(uniq))
    cat = pd.qcut(s, q=bins, duplicates="drop")
    if not hasattr(cat, "cat"):
        return pd.Series([f"{prefix}_flat"] * len(s), index=s.index)
    n = len(cat.cat.categories)
    labels = [f"{prefix}_q{i+1}" for i in range(n)]
    return cat.cat.rename_categories(labels).astype(str)


def _load_panel_data(
    strict_root: Path,
    eval_audit_root: Path,
    panel: str,
    mode: str,
) -> pd.DataFrame:
    decision_path = strict_root / f"{panel}_{mode}_decisions.csv"
    state_path = eval_audit_root / f"{panel}_slot3_state_summary.csv"
    cand_path = eval_audit_root / f"{panel}_slot3_candidate_rows.csv"
    decisions = pd.read_csv(decision_path)
    states = pd.read_csv(state_path)
    cands = pd.read_csv(cand_path)
    cands = cands[cands["policy_source"].astype(str) == "teacher"].copy()
    states = states[states["policy_source"].astype(str) == "teacher"].copy()

    merged = decisions.merge(
        states[
            [
                "state_key",
                "episode_index",
                "remaining_budget",
                "posterior_entropy",
                "top1_top2_margin",
                "best_slot3_local",
                "best_delta_return_vs_teacher_slot3",
                "best_delta_success_vs_teacher_slot3",
            ]
        ],
        on="state_key",
        how="left",
        validate="one_to_one",
    )
    chosen = cands[
        [
            "state_key",
            "candidate_slot3_local",
            "delta_return_vs_teacher_slot3",
            "delta_success_vs_teacher_slot3",
        ]
    ].rename(
        columns={
            "candidate_slot3_local": "pred_slot3_local",
            "delta_return_vs_teacher_slot3": "chosen_delta_return",
            "delta_success_vs_teacher_slot3": "chosen_delta_success",
        }
    )
    merged = merged.merge(chosen, on=["state_key", "pred_slot3_local"], how="left", validate="many_to_one")
    merged["oracle_replace_needed"] = merged["best_delta_return_vs_teacher_slot3"].astype(float) > 0.0
    merged["oracle_replace_needed_success"] = merged["best_delta_success_vs_teacher_slot3"].astype(float) > 0.0
    merged["pred_replace"] = merged["replace_applied"].astype(float) > 0.5
    merged["chosen_is_oracle_best"] = merged["pred_slot3_local"].astype(int) == merged["best_slot3_local"].astype(int)
    merged["chosen_positive"] = merged["chosen_delta_return"].astype(float) > 0.0
    merged["chosen_negative"] = merged["chosen_delta_return"].astype(float) < 0.0
    merged["chosen_nonbest_positive"] = merged["chosen_positive"] & (~merged["chosen_is_oracle_best"])
    merged["episode_bucket"] = merged["episode_index"].apply(lambda v: f"ep{int(v)}")
    merged["remaining_budget_bucket"] = pd.cut(
        merged["remaining_budget"].astype(float),
        bins=[-np.inf, 10, 20, 30, 40, 60, np.inf],
        labels=["<=10", "11-20", "21-30", "31-40", "41-60", ">60"],
    ).astype(str)
    merged["entropy_bucket"] = _qcut_labels(merged["posterior_entropy"], q=3, prefix="entropy")
    merged["margin_bucket"] = _qcut_labels(merged["top1_top2_margin"], q=3, prefix="margin")
    merged["positive_regret_bucket"] = merged["oracle_replace_needed"].map({True: "positive_regret", False: "non_positive_regret"})
    return merged


def _decision_metrics(df: pd.DataFrame) -> Dict[str, float]:
    pos = df["oracle_replace_needed"]
    pred = df["pred_replace"]
    tp = int((pos & pred).sum())
    fn = int((pos & (~pred)).sum())
    fp = int(((~pos) & pred).sum())
    tn = int(((~pos) & (~pred)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "states": int(len(df)),
        "oracle_replace_states": int(pos.sum()),
        "oracle_keep_states": int((~pos).sum()),
        "tp_replace": tp,
        "fn_keep_when_oracle_replace": fn,
        "fp_replace_when_oracle_keep": fp,
        "tn_keep": tn,
        "precision_replace": precision,
        "recall_replace": recall,
        "f1_replace": _f1(precision, recall),
        "keep_rate_overall": float((~pred).mean()) if len(df) else 0.0,
        "replace_rate_overall": float(pred.mean()) if len(df) else 0.0,
    }


def _selection_metrics(df: pd.DataFrame) -> Dict[str, float]:
    sub = df[df["oracle_replace_needed"]].copy()
    if len(sub) <= 0:
        return {
            "oracle_replace_states": 0,
            "hit_oracle_best_rate": 0.0,
            "hit_nonbest_positive_rate": 0.0,
            "hit_any_positive_rate": 0.0,
            "hit_negative_rate": 0.0,
            "hit_zero_rate": 0.0,
            "local_capture_return_ratio": 0.0,
            "local_capture_success_ratio": 0.0,
            "mean_best_delta_return": 0.0,
            "mean_chosen_delta_return": 0.0,
            "mean_best_delta_success": 0.0,
            "mean_chosen_delta_success": 0.0,
        }
    chosen = sub["chosen_delta_return"].astype(float)
    best = sub["best_delta_return_vs_teacher_slot3"].astype(float)
    chosen_s = sub["chosen_delta_success"].astype(float)
    best_s = sub["best_delta_success_vs_teacher_slot3"].astype(float)
    return {
        "oracle_replace_states": int(len(sub)),
        "hit_oracle_best_rate": float(sub["chosen_is_oracle_best"].mean()),
        "hit_nonbest_positive_rate": float(sub["chosen_nonbest_positive"].mean()),
        "hit_any_positive_rate": float(sub["chosen_positive"].mean()),
        "hit_negative_rate": float(sub["chosen_negative"].mean()),
        "hit_zero_rate": float((np.isclose(chosen.values, 0.0)).mean()),
        "local_capture_return_ratio": _safe_div(float(chosen.sum()), float(best.sum())),
        "local_capture_success_ratio": _safe_div(float(chosen_s.sum()), float(best_s.sum())),
        "mean_best_delta_return": float(best.mean()),
        "mean_chosen_delta_return": float(chosen.mean()),
        "mean_best_delta_success": float(best_s.mean()),
        "mean_chosen_delta_success": float(chosen_s.mean()),
    }


def _bucket_table(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for bucket, g in df.groupby(axis, dropna=False):
        metrics = _decision_metrics(g)
        sel = _selection_metrics(g)
        rows.append(
            {
                "axis": axis,
                "bucket": str(bucket),
                "states": metrics["states"],
                "oracle_replace_states": metrics["oracle_replace_states"],
                "oracle_replace_frac": _safe_div(metrics["oracle_replace_states"], metrics["states"]),
                "replace_rate": metrics["replace_rate_overall"],
                "precision_replace": metrics["precision_replace"],
                "recall_replace": metrics["recall_replace"],
                "f1_replace": metrics["f1_replace"],
                "hit_oracle_best_rate": sel["hit_oracle_best_rate"],
                "hit_any_positive_rate": sel["hit_any_positive_rate"],
                "hit_negative_rate": sel["hit_negative_rate"],
                "local_capture_return_ratio": sel["local_capture_return_ratio"],
                "local_capture_success_ratio": sel["local_capture_success_ratio"],
                "mean_best_delta_return": sel["mean_best_delta_return"],
                "mean_chosen_delta_return": sel["mean_chosen_delta_return"],
                "mean_best_delta_success": sel["mean_best_delta_success"],
                "mean_chosen_delta_success": sel["mean_chosen_delta_success"],
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["states", "bucket"], ascending=[False, True]).reset_index(drop=True)
    return out


def _panel_upper_bound_state_stats(state_df: pd.DataFrame) -> Dict[str, float]:
    best_r = state_df["best_delta_return_vs_teacher_slot3"].astype(float)
    best_s = state_df["best_delta_success_vs_teacher_slot3"].astype(float)
    pos_r = best_r[best_r > 0.0]
    pos_s = best_s[best_s > 0.0]
    return {
        "state_count": int(len(state_df)),
        "positive_regret_state_fraction": float((best_r > 0.0).mean()) if len(best_r) else 0.0,
        "positive_regret_mean_delta_return_conditional": float(pos_r.mean()) if len(pos_r) else 0.0,
        "positive_regret_mean_delta_return_all_states": float(best_r.mean()) if len(best_r) else 0.0,
        "positive_success_state_fraction": float((best_s > 0.0).mean()) if len(best_s) else 0.0,
        "positive_success_mean_delta_conditional": float(pos_s.mean()) if len(pos_s) else 0.0,
        "positive_success_mean_delta_all_states": float(best_s.mean()) if len(best_s) else 0.0,
    }


def _panel_capture_vs_oracle(panel_summary: Dict[str, Any], mode: str) -> Dict[str, float]:
    teacher = panel_summary["teacher"]
    oracle = panel_summary["oracle_slot3_upper_bound"]
    learner = panel_summary[mode]
    return {
        "teacher_success": float(teacher["success_rate"]),
        "teacher_return_r0": float(teacher["avg_return_r0"]),
        "oracle_success": float(oracle["success_rate"]),
        "oracle_return_r0": float(oracle["avg_return_r0"]),
        "oracle_delta_success": float(oracle["delta_success_vs_teacher"]),
        "oracle_delta_return_r0": float(oracle["delta_return_r0_vs_teacher"]),
        "learner_success": float(learner["success_rate"]),
        "learner_return_r0": float(learner["avg_return_r0"]),
        "learner_delta_success": float(learner["delta_success_vs_teacher"]),
        "learner_delta_return_r0": float(learner["delta_return_r0_vs_teacher"]),
        "capture_ratio_success": _safe_div(
            float(learner["delta_success_vs_teacher"]),
            float(oracle["delta_success_vs_teacher"]),
        ),
        "capture_ratio_return_r0": _safe_div(
            float(learner["delta_return_r0_vs_teacher"]),
            float(oracle["delta_return_r0_vs_teacher"]),
        ),
    }


def _headroom_compare_table(
    strict_summary: Dict[str, Any],
    exact_summary: Optional[Dict[str, Any]],
    panel_pairs: Iterable[Tuple[str, Optional[str]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strict_panel, exact_panel in panel_pairs:
        strict = strict_summary["panel_results"][strict_panel]["oracle_slot3_upper_bound"]
        rows.append(
            {
                "source": "strict",
                "panel": strict_panel,
                "oracle_success_rate": float(strict["success_rate"]),
                "oracle_avg_hit_round_conditional": float(strict["avg_hit_round_conditional"]),
                "oracle_avg_return_r0": float(strict["avg_return_r0"]),
                "oracle_delta_success_vs_teacher": float(strict["delta_success_vs_teacher"]),
                "oracle_delta_return_r0_vs_teacher": float(strict["delta_return_r0_vs_teacher"]),
            }
        )
        if exact_summary is not None and exact_panel is not None:
            ex = exact_summary["panel_results"][exact_panel]["oracle_slot3_upper_bound"]
            rows.append(
                {
                    "source": "exact136",
                    "panel": exact_panel,
                    "oracle_success_rate": float(ex["success_rate"]),
                    "oracle_avg_hit_round_conditional": float(ex["avg_hit_round_conditional"]),
                    "oracle_avg_return_r0": float(ex["avg_return_r0"]),
                    "oracle_delta_success_vs_teacher": float(ex["delta_success_vs_teacher"]),
                    "oracle_delta_return_r0_vs_teacher": float(ex["delta_return_r0_vs_teacher"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    strict_summary_path = Path(args.strict_summary)
    strict_root = Path(args.strict_root)
    eval_audit_root = Path(args.eval_audit_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_summary = json.loads(strict_summary_path.read_text(encoding="utf-8"))
    exact_summary = None
    if str(args.exact_summary).strip():
        exact_summary = json.loads(Path(args.exact_summary).read_text(encoding="utf-8"))

    panels = ["val_B30", "val_B60"]
    modes = ["ungated", "gated_main"]
    diagnosis: Dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "inputs": {
            "strict_summary": str(strict_summary_path),
            "strict_root": str(strict_root),
            "eval_audit_root": str(eval_audit_root),
            "exact_summary": str(args.exact_summary) if str(args.exact_summary).strip() else None,
        },
        "panel_mode_metrics": {},
        "upper_bound_state_stats": {},
    }

    all_bucket_rows: List[pd.DataFrame] = []
    all_decision_rows: List[Dict[str, Any]] = []
    all_selection_rows: List[Dict[str, Any]] = []
    capture_rows: List[Dict[str, Any]] = []

    for panel in panels:
        state_df = pd.read_csv(eval_audit_root / f"{panel}_slot3_state_summary.csv")
        state_df = state_df[state_df["policy_source"].astype(str) == "teacher"].copy()
        diagnosis["upper_bound_state_stats"][panel] = _panel_upper_bound_state_stats(state_df)

        diagnosis["panel_mode_metrics"][panel] = {}
        for mode in modes:
            full_df = _load_panel_data(strict_root=strict_root, eval_audit_root=eval_audit_root, panel=panel, mode=mode)
            dm = _decision_metrics(full_df)
            sm = _selection_metrics(full_df)
            diagnosis["panel_mode_metrics"][panel][mode] = {"decision_quality": dm, "selection_quality": sm}

            all_decision_rows.append({"panel": panel, "mode": mode, **dm})
            all_selection_rows.append({"panel": panel, "mode": mode, **sm})

            for axis in [
                "episode_bucket",
                "remaining_budget_bucket",
                "entropy_bucket",
                "margin_bucket",
                "positive_regret_bucket",
            ]:
                bt = _bucket_table(full_df, axis=axis)
                if len(bt):
                    bt.insert(0, "mode", mode)
                    bt.insert(0, "panel", panel)
                    all_bucket_rows.append(bt)

            cap = _panel_capture_vs_oracle(strict_summary["panel_results"][panel], mode=mode)
            capture_rows.append({"panel": panel, "mode": mode, **cap})

    decision_df = pd.DataFrame(all_decision_rows)
    selection_df = pd.DataFrame(all_selection_rows)
    capture_df = pd.DataFrame(capture_rows)
    buckets_df = pd.concat(all_bucket_rows, ignore_index=True) if all_bucket_rows else pd.DataFrame()

    headroom_df = _headroom_compare_table(
        strict_summary=strict_summary,
        exact_summary=exact_summary,
        panel_pairs=[("val_B30", "exact136_B30"), ("val_B60", "exact136_B60")],
    )

    decision_df.to_csv(output_dir / "capture_gap_decision_quality.csv", index=False)
    selection_df.to_csv(output_dir / "capture_gap_selection_quality.csv", index=False)
    capture_df.to_csv(output_dir / "capture_gap_panel_capture_ratio.csv", index=False)
    headroom_df.to_csv(output_dir / "headroom_oracle_compare.csv", index=False)
    if len(buckets_df):
        buckets_df.to_csv(output_dir / "capture_gap_bucket_breakdown.csv", index=False)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_mode_metrics": diagnosis["panel_mode_metrics"],
        "upper_bound_state_stats": diagnosis["upper_bound_state_stats"],
        "artifacts": {
            "decision_quality_csv": str(output_dir / "capture_gap_decision_quality.csv"),
            "selection_quality_csv": str(output_dir / "capture_gap_selection_quality.csv"),
            "panel_capture_ratio_csv": str(output_dir / "capture_gap_panel_capture_ratio.csv"),
            "headroom_compare_csv": str(output_dir / "headroom_oracle_compare.csv"),
            "bucket_breakdown_csv": str(output_dir / "capture_gap_bucket_breakdown.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)

    # Lightweight markdown for fast human read.
    md_lines: List[str] = []
    md_lines.append(f"# {RUNNER_VERSION}")
    md_lines.append("")
    md_lines.append("## Inputs")
    md_lines.append(f"- strict_summary: `{strict_summary_path}`")
    md_lines.append(f"- strict_root: `{strict_root}`")
    md_lines.append(f"- eval_audit_root: `{eval_audit_root}`")
    if exact_summary is not None:
        md_lines.append(f"- exact_summary: `{args.exact_summary}`")
    md_lines.append("")
    md_lines.append("## Headroom Compare")
    if len(headroom_df):
        md_lines.append(headroom_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Capture Ratio")
    if len(capture_df):
        md_lines.append(capture_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Decision Quality")
    if len(decision_df):
        md_lines.append(decision_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Selection Quality")
    if len(selection_df):
        md_lines.append(selection_df.to_markdown(index=False))
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
