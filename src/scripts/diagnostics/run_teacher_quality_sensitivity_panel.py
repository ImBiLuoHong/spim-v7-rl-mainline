from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded teacher-quality sensitivity panel on one 500-train subset.")
    parser.add_argument("--baseline-case-csv", type=str, required=True)
    parser.add_argument("--oracle-step-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def baseline_summary(case_df: pd.DataFrame) -> Dict[str, Any]:
    valid = case_df[case_df["valid_case"] == True].copy()
    total = int(len(case_df))
    return {
        "source": "frozen_pbest_baseline_init",
        "case_count": total,
        "valid_final_ranking_case_count": int(len(valid)),
        "valid_final_ranking_case_fraction": float(len(valid) / max(total, 1)),
        "source_retained_final_fraction": float((valid["final_true_source_rank"].notna()).mean()) if len(valid) else float("nan"),
        "final_candidate_size_mean": float(valid["final_pre_action_valid_size"].mean()) if len(valid) else float("nan"),
        "final_candidate_fraction_mean": float((valid["final_pre_action_valid_size"] / valid["total_nodes"]).mean()) if len(valid) else float("nan"),
        "final_revealed_ratio_mean": float(valid["final_revealed_ratio"].mean()) if len(valid) else float("nan"),
        "final_sampled_ratio_mean": float(valid["final_sampled_ratio"].mean()) if len(valid) else float("nan"),
        "final_top1_hit": float(valid["final_top1_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "final_top5_hit": float(valid["final_top5_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "final_mrr": float(valid["final_mrr"].mean()) if len(valid) else float("nan"),
        "final_true_source_rank_mean": float(valid["final_true_source_rank"].mean()) if len(valid) else float("nan"),
    }


def oracle_summary(step_df: pd.DataFrame) -> Dict[str, Any]:
    total_cases = int(step_df["case_id"].nunique())
    last = (
        step_df.sort_values(["case_id", "round_index"])
        .groupby("case_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    valid = last[last["oracle_true_source_rank_after"].notna()].copy()
    return {
        "source": "task_defined_oracle_candidate",
        "case_count": total_cases,
        "valid_final_ranking_case_count": int(len(valid)),
        "valid_final_ranking_case_fraction": float(len(valid) / max(total_cases, 1)),
        "source_retained_final_fraction": float((valid["oracle_true_source_rank_after"].notna()).mean()) if len(valid) else float("nan"),
        "final_candidate_size_mean": float(valid["candidate_size_after_oracle"].mean()) if len(valid) else float("nan"),
        "final_candidate_fraction_mean": float((valid["candidate_size_after_oracle"] / valid["candidate_size"]).mean()) if len(valid) else float("nan"),
        "final_revealed_ratio_mean": float("nan"),
        "final_sampled_ratio_mean": float("nan"),
        "final_top1_hit": float(valid["oracle_top1_after"].mean()) if len(valid) else float("nan"),
        "final_top5_hit": float(valid["oracle_top5_after"].mean()) if len(valid) else float("nan"),
        "final_mrr": float(valid["oracle_mrr_after"].mean()) if len(valid) else float("nan"),
        "final_true_source_rank_mean": float(valid["oracle_true_source_rank_after"].mean()) if len(valid) else float("nan"),
        "oracle_vs_frozen_delta_mrr_mean": float(step_df["oracle_vs_frozen_delta_mrr"].mean()) if len(step_df) else float("nan"),
        "oracle_vs_frozen_delta_top1_mean": float(step_df["oracle_vs_frozen_delta_top1"].mean()) if len(step_df) else float("nan"),
        "oracle_vs_frozen_delta_top5_mean": float(step_df["oracle_vs_frozen_delta_top5"].mean()) if len(step_df) else float("nan"),
        "oracle_vs_frozen_delta_rank_mean": float(step_df["oracle_vs_frozen_delta_rank"].mean()) if len(step_df) else float("nan"),
        "candidate_shrink_frac_after_oracle_mean": float(step_df["candidate_shrink_frac_after_oracle"].mean()) if len(step_df) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_case = pd.read_csv(args.baseline_case_csv)
    oracle_step = pd.read_csv(args.oracle_step_csv)

    left = baseline_summary(baseline_case)
    right = oracle_summary(oracle_step)
    delta = {}
    for key in [
        "valid_final_ranking_case_fraction",
        "final_candidate_size_mean",
        "final_candidate_fraction_mean",
        "final_top1_hit",
        "final_top5_hit",
        "final_mrr",
        "final_true_source_rank_mean",
    ]:
        if key in left and key in right:
            lv = safe_float(left[key])
            rv = safe_float(right[key])
            delta[key] = rv - lv if key != "final_true_source_rank_mean" else lv - rv

    summary = {
        "baseline": left,
        "candidate": right,
        "delta_candidate_minus_baseline": delta,
        "panel_version": "teacher_quality_sensitivity_panel_v1",
    }
    write_json(output_dir / "teacher_quality_summary.json", summary)
    pd.DataFrame(
        [
            {"source": "baseline", **left},
            {"source": "candidate", **right},
        ]
    ).to_csv(output_dir / "teacher_quality_summary.csv", index=False)


if __name__ == "__main__":
    main()
