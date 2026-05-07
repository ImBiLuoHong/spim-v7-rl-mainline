from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def _read_summary(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["summary"] if "summary" in obj else obj


def _parse_action_list(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(sorted(int(v) for v in json.loads(value)))
    return tuple()


def _load_case_step(dir_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(dir_path / "case_rows.csv"), pd.read_csv(dir_path / "step_rows.csv")


def _avg_return_r0(case_df: pd.DataFrame) -> float:
    return float((case_df["success_rate"].astype(float) - case_df["budget_used"].astype(float) / 30.0).mean())


def _wins_losses(candidate_case: pd.DataFrame, anchor_case: pd.DataFrame) -> Dict[str, Any]:
    merged = candidate_case[["case_id", "success_rate", "hit_round", "budget_used"]].merge(
        anchor_case[["case_id", "success_rate", "hit_round", "budget_used"]],
        on="case_id",
        suffixes=("_cand", "_anchor"),
    )
    wins = merged["success_rate_cand"] > merged["success_rate_anchor"]
    losses = merged["success_rate_cand"] < merged["success_rate_anchor"]
    ties = ~(wins | losses)
    both_hit = (merged["success_rate_cand"] > 0.5) & (merged["success_rate_anchor"] > 0.5)
    return {
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "ties": int(ties.sum()),
        "net_flip": int(wins.sum() - losses.sum()),
        "both_hit_mean_hit_round_delta": (
            float((merged.loc[both_hit, "hit_round_cand"] - merged.loc[both_hit, "hit_round_anchor"]).mean())
            if bool(both_hit.any())
            else None
        ),
        "merged": merged,
        "wins_mask": wins,
        "losses_mask": losses,
    }


def _round_prefix_metrics(candidate_step: pd.DataFrame, anchor_step: pd.DataFrame, max_round: int) -> Dict[str, Any]:
    cand = candidate_step[candidate_step["round_index"] <= max_round].copy()
    base = anchor_step[anchor_step["round_index"] <= max_round].copy()
    cand["action_set"] = cand["selected_global_ids"].apply(_parse_action_list)
    base["action_set"] = base["selected_global_ids"].apply(_parse_action_list)
    merged = cand[["case_id", "round_index", "action_set"]].merge(
        base[["case_id", "round_index", "action_set"]],
        on=["case_id", "round_index"],
        suffixes=("_cand", "_anchor"),
    )
    if merged.empty:
        return {
            "agreement_rate": None,
            "different_case_rate": None,
            "different_step_rate": None,
        }
    merged["same_action_set"] = merged["action_set_cand"] == merged["action_set_anchor"]
    case_change = merged.groupby("case_id")["same_action_set"].all()
    return {
        "agreement_rate": float(merged["same_action_set"].mean()),
        "different_case_rate": float((~case_change).mean()),
        "different_step_rate": float((~merged["same_action_set"]).mean()),
    }


def _early_hit_metrics(candidate_case: pd.DataFrame, anchor_case: pd.DataFrame, max_round: int) -> Dict[str, Any]:
    merged = candidate_case[["case_id", "success_rate", "hit_round"]].merge(
        anchor_case[["case_id", "success_rate", "hit_round"]],
        on="case_id",
        suffixes=("_cand", "_anchor"),
    )
    cand_early = (merged["success_rate_cand"] > 0.5) & (merged["hit_round_cand"] <= max_round)
    anchor_early = (merged["success_rate_anchor"] > 0.5) & (merged["hit_round_anchor"] <= max_round)
    return {
        "candidate_early_hit_rate": float(cand_early.mean()),
        "anchor_early_hit_rate": float(anchor_early.mean()),
        "delta_early_hit_rate": float(cand_early.mean() - anchor_early.mean()),
    }


def _early_observation_proxy(candidate_step: pd.DataFrame, anchor_step: pd.DataFrame, max_round: int) -> Dict[str, Any]:
    cand = candidate_step[candidate_step["round_index"] <= max_round].copy()
    base = anchor_step[anchor_step["round_index"] <= max_round].copy()
    cand = cand.groupby("case_id", as_index=False)[["positive_count", "negative_count"]].max()
    base = base.groupby("case_id", as_index=False)[["positive_count", "negative_count"]].max()
    merged = cand.merge(base, on="case_id", suffixes=("_cand", "_anchor"))
    if merged.empty:
        return {
            "candidate_positive_mean": None,
            "anchor_positive_mean": None,
            "delta_positive_mean": None,
            "candidate_negative_mean": None,
            "anchor_negative_mean": None,
            "delta_negative_mean": None,
        }
    return {
        "candidate_positive_mean": float(merged["positive_count_cand"].mean()),
        "anchor_positive_mean": float(merged["positive_count_anchor"].mean()),
        "delta_positive_mean": float((merged["positive_count_cand"] - merged["positive_count_anchor"]).mean()),
        "candidate_negative_mean": float(merged["negative_count_cand"].mean()),
        "anchor_negative_mean": float(merged["negative_count_anchor"].mean()),
        "delta_negative_mean": float((merged["negative_count_cand"] - merged["negative_count_anchor"]).mean()),
    }


def _first_round_state_compare(candidate_step: pd.DataFrame, anchor_step: pd.DataFrame, case_ids: Iterable[str]) -> Dict[str, Any]:
    wanted = set(str(v) for v in case_ids)
    cand = candidate_step[candidate_step["round_index"] == 1][["case_id", "posterior_entropy", "top1_top2_margin"]].copy()
    base = anchor_step[anchor_step["round_index"] == 1][["case_id", "posterior_entropy", "top1_top2_margin"]].copy()
    merged = cand.merge(base, on="case_id", suffixes=("_cand", "_anchor"))
    subset = merged[merged["case_id"].isin(wanted)]
    if subset.empty:
        return {
            "case_count": 0,
            "anchor_entropy_mean": None,
            "anchor_margin_mean": None,
            "candidate_entropy_mean": None,
            "candidate_margin_mean": None,
        }
    return {
        "case_count": int(len(subset)),
        "anchor_entropy_mean": float(subset["posterior_entropy_anchor"].mean()),
        "anchor_margin_mean": float(subset["top1_top2_margin_anchor"].mean()),
        "candidate_entropy_mean": float(subset["posterior_entropy_cand"].mean()),
        "candidate_margin_mean": float(subset["top1_top2_margin_cand"].mean()),
    }


def _high_entropy_low_margin_gain(candidate_case: pd.DataFrame, anchor_case: pd.DataFrame, anchor_step: pd.DataFrame) -> Dict[str, Any]:
    merged = candidate_case[["case_id", "success_rate"]].merge(
        anchor_case[["case_id", "success_rate"]],
        on="case_id",
        suffixes=("_cand", "_anchor"),
    )
    step1 = anchor_step[anchor_step["round_index"] == 1][["case_id", "posterior_entropy", "top1_top2_margin"]].copy()
    joined = merged.merge(step1, on="case_id", how="inner")
    if joined.empty:
        return {"subset_case_count": 0, "candidate_sr": None, "anchor_sr": None, "delta_sr": None}
    entropy_thr = float(joined["posterior_entropy"].median())
    margin_thr = float(joined["top1_top2_margin"].median())
    subset = joined[(joined["posterior_entropy"] >= entropy_thr) & (joined["top1_top2_margin"] <= margin_thr)]
    if subset.empty:
        return {"subset_case_count": 0, "candidate_sr": None, "anchor_sr": None, "delta_sr": None}
    cand_sr = float(subset["success_rate_cand"].mean())
    base_sr = float(subset["success_rate_anchor"].mean())
    return {
        "subset_case_count": int(len(subset)),
        "entropy_threshold_median": entropy_thr,
        "margin_threshold_median": margin_thr,
        "candidate_sr": cand_sr,
        "anchor_sr": base_sr,
        "delta_sr": float(cand_sr - base_sr),
    }


def analyze_candidate(
    *,
    label: str,
    candidate_dir: Path,
    anchor_dir: Path,
    teacher_full_dir: Path,
    teacher_slate_dir: Path,
) -> Dict[str, Any]:
    cand_case, cand_step = _load_case_step(candidate_dir)
    anchor_case, anchor_step = _load_case_step(anchor_dir)
    teacher_full_summary = _read_summary(teacher_full_dir / "summary.json")
    teacher_slate_summary = _read_summary(teacher_slate_dir / "summary.json")
    candidate_summary = _read_summary(candidate_dir / "summary.json")
    win_stats = _wins_losses(cand_case, anchor_case)
    win_case_ids = win_stats["merged"].loc[win_stats["wins_mask"], "case_id"].tolist()
    loss_case_ids = win_stats["merged"].loc[win_stats["losses_mask"], "case_id"].tolist()
    return {
        "label": label,
        "candidate_dir": str(candidate_dir),
        "success_rate": float(candidate_summary["success_rate"]),
        "delta_vs_teacher_full": float(candidate_summary["success_rate"] - float(teacher_full_summary["success_rate"])),
        "delta_vs_teacher_slate": float(candidate_summary["success_rate"] - float(teacher_slate_summary["success_rate"])),
        "delta_vs_anchor": float(candidate_summary["success_rate"] - float(_read_summary(anchor_dir / "summary.json")["success_rate"])),
        "avg_return_r0": _avg_return_r0(cand_case),
        "avg_hit_round_conditional": float(candidate_summary["avg_hit_round_conditional"]),
        "wins": int(win_stats["wins"]),
        "losses": int(win_stats["losses"]),
        "ties": int(win_stats["ties"]),
        "net_flip": int(win_stats["net_flip"]),
        "exact_action_set_agreement_global": _round_prefix_metrics(
            candidate_step=cand_step,
            anchor_step=anchor_step,
            max_round=int(cand_step["round_index"].max()),
        )["agreement_rate"],
        "round1_action_set": _round_prefix_metrics(cand_step, anchor_step, 1),
        "round1_2_action_set": _round_prefix_metrics(cand_step, anchor_step, 2),
        "round1_3_action_set": _round_prefix_metrics(cand_step, anchor_step, 3),
        "round1_2_early_hit": _early_hit_metrics(cand_case, anchor_case, 2),
        "round1_3_early_hit": _early_hit_metrics(cand_case, anchor_case, 3),
        "round1_2_observation_proxy": _early_observation_proxy(cand_step, anchor_step, 2),
        "round1_3_observation_proxy": _early_observation_proxy(cand_step, anchor_step, 3),
        "both_hit_hit_round_delta": win_stats["both_hit_mean_hit_round_delta"],
        "wins_first_round_state": _first_round_state_compare(cand_step, anchor_step, win_case_ids),
        "losses_first_round_state": _first_round_state_compare(cand_step, anchor_step, loss_case_ids),
        "high_entropy_low_margin_gain": _high_entropy_low_margin_gain(cand_case, anchor_case, anchor_step),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze bounded v3 early-stage specialist strict-eval outputs.")
    p.add_argument("--anchor-dir", type=Path, required=True)
    p.add_argument("--teacher-full-dir", type=Path, required=True)
    p.add_argument("--teacher-slate-dir", type=Path, required=True)
    p.add_argument("--candidate", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out: Dict[str, Any] = {"candidates": []}
    for label, dir_str in args.candidate:
        out["candidates"].append(
            analyze_candidate(
                label=label,
                candidate_dir=Path(dir_str),
                anchor_dir=args.anchor_dir,
                teacher_full_dir=args.teacher_full_dir,
                teacher_slate_dir=args.teacher_slate_dir,
            )
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
