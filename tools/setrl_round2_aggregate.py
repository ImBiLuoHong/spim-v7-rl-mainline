#!/usr/bin/env python3
"""Aggregate strict val_B30 metrics across multiple run roots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

BUDGET_B30 = 30.0
TIE_TOL = 1e-12


class AggregationError(RuntimeError):
    """Raised for invalid inputs or missing/invalid artifacts."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True, help="Run root path (repeatable).")
    parser.add_argument("--label", action="append", required=True, help="Label for run root (repeatable).")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise AggregationError(f"Missing required file: {path}")


def read_summary_sr(path: Path) -> Tuple[float, float]:
    require_file(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise AggregationError(f"Invalid summary.json (missing object field 'summary'): {path}")
    if "success_rate" not in summary or "avg_hit_round_conditional" not in summary:
        raise AggregationError(
            "Invalid summary.json (missing 'summary.success_rate' or "
            f"'summary.avg_hit_round_conditional'): {path}"
        )
    return float(summary["success_rate"]), float(summary["avg_hit_round_conditional"])


def load_case_rows(path: Path) -> Dict[str, float]:
    require_file(path)
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"case_id", "success_rate", "budget_used"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise AggregationError(f"Missing columns {sorted(missing)} in {path}")
        for i, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            if not case_id:
                raise AggregationError(f"Empty case_id at {path}:{i}")
            if case_id in out:
                raise AggregationError(f"Duplicate case_id '{case_id}' in {path}")
            try:
                success = float(row["success_rate"])
                budget_used = float(row["budget_used"])
            except Exception as exc:
                raise AggregationError(f"Invalid numeric value at {path}:{i}: {exc}") from exc
            if budget_used < 0.0:
                raise AggregationError(f"budget_used < 0 at {path}:{i}")
            is_success = success > 0.5
            r0_return = (1.0 - budget_used / BUDGET_B30) if is_success else (-budget_used / BUDGET_B30)
            out[case_id] = r0_return
    if not out:
        raise AggregationError(f"No rows found in {path}")
    return out


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values))


def compute_pair_outcome(target: Dict[str, float], baseline: Dict[str, float], name: str) -> Tuple[int, int, int, int]:
    target_ids = set(target)
    baseline_ids = set(baseline)
    if target_ids != baseline_ids:
        missing_in_target = sorted(baseline_ids - target_ids)
        missing_in_base = sorted(target_ids - baseline_ids)
        raise AggregationError(
            f"Case-id mismatch for {name}: "
            f"missing_in_target={missing_in_target[:5]} "
            f"missing_in_baseline={missing_in_base[:5]}"
        )

    wins = 0
    losses = 0
    ties = 0
    for case_id in target_ids:
        diff = target[case_id] - baseline[case_id]
        if diff > TIE_TOL:
            wins += 1
        elif diff < -TIE_TOL:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, wins - losses


def detect_rl_dir(strict_root: Path) -> str:
    candidates = sorted(p.name for p in strict_root.iterdir() if p.is_dir() and p.name.startswith("rl_"))
    if not candidates:
        raise AggregationError(f"No rl_* subdirectory found under {strict_root}")
    if len(candidates) > 1:
        raise AggregationError(f"Multiple rl_* subdirectories found under {strict_root}: {candidates}")
    return candidates[0]


def build_run_metrics(label: str, run_root: Path) -> Dict[str, object]:
    strict_root = run_root / "strict_eval_val_B30"
    if not strict_root.is_dir():
        raise AggregationError(f"Missing directory: {strict_root}")

    rl_dir = detect_rl_dir(strict_root)
    dirs = {
        "teacher_full": strict_root / "teacher_full",
        "teacher_slate": strict_root / "teacher_slate",
        "rl": strict_root / rl_dir,
    }

    sr_tf, hit_tf = read_summary_sr(dirs["teacher_full"] / "summary.json")
    sr_ts, hit_ts = read_summary_sr(dirs["teacher_slate"] / "summary.json")
    sr_rl, hit_rl = read_summary_sr(dirs["rl"] / "summary.json")

    ret_tf_map = load_case_rows(dirs["teacher_full"] / "case_rows.csv")
    ret_ts_map = load_case_rows(dirs["teacher_slate"] / "case_rows.csv")
    ret_rl_map = load_case_rows(dirs["rl"] / "case_rows.csv")

    avg_ret_tf = mean(list(ret_tf_map.values()))
    avg_ret_ts = mean(list(ret_ts_map.values()))
    avg_ret_rl = mean(list(ret_rl_map.values()))

    w_tf, l_tf, t_tf, nf_tf = compute_pair_outcome(ret_rl_map, ret_tf_map, "rl_vs_teacher_full")
    w_ts, l_ts, t_ts, nf_ts = compute_pair_outcome(ret_rl_map, ret_ts_map, "rl_vs_teacher_slate")

    return {
        "label": label,
        "run_root": str(run_root),
        "strict_eval_dir": str(strict_root),
        "rl_subdir": rl_dir,
        "teacher_full_sr": sr_tf,
        "teacher_slate_sr": sr_ts,
        "rl_sr": sr_rl,
        "delta_vs_teacher_full": sr_rl - sr_tf,
        "delta_vs_teacher_slate": sr_rl - sr_ts,
        "teacher_full_avg_hit_round_conditional": hit_tf,
        "teacher_slate_avg_hit_round_conditional": hit_ts,
        "rl_avg_hit_round_conditional": hit_rl,
        "teacher_full_avg_return_r0": avg_ret_tf,
        "teacher_slate_avg_return_r0": avg_ret_ts,
        "rl_avg_return_r0": avg_ret_rl,
        "wins_vs_teacher_full": w_tf,
        "losses_vs_teacher_full": l_tf,
        "ties_vs_teacher_full": t_tf,
        "net_flip_vs_teacher_full": nf_tf,
        "wins_vs_teacher_slate": w_ts,
        "losses_vs_teacher_slate": l_ts,
        "ties_vs_teacher_slate": t_ts,
        "net_flip_vs_teacher_slate": nf_ts,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise AggregationError("No rows to write")
    fieldnames = [
        "label",
        "run_root",
        "strict_eval_dir",
        "rl_subdir",
        "teacher_full_sr",
        "teacher_slate_sr",
        "rl_sr",
        "delta_vs_teacher_full",
        "delta_vs_teacher_slate",
        "teacher_full_avg_hit_round_conditional",
        "teacher_slate_avg_hit_round_conditional",
        "rl_avg_hit_round_conditional",
        "teacher_full_avg_return_r0",
        "teacher_slate_avg_return_r0",
        "rl_avg_return_r0",
        "wins_vs_teacher_full",
        "losses_vs_teacher_full",
        "ties_vs_teacher_full",
        "net_flip_vs_teacher_full",
        "wins_vs_teacher_slate",
        "losses_vs_teacher_slate",
        "ties_vs_teacher_slate",
        "net_flip_vs_teacher_slate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if len(args.run_root) != len(args.label):
        raise AggregationError(
            "--run-root and --label must have the same number of values "
            f"(got {len(args.run_root)} and {len(args.label)})"
        )

    seen_labels = set()
    for label in args.label:
        if label in seen_labels:
            raise AggregationError(f"Duplicate label: {label}")
        seen_labels.add(label)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run: List[Dict[str, object]] = []
    for label, run_root_str in zip(args.label, args.run_root):
        run_root = Path(run_root_str)
        if not run_root.is_dir():
            raise AggregationError(f"Run root is not a directory: {run_root}")
        per_run.append(build_run_metrics(label=label, run_root=run_root))

    rl_srs = [float(row["rl_sr"]) for row in per_run]
    aggregate = {
        "run_count": len(per_run),
        "labels": [row["label"] for row in per_run],
        "rl_sr": {
            "mean": mean(rl_srs),
            "median": float(statistics.median(rl_srs)),
            "std": float(statistics.pstdev(rl_srs)) if len(rl_srs) > 1 else 0.0,
        },
    }

    write_csv(out_dir / "b30_compare_table.csv", per_run)
    with (out_dir / "per_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2, sort_keys=True)
    with (out_dir / "aggregate_stats.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    print(f"Wrote {out_dir / 'b30_compare_table.csv'}")
    print(f"Wrote {out_dir / 'per_run_summary.json'}")
    print(f"Wrote {out_dir / 'aggregate_stats.json'}")


if __name__ == "__main__":
    try:
        main()
    except AggregationError as exc:
        raise SystemExit(f"ERROR: {exc}")
