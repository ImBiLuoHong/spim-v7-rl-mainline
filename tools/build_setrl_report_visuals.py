#!/usr/bin/env python3
"""Build report-ready visualizations for set-level RL strict val_B30 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


B30_BUDGET = 30
TIE_TOL = 1e-12


@dataclass
class PolicyEval:
    run_name: str
    policy_key: str
    summary_path: Path
    case_rows_path: Path
    success_rate: float
    avg_hit_round_conditional: float


@dataclass
class RunRecord:
    run_name: str
    run_dir: Path
    teacher_full: Optional[PolicyEval]
    teacher_slate: Optional[PolicyEval]
    rl: Optional[PolicyEval]
    seed: Optional[int]
    slate: str
    train_size: Optional[int]
    variant: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/spim_set_level_rl_mainline",
        help="Root directory containing set-level RL run folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/spim_set_level_rl_mainline/report_visuals_20260415",
        help="Output directory for tables and figures.",
    )
    parser.add_argument(
        "--run-glob",
        action="append",
        default=["20260414_*", "20260415_*"],
        help="Run name glob(s) under artifacts root. Repeatable.",
    )
    return parser.parse_args()


def read_summary(path: Path) -> Tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    return float(summary["success_rate"]), float(summary["avg_hit_round_conditional"])


def load_returns_from_case_rows(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row["case_id"]).strip()
            success = float(row["success_rate"]) > 0.5
            budget_used = float(row["budget_used"])
            ret = (1.0 - budget_used / B30_BUDGET) if success else (-budget_used / B30_BUDGET)
            out[case_id] = ret
    return out


def load_success_from_case_rows(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row["case_id"]).strip()
            out[case_id] = 1 if float(row["success_rate"]) > 0.5 else 0
    return out


def load_budget_success_curve_from_case_rows(path: Path) -> List[Tuple[int, float]]:
    rows: List[Tuple[bool, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["success_rate"]) > 0.5, float(row["budget_used"])))
    total = len(rows)
    curve = []
    for b in range(1, B30_BUDGET + 1):
        hit = sum(1 for ok, used in rows if ok and used <= b + 1e-9)
        curve.append((b, hit / total if total else 0.0))
    return curve


def discover_policy(strict_dir: Path, policy_key: str) -> Optional[PolicyEval]:
    matches = []
    for sub in strict_dir.iterdir():
        if not sub.is_dir():
            continue
        if policy_key == "rl":
            if sub.name.startswith("rl_"):
                matches.append(sub)
        elif sub.name == policy_key:
            matches.append(sub)
    if not matches:
        return None
    if policy_key == "rl" and len(matches) > 1:
        raise RuntimeError(f"Multiple rl_* folders under {strict_dir}: {[m.name for m in matches]}")
    target = sorted(matches)[0]
    summary_path = target / "summary.json"
    case_rows_path = target / "case_rows.csv"
    if not summary_path.is_file() or not case_rows_path.is_file():
        return None
    sr, hit = read_summary(summary_path)
    return PolicyEval(
        run_name=strict_dir.parent.name,
        policy_key=target.name,
        summary_path=summary_path,
        case_rows_path=case_rows_path,
        success_rate=sr,
        avg_hit_round_conditional=hit,
    )


def parse_run_meta(run_name: str) -> Tuple[Optional[int], str, Optional[int], str]:
    seed_match = re.search(r"seed(\d+)", run_name)
    seed = int(seed_match.group(1)) if seed_match else None
    train_match = re.search(r"train(\d+)", run_name)
    train_size = int(train_match.group(1)) if train_match else None
    if "slateA" in run_name:
        slate = "slateA"
    elif "slateB" in run_name:
        slate = "slateB"
    else:
        slate = "legacy"
    variant = run_name.split("setrl_", 1)[-1] if "setrl_" in run_name else run_name
    return seed, slate, train_size, variant


def load_runs(artifacts_root: Path, run_globs: List[str]) -> List[RunRecord]:
    run_dirs: Dict[str, Path] = {}
    for g in run_globs:
        for run_dir in artifacts_root.glob(g):
            if run_dir.is_dir():
                run_dirs[run_dir.name] = run_dir
    out: List[RunRecord] = []
    for run_name in sorted(run_dirs):
        run_dir = run_dirs[run_name]
        strict_dir = run_dir / "strict_eval_val_B30"
        if not strict_dir.is_dir():
            continue
        seed, slate, train_size, variant = parse_run_meta(run_name)
        out.append(
            RunRecord(
                run_name=run_name,
                run_dir=run_dir,
                teacher_full=discover_policy(strict_dir, "teacher_full"),
                teacher_slate=discover_policy(strict_dir, "teacher_slate"),
                rl=discover_policy(strict_dir, "rl"),
                seed=seed,
                slate=slate,
                train_size=train_size,
                variant=variant,
            )
        )
    return out


def compute_wlt(target: Dict[str, float], base: Dict[str, float]) -> Tuple[int, int, int, int]:
    ids_t = set(target)
    ids_b = set(base)
    if ids_t != ids_b:
        raise RuntimeError("Case id mismatch between target and baseline when computing win/loss/tie.")
    wins = losses = ties = 0
    for k in ids_t:
        d = target[k] - base[k]
        if d > TIE_TOL:
            wins += 1
        elif d < -TIE_TOL:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, wins - losses


def ensure_teacher_full_fallback(run_records: List[RunRecord]) -> None:
    canonical: Optional[PolicyEval] = None
    for r in run_records:
        if r.teacher_full is not None:
            canonical = r.teacher_full
            break
    if canonical is None:
        return
    for r in run_records:
        if r.teacher_full is None:
            r.teacher_full = canonical


def build_summary_rows(run_records: List[RunRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in run_records:
        if r.rl is None:
            continue
        tf_sr = r.teacher_full.success_rate if r.teacher_full else math.nan
        ts_sr = r.teacher_slate.success_rate if r.teacher_slate else math.nan
        rl_sr = r.rl.success_rate

        tf_ret_mean = ts_ret_mean = rl_ret_mean = math.nan
        w_tf = l_tf = t_tf = nf_tf = None
        w_ts = l_ts = t_ts = nf_ts = None
        sw_tf = sl_tf = st_tf = snf_tf = None
        sw_ts = sl_ts = st_ts = snf_ts = None

        rl_ret_map = load_returns_from_case_rows(r.rl.case_rows_path)
        rl_ret_mean = mean(list(rl_ret_map.values()))
        rl_success_map = load_success_from_case_rows(r.rl.case_rows_path)

        if r.teacher_full:
            tf_ret_map = load_returns_from_case_rows(r.teacher_full.case_rows_path)
            tf_ret_mean = mean(list(tf_ret_map.values()))
            w_tf, l_tf, t_tf, nf_tf = compute_wlt(rl_ret_map, tf_ret_map)
            tf_success_map = load_success_from_case_rows(r.teacher_full.case_rows_path)
            sw_tf, sl_tf, st_tf, snf_tf = compute_wlt(rl_success_map, tf_success_map)
        if r.teacher_slate:
            ts_ret_map = load_returns_from_case_rows(r.teacher_slate.case_rows_path)
            ts_ret_mean = mean(list(ts_ret_map.values()))
            w_ts, l_ts, t_ts, nf_ts = compute_wlt(rl_ret_map, ts_ret_map)
            ts_success_map = load_success_from_case_rows(r.teacher_slate.case_rows_path)
            sw_ts, sl_ts, st_ts, snf_ts = compute_wlt(rl_success_map, ts_success_map)

        rows.append(
            {
                "run_name": r.run_name,
                "seed": r.seed,
                "slate": r.slate,
                "train_size": r.train_size,
                "variant": r.variant,
                "teacher_full_sr": tf_sr,
                "teacher_slate_sr": ts_sr,
                "rl_sr": rl_sr,
                "delta_vs_teacher_full": rl_sr - tf_sr if not math.isnan(tf_sr) else math.nan,
                "delta_vs_teacher_slate": rl_sr - ts_sr if not math.isnan(ts_sr) else math.nan,
                "teacher_full_avg_hit_round": r.teacher_full.avg_hit_round_conditional if r.teacher_full else math.nan,
                "teacher_slate_avg_hit_round": r.teacher_slate.avg_hit_round_conditional if r.teacher_slate else math.nan,
                "rl_avg_hit_round": r.rl.avg_hit_round_conditional,
                "teacher_full_avg_return_r0": tf_ret_mean,
                "teacher_slate_avg_return_r0": ts_ret_mean,
                "rl_avg_return_r0": rl_ret_mean,
                "wins_vs_teacher_full": w_tf,
                "losses_vs_teacher_full": l_tf,
                "ties_vs_teacher_full": t_tf,
                "net_flip_vs_teacher_full": nf_tf,
                "success_wins_vs_teacher_full": sw_tf,
                "success_losses_vs_teacher_full": sl_tf,
                "success_ties_vs_teacher_full": st_tf,
                "success_net_flip_vs_teacher_full": snf_tf,
                "wins_vs_teacher_slate": w_ts,
                "losses_vs_teacher_slate": l_ts,
                "ties_vs_teacher_slate": t_ts,
                "net_flip_vs_teacher_slate": nf_ts,
                "success_wins_vs_teacher_slate": sw_ts,
                "success_losses_vs_teacher_slate": sl_ts,
                "success_ties_vs_teacher_slate": st_ts,
                "success_net_flip_vs_teacher_slate": snf_ts,
            }
        )
    return rows


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No summary rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sr_over_runs(rows: List[Dict[str, object]], out_path: Path) -> None:
    x = list(range(len(rows)))
    labels = [str(r["run_name"]).replace("20260415_", "").replace("20260414_", "") for r in rows]
    tf = [float(r["teacher_full_sr"]) for r in rows]
    ts = [float(r["teacher_slate_sr"]) for r in rows]
    rl = [float(r["rl_sr"]) for r in rows]

    plt.figure(figsize=(16, 6))
    plt.plot(x, tf, marker="o", linewidth=2, label="teacher_full")
    plt.plot(x, ts, marker="o", linewidth=2, label="teacher_slate")
    plt.plot(x, rl, marker="o", linewidth=2.5, label="set-level RL")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0.86, 0.93)
    plt.ylabel("Strict val_B30 Success Rate")
    plt.title("Set-Level RL Mainline: Strict val_B30 SR Across Runs")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_delta_vs_teacher_full(rows: List[Dict[str, object]], out_path: Path) -> None:
    xs = []
    ys = []
    cs = []
    for i, r in enumerate(rows):
        delta = float(r["delta_vs_teacher_full"])
        if math.isnan(delta):
            continue
        xs.append(i)
        ys.append(delta * 100.0)
        cs.append("#2ca02c" if delta > 0 else "#d62728")

    labels = [str(r["run_name"]).replace("20260415_", "").replace("20260414_", "") for r in rows]
    plt.figure(figsize=(16, 6))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.bar(xs, ys, color=cs, alpha=0.85)
    plt.xticks(list(range(len(rows))), labels, rotation=45, ha="right")
    plt.ylabel("Delta SR vs teacher_full (percentage points)")
    plt.title("Strict val_B30 Improvement of Set-Level RL Over teacher_full")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_netflip_vs_teacher_full(rows: List[Dict[str, object]], out_path: Path) -> None:
    xs = []
    ys = []
    colors = []
    for i, r in enumerate(rows):
        val = r["success_net_flip_vs_teacher_full"]
        if val is None:
            continue
        v = int(val)
        xs.append(i)
        ys.append(v)
        colors.append("#1f77b4" if v >= 0 else "#d62728")

    labels = [str(r["run_name"]).replace("20260415_", "").replace("20260414_", "") for r in rows]
    plt.figure(figsize=(16, 6))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.bar(xs, ys, color=colors, alpha=0.85)
    plt.xticks(list(range(len(rows))), labels, rotation=45, ha="right")
    plt.ylabel("Success net flip vs teacher_full (wins - losses)")
    plt.title("Case-Level Success Net Flip Under Strict val_B30")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_budget_curves(run_records: List[RunRecord], out_path: Path) -> None:
    targets = [r for r in run_records if ("killcorroborate" in r.run_name and r.rl is not None)]
    if not targets:
        return
    targets = sorted(targets, key=lambda r: r.run_name)
    plt.figure(figsize=(10, 6))

    teacher_ref = targets[0].teacher_full
    if teacher_ref:
        curve = load_budget_success_curve_from_case_rows(teacher_ref.case_rows_path)
        plt.plot([b for b, _ in curve], [v for _, v in curve], linewidth=2.3, label="teacher_full", color="#444444")

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for idx, r in enumerate(targets):
        curve = load_budget_success_curve_from_case_rows(r.rl.case_rows_path)
        label = f"RL {r.run_name.replace('20260414_', '')}"
        plt.plot([b for b, _ in curve], [v for _, v in curve], linewidth=2.0, label=label, color=palette[idx % len(palette)])

    plt.xlim(1, 30)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Sample Budget")
    plt.ylabel("Cumulative Success Rate")
    plt.title("Strict val_B30 Budget Curves (Kill/Corroborate Seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def write_claim_snapshot(rows: List[Dict[str, object]], out_path: Path) -> None:
    by_run = {str(r["run_name"]): r for r in rows}
    lines = []
    lines.append("# Set-Level RL Report Snapshot (Strict val_B30)")
    lines.append("")
    lines.append("## [proven]")
    for key in [
        "20260414_killcorroborate_seed45_rerun_v1",
        "20260414_killcorroborate_seed46_rerun_v1",
    ]:
        row = by_run.get(key)
        if not row:
            continue
        lines.append(
            "- "
            f"{key}: rl_sr={float(row['rl_sr']):.6f}, teacher_full={float(row['teacher_full_sr']):.6f}, "
            f"delta={float(row['delta_vs_teacher_full']):+.6f}, "
            f"success_flip={int(row['success_net_flip_vs_teacher_full']) if row['success_net_flip_vs_teacher_full'] is not None else 'NA'}, "
            f"success_w/l/t="
            f"{int(row['success_wins_vs_teacher_full']) if row['success_wins_vs_teacher_full'] is not None else 'NA'}/"
            f"{int(row['success_losses_vs_teacher_full']) if row['success_losses_vs_teacher_full'] is not None else 'NA'}/"
            f"{int(row['success_ties_vs_teacher_full']) if row['success_ties_vs_teacher_full'] is not None else 'NA'}"
        )
    lines.append("")
    lines.append("## [partially proven]")
    lines.append("- 2026-04-15 slate/scale extensions show mixed magnitudes; positive runs exist but stability across seeds/settings is not locked.")
    lines.append("")
    lines.append("## [not proved]")
    lines.append("- Not yet proven that current set-level RL recipe is consistently superior across broader seeds and larger train scales.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records = load_runs(artifacts_root, args.run_glob)
    if not run_records:
        raise SystemExit("No runs discovered.")
    ensure_teacher_full_fallback(run_records)

    rows = build_summary_rows(run_records)
    rows = sorted(rows, key=lambda r: str(r["run_name"]))

    write_summary_csv(out_dir / "strict_b30_summary_table.csv", rows)
    plot_sr_over_runs(rows, out_dir / "fig1_sr_over_runs.png")
    plot_delta_vs_teacher_full(rows, out_dir / "fig2_delta_vs_teacher_full_pp.png")
    plot_netflip_vs_teacher_full(rows, out_dir / "fig3_netflip_vs_teacher_full.png")
    plot_budget_curves(run_records, out_dir / "fig4_killcorroborate_budget_curves.png")
    write_claim_snapshot(rows, out_dir / "claim_snapshot.md")

    print(f"Wrote summary: {out_dir / 'strict_b30_summary_table.csv'}")
    print(f"Wrote figures under: {out_dir}")


if __name__ == "__main__":
    main()
