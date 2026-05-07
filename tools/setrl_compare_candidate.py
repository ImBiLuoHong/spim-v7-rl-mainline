#!/usr/bin/env python3
"""Compare one strict-val candidate RL run against fixed teacher/baseline references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

BUDGET_B30 = 30.0
TIE_TOL = 1e-12


def read_summary(path: Path) -> Tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    s = payload["summary"]
    return float(s["success_rate"]), float(s["avg_hit_round_conditional"])


def load_case_return_map(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row["case_id"]).strip()
            success = float(row["success_rate"]) > 0.5
            budget_used = float(row["budget_used"])
            out[cid] = (1.0 - budget_used / BUDGET_B30) if success else (-budget_used / BUDGET_B30)
    return out


def pair_outcome(a: Dict[str, float], b: Dict[str, float]) -> Tuple[int, int, int, int]:
    ids = set(a.keys())
    if ids != set(b.keys()):
        raise RuntimeError("case_id mismatch between compared maps")
    w = l = t = 0
    for cid in ids:
        d = a[cid] - b[cid]
        if d > TIE_TOL:
            w += 1
        elif d < -TIE_TOL:
            l += 1
        else:
            t += 1
    return w, l, t, w - l


def avg(v: Dict[str, float]) -> float:
    vals = list(v.values())
    return float(sum(vals) / max(len(vals), 1))


def detect_rl_dir(strict_dir: Path) -> Path:
    cands = sorted([p for p in strict_dir.iterdir() if p.is_dir() and p.name.startswith("rl_")])
    if len(cands) != 1:
        raise RuntimeError(f"Expected exactly one rl_* dir in {strict_dir}, found {len(cands)}")
    return cands[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--candidate-run-root", required=True)
    p.add_argument("--teacher-full-dir", required=True)
    p.add_argument("--teacher-slate-dir", required=True)
    p.add_argument("--baseline-rl-dir", default="")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    cand_root = Path(args.candidate_run_root)
    cand_strict = cand_root / "strict_eval_val_B30"
    cand_rl = detect_rl_dir(cand_strict)

    tf = Path(args.teacher_full_dir)
    ts = Path(args.teacher_slate_dir)
    br = Path(args.baseline_rl_dir) if str(args.baseline_rl_dir).strip() else None

    sr_rl, hit_rl = read_summary(cand_rl / "summary.json")
    sr_tf, hit_tf = read_summary(tf / "summary.json")
    sr_ts, hit_ts = read_summary(ts / "summary.json")

    ret_rl = load_case_return_map(cand_rl / "case_rows.csv")
    ret_tf = load_case_return_map(tf / "case_rows.csv")
    ret_ts = load_case_return_map(ts / "case_rows.csv")

    w_tf, l_tf, t_tf, nf_tf = pair_outcome(ret_rl, ret_tf)
    w_ts, l_ts, t_ts, nf_ts = pair_outcome(ret_rl, ret_ts)

    out = {
        "label": str(args.label),
        "candidate_run_root": str(cand_root),
        "candidate_rl_dir": str(cand_rl),
        "teacher_full_dir": str(tf),
        "teacher_slate_dir": str(ts),
        "teacher_full_sr": sr_tf,
        "teacher_slate_sr": sr_ts,
        "candidate_sr": sr_rl,
        "delta_vs_teacher_full": sr_rl - sr_tf,
        "delta_vs_teacher_slate": sr_rl - sr_ts,
        "teacher_full_avg_hit_round_conditional": hit_tf,
        "teacher_slate_avg_hit_round_conditional": hit_ts,
        "candidate_avg_hit_round_conditional": hit_rl,
        "teacher_full_avg_return_r0": avg(ret_tf),
        "teacher_slate_avg_return_r0": avg(ret_ts),
        "candidate_avg_return_r0": avg(ret_rl),
        "wins_vs_teacher_full": w_tf,
        "losses_vs_teacher_full": l_tf,
        "ties_vs_teacher_full": t_tf,
        "net_flip_vs_teacher_full": nf_tf,
        "wins_vs_teacher_slate": w_ts,
        "losses_vs_teacher_slate": l_ts,
        "ties_vs_teacher_slate": t_ts,
        "net_flip_vs_teacher_slate": nf_ts,
    }

    if br is not None:
        sr_br, _ = read_summary(br / "summary.json")
        ret_br = load_case_return_map(br / "case_rows.csv")
        w_br, l_br, t_br, nf_br = pair_outcome(ret_rl, ret_br)
        out.update(
            {
                "baseline_3c_rl_dir": str(br),
                "baseline_3c_sr": sr_br,
                "delta_vs_baseline_3c": sr_rl - sr_br,
                "wins_vs_baseline_3c": w_br,
                "losses_vs_baseline_3c": l_br,
                "ties_vs_baseline_3c": t_br,
                "net_flip_vs_baseline_3c": nf_br,
            }
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
