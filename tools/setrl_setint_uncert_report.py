#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

B30 = 30.0
TIE_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bounded S/U vs anchor report for strict val_B30.")
    p.add_argument("--anchor-label", required=True)
    p.add_argument("--anchor-dir", required=True)
    p.add_argument("--teacher-full-dir", required=True)
    p.add_argument("--teacher-slate-dir", required=True)
    p.add_argument("--candidate-label", action="append", required=True)
    p.add_argument("--candidate-dir", action="append", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def read_summary(strict_dir: Path) -> Dict[str, float]:
    payload = json.loads((strict_dir / "summary.json").read_text(encoding="utf-8"))
    s = payload["summary"]
    return {
        "success_rate": float(s["success_rate"]),
        "avg_hit_round_conditional": float(s["avg_hit_round_conditional"]),
    }


def load_case_df(strict_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(strict_dir / "case_rows.csv")
    df["return_r0"] = (df["success_rate"] > 0.5).astype(float) - (df["budget_used"].astype(float) / B30)
    return df


def load_step_df(strict_dir: Path) -> pd.DataFrame:
    return pd.read_csv(strict_dir / "step_rows.csv")


def pair_outcome(case_a: pd.DataFrame, case_b: pd.DataFrame) -> Tuple[int, int, int, int]:
    cols = ["case_id", "return_r0"]
    m = case_a[cols].merge(case_b[cols], on="case_id", suffixes=("_a", "_b"))
    d = m["return_r0_a"] - m["return_r0_b"]
    wins = int((d > TIE_TOL).sum())
    losses = int((d < -TIE_TOL).sum())
    ties = int((d.abs() <= TIE_TOL).sum())
    return wins, losses, ties, wins - losses


def first_round_metrics(step_df: pd.DataFrame) -> Dict[str, float]:
    r1 = step_df[step_df["round_index"] == 1]
    return {
        "first_round_entropy": float(r1["posterior_entropy"].mean()),
        "first_round_top1_mass": float(r1["top1_mass"].mean()),
        "first_round_top1_top2_margin": float(r1["top1_top2_margin"].mean()),
    }


def final_metrics(case_df: pd.DataFrame, step_df: pd.DataFrame) -> Dict[str, float]:
    tail = step_df.sort_values(["case_id", "round_index"]).groupby("case_id", as_index=False).tail(1)
    return {
        "final_entropy": float(case_df["final_entropy"].mean()),
        "final_top1_mass": float(case_df["final_top1_mass"].mean()),
        "final_top1_top2_margin": float(tail["top1_top2_margin"].mean()),
    }


def round_delta_metrics(step_df: pd.DataFrame) -> Dict[str, float]:
    pivot = step_df.pivot_table(index="case_id", columns="round_index", values=["posterior_entropy", "top1_mass"], aggfunc="first")
    out: Dict[str, float] = {}
    for k, v in [(1, 2), (2, 3)]:
        if ("posterior_entropy", k) in pivot.columns and ("posterior_entropy", v) in pivot.columns:
            out[f"entropy_delta_r{v}_minus_r{k}"] = float((pivot[("posterior_entropy", v)] - pivot[("posterior_entropy", k)]).mean())
        if ("top1_mass", k) in pivot.columns and ("top1_mass", v) in pivot.columns:
            out[f"top1_mass_delta_r{v}_minus_r{k}"] = float((pivot[("top1_mass", v)] - pivot[("top1_mass", k)]).mean())
    return out


def action_set_agreement(step_anchor: pd.DataFrame, step_cand: pd.DataFrame) -> float:
    a = step_anchor[["case_id", "round_index", "selected_global_ids"]]
    b = step_cand[["case_id", "round_index", "selected_global_ids"]]
    m = a.merge(b, on=["case_id", "round_index"], suffixes=("_a", "_b"))
    if len(m) <= 0:
        return 0.0
    return float((m["selected_global_ids_a"] == m["selected_global_ids_b"]).mean())


def attribution(
    case_anchor: pd.DataFrame,
    case_cand: pd.DataFrame,
    step_anchor: pd.DataFrame,
    step_cand: pd.DataFrame,
) -> Dict[str, float]:
    a = case_anchor[["case_id", "success_rate", "hit_round", "return_r0"]].rename(
        columns={"success_rate": "succ_a", "hit_round": "hit_a", "return_r0": "ret_a"}
    )
    c = case_cand[["case_id", "success_rate", "hit_round", "return_r0"]].rename(
        columns={"success_rate": "succ_c", "hit_round": "hit_c", "return_r0": "ret_c"}
    )
    m = a.merge(c, on="case_id")
    m["win_case"] = (m["succ_c"] > m["succ_a"]).astype(int)
    m["loss_case"] = (m["succ_c"] < m["succ_a"]).astype(int)
    m["ret_delta"] = m["ret_c"] - m["ret_a"]

    r1 = step_anchor[step_anchor["round_index"] == 1][["case_id", "posterior_entropy", "top1_top2_margin"]].copy()
    m = m.merge(r1, on="case_id", how="left")
    ent_med = float(m["posterior_entropy"].median())
    high_entropy = m["posterior_entropy"] >= ent_med
    low_margin = m["top1_top2_margin"] <= 0.06
    hard_mask = high_entropy & low_margin
    wins_hard = int((m["ret_delta"][hard_mask] > TIE_TOL).sum())
    losses_hard = int((m["ret_delta"][hard_mask] < -TIE_TOL).sum())

    both_hit = (m["succ_a"] > 0.5) & (m["succ_c"] > 0.5) & m["hit_a"].notna() & m["hit_c"].notna()
    hit_delta = float((m.loc[both_hit, "hit_c"] - m.loc[both_hit, "hit_a"]).mean()) if bool(both_hit.any()) else 0.0
    win_rows = m[m["win_case"] > 0]
    loss_rows = m[m["loss_case"] > 0]
    return {
        "exact_action_set_agreement_vs_anchor": action_set_agreement(step_anchor=step_anchor, step_cand=step_cand),
        "win_case_first_round_entropy_mean": float(win_rows["posterior_entropy"].mean()) if len(win_rows) > 0 else 0.0,
        "loss_case_first_round_entropy_mean": float(loss_rows["posterior_entropy"].mean()) if len(loss_rows) > 0 else 0.0,
        "win_case_first_round_margin_mean": float(win_rows["top1_top2_margin"].mean()) if len(win_rows) > 0 else 0.0,
        "loss_case_first_round_margin_mean": float(loss_rows["top1_top2_margin"].mean()) if len(loss_rows) > 0 else 0.0,
        "high_entropy_threshold_anchor_round1": ent_med,
        "hard_state_case_count": int(hard_mask.sum()),
        "hard_state_wins": wins_hard,
        "hard_state_losses": losses_hard,
        "hard_state_net_flip": int(wins_hard - losses_hard),
        "avg_hit_round_delta_vs_anchor_on_both_hit": hit_delta,
    }


def main() -> None:
    args = parse_args()
    if len(args.candidate_label) != len(args.candidate_dir):
        raise SystemExit("--candidate-label and --candidate-dir must have same count.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_dir = Path(args.anchor_dir)
    teacher_full_dir = Path(args.teacher_full_dir)
    teacher_slate_dir = Path(args.teacher_slate_dir)
    anchor_label = str(args.anchor_label)

    anchor_summary = read_summary(anchor_dir)
    teacher_full_summary = read_summary(teacher_full_dir)
    teacher_slate_summary = read_summary(teacher_slate_dir)
    anchor_case = load_case_df(anchor_dir)
    teacher_full_case = load_case_df(teacher_full_dir)
    teacher_slate_case = load_case_df(teacher_slate_dir)
    anchor_step = load_step_df(anchor_dir)

    rows_main: List[Dict[str, float]] = []
    rows_dyn: List[Dict[str, float]] = []
    rows_attr: List[Dict[str, float]] = []

    for label, d in zip(args.candidate_label, args.candidate_dir):
        strict_dir = Path(d)
        s = read_summary(strict_dir)
        case = load_case_df(strict_dir)
        step = load_step_df(strict_dir)
        w_tf, l_tf, t_tf, nf_tf = pair_outcome(case, teacher_full_case)
        w_ts, l_ts, t_ts, nf_ts = pair_outcome(case, teacher_slate_case)
        w_a, l_a, t_a, nf_a = pair_outcome(case, anchor_case)

        rows_main.append(
            {
                "label": label,
                "strict_dir": str(strict_dir),
                "success_rate": s["success_rate"],
                "delta_vs_teacher_full": s["success_rate"] - teacher_full_summary["success_rate"],
                "delta_vs_teacher_slate": s["success_rate"] - teacher_slate_summary["success_rate"],
                "delta_vs_anchor": s["success_rate"] - anchor_summary["success_rate"],
                "avg_return_r0": float(case["return_r0"].mean()),
                "avg_hit_round_conditional": s["avg_hit_round_conditional"],
                "wins_vs_teacher_full": w_tf,
                "losses_vs_teacher_full": l_tf,
                "ties_vs_teacher_full": t_tf,
                "net_flip_vs_teacher_full": nf_tf,
                "wins_vs_teacher_slate": w_ts,
                "losses_vs_teacher_slate": l_ts,
                "ties_vs_teacher_slate": t_ts,
                "net_flip_vs_teacher_slate": nf_ts,
                "wins_vs_anchor": w_a,
                "losses_vs_anchor": l_a,
                "ties_vs_anchor": t_a,
                "net_flip_vs_anchor": nf_a,
            }
        )

        dyn = {"label": label}
        dyn.update(first_round_metrics(step))
        dyn.update(final_metrics(case, step))
        dyn.update(round_delta_metrics(step))
        rows_dyn.append(dyn)

        attr = {"label": label}
        attr.update(attribution(case_anchor=anchor_case, case_cand=case, step_anchor=anchor_step, step_cand=step))
        rows_attr.append(attr)

    main_df = pd.DataFrame(rows_main).sort_values("delta_vs_anchor", ascending=False)
    dyn_df = pd.DataFrame(rows_dyn)
    attr_df = pd.DataFrame(rows_attr)

    stage3_trigger = False
    best_label = str(main_df.iloc[0]["label"]) if len(main_df) > 0 else ""
    if len(main_df) > 0:
        best = main_df.iloc[0]
        delta_sr = float(best["delta_vs_anchor"])
        best_attr = attr_df[attr_df["label"] == best_label].iloc[0]
        delta_ret = float(best["avg_return_r0"]) - float(anchor_case["return_r0"].mean())
        delta_hit = float(best["avg_hit_round_conditional"]) - float(anchor_summary["avg_hit_round_conditional"])
        net_flip = int(best["net_flip_vs_anchor"])
        if delta_sr > 0.0:
            stage3_trigger = True
        elif abs(delta_sr) <= TIE_TOL and delta_ret > 0.0 and delta_hit < 0.0 and net_flip >= 0:
            stage3_trigger = True
        decision = {
            "best_label_seed45": best_label,
            "best_delta_sr_vs_anchor": delta_sr,
            "best_delta_return_vs_anchor": delta_ret,
            "best_delta_hit_round_vs_anchor": delta_hit,
            "best_net_flip_vs_anchor": net_flip,
            "best_exact_action_set_agreement_vs_anchor": float(best_attr["exact_action_set_agreement_vs_anchor"]),
            "stage3_seed46_triggered": bool(stage3_trigger),
            "anchor_label": anchor_label,
            "anchor_sr": float(anchor_summary["success_rate"]),
            "anchor_avg_return_r0": float(anchor_case["return_r0"].mean()),
            "anchor_avg_hit_round_conditional": float(anchor_summary["avg_hit_round_conditional"]),
        }
    else:
        decision = {"stage3_seed46_triggered": False}

    main_df.to_csv(out_dir / "main_metrics_seed45.csv", index=False)
    dyn_df.to_csv(out_dir / "dynamic_metrics_seed45.csv", index=False)
    attr_df.to_csv(out_dir / "attribution_metrics_seed45.csv", index=False)
    (out_dir / "stage3_decision_seed45.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
