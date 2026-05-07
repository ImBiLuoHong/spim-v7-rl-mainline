from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    extra_train_args: Tuple[str, ...]
    extra_eval_args: Tuple[str, ...]


BASELINE_ROOT_BY_SEED = {
    45: PROJECT_ROOT / "artifacts" / "spim_set_level_rl_mainline" / "20260415_causal_exp3C_train4823_seed45_random_valueonly_v1",
    46: PROJECT_ROOT / "artifacts" / "spim_set_level_rl_mainline" / "20260415_causal_exp3C_train4823_seed46_random_valueonly_v1",
}
BASELINE_POLICY_NAME_BY_SEED = {
    45: "rl_set_seed45_random_valueonly_train4823",
    46: "rl_set_seed46_random_valueonly_train4823",
}
STRONGER_V2_ROOT_BY_SEED = {
    45: PROJECT_ROOT / "artifacts" / "spim_set_level_rl_mainline" / "20260416_critic_value_stage1_seed45_v2" / "stronger_value_head_v2",
    46: PROJECT_ROOT / "artifacts" / "spim_set_level_rl_mainline" / "20260416_critic_value_stage3_seed46_v1" / "stronger_value_head_v2",
}
STRONGER_V2_POLICY_NAME_BY_SEED = {
    45: "rl_set_seed45_stronger_value_head_v2_train4823",
    46: "rl_set_seed46_stronger_value_head_v2_train4823",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bounded critic-system strengthening run on fixed 3C mainline.")
    p.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "artifacts" / "spim_set_level_rl_mainline" / "20260416_critic_system_report_v1"))
    p.add_argument("--stage1-root", type=str, default=str(PROJECT_ROOT / "artifacts" / "spim_set_level_rl_mainline" / "20260416_critic_system_stage1_seed45_v1"))
    p.add_argument("--stage3-root", type=str, default=str(PROJECT_ROOT / "artifacts" / "spim_set_level_rl_mainline" / "20260416_critic_system_stage3_seed46_v1"))
    p.add_argument("--include-value-coeff-v2", action="store_true")
    p.add_argument("--max-top2", type=int, default=2)
    return p.parse_args()


def _run_cmd(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} (see {log_path})")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_dirs(run_root: Path, policy_name: str) -> Dict[str, Path]:
    strict_root = run_root / "strict_eval_val_B30"
    return {
        "teacher_full": strict_root / "teacher_full",
        "teacher_slate": strict_root / "teacher_slate",
        "policy": strict_root / policy_name,
    }


def _ensure_strict_eval(
    *,
    run_root: Path,
    policy_name: str,
    seed: int,
    checkpoint: Optional[Path],
    eval_args: Sequence[str],
    logs_root: Path,
) -> Dict[str, Path]:
    dirs = _strict_dirs(run_root, policy_name)
    if not (dirs["teacher_full"] / "summary.json").exists():
        cmd = [
            "python",
            "src/scripts/run_spim_policy_eval_strict.py",
            "--output-dir",
            str(dirs["teacher_full"]),
            "--seed",
            str(seed),
            "--split",
            "val",
            "--case-limit",
            "0",
            "--policy-mode",
            "teacher",
            "--policy-name",
            "teacher_full",
            "--teacher-family",
            "hsr_soft_scenario_posterior_v3",
            "--reward-family",
            "reward_r0_terminal_step",
            "--slate-size",
            "10",
            "--slate-top-posterior-k",
            "8",
            "--slate-high-disagreement-k",
            "1",
            "--slate-novelty-k",
            "1",
        ]
        _run_cmd(cmd, logs_root / f"{policy_name}_teacher_full_strict_eval.log")
    if not (dirs["teacher_slate"] / "summary.json").exists():
        cmd = [
            "python",
            "src/scripts/run_spim_policy_eval_strict.py",
            "--output-dir",
            str(dirs["teacher_slate"]),
            "--seed",
            str(seed),
            "--split",
            "val",
            "--case-limit",
            "0",
            "--policy-mode",
            "teacher_slate",
            "--policy-name",
            "teacher_slate",
            "--teacher-family",
            "hsr_soft_scenario_posterior_v3",
            "--reward-family",
            "reward_r0_terminal_step",
            "--slate-size",
            "10",
            "--slate-top-posterior-k",
            "8",
            "--slate-high-disagreement-k",
            "1",
            "--slate-novelty-k",
            "1",
        ]
        _run_cmd(cmd, logs_root / f"{policy_name}_teacher_slate_strict_eval.log")
    if checkpoint is not None and not (dirs["policy"] / "summary.json").exists():
        cmd = [
            "python",
            "src/scripts/run_spim_policy_eval_strict.py",
            "--output-dir",
            str(dirs["policy"]),
            "--seed",
            str(seed),
            "--split",
            "val",
            "--case-limit",
            "0",
            "--policy-mode",
            "rl",
            "--policy-name",
            policy_name,
            "--checkpoint",
            str(checkpoint),
            "--teacher-family",
            "hsr_soft_scenario_posterior_v3",
            "--reward-family",
            "reward_r0_terminal_step",
            "--slate-size",
            "10",
            "--slate-top-posterior-k",
            "8",
            "--slate-high-disagreement-k",
            "1",
            "--slate-novelty-k",
            "1",
            *eval_args,
        ]
        _run_cmd(cmd, logs_root / f"{policy_name}_strict_eval.log")
    return dirs


def _compare_case_wins(candidate_case_csv: Path, baseline_case_csv: Path) -> Tuple[int, int, int, int]:
    cand = pd.read_csv(candidate_case_csv)[["case_id", "success_rate"]].rename(columns={"success_rate": "sr_c"})
    base = pd.read_csv(baseline_case_csv)[["case_id", "success_rate"]].rename(columns={"success_rate": "sr_b"})
    merged = cand.merge(base, on="case_id", how="inner")
    wins = int((merged["sr_c"] > merged["sr_b"]).sum())
    losses = int((merged["sr_c"] < merged["sr_b"]).sum())
    ties = int((merged["sr_c"] == merged["sr_b"]).sum())
    return wins, losses, ties, wins - losses


def _avg_return_r0(case_csv: Path) -> float:
    df = pd.read_csv(case_csv, usecols=["success_rate", "budget_used"])
    ret = df["success_rate"].astype(float) - (df["budget_used"].astype(float) / 30.0)
    return float(ret.mean())


def _extract_metrics(seed: int, label: str, dirs: Dict[str, Path], baseline_policy_dir: Path) -> Dict[str, Any]:
    teacher = _read_json(dirs["teacher_full"] / "summary.json")
    slate = _read_json(dirs["teacher_slate"] / "summary.json")
    policy = _read_json(dirs["policy"] / "summary.json")
    base = _read_json(baseline_policy_dir / "summary.json")
    wins, losses, ties, net = _compare_case_wins(dirs["policy"] / "case_rows.csv", baseline_policy_dir / "case_rows.csv")
    policy_sr = float(policy["summary"]["success_rate"] if "summary" in policy else policy["success_rate"])
    teacher_sr = float(teacher["summary"]["success_rate"] if "summary" in teacher else teacher["success_rate"])
    slate_sr = float(slate["summary"]["success_rate"] if "summary" in slate else slate["success_rate"])
    base_sr = float(base["summary"]["success_rate"] if "summary" in base else base["success_rate"])
    return {
        "seed": int(seed),
        "candidate": str(label),
        "teacher_full_sr": teacher_sr,
        "teacher_slate_sr": slate_sr,
        "baseline_3c_sr": base_sr,
        "candidate_sr": policy_sr,
        "delta_vs_teacher_full": float(policy_sr - teacher_sr),
        "delta_vs_teacher_slate": float(policy_sr - slate_sr),
        "delta_vs_baseline_3c": float(policy_sr - base_sr),
        "wins_vs_baseline_3c": int(wins),
        "losses_vs_baseline_3c": int(losses),
        "ties_vs_baseline_3c": int(ties),
        "net_flip_vs_baseline_3c": int(net),
        "avg_hit_round_conditional": float((policy["summary"]["avg_hit_round_conditional"] if "summary" in policy else policy["avg_hit_round_conditional"])),
        "avg_return_r0": _avg_return_r0(dirs["policy"] / "case_rows.csv"),
    }


def _build_stage1_specs(include_value_coeff_v2: bool) -> List[CandidateSpec]:
    specs = [
        CandidateSpec(
            label="separate_critic_trunk_v1",
            extra_train_args=(
                "--critic-trunk-depth", "2",
                "--critic-trunk-hidden-dim", "256",
            ),
            extra_eval_args=(
                "--critic-trunk-depth", "2",
                "--critic-trunk-hidden-dim", "256",
            ),
        ),
        CandidateSpec(
            label="separate_critic_trunk_v2",
            extra_train_args=(
                "--value-mlp-depth", "3",
                "--value-head-width-mult", "2.0",
                "--critic-trunk-depth", "3",
                "--critic-trunk-hidden-dim", "256",
            ),
            extra_eval_args=(
                "--value-mlp-depth", "3",
                "--value-head-width-mult", "2.0",
                "--critic-trunk-depth", "3",
                "--critic-trunk-hidden-dim", "256",
            ),
        ),
        CandidateSpec(
            label="critic_warmup_v1",
            extra_train_args=(
                "--value-mlp-depth", "3",
                "--value-head-width-mult", "2.0",
                "--critic-warmup-epochs", "1",
            ),
            extra_eval_args=(
                "--value-mlp-depth", "3",
                "--value-head-width-mult", "2.0",
            ),
        ),
        CandidateSpec(
            label="actor_critic_update_ratio_v1",
            extra_train_args=(
                "--value-mlp-depth", "3",
                "--value-head-width-mult", "2.0",
                "--rl-critic-extra-updates", "1",
            ),
            extra_eval_args=(
                "--value-mlp-depth", "3",
                "--value-head-width-mult", "2.0",
            ),
        ),
    ]
    if include_value_coeff_v2:
        specs.append(
            CandidateSpec(
                label="value_coeff_tuned_v2",
                extra_train_args=(
                    "--value-mlp-depth", "3",
                    "--value-head-width-mult", "2.0",
                    "--rl-value-coef", "0.7",
                ),
                extra_eval_args=(
                    "--value-mlp-depth", "3",
                    "--value-head-width-mult", "2.0",
                ),
            )
        )
    return specs


def _run_train_candidate(spec: CandidateSpec, seed: int, run_root: Path, logs_root: Path) -> Path:
    checkpoint = run_root / "checkpoints" / "rl_student_final.pt"
    if not checkpoint.exists():
        cmd = [
            "python",
            "src/scripts/run_spim_teacher_imitation_rl_pilot.py",
            "--output-dir",
            str(run_root),
            "--seed",
            str(seed),
            "--runner-version-tag",
            f"critic_system_{spec.label}_seed{seed}",
            "--teacher-family",
            "hsr_soft_scenario_posterior_v3",
            "--train-full-max-cases",
            "4823",
            "--train-full-cache-version",
            f"setrl_main_train4823_full_seed{seed}_v1",
            "--reward-family",
            "reward_r0_terminal_step",
            "--rl-init-mode",
            "random_init",
            "--advantage-baseline",
            "value_only",
            "--slate-size",
            "10",
            "--slate-top-posterior-k",
            "8",
            "--slate-high-disagreement-k",
            "1",
            "--slate-novelty-k",
            "1",
            "--save-final-checkpoint",
            str(checkpoint),
            "--save-rl-epoch-checkpoints-dir",
            str(run_root / "checkpoints" / "rl_epochs"),
            *spec.extra_train_args,
        ]
        _run_cmd(cmd, logs_root / f"{spec.label}_seed{seed}_train.log")
    return checkpoint


def _run_overlap_gate() -> Dict[str, Any]:
    train_ids = [line.strip() for line in (PROJECT_ROOT / "data" / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    val_ids = [line.strip() for line in (PROJECT_ROOT / "data" / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_scenarios = {row.split(":")[0] for row in train_ids}
    val_scenarios = {row.split(":")[0] for row in val_ids}
    return {
        "train_row_count": int(len(train_ids)),
        "val_row_count": int(len(val_ids)),
        "train_group_count": int(len(train_scenarios)),
        "val_group_count": int(len(val_scenarios)),
        "group_overlap": int(len(train_scenarios.intersection(val_scenarios))),
    }


def _stage2_top2(df: pd.DataFrame, max_top2: int) -> List[str]:
    stable = pd.Series([True] * len(df))
    keep = (df["delta_vs_baseline_3c"] > -0.003) & (df["delta_vs_teacher_full"] > 0.0) & stable
    filtered = df[keep].sort_values(["delta_vs_baseline_3c", "candidate_sr"], ascending=False)
    return filtered["candidate"].head(max_top2).tolist()


def _minimal_attribution(candidate_dir: Path, baseline_dir: Path, out_json: Path) -> None:
    step_c = pd.read_csv(candidate_dir / "step_rows.csv")
    step_b = pd.read_csv(baseline_dir / "step_rows.csv")
    case_c = pd.read_csv(candidate_dir / "case_rows.csv")[["case_id", "success_rate"]].rename(columns={"success_rate": "sr_c"})
    case_b = pd.read_csv(baseline_dir / "case_rows.csv")[["case_id", "success_rate"]].rename(columns={"success_rate": "sr_b"})
    diff = case_c.merge(case_b, on="case_id", how="inner")
    diff["delta"] = diff["sr_c"] - diff["sr_b"]
    b_first = step_b[step_b["round_index"] == 1][["case_id", "posterior_entropy_norm", "top1_top2_margin"]]
    merged = diff.merge(b_first, on="case_id", how="left")

    def summarize(mask: pd.Series, name: str) -> Dict[str, Any]:
        sub = merged[mask]
        wins = int((sub["delta"] > 0).sum())
        losses = int((sub["delta"] < 0).sum())
        return {
            "bucket": name,
            "count": int(len(sub)),
            "wins": wins,
            "losses": losses,
            "net": int(wins - losses),
        }

    rows = [
        summarize(merged["posterior_entropy_norm"] >= 0.35, "high_entropy"),
        summarize(merged["top1_top2_margin"] <= 0.06, "low_margin"),
        summarize((merged["posterior_entropy_norm"] >= 0.35) & (merged["top1_top2_margin"] <= 0.06), "intersection"),
    ]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    stage1_root = Path(args.stage1_root)
    stage3_root = Path(args.stage3_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    gate = _run_overlap_gate()
    (output_root / "formal_split_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    stage1_rows: List[Dict[str, Any]] = []

    baseline_seed45_dirs = _strict_dirs(BASELINE_ROOT_BY_SEED[45], BASELINE_POLICY_NAME_BY_SEED[45])
    stronger_seed45_dirs = _strict_dirs(STRONGER_V2_ROOT_BY_SEED[45], STRONGER_V2_POLICY_NAME_BY_SEED[45])
    stronger_seed45_dirs["teacher_full"] = baseline_seed45_dirs["teacher_full"]
    stronger_seed45_dirs["teacher_slate"] = baseline_seed45_dirs["teacher_slate"]

    stage1_rows.append(
        _extract_metrics(
            seed=45,
            label="stronger_value_head_v2",
            dirs=stronger_seed45_dirs,
            baseline_policy_dir=baseline_seed45_dirs["policy"],
        )
    )

    specs = _build_stage1_specs(include_value_coeff_v2=bool(args.include_value_coeff_v2))
    for spec in specs:
        run_root = stage1_root / spec.label
        checkpoint = _run_train_candidate(spec, 45, run_root, output_root / "logs")
        policy_name = f"rl_set_seed45_{spec.label}_train4823"
        dirs = _ensure_strict_eval(
            run_root=run_root,
            policy_name=policy_name,
            seed=45,
            checkpoint=checkpoint,
            eval_args=spec.extra_eval_args,
            logs_root=output_root / "logs",
        )
        row = _extract_metrics(
            seed=45,
            label=spec.label,
            dirs=dirs,
            baseline_policy_dir=baseline_seed45_dirs["policy"],
        )
        stage1_rows.append(row)

    stage1_df = pd.DataFrame(stage1_rows).sort_values("delta_vs_baseline_3c", ascending=False)
    stage1_df.to_csv(output_root / "stage1_seed45_screen.csv", index=False)

    top2 = _stage2_top2(stage1_df, int(args.max_top2))
    (output_root / "stage2_top2_selection.json").write_text(json.dumps({"top2": top2}, indent=2), encoding="utf-8")

    stage3_rows: List[Dict[str, Any]] = []
    baseline_seed46_dirs = _strict_dirs(BASELINE_ROOT_BY_SEED[46], BASELINE_POLICY_NAME_BY_SEED[46])
    for label in top2:
        if label == "stronger_value_head_v2":
            dirs = _strict_dirs(STRONGER_V2_ROOT_BY_SEED[46], STRONGER_V2_POLICY_NAME_BY_SEED[46])
            dirs["teacher_full"] = baseline_seed46_dirs["teacher_full"]
            dirs["teacher_slate"] = baseline_seed46_dirs["teacher_slate"]
            row = _extract_metrics(
                seed=46,
                label=label,
                dirs=dirs,
                baseline_policy_dir=baseline_seed46_dirs["policy"],
            )
            stage3_rows.append(row)
            continue
        spec = [s for s in specs if s.label == label][0]
        run_root = stage3_root / spec.label
        checkpoint = _run_train_candidate(spec, 46, run_root, output_root / "logs")
        policy_name = f"rl_set_seed46_{spec.label}_train4823"
        dirs = _ensure_strict_eval(
            run_root=run_root,
            policy_name=policy_name,
            seed=46,
            checkpoint=checkpoint,
            eval_args=spec.extra_eval_args,
            logs_root=output_root / "logs",
        )
        row = _extract_metrics(
            seed=46,
            label=label,
            dirs=dirs,
            baseline_policy_dir=baseline_seed46_dirs["policy"],
        )
        stage3_rows.append(row)

    stage3_df = pd.DataFrame(stage3_rows).sort_values("delta_vs_baseline_3c", ascending=False)
    stage3_df.to_csv(output_root / "stage3_seed46_corroboration.csv", index=False)

    ranking_df = pd.concat([stage1_df[["candidate", "delta_vs_baseline_3c", "delta_vs_teacher_full", "candidate_sr"]], stage3_df[["candidate", "delta_vs_baseline_3c", "delta_vs_teacher_full", "candidate_sr"]]], ignore_index=True)
    ranking = (
        ranking_df.groupby("candidate", as_index=False)
        .agg(
            mean_delta_vs_baseline_3c=("delta_vs_baseline_3c", "mean"),
            mean_delta_vs_teacher_full=("delta_vs_teacher_full", "mean"),
            mean_candidate_sr=("candidate_sr", "mean"),
            corroborated_seeds=("candidate", "count"),
        )
        .sort_values(["mean_delta_vs_baseline_3c", "mean_candidate_sr"], ascending=False)
    )
    ranking.to_csv(output_root / "critic_system_ranking.csv", index=False)

    if not ranking.empty:
        best_label = str(ranking.iloc[0]["candidate"])
        if best_label != "stronger_value_head_v2":
            if best_label in top2:
                if best_label == "stronger_value_head_v2":
                    pass
                else:
                    seed45_cand_dir = (stage1_root / best_label / "strict_eval_val_B30" / f"rl_set_seed45_{best_label}_train4823")
                    seed45_base_dir = baseline_seed45_dirs["policy"]
                    _minimal_attribution(seed45_cand_dir, seed45_base_dir, output_root / "stage4_min_attr_seed45.json")

                    if best_label in [str(x) for x in stage3_df["candidate"].tolist()]:
                        seed46_cand_dir = (stage3_root / best_label / "strict_eval_val_B30" / f"rl_set_seed46_{best_label}_train4823")
                        if not seed46_cand_dir.exists():
                            seed46_cand_dir = _strict_dirs(STRONGER_V2_ROOT_BY_SEED[46], STRONGER_V2_POLICY_NAME_BY_SEED[46])["policy"]
                        _minimal_attribution(seed46_cand_dir, baseline_seed46_dirs["policy"], output_root / "stage4_min_attr_seed46.json")


if __name__ == "__main__":
    main()
