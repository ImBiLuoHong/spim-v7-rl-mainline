from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "run_spim_teacher_imitation_rl_pilot.py"

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact136_h3_formal_v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
DEFAULT_PRECHECK_ROOT = PROJECT_ROOT / "artifacts" / "spim_teacher_precheck" / "20260410_trainfull512_v3_vs_v1_b30_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "spim_reward_family_teacher_control" / "20260411_exact136_b30_reward_family_teacher_control_v1"

RUNNER_VERSION = "spim_reward_family_teacher_control_v1"
PANEL_VERSION = "exact136_b30_spim_native_reward_family_teacher_control_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded 2-stage SPIM reward-family sweep + teacher vs no-teacher control via pilot runner."
    )
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--precheck-root", type=str, default=str(DEFAULT_PRECHECK_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)

    parser.add_argument("--train-full-max-cases", type=int, default=320)
    parser.add_argument("--train-full-cache-version", type=str, default="train_full_rlpilot_n320_reward_family_v1")
    parser.add_argument("--rl-epochs", type=int, default=6)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--rl-minibatch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, float):
        if math.isnan(value):
            return default
        return float(value)
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return default
    try:
        val = float(text)
    except ValueError:
        return default
    if math.isnan(val):
        return default
    return val


def run_command(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def command_to_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def build_common_args(args: argparse.Namespace, run_dir: Path) -> List[str]:
    return [
        str(PILOT_SCRIPT),
        "--source-root",
        str(Path(args.source_root)),
        "--cache-dir",
        str(Path(args.cache_dir)),
        "--precheck-root",
        str(Path(args.precheck_root)),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(int(args.seed)),
        "--train-full-max-cases",
        str(int(args.train_full_max_cases)),
        "--train-full-cache-version",
        str(args.train_full_cache_version),
        "--bc-epochs",
        str(int(args.bc_epochs)),
        "--bc-batch-size",
        str(int(args.bc_batch_size)),
        "--rl-epochs",
        str(int(args.rl_epochs)),
        "--rl-minibatch-size",
        str(int(args.rl_minibatch_size)),
        "--device",
        str(args.device),
    ]


def build_prep_bc_command(args: argparse.Namespace, run_dir: Path, save_ckpt_path: Path) -> List[str]:
    cmd = [sys.executable]
    cmd.extend(build_common_args(args, run_dir))
    cmd.extend(
        [
            "--rl-init-mode",
            "teacher_warm_start",
            "--reward-family",
            "reward_r0_terminal_step",
            "--reward-lambda-cover",
            "0.0",
            "--reward-lambda-error",
            "0.0",
            "--rl-epochs",
            "0",
            "--save-bc-checkpoint",
            str(save_ckpt_path),
        ]
    )
    return cmd


def build_stage_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    reward_family: str,
    lambda_cover: float,
    lambda_error: float,
    init_mode: str,
    load_ckpt: Optional[Path],
) -> List[str]:
    cmd = [sys.executable]
    cmd.extend(build_common_args(args, run_dir))
    cmd.extend(
        [
            "--reward-family",
            str(reward_family),
            "--reward-lambda-cover",
            str(float(lambda_cover)),
            "--reward-lambda-error",
            str(float(lambda_error)),
            "--rl-init-mode",
            str(init_mode),
            "--rl-imitation-anchor-start",
            "0",
            "--rl-imitation-anchor-end",
            "0",
            "--skip-bc-train",
        ]
    )
    if load_ckpt is not None:
        cmd.extend(["--load-bc-checkpoint", str(load_ckpt)])
    return cmd


def get_epoch_reach_teacher(rl_history_rows: List[Dict[str, str]], teacher_sr: float) -> Optional[float]:
    for row in rl_history_rows:
        epoch = as_float(row.get("epoch"), default=None)
        train_sr = as_float(row.get("train_success_rate"), default=None)
        if epoch is None or train_sr is None:
            continue
        if train_sr >= teacher_sr:
            return float(epoch)
    return None


def safe_inf_for_none(val: Optional[float]) -> float:
    if val is None:
        return float("inf")
    return float(val)


def noise_flag(improvement: float, rl_history_rows: List[Dict[str, str]]) -> bool:
    abs_improve = abs(float(improvement))
    if abs_improve < 0.005:
        return True
    if improvement <= 0.0:
        return False
    train_srs = [as_float(r.get("train_success_rate"), default=None) for r in rl_history_rows]
    train_srs = [x for x in train_srs if x is not None]
    if not train_srs:
        return False
    last = float(train_srs[-1])
    peak = float(max(train_srs))
    return last < (peak - 0.03)


def load_run_bundle(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    policy_summary_path = run_dir / "policy_summary.csv"
    rl_history_path = run_dir / "rl_train_history.csv"

    summary = read_json(summary_path)
    policy_rows = read_csv_rows(policy_summary_path)
    rl_history_rows = read_csv_rows(rl_history_path)

    teacher_summary = summary.get("teacher_summary") or {}
    rl_summary = summary.get("rl_summary") or {}

    teacher_sr = as_float(teacher_summary.get("success_rate"), default=0.0) or 0.0
    rl_sr = as_float(rl_summary.get("success_rate"), default=0.0) or 0.0
    budget = as_float(rl_summary.get("budget_used_mean"), default=float("inf"))
    avg_hit_round = as_float(rl_summary.get("avg_hit_round_conditional"), default=None)

    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "policy_summary_path": str(policy_summary_path),
        "rl_history_path": str(rl_history_path),
        "summary": summary,
        "policy_rows": policy_rows,
        "rl_history_rows": rl_history_rows,
        "teacher_sr": float(teacher_sr),
        "rl_sr": float(rl_sr),
        "budget_used_mean": float(budget) if budget is not None else float("inf"),
        "avg_hit_round_conditional": avg_hit_round,
    }


def rank_stage1(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            -float(r["rl_sr"]),
            float(r["budget_used_mean"]),
            safe_inf_for_none(r["avg_hit_round_conditional"]),
        ),
    )


def stage1_judgment(ranked: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ranked:
        return {
            "best_reward_family": None,
            "best_is_stable": False,
            "noise_like_arms": [],
            "shaping_has_value": "not_proved",
        }

    best = ranked[0]
    noise_like_arms = [r["arm_name"] for r in ranked if bool(r["noise_flag"]) ]
    best_is_stable = not bool(best["noise_flag"])

    if best["reward_family"] == "reward_r0_terminal_step":
        shaping_value = "not_proved"
    elif best["improvement_vs_teacher"] > 0.0 and best_is_stable:
        shaping_value = "proved"
    else:
        shaping_value = "partially_proved"

    return {
        "best_reward_family": best["reward_family"],
        "best_arm_name": best["arm_name"],
        "best_is_stable": bool(best_is_stable),
        "noise_like_arms": noise_like_arms,
        "shaping_has_value": shaping_value,
    }


def stage2_judgment(warm: Dict[str, Any], random_arm: Dict[str, Any]) -> Dict[str, Any]:
    warm_sr = float(warm["rl_sr"])
    rand_sr = float(random_arm["rl_sr"])
    warm_epoch = warm.get("epoch_reach_teacher")
    rand_epoch = random_arm.get("epoch_reach_teacher")

    speed_status = "not_proved"
    if warm_epoch is not None and rand_epoch is not None and warm_epoch < rand_epoch:
        speed_status = "proved"
    elif warm_epoch is not None and rand_epoch is None:
        speed_status = "proved"
    elif warm_epoch is not None or rand_epoch is not None:
        speed_status = "partially_proved"

    final_strength_status = "proved" if warm_sr > rand_sr else ("partially_proved" if warm_sr == rand_sr else "not_proved")
    random_learnable = bool(random_arm["random_init_learnable"])

    if final_strength_status == "proved" and speed_status == "proved":
        warm_start_value = "strongly_useful"
    elif final_strength_status in {"proved", "partially_proved"} or speed_status in {"proved", "partially_proved"}:
        warm_start_value = "useful_but_bounded"
    else:
        warm_start_value = "not_clearly_needed"

    return {
        "warm_start_faster": speed_status,
        "warm_start_stronger_final": final_strength_status,
        "random_init_learnable": random_learnable,
        "warm_start_value": warm_start_value,
    }


def format_float(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{digits}f}"


def build_markdown_report(
    *,
    args: argparse.Namespace,
    prep: Dict[str, Any],
    stage1_ranked: List[Dict[str, Any]],
    stage1_eval: Dict[str, Any],
    stage2_rows: List[Dict[str, Any]],
    stage2_eval: Dict[str, Any],
    commands: List[str],
    artifacts: Dict[str, Any],
    source_root: str,
    cache_dir: str,
    foundation_graph_path: str,
) -> str:
    lines: List[str] = []
    lines.append("# SPIM Reward Family + Teacher Control Report")
    lines.append("")
    lines.append("## Run Meta")
    lines.append(f"- runner_version: {RUNNER_VERSION}")
    lines.append(f"- panel_version: {PANEL_VERSION}")
    lines.append(f"- seed: {int(args.seed)}")
    lines.append(f"- source_root: {source_root}")
    lines.append(f"- cache_dir: {cache_dir}")
    lines.append(f"- precheck_root: {str(Path(args.precheck_root))}")
    lines.append(f"- train_full_cache_version: {str(args.train_full_cache_version)}")
    lines.append(f"- foundation_graph_path: {foundation_graph_path}")
    lines.append(f"- protocol: exact136, B30, shared SPIM-native policy/action contract from pilot")
    lines.append("")

    lines.append("## Stage0 Prep BC")
    lines.append(f"- prep_run_dir: {prep['run_dir']}")
    lines.append(f"- bc_checkpoint: {prep['bc_checkpoint_path']}")
    lines.append("")

    lines.append("## Reward Family Definitions")
    lines.append("- reward_r0_terminal_step: lambda_cover=0.0, lambda_error=0.0")
    lines.append("- reward_r1_cover_shrink: lambda_cover=0.1, lambda_error=0.0")
    lines.append("- reward_r2_topk_scenario_error_improve: lambda_cover=0.0, lambda_error=0.1")
    lines.append("- reward_r3_cover_plus_error: lambda_cover=0.05, lambda_error=0.05")
    lines.append("")

    lines.append("## Stage1 Results")
    lines.append("| rank | arm | reward_family | rl_sr | teacher_sr | delta_rl_minus_teacher | budget_used_mean | avg_hit_round_conditional | noise_flag |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for i, row in enumerate(stage1_ranked, start=1):
        lines.append(
            "| "
            f"{i} | {row['arm_name']} | {row['reward_family']} | {format_float(row['rl_sr'])} | {format_float(row['teacher_sr'])} | "
            f"{format_float(row['improvement_vs_teacher'])} | {format_float(row['budget_used_mean'])} | {format_float(row['avg_hit_round_conditional'])} | {str(bool(row['noise_flag']))} |"
        )
    lines.append(f"- best_reward_family: {stage1_eval.get('best_reward_family')}")
    lines.append(f"- best_is_stable: {str(stage1_eval.get('best_is_stable'))}")
    lines.append(f"- noise_like_arms: {', '.join(stage1_eval.get('noise_like_arms', [])) if stage1_eval.get('noise_like_arms') else 'none'}")
    lines.append(f"- shaping_has_value: {stage1_eval.get('shaping_has_value')}")
    lines.append("")

    lines.append("## Stage2 Results")
    lines.append("| arm | init_mode | reward_family | rl_sr | teacher_sr | delta_rl_minus_teacher | epoch_reach_teacher | random_init_learnable |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for row in stage2_rows:
        lines.append(
            "| "
            f"{row['arm_name']} | {row['init_mode']} | {row['reward_family']} | {format_float(row['rl_sr'])} | {format_float(row['teacher_sr'])} | "
            f"{format_float(row['improvement_vs_teacher'])} | {format_float(row.get('epoch_reach_teacher'))} | {str(bool(row.get('random_init_learnable', False)))} |"
        )
    lines.append(f"- warm_start_faster: {stage2_eval.get('warm_start_faster')}")
    lines.append(f"- warm_start_stronger_final: {stage2_eval.get('warm_start_stronger_final')}")
    lines.append(f"- random_init_learnable: {str(stage2_eval.get('random_init_learnable'))}")
    lines.append(f"- warm_start_value: {stage2_eval.get('warm_start_value')}")
    lines.append("")

    lines.append("## Proof Boundary")
    lines.append(f"- reward_family_most_promising: {stage1_eval.get('best_reward_family')} [{ 'proved' if stage1_eval.get('best_reward_family') else 'not_proved' }]")
    lines.append(f"- reward_shaping_value: {stage1_eval.get('shaping_has_value')}")
    lines.append(f"- teacher_warm_start_is_good: {stage2_eval.get('warm_start_value')}")
    lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- summary_json: {artifacts.get('summary_json')}")
    lines.append(f"- report_md: {artifacts.get('report_md')}")
    for key, value in artifacts.items():
        if key in {"summary_json", "report_md"}:
            continue
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Commands")
    for cmd in commands:
        lines.append(f"- {cmd}")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: List[str] = []

    prep_dir = output_dir / "prep_bc"
    prep_ckpt = prep_dir / "bc_teacher_warm_start.pt"
    prep_cmd = build_prep_bc_command(args, prep_dir, prep_ckpt)
    run_command(prep_cmd)
    commands.append(command_to_str(prep_cmd))

    prep_summary = load_run_bundle(prep_dir)
    prep_info = {
        "run_dir": str(prep_dir),
        "summary_path": prep_summary["summary_path"],
        "policy_summary_path": prep_summary["policy_summary_path"],
        "rl_history_path": prep_summary["rl_history_path"],
        "bc_checkpoint_path": str(prep_ckpt),
    }

    stage1_reward_cfgs: List[Tuple[str, str, float, float]] = [
        ("reward_r0_terminal_step", "reward_r0_terminal_step", 0.0, 0.0),
        ("reward_r1_cover_shrink", "reward_r1_cover_shrink", 0.1, 0.0),
        ("reward_r2_topk_scenario_error_improve", "reward_r2_topk_scenario_error_improve", 0.0, 0.1),
        ("reward_r3_cover_plus_error", "reward_r3_cover_plus_error", 0.05, 0.05),
    ]

    stage1_rows: List[Dict[str, Any]] = []
    for arm_name, reward_family, lambda_cover, lambda_error in stage1_reward_cfgs:
        run_dir = output_dir / "stage1" / arm_name
        cmd = build_stage_command(
            args,
            run_dir=run_dir,
            reward_family=reward_family,
            lambda_cover=lambda_cover,
            lambda_error=lambda_error,
            init_mode="teacher_warm_start",
            load_ckpt=prep_ckpt,
        )
        run_command(cmd)
        commands.append(command_to_str(cmd))

        bundle = load_run_bundle(run_dir)
        improvement = float(bundle["rl_sr"] - bundle["teacher_sr"])
        stage1_rows.append(
            {
                "arm_name": arm_name,
                "reward_family": reward_family,
                "lambda_cover": float(lambda_cover),
                "lambda_error": float(lambda_error),
                "run_dir": str(run_dir),
                "summary_path": bundle["summary_path"],
                "policy_summary_path": bundle["policy_summary_path"],
                "rl_history_path": bundle["rl_history_path"],
                "teacher_sr": float(bundle["teacher_sr"]),
                "rl_sr": float(bundle["rl_sr"]),
                "improvement_vs_teacher": improvement,
                "budget_used_mean": float(bundle["budget_used_mean"]),
                "avg_hit_round_conditional": bundle["avg_hit_round_conditional"],
                "noise_flag": bool(noise_flag(improvement, bundle["rl_history_rows"])),
                "policy_summary_rows": len(bundle["policy_rows"]),
                "summary": bundle["summary"],
            }
        )

    stage1_ranked = rank_stage1(stage1_rows)
    stage1_eval = stage1_judgment(stage1_ranked)

    best_reward = stage1_eval.get("best_reward_family")
    if best_reward is None:
        raise RuntimeError("Stage1 did not produce a best reward family.")

    best_cfg = next((x for x in stage1_reward_cfgs if x[1] == best_reward), None)
    if best_cfg is None:
        raise RuntimeError(f"Best reward family {best_reward} not found in fixed stage1 configs.")
    _, _, best_lambda_cover, best_lambda_error = best_cfg

    stage2_rows: List[Dict[str, Any]] = []

    stage2_arms = [
        {
            "arm_name": "teacher_warm_start",
            "init_mode": "teacher_warm_start",
            "load_ckpt": prep_ckpt,
        },
        {
            "arm_name": "random_init",
            "init_mode": "random_init",
            "load_ckpt": None,
        },
    ]

    for arm in stage2_arms:
        run_dir = output_dir / "stage2" / str(arm["arm_name"])
        cmd = build_stage_command(
            args,
            run_dir=run_dir,
            reward_family=best_reward,
            lambda_cover=float(best_lambda_cover),
            lambda_error=float(best_lambda_error),
            init_mode=str(arm["init_mode"]),
            load_ckpt=arm["load_ckpt"],
        )
        run_command(cmd)
        commands.append(command_to_str(cmd))

        bundle = load_run_bundle(run_dir)
        teacher_sr = float(bundle["teacher_sr"])
        rl_sr = float(bundle["rl_sr"])
        stage2_rows.append(
            {
                "arm_name": str(arm["arm_name"]),
                "init_mode": str(arm["init_mode"]),
                "reward_family": str(best_reward),
                "lambda_cover": float(best_lambda_cover),
                "lambda_error": float(best_lambda_error),
                "run_dir": str(run_dir),
                "summary_path": bundle["summary_path"],
                "policy_summary_path": bundle["policy_summary_path"],
                "rl_history_path": bundle["rl_history_path"],
                "teacher_sr": teacher_sr,
                "rl_sr": rl_sr,
                "improvement_vs_teacher": float(rl_sr - teacher_sr),
                "epoch_reach_teacher": get_epoch_reach_teacher(bundle["rl_history_rows"], teacher_sr),
                "random_init_learnable": bool(str(arm["arm_name"]) == "random_init" and rl_sr > 0.05),
                "policy_summary_rows": len(bundle["policy_rows"]),
                "summary": bundle["summary"],
            }
        )

    warm_row = next(r for r in stage2_rows if r["arm_name"] == "teacher_warm_start")
    random_row = next(r for r in stage2_rows if r["arm_name"] == "random_init")
    stage2_eval = stage2_judgment(warm_row, random_row)

    foundation_graph_path = ""
    if stage1_ranked:
        foundation_graph_path = str(stage1_ranked[0]["summary"].get("foundation_graph_path", ""))

    artifacts: Dict[str, Any] = {
        "prep_summary": prep_info["summary_path"],
        "prep_policy_summary": prep_info["policy_summary_path"],
        "prep_rl_history": prep_info["rl_history_path"],
        "prep_bc_checkpoint": prep_info["bc_checkpoint_path"],
    }

    for row in stage1_ranked:
        prefix = f"stage1_{row['arm_name']}"
        artifacts[f"{prefix}_summary"] = row["summary_path"]
        artifacts[f"{prefix}_policy_summary"] = row["policy_summary_path"]
        artifacts[f"{prefix}_rl_history"] = row["rl_history_path"]

    for row in stage2_rows:
        prefix = f"stage2_{row['arm_name']}"
        artifacts[f"{prefix}_summary"] = row["summary_path"]
        artifacts[f"{prefix}_policy_summary"] = row["policy_summary_path"]
        artifacts[f"{prefix}_rl_history"] = row["rl_history_path"]

    proof_boundary = {
        "reward_family_most_promising": {
            "status": "proved" if stage1_eval.get("best_reward_family") is not None else "not_proved",
            "value": stage1_eval.get("best_reward_family"),
        },
        "reward_shaping_value": {
            "status": str(stage1_eval.get("shaping_has_value")),
            "value": str(stage1_eval.get("shaping_has_value")),
        },
        "teacher_warm_start_is_good": {
            "status": (
                "proved"
                if stage2_eval.get("warm_start_value") == "strongly_useful"
                else "partially_proved"
                if stage2_eval.get("warm_start_value") == "useful_but_bounded"
                else "not_proved"
            ),
            "value": stage2_eval.get("warm_start_value"),
        },
    }

    summary: Dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "source_root": str(Path(args.source_root)),
        "cache_dir": str(Path(args.cache_dir)),
        "precheck_root": str(Path(args.precheck_root)),
        "train_full_max_cases": int(args.train_full_max_cases),
        "train_full_cache_version": str(args.train_full_cache_version),
        "rl_epochs": int(args.rl_epochs),
        "bc_epochs": int(args.bc_epochs),
        "bc_batch_size": int(args.bc_batch_size),
        "rl_minibatch_size": int(args.rl_minibatch_size),
        "device": str(args.device),
        "protocol": {
            "dataset_panel": "exact136",
            "budget": "B30",
            "input_contract": "SPIM-native",
            "action_contract": "sequential masked selection without replacement, 3 actions/round",
            "policy_contract": "shared scorer MLP + value head inherited from pilot",
            "fixed_conditions": "all arms share pilot protocol; only reward family and init mode vary as specified",
        },
        "stage0_prep_bc": prep_info,
        "reward_family_definitions": [
            {
                "reward_family": "reward_r0_terminal_step",
                "lambda_cover": 0.0,
                "lambda_error": 0.0,
            },
            {
                "reward_family": "reward_r1_cover_shrink",
                "lambda_cover": 0.1,
                "lambda_error": 0.0,
            },
            {
                "reward_family": "reward_r2_topk_scenario_error_improve",
                "lambda_cover": 0.0,
                "lambda_error": 0.1,
            },
            {
                "reward_family": "reward_r3_cover_plus_error",
                "lambda_cover": 0.05,
                "lambda_error": 0.05,
            },
        ],
        "stage1": {
            "arms": [
                {
                    k: v
                    for k, v in row.items()
                    if k != "summary"
                }
                for row in stage1_ranked
            ],
            "best_reward_family": stage1_eval.get("best_reward_family"),
            "best_arm_name": stage1_eval.get("best_arm_name"),
            "best_is_stable": bool(stage1_eval.get("best_is_stable", False)),
            "noise_like_arms": list(stage1_eval.get("noise_like_arms", [])),
            "shaping_has_value": stage1_eval.get("shaping_has_value"),
        },
        "stage2": {
            "best_reward_from_stage1": best_reward,
            "arms": [
                {
                    k: v
                    for k, v in row.items()
                    if k != "summary"
                }
                for row in stage2_rows
            ],
            "comparisons": stage2_eval,
        },
        "proof_boundary": proof_boundary,
        "current_judgment": {
            "which_reward_most_promising": stage1_eval.get("best_reward_family"),
            "does_shaping_help": stage1_eval.get("shaping_has_value"),
            "teacher_warm_start_good_idea": stage2_eval.get("warm_start_value"),
            "next_step_focus": (
                "reward_deepen"
                if stage1_eval.get("best_reward_family") != "reward_r0_terminal_step"
                else "training_robustness_lock"
            ),
        },
        "artifacts": artifacts,
        "code_paths": [
            str(PILOT_SCRIPT),
            str(Path(__file__).resolve()),
        ],
        "commands": commands,
        "foundation_graph_path": foundation_graph_path,
    }

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    artifacts["summary_json"] = str(summary_path)
    artifacts["report_md"] = str(report_path)

    report_text = build_markdown_report(
        args=args,
        prep=prep_info,
        stage1_ranked=stage1_ranked,
        stage1_eval=stage1_eval,
        stage2_rows=stage2_rows,
        stage2_eval=stage2_eval,
        commands=commands,
        artifacts=artifacts,
        source_root=str(Path(args.source_root)),
        cache_dir=str(Path(args.cache_dir)),
        foundation_graph_path=foundation_graph_path,
    )
    report_path.write_text(report_text, encoding="utf-8")

    summary["artifacts"] = artifacts
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
