from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv
from src.scripts.run_posterior_like_belief_audit import load_runtime_context
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    CaseRecord,
    collect_dataset_assets,
    colon_case_id_from_data,
    read_json,
)
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_PRECHECK_ROOT,
    DEFAULT_SOURCE_ROOT,
    EXTENDED_SOFT_SCENARIO_FAMILIES,
    GLOBAL_FEATURE_NAMES,
    LOCAL_FEATURE_NAMES_BASE,
    LOCAL_FEATURE_NAMES_SURROGATE,
    SpimNativePolicy,
    auto_select_teacher,
    get_device,
    run_policy_on_cases,
    summarize_case_metrics,
)


RUNNER_VERSION = "spim_policy_eval_v1"
PANEL_VERSION = "spim_policy_eval_cross_split_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SPIM teacher or trained student checkpoint on exact136/train/val/test splits.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--precheck-root", type=str, default=str(DEFAULT_PRECHECK_ROOT))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--split", type=str, default="val", choices=["exact136", "train", "val", "test"])
    parser.add_argument("--train-max-cases", type=int, default=0)
    parser.add_argument("--train-cache-version", type=str, default="")

    parser.add_argument("--policy-mode", type=str, default="student", choices=["teacher", "student"])
    parser.add_argument("--policy-name", type=str, default="student_eval")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument(
        "--teacher-family",
        type=str,
        default="auto",
        choices=["auto", "hsr_soft_scenario_posterior_v3", *EXTENDED_SOFT_SCENARIO_FAMILIES, "hsr_soft_scenario_posterior_v6", "hsr_paper_topk_ema_v1"],
    )
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--actions-per-round", type=int, default=3)

    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)

    parser.add_argument("--top-source-k", type=int, default=8)
    parser.add_argument("--include-surrogate-features", action="store_true")

    parser.add_argument("--hit-reward", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=-1.0 / 30.0)
    parser.add_argument(
        "--reward-family",
        type=str,
        default="reward_r0_terminal_step",
        choices=[
            "reward_r0_terminal_step",
            "reward_r1_cover_shrink",
            "reward_r2_topk_scenario_error_improve",
            "reward_r3_cover_plus_error",
        ],
    )
    parser.add_argument("--reward-lambda-cover", type=float, default=0.0)
    parser.add_argument("--reward-lambda-error", type=float, default=0.0)
    parser.add_argument("--reward-cover-delta-clip", type=float, default=0.2)
    parser.add_argument("--reward-error-delta-clip", type=float, default=2.0)
    parser.add_argument("--reward-topk-fraction", type=float, default=0.12)
    parser.add_argument("--reward-time-tol-min", type=float, default=30.0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _life_support_clean(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    life_support = out.get("life_support")
    if isinstance(life_support, dict) and str(life_support.get("profile")) == "custom_direct_edit":
        out["life_support"] = {k: v for k, v in life_support.items() if k != "profile"}
    return out


def _load_cfg_from_source(source_root: Path) -> Path:
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    return Path(oracle_manifest["config_path"])


def _load_split_cases(
    *,
    cfg_path: Path,
    cache_dir: Path,
    split: str,
    train_max_cases: int,
    train_cache_version: str,
) -> Tuple[List[CaseRecord], Dict[str, Any], Dict[str, Any]]:
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    payload = _life_support_clean(payload)

    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(payload)
    cfg.training.enable_eval = False
    cfg.training.train_only = bool(split == "train")
    cfg.training.enable_wandb = False

    cfg.data.skip_lmdb = False
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    if split == "train":
        if int(train_max_cases) > 0:
            cfg.data.max_samples = int(train_max_cases)
        if str(train_cache_version).strip():
            cfg.data.cache_version = str(train_cache_version).strip()
    else:
        # Held-out lanes must keep full split cardinality; do not inherit train caps.
        cfg.data.max_samples = None
    cfg.data.rebuild_cache = False
    cfg.paths.cache_dir = str(cache_dir)

    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        eval_batch_size=1,
        skip_lmdb=False,
        train_only=bool(split == "train"),
    )
    if split == "train":
        dataset = train_loader.dataset
    elif split == "val":
        if val_loader is None:
            raise RuntimeError("val loader is None")
        dataset = val_loader.dataset
    elif split == "test":
        if test_loader is None:
            raise RuntimeError("test loader is None")
        dataset = test_loader.dataset
    else:
        raise ValueError(f"Unsupported split: {split}")

    assets = collect_dataset_assets(dataset)
    if assets.get("topology") is None:
        assets["topology"] = HydraulicTopology(cfg.paths.foundation_path)

    split_tag = str(split)
    cases: List[CaseRecord] = []
    for dataset_idx in range(len(dataset)):
        data = dataset[dataset_idx]
        case_id, scenario_id, part_id = colon_case_id_from_data(data, split_tag, dataset_idx)
        cases.append(
            CaseRecord(
                case_id=str(case_id),
                scenario_id=int(scenario_id),
                part_id=int(part_id),
                dataset_index=int(dataset_idx),
                data=deepcopy(data),
            )
        )
    meta = {
        "cfg_path": str(cfg_path),
        "split_dir": str(getattr(cfg.paths, "split_dir", "")),
        "cache_version": str(getattr(cfg.data, "cache_version", "")),
        "train_max_cases": int(train_max_cases),
        "split": str(split),
    }
    return cases, assets, meta


def build_runtime(
    *,
    source_root: Path,
    cache_dir: Path,
    split: str,
    num_rounds: int,
    actions_per_round: int,
    train_max_cases: int,
    train_cache_version: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if split == "exact136":
        runtime = load_runtime_context(source_root, cache_dir)
        runtime["num_episodes"] = int(num_rounds)
        runtime["action_budget"] = int(actions_per_round)
        meta = {
            "split": "exact136",
            "cfg_path": None,
            "split_dir": None,
            "cache_version": None,
        }
        return runtime, meta

    cfg_path = _load_cfg_from_source(source_root)
    cases, dataset_assets, split_meta = _load_split_cases(
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        split=split,
        train_max_cases=int(train_max_cases),
        train_cache_version=str(train_cache_version),
    )
    exact_runtime = load_runtime_context(source_root, cache_dir)
    runtime = {
        "cases": cases,
        "dataset_assets": dataset_assets,
        "num_episodes": int(num_rounds),
        "action_budget": int(actions_per_round),
        "episode_duration_min": float(exact_runtime["episode_duration_min"]),
        "frontier_role_mode": str(exact_runtime["frontier_role_mode"]),
    }
    return runtime, split_meta


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    precheck_root = Path(args.precheck_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(str(args.device))

    if str(args.teacher_family) == "auto":
        teacher_decision = auto_select_teacher(precheck_root)
        teacher_family = str(teacher_decision["selected_family"])
    else:
        teacher_family = str(args.teacher_family)
        teacher_decision = {
            "selected_family": teacher_family,
            "selection_rule": "manual_override",
        }

    runtime, split_meta = build_runtime(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.split),
        num_rounds=int(args.num_rounds),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=int(args.train_max_cases),
        train_cache_version=str(args.train_cache_version),
    )

    model: Optional[SpimNativePolicy] = None
    local_feature_names = list(LOCAL_FEATURE_NAMES_BASE)
    if bool(args.include_surrogate_features):
        local_feature_names.extend(LOCAL_FEATURE_NAMES_SURROGATE)

    if str(args.policy_mode) == "student":
        ckpt = str(args.checkpoint).strip()
        if not ckpt:
            raise ValueError("--checkpoint is required for --policy-mode student")
        checkpoint_path = Path(ckpt)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = SpimNativePolicy(
            global_dim=len(GLOBAL_FEATURE_NAMES),
            local_dim=len(local_feature_names),
            hidden_dim=int(args.hidden_dim),
        ).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)

    env = CleanTwoChannelEvidenceEnv()
    eval_out = run_policy_on_cases(
        cases=runtime["cases"],
        family=teacher_family,
        runtime=runtime,
        env=env,
        policy_name=str(args.policy_name),
        model=model,
        deterministic=True,
        base_seed=int(args.seed),
        include_surrogate_features=bool(args.include_surrogate_features),
        top_source_k=int(args.top_source_k),
        paper_like_alpha=float(args.paper_like_alpha),
        paper_like_topk_fraction=float(args.paper_like_topk_fraction),
        paper_like_time_tol_min=float(args.paper_like_time_tol_min),
        soft_scenario_beta=float(args.soft_scenario_beta),
        hit_reward=float(args.hit_reward),
        step_penalty=float(args.step_penalty),
        reward_family=str(args.reward_family),
        reward_lambda_cover=float(args.reward_lambda_cover),
        reward_lambda_error=float(args.reward_lambda_error),
        reward_cover_delta_clip=float(args.reward_cover_delta_clip),
        reward_error_delta_clip=float(args.reward_error_delta_clip),
        reward_topk_fraction=float(args.reward_topk_fraction),
        reward_time_tol_min=float(args.reward_time_tol_min),
        device=device,
        collect_transitions=False,
    )
    summary = summarize_case_metrics(
        eval_out["case_rows"],
        num_rounds=int(args.num_rounds),
        action_budget=int(args.actions_per_round),
    )

    import pandas as pd

    pd.DataFrame(eval_out["case_rows"]).to_csv(output_dir / "case_rows.csv", index=False)
    pd.DataFrame(eval_out["step_rows"]).to_csv(output_dir / "step_rows.csv", index=False)

    payload = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "source_root": str(source_root),
        "cache_dir": str(cache_dir),
        "seed": int(args.seed),
        "device": str(device),
        "split": str(args.split),
        "split_case_count": int(len(runtime["cases"])),
        "split_meta": split_meta,
        "teacher_family": teacher_family,
        "teacher_decision": teacher_decision,
        "policy_mode": str(args.policy_mode),
        "policy_name": str(args.policy_name),
        "checkpoint": None if str(args.policy_mode) == "teacher" else str(Path(str(args.checkpoint))),
        "protocol": {
            "num_rounds": int(args.num_rounds),
            "actions_per_round": int(args.actions_per_round),
            "budget": int(args.num_rounds) * int(args.actions_per_round),
        },
        "reward": {
            "family": str(args.reward_family),
            "lambda_cover": float(args.reward_lambda_cover),
            "lambda_error": float(args.reward_lambda_error),
        },
        "summary": summary,
        "artifacts": {
            "case_rows": str(output_dir / "case_rows.csv"),
            "step_rows": str(output_dir / "step_rows.csv"),
        },
    }
    write_json(output_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
