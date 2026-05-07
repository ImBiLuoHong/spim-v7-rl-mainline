from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_authoritative_hsr_baseline import resolve_foundation_graph_path
from src.scripts.run_posterior_like_belief_audit import load_runtime_context
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    CaseRecord,
    collect_dataset_assets,
    colon_case_id_from_data,
    make_rollout_state,
    read_json,
)
from src.scripts.run_spim_family_sweep import (
    PaperLikeHSRState,
    _compute_scenario_error,
    _belief_metrics,
    _build_clean_candidate_mask,
    _extract_trigger_global,
    _paper_topk_ema_posterior,
    _pick_topk_unsampled,
    _soft_scenario_posterior,
    _soft_scenario_posterior_v6,
)

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact136_h3_formal_v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
DEFAULT_PRECHECK_ROOT = PROJECT_ROOT / "artifacts" / "spim_teacher_precheck" / "20260410_trainfull512_v3_vs_v1_b30_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "spim_teacher_imitation_rl_pilot" / "20260410_exact136_b30_pilot_v1"
RUNNER_VERSION = "spim_teacher_imitation_rl_pilot_v1"
PANEL_VERSION = "exact136_b30_spim_native_teacher_imitation_rl_v1"

GLOBAL_FEATURE_NAMES = [
    "round_index_norm",
    "remaining_budget_norm",
    "candidate_count_norm",
    "candidate_ratio",
    "positive_count_norm",
    "negative_count_norm",
    "elapsed_time_norm",
    "posterior_entropy_norm",
    "mass_cover_0p7_ratio",
    "top1_mass",
    "top3_mass",
    "top1_top2_margin",
]

LOCAL_FEATURE_NAMES_BASE = [
    "posterior_mass",
    "posterior_rank_percentile",
    "expected_positive_prob",
    "disagreement_score",
    "distance_to_trigger_norm",
    "distance_to_nearest_positive_norm",
    "distance_to_nearest_negative_norm",
    "legal_flag",
    "sampled_flag",
]

LOCAL_FEATURE_NAMES_SURROGATE = [
    "expected_cover_shrink_surrogate",
    "expected_entropy_change_surrogate",
]

GLOBAL_FEATURE_NAMES_UNCERTAINTY_REGIME = [
    "top5_mass",
    "effective_support_size_ratio",
    "high_quality_cover_0p9_ratio",
    "entropy_delta_prev1",
    "entropy_delta_prev2",
    "top1_mass_delta_prev1",
    "top1_mass_delta_prev2",
]

LOCAL_FEATURE_NAMES_UNCERTAINTY_REGIME = [
    "top1_gap",
    "cum_mass_rank_ratio",
    "cover_0p9_member",
    "one_step_entropy_drop_estimate",
    "witness_distance_mean_norm",
    "novelty_redundancy_score",
]

EXTENDED_SOFT_SCENARIO_FAMILIES = [f"hsr_soft_scenario_posterior_v7_{n}offset" for n in range(5, 22, 2)]


def get_global_feature_names(include_uncertainty_regime_features: bool) -> List[str]:
    names = list(GLOBAL_FEATURE_NAMES)
    if bool(include_uncertainty_regime_features):
        names.extend(GLOBAL_FEATURE_NAMES_UNCERTAINTY_REGIME)
    return names


def get_local_feature_names(
    *,
    include_surrogate_features: bool,
    include_uncertainty_regime_features: bool,
) -> List[str]:
    names = list(LOCAL_FEATURE_NAMES_BASE)
    if bool(include_surrogate_features):
        names.extend(LOCAL_FEATURE_NAMES_SURROGATE)
    if bool(include_uncertainty_regime_features):
        names.extend(LOCAL_FEATURE_NAMES_UNCERTAINTY_REGIME)
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal SPIM-native teacher imitation -> RL fine-tune pilot on exact136 B30."
    )
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--precheck-root", type=str, default=str(DEFAULT_PRECHECK_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument(
        "--runner-version-tag",
        type=str,
        default="conservative_corrective_rl_v1",
        help="User-facing run tag persisted into summary.",
    )

    parser.add_argument(
        "--teacher-family",
        type=str,
        default="auto",
        choices=["auto", "hsr_soft_scenario_posterior_v3", *EXTENDED_SOFT_SCENARIO_FAMILIES, "hsr_soft_scenario_posterior_v6", "hsr_paper_topk_ema_v1"],
    )
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--actions-per-round", type=int, default=3)

    parser.add_argument("--train-full-max-cases", type=int, default=256)
    parser.add_argument("--train-full-cache-version", type=str, default="train_full_rlpilot_n256_v1")

    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)

    parser.add_argument("--top-source-k", type=int, default=8)
    parser.add_argument("--include-surrogate-features", action="store_true")
    parser.add_argument("--use-uncertainty-regime-features", action="store_true")
    parser.add_argument(
        "--rl-policy-mode",
        type=str,
        default="free",
        choices=["free", "conservative_corrective_rl_v1"],
    )
    parser.add_argument("--corrective-candidate-topk", type=int, default=6)
    parser.add_argument("--ambiguity-top1-top2-max", type=float, default=0.06)
    parser.add_argument("--ambiguity-min-candidate-count", type=int, default=3)
    parser.add_argument("--slate-size", type=int, default=10)
    parser.add_argument("--slate-top-posterior-k", type=int, default=6)
    parser.add_argument("--slate-high-disagreement-k", type=int, default=3)
    parser.add_argument("--slate-novelty-k", type=int, default=2)
    parser.add_argument("--enable-early-stage-specialist-head", action="store_true")
    parser.add_argument("--early-stage-round-cutoff", type=int, default=0)
    parser.add_argument("--early-stage-slate-top-posterior-k", type=int, default=-1)
    parser.add_argument("--early-stage-slate-high-disagreement-k", type=int, default=-1)
    parser.add_argument("--early-stage-slate-novelty-k", type=int, default=-1)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--policy-arch", type=str, default="separate_heads", choices=["separate_heads", "shared_trunk"])
    parser.add_argument("--policy-mlp-depth", type=int, default=2)
    parser.add_argument("--value-mlp-depth", type=int, default=2)
    parser.add_argument("--value-head-width-mult", type=float, default=1.0)
    parser.add_argument("--critic-trunk-depth", type=int, default=0)
    parser.add_argument("--critic-trunk-hidden-dim", type=int, default=0)
    parser.add_argument("--policy-dropout", type=float, default=0.0)
    parser.add_argument("--policy-norm", type=str, default="none", choices=["none", "layernorm", "rmsnorm"])
    parser.add_argument("--candidate-encoder", type=str, default="none", choices=["none", "self_attention"])
    parser.add_argument("--candidate-attn-heads", type=int, default=4)
    parser.add_argument("--enable-regime-head", action="store_true")
    parser.add_argument("--regime-head-classes", type=int, default=3)
    parser.add_argument("--regime-embed-dim", type=int, default=12)
    parser.add_argument(
        "--arch-backbone",
        type=str,
        default="baseline_mlp",
        choices=[
            "baseline_mlp",
            "residual_mlp_control",
            "slate_transformer_lite",
            "graphsage_lite",
            "gat_lite",
            "cnn_lite",
        ],
    )
    parser.add_argument("--residual-hidden-dim", type=int, default=256)
    parser.add_argument("--residual-depth", type=int, default=4)
    parser.add_argument("--residual-head-dim", type=int, default=128)
    parser.add_argument("--transformer-token-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=256)
    parser.add_argument("--graph-hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--graph-max-subgraph-nodes", type=int, default=512)
    parser.add_argument("--graph-use-onehop", action="store_true")
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--cnn-norm", type=str, default="layernorm", choices=["layernorm", "groupnorm"])
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-recovery-epochs", type=int, default=4)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--bc-batch-size", type=int, default=128)

    parser.add_argument("--rl-epochs", type=int, default=4)
    parser.add_argument("--rl-lr", type=float, default=1e-4)
    parser.add_argument("--rl-gamma", type=float, default=0.97)
    parser.add_argument("--rl-clip-range", type=float, default=0.2)
    parser.add_argument("--rl-update-epochs", type=int, default=2)
    parser.add_argument("--rl-minibatch-size", type=int, default=64)
    parser.add_argument("--rl-value-coef", type=float, default=0.5)
    parser.add_argument("--rl-critic-extra-updates", type=int, default=0)
    parser.add_argument("--critic-warmup-epochs", type=int, default=0)
    parser.add_argument("--rl-entropy-coef-start", type=float, default=0.02)
    parser.add_argument("--rl-entropy-coef-end", type=float, default=0.002)
    parser.add_argument("--rl-imitation-anchor-start", type=float, default=0.05)
    parser.add_argument("--rl-imitation-anchor-end", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rl-init-mode", type=str, default="teacher_warm_start", choices=["teacher_warm_start", "random_init"])
    parser.add_argument("--rl-early-stop-patience", type=int, default=2)
    parser.add_argument("--advantage-baseline", type=str, default="greedy_relative", choices=["value_only", "greedy_relative"])
    parser.add_argument(
        "--ambiguity-weighting-mode",
        type=str,
        default="none",
        choices=["none", "entropy_margin_focus"],
    )
    parser.add_argument("--ambiguity-weight-alpha", type=float, default=0.0)
    parser.add_argument("--ambiguity-entropy-threshold", type=float, default=0.35)
    parser.add_argument("--ambiguity-margin-threshold", type=float, default=0.06)
    parser.add_argument("--ambiguity-entropy-temp", type=float, default=0.08)
    parser.add_argument("--ambiguity-margin-temp", type=float, default=0.02)
    parser.add_argument("--early-stage-set-aux-weight", type=float, default=0.0)
    parser.add_argument("--early-stage-set-adv-mode", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument(
        "--early-stage-set-aux-ambiguity-mode",
        type=str,
        default="none",
        choices=["none", "entropy_margin_focus"],
    )
    parser.add_argument("--early-stage-set-aux-ambiguity-alpha", type=float, default=0.0)

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
    parser.add_argument("--reward-lambda-cover", type=float, default=0.1)
    parser.add_argument("--reward-lambda-error", type=float, default=0.1)
    parser.add_argument("--reward-cover-delta-clip", type=float, default=0.2)
    parser.add_argument("--reward-error-delta-clip", type=float, default=2.0)
    parser.add_argument("--reward-topk-fraction", type=float, default=0.12)
    parser.add_argument("--reward-time-tol-min", type=float, default=30.0)
    parser.add_argument(
        "--decode-consistency-mode",
        type=str,
        default="none",
        choices=["none", "beam_winner_distill", "critic_preference"],
    )
    parser.add_argument("--decode-consistency-weight", type=float, default=0.0)
    parser.add_argument("--decode-consistency-margin", type=float, default=0.05)
    parser.add_argument("--decode-consistency-beam-width", type=int, default=4)
    parser.add_argument("--decode-consistency-topk-per-slot", type=int, default=4)
    parser.add_argument("--decode-consistency-logprob-weight", type=float, default=1.0)
    parser.add_argument("--decode-consistency-value-weight", type=float, default=4.0)
    parser.add_argument("--decode-consistency-max-margin", type=float, default=0.10)
    parser.add_argument("--decode-consistency-max-round", type=int, default=6)
    parser.add_argument("--decode-consistency-min-entropy", type=float, default=0.0)
    parser.add_argument("--decode-consistency-max-preview-candidates", type=int, default=0)
    parser.add_argument("--decode-consistency-max-trigger-states", type=int, default=0)
    parser.add_argument("--skip-bc-train", action="store_true")
    parser.add_argument("--load-bc-checkpoint", type=str, default="")
    parser.add_argument("--save-bc-checkpoint", type=str, default="")
    parser.add_argument("--save-bc-epoch-checkpoints-dir", type=str, default="")
    parser.add_argument("--save-final-checkpoint", type=str, default="")
    parser.add_argument("--save-rl-epoch-checkpoints-dir", type=str, default="")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-b60", action="store_true")
    parser.add_argument("--b60-num-rounds", type=int, default=20)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint_with_specialist_compat(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    state_dict = torch.load(checkpoint_path, map_location=device)
    checkpoint_keys = set(state_dict.keys())

    copied_pairs: List[Tuple[str, str]] = []
    for dst_key, src_key in [
        ("early_stage_action_mlp.0.weight", "action_mlp.0.weight"),
        ("early_stage_action_mlp.0.bias", "action_mlp.0.bias"),
        ("early_stage_action_mlp.3.weight", "action_mlp.3.weight"),
        ("early_stage_action_mlp.3.bias", "action_mlp.3.bias"),
        ("early_stage_action_mlp.6.weight", "action_mlp.6.weight"),
        ("early_stage_action_mlp.6.bias", "action_mlp.6.bias"),
        ("early_stage_action_head.weight", "action_head.weight"),
        ("early_stage_action_head.bias", "action_head.bias"),
    ]:
        if dst_key not in checkpoint_keys and src_key in checkpoint_keys:
            state_dict[dst_key] = state_dict[src_key].clone()
            copied_pairs.append((dst_key, src_key))

    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    missing = [key for key in missing if key not in {dst for dst, _ in copied_pairs}]
    model_keys = set(model.state_dict().keys())
    unexpected = [
        key
        for key in unexpected
        if not (key.startswith("early_stage_") and key not in model_keys)
    ]

    if missing or unexpected:
        raise RuntimeError(
            "Incompatible checkpoint load for continuation: "
            f"missing={missing}, unexpected={unexpected}, copied_pairs={copied_pairs}"
        )
    return {
        "path": str(checkpoint_path),
        "copied_pairs": [{"dst": dst, "src": src} for dst, src in copied_pairs],
        "strict": False,
    }


def _life_support_clean(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    life_support = out.get("life_support")
    if isinstance(life_support, dict) and str(life_support.get("profile")) == "custom_direct_edit":
        out["life_support"] = {k: v for k, v in life_support.items() if k != "profile"}
    return out


def load_train_full_cases(
    *,
    source_root: Path,
    cache_dir: Path,
    max_cases: int,
    cache_version: str,
) -> Tuple[List[CaseRecord], Dict[str, Any], Path]:
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])

    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    payload = _life_support_clean(payload)

    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(payload)
    cfg.training.enable_eval = False
    cfg.training.train_only = True
    cfg.training.enable_wandb = False

    cfg.data.skip_lmdb = False
    cfg.data.max_samples = int(max_cases)
    cfg.data.cache_version = str(cache_version)
    cfg.data.rebuild_cache = False
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False

    cfg.paths.cache_dir = str(cache_dir)
    train_loader, _, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        eval_batch_size=1,
        skip_lmdb=False,
        train_only=True,
    )

    dataset = train_loader.dataset
    assets = collect_dataset_assets(dataset)
    if assets.get("topology") is None:
        assets["topology"] = HydraulicTopology(cfg.paths.foundation_path)

    cases: List[CaseRecord] = []
    for dataset_idx in range(len(dataset)):
        data = dataset[dataset_idx]
        case_id, scenario_id, part_id = colon_case_id_from_data(data, "train", dataset_idx)
        cases.append(
            CaseRecord(
                case_id=str(case_id),
                scenario_id=int(scenario_id),
                part_id=int(part_id),
                dataset_index=int(dataset_idx),
                data=deepcopy(data),
            )
        )
    return cases, assets, cfg_path


def _safe_margin_top1_top2(probs: torch.Tensor, mask: torch.Tensor) -> float:
    vals = probs.view(-1).float().cpu()
    mask = mask.view(-1).bool().cpu()
    valid = vals[mask]
    if valid.numel() <= 0:
        return 0.0
    if valid.numel() == 1:
        return float(valid[0].item())
    top2 = torch.topk(valid, k=2).values
    return float(top2[0].item() - top2[1].item())


def _normalize_distance(distance: torch.Tensor, max_time: float) -> torch.Tensor:
    out = torch.ones_like(distance, dtype=torch.float32)
    finite = torch.isfinite(distance)
    if bool(finite.any()):
        out[finite] = (distance[finite].float() / max(float(max_time), 1e-6)).clamp(0.0, 1.0)
    return out


def _extract_history_node_locals(records: Sequence[Any], num_nodes: int) -> List[int]:
    seen = set()
    out: List[int] = []
    for row in records:
        idx = int(getattr(row, "node_local_idx", -1))
        if idx < 0 or idx >= int(num_nodes) or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def compute_teacher_belief(
    *,
    family: str,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    onset_offsets_min: Sequence[float],
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
) -> Dict[str, Any]:
    if family == "hsr_paper_topk_ema_v1":
        return _paper_topk_ema_posterior(
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=list(onset_offsets_min),
            alpha=float(paper_like_alpha),
            topk_fraction=float(paper_like_topk_fraction),
            time_tol_min=float(paper_like_time_tol_min),
        )
    if family == "hsr_soft_scenario_posterior_v3":
        return _soft_scenario_posterior(
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=list(onset_offsets_min),
            alpha=float(paper_like_alpha),
            time_tol_min=float(paper_like_time_tol_min),
            beta=float(soft_scenario_beta),
        )
    if family in EXTENDED_SOFT_SCENARIO_FAMILIES:
        return _soft_scenario_posterior(
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=list(onset_offsets_min),
            alpha=float(paper_like_alpha),
            time_tol_min=float(paper_like_time_tol_min),
            beta=float(soft_scenario_beta),
        )
    if family == "hsr_soft_scenario_posterior_v6":
        return _soft_scenario_posterior_v6(
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=list(onset_offsets_min),
            alpha=float(paper_like_alpha),
            time_tol_min=float(paper_like_time_tol_min),
            beta=float(soft_scenario_beta),
        )
    raise ValueError(f"Unsupported teacher family: {family}")


def resolve_onset_grid(*, family: str, episode_duration_min: float) -> List[float]:
    delta = float(episode_duration_min)
    match = re.fullmatch(r"hsr_soft_scenario_posterior_v7_(\d+)offset", str(family))
    if match is not None:
        count = int(match.group(1))
        if count >= 5 and count % 2 == 1:
            half = count // 2
            return [float(k) * delta for k in range(-half, half + 1)]
    return [-1.0 * delta, 0.0, 1.0 * delta]


def build_spim_native_state(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    belief_ctx: Dict[str, Any],
    trigger_global: Optional[int],
    gate: DynamicReachabilityRuleModule,
    num_rounds: int,
    action_budget: int,
    episode_duration_min: float,
    top_source_k: int,
    include_surrogate_features: bool,
    include_uncertainty_regime_features: bool,
    source_local: Optional[int],
    prev_entropy: Optional[float] = None,
    prev2_entropy: Optional[float] = None,
    prev_top1_mass: Optional[float] = None,
    prev2_top1_mass: Optional[float] = None,
) -> Dict[str, Any]:
    belief = belief_ctx["belief"].view(-1).float().cpu()
    candidate_mask = belief_ctx["candidate_mask"].view(-1).bool().cpu()
    sampled_mask = rollout.revealed_mask.view(-1).bool().cpu()
    available_mask = candidate_mask & (~sampled_mask)
    if not bool(available_mask.any()):
        available_mask = candidate_mask & (~sampled_mask)

    num_nodes = int(rollout.num_nodes)
    info = state["info"]
    round_index = int(info["episode"]) + 1
    total_budget = int(num_rounds) * int(action_budget)
    used_budget = int(rollout.revealed_mask.sum().item())
    remaining_budget = max(int(total_budget) - int(used_budget), 0)

    metrics = _belief_metrics(belief, candidate_mask, source_local, threshold=0.7)
    top1_top2_margin = _safe_margin_top1_top2(belief, candidate_mask)

    positive_count = len(history.positive_records())
    negative_count = len(history.safe_records())

    candidate_count = int(available_mask.sum().item())
    candidate_ratio = float(candidate_count / max(num_nodes, 1))
    entropy = float(metrics["entropy"])
    entropy_norm = float(entropy / math.log(max(candidate_count, 2))) if candidate_count > 1 else 0.0
    top1_mass = float(metrics["top1_mass"])

    elapsed_time = float(info["time_min"])
    horizon_time = float(num_rounds) * float(episode_duration_min)

    ordered_candidates = [int(v) for v in metrics["ordered_candidates"]]
    rank_percentile = torch.zeros(num_nodes, dtype=torch.float32)
    denom = max(len(ordered_candidates) - 1, 1)
    for rank, local_idx in enumerate(ordered_candidates):
        rank_percentile[int(local_idx)] = float(1.0 - (float(rank) / float(denom)))

    trigger_local: Optional[int] = None
    if trigger_global is not None:
        gids = rollout.g_ids.detach().cpu().view(-1)
        matches = (gids == int(trigger_global)).nonzero(as_tuple=True)[0]
        if matches.numel() > 0:
            trigger_local = int(matches[0].item())

    positive_locals = _extract_history_node_locals(history.positive_records(), num_nodes)
    negative_locals = _extract_history_node_locals(history.safe_records(), num_nodes)

    top_source_locals = []
    for idx in ordered_candidates:
        if bool(candidate_mask[int(idx)].item()):
            top_source_locals.append(int(idx))
        if len(top_source_locals) >= int(max(top_source_k, 1)):
            break

    seed_union: List[int] = []
    for bucket in [
        ([] if trigger_local is None else [int(trigger_local)]),
        positive_locals,
        negative_locals,
        top_source_locals,
    ]:
        for idx in bucket:
            if idx not in seed_union:
                seed_union.append(int(idx))

    dist_matrix = None
    seed_to_col: Dict[int, int] = {}
    if seed_union:
        seed_tensor = torch.tensor(seed_union, dtype=torch.long, device=state["edge_index"].device)
        dist_matrix = gate.compute_distance_matrix(
            seed_indices=seed_tensor,
            physics_context=state["phys_ctx"],
            num_nodes=num_nodes,
        ).detach().cpu().float()
        seed_to_col = {int(seed): int(col) for col, seed in enumerate(seed_union)}

    inf_vec = torch.full((num_nodes,), float("inf"), dtype=torch.float32)

    def _col_or_inf(seed: Optional[int]) -> torch.Tensor:
        if dist_matrix is None or seed is None or int(seed) not in seed_to_col:
            return inf_vec.clone()
        return dist_matrix[:, int(seed_to_col[int(seed)])].clone()

    dist_to_trigger = _col_or_inf(trigger_local)

    if positive_locals and dist_matrix is not None:
        cols = [seed_to_col[int(idx)] for idx in positive_locals if int(idx) in seed_to_col]
        if cols:
            dist_to_positive = torch.min(dist_matrix[:, cols], dim=1).values
        else:
            dist_to_positive = inf_vec.clone()
    else:
        dist_to_positive = inf_vec.clone()

    if negative_locals and dist_matrix is not None:
        cols = [seed_to_col[int(idx)] for idx in negative_locals if int(idx) in seed_to_col]
        if cols:
            dist_to_negative = torch.min(dist_matrix[:, cols], dim=1).values
        else:
            dist_to_negative = inf_vec.clone()
    else:
        dist_to_negative = inf_vec.clone()

    if top_source_locals and dist_matrix is not None:
        cols = [seed_to_col[int(idx)] for idx in top_source_locals if int(idx) in seed_to_col]
        source_weights = belief[torch.tensor(top_source_locals, dtype=torch.long)].clamp_min(0.0)
        if float(source_weights.sum().item()) <= 1e-12:
            source_weights = torch.ones_like(source_weights) / max(int(source_weights.numel()), 1)
        else:
            source_weights = source_weights / source_weights.sum().clamp_min(1e-12)
        dist_to_top = dist_matrix[:, cols]
        arrive_prob = torch.sigmoid((float(elapsed_time) - dist_to_top) / max(float(episode_duration_min), 1e-6))
        expected_positive = (arrive_prob * source_weights.view(1, -1)).sum(dim=1).clamp(0.0, 1.0)
    else:
        expected_positive = belief.clone().clamp(0.0, 1.0)

    disagreement = torch.minimum(expected_positive, 1.0 - expected_positive)

    sorted_available = belief[available_mask]
    if sorted_available.numel() > 0:
        sorted_available = torch.sort(sorted_available, descending=True).values
        top5_mass = float(sorted_available[: min(5, int(sorted_available.numel()))].sum().item())
        inv_sq_sum = float(torch.sum(sorted_available * sorted_available).item())
        eff_support = float(1.0 / max(inv_sq_sum, 1e-12))
        eff_support_ratio = float(eff_support / max(float(candidate_count), 1.0))
        cum = torch.cumsum(sorted_available, dim=0)
        cover_idx = int((cum >= 0.9).nonzero(as_tuple=True)[0][0].item()) + 1 if bool((cum >= 0.9).any()) else int(cum.numel())
        high_quality_cover_ratio = float(cover_idx / max(int(sorted_available.numel()), 1))
    else:
        top5_mass = 0.0
        eff_support_ratio = 0.0
        high_quality_cover_ratio = 0.0

    entropy_delta_prev1 = float(entropy - float(prev_entropy)) if prev_entropy is not None else 0.0
    entropy_delta_prev2 = float(entropy - float(prev2_entropy)) if prev2_entropy is not None else 0.0
    top1_mass_delta_prev1 = float(top1_mass - float(prev_top1_mass)) if prev_top1_mass is not None else 0.0
    top1_mass_delta_prev2 = float(top1_mass - float(prev2_top1_mass)) if prev2_top1_mass is not None else 0.0

    top1_gap = (top1_mass - belief).clamp_min(0.0)
    local_sorted = torch.argsort(belief, descending=True)
    cum_mass_rank_ratio = torch.zeros(num_nodes, dtype=torch.float32)
    cover_0p9_member = torch.zeros(num_nodes, dtype=torch.float32)
    if int(local_sorted.numel()) > 0:
        sorted_belief = belief[local_sorted]
        cumsum = torch.cumsum(sorted_belief, dim=0)
        cum_mass_rank_ratio[local_sorted] = cumsum
        cover_cut = int((cumsum >= 0.9).nonzero(as_tuple=True)[0][0].item()) if bool((cumsum >= 0.9).any()) else int(cumsum.numel() - 1)
        cover_0p9_member[local_sorted[: cover_cut + 1]] = 1.0
    one_step_entropy_drop_estimate = (disagreement * (1.0 - rank_percentile) * float(entropy_norm)).clamp(0.0, 1.0)
    witness_distance_mean_norm = (
        (_normalize_distance(dist_to_trigger, horizon_time) + _normalize_distance(dist_to_positive, horizon_time) + _normalize_distance(dist_to_negative, horizon_time))
        / 3.0
    )
    novelty_redundancy_score = (
        0.5 * _normalize_distance(dist_to_positive, horizon_time)
        + 0.5 * _normalize_distance(dist_to_negative, horizon_time)
        - 0.25 * _normalize_distance(dist_to_trigger, horizon_time)
    ).clamp(0.0, 1.0)

    local_features = [
        belief,
        rank_percentile,
        expected_positive,
        disagreement,
        _normalize_distance(dist_to_trigger, horizon_time),
        _normalize_distance(dist_to_positive, horizon_time),
        _normalize_distance(dist_to_negative, horizon_time),
        available_mask.float(),
        sampled_mask.float(),
    ]

    if include_surrogate_features:
        cover_shrink_surr = expected_positive * (1.0 - belief)
        entropy_change_surr = disagreement * (1.0 - rank_percentile)
        local_features.extend([cover_shrink_surr, entropy_change_surr])
    if include_uncertainty_regime_features:
        local_features.extend(
            [
                top1_gap,
                cum_mass_rank_ratio,
                cover_0p9_member,
                one_step_entropy_drop_estimate,
                witness_distance_mean_norm,
                novelty_redundancy_score,
            ]
        )

    global_features = torch.tensor(
        [
            float(round_index / max(int(num_rounds), 1)),
            float(remaining_budget / max(int(total_budget), 1)),
            float(candidate_count / max(int(num_nodes), 1)),
            float(candidate_ratio),
            float(positive_count / max(int(total_budget), 1)),
            float(negative_count / max(int(total_budget), 1)),
            float(elapsed_time / max(float(horizon_time), 1e-6)),
            float(entropy_norm),
            float(metrics["mass_cover_size_ratio"]),
            float(top1_mass),
            float(metrics["top3_mass"]),
            float(top1_top2_margin),
        ],
        dtype=torch.float32,
    )
    if include_uncertainty_regime_features:
        global_features = torch.cat(
            [
                global_features,
                torch.tensor(
                    [
                        float(top5_mass),
                        float(eff_support_ratio),
                        float(high_quality_cover_ratio),
                        float(entropy_delta_prev1),
                        float(entropy_delta_prev2),
                        float(top1_mass_delta_prev1),
                        float(top1_mass_delta_prev2),
                    ],
                    dtype=torch.float32,
                ),
            ],
            dim=0,
        )

    return {
        "global_features": global_features,
        "local_features": torch.stack(local_features, dim=1).float(),
        "available_mask": available_mask.bool(),
        "graph_edge_index": state["edge_index"].detach().cpu().long(),
        "graph_evidence_nodes": sorted(
            set(
                ([] if trigger_local is None else [int(trigger_local)])
                + [int(v) for v in positive_locals]
                + [int(v) for v in negative_locals]
                + [int(v) for v in top_source_locals]
            )
        ),
        "diagnostics": {
            "round_index": int(round_index),
            "remaining_budget": int(remaining_budget),
            "candidate_count": int(candidate_count),
            "candidate_ratio": float(candidate_ratio),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "elapsed_time_since_trigger": float(elapsed_time),
            "posterior_entropy": float(entropy),
            "mass_cover_0p7": float(metrics["mass_cover_size_ratio"]),
            "top1_mass": float(top1_mass),
            "top3_mass": float(metrics["top3_mass"]),
            "top5_mass": float(top5_mass),
            "top1_top2_margin": float(top1_top2_margin),
            "effective_support_size_ratio": float(eff_support_ratio),
            "high_quality_cover_0p9_ratio": float(high_quality_cover_ratio),
            "entropy_delta_prev1": float(entropy_delta_prev1),
            "entropy_delta_prev2": float(entropy_delta_prev2),
            "top1_mass_delta_prev1": float(top1_mass_delta_prev1),
            "top1_mass_delta_prev2": float(top1_mass_delta_prev2),
        },
    }


def build_controlled_slate_mask(
    *,
    spim_state: Dict[str, Any],
    belief_ctx: Dict[str, Any],
    slate_size: int,
    top_posterior_k: int,
    high_disagreement_k: int,
    novelty_k: int,
    round_index: int = 1,
    early_stage_round_cutoff: int = 0,
    early_stage_top_posterior_k: Optional[int] = None,
    early_stage_high_disagreement_k: Optional[int] = None,
    early_stage_novelty_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a bounded candidate slate for set-level policy decisions.

    Slate uses belief-side statistics for proposal only; reward remains terminal-task aligned.
    """
    round_index = int(round_index)
    early_stage_active = bool(int(early_stage_round_cutoff) > 0 and round_index <= int(early_stage_round_cutoff))
    eff_top_posterior_k = int(early_stage_top_posterior_k) if early_stage_active and early_stage_top_posterior_k is not None else int(top_posterior_k)
    eff_high_disagreement_k = (
        int(early_stage_high_disagreement_k)
        if early_stage_active and early_stage_high_disagreement_k is not None
        else int(high_disagreement_k)
    )
    eff_novelty_k = int(early_stage_novelty_k) if early_stage_active and early_stage_novelty_k is not None else int(novelty_k)

    available_mask = spim_state["available_mask"].view(-1).bool().cpu()
    available_idx = torch.nonzero(available_mask, as_tuple=True)[0]
    requested = max(int(slate_size), 1)
    if int(available_idx.numel()) <= requested:
        return {
            "slate_mask": available_mask.clone(),
            "slate_indices": [int(v) for v in available_idx.tolist()],
            "diagnostics": {
                "slate_requested": int(requested),
                "slate_size": int(available_idx.numel()),
                "available_size": int(available_idx.numel()),
                "posterior_take": int(min(int(eff_top_posterior_k), int(available_idx.numel()))),
                "disagreement_take": 0,
                "novelty_take": 0,
                "fill_take": 0,
                "round_index": int(round_index),
                "early_stage_slate_active": float(early_stage_active),
                "effective_top_posterior_k": int(eff_top_posterior_k),
                "effective_high_disagreement_k": int(eff_high_disagreement_k),
                "effective_novelty_k": int(eff_novelty_k),
            },
        }

    local_features = spim_state["local_features"].detach().cpu().float()
    posterior = belief_ctx["belief"].view(-1).float().cpu()
    disagreement = local_features[:, 3]
    novelty = 0.5 * local_features[:, 4] + 0.5 * local_features[:, 5]
    available_set = set(int(v) for v in available_idx.tolist())
    selected: List[int] = []
    selected_set: set[int] = set()

    def _pick_from_scores(score: torch.Tensor, take_k: int) -> int:
        take = max(int(take_k), 0)
        if take <= 0:
            return 0
        cand = [int(v) for v in available_idx.tolist() if int(v) not in selected_set]
        if not cand:
            return 0
        cand_tensor = torch.tensor(cand, dtype=torch.long)
        order = torch.argsort(score[cand_tensor], descending=True)
        added = 0
        for pos in order.tolist():
            idx = int(cand[pos])
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            added += 1
            if len(selected) >= requested or added >= take:
                break
        return added

    posterior_take = _pick_from_scores(posterior, int(eff_top_posterior_k))
    disagreement_take = _pick_from_scores(disagreement, int(eff_high_disagreement_k))
    novelty_take = _pick_from_scores(novelty, int(eff_novelty_k))

    fill_take = 0
    if len(selected) < requested:
        cand = [int(v) for v in available_idx.tolist() if int(v) not in selected_set]
        if cand:
            cand_tensor = torch.tensor(cand, dtype=torch.long)
            order = torch.argsort(posterior[cand_tensor], descending=True)
            for pos in order.tolist():
                idx = int(cand[pos])
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                fill_take += 1
                if len(selected) >= requested:
                    break

    # Safety: keep only legal unsampled available candidates.
    selected = [int(v) for v in selected if int(v) in available_set][:requested]
    slate_mask = torch.zeros_like(available_mask, dtype=torch.bool)
    if selected:
        slate_mask[torch.tensor(selected, dtype=torch.long)] = True
    else:
        slate_mask = available_mask.clone()
        selected = [int(v) for v in available_idx.tolist()]

    return {
        "slate_mask": slate_mask,
        "slate_indices": selected,
        "diagnostics": {
            "slate_requested": int(requested),
            "slate_size": int(len(selected)),
            "available_size": int(available_idx.numel()),
            "posterior_take": int(posterior_take),
            "disagreement_take": int(disagreement_take),
            "novelty_take": int(novelty_take),
            "fill_take": int(fill_take),
            "round_index": int(round_index),
            "early_stage_slate_active": float(early_stage_active),
            "effective_top_posterior_k": int(eff_top_posterior_k),
            "effective_high_disagreement_k": int(eff_high_disagreement_k),
            "effective_novelty_k": int(eff_novelty_k),
        },
    }


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(int(dim), int(dim))
        self.fc2 = nn.Linear(int(dim), int(dim))
        self.norm1 = nn.LayerNorm(int(dim))
        self.norm2 = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.gelu(self.fc1(x)))
        h = self.drop(h)
        h = self.norm2(F.gelu(self.fc2(h)))
        h = self.drop(h)
        return x + h


class _GraphSAGELiteLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(int(dim), int(dim))
        self.nei_lin = nn.Linear(int(dim), int(dim))
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() <= 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(h)
        deg = torch.zeros((h.size(0), 1), dtype=h.dtype, device=h.device)
        agg.index_add_(0, dst, h[src])
        deg.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=h.dtype, device=h.device))
        agg = agg / deg.clamp_min(1.0)
        out = self.self_lin(h) + self.nei_lin(agg)
        out = self.drop(self.norm(F.gelu(out)))
        return h + out


class _GATLiteLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.heads = max(1, int(heads))
        self.dim_head = max(1, self.dim // self.heads)
        self.total_dim = self.heads * self.dim_head
        self.lin = nn.Linear(self.dim, self.total_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(self.heads, self.dim_head))
        self.a_dst = nn.Parameter(torch.zeros(self.heads, self.dim_head))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.out = nn.Linear(self.total_dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.feat_drop = nn.Dropout(float(dropout))
        self.attn_drop = nn.Dropout(float(dropout))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() <= 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        z = self.feat_drop(self.lin(h)).view(-1, self.heads, self.dim_head)
        e = F.elu((z[src] * self.a_src).sum(dim=-1) + (z[dst] * self.a_dst).sum(dim=-1))
        alpha = torch.zeros_like(e)
        for node in torch.unique(dst):
            mask = dst == node
            alpha[mask] = torch.softmax(e[mask], dim=0)
        alpha = self.attn_drop(alpha)
        out = torch.zeros((h.size(0), self.heads, self.dim_head), dtype=h.dtype, device=h.device)
        msg = z[src] * alpha.unsqueeze(-1)
        out.index_add_(0, dst, msg)
        out = self.out(out.reshape(h.size(0), self.total_dim))
        out = self.norm(F.gelu(out))
        return h + out


class _CNNResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = max(0, int(kernel_size) // 2)
        self.conv1 = nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad)
        self.conv2 = nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.conv1(x)))
        h = self.drop(F.gelu(self.conv2(h)))
        return x + h


class SpimNativePolicy(nn.Module):
    def __init__(
        self,
        global_dim: int,
        local_dim: int,
        hidden_dim: int = 128,
        policy_arch: str = "separate_heads",
        policy_mlp_depth: int = 2,
        value_mlp_depth: int = 2,
        value_head_width_mult: float = 1.0,
        critic_trunk_depth: int = 0,
        critic_trunk_hidden_dim: int = 0,
        policy_dropout: float = 0.0,
        policy_norm: str = "none",
        candidate_encoder: str = "none",
        candidate_attn_heads: int = 4,
        enable_regime_head: bool = False,
        regime_head_classes: int = 3,
        regime_embed_dim: int = 12,
        arch_backbone: str = "baseline_mlp",
        residual_hidden_dim: int = 256,
        residual_depth: int = 4,
        residual_head_dim: int = 128,
        transformer_token_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn_dim: int = 256,
        graph_hidden_dim: int = 128,
        graph_layers: int = 2,
        graph_heads: int = 4,
        graph_max_subgraph_nodes: int = 512,
        graph_use_onehop: bool = False,
        cnn_channels: int = 128,
        cnn_kernel_size: int = 3,
        cnn_norm: str = "layernorm",
        enable_early_stage_specialist_head: bool = False,
        early_stage_round_cutoff: int = 0,
    ):
        super().__init__()
        self.global_dim = int(global_dim)
        self.local_dim = int(local_dim)
        self.hidden_dim = int(hidden_dim)
        self.policy_arch = str(policy_arch)
        self.candidate_encoder = str(candidate_encoder)
        self.policy_norm = str(policy_norm)
        self.policy_dropout = max(float(policy_dropout), 0.0)
        self.policy_mlp_depth = max(int(policy_mlp_depth), 1)
        self.value_mlp_depth = max(int(value_mlp_depth), 1)
        self.value_hidden_dim = max(8, int(round(float(hidden_dim) * max(float(value_head_width_mult), 0.25))))
        self.critic_trunk_depth = max(int(critic_trunk_depth), 0)
        self.critic_trunk_hidden_dim = max(8, int(critic_trunk_hidden_dim)) if int(critic_trunk_hidden_dim) > 0 else self.value_hidden_dim
        self.enable_regime_head = bool(enable_regime_head)
        self.regime_head_classes = max(2, int(regime_head_classes))
        self.regime_embed_dim = max(4, int(regime_embed_dim))
        self.actor_global_dim = self.global_dim + (self.regime_embed_dim if self.enable_regime_head else 0)
        self.arch_backbone = str(arch_backbone)
        self.graph_use_onehop = bool(graph_use_onehop)
        self.graph_max_subgraph_nodes = max(32, int(graph_max_subgraph_nodes))
        self.cnn_norm = str(cnn_norm)
        self.enable_early_stage_specialist_head = bool(enable_early_stage_specialist_head)
        self.early_stage_round_cutoff = max(int(early_stage_round_cutoff), 0)
        self.regime_head = nn.Linear(self.global_dim, self.regime_head_classes) if self.enable_regime_head else None
        self.regime_embedding = nn.Embedding(self.regime_head_classes, self.regime_embed_dim) if self.enable_regime_head else None

        # Keep exact baseline parameter names for strict checkpoint compatibility.
        self.local_encoder = None
        self.global_to_local = None
        self.candidate_attention = None
        self.shared_trunk = None
        self.action_head = None
        self.value_head = None
        self.action_mlp = None
        self.early_stage_action_mlp = None
        self.value_mlp = None
        self.critic_trunk = None
        self.early_stage_action_head = None

        if self.arch_backbone == "baseline_mlp":
            local_action_dim = self.local_dim
            if self.candidate_encoder == "self_attention":
                self.local_encoder = nn.Linear(self.local_dim, self.hidden_dim)
                self.global_to_local = nn.Linear(self.actor_global_dim, self.hidden_dim)
                attn_heads = max(1, int(candidate_attn_heads))
                while self.hidden_dim % attn_heads != 0 and attn_heads > 1:
                    attn_heads -= 1
                self.candidate_attention = nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=attn_heads,
                    dropout=self.policy_dropout,
                    batch_first=True,
                )
                local_action_dim = self.hidden_dim

            action_in_dim = self.actor_global_dim + local_action_dim
            if self.policy_arch == "shared_trunk":
                self.shared_trunk = self._build_mlp(
                    in_dim=action_in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    depth=self.policy_mlp_depth,
                    dropout=self.policy_dropout,
                    norm=self.policy_norm,
                )
                self.action_head = nn.Linear(self.hidden_dim, 1)
                if self.enable_early_stage_specialist_head:
                    self.early_stage_action_head = nn.Linear(self.hidden_dim, 1)
                self.value_head = self._build_mlp(
                    in_dim=self.hidden_dim,
                    hidden_dim=self.value_hidden_dim,
                    out_dim=1,
                    depth=self.value_mlp_depth,
                    dropout=self.policy_dropout,
                    norm=self.policy_norm,
                )
            else:
                self.action_mlp = self._build_mlp(
                    in_dim=action_in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=1,
                    depth=self.policy_mlp_depth,
                    dropout=self.policy_dropout,
                    norm=self.policy_norm,
                )
                if self.enable_early_stage_specialist_head:
                    self.early_stage_action_mlp = self._build_mlp(
                        in_dim=action_in_dim,
                        hidden_dim=self.hidden_dim,
                        out_dim=1,
                        depth=self.policy_mlp_depth,
                        dropout=self.policy_dropout,
                        norm=self.policy_norm,
                    )
                if self.critic_trunk_depth > 0:
                    self.critic_trunk = self._build_mlp(
                        in_dim=self.actor_global_dim + local_action_dim,
                        hidden_dim=self.critic_trunk_hidden_dim,
                        out_dim=self.critic_trunk_hidden_dim,
                        depth=self.critic_trunk_depth,
                        dropout=self.policy_dropout,
                        norm=self.policy_norm,
                    )
                self.value_mlp = self._build_mlp(
                    in_dim=self.critic_trunk_hidden_dim if self.critic_trunk is not None else self.actor_global_dim,
                    hidden_dim=self.value_hidden_dim,
                    out_dim=1,
                    depth=self.value_mlp_depth,
                    dropout=self.policy_dropout,
                    norm=self.policy_norm,
                )
        elif self.arch_backbone == "residual_mlp_control":
            if self.enable_early_stage_specialist_head:
                raise ValueError("early-stage specialist head is only supported for baseline_mlp backbone")
            d_h = int(residual_hidden_dim)
            d_head = int(residual_head_dim)
            self.res_in = nn.Linear(self.actor_global_dim + self.local_dim, d_h)
            n_blocks = max(1, int(residual_depth) // 2)
            self.res_blocks = nn.ModuleList([_ResidualBlock(d_h, self.policy_dropout) for _ in range(n_blocks)])
            self.res_actor_head = nn.Sequential(
                nn.Linear(d_h, d_head),
                nn.GELU(),
                nn.Linear(d_head, d_head),
                nn.GELU(),
                nn.Linear(d_head, 1),
            )
            self.res_value_head = nn.Sequential(
                nn.Linear(self.actor_global_dim + d_h, d_head),
                nn.GELU(),
                nn.Linear(d_head, d_head),
                nn.GELU(),
                nn.Linear(d_head, 1),
            )
        elif self.arch_backbone == "slate_transformer_lite":
            if self.enable_early_stage_specialist_head:
                raise ValueError("early-stage specialist head is only supported for baseline_mlp backbone")
            d_t = int(transformer_token_dim)
            self.tr_local = nn.Linear(self.local_dim, d_t)
            self.tr_global = nn.Linear(self.actor_global_dim, d_t)
            self.tr_rank = nn.Embedding(64, d_t)
            layer = nn.TransformerEncoderLayer(
                d_model=d_t,
                nhead=max(1, int(transformer_heads)),
                dim_feedforward=int(transformer_ffn_dim),
                dropout=self.policy_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.tr_encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(transformer_layers)))
            self.tr_actor_head = nn.Sequential(nn.Linear(d_t, 128), nn.GELU(), nn.Linear(128, 1))
            self.tr_value_head = nn.Sequential(nn.Linear(2 * d_t, 128), nn.GELU(), nn.Linear(128, 1))
        elif self.arch_backbone == "graphsage_lite":
            if self.enable_early_stage_specialist_head:
                raise ValueError("early-stage specialist head is only supported for baseline_mlp backbone")
            d_g = int(graph_hidden_dim)
            self.gs_local = nn.Linear(self.local_dim, d_g)
            self.gs_layers = nn.ModuleList([_GraphSAGELiteLayer(d_g, self.policy_dropout) for _ in range(max(1, int(graph_layers)))])
            self.gs_actor_head = nn.Sequential(nn.Linear(d_g, 128), nn.GELU(), nn.Linear(128, 1))
            self.gs_value_head = nn.Sequential(nn.Linear(self.actor_global_dim + d_g, 128), nn.GELU(), nn.Linear(128, 1))
        elif self.arch_backbone == "gat_lite":
            if self.enable_early_stage_specialist_head:
                raise ValueError("early-stage specialist head is only supported for baseline_mlp backbone")
            d_g = int(graph_hidden_dim)
            self.gat_local = nn.Linear(self.local_dim, d_g)
            self.gat_layers = nn.ModuleList([_GATLiteLayer(d_g, int(graph_heads), self.policy_dropout) for _ in range(max(1, int(graph_layers)))])
            self.gat_actor_head = nn.Sequential(nn.Linear(d_g, 128), nn.GELU(), nn.Linear(128, 1))
            self.gat_value_head = nn.Sequential(nn.Linear(self.actor_global_dim + d_g, 128), nn.GELU(), nn.Linear(128, 1))
        elif self.arch_backbone == "cnn_lite":
            if self.enable_early_stage_specialist_head:
                raise ValueError("early-stage specialist head is only supported for baseline_mlp backbone")
            d_c = int(cnn_channels)
            self.cnn_local = nn.Linear(self.local_dim, d_c)
            self.cnn_blocks = nn.ModuleList([_CNNResBlock1d(d_c, int(cnn_kernel_size), self.policy_dropout) for _ in range(2)])
            self.cnn_ln = nn.LayerNorm(d_c)
            self.cnn_gn = nn.GroupNorm(8, d_c)
            self.cnn_actor_head = nn.Sequential(nn.Linear(d_c, 128), nn.GELU(), nn.Linear(128, 1))
            self.cnn_value_head = nn.Sequential(nn.Linear(self.actor_global_dim + d_c, 128), nn.GELU(), nn.Linear(128, 1))
        else:
            raise ValueError(f"Unsupported arch_backbone: {self.arch_backbone}")

    def _use_early_stage_specialist(self, round_index: Optional[int]) -> bool:
        return bool(
            self.enable_early_stage_specialist_head
            and self.early_stage_round_cutoff > 0
            and round_index is not None
            and int(round_index) <= self.early_stage_round_cutoff
        )

    @staticmethod
    def _make_norm(norm: str, dim: int) -> nn.Module:
        name = str(norm).lower()
        if name == "layernorm":
            return nn.LayerNorm(int(dim))
        if name == "rmsnorm":
            if hasattr(nn, "RMSNorm"):
                return nn.RMSNorm(int(dim))
            return nn.LayerNorm(int(dim))
        return nn.Identity()

    @classmethod
    def _build_mlp(
        cls,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        dropout: float,
        norm: str,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        d_in = int(in_dim)
        d_hidden = int(hidden_dim)
        for _ in range(max(int(depth), 1)):
            layers.append(nn.Linear(d_in, d_hidden))
            layers.append(cls._make_norm(norm, d_hidden))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            d_in = d_hidden
        layers.append(nn.Linear(d_in, int(out_dim)))
        return nn.Sequential(*layers)

    def _encode_local_features(self, global_features: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        if self.candidate_encoder != "self_attention":
            return local_features
        assert self.local_encoder is not None and self.global_to_local is not None and self.candidate_attention is not None
        h = self.local_encoder(local_features)
        g = self.global_to_local(global_features.view(1, -1)).expand(h.size(0), -1)
        h = h + g
        attn_out, _ = self.candidate_attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
        return attn_out.squeeze(0)

    def _compute_regime_condition(self, global_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not self.enable_regime_head:
            return global_features, {"enabled": False}
        assert self.regime_head is not None and self.regime_embedding is not None
        logits = self.regime_head(global_features.view(1, -1)).view(-1)
        probs = torch.softmax(logits, dim=0)
        emb = torch.matmul(probs.view(1, -1), self.regime_embedding.weight).view(-1)
        conditioned = torch.cat([global_features, emb], dim=0)
        regime_id = int(torch.argmax(probs).item())
        return conditioned, {
            "enabled": True,
            "regime_logits": logits,
            "regime_probs": probs,
            "regime_id": regime_id,
        }

    def _build_induced_subgraph(
        self,
        *,
        available_mask: torch.Tensor,
        graph_bundle: Optional[Dict[str, Any]],
        num_nodes: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if graph_bundle is None:
            return None
        edge_index = graph_bundle.get("edge_index", None)
        if edge_index is None:
            return None
        edge_index = edge_index.to(device=device, dtype=torch.long)
        seed = torch.nonzero(available_mask.view(-1).bool(), as_tuple=True)[0]
        ev = graph_bundle.get("evidence_nodes", [])
        if len(ev) > 0:
            ev_t = torch.tensor([int(v) for v in ev], dtype=torch.long, device=device)
            seed = torch.unique(torch.cat([seed, ev_t], dim=0))
        if seed.numel() <= 0:
            return None

        if self.graph_use_onehop:
            src, dst = edge_index[0], edge_index[1]
            keep = torch.zeros((num_nodes,), dtype=torch.bool, device=device)
            keep[seed] = True
            touch = keep[src] | keep[dst]
            onehop = torch.unique(torch.cat([seed, src[touch], dst[touch]], dim=0))
            seed = onehop
        if seed.numel() > int(self.graph_max_subgraph_nodes):
            seed = seed[: int(self.graph_max_subgraph_nodes)]
        seed = torch.unique(seed)

        map_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        map_idx[seed] = torch.arange(seed.numel(), dtype=torch.long, device=device)
        src, dst = edge_index[0], edge_index[1]
        src_sub = map_idx[src]
        dst_sub = map_idx[dst]
        mask = (src_sub >= 0) & (dst_sub >= 0)
        sub_edge = torch.stack([src_sub[mask], dst_sub[mask]], dim=0) if bool(mask.any()) else torch.zeros((2, 0), dtype=torch.long, device=device)
        return seed, sub_edge

    def _score_and_value(
        self,
        *,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
        graph_bundle: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        conditioned_global, regime_info = self._compute_regime_condition(global_features)
        if available_mask is None:
            available = torch.ones((local_features.size(0),), dtype=torch.bool, device=local_features.device)
        else:
            available = available_mask.view(-1).bool()
        avail_idx = torch.nonzero(available, as_tuple=True)[0]

        if self.arch_backbone == "baseline_mlp":
            encoded_local = self._encode_local_features(conditioned_global, local_features)
            g = conditioned_global.view(1, -1).expand(encoded_local.size(0), -1)
            x = torch.cat([g, encoded_local], dim=1)
            if self.policy_arch == "shared_trunk":
                assert self.shared_trunk is not None and self.action_head is not None and self.value_head is not None
                trunk = self.shared_trunk(x)
                if self._use_early_stage_specialist(round_index):
                    assert self.early_stage_action_head is not None
                    logits = self.early_stage_action_head(trunk).view(-1)
                else:
                    logits = self.action_head(trunk).view(-1)
                if avail_idx.numel() > 0:
                    pooled = trunk[avail_idx].mean(dim=0, keepdim=True)
                else:
                    pooled = trunk.mean(dim=0, keepdim=True) if trunk.size(0) > 0 else global_features.new_zeros(1, self.hidden_dim)
                value = self.value_head(pooled).view(())
                return logits, value, regime_info
            assert self.action_mlp is not None and self.value_mlp is not None
            if self._use_early_stage_specialist(round_index):
                assert self.early_stage_action_mlp is not None
                logits = self.early_stage_action_mlp(x).view(-1)
            else:
                logits = self.action_mlp(x).view(-1)
            if avail_idx.numel() > 0:
                pooled_local = encoded_local[avail_idx].mean(dim=0)
            else:
                pooled_local = encoded_local.mean(dim=0)
            critic_input = torch.cat([conditioned_global, pooled_local], dim=0).view(1, -1)
            if self.critic_trunk is not None:
                critic_input = self.critic_trunk(critic_input)
            value = self.value_mlp(critic_input if self.critic_trunk is not None else conditioned_global.view(1, -1)).view(())
            return logits, value, regime_info

        if self.arch_backbone == "residual_mlp_control":
            h = torch.cat([conditioned_global.view(1, -1).expand(local_features.size(0), -1), local_features], dim=1)
            h = self.res_in(h)
            for blk in self.res_blocks:
                h = blk(h)
            logits = self.res_actor_head(h).view(-1)
            pooled = h[avail_idx].mean(dim=0) if avail_idx.numel() > 0 else h.mean(dim=0)
            value = self.res_value_head(torch.cat([conditioned_global, pooled], dim=0).view(1, -1)).view(())
            return logits, value, regime_info

        if self.arch_backbone == "slate_transformer_lite":
            h = self.tr_local(local_features)
            if avail_idx.numel() > 0:
                cand = h[avail_idx]
                rank_bucket = (local_features[avail_idx, 1].clamp(0.0, 1.0) * 63.0).long()
                cand = cand + self.tr_rank(rank_bucket)
                gtok = self.tr_global(conditioned_global).view(1, -1)
                tokens = torch.cat([gtok, cand], dim=0).unsqueeze(0)
                out = self.tr_encoder(tokens).squeeze(0)
                g_out = out[0]
                h[avail_idx] = out[1:]
            else:
                g_out = self.tr_global(conditioned_global)
            logits = self.tr_actor_head(h).view(-1)
            pooled = h[avail_idx].mean(dim=0) if avail_idx.numel() > 0 else h.mean(dim=0)
            value = self.tr_value_head(torch.cat([g_out, pooled], dim=0).view(1, -1)).view(())
            return logits, value, regime_info

        if self.arch_backbone == "graphsage_lite":
            h = self.gs_local(local_features)
            induced = self._build_induced_subgraph(
                available_mask=available,
                graph_bundle=graph_bundle,
                num_nodes=local_features.size(0),
                device=local_features.device,
            )
            if induced is not None:
                sub_nodes, sub_edge = induced
                sub_h = h[sub_nodes]
                for layer in self.gs_layers:
                    sub_h = layer(sub_h, sub_edge)
                h[sub_nodes] = sub_h
            logits = self.gs_actor_head(h).view(-1)
            pooled = h[avail_idx].mean(dim=0) if avail_idx.numel() > 0 else h.mean(dim=0)
            value = self.gs_value_head(torch.cat([conditioned_global, pooled], dim=0).view(1, -1)).view(())
            return logits, value, regime_info

        if self.arch_backbone == "gat_lite":
            h = self.gat_local(local_features)
            induced = self._build_induced_subgraph(
                available_mask=available,
                graph_bundle=graph_bundle,
                num_nodes=local_features.size(0),
                device=local_features.device,
            )
            if induced is not None:
                sub_nodes, sub_edge = induced
                sub_h = h[sub_nodes]
                for layer in self.gat_layers:
                    sub_h = layer(sub_h, sub_edge)
                h[sub_nodes] = sub_h
            logits = self.gat_actor_head(h).view(-1)
            pooled = h[avail_idx].mean(dim=0) if avail_idx.numel() > 0 else h.mean(dim=0)
            value = self.gat_value_head(torch.cat([conditioned_global, pooled], dim=0).view(1, -1)).view(())
            return logits, value, regime_info

        if self.arch_backbone == "cnn_lite":
            h = self.cnn_local(local_features)
            if avail_idx.numel() > 0:
                sort_score = local_features[avail_idx, 1]
                order = torch.argsort(sort_score, descending=True)
                sorted_idx = avail_idx[order]
                seq = h[sorted_idx].transpose(0, 1).unsqueeze(0)
                for blk in self.cnn_blocks:
                    seq = blk(seq)
                seq = seq.squeeze(0).transpose(0, 1)
                if self.cnn_norm == "groupnorm":
                    seq = self.cnn_gn(seq.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
                else:
                    seq = self.cnn_ln(seq)
                h[sorted_idx] = seq
            logits = self.cnn_actor_head(h).view(-1)
            pooled = h[avail_idx].mean(dim=0) if avail_idx.numel() > 0 else h.mean(dim=0)
            value = self.cnn_value_head(torch.cat([conditioned_global, pooled], dim=0).view(1, -1)).view(())
            return logits, value, regime_info

        raise RuntimeError(f"Unsupported arch_backbone: {self.arch_backbone}")

    def score_actions(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
        graph_bundle: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
    ) -> torch.Tensor:
        logits, _ = self._score_and_value(
            global_features=global_features,
            local_features=local_features,
            available_mask=available_mask,
            graph_bundle=graph_bundle,
            round_index=round_index,
        )[:2]
        return logits

    def evaluate_actions(
        self,
        *,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        available_mask: torch.Tensor,
        selected_actions: Sequence[int],
        action_budget: int,
        graph_bundle: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        logits, value, regime_info = self._score_and_value(
            global_features=global_features,
            local_features=local_features,
            available_mask=available_mask,
            graph_bundle=graph_bundle,
            round_index=round_index,
        )
        available = available_mask.view(-1).bool().clone()
        log_prob = logits.new_tensor(0.0)
        entropy = logits.new_tensor(0.0)

        for slot_idx, action in enumerate(list(selected_actions)[: int(action_budget)]):
            candidate_idx = torch.nonzero(available, as_tuple=True)[0]
            if candidate_idx.numel() <= 0:
                break
            candidate_logits = logits[candidate_idx]
            probs = torch.softmax(candidate_logits, dim=0)
            action_pos = (candidate_idx == int(action)).nonzero(as_tuple=True)[0]
            if action_pos.numel() != 1:
                break
            pos = action_pos[0]
            log_prob = log_prob + torch.log(probs[pos].clamp_min(1e-12))
            entropy = entropy - (probs * torch.log(probs.clamp_min(1e-12))).sum()
            available[int(action)] = False

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
            "regime_id": regime_info.get("regime_id", None),
            "regime_probs": None
            if regime_info.get("regime_probs", None) is None
            else regime_info["regime_probs"].detach().cpu().tolist(),
        }

    def act(
        self,
        *,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        available_mask: torch.Tensor,
        action_budget: int,
        deterministic: bool,
        generator: Optional[torch.Generator],
        graph_bundle: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        logits, value, regime_info = self._score_and_value(
            global_features=global_features,
            local_features=local_features,
            available_mask=available_mask,
            graph_bundle=graph_bundle,
            round_index=round_index,
        )
        available = available_mask.view(-1).bool().clone()
        selected: List[int] = []
        log_prob = logits.new_tensor(0.0)
        entropy = logits.new_tensor(0.0)

        for _ in range(int(action_budget)):
            candidate_idx = torch.nonzero(available, as_tuple=True)[0]
            if candidate_idx.numel() <= 0:
                break
            candidate_logits = logits[candidate_idx]
            probs = torch.softmax(candidate_logits, dim=0)
            dist_entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
            entropy = entropy + dist_entropy

            if bool(deterministic):
                local_pos = int(torch.argmax(candidate_logits).item())
            else:
                sampled = torch.multinomial(
                    probs.detach().cpu(),
                    num_samples=1,
                    replacement=False,
                    generator=generator,
                )
                local_pos = int(sampled.view(-1)[0].item())

            chosen = int(candidate_idx[local_pos].item())
            selected.append(chosen)
            log_prob = log_prob + torch.log(probs[local_pos].clamp_min(1e-12))
            available[chosen] = False

        return {
            "actions": selected,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
            "regime_id": regime_info.get("regime_id", None),
            "regime_probs": None
            if regime_info.get("regime_probs", None) is None
            else regime_info["regime_probs"].detach().cpu().tolist(),
        }

    def evaluate_single_choice(
        self,
        *,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        candidate_indices: Sequence[int],
        selected_action: int,
        available_mask: Optional[torch.Tensor] = None,
        graph_bundle: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        logits, value, regime_info = self._score_and_value(
            global_features=global_features,
            local_features=local_features,
            available_mask=available_mask,
            graph_bundle=graph_bundle,
            round_index=round_index,
        )
        if len(candidate_indices) <= 0:
            zero = logits.new_tensor(0.0)
            return {
                "log_prob": zero,
                "entropy": zero,
                "value": value,
                "logits": logits,
                "regime_id": regime_info.get("regime_id", None),
                "regime_probs": None
                if regime_info.get("regime_probs", None) is None
                else regime_info["regime_probs"].detach().cpu().tolist(),
            }
        idx = torch.tensor([int(v) for v in candidate_indices], dtype=torch.long, device=logits.device)
        cand_logits = logits[idx]
        probs = torch.softmax(cand_logits, dim=0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        chosen_pos = (idx == int(selected_action)).nonzero(as_tuple=True)[0]
        if chosen_pos.numel() != 1:
            log_prob = logits.new_tensor(0.0)
        else:
            log_prob = torch.log(probs[chosen_pos[0]].clamp_min(1e-12))
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "logits": logits,
            "regime_id": regime_info.get("regime_id", None),
            "regime_probs": None
            if regime_info.get("regime_probs", None) is None
            else regime_info["regime_probs"].detach().cpu().tolist(),
        }


def _enumerate_beam_candidates(
    *,
    logits: torch.Tensor,
    available_mask: torch.Tensor,
    action_budget: int,
    beam_width: int,
    topk_per_slot: int,
) -> List[Dict[str, Any]]:
    beam: List[Tuple[List[int], float, torch.Tensor]] = [([], 0.0, available_mask.view(-1).bool().clone())]
    for _ in range(int(action_budget)):
        expanded: List[Tuple[List[int], float, torch.Tensor]] = []
        for prefix, score, avail in beam:
            idx = torch.nonzero(avail, as_tuple=True)[0]
            if idx.numel() <= 0:
                expanded.append((list(prefix), float(score), avail.clone()))
                continue
            cand_logits = logits[idx]
            k = min(int(max(1, topk_per_slot)), int(idx.numel()))
            top_pos = torch.topk(cand_logits, k=k, dim=0).indices
            probs = torch.softmax(cand_logits, dim=0)
            for pos in top_pos.tolist():
                chosen = int(idx[int(pos)].item())
                next_avail = avail.clone()
                next_avail[chosen] = False
                expanded.append(
                    (
                        list(prefix) + [chosen],
                        float(score + torch.log(probs[int(pos)].clamp_min(1e-12)).item()),
                        next_avail,
                    )
                )
        expanded.sort(key=lambda item: item[1], reverse=True)
        beam = expanded[: int(max(1, beam_width))]
    out: List[Dict[str, Any]] = []
    seen = set()
    for actions, seq_logprob, _ in beam:
        key = tuple(int(v) for v in actions)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"actions": list(key), "seq_logprob": float(seq_logprob), "origin": "beam"})
    return out


def _pack_preview_state(spim_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "global_features": spim_state["global_features"].detach().cpu(),
        "local_features": spim_state["local_features"].detach().cpu(),
        "available_mask": spim_state["available_mask"].detach().cpu(),
        "graph_edge_index": spim_state["graph_edge_index"].detach().cpu(),
        "graph_evidence_nodes": list(spim_state["graph_evidence_nodes"]),
        "diagnostics": {
            "posterior_entropy": float(spim_state["diagnostics"]["posterior_entropy"]),
            "top1_top2_margin": float(spim_state["diagnostics"]["top1_top2_margin"]),
            "top1_mass": float(spim_state["diagnostics"]["top1_mass"]),
        },
    }


def _preview_candidate_set(
    *,
    candidate: Dict[str, Any],
    case: CaseRecord,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    env: CleanTwoChannelEvidenceEnv,
    family: str,
    trigger_global: Optional[int],
    onset_grid: Sequence[float],
    gate: DynamicReachabilityRuleModule,
    runtime: Dict[str, Any],
    source_local: Optional[int],
    include_surrogate_features: bool,
    include_uncertainty_regime_features: bool,
    prev_entropy: Optional[float],
    prev2_entropy: Optional[float],
    prev_top1_mass: Optional[float],
    prev2_top1_mass: Optional[float],
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
    device: torch.device,
    model: SpimNativePolicy,
    top_source_k: int,
    logprob_weight: float,
    value_weight: float,
    store_preview_state: bool,
) -> Dict[str, Any]:
    preview_timing: Dict[str, float] = {}
    t0 = time.perf_counter()
    t = time.perf_counter()
    tmp_rollout = deepcopy(rollout)
    tmp_history = deepcopy(history)
    tmp_paper_state = PaperLikeHSRState(source_prior=None)
    preview_timing["deepcopy_s"] = float(time.perf_counter() - t)
    t = time.perf_counter()
    tmp_rollout.step_with_actions(
        list(candidate["actions"]),
        sample_types=[f"decode_consistency_slot_{i}" for i in range(len(candidate["actions"]))],
    )
    if tmp_rollout.history_steps:
        tmp_history.append_from_history_step(tmp_rollout.history_steps[-1])
    preview_timing["step_history_s"] = float(time.perf_counter() - t)
    t = time.perf_counter()
    next_state = make_rollout_state(
        case=case,
        rollout=tmp_rollout,
        history=tmp_history,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        num_episodes=int(runtime["num_episodes"]),
        action_budget=int(runtime["action_budget"]),
        frontier_role_mode=str(runtime["frontier_role_mode"]),
    )
    preview_timing["make_rollout_state_s"] = float(time.perf_counter() - t)
    t = time.perf_counter()
    next_belief = compute_teacher_belief(
        family=family,
        rollout=tmp_rollout,
        state=next_state,
        history=tmp_history,
        trigger_global=trigger_global,
        paper_state=tmp_paper_state,
        onset_offsets_min=onset_grid,
        paper_like_alpha=float(paper_like_alpha),
        paper_like_topk_fraction=float(paper_like_topk_fraction),
        paper_like_time_tol_min=float(paper_like_time_tol_min),
        soft_scenario_beta=float(soft_scenario_beta),
    )
    preview_timing["teacher_belief_s"] = float(time.perf_counter() - t)
    t = time.perf_counter()
    next_spim_state = build_spim_native_state(
        rollout=tmp_rollout,
        state=next_state,
        history=tmp_history,
        belief_ctx=next_belief,
        trigger_global=trigger_global,
        gate=gate,
        num_rounds=int(runtime["num_episodes"]),
        action_budget=int(runtime["action_budget"]),
        episode_duration_min=float(runtime["episode_duration_min"]),
        top_source_k=int(top_source_k),
        include_surrogate_features=bool(include_surrogate_features),
        include_uncertainty_regime_features=bool(include_uncertainty_regime_features),
        source_local=source_local,
        prev_entropy=prev_entropy,
        prev2_entropy=prev2_entropy,
        prev_top1_mass=prev_top1_mass,
        prev2_top1_mass=prev2_top1_mass,
    )
    preview_timing["spim_state_build_s"] = float(time.perf_counter() - t)
    t = time.perf_counter()
    with torch.no_grad():
        preview = model.act(
            global_features=next_spim_state["global_features"].to(device),
            local_features=next_spim_state["local_features"].to(device),
            available_mask=next_spim_state["available_mask"].to(device),
            action_budget=0,
            deterministic=True,
            generator=None,
            graph_bundle={
                "edge_index": next_spim_state["graph_edge_index"].to(device),
                "evidence_nodes": list(next_spim_state["graph_evidence_nodes"]),
            },
        )
    preview_timing["model_value_s"] = float(time.perf_counter() - t)
    post_value = float(preview["value"].detach().cpu().item())
    score = float(logprob_weight) * float(candidate["seq_logprob"]) + float(value_weight) * post_value
    preview_timing["total_s"] = float(time.perf_counter() - t0)
    return {
        **candidate,
        "post_value": float(post_value),
        "score": float(score),
        "preview_state": _pack_preview_state(next_spim_state) if bool(store_preview_state) else None,
        "preview_timing": preview_timing,
    }


def _compute_action_sequence_ce_loss(
    *,
    model: SpimNativePolicy,
    transition: Dict[str, Any],
    target_actions: Sequence[int],
    device: torch.device,
    action_budget: int,
) -> Tuple[Optional[torch.Tensor], float]:
    target_actions = [int(v) for v in list(target_actions)[: int(action_budget)]]
    if not target_actions:
        return None, 0.0

    global_features = transition["global_features"].to(device)
    local_features = transition["local_features"].to(device)
    available = transition["available_mask"].to(device).view(-1).bool().clone()
    graph_bundle = {
        "edge_index": transition["graph_edge_index"].to(device),
        "evidence_nodes": list(transition.get("graph_evidence_nodes", [])),
    }

    logits = model.score_actions(global_features, local_features, available_mask=available, graph_bundle=graph_bundle)
    losses: List[torch.Tensor] = []
    slot_match = 0.0
    slot_count = 0.0
    for action in target_actions:
        candidate_idx = torch.nonzero(available, as_tuple=True)[0]
        if candidate_idx.numel() <= 0:
            break
        candidate_logits = logits[candidate_idx]
        target = (candidate_idx == int(action)).nonzero(as_tuple=True)[0]
        if target.numel() != 1:
            break
        target_idx = target[0].view(1)
        losses.append(F.cross_entropy(candidate_logits.view(1, -1), target_idx))
        pred_idx = int(candidate_idx[int(torch.argmax(candidate_logits).item())].item())
        slot_match += float(pred_idx == int(action))
        slot_count += 1.0
        available[int(action)] = False
    if not losses:
        return None, 0.0
    loss = torch.stack(losses).mean()
    match_rate = float(slot_match / max(slot_count, 1.0))
    return loss, match_rate


def _evaluate_packed_state_value(
    *,
    model: SpimNativePolicy,
    packed_state: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    out = model.act(
        global_features=packed_state["global_features"].to(device),
        local_features=packed_state["local_features"].to(device),
        available_mask=packed_state["available_mask"].to(device),
        action_budget=0,
        deterministic=True,
        generator=None,
        round_index=int(packed_state["diagnostics"]["round_index"]),
        graph_bundle={
            "edge_index": packed_state["graph_edge_index"].to(device),
            "evidence_nodes": list(packed_state.get("graph_evidence_nodes", [])),
        },
    )
    return out["value"]


def _lin_anneal(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    alpha = float(step - 1) / float(total_steps - 1)
    return float((1.0 - alpha) * float(start) + alpha * float(end))


def _posterior_topk_unsampled(
    *,
    belief: torch.Tensor,
    candidate_mask: torch.Tensor,
    rollout: PracticalRollout,
    topk: int,
) -> List[int]:
    available = candidate_mask.view(-1).bool().cpu() & (~rollout.revealed_mask.view(-1).bool().cpu())
    idx = torch.nonzero(available, as_tuple=True)[0]
    if idx.numel() <= 0:
        return []
    values = belief.view(-1).float().cpu()[idx]
    k = min(int(max(1, topk)), int(idx.numel()))
    top_local = torch.topk(values, k=k, dim=0).indices
    return [int(idx[int(i)].item()) for i in top_local]


def _case_rollout_seed(base_seed: int, case_index: int, episode_index: int) -> int:
    return int(base_seed) * 1000003 + int(case_index) * 9973 + int(episode_index) * 97


def _clip_unit(value: float, clip_value: float) -> float:
    clip_v = max(float(clip_value), 1e-9)
    return float(max(min(float(value), clip_v), -clip_v) / clip_v)


def _mean_topk_scenario_error(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    belief_ctx: Dict[str, Any],
    onset_offsets_min: Sequence[float],
    topk_fraction: float,
    time_tol_min: float,
) -> float:
    base_mask = _build_clean_candidate_mask(rollout=rollout, state=state, trigger_global=trigger_global)
    candidate_mask = base_mask & belief_ctx["candidate_mask"].view(-1).bool().cpu()
    candidate_idx = torch.nonzero(candidate_mask, as_tuple=True)[0].long()
    if candidate_idx.numel() <= 0:
        return 0.0
    k = max(1, int(math.ceil(float(topk_fraction) * float(candidate_idx.numel()))))
    k = min(k, int(candidate_idx.numel()))
    belief_local = belief_ctx["belief"].view(-1).float().cpu()[candidate_idx]
    top_local = torch.topk(belief_local, k=k, dim=0).indices
    top_candidate_idx = candidate_idx[top_local]
    scenario_error = _compute_scenario_error(
        rollout=rollout,
        history=history,
        candidate_idx=top_candidate_idx,
        onset_offsets_min=[float(v) for v in onset_offsets_min],
        time_tol_min=float(time_tol_min),
    )
    if scenario_error.numel() <= 0:
        return 0.0
    per_source_error = scenario_error.float().mean(dim=1)
    return float(per_source_error.mean().item()) if per_source_error.numel() > 0 else 0.0


def _compute_reward_by_family(
    *,
    reward_family: str,
    hit_reward: float,
    step_penalty: float,
    selected_count: int,
    round_hit: bool,
    pre_cover_ratio: float,
    post_cover_ratio: float,
    pre_topk_error: float,
    post_topk_error: float,
    lambda_cover: float,
    lambda_error: float,
    cover_delta_clip: float,
    error_delta_clip: float,
) -> Dict[str, float]:
    base_reward = float(step_penalty) * float(selected_count)
    if bool(round_hit):
        base_reward += float(hit_reward)

    delta_cover = float(pre_cover_ratio - post_cover_ratio)
    delta_error = float(pre_topk_error - post_topk_error)
    delta_cover_norm = _clip_unit(delta_cover, float(cover_delta_clip))
    delta_error_norm = _clip_unit(delta_error, float(error_delta_clip))

    reward_cover = 0.0
    reward_error = 0.0
    if reward_family == "reward_r1_cover_shrink":
        reward_cover = float(lambda_cover) * float(delta_cover_norm)
    elif reward_family == "reward_r2_topk_scenario_error_improve":
        reward_error = float(lambda_error) * float(delta_error_norm)
    elif reward_family == "reward_r3_cover_plus_error":
        reward_cover = float(lambda_cover) * float(delta_cover_norm)
        reward_error = float(lambda_error) * float(delta_error_norm)
    elif reward_family != "reward_r0_terminal_step":
        raise ValueError(f"Unsupported reward family: {reward_family}")

    total_reward = float(base_reward + reward_cover + reward_error)
    return {
        "reward_total": float(total_reward),
        "reward_base": float(base_reward),
        "reward_cover_term": float(reward_cover),
        "reward_error_term": float(reward_error),
        "delta_cover_raw": float(delta_cover),
        "delta_cover_norm": float(delta_cover_norm),
        "delta_error_raw": float(delta_error),
        "delta_error_norm": float(delta_error_norm),
        "pre_cover_ratio": float(pre_cover_ratio),
        "post_cover_ratio": float(post_cover_ratio),
        "pre_topk_error": float(pre_topk_error),
        "post_topk_error": float(post_topk_error),
    }


def run_case_policy(
    *,
    case: CaseRecord,
    case_index: int,
    family: str,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    policy_name: str,
    model: Optional[SpimNativePolicy],
    deterministic: bool,
    base_seed: int,
    include_surrogate_features: bool,
    include_uncertainty_regime_features: bool,
    top_source_k: int,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
    hit_reward: float,
    step_penalty: float,
    reward_family: str,
    reward_lambda_cover: float,
    reward_lambda_error: float,
    reward_cover_delta_clip: float,
    reward_error_delta_clip: float,
    reward_topk_fraction: float,
    reward_time_tol_min: float,
    rl_policy_mode: str,
    corrective_candidate_topk: int,
    ambiguity_top1_top2_max: float,
    ambiguity_min_candidate_count: int,
    device: torch.device,
    collect_transitions: bool,
    slate_size: int = 10,
    slate_top_posterior_k: int = 6,
    slate_high_disagreement_k: int = 3,
    slate_novelty_k: int = 2,
    early_stage_round_cutoff: int = 0,
    early_stage_slate_top_posterior_k: Optional[int] = None,
    early_stage_slate_high_disagreement_k: Optional[int] = None,
    early_stage_slate_novelty_k: Optional[int] = None,
    decode_consistency_mode: str = "none",
    decode_consistency_beam_width: int = 4,
    decode_consistency_topk_per_slot: int = 4,
    decode_consistency_logprob_weight: float = 1.0,
    decode_consistency_value_weight: float = 4.0,
    decode_consistency_max_margin: float = 0.10,
    decode_consistency_max_round: int = 6,
    decode_consistency_min_entropy: float = 0.0,
    decode_consistency_max_preview_candidates: int = 0,
    decode_consistency_tracker: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rollout = PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=runtime["dataset_assets"]["global_edge_index"],
        stt_dynamic_series=runtime["dataset_assets"]["stt_dynamic_series"],
        num_global_nodes=int(runtime["dataset_assets"]["num_global_nodes"]),
        num_episodes=int(runtime["num_episodes"]),
        samples_per_episode=int(runtime["action_budget"]),
        episode_duration_min=float(runtime["episode_duration_min"]),
    )
    history = ObservationWitnessHistory()
    gate = DynamicReachabilityRuleModule()
    trigger_global = _extract_trigger_global(case.data)
    source_local = resolve_source_local_idx(rollout)
    source_global = None if source_local is None else int(rollout.g_ids[int(source_local)].item())

    onset_grid = resolve_onset_grid(
        family=str(family),
        episode_duration_min=float(runtime["episode_duration_min"]),
    )
    paper_state = PaperLikeHSRState(source_prior=None)

    step_rows: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []

    hit_round: Optional[int] = None
    hit_sample_index: Optional[int] = None
    budget_used = 0
    termination_reason = "budget_exhausted"
    prev_entropy: Optional[float] = None
    prev2_entropy: Optional[float] = None
    prev_top1_mass: Optional[float] = None
    prev2_top1_mass: Optional[float] = None
    case_decode_stats = {
        "triggered_states": 0,
        "rewrites": 0,
        "preview_candidates": 0,
        "greedy_model_s": 0.0,
        "beam_enum_s": 0.0,
        "preview_total_s": 0.0,
        "preview_deepcopy_s": 0.0,
        "preview_step_history_s": 0.0,
        "preview_make_rollout_state_s": 0.0,
        "preview_teacher_belief_s": 0.0,
        "preview_spim_state_build_s": 0.0,
        "preview_model_value_s": 0.0,
        "skipped_budget": 0,
    }

    for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
        state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=runtime["dataset_assets"]["topology"],
            num_episodes=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
            frontier_role_mode=str(runtime["frontier_role_mode"]),
        )

        if int(state["valid_mask"].sum().item()) <= 0:
            termination_reason = "no_valid_nodes"
            break

        belief_ctx = compute_teacher_belief(
            family=family,
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=onset_grid,
            paper_like_alpha=float(paper_like_alpha),
            paper_like_topk_fraction=float(paper_like_topk_fraction),
            paper_like_time_tol_min=float(paper_like_time_tol_min),
            soft_scenario_beta=float(soft_scenario_beta),
        )

        spim_state = build_spim_native_state(
            rollout=rollout,
            state=state,
            history=history,
            belief_ctx=belief_ctx,
            trigger_global=trigger_global,
            gate=gate,
            num_rounds=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
            episode_duration_min=float(runtime["episode_duration_min"]),
            top_source_k=int(top_source_k),
            include_surrogate_features=bool(include_surrogate_features),
            include_uncertainty_regime_features=bool(include_uncertainty_regime_features),
            source_local=source_local,
            prev_entropy=prev_entropy,
            prev2_entropy=prev2_entropy,
            prev_top1_mass=prev_top1_mass,
            prev2_top1_mass=prev2_top1_mass,
        )
        need_cover_term = str(reward_family) in {"reward_r1_cover_shrink", "reward_r3_cover_plus_error"}
        need_error_term = str(reward_family) in {"reward_r2_topk_scenario_error_improve", "reward_r3_cover_plus_error"}
        pre_cover_ratio = float(spim_state["diagnostics"]["mass_cover_0p7"])
        pre_topk_error = 0.0
        if need_error_term:
            pre_topk_error = _mean_topk_scenario_error(
                rollout=rollout,
                state=state,
                history=history,
                trigger_global=trigger_global,
                belief_ctx=belief_ctx,
                onset_offsets_min=onset_grid,
                topk_fraction=float(reward_topk_fraction),
                time_tol_min=float(reward_time_tol_min),
            )

        teacher_actions = _pick_topk_unsampled(
            belief_ctx["belief"],
            belief_ctx["candidate_mask"],
            rollout,
            int(runtime["action_budget"]),
        )
        teacher_actions = [int(v) for v in teacher_actions]
        if not teacher_actions:
            termination_reason = "teacher_no_action"
            break

        gate_triggered = False
        corrective_pool: List[int] = []
        corrective_replaced = False
        action_agreement_rate = 1.0
        policy_out = None
        slate_info = {
            "slate_requested": 0,
            "slate_size": int(spim_state["available_mask"].sum().item()),
            "available_size": int(spim_state["available_mask"].sum().item()),
            "posterior_take": 0,
            "disagreement_take": 0,
            "novelty_take": 0,
            "fill_take": 0,
        }
        policy_available_mask = spim_state["available_mask"]

        if policy_name == "teacher":
            selected_actions = list(teacher_actions)
        else:
            if model is None:
                raise ValueError("model is required for non-teacher policy")
            slate_out = build_controlled_slate_mask(
                spim_state=spim_state,
                belief_ctx=belief_ctx,
                slate_size=int(slate_size),
                top_posterior_k=int(slate_top_posterior_k),
                high_disagreement_k=int(slate_high_disagreement_k),
                novelty_k=int(slate_novelty_k),
                round_index=int(spim_state["diagnostics"]["round_index"]),
                early_stage_round_cutoff=int(early_stage_round_cutoff),
                early_stage_top_posterior_k=early_stage_slate_top_posterior_k,
                early_stage_high_disagreement_k=early_stage_slate_high_disagreement_k,
                early_stage_novelty_k=early_stage_slate_novelty_k,
            )
            policy_available_mask = slate_out["slate_mask"].bool()
            slate_info = dict(slate_out["diagnostics"])
            if str(rl_policy_mode) == "free":
                generator = None
                if not bool(deterministic):
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(_case_rollout_seed(base_seed, case_index, episode_idx))
                with torch.no_grad():
                    policy_out = model.act(
                        global_features=spim_state["global_features"].to(device),
                        local_features=spim_state["local_features"].to(device),
                        available_mask=policy_available_mask.to(device),
                        action_budget=int(runtime["action_budget"]),
                        deterministic=bool(deterministic),
                        generator=generator,
                        round_index=int(spim_state["diagnostics"]["round_index"]),
                        graph_bundle={
                            "edge_index": spim_state["graph_edge_index"].to(device),
                            "evidence_nodes": list(spim_state["graph_evidence_nodes"]),
                        },
                    )
                selected_actions = [int(v) for v in policy_out["actions"]]
                if not selected_actions:
                    termination_reason = "policy_no_action"
                    break
                if len(teacher_actions) > 0:
                    aligned_slots = min(len(selected_actions), len(teacher_actions))
                    if aligned_slots > 0:
                        same = sum(int(selected_actions[i] == teacher_actions[i]) for i in range(aligned_slots))
                        action_agreement_rate = float(same / float(aligned_slots))
            else:
                # conservative_corrective_rl_v1: first two actions fixed to teacher; RL can only replace slot-3 under ambiguity gate
                selected_actions = list(teacher_actions)
                action_budget = int(runtime["action_budget"])
                can_attempt = (
                    len(teacher_actions) >= 3
                    and int(action_budget) >= 3
                    and float(spim_state["diagnostics"]["candidate_count"]) >= float(ambiguity_min_candidate_count)
                    and float(spim_state["diagnostics"]["top1_top2_margin"]) <= float(ambiguity_top1_top2_max)
                )
                if can_attempt:
                    gate_triggered = True
                    candidate_pool = _posterior_topk_unsampled(
                        belief=belief_ctx["belief"],
                        candidate_mask=belief_ctx["candidate_mask"],
                        rollout=rollout,
                        topk=int(corrective_candidate_topk),
                    )
                    blocked = {int(teacher_actions[0]), int(teacher_actions[1])}
                    corrective_pool = [int(v) for v in candidate_pool if int(v) not in blocked]
                    if int(teacher_actions[2]) not in corrective_pool:
                        corrective_pool.append(int(teacher_actions[2]))
                    if corrective_pool:
                        with torch.no_grad():
                            logits = model.score_actions(
                                spim_state["global_features"].to(device),
                                spim_state["local_features"].to(device),
                                available_mask=policy_available_mask.to(device),
                                round_index=int(spim_state["diagnostics"]["round_index"]),
                                graph_bundle={
                                    "edge_index": spim_state["graph_edge_index"].to(device),
                                    "evidence_nodes": list(spim_state["graph_evidence_nodes"]),
                                },
                            )
                            pool_tensor = torch.tensor(corrective_pool, dtype=torch.long, device=device)
                            pool_logits = logits[pool_tensor]
                            pool_probs = torch.softmax(pool_logits, dim=0)
                            entropy = -(pool_probs * torch.log(pool_probs.clamp_min(1e-12))).sum()
                            if bool(deterministic):
                                chosen_pos = int(torch.argmax(pool_logits).item())
                            else:
                                gen = torch.Generator(device="cpu")
                                gen.manual_seed(_case_rollout_seed(base_seed, case_index, episode_idx))
                                sampled = torch.multinomial(
                                    pool_probs.detach().cpu(),
                                    num_samples=1,
                                    replacement=False,
                                    generator=gen,
                                )
                                chosen_pos = int(sampled.view(-1)[0].item())
                            chosen_action = int(corrective_pool[chosen_pos])
                            selected_actions[2] = int(chosen_action)
                            corrective_replaced = int(chosen_action) != int(teacher_actions[2])
                            eval_one = model.evaluate_single_choice(
                                global_features=spim_state["global_features"].to(device),
                                local_features=spim_state["local_features"].to(device),
                                candidate_indices=corrective_pool,
                                selected_action=chosen_action,
                                available_mask=policy_available_mask.to(device),
                                round_index=int(spim_state["diagnostics"]["round_index"]),
                                graph_bundle={
                                    "edge_index": spim_state["graph_edge_index"].to(device),
                                    "evidence_nodes": list(spim_state["graph_evidence_nodes"]),
                                },
                            )
                            log_prob = eval_one["log_prob"]
                            value = eval_one["value"]
                            policy_out = {
                                "actions": list(selected_actions),
                                "log_prob": log_prob,
                                "entropy": entropy,
                                "value": value,
                                "candidate_pool": list(corrective_pool),
                            }
                same = sum(int(a == b) for a, b in zip(selected_actions, teacher_actions[: len(selected_actions)]))
                action_agreement_rate = float(same / max(len(selected_actions), 1))

        selected_global_ids = [int(rollout.g_ids[int(v)].item()) for v in selected_actions]
        policy_slate_local_ids = [int(v) for v in torch.nonzero(policy_available_mask.view(-1).bool(), as_tuple=True)[0].tolist()]
        policy_slate_global_ids = [int(rollout.g_ids[int(v)].item()) for v in policy_slate_local_ids]
        regime_id = None if policy_out is None else policy_out.get("regime_id", None)
        regime_probs = None if policy_out is None else policy_out.get("regime_probs", None)
        decode_consistency_bundle: Optional[Dict[str, Any]] = None
        if (
            collect_transitions
            and model is not None
            and str(policy_name) != "teacher"
            and str(decode_consistency_mode) != "none"
            and str(rl_policy_mode) == "free"
            and int(episode_idx) <= int(decode_consistency_max_round)
            and float(spim_state["diagnostics"]["top1_top2_margin"]) <= float(decode_consistency_max_margin)
            and float(spim_state["diagnostics"]["posterior_entropy"]) >= float(decode_consistency_min_entropy)
            and int(policy_available_mask.sum().item()) > int(runtime["action_budget"])
        ):
            budget_ok = True
            if decode_consistency_tracker is not None and int(decode_consistency_tracker.get("max_trigger_states", 0)) > 0:
                budget_ok = int(decode_consistency_tracker.get("triggered_states", 0)) < int(
                    decode_consistency_tracker["max_trigger_states"]
                )
            if not budget_ok:
                case_decode_stats["skipped_budget"] += 1
            else:
                t_greedy = time.perf_counter()
                with torch.no_grad():
                    greedy_out = model.act(
                        global_features=spim_state["global_features"].to(device),
                        local_features=spim_state["local_features"].to(device),
                        available_mask=policy_available_mask.to(device),
                        action_budget=int(runtime["action_budget"]),
                        deterministic=True,
                        generator=None,
                        round_index=int(spim_state["diagnostics"]["round_index"]),
                        graph_bundle={
                            "edge_index": spim_state["graph_edge_index"].to(device),
                            "evidence_nodes": list(spim_state["graph_evidence_nodes"]),
                        },
                    )
                greedy_elapsed = float(time.perf_counter() - t_greedy)
                case_decode_stats["greedy_model_s"] += greedy_elapsed
                if decode_consistency_tracker is not None:
                    decode_consistency_tracker["greedy_model_s"] = float(
                        decode_consistency_tracker.get("greedy_model_s", 0.0) + greedy_elapsed
                    )
                greedy_actions = [int(v) for v in greedy_out["actions"]]
                greedy_seq_logprob = float(greedy_out["log_prob"].detach().cpu().item())
                candidates: List[Dict[str, Any]] = [
                    {"actions": list(greedy_actions), "seq_logprob": float(greedy_seq_logprob), "origin": "greedy"}
                ]
                t_beam = time.perf_counter()
                candidates.extend(
                    _enumerate_beam_candidates(
                        logits=greedy_out["logits"],
                        available_mask=policy_available_mask,
                        action_budget=int(runtime["action_budget"]),
                        beam_width=int(decode_consistency_beam_width),
                        topk_per_slot=int(decode_consistency_topk_per_slot),
                    )
                )
                beam_elapsed = float(time.perf_counter() - t_beam)
                case_decode_stats["beam_enum_s"] += beam_elapsed
                if decode_consistency_tracker is not None:
                    decode_consistency_tracker["beam_enum_s"] = float(
                        decode_consistency_tracker.get("beam_enum_s", 0.0) + beam_elapsed
                    )
                deduped: List[Dict[str, Any]] = []
                seen = set()
                for cand in candidates:
                    key = tuple(int(v) for v in cand["actions"])
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    deduped.append(cand)
                if int(decode_consistency_max_preview_candidates) > 0 and len(deduped) > int(
                    decode_consistency_max_preview_candidates
                ):
                    greedy_cand = deduped[0]
                    non_greedy = sorted(deduped[1:], key=lambda row: float(row["seq_logprob"]), reverse=True)
                    keep = max(int(decode_consistency_max_preview_candidates) - 1, 0)
                    deduped = [greedy_cand] + non_greedy[:keep]
                scored = []
                store_preview_state = str(decode_consistency_mode) == "critic_preference"
                for cand in deduped:
                    preview_row = _preview_candidate_set(
                        candidate=cand,
                        case=case,
                        rollout=rollout,
                        history=history,
                        env=env,
                        family=family,
                        trigger_global=trigger_global,
                        onset_grid=onset_grid,
                        gate=gate,
                        runtime=runtime,
                        source_local=source_local,
                        include_surrogate_features=bool(include_surrogate_features),
                        include_uncertainty_regime_features=bool(include_uncertainty_regime_features),
                        prev_entropy=prev_entropy,
                        prev2_entropy=prev2_entropy,
                        prev_top1_mass=prev_top1_mass,
                        prev2_top1_mass=prev2_top1_mass,
                        paper_like_alpha=float(paper_like_alpha),
                        paper_like_topk_fraction=float(paper_like_topk_fraction),
                        paper_like_time_tol_min=float(paper_like_time_tol_min),
                        soft_scenario_beta=float(soft_scenario_beta),
                        device=device,
                        model=model,
                        top_source_k=int(top_source_k),
                        logprob_weight=float(decode_consistency_logprob_weight),
                        value_weight=float(decode_consistency_value_weight),
                        store_preview_state=bool(store_preview_state),
                    )
                    timing = dict(preview_row.get("preview_timing", {}))
                    case_decode_stats["preview_total_s"] += float(timing.get("total_s", 0.0))
                    case_decode_stats["preview_deepcopy_s"] += float(timing.get("deepcopy_s", 0.0))
                    case_decode_stats["preview_step_history_s"] += float(timing.get("step_history_s", 0.0))
                    case_decode_stats["preview_make_rollout_state_s"] += float(timing.get("make_rollout_state_s", 0.0))
                    case_decode_stats["preview_teacher_belief_s"] += float(timing.get("teacher_belief_s", 0.0))
                    case_decode_stats["preview_spim_state_build_s"] += float(timing.get("spim_state_build_s", 0.0))
                    case_decode_stats["preview_model_value_s"] += float(timing.get("model_value_s", 0.0))
                    scored.append(preview_row)
                case_decode_stats["triggered_states"] += 1
                case_decode_stats["preview_candidates"] += int(len(scored))
                if decode_consistency_tracker is not None:
                    decode_consistency_tracker["triggered_states"] = int(
                        decode_consistency_tracker.get("triggered_states", 0) + 1
                    )
                scored.sort(key=lambda row: (float(row["score"]), float(row["seq_logprob"])), reverse=True)
                if scored:
                    winner = scored[0]
                    greedy_scored = next((row for row in scored if list(row["actions"]) == list(greedy_actions)), None)
                    rewrite = bool(list(winner["actions"]) != list(greedy_actions))
                    if rewrite:
                        case_decode_stats["rewrites"] += 1
                        if decode_consistency_tracker is not None:
                            decode_consistency_tracker["rewrites"] = int(
                                decode_consistency_tracker.get("rewrites", 0) + 1
                            )
                    decode_consistency_bundle = {
                        "candidate_count": int(len(scored)),
                        "rewrite": rewrite,
                        "greedy_actions": [int(v) for v in greedy_actions],
                        "winner_actions": [int(v) for v in winner["actions"]],
                        "greedy_score": None if greedy_scored is None else float(greedy_scored["score"]),
                        "winner_score": float(winner["score"]),
                        "greedy_post_value": None if greedy_scored is None else float(greedy_scored["post_value"]),
                        "winner_post_value": float(winner["post_value"]),
                        "greedy_preview_state": None if greedy_scored is None else greedy_scored["preview_state"],
                        "winner_preview_state": winner["preview_state"],
                    }
        round_hit = source_local is not None and int(source_local) in set(selected_actions)
        if round_hit and hit_round is None:
            hit_round = int(episode_idx)
            source_slot = selected_actions.index(int(source_local)) + 1
            hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

        rollout.step_with_actions(
            selected_actions,
            sample_types=[f"{policy_name}_slot_{i}" for i in range(len(selected_actions))],
        )
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])

        post_cover_ratio = float(pre_cover_ratio)
        post_topk_error = float(pre_topk_error)
        if need_cover_term or need_error_term:
            post_paper_state = PaperLikeHSRState(
                source_prior=None if paper_state.source_prior is None else paper_state.source_prior.clone()
            )
            post_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=runtime["dataset_assets"]["topology"],
                num_episodes=int(runtime["num_episodes"]),
                action_budget=int(runtime["action_budget"]),
                frontier_role_mode=str(runtime["frontier_role_mode"]),
            )
            post_belief_ctx = compute_teacher_belief(
                family=family,
                rollout=rollout,
                state=post_state,
                history=history,
                trigger_global=trigger_global,
                paper_state=post_paper_state,
                onset_offsets_min=onset_grid,
                paper_like_alpha=float(paper_like_alpha),
                paper_like_topk_fraction=float(paper_like_topk_fraction),
                paper_like_time_tol_min=float(paper_like_time_tol_min),
                soft_scenario_beta=float(soft_scenario_beta),
            )
            if need_cover_term:
                post_metrics = _belief_metrics(post_belief_ctx["belief"], post_belief_ctx["candidate_mask"], source_local)
                post_cover_ratio = float(post_metrics["mass_cover_size_ratio"])
            if need_error_term:
                post_topk_error = _mean_topk_scenario_error(
                    rollout=rollout,
                    state=post_state,
                    history=history,
                    trigger_global=trigger_global,
                    belief_ctx=post_belief_ctx,
                    onset_offsets_min=onset_grid,
                    topk_fraction=float(reward_topk_fraction),
                    time_tol_min=float(reward_time_tol_min),
                )
        reward_parts = _compute_reward_by_family(
            reward_family=str(reward_family),
            hit_reward=float(hit_reward),
            step_penalty=float(step_penalty),
            selected_count=int(len(selected_actions)),
            round_hit=bool(round_hit),
            pre_cover_ratio=float(pre_cover_ratio),
            post_cover_ratio=float(post_cover_ratio),
            pre_topk_error=float(pre_topk_error),
            post_topk_error=float(post_topk_error),
            lambda_cover=float(reward_lambda_cover),
            lambda_error=float(reward_lambda_error),
            cover_delta_clip=float(reward_cover_delta_clip),
            error_delta_clip=float(reward_error_delta_clip),
        )
        step_reward = float(reward_parts["reward_total"])

        budget_used += int(len(selected_actions))

        teacher_match = float(list(selected_actions) == list(teacher_actions))
        step_rows.append(
            {
                "case_id": case.case_id,
                "policy_name": str(policy_name),
                "episode_index": int(episode_idx),
                "selected_local_ids": json.dumps([int(v) for v in selected_actions]),
                "selected_global_ids": json.dumps([int(v) for v in selected_global_ids]),
                "selected_count": int(len(selected_actions)),
                "source_hit_in_round": float(bool(round_hit)),
                "hit_sample_index": None if not round_hit else int(hit_sample_index),
                "reward": float(step_reward),
                "reward_base": float(reward_parts["reward_base"]),
                "reward_cover_term": float(reward_parts["reward_cover_term"]),
                "reward_error_term": float(reward_parts["reward_error_term"]),
                "delta_cover_raw": float(reward_parts["delta_cover_raw"]),
                "delta_cover_norm": float(reward_parts["delta_cover_norm"]),
                "delta_error_raw": float(reward_parts["delta_error_raw"]),
                "delta_error_norm": float(reward_parts["delta_error_norm"]),
                "pre_topk_error": float(reward_parts["pre_topk_error"]),
                "post_topk_error": float(reward_parts["post_topk_error"]),
                "teacher_exact_match": float(teacher_match),
                "action_agreement_rate": float(action_agreement_rate),
                "ambiguity_gate_triggered": float(bool(gate_triggered)),
                "corrective_candidate_pool_size": int(len(corrective_pool)),
                "corrective_third_replaced": float(bool(corrective_replaced)),
                "policy_slate_size": int(slate_info["slate_size"]),
                "policy_slate_available_size": int(slate_info["available_size"]),
                "policy_slate_posterior_take": int(slate_info["posterior_take"]),
                "policy_slate_disagreement_take": int(slate_info["disagreement_take"]),
                "policy_slate_novelty_take": int(slate_info["novelty_take"]),
                "policy_slate_fill_take": int(slate_info["fill_take"]),
                "policy_slate_early_stage_active": float(slate_info.get("early_stage_slate_active", 0.0)),
                "policy_slate_effective_top_posterior_k": int(slate_info.get("effective_top_posterior_k", slate_top_posterior_k)),
                "policy_slate_effective_high_disagreement_k": int(slate_info.get("effective_high_disagreement_k", slate_high_disagreement_k)),
                "policy_slate_effective_novelty_k": int(slate_info.get("effective_novelty_k", slate_novelty_k)),
                "policy_slate_local_ids": json.dumps(policy_slate_local_ids),
                "policy_slate_global_ids": json.dumps(policy_slate_global_ids),
                "regime_id": None if regime_id is None else int(regime_id),
                "regime_probs": None if regime_probs is None else json.dumps([float(v) for v in regime_probs]),
                **spim_state["diagnostics"],
            }
        )

        if collect_transitions:
            transition_row = {
                "global_features": spim_state["global_features"].detach().cpu(),
                "local_features": spim_state["local_features"].detach().cpu(),
                "available_mask": policy_available_mask.detach().cpu(),
                "graph_edge_index": spim_state["graph_edge_index"].detach().cpu(),
                "graph_evidence_nodes": list(spim_state["graph_evidence_nodes"]),
                "actions": [int(v) for v in selected_actions],
                "teacher_actions": [int(v) for v in teacher_actions],
                "old_log_prob": (
                    float(policy_out["log_prob"].detach().cpu().item()) if policy_out is not None else 0.0
                ),
                "old_value": (
                    float(policy_out["value"].detach().cpu().item()) if policy_out is not None else 0.0
                ),
                "reward": float(step_reward),
                "done": bool(round_hit),
                "case_id": str(case.case_id),
                "episode_index": int(episode_idx),
                "round_index": int(spim_state["diagnostics"]["round_index"]),
                "rl_policy_mode": str(rl_policy_mode),
                "gate_triggered": bool(gate_triggered),
                "action_agreement_rate": float(action_agreement_rate),
                "corrective_third_replaced": bool(corrective_replaced),
                "corrective_candidate_pool": [int(v) for v in corrective_pool],
                "policy_slate_size": int(slate_info["slate_size"]),
                "policy_slate_available_size": int(slate_info["available_size"]),
                "ambiguity_top1_top2_margin": float(spim_state["diagnostics"]["top1_top2_margin"]),
                "ambiguity_candidate_count": int(spim_state["diagnostics"]["candidate_count"]),
                "ambiguity_entropy_norm": float(spim_state["diagnostics"]["posterior_entropy"])
                / max(math.log(max(int(spim_state["diagnostics"]["candidate_count"]), 2)), 1e-6),
                "regime_id": None if regime_id is None else int(regime_id),
                "regime_probs": None if regime_probs is None else [float(v) for v in regime_probs],
                "decode_consistency_mode": str(decode_consistency_mode),
                "decode_rewrite": False if decode_consistency_bundle is None else bool(decode_consistency_bundle["rewrite"]),
                "decode_candidate_count": 0 if decode_consistency_bundle is None else int(decode_consistency_bundle["candidate_count"]),
                "decode_greedy_actions": [] if decode_consistency_bundle is None else list(decode_consistency_bundle["greedy_actions"]),
                "decode_winner_actions": [] if decode_consistency_bundle is None else list(decode_consistency_bundle["winner_actions"]),
                "decode_greedy_score": None if decode_consistency_bundle is None else decode_consistency_bundle["greedy_score"],
                "decode_winner_score": None if decode_consistency_bundle is None else decode_consistency_bundle["winner_score"],
                "decode_greedy_post_value": None if decode_consistency_bundle is None else decode_consistency_bundle["greedy_post_value"],
                "decode_winner_post_value": None if decode_consistency_bundle is None else decode_consistency_bundle["winner_post_value"],
            }
            if decode_consistency_bundle is not None and bool(decode_consistency_bundle["rewrite"]):
                transition_row["decode_greedy_preview_state"] = decode_consistency_bundle["greedy_preview_state"]
                transition_row["decode_winner_preview_state"] = decode_consistency_bundle["winner_preview_state"]
            if len(teacher_actions) >= 3 and len(selected_actions) >= 3:
                transition_row["teacher_third_action"] = int(teacher_actions[2])
                transition_row["selected_third_action"] = int(selected_actions[2])
            if str(rl_policy_mode) == "conservative_corrective_rl_v1" and not bool(gate_triggered):
                # No RL decision happened; do not add a no-op transition.
                pass
            else:
                transitions.append(transition_row)

        if round_hit:
            termination_reason = "source_hit"
            break

        prev2_entropy = prev_entropy
        prev_entropy = float(spim_state["diagnostics"]["posterior_entropy"])
        prev2_top1_mass = prev_top1_mass
        prev_top1_mass = float(spim_state["diagnostics"]["top1_mass"])

    final_state = make_rollout_state(
        case=case,
        rollout=rollout,
        history=history,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        num_episodes=int(runtime["num_episodes"]),
        action_budget=int(runtime["action_budget"]),
        frontier_role_mode=str(runtime["frontier_role_mode"]),
    )
    final_belief = compute_teacher_belief(
        family=family,
        rollout=rollout,
        state=final_state,
        history=history,
        trigger_global=trigger_global,
        paper_state=paper_state,
        onset_offsets_min=onset_grid,
        paper_like_alpha=float(paper_like_alpha),
        paper_like_topk_fraction=float(paper_like_topk_fraction),
        paper_like_time_tol_min=float(paper_like_time_tol_min),
        soft_scenario_beta=float(soft_scenario_beta),
    )
    final_metrics = _belief_metrics(final_belief["belief"], final_belief["candidate_mask"], source_local)

    case_row = {
        "case_id": case.case_id,
        "scenario_id": int(case.scenario_id),
        "part_id": int(case.part_id),
        "policy_name": str(policy_name),
        "success_rate": float(hit_round is not None),
        "hit_round": None if hit_round is None else int(hit_round),
        "hit_sample_index": None if hit_sample_index is None else int(hit_sample_index),
        "budget_used": int(budget_used),
        "avg_step_reward": float(sum(float(row["reward"]) for row in step_rows) / max(len(step_rows), 1)),
        "teacher_exact_match_rate": float(sum(float(row["teacher_exact_match"]) for row in step_rows) / max(len(step_rows), 1)),
        "action_agreement_rate": float(sum(float(row["action_agreement_rate"]) for row in step_rows) / max(len(step_rows), 1)),
        "ambiguity_gate_trigger_rate": float(sum(float(row["ambiguity_gate_triggered"]) for row in step_rows) / max(len(step_rows), 1)),
        "corrective_third_replace_rate": float(sum(float(row["corrective_third_replaced"]) for row in step_rows) / max(len(step_rows), 1)),
        "final_top1_mass": float(final_metrics["top1_mass"]),
        "final_top3_mass": float(final_metrics["top3_mass"]),
        "final_entropy": float(final_metrics["entropy"]),
        "termination_reason": str(termination_reason),
        "source_global_id": source_global,
        "trigger_global_id": trigger_global,
    }
    return {
        "case_row": case_row,
        "step_rows": step_rows,
        "transitions": transitions,
        "decode_consistency_stats": case_decode_stats,
    }


def run_policy_on_cases(
    *,
    cases: Sequence[CaseRecord],
    family: str,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    policy_name: str,
    model: Optional[SpimNativePolicy],
    deterministic: bool,
    base_seed: int,
    include_surrogate_features: bool,
    include_uncertainty_regime_features: bool,
    top_source_k: int,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
    hit_reward: float,
    step_penalty: float,
    reward_family: str,
    reward_lambda_cover: float,
    reward_lambda_error: float,
    reward_cover_delta_clip: float,
    reward_error_delta_clip: float,
    reward_topk_fraction: float,
    reward_time_tol_min: float,
    rl_policy_mode: str,
    corrective_candidate_topk: int,
    ambiguity_top1_top2_max: float,
    ambiguity_min_candidate_count: int,
    device: torch.device,
    collect_transitions: bool,
    slate_size: int = 10,
    slate_top_posterior_k: int = 6,
    slate_high_disagreement_k: int = 3,
    slate_novelty_k: int = 2,
    early_stage_round_cutoff: int = 0,
    early_stage_slate_top_posterior_k: Optional[int] = None,
    early_stage_slate_high_disagreement_k: Optional[int] = None,
    early_stage_slate_novelty_k: Optional[int] = None,
    decode_consistency_mode: str = "none",
    decode_consistency_beam_width: int = 4,
    decode_consistency_topk_per_slot: int = 4,
    decode_consistency_logprob_weight: float = 1.0,
    decode_consistency_value_weight: float = 4.0,
    decode_consistency_max_margin: float = 0.10,
    decode_consistency_max_round: int = 6,
    decode_consistency_min_entropy: float = 0.0,
    decode_consistency_max_preview_candidates: int = 0,
    decode_consistency_max_trigger_states: int = 0,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    all_transitions: List[Dict[str, Any]] = []
    decode_consistency_stats = {
        "triggered_states": 0,
        "rewrites": 0,
        "preview_candidates": 0,
        "greedy_model_s": 0.0,
        "beam_enum_s": 0.0,
        "preview_total_s": 0.0,
        "preview_deepcopy_s": 0.0,
        "preview_step_history_s": 0.0,
        "preview_make_rollout_state_s": 0.0,
        "preview_teacher_belief_s": 0.0,
        "preview_spim_state_build_s": 0.0,
        "preview_model_value_s": 0.0,
        "skipped_budget": 0,
        "max_trigger_states": int(max(decode_consistency_max_trigger_states, 0)),
    }

    for case_index, case in enumerate(cases):
        out = run_case_policy(
            case=case,
            case_index=int(case_index),
            family=family,
            runtime=runtime,
            env=env,
            policy_name=policy_name,
            model=model,
            deterministic=bool(deterministic),
            base_seed=int(base_seed),
            include_surrogate_features=bool(include_surrogate_features),
            include_uncertainty_regime_features=bool(include_uncertainty_regime_features),
            top_source_k=int(top_source_k),
            paper_like_alpha=float(paper_like_alpha),
            paper_like_topk_fraction=float(paper_like_topk_fraction),
            paper_like_time_tol_min=float(paper_like_time_tol_min),
            soft_scenario_beta=float(soft_scenario_beta),
            hit_reward=float(hit_reward),
            step_penalty=float(step_penalty),
            reward_family=str(reward_family),
            reward_lambda_cover=float(reward_lambda_cover),
            reward_lambda_error=float(reward_lambda_error),
            reward_cover_delta_clip=float(reward_cover_delta_clip),
            reward_error_delta_clip=float(reward_error_delta_clip),
            reward_topk_fraction=float(reward_topk_fraction),
            reward_time_tol_min=float(reward_time_tol_min),
            rl_policy_mode=str(rl_policy_mode),
            corrective_candidate_topk=int(corrective_candidate_topk),
            ambiguity_top1_top2_max=float(ambiguity_top1_top2_max),
            ambiguity_min_candidate_count=int(ambiguity_min_candidate_count),
            device=device,
            collect_transitions=bool(collect_transitions),
            slate_size=int(slate_size),
            slate_top_posterior_k=int(slate_top_posterior_k),
            slate_high_disagreement_k=int(slate_high_disagreement_k),
            slate_novelty_k=int(slate_novelty_k),
            early_stage_round_cutoff=int(early_stage_round_cutoff),
            early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
            early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
            early_stage_slate_novelty_k=early_stage_slate_novelty_k,
            decode_consistency_mode=str(decode_consistency_mode),
            decode_consistency_beam_width=int(decode_consistency_beam_width),
            decode_consistency_topk_per_slot=int(decode_consistency_topk_per_slot),
            decode_consistency_logprob_weight=float(decode_consistency_logprob_weight),
            decode_consistency_value_weight=float(decode_consistency_value_weight),
            decode_consistency_max_margin=float(decode_consistency_max_margin),
            decode_consistency_max_round=int(decode_consistency_max_round),
            decode_consistency_min_entropy=float(decode_consistency_min_entropy),
            decode_consistency_max_preview_candidates=int(decode_consistency_max_preview_candidates),
            decode_consistency_tracker=decode_consistency_stats,
        )
        case_rows.append(out["case_row"])
        step_rows.extend(out["step_rows"])
        all_transitions.extend(out["transitions"])
        case_stats = out.get("decode_consistency_stats", {})
        decode_consistency_stats["preview_candidates"] = int(
            decode_consistency_stats.get("preview_candidates", 0) + int(case_stats.get("preview_candidates", 0))
        )
        decode_consistency_stats["skipped_budget"] = int(
            decode_consistency_stats.get("skipped_budget", 0) + int(case_stats.get("skipped_budget", 0))
        )
        for key in [
            "preview_total_s",
            "preview_deepcopy_s",
            "preview_step_history_s",
            "preview_make_rollout_state_s",
            "preview_teacher_belief_s",
            "preview_spim_state_build_s",
            "preview_model_value_s",
        ]:
            decode_consistency_stats[key] = float(
                decode_consistency_stats.get(key, 0.0) + float(case_stats.get(key, 0.0))
            )

    return {
        "case_rows": case_rows,
        "step_rows": step_rows,
        "transitions": all_transitions,
        "decode_consistency_stats": decode_consistency_stats,
    }


def summarize_case_metrics(case_rows: Sequence[Dict[str, Any]], num_rounds: int, action_budget: int) -> Dict[str, Any]:
    if not case_rows:
        return {
            "case_count": 0,
            "success_rate": 0.0,
            "avg_hit_round_conditional": None,
            "budget_used_mean": 0.0,
            "teacher_exact_match_rate_mean": 0.0,
        }
    df = pd.DataFrame(case_rows)
    hit_mask = df["success_rate"] > 0.5
    summary = {
        "case_count": int(len(df)),
        "success_rate": float(df["success_rate"].mean()),
        "avg_hit_round_conditional": float(df.loc[hit_mask, "hit_round"].mean()) if bool(hit_mask.any()) else None,
        "budget_used_mean": float(df["budget_used"].mean()),
        "teacher_exact_match_rate_mean": float(df["teacher_exact_match_rate"].mean()),
        "action_agreement_rate_mean": float(df["action_agreement_rate"].mean()) if "action_agreement_rate" in df.columns else None,
        "ambiguity_gate_trigger_rate_mean": float(df["ambiguity_gate_trigger_rate"].mean()) if "ambiguity_gate_trigger_rate" in df.columns else None,
        "corrective_third_replace_rate_mean": float(df["corrective_third_replace_rate"].mean()) if "corrective_third_replace_rate" in df.columns else None,
        "termination_reason_counts": df["termination_reason"].value_counts().to_dict(),
    }

    round_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []

    for r in range(1, int(num_rounds) + 1):
        mask = df["hit_round"].fillna(10**9) <= int(r)
        round_rows.append(
            {
                "round_index": int(r),
                "cumulative_success_rate": float(mask.mean()),
            }
        )

    total_budget = int(num_rounds) * int(action_budget)
    for b in range(1, total_budget + 1):
        hit_sample = df["hit_sample_index"].fillna(10**9)
        hit_round = df["hit_round"].fillna(10**9)
        budget_hit = (hit_sample <= int(b)) | (hit_round <= math.ceil(float(b) / float(action_budget)))
        budget_rows.append(
            {
                "sample_budget": int(b),
                "cumulative_success_rate": float(budget_hit.mean()),
            }
        )

    summary["round_curve"] = round_rows
    summary["budget_curve"] = budget_rows
    return summary


def _compute_bc_loss(
    model: SpimNativePolicy,
    transition: Dict[str, Any],
    device: torch.device,
    action_budget: int,
) -> Tuple[Optional[torch.Tensor], float]:
    return _compute_action_sequence_ce_loss(
        model=model,
        transition=transition,
        target_actions=transition.get("teacher_actions", []),
        device=device,
        action_budget=int(action_budget),
    )


def _compute_corrective_anchor_loss(
    *,
    model: SpimNativePolicy,
    transition: Dict[str, Any],
    device: torch.device,
) -> Optional[torch.Tensor]:
    pool = [int(v) for v in transition.get("corrective_candidate_pool", [])]
    teacher_third = transition.get("teacher_third_action", None)
    if teacher_third is None or len(pool) <= 0:
        return None
    global_features = transition["global_features"].to(device)
    local_features = transition["local_features"].to(device)
    available = transition["available_mask"].to(device).view(-1).bool().clone()
    graph_bundle = {
        "edge_index": transition["graph_edge_index"].to(device),
        "evidence_nodes": list(transition.get("graph_evidence_nodes", [])),
    }
    logits = model.score_actions(global_features, local_features, available_mask=available, graph_bundle=graph_bundle)
    idx = torch.tensor(pool, dtype=torch.long, device=device)
    cand_logits = logits[idx]
    target = (idx == int(teacher_third)).nonzero(as_tuple=True)[0]
    if target.numel() != 1:
        return None
    return F.cross_entropy(cand_logits.view(1, -1), target[0].view(1))


def train_behavior_cloning(
    *,
    model: SpimNativePolicy,
    transitions: Sequence[Dict[str, Any]],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    action_budget: int,
    grad_clip: float,
    save_epoch_checkpoints_dir: Optional[Path] = None,
    checkpoint_prefix: str = "bc_epoch",
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    history: List[Dict[str, float]] = []
    idxs = list(range(len(transitions)))

    for epoch in range(1, int(epochs) + 1):
        random.shuffle(idxs)
        epoch_loss_sum = 0.0
        epoch_match_sum = 0.0
        epoch_items = 0

        for start in range(0, len(idxs), int(batch_size)):
            chunk = idxs[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            batch_losses: List[torch.Tensor] = []
            batch_match = 0.0
            batch_items = 0
            for idx in chunk:
                loss, match_rate = _compute_bc_loss(
                    model=model,
                    transition=transitions[idx],
                    device=device,
                    action_budget=int(action_budget),
                )
                if loss is None:
                    continue
                batch_losses.append(loss)
                batch_match += float(match_rate)
                batch_items += 1

            if not batch_losses:
                continue

            loss_mean = torch.stack(batch_losses).mean()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            epoch_loss_sum += float(loss_mean.detach().cpu().item()) * float(batch_items)
            epoch_match_sum += float(batch_match)
            epoch_items += int(batch_items)

        history.append(
            {
                "epoch": float(epoch),
                "bc_loss": float(epoch_loss_sum / max(epoch_items, 1)),
                "slot_match_rate": float(epoch_match_sum / max(epoch_items, 1)),
                "effective_batches": float(epoch_items),
            }
        )
        if save_epoch_checkpoints_dir is not None:
            save_epoch_checkpoints_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_epoch_checkpoints_dir / f"{checkpoint_prefix}_{int(epoch):03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
    return history


def _compute_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    out: List[float] = []
    running = 0.0
    for reward in reversed(list(rewards)):
        running = float(reward) + float(gamma) * float(running)
        out.append(float(running))
    out.reverse()
    return out


def _build_case_return_lookup(
    transitions: Sequence[Dict[str, Any]],
    gamma: float,
) -> Dict[str, Dict[int, float]]:
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in transitions:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    lookup: Dict[str, Dict[int, float]] = {}
    for case_id, rows in by_case.items():
        ordered = sorted(rows, key=lambda r: int(r["episode_index"]))
        rewards = [float(r["reward"]) for r in ordered]
        returns = _compute_returns(rewards, gamma=float(gamma))
        per_ep: Dict[int, float] = {}
        for row, ret in zip(ordered, returns):
            per_ep[int(row["episode_index"])] = float(ret)
        lookup[str(case_id)] = per_ep
    return lookup


def _prepare_ppo_targets(
    transitions: List[Dict[str, Any]],
    gamma: float,
    baseline_returns: Optional[Dict[str, Dict[int, float]]] = None,
) -> None:
    if not transitions:
        return
    by_case: Dict[str, List[int]] = {}
    for idx, row in enumerate(transitions):
        by_case.setdefault(str(row["case_id"]), []).append(idx)

    for case_id, idxs in by_case.items():
        ordered = sorted(idxs, key=lambda i: int(transitions[i]["episode_index"]))
        rewards = [float(transitions[i]["reward"]) for i in ordered]
        returns = _compute_returns(rewards, gamma=float(gamma))
        for i, ret in zip(ordered, returns):
            case_id_i = str(transitions[i]["case_id"])
            ep_i = int(transitions[i]["episode_index"])
            baseline = 0.0
            if baseline_returns is not None:
                baseline = float(baseline_returns.get(case_id_i, {}).get(ep_i, 0.0))
            delta_return = float(ret - baseline)
            transitions[i]["raw_return"] = float(ret)
            transitions[i]["baseline_return"] = float(baseline)
            transitions[i]["return"] = float(delta_return)
            transitions[i]["advantage"] = float(delta_return - float(transitions[i]["old_value"]))

    adv = torch.tensor([float(row.get("advantage", 0.0)) for row in transitions], dtype=torch.float32)
    adv_mean = float(adv.mean().item()) if adv.numel() > 0 else 0.0
    adv_std = float(adv.std(unbiased=False).item()) if adv.numel() > 0 else 1.0
    adv_std = max(adv_std, 1e-6)
    for row in transitions:
        row["advantage_norm"] = float((float(row.get("advantage", 0.0)) - adv_mean) / adv_std)


def _entropy_margin_focus_weight(
    *,
    entropy_norm: float,
    margin: float,
    alpha: float,
    entropy_threshold: float,
    margin_threshold: float,
    entropy_temp: float,
    margin_temp: float,
    device: torch.device,
) -> float:
    if float(alpha) <= 0.0:
        return 1.0
    entropy_gate = torch.sigmoid(
        torch.tensor(
            (float(entropy_norm) - float(entropy_threshold)) / max(float(entropy_temp), 1e-6),
            dtype=torch.float32,
            device=device,
        )
    )
    margin_gate = torch.sigmoid(
        torch.tensor(
            (float(margin_threshold) - float(margin)) / max(float(margin_temp), 1e-6),
            dtype=torch.float32,
            device=device,
        )
    )
    return float(1.0 + float(alpha) * float((entropy_gate * margin_gate).item()))


def ppo_update(
    *,
    model: SpimNativePolicy,
    transitions: Sequence[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    action_budget: int,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    imitation_anchor_weight: float,
    update_epochs: int,
    minibatch_size: int,
    grad_clip: float,
    actor_loss_coef: float,
    ambiguity_weighting_mode: str,
    ambiguity_weight_alpha: float,
    ambiguity_entropy_threshold: float,
    ambiguity_margin_threshold: float,
    ambiguity_entropy_temp: float,
    ambiguity_margin_temp: float,
    early_stage_round_cutoff: int,
    early_stage_set_aux_weight: float,
    early_stage_set_adv_mode: str,
    early_stage_set_aux_ambiguity_mode: str,
    early_stage_set_aux_ambiguity_alpha: float,
    decode_consistency_mode: str,
    decode_consistency_weight: float,
    decode_consistency_margin: float,
) -> Dict[str, float]:
    idxs = list(range(len(transitions)))
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    anchor_loss_sum = 0.0
    decode_aux_loss_sum = 0.0
    sample_weight_sum = 0.0
    sample_weight_count = 0
    set_aux_loss_sum = 0.0
    set_aux_weight_sum = 0.0
    set_aux_ratio_sum = 0.0
    set_aux_clip_frac_sum = 0.0
    set_aux_count = 0
    update_steps = 0
    set_aux_adv_lookup: Dict[int, float] = {}

    if float(early_stage_set_aux_weight) > 0.0 and int(early_stage_round_cutoff) > 0:
        eligible_rows: List[Tuple[int, float]] = []
        for idx, tr in enumerate(transitions):
            round_index = int(tr.get("round_index", tr.get("episode_index", 1)))
            if round_index > int(early_stage_round_cutoff):
                continue
            action_count = max(len(list(tr.get("actions", []))), 1)
            raw_advantage = float(tr.get("advantage", 0.0))
            if str(early_stage_set_adv_mode) == "sum":
                raw_advantage = float(raw_advantage * float(action_count))
            eligible_rows.append((idx, raw_advantage))
        if eligible_rows:
            eligible_adv = torch.tensor([row[1] for row in eligible_rows], dtype=torch.float32)
            adv_mean = float(eligible_adv.mean().item())
            adv_std = max(float(eligible_adv.std(unbiased=False).item()), 1e-6)
            for idx, raw_advantage in eligible_rows:
                set_aux_adv_lookup[idx] = float((float(raw_advantage) - adv_mean) / adv_std)

    for _ in range(int(update_epochs)):
        random.shuffle(idxs)
        for start in range(0, len(idxs), int(minibatch_size)):
            chunk = idxs[start : start + int(minibatch_size)]
            optimizer.zero_grad(set_to_none=True)
            losses: List[torch.Tensor] = []
            chunk_policy = 0.0
            chunk_value = 0.0
            chunk_entropy = 0.0
            chunk_anchor = 0.0
            chunk_decode_aux = 0.0
            chunk_set_aux = 0.0
            chunk_count = 0

            for idx in chunk:
                tr = transitions[idx]
                graph_bundle = {
                    "edge_index": tr["graph_edge_index"].to(device),
                    "evidence_nodes": list(tr.get("graph_evidence_nodes", [])),
                }
                rl_mode = str(tr.get("rl_policy_mode", "free"))
                if rl_mode == "conservative_corrective_rl_v1":
                    if not bool(tr.get("gate_triggered", False)):
                        continue
                    out = model.evaluate_single_choice(
                        global_features=tr["global_features"].to(device),
                        local_features=tr["local_features"].to(device),
                        candidate_indices=[int(v) for v in tr.get("corrective_candidate_pool", [])],
                        selected_action=int(tr.get("selected_third_action", -1)),
                        available_mask=tr["available_mask"].to(device),
                        round_index=int(tr.get("round_index", tr.get("episode_index", 1))),
                        graph_bundle=graph_bundle,
                    )
                else:
                    out = model.evaluate_actions(
                        global_features=tr["global_features"].to(device),
                        local_features=tr["local_features"].to(device),
                        available_mask=tr["available_mask"].to(device),
                        selected_actions=tr["actions"],
                        action_budget=int(action_budget),
                        round_index=int(tr.get("round_index", tr.get("episode_index", 1))),
                        graph_bundle=graph_bundle,
                    )
                new_log_prob = out["log_prob"]
                new_value = out["value"]
                entropy = out["entropy"]

                old_log_prob = torch.tensor(float(tr["old_log_prob"]), device=device, dtype=torch.float32)
                advantage = torch.tensor(float(tr["advantage_norm"]), device=device, dtype=torch.float32)
                target_return = torch.tensor(float(tr["return"]), device=device, dtype=torch.float32)
                sample_weight = 1.0
                apply_ambiguity_focus = (
                    str(ambiguity_weighting_mode) == "entropy_margin_focus"
                    and int(early_stage_round_cutoff) > 0
                    and int(tr.get("round_index", tr.get("episode_index", 1))) <= int(early_stage_round_cutoff)
                )
                if apply_ambiguity_focus:
                    entropy_norm = float(tr.get("ambiguity_entropy_norm", 0.0))
                    margin = float(tr.get("ambiguity_top1_top2_margin", 1.0))
                    sample_weight = _entropy_margin_focus_weight(
                        entropy_norm=entropy_norm,
                        margin=margin,
                        alpha=float(ambiguity_weight_alpha),
                        entropy_threshold=float(ambiguity_entropy_threshold),
                        margin_threshold=float(ambiguity_margin_threshold),
                        entropy_temp=float(ambiguity_entropy_temp),
                        margin_temp=float(ambiguity_margin_temp),
                        device=device,
                    )
                w = torch.tensor(float(sample_weight), device=device, dtype=torch.float32)

                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * advantage
                policy_loss = -torch.minimum(surr1, surr2)
                value_loss = F.mse_loss(new_value, target_return)
                actor_term = float(actor_loss_coef) * (policy_loss - float(entropy_coef) * entropy)

                loss = w * actor_term + float(value_coef) * value_loss
                set_aux_loss_value = 0.0

                if idx in set_aux_adv_lookup:
                    set_advantage = torch.tensor(float(set_aux_adv_lookup[idx]), device=device, dtype=torch.float32)
                    set_aux_weight = 1.0
                    if (
                        str(early_stage_set_aux_ambiguity_mode) == "entropy_margin_focus"
                        and float(early_stage_set_aux_ambiguity_alpha) > 0.0
                    ):
                        set_aux_weight = _entropy_margin_focus_weight(
                            entropy_norm=float(tr.get("ambiguity_entropy_norm", 0.0)),
                            margin=float(tr.get("ambiguity_top1_top2_margin", 1.0)),
                            alpha=float(early_stage_set_aux_ambiguity_alpha),
                            entropy_threshold=float(ambiguity_entropy_threshold),
                            margin_threshold=float(ambiguity_margin_threshold),
                            entropy_temp=float(ambiguity_entropy_temp),
                            margin_temp=float(ambiguity_margin_temp),
                            device=device,
                        )
                    set_surr1 = ratio * set_advantage
                    set_surr2 = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * set_advantage
                    set_policy_loss = -torch.minimum(set_surr1, set_surr2)
                    loss = loss + float(early_stage_set_aux_weight) * float(set_aux_weight) * set_policy_loss
                    set_aux_loss_value = float(set_policy_loss.detach().cpu().item())
                    set_aux_loss_sum += float(set_aux_loss_value)
                    set_aux_weight_sum += float(set_aux_weight)
                    set_aux_ratio_sum += float(ratio.detach().cpu().item())
                    set_aux_clip_frac_sum += float(
                        (torch.abs(ratio.detach() - 1.0) > float(clip_range)).float().cpu().item()
                    )
                    set_aux_count += 1

                anchor_loss_value = 0.0
                if float(actor_loss_coef) > 0.0 and float(imitation_anchor_weight) > 0.0:
                    if rl_mode == "conservative_corrective_rl_v1":
                        anchor_loss = _compute_corrective_anchor_loss(
                            model=model,
                            transition=tr,
                            device=device,
                        )
                        if anchor_loss is not None:
                            loss = loss + float(imitation_anchor_weight) * anchor_loss
                            anchor_loss_value = float(anchor_loss.detach().cpu().item())
                    else:
                        bc_loss, _ = _compute_bc_loss(
                            model=model,
                            transition=tr,
                            device=device,
                            action_budget=int(action_budget),
                        )
                        if bc_loss is not None:
                            loss = loss + float(imitation_anchor_weight) * bc_loss
                            anchor_loss_value = float(bc_loss.detach().cpu().item())

                decode_aux_value = 0.0
                if float(decode_consistency_weight) > 0.0 and bool(tr.get("decode_rewrite", False)):
                    if str(decode_consistency_mode) == "beam_winner_distill" and float(actor_loss_coef) > 0.0:
                        decode_loss, _ = _compute_action_sequence_ce_loss(
                            model=model,
                            transition=tr,
                            target_actions=tr.get("decode_winner_actions", []),
                            device=device,
                            action_budget=int(action_budget),
                        )
                        if decode_loss is not None:
                            loss = loss + float(decode_consistency_weight) * decode_loss
                            decode_aux_value = float(decode_loss.detach().cpu().item())
                    elif str(decode_consistency_mode) == "critic_preference":
                        winner_state = tr.get("decode_winner_preview_state", None)
                        greedy_state = tr.get("decode_greedy_preview_state", None)
                        if isinstance(winner_state, dict) and isinstance(greedy_state, dict):
                            winner_value = _evaluate_packed_state_value(
                                model=model,
                                packed_state=winner_state,
                                device=device,
                            )
                            greedy_value = _evaluate_packed_state_value(
                                model=model,
                                packed_state=greedy_state,
                                device=device,
                            )
                            margin = winner_value - greedy_value
                            decode_loss = F.relu(
                                torch.tensor(float(decode_consistency_margin), device=device, dtype=torch.float32) - margin
                            )
                            loss = loss + float(decode_consistency_weight) * decode_loss
                            decode_aux_value = float(decode_loss.detach().cpu().item())

                losses.append(loss)
                chunk_policy += float(policy_loss.detach().cpu().item())
                chunk_value += float(value_loss.detach().cpu().item())
                chunk_entropy += float(entropy.detach().cpu().item())
                chunk_anchor += float(anchor_loss_value)
                chunk_decode_aux += float(decode_aux_value)
                chunk_set_aux += float(set_aux_loss_value)
                sample_weight_sum += float(sample_weight)
                sample_weight_count += 1
                chunk_count += 1

            if not losses:
                continue

            loss_mean = torch.stack(losses).mean()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            denom = max(chunk_count, 1)
            policy_loss_sum += float(chunk_policy / denom)
            value_loss_sum += float(chunk_value / denom)
            entropy_sum += float(chunk_entropy / denom)
            anchor_loss_sum += float(chunk_anchor / denom)
            decode_aux_loss_sum += float(chunk_decode_aux / denom)
            set_aux_loss_sum += float(chunk_set_aux / denom)
            update_steps += 1

    denom = max(update_steps, 1)
    return {
        "ppo_policy_loss": float(policy_loss_sum / denom),
        "ppo_value_loss": float(value_loss_sum / denom),
        "ppo_entropy": float(entropy_sum / denom),
        "ppo_anchor_loss": float(anchor_loss_sum / denom),
        "ppo_decode_aux_loss": float(decode_aux_loss_sum / denom),
        "ppo_sample_weight_mean": float(sample_weight_sum / max(sample_weight_count, 1)),
        "ppo_set_aux_loss": float(set_aux_loss_sum / denom),
        "ppo_set_aux_weight_mean": float(set_aux_weight_sum / max(set_aux_count, 1)),
        "ppo_set_aux_ratio_mean": float(set_aux_ratio_sum / max(set_aux_count, 1)),
        "ppo_set_aux_clip_frac": float(set_aux_clip_frac_sum / max(set_aux_count, 1)),
        "ppo_set_aux_active_fraction": float(set_aux_count / max(sample_weight_count, 1)),
        "ppo_update_steps": float(update_steps),
    }


def auto_select_teacher(precheck_root: Path) -> Dict[str, Any]:
    summary_csv = precheck_root / "family_summary.csv"
    if not summary_csv.exists():
        return {
            "selected_family": "hsr_soft_scenario_posterior_v3",
            "selection_rule": "precheck_missing_default_v3",
            "v3_success_rate": None,
            "v1_success_rate": None,
            "precheck_path": str(summary_csv),
        }

    df = pd.read_csv(summary_csv)
    v3 = df[df["family"] == "hsr_soft_scenario_posterior_v3"]
    v1 = df[df["family"] == "hsr_paper_topk_ema_v1"]
    if len(v3) == 0 or len(v1) == 0:
        return {
            "selected_family": "hsr_soft_scenario_posterior_v3",
            "selection_rule": "precheck_incomplete_default_v3",
            "v3_success_rate": None,
            "v1_success_rate": None,
            "precheck_path": str(summary_csv),
        }

    v3_sr = float(v3.iloc[0]["success_rate"])
    v1_sr = float(v1.iloc[0]["success_rate"])
    selected = "hsr_soft_scenario_posterior_v3" if v3_sr >= v1_sr else "hsr_paper_topk_ema_v1"
    return {
        "selected_family": str(selected),
        "selection_rule": "v3_not_weaker_choose_v3" if selected == "hsr_soft_scenario_posterior_v3" else "v3_weaker_choose_v1",
        "v3_success_rate": float(v3_sr),
        "v1_success_rate": float(v1_sr),
        "precheck_path": str(summary_csv),
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    precheck_root = Path(args.precheck_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(str(args.device))

    exact_runtime = load_runtime_context(source_root, cache_dir)
    exact_runtime["num_episodes"] = int(args.num_rounds)
    exact_runtime["action_budget"] = int(args.actions_per_round)

    train_cases, train_assets, train_cfg_path = load_train_full_cases(
        source_root=source_root,
        cache_dir=cache_dir,
        max_cases=int(args.train_full_max_cases),
        cache_version=str(args.train_full_cache_version),
    )
    train_runtime = {
        "cases": train_cases,
        "dataset_assets": train_assets,
        "num_episodes": int(args.num_rounds),
        "action_budget": int(args.actions_per_round),
        "episode_duration_min": float(exact_runtime["episode_duration_min"]),
        "frontier_role_mode": str(exact_runtime["frontier_role_mode"]),
    }

    if str(args.teacher_family) == "auto":
        teacher_decision = auto_select_teacher(precheck_root)
        teacher_family = str(teacher_decision["selected_family"])
    else:
        teacher_family = str(args.teacher_family)
        teacher_decision = {
            "selected_family": str(teacher_family),
            "selection_rule": "manual_override",
            "v3_success_rate": None,
            "v1_success_rate": None,
            "precheck_path": str(precheck_root / "family_summary.csv"),
        }
    write_json(output_dir / "teacher_precheck_decision.json", teacher_decision)

    env = CleanTwoChannelEvidenceEnv()
    global_feature_names = get_global_feature_names(bool(args.use_uncertainty_regime_features))
    local_feature_names = get_local_feature_names(
        include_surrogate_features=bool(args.include_surrogate_features),
        include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
    )

    model = SpimNativePolicy(
        global_dim=len(global_feature_names),
        local_dim=len(local_feature_names),
        hidden_dim=int(args.hidden_dim),
        policy_arch=str(args.policy_arch),
        policy_mlp_depth=int(args.policy_mlp_depth),
        value_mlp_depth=int(args.value_mlp_depth),
        value_head_width_mult=float(args.value_head_width_mult),
        critic_trunk_depth=int(args.critic_trunk_depth),
        critic_trunk_hidden_dim=int(args.critic_trunk_hidden_dim),
        policy_dropout=float(args.policy_dropout),
        policy_norm=str(args.policy_norm),
        candidate_encoder=str(args.candidate_encoder),
        candidate_attn_heads=int(args.candidate_attn_heads),
        enable_regime_head=bool(args.enable_regime_head),
        regime_head_classes=int(args.regime_head_classes),
        regime_embed_dim=int(args.regime_embed_dim),
        arch_backbone=str(args.arch_backbone),
        residual_hidden_dim=int(args.residual_hidden_dim),
        residual_depth=int(args.residual_depth),
        residual_head_dim=int(args.residual_head_dim),
        transformer_token_dim=int(args.transformer_token_dim),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        transformer_ffn_dim=int(args.transformer_ffn_dim),
        graph_hidden_dim=int(args.graph_hidden_dim),
        graph_layers=int(args.graph_layers),
        graph_heads=int(args.graph_heads),
        graph_max_subgraph_nodes=int(args.graph_max_subgraph_nodes),
        graph_use_onehop=bool(args.graph_use_onehop),
        cnn_channels=int(args.cnn_channels),
        cnn_kernel_size=int(args.cnn_kernel_size),
        cnn_norm=str(args.cnn_norm),
        enable_early_stage_specialist_head=bool(args.enable_early_stage_specialist_head),
        early_stage_round_cutoff=int(args.early_stage_round_cutoff),
    ).to(device)
    early_stage_slate_top_posterior_k = None if int(args.early_stage_slate_top_posterior_k) < 0 else int(args.early_stage_slate_top_posterior_k)
    early_stage_slate_high_disagreement_k = None if int(args.early_stage_slate_high_disagreement_k) < 0 else int(args.early_stage_slate_high_disagreement_k)
    early_stage_slate_novelty_k = None if int(args.early_stage_slate_novelty_k) < 0 else int(args.early_stage_slate_novelty_k)

    teacher_eval = run_policy_on_cases(
        cases=exact_runtime["cases"],
        family=teacher_family,
        runtime=exact_runtime,
        env=env,
        policy_name="teacher",
        model=None,
        deterministic=True,
        base_seed=int(args.seed),
        include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
        rl_policy_mode=str(args.rl_policy_mode),
        corrective_candidate_topk=int(args.corrective_candidate_topk),
        ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
        ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
        device=device,
        collect_transitions=False,
        slate_size=int(args.slate_size),
        slate_top_posterior_k=int(args.slate_top_posterior_k),
        slate_high_disagreement_k=int(args.slate_high_disagreement_k),
        slate_novelty_k=int(args.slate_novelty_k),
        early_stage_round_cutoff=int(args.early_stage_round_cutoff),
        early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
        early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
        early_stage_slate_novelty_k=early_stage_slate_novelty_k,
    )

    teacher_summary = summarize_case_metrics(
        teacher_eval["case_rows"],
        num_rounds=int(args.num_rounds),
        action_budget=int(args.actions_per_round),
    )

    pd.DataFrame(teacher_eval["case_rows"]).to_csv(output_dir / "teacher_exact136_case_rows.csv", index=False)
    pd.DataFrame(teacher_eval["step_rows"]).to_csv(output_dir / "teacher_exact136_step_rows.csv", index=False)

    load_bc_checkpoint = str(args.load_bc_checkpoint).strip()
    save_bc_checkpoint = str(args.save_bc_checkpoint).strip()
    save_bc_epoch_ckpt_dir: Optional[Path] = None
    save_bc_epoch_ckpt_arg = str(args.save_bc_epoch_checkpoints_dir).strip()
    if save_bc_epoch_ckpt_arg:
        save_bc_epoch_ckpt_dir = Path(save_bc_epoch_ckpt_arg)
        save_bc_epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if load_bc_checkpoint:
        checkpoint_path = Path(load_bc_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {checkpoint_path}")
        load_checkpoint_with_specialist_compat(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    bc_history: List[Dict[str, float]] = []
    bc_transitions: List[Dict[str, Any]] = []
    if str(args.rl_init_mode) == "teacher_warm_start" and not bool(args.skip_bc_train) and not load_bc_checkpoint:
        teacher_train_with_transitions = run_policy_on_cases(
            cases=train_runtime["cases"],
            family=teacher_family,
            runtime=train_runtime,
            env=env,
            policy_name="teacher",
            model=None,
            deterministic=True,
            base_seed=int(args.seed),
            include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
            rl_policy_mode=str(args.rl_policy_mode),
            corrective_candidate_topk=int(args.corrective_candidate_topk),
            ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
            ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
            device=device,
            collect_transitions=True,
            early_stage_round_cutoff=int(args.early_stage_round_cutoff),
            early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
            early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
            early_stage_slate_novelty_k=early_stage_slate_novelty_k,
        )
        bc_transitions = list(teacher_train_with_transitions["transitions"])
        bc_history = train_behavior_cloning(
            model=model,
            transitions=bc_transitions,
            device=device,
            epochs=int(args.bc_epochs),
            lr=float(args.bc_lr),
            batch_size=int(args.bc_batch_size),
            action_budget=int(args.actions_per_round),
            grad_clip=float(args.grad_clip),
            save_epoch_checkpoints_dir=save_bc_epoch_ckpt_dir,
            checkpoint_prefix="bc_main_epoch",
        )

    bc_eval_policy = "bc_student" if str(args.rl_init_mode) == "teacher_warm_start" else "random_init_student_pre_rl"
    bc_eval = run_policy_on_cases(
        cases=exact_runtime["cases"],
        family=teacher_family,
        runtime=exact_runtime,
        env=env,
        policy_name=bc_eval_policy,
        model=model,
        deterministic=True,
        base_seed=int(args.seed),
        include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
        rl_policy_mode=str(args.rl_policy_mode),
        corrective_candidate_topk=int(args.corrective_candidate_topk),
        ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
        ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
        device=device,
        collect_transitions=False,
        early_stage_round_cutoff=int(args.early_stage_round_cutoff),
        early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
        early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
        early_stage_slate_novelty_k=early_stage_slate_novelty_k,
    )

    bc_summary = summarize_case_metrics(
        bc_eval["case_rows"],
        num_rounds=int(args.num_rounds),
        action_budget=int(args.actions_per_round),
    )

    teacher_sr = float(teacher_summary["success_rate"])
    bc_sr = float(bc_summary["success_rate"])
    bc_ratio = float(bc_sr / max(teacher_sr, 1e-9))

    if (
        str(args.rl_init_mode) == "teacher_warm_start"
        and bc_ratio < 0.95
        and int(args.bc_recovery_epochs) > 0
        and len(bc_transitions) > 0
    ):
        recovery_history = train_behavior_cloning(
            model=model,
            transitions=bc_transitions,
            device=device,
            epochs=int(args.bc_recovery_epochs),
            lr=float(args.bc_lr),
            batch_size=int(args.bc_batch_size),
            action_budget=int(args.actions_per_round),
            grad_clip=float(args.grad_clip),
            save_epoch_checkpoints_dir=save_bc_epoch_ckpt_dir,
            checkpoint_prefix="bc_recovery_epoch",
        )
        for row in recovery_history:
            row["phase"] = "recovery"
        for row in bc_history:
            row["phase"] = "main"
        bc_history.extend(recovery_history)

        bc_eval = run_policy_on_cases(
            cases=exact_runtime["cases"],
            family=teacher_family,
            runtime=exact_runtime,
            env=env,
            policy_name="bc_student",
            model=model,
            deterministic=True,
            base_seed=int(args.seed),
            include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
            rl_policy_mode=str(args.rl_policy_mode),
            corrective_candidate_topk=int(args.corrective_candidate_topk),
            ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
            ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
            device=device,
            collect_transitions=False,
            slate_size=int(args.slate_size),
            slate_top_posterior_k=int(args.slate_top_posterior_k),
            slate_high_disagreement_k=int(args.slate_high_disagreement_k),
            slate_novelty_k=int(args.slate_novelty_k),
            early_stage_round_cutoff=int(args.early_stage_round_cutoff),
            early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
            early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
            early_stage_slate_novelty_k=early_stage_slate_novelty_k,
        )
        bc_summary = summarize_case_metrics(
            bc_eval["case_rows"],
            num_rounds=int(args.num_rounds),
            action_budget=int(args.actions_per_round),
        )
        bc_sr = float(bc_summary["success_rate"])
        bc_ratio = float(bc_sr / max(teacher_sr, 1e-9))

    if save_bc_checkpoint:
        checkpoint_path = Path(save_bc_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    pd.DataFrame(bc_eval["case_rows"]).to_csv(output_dir / "bc_exact136_case_rows.csv", index=False)
    pd.DataFrame(bc_eval["step_rows"]).to_csv(output_dir / "bc_exact136_step_rows.csv", index=False)
    pd.DataFrame(bc_history).to_csv(output_dir / "bc_train_history.csv", index=False)
    bc_state_dict = deepcopy(model.state_dict())

    rl_summary = None
    rl_eval = {"case_rows": [], "step_rows": []}
    rl_history: List[Dict[str, float]] = []
    last_warm_decode_stats = None
    last_train_decode_stats = None

    if str(args.rl_init_mode) == "random_init" or bc_ratio >= 0.95:
        rl_optimizer = torch.optim.Adam(model.parameters(), lr=float(args.rl_lr))
        rl_epoch_ckpt_dir: Optional[Path] = None
        save_rl_epoch_ckpt_dir = str(args.save_rl_epoch_checkpoints_dir).strip()
        if save_rl_epoch_ckpt_dir:
            rl_epoch_ckpt_dir = Path(save_rl_epoch_ckpt_dir)
            rl_epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_state_dict = deepcopy(model.state_dict())
        best_heldout_sr = -1.0
        best_epoch = 0
        stale_epochs = 0
        baseline_returns_lookup: Optional[Dict[str, Dict[int, float]]] = None
        if str(args.advantage_baseline) == "greedy_relative":
            greedy_rollout = run_policy_on_cases(
                cases=train_runtime["cases"],
                family=teacher_family,
                runtime=train_runtime,
                env=env,
                policy_name="teacher",
                model=None,
                deterministic=True,
                base_seed=int(args.seed),
                include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
                rl_policy_mode=str(args.rl_policy_mode),
                corrective_candidate_topk=int(args.corrective_candidate_topk),
                ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
                ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
                device=device,
                collect_transitions=True,
                slate_size=int(args.slate_size),
                slate_top_posterior_k=int(args.slate_top_posterior_k),
                slate_high_disagreement_k=int(args.slate_high_disagreement_k),
                slate_novelty_k=int(args.slate_novelty_k),
                early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
                early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
                early_stage_slate_novelty_k=early_stage_slate_novelty_k,
            )
            baseline_returns_lookup = _build_case_return_lookup(
                greedy_rollout["transitions"],
                gamma=float(args.rl_gamma),
            )

        if int(args.critic_warmup_epochs) > 0:
            for warm_epoch in range(1, int(args.critic_warmup_epochs) + 1):
                warm_rollout = run_policy_on_cases(
                    cases=train_runtime["cases"],
                    family=teacher_family,
                    runtime=train_runtime,
                    env=env,
                    policy_name="rl_student",
                    model=model,
                    deterministic=False,
                    base_seed=int(args.seed) + 70000 + int(warm_epoch) * 100,
                    include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
                    rl_policy_mode=str(args.rl_policy_mode),
                    corrective_candidate_topk=int(args.corrective_candidate_topk),
                    ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
                    ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
                    device=device,
                    collect_transitions=True,
                    slate_size=int(args.slate_size),
                    slate_top_posterior_k=int(args.slate_top_posterior_k),
                    slate_high_disagreement_k=int(args.slate_high_disagreement_k),
                    slate_novelty_k=int(args.slate_novelty_k),
                    early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                    early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
                    early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
                    early_stage_slate_novelty_k=early_stage_slate_novelty_k,
                    decode_consistency_mode=str(args.decode_consistency_mode),
                    decode_consistency_beam_width=int(args.decode_consistency_beam_width),
                    decode_consistency_topk_per_slot=int(args.decode_consistency_topk_per_slot),
                    decode_consistency_logprob_weight=float(args.decode_consistency_logprob_weight),
                    decode_consistency_value_weight=float(args.decode_consistency_value_weight),
                    decode_consistency_max_margin=float(args.decode_consistency_max_margin),
                    decode_consistency_max_round=int(args.decode_consistency_max_round),
                    decode_consistency_min_entropy=float(args.decode_consistency_min_entropy),
                    decode_consistency_max_preview_candidates=int(args.decode_consistency_max_preview_candidates),
                    decode_consistency_max_trigger_states=int(args.decode_consistency_max_trigger_states),
                )
                warm_transitions = list(warm_rollout["transitions"])
                last_warm_decode_stats = warm_rollout.get("decode_consistency_stats", None)
                _prepare_ppo_targets(
                    warm_transitions,
                    gamma=float(args.rl_gamma),
                    baseline_returns=baseline_returns_lookup,
                )
                warm_stats = ppo_update(
                    model=model,
                    transitions=warm_transitions,
                    optimizer=rl_optimizer,
                    device=device,
                    action_budget=int(args.actions_per_round),
                    clip_range=float(args.rl_clip_range),
                    value_coef=float(args.rl_value_coef),
                    entropy_coef=0.0,
                    imitation_anchor_weight=0.0,
                    update_epochs=int(args.rl_update_epochs),
                    minibatch_size=int(args.rl_minibatch_size),
                    grad_clip=float(args.grad_clip),
                    actor_loss_coef=0.0,
                    ambiguity_weighting_mode=str(args.ambiguity_weighting_mode),
                    ambiguity_weight_alpha=float(args.ambiguity_weight_alpha),
                    ambiguity_entropy_threshold=float(args.ambiguity_entropy_threshold),
                    ambiguity_margin_threshold=float(args.ambiguity_margin_threshold),
                    ambiguity_entropy_temp=float(args.ambiguity_entropy_temp),
                    ambiguity_margin_temp=float(args.ambiguity_margin_temp),
                    early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                    early_stage_set_aux_weight=float(args.early_stage_set_aux_weight),
                    early_stage_set_adv_mode=str(args.early_stage_set_adv_mode),
                    early_stage_set_aux_ambiguity_mode=str(args.early_stage_set_aux_ambiguity_mode),
                    early_stage_set_aux_ambiguity_alpha=float(args.early_stage_set_aux_ambiguity_alpha),
                    decode_consistency_mode=str(args.decode_consistency_mode),
                    decode_consistency_weight=float(args.decode_consistency_weight),
                    decode_consistency_margin=float(args.decode_consistency_margin),
                )
                rl_history.append(
                    {
                        "epoch": float(-int(warm_epoch)),
                        "phase": "critic_warmup",
                        "train_success_rate": None,
                        "train_avg_hit_round_conditional": None,
                        "heldout_success_rate": None,
                        "heldout_avg_hit_round_conditional": None,
                        "heldout_teacher_exact_match_rate": None,
                        "heldout_action_agreement_rate": None,
                        "heldout_ambiguity_gate_trigger_rate": None,
                        "heldout_corrective_third_replace_rate": None,
                        "entropy_coef": 0.0,
                        "imitation_anchor": 0.0,
                        "advantage_baseline": str(args.advantage_baseline),
                        **warm_stats,
                    }
                )

        for epoch in range(1, int(args.rl_epochs) + 1):
            entropy_coef = _lin_anneal(
                float(args.rl_entropy_coef_start),
                float(args.rl_entropy_coef_end),
                int(epoch),
                int(args.rl_epochs),
            )
            imitation_anchor = _lin_anneal(
                float(args.rl_imitation_anchor_start),
                float(args.rl_imitation_anchor_end),
                int(epoch),
                int(args.rl_epochs),
            )

            train_rollout = run_policy_on_cases(
                cases=train_runtime["cases"],
                family=teacher_family,
                runtime=train_runtime,
                env=env,
                policy_name="rl_student",
                model=model,
                deterministic=False,
                base_seed=int(args.seed) + int(epoch) * 100,
                include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
                rl_policy_mode=str(args.rl_policy_mode),
                corrective_candidate_topk=int(args.corrective_candidate_topk),
                ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
                ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
                device=device,
                collect_transitions=True,
                slate_size=int(args.slate_size),
                slate_top_posterior_k=int(args.slate_top_posterior_k),
                slate_high_disagreement_k=int(args.slate_high_disagreement_k),
                slate_novelty_k=int(args.slate_novelty_k),
                early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
                early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
                early_stage_slate_novelty_k=early_stage_slate_novelty_k,
                decode_consistency_mode=str(args.decode_consistency_mode),
                decode_consistency_beam_width=int(args.decode_consistency_beam_width),
                decode_consistency_topk_per_slot=int(args.decode_consistency_topk_per_slot),
                decode_consistency_logprob_weight=float(args.decode_consistency_logprob_weight),
                decode_consistency_value_weight=float(args.decode_consistency_value_weight),
                decode_consistency_max_margin=float(args.decode_consistency_max_margin),
                decode_consistency_max_round=int(args.decode_consistency_max_round),
                decode_consistency_min_entropy=float(args.decode_consistency_min_entropy),
                decode_consistency_max_preview_candidates=int(args.decode_consistency_max_preview_candidates),
                decode_consistency_max_trigger_states=int(args.decode_consistency_max_trigger_states),
            )
            transitions = list(train_rollout["transitions"])
            last_train_decode_stats = train_rollout.get("decode_consistency_stats", None)
            _prepare_ppo_targets(
                transitions,
                gamma=float(args.rl_gamma),
                baseline_returns=baseline_returns_lookup,
            )

            ppo_stats = ppo_update(
                model=model,
                transitions=transitions,
                optimizer=rl_optimizer,
                device=device,
                action_budget=int(args.actions_per_round),
                clip_range=float(args.rl_clip_range),
                value_coef=float(args.rl_value_coef),
                entropy_coef=float(entropy_coef),
                imitation_anchor_weight=float(imitation_anchor),
                update_epochs=int(args.rl_update_epochs),
                minibatch_size=int(args.rl_minibatch_size),
                grad_clip=float(args.grad_clip),
                actor_loss_coef=1.0,
                ambiguity_weighting_mode=str(args.ambiguity_weighting_mode),
                ambiguity_weight_alpha=float(args.ambiguity_weight_alpha),
                ambiguity_entropy_threshold=float(args.ambiguity_entropy_threshold),
                ambiguity_margin_threshold=float(args.ambiguity_margin_threshold),
                ambiguity_entropy_temp=float(args.ambiguity_entropy_temp),
                ambiguity_margin_temp=float(args.ambiguity_margin_temp),
                early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                early_stage_set_aux_weight=float(args.early_stage_set_aux_weight),
                early_stage_set_adv_mode=str(args.early_stage_set_adv_mode),
                early_stage_set_aux_ambiguity_mode=str(args.early_stage_set_aux_ambiguity_mode),
                early_stage_set_aux_ambiguity_alpha=float(args.early_stage_set_aux_ambiguity_alpha),
                decode_consistency_mode=str(args.decode_consistency_mode),
                decode_consistency_weight=float(args.decode_consistency_weight),
                decode_consistency_margin=float(args.decode_consistency_margin),
            )
            if int(args.rl_critic_extra_updates) > 0:
                for _ in range(int(args.rl_critic_extra_updates)):
                    extra_stats = ppo_update(
                        model=model,
                        transitions=transitions,
                        optimizer=rl_optimizer,
                        device=device,
                        action_budget=int(args.actions_per_round),
                        clip_range=float(args.rl_clip_range),
                        value_coef=float(args.rl_value_coef),
                        entropy_coef=0.0,
                        imitation_anchor_weight=0.0,
                        update_epochs=int(args.rl_update_epochs),
                        minibatch_size=int(args.rl_minibatch_size),
                        grad_clip=float(args.grad_clip),
                        actor_loss_coef=0.0,
                        ambiguity_weighting_mode=str(args.ambiguity_weighting_mode),
                        ambiguity_weight_alpha=float(args.ambiguity_weight_alpha),
                        ambiguity_entropy_threshold=float(args.ambiguity_entropy_threshold),
                        ambiguity_margin_threshold=float(args.ambiguity_margin_threshold),
                        ambiguity_entropy_temp=float(args.ambiguity_entropy_temp),
                        ambiguity_margin_temp=float(args.ambiguity_margin_temp),
                        early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                        early_stage_set_aux_weight=float(args.early_stage_set_aux_weight),
                        early_stage_set_adv_mode=str(args.early_stage_set_adv_mode),
                        early_stage_set_aux_ambiguity_mode=str(args.early_stage_set_aux_ambiguity_mode),
                        early_stage_set_aux_ambiguity_alpha=float(args.early_stage_set_aux_ambiguity_alpha),
                        decode_consistency_mode=str(args.decode_consistency_mode),
                        decode_consistency_weight=float(args.decode_consistency_weight),
                        decode_consistency_margin=float(args.decode_consistency_margin),
                    )
                ppo_stats["ppo_value_loss"] = float(extra_stats.get("ppo_value_loss", ppo_stats["ppo_value_loss"]))

            train_summary = summarize_case_metrics(
                train_rollout["case_rows"],
                num_rounds=int(args.num_rounds),
                action_budget=int(args.actions_per_round),
            )

            heldout_epoch_eval = run_policy_on_cases(
                cases=exact_runtime["cases"],
                family=teacher_family,
                runtime=exact_runtime,
                env=env,
                policy_name="rl_student",
                model=model,
                deterministic=True,
                base_seed=int(args.seed),
                include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
                rl_policy_mode=str(args.rl_policy_mode),
                corrective_candidate_topk=int(args.corrective_candidate_topk),
                ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
                ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
                device=device,
                collect_transitions=False,
                slate_size=int(args.slate_size),
                slate_top_posterior_k=int(args.slate_top_posterior_k),
                slate_high_disagreement_k=int(args.slate_high_disagreement_k),
                slate_novelty_k=int(args.slate_novelty_k),
                early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
                early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
                early_stage_slate_novelty_k=early_stage_slate_novelty_k,
            )
            heldout_summary = summarize_case_metrics(
                heldout_epoch_eval["case_rows"],
                num_rounds=int(args.num_rounds),
                action_budget=int(args.actions_per_round),
            )
            heldout_sr = float(heldout_summary["success_rate"])

            rl_history.append(
                {
                    "epoch": float(epoch),
                    "train_success_rate": float(train_summary["success_rate"]),
                    "train_avg_hit_round_conditional": (
                        None
                        if train_summary["avg_hit_round_conditional"] is None
                        else float(train_summary["avg_hit_round_conditional"])
                    ),
                    "heldout_success_rate": float(heldout_sr),
                    "heldout_avg_hit_round_conditional": (
                        None
                        if heldout_summary["avg_hit_round_conditional"] is None
                        else float(heldout_summary["avg_hit_round_conditional"])
                    ),
                    "heldout_teacher_exact_match_rate": (
                        None
                        if heldout_summary["teacher_exact_match_rate_mean"] is None
                        else float(heldout_summary["teacher_exact_match_rate_mean"])
                    ),
                    "heldout_action_agreement_rate": (
                        None
                        if heldout_summary["action_agreement_rate_mean"] is None
                        else float(heldout_summary["action_agreement_rate_mean"])
                    ),
                    "heldout_ambiguity_gate_trigger_rate": (
                        None
                        if heldout_summary["ambiguity_gate_trigger_rate_mean"] is None
                        else float(heldout_summary["ambiguity_gate_trigger_rate_mean"])
                    ),
                    "heldout_corrective_third_replace_rate": (
                        None
                        if heldout_summary["corrective_third_replace_rate_mean"] is None
                        else float(heldout_summary["corrective_third_replace_rate_mean"])
                    ),
                    "entropy_coef": float(entropy_coef),
                    "imitation_anchor": float(imitation_anchor),
                    "advantage_baseline": str(args.advantage_baseline),
                    **ppo_stats,
                }
            )
            if heldout_sr > (best_heldout_sr + 1e-12):
                best_heldout_sr = float(heldout_sr)
                best_epoch = int(epoch)
                stale_epochs = 0
                best_state_dict = deepcopy(model.state_dict())
            else:
                stale_epochs += 1
            if rl_epoch_ckpt_dir is not None:
                torch.save(model.state_dict(), rl_epoch_ckpt_dir / f"epoch_{int(epoch):03d}.pt")
            if int(args.rl_early_stop_patience) > 0 and stale_epochs >= int(args.rl_early_stop_patience):
                break

        if best_epoch > 0:
            model.load_state_dict(best_state_dict)

        rl_eval = run_policy_on_cases(
            cases=exact_runtime["cases"],
            family=teacher_family,
            runtime=exact_runtime,
            env=env,
            policy_name="rl_student",
            model=model,
            deterministic=True,
            base_seed=int(args.seed),
            include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
            rl_policy_mode=str(args.rl_policy_mode),
            corrective_candidate_topk=int(args.corrective_candidate_topk),
            ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
            ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
            device=device,
            collect_transitions=False,
            slate_size=int(args.slate_size),
            slate_top_posterior_k=int(args.slate_top_posterior_k),
            slate_high_disagreement_k=int(args.slate_high_disagreement_k),
            slate_novelty_k=int(args.slate_novelty_k),
            early_stage_round_cutoff=int(args.early_stage_round_cutoff),
            early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
            early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
            early_stage_slate_novelty_k=early_stage_slate_novelty_k,
        )

        rl_summary = summarize_case_metrics(
            rl_eval["case_rows"],
            num_rounds=int(args.num_rounds),
            action_budget=int(args.actions_per_round),
        )
        if rl_summary is not None:
            rl_summary["selected_by_epoch"] = int(best_epoch) if best_epoch > 0 else None
            rl_summary["selected_by_heldout_success_rate"] = float(best_heldout_sr) if best_epoch > 0 else None

    pd.DataFrame(rl_eval["case_rows"]).to_csv(output_dir / "rl_exact136_case_rows.csv", index=False)
    pd.DataFrame(rl_eval["step_rows"]).to_csv(output_dir / "rl_exact136_step_rows.csv", index=False)
    pd.DataFrame(rl_history).to_csv(output_dir / "rl_train_history.csv", index=False)

    save_final_checkpoint = str(args.save_final_checkpoint).strip()
    if save_final_checkpoint:
        final_checkpoint_path = Path(save_final_checkpoint)
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_checkpoint_path)

    policy_rows = [
        {"policy_name": "teacher", **teacher_summary},
        {"policy_name": "bc_student", **bc_summary},
    ]
    if rl_summary is not None:
        policy_rows.append({"policy_name": "rl_student", **rl_summary})
    policy_summary_df = pd.DataFrame(policy_rows)
    policy_summary_df.to_csv(output_dir / "policy_summary.csv", index=False)

    b60_summaries: Dict[str, Any] = {}
    if bool(args.eval_b60):
        b60_runtime = dict(exact_runtime)
        b60_runtime["num_episodes"] = int(args.b60_num_rounds)
        b60_runtime["action_budget"] = int(args.actions_per_round)
        teacher_b60_eval = run_policy_on_cases(
            cases=exact_runtime["cases"],
            family=teacher_family,
            runtime=b60_runtime,
            env=env,
            policy_name="teacher",
            model=None,
            deterministic=True,
            base_seed=int(args.seed),
            include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
            rl_policy_mode=str(args.rl_policy_mode),
            corrective_candidate_topk=int(args.corrective_candidate_topk),
            ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
            ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
            device=device,
            collect_transitions=False,
            slate_size=int(args.slate_size),
            slate_top_posterior_k=int(args.slate_top_posterior_k),
            slate_high_disagreement_k=int(args.slate_high_disagreement_k),
            slate_novelty_k=int(args.slate_novelty_k),
            early_stage_round_cutoff=int(args.early_stage_round_cutoff),
            early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
            early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
            early_stage_slate_novelty_k=early_stage_slate_novelty_k,
        )
        bc_eval_model = SpimNativePolicy(
            global_dim=len(global_feature_names),
            local_dim=len(local_feature_names),
            hidden_dim=int(args.hidden_dim),
            policy_arch=str(args.policy_arch),
            policy_mlp_depth=int(args.policy_mlp_depth),
            value_mlp_depth=int(args.value_mlp_depth),
            value_head_width_mult=float(args.value_head_width_mult),
            critic_trunk_depth=int(args.critic_trunk_depth),
            critic_trunk_hidden_dim=int(args.critic_trunk_hidden_dim),
            policy_dropout=float(args.policy_dropout),
            policy_norm=str(args.policy_norm),
            candidate_encoder=str(args.candidate_encoder),
            candidate_attn_heads=int(args.candidate_attn_heads),
            enable_regime_head=bool(args.enable_regime_head),
            regime_head_classes=int(args.regime_head_classes),
            regime_embed_dim=int(args.regime_embed_dim),
            arch_backbone=str(args.arch_backbone),
            residual_hidden_dim=int(args.residual_hidden_dim),
            residual_depth=int(args.residual_depth),
            residual_head_dim=int(args.residual_head_dim),
            transformer_token_dim=int(args.transformer_token_dim),
            transformer_layers=int(args.transformer_layers),
            transformer_heads=int(args.transformer_heads),
            transformer_ffn_dim=int(args.transformer_ffn_dim),
            graph_hidden_dim=int(args.graph_hidden_dim),
            graph_layers=int(args.graph_layers),
            graph_heads=int(args.graph_heads),
            graph_max_subgraph_nodes=int(args.graph_max_subgraph_nodes),
            graph_use_onehop=bool(args.graph_use_onehop),
            cnn_channels=int(args.cnn_channels),
            cnn_kernel_size=int(args.cnn_kernel_size),
            cnn_norm=str(args.cnn_norm),
        ).to(device)
        bc_eval_model.load_state_dict(bc_state_dict)
        bc_b60_eval = run_policy_on_cases(
            cases=exact_runtime["cases"],
            family=teacher_family,
            runtime=b60_runtime,
            env=env,
            policy_name="bc_student",
            model=bc_eval_model,
            deterministic=True,
            base_seed=int(args.seed),
            include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
            rl_policy_mode=str(args.rl_policy_mode),
            corrective_candidate_topk=int(args.corrective_candidate_topk),
            ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
            ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
            device=device,
            collect_transitions=False,
            slate_size=int(args.slate_size),
            slate_top_posterior_k=int(args.slate_top_posterior_k),
            slate_high_disagreement_k=int(args.slate_high_disagreement_k),
            slate_novelty_k=int(args.slate_novelty_k),
        )
        b60_summaries["teacher"] = summarize_case_metrics(
            teacher_b60_eval["case_rows"],
            num_rounds=int(args.b60_num_rounds),
            action_budget=int(args.actions_per_round),
        )
        b60_summaries["bc_student"] = summarize_case_metrics(
            bc_b60_eval["case_rows"],
            num_rounds=int(args.b60_num_rounds),
            action_budget=int(args.actions_per_round),
        )
        if rl_summary is not None:
            rl_b60_eval = run_policy_on_cases(
                cases=exact_runtime["cases"],
                family=teacher_family,
                runtime=b60_runtime,
                env=env,
                policy_name="rl_student",
                model=model,
                deterministic=True,
                base_seed=int(args.seed),
                include_surrogate_features=bool(args.include_surrogate_features),
            include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
                rl_policy_mode=str(args.rl_policy_mode),
                corrective_candidate_topk=int(args.corrective_candidate_topk),
                ambiguity_top1_top2_max=float(args.ambiguity_top1_top2_max),
                ambiguity_min_candidate_count=int(args.ambiguity_min_candidate_count),
                device=device,
                collect_transitions=False,
                slate_size=int(args.slate_size),
                slate_top_posterior_k=int(args.slate_top_posterior_k),
                slate_high_disagreement_k=int(args.slate_high_disagreement_k),
                slate_novelty_k=int(args.slate_novelty_k),
                early_stage_round_cutoff=int(args.early_stage_round_cutoff),
                early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
                early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
                early_stage_slate_novelty_k=early_stage_slate_novelty_k,
            )
            b60_summaries["rl_student"] = summarize_case_metrics(
                rl_b60_eval["case_rows"],
                num_rounds=int(args.b60_num_rounds),
                action_budget=int(args.actions_per_round),
            )
        b60_rows = [{"policy_name": name, **summary} for name, summary in b60_summaries.items()]
        pd.DataFrame(b60_rows).to_csv(output_dir / "policy_summary_b60.csv", index=False)

    round_curve_rows: List[Dict[str, Any]] = []
    budget_curve_rows: List[Dict[str, Any]] = []
    for name, summary in [
        ("teacher", teacher_summary),
        ("bc_student", bc_summary),
        ("rl_student", rl_summary if rl_summary is not None else None),
    ]:
        if summary is None:
            continue
        for row in summary.get("round_curve", []):
            round_curve_rows.append({"policy_name": name, **row})
        for row in summary.get("budget_curve", []):
            budget_curve_rows.append({"policy_name": name, **row})
    pd.DataFrame(round_curve_rows).to_csv(output_dir / "roundwise_success_curve.csv", index=False)
    pd.DataFrame(budget_curve_rows).to_csv(output_dir / "budget_success_curve.csv", index=False)

    teacher_sr = float(teacher_summary["success_rate"])
    bc_sr = float(bc_summary["success_rate"])
    rl_sr = None if rl_summary is None else float(rl_summary["success_rate"])

    claims = {
        "teacher_selected": str(teacher_family),
        "bc_reaches_95pct_teacher": bool((bc_sr / max(teacher_sr, 1e-9)) >= 0.95) if str(args.rl_init_mode) == "teacher_warm_start" else None,
        "pure_task_reward_trainable": bool(rl_summary is not None),
        "rl_reaches_teacher": bool(rl_sr is not None and rl_sr >= teacher_sr),
        "rl_exceeds_teacher": bool(rl_sr is not None and rl_sr > teacher_sr),
        "b60_checked": bool(args.eval_b60),
        "rl_not_hurting_b60_vs_bc": (
            None
            if (not bool(args.eval_b60) or rl_summary is None or "rl_student" not in b60_summaries)
            else bool(float(b60_summaries["rl_student"]["success_rate"]) >= float(b60_summaries["bc_student"]["success_rate"]))
        ),
    }

    summary = {
        "runner_version": RUNNER_VERSION,
        "runner_version_tag": str(args.runner_version_tag),
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "device": str(device),
        "source_root": str(source_root),
        "cache_dir": str(cache_dir),
        "train_full_cache_version": str(args.train_full_cache_version),
        "foundation_graph_path": str(resolve_foundation_graph_path(source_root)),
        "train_full_case_count": int(len(train_runtime["cases"])),
        "exact136_case_count": int(len(exact_runtime["cases"])),
        "teacher_precheck": teacher_decision,
        "teacher_family": str(teacher_family),
        "protocol": {
            "num_rounds": int(args.num_rounds),
            "actions_per_round": int(args.actions_per_round),
            "budget": int(args.num_rounds) * int(args.actions_per_round),
            "success_definition": "direct source hit under B30 budget",
        },
        "reward": {
            "family": str(args.reward_family),
            "hit_reward": float(args.hit_reward),
            "step_penalty": float(args.step_penalty),
            "lambda_cover": float(args.reward_lambda_cover),
            "lambda_error": float(args.reward_lambda_error),
            "cover_delta_clip": float(args.reward_cover_delta_clip),
            "error_delta_clip": float(args.reward_error_delta_clip),
            "topk_fraction": float(args.reward_topk_fraction),
            "time_tol_min": float(args.reward_time_tol_min),
            "formula": "base = 1[source_hit]*hit_reward + sampled_action_count*step_penalty; optional shaping uses clipped/normalized delta_cover and delta_topk_scenario_error",
        },
        "feature_spec": {
            "global_features": list(global_feature_names),
            "local_features": list(local_feature_names),
        },
        "policy_spec": {
            "architecture": str(args.arch_backbone),
            "policy_arch": str(args.policy_arch),
            "policy_mlp_depth": int(args.policy_mlp_depth),
            "value_mlp_depth": int(args.value_mlp_depth),
            "hidden_dim": int(args.hidden_dim),
            "value_head_width_mult": float(args.value_head_width_mult),
            "critic_trunk_depth": int(args.critic_trunk_depth),
            "critic_trunk_hidden_dim": int(args.critic_trunk_hidden_dim),
            "policy_dropout": float(args.policy_dropout),
            "policy_norm": str(args.policy_norm),
            "candidate_encoder": str(args.candidate_encoder),
            "candidate_attn_heads": int(args.candidate_attn_heads),
            "enable_early_stage_specialist_head": bool(args.enable_early_stage_specialist_head),
            "early_stage_round_cutoff": int(args.early_stage_round_cutoff),
            "enable_regime_head": bool(args.enable_regime_head),
            "regime_head_classes": int(args.regime_head_classes),
            "regime_embed_dim": int(args.regime_embed_dim),
            "arch_hparams": {
                "residual_hidden_dim": int(args.residual_hidden_dim),
                "residual_depth": int(args.residual_depth),
                "residual_head_dim": int(args.residual_head_dim),
                "transformer_token_dim": int(args.transformer_token_dim),
                "transformer_layers": int(args.transformer_layers),
                "transformer_heads": int(args.transformer_heads),
                "transformer_ffn_dim": int(args.transformer_ffn_dim),
                "graph_hidden_dim": int(args.graph_hidden_dim),
                "graph_layers": int(args.graph_layers),
                "graph_heads": int(args.graph_heads),
                "graph_max_subgraph_nodes": int(args.graph_max_subgraph_nodes),
                "graph_use_onehop": bool(args.graph_use_onehop),
                "cnn_channels": int(args.cnn_channels),
                "cnn_kernel_size": int(args.cnn_kernel_size),
                "cnn_norm": str(args.cnn_norm),
            },
            "action_parameterization": "sequential masked selection without replacement",
            "teacher_student_rl_consistency": "student policy uses bounded slate->set-level autoregressive 3-pick; teacher remains posterior_greedy full-available baseline",
            "rl_init_mode": str(args.rl_init_mode),
            "rl_policy_mode": str(args.rl_policy_mode),
            "corrective_candidate_topk": int(args.corrective_candidate_topk),
            "ambiguity_gate": "top1_top2_margin <= threshold AND candidate_count >= min_count",
            "ambiguity_top1_top2_max": float(args.ambiguity_top1_top2_max),
            "ambiguity_min_candidate_count": int(args.ambiguity_min_candidate_count),
            "candidate_slate": {
                "slate_size": int(args.slate_size),
                "top_posterior_k": int(args.slate_top_posterior_k),
                "high_disagreement_k": int(args.slate_high_disagreement_k),
                "novelty_k": int(args.slate_novelty_k),
            },
            "early_stage_candidate_slate": {
                "round_cutoff": int(args.early_stage_round_cutoff),
                "top_posterior_k": early_stage_slate_top_posterior_k,
                "high_disagreement_k": early_stage_slate_high_disagreement_k,
                "novelty_k": early_stage_slate_novelty_k,
            },
            "advantage_baseline": str(args.advantage_baseline),
            "ambiguity_weighting": {
                "mode": str(args.ambiguity_weighting_mode),
                "alpha": float(args.ambiguity_weight_alpha),
                "entropy_threshold": float(args.ambiguity_entropy_threshold),
                "margin_threshold": float(args.ambiguity_margin_threshold),
                "entropy_temp": float(args.ambiguity_entropy_temp),
                "margin_temp": float(args.ambiguity_margin_temp),
            },
            "early_stage_set_aux": {
                "weight": float(args.early_stage_set_aux_weight),
                "adv_mode": str(args.early_stage_set_adv_mode),
                "ambiguity_mode": str(args.early_stage_set_aux_ambiguity_mode),
                "ambiguity_alpha": float(args.early_stage_set_aux_ambiguity_alpha),
            },
            "rl_teacher_anchor_start": float(args.rl_imitation_anchor_start),
            "rl_teacher_anchor_end": float(args.rl_imitation_anchor_end),
            "rl_early_stop_patience": int(args.rl_early_stop_patience),
            "critic_warmup_epochs": int(args.critic_warmup_epochs),
            "rl_critic_extra_updates": int(args.rl_critic_extra_updates),
            "decode_consistency": {
                "mode": str(args.decode_consistency_mode),
                "weight": float(args.decode_consistency_weight),
                "margin": float(args.decode_consistency_margin),
                "beam_width": int(args.decode_consistency_beam_width),
                "topk_per_slot": int(args.decode_consistency_topk_per_slot),
                "logprob_weight": float(args.decode_consistency_logprob_weight),
                "value_weight": float(args.decode_consistency_value_weight),
                "max_margin": float(args.decode_consistency_max_margin),
                "max_round": int(args.decode_consistency_max_round),
                "min_entropy": float(args.decode_consistency_min_entropy),
                "max_preview_candidates": int(args.decode_consistency_max_preview_candidates),
                "max_trigger_states": int(args.decode_consistency_max_trigger_states),
            },
        },
        "teacher_summary": teacher_summary,
        "bc_summary": bc_summary,
        "rl_summary": rl_summary,
        "b60_summaries": b60_summaries if bool(args.eval_b60) else None,
        "imitation_gap": {
            "bc_minus_teacher_sr": float(bc_sr - teacher_sr),
            "bc_over_teacher_ratio": float(bc_sr / max(teacher_sr, 1e-9)),
            "bc_teacher_match_rate": float(bc_summary["teacher_exact_match_rate_mean"]),
        },
        "rl_gap": {
            "rl_minus_teacher_sr": None if rl_sr is None else float(rl_sr - teacher_sr),
            "rl_over_teacher_ratio": None if rl_sr is None else float(rl_sr / max(teacher_sr, 1e-9)),
            "rl_teacher_match_rate": None if rl_summary is None else float(rl_summary["teacher_exact_match_rate_mean"]),
        },
        "decode_consistency_stats": {
            "critic_warmup": last_warm_decode_stats,
            "train": last_train_decode_stats,
        },
        "claims": claims,
        "artifacts": {
            "policy_summary": str(output_dir / "policy_summary.csv"),
            "teacher_case_rows": str(output_dir / "teacher_exact136_case_rows.csv"),
            "bc_case_rows": str(output_dir / "bc_exact136_case_rows.csv"),
            "rl_case_rows": str(output_dir / "rl_exact136_case_rows.csv"),
            "round_curve": str(output_dir / "roundwise_success_curve.csv"),
            "budget_curve": str(output_dir / "budget_success_curve.csv"),
            "policy_summary_b60": str(output_dir / "policy_summary_b60.csv") if bool(args.eval_b60) else None,
            "bc_train_history": str(output_dir / "bc_train_history.csv"),
            "rl_train_history": str(output_dir / "rl_train_history.csv"),
            "load_bc_checkpoint": None if not load_bc_checkpoint else str(Path(load_bc_checkpoint)),
            "save_bc_checkpoint": None if not save_bc_checkpoint else str(Path(save_bc_checkpoint)),
            "save_bc_epoch_checkpoints_dir": None if save_bc_epoch_ckpt_dir is None else str(save_bc_epoch_ckpt_dir),
            "save_final_checkpoint": None if not save_final_checkpoint else str(Path(save_final_checkpoint)),
        },
        "code_paths": [
            "src/scripts/run_spim_teacher_imitation_rl_pilot.py",
            "src/scripts/run_spim_family_sweep.py",
            "src/scripts/run_posterior_like_belief_audit.py",
            "src/scripts/run_reasoner_same_case_stronger_source_overfit.py",
            "src/scripts/audit/utils_practical_rollout.py",
            "src/modeling/evidence/dynamic_reachability.py",
        ],
        "train_cfg_path": str(train_cfg_path),
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
