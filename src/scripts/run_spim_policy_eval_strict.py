from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
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
    _belief_metrics,
    _build_clean_candidate_mask,
    _compute_scenario_error,
    _extract_trigger_global,
    _pick_topk_unsampled,
    _soft_scenario_posterior,
)
from src.scripts.run_spim_policy_eval import build_runtime, summarize_case_metrics, write_json
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_PRECHECK_ROOT,
    DEFAULT_SOURCE_ROOT,
    EXTENDED_SOFT_SCENARIO_FAMILIES,
    SpimNativePolicy,
    build_controlled_slate_mask,
    get_global_feature_names,
    get_local_feature_names,
    _case_rollout_seed,
    _clip_unit,
    _compute_reward_by_family,
    _mean_topk_scenario_error,
    _lin_anneal,
    auto_select_teacher,
    build_spim_native_state,
    compute_teacher_belief,
    get_device,
    resolve_onset_grid,
    seed_everything,
)

RUNNER_VERSION = "spim_policy_eval_strict_v1"
PANEL_VERSION = "spim_policy_eval_strict_cross_split_v1"

MODE_TO_BRANCH = {
    "teacher": "teacher_algorithmic",
    "teacher_slate": "teacher_algorithmic_slate_restricted",
    "bc": "student_checkpoint_bc",
    "rl": "student_checkpoint_rl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict SPIM policy eval runner with mode/name separation and checkpoint gating."
    )
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--precheck-root", type=str, default=str(DEFAULT_PRECHECK_ROOT))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--split", type=str, default="val", choices=["exact136", "train", "val", "test"])
    parser.add_argument("--train-max-cases", type=int, default=0)
    parser.add_argument("--train-cache-version", type=str, default="")
    parser.add_argument("--case-limit", type=int, default=8)
    parser.add_argument("--trace-case-limit", type=int, default=2)
    parser.add_argument("--trace-step-limit", type=int, default=2)

    parser.add_argument("--policy-mode", type=str, default="teacher", choices=["teacher", "teacher_slate", "bc", "rl"])
    parser.add_argument("--policy-name", type=str, default="strict_eval")
    parser.add_argument("--checkpoint", type=str, default="")
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
    parser.add_argument("--use-uncertainty-regime-features", action="store_true")
    parser.add_argument("--slate-size", type=int, default=10)
    parser.add_argument("--slate-top-posterior-k", type=int, default=6)
    parser.add_argument("--slate-high-disagreement-k", type=int, default=3)
    parser.add_argument("--slate-novelty-k", type=int, default=2)
    parser.add_argument("--enable-early-stage-specialist-head", action="store_true")
    parser.add_argument("--early-stage-round-cutoff", type=int, default=0)
    parser.add_argument("--early-stage-slate-top-posterior-k", type=int, default=-1)
    parser.add_argument("--early-stage-slate-high-disagreement-k", type=int, default=-1)
    parser.add_argument("--early-stage-slate-novelty-k", type=int, default=-1)
    parser.add_argument("--enable-regime-head", action="store_true")
    parser.add_argument("--regime-head-classes", type=int, default=3)
    parser.add_argument("--regime-embed-dim", type=int, default=12)

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
    parser.add_argument(
        "--decode-mode",
        type=str,
        default="greedy",
        choices=["greedy", "beam_rerank", "sample_rerank"],
    )
    parser.add_argument("--decode-beam-width", type=int, default=4)
    parser.add_argument("--decode-topk-per-slot", type=int, default=4)
    parser.add_argument("--decode-sample-candidates", type=int, default=6)
    parser.add_argument("--decode-sample-topk", type=int, default=6)
    parser.add_argument("--decode-logprob-weight", type=float, default=1.0)
    parser.add_argument("--decode-value-weight", type=float, default=1.0)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cfg_from_source(source_root: Path) -> Path:
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    return Path(oracle_manifest["config_path"])


def _candidate_sequence_logprob(
    *,
    logits: torch.Tensor,
    available_mask: torch.Tensor,
    actions: Sequence[int],
) -> float:
    available = available_mask.view(-1).bool().clone()
    total = 0.0
    for action in list(actions):
        idx = torch.nonzero(available, as_tuple=True)[0]
        if idx.numel() <= 0:
            break
        candidate_logits = logits[idx]
        probs = torch.softmax(candidate_logits, dim=0)
        pos = (idx == int(action)).nonzero(as_tuple=True)[0]
        if pos.numel() != 1:
            return float("-inf")
        total += float(torch.log(probs[pos[0]].clamp_min(1e-12)).item())
        available[int(action)] = False
    return float(total)


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


def _sample_candidate_sets(
    *,
    logits: torch.Tensor,
    available_mask: torch.Tensor,
    action_budget: int,
    num_candidates: int,
    sample_topk: int,
    generator: torch.Generator,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    max_attempts = max(int(num_candidates) * 4, 8)
    while len(out) < int(max(1, num_candidates)) and attempts < max_attempts:
        attempts += 1
        available = available_mask.view(-1).bool().clone()
        actions: List[int] = []
        seq_logprob = 0.0
        for _ in range(int(action_budget)):
            idx = torch.nonzero(available, as_tuple=True)[0]
            if idx.numel() <= 0:
                break
            cand_logits = logits[idx]
            k = min(int(max(1, sample_topk)), int(idx.numel()))
            top_vals, top_pos = torch.topk(cand_logits, k=k, dim=0)
            probs = torch.softmax(top_vals, dim=0)
            sampled = torch.multinomial(probs.detach().cpu(), num_samples=1, replacement=False, generator=generator)
            local = int(sampled.view(-1)[0].item())
            chosen = int(idx[int(top_pos[local].item())].item())
            actions.append(chosen)
            seq_logprob += float(torch.log(probs[local].clamp_min(1e-12)).item())
            available[chosen] = False
        key = tuple(int(v) for v in actions)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"actions": list(key), "seq_logprob": float(seq_logprob), "origin": "sample"})
    return out


def _score_candidate_set(
    *,
    candidate: Dict[str, Any],
    case: CaseRecord,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    env: CleanTwoChannelEvidenceEnv,
    family: str,
    trigger_global: int,
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
) -> Dict[str, Any]:
    tmp_rollout = deepcopy(rollout)
    tmp_history = deepcopy(history)
    tmp_paper_state = PaperLikeHSRState(source_prior=None)
    tmp_rollout.step_with_actions(
        list(candidate["actions"]),
        sample_types=[f"decode_preview_slot_{i}" for i in range(len(candidate["actions"]))],
    )
    if tmp_rollout.history_steps:
        tmp_history.append_from_history_step(tmp_rollout.history_steps[-1])
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
    with torch.no_grad():
        preview = model.act(
            global_features=next_spim_state["global_features"].to(device),
            local_features=next_spim_state["local_features"].to(device),
            available_mask=next_spim_state["available_mask"].to(device),
            action_budget=0,
            deterministic=True,
            generator=None,
            round_index=int(next_spim_state["diagnostics"]["round_index"]),
            graph_bundle={
                "edge_index": next_spim_state["graph_edge_index"].to(device),
                "evidence_nodes": list(next_spim_state["graph_evidence_nodes"]),
            },
        )
    post_value = float(preview["value"].detach().cpu().item())
    score = float(logprob_weight) * float(candidate["seq_logprob"]) + float(value_weight) * post_value
    return {
        **candidate,
        "post_value": post_value,
        "score": score,
        "post_entropy": float(next_spim_state["diagnostics"]["posterior_entropy"]),
        "post_margin": float(next_spim_state["diagnostics"]["top1_top2_margin"]),
        "post_top1_mass": float(next_spim_state["diagnostics"]["top1_mass"]),
    }


def _load_split_cases(
    *,
    cfg_path: Path,
    cache_dir: Path,
    split: str,
    train_max_cases: int,
    train_cache_version: str,
    case_limit: int,
) -> Tuple[List[CaseRecord], Dict[str, Any], Dict[str, Any]]:
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if isinstance(payload.get("life_support"), dict) and str(payload["life_support"].get("profile")) == "custom_direct_edit":
        payload = dict(payload)
        payload["life_support"] = {k: v for k, v in payload["life_support"].items() if k != "profile"}

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
    limit = len(dataset) if int(case_limit) <= 0 else min(int(case_limit), len(dataset))
    for dataset_idx in range(limit):
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
        "case_limit": int(case_limit),
        "loaded_case_count": int(len(cases)),
        "full_dataset_count": int(len(dataset)),
    }
    return cases, assets, meta


def build_runtime_strict(
    *,
    source_root: Path,
    cache_dir: Path,
    split: str,
    num_rounds: int,
    actions_per_round: int,
    train_max_cases: int,
    train_cache_version: str,
    case_limit: int,
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
            "case_limit": int(case_limit),
            "loaded_case_count": int(len(runtime.get("cases", []))),
        }
        if int(case_limit) > 0:
            runtime["cases"] = list(runtime["cases"][: int(case_limit)])
        return runtime, meta

    cfg_path = _load_cfg_from_source(source_root)
    cases, dataset_assets, split_meta = _load_split_cases(
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        split=split,
        train_max_cases=int(train_max_cases),
        train_cache_version=str(train_cache_version),
        case_limit=int(case_limit),
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


def _resolve_mode_branch(policy_mode: str, checkpoint: str) -> Tuple[str, Optional[Path]]:
    if policy_mode not in MODE_TO_BRANCH:
        raise AssertionError(f"Unsupported policy_mode: {policy_mode}")
    if policy_mode in {"teacher", "teacher_slate"}:
        if str(checkpoint).strip():
            raise AssertionError(f"{policy_mode} mode must not accept a checkpoint")
        return MODE_TO_BRANCH[policy_mode], None
    ckpt = Path(str(checkpoint).strip())
    if not str(checkpoint).strip():
        raise AssertionError(f"{policy_mode} mode requires a checkpoint")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return MODE_TO_BRANCH[policy_mode], ckpt


def _load_model_for_mode(
    *,
    policy_mode: str,
    checkpoint: Optional[Path],
    device: torch.device,
    hidden_dim: int,
    include_surrogate_features: bool,
    include_uncertainty_regime_features: bool,
    policy_arch: str,
    policy_mlp_depth: int,
    value_mlp_depth: int,
    value_head_width_mult: float,
    critic_trunk_depth: int,
    critic_trunk_hidden_dim: int,
    policy_dropout: float,
    policy_norm: str,
    candidate_encoder: str,
    candidate_attn_heads: int,
    enable_early_stage_specialist_head: bool,
    early_stage_round_cutoff: int,
    enable_regime_head: bool,
    regime_head_classes: int,
    regime_embed_dim: int,
    arch_backbone: str,
    residual_hidden_dim: int,
    residual_depth: int,
    residual_head_dim: int,
    transformer_token_dim: int,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_dim: int,
    graph_hidden_dim: int,
    graph_layers: int,
    graph_heads: int,
    graph_max_subgraph_nodes: int,
    graph_use_onehop: bool,
    cnn_channels: int,
    cnn_kernel_size: int,
    cnn_norm: str,
) -> Tuple[Optional[SpimNativePolicy], Dict[str, Any]]:
    checkpoint_info: Dict[str, Any] = {
        "path": None if checkpoint is None else str(checkpoint),
        "load_status": "not_applicable" if checkpoint is None else "pending",
        "sha256": None,
        "size_bytes": None,
        "strict_load": False,
    }
    if policy_mode in {"teacher", "teacher_slate"}:
        return None, checkpoint_info

    assert checkpoint is not None
    checkpoint_info["sha256"] = _sha256_file(checkpoint)
    checkpoint_info["size_bytes"] = int(checkpoint.stat().st_size)

    global_feature_names = get_global_feature_names(bool(include_uncertainty_regime_features))
    local_feature_names = get_local_feature_names(
        include_surrogate_features=bool(include_surrogate_features),
        include_uncertainty_regime_features=bool(include_uncertainty_regime_features),
    )

    model = SpimNativePolicy(
        global_dim=len(global_feature_names),
        local_dim=len(local_feature_names),
        hidden_dim=int(hidden_dim),
        policy_arch=str(policy_arch),
        policy_mlp_depth=int(policy_mlp_depth),
        value_mlp_depth=int(value_mlp_depth),
        value_head_width_mult=float(value_head_width_mult),
        critic_trunk_depth=int(critic_trunk_depth),
        critic_trunk_hidden_dim=int(critic_trunk_hidden_dim),
        policy_dropout=float(policy_dropout),
        policy_norm=str(policy_norm),
        candidate_encoder=str(candidate_encoder),
        candidate_attn_heads=int(candidate_attn_heads),
        enable_early_stage_specialist_head=bool(enable_early_stage_specialist_head),
        early_stage_round_cutoff=int(early_stage_round_cutoff),
        enable_regime_head=bool(enable_regime_head),
        regime_head_classes=int(regime_head_classes),
        regime_embed_dim=int(regime_embed_dim),
        arch_backbone=str(arch_backbone),
        residual_hidden_dim=int(residual_hidden_dim),
        residual_depth=int(residual_depth),
        residual_head_dim=int(residual_head_dim),
        transformer_token_dim=int(transformer_token_dim),
        transformer_layers=int(transformer_layers),
        transformer_heads=int(transformer_heads),
        transformer_ffn_dim=int(transformer_ffn_dim),
        graph_hidden_dim=int(graph_hidden_dim),
        graph_layers=int(graph_layers),
        graph_heads=int(graph_heads),
        graph_max_subgraph_nodes=int(graph_max_subgraph_nodes),
        graph_use_onehop=bool(graph_use_onehop),
        cnn_channels=int(cnn_channels),
        cnn_kernel_size=int(cnn_kernel_size),
        cnn_norm=str(cnn_norm),
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    checkpoint_info["load_status"] = "loaded"
    checkpoint_info["strict_load"] = True
    return model, checkpoint_info


def _compute_mode_trace_id(case_id: str, episode_index: int) -> str:
    return f"{case_id}::ep{episode_index}"


def run_case_policy_strict(
    *,
    case: CaseRecord,
    case_index: int,
    family: str,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    policy_mode: str,
    requested_policy_name: str,
    model: Optional[SpimNativePolicy],
    branch_taken: str,
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
    device: torch.device,
    collect_transitions: bool,
    trace_case_limit: int,
    trace_step_limit: int,
    checkpoint_info: Dict[str, Any],
    slate_size: int = 10,
    slate_top_posterior_k: int = 6,
    slate_high_disagreement_k: int = 3,
    slate_novelty_k: int = 2,
    early_stage_round_cutoff: int = 0,
    early_stage_slate_top_posterior_k: Optional[int] = None,
    early_stage_slate_high_disagreement_k: Optional[int] = None,
    early_stage_slate_novelty_k: Optional[int] = None,
    decode_mode: str = "greedy",
    decode_beam_width: int = 4,
    decode_topk_per_slot: int = 4,
    decode_sample_candidates: int = 6,
    decode_sample_topk: int = 6,
    decode_logprob_weight: float = 1.0,
    decode_value_weight: float = 1.0,
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
    step_trace_rows: List[Dict[str, Any]] = []

    hit_round: Optional[int] = None
    hit_sample_index: Optional[int] = None
    budget_used = 0
    termination_reason = "budget_exhausted"
    prev_entropy: Optional[float] = None
    prev2_entropy: Optional[float] = None
    prev_top1_mass: Optional[float] = None
    prev2_top1_mass: Optional[float] = None

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
        if policy_mode == "teacher":
            selected_actions = list(teacher_actions)
            action_source = "teacher_actions"
        elif policy_mode == "teacher_slate":
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
            teacher_candidate_mask = belief_ctx["candidate_mask"].view(-1).bool().cpu() & policy_available_mask.view(-1).bool().cpu()
            selected_actions = _pick_topk_unsampled(
                belief_ctx["belief"],
                teacher_candidate_mask,
                rollout,
                int(runtime["action_budget"]),
            )
            selected_actions = [int(v) for v in selected_actions]
            action_source = "teacher_actions_with_slate"
            if not selected_actions:
                termination_reason = "teacher_no_action"
                break
        else:
            assert model is not None, f"model is required for policy_mode={policy_mode}"
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
            greedy_actions = [int(v) for v in policy_out["actions"]]
            selected_actions = list(greedy_actions)
            greedy_seq_logprob = _candidate_sequence_logprob(
                logits=policy_out["logits"],
                available_mask=policy_available_mask,
                actions=greedy_actions,
            )
            decode_diag: Dict[str, Any] = {
                "decode_mode": str(decode_mode),
                "candidate_count": 1,
                "selected_rank": 1,
                "rewrite": False,
                "selected_equals_greedy": True,
                "greedy_actions": list(greedy_actions),
                "selected_seq_logprob": float(greedy_seq_logprob),
                "selected_post_value": None,
                "selected_score": None,
                "greedy_seq_logprob": float(greedy_seq_logprob),
                "greedy_post_value": None,
                "greedy_score": None,
                "candidate_summary": None,
            }
            if str(decode_mode) != "greedy":
                candidates: List[Dict[str, Any]] = [
                    {"actions": list(greedy_actions), "seq_logprob": float(greedy_seq_logprob), "origin": "greedy"}
                ]
                if str(decode_mode) == "beam_rerank":
                    candidates.extend(
                        _enumerate_beam_candidates(
                            logits=policy_out["logits"],
                            available_mask=policy_available_mask,
                            action_budget=int(runtime["action_budget"]),
                            beam_width=int(decode_beam_width),
                            topk_per_slot=int(decode_topk_per_slot),
                        )
                    )
                else:
                    cand_generator = torch.Generator(device="cpu")
                    cand_generator.manual_seed(_case_rollout_seed(base_seed + 17, case_index, episode_idx))
                    candidates.extend(
                        _sample_candidate_sets(
                            logits=policy_out["logits"],
                            available_mask=policy_available_mask,
                            action_budget=int(runtime["action_budget"]),
                            num_candidates=int(decode_sample_candidates),
                            sample_topk=int(decode_sample_topk),
                            generator=cand_generator,
                        )
                    )
                deduped: List[Dict[str, Any]] = []
                seen = set()
                for cand in candidates:
                    key = tuple(int(v) for v in cand["actions"])
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    deduped.append({"actions": list(key), "seq_logprob": float(cand["seq_logprob"]), "origin": str(cand["origin"])})
                scored = [
                    _score_candidate_set(
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
                        logprob_weight=float(decode_logprob_weight),
                        value_weight=float(decode_value_weight),
                    )
                    for cand in deduped
                ]
                scored.sort(key=lambda row: (float(row["score"]), float(row["seq_logprob"])), reverse=True)
                best = scored[0]
                selected_actions = [int(v) for v in best["actions"]]
                greedy_scored = next((row for row in scored if list(row["actions"]) == list(greedy_actions)), None)
                decode_diag.update(
                    {
                        "candidate_count": int(len(scored)),
                        "rewrite": bool(list(selected_actions) != list(greedy_actions)),
                        "selected_equals_greedy": bool(list(selected_actions) == list(greedy_actions)),
                        "selected_seq_logprob": float(best["seq_logprob"]),
                        "selected_post_value": float(best["post_value"]),
                        "selected_score": float(best["score"]),
                        "greedy_post_value": None if greedy_scored is None else float(greedy_scored["post_value"]),
                        "greedy_score": None if greedy_scored is None else float(greedy_scored["score"]),
                        "candidate_summary": json.dumps(
                            [
                                {
                                    "actions": [int(v) for v in row["actions"]],
                                    "origin": str(row["origin"]),
                                    "seq_logprob": float(row["seq_logprob"]),
                                    "post_value": float(row["post_value"]),
                                    "score": float(row["score"]),
                                }
                                for row in scored[: min(6, len(scored))]
                            ]
                        ),
                    }
                )
            action_source = "model.act"
            if not selected_actions:
                termination_reason = "policy_no_action"
                break

        policy_slate_local_ids = [int(v) for v in torch.nonzero(policy_available_mask.view(-1).bool(), as_tuple=True)[0].tolist()]
        policy_slate_global_ids = [int(rollout.g_ids[int(v)].item()) for v in policy_slate_local_ids]
        selected_global_ids = [int(rollout.g_ids[int(v)].item()) for v in selected_actions]
        regime_id = None if policy_out is None else policy_out.get("regime_id", None)
        regime_probs = None if policy_out is None else policy_out.get("regime_probs", None)
        round_hit = source_local is not None and int(source_local) in set(selected_actions)
        if round_hit and hit_round is None:
            hit_round = int(episode_idx)
            source_slot = selected_actions.index(int(source_local)) + 1
            hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

        rollout.step_with_actions(
            selected_actions,
            sample_types=[f"{requested_policy_name}_slot_{i}" for i in range(len(selected_actions))],
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
        step_row = {
            "case_id": case.case_id,
            "policy_name": str(requested_policy_name),
            "policy_mode": str(policy_mode),
            "branch_taken": str(branch_taken),
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
            "decode_mode": None if policy_out is None else str(decode_diag["decode_mode"]),
            "decode_candidate_count": None if policy_out is None else int(decode_diag["candidate_count"]),
            "decode_selected_rank": None if policy_out is None else int(decode_diag["selected_rank"]),
            "decode_rewrite": None if policy_out is None else float(bool(decode_diag["rewrite"])),
            "decode_selected_equals_greedy": None if policy_out is None else float(bool(decode_diag["selected_equals_greedy"])),
            "decode_greedy_actions": None if policy_out is None else json.dumps([int(v) for v in decode_diag["greedy_actions"]]),
            "decode_selected_seq_logprob": None if policy_out is None else float(decode_diag["selected_seq_logprob"]),
            "decode_selected_post_value": None if policy_out is None or decode_diag["selected_post_value"] is None else float(decode_diag["selected_post_value"]),
            "decode_selected_score": None if policy_out is None or decode_diag["selected_score"] is None else float(decode_diag["selected_score"]),
            "decode_greedy_seq_logprob": None if policy_out is None else float(decode_diag["greedy_seq_logprob"]),
            "decode_greedy_post_value": None if policy_out is None or decode_diag["greedy_post_value"] is None else float(decode_diag["greedy_post_value"]),
            "decode_greedy_score": None if policy_out is None or decode_diag["greedy_score"] is None else float(decode_diag["greedy_score"]),
            "decode_candidate_summary": None if policy_out is None else decode_diag["candidate_summary"],
            **spim_state["diagnostics"],
        }
        step_rows.append(step_row)

        if collect_transitions:
            transitions.append(
                {
                    "global_features": spim_state["global_features"].detach().cpu(),
                    "local_features": spim_state["local_features"].detach().cpu(),
                    "available_mask": policy_available_mask.detach().cpu(),
                    "actions": [int(v) for v in selected_actions],
                    "teacher_actions": [int(v) for v in teacher_actions],
                    "old_log_prob": float(policy_out["log_prob"].detach().cpu().item()) if policy_out is not None else 0.0,
                    "old_value": float(policy_out["value"].detach().cpu().item()) if policy_out is not None else 0.0,
                    "reward": float(step_reward),
                    "done": bool(round_hit),
                    "case_id": str(case.case_id),
                    "episode_index": int(episode_idx),
                }
            )

        if int(case_index) < int(trace_case_limit) and int(episode_idx) <= int(trace_step_limit):
            step_trace_rows.append(
                {
                    "trace_id": _compute_mode_trace_id(case.case_id, episode_idx),
                    "case_index": int(case_index),
                    "case_id": case.case_id,
                    "episode_index": int(episode_idx),
                    "policy_mode": str(policy_mode),
                    "requested_policy_name": str(requested_policy_name),
                    "branch_taken": str(branch_taken),
                    "action_source": str(action_source),
                    "policy_out_present": bool(policy_out is not None),
                    "checkpoint_path": checkpoint_info["path"],
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "selected_local_ids": [int(v) for v in selected_actions],
                    "teacher_local_ids": [int(v) for v in teacher_actions],
                    "selected_equals_teacher": bool(list(selected_actions) == list(teacher_actions)),
                    "budget_used_after_step": int(budget_used),
                }
            )

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
        "policy_name": str(requested_policy_name),
        "policy_mode": str(policy_mode),
        "branch_taken": str(branch_taken),
        "success_rate": float(hit_round is not None),
        "hit_round": None if hit_round is None else int(hit_round),
        "hit_sample_index": None if hit_sample_index is None else int(hit_sample_index),
        "budget_used": int(budget_used),
        "avg_step_reward": float(sum(float(row["reward"]) for row in step_rows) / max(len(step_rows), 1)),
        "teacher_exact_match_rate": float(sum(float(row["teacher_exact_match"]) for row in step_rows) / max(len(step_rows), 1)),
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
        "step_trace_rows": step_trace_rows,
    }


def run_policy_on_cases_strict(
    *,
    cases: Sequence[CaseRecord],
    family: str,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    policy_mode: str,
    requested_policy_name: str,
    model: Optional[SpimNativePolicy],
    branch_taken: str,
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
    device: torch.device,
    collect_transitions: bool,
    trace_case_limit: int,
    trace_step_limit: int,
    checkpoint_info: Dict[str, Any],
    slate_size: int = 10,
    slate_top_posterior_k: int = 6,
    slate_high_disagreement_k: int = 3,
    slate_novelty_k: int = 2,
    early_stage_round_cutoff: int = 0,
    early_stage_slate_top_posterior_k: Optional[int] = None,
    early_stage_slate_high_disagreement_k: Optional[int] = None,
    early_stage_slate_novelty_k: Optional[int] = None,
    decode_mode: str = "greedy",
    decode_beam_width: int = 4,
    decode_topk_per_slot: int = 4,
    decode_sample_candidates: int = 6,
    decode_sample_topk: int = 6,
    decode_logprob_weight: float = 1.0,
    decode_value_weight: float = 1.0,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    all_transitions: List[Dict[str, Any]] = []
    trace_case_rows: List[Dict[str, Any]] = []
    trace_step_rows: List[Dict[str, Any]] = []

    for case_index, case in enumerate(cases):
        out = run_case_policy_strict(
            case=case,
            case_index=int(case_index),
            family=family,
            runtime=runtime,
            env=env,
            policy_mode=str(policy_mode),
            requested_policy_name=str(requested_policy_name),
            model=model,
            branch_taken=str(branch_taken),
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
            device=device,
            collect_transitions=bool(collect_transitions),
            trace_case_limit=int(trace_case_limit),
            trace_step_limit=int(trace_step_limit),
            checkpoint_info=checkpoint_info,
            slate_size=int(slate_size),
            slate_top_posterior_k=int(slate_top_posterior_k),
            slate_high_disagreement_k=int(slate_high_disagreement_k),
            slate_novelty_k=int(slate_novelty_k),
            early_stage_round_cutoff=int(early_stage_round_cutoff),
            early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
            early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
            early_stage_slate_novelty_k=early_stage_slate_novelty_k,
            decode_mode=str(decode_mode),
            decode_beam_width=int(decode_beam_width),
            decode_topk_per_slot=int(decode_topk_per_slot),
            decode_sample_candidates=int(decode_sample_candidates),
            decode_sample_topk=int(decode_sample_topk),
            decode_logprob_weight=float(decode_logprob_weight),
            decode_value_weight=float(decode_value_weight),
        )
        case_rows.append(out["case_row"])
        step_rows.extend(out["step_rows"])
        all_transitions.extend(out["transitions"])
        if int(case_index) < int(trace_case_limit):
            trace_case_rows.append(
                {
                    "case_index": int(case_index),
                    "case_id": case.case_id,
                    "policy_mode": str(policy_mode),
                    "requested_policy_name": str(requested_policy_name),
                    "branch_taken": str(branch_taken),
                    "checkpoint_path": checkpoint_info["path"],
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "checkpoint_load_status": checkpoint_info["load_status"],
                    "step_trace_count": int(sum(1 for row in out["step_trace_rows"] if row["case_index"] == int(case_index))),
                }
            )
        trace_step_rows.extend(out["step_trace_rows"])

    return {
        "case_rows": case_rows,
        "step_rows": step_rows,
        "transitions": all_transitions,
        "trace": {
            "case_rows": trace_case_rows,
            "step_rows": trace_step_rows,
        },
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

    if str(args.teacher_family) == "auto":
        teacher_decision = auto_select_teacher(precheck_root)
        teacher_family = str(teacher_decision["selected_family"])
    else:
        teacher_family = str(args.teacher_family)
        teacher_decision = {
            "selected_family": teacher_family,
            "selection_rule": "manual_override",
            "v3_success_rate": None,
            "v1_success_rate": None,
            "precheck_path": str(precheck_root / "family_summary.csv"),
        }

    branch_taken, checkpoint_path = _resolve_mode_branch(str(args.policy_mode), str(args.checkpoint))

    runtime, split_meta = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.split),
        num_rounds=int(args.num_rounds),
        actions_per_round=int(args.actions_per_round),
        train_max_cases=int(args.train_max_cases),
        train_cache_version=str(args.train_cache_version),
        case_limit=int(args.case_limit),
    )

    model, checkpoint_info = _load_model_for_mode(
        policy_mode=str(args.policy_mode),
        checkpoint=checkpoint_path,
        device=device,
        hidden_dim=int(args.hidden_dim),
        include_surrogate_features=bool(args.include_surrogate_features),
        include_uncertainty_regime_features=bool(args.use_uncertainty_regime_features),
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
        enable_early_stage_specialist_head=bool(args.enable_early_stage_specialist_head),
        early_stage_round_cutoff=int(args.early_stage_round_cutoff),
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
    )

    env = CleanTwoChannelEvidenceEnv()
    early_stage_slate_top_posterior_k = None if int(args.early_stage_slate_top_posterior_k) < 0 else int(args.early_stage_slate_top_posterior_k)
    early_stage_slate_high_disagreement_k = None if int(args.early_stage_slate_high_disagreement_k) < 0 else int(args.early_stage_slate_high_disagreement_k)
    early_stage_slate_novelty_k = None if int(args.early_stage_slate_novelty_k) < 0 else int(args.early_stage_slate_novelty_k)
    eval_out = run_policy_on_cases_strict(
        cases=runtime["cases"],
        family=teacher_family,
        runtime=runtime,
        env=env,
        policy_mode=str(args.policy_mode),
        requested_policy_name=str(args.policy_name),
        model=model,
        branch_taken=str(branch_taken),
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
        device=device,
        collect_transitions=False,
        trace_case_limit=int(args.trace_case_limit),
        trace_step_limit=int(args.trace_step_limit),
        checkpoint_info=checkpoint_info,
        slate_size=int(args.slate_size),
        slate_top_posterior_k=int(args.slate_top_posterior_k),
        slate_high_disagreement_k=int(args.slate_high_disagreement_k),
        slate_novelty_k=int(args.slate_novelty_k),
        early_stage_round_cutoff=int(args.early_stage_round_cutoff),
        early_stage_slate_top_posterior_k=early_stage_slate_top_posterior_k,
        early_stage_slate_high_disagreement_k=early_stage_slate_high_disagreement_k,
        early_stage_slate_novelty_k=early_stage_slate_novelty_k,
        decode_mode=str(args.decode_mode),
        decode_beam_width=int(args.decode_beam_width),
        decode_topk_per_slot=int(args.decode_topk_per_slot),
        decode_sample_candidates=int(args.decode_sample_candidates),
        decode_sample_topk=int(args.decode_sample_topk),
        decode_logprob_weight=float(args.decode_logprob_weight),
        decode_value_weight=float(args.decode_value_weight),
    )

    summary = summarize_case_metrics(
        eval_out["case_rows"],
        num_rounds=int(args.num_rounds),
        action_budget=int(args.actions_per_round),
    )

    pd.DataFrame(eval_out["case_rows"]).to_csv(output_dir / "case_rows.csv", index=False)
    pd.DataFrame(eval_out["step_rows"]).to_csv(output_dir / "step_rows.csv", index=False)

    trace_payload = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "source_root": str(source_root),
        "cache_dir": str(cache_dir),
        "seed": int(args.seed),
        "device": str(device),
        "split": str(args.split),
        "split_meta": split_meta,
        "teacher_family": str(teacher_family),
        "teacher_decision": teacher_decision,
        "policy_mode": str(args.policy_mode),
        "requested_policy_name": str(args.policy_name),
        "branch_taken": str(branch_taken),
        "candidate_slate": {
            "slate_size": int(args.slate_size),
            "top_posterior_k": int(args.slate_top_posterior_k),
            "high_disagreement_k": int(args.slate_high_disagreement_k),
            "novelty_k": int(args.slate_novelty_k),
            "early_stage_round_cutoff": int(args.early_stage_round_cutoff),
            "early_stage_top_posterior_k": early_stage_slate_top_posterior_k,
            "early_stage_high_disagreement_k": early_stage_slate_high_disagreement_k,
            "early_stage_novelty_k": early_stage_slate_novelty_k,
        },
        "decode_spec": {
            "decode_mode": str(args.decode_mode),
            "beam_width": int(args.decode_beam_width),
            "topk_per_slot": int(args.decode_topk_per_slot),
            "sample_candidates": int(args.decode_sample_candidates),
            "sample_topk": int(args.decode_sample_topk),
            "logprob_weight": float(args.decode_logprob_weight),
            "value_weight": float(args.decode_value_weight),
        },
        "policy_spec": {
            "include_surrogate_features": bool(args.include_surrogate_features),
            "use_uncertainty_regime_features": bool(args.use_uncertainty_regime_features),
            "enable_regime_head": bool(args.enable_regime_head),
            "regime_head_classes": int(args.regime_head_classes),
            "regime_embed_dim": int(args.regime_embed_dim),
            "candidate_encoder": str(args.candidate_encoder),
        },
        "checkpoint": checkpoint_info,
        "strict_assertions": {
            "teacher_rejects_checkpoint": bool(args.policy_mode not in {"teacher", "teacher_slate"} or not str(args.checkpoint).strip()),
            "student_requires_checkpoint": bool(args.policy_mode in {"teacher", "teacher_slate"} or bool(str(args.checkpoint).strip())),
            "policy_name_not_used_for_branching": True,
        },
        "trace_limits": {
            "case_limit": int(args.case_limit),
            "trace_case_limit": int(args.trace_case_limit),
            "trace_step_limit": int(args.trace_step_limit),
        },
        "trace_evidence": eval_out["trace"],
        "summary": summary,
        "artifacts": {
            "case_rows": str(output_dir / "case_rows.csv"),
            "step_rows": str(output_dir / "step_rows.csv"),
        },
    }
    write_json(output_dir / "strict_mode_trace.json", trace_payload)
    write_json(
        output_dir / "summary.json",
        {
            "runner_version": RUNNER_VERSION,
            "panel_version": PANEL_VERSION,
            "source_root": str(source_root),
            "cache_dir": str(cache_dir),
            "seed": int(args.seed),
            "device": str(device),
            "split": str(args.split),
            "split_case_count": int(len(runtime["cases"])),
            "split_meta": split_meta,
            "teacher_family": str(teacher_family),
            "teacher_decision": teacher_decision,
            "policy_mode": str(args.policy_mode),
            "requested_policy_name": str(args.policy_name),
            "branch_taken": str(branch_taken),
            "candidate_slate": {
                "slate_size": int(args.slate_size),
                "top_posterior_k": int(args.slate_top_posterior_k),
                "high_disagreement_k": int(args.slate_high_disagreement_k),
                "novelty_k": int(args.slate_novelty_k),
            },
            "decode_spec": {
                "decode_mode": str(args.decode_mode),
                "beam_width": int(args.decode_beam_width),
                "topk_per_slot": int(args.decode_topk_per_slot),
                "sample_candidates": int(args.decode_sample_candidates),
                "sample_topk": int(args.decode_sample_topk),
                "logprob_weight": float(args.decode_logprob_weight),
                "value_weight": float(args.decode_value_weight),
            },
            "policy_spec": {
                "include_surrogate_features": bool(args.include_surrogate_features),
                "use_uncertainty_regime_features": bool(args.use_uncertainty_regime_features),
                "enable_regime_head": bool(args.enable_regime_head),
                "regime_head_classes": int(args.regime_head_classes),
                "regime_embed_dim": int(args.regime_embed_dim),
                "candidate_encoder": str(args.candidate_encoder),
            },
            "checkpoint": checkpoint_info,
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
                "strict_mode_trace": str(output_dir / "strict_mode_trace.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
