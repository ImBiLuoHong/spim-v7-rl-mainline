from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

LEGACY_HSR_ROOT = PROJECT_ROOT / "tools" / "legacy" / "src_baselines_archive"
if str(LEGACY_HSR_ROOT) not in sys.path:
    sys.path.append(str(LEGACY_HSR_ROOT))

from hsr_agent import HSRAgent

from src.modeling.belief_updaters.evidence_posterior_like import _evidence_contrast_scalar, _masked_zscore
from src.modeling.belief_updaters.pure_likelihood_bayes import PureLikelihoodBayesBelief
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory, WitnessRecord
from src.modeling.loop.navigator_vnext_contract import build_candidate_semantics, default_reward_contract, tensor_attr
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_authoritative_hsr_baseline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SOURCE_ROOT,
    _extract_trigger_global,
    build_hsr_agent,
    load_foundation_graph,
    resolve_foundation_graph_path,
    read_json,
)
from src.scripts.run_posterior_like_belief_audit import load_frozen_reasoner, load_runtime_context, write_json
from src.scripts.run_reasoner_same_case_stronger_source_overfit import TempGraph, build_state_input, move_payload, make_rollout_state
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    CaseRecord,
    collect_dataset_assets,
    colon_case_id_from_data,
)
from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.data.v6.topology import HydraulicTopology
import yaml


DEFAULT_ACCEPTABILITY_ROOT = PROJECT_ROOT / "artifacts" / "posterior_like_belief_acceptability_audit" / "20260407_exact136_belief_acceptability_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "spim_family_sweep" / "20260409_exact136_b30_spim_sweep_v1"
RUNNER_VERSION = "spim_family_sweep_v1"
PANEL_VERSION = "exact136_authoritative_b30_spim_sweep_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded SPIM family sweep under authoritative exact136 B30 posterior_greedy.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--actions-per-round", type=int, default=3)
    parser.add_argument("--progress-every-cases", type=int, default=10)
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--rank-weighted-beta", type=float, default=2.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)
    parser.add_argument("--binary-loglik-eps-fp", type=float, default=0.02)
    parser.add_argument("--binary-loglik-eps-fn", type=float, default=0.05)
    parser.add_argument("--binary-loglik-prior-mix", type=float, default=0.1)
    parser.add_argument("--full-train-split", action="store_true")
    return parser.parse_args()


def _contract_cfg() -> Dict[str, float]:
    cfg = default_reward_contract()
    cfg.update({"support_plausible_delta": 0.25, "not_ruled_out_threshold": 0.5})
    return cfg


def _load_calibrated_fused(acceptability_root: Path) -> tuple[Dict[str, float], float]:
    payload = json.loads((acceptability_root / "summary.json").read_text())
    return dict(payload["head_definitions"]["calibrated_fused_posterior"]), float(payload["head_definitions"]["logits_only_posterior"]["temperature"])


def _seed_case(seed: int, family: str, case_id: str) -> None:
    digest = hashlib.sha256(f"{seed}:{family}:{case_id}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:8], 16)
    np.random.seed(local_seed)
    torch.manual_seed(int(local_seed % (2**31)))


def _safe_softmax(scores: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    out = torch.zeros_like(scores.view(-1).float())
    mask = mask.view(-1).bool()
    if bool(mask.any()):
        out[mask] = torch.softmax(scores[mask] / max(float(temperature), 1e-6), dim=0)
    return out


def _normalize_distribution(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(probs.view(-1).float())
    mask = mask.view(-1).bool()
    vals = probs.view(-1).float().clone()
    vals[~mask] = 0.0
    denom = float(vals.sum().item())
    if denom <= 1e-12:
        out[mask] = 1.0 / max(int(mask.sum().item()), 1)
    else:
        out[mask] = vals[mask] / denom
    return out


def _compute_components(state: Dict[str, Any], reasoner_module, device: torch.device) -> Dict[str, torch.Tensor]:
    valid_mask = state["valid_mask"].view(-1).bool().cpu()
    graph = TempGraph(state["edge_index"], int(valid_mask.numel()), device)
    state_input = move_payload(build_state_input(state), device)
    physics_ctx = move_payload(state["phys_ctx"].__dict__, device)
    with torch.no_grad():
        out = reasoner_module(state_input, graph, physics_ctx=physics_ctx)
    reasoner_logits = out["logits"].detach().float().view(-1).cpu()
    payload = build_clean_aligned_feature_payload(
        build_state_input(state),
        batch_index=torch.zeros(int(valid_mask.numel()), dtype=torch.long),
        edge_index=state["edge_index"].view(2, -1).long(),
        physics_ctx=state["phys_ctx"].__dict__,
        frontier_mode="unresolved_without_pair",
    )
    semantics = build_candidate_semantics(
        evidence_state=state["evidence_state"],
        constraint_state=state["constraint_state"],
        valid_mask=valid_mask,
        batch=torch.zeros(int(valid_mask.numel()), dtype=torch.long),
        contract_cfg=_contract_cfg(),
    )
    candidate_mask = semantics["candidate_mask"].view(-1).bool().cpu()
    if not bool(candidate_mask.any()):
        candidate_mask = valid_mask.clone()
    q_score = semantics["q_score"].view(-1).float().cpu()
    contradiction = tensor_attr(state["evidence_state"], "contradiction_score", valid_mask.float(), default=0.0).cpu()
    contrast_signal = _evidence_contrast_scalar(payload["node_features"].cpu(), valid_mask)
    return {
        "valid_mask": valid_mask,
        "candidate_mask": candidate_mask,
        "reasoner_logits": reasoner_logits,
        "q_score": q_score,
        "contradiction_score": contradiction,
        "contrast_signal": contrast_signal,
    }


def _belief_metrics(probs: torch.Tensor, mask: torch.Tensor, source_local: int | None, threshold: float = 0.7) -> Dict[str, Any]:
    probs = _normalize_distribution(probs, mask)
    mask = mask.view(-1).bool()
    valid = probs[mask].clamp_min(1e-12)
    entropy = float((-(valid * torch.log(valid))).sum().item()) if valid.numel() > 0 else 0.0
    eff_support = float(math.exp(entropy)) if entropy > -1e8 else 0.0
    norm_entropy = float(entropy / math.log(max(int(valid.numel()), 2))) if valid.numel() > 1 else 0.0
    order = torch.argsort(probs, descending=True)
    order = order[mask[order]]
    ordered_vals = probs[order]
    csum = torch.cumsum(ordered_vals, dim=0) if ordered_vals.numel() > 0 else torch.tensor([], dtype=torch.float32)
    hits = (csum >= float(threshold)).nonzero(as_tuple=True)[0] if ordered_vals.numel() > 0 else torch.tensor([], dtype=torch.long)
    cover_idx = int(hits[0].item()) + 1 if hits.numel() > 0 else int(ordered_vals.numel())
    rank = None
    true_mass = None
    if source_local is not None and bool(mask[int(source_local)].item()):
        pos = (order == int(source_local)).nonzero(as_tuple=True)[0]
        if pos.numel() > 0:
            rank = int(pos[0].item()) + 1
            true_mass = float(probs[int(source_local)].item())
    return {
        "belief": probs,
        "ordered_candidates": [int(v) for v in order.tolist()],
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "effective_support": eff_support,
        "top1_mass": float(ordered_vals[:1].sum().item()) if ordered_vals.numel() > 0 else 0.0,
        "top3_mass": float(ordered_vals[: min(3, ordered_vals.numel())].sum().item()) if ordered_vals.numel() > 0 else 0.0,
        "top5_mass": float(ordered_vals[: min(5, ordered_vals.numel())].sum().item()) if ordered_vals.numel() > 0 else 0.0,
        "mass_cover_size_ratio": float(cover_idx / max(int(mask.sum().item()), 1)),
        "mass_cover_size": int(cover_idx),
        "true_rank": rank,
        "true_mass": true_mass,
    }


def _pick_topk_unsampled(probs: torch.Tensor, mask: torch.Tensor, rollout: PracticalRollout, k: int) -> List[int]:
    order = torch.argsort(probs.view(-1), descending=True)
    chosen: List[int] = []
    for idx in order.tolist():
        if not bool(mask[int(idx)].item()):
            continue
        if bool(rollout.revealed_mask[int(idx)].item()):
            continue
        chosen.append(int(idx))
        if len(chosen) >= int(k):
            break
    return chosen


def _calibrated_fused_probs(components: Dict[str, torch.Tensor], params: Dict[str, float]) -> torch.Tensor:
    mask = components["candidate_mask"]
    q_z = _masked_zscore(components["q_score"], mask)
    l_z = _masked_zscore(components["reasoner_logits"], mask)
    c_z = _masked_zscore(components["contrast_signal"], mask)
    d_z = _masked_zscore(components["contradiction_score"], mask)
    energy = (
        float(params["lambda_reasoner"]) * l_z
        + float(params["lambda_q"]) * q_z
        + float(params["lambda_contrast"]) * c_z
        - float(params["lambda_contradiction"]) * d_z
    )
    return _safe_softmax(energy, mask, float(params["temperature"]))


def _hsr_hard_posterior(rollout: PracticalRollout, agent: HSRAgent) -> Dict[str, Any]:
    probs = torch.zeros(int(rollout.num_nodes), dtype=torch.float32)
    allowed_local = []
    global_to_local = {int(g.item()): int(i) for i, g in enumerate(rollout.g_ids.view(-1))}
    for gid in sorted(agent.candidate_set):
        local = global_to_local.get(int(gid))
        if local is None or bool(rollout.revealed_mask[int(local)].item()):
            continue
        allowed_local.append(int(local))
    if allowed_local:
        mass = 1.0 / float(len(allowed_local))
        probs[allowed_local] = mass
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[allowed_local] = True
    return {"belief": probs, "candidate_mask": mask, "candidate_fraction": float(len(allowed_local) / max(int(rollout.num_nodes), 1))}


@dataclass
class PaperLikeHSRState:
    source_prior: Optional[torch.Tensor] = None
    trigger_seeded_positive: bool = False


def _trigger_global_to_local(rollout: PracticalRollout, trigger_global: Optional[int]) -> Optional[int]:
    if trigger_global is None:
        return None
    for local_idx, gid in enumerate(rollout.g_ids.view(-1).tolist()):
        if int(gid) == int(trigger_global):
            return int(local_idx)
    return None


def _resolve_onset_grid(*, family: str, episode_duration_min: float) -> List[float]:
    delta = float(episode_duration_min)
    if family in {"hsr_soft_scenario_posterior_v7_5offset"}:
        return [-2.0 * delta, -1.0 * delta, 0.0, 1.0 * delta, 2.0 * delta]
    if family in {"hsr_soft_scenario_posterior_v7_7offset"}:
        return [-3.0 * delta, -2.0 * delta, -1.0 * delta, 0.0, 1.0 * delta, 2.0 * delta, 3.0 * delta]
    return [-1.0 * delta, 0.0, 1.0 * delta]


def _inject_trigger_positive_witness_once(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
) -> None:
    if bool(paper_state.trigger_seeded_positive):
        return
    trigger_local = _trigger_global_to_local(rollout, trigger_global)
    if trigger_local is None:
        paper_state.trigger_seeded_positive = True
        return
    info = state.get("info", {})
    episode_id = int(info.get("episode", 0))
    abs_time = float(info.get("time_min", 0.0))
    snapshot_idx = int(info.get("t_snapshot_idx", 0))
    history.records.insert(
        0,
        WitnessRecord(
            node_local_idx=int(trigger_local),
            node_global_idx=int(trigger_global),
            absolute_time_min=abs_time,
            episode_id=episode_id,
            absolute_snapshot_idx=snapshot_idx,
            label="positive",
            confidence=1.0,
            t_snapshot_idx=snapshot_idx,
            phys_ctx=state["phys_ctx"],
        ),
    )
    paper_state.trigger_seeded_positive = True


def _dynamic_source_loglik(
    *,
    record,
    arrival: torch.Tensor,
    onset_offset_min: float,
    tau_min: float,
    eps_fp: float,
    eps_fn: float,
) -> torch.Tensor:
    delta = float(record.absolute_time_min) - float(onset_offset_min) - arrival
    p_arrived = torch.sigmoid(delta / max(float(tau_min), 1e-6))
    p_pos = eps_fp + (1.0 - eps_fp - eps_fn) * p_arrived
    p_pos = p_pos.clamp(1e-9, 1.0 - 1e-9)
    if str(record.label) == "positive":
        return torch.log(p_pos)
    return torch.log(1.0 - p_pos)


def _uniform_prior_on_mask(mask: torch.Tensor) -> torch.Tensor:
    prior = torch.zeros(mask.numel(), dtype=torch.float32)
    count = int(mask.view(-1).bool().sum().item())
    if count > 0:
        prior[mask.view(-1).bool()] = 1.0 / float(count)
    return prior


def _build_clean_candidate_mask(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    trigger_global: Optional[int],
) -> torch.Tensor:
    num_nodes = int(rollout.num_nodes)
    valid_mask = state["valid_mask"].view(-1).bool().cpu()
    constraint_state = state["constraint_state"]
    phys_ctx = state["phys_ctx"]
    feasible = phys_ctx.feasible_mask.view(-1).bool().cpu() if phys_ctx.feasible_mask is not None else valid_mask.clone()
    confirmed_non_source = (
        constraint_state.confirmed_non_source_mask.view(-1).bool().cpu()
        if constraint_state is not None and getattr(constraint_state, "confirmed_non_source_mask", None) is not None
        else torch.zeros(num_nodes, dtype=torch.bool)
    )
    mask = feasible & (~confirmed_non_source)
    mask &= (~rollout.revealed_mask.view(-1).cpu())
    if trigger_global is None:
        return mask
    trigger_local = None
    for local_idx, gid in enumerate(rollout.g_ids.view(-1).tolist()):
        if int(gid) == int(trigger_global):
            trigger_local = int(local_idx)
            break
    if trigger_local is None:
        return mask
    gate = DynamicReachabilityRuleModule()
    seed = torch.tensor([int(trigger_local)], dtype=torch.long)
    dist_to_trigger = gate.compute_distance_matrix(seed, phys_ctx, num_nodes).cpu()[:, 0]
    trigger_reachable = dist_to_trigger < float(gate.infinity / 2)
    return mask & trigger_reachable


def _compute_scenario_error(
    *,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    candidate_idx: torch.Tensor,
    onset_offsets_min: List[float],
    time_tol_min: float,
) -> torch.Tensor:
    num_nodes = int(rollout.num_nodes)
    onset_grid = [float(v) for v in onset_offsets_min]
    if candidate_idx.numel() <= 0 or len(onset_grid) <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    gate = DynamicReachabilityRuleModule()
    scenario_error = torch.zeros((int(candidate_idx.numel()), int(len(onset_grid))), dtype=torch.float32)
    tol = float(max(time_tol_min, 1e-6))
    for record in history.records:
        seed = torch.tensor([int(record.node_local_idx)], dtype=torch.long)
        dist_matrix = gate.compute_distance_matrix(seed, record.phys_ctx, num_nodes).cpu()
        arrival = dist_matrix[candidate_idx, 0]
        finite = arrival < float(gate.infinity / 2)
        observed_positive = 1.0 if str(record.label) == "positive" else 0.0
        t_obs = float(record.absolute_time_min)
        for onset_col, onset_offset in enumerate(onset_grid):
            slack = (t_obs - onset_offset) - arrival
            expected_positive = torch.sigmoid(slack / tol)
            expected_positive = torch.where(finite, expected_positive, torch.zeros_like(expected_positive))
            scenario_error[:, onset_col] += torch.abs(expected_positive - observed_positive)
    return scenario_error


def _compute_scenario_log_joint(
    *,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    candidate_idx: torch.Tensor,
    onset_offsets_min: List[float],
    tau_min: float,
    eps_fp: float,
    eps_fn: float,
) -> torch.Tensor:
    num_nodes = int(rollout.num_nodes)
    onset_grid = [float(v) for v in onset_offsets_min]
    if candidate_idx.numel() <= 0 or len(onset_grid) <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    log_joint = torch.full(
        (int(candidate_idx.numel()), int(len(onset_grid))),
        -math.log(float(candidate_idx.numel())) - math.log(float(len(onset_grid))),
        dtype=torch.float32,
    )
    gate = DynamicReachabilityRuleModule()
    for record in history.records:
        seed = torch.tensor([int(record.node_local_idx)], dtype=torch.long)
        dist_matrix = gate.compute_distance_matrix(seed, record.phys_ctx, num_nodes).cpu()
        arrival = dist_matrix[candidate_idx, 0]
        for onset_col, onset_offset in enumerate(onset_grid):
            log_joint[:, onset_col] += _dynamic_source_loglik(
                record=record,
                arrival=arrival,
                onset_offset_min=float(onset_offset),
                tau_min=float(tau_min),
                eps_fp=float(eps_fp),
                eps_fn=float(eps_fn),
            )
    return log_joint


def _mix_with_prior(posterior: torch.Tensor, prior: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha = float(min(max(alpha, 0.0), 1.0))
    return alpha * posterior + (1.0 - alpha) * prior


def _paper_topk_ema_posterior(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    onset_offsets_min: List[float],
    alpha: float,
    topk_fraction: float,
    time_tol_min: float,
) -> Dict[str, Any]:
    num_nodes = int(rollout.num_nodes)
    base_mask = _build_clean_candidate_mask(rollout=rollout, state=state, trigger_global=trigger_global)
    candidate_idx = torch.nonzero(base_mask, as_tuple=True)[0].long()
    if candidate_idx.numel() <= 0:
        return {"belief": torch.zeros(num_nodes, dtype=torch.float32), "candidate_mask": base_mask}

    onset_grid = [float(v) for v in onset_offsets_min]
    scenario_count = int(candidate_idx.numel()) * int(len(onset_grid))
    if scenario_count <= 0:
        return {"belief": torch.zeros(num_nodes, dtype=torch.float32), "candidate_mask": base_mask}
    scenario_error = _compute_scenario_error(
        rollout=rollout,
        history=history,
        candidate_idx=candidate_idx,
        onset_offsets_min=onset_grid,
        time_tol_min=float(time_tol_min),
    )
    flat_error = scenario_error.view(-1)
    keep_top = max(1, int(math.ceil(float(topk_fraction) * float(flat_error.numel()))))
    keep_top = min(keep_top, int(flat_error.numel()))
    _, top_idx = torch.topk(-flat_error, k=keep_top, dim=0)
    rows = top_idx // int(len(onset_grid))
    src_counts = torch.bincount(rows, minlength=int(candidate_idx.numel())).float()
    p_hat_local = src_counts / src_counts.sum().clamp_min(1e-9)

    p_hat = torch.zeros(num_nodes, dtype=torch.float32)
    p_hat[candidate_idx] = p_hat_local
    p_hat = _normalize_distribution(p_hat, base_mask)

    if paper_state.source_prior is None:
        prior = _uniform_prior_on_mask(base_mask)
    else:
        prior = paper_state.source_prior.view(-1).float().cpu().clone()
        prior = _normalize_distribution(prior, base_mask)
    belief = _mix_with_prior(p_hat, prior, float(alpha))
    belief = _normalize_distribution(belief, base_mask)
    paper_state.source_prior = belief.clone()
    return {"belief": belief, "candidate_mask": base_mask}


def _rank_weighted_topk_posterior(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    onset_offsets_min: List[float],
    alpha: float,
    topk_fraction: float,
    time_tol_min: float,
    beta: float,
) -> Dict[str, Any]:
    num_nodes = int(rollout.num_nodes)
    base_mask = _build_clean_candidate_mask(rollout=rollout, state=state, trigger_global=trigger_global)
    candidate_idx = torch.nonzero(base_mask, as_tuple=True)[0].long()
    if candidate_idx.numel() <= 0:
        return {"belief": torch.zeros(num_nodes, dtype=torch.float32), "candidate_mask": base_mask}
    onset_grid = [float(v) for v in onset_offsets_min]
    scenario_error = _compute_scenario_error(
        rollout=rollout,
        history=history,
        candidate_idx=candidate_idx,
        onset_offsets_min=onset_grid,
        time_tol_min=float(time_tol_min),
    )
    flat_error = scenario_error.view(-1)
    keep_top = max(1, int(math.ceil(float(topk_fraction) * float(flat_error.numel()))))
    keep_top = min(keep_top, int(flat_error.numel()))
    _, top_idx = torch.topk(-flat_error, k=keep_top, dim=0)
    top_err = flat_error[top_idx]
    norm = (top_err - float(top_err.min().item())) / max(float((top_err.max() - top_err.min()).item()), 1e-6)
    top_w = torch.exp(-float(max(beta, 1e-6)) * norm)
    rows = top_idx // int(len(onset_grid))
    weighted_src = torch.zeros(int(candidate_idx.numel()), dtype=torch.float32)
    weighted_src.scatter_add_(0, rows.long(), top_w.float())
    p_hat_local = weighted_src / weighted_src.sum().clamp_min(1e-9)
    p_hat = torch.zeros(num_nodes, dtype=torch.float32)
    p_hat[candidate_idx] = p_hat_local
    p_hat = _normalize_distribution(p_hat, base_mask)
    if paper_state.source_prior is None:
        prior = _uniform_prior_on_mask(base_mask)
    else:
        prior = _normalize_distribution(paper_state.source_prior.view(-1).float().cpu().clone(), base_mask)
    belief = _mix_with_prior(p_hat, prior, float(alpha))
    belief = _normalize_distribution(belief, base_mask)
    paper_state.source_prior = belief.clone()
    return {"belief": belief, "candidate_mask": base_mask}


def _soft_scenario_posterior(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    onset_offsets_min: List[float],
    alpha: float,
    time_tol_min: float,
    beta: float,
) -> Dict[str, Any]:
    num_nodes = int(rollout.num_nodes)
    base_mask = _build_clean_candidate_mask(rollout=rollout, state=state, trigger_global=trigger_global)
    candidate_idx = torch.nonzero(base_mask, as_tuple=True)[0].long()
    if candidate_idx.numel() <= 0:
        return {"belief": torch.zeros(num_nodes, dtype=torch.float32), "candidate_mask": base_mask}
    onset_grid = [float(v) for v in onset_offsets_min]
    scenario_error = _compute_scenario_error(
        rollout=rollout,
        history=history,
        candidate_idx=candidate_idx,
        onset_offsets_min=onset_grid,
        time_tol_min=float(time_tol_min),
    )
    flat_error = scenario_error.view(-1)
    shifted = flat_error - float(flat_error.min().item())
    weights = torch.exp(-float(max(beta, 1e-6)) * shifted)
    rows = torch.arange(int(flat_error.numel()), dtype=torch.long) // int(len(onset_grid))
    src_weight = torch.zeros(int(candidate_idx.numel()), dtype=torch.float32)
    src_weight.scatter_add_(0, rows, weights.float())
    p_hat_local = src_weight / src_weight.sum().clamp_min(1e-9)
    p_hat = torch.zeros(num_nodes, dtype=torch.float32)
    p_hat[candidate_idx] = p_hat_local
    p_hat = _normalize_distribution(p_hat, base_mask)
    if paper_state.source_prior is None:
        prior = _uniform_prior_on_mask(base_mask)
    else:
        prior = _normalize_distribution(paper_state.source_prior.view(-1).float().cpu().clone(), base_mask)
    belief = _mix_with_prior(p_hat, prior, float(alpha))
    belief = _normalize_distribution(belief, base_mask)
    paper_state.source_prior = belief.clone()
    return {"belief": belief, "candidate_mask": base_mask}


def _soft_scenario_posterior_v6(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    onset_offsets_min: List[float],
    alpha: float,
    time_tol_min: float,
    beta: float,
) -> Dict[str, Any]:
    # SPIM v6: same as v3 but seed trigger as the first positive witness once.
    _inject_trigger_positive_witness_once(
        rollout=rollout,
        state=state,
        history=history,
        trigger_global=trigger_global,
        paper_state=paper_state,
    )
    return _soft_scenario_posterior(
        rollout=rollout,
        state=state,
        history=history,
        trigger_global=trigger_global,
        paper_state=paper_state,
        onset_offsets_min=onset_offsets_min,
        alpha=alpha,
        time_tol_min=time_tol_min,
        beta=beta,
    )


def _binary_loglik_source_only_posterior(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    tau_min: float,
    eps_fp: float,
    eps_fn: float,
    prior_mix: float,
) -> Dict[str, Any]:
    num_nodes = int(rollout.num_nodes)
    base_mask = _build_clean_candidate_mask(rollout=rollout, state=state, trigger_global=trigger_global)
    candidate_idx = torch.nonzero(base_mask, as_tuple=True)[0].long()
    if candidate_idx.numel() <= 0:
        return {"belief": torch.zeros(num_nodes, dtype=torch.float32), "candidate_mask": base_mask}
    logp = torch.full((int(candidate_idx.numel()),), -math.log(float(candidate_idx.numel())), dtype=torch.float32)
    gate = DynamicReachabilityRuleModule()
    for record in history.records:
        seed = torch.tensor([int(record.node_local_idx)], dtype=torch.long)
        dist_matrix = gate.compute_distance_matrix(seed, record.phys_ctx, num_nodes).cpu()
        arrival = dist_matrix[candidate_idx, 0]
        logp += _dynamic_source_loglik(
            record=record,
            arrival=arrival,
            onset_offset_min=0.0,
            tau_min=float(tau_min),
            eps_fp=float(eps_fp),
            eps_fn=float(eps_fn),
        )
    posterior_local = torch.softmax(logp, dim=0)
    posterior = torch.zeros(num_nodes, dtype=torch.float32)
    posterior[candidate_idx] = posterior_local
    posterior = _normalize_distribution(posterior, base_mask)
    if paper_state.source_prior is None:
        prior = _uniform_prior_on_mask(base_mask)
    else:
        prior = _normalize_distribution(paper_state.source_prior.view(-1).float().cpu().clone(), base_mask)
    belief = _mix_with_prior(posterior, prior, 1.0 - float(min(max(prior_mix, 0.0), 1.0)))
    belief = _normalize_distribution(belief, base_mask)
    paper_state.source_prior = belief.clone()
    return {"belief": belief, "candidate_mask": base_mask}


def _binary_loglik_source_onset_posterior(
    *,
    rollout: PracticalRollout,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    trigger_global: Optional[int],
    paper_state: PaperLikeHSRState,
    onset_offsets_min: List[float],
    tau_min: float,
    eps_fp: float,
    eps_fn: float,
    prior_mix: float,
) -> Dict[str, Any]:
    num_nodes = int(rollout.num_nodes)
    base_mask = _build_clean_candidate_mask(rollout=rollout, state=state, trigger_global=trigger_global)
    candidate_idx = torch.nonzero(base_mask, as_tuple=True)[0].long()
    if candidate_idx.numel() <= 0:
        return {"belief": torch.zeros(num_nodes, dtype=torch.float32), "candidate_mask": base_mask}
    log_joint = _compute_scenario_log_joint(
        rollout=rollout,
        history=history,
        candidate_idx=candidate_idx,
        onset_offsets_min=[float(v) for v in onset_offsets_min],
        tau_min=float(tau_min),
        eps_fp=float(eps_fp),
        eps_fn=float(eps_fn),
    )
    joint = torch.softmax(log_joint.view(-1), dim=0).view_as(log_joint)
    posterior_local = joint.sum(dim=1)
    posterior = torch.zeros(num_nodes, dtype=torch.float32)
    posterior[candidate_idx] = posterior_local
    posterior = _normalize_distribution(posterior, base_mask)
    if paper_state.source_prior is None:
        prior = _uniform_prior_on_mask(base_mask)
    else:
        prior = _normalize_distribution(paper_state.source_prior.view(-1).float().cpu().clone(), base_mask)
    belief = _mix_with_prior(posterior, prior, 1.0 - float(min(max(prior_mix, 0.0), 1.0)))
    belief = _normalize_distribution(belief, base_mask)
    paper_state.source_prior = belief.clone()
    return {"belief": belief, "candidate_mask": base_mask}


def _extract_source_global(rollout: PracticalRollout) -> Optional[int]:
    source_local = resolve_source_local_idx(rollout)
    if source_local is None:
        return None
    return int(rollout.g_ids[int(source_local)].item())


def _load_full_train_cases(*, source_root: Path, cache_dir: Path) -> tuple[List[CaseRecord], Dict[str, Any], Any]:
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    life_support = payload.get("life_support")
    if isinstance(life_support, dict) and str(life_support.get("profile")) == "custom_direct_edit":
        payload["life_support"] = {k: v for k, v in life_support.items() if k != "profile"}
    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(payload)
    cfg.training.enable_eval = False
    cfg.training.train_only = True
    cfg.training.enable_wandb = False
    cfg.data.skip_lmdb = False
    cfg.data.max_samples = None
    cfg.data.cache_version = "train_full_4823_paperlike_v1"
    cfg.data.rebuild_cache = True
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
                data=None,
            )
        )
    return cases, assets, dataset


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    authoritative_ref = PROJECT_ROOT / "artifacts" / "authoritative_hsr_baseline" / "20260409_exact136_hsr_authoritative_v1"

    runtime = load_runtime_context(source_root, cache_dir)
    runtime["num_episodes"] = int(args.num_rounds)
    runtime["action_budget"] = int(args.actions_per_round)
    runtime["full_train_dataset"] = None
    if bool(args.full_train_split):
        full_cases, full_assets, full_dataset = _load_full_train_cases(source_root=source_root, cache_dir=cache_dir)
        runtime["cases"] = full_cases
        runtime["dataset_assets"] = full_assets
        runtime["full_train_dataset"] = full_dataset
    if int(args.max_cases) > 0:
        runtime["cases"] = runtime["cases"][: int(args.max_cases)]
    foundation_graph_path = resolve_foundation_graph_path(source_root)
    families = [
        {"name": "hsr_paper_topk_ema_v1", "track": "clean"},
        {"name": "hsr_rank_weighted_topk_v2", "track": "clean"},
        {"name": "hsr_soft_scenario_posterior_v3", "track": "clean"},
        {"name": "hsr_soft_scenario_posterior_v7_5offset", "track": "clean"},
        {"name": "hsr_soft_scenario_posterior_v7_7offset", "track": "clean"},
        {"name": "hsr_soft_scenario_posterior_v6", "track": "clean"},
        {"name": "hsr_binary_loglik_posterior_source_only_v4", "track": "clean"},
        {"name": "hsr_binary_loglik_posterior_source_onset_v4", "track": "clean"},
    ]
    if str(args.families).strip():
        keep = {part.strip() for part in str(args.families).split(",") if part.strip()}
        families = [fam for fam in families if fam["name"] in keep]

    protocol_audit = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "authoritative_reference": {
            "protocol_audit": str(authoritative_ref / "protocol_audit.json"),
            "summary": str(authoritative_ref / "summary.json"),
            "why_this_reference": "official exact136 + B30 authoritative_hsr_baseline artifact used as clean contract anchor",
        },
        "shared_protocol": {
            "split": ("train-full current split panel" if bool(args.full_train_split) else "train-only exact136 replayable same-case panel"),
            "case_count": int(len(runtime["cases"])),
            "source": (
                str(source_root / "same_case_replayable_manifest.csv")
                if not bool(args.full_train_split)
                else str(PROJECT_ROOT / "data" / "train.txt")
            ),
            "num_rounds": int(args.num_rounds),
            "actions_per_round": int(args.actions_per_round),
            "episode_duration_min": float(runtime["episode_duration_min"]),
            "total_budget": int(args.num_rounds) * int(args.actions_per_round),
            "success_definition": "budgeted direct source hit under posterior_greedy",
            "fixed_policy_template": "posterior_greedy: select highest posterior mass valid unsampled nodes each round",
        },
        "families": [fam["name"] for fam in families],
    }
    write_json(output_dir / "protocol_audit.json", protocol_audit)

    provenance_audit = {
        "clean_track": {
            "hsr_paper_topk_ema_v1": {
                "source": "history + phys_ctx pseudo-scenarios; top-K scenario frequency -> p_hat; EMA update",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "onset_grid_mode": "[-episode_duration,0,+episode_duration]",
                "paper_like_alpha": float(args.paper_like_alpha),
                "paper_like_topk_fraction": float(args.paper_like_topk_fraction),
                "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
            },
            "hsr_rank_weighted_topk_v2": {
                "source": "same pseudo-scenarios + top-K with error-weighted aggregation + EMA",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "onset_grid_mode": "[-episode_duration,0,+episode_duration]",
                "paper_like_alpha": float(args.paper_like_alpha),
                "paper_like_topk_fraction": float(args.paper_like_topk_fraction),
                "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
                "rank_weighted_beta": float(args.rank_weighted_beta),
            },
            "hsr_soft_scenario_posterior_v3": {
                "source": "same pseudo-scenarios + all-scenario soft weighting exp(-beta*error) + EMA",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "onset_grid_mode": "[-episode_duration,0,+episode_duration]",
                "paper_like_alpha": float(args.paper_like_alpha),
                "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
                "soft_scenario_beta": float(args.soft_scenario_beta),
            },
            "hsr_soft_scenario_posterior_v7_5offset": {
                "source": "v3 enlarged onset grid with 5 offsets [-2,-1,0,+1,+2]×episode_duration + all-scenario soft weighting + EMA",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "onset_grid_mode": "[-2,-1,0,+1,+2] * episode_duration",
                "paper_like_alpha": float(args.paper_like_alpha),
                "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
                "soft_scenario_beta": float(args.soft_scenario_beta),
            },
            "hsr_soft_scenario_posterior_v7_7offset": {
                "source": "v3 enlarged onset grid with 7 offsets [-3,-2,-1,0,+1,+2,+3]×episode_duration + all-scenario soft weighting + EMA",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "onset_grid_mode": "[-3,-2,-1,0,+1,+2,+3] * episode_duration",
                "paper_like_alpha": float(args.paper_like_alpha),
                "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
                "soft_scenario_beta": float(args.soft_scenario_beta),
            },
            "hsr_soft_scenario_posterior_v6": {
                "source": "v3 + trigger injected as episode0 positive witness once, then same all-scenario soft weighting + EMA",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "onset_grid_mode": "[-episode_duration,0,+episode_duration]",
                "paper_like_alpha": float(args.paper_like_alpha),
                "paper_like_time_tol_min": float(args.paper_like_time_tol_min),
                "soft_scenario_beta": float(args.soft_scenario_beta),
            },
            "hsr_binary_loglik_posterior_source_only_v4": {
                "source": "source-only Bernoulli log-likelihood over binary observations + mild prior damping",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "tau_min": float(runtime["episode_duration_min"]),
                "binary_loglik_eps_fp": float(args.binary_loglik_eps_fp),
                "binary_loglik_eps_fn": float(args.binary_loglik_eps_fn),
                "binary_loglik_prior_mix": float(args.binary_loglik_prior_mix),
            },
            "hsr_binary_loglik_posterior_source_onset_v4": {
                "source": "source×onset Bernoulli log-likelihood + source marginalization + mild prior damping",
                "reasoner_logits_used": False,
                "clean_label": True,
                "trigger_anchor_used": True,
                "onset_grid_mode": "[-episode_duration,0,+episode_duration]",
                "tau_min": float(runtime["episode_duration_min"]),
                "binary_loglik_eps_fp": float(args.binary_loglik_eps_fp),
                "binary_loglik_eps_fn": float(args.binary_loglik_eps_fn),
                "binary_loglik_prior_mix": float(args.binary_loglik_prior_mix),
            },
        },
    }
    write_json(output_dir / "provenance_audit.json", provenance_audit)

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    for fam in families:
        family = fam["name"]
        for case_idx, case in enumerate(runtime["cases"], start=1):
            _seed_case(int(args.seed), family, case.case_id)
            case_data = case.data
            if case_data is None:
                dataset = runtime.get("full_train_dataset")
                if dataset is None:
                    raise RuntimeError("full_train_dataset is missing while case.data is None.")
                case_data = dataset[int(case.dataset_index)]
            rollout = PracticalRollout(
                event_data=deepcopy(case_data),
                global_edge_index=runtime["dataset_assets"]["global_edge_index"],
                stt_dynamic_series=runtime["dataset_assets"]["stt_dynamic_series"],
                num_global_nodes=int(runtime["dataset_assets"]["num_global_nodes"]),
                num_episodes=int(runtime["num_episodes"]),
                samples_per_episode=int(runtime["action_budget"]),
                episode_duration_min=float(runtime["episode_duration_min"]),
            )
            history = ObservationWitnessHistory()
            source_global = _extract_source_global(rollout)
            source_local = resolve_source_local_idx(rollout)
            trigger_global = _extract_trigger_global(case_data)
            hit_round = None
            final_candidate_fraction = None
            final_source_in_candidates = None
            paper_like_state = PaperLikeHSRState(source_prior=None)
            onset_grid = _resolve_onset_grid(
                family=family,
                episode_duration_min=float(runtime["episode_duration_min"]),
            )

            for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
                state = make_rollout_state(
                    case=case,
                    rollout=rollout,
                    history=history,
                    env=env,
                    topology=topology,
                    num_episodes=runtime["num_episodes"],
                    action_budget=runtime["action_budget"],
                    frontier_role_mode=runtime["frontier_role_mode"],
                )
                if int(state["valid_mask"].sum().item()) <= 0:
                    break
                if family == "hsr_paper_topk_ema_v1":
                    belief_ctx = _paper_topk_ema_posterior(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        onset_offsets_min=onset_grid,
                        alpha=float(args.paper_like_alpha),
                        topk_fraction=float(args.paper_like_topk_fraction),
                        time_tol_min=float(args.paper_like_time_tol_min),
                    )
                elif family == "hsr_rank_weighted_topk_v2":
                    belief_ctx = _rank_weighted_topk_posterior(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        onset_offsets_min=onset_grid,
                        alpha=float(args.paper_like_alpha),
                        topk_fraction=float(args.paper_like_topk_fraction),
                        time_tol_min=float(args.paper_like_time_tol_min),
                        beta=float(args.rank_weighted_beta),
                    )
                elif family == "hsr_soft_scenario_posterior_v3":
                    belief_ctx = _soft_scenario_posterior(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        onset_offsets_min=onset_grid,
                        alpha=float(args.paper_like_alpha),
                        time_tol_min=float(args.paper_like_time_tol_min),
                        beta=float(args.soft_scenario_beta),
                    )
                elif family == "hsr_soft_scenario_posterior_v7_5offset":
                    belief_ctx = _soft_scenario_posterior(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        onset_offsets_min=onset_grid,
                        alpha=float(args.paper_like_alpha),
                        time_tol_min=float(args.paper_like_time_tol_min),
                        beta=float(args.soft_scenario_beta),
                    )
                elif family == "hsr_soft_scenario_posterior_v7_7offset":
                    belief_ctx = _soft_scenario_posterior(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        onset_offsets_min=onset_grid,
                        alpha=float(args.paper_like_alpha),
                        time_tol_min=float(args.paper_like_time_tol_min),
                        beta=float(args.soft_scenario_beta),
                    )
                elif family == "hsr_soft_scenario_posterior_v6":
                    belief_ctx = _soft_scenario_posterior_v6(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        onset_offsets_min=onset_grid,
                        alpha=float(args.paper_like_alpha),
                        time_tol_min=float(args.paper_like_time_tol_min),
                        beta=float(args.soft_scenario_beta),
                    )
                elif family == "hsr_binary_loglik_posterior_source_only_v4":
                    belief_ctx = _binary_loglik_source_only_posterior(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        tau_min=float(runtime["episode_duration_min"]),
                        eps_fp=float(args.binary_loglik_eps_fp),
                        eps_fn=float(args.binary_loglik_eps_fn),
                        prior_mix=float(args.binary_loglik_prior_mix),
                    )
                elif family == "hsr_binary_loglik_posterior_source_onset_v4":
                    belief_ctx = _binary_loglik_source_onset_posterior(
                        rollout=rollout,
                        state=state,
                        history=history,
                        trigger_global=trigger_global,
                        paper_state=paper_like_state,
                        onset_offsets_min=onset_grid,
                        tau_min=float(runtime["episode_duration_min"]),
                        eps_fp=float(args.binary_loglik_eps_fp),
                        eps_fn=float(args.binary_loglik_eps_fn),
                        prior_mix=float(args.binary_loglik_prior_mix),
                    )
                else:
                    raise ValueError(family)

                metrics = _belief_metrics(belief_ctx["belief"], belief_ctx["candidate_mask"], source_local)
                selected_local = _pick_topk_unsampled(belief_ctx["belief"], belief_ctx["candidate_mask"], rollout, int(runtime["action_budget"]))
                if not selected_local:
                    break
                selected_global = [int(rollout.g_ids[int(idx)].item()) for idx in selected_local]
                round_hit = source_global is not None and int(source_global) in set(selected_global)
                if round_hit and hit_round is None:
                    hit_round = int(episode_idx)
                rollout.step_with_actions(selected_local, sample_types=[f"{family}_slot_{i}" for i in range(len(selected_local))])
                if rollout.history_steps:
                    history.append_from_history_step(rollout.history_steps[-1])
                step_rows.append(
                    {
                        "family": family,
                        "track": fam["track"],
                        "case_id": case.case_id,
                        "episode_index": int(episode_idx),
                        "selected_global_ids": json.dumps(selected_global),
                        "selected_local_ids": json.dumps(selected_local),
                        "source_hit_in_round": float(bool(round_hit)),
                        "posterior_top1_mass": float(metrics["top1_mass"]),
                        "posterior_top3_mass": float(metrics["top3_mass"]),
                        "normalized_entropy": float(metrics["normalized_entropy"]),
                        "mass_cover_size_ratio": float(metrics["mass_cover_size_ratio"]),
                        "candidate_fraction": belief_ctx.get("candidate_fraction"),
                        "source_in_candidate_set": (
                            float(source_local is not None and bool(belief_ctx["candidate_mask"][int(source_local)].item()))
                            if source_local is not None
                            else None
                        ),
                    }
                )

            final_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=runtime["num_episodes"],
                action_budget=runtime["action_budget"],
                frontier_role_mode=runtime["frontier_role_mode"],
            )
            if family == "hsr_paper_topk_ema_v1":
                final_belief = _paper_topk_ema_posterior(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    onset_offsets_min=onset_grid,
                    alpha=float(args.paper_like_alpha),
                    topk_fraction=float(args.paper_like_topk_fraction),
                    time_tol_min=float(args.paper_like_time_tol_min),
                )
            elif family == "hsr_rank_weighted_topk_v2":
                final_belief = _rank_weighted_topk_posterior(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    onset_offsets_min=onset_grid,
                    alpha=float(args.paper_like_alpha),
                    topk_fraction=float(args.paper_like_topk_fraction),
                    time_tol_min=float(args.paper_like_time_tol_min),
                    beta=float(args.rank_weighted_beta),
                )
            elif family == "hsr_soft_scenario_posterior_v3":
                final_belief = _soft_scenario_posterior(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    onset_offsets_min=onset_grid,
                    alpha=float(args.paper_like_alpha),
                    time_tol_min=float(args.paper_like_time_tol_min),
                    beta=float(args.soft_scenario_beta),
                )
            elif family == "hsr_soft_scenario_posterior_v7_5offset":
                final_belief = _soft_scenario_posterior(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    onset_offsets_min=onset_grid,
                    alpha=float(args.paper_like_alpha),
                    time_tol_min=float(args.paper_like_time_tol_min),
                    beta=float(args.soft_scenario_beta),
                )
            elif family == "hsr_soft_scenario_posterior_v7_7offset":
                final_belief = _soft_scenario_posterior(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    onset_offsets_min=onset_grid,
                    alpha=float(args.paper_like_alpha),
                    time_tol_min=float(args.paper_like_time_tol_min),
                    beta=float(args.soft_scenario_beta),
                )
            elif family == "hsr_soft_scenario_posterior_v6":
                final_belief = _soft_scenario_posterior_v6(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    onset_offsets_min=onset_grid,
                    alpha=float(args.paper_like_alpha),
                    time_tol_min=float(args.paper_like_time_tol_min),
                    beta=float(args.soft_scenario_beta),
                )
            elif family == "hsr_binary_loglik_posterior_source_only_v4":
                final_belief = _binary_loglik_source_only_posterior(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    tau_min=float(runtime["episode_duration_min"]),
                    eps_fp=float(args.binary_loglik_eps_fp),
                    eps_fn=float(args.binary_loglik_eps_fn),
                    prior_mix=float(args.binary_loglik_prior_mix),
                )
            elif family == "hsr_binary_loglik_posterior_source_onset_v4":
                final_belief = _binary_loglik_source_onset_posterior(
                    rollout=rollout,
                    state=final_state,
                    history=history,
                    trigger_global=trigger_global,
                    paper_state=paper_like_state,
                    onset_offsets_min=onset_grid,
                    tau_min=float(runtime["episode_duration_min"]),
                    eps_fp=float(args.binary_loglik_eps_fp),
                    eps_fn=float(args.binary_loglik_eps_fn),
                    prior_mix=float(args.binary_loglik_prior_mix),
                )
            else:
                raise ValueError(family)

            final_metrics = _belief_metrics(final_belief["belief"], final_belief["candidate_mask"], source_local)
            case_rows.append(
                {
                    "family": family,
                    "track": fam["track"],
                    "case_id": case.case_id,
                    "success_rate": float(hit_round is not None),
                    "hit_round": hit_round,
                    "final_top1_hit": float((final_metrics["true_rank"] or 10**9) <= 1),
                    "final_top3_hit": float((final_metrics["true_rank"] or 10**9) <= 3),
                    "final_top5_hit": float((final_metrics["true_rank"] or 10**9) <= 5),
                    "final_mrr": float(1.0 / final_metrics["true_rank"]) if final_metrics["true_rank"] is not None else 0.0,
                    "final_true_mass": final_metrics["true_mass"],
                    "final_normalized_entropy": float(final_metrics["normalized_entropy"]),
                    "final_mass_cover_size_ratio": float(final_metrics["mass_cover_size_ratio"]),
                    "final_candidate_fraction": final_candidate_fraction,
                    "final_source_in_candidate_set_rate": final_source_in_candidates,
                    "budget_used": float(rollout.revealed_mask.sum().item()),
                }
            )
            if case_idx % int(args.progress_every_cases) == 0 or case_idx == len(runtime["cases"]):
                pd.DataFrame(case_rows).to_csv(output_dir / "case_rows.partial.csv", index=False)
                pd.DataFrame(step_rows).to_csv(output_dir / "step_rows.partial.csv", index=False)
                print(f"[progress] family={family} case={case_idx}/{len(runtime['cases'])}", flush=True)

    case_df = pd.DataFrame(case_rows)
    step_df = pd.DataFrame(step_rows)
    case_df.to_csv(output_dir / "case_rows.csv", index=False)
    step_df.to_csv(output_dir / "step_rows.csv", index=False)

    summary_rows = []
    round_rows = []
    budget_rows = []
    for family, sub in case_df.groupby("family"):
        fam_step = step_df[step_df["family"] == family]
        row = {
            "family": family,
            "track": sub["track"].iloc[0],
            "case_count": int(len(sub)),
            "success_rate": float(sub["success_rate"].mean()),
            "avg_hit_round_conditional": float(sub["hit_round"].dropna().mean()) if sub["hit_round"].notna().any() else None,
            "final_top1_hit": float(sub["final_top1_hit"].mean()),
            "final_top3_hit": float(sub["final_top3_hit"].mean()),
            "final_top5_hit": float(sub["final_top5_hit"].mean()),
            "final_mrr": float(sub["final_mrr"].mean()),
            "final_true_mass_mean": float(sub["final_true_mass"].dropna().mean()) if sub["final_true_mass"].notna().any() else None,
            "final_normalized_entropy_mean": float(sub["final_normalized_entropy"].mean()),
            "final_mass_cover_size_ratio_mean": float(sub["final_mass_cover_size_ratio"].mean()),
            "final_candidate_fraction_mean": float(sub["final_candidate_fraction"].dropna().mean()) if sub["final_candidate_fraction"].notna().any() else None,
            "final_source_in_candidate_set_rate": float(sub["final_source_in_candidate_set_rate"].dropna().mean()) if sub["final_source_in_candidate_set_rate"].notna().any() else None,
        }
        summary_rows.append(row)
        for r in range(1, int(args.num_rounds) + 1):
            hit_mask = sub["hit_round"].fillna(10**9) <= int(r)
            round_rows.append({"family": family, "round_index": int(r), "cumulative_success_rate": float(hit_mask.mean())})
        for b in range(1, int(args.num_rounds) * int(args.actions_per_round) + 1):
            hit_mask = sub["hit_round"].fillna(10**9) <= math.ceil(float(b) / float(args.actions_per_round))
            budget_rows.append({"family": family, "sample_budget": int(b), "cumulative_success_rate": float(hit_mask.mean())})

    summary_df = pd.DataFrame(summary_rows).sort_values(["track", "success_rate", "final_mrr"], ascending=[True, False, False])
    summary_df.to_csv(output_dir / "family_summary.csv", index=False)
    pd.DataFrame(round_rows).to_csv(output_dir / "roundwise_success_curve.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(output_dir / "budget_success_curve.csv", index=False)

    clean_df = summary_df[summary_df["track"] == "clean"].sort_values(["success_rate", "final_mrr"], ascending=[False, False])
    exploratory_df = summary_df.sort_values(["success_rate", "final_mrr"], ascending=[False, False])
    clean_df.to_csv(output_dir / "clean_leaderboard.csv", index=False)
    exploratory_df.to_csv(output_dir / "exploratory_leaderboard.csv", index=False)

    sanity_checks = []
    for family, sub in case_df.groupby("family"):
        fam_step = step_df[step_df["family"] == family]
        budgets = budget_rows
        fam_budget = [row for row in budgets if row["family"] == family]
        mono = all(fam_budget[i]["cumulative_success_rate"] <= fam_budget[i + 1]["cumulative_success_rate"] + 1e-12 for i in range(len(fam_budget) - 1))
        sanity_checks.append(
            {
                "family": family,
                "case_count_matches": bool(len(sub) == len(runtime["cases"])),
                "budget_curve_monotonic": bool(mono),
                "nan_in_core_metrics": bool(
                    sub[["success_rate", "final_top1_hit", "final_top3_hit", "final_top5_hit", "final_mrr"]].isna().any().any()
                ),
                "high_sr_low_rank_flag": bool(float(sub["success_rate"].mean()) >= 0.75 and float(sub["final_top5_hit"].mean()) <= 0.05),
                "steps_recorded": int(len(fam_step)),
            }
        )
    write_json(output_dir / "sanity_checks.json", {"checks": sanity_checks})

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "cache_version": cache_dir.name,
        "source_root": str(source_root),
        "foundation_graph_path": str(foundation_graph_path),
        "reasoner_checkpoint": None,
        "families": families,
        "clean_winner": clean_df.iloc[0].to_dict(),
        "exploratory_winner": exploratory_df.iloc[0].to_dict(),
        "authoritative_reference": {
            "summary": str(authoritative_ref / "summary.json"),
            "protocol_audit": str(authoritative_ref / "protocol_audit.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
