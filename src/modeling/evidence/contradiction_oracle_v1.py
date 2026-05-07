"""
Contradiction oracle / practical compare utilities.

EvidenceState v1 status:
- contradiction is frozen as a sparse auxiliary / audit-explanation branch
- these routines remain for audit compare, explainability, and future observability research
- they are not the default training or ranking path
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.state.schema import PhysicsContext

DEFAULT_SAFE_VIOLATION_TAU_MIN = 15.0
DEFAULT_TOP_K_WITNESSES = 5
DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE = "witness_time_physctx"
DEFAULT_TOP_K_MARGIN_SUMMARY = 3
DEFAULT_WITNESS_MINING_MODE = "baseline_latest_safe"
DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN = 60.0
DEFAULT_MINED_TOP_K_SAFE_WITNESSES = 3
DEFAULT_ADMISSIBILITY_MODE = "baseline_admissibility"
DEFAULT_TIME_BRACKETING_RELAX_SLACK_MIN = 45.0
DEFAULT_RELAXED_FRONTIER_WINDOW_TAU_MIN = 120.0
VALID_ORACLE_HISTORY_PHYSCTX_MODES = {
    "witness_time_physctx",
    "current_time_physctx",
}
VALID_WITNESS_MINING_MODES = {
    "baseline_latest_safe",
    "candidate_conditioned_frontier_safe",
    "candidate_conditioned_topk_safe",
}
VALID_ADMISSIBILITY_MODES = {
    "baseline_admissibility",
    "topology_relaxed_compare",
    "time_bracketing_relaxed_compare",
    "frontier_window_relaxed_compare",
    "union_relaxed_upper_bound",
}
DEFAULT_ADMISSIBILITY_COMPARE_MODES = (
    "baseline_admissibility",
    "topology_relaxed_compare",
    "time_bracketing_relaxed_compare",
    "frontier_window_relaxed_compare",
    "union_relaxed_upper_bound",
)


@dataclass
class OracleHistorySample:
    local_idx: int
    global_idx: int
    time_min: float
    t_snapshot_idx: int
    is_positive: bool
    is_safe: bool
    sample_type: str
    concentration: float
    signal: float


@dataclass
class OracleHistoryStep:
    episode: int
    time_min: float
    t_snapshot_idx: int
    phys_ctx: PhysicsContext
    samples: List[OracleHistorySample]
    absolute_snapshot_idx: Optional[int] = None


@dataclass(frozen=True)
class PracticalContradictionV2Config:
    label: str = "safe_dominant_norm"
    gap_cap_min: float = 60.0
    gap_log_tau_min: float = 15.0
    soft_count_tau_min: float = 12.0
    near_safe_tau_min: float = 10.0
    near_safe_slack_min: float = 20.0
    alpha_gap: float = 0.35
    beta_near_safe: float = 1.0
    gamma_soft_count: float = 0.75
    normalize_by_eligible_safe_count: bool = True
    top_k_margin_summary: int = DEFAULT_TOP_K_MARGIN_SUMMARY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "gap_cap_min": float(self.gap_cap_min),
            "gap_log_tau_min": float(self.gap_log_tau_min),
            "soft_count_tau_min": float(self.soft_count_tau_min),
            "near_safe_tau_min": float(self.near_safe_tau_min),
            "near_safe_slack_min": float(self.near_safe_slack_min),
            "alpha_gap": float(self.alpha_gap),
            "beta_near_safe": float(self.beta_near_safe),
            "gamma_soft_count": float(self.gamma_soft_count),
            "normalize_by_eligible_safe_count": bool(self.normalize_by_eligible_safe_count),
            "top_k_margin_summary": int(self.top_k_margin_summary),
        }


@dataclass(frozen=True)
class AdmissibilityCompareConfig:
    time_bracketing_relax_slack_min: float = DEFAULT_TIME_BRACKETING_RELAX_SLACK_MIN
    frontier_window_relax_tau_min: float = DEFAULT_RELAXED_FRONTIER_WINDOW_TAU_MIN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_bracketing_relax_slack_min": float(self.time_bracketing_relax_slack_min),
            "frontier_window_relax_tau_min": float(self.frontier_window_relax_tau_min),
        }


def _validate_phys_ctx_mode(phys_ctx_mode: str) -> str:
    mode = str(phys_ctx_mode)
    if mode not in VALID_ORACLE_HISTORY_PHYSCTX_MODES:
        raise ValueError(f"Unknown phys_ctx_mode: {mode}")
    return mode


def _validate_witness_mining_mode(witness_mining_mode: str) -> str:
    mode = str(witness_mining_mode)
    if mode not in VALID_WITNESS_MINING_MODES:
        raise ValueError(f"Unknown witness_mining_mode: {mode}")
    return mode


def _validate_admissibility_mode(admissibility_mode: str) -> str:
    mode = str(admissibility_mode)
    if mode not in VALID_ADMISSIBILITY_MODES:
        raise ValueError(f"Unknown admissibility_mode: {mode}")
    return mode


def _resolve_admissibility_mode_args(
    admissibility_mode: str,
    frontier_safe_close_tau_min: float,
    compare_config: AdmissibilityCompareConfig,
) -> Dict[str, float]:
    mode = _validate_admissibility_mode(admissibility_mode)
    args = {
        "use_history_safe_view": 0.0,
        "time_bracketing_margin_floor_min": 0.0,
        "frontier_close_tau_min": float(frontier_safe_close_tau_min),
    }
    if mode == "topology_relaxed_compare":
        args["use_history_safe_view"] = 1.0
        return args
    if mode == "time_bracketing_relaxed_compare":
        args["time_bracketing_margin_floor_min"] = -float(compare_config.time_bracketing_relax_slack_min)
        return args
    if mode == "frontier_window_relaxed_compare":
        args["frontier_close_tau_min"] = max(
            float(frontier_safe_close_tau_min),
            float(compare_config.frontier_window_relax_tau_min),
        )
        return args
    if mode == "union_relaxed_upper_bound":
        args["use_history_safe_view"] = 1.0
        args["time_bracketing_margin_floor_min"] = -float(compare_config.time_bracketing_relax_slack_min)
        args["frontier_close_tau_min"] = max(
            float(frontier_safe_close_tau_min),
            float(compare_config.frontier_window_relax_tau_min),
        )
        return args
    return args


def _resolve_distance_weights(phys_ctx: PhysicsContext) -> torch.Tensor:
    if phys_ctx.stt_dynamic is not None:
        return torch.abs(phys_ctx.stt_dynamic.view(-1))
    if phys_ctx.stt_median is not None:
        return torch.expm1(phys_ctx.stt_median)
    if phys_ctx.edge_attr is not None and phys_ctx.edge_attr.size(1) > 0:
        return torch.expm1(phys_ctx.edge_attr[:, 0])
    return torch.ones_like(phys_ctx.edge_index[0], dtype=torch.float) * 20.0


def compress_oracle_history_steps(history_steps: Sequence[OracleHistoryStep]) -> Dict[str, List[Dict[str, Any]]]:
    positive_by_node: Dict[int, Dict[str, Any]] = {}
    safe_by_node: Dict[int, Dict[str, Any]] = {}

    for step_idx, step in enumerate(history_steps):
        for sample in step.samples:
            record = {
                "local_idx": int(sample.local_idx),
                "global_idx": int(sample.global_idx),
                "episode": int(step.episode),
                "time_min": float(sample.time_min),
                "t_snapshot_idx": int(sample.t_snapshot_idx),
                "sample_type": str(sample.sample_type),
                "concentration": float(sample.concentration),
                "signal": float(sample.signal),
                "phys_ctx": step.phys_ctx,
                "history_step_idx": int(step_idx),
            }
            if sample.is_positive:
                existing = positive_by_node.get(sample.local_idx)
                if existing is None or float(sample.time_min) < float(existing["time_min"]):
                    positive_by_node[sample.local_idx] = record
            if sample.is_safe:
                existing = safe_by_node.get(sample.local_idx)
                if existing is None or float(sample.time_min) > float(existing["time_min"]):
                    safe_by_node[sample.local_idx] = record

    positive_records = sorted(
        positive_by_node.values(),
        key=lambda row: (float(row["time_min"]), int(row["local_idx"])),
    )
    safe_records = sorted(
        safe_by_node.values(),
        key=lambda row: (float(row["time_min"]), int(row["local_idx"])),
    )
    return {
        "positive_records": positive_records,
        "safe_records": safe_records,
    }


def _stack_arrival_times(
    reachability_module: DynamicReachabilityRuleModule,
    records: Sequence[Dict[str, Any]],
    num_nodes: int,
    device: torch.device,
    phys_ctx_mode: str = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    current_phys_ctx: Optional[PhysicsContext] = None,
) -> torch.Tensor:
    if not records:
        return torch.zeros((num_nodes, 0), device=device)

    mode = _validate_phys_ctx_mode(phys_ctx_mode)
    arrivals: List[torch.Tensor] = []
    for record in records:
        seed = torch.zeros(num_nodes, device=device)
        seed[int(record["local_idx"])] = 1.0
        phys_ctx = current_phys_ctx if mode == "current_time_physctx" else record["phys_ctx"]
        if phys_ctx is None:
            raise ValueError("current_phys_ctx must be provided when phys_ctx_mode='current_time_physctx'")
        weights = _resolve_distance_weights(phys_ctx).to(device)
        dist = reachability_module.compute_distance(seed, phys_ctx, weights, num_nodes)
        arrivals.append(dist)
    return torch.stack(arrivals, dim=1)


def _stack_arrival_times_under_fixed_phys_ctx(
    reachability_module: DynamicReachabilityRuleModule,
    records: Sequence[Dict[str, Any]],
    num_nodes: int,
    device: torch.device,
    phys_ctx: PhysicsContext,
) -> torch.Tensor:
    if not records:
        return torch.zeros((num_nodes, 0), device=device)
    if phys_ctx.stt_dynamic is None:
        return _stack_arrival_times(
            reachability_module=reachability_module,
            records=records,
            num_nodes=num_nodes,
            device=device,
            phys_ctx_mode="current_time_physctx",
            current_phys_ctx=phys_ctx,
        )

    seed_indices = [int(record["local_idx"]) for record in records]
    if not seed_indices:
        return torch.zeros((num_nodes, 0), device=device)

    adj_rev = reachability_module._build_scipy_reverse_graph(
        phys_ctx.edge_index,
        phys_ctx.stt_dynamic.view(-1),
        num_nodes,
    )
    dist_matrix = reachability_module._run_scipy_dijkstra(adj_rev, seed_indices)
    dist_tensor = torch.from_numpy(dist_matrix).float().to(device)
    if dist_tensor.ndim == 1:
        dist_tensor = dist_tensor.unsqueeze(0)
    return dist_tensor.transpose(0, 1).contiguous()


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros((values.size(0),))
    masked = torch.where(mask, values, torch.full_like(values, float("-inf")))
    topk = min(max(int(k), 1), values.size(1))
    topk_values, _ = torch.topk(masked, k=topk, dim=1)
    valid_topk = torch.isfinite(topk_values)
    safe_topk_values = torch.where(valid_topk, topk_values, torch.zeros_like(topk_values))
    denom = valid_topk.float().sum(dim=1).clamp_min(1.0)
    return safe_topk_values.sum(dim=1) / denom


def _apply_candidate_mask(
    result: Dict[str, Any],
    suspect_pool: Optional[torch.Tensor],
    keys: Sequence[str],
) -> Dict[str, Any]:
    if suspect_pool is None:
        return result
    pool = suspect_pool.to(result["interval_gap"].device)
    for key in keys:
        if key in result and isinstance(result[key], torch.Tensor) and result[key].shape == pool.shape:
            result[key] = result[key] * pool
    return result


def _empty_history_contradiction_result(
    num_nodes: int,
    device: torch.device,
    compressed_history: Dict[str, List[Dict[str, Any]]],
    safe_violation_tau_min: float,
    phys_ctx_mode: str,
    witness_mining_mode: str,
    frontier_safe_close_tau_min: float,
    mined_top_k_safe_witnesses: int,
) -> Dict[str, Any]:
    zero = torch.zeros(num_nodes, device=device)
    neg_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    neg_inf = torch.full((num_nodes,), float("-inf"), device=device)
    pos_inf = torch.full((num_nodes,), float("inf"), device=device)
    empty_matrix = torch.zeros((num_nodes, 0), device=device)
    empty_bool = torch.zeros((num_nodes, 0), dtype=torch.bool, device=device)
    has_safe_observation_anywhere = torch.full(
        (num_nodes,),
        1.0 if len(compressed_history["safe_records"]) > 0 else 0.0,
        device=device,
    )
    return {
        "mode": "history_base",
        "interval_gap": zero,
        "interval_gap_term": zero,
        "safe_violation": zero,
        "violated_safe_count": zero,
        "positive_reachable_count": zero,
        "safe_reachable_count": zero,
        "pair_count": zero,
        "positive_margin_pair_count": zero,
        "non_positive_margin_pair_count": zero,
        "pair_available": zero,
        "positive_margin_available": zero,
        "upper_bound_finite": zero,
        "lower_bound_finite": zero,
        "interval_bounds_available": zero,
        "interval_regime_available": zero,
        "eligible_safe_witness_count": zero,
        "positive_margin_count": zero,
        "best_margin_topk_mean": zero,
        "top_witness_margin": zero,
        "top_witness_safe_local_idx": neg_idx,
        "top_witness_pos_local_idx": neg_idx,
        "upper_bound": pos_inf,
        "lower_bound": neg_inf,
        "best_margin_per_safe": empty_matrix,
        "best_pos_idx_per_safe": empty_matrix.long(),
        "safe_pair_available": empty_bool,
        "compressed_history": compressed_history,
        "positive_count": len(compressed_history["positive_records"]),
        "safe_count": len(compressed_history["safe_records"]),
        "safe_violation_tau_min": float(safe_violation_tau_min),
        "phys_ctx_mode": str(phys_ctx_mode),
        "witness_mining_mode": str(witness_mining_mode),
        "frontier_safe_close_tau_min": float(frontier_safe_close_tau_min),
        "mined_top_k_safe_witnesses": int(mined_top_k_safe_witnesses),
        "mined_safe_candidate_count": zero,
        "hydraulic_comparable_safe_count": zero,
        "front_close_safe_count": zero,
        "frontier_safe_count": zero,
        "selected_safe_witness_count": zero,
        "topk_safe_count": zero,
        "pair_available_after_mining": zero,
        "positive_margin_available_after_mining": zero,
        "time_bracketing_safe_count": zero,
        "frontier_window_safe_count": zero,
        "has_safe_observation_anywhere": has_safe_observation_anywhere,
        "has_hydraulically_comparable_safe": zero,
        "has_time_bracketing_safe": zero,
        "has_frontier_window_safe": zero,
        "pair_available_under_mode": zero,
        "positive_margin_available_under_mode": zero,
        "admissibility_mode": DEFAULT_ADMISSIBILITY_MODE,
    }


def _build_candidate_conditioned_pair_stats(
    reachability_module: DynamicReachabilityRuleModule,
    positive_records: Sequence[Dict[str, Any]],
    safe_records: Sequence[Dict[str, Any]],
    pos_arrivals: torch.Tensor,
    pos_finite: torch.Tensor,
    pos_times: torch.Tensor,
    safe_arrivals_history: torch.Tensor,
    safe_finite_history: torch.Tensor,
    safe_times: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    inf_thresh = float(reachability_module.infinity / 2)
    safe_arrivals_per_pos_ctx: List[torch.Tensor] = []
    for pos_record in positive_records:
        safe_arrivals_per_pos_ctx.append(
            _stack_arrival_times_under_fixed_phys_ctx(
                reachability_module=reachability_module,
                records=safe_records,
                num_nodes=num_nodes,
                device=device,
                phys_ctx=pos_record["phys_ctx"],
            )
        )
    safe_arrivals_posctx = torch.stack(safe_arrivals_per_pos_ctx, dim=2)
    safe_finite_posctx = safe_arrivals_posctx < inf_thresh
    comparable_pair_posctx = safe_finite_posctx & pos_finite.unsqueeze(1)
    obs_margin = safe_times.view(1, -1, 1) - pos_times.view(1, 1, -1)
    frontier_margin_posctx = obs_margin - (safe_arrivals_posctx - pos_arrivals.unsqueeze(1))
    frontier_margin_posctx = frontier_margin_posctx.masked_fill(~comparable_pair_posctx, float("-inf"))
    arrival_gap_posctx = torch.abs(safe_arrivals_posctx - pos_arrivals.unsqueeze(1))
    arrival_gap_posctx = arrival_gap_posctx.masked_fill(~comparable_pair_posctx, float("inf"))

    comparable_pair_history = safe_finite_history.unsqueeze(2) & pos_finite.unsqueeze(1)
    safe_arrivals_history_3d = safe_arrivals_history.unsqueeze(2)
    frontier_margin_history = obs_margin - (safe_arrivals_history_3d - pos_arrivals.unsqueeze(1))
    frontier_margin_history = frontier_margin_history.masked_fill(~comparable_pair_history, float("-inf"))
    arrival_gap_history = torch.abs(safe_arrivals_history_3d - pos_arrivals.unsqueeze(1))
    arrival_gap_history = arrival_gap_history.masked_fill(~comparable_pair_history, float("inf"))

    return {
        "comparable_pair_posctx": comparable_pair_posctx,
        "frontier_margin_posctx": frontier_margin_posctx,
        "arrival_gap_posctx": arrival_gap_posctx,
        "comparable_pair_history": comparable_pair_history,
        "frontier_margin_history": frontier_margin_history,
        "arrival_gap_history": arrival_gap_history,
    }


def _combine_candidate_conditioned_pair_views(
    pair_stats: Dict[str, torch.Tensor],
    use_history_safe_view: bool,
) -> Dict[str, torch.Tensor]:
    comparable_pair = pair_stats["comparable_pair_posctx"]
    frontier_margin = pair_stats["frontier_margin_posctx"]
    arrival_gap = pair_stats["arrival_gap_posctx"]
    if not use_history_safe_view:
        return {
            "comparable_pair": comparable_pair,
            "frontier_margin": frontier_margin,
            "arrival_gap": arrival_gap,
        }

    comparable_pair = comparable_pair | pair_stats["comparable_pair_history"]
    frontier_margin = torch.maximum(
        pair_stats["frontier_margin_posctx"],
        pair_stats["frontier_margin_history"],
    )
    frontier_margin = frontier_margin.masked_fill(~comparable_pair, float("-inf"))
    arrival_gap = torch.minimum(
        pair_stats["arrival_gap_posctx"],
        pair_stats["arrival_gap_history"],
    )
    arrival_gap = arrival_gap.masked_fill(~comparable_pair, float("inf"))
    return {
        "comparable_pair": comparable_pair,
        "frontier_margin": frontier_margin,
        "arrival_gap": arrival_gap,
    }


def _evaluate_candidate_conditioned_admissibility_mode(
    positive_records: Sequence[Dict[str, Any]],
    safe_records: Sequence[Dict[str, Any]],
    comparable_pair: torch.Tensor,
    frontier_margin: torch.Tensor,
    arrival_gap: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    safe_violation_tau_min: float,
    witness_mining_mode: str,
    frontier_close_tau_min: float,
    mined_top_k_safe_witnesses: int,
    admissibility_mode: str,
    time_bracketing_margin_floor_min: float,
) -> Dict[str, Any]:
    zero = torch.zeros(num_nodes, device=device)
    neg_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    neg_inf = torch.full((num_nodes, len(safe_records)), float("-inf"), device=device)
    if not positive_records or not safe_records:
        return {
            "safe_violation": zero,
            "violated_safe_count": zero,
            "safe_reachable_count": zero,
            "pair_count": zero,
            "positive_margin_pair_count": zero,
            "non_positive_margin_pair_count": zero,
            "pair_available": zero,
            "positive_margin_available": zero,
            "eligible_safe_witness_count": zero,
            "positive_margin_count": zero,
            "best_margin_topk_mean": zero,
            "top_witness_margin": zero,
            "top_witness_safe_local_idx": neg_idx,
            "top_witness_pos_local_idx": neg_idx,
            "best_margin_per_safe": neg_inf,
            "best_pos_idx_per_safe": neg_inf.long(),
            "safe_pair_available": torch.zeros((num_nodes, len(safe_records)), dtype=torch.bool, device=device),
            "mined_safe_candidate_count": zero,
            "hydraulic_comparable_safe_count": zero,
            "front_close_safe_count": zero,
            "frontier_safe_count": zero,
            "selected_safe_witness_count": zero,
            "topk_safe_count": zero,
            "pair_available_after_mining": zero,
            "positive_margin_available_after_mining": zero,
            "time_bracketing_safe_count": zero,
            "frontier_window_safe_count": zero,
            "has_safe_observation_anywhere": zero,
            "has_hydraulically_comparable_safe": zero,
            "has_time_bracketing_safe": zero,
            "has_frontier_window_safe": zero,
            "pair_available_under_mode": zero,
            "positive_margin_available_under_mode": zero,
            "admissibility_mode": str(admissibility_mode),
            "time_bracketing_margin_floor_min": float(time_bracketing_margin_floor_min),
            "admissibility_frontier_close_tau_min": float(frontier_close_tau_min),
        }

    positive_margin_pair = comparable_pair & (frontier_margin > 0.0)
    time_bracketing_pair = comparable_pair & (frontier_margin >= float(time_bracketing_margin_floor_min))
    front_close_pair = comparable_pair & (arrival_gap <= float(frontier_close_tau_min))
    mining_pair = time_bracketing_pair | front_close_pair

    safe_comparable = comparable_pair.any(dim=2)
    safe_time_bracketed = time_bracketing_pair.any(dim=2)
    safe_front_close = front_close_pair.any(dim=2)
    safe_mined = mining_pair.any(dim=2)

    score = (
        100.0 * front_close_pair.float()
        + 10.0 * positive_margin_pair.float()
        + torch.clamp(
            frontier_margin / max(float(safe_violation_tau_min), 1e-6),
            min=-10.0,
            max=10.0,
        )
        - arrival_gap / max(float(frontier_close_tau_min), 1.0)
    )
    score = score.masked_fill(~mining_pair, float("-inf"))

    best_score_per_safe, best_pos_idx_per_safe = score.max(dim=2)
    best_margin_per_safe = frontier_margin.gather(2, best_pos_idx_per_safe.unsqueeze(2)).squeeze(2)
    best_margin_per_safe = torch.where(
        safe_mined,
        best_margin_per_safe,
        torch.full_like(best_margin_per_safe, float("-inf")),
    )
    best_pos_idx_per_safe = torch.where(safe_mined, best_pos_idx_per_safe, neg_idx.unsqueeze(1))

    selected_k = 1 if witness_mining_mode == "candidate_conditioned_frontier_safe" else max(
        int(mined_top_k_safe_witnesses), 1
    )
    safe_scores = best_score_per_safe.masked_fill(~safe_mined, float("-inf"))
    topk = min(selected_k, safe_scores.size(1))
    selected_safe_mask = torch.zeros_like(safe_mined)
    if topk > 0:
        topk_values, topk_idx = torch.topk(safe_scores, k=topk, dim=1)
        valid_topk = torch.isfinite(topk_values)
        selected_safe_mask.scatter_(1, topk_idx, valid_topk)

    selected_best_margin_per_safe = torch.where(
        selected_safe_mask,
        best_margin_per_safe,
        torch.full_like(best_margin_per_safe, float("-inf")),
    )
    selected_best_pos_idx_per_safe = torch.where(
        selected_safe_mask,
        best_pos_idx_per_safe,
        neg_idx.unsqueeze(1),
    )

    safe_violation_terms = torch.where(
        selected_safe_mask,
        F.softplus(selected_best_margin_per_safe / max(float(safe_violation_tau_min), 1e-6)),
        torch.zeros_like(selected_best_margin_per_safe),
    )
    safe_violation = safe_violation_terms.sum(dim=1)
    selected_positive_margin_count = (selected_safe_mask & (selected_best_margin_per_safe > 0.0)).float().sum(dim=1)
    pair_count = mining_pair.float().sum(dim=(1, 2))
    positive_margin_pair_count = (mining_pair & positive_margin_pair).float().sum(dim=(1, 2))
    non_positive_margin_pair_count = torch.clamp(pair_count - positive_margin_pair_count, min=0.0)
    pair_available = (pair_count > 0.0).float()
    positive_margin_available = (positive_margin_pair_count > 0.0).float()
    best_margin_topk_mean = _masked_topk_mean(
        selected_best_margin_per_safe,
        selected_safe_mask,
        k=max(int(mined_top_k_safe_witnesses), 1),
    )

    has_selected_witness = selected_safe_mask.any(dim=1)
    top_witness_margin, top_safe_pos = selected_best_margin_per_safe.max(dim=1)
    top_witness_margin = torch.where(has_selected_witness, top_witness_margin, torch.zeros_like(top_witness_margin))
    top_safe_pos = torch.where(has_selected_witness, top_safe_pos, neg_idx)
    top_pos_pos = selected_best_pos_idx_per_safe.gather(
        1,
        top_safe_pos.clamp_min(0).unsqueeze(1),
    ).squeeze(1)
    top_pos_pos = torch.where(has_selected_witness, top_pos_pos, neg_idx)

    safe_local_idx_lookup = torch.tensor(
        [int(record["local_idx"]) for record in safe_records],
        dtype=torch.long,
        device=device,
    )
    pos_local_idx_lookup = torch.tensor(
        [int(record["local_idx"]) for record in positive_records],
        dtype=torch.long,
        device=device,
    )
    top_witness_safe_local_idx = torch.where(
        top_safe_pos >= 0,
        safe_local_idx_lookup[top_safe_pos.clamp_min(0)],
        neg_idx,
    )
    top_witness_pos_local_idx = torch.where(
        top_pos_pos >= 0,
        pos_local_idx_lookup[top_pos_pos.clamp_min(0)],
        neg_idx,
    )

    selected_safe_witness_count = selected_safe_mask.float().sum(dim=1)
    mined_safe_candidate_count = safe_mined.float().sum(dim=1)
    hydraulic_comparable_safe_count = safe_comparable.float().sum(dim=1)
    time_bracketing_safe_count = safe_time_bracketed.float().sum(dim=1)
    frontier_window_safe_count = safe_front_close.float().sum(dim=1)

    return {
        "safe_violation": safe_violation,
        "violated_safe_count": selected_positive_margin_count,
        "safe_reachable_count": hydraulic_comparable_safe_count,
        "pair_count": pair_count,
        "positive_margin_pair_count": positive_margin_pair_count,
        "non_positive_margin_pair_count": non_positive_margin_pair_count,
        "pair_available": pair_available,
        "positive_margin_available": positive_margin_available,
        "eligible_safe_witness_count": selected_safe_witness_count,
        "positive_margin_count": selected_positive_margin_count,
        "best_margin_topk_mean": best_margin_topk_mean,
        "top_witness_margin": top_witness_margin,
        "top_witness_safe_local_idx": top_witness_safe_local_idx,
        "top_witness_pos_local_idx": top_witness_pos_local_idx,
        "best_margin_per_safe": selected_best_margin_per_safe,
        "best_pos_idx_per_safe": selected_best_pos_idx_per_safe,
        "safe_pair_available": selected_safe_mask,
        "mined_safe_candidate_count": mined_safe_candidate_count,
        "hydraulic_comparable_safe_count": hydraulic_comparable_safe_count,
        "front_close_safe_count": frontier_window_safe_count,
        "frontier_safe_count": frontier_window_safe_count,
        "selected_safe_witness_count": selected_safe_witness_count,
        "topk_safe_count": selected_safe_witness_count,
        "pair_available_after_mining": pair_available,
        "positive_margin_available_after_mining": positive_margin_available,
        "time_bracketing_safe_count": time_bracketing_safe_count,
        "frontier_window_safe_count": frontier_window_safe_count,
        "has_safe_observation_anywhere": torch.full(
            (num_nodes,),
            1.0 if len(safe_records) > 0 else 0.0,
            device=device,
        ),
        "has_hydraulically_comparable_safe": (hydraulic_comparable_safe_count > 0.0).float(),
        "has_time_bracketing_safe": (time_bracketing_safe_count > 0.0).float(),
        "has_frontier_window_safe": (frontier_window_safe_count > 0.0).float(),
        "pair_available_under_mode": pair_available,
        "positive_margin_available_under_mode": positive_margin_available,
        "admissibility_mode": str(admissibility_mode),
        "time_bracketing_margin_floor_min": float(time_bracketing_margin_floor_min),
        "admissibility_frontier_close_tau_min": float(frontier_close_tau_min),
    }


def _mine_candidate_conditioned_safe_witnesses(
    reachability_module: DynamicReachabilityRuleModule,
    positive_records: Sequence[Dict[str, Any]],
    safe_records: Sequence[Dict[str, Any]],
    pos_arrivals: torch.Tensor,
    pos_finite: torch.Tensor,
    pos_times: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    safe_violation_tau_min: float,
    witness_mining_mode: str,
    frontier_safe_close_tau_min: float,
    mined_top_k_safe_witnesses: int,
    safe_arrivals_history: torch.Tensor,
    safe_finite_history: torch.Tensor,
    admissibility_mode: str = DEFAULT_ADMISSIBILITY_MODE,
    compare_config: Optional[AdmissibilityCompareConfig] = None,
    diagnostic_compare_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    compare_cfg = compare_config or AdmissibilityCompareConfig()
    pair_stats = _build_candidate_conditioned_pair_stats(
        reachability_module=reachability_module,
        positive_records=positive_records,
        safe_records=safe_records,
        pos_arrivals=pos_arrivals,
        pos_finite=pos_finite,
        pos_times=pos_times,
        safe_arrivals_history=safe_arrivals_history,
        safe_finite_history=safe_finite_history,
        safe_times=torch.tensor(
            [float(record["time_min"]) for record in safe_records],
            dtype=torch.float,
            device=device,
        ) if safe_records else torch.zeros((0,), dtype=torch.float, device=device),
        num_nodes=num_nodes,
        device=device,
    )

    def build_mode_fields(mode_name: str) -> Dict[str, Any]:
        mode_args = _resolve_admissibility_mode_args(
            admissibility_mode=mode_name,
            frontier_safe_close_tau_min=frontier_safe_close_tau_min,
            compare_config=compare_cfg,
        )
        view = _combine_candidate_conditioned_pair_views(
            pair_stats=pair_stats,
            use_history_safe_view=bool(mode_args["use_history_safe_view"]),
        )
        return _evaluate_candidate_conditioned_admissibility_mode(
            positive_records=positive_records,
            safe_records=safe_records,
            comparable_pair=view["comparable_pair"],
            frontier_margin=view["frontier_margin"],
            arrival_gap=view["arrival_gap"],
            num_nodes=num_nodes,
            device=device,
            safe_violation_tau_min=safe_violation_tau_min,
            witness_mining_mode=witness_mining_mode,
            frontier_close_tau_min=float(mode_args["frontier_close_tau_min"]),
            mined_top_k_safe_witnesses=mined_top_k_safe_witnesses,
            admissibility_mode=mode_name,
            time_bracketing_margin_floor_min=float(mode_args["time_bracketing_margin_floor_min"]),
        )

    active_mode = _validate_admissibility_mode(admissibility_mode)
    result = build_mode_fields(active_mode)
    if diagnostic_compare_modes is not None:
        result["admissibility_compare"] = {
            _validate_admissibility_mode(mode_name): build_mode_fields(mode_name)
            for mode_name in diagnostic_compare_modes
        }
    return result


def _compute_history_conditioned_contradiction_base(
    reachability_module: DynamicReachabilityRuleModule,
    history_steps: Sequence[OracleHistoryStep],
    num_nodes: int,
    safe_violation_tau_min: float = DEFAULT_SAFE_VIOLATION_TAU_MIN,
    suspect_pool: Optional[torch.Tensor] = None,
    phys_ctx_mode: str = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    witness_mining_mode: str = DEFAULT_WITNESS_MINING_MODE,
    frontier_safe_close_tau_min: float = DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN,
    mined_top_k_safe_witnesses: int = DEFAULT_MINED_TOP_K_SAFE_WITNESSES,
) -> Dict[str, Any]:
    if not history_steps:
        raise ValueError("history_steps must contain at least one OracleHistoryStep")

    mode = _validate_phys_ctx_mode(phys_ctx_mode)
    mining_mode = _validate_witness_mining_mode(witness_mining_mode)
    device = history_steps[-1].phys_ctx.edge_index.device
    compressed_history = compress_oracle_history_steps(history_steps)
    positive_records = compressed_history["positive_records"]
    safe_records = compressed_history["safe_records"]

    if not positive_records or not safe_records:
        return _empty_history_contradiction_result(
            num_nodes=num_nodes,
            device=device,
            compressed_history=compressed_history,
            safe_violation_tau_min=safe_violation_tau_min,
            phys_ctx_mode=mode,
            witness_mining_mode=mining_mode,
            frontier_safe_close_tau_min=frontier_safe_close_tau_min,
            mined_top_k_safe_witnesses=mined_top_k_safe_witnesses,
        )

    pos_times = torch.tensor(
        [float(record["time_min"]) for record in positive_records],
        dtype=torch.float,
        device=device,
    )
    safe_times = torch.tensor(
        [float(record["time_min"]) for record in safe_records],
        dtype=torch.float,
        device=device,
    )

    pos_arrivals = _stack_arrival_times(
        reachability_module=reachability_module,
        records=positive_records,
        num_nodes=num_nodes,
        device=device,
        phys_ctx_mode=mode,
        current_phys_ctx=history_steps[-1].phys_ctx,
    )
    safe_arrivals = _stack_arrival_times(
        reachability_module=reachability_module,
        records=safe_records,
        num_nodes=num_nodes,
        device=device,
        phys_ctx_mode=mode,
        current_phys_ctx=history_steps[-1].phys_ctx,
    )

    inf_thresh = float(reachability_module.infinity / 2)
    pos_finite = pos_arrivals < inf_thresh
    safe_finite = safe_arrivals < inf_thresh

    pos_upper = pos_times.unsqueeze(0) - pos_arrivals
    pos_upper = pos_upper.masked_fill(~pos_finite, float("inf"))
    upper_bound = pos_upper.min(dim=1).values

    safe_lower = safe_times.unsqueeze(0) - safe_arrivals
    safe_lower = safe_lower.masked_fill(~safe_finite, float("-inf"))
    lower_bound = safe_lower.max(dim=1).values

    interval_gap = torch.relu(lower_bound - upper_bound)
    interval_gap_term = interval_gap / max(float(safe_violation_tau_min), 1.0)

    pair_valid = safe_finite.unsqueeze(2) & pos_finite.unsqueeze(1)
    obs_margin = safe_times.view(1, -1, 1) - pos_times.view(1, 1, -1)
    arrival_margin = safe_arrivals.unsqueeze(2) - pos_arrivals.unsqueeze(1)
    pair_margin = obs_margin - arrival_margin
    pair_margin = pair_margin.masked_fill(~pair_valid, float("-inf"))

    best_margin_per_safe, best_pos_idx_per_safe = pair_margin.max(dim=2)
    safe_pair_available = pair_valid.any(dim=2)
    best_margin_per_safe = torch.where(
        safe_pair_available,
        best_margin_per_safe,
        torch.full_like(best_margin_per_safe, float("-inf")),
    )

    safe_violation_terms = torch.where(
        safe_pair_available,
        F.softplus(best_margin_per_safe / max(float(safe_violation_tau_min), 1e-6)),
        torch.zeros_like(best_margin_per_safe),
    )
    safe_violation = safe_violation_terms.sum(dim=1)
    positive_reachable_count = pos_finite.float().sum(dim=1)
    safe_reachable_count = safe_finite.float().sum(dim=1)
    pair_count = pair_valid.float().sum(dim=(1, 2))
    positive_margin_pair_count = (pair_valid & (pair_margin > 0.0)).float().sum(dim=(1, 2))
    non_positive_margin_pair_count = torch.clamp(pair_count - positive_margin_pair_count, min=0.0)
    pair_available = (pair_count > 0.0).float()
    positive_margin_available = (positive_margin_pair_count > 0.0).float()
    upper_bound_finite = torch.isfinite(upper_bound).float()
    lower_bound_finite = torch.isfinite(lower_bound).float()
    interval_bounds_available = upper_bound_finite * lower_bound_finite
    interval_regime_available = (interval_gap > 0.0).float()
    violated_safe_count = (best_margin_per_safe > 0.0).float().sum(dim=1)
    eligible_safe_witness_count = safe_pair_available.float().sum(dim=1)
    positive_margin_count = violated_safe_count.clone()
    best_margin_topk_mean = _masked_topk_mean(
        best_margin_per_safe,
        safe_pair_available,
        k=DEFAULT_TOP_K_MARGIN_SUMMARY,
    )

    has_any_witness = safe_pair_available.any(dim=1)
    neg_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    top_witness_margin, top_safe_pos = best_margin_per_safe.max(dim=1)
    top_witness_margin = torch.where(has_any_witness, top_witness_margin, torch.zeros_like(top_witness_margin))
    top_safe_pos = torch.where(has_any_witness, top_safe_pos, neg_idx)
    gather_pos = top_safe_pos.clamp_min(0).unsqueeze(1)
    top_pos_pos = best_pos_idx_per_safe.gather(1, gather_pos).squeeze(1)
    top_pos_pos = torch.where(has_any_witness, top_pos_pos, neg_idx)

    safe_local_idx_lookup = torch.tensor(
        [int(record["local_idx"]) for record in safe_records],
        dtype=torch.long,
        device=device,
    )
    pos_local_idx_lookup = torch.tensor(
        [int(record["local_idx"]) for record in positive_records],
        dtype=torch.long,
        device=device,
    )
    top_witness_safe_local_idx = torch.where(
        top_safe_pos >= 0,
        safe_local_idx_lookup[top_safe_pos.clamp_min(0)],
        neg_idx,
    )
    top_witness_pos_local_idx = torch.where(
        top_pos_pos >= 0,
        pos_local_idx_lookup[top_pos_pos.clamp_min(0)],
        neg_idx,
    )

    result = {
        "mode": "history_base",
        "interval_gap": interval_gap,
        "interval_gap_term": interval_gap_term,
        "safe_violation": safe_violation,
        "violated_safe_count": violated_safe_count,
        "positive_reachable_count": positive_reachable_count,
        "safe_reachable_count": safe_reachable_count,
        "pair_count": pair_count,
        "positive_margin_pair_count": positive_margin_pair_count,
        "non_positive_margin_pair_count": non_positive_margin_pair_count,
        "pair_available": pair_available,
        "positive_margin_available": positive_margin_available,
        "upper_bound_finite": upper_bound_finite,
        "lower_bound_finite": lower_bound_finite,
        "interval_bounds_available": interval_bounds_available,
        "interval_regime_available": interval_regime_available,
        "eligible_safe_witness_count": eligible_safe_witness_count,
        "positive_margin_count": positive_margin_count,
        "best_margin_topk_mean": best_margin_topk_mean,
        "top_witness_margin": top_witness_margin,
        "top_witness_safe_local_idx": top_witness_safe_local_idx,
        "top_witness_pos_local_idx": top_witness_pos_local_idx,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "best_margin_per_safe": best_margin_per_safe,
        "best_pos_idx_per_safe": best_pos_idx_per_safe,
        "safe_pair_available": safe_pair_available,
        "compressed_history": compressed_history,
        "positive_count": len(positive_records),
        "safe_count": len(safe_records),
        "safe_violation_tau_min": float(safe_violation_tau_min),
        "phys_ctx_mode": mode,
        "witness_mining_mode": mining_mode,
        "frontier_safe_close_tau_min": float(frontier_safe_close_tau_min),
        "mined_top_k_safe_witnesses": int(mined_top_k_safe_witnesses),
        "mined_safe_candidate_count": safe_reachable_count,
        "hydraulic_comparable_safe_count": safe_reachable_count,
        "front_close_safe_count": torch.zeros_like(safe_reachable_count),
        "frontier_safe_count": torch.zeros_like(safe_reachable_count),
        "selected_safe_witness_count": eligible_safe_witness_count,
        "topk_safe_count": eligible_safe_witness_count,
        "pair_available_after_mining": pair_available,
        "positive_margin_available_after_mining": positive_margin_available,
        "time_bracketing_safe_count": violated_safe_count.clone(),
        "frontier_window_safe_count": torch.zeros_like(safe_reachable_count),
        "has_safe_observation_anywhere": torch.full(
            (num_nodes,),
            1.0 if len(safe_records) > 0 else 0.0,
            device=device,
        ),
        "has_hydraulically_comparable_safe": (safe_reachable_count > 0.0).float(),
        "has_time_bracketing_safe": (violated_safe_count > 0.0).float(),
        "has_frontier_window_safe": torch.zeros_like(safe_reachable_count),
        "pair_available_under_mode": pair_available,
        "positive_margin_available_under_mode": positive_margin_available,
        "admissibility_mode": DEFAULT_ADMISSIBILITY_MODE,
        "_diag_pos_arrivals": pos_arrivals,
        "_diag_safe_arrivals": safe_arrivals,
        "_diag_pos_finite": pos_finite,
        "_diag_safe_finite": safe_finite,
        "_diag_pos_times": pos_times,
        "_diag_safe_times": safe_times,
    }
    if mining_mode != DEFAULT_WITNESS_MINING_MODE:
        mining_fields = _mine_candidate_conditioned_safe_witnesses(
            reachability_module=reachability_module,
            positive_records=positive_records,
            safe_records=safe_records,
            pos_arrivals=pos_arrivals,
            pos_finite=pos_finite,
            pos_times=pos_times,
            num_nodes=num_nodes,
            device=device,
            safe_violation_tau_min=safe_violation_tau_min,
            witness_mining_mode=mining_mode,
            frontier_safe_close_tau_min=frontier_safe_close_tau_min,
            mined_top_k_safe_witnesses=mined_top_k_safe_witnesses,
            safe_arrivals_history=safe_arrivals,
            safe_finite_history=safe_finite,
        )
        result.update(mining_fields)
    return _apply_candidate_mask(
        result,
        suspect_pool,
        keys=[
            "interval_gap",
            "interval_gap_term",
            "safe_violation",
            "violated_safe_count",
            "positive_reachable_count",
            "safe_reachable_count",
            "pair_count",
            "positive_margin_pair_count",
            "non_positive_margin_pair_count",
            "pair_available",
            "positive_margin_available",
            "upper_bound_finite",
            "lower_bound_finite",
            "interval_bounds_available",
            "interval_regime_available",
            "eligible_safe_witness_count",
            "positive_margin_count",
            "best_margin_topk_mean",
            "top_witness_margin",
            "mined_safe_candidate_count",
            "hydraulic_comparable_safe_count",
            "front_close_safe_count",
            "frontier_safe_count",
            "selected_safe_witness_count",
            "topk_safe_count",
            "pair_available_after_mining",
            "positive_margin_available_after_mining",
            "time_bracketing_safe_count",
            "frontier_window_safe_count",
            "has_safe_observation_anywhere",
            "has_hydraulically_comparable_safe",
            "has_time_bracketing_safe",
            "has_frontier_window_safe",
            "pair_available_under_mode",
            "positive_margin_available_under_mode",
        ],
    )


def compute_oracle_v1_contradiction(
    reachability_module: DynamicReachabilityRuleModule,
    history_steps: Sequence[OracleHistoryStep],
    num_nodes: int,
    safe_violation_tau_min: float = DEFAULT_SAFE_VIOLATION_TAU_MIN,
    suspect_pool: Optional[torch.Tensor] = None,
    phys_ctx_mode: str = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
) -> Dict[str, Any]:
    base_result = _compute_history_conditioned_contradiction_base(
        reachability_module=reachability_module,
        history_steps=history_steps,
        num_nodes=num_nodes,
        safe_violation_tau_min=safe_violation_tau_min,
        suspect_pool=suspect_pool,
        phys_ctx_mode=phys_ctx_mode,
    )
    total = base_result["interval_gap_term"] + base_result["safe_violation"]
    arrival_gate = ((base_result["interval_gap"] > 0.0) | (base_result["violated_safe_count"] > 0.0)).float()
    result = dict(base_result)
    result.update(
        {
            "mode": "oracle_v1",
            "total": total,
            "soft": base_result["safe_violation"],
            "hard": base_result["interval_gap_term"],
            "arrival_gate": arrival_gate,
        }
    )
    return result


def derive_practical_v2_contradiction(
    base_result: Dict[str, Any],
    config: Optional[PracticalContradictionV2Config] = None,
) -> Dict[str, Any]:
    cfg = config or PracticalContradictionV2Config()
    safe_pair_available = base_result["safe_pair_available"]
    best_margin_per_safe = base_result["best_margin_per_safe"]
    tau_count = max(float(cfg.soft_count_tau_min), 1e-6)
    tau_safe = max(float(cfg.near_safe_tau_min), 1e-6)
    gap_log_tau = max(float(cfg.gap_log_tau_min), 1e-6)
    gap_cap = max(float(cfg.gap_cap_min), 1e-6)

    soft_violated_safe_terms = torch.where(
        safe_pair_available,
        torch.sigmoid(best_margin_per_safe / tau_count),
        torch.zeros_like(best_margin_per_safe),
    )
    soft_violated_safe_count = soft_violated_safe_terms.sum(dim=1)

    near_safe_terms = torch.where(
        safe_pair_available,
        F.softplus((best_margin_per_safe + float(cfg.near_safe_slack_min)) / tau_safe),
        torch.zeros_like(best_margin_per_safe),
    )
    near_safe_mass = near_safe_terms.sum(dim=1)

    interval_gap = base_result["interval_gap"]
    interval_gap_capped = torch.clamp(interval_gap, min=0.0, max=gap_cap)
    interval_gap_log = torch.log1p(interval_gap / gap_log_tau)
    gap_component = float(cfg.alpha_gap) * interval_gap_log
    safe_component = (
        float(cfg.beta_near_safe) * near_safe_mass
        + float(cfg.gamma_soft_count) * soft_violated_safe_count
    )
    total_raw = gap_component + safe_component
    denom = torch.clamp(base_result["eligible_safe_witness_count"], min=1.0)
    total_norm = total_raw / denom
    total = total_norm if cfg.normalize_by_eligible_safe_count else total_raw

    dominant_component_flag = torch.zeros_like(interval_gap, dtype=torch.long)
    positive_total = total > 0.0
    dominant_component_flag = torch.where(
        positive_total & (gap_component > safe_component),
        torch.ones_like(dominant_component_flag),
        dominant_component_flag,
    )
    dominant_component_flag = torch.where(
        positive_total & (safe_component >= gap_component),
        torch.full_like(dominant_component_flag, 2),
        dominant_component_flag,
    )
    arrival_gate = (
        (interval_gap > 0.0)
        | (soft_violated_safe_count > 0.0)
        | (near_safe_mass > 0.0)
    ).float()
    safe_regime_available = ((soft_violated_safe_count > 0.0) | (near_safe_mass > 0.0)).float()

    result = dict(base_result)
    result.update(
        {
            "mode": "practical_v2",
            "total": total,
            "soft": safe_component,
            "hard": gap_component,
            "arrival_gate": arrival_gate,
            "soft_violated_safe_count": soft_violated_safe_count,
            "near_safe_mass": near_safe_mass,
            "interval_gap_capped": interval_gap_capped,
            "interval_gap_log": interval_gap_log,
            "gap_component": gap_component,
            "safe_component": safe_component,
            "safe_regime_available": safe_regime_available,
            "total_practical_v2_raw": total_raw,
            "total_practical_v2_norm": total_norm,
            "dominant_component_flag": dominant_component_flag,
            "practical_v2_config": cfg.to_dict(),
        }
    )
    return result


def compute_practical_v2_contradiction(
    reachability_module: DynamicReachabilityRuleModule,
    history_steps: Sequence[OracleHistoryStep],
    num_nodes: int,
    safe_violation_tau_min: float = DEFAULT_SAFE_VIOLATION_TAU_MIN,
    suspect_pool: Optional[torch.Tensor] = None,
    phys_ctx_mode: str = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    config: Optional[PracticalContradictionV2Config] = None,
    witness_mining_mode: str = DEFAULT_WITNESS_MINING_MODE,
    frontier_safe_close_tau_min: float = DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN,
    mined_top_k_safe_witnesses: int = DEFAULT_MINED_TOP_K_SAFE_WITNESSES,
    admissibility_mode: str = DEFAULT_ADMISSIBILITY_MODE,
    admissibility_compare_config: Optional[AdmissibilityCompareConfig] = None,
) -> Dict[str, Any]:
    base_result = _compute_history_conditioned_contradiction_base(
        reachability_module=reachability_module,
        history_steps=history_steps,
        num_nodes=num_nodes,
        safe_violation_tau_min=safe_violation_tau_min,
        suspect_pool=suspect_pool,
        phys_ctx_mode=phys_ctx_mode,
        witness_mining_mode=witness_mining_mode,
        frontier_safe_close_tau_min=frontier_safe_close_tau_min,
        mined_top_k_safe_witnesses=mined_top_k_safe_witnesses,
    )
    if witness_mining_mode != DEFAULT_WITNESS_MINING_MODE:
        mode = _validate_admissibility_mode(admissibility_mode)
        compressed_history = base_result["compressed_history"]
        if (
            mode != DEFAULT_ADMISSIBILITY_MODE
            and compressed_history["positive_records"]
            and compressed_history["safe_records"]
        ):
            mining_fields = _mine_candidate_conditioned_safe_witnesses(
                reachability_module=reachability_module,
                positive_records=compressed_history["positive_records"],
                safe_records=compressed_history["safe_records"],
                pos_arrivals=base_result["_diag_pos_arrivals"],
                pos_finite=base_result["_diag_pos_finite"],
                pos_times=base_result["_diag_pos_times"],
                num_nodes=num_nodes,
                device=base_result["interval_gap"].device,
                safe_violation_tau_min=safe_violation_tau_min,
                witness_mining_mode=witness_mining_mode,
                frontier_safe_close_tau_min=frontier_safe_close_tau_min,
                mined_top_k_safe_witnesses=mined_top_k_safe_witnesses,
                safe_arrivals_history=base_result["_diag_safe_arrivals"],
                safe_finite_history=base_result["_diag_safe_finite"],
                admissibility_mode=mode,
                compare_config=admissibility_compare_config,
            )
            base_result.update(mining_fields)
    return derive_practical_v2_contradiction(base_result, config=config)


def compute_practical_v2_admissibility_compare(
    reachability_module: DynamicReachabilityRuleModule,
    history_steps: Sequence[OracleHistoryStep],
    num_nodes: int,
    safe_violation_tau_min: float = DEFAULT_SAFE_VIOLATION_TAU_MIN,
    suspect_pool: Optional[torch.Tensor] = None,
    phys_ctx_mode: str = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    config: Optional[PracticalContradictionV2Config] = None,
    frontier_safe_close_tau_min: float = DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN,
    compare_config: Optional[AdmissibilityCompareConfig] = None,
    admissibility_modes: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    modes = tuple(
        _validate_admissibility_mode(mode_name)
        for mode_name in (admissibility_modes or DEFAULT_ADMISSIBILITY_COMPARE_MODES)
    )
    compare_cfg = compare_config or AdmissibilityCompareConfig()
    base_result = _compute_history_conditioned_contradiction_base(
        reachability_module=reachability_module,
        history_steps=history_steps,
        num_nodes=num_nodes,
        safe_violation_tau_min=safe_violation_tau_min,
        suspect_pool=suspect_pool,
        phys_ctx_mode=phys_ctx_mode,
        witness_mining_mode=DEFAULT_WITNESS_MINING_MODE,
        frontier_safe_close_tau_min=frontier_safe_close_tau_min,
        mined_top_k_safe_witnesses=1,
    )
    compressed_history = base_result["compressed_history"]
    if not compressed_history["positive_records"] or not compressed_history["safe_records"]:
        empty_compare: Dict[str, Dict[str, Any]] = {}
        for mode_name in modes:
            mode_result = dict(base_result)
            mode_result.update(
                _evaluate_candidate_conditioned_admissibility_mode(
                    positive_records=compressed_history["positive_records"],
                    safe_records=compressed_history["safe_records"],
                    comparable_pair=torch.zeros((num_nodes, 0, 0), dtype=torch.bool, device=base_result["interval_gap"].device),
                    frontier_margin=torch.zeros((num_nodes, 0, 0), device=base_result["interval_gap"].device),
                    arrival_gap=torch.zeros((num_nodes, 0, 0), device=base_result["interval_gap"].device),
                    num_nodes=num_nodes,
                    device=base_result["interval_gap"].device,
                    safe_violation_tau_min=safe_violation_tau_min,
                    witness_mining_mode="candidate_conditioned_frontier_safe",
                    frontier_close_tau_min=frontier_safe_close_tau_min,
                    mined_top_k_safe_witnesses=1,
                    admissibility_mode=mode_name,
                    time_bracketing_margin_floor_min=0.0,
                )
            )
            mode_result["admissibility_compare_config"] = compare_cfg.to_dict()
            empty_compare[mode_name] = derive_practical_v2_contradiction(mode_result, config=config)
        return empty_compare

    mining_fields = _mine_candidate_conditioned_safe_witnesses(
        reachability_module=reachability_module,
        positive_records=compressed_history["positive_records"],
        safe_records=compressed_history["safe_records"],
        pos_arrivals=base_result["_diag_pos_arrivals"],
        pos_finite=base_result["_diag_pos_finite"],
        pos_times=base_result["_diag_pos_times"],
        num_nodes=num_nodes,
        device=base_result["interval_gap"].device,
        safe_violation_tau_min=safe_violation_tau_min,
        witness_mining_mode="candidate_conditioned_frontier_safe",
        frontier_safe_close_tau_min=frontier_safe_close_tau_min,
        mined_top_k_safe_witnesses=1,
        safe_arrivals_history=base_result["_diag_safe_arrivals"],
        safe_finite_history=base_result["_diag_safe_finite"],
        admissibility_mode=DEFAULT_ADMISSIBILITY_MODE,
        compare_config=compare_cfg,
        diagnostic_compare_modes=modes,
    )
    compare_results: Dict[str, Dict[str, Any]] = {}
    for mode_name, mode_fields in mining_fields.get("admissibility_compare", {}).items():
        mode_result = dict(base_result)
        mode_result.update(mode_fields)
        mode_result["admissibility_compare_config"] = compare_cfg.to_dict()
        compare_results[mode_name] = derive_practical_v2_contradiction(mode_result, config=config)
    return compare_results


def extract_candidate_top_witnesses(
    oracle_v1_result: Dict[str, Any],
    candidate_idx: int,
    top_k: int = DEFAULT_TOP_K_WITNESSES,
) -> List[Dict[str, Any]]:
    compressed_history = oracle_v1_result["compressed_history"]
    positive_records = compressed_history["positive_records"]
    safe_records = compressed_history["safe_records"]
    if not positive_records or not safe_records:
        return []

    candidate_idx = int(candidate_idx)
    best_margin = oracle_v1_result["best_margin_per_safe"][candidate_idx]
    best_pos_idx = oracle_v1_result["best_pos_idx_per_safe"][candidate_idx]
    safe_pair_available = oracle_v1_result["safe_pair_available"][candidate_idx]
    if not bool(safe_pair_available.any().item()):
        return []

    order = torch.argsort(best_margin, descending=True)
    witnesses: List[Dict[str, Any]] = []
    for safe_pos in order.tolist():
        if len(witnesses) >= int(top_k):
            break
        if not bool(safe_pair_available[safe_pos].item()):
            continue
        pos_pos = int(best_pos_idx[safe_pos].item())
        safe_record = safe_records[safe_pos]
        pos_record = positive_records[pos_pos]
        witnesses.append(
            {
                "safe_local_idx": int(safe_record["local_idx"]),
                "safe_global_idx": int(safe_record["global_idx"]),
                "safe_time_min": float(safe_record["time_min"]),
                "safe_episode": int(safe_record["episode"]),
                "positive_local_idx": int(pos_record["local_idx"]),
                "positive_global_idx": int(pos_record["global_idx"]),
                "positive_time_min": float(pos_record["time_min"]),
                "positive_episode": int(pos_record["episode"]),
                "margin": float(best_margin[safe_pos].item()),
            }
        )
    return witnesses
