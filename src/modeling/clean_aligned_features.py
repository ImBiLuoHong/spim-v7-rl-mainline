from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from src.modeling.navigators.clean_v1 import (
    bound_nonnegative_score,
    derive_two_channel_features,
)


NODE_FEATURE_DIM = 21
GRAPH_FEATURE_DIM = 6


def state_attr(
    state_obj: Any,
    key: str,
    reference: torch.Tensor,
    default: float = 0.0,
) -> torch.Tensor:
    if state_obj is None:
        return torch.full_like(reference.view(-1).float(), float(default))
    if isinstance(state_obj, dict):
        value = state_obj.get(key)
    else:
        value = getattr(state_obj, key, None)
    if value is None:
        return torch.full_like(reference.view(-1).float(), float(default))
    return value.view(-1).to(device=reference.device, dtype=torch.float32)


def build_clean_aligned_feature_payload(
    state: Dict[str, Any],
    *,
    batch_index: torch.Tensor,
    edge_index: torch.Tensor,
    physics_ctx: Optional[Dict[str, Any]],
    frontier_mode: str = "unresolved_without_pair",
) -> Dict[str, torch.Tensor]:
    reference = state["valid_mask"].view(-1).float()
    graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    evidence_state = state.get("evidence_state")
    obs_state = state.get("observation_state")
    constraint_state = state.get("constraint_state")

    support_score = state_attr(evidence_state, "support_score", reference, default=0.0)
    contradiction_score = state_attr(evidence_state, "contradiction_score", reference, default=0.0)
    derived = derive_two_channel_features(support_score, contradiction_score)

    support_focus = state_attr(evidence_state, "support_focus_term", reference, default=0.0)
    support_timing = state_attr(evidence_state, "support_timing_term", reference, default=0.0)
    support_coverage = state_attr(evidence_state, "support_coverage_term", reference, default=0.0)
    contradiction_soft = state_attr(evidence_state, "contradiction_toxic_term", reference, default=0.0)
    contradiction_hard = state_attr(evidence_state, "contradiction_clean_term", reference, default=0.0)
    arrival_gate = state_attr(evidence_state, "arrival_gate", reference, default=0.0).clamp(0.0, 1.0)

    if physics_ctx is not None and isinstance(physics_ctx.get("feasible_mask"), torch.Tensor):
        feasible_mask = physics_ctx["feasible_mask"].view(-1).float()
    else:
        feasible_mask = torch.ones_like(reference)

    if constraint_state is None:
        sampled_mask = torch.zeros_like(reference)
        no_resample_mask = torch.zeros_like(reference)
    else:
        sampled_mask = constraint_state.sampled_mask.view(-1).float()
        no_resample_mask = constraint_state.no_resample_mask.view(-1).float()

    positive_anchor_potential = bound_nonnegative_score(support_focus + support_timing)
    safe_pair_potential = bound_nonnegative_score(contradiction_score + contradiction_hard)
    positive_reachability = bound_nonnegative_score(support_coverage)
    safe_reachability = arrival_gate
    positive_distance_summary = bound_nonnegative_score(support_timing)
    safe_distance_summary = bound_nonnegative_score(contradiction_soft + contradiction_hard)
    pair_available = arrival_gate
    eligible_safe_witness_count_bounded = bound_nonnegative_score(contradiction_soft)
    top_pair_margin_bounded = bound_nonnegative_score(contradiction_hard)

    if edge_index.numel() == 0:
        degree_norm = torch.zeros_like(reference)
    else:
        src, dst = edge_index
        degree = (
            torch.bincount(src, minlength=int(reference.numel())).float()
            + torch.bincount(dst, minlength=int(reference.numel())).float()
        )
        degree_max = scatter_max(degree, batch_index, dim=0, dim_size=graph_count)[0]
        degree_den = degree_max[batch_index].clamp_min(1.0)
        degree_norm = degree / degree_den

    node_features = torch.stack(
        [
            support_score,
            contradiction_score,
            derived["support_bounded"],
            derived["contradiction_bounded"],
            derived["live_plausibility"],
            derived["conflict_mass"],
            derived["ignorance_mass"],
            positive_anchor_potential,
            safe_pair_potential,
            positive_reachability,
            safe_reachability,
            positive_distance_summary,
            safe_distance_summary,
            pair_available,
            eligible_safe_witness_count_bounded,
            top_pair_margin_bounded,
            feasible_mask.float(),
            sampled_mask.float(),
            no_resample_mask.float(),
            degree_norm.float(),
            reference.float(),
        ],
        dim=1,
    ).float()
    node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    if frontier_mode == "conflict_mass":
        frontier_role_potential = derived["conflict_mass"]
    else:
        frontier_role_potential = derived["unresolved_mass"] * (1.0 - pair_available)
    role_potentials = torch.stack(
        [
            positive_anchor_potential,
            frontier_role_potential,
            safe_pair_potential,
        ],
        dim=1,
    ).float()
    role_potentials = torch.nan_to_num(role_potentials, nan=0.0, posinf=0.0, neginf=0.0)

    if obs_state is not None:
        positive_count_by_graph = scatter_sum(
            obs_state.toxic_positive_flag.view(-1).float(),
            batch_index,
            dim=0,
            dim_size=graph_count,
        )
        safe_count_by_graph = scatter_sum(
            obs_state.toxic_negative_flag.view(-1).float(),
            batch_index,
            dim=0,
            dim_size=graph_count,
        )
    else:
        positive_count_by_graph = torch.zeros((graph_count,), device=batch_index.device, dtype=torch.float32)
        safe_count_by_graph = torch.zeros((graph_count,), device=batch_index.device, dtype=torch.float32)

    summary = state.get("nav_state_summary")
    if summary is not None:
        summary = summary.float()
        budget_norm = summary[:, 4] if summary.size(1) > 4 else torch.zeros((graph_count,), device=batch_index.device)
        step_norm = summary[:, 5] if summary.size(1) > 5 else torch.zeros((graph_count,), device=batch_index.device)
    else:
        budget_norm = torch.zeros((graph_count,), device=batch_index.device)
        step_norm = torch.zeros((graph_count,), device=batch_index.device)
    candidate_fraction_by_graph = scatter_mean(reference, batch_index, dim=0, dim_size=graph_count)
    graph_sizes = torch.bincount(batch_index, minlength=graph_count).float().clamp_min(1.0)
    graph_features_by_graph = torch.stack(
        [
            step_norm,
            budget_norm,
            positive_count_by_graph / graph_sizes,
            safe_count_by_graph / graph_sizes,
            candidate_fraction_by_graph,
            step_norm,
        ],
        dim=1,
    ).float()
    graph_features_by_graph = torch.nan_to_num(graph_features_by_graph, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "node_features": node_features,
        "role_potentials": role_potentials,
        "valid_mask": reference.bool(),
        "graph_features_by_graph": graph_features_by_graph,
        "positive_count_by_graph": positive_count_by_graph.float(),
        "safe_count_by_graph": safe_count_by_graph.float(),
    }
