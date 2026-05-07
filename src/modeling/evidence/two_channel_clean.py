from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.builder import EvidenceBuilder
from src.modeling.evidence.contradiction_oracle_v1 import OracleHistoryStep
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.state.schema import ConstraintState, EvidenceStateMini, ObservationState, PhysicsContext
import src.scripts.audit.run_support_score_v21_oracle_sweep as support_v21
import src.scripts.audit.run_support_score_v21_practical_audit as support_v21_practical


@dataclass
class WitnessRecord:
    node_local_idx: int
    node_global_idx: int
    absolute_time_min: float
    episode_id: int
    absolute_snapshot_idx: int
    label: str
    confidence: float
    t_snapshot_idx: int
    phys_ctx: PhysicsContext


@dataclass(frozen=True)
class SupportScoreContext:
    """
    Extra context required to ground support_body_v2.1 on the real topology assets.
    """

    global_node_ids: torch.Tensor
    absolute_snapshot_idx: int
    topology: HydraulicTopology


class ObservationWitnessHistory:
    """
    Layer A: observation/witness memory accumulated across episodes.
    """

    def __init__(self) -> None:
        self.records: List[WitnessRecord] = []

    def append_from_history_step(self, step: OracleHistoryStep) -> None:
        for sample in step.samples:
            if sample.is_positive:
                label = "positive"
            elif sample.is_safe:
                label = "safe"
            else:
                continue
            conf = max(float(sample.concentration), 0.0)
            conf = conf / (1.0 + conf)
            self.records.append(
                WitnessRecord(
                    node_local_idx=int(sample.local_idx),
                    node_global_idx=int(sample.global_idx),
                    absolute_time_min=float(sample.time_min),
                    episode_id=int(step.episode),
                    absolute_snapshot_idx=int(
                        step.absolute_snapshot_idx
                        if step.absolute_snapshot_idx is not None
                        else step.t_snapshot_idx
                    ),
                    label=label,
                    confidence=float(conf),
                    t_snapshot_idx=int(sample.t_snapshot_idx),
                    phys_ctx=step.phys_ctx,
                )
            )

    def positive_records(self) -> List[WitnessRecord]:
        return [row for row in self.records if row.label == "positive"]

    def safe_records(self) -> List[WitnessRecord]:
        return [row for row in self.records if row.label == "safe"]


class PhysicsEvidenceGate:
    """
    Layer B: physics/evidence-validity gating and TT lookup.
    """

    def __init__(self, reachability: Optional[DynamicReachabilityRuleModule] = None) -> None:
        self.reachability = reachability or DynamicReachabilityRuleModule()

    def _resolve_distance_weights(self, phys_ctx: PhysicsContext) -> torch.Tensor:
        if phys_ctx.stt_dynamic is not None:
            return torch.abs(phys_ctx.stt_dynamic.view(-1))
        if phys_ctx.stt_median is not None:
            return torch.expm1(phys_ctx.stt_median)
        if phys_ctx.edge_attr is not None and phys_ctx.edge_attr.size(1) > 0:
            return torch.expm1(phys_ctx.edge_attr[:, 0])
        return torch.ones_like(phys_ctx.edge_index[0], dtype=torch.float) * 20.0

    def stack_arrival_times(
        self,
        records: Sequence[WitnessRecord],
        num_nodes: int,
        device: torch.device,
        phys_ctx_mode: str = "witness_time_physctx",
        current_phys_ctx: Optional[PhysicsContext] = None,
    ) -> torch.Tensor:
        if len(records) == 0:
            return torch.zeros((num_nodes, 0), device=device)
        mode = str(phys_ctx_mode)
        grouped_records: Dict[int, List[tuple[int, WitnessRecord]]] = {}
        for record_idx, record in enumerate(records):
            phys_ctx = current_phys_ctx if mode == "current_time_physctx" else record.phys_ctx
            if phys_ctx is None:
                raise ValueError("current_phys_ctx is required when phys_ctx_mode='current_time_physctx'")
            grouped_records.setdefault(id(phys_ctx), []).append((record_idx, record))

        arrivals: List[Optional[torch.Tensor]] = [None] * len(records)
        for record_group in grouped_records.values():
            phys_ctx = current_phys_ctx if mode == "current_time_physctx" else record_group[0][1].phys_ctx
            seed_indices = torch.tensor(
                [int(record.node_local_idx) for _, record in record_group],
                device=device,
                dtype=torch.long,
            )
            dist_matrix = self.reachability.compute_distance_matrix(
                seed_indices=seed_indices,
                physics_context=phys_ctx,
                num_nodes=num_nodes,
            )
            for local_col, (record_idx, _) in enumerate(record_group):
                arrivals[record_idx] = dist_matrix[:, local_col]
        return torch.stack([col for col in arrivals if col is not None], dim=1)


class RuntimeConstraintLayer:
    """
    Layer C: runtime hard constraints (not evidence semantics).
    """

    def build_valid_mask(
        self,
        num_nodes: int,
        device: torch.device,
        constraint_state: Optional[ConstraintState] = None,
    ) -> torch.Tensor:
        if constraint_state is None:
            return torch.ones(num_nodes, device=device, dtype=torch.float32)
        valid = torch.ones(num_nodes, device=device, dtype=torch.float32)
        if constraint_state.confirmed_non_source_mask is not None:
            valid = valid * (1.0 - constraint_state.confirmed_non_source_mask.view(-1).float())
        if constraint_state.no_resample_mask is not None:
            valid = valid * (1.0 - constraint_state.no_resample_mask.view(-1).float())
        return valid.clamp(0.0, 1.0)


class CleanTwoChannelEvidenceEnv:
    """
    Layer D: minimal evidence state with support_score and contradiction_score only.
    """

    def __init__(
        self,
        evidence_builder: Optional[EvidenceBuilder] = None,
        physics_gate: Optional[PhysicsEvidenceGate] = None,
        runtime_layer: Optional[RuntimeConstraintLayer] = None,
        contradiction_aggregation_mode: str = "pair_topk_mean",
        contradiction_top_k_pairs: int = 8,
        frontier_top_m: int = 3,
        contradiction_time_decay_min: float = 135.0,
        tau_anchor_min: float = 30.0,
        tau_safe_gate_min: float = 30.0,
        tau_margin_min: float = 15.0,
        safe_gate_slack_min: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.physics_gate = physics_gate or PhysicsEvidenceGate(self.evidence_builder.dynamic_reachability)
        self.runtime_layer = runtime_layer or RuntimeConstraintLayer()
        self.contradiction_aggregation_mode = str(contradiction_aggregation_mode)
        if self.contradiction_aggregation_mode not in {
            "pair_topk_mean",
            "safe_topk_best_positive",
            "interval_recency_bracket",
            "frontier_adjacent_interval",
            "multi_frontier_interval_state",
        }:
            raise ValueError(
                f"Unsupported contradiction_aggregation_mode: {self.contradiction_aggregation_mode}"
            )
        self.contradiction_top_k_pairs = int(max(1, contradiction_top_k_pairs))
        self.frontier_top_m = int(max(1, frontier_top_m))
        self.contradiction_time_decay_min = float(max(contradiction_time_decay_min, 1e-6))
        self.tau_anchor_min = float(max(tau_anchor_min, 1e-6))
        self.tau_safe_gate_min = float(max(tau_safe_gate_min, 1e-6))
        self.tau_margin_min = float(max(tau_margin_min, 1e-6))
        self.safe_gate_slack_min = float(safe_gate_slack_min)
        self.eps = float(eps)

    def _prepare_support_history_payload(
        self,
        history: ObservationWitnessHistory,
        physics_context: PhysicsContext,
        support_context: SupportScoreContext,
    ) -> Dict[str, Any]:
        positives = history.positive_records()
        device = physics_context.edge_index.device
        num_nodes = int(physics_context.batch.numel()) if physics_context.batch is not None else int(
            support_context.global_node_ids.numel()
        )
        if len(positives) == 0:
            return {
                "num_nodes": num_nodes,
                "device": device,
                "witness_count": 0,
                "physical_time_matrix": None,
                "virtual_time_matrix": None,
            }

        subgraph_global_ids = support_context.global_node_ids.detach().cpu()
        physical_time_cols: List[Optional[torch.Tensor]] = [None] * len(positives)
        virtual_time_cols: List[torch.Tensor] = []
        witness_time_mins: List[float] = []
        grouped_records: Dict[int, List[tuple[int, WitnessRecord]]] = {}

        for record_idx, record in enumerate(positives):
            phys_ctx = record.phys_ctx
            num_record_nodes = int(phys_ctx.batch.numel()) if phys_ctx.batch is not None else num_nodes
            if num_record_nodes != num_nodes:
                raise ValueError(
                    f"History witness node-count mismatch: expected {num_nodes}, got {num_record_nodes}"
                )
            grouped_records.setdefault(id(phys_ctx), []).append((record_idx, record))
            virt_dist = support_v21_practical.rerun.build_virtual_time_lookup(
                topology=support_context.topology,
                witness_global_id=int(record.node_global_idx),
                subgraph_global_ids=subgraph_global_ids,
                witness_local_idx=int(record.node_local_idx),
                t_abs_idx=int(record.absolute_snapshot_idx),
                num_nodes=num_nodes,
                device=device,
            )
            virtual_time_cols.append(virt_dist)
            witness_time_mins.append(float(record.absolute_time_min))

        for record_group in grouped_records.values():
            phys_ctx = record_group[0][1].phys_ctx
            seed_indices = torch.tensor(
                [int(record.node_local_idx) for _, record in record_group],
                device=device,
                dtype=torch.long,
            )
            dist_matrix = self.physics_gate.reachability.compute_distance_matrix(
                seed_indices=seed_indices,
                physics_context=phys_ctx,
                num_nodes=num_nodes,
            )
            for local_col, (record_idx, _) in enumerate(record_group):
                physical_time_cols[record_idx] = dist_matrix[:, local_col]

        return {
            "num_nodes": num_nodes,
            "device": device,
            "witness_count": len(positives),
            "physical_time_matrix": torch.stack([col for col in physical_time_cols if col is not None], dim=1),
            "virtual_time_matrix": torch.stack(virtual_time_cols, dim=1),
            "witness_time_mins": torch.tensor(witness_time_mins, dtype=torch.float32, device=device),
        }

    def compute_support_score(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        current_time_min: float,
        support_context: SupportScoreContext,
        history: Optional[ObservationWitnessHistory] = None,
    ) -> Dict[str, torch.Tensor]:
        if history is not None:
            payload = self._prepare_support_history_payload(
                history=history,
                physics_context=physics_context,
                support_context=support_context,
            )
            return support_v21.compute_support_v21_from_payload(
                payload=payload,
                current_time_min=float(current_time_min),
                config=support_v21_practical.BEST_V21_CONFIG,
            )

        rollout_adapter = SimpleNamespace(
            g_ids=support_context.global_node_ids,
            reachability_module=self.physics_gate.reachability,
        )
        witness_strength = torch.abs(observation_state.chlorine_deviation)
        truth_positive_mask = observation_state.toxic_positive_flag > 0.5
        payload = support_v21.prepare_v2_payload(
            rollout=rollout_adapter,
            phys_ctx=physics_context,
            truth_positive_mask=truth_positive_mask,
            witness_strength=witness_strength,
            t_abs_idx=int(support_context.absolute_snapshot_idx),
            topology=support_context.topology,
        )
        support_res = support_v21.compute_support_v21_from_payload(
            payload=payload,
            current_time_min=float(current_time_min),
            config=support_v21_practical.BEST_V21_CONFIG,
        )
        return support_res

    def compute_contradiction_score(
        self,
        history: ObservationWitnessHistory,
        num_nodes: int,
        device: torch.device,
        phys_ctx_mode: str = "witness_time_physctx",
        current_phys_ctx: Optional[PhysicsContext] = None,
        current_time_min: Optional[float] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        positives = history.positive_records()
        safes = history.safe_records()
        zero = torch.zeros(num_nodes, device=device)
        if len(positives) == 0 or len(safes) == 0:
            return {
                "contradiction_score": zero,
                "pair_available": zero,
                "contradiction_nonzero": zero,
                "eligible_safe_witness_count": zero,
                "top_pair_margin": zero,
                "pair_count": zero,
                "positive_margin_pair_count": zero,
                "aggregate_contributor_count": zero,
                "feasible_upper_bound": zero,
                "feasible_lower_bound": zero,
                "violation_margin": zero,
                "upper_bound_strength": zero,
                "lower_bound_strength": zero,
                "frontier_active": zero,
                "frontier_set_size": zero,
                "frontier_selected_positive_idx_tensor": torch.full(
                    (num_nodes, 0),
                    -1,
                    device=device,
                    dtype=torch.long,
                ),
                "frontier_selected_count_tensor": zero.view(num_nodes, 1)[:, :0],
                "frontier_selected_positive_idx_topm_tensor": torch.full(
                    (num_nodes, 0, 0),
                    -1,
                    device=device,
                    dtype=torch.long,
                ),
                "frontier_selected_upper_bound_topm_tensor": zero.view(num_nodes, 1, 1)[:, :0, :0],
                "frontier_selected_violation_topm_tensor": zero.view(num_nodes, 1, 1)[:, :0, :0],
                "frontier_selected_score_topm_tensor": zero.view(num_nodes, 1, 1)[:, :0, :0],
                "frontier_upper_bound_tensor": zero.view(num_nodes, 1)[:, :0],
                "frontier_safe_lower_bound_tensor": zero.view(num_nodes, 1)[:, :0],
                "frontier_violation_tensor": zero.view(num_nodes, 1)[:, :0],
                "frontier_score_tensor": zero.view(num_nodes, 1)[:, :0],
                "frontier_anchor_tensor": zero.view(num_nodes, 1)[:, :0],
                "frontier_safe_gate_tensor": zero.view(num_nodes, 1)[:, :0],
                "pair_valid_tensor": zero.view(num_nodes, 1, 1)[:, :0, :0],
            }

        pos_arrivals = self.physics_gate.stack_arrival_times(
            positives,
            num_nodes=num_nodes,
            device=device,
            phys_ctx_mode=phys_ctx_mode,
            current_phys_ctx=current_phys_ctx,
        )
        safe_arrivals = self.physics_gate.stack_arrival_times(
            safes,
            num_nodes=num_nodes,
            device=device,
            phys_ctx_mode=phys_ctx_mode,
            current_phys_ctx=current_phys_ctx,
        )
        inf_thresh = float(self.physics_gate.reachability.infinity / 2)
        pos_finite = pos_arrivals < inf_thresh
        safe_finite = safe_arrivals < inf_thresh
        t_pos = torch.tensor([float(row.absolute_time_min) for row in positives], device=device, dtype=torch.float32)
        t_safe = torch.tensor([float(row.absolute_time_min) for row in safes], device=device, dtype=torch.float32)
        pos_conf = torch.tensor([float(row.confidence) for row in positives], device=device, dtype=torch.float32)

        anchor_raw = (t_pos.unsqueeze(0) - pos_arrivals) / self.tau_anchor_min
        anchor_strength = torch.sigmoid(anchor_raw) * pos_conf.unsqueeze(0)
        safe_gate_raw = (t_safe.unsqueeze(0) - safe_arrivals + self.safe_gate_slack_min) / self.tau_safe_gate_min
        safe_gate = torch.sigmoid(safe_gate_raw)

        pair_valid = safe_finite.unsqueeze(2) & pos_finite.unsqueeze(1)
        obs_margin = t_safe.view(1, -1, 1) - t_pos.view(1, 1, -1)
        arrival_margin = safe_arrivals.unsqueeze(2) - pos_arrivals.unsqueeze(1)
        delta = obs_margin - arrival_margin
        margin_term = F.softplus(delta / self.tau_margin_min)
        positive_upper_bound = t_pos.unsqueeze(0) - pos_arrivals
        safe_lower_bound = t_safe.unsqueeze(0) - safe_arrivals + self.safe_gate_slack_min

        pair_score = anchor_strength.unsqueeze(1) * safe_gate.unsqueeze(2) * margin_term
        pair_score = torch.where(pair_valid, pair_score, torch.zeros_like(pair_score))

        flat_score = pair_score.view(num_nodes, -1)
        flat_valid = pair_valid.view(num_nodes, -1)
        valid_pair_count = flat_valid.float().sum(dim=1)
        safe_has_pair = pair_valid.any(dim=2)
        finite_positive_count = pos_finite.float().sum(dim=1)
        finite_safe_count = safe_finite.float().sum(dim=1)

        if self.contradiction_aggregation_mode == "pair_topk_mean":
            k = min(int(self.contradiction_top_k_pairs), int(flat_score.size(1)))
            top_vals, _ = torch.topk(
                torch.where(flat_valid, flat_score, torch.full_like(flat_score, float("-inf"))),
                k=k,
                dim=1,
            )
            top_finite = torch.isfinite(top_vals)
            top_vals_safe = torch.where(top_finite, top_vals, torch.zeros_like(top_vals))
            contradiction_score = torch.where(
                valid_pair_count > 0.0,
                top_vals_safe.sum(dim=1) / top_finite.float().sum(dim=1).clamp_min(1.0),
                torch.zeros(num_nodes, device=device),
            )
            aggregate_contributor_count = top_finite.float().sum(dim=1)
            feasible_upper_bound = torch.zeros(num_nodes, device=device)
            feasible_lower_bound = torch.zeros(num_nodes, device=device)
            violation_margin = torch.zeros(num_nodes, device=device)
            upper_bound_strength = torch.zeros(num_nodes, device=device)
            lower_bound_strength = torch.zeros(num_nodes, device=device)
            frontier_active = torch.zeros(num_nodes, device=device)
            frontier_set_size = torch.zeros(num_nodes, device=device)
            frontier_selected_positive_idx_tensor = torch.full(
                (num_nodes, len(safes)),
                -1,
                device=device,
                dtype=torch.long,
            )
            frontier_selected_count_tensor = torch.zeros(num_nodes, len(safes), device=device)
            frontier_selected_positive_idx_topm_tensor = torch.full(
                (num_nodes, len(safes), 1),
                -1,
                device=device,
                dtype=torch.long,
            )
            frontier_selected_upper_bound_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
            frontier_selected_violation_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
            frontier_selected_score_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
            frontier_upper_bound_tensor = torch.zeros(num_nodes, len(safes), device=device)
            frontier_safe_lower_bound_tensor = torch.zeros(num_nodes, len(safes), device=device)
            frontier_violation_tensor = torch.zeros(num_nodes, len(safes), device=device)
            frontier_score_tensor = torch.zeros(num_nodes, len(safes), device=device)
            frontier_anchor_tensor = torch.zeros(num_nodes, len(safes), device=device)
            frontier_safe_gate_tensor = torch.zeros(num_nodes, len(safes), device=device)
        else:
            if self.contradiction_aggregation_mode == "safe_topk_best_positive":
                safe_best_pair = torch.where(
                    pair_valid,
                    pair_score,
                    torch.full_like(pair_score, float("-inf")),
                ).amax(dim=2)
                k = min(int(self.contradiction_top_k_pairs), int(safe_best_pair.size(1)))
                top_safe_vals, _ = torch.topk(
                    torch.where(safe_has_pair, safe_best_pair, torch.full_like(safe_best_pair, float("-inf"))),
                    k=k,
                    dim=1,
                )
                top_safe_finite = torch.isfinite(top_safe_vals)
                top_safe_vals = torch.where(top_safe_finite, top_safe_vals, torch.zeros_like(top_safe_vals))
                contradiction_score = torch.where(
                    safe_has_pair.float().sum(dim=1) > 0.0,
                    top_safe_vals.sum(dim=1) / top_safe_finite.float().sum(dim=1).clamp_min(1.0),
                    torch.zeros(num_nodes, device=device),
                )
                aggregate_contributor_count = top_safe_finite.float().sum(dim=1)
                feasible_upper_bound = torch.zeros(num_nodes, device=device)
                feasible_lower_bound = torch.zeros(num_nodes, device=device)
                violation_margin = torch.zeros(num_nodes, device=device)
                upper_bound_strength = torch.zeros(num_nodes, device=device)
                lower_bound_strength = torch.zeros(num_nodes, device=device)
                frontier_active = torch.zeros(num_nodes, device=device)
                frontier_set_size = torch.zeros(num_nodes, device=device)
                frontier_selected_positive_idx_tensor = torch.full(
                    (num_nodes, len(safes)),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                frontier_selected_count_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_selected_positive_idx_topm_tensor = torch.full(
                    (num_nodes, len(safes), 1),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                frontier_selected_upper_bound_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
                frontier_selected_violation_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
                frontier_selected_score_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
                frontier_upper_bound_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_safe_lower_bound_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_violation_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_score_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_anchor_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_safe_gate_tensor = torch.zeros(num_nodes, len(safes), device=device)
            elif self.contradiction_aggregation_mode == "frontier_adjacent_interval":
                safe_lower_expanded = safe_lower_bound.unsqueeze(2)
                positive_upper_expanded = positive_upper_bound.unsqueeze(1)
                frontier_violation_raw = safe_lower_expanded - positive_upper_expanded
                violated_mask = pair_valid & (frontier_violation_raw > 0.0)
                frontier_upper_candidates = torch.where(
                    violated_mask,
                    positive_upper_expanded,
                    torch.full_like(positive_upper_expanded, float("-inf")),
                )
                frontier_upper_bound_tensor, frontier_pos_idx = frontier_upper_candidates.max(dim=2)
                frontier_has_violation = torch.isfinite(frontier_upper_bound_tensor)
                safe_idx_expanded = frontier_pos_idx.unsqueeze(-1)
                frontier_anchor_tensor = torch.gather(anchor_strength, 1, frontier_pos_idx).where(
                    frontier_has_violation,
                    torch.zeros_like(frontier_upper_bound_tensor),
                )
                frontier_safe_gate_tensor = safe_gate.where(
                    frontier_has_violation,
                    torch.zeros_like(safe_gate),
                )
                frontier_selected_count_tensor = frontier_has_violation.float()
                frontier_selected_positive_idx_topm_tensor = frontier_pos_idx.unsqueeze(2)
                frontier_selected_upper_bound_topm_tensor = frontier_upper_bound_tensor.unsqueeze(2)
                frontier_safe_lower_bound_tensor = safe_lower_bound.where(
                    frontier_has_violation,
                    torch.zeros_like(safe_lower_bound),
                )
                frontier_violation_tensor = torch.relu(
                    frontier_safe_lower_bound_tensor - frontier_upper_bound_tensor.where(
                        frontier_has_violation,
                        torch.zeros_like(frontier_upper_bound_tensor),
                    )
                )
                frontier_margin_term = torch.relu(frontier_violation_tensor / self.tau_margin_min)
                frontier_score_tensor = frontier_anchor_tensor * frontier_safe_gate_tensor * frontier_margin_term
                frontier_selected_violation_topm_tensor = frontier_violation_tensor.unsqueeze(2)
                frontier_selected_score_topm_tensor = frontier_score_tensor.unsqueeze(2)
                safe_count = int(frontier_score_tensor.size(1))
                k = min(int(self.contradiction_top_k_pairs), safe_count)
                top_frontier_vals, _ = torch.topk(
                    torch.where(
                        frontier_has_violation,
                        frontier_score_tensor,
                        torch.full_like(frontier_score_tensor, float("-inf")),
                    ),
                    k=k,
                    dim=1,
                )
                top_frontier_finite = torch.isfinite(top_frontier_vals)
                top_frontier_vals = torch.where(top_frontier_finite, top_frontier_vals, torch.zeros_like(top_frontier_vals))
                contradiction_score = torch.where(
                    frontier_has_violation.float().sum(dim=1) > 0.0,
                    top_frontier_vals.sum(dim=1) / top_frontier_finite.float().sum(dim=1).clamp_min(1.0),
                    torch.zeros(num_nodes, device=device),
                )
                aggregate_contributor_count = top_frontier_finite.float().sum(dim=1)
                best_safe_idx = torch.argmax(
                    torch.where(
                        frontier_has_violation,
                        frontier_score_tensor,
                        torch.full_like(frontier_score_tensor, float("-inf")),
                    ),
                    dim=1,
                )
                gather_idx = best_safe_idx.view(-1, 1)
                feasible_upper_bound = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(frontier_upper_bound_tensor, 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                feasible_lower_bound = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(frontier_safe_lower_bound_tensor, 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                violation_margin = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(frontier_violation_tensor, 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                upper_bound_strength = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(frontier_anchor_tensor, 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                lower_bound_strength = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(frontier_safe_gate_tensor, 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                frontier_active = frontier_has_violation.any(dim=1).float()
                frontier_set_size = (frontier_has_violation.float().sum(dim=1) > 0.0).float() * 0.0
                for cand_idx in range(num_nodes):
                    cand_pos = frontier_pos_idx[cand_idx][frontier_has_violation[cand_idx]]
                    if cand_pos.numel() > 0:
                        frontier_set_size[cand_idx] = float(cand_pos.unique().numel())
                frontier_selected_positive_idx_tensor = torch.where(
                    frontier_has_violation,
                    frontier_pos_idx,
                    torch.full_like(frontier_pos_idx, -1),
                )
                frontier_upper_bound_tensor = torch.where(
                    frontier_has_violation,
                    frontier_upper_bound_tensor,
                    torch.zeros_like(frontier_upper_bound_tensor),
                )
            elif self.contradiction_aggregation_mode == "multi_frontier_interval_state":
                safe_lower_expanded = safe_lower_bound.unsqueeze(2)
                positive_upper_expanded = positive_upper_bound.unsqueeze(1)
                frontier_violation_raw = safe_lower_expanded - positive_upper_expanded
                violated_mask = pair_valid & (frontier_violation_raw > 0.0)
                frontier_top_m = min(int(self.frontier_top_m), int(positive_upper_expanded.size(2)))
                frontier_upper_candidates = torch.where(
                    violated_mask,
                    positive_upper_expanded,
                    torch.full_like(positive_upper_expanded, float("-inf")),
                )
                frontier_selected_upper_bound_topm_tensor, frontier_selected_positive_idx_topm_tensor = torch.topk(
                    frontier_upper_candidates,
                    k=frontier_top_m,
                    dim=2,
                )
                frontier_selected_valid = torch.isfinite(frontier_selected_upper_bound_topm_tensor)
                frontier_selected_positive_idx_topm_tensor = torch.where(
                    frontier_selected_valid,
                    frontier_selected_positive_idx_topm_tensor,
                    torch.full_like(frontier_selected_positive_idx_topm_tensor, -1),
                )
                frontier_selected_count_tensor = frontier_selected_valid.float().sum(dim=2)
                expanded_anchor = anchor_strength.unsqueeze(1).expand(-1, len(safes), -1)
                gathered_anchor = torch.gather(
                    expanded_anchor,
                    2,
                    torch.clamp(frontier_selected_positive_idx_topm_tensor, min=0),
                )
                gathered_violation = torch.gather(
                    frontier_violation_raw,
                    2,
                    torch.clamp(frontier_selected_positive_idx_topm_tensor, min=0),
                )
                frontier_selected_upper_bound_topm_tensor = torch.where(
                    frontier_selected_valid,
                    frontier_selected_upper_bound_topm_tensor,
                    torch.zeros_like(frontier_selected_upper_bound_topm_tensor),
                )
                frontier_selected_violation_topm_tensor = torch.where(
                    frontier_selected_valid,
                    torch.relu(gathered_violation),
                    torch.zeros_like(gathered_violation),
                )
                frontier_anchor_topm = torch.where(
                    frontier_selected_valid,
                    gathered_anchor,
                    torch.zeros_like(gathered_anchor),
                )
                frontier_safe_gate_topm = safe_gate.unsqueeze(2).expand_as(frontier_anchor_topm)
                frontier_selected_score_topm_tensor = (
                    frontier_anchor_topm
                    * frontier_safe_gate_topm
                    * torch.relu(frontier_selected_violation_topm_tensor / self.tau_margin_min)
                )
                frontier_safe_score = frontier_selected_score_topm_tensor.sum(dim=2) / torch.sqrt(
                    frontier_selected_count_tensor.clamp_min(1.0)
                )
                frontier_has_violation = frontier_selected_count_tensor > 0.0
                safe_count = int(frontier_safe_score.size(1))
                k = min(int(self.contradiction_top_k_pairs), safe_count)
                top_frontier_vals, top_frontier_idx = torch.topk(
                    torch.where(
                        frontier_has_violation,
                        frontier_safe_score,
                        torch.full_like(frontier_safe_score, float("-inf")),
                    ),
                    k=k,
                    dim=1,
                )
                top_frontier_finite = torch.isfinite(top_frontier_vals)
                top_frontier_vals = torch.where(top_frontier_finite, top_frontier_vals, torch.zeros_like(top_frontier_vals))
                contradiction_score = torch.where(
                    frontier_has_violation.float().sum(dim=1) > 0.0,
                    top_frontier_vals.sum(dim=1) / top_frontier_finite.float().sum(dim=1).clamp_min(1.0),
                    torch.zeros(num_nodes, device=device),
                )
                top_safe_selected_count = torch.gather(frontier_selected_count_tensor, 1, top_frontier_idx)
                aggregate_contributor_count = torch.where(
                    top_frontier_finite,
                    top_safe_selected_count,
                    torch.zeros_like(top_safe_selected_count),
                ).sum(dim=1)
                best_safe_idx = torch.argmax(
                    torch.where(
                        frontier_has_violation,
                        frontier_safe_score,
                        torch.full_like(frontier_safe_score, float("-inf")),
                    ),
                    dim=1,
                )
                gather_idx = best_safe_idx.view(-1, 1)
                feasible_upper_bound = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(
                        frontier_selected_upper_bound_topm_tensor[:, :, 0],
                        1,
                        gather_idx,
                    ).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                feasible_lower_bound = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(safe_lower_bound, 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                violation_margin = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(
                        frontier_selected_violation_topm_tensor[:, :, 0],
                        1,
                        gather_idx,
                    ).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                upper_bound_strength = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(frontier_anchor_topm[:, :, 0], 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                lower_bound_strength = torch.where(
                    frontier_has_violation.any(dim=1),
                    torch.gather(safe_gate, 1, gather_idx).squeeze(1),
                    torch.zeros(num_nodes, device=device),
                )
                frontier_active = frontier_has_violation.any(dim=1).float()
                frontier_set_size = torch.zeros(num_nodes, device=device)
                for cand_idx in range(num_nodes):
                    cand_pos = frontier_selected_positive_idx_topm_tensor[cand_idx][frontier_selected_valid[cand_idx]]
                    if cand_pos.numel() > 0:
                        frontier_set_size[cand_idx] = float(cand_pos.unique().numel())
                frontier_selected_positive_idx_tensor = frontier_selected_positive_idx_topm_tensor[:, :, 0]
                frontier_upper_bound_tensor = frontier_selected_upper_bound_topm_tensor[:, :, 0]
                frontier_safe_lower_bound_tensor = torch.where(
                    frontier_has_violation,
                    safe_lower_bound,
                    torch.zeros_like(safe_lower_bound),
                )
                frontier_violation_tensor = frontier_selected_violation_topm_tensor[:, :, 0]
                frontier_score_tensor = frontier_safe_score
                frontier_anchor_tensor = frontier_anchor_topm[:, :, 0]
                frontier_safe_gate_tensor = torch.where(
                    frontier_has_violation,
                    safe_gate,
                    torch.zeros_like(safe_gate),
                )
            else:
                current_time_value = float(
                    current_time_min
                    if current_time_min is not None
                    else max(
                        max((float(row.absolute_time_min) for row in positives), default=0.0),
                        max((float(row.absolute_time_min) for row in safes), default=0.0),
                    )
                )
                pos_age = torch.clamp(
                    current_time_value - t_pos,
                    min=0.0,
                )
                safe_age = torch.clamp(
                    current_time_value - t_safe,
                    min=0.0,
                )
                pos_decay = torch.exp(-pos_age / self.contradiction_time_decay_min).unsqueeze(0)
                safe_decay = torch.exp(-safe_age / self.contradiction_time_decay_min).unsqueeze(0)
                upper_weight = anchor_strength * pos_decay
                lower_weight = safe_gate * safe_decay

                neg_inf = torch.full_like(positive_upper_bound, float("-inf"))
                pos_log_weight = torch.where(
                    pos_finite,
                    torch.log(torch.clamp(upper_weight, min=self.eps)),
                    neg_inf,
                )
                safe_log_weight = torch.where(
                    safe_finite,
                    torch.log(torch.clamp(lower_weight, min=self.eps)),
                    torch.full_like(safe_lower_bound, float("-inf")),
                )
                bound_tau = self.tau_anchor_min
                pos_norm = torch.logsumexp(pos_log_weight, dim=1)
                safe_norm = torch.logsumexp(safe_log_weight, dim=1)
                pos_softmin_term = torch.where(
                    pos_finite,
                    pos_log_weight - positive_upper_bound / bound_tau,
                    torch.full_like(pos_log_weight, float("-inf")),
                )
                safe_softmax_term = torch.where(
                    safe_finite,
                    safe_log_weight + safe_lower_bound / bound_tau,
                    torch.full_like(safe_log_weight, float("-inf")),
                )
                feasible_upper_bound = torch.where(
                    finite_positive_count > 0.0,
                    -bound_tau
                    * (
                        torch.logsumexp(pos_softmin_term, dim=1)
                        - pos_norm
                    ),
                    torch.zeros(num_nodes, device=device),
                )
                feasible_lower_bound = torch.where(
                    finite_safe_count > 0.0,
                    bound_tau
                    * (
                        torch.logsumexp(safe_softmax_term, dim=1)
                        - safe_norm
                    ),
                    torch.zeros(num_nodes, device=device),
                )
                violation_margin = feasible_lower_bound - feasible_upper_bound
                upper_bound_strength = torch.where(
                    finite_positive_count > 0.0,
                    torch.max(torch.where(pos_finite, upper_weight, torch.zeros_like(upper_weight)), dim=1).values,
                    torch.zeros(num_nodes, device=device),
                )
                lower_bound_strength = torch.where(
                    finite_safe_count > 0.0,
                    torch.max(torch.where(safe_finite, lower_weight, torch.zeros_like(lower_weight)), dim=1).values,
                    torch.zeros(num_nodes, device=device),
                )
                bracket_activity = torch.sqrt(torch.clamp(upper_bound_strength * lower_bound_strength, min=0.0))
                contradiction_score = bracket_activity * torch.relu(violation_margin / self.tau_margin_min)
                contradiction_score = torch.where(
                    (finite_positive_count > 0.0) & (finite_safe_count > 0.0),
                    contradiction_score,
                    torch.zeros(num_nodes, device=device),
                )
                aggregate_contributor_count = (finite_positive_count > 0.0).float() + (finite_safe_count > 0.0).float()
                frontier_active = torch.zeros(num_nodes, device=device)
                frontier_set_size = torch.zeros(num_nodes, device=device)
                frontier_selected_positive_idx_tensor = torch.full(
                    (num_nodes, len(safes)),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                frontier_selected_count_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_selected_positive_idx_topm_tensor = torch.full(
                    (num_nodes, len(safes), 1),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                frontier_selected_upper_bound_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
                frontier_selected_violation_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
                frontier_selected_score_topm_tensor = torch.zeros(num_nodes, len(safes), 1, device=device)
                frontier_upper_bound_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_safe_lower_bound_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_violation_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_score_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_anchor_tensor = torch.zeros(num_nodes, len(safes), device=device)
                frontier_safe_gate_tensor = torch.zeros(num_nodes, len(safes), device=device)

        eligible_safe_witness_count = safe_has_pair.float().sum(dim=1)
        pair_count = valid_pair_count
        positive_margin_pair_count = (pair_valid & (delta > 0.0)).float().sum(dim=(1, 2))
        top_pair_margin = torch.where(
            valid_pair_count > 0.0,
            delta.masked_fill(~pair_valid, float("-inf")).amax(dim=(1, 2)),
            torch.zeros(num_nodes, device=device),
        )
        pair_available = (pair_count > 0.0).float()
        contradiction_nonzero = (contradiction_score > self.eps).float()

        if candidate_mask is not None:
            mask = candidate_mask.view(-1).float().to(device)
            contradiction_score = contradiction_score * mask
            pair_available = pair_available * mask
            contradiction_nonzero = contradiction_nonzero * mask
            eligible_safe_witness_count = eligible_safe_witness_count * mask
            pair_count = pair_count * mask
            positive_margin_pair_count = positive_margin_pair_count * mask
            top_pair_margin = top_pair_margin * mask
            aggregate_contributor_count = aggregate_contributor_count * mask
            feasible_upper_bound = feasible_upper_bound * mask
            feasible_lower_bound = feasible_lower_bound * mask
            violation_margin = violation_margin * mask
            upper_bound_strength = upper_bound_strength * mask
            lower_bound_strength = lower_bound_strength * mask
            frontier_active = frontier_active * mask
            frontier_set_size = frontier_set_size * mask
            frontier_selected_count_tensor = frontier_selected_count_tensor * mask.view(-1, 1)
            frontier_selected_upper_bound_topm_tensor = frontier_selected_upper_bound_topm_tensor * mask.view(-1, 1, 1)
            frontier_selected_violation_topm_tensor = frontier_selected_violation_topm_tensor * mask.view(-1, 1, 1)
            frontier_selected_score_topm_tensor = frontier_selected_score_topm_tensor * mask.view(-1, 1, 1)
            frontier_upper_bound_tensor = frontier_upper_bound_tensor * mask.view(-1, 1)
            frontier_safe_lower_bound_tensor = frontier_safe_lower_bound_tensor * mask.view(-1, 1)
            frontier_violation_tensor = frontier_violation_tensor * mask.view(-1, 1)
            frontier_score_tensor = frontier_score_tensor * mask.view(-1, 1)
            frontier_anchor_tensor = frontier_anchor_tensor * mask.view(-1, 1)
            frontier_safe_gate_tensor = frontier_safe_gate_tensor * mask.view(-1, 1)
            frontier_selected_positive_idx_tensor = torch.where(
                mask.view(-1, 1).bool(),
                frontier_selected_positive_idx_tensor,
                torch.full_like(frontier_selected_positive_idx_tensor, -1),
            )
            frontier_selected_positive_idx_topm_tensor = torch.where(
                mask.view(-1, 1, 1).bool(),
                frontier_selected_positive_idx_topm_tensor,
                torch.full_like(frontier_selected_positive_idx_topm_tensor, -1),
            )
            pair_valid = pair_valid & mask.view(-1, 1, 1).bool()

        return {
            "contradiction_score": contradiction_score,
            "pair_available": pair_available,
            "contradiction_nonzero": contradiction_nonzero,
            "eligible_safe_witness_count": eligible_safe_witness_count,
            "top_pair_margin": top_pair_margin,
            "pair_count": pair_count,
            "positive_margin_pair_count": positive_margin_pair_count,
            "aggregate_contributor_count": aggregate_contributor_count,
            "feasible_upper_bound": feasible_upper_bound,
            "feasible_lower_bound": feasible_lower_bound,
            "violation_margin": violation_margin,
            "upper_bound_strength": upper_bound_strength,
            "lower_bound_strength": lower_bound_strength,
            "frontier_active": frontier_active,
            "frontier_set_size": frontier_set_size,
            "frontier_selected_positive_idx_tensor": frontier_selected_positive_idx_tensor,
            "frontier_selected_count_tensor": frontier_selected_count_tensor,
            "frontier_selected_positive_idx_topm_tensor": frontier_selected_positive_idx_topm_tensor,
            "frontier_selected_upper_bound_topm_tensor": frontier_selected_upper_bound_topm_tensor,
            "frontier_selected_violation_topm_tensor": frontier_selected_violation_topm_tensor,
            "frontier_selected_score_topm_tensor": frontier_selected_score_topm_tensor,
            "frontier_upper_bound_tensor": frontier_upper_bound_tensor,
            "frontier_safe_lower_bound_tensor": frontier_safe_lower_bound_tensor,
            "frontier_violation_tensor": frontier_violation_tensor,
            "frontier_score_tensor": frontier_score_tensor,
            "frontier_anchor_tensor": frontier_anchor_tensor,
            "frontier_safe_gate_tensor": frontier_safe_gate_tensor,
            "pair_valid_tensor": pair_valid.float(),
        }

    def build_evidence_state_mini(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        history: ObservationWitnessHistory,
        current_time_min: float,
        support_context: SupportScoreContext,
        constraint_state: Optional[ConstraintState] = None,
        phys_ctx_mode: str = "witness_time_physctx",
    ) -> Dict[str, Any]:
        support_res = self.compute_support_score(
            observation_state=observation_state,
            physics_context=physics_context,
            current_time_min=current_time_min,
            support_context=support_context,
            history=history,
        )
        valid_mask = self.runtime_layer.build_valid_mask(
            num_nodes=int(observation_state.observed_flag.numel()),
            device=observation_state.observed_flag.device,
            constraint_state=constraint_state,
        )
        contradiction = self.compute_contradiction_score(
            history=history,
            num_nodes=int(observation_state.observed_flag.numel()),
            device=observation_state.observed_flag.device,
            phys_ctx_mode=phys_ctx_mode,
            current_phys_ctx=physics_context,
            current_time_min=float(current_time_min),
            candidate_mask=None,
        )
        evidence_mini = EvidenceStateMini(
            support_score=support_res["total"],
            contradiction_score=contradiction["contradiction_score"],
        )
        return {
            "evidence_state_mini": evidence_mini,
            "support_score": evidence_mini.support_score,
            "contradiction_score": evidence_mini.contradiction_score,
            "support_meta": support_res,
            "contradiction_meta": contradiction,
            "runtime_constraints": {
                "valid_mask": valid_mask,
            },
        }
