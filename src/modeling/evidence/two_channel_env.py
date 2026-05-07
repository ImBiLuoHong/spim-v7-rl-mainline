from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.modeling.evidence.builder import EvidenceBuilder
from src.modeling.evidence.contradiction_oracle_v1 import (
    DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    DEFAULT_SAFE_VIOLATION_TAU_MIN,
    PracticalContradictionV2Config,
)
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.state.schema import ConstraintState, ObservationState, PhysicsContext


@dataclass
class ObservationWitnessHistory:
    """
    Layer A: observation and witness history (episode-cumulative memory).
    """

    history_steps: Sequence[Any]


@dataclass
class PhysicsEvidenceValidity:
    """
    Layer B: physics/evidence-validity gating diagnostics.
    """

    pair_available: torch.Tensor
    positive_margin_available: torch.Tensor
    eligible_safe_witness_count: torch.Tensor
    top_witness_margin: torch.Tensor


@dataclass
class RuntimeConstraintView:
    """
    Layer C: runtime hard constraints (kept outside evidence semantics).
    """

    constraint_state: Optional[ConstraintState] = None


@dataclass
class EvidenceStateMini:
    """
    Layer D: official evidence semantics.
    """

    support_score: torch.Tensor
    contradiction_score: torch.Tensor


class TwoChannelEvidenceEnvironment:
    """
    Clean navigator-only two-channel evidence environment:
    - support_score
    - contradiction_score
    """

    def __init__(
        self,
        cfg=None,
        evidence_builder: Optional[EvidenceBuilder] = None,
        practical_v2_config: Optional[PracticalContradictionV2Config] = None,
        safe_violation_tau_min: float = DEFAULT_SAFE_VIOLATION_TAU_MIN,
        history_phys_ctx_mode: str = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    ):
        self.cfg = cfg
        self.evidence_builder = evidence_builder or EvidenceBuilder(cfg)
        self.reachability_module = DynamicReachabilityRuleModule()
        self.practical_v2_config = practical_v2_config or PracticalContradictionV2Config(
            label="two_channel_mini_default",
            normalize_by_eligible_safe_count=True,
        )
        self.safe_violation_tau_min = float(safe_violation_tau_min)
        self.history_phys_ctx_mode = str(history_phys_ctx_mode)

    @staticmethod
    def _candidate_mask(num_nodes: int, device: torch.device, candidate_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if candidate_mask is None:
            return torch.ones(num_nodes, device=device)
        return candidate_mask.view(-1).float().to(device)

    @staticmethod
    def _as_t_sim(abs_time_min: Optional[float], device: torch.device) -> Optional[torch.Tensor]:
        if abs_time_min is None:
            return None
        return torch.tensor([float(abs_time_min)], dtype=torch.float32, device=device)

    def compute_reachability(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        abs_time_min: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        if physics_context.batch is not None:
            batch = physics_context.batch
        else:
            batch = torch.zeros(
                observation_state.observed_flag.size(0),
                dtype=torch.long,
                device=observation_state.observed_flag.device,
            )
        t_sim = self._as_t_sim(abs_time_min, observation_state.observed_flag.device)
        if physics_context.stt_dynamic is not None:
            return self.evidence_builder.dynamic_reachability.compute_reachability(
                observation_state,
                physics_context,
                t_sim,
                batch,
            )
        return self.evidence_builder.reachability.compute_reachability(
            observation_state,
            physics_context,
            t_sim,
            batch,
        )

    def compute_support_score(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        abs_time_min: Optional[float],
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        num_nodes = int(observation_state.observed_flag.numel())
        mask = self._candidate_mask(num_nodes, observation_state.observed_flag.device, candidate_mask)
        t_sim = self._as_t_sim(abs_time_min, observation_state.observed_flag.device)
        reach = self.compute_reachability(observation_state, physics_context, abs_time_min)
        support_res = self.evidence_builder.compute_support_score(
            observation_state,
            physics_context,
            mask,
            reach,
            t_sim=t_sim,
        )
        return {"reachability": reach, "support_result": support_res}

    def compute_contradiction_score(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        abs_time_min: Optional[float],
        history: ObservationWitnessHistory,
        candidate_mask: Optional[torch.Tensor] = None,
        history_mode: str = "cumulative",
    ) -> Dict[str, Any]:
        history_steps: Sequence[Any] = history.history_steps
        if history_mode == "snapshot":
            history_steps = history_steps[-1:] if history_steps else []
        if history_mode not in {"snapshot", "cumulative"}:
            raise ValueError(f"Unknown history_mode: {history_mode}")

        num_nodes = int(observation_state.observed_flag.numel())
        mask = self._candidate_mask(num_nodes, observation_state.observed_flag.device, candidate_mask)
        t_sim = self._as_t_sim(abs_time_min, observation_state.observed_flag.device)
        reach = self.compute_reachability(observation_state, physics_context, abs_time_min)
        contra_res = self.evidence_builder.compute_contradiction_score(
            observation_state,
            physics_context,
            mask,
            reach,
            t_sim=t_sim,
            contradiction_mode="practical_v2",
            oracle_history_steps=history_steps,
            safe_violation_tau_min=self.safe_violation_tau_min,
            history_phys_ctx_mode=self.history_phys_ctx_mode,
            practical_v2_config=self.practical_v2_config,
        )
        return {"reachability": reach, "contradiction_result": contra_res}

    def build_evidence_state_mini(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        abs_time_min: Optional[float],
        history: ObservationWitnessHistory,
        candidate_mask: Optional[torch.Tensor] = None,
        history_mode: str = "cumulative",
    ) -> Dict[str, Any]:
        support_pack = self.compute_support_score(
            observation_state=observation_state,
            physics_context=physics_context,
            abs_time_min=abs_time_min,
            candidate_mask=candidate_mask,
        )
        contra_pack = self.compute_contradiction_score(
            observation_state=observation_state,
            physics_context=physics_context,
            abs_time_min=abs_time_min,
            history=history,
            candidate_mask=candidate_mask,
            history_mode=history_mode,
        )
        support_res = support_pack["support_result"]
        contra_res = contra_pack["contradiction_result"]
        mini = EvidenceStateMini(
            support_score=support_res["total"],
            contradiction_score=contra_res["total"],
        )
        physics_gate = PhysicsEvidenceValidity(
            pair_available=contra_res["pair_available"],
            positive_margin_available=contra_res["positive_margin_available"],
            eligible_safe_witness_count=contra_res["eligible_safe_witness_count"],
            top_witness_margin=contra_res["top_witness_margin"],
        )
        return {
            "evidence_state_mini": mini,
            "physics_validity": physics_gate,
            "support_result": support_res,
            "contradiction_result": contra_res,
            "reachability": support_pack["reachability"],
        }
