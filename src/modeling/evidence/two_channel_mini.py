from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.modeling.evidence.builder import EvidenceBuilder
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule


@dataclass
class WitnessRecord:
    local_idx: int
    global_idx: int
    absolute_time_min: float
    episode_id: int
    label: str
    confidence: float
    t_snapshot_idx: int
    concentration: float
    signal: float
    phys_ctx: Any


@dataclass
class WitnessHistoryMemory:
    positive_records: List[WitnessRecord] = field(default_factory=list)
    safe_records: List[WitnessRecord] = field(default_factory=list)

    def clear(self) -> None:
        self.positive_records.clear()
        self.safe_records.clear()

    def add_history_step(self, history_step: Any) -> None:
        for sample in history_step.samples:
            concentration = float(sample.concentration)
            if bool(sample.is_positive):
                confidence = float(torch.sigmoid(torch.tensor((concentration - 0.1) / 0.05)).item())
                record = WitnessRecord(
                    local_idx=int(sample.local_idx),
                    global_idx=int(sample.global_idx),
                    absolute_time_min=float(sample.time_min),
                    episode_id=int(history_step.episode),
                    label="positive",
                    confidence=confidence,
                    t_snapshot_idx=int(sample.t_snapshot_idx),
                    concentration=concentration,
                    signal=float(sample.signal),
                    phys_ctx=history_step.phys_ctx,
                )
                self.positive_records.append(record)
            if bool(sample.is_safe):
                confidence = float(torch.sigmoid(torch.tensor((0.1 - concentration) / 0.05)).item())
                record = WitnessRecord(
                    local_idx=int(sample.local_idx),
                    global_idx=int(sample.global_idx),
                    absolute_time_min=float(sample.time_min),
                    episode_id=int(history_step.episode),
                    label="safe",
                    confidence=confidence,
                    t_snapshot_idx=int(sample.t_snapshot_idx),
                    concentration=concentration,
                    signal=float(sample.signal),
                    phys_ctx=history_step.phys_ctx,
                )
                self.safe_records.append(record)


@dataclass
class EvidenceStateMini:
    support_score: torch.Tensor
    contradiction_score: torch.Tensor


class TwoChannelEvidenceMiniEnv:
    """
    Clean two-channel environment:
    - Layer A: witness history memory
    - Layer B: pair-level physics validity gate
    - Layer C: runtime constraints (consumed externally; not folded into evidence semantics)
    - Layer D: EvidenceStateMini = {support_score, contradiction_score}
    """

    def __init__(
        self,
        cfg: Optional[Any] = None,
        contradiction_top_k_pairs: int = 5,
        contradiction_margin_tau_min: float = 15.0,
        anchor_tau_min: float = 20.0,
        safe_gate_tau_min: float = 20.0,
    ):
        self.cfg = cfg
        self.evidence_builder = EvidenceBuilder(cfg)
        self.reachability = DynamicReachabilityRuleModule()
        self.history = WitnessHistoryMemory()
        self.contradiction_top_k_pairs = max(int(contradiction_top_k_pairs), 1)
        self.contradiction_margin_tau_min = float(contradiction_margin_tau_min)
        self.anchor_tau_min = float(anchor_tau_min)
        self.safe_gate_tau_min = float(safe_gate_tau_min)

    def clear_history(self) -> None:
        self.history.clear()

    def add_history_step(self, history_step: Any) -> None:
        self.history.add_history_step(history_step)

    def compute_support_score(
        self,
        observation_state: Any,
        physics_context: Any,
        current_time_min: float,
    ) -> torch.Tensor:
        t_sim = torch.tensor([float(current_time_min)], device=observation_state.observed_flag.device)
        support = self.evidence_builder.compute_support_score(
            observation_state,
            physics_context,
            None,
            {},
            t_sim=t_sim,
        )
        return support["total"]

    def _distance_to_witness(self, witness: WitnessRecord, num_nodes: int, device: torch.device) -> torch.Tensor:
        seed = torch.zeros(num_nodes, device=device)
        seed[int(witness.local_idx)] = 1.0
        phys_ctx = witness.phys_ctx
        if phys_ctx.stt_dynamic is not None:
            weights = torch.abs(phys_ctx.stt_dynamic.view(-1)).to(device)
        elif phys_ctx.stt_median is not None:
            weights = torch.expm1(phys_ctx.stt_median).to(device)
        elif phys_ctx.edge_attr is not None and phys_ctx.edge_attr.size(1) > 0:
            weights = torch.expm1(phys_ctx.edge_attr[:, 0]).to(device)
        else:
            weights = torch.ones(phys_ctx.edge_index.size(1), device=device) * 20.0
        return self.reachability.compute_distance(seed, phys_ctx, weights, num_nodes)

    def compute_contradiction_score(
        self,
        num_nodes: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        pos_records = self.history.positive_records
        safe_records = self.history.safe_records
        zero = torch.zeros(num_nodes, device=device)
        if len(pos_records) == 0 or len(safe_records) == 0:
            return {
                "total": zero,
                "pair_available": zero,
                "eligible_safe_witness_count": zero,
                "top_pair_margin": zero,
                "positive_count": torch.tensor(float(len(pos_records)), device=device),
                "safe_count": torch.tensor(float(len(safe_records)), device=device),
            }

        pos_dist = torch.stack(
            [self._distance_to_witness(rec, num_nodes, device) for rec in pos_records],
            dim=1,
        )  # [N, P]
        safe_dist = torch.stack(
            [self._distance_to_witness(rec, num_nodes, device) for rec in safe_records],
            dim=1,
        )  # [N, S]

        pos_time = torch.tensor([rec.absolute_time_min for rec in pos_records], device=device, dtype=torch.float32)
        safe_time = torch.tensor([rec.absolute_time_min for rec in safe_records], device=device, dtype=torch.float32)
        pos_conf = torch.tensor([rec.confidence for rec in pos_records], device=device, dtype=torch.float32)
        safe_conf = torch.tensor([rec.confidence for rec in safe_records], device=device, dtype=torch.float32)

        finite_pos = torch.isfinite(pos_dist) & (pos_dist < 1e8)
        finite_safe = torch.isfinite(safe_dist) & (safe_dist < 1e8)

        # A(c,p): candidate-conditioned positive anchor strength.
        anchor_slack = pos_time.view(1, -1) - pos_dist
        anchor = torch.sigmoid(anchor_slack / self.anchor_tau_min)
        anchor = anchor * pos_conf.view(1, -1) * finite_pos.float()

        # G(c,s): safe witness validity gate (topology/time/arrival + safe validity).
        safe_slack = safe_time.view(1, -1) - safe_dist
        safe_gate = torch.sigmoid(safe_slack / self.safe_gate_tau_min)
        safe_gate = safe_gate * safe_conf.view(1, -1) * finite_safe.float()

        # Delta(c,p,s) = (t_s - t_p) - (TT(c->s) - TT(c->p)).
        time_term = safe_time.view(1, 1, -1) - pos_time.view(1, -1, 1)
        tt_term = safe_dist.unsqueeze(1) - pos_dist.unsqueeze(2)
        delta = time_term - tt_term

        pair_valid = finite_pos.unsqueeze(2) & finite_safe.unsqueeze(1)
        pair_gate = anchor.unsqueeze(2) * safe_gate.unsqueeze(1) * pair_valid.float()
        margin_term = torch.sigmoid(delta / self.contradiction_margin_tau_min)
        pair_score = pair_gate * margin_term

        contradiction = torch.zeros(num_nodes, device=device)
        pair_available = torch.zeros(num_nodes, device=device)
        top_pair_margin = torch.zeros(num_nodes, device=device)
        eligible_safe_witness_count = torch.zeros(num_nodes, device=device)

        for cand_idx in range(num_nodes):
            valid_mask = pair_valid[cand_idx]
            if not bool(valid_mask.any()):
                continue
            valid_scores = pair_score[cand_idx][valid_mask]
            k = min(self.contradiction_top_k_pairs, int(valid_scores.numel()))
            top_scores, _ = torch.topk(valid_scores, k=k, largest=True, sorted=False)
            contradiction[cand_idx] = top_scores.mean()
            pair_available[cand_idx] = 1.0
            valid_delta = delta[cand_idx][valid_mask]
            top_pair_margin[cand_idx] = valid_delta.max()
            safe_any = valid_mask.any(dim=0)
            eligible_safe_witness_count[cand_idx] = safe_any.float().sum()

        return {
            "total": contradiction,
            "pair_available": pair_available,
            "eligible_safe_witness_count": eligible_safe_witness_count,
            "top_pair_margin": top_pair_margin,
            "positive_count": torch.tensor(float(len(pos_records)), device=device),
            "safe_count": torch.tensor(float(len(safe_records)), device=device),
        }

    def build_evidence_state_mini(
        self,
        observation_state: Any,
        physics_context: Any,
        current_time_min: float,
    ) -> EvidenceStateMini:
        support_score = self.compute_support_score(
            observation_state=observation_state,
            physics_context=physics_context,
            current_time_min=current_time_min,
        )
        contradiction_res = self.compute_contradiction_score(
            num_nodes=int(support_score.numel()),
            device=support_score.device,
        )
        return EvidenceStateMini(
            support_score=support_score,
            contradiction_score=contradiction_res["total"],
        )
