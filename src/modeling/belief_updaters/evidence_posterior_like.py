from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from src.modeling.interfaces.belief_updater import BeliefStateBase, BeliefUpdaterBase, BeliefUpdaterCapabilities
from src.modeling.loop.navigator_vnext_contract import build_candidate_semantics, default_reward_contract, tensor_attr
from src.modeling.registry import BELIEF_UPDATER_REGISTRY


def _masked_zscore(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    values = values.view(-1).float()
    mask = mask.view(-1).bool()
    out = torch.zeros_like(values)
    if not bool(mask.any()):
        return out
    masked_vals = values[mask]
    mean = masked_vals.mean()
    std = masked_vals.std(unbiased=False).clamp_min(float(eps))
    out[mask] = (masked_vals - mean) / std
    return out


def _evidence_contrast_scalar(node_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    core = node_features[:, :7].float()
    valid = valid_mask.view(-1).bool()
    if not bool(valid.any()):
        return torch.zeros(node_features.size(0), device=node_features.device, dtype=torch.float32)

    valid_core = core[valid]
    means = valid_core.mean(dim=0, keepdim=True)
    stds = valid_core.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    maxs = valid_core.max(dim=0, keepdim=True).values

    percentile_parts = []
    for dim_idx in range(valid_core.size(1)):
        col_valid = valid_core[:, dim_idx]
        col_all = core[:, dim_idx : dim_idx + 1]
        percentile_parts.append((col_valid.view(1, -1) <= col_all).float().mean(dim=1, keepdim=True))
    percentile_rank = torch.cat(percentile_parts, dim=1)
    z_score = (core - means) / stds
    gap_to_max = core - maxs

    positive_dims = [0, 2, 4]
    negative_dims = [1, 3, 5, 6]
    positive_strength = (
        percentile_rank[:, positive_dims].mean(dim=1)
        + z_score[:, positive_dims].mean(dim=1)
        + gap_to_max[:, positive_dims].mean(dim=1)
    )
    negative_strength = (
        percentile_rank[:, negative_dims].mean(dim=1)
        + z_score[:, negative_dims].mean(dim=1)
        + gap_to_max[:, negative_dims].mean(dim=1)
    )
    return (positive_strength - negative_strength).float()


@dataclass
class EvidencePosteriorLikeState(BeliefStateBase):
    prev_belief: Optional[torch.Tensor]
    prev_entropy: Optional[torch.Tensor]
    step_count: int = 0

    def detach(self):
        return EvidencePosteriorLikeState(
            prev_belief=None if self.prev_belief is None else self.prev_belief.detach(),
            prev_entropy=None if self.prev_entropy is None else self.prev_entropy.detach(),
            step_count=self.step_count,
        )

    def to(self, device: torch.device):
        return EvidencePosteriorLikeState(
            prev_belief=None if self.prev_belief is None else self.prev_belief.to(device),
            prev_entropy=None if self.prev_entropy is None else self.prev_entropy.to(device),
            step_count=self.step_count,
        )


@BELIEF_UPDATER_REGISTRY.register("evidence_posterior_like")
class EvidencePosteriorLikeBelief(BeliefUpdaterBase):
    """
    Posterior-like candidate-source belief map built on current evidence semantics.

    Hard prior / support:
    - candidate_mask from build_candidate_semantics

    Soft evidence:
    - q_score
    - reasoner logits
    - evidence-core contrast scalar
    - contradiction score (negative energy contribution)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        lambda_q: float = 0.75,
        lambda_reasoner: float = 1.00,
        lambda_contrast: float = 0.50,
        lambda_contradiction: float = 0.25,
        support_plausible_delta: float = 0.25,
        not_ruled_out_threshold: float = 0.5,
        confusion_logit_delta: float = 1.0,
    ):
        super().__init__()
        self.temperature = float(max(temperature, 1e-6))
        self.lambda_q = float(lambda_q)
        self.lambda_reasoner = float(lambda_reasoner)
        self.lambda_contrast = float(lambda_contrast)
        self.lambda_contradiction = float(lambda_contradiction)
        self.support_plausible_delta = float(support_plausible_delta)
        self.not_ruled_out_threshold = float(not_ruled_out_threshold)
        self.confusion_logit_delta = float(max(confusion_logit_delta, 1e-6))

    def init_state(
        self,
        batch_size: int,
        num_nodes: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> EvidencePosteriorLikeState:
        return EvidencePosteriorLikeState(prev_belief=None, prev_entropy=None, step_count=0)

    def _step_impl(
        self,
        state: EvidencePosteriorLikeState,
        step_in: Dict[str, Any],
    ) -> Tuple[EvidencePosteriorLikeState, Dict[str, Any]]:
        valid_mask = step_in["valid_mask"].view(-1).bool()
        reference = valid_mask.float()
        batch = step_in.get("batch")
        if batch is None:
            batch = torch.zeros_like(reference, dtype=torch.long)
        else:
            batch = batch.view(-1).long().to(device=reference.device)

        evidence_state = step_in.get("evidence_state")
        constraint_state = step_in.get("constraint_state")
        reasoner_logits = step_in.get("reasoner_logits")
        if reasoner_logits is None:
            reasoner_logits = torch.zeros_like(reference)
        reasoner_logits = reasoner_logits.view(-1).float().to(device=reference.device)

        node_features = step_in.get("node_features")
        if node_features is None:
            node_features = torch.zeros(reference.numel(), 21, device=reference.device, dtype=torch.float32)
        node_features = node_features.float().to(device=reference.device)

        contract_cfg = default_reward_contract()
        contract_cfg.update(
            {
                "support_plausible_delta": self.support_plausible_delta,
                "not_ruled_out_threshold": self.not_ruled_out_threshold,
            }
        )
        semantics = build_candidate_semantics(
            evidence_state=evidence_state,
            constraint_state=constraint_state,
            valid_mask=valid_mask,
            batch=batch,
            contract_cfg=contract_cfg,
        )
        candidate_mask = semantics["candidate_mask"].view(-1).bool()
        if not bool(candidate_mask.any()):
            candidate_mask = valid_mask

        q_score = semantics["q_score"].view(-1).float()
        contradiction_score = tensor_attr(evidence_state, "contradiction_score", reference, default=0.0)
        contrast_signal = _evidence_contrast_scalar(node_features, valid_mask)

        q_z = _masked_zscore(q_score, candidate_mask)
        logits_z = _masked_zscore(reasoner_logits, candidate_mask)
        contrast_z = _masked_zscore(contrast_signal, candidate_mask)
        contradiction_z = _masked_zscore(contradiction_score, candidate_mask)

        energy = (
            self.lambda_q * q_z
            + self.lambda_reasoner * logits_z
            + self.lambda_contrast * contrast_z
            - self.lambda_contradiction * contradiction_z
        )

        masked_energy = energy.clone()
        masked_energy[~candidate_mask] = -float("inf")
        belief = torch.zeros_like(masked_energy)
        if bool(candidate_mask.any()):
            belief[candidate_mask] = torch.softmax(masked_energy[candidate_mask] / self.temperature, dim=0)

        entropy = torch.tensor(0.0, device=belief.device)
        top1_mass = torch.tensor(0.0, device=belief.device)
        top3_mass = torch.tensor(0.0, device=belief.device)
        top5_mass = torch.tensor(0.0, device=belief.device)
        cluster_mask = torch.zeros_like(candidate_mask)
        cluster_mass = torch.tensor(0.0, device=belief.device)
        cluster_count = torch.tensor(0.0, device=belief.device)
        hardest_confuser_idx = None
        order = torch.empty(0, dtype=torch.long, device=belief.device)
        if bool(candidate_mask.any()):
            probs = belief[candidate_mask]
            entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum()
            top1_mass = probs.max()
            topk = min(3, int(probs.numel()))
            if topk > 0:
                top3_mass = torch.topk(probs, k=topk).values.sum()
            topk5 = min(5, int(probs.numel()))
            if topk5 > 0:
                top5_mass = torch.topk(probs, k=topk5).values.sum()
            top1_energy = float(masked_energy[candidate_mask].max().item())
            cluster_mask = candidate_mask & (masked_energy >= top1_energy - self.confusion_logit_delta)
            cluster_mass = belief[cluster_mask].sum()
            cluster_count = cluster_mask.float().sum()
            order = torch.argsort(masked_energy, descending=True)
            order = order[torch.isfinite(masked_energy[order])]

        new_state = EvidencePosteriorLikeState(
            prev_belief=belief.detach(),
            prev_entropy=entropy.detach().view(1),
            step_count=int(state.step_count) + 1,
        )
        belief_ctx = {
            "belief": belief,
            "candidate_mask": candidate_mask,
            "energy": energy,
            "q_score": q_score,
            "contrast_signal": contrast_signal,
            "entropy": entropy,
            "top1_mass": top1_mass,
            "top3_mass": top3_mass,
            "top5_mass": top5_mass,
            "cluster_mask": cluster_mask.float(),
            "cluster_mass": cluster_mass,
            "cluster_count": cluster_count,
            "ordered_candidates": order,
            "hardest_confuser_idx": hardest_confuser_idx,
            "audit": {
                "temperature": float(self.temperature),
                "lambda_q": float(self.lambda_q),
                "lambda_reasoner": float(self.lambda_reasoner),
                "lambda_contrast": float(self.lambda_contrast),
                "lambda_contradiction": float(self.lambda_contradiction),
                "candidate_count": int(candidate_mask.float().sum().item()),
                "entropy": float(entropy.item()),
                "top1_mass": float(top1_mass.item()),
                "top3_mass": float(top3_mass.item()),
                "top5_mass": float(top5_mass.item()),
                "cluster_mass": float(cluster_mass.item()),
                "cluster_count": int(cluster_count.item()),
            },
        }
        return new_state, belief_ctx

    def capabilities(self) -> BeliefUpdaterCapabilities:
        return {
            "provides_node_belief": True,
            "provides_global_belief": True,
            "provides_memory_bank": False,
            "output_fields": [
                "belief",
                "entropy",
                "top1_mass",
                "top3_mass",
                "top5_mass",
                "cluster_mask",
                "cluster_mass",
                "cluster_count",
                "energy",
                "q_score",
                "contrast_signal",
            ],
        }
