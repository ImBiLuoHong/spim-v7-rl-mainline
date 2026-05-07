import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum


def _cfg_get(cfg_obj, key: str, default):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


class SharedEvidenceResidualRefiner(nn.Module):
    """
    Minimal shared residual refiner for live EvidenceState fields.
    The shared trunk is node-wise, and only support_score / contradiction_score are
    updated. suspect_pool is forwarded unchanged as a frozen structural prior.
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = getattr(cfg, "model", None)
        refiner_cfg = _cfg_get(model_cfg, "evidence_refiner", {})

        hidden_dim = int(_cfg_get(refiner_cfg, "hidden_dim", 32))
        self.support_delta_scale = float(_cfg_get(refiner_cfg, "support_delta_scale", 1.0))
        self.contradiction_delta_scale = float(
            _cfg_get(refiner_cfg, "contradiction_delta_scale", _cfg_get(refiner_cfg, "suspect_delta_scale", 0.25))
        )
        self.support_focus_floor = float(_cfg_get(refiner_cfg, "support_focus_floor", 0.25))

        input_dim = 12
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.support_head = nn.Linear(hidden_dim, 1)
        self.contradiction_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.support_head.weight)
        nn.init.zeros_(self.support_head.bias)
        nn.init.zeros_(self.contradiction_head.weight)
        nn.init.zeros_(self.contradiction_head.bias)

    def _flat(self, tensor, reference):
        if tensor is None:
            return torch.zeros_like(reference)
        return tensor.float().view(-1)

    def _center_per_graph(self, tensor, batch_index):
        if batch_index is None:
            return tensor - tensor.mean()
        batch = batch_index.view(-1).to(device=tensor.device, dtype=torch.long)
        if batch.numel() != tensor.numel():
            return tensor - tensor.mean()
        graph_mean = scatter_mean(tensor, batch, dim=0)
        return tensor - graph_mean[batch]

    def _match_total_mass(self, base_value, refined_value, batch_index):
        if batch_index is None:
            base_mass = base_value.sum()
            refined_mass = refined_value.sum().clamp_min(1e-12)
            return refined_value * (base_mass / refined_mass)

        batch = batch_index.view(-1).to(device=base_value.device, dtype=torch.long)
        if batch.numel() != base_value.numel():
            return self._match_total_mass(base_value, refined_value, batch_index=None)

        base_mass = scatter_sum(base_value, batch, dim=0)
        refined_mass = scatter_sum(refined_value, batch, dim=0).clamp_min(1e-12)
        scale = base_mass / refined_mass
        return refined_value * scale[batch]

    def forward(self, evidence_state, observation_state, constraint_state=None, batch_index=None):
        base_support = evidence_state.support_score.float().view(-1)
        frozen_suspect = evidence_state.suspect_pool.float().view(-1).detach()
        base_contradiction = evidence_state.contradiction_score.float().view(-1)
        topology_gate = self._flat(evidence_state.topology_gate, base_support)
        coarse_time_gate = self._flat(evidence_state.coarse_time_gate, base_support)
        not_ruled_out_gate = self._flat(evidence_state.not_ruled_out_gate, base_support)

        feats = [
            base_support,
            frozen_suspect,
            base_contradiction,
            topology_gate,
            coarse_time_gate,
            not_ruled_out_gate,
            self._flat(observation_state.observed_flag, base_support),
            self._flat(observation_state.toxic_positive_flag, base_support),
            self._flat(observation_state.toxic_negative_flag, base_support),
            self._flat(observation_state.freshness, base_support),
            self._flat(getattr(constraint_state, "confirmed_non_source_mask", None), base_support),
            self._flat(getattr(constraint_state, "confirmed_source_mask", None), base_support),
        ]
        x = torch.stack(feats, dim=-1)
        hidden = self.shared_mlp(x)

        support_focus = self._flat(getattr(evidence_state, "support_focus_term", None), base_support)
        support_focus = torch.clamp(support_focus, min=0.0, max=1.0)
        support_gate = self.support_focus_floor + (1.0 - self.support_focus_floor) * support_focus
        support_delta = torch.tanh(self.support_head(hidden)).view(-1)
        support_delta = support_delta * support_gate * self.support_delta_scale
        support_delta = self._center_per_graph(support_delta, batch_index)
        refined_support = self._match_total_mass(
            base_support,
            torch.clamp(base_support + support_delta, min=0.0),
            batch_index,
        )
        support_delta = refined_support - base_support

        contradiction_delta = torch.tanh(self.contradiction_head(hidden)).view(-1)
        contradiction_delta = self._center_per_graph(contradiction_delta, batch_index)
        contradiction_pressure = torch.clamp(base_contradiction + (1.0 - not_ruled_out_gate), min=0.0, max=1.0)
        contradiction_gate = 0.25 + 0.75 * contradiction_pressure
        contradiction_delta = contradiction_delta * contradiction_gate * self.contradiction_delta_scale
        refined_contradiction = torch.clamp(base_contradiction + contradiction_delta, min=0.0)
        contradiction_delta = refined_contradiction - base_contradiction

        suspect_delta = torch.zeros_like(frozen_suspect)

        return {
            "support_score": refined_support,
            "suspect_pool": frozen_suspect,
            "contradiction_score": refined_contradiction,
            "support_delta": support_delta,
            "suspect_delta": suspect_delta,
            "contradiction_delta": contradiction_delta,
            "suspect_canonical_latent": frozen_suspect,
        }
