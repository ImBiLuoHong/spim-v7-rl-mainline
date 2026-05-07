import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import softmax as gnn_softmax
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from src.modeling.interfaces.base import NavigatorBase
from src.modeling.navigators.backbones import SageBackbone
from src.modeling.registry import NAVIGATOR_REGISTRY


def _cfg_get(cfg_obj: Any, key: str, default: Any) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _state_value(state_obj: Any, key: str, reference: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if state_obj is None:
        return torch.full_like(reference.view(-1), float(default), dtype=torch.float32)
    if isinstance(state_obj, dict):
        value = state_obj.get(key)
    else:
        value = getattr(state_obj, key, None)
    if value is None:
        return torch.full_like(reference.view(-1), float(default), dtype=torch.float32)
    return value.view(-1).to(device=reference.device, dtype=torch.float32)


@NAVIGATOR_REGISTRY.register("navigator_vnext")
class NavigatorVNext(NavigatorBase):
    """
    Navigator-only actor-critic policy over the current fused state semantics.

    The existing `backbone` attribute is preserved for Phase45 re-embedding compatibility.
    The actual RL policy uses a fused-space GraphSAGE encoder over `h_fused` plus semantic
    node features, then samples up to K nodes autoregressively without replacement.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._profile_enabled = os.environ.get("NAVIGATOR_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
        self._profile: Dict[str, float] = {}
        nav_cfg = getattr(cfg.model, "navigator", {})
        vnext_cfg = getattr(cfg.model, "navigator_vnext", {})

        def merged_cfg(key: str, default: Any) -> Any:
            vnext_val = _cfg_get(vnext_cfg, key, None)
            if vnext_val is not None:
                return vnext_val
            return _cfg_get(nav_cfg, key, default)

        hidden_dim = int(merged_cfg("hidden_dim", 128))
        actor_hidden_dim = int(merged_cfg("actor_hidden_dim", hidden_dim))
        critic_hidden_dim = int(merged_cfg("critic_hidden_dim", hidden_dim))
        policy_layers = int(merged_cfg("policy_layers", 3))
        self.max_picks = int(getattr(cfg.model, "sample_budget", 3))
        self.greedy_eval = bool(merged_cfg("greedy_eval", True))
        self.source_like_bonus_scale = float(merged_cfg("source_like_bonus_scale", 0.0))
        self.source_like_bonus_later_round_multiplier = float(
            merged_cfg("source_like_bonus_later_round_multiplier", 1.0)
        )
        self.flat_drift_penalty_scale = float(merged_cfg("flat_drift_penalty_scale", 0.0))
        self.topology_bonus_scale = float(merged_cfg("topology_bonus_scale", 0.0))
        self.topology_bonus_later_round_multiplier = float(
            merged_cfg("topology_bonus_later_round_multiplier", 1.0)
        )
        self.evidence_gate_scale = float(merged_cfg("evidence_gate_scale", 0.25))
        self.evidence_bias_scale = float(merged_cfg("evidence_bias_scale", 0.10))
        self.use_aux_privileged_critic = bool(merged_cfg("use_aux_privileged_critic", False))

        # Preserve the old observation backbone interface so EpisodeRunner can keep
        # re-embedding raw observations after each environment state update.
        self.backbone = SageBackbone(cfg)

        # Base graph encoder consumes deployable observation/constraint/runtime features only.
        self.base_feature_names = [
            "observed_flag",
            "chlorine_deviation",
            "toxic_positive_flag",
            "toxic_negative_flag",
            "freshness",
            "confirmed_non_source_mask",
            "confirmed_source_mask",
            "sampled_mask",
            "no_resample_mask",
            "valid_mask",
        ]
        # Evidence controller is a separate stream that injects nodewise gate/bias.
        self.evidence_feature_names = [
            "support_score",
            "uncertainty_gap",
            "suspect_pool",
            "not_ruled_out_gate",
            "topology_gate",
            "arrival_gate",
            "contradiction_score",
        ]
        base_dim = len(self.base_feature_names)
        summary_dim = int(getattr(cfg.model, "nav_state_summary", {}).get("dim", 6))
        evidence_dim = len(self.evidence_feature_names)

        self.node_input_proj = nn.Sequential(
            nn.Linear(hidden_dim + base_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.evidence_gate_proj = nn.Linear(evidence_dim, hidden_dim)
        self.evidence_bias_proj = nn.Linear(evidence_dim, hidden_dim)

        self.policy_encoder = nn.ModuleList()
        self.policy_norms = nn.ModuleList()
        for _ in range(max(policy_layers, 1)):
            self.policy_encoder.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
            self.policy_norms.append(nn.LayerNorm(hidden_dim))

        actor_input_dim = hidden_dim * 2 + summary_dim + 2
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_input_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1),
        )
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_dim + summary_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1),
        )
        self.aux_critic_mlp = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + summary_dim, critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(critic_hidden_dim, critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(critic_hidden_dim, 1),
            )
            if self.use_aux_privileged_critic
            else None
        )

    def _profile_tic(self, reference: torch.Tensor | None = None) -> float | None:
        if not self._profile_enabled:
            return None
        if reference is not None and reference.device.type == "cuda":
            torch.cuda.synchronize(reference.device)
        return time.perf_counter()

    def _profile_add(self, key: str, start_time: float | None, reference: torch.Tensor | None = None) -> None:
        if start_time is None:
            return
        if reference is not None and reference.device.type == "cuda":
            torch.cuda.synchronize(reference.device)
        self._profile[key] = self._profile.get(key, 0.0) + float(time.perf_counter() - start_time)

    def _profile_inc(self, key: str, value: float = 1.0) -> None:
        if not self._profile_enabled:
            return
        self._profile[key] = self._profile.get(key, 0.0) + float(value)

    def reset_profile(self) -> None:
        self._profile.clear()

    def get_profile(self) -> Dict[str, float]:
        return dict(self._profile)

    def _build_base_features(self, state: Dict[str, Any]) -> torch.Tensor:
        h_fused = state["h_fused"]
        obs_state = state.get("observation_state")
        constraint_state = state.get("constraint_state")
        valid_mask = state.get("valid_mask")

        features = [
            _state_value(obs_state, "observed_flag", h_fused, default=0.0),
            _state_value(obs_state, "chlorine_deviation", h_fused, default=0.0),
            _state_value(obs_state, "toxic_positive_flag", h_fused, default=0.0),
            _state_value(obs_state, "toxic_negative_flag", h_fused, default=0.0),
            _state_value(obs_state, "freshness", h_fused, default=0.0),
            _state_value(constraint_state, "confirmed_non_source_mask", h_fused, default=0.0),
            _state_value(constraint_state, "confirmed_source_mask", h_fused, default=0.0),
            _state_value(constraint_state, "sampled_mask", h_fused, default=0.0),
            _state_value(constraint_state, "no_resample_mask", h_fused, default=0.0),
            (
                valid_mask.view(-1).to(device=h_fused.device, dtype=torch.float32)
                if valid_mask is not None
                else torch.ones(h_fused.size(0), device=h_fused.device, dtype=torch.float32)
            ),
        ]
        return torch.stack(features, dim=-1)

    def _build_evidence_features(self, state: Dict[str, Any]) -> torch.Tensor:
        h_fused = state["h_fused"]
        evidence_state = state.get("evidence_state")
        features = [
            _state_value(evidence_state, "support_score", h_fused, default=0.0),
            _state_value(evidence_state, "uncertainty_gap", h_fused, default=0.0),
            _state_value(evidence_state, "suspect_pool", h_fused, default=0.0),
            _state_value(evidence_state, "not_ruled_out_gate", h_fused, default=1.0),
            _state_value(evidence_state, "topology_gate", h_fused, default=0.0),
            _state_value(evidence_state, "arrival_gate", h_fused, default=0.0),
            _state_value(evidence_state, "contradiction_score", h_fused, default=0.0),
        ]
        return torch.stack(features, dim=-1)

    def _encode_policy_state(self, state: Dict[str, Any]) -> torch.Tensor:
        h_fused = state["h_fused"]
        edge_index = state["fused_edge_index"]
        t_prof = self._profile_tic(h_fused)
        x_base = self._build_base_features(state)
        self._profile_add("feature_build_s", t_prof, h_fused)
        t_prof = self._profile_tic(h_fused)
        x_evidence = self._build_evidence_features(state)
        self._profile_add("evidence_feature_build_s", t_prof, h_fused)
        t_prof = self._profile_tic(h_fused)
        x = torch.cat([h_fused.float(), x_base], dim=-1)
        self._profile_add("encoder_materialize_s", t_prof, h_fused)
        t_prof = self._profile_tic(h_fused)
        x = self.node_input_proj(x)
        self._profile_add("node_input_proj_s", t_prof, h_fused)
        t_prof = self._profile_tic(h_fused)
        evidence_gate = torch.sigmoid(self.evidence_gate_proj(x_evidence))
        evidence_bias = torch.tanh(self.evidence_bias_proj(x_evidence))
        self._profile_add("evidence_controller_s", t_prof, h_fused)
        emit_action_audit_scalars = bool(state.get("emit_action_audit_scalars", True))
        if self._profile_enabled or emit_action_audit_scalars:
            self._last_evidence_gate_mean = float(evidence_gate.detach().mean().cpu())
        else:
            self._last_evidence_gate_mean = 0.0
        x = x * (1.0 + self.evidence_gate_scale * evidence_gate)
        x = x + self.evidence_bias_scale * evidence_bias
        for conv, norm in zip(self.policy_encoder, self.policy_norms):
            t_prof = self._profile_tic(h_fused)
            residual = x
            x = conv(x, edge_index)
            x = norm(F.relu(x) + residual)
            self._profile_add("policy_conv_s", t_prof, h_fused)
        return x

    def _privileged_source_context(
        self,
        h_policy: torch.Tensor,
        batch: torch.Tensor,
        graph_count: int,
        critic_privileged: Dict[str, Any],
    ) -> torch.Tensor:
        true_source_mask = critic_privileged.get("true_source_mask")
        if true_source_mask is None:
            return torch.zeros(graph_count, h_policy.size(-1), device=h_policy.device, dtype=h_policy.dtype)
        true_source_mask = true_source_mask.view(-1).to(device=h_policy.device, dtype=h_policy.dtype).clamp(0.0, 1.0)
        weighted = h_policy * true_source_mask.unsqueeze(-1)
        source_sum = scatter_mean(weighted, batch, dim=0, dim_size=graph_count)
        source_count = scatter_mean(true_source_mask.unsqueeze(-1), batch, dim=0, dim_size=graph_count).clamp_min(1e-6)
        return source_sum / source_count

    def _local_logit_bias(self, state: Dict[str, Any], batch: torch.Tensor, graph_count: int) -> torch.Tensor:
        h_fused = state["h_fused"]
        bias = torch.zeros(h_fused.size(0), device=h_fused.device, dtype=torch.float32)
        if (
            abs(self.source_like_bonus_scale) <= 1e-12
            and abs(self.flat_drift_penalty_scale) <= 1e-12
            and abs(self.topology_bonus_scale) <= 1e-12
        ):
            return bias

        obs_state = state.get("observation_state")
        evidence_state = state.get("evidence_state")
        constraint_state = state.get("constraint_state")

        observed = _state_value(obs_state, "observed_flag", h_fused, default=0.0).clamp(0.0, 1.0)
        suspect = _state_value(evidence_state, "suspect_pool", h_fused, default=0.0).clamp(0.0, 1.0)
        topology = _state_value(evidence_state, "topology_gate", h_fused, default=0.0).clamp(0.0, 1.0)
        not_ruled_out = _state_value(evidence_state, "not_ruled_out_gate", h_fused, default=1.0).clamp(0.0, 1.0)
        sampled = _state_value(constraint_state, "sampled_mask", h_fused, default=0.0).clamp(0.0, 1.0)
        valid_mask = state.get("valid_mask")
        if valid_mask is None:
            valid_float = torch.ones(h_fused.size(0), device=h_fused.device, dtype=torch.float32)
        else:
            valid_float = valid_mask.view(-1).to(device=h_fused.device, dtype=torch.float32).clamp(0.0, 1.0)

        # Source-like cue: keep it strictly non-oracle and local to existing evidence fields.
        source_like = suspect * topology * not_ruled_out * (1.0 - observed) * (1.0 - sampled) * valid_float
        graph_has_history = scatter_mean(sampled, batch, dim=0, dim_size=graph_count)
        later_round = (graph_has_history[batch] > 0.0).to(dtype=torch.float32)
        if abs(self.source_like_bonus_scale) > 1e-12:
            later_multiplier = 1.0 + later_round * max(self.source_like_bonus_later_round_multiplier - 1.0, 0.0)
            bias = bias + self.source_like_bonus_scale * source_like * later_multiplier

        if abs(self.topology_bonus_scale) > 1e-12:
            topology_like = topology * (1.0 - observed) * (1.0 - sampled) * valid_float
            later_multiplier = 1.0 + later_round * max(self.topology_bonus_later_round_multiplier - 1.0, 0.0)
            bias = bias + self.topology_bonus_scale * topology_like * later_multiplier

        if abs(self.flat_drift_penalty_scale) > 1e-12:
            flat_drift = (1.0 - suspect) * (1.0 - topology) * (1.0 - observed) * (1.0 - sampled) * valid_float
            bias = bias - self.flat_drift_penalty_scale * flat_drift
        return bias

    def _select_pick(
        self,
        pick_logits: torch.Tensor,
        batch: torch.Tensor,
        available_mask: torch.Tensor,
        graph_count: int,
    ):
        t_prof = self._profile_tic(pick_logits)
        log_prob = pick_logits.new_zeros(graph_count)
        entropy = pick_logits.new_zeros(graph_count)
        probs_dense = pick_logits.new_zeros(pick_logits.size(0))
        candidate_idx = torch.nonzero(available_mask, as_tuple=True)[0]
        if candidate_idx.numel() == 0:
            empty = torch.empty(0, device=pick_logits.device, dtype=torch.long)
            self._profile_add("sampling_s", t_prof, pick_logits)
            return empty, empty, log_prob, entropy, probs_dense

        candidate_batch = batch[candidate_idx].view(-1).long()
        candidate_logits = pick_logits[candidate_idx].float()
        candidate_counts = scatter_sum(
            torch.ones_like(candidate_logits),
            candidate_batch,
            dim=0,
            dim_size=graph_count,
        )
        valid_graph_mask = candidate_counts > 0.0
        self._profile_inc("sample_candidate_graphs", float(valid_graph_mask.sum().detach().cpu()))
        self._profile_inc("sample_candidate_nodes", float(candidate_idx.numel()))

        finite_mask = torch.isfinite(candidate_logits)
        has_finite = scatter_max(
            finite_mask.float(),
            candidate_batch,
            dim=0,
            dim_size=graph_count,
        )[0] > 0.0
        finite_fill = torch.full_like(candidate_logits, float("inf"))
        min_finite = scatter_min(
            torch.where(finite_mask, candidate_logits, finite_fill),
            candidate_batch,
            dim=0,
            dim_size=graph_count,
        )[0]
        fallback_fill = min_finite[candidate_batch] - 20.0
        safe_logits = torch.where(finite_mask, candidate_logits, fallback_fill)
        probs = gnn_softmax(safe_logits, candidate_batch, num_nodes=graph_count)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        uniform_graph_mask = valid_graph_mask & (~has_finite)
        prob_mass = scatter_sum(probs, candidate_batch, dim=0, dim_size=graph_count)
        bad_mass_mask = valid_graph_mask & (~torch.isfinite(prob_mass) | (prob_mass <= 0.0))
        fallback_graph_mask = uniform_graph_mask | bad_mass_mask
        if bool(fallback_graph_mask.any()):
            fallback_mask = fallback_graph_mask[candidate_batch]
            probs[fallback_mask] = 1.0 / candidate_counts[candidate_batch[fallback_mask]].clamp_min(1.0)
            prob_mass = scatter_sum(probs, candidate_batch, dim=0, dim_size=graph_count)
            self._profile_inc("sample_uniform_fallback_graphs", float(fallback_graph_mask.sum().detach().cpu()))

        probs = probs / prob_mass[candidate_batch].clamp_min(1e-12)
        probs_dense[candidate_idx] = probs.to(dtype=probs_dense.dtype)
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -scatter_sum(probs * log_probs, candidate_batch, dim=0, dim_size=graph_count)

        if self.training or not self.greedy_eval:
            rand = torch.rand_like(probs).clamp_(1e-6, 1.0 - 1e-6)
            sample_scores = log_probs + (-torch.log(-torch.log(rand)))
        else:
            sample_scores = probs

        _, selected_pos = scatter_max(sample_scores, candidate_batch, dim=0, dim_size=graph_count)
        selected_graph_tensor = torch.nonzero(valid_graph_mask, as_tuple=True)[0]
        selected_candidate_pos = selected_pos[selected_graph_tensor]
        selected_tensor = candidate_idx[selected_candidate_pos]
        log_prob[selected_graph_tensor] = log_probs[selected_candidate_pos]
        self._profile_add("sampling_s", t_prof, pick_logits)
        return selected_tensor, selected_graph_tensor, log_prob, entropy, probs_dense

    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        h_policy = self._encode_policy_state(state)
        batch = state.get("batch")
        if batch is None:
            batch = torch.zeros(h_policy.size(0), device=h_policy.device, dtype=torch.long)
        batch = batch.view(-1).long()
        nav_state_summary = state.get("nav_state_summary")
        if nav_state_summary is not None:
            graph_count = int(nav_state_summary.size(0))
        else:
            graph_count = int(batch.max().detach().cpu()) + 1 if batch.numel() > 0 else 1

        t_prof = self._profile_tic(h_policy)
        graph_context = scatter_mean(h_policy, batch, dim=0, dim_size=graph_count)
        self._profile_add("graph_pool_s", t_prof, h_policy)
        if nav_state_summary is None:
            nav_state_summary = torch.zeros(graph_count, 6, device=h_policy.device, dtype=torch.float32)
        else:
            nav_state_summary = nav_state_summary.to(device=h_policy.device, dtype=torch.float32)
            if nav_state_summary.size(0) == 1 and graph_count > 1:
                nav_state_summary = nav_state_summary.repeat(graph_count, 1)

        t_prof = self._profile_tic(h_policy)
        critic_input = torch.cat([graph_context, nav_state_summary], dim=-1)
        value = self.critic_mlp(critic_input).view(-1)
        self._profile_add("critic_s", t_prof, h_policy)
        aux_value = None
        critic_privileged = state.get("critic_privileged")
        if self.aux_critic_mlp is not None and critic_privileged is not None:
            t_prof = self._profile_tic(h_policy)
            source_context = self._privileged_source_context(
                h_policy,
                batch,
                graph_count,
                critic_privileged,
            )
            aux_input = torch.cat([graph_context, source_context, nav_state_summary], dim=-1)
            aux_value = self.aux_critic_mlp(aux_input).view(-1)
            self._profile_add("aux_critic_s", t_prof, h_policy)

        valid_mask = state.get("valid_mask")
        t_prof = self._profile_tic(h_policy)
        if valid_mask is None:
            valid_mask = torch.ones(h_policy.size(0), device=h_policy.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.view(-1).bool().to(h_policy.device)
        self._profile_add("valid_mask_prep_s", t_prof, h_policy)

        max_k = int(state.get("k", self.max_picks))
        available_mask = valid_mask.clone()
        selected_indices_all: List[torch.Tensor] = []
        first_pick_logits = None
        first_pick_probs = None
        graph_context_per_node = graph_context[batch]
        nav_summary_per_node = nav_state_summary[batch]
        t_prof = self._profile_tic(h_policy)
        local_bias = self._local_logit_bias(state, batch, graph_count)
        self._profile_add("local_bias_s", t_prof, h_policy)
        action_log_prob = value.new_zeros(graph_count)
        action_entropy = value.new_zeros(graph_count)

        for pick_idx in range(max(max_k, 1)):
            if not bool(available_mask.any()):
                break
            self._profile_inc("pick_iterations")
            self._profile_inc("pick_available_nodes", float(available_mask.sum().detach().cpu()))

            t_prof = self._profile_tic(h_policy)
            pick_fraction = torch.full(
                (h_policy.size(0), 1),
                float(pick_idx) / float(max(max_k, 1)),
                device=h_policy.device,
                dtype=torch.float32,
            )
            already_selected = (~available_mask).float().unsqueeze(-1)
            actor_input = torch.cat(
                [
                    h_policy,
                    graph_context_per_node,
                    nav_summary_per_node,
                    pick_fraction,
                    already_selected,
                ],
                dim=-1,
            )
            self._profile_add("actor_input_materialize_s", t_prof, h_policy)
            t_prof = self._profile_tic(h_policy)
            pick_logits = self.actor_mlp(actor_input).view(-1)
            self._profile_add("actor_head_s", t_prof, h_policy)
            t_prof = self._profile_tic(h_policy)
            pick_logits = pick_logits + local_bias
            masked_logits = pick_logits.masked_fill(~available_mask, -1e9)
            self._profile_add("masking_s", t_prof, h_policy)
            selected_idx, _, pick_log_prob, pick_entropy, pick_probs = self._select_pick(
                masked_logits,
                batch,
                available_mask,
                graph_count,
            )
            if first_pick_logits is None:
                first_pick_logits = masked_logits
            if first_pick_probs is None:
                first_pick_probs = pick_probs
            action_log_prob = action_log_prob + pick_log_prob
            action_entropy = action_entropy + pick_entropy
            if selected_idx.numel() == 0:
                break
            available_mask[selected_idx] = False
            selected_indices_all.append(selected_idx)

        if selected_indices_all:
            selected_indices = torch.cat(selected_indices_all, dim=0)
        else:
            selected_indices = torch.empty(0, device=h_policy.device, dtype=torch.long)

        t_prof = self._profile_tic(value)
        self._profile_add("action_reduce_s", t_prof, value)

        nav_probs = first_pick_probs if first_pick_probs is not None else torch.zeros(h_policy.size(0), device=h_policy.device)
        final_logits = first_pick_logits if first_pick_logits is not None else torch.full(
            (h_policy.size(0),),
            -1e9,
            device=h_policy.device,
            dtype=torch.float32,
        )

        t_prof = self._profile_tic(value)
        mean_entropy = (
            float(action_entropy.detach().mean().cpu())
            if action_entropy.numel() > 0 and (self._profile_enabled or bool(state.get("emit_action_audit_scalars", True)))
            else 0.0
        )
        self._profile_add("action_audit_s", t_prof, value)
        return {
            "logits": final_logits.view(-1, 1),
            "selected_indices": selected_indices,
            "nav_probs": nav_probs,
            "value": value,
            "aux_value": aux_value,
            "action_log_prob": action_log_prob,
            "action_entropy": action_entropy,
            "selected_log_prob": action_log_prob,
            "policy_entropy": action_entropy,
            "action_audit": {
                "selected_count": int(selected_indices.numel()),
                "mean_entropy": mean_entropy,
                "autoregressive_picks": int(len(selected_indices_all)),
                "policy_encoder": "graphsage_fused_state",
                "evidence_controller_mean": float(getattr(self, "_last_evidence_gate_mean", 0.0)),
            },
        }

    def capabilities(self):
        return {
            "supports_soft_actions": False,
            "supports_without_replacement": True,
            "output_fields": [
                "selected_indices",
                "nav_probs",
                "value",
                "aux_value",
                "action_log_prob",
                "action_entropy",
                "action_audit",
            ],
        }
