from typing import Any, Dict, Optional

import torch
from torch_scatter import scatter_max, scatter_min, scatter_sum

from src.modeling.evidence.builder import EvidenceBuilder
from src.modeling.state.schema import ObservationState, PhysicsContext


def _cfg_get(cfg_obj, key: str, default):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


class OracleLabelBuilder:
    """
    Minimal oracle-supervised evidence label builder.

    Support teacher:
    - Reuses the current support semantics on an oracleized observation snapshot.

    Rebuttal teacher:
    - Reuses the current legacy contradiction kernel on the same oracleized snapshot.
    """

    def __init__(self, cfg=None, evidence_builder: Optional[EvidenceBuilder] = None):
        self.cfg = cfg
        self.evidence_builder = evidence_builder or EvidenceBuilder(cfg)
        physics_cfg = getattr(cfg, "physics", None)
        env_cfg = getattr(physics_cfg, "env", None) if physics_cfg is not None else None
        loss_cfg = getattr(cfg, "loss", None)
        loss_params = _cfg_get(loss_cfg, "params", {}) if loss_cfg is not None else {}
        evidence_oracle_cfg = _cfg_get(loss_params, "evidence_oracle", {})
        self.positive_threshold = float(
            _cfg_get(env_cfg, "sensor_reading_threshold", _cfg_get(physics_cfg, "poison_threshold", 0.1))
        )
        self.eps = 1e-6
        self.rebuttal_density_quantile = float(
            _cfg_get(evidence_oracle_cfg, "rebuttal_density_quantile", 0.75)
        )
        self.rebuttal_density_min_positive = int(
            _cfg_get(evidence_oracle_cfg, "rebuttal_density_min_positive", 8)
        )
        self.rebuttal_tail_quantile = float(
            _cfg_get(evidence_oracle_cfg, "rebuttal_tail_quantile", 0.95)
        )
        self.rebuttal_tail_scale_floor = float(
            _cfg_get(evidence_oracle_cfg, "rebuttal_tail_scale_floor", 1.0)
        )

    def _compress_rebuttal_target(self, values: torch.Tensor) -> torch.Tensor:
        # Keep ordering while removing heavy-tail magnitude that dominates Phase A alignment.
        return torch.log1p(values.detach().float().clamp_min(0.0))

    def _shape_rebuttal_target_density(self, values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if values is None or values.numel() == 0:
            return values

        shaped = values.detach().float().clone()
        batch = batch.view(-1).to(device=shaped.device, dtype=torch.long)
        if batch.numel() != shaped.numel():
            return shaped

        quantile = min(max(self.rebuttal_density_quantile, 0.0), 1.0)
        if quantile <= 0.0:
            return shaped

        for graph_id in torch.unique(batch):
            graph_mask = batch == graph_id
            graph_values = shaped[graph_mask]
            positive_mask = graph_values > self.eps
            positive_values = graph_values[positive_mask]
            if positive_values.numel() < self.rebuttal_density_min_positive:
                continue

            floor = torch.quantile(positive_values, quantile)
            if float(floor.item()) <= self.eps:
                continue

            residual = (graph_values - floor).clamp_min(0.0)
            shaped[graph_mask] = self._calibrate_rebuttal_tail(residual)
        return shaped

    def _calibrate_rebuttal_tail(self, values: torch.Tensor) -> torch.Tensor:
        if values is None or values.numel() == 0:
            return values

        positive_values = values[values > self.eps]
        if positive_values.numel() < self.rebuttal_density_min_positive:
            return values

        quantile = min(max(self.rebuttal_tail_quantile, 0.0), 1.0)
        if quantile <= 0.0:
            return values

        tail_scale = torch.quantile(positive_values, quantile).clamp_min(self.rebuttal_tail_scale_floor)
        if float(tail_scale.item()) <= self.eps:
            return values
        return values / tail_scale

    def _expand_graph_tensor(self, value, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if value is None:
            return None
        try:
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
        except Exception:
            return None
        tensor = tensor.to(device=device, dtype=dtype).view(-1)
        if tensor.numel() == batch_size:
            return tensor
        if tensor.numel() == 1:
            return tensor.repeat(batch_size)
        return None

    def _resolve_snapshot_indices(
        self,
        x_raw: torch.Tensor,
        t_sim: Optional[torch.Tensor],
        view_batch: torch.Tensor,
        trigger_time_step,
        step_seconds,
    ) -> Optional[torch.Tensor]:
        if x_raw is None or x_raw.dim() != 3 or view_batch is None:
            return None
        num_graphs = int(view_batch.max().item()) + 1 if view_batch.numel() > 0 else 1
        device = x_raw.device

        trigger_steps = self._expand_graph_tensor(trigger_time_step, num_graphs, device, torch.long)
        if trigger_steps is None:
            trigger_steps = torch.zeros(num_graphs, device=device, dtype=torch.long)

        step_seconds_tensor = self._expand_graph_tensor(step_seconds, num_graphs, device, torch.float)
        if step_seconds_tensor is None:
            step_seconds_tensor = torch.full((num_graphs,), 900.0, device=device)
        step_seconds_tensor = step_seconds_tensor.clamp_min(1.0)

        if t_sim is None:
            t_sim_tensor = torch.zeros(num_graphs, device=device)
        else:
            t_sim_tensor = self._expand_graph_tensor(t_sim, num_graphs, device, torch.float)
            if t_sim_tensor is None:
                return None

        t_sim_steps = torch.div(t_sim_tensor * 60.0, step_seconds_tensor, rounding_mode="floor").to(torch.long)
        t_idx_nodes = trigger_steps[view_batch.long()] + t_sim_steps[view_batch.long()]
        return t_idx_nodes.clamp_(min=0, max=x_raw.size(1) - 1)

    def _fuse_current_snapshot(
        self,
        observation_state: ObservationState,
        x_raw: torch.Tensor,
        inverse_indices: torch.Tensor,
        t_idx_nodes: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if x_raw is None or inverse_indices is None or t_idx_nodes is None:
            return None

        num_fused = int(observation_state.observed_flag.numel())
        device = observation_state.observed_flag.device
        valid_raw = inverse_indices.view(-1) >= 0
        if not bool(valid_raw.any()):
            return None

        raw_idx = valid_raw.nonzero(as_tuple=True)[0]
        fused_idx = inverse_indices[raw_idx].to(device=device, dtype=torch.long)
        t_idx = t_idx_nodes[raw_idx].to(device=device, dtype=torch.long)

        signal_raw = x_raw[raw_idx, t_idx, 0].to(device=device, dtype=torch.float)
        conc_raw = x_raw[raw_idx, t_idx, 1].to(device=device, dtype=torch.float)
        positive_raw = (conc_raw > self.positive_threshold).float()
        presence_raw = torch.ones_like(signal_raw)

        signal_min, _ = scatter_min(signal_raw, fused_idx, dim=0, dim_size=num_fused)
        presence_fused = scatter_max(presence_raw, fused_idx, dim=0, dim_size=num_fused)[0]
        positive_fused = scatter_max(positive_raw, fused_idx, dim=0, dim_size=num_fused)[0]

        signal_fused = torch.where(
            presence_fused > 0.0,
            signal_min,
            torch.zeros(num_fused, device=device),
        )
        negative_fused = (presence_fused > 0.0).float() * (1.0 - positive_fused)
        return {
            "signal_fused": signal_fused,
            "observed_fused": presence_fused,
            "positive_fused": (positive_fused > 0.5),
            "negative_fused": (negative_fused > 0.5),
        }

    def _build_oracle_snapshot_observation(
        self,
        obs_partial: ObservationState,
        signal_fused: torch.Tensor,
        observed_fused: torch.Tensor,
        positive_fused: torch.Tensor,
        negative_fused: torch.Tensor,
    ) -> ObservationState:
        observed_float = observed_fused.float()
        positive_float = positive_fused.float()
        negative_float = negative_fused.float()
        oracle_observed = torch.maximum(obs_partial.observed_flag.float().view(-1), observed_float)
        oracle_positive = torch.maximum(obs_partial.toxic_positive_flag.float().view(-1), positive_float)
        oracle_negative = torch.maximum(obs_partial.toxic_negative_flag.float().view(-1), negative_float)
        oracle_chlorine = obs_partial.chlorine_deviation.float().view(-1).clone()
        oracle_chlorine[observed_fused > 0.5] = signal_fused[observed_fused > 0.5]
        oracle_freshness = torch.maximum(obs_partial.freshness.float().view(-1), observed_float)
        oracle_anchor = None if obs_partial.anchor is None else obs_partial.anchor.float().view(-1).clone()
        return ObservationState(
            observed_flag=oracle_observed,
            chlorine_deviation=oracle_chlorine,
            toxic_positive_flag=oracle_positive,
            toxic_negative_flag=oracle_negative,
            freshness=oracle_freshness,
            anchor=oracle_anchor,
        )

    def _compute_reachability(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        t_sim: Optional[torch.Tensor],
    ):
        if physics_context.batch is not None:
            batch = physics_context.batch
        else:
            batch = torch.zeros(
                observation_state.observed_flag.size(0),
                dtype=torch.long,
                device=observation_state.observed_flag.device,
            )
        if physics_context.stt_dynamic is not None:
            return self.evidence_builder.dynamic_reachability.compute_reachability_bundle(
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
        ), None

    def build(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        t_sim: Optional[torch.Tensor],
        inverse_indices: torch.Tensor,
        x_raw: Optional[torch.Tensor],
        view_batch: Optional[torch.Tensor],
        trigger_time_step,
        step_seconds,
    ) -> Optional[Dict[str, Any]]:
        if x_raw is None or view_batch is None or inverse_indices is None:
            return None

        t_idx_nodes = self._resolve_snapshot_indices(
            x_raw=x_raw,
            t_sim=t_sim,
            view_batch=view_batch,
            trigger_time_step=trigger_time_step,
            step_seconds=step_seconds,
        )
        if t_idx_nodes is None:
            return None

        snapshot = self._fuse_current_snapshot(
            observation_state=observation_state,
            x_raw=x_raw,
            inverse_indices=inverse_indices,
            t_idx_nodes=t_idx_nodes,
        )
        if snapshot is None:
            return None

        oracle_obs = self._build_oracle_snapshot_observation(
            obs_partial=observation_state,
            signal_fused=snapshot["signal_fused"],
            observed_fused=snapshot["observed_fused"],
            positive_fused=snapshot["positive_fused"],
            negative_fused=snapshot["negative_fused"],
        )
        reach_res, support_distance_matrix = self._compute_reachability(oracle_obs, physics_context, t_sim)
        support_res = self.evidence_builder.compute_support_score(
            oracle_obs,
            physics_context,
            None,
            reach_res,
            t_sim,
            precomputed_distance_matrix=support_distance_matrix,
        )
        rebuttal_res = self.evidence_builder.compute_contradiction_score(
            oracle_obs,
            physics_context,
            None,
            reach_res,
            t_sim,
        )
        support_target = support_res["total"].detach()
        rebuttal_target = self._compress_rebuttal_target(rebuttal_res["total"])

        batch = physics_context.batch
        if batch is None:
            batch = torch.zeros(support_target.size(0), device=support_target.device, dtype=torch.long)
        batch = batch.view(-1).to(device=support_target.device, dtype=torch.long)
        rebuttal_target = self._shape_rebuttal_target_density(rebuttal_target, batch)

        support_graph_mass = scatter_sum(support_target.abs(), batch, dim=0)
        rebuttal_nonzero = (rebuttal_target > self.eps).float()
        positive_ratio = snapshot["positive_fused"].float().mean()
        negative_ratio = snapshot["negative_fused"].float().mean()

        stats = {
            "support_nonzero_ratio": float((support_target.abs() > self.eps).float().mean().item()),
            "support_graph_active_ratio": float((support_graph_mass > self.eps).float().mean().item()),
            "support_max": float(support_target.max().item()) if support_target.numel() > 0 else 0.0,
            "rebuttal_nonzero_ratio": float(rebuttal_nonzero.mean().item()) if rebuttal_nonzero.numel() > 0 else 0.0,
            "rebuttal_max": float(rebuttal_target.max().item()) if rebuttal_target.numel() > 0 else 0.0,
            "oracle_positive_ratio": float(positive_ratio.item()),
            "oracle_negative_ratio": float(negative_ratio.item()),
        }

        return {
            "support_target": support_target,
            "rebuttal_target": rebuttal_target,
            "oracle_observed_mask": snapshot["observed_fused"].detach().float(),
            "oracle_positive_mask": snapshot["positive_fused"].detach().float(),
            "oracle_negative_mask": snapshot["negative_fused"].detach().float(),
            "stats": stats,
        }
