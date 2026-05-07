import copy
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from src.modeling.clean_aligned_features import (
    GRAPH_FEATURE_DIM,
    NODE_FEATURE_DIM,
    build_clean_aligned_feature_payload,
)
from src.modeling.interfaces.base import NavigatorBase, NavigatorCapabilities
from src.modeling.navigators.clean_v1 import (
    CleanNavigatorV1,
    bound_nonnegative_score,
    derive_two_channel_features,
)
from src.modeling.registry import NAVIGATOR_REGISTRY, NAV_BACKBONE_REGISTRY


def _cfg_get(cfg_obj: Any, key: str, default: Any) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _state_attr(state_obj: Any, key: str, reference: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if state_obj is None:
        return torch.full_like(reference.view(-1).float(), float(default))
    if isinstance(state_obj, dict):
        value = state_obj.get(key)
    else:
        value = getattr(state_obj, key, None)
    if value is None:
        return torch.full_like(reference.view(-1).float(), float(default))
    return value.view(-1).to(device=reference.device, dtype=torch.float32)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_bridge_role_prior(
    role_potentials: Optional[torch.Tensor],
    *,
    coupling_active: bool,
    mode: str,
    weight: float,
    step_index: int,
    max_episode: int,
) -> Optional[torch.Tensor]:
    if role_potentials is None or float(weight) <= 0.0:
        return role_potentials
    if mode == "frontier_slot0_bias":
        should_apply = bool(coupling_active)
    elif mode == "early_frontier_slot0_bias":
        should_apply = int(step_index) <= int(max_episode)
    else:
        should_apply = False
    if not should_apply:
        return role_potentials
    if role_potentials.dim() != 2 or role_potentials.size(1) < 2:
        return role_potentials
    adjusted = role_potentials.clone()
    adjusted[:, 0] = adjusted[:, 0] + float(weight) * adjusted[:, 1]
    return adjusted


@NAVIGATOR_REGISTRY.register("frozen_clean_v1_bridge")
class FrozenCleanNavigatorBridge(NavigatorBase):
    """
    Adapter that lets the Phase45 closed loop consume a frozen standalone
    CleanNavigatorV1 checkpoint without changing the reasoner lane.

    This bridge intentionally keeps the reasoner architecture untouched and only
    translates the active clean-environment state contract into the minimal
    CleanNavigatorV1 feature contract needed for frozen sampling.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nav_cfg = getattr(cfg.model, "navigator", {})

        control_summary_path = _cfg_get(nav_cfg, "bridge_control_summary_path", None)
        control_dir = _cfg_get(nav_cfg, "bridge_control_dir", None)
        if control_summary_path is None and control_dir is not None:
            control_summary_path = str(Path(control_dir) / "summary.json")
        if control_summary_path is None:
            raise ValueError(
                "FrozenCleanNavigatorBridge requires model.navigator.bridge_control_summary_path "
                "or model.navigator.bridge_control_dir."
            )

        control_summary = _load_json(Path(control_summary_path))
        control_args = dict(control_summary.get("args", {}))
        if not control_args:
            raise ValueError(f"Control summary {control_summary_path} does not contain args.")
        self.control_args = control_args

        self.force_deterministic_train = bool(
            _cfg_get(nav_cfg, "bridge_deterministic_train", True)
        )
        self.force_deterministic_eval = bool(
            _cfg_get(nav_cfg, "bridge_deterministic_eval", True)
        )
        self.freeze_clean_navigator = bool(
            _cfg_get(nav_cfg, "bridge_freeze_clean_navigator", True)
        )
        self.training_sampling_mode = str(
            _cfg_get(
                nav_cfg,
                "bridge_training_sampling_mode",
                control_args.get("training_sampling_mode", "explicit_cpu_multinomial"),
            )
        )
        self.trainable_clean_navigator = bool(
            _cfg_get(nav_cfg, "bridge_trainable_clean_navigator", False)
        )
        if self.trainable_clean_navigator:
            self.freeze_clean_navigator = False
        self.freeze_bridge_backbone = bool(
            _cfg_get(nav_cfg, "bridge_freeze_backbone", False)
        )
        self.coupling_horizon_k = max(0, int(_cfg_get(nav_cfg, "bridge_coupling_horizon_k", 0)))
        self.coupling_train_sampling_mode = str(
            _cfg_get(nav_cfg, "bridge_coupling_train_sampling_mode", "gumbel_max")
        )
        self.reference_after_coupling = bool(
            _cfg_get(nav_cfg, "bridge_reference_after_coupling", self.coupling_horizon_k > 0)
        )
        self.frontier_mode = str(
            _cfg_get(nav_cfg, "bridge_frontier_mode", "unresolved_without_pair")
        )
        self.heuristic_prior_mode = str(
            _cfg_get(nav_cfg, "bridge_heuristic_prior_mode", "none")
        )
        self.heuristic_prior_weight = float(
            _cfg_get(nav_cfg, "bridge_heuristic_prior_weight", 0.0)
        )
        self.heuristic_prior_max_episode = int(
            _cfg_get(nav_cfg, "bridge_heuristic_prior_max_episode", 3)
        )

        backbone_type = str(_cfg_get(nav_cfg, "backbone_type", "sage_backbone"))
        backbone_cls = NAV_BACKBONE_REGISTRY.get(backbone_type)
        self.backbone = backbone_cls(cfg)

        self.clean_navigator = CleanNavigatorV1(
            node_feature_dim=NODE_FEATURE_DIM,
            graph_feature_dim=GRAPH_FEATURE_DIM,
            hidden_dim=int(control_args["hidden_dim"]),
            num_layers=int(control_args["num_layers"]),
            num_slots=int(control_args["action_budget"]),
            greedy_eval=True,
            role_mode=str(control_args.get("role_mode", "none")),
            role_bias_weight=float(control_args.get("role_bias_weight", 0.0)),
            diversity_mode=str(control_args.get("diversity_mode", "none")),
            diversity_penalty_weight=float(control_args.get("diversity_penalty_weight", 0.0)),
            complementarity_mode=str(control_args.get("complementarity_mode", "none")),
            complementarity_penalty_weight=float(control_args.get("complementarity_penalty_weight", 0.0)),
            credit_mode=str(control_args.get("credit_mode", "state_value")),
        )

        load_path = _cfg_get(nav_cfg, "bridge_standalone_checkpoint_path", None)
        if load_path:
            self.load_frozen_checkpoint(load_path)

        if self.freeze_clean_navigator:
            for param in self.clean_navigator.parameters():
                param.requires_grad = False
        if self.freeze_bridge_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.reference_navigator: Optional[CleanNavigatorV1] = None

    def load_frozen_checkpoint(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        self.clean_navigator.load_state_dict(state, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clean_navigator:
            # Keep the frozen policy path in eval mode to avoid any train/eval drift.
            self.clean_navigator.eval()
        else:
            self.clean_navigator.train(mode)
        if self.reference_navigator is not None:
            self.reference_navigator.eval()
        return self

    def _ensure_reference_navigator(self) -> CleanNavigatorV1:
        if self.reference_navigator is None:
            self.reference_navigator = copy.deepcopy(self.clean_navigator)
            for param in self.reference_navigator.parameters():
                param.requires_grad = False
            self.reference_navigator.eval()
        return self.reference_navigator

    def capabilities(self) -> NavigatorCapabilities:
        return {
            "supports_without_replacement": True,
            "output_fields": [
                "logits",
                "selected_indices",
                "selected_log_prob",
                "policy_entropy",
                "value",
                "aux_value",
                "y_action",
            ],
        }

    def _degree_norm(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        if num_nodes <= 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, device=device, dtype=torch.float32)
        src, dst = edge_index
        degree = torch.bincount(src, minlength=num_nodes).float() + torch.bincount(dst, minlength=num_nodes).float()
        max_degree = degree.max().clamp_min(1.0)
        return (degree / max_degree).to(device=device, dtype=torch.float32)

    def _build_node_features(
        self,
        nav_state: Dict[str, Any],
        graph_nodes: torch.Tensor,
        local_edge_index: torch.Tensor,
        physics_ctx: Optional[Dict[str, Any]],
    ):
        valid_mask = nav_state["valid_mask"].view(-1)[graph_nodes].float()
        evidence_state = nav_state.get("evidence_state")
        obs_state = nav_state.get("observation_state")
        constraint_state = nav_state.get("constraint_state")

        support_score = _state_attr(evidence_state, "support_score", valid_mask, default=0.0)
        contradiction_score = _state_attr(evidence_state, "contradiction_score", valid_mask, default=0.0)
        support_score = support_score[graph_nodes]
        contradiction_score = contradiction_score[graph_nodes]
        derived = derive_two_channel_features(support_score, contradiction_score)

        support_focus = _state_attr(evidence_state, "support_focus_term", valid_mask, default=0.0)[graph_nodes]
        support_timing = _state_attr(evidence_state, "support_timing_term", valid_mask, default=0.0)[graph_nodes]
        support_coverage = _state_attr(evidence_state, "support_coverage_term", valid_mask, default=0.0)[graph_nodes]
        contradiction_soft = _state_attr(evidence_state, "contradiction_toxic_term", valid_mask, default=0.0)[graph_nodes]
        contradiction_hard = _state_attr(evidence_state, "contradiction_clean_term", valid_mask, default=0.0)[graph_nodes]
        arrival_gate = _state_attr(evidence_state, "arrival_gate", valid_mask, default=0.0)[graph_nodes].clamp(0.0, 1.0)
        if physics_ctx is not None and isinstance(physics_ctx.get("feasible_mask"), torch.Tensor):
            feasible_mask = physics_ctx["feasible_mask"].view(-1)[graph_nodes].float()
        else:
            feasible_mask = torch.ones_like(valid_mask)

        if constraint_state is None:
            sampled_mask = torch.zeros_like(valid_mask)
            no_resample_mask = torch.zeros_like(valid_mask)
        else:
            sampled_mask = constraint_state.sampled_mask.view(-1)[graph_nodes].float()
            no_resample_mask = constraint_state.no_resample_mask.view(-1)[graph_nodes].float()

        positive_anchor_potential = bound_nonnegative_score(support_focus + support_timing)
        safe_pair_potential = bound_nonnegative_score(contradiction_score + contradiction_hard)
        positive_reachability = bound_nonnegative_score(support_coverage)
        safe_reachability = arrival_gate
        positive_distance_summary = bound_nonnegative_score(support_timing)
        safe_distance_summary = bound_nonnegative_score(contradiction_soft + contradiction_hard)
        pair_available = arrival_gate
        eligible_safe_witness_count_bounded = bound_nonnegative_score(contradiction_soft)
        top_pair_margin_bounded = bound_nonnegative_score(contradiction_hard)
        degree_norm = self._degree_norm(local_edge_index, int(graph_nodes.numel()), graph_nodes.device)

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
                degree_norm,
                valid_mask.float(),
            ],
            dim=1,
        )
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.frontier_mode == "conflict_mass":
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
        )
        role_potentials = torch.nan_to_num(role_potentials, nan=0.0, posinf=0.0, neginf=0.0)

        positive_count = torch.zeros((), device=graph_nodes.device)
        safe_count = torch.zeros((), device=graph_nodes.device)
        if obs_state is not None:
            positive_count = obs_state.toxic_positive_flag.view(-1)[graph_nodes].float().sum()
            safe_count = obs_state.toxic_negative_flag.view(-1)[graph_nodes].float().sum()

        return {
            "node_features": node_features,
            "role_potentials": role_potentials,
            "valid_mask": valid_mask.bool(),
            "positive_count": positive_count,
            "safe_count": safe_count,
        }

    def _build_graph_features(
        self,
        nav_state: Dict[str, Any],
        graph_idx: int,
        graph_payload: Dict[str, Any],
    ) -> torch.Tensor:
        summary = nav_state.get("nav_state_summary")
        if summary is not None:
            summary_row = summary[int(graph_idx)].view(-1).float()
            budget_norm = summary_row[4] if summary_row.numel() > 4 else torch.tensor(0.0, device=summary.device)
            step_norm = summary_row[5] if summary_row.numel() > 5 else torch.tensor(0.0, device=summary.device)
        else:
            device = graph_payload["node_features"].device
            budget_norm = torch.tensor(0.0, device=device)
            step_norm = torch.tensor(0.0, device=device)

        num_nodes = max(int(graph_payload["node_features"].size(0)), 1)
        candidate_fraction = graph_payload["valid_mask"].float().mean()
        return torch.stack(
            [
                step_norm,
                budget_norm,
                graph_payload["positive_count"] / float(num_nodes),
                graph_payload["safe_count"] / float(num_nodes),
                candidate_fraction,
                step_norm,
            ],
            dim=0,
        ).float().nan_to_num(0.0, 0.0, 0.0)

    def _build_global_node_payload(
        self,
        nav_state: Dict[str, Any],
        batch_index: torch.Tensor,
        edge_index: torch.Tensor,
        physics_ctx: Optional[Dict[str, Any]],
        graph_count: int,
    ) -> Dict[str, torch.Tensor]:
        reference = nav_state["valid_mask"].view(-1).float()
        evidence_state = nav_state.get("evidence_state")
        obs_state = nav_state.get("observation_state")
        constraint_state = nav_state.get("constraint_state")

        support_score = _state_attr(evidence_state, "support_score", reference, default=0.0)
        contradiction_score = _state_attr(evidence_state, "contradiction_score", reference, default=0.0)
        derived = derive_two_channel_features(support_score, contradiction_score)

        support_focus = _state_attr(evidence_state, "support_focus_term", reference, default=0.0)
        support_timing = _state_attr(evidence_state, "support_timing_term", reference, default=0.0)
        support_coverage = _state_attr(evidence_state, "support_coverage_term", reference, default=0.0)
        contradiction_soft = _state_attr(evidence_state, "contradiction_toxic_term", reference, default=0.0)
        contradiction_hard = _state_attr(evidence_state, "contradiction_clean_term", reference, default=0.0)
        arrival_gate = _state_attr(evidence_state, "arrival_gate", reference, default=0.0).clamp(0.0, 1.0)

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

        if self.frontier_mode == "conflict_mass":
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

        return {
            "node_features": node_features,
            "role_potentials": role_potentials,
            "valid_mask": reference.bool(),
            "positive_count_by_graph": positive_count_by_graph.float(),
            "safe_count_by_graph": safe_count_by_graph.float(),
        }

    def forward(self, state: Dict[str, Any], graph: Any, physics_ctx: Dict[str, Any] = None) -> Dict[str, Any]:
        batch_index = state["batch"].view(-1).long()
        edge_index = state["fused_edge_index"].view(2, -1).long()
        num_nodes = int(batch_index.numel())
        graph_count = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        device = batch_index.device
        step_index = int(state.get("episode_index", state.get("current_step", 0)) or 0)
        coupling_active = bool(
            self.trainable_clean_navigator
            and self.coupling_horizon_k > 0
            and step_index < self.coupling_horizon_k
        )
        policy_nav = self.clean_navigator
        if not coupling_active and self.reference_after_coupling:
            policy_nav = self._ensure_reference_navigator()

        global_logits = torch.full((num_nodes,), -1e9, device=device, dtype=torch.float32)
        global_probs = torch.zeros((num_nodes,), device=device, dtype=torch.float32)
        action_mask = torch.zeros((num_nodes, 1), device=device, dtype=torch.float32)
        selected_global: list[torch.Tensor] = []
        log_prob = torch.zeros(graph_count, device=device, dtype=torch.float32)
        entropy = torch.zeros(graph_count, device=device, dtype=torch.float32)
        value = torch.zeros(graph_count, device=device, dtype=torch.float32)
        aux_value = torch.zeros(graph_count, device=device, dtype=torch.float32)

        deterministic = self.force_deterministic_train if self.training else self.force_deterministic_eval
        if coupling_active:
            deterministic = not self.training
        batch_is_contiguous = bool(
            batch_index.numel() == 0 or torch.all(batch_index[1:] >= batch_index[:-1]).item()
        )
        graph_ptr = None
        edge_graph_index = None
        filtered_edge_index = edge_index
        if batch_is_contiguous and graph_count > 0:
            graph_sizes = torch.bincount(batch_index, minlength=graph_count)
            graph_ptr = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    graph_sizes.cumsum(dim=0),
                ],
                dim=0,
            )
            edge_graph_index = batch_index[edge_index[0]]
            same_graph_edge = edge_graph_index == batch_index[edge_index[1]]
            if not bool(same_graph_edge.all()):
                filtered_edge_index = edge_index[:, same_graph_edge]
                edge_graph_index = edge_graph_index[same_graph_edge]

        graph_payloads: list[Optional[Dict[str, Any]]] = [None] * graph_count
        encoded_h = None
        global_payload = build_clean_aligned_feature_payload(
            state,
            batch_index=batch_index,
            edge_index=filtered_edge_index,
            physics_ctx=physics_ctx,
            frontier_mode=self.frontier_mode,
        )
        global_node_features = global_payload["node_features"].float()
        global_role_potentials = global_payload["role_potentials"].float()
        global_valid_mask = global_payload["valid_mask"].bool()
        graph_features_by_graph = global_payload["graph_features_by_graph"].float()
        global_role_potentials = apply_bridge_role_prior(
            global_role_potentials,
            coupling_active=coupling_active,
            mode=self.heuristic_prior_mode,
            weight=self.heuristic_prior_weight,
            step_index=step_index,
            max_episode=self.heuristic_prior_max_episode,
        )

        positive_count_by_graph = global_payload["positive_count_by_graph"].float()
        safe_count_by_graph = global_payload["safe_count_by_graph"].float()
        grad_context = torch.inference_mode if (self.freeze_clean_navigator and not coupling_active) else nullcontext

        for graph_idx in range(graph_count):
            if graph_ptr is not None:
                start = int(graph_ptr[graph_idx].item())
                end = int(graph_ptr[graph_idx + 1].item())
                node_indices = torch.arange(start, end, device=device, dtype=torch.long)
                if edge_graph_index is not None:
                    local_edge_index = filtered_edge_index[:, edge_graph_index == graph_idx] - start
                else:
                    local_edge_index = filtered_edge_index.new_zeros((2, 0))
            else:
                node_mask = batch_index == int(graph_idx)
                node_indices = torch.nonzero(node_mask, as_tuple=True)[0]
                if node_indices.numel() > 0:
                    global_to_local = torch.full(
                        (batch_index.numel(),),
                        -1,
                        dtype=torch.long,
                        device=device,
                    )
                    global_to_local[node_indices] = torch.arange(node_indices.numel(), device=device)
                    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                    local_edge_index = global_to_local[edge_index[:, edge_mask]]
                    if local_edge_index.numel() > 0:
                        valid_edge = (local_edge_index >= 0).all(dim=0)
                        valid_edge = valid_edge & (local_edge_index[0] < node_indices.numel()) & (local_edge_index[1] < node_indices.numel())
                        local_edge_index = local_edge_index[:, valid_edge]
                else:
                    local_edge_index = edge_index.new_zeros((2, 0))
            if node_indices.numel() == 0:
                continue

            graph_payload = {
                "node_indices": node_indices,
                "node_features": global_node_features[node_indices],
                "role_potentials": global_role_potentials[node_indices],
                "valid_mask": global_valid_mask[node_indices],
                "graph_features": graph_features_by_graph[graph_idx],
            }
            graph_payloads[graph_idx] = graph_payload

        if num_nodes > 0:
            with grad_context():
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    encoded_h = policy_nav.encode(global_node_features.float(), filtered_edge_index.long())

        fast_policy_path = (
            deterministic
            and str(policy_nav.diversity_mode) == "none"
            and float(policy_nav.diversity_penalty_weight) <= 0.0
            and str(policy_nav.complementarity_mode) == "none"
            and float(policy_nav.complementarity_penalty_weight) <= 0.0
            and str(policy_nav.credit_mode) == "state_value"
            and encoded_h is not None
            and not coupling_active
        )
        if fast_policy_path and graph_count > 0:
            graph_contexts = scatter_mean(encoded_h, batch_index, dim=0, dim_size=graph_count).float()
            graph_has_valid = scatter_max(global_valid_mask.float(), batch_index, dim=0, dim_size=graph_count)[0] > 0.5
            value_inputs = torch.cat([graph_contexts, graph_features_by_graph], dim=1)
            value.copy_(policy_nav.value_head(value_inputs).view(-1).float())
            aux_value.copy_(value)

            graph_ctx_expand = graph_contexts[batch_index]
            graph_feat_expand = graph_features_by_graph[batch_index]
            available = global_valid_mask.clone()
            selected_per_graph: list[list[int]] = [[] for _ in range(graph_count)]
            first_slot_logits = None
            with grad_context():
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    for slot_idx in range(policy_nav.num_slots):
                        if not bool(available.any()):
                            break
                        slot_embed = policy_nav.slot_embeddings[slot_idx].unsqueeze(0).expand(num_nodes, -1)
                        slot_input = torch.cat([encoded_h, graph_ctx_expand, graph_feat_expand, slot_embed], dim=-1)
                        logits = policy_nav.slot_heads[slot_idx](slot_input)
                        if (
                            str(policy_nav.role_mode) == "slot_bias"
                            and float(policy_nav.role_bias_weight) > 0.0
                            and slot_idx < int(global_role_potentials.size(1))
                        ):
                            logits = logits + float(policy_nav.role_bias_weight) * global_role_potentials[:, slot_idx]
                        masked_logits = logits.masked_fill(~available, -1e9)
                        if slot_idx == 0:
                            first_slot_logits = logits.masked_fill(~global_valid_mask, -1e9)
                        best_value, best_index = scatter_max(masked_logits, batch_index, dim=0, dim_size=graph_count)
                        chosen_graphs = torch.nonzero(graph_has_valid & (best_value > -1e8), as_tuple=True)[0]
                        if chosen_graphs.numel() == 0:
                            break
                        chosen_nodes = best_index[chosen_graphs]
                        available[chosen_nodes] = False
                        for graph_id, node_id in zip(chosen_graphs.tolist(), chosen_nodes.tolist()):
                            selected_per_graph[int(graph_id)].append(int(node_id))

            if first_slot_logits is not None:
                safe_first_slot_logits = torch.nan_to_num(first_slot_logits, nan=-1e9, posinf=0.0, neginf=-1e9)
                global_logits.copy_(safe_first_slot_logits)

            selected_global = []
            for graph_idx, selected_list in enumerate(selected_per_graph):
                if not selected_list:
                    continue
                selected_nodes = torch.tensor(selected_list, dtype=torch.long, device=device)
                selected_global.append(selected_nodes)
                action_mask[selected_nodes] = 1.0

            selected_indices = (
                torch.cat(selected_global, dim=0)
                if selected_global
                else torch.empty(0, dtype=torch.long, device=device)
            )
            return {
                "logits": global_logits.view(-1, 1),
                "nav_probs": global_probs.view(-1, 1),
                "selected_indices": selected_indices,
                "selected_log_prob": log_prob,
                "policy_entropy": entropy,
                "value": value,
                "aux_value": aux_value,
                "y_action": action_mask,
                "coupling_active": False,
            }

        log_prob_rows = [torch.zeros((), device=device, dtype=torch.float32) for _ in range(graph_count)]
        entropy_rows = [torch.zeros((), device=device, dtype=torch.float32) for _ in range(graph_count)]
        value_rows = [torch.zeros((), device=device, dtype=torch.float32) for _ in range(graph_count)]
        aux_value_rows = [torch.zeros((), device=device, dtype=torch.float32) for _ in range(graph_count)]

        for graph_idx in range(graph_count):
            graph_payload = graph_payloads[graph_idx]
            if graph_payload is None:
                continue
            node_indices = graph_payload["node_indices"]
            local_valid = graph_payload["valid_mask"]
            if not bool(local_valid.any()):
                continue

            graph_features = graph_payload["graph_features"]
            local_h = encoded_h[node_indices]
            graph_context = local_h.mean(dim=0)
            graph_ctx_expand = graph_context.unsqueeze(0).expand(local_h.size(0), -1)
            graph_feat_expand = graph_features.unsqueeze(0).expand(local_h.size(0), -1)
            value_pred = policy_nav.value_head(torch.cat([graph_context, graph_features], dim=0)).view(())
            policy_ctx = {
                "node_features": graph_payload["node_features"].float(),
                "valid_mask": local_valid.bool(),
                "graph_features": graph_features.float(),
                "h": local_h,
                "graph_context": graph_context,
                "value": value_pred,
                "graph_ctx_expand": graph_ctx_expand,
                "graph_feat_expand": graph_feat_expand,
                "normalized_redundancy": None,
                "role_aware_overlap_features": None,
                "role_potentials": graph_payload["role_potentials"].float(),
            }
            with grad_context():
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    policy_out = policy_nav._run_policy_from_context(
                        policy_ctx=policy_ctx,
                        deterministic=deterministic,
                        generator=None,
                        training_sampling_mode=(
                            self.coupling_train_sampling_mode
                            if coupling_active and self.training
                            else self.training_sampling_mode
                        ),
                        compute_sampling_stats=(not deterministic) or coupling_active,
                        slot_logits_limit=1,
                    )

            slot_logits = policy_out.get("slot_logits") or []
            if slot_logits:
                first_slot_logits = torch.nan_to_num(slot_logits[0].to(device=device, dtype=torch.float32), nan=-1e9, posinf=0.0, neginf=-1e9)
                global_logits[node_indices] = first_slot_logits
                global_probs[node_indices] = torch.softmax(first_slot_logits.masked_fill(~local_valid, -1e9), dim=0)
            else:
                fallback_logits = graph_payload["node_features"][:, 0].float()
                global_logits[node_indices] = fallback_logits

            local_selected = policy_out.get("selected_indices", torch.empty(0, dtype=torch.long, device=device)).view(-1).long()
            if local_selected.numel() > 0:
                selected_nodes = node_indices[local_selected]
                selected_global.append(selected_nodes)
                action_mask[selected_nodes] = 1.0

            value_item = policy_out.get("value", torch.tensor(0.0, device=device)).view(()).float()
            log_prob_rows[graph_idx] = policy_out.get("log_prob", torch.tensor(0.0, device=device)).view(()).float()
            entropy_rows[graph_idx] = policy_out.get("entropy", torch.tensor(0.0, device=device)).view(()).float()
            value_rows[graph_idx] = value_item
            aux_value_rows[graph_idx] = policy_out.get("set_value", value_item).view(()).float()

        if graph_count > 0:
            log_prob = torch.stack(log_prob_rows)
            entropy = torch.stack(entropy_rows)
            value = torch.stack(value_rows)
            aux_value = torch.stack(aux_value_rows)

        selected_indices = (
            torch.cat(selected_global, dim=0)
            if selected_global
            else torch.empty(0, dtype=torch.long, device=device)
        )
        return {
            "logits": global_logits.view(-1, 1),
            "nav_probs": global_probs.view(-1, 1),
            "selected_indices": selected_indices,
            "selected_log_prob": log_prob,
            "policy_entropy": entropy,
            "value": value,
            "aux_value": aux_value,
            "y_action": action_mask,
            "coupling_active": coupling_active,
        }
