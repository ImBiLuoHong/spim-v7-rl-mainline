import copy

import torch
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_min
from torch_geometric.utils import softmax as gnn_softmax
from src.modeling.evidence.oracle_label_builder import OracleLabelBuilder
from src.modeling.loop.orchestration.state_builders import StateBuilder
from src.modeling.loop.orchestration.state_updates import StateUpdater, build_runtime_verdict_payload
from src.modeling.loop.orchestration.module_dispatch import ModuleDispatcher
from src.modeling.loop.orchestration.metrics_flow import MetricsFlow
from src.modeling.loop.audit import SystemAuditor

import time


def _cfg_get(cfg_obj, key: str, default):
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


class TempGraph:
    __slots__ = ("x", "edge_index", "edge_attr", "batch")

    def __init__(self, x, edge_index, edge_attr, batch):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch

class EpisodeRunner:
    """
    Orchestrates the episode loop using specialized components.
    """
    def __init__(self, model):
        self.model = model
        self.cfg = model.cfg
        
        # Components
        self.state_builder = StateBuilder(model)
        self.state_updater = StateUpdater(model)
        self.module_dispatcher = ModuleDispatcher(model)
        self.metrics_flow = MetricsFlow(model.cfg)
        self.auditor = SystemAuditor(model.cfg)
        self.oracle_label_builder = OracleLabelBuilder(
            cfg=model.cfg,
            evidence_builder=self.state_builder.evidence_builder,
        )
        self.runtime_contract_model = None
        loss_cfg = getattr(self.cfg, "loss", None)
        loss_weights = getattr(loss_cfg, "weights", {}) if loss_cfg is not None else {}
        self.reward_path_enabled = float(loss_weights.get("w_hit", 0.0)) > 0.0

    def _evidence_oracle_enabled(self) -> bool:
        loss_cfg = getattr(self.cfg, "loss", None)
        params_cfg = getattr(loss_cfg, "params", {}) if loss_cfg is not None else {}
        evidence_oracle_cfg = _cfg_get(params_cfg, "evidence_oracle", {})
        return bool(_cfg_get(evidence_oracle_cfg, "enabled", False))

    def _maybe_refine_evidence_state(self, evidence_state, observation_state, constraint_state, batch_index=None):
        refiner = getattr(self.model, "evidence_refiner", None)
        if refiner is None or evidence_state is None:
            return evidence_state

        refined = refiner(evidence_state, observation_state, constraint_state, batch_index=batch_index)
        evidence_state.base_support_score = evidence_state.support_score.detach()
        evidence_state.base_suspect_pool = evidence_state.suspect_pool.detach()
        evidence_state.base_contradiction_score = evidence_state.contradiction_score.detach()
        evidence_state.support_score_delta = refined["support_delta"].detach()
        evidence_state.suspect_pool_delta = refined["suspect_delta"].detach()
        evidence_state.contradiction_score_delta = refined["contradiction_delta"].detach()
        evidence_state.suspect_canonical_latent = refined.get("suspect_canonical_latent")
        evidence_state.support_score = refined["support_score"]
        evidence_state.suspect_pool = refined["suspect_pool"]
        evidence_state.contradiction_score = refined["contradiction_score"]
        return evidence_state

    def _sanitize_selected_nodes(self, selected_nodes, num_nodes):
        if selected_nodes is None:
            return torch.tensor([], dtype=torch.long)
        if not isinstance(selected_nodes, torch.Tensor):
            selected_nodes = torch.as_tensor(selected_nodes, dtype=torch.long)
        selected_nodes = selected_nodes.view(-1).long()
        if selected_nodes.numel() == 0:
            return selected_nodes
        valid_mask = (selected_nodes >= 0) & (selected_nodes < int(num_nodes))
        return selected_nodes[valid_mask]

    def _append_selected_nodes(self, step_selected_indices, confirmation_mask, selected_nodes, num_nodes, mask_logits=None):
        selected_nodes = self._sanitize_selected_nodes(selected_nodes, num_nodes)
        if selected_nodes.numel() == 0:
            return selected_nodes
        confirmation_mask[selected_nodes] = 1.0
        if mask_logits is not None:
            mask_logits[selected_nodes] = -float("inf")
        step_selected_indices.append(selected_nodes)
        return selected_nodes

    def _clone_runtime_value(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: self._clone_runtime_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone_runtime_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_runtime_value(item) for item in value)
        if hasattr(value, "__dataclass_fields__"):
            return type(value)(
                **{
                    key: self._clone_runtime_value(getattr(value, key))
                    for key in value.__dataclass_fields__.keys()
                }
            )
        return copy.deepcopy(value)

    def _build_resume_state(
        self,
        *,
        step_index,
        h_fused,
        x_nav_raw,
        t_sim,
        acc_mask_local,
        constraint_state,
        causal_anchors,
        anchor_times,
        hit_before_t,
        graph_success,
        reasoner_memory_state,
        nav_memory_state,
        steps_taken,
    ):
        return {
            "resume_step_index": int(step_index),
            "resume_h_fused": h_fused.detach().clone(),
            "resume_x_nav_raw": x_nav_raw.detach().clone(),
            "t_sim": t_sim.detach().clone(),
            "accumulated_mask": acc_mask_local.detach().clone(),
            "constraint_state": self._clone_runtime_value(constraint_state),
            "causal_anchors": causal_anchors.detach().clone(),
            "anchor_times": anchor_times.detach().clone(),
            "hit_before_t": hit_before_t.detach().clone(),
            "graph_success": graph_success.detach().clone(),
            "reasoner_memory_state": self._clone_runtime_value(reasoner_memory_state),
            "nav_memory_state": self._clone_runtime_value(nav_memory_state),
            "steps_taken": steps_taken.detach().clone(),
        }

    def _route_forced_actions(
        self,
        *,
        forced_indices,
        logits_fused,
        fused_batch,
        curr_batch_size,
        current_action_k,
        valid_mask_final,
    ):
        num_nodes = logits_fused.view(-1).size(0)
        confirmation_mask = torch.zeros_like(logits_fused.view(-1))
        forced_indices = self._sanitize_selected_nodes(forced_indices, num_nodes).to(
            device=logits_fused.device,
            dtype=torch.long,
        )
        if forced_indices.numel() <= 0:
            return confirmation_mask.view_as(logits_fused), None, {
                "action_policy_branch": "forced_empty",
                "nav_out_consumed": False,
                "forced_action_used": True,
                "forced_candidate_count": 0,
                "forced_admitted_count": 0,
            }

        keep_mask = valid_mask_final.view(-1)[forced_indices]
        forced_indices = forced_indices[keep_mask]
        if forced_indices.numel() <= 0:
            return confirmation_mask.view_as(logits_fused), None, {
                "action_policy_branch": "forced_invalid",
                "nav_out_consumed": False,
                "forced_action_used": True,
                "forced_candidate_count": 0,
                "forced_admitted_count": 0,
            }

        counts = torch.zeros(curr_batch_size, dtype=torch.long, device=logits_fused.device)
        kept = []
        for node_idx in forced_indices.tolist():
            graph_idx = int(fused_batch[int(node_idx)].item())
            if counts[graph_idx] >= int(max(current_action_k, 1)):
                continue
            kept.append(int(node_idx))
            counts[graph_idx] += 1
        if not kept:
            return confirmation_mask.view_as(logits_fused), None, {
                "action_policy_branch": "forced_over_budget",
                "nav_out_consumed": False,
                "forced_action_used": True,
                "forced_candidate_count": int(forced_indices.numel()),
                "forced_admitted_count": 0,
            }
        kept_tensor = torch.tensor(kept, device=logits_fused.device, dtype=torch.long)
        confirmation_mask[kept_tensor] = 1.0
        return confirmation_mask.view_as(logits_fused), kept_tensor, {
            "action_policy_branch": "forced_override",
            "nav_out_consumed": True,
            "forced_action_used": True,
            "forced_candidate_count": int(forced_indices.numel()),
            "forced_admitted_count": int(kept_tensor.numel()),
        }

    def _iter_per_graph_topk(self, logits, fused_batch, curr_batch_size, k, valid_mask=None):
        temp_logits = logits.view(-1).clone()
        if valid_mask is not None:
            temp_logits[~valid_mask.view(-1)] = -float("inf")

        idx_per_round = []
        val_per_round = []
        for _ in range(max(int(k), 0)):
            vals, idx = scatter_max(temp_logits, fused_batch, dim=0, dim_size=curr_batch_size)
            invalid = vals <= -1e9
            idx_safe = idx.clone()
            idx_safe[invalid] = -1
            idx_per_round.append(idx_safe)
            val_per_round.append(vals.clone())
            valid_idx = idx[~invalid]
            valid_idx = self._sanitize_selected_nodes(valid_idx, temp_logits.numel())
            if valid_idx.numel() > 0:
                temp_logits[valid_idx] = -float("inf")
        return idx_per_round, val_per_round

    def _nav_guided_guard_mode(self, action_policy):
        if action_policy == "nav_guided_hard_early":
            return "hard_early"
        if action_policy == "nav_guided_harm_guard":
            return "harm_guard"
        if action_policy == "nav_guided_hybrid_guard":
            return "hybrid_guard"
        return None

    def _nav_guided_proxy_mode(self, action_policy):
        if action_policy == "nav_guided_proxy_shrink":
            return "shrink"
        if action_policy == "nav_guided_proxy_source":
            return "source"
        if action_policy == "nav_guided_proxy_exclusion":
            return "exclusion"
        if action_policy == "nav_guided_proxy_hybrid":
            return "hybrid"
        if action_policy == "nav_guided_proxy_source_learned":
            return "source_learned"
        return None

    def _normalize_per_graph(self, tensor, fused_batch, curr_batch_size):
        tensor = tensor.view(-1).float()
        finite_mask = torch.isfinite(tensor)
        safe_tensor = torch.where(finite_mask, tensor, torch.zeros_like(tensor))
        neg_fill = torch.full_like(safe_tensor, -1e9)
        pos_fill = torch.full_like(safe_tensor, 1e9)
        max_vals = scatter_max(
            torch.where(finite_mask, safe_tensor, neg_fill),
            fused_batch,
            dim=0,
            dim_size=curr_batch_size,
        )[0]
        min_vals = scatter_min(
            torch.where(finite_mask, safe_tensor, pos_fill),
            fused_batch,
            dim=0,
            dim_size=curr_batch_size,
        )[0]
        empty_graph = max_vals <= -1e8
        max_vals = torch.where(empty_graph, torch.zeros_like(max_vals), max_vals)
        min_vals = torch.where(empty_graph, torch.zeros_like(min_vals), min_vals)
        denom = (max_vals - min_vals).clamp_min(1e-6)
        norm = ((safe_tensor - min_vals[fused_batch]) / denom[fused_batch]).clamp(0.0, 1.0)
        return torch.where(finite_mask, norm, torch.zeros_like(norm))

    def _build_nav_summary(
        self,
        obs_state,
        evidence_state,
        constraint_state,
        valid_mask,
        fused_batch,
        curr_batch_size,
        steps_taken,
        max_episodes,
        step_index,
    ):
        device = valid_mask.device
        observed_ratio = scatter_mean(
            obs_state.observed_flag.view(-1).float(),
            fused_batch,
            dim=0,
            dim_size=curr_batch_size,
        )
        valid_ratio = scatter_mean(
            valid_mask.view(-1).float(),
            fused_batch,
            dim=0,
            dim_size=curr_batch_size,
        )
        support = self.metrics_flow._metric_tensor(evidence_state, "support_score", valid_mask, default=0.0)
        not_ruled_out = self.metrics_flow._metric_tensor(evidence_state, "not_ruled_out_gate", valid_mask, default=1.0)
        uncertainty = self.metrics_flow._metric_tensor(evidence_state, "uncertainty_gap", valid_mask, default=0.0)
        suspect = self.metrics_flow._metric_tensor(evidence_state, "suspect_pool", valid_mask, default=0.0)
        support_q = support.clamp_min(0.0) * not_ruled_out.clamp(0.0, 1.0)
        masked_support_q = support_q.clone()
        masked_support_q[~valid_mask.view(-1)] = -float("inf")
        top1_vals, top1_idx = scatter_max(masked_support_q, fused_batch, dim=0, dim_size=curr_batch_size)
        second_best = masked_support_q.clone()
        valid_top1 = (
            (top1_vals > -1e9)
            & (top1_idx >= 0)
            & (top1_idx < masked_support_q.numel())
        )
        if bool(valid_top1.any()):
            second_best[top1_idx[valid_top1]] = -float("inf")
        top2_vals = scatter_max(second_best, fused_batch, dim=0, dim_size=curr_batch_size)[0]
        support_margin = torch.where(top2_vals > -1e9, top1_vals - top2_vals, torch.zeros_like(top1_vals))
        core_mass = self.metrics_flow._core_weight_mass_per_graph(
            support_q,
            uncertainty,
            suspect,
            not_ruled_out,
            valid_mask.view(-1),
            fused_batch,
            curr_batch_size,
        ).to(device=device, dtype=torch.float32)
        budget_total = float(max(max_episodes, 1))
        budget_norm = (budget_total - steps_taken) / budget_total
        step_norm = torch.full(
            (curr_batch_size,),
            float(step_index) / float(max(max_episodes, 1)),
            device=device,
            dtype=torch.float32,
        )
        sampled_ratio = scatter_mean(
            constraint_state.sampled_mask.view(-1).float(),
            fused_batch,
            dim=0,
            dim_size=curr_batch_size,
        )
        return torch.stack(
            [
                observed_ratio,
                valid_ratio,
                support_margin.clamp(-10.0, 10.0),
                core_mass,
                budget_norm,
                step_norm,
            ],
            dim=1,
        )

    def _build_critic_privileged(
        self,
        fused_batch,
        fused_global_ids,
        graph_source_global_ids,
        device,
    ):
        if fused_global_ids is None or graph_source_global_ids is None:
            return None
        batch_flat = fused_batch.view(-1).long()
        graph_source_tensor = torch.as_tensor(
            graph_source_global_ids,
            device=device,
            dtype=fused_global_ids.dtype,
        ).view(-1)
        graph_count = int(batch_flat.max().item()) + 1 if batch_flat.numel() > 0 else 0
        if graph_source_tensor.numel() == 1 and graph_count > 1:
            graph_source_tensor = graph_source_tensor.repeat(graph_count)
        if graph_source_tensor.numel() != graph_count:
            return None
        source_per_node = graph_source_tensor[batch_flat]
        true_source_mask = (
            fused_global_ids.view(-1).to(device=device, dtype=source_per_node.dtype)
            == source_per_node
        ).float()
        return {
            "true_source_mask": true_source_mask,
        }

    def _clean_nav_mainline_enabled(self) -> bool:
        reward_cfg = self.metrics_flow._navigator_reward_cfg()
        reward_mode = str(reward_cfg.get("mode", ""))
        return self.metrics_flow._clean_mainline_reward_mode(reward_mode)

    def _aggregate_official_episode_metrics(self, trajectory_data, budget_used):
        if not trajectory_data:
            return {}

        first_bundle = trajectory_data[0].get("nav_reward_bundle") or {}
        last_bundle = trajectory_data[-1].get("nav_reward_bundle") or {}
        graph_count = int(budget_used.numel()) if isinstance(budget_used, torch.Tensor) else 0
        if graph_count <= 0:
            return {}

        def _bundle_tensor(bundle, key, fallback=None):
            value = bundle.get(key, fallback)
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.view(-1).detach().float().cpu()
            return torch.as_tensor(value, dtype=torch.float32).view(-1).cpu()

        core_mass_before = _bundle_tensor(first_bundle, "core_mass_before")
        core_size_before = _bundle_tensor(first_bundle, "core_size_before")
        uncertainty_before = _bundle_tensor(first_bundle, "uncertainty_core_before")

        core_mass_after = _bundle_tensor(last_bundle, "terminal_core_mass_final")
        if core_mass_after is None:
            core_mass_after = _bundle_tensor(last_bundle, "core_mass_after")
        core_size_after = _bundle_tensor(last_bundle, "terminal_core_size_final")
        if core_size_after is None:
            core_size_after = _bundle_tensor(last_bundle, "core_size_after")
        uncertainty_after = _bundle_tensor(last_bundle, "terminal_uncertainty_final")
        if uncertainty_after is None:
            uncertainty_after = _bundle_tensor(last_bundle, "uncertainty_core_after")

        if core_mass_before is None or core_mass_after is None:
            return {}

        closure_success = _bundle_tensor(last_bundle, "terminal_closure_success")
        decisive_closure = _bundle_tensor(last_bundle, "terminal_early_closure")
        terminal_budget_bonus = _bundle_tensor(last_bundle, "terminal_budget_bonus")
        if closure_success is None:
            closure_success = torch.zeros(graph_count, dtype=torch.float32)
        if decisive_closure is None:
            decisive_closure = torch.zeros(graph_count, dtype=torch.float32)
        if terminal_budget_bonus is None:
            terminal_budget_bonus = torch.zeros(graph_count, dtype=torch.float32)

        harmful_acc = torch.zeros(graph_count, dtype=torch.float32)
        focus_acc = torch.zeros(graph_count, dtype=torch.float32)
        waste_acc = torch.zeros(graph_count, dtype=torch.float32)
        empty_acc = torch.zeros(graph_count, dtype=torch.float32)
        count_acc = torch.zeros(graph_count, dtype=torch.float32)
        budget_to_closure = budget_used.detach().float().cpu().clone()

        for step in trajectory_data:
            bundle = step.get("nav_reward_bundle") or {}
            harmful = _bundle_tensor(bundle, "harmful_drift")
            focus = _bundle_tensor(bundle, "focus_core_delta")
            waste = _bundle_tensor(bundle, "wasted_budget_fraction")
            empty = _bundle_tensor(bundle, "diag_empty_selection")
            closure_after = _bundle_tensor(bundle, "closure_reached_after")
            budget_after = _bundle_tensor(bundle, "budget_used_after")
            if harmful is not None:
                harmful_acc += harmful
                count_acc += 1.0
            if focus is not None:
                focus_acc += focus
            if waste is not None:
                waste_acc += waste
            if empty is not None:
                empty_acc += empty
            if closure_after is not None and budget_after is not None:
                new_closure = (closure_after > 0.5) & (budget_to_closure >= budget_used.detach().float().cpu())
                budget_to_closure[new_closure] = budget_after[new_closure]

        denom = count_acc.clamp_min(1.0)
        core_mass_delta = core_mass_before - core_mass_after
        budget_used_cpu = budget_used.detach().float().cpu()
        return {
            "raw_core_mass_before": core_mass_before,
            "raw_core_mass_after": core_mass_after,
            "raw_core_mass_delta": core_mass_delta,
            "raw_core_size_before": torch.zeros_like(core_mass_before) if core_size_before is None else core_size_before,
            "raw_core_size_after": torch.zeros_like(core_mass_after) if core_size_after is None else core_size_after,
            "raw_core_size_delta": (
                torch.zeros_like(core_mass_delta)
                if core_size_before is None or core_size_after is None
                else core_size_before - core_size_after
            ),
            "raw_uncertainty_before": torch.zeros_like(core_mass_before) if uncertainty_before is None else uncertainty_before,
            "raw_uncertainty_after": torch.zeros_like(core_mass_after) if uncertainty_after is None else uncertainty_after,
            "raw_uncertainty_collapse": (
                torch.zeros_like(core_mass_delta)
                if uncertainty_before is None or uncertainty_after is None
                else uncertainty_before - uncertainty_after
            ),
            "raw_closure_success": closure_success,
            "raw_decisive_closure": decisive_closure,
            "raw_budget_to_closure": budget_to_closure,
            "raw_budget_efficiency": core_mass_delta / budget_used_cpu.clamp_min(1.0),
            "raw_evidence_gain_per_sample": core_mass_delta / budget_used_cpu.clamp_min(1.0),
            "raw_harmful_drift": harmful_acc / denom,
            "raw_focus_core_delta": focus_acc / denom,
            "raw_wasted_budget_fraction": waste_acc / denom,
            "raw_empty_selection_fraction": empty_acc / denom,
            "raw_terminal_budget_bonus": terminal_budget_bonus,
        }

    def _build_runtime_proxy_feature_ctx(
        self,
        logits_fused,
        nav_logits,
        evidence_state,
        fused_batch,
        curr_batch_size,
        valid_mask_final,
        step_index=0,
    ):
        logits_flat = logits_fused.view(-1).float()
        valid_flat = valid_mask_final.view(-1).bool()
        masked_logits = logits_flat.clone()
        masked_logits[~valid_flat] = -float("inf")

        top_idx_rounds, top_val_rounds = self._iter_per_graph_topk(
            masked_logits,
            fused_batch,
            curr_batch_size,
            2,
            valid_mask=valid_flat,
        )
        top1_vals = top_val_rounds[0] if top_val_rounds else torch.full(
            (curr_batch_size,),
            -float("inf"),
            device=logits_flat.device,
        )
        top2_vals = top_val_rounds[1] if len(top_val_rounds) > 1 else torch.full_like(top1_vals, -float("inf"))
        top1_margin = torch.where(top2_vals > -1e9, top1_vals - top2_vals, torch.zeros_like(top1_vals))
        fallback_vals = torch.where(top2_vals > -1e9, top2_vals, top1_vals)

        plausible_delta = 1.0
        top1_per_node = top1_vals[fused_batch]
        plausible_floor = top1_per_node - plausible_delta
        plausible_weight = ((logits_flat - plausible_floor) / plausible_delta).clamp(0.0, 1.0)
        plausible_count = scatter_sum(
            ((logits_flat >= plausible_floor) & valid_flat).float(),
            fused_batch,
            dim=0,
            dim_size=curr_batch_size,
        )
        explore_graph_mask = (top1_margin <= 0.75) | (plausible_count >= 6.0)

        def get_ev(name, default):
            if evidence_state is None:
                return default
            value = getattr(evidence_state, name, None)
            if value is None:
                return default
            return value.view(-1).float()

        zeros = torch.zeros_like(logits_flat)
        ones = torch.ones_like(logits_flat)
        uncertainty = get_ev("uncertainty_gap", zeros).clamp(0.0, 1.0)
        not_ruled_out = get_ev("not_ruled_out_gate", ones).clamp(0.0, 1.0)
        support_norm = self._normalize_per_graph(get_ev("support_score", zeros), fused_batch, curr_batch_size)
        contradiction_toxic_norm = self._normalize_per_graph(
            get_ev("contradiction_toxic_term", zeros),
            fused_batch,
            curr_batch_size,
        )
        arrival_gate = get_ev("arrival_gate", zeros).clamp(0.0, 1.0)
        exclusion_aux = 0.5 * contradiction_toxic_norm + 0.5 * arrival_gate

        top1_margin_norm = (top1_margin / (top1_margin + 1.0)).clamp(0.0, 1.0)
        plausible_count_norm = (plausible_count / (plausible_count + 1.0)).clamp(0.0, 1.0)
        reasoner_norm = self._normalize_per_graph(logits_flat, fused_batch, curr_batch_size)
        if nav_logits is not None:
            nav_norm = self._normalize_per_graph(nav_logits.view(-1).float(), fused_batch, curr_batch_size)
        else:
            nav_norm = zeros
        fallback_gap = torch.sigmoid(logits_flat - fallback_vals[fused_batch])
        step_norm = torch.full_like(logits_flat, float(step_index) / 4.0)
        shrink_core = uncertainty * not_ruled_out * plausible_weight
        feature_matrix = torch.stack(
            [
                uncertainty,
                not_ruled_out,
                plausible_weight,
                support_norm,
                contradiction_toxic_norm,
                arrival_gate,
                top1_margin_norm[fused_batch],
                plausible_count_norm[fused_batch],
                reasoner_norm,
                nav_norm,
                fallback_gap,
                step_norm,
            ],
            dim=1,
        )
        return {
            "feature_matrix": feature_matrix,
            "support_norm": support_norm,
            "contradiction_toxic_norm": contradiction_toxic_norm,
            "arrival_gate": arrival_gate,
            "exclusion_aux": exclusion_aux,
            "shrink_core": shrink_core,
            "explore_graph_mask": explore_graph_mask,
            "plausible_count": plausible_count,
            "top1_margin": top1_margin,
            "valid_mask": valid_flat,
        }

    def _build_runtime_proxy_contract(
        self,
        proxy_mode,
        logits_fused,
        nav_logits,
        evidence_state,
        fused_batch,
        curr_batch_size,
        valid_mask_final,
        step_index=0,
    ):
        ctx = self._build_runtime_proxy_feature_ctx(
            logits_fused,
            nav_logits,
            evidence_state,
            fused_batch,
            curr_batch_size,
            valid_mask_final,
            step_index=step_index,
        )
        feature_matrix = ctx["feature_matrix"]
        support_norm = ctx["support_norm"]
        exclusion_aux = ctx["exclusion_aux"]
        shrink_core = ctx["shrink_core"]

        if proxy_mode == "shrink":
            proxy_score = shrink_core
        elif proxy_mode == "source":
            proxy_score = shrink_core * (1.0 + 0.5 * support_norm)
        elif proxy_mode == "exclusion":
            proxy_score = shrink_core * (1.0 + 0.5 * exclusion_aux)
        elif proxy_mode == "source_learned":
            if self.runtime_contract_model is None:
                proxy_score = shrink_core * (1.0 + 0.5 * support_norm)
            else:
                with torch.no_grad():
                    proxy_score = self.runtime_contract_model(feature_matrix).view(-1)
        else:
            proxy_score = shrink_core * (1.0 + 0.25 * support_norm + 0.25 * exclusion_aux)

        proxy_score = proxy_score.masked_fill(~ctx["valid_mask"], -float("inf"))
        ctx["proxy_score"] = proxy_score
        return ctx

    def _route_actions(
        self,
        action_policy,
        logits_fused,
        nav_logits,
        nav_out,
        evidence_state,
        active_mask_t,
        fused_batch,
        curr_batch_size,
        current_action_k,
        valid_mask_final,
        step_index,
    ):
        num_nodes = logits_fused.view(-1).size(0)
        confirmation_mask = torch.zeros_like(logits_fused.view(-1))
        step_selected_indices = []
        guard_mode = self._nav_guided_guard_mode(action_policy)
        proxy_mode = self._nav_guided_proxy_mode(action_policy)
        routing_meta = {
            "action_policy_branch": "greedy",
            "nav_out_consumed": False,
            "guardrail_mode": guard_mode or "none",
            "proxy_contract_mode": proxy_mode or "none",
            "navigator_proposed_count": 0,
            "navigator_admitted_count": 0,
            "navigator_blocked_count": 0,
            "fallback_to_exploitation_count": 0,
            "proxy_eligible_graph_count": 0,
        }

        if action_policy == "nav_only":
            routing_meta["action_policy_branch"] = "nav_only"
            nav_indices_flat = nav_out.get("selected_indices") if nav_out is not None else None
            if nav_indices_flat is not None:
                nav_indices_flat = self._sanitize_selected_nodes(nav_indices_flat, num_nodes)
            else:
                nav_indices_flat = torch.tensor([], dtype=torch.long, device=logits_fused.device)

            routing_meta["navigator_proposed_count"] = int(nav_indices_flat.numel())
            if nav_indices_flat.numel() > 0:
                # Navigator-only rollouts should respect navigator validity masks rather than
                # suppressing proposals via the episode active gate.
                admitted = nav_indices_flat[valid_mask_final[nav_indices_flat]]
                admitted = self._append_selected_nodes(
                    step_selected_indices,
                    confirmation_mask,
                    admitted,
                    num_nodes,
                )
                routing_meta["nav_out_consumed"] = bool(admitted.numel() > 0)
                routing_meta["navigator_admitted_count"] = int(admitted.numel())

            final_step_indices = None
            if step_selected_indices:
                final_step_indices = torch.cat(step_selected_indices, dim=0)
            return confirmation_mask.view_as(logits_fused), final_step_indices, routing_meta

        if action_policy == "nav_guided" or guard_mode is not None or proxy_mode is not None:
            routing_meta["action_policy_branch"] = action_policy
            greedy_idx_rounds, greedy_val_rounds = self._iter_per_graph_topk(
                logits_fused.view(-1),
                fused_batch,
                curr_batch_size,
                max(int(current_action_k), 1),
                valid_mask=valid_mask_final.view(-1),
            )

            preserve_all_exploit = (
                action_policy == "greedy"
                or (guard_mode == "hard_early" and int(step_index) < 2)
                or (guard_mode == "hybrid_guard" and int(step_index) < 1)
                or (proxy_mode is not None and int(step_index) < 1)
            )

            if preserve_all_exploit:
                for slot in range(max(int(current_action_k), 1)):
                    if slot >= len(greedy_idx_rounds):
                        break
                    idx_round = greedy_idx_rounds[slot]
                    valid_selection = (idx_round >= 0) & active_mask_t
                    if not valid_selection.any():
                        continue
                    self._append_selected_nodes(
                        step_selected_indices,
                        confirmation_mask,
                        idx_round[valid_selection],
                        num_nodes,
                    )
                final_step_indices = None
                if step_selected_indices:
                    final_step_indices = torch.cat(step_selected_indices, dim=0)
                return confirmation_mask.view_as(logits_fused), final_step_indices, routing_meta

            rea_idx = greedy_idx_rounds[0] if greedy_idx_rounds else torch.full(
                (curr_batch_size,),
                -1,
                dtype=torch.long,
                device=logits_fused.device,
            )
            valid_selection = (rea_idx >= 0) & active_mask_t
            if valid_selection.any():
                self._append_selected_nodes(
                    step_selected_indices,
                    confirmation_mask,
                    rea_idx[valid_selection],
                    num_nodes,
                )

            if int(current_action_k) > 1 and nav_logits is not None:
                expl_logits = nav_logits.view(-1).clone()
                expl_logits[~valid_mask_final.view(-1)] = -float("inf")
                expl_logits[confirmation_mask > 0.5] = -float("inf")
                reasoner_logits_flat = logits_fused.view(-1)
                logit_margin = 0.5
                use_harm_guard = (
                    guard_mode == "harm_guard"
                    or (guard_mode == "hybrid_guard" and int(step_index) >= 1)
                )
                proxy_ctx = None
                proxy_scores = None
                explore_graph_mask = torch.ones(curr_batch_size, dtype=torch.bool, device=logits_fused.device)
                proxy_pool_k = max(int(current_action_k) + 1, 4)
                if proxy_mode is not None:
                    proxy_ctx = self._build_runtime_proxy_contract(
                        proxy_mode,
                        logits_fused,
                        nav_logits,
                        evidence_state,
                        fused_batch,
                        curr_batch_size,
                        valid_mask_final,
                        step_index=step_index,
                    )
                    proxy_scores = proxy_ctx["proxy_score"].clone()
                    proxy_scores[confirmation_mask > 0.5] = -float("inf")
                    explore_graph_mask = proxy_ctx["explore_graph_mask"].view(-1).bool()
                    routing_meta["proxy_eligible_graph_count"] = int((active_mask_t & explore_graph_mask).sum().item())

                for slot in range(max(int(current_action_k) - 1, 0)):
                    if proxy_mode is not None:
                        nav_pool_rounds, _ = self._iter_per_graph_topk(
                            expl_logits,
                            fused_batch,
                            curr_batch_size,
                            proxy_pool_k,
                            valid_mask=valid_mask_final.view(-1),
                        )
                        nav_idx = torch.full(
                            (curr_batch_size,),
                            -1,
                            dtype=torch.long,
                            device=logits_fused.device,
                        )
                        nav_vals = torch.full(
                            (curr_batch_size,),
                            -float("inf"),
                            dtype=logits_fused.dtype,
                            device=logits_fused.device,
                        )
                        for idx_round in nav_pool_rounds:
                            round_valid = idx_round >= 0
                            if not round_valid.any():
                                continue
                            safe_idx = idx_round.clamp(0, num_nodes - 1)
                            round_scores = torch.full_like(nav_vals, -float("inf"))
                            round_scores[round_valid] = proxy_scores[safe_idx[round_valid]]
                            better_mask = round_scores > nav_vals
                            nav_vals = torch.where(better_mask, round_scores, nav_vals)
                            nav_idx = torch.where(better_mask, idx_round, nav_idx)
                        valid_nav = (nav_vals > 1e-8) & active_mask_t & explore_graph_mask
                    else:
                        nav_vals, nav_idx = scatter_max(expl_logits, fused_batch, dim=0, dim_size=curr_batch_size)
                        valid_nav = (nav_vals > -1e9) & active_mask_t
                    if not valid_nav.any():
                        break

                    routing_meta["navigator_proposed_count"] += int(valid_nav.sum().item())
                    nav_idx_safe = nav_idx.clamp(0, num_nodes - 1)
                    fallback_idx = None
                    fallback_valid = torch.zeros_like(valid_nav)
                    if slot + 1 < len(greedy_idx_rounds):
                        fallback_idx = greedy_idx_rounds[slot + 1]
                        fallback_valid = (fallback_idx >= 0) & active_mask_t

                    admit_mask = valid_nav.clone()
                    blocked_mask = torch.zeros_like(valid_nav)
                    fallback_mask = torch.zeros_like(valid_nav)

                    if proxy_mode is not None and fallback_idx is not None:
                        fallback_safe = fallback_idx.clamp(0, num_nodes - 1)
                        fallback_proxy = torch.full_like(nav_vals, -float("inf"))
                        fallback_proxy[fallback_valid] = proxy_scores[fallback_safe[fallback_valid]]
                        admit_mask = valid_nav & (
                            (~fallback_valid)
                            | (nav_vals >= fallback_proxy)
                        )
                        blocked_mask = valid_nav & (~admit_mask) & fallback_valid
                        fallback_mask = blocked_mask.clone()
                    elif use_harm_guard and fallback_idx is not None:
                        fallback_safe = fallback_idx.clamp(0, num_nodes - 1)
                        nav_reasoner_val = reasoner_logits_flat[nav_idx_safe]
                        fallback_reasoner_val = reasoner_logits_flat[fallback_safe]
                        admit_mask = valid_nav & (
                            (~fallback_valid)
                            | (nav_reasoner_val >= (fallback_reasoner_val - logit_margin))
                        )
                        blocked_mask = valid_nav & (~admit_mask) & fallback_valid
                        fallback_mask = blocked_mask.clone()

                    if admit_mask.any():
                        selected = self._append_selected_nodes(
                            step_selected_indices,
                            confirmation_mask,
                            nav_idx[admit_mask],
                            num_nodes,
                            mask_logits=expl_logits,
                        )
                        if selected.numel() > 0:
                            routing_meta["nav_out_consumed"] = True
                            routing_meta["navigator_admitted_count"] += int(admit_mask.sum().item())

                    if fallback_mask.any() and fallback_idx is not None:
                        selected = self._append_selected_nodes(
                            step_selected_indices,
                            confirmation_mask,
                            fallback_idx[fallback_mask],
                            num_nodes,
                            mask_logits=expl_logits,
                        )
                        if selected.numel() > 0:
                            routing_meta["fallback_to_exploitation_count"] += int(fallback_mask.sum().item())
                            routing_meta["navigator_blocked_count"] += int(blocked_mask.sum().item())
                    valid_nav_idx = nav_idx[valid_nav]
                    valid_nav_idx = self._sanitize_selected_nodes(valid_nav_idx, num_nodes)
                    if valid_nav_idx.numel() > 0:
                        expl_logits[valid_nav_idx] = -float("inf")
                        if proxy_scores is not None:
                            proxy_scores[valid_nav_idx] = -float("inf")

        elif action_policy == "navigator_only":
            routing_meta["action_policy_branch"] = "navigator_only"
            nav_indices_flat = nav_out.get("selected_indices") if nav_out is not None else None
            nav_indices_flat = self._sanitize_selected_nodes(nav_indices_flat, num_nodes)
            if nav_indices_flat.numel() > 0:
                kept_indices = []
                per_graph_counts = torch.zeros(curr_batch_size, dtype=torch.long, device=logits_fused.device)
                for node_idx in nav_indices_flat.view(-1):
                    graph_idx = int(fused_batch[int(node_idx)].item())
                    if not bool(active_mask_t[graph_idx].item()):
                        continue
                    if not bool(valid_mask_final[int(node_idx)].item()):
                        continue
                    if per_graph_counts[graph_idx] >= int(max(current_action_k, 1)):
                        continue
                    kept_indices.append(node_idx.view(1))
                    per_graph_counts[graph_idx] += 1
                if kept_indices:
                    selected = self._append_selected_nodes(
                        step_selected_indices,
                        confirmation_mask,
                        torch.cat(kept_indices, dim=0),
                        num_nodes,
                    )
                    if selected.numel() > 0:
                        routing_meta["nav_out_consumed"] = True
                        routing_meta["navigator_admitted_count"] = int(selected.numel())

        elif action_policy == "random_valid":
            routing_meta["action_policy_branch"] = "random_valid"
            temp_logits = torch.rand_like(logits_fused.view(-1))
            invalid_mask = (~valid_mask_final.view(-1).bool()) | (~active_mask_t[fused_batch].bool())
            temp_logits[invalid_mask] = -float("inf")
            for _ in range(max(int(current_action_k), 1)):
                rand_vals, rand_idx = scatter_max(temp_logits, fused_batch, dim=0, dim_size=curr_batch_size)
                valid_selection = (rand_vals > -1e9) & active_mask_t
                if not valid_selection.any():
                    break
                selected = self._append_selected_nodes(
                    step_selected_indices,
                    confirmation_mask,
                    rand_idx[valid_selection],
                    num_nodes,
                    mask_logits=temp_logits,
                )
                if selected.numel() > 0:
                    routing_meta["random_valid_admitted_count"] = routing_meta.get("random_valid_admitted_count", 0) + int(selected.numel())

        elif action_policy == "learned":
            routing_meta["action_policy_branch"] = "learned"
            budget_k = max(int(current_action_k), 1)
            logit_max_abs = getattr(self.cfg.training, "logit_max_abs", 20.0)
            eps_prob = getattr(self.cfg.training, "eps_prob", 1e-8)

            temp_logits = torch.clamp(logits_fused.view(-1).clone(), min=-logit_max_abs, max=logit_max_abs)
            probs = gnn_softmax(temp_logits, fused_batch)
            p_max, _ = scatter_max(probs, fused_batch, dim=0, dim_size=curr_batch_size)
            log_p = torch.log(probs + eps_prob)
            entropy_per_node = -probs * log_p
            h_sum = scatter_sum(entropy_per_node, fused_batch, dim=0, dim_size=curr_batch_size)
            n_graph = scatter_sum(torch.ones_like(probs), fused_batch, dim=0, dim_size=curr_batch_size)
            h_norm = h_sum / torch.log(n_graph + eps_prob)

            csm_cfg = getattr(self.cfg.model, "cognitive_state_machine", {})
            if hasattr(csm_cfg, "get"):
                p1_h = csm_cfg.get("phase1_h_threshold", 0.8)
                p1_p = csm_cfg.get("phase1_p_threshold", 0.3)
                p3_h = csm_cfg.get("phase3_h_threshold", 0.4)
                p3_p = csm_cfg.get("phase3_p_threshold", 0.8)
            else:
                p1_h = getattr(csm_cfg, "phase1_h_threshold", 0.8)
                p1_p = getattr(csm_cfg, "phase1_p_threshold", 0.3)
                p3_h = getattr(csm_cfg, "phase3_h_threshold", 0.4)
                p3_p = getattr(csm_cfg, "phase3_p_threshold", 0.8)

            phase1_mask = (h_norm > p1_h) | (p_max < p1_p)
            phase3_mask = (h_norm <= p3_h) | (p_max > p3_p)
            phase2_mask = (~phase1_mask) & (~phase3_mask)

            rea_candidates = []
            temp_rea_logits = temp_logits.clone()
            for _ in range(budget_k):
                rea_vals, rea_idx = scatter_max(temp_rea_logits, fused_batch, dim=0, dim_size=curr_batch_size)
                valid_selection = rea_vals > -1e9
                if not valid_selection.any():
                    break
                rea_candidates.append(rea_idx)
                valid_idx = rea_idx[valid_selection]
                valid_idx = self._sanitize_selected_nodes(valid_idx, num_nodes)
                if valid_idx.numel() > 0:
                    temp_rea_logits[valid_idx] = -float("inf")

            final_indices_list = []
            nav_indices_flat = nav_out.get("selected_indices") if nav_out is not None else None

            if nav_indices_flat is not None:
                nav_indices_flat = self._sanitize_selected_nodes(nav_indices_flat, num_nodes)
            else:
                nav_indices_flat = torch.tensor([], device=logits_fused.device, dtype=torch.long)

            if nav_indices_flat.numel() > 0:
                nav_batch = fused_batch[nav_indices_flat]
                p1_nodes = phase1_mask[nav_batch]
                if p1_nodes.any():
                    final_indices_list.append(nav_indices_flat[p1_nodes])
                    routing_meta["nav_out_consumed"] = True

            for idx_b in rea_candidates:
                if idx_b.size(0) != phase3_mask.size(0):
                    continue
                safe_idx_b = idx_b.clamp(0, num_nodes - 1)
                p3_valid = phase3_mask & (idx_b >= 0) & (idx_b < num_nodes) & (temp_logits[safe_idx_b] > -1e9)
                if p3_valid.any():
                    final_indices_list.append(idx_b[p3_valid])

            if rea_candidates:
                idx_b = rea_candidates[0]
                p2_valid = phase2_mask & (idx_b >= 0) & (idx_b < num_nodes)
                if p2_valid.any():
                    final_indices_list.append(idx_b[p2_valid])

            if nav_indices_flat.numel() > 0:
                nav_batch = fused_batch[nav_indices_flat]
                p2_nodes = phase2_mask[nav_batch]
                if p2_nodes.any():
                    nav_indices_cpu = nav_indices_flat.detach().cpu()
                    nav_batch_cpu = nav_batch.detach().cpu()
                    p2_mask_cpu = phase2_mask.detach().cpu()
                    phase3_mask_cpu = phase3_mask.detach().cpu()
                    keep_mask_cpu = torch.ones(nav_indices_flat.size(0), dtype=torch.bool)
                    p2_counts = torch.zeros(curr_batch_size, device="cpu")
                    nav_budget_phase2 = max(budget_k - 1, 0)
                    for i, (_, graph_id) in enumerate(zip(nav_indices_cpu, nav_batch_cpu)):
                        graph_id_int = int(graph_id.item())
                        if p2_mask_cpu[graph_id_int]:
                            if p2_counts[graph_id_int] < nav_budget_phase2:
                                p2_counts[graph_id_int] += 1
                            else:
                                keep_mask_cpu[i] = False
                        if phase3_mask_cpu[graph_id_int]:
                            keep_mask_cpu[i] = False
                    keep_mask = keep_mask_cpu.to(nav_indices_flat.device)
                    kept = nav_indices_flat[keep_mask]
                    if kept.numel() > 0:
                        final_indices_list.append(kept)
                        routing_meta["nav_out_consumed"] = True

            if final_indices_list:
                all_selected = torch.cat(final_indices_list, dim=0)
                self._append_selected_nodes(
                    step_selected_indices,
                    confirmation_mask,
                    all_selected,
                    num_nodes,
                )

        else:
            temp_logits = logits_fused.view(-1).clone()
            for _ in range(max(int(current_action_k), 1)):
                rea_vals, rea_idx = scatter_max(temp_logits, fused_batch, dim=0, dim_size=curr_batch_size)
                valid_selection = (rea_vals > -1e9) & active_mask_t
                if not valid_selection.any():
                    break
                self._append_selected_nodes(
                    step_selected_indices,
                    confirmation_mask,
                    rea_idx[valid_selection],
                    num_nodes,
                    mask_logits=temp_logits,
                )

        final_step_indices = None
        if step_selected_indices:
            final_step_indices = torch.cat(step_selected_indices, dim=0)
        return confirmation_mask.view_as(logits_fused), final_step_indices, routing_meta

    def run(
        self,
        max_episodes,
        static_ctx,
        dynamic_state,
        inference_mode=False,
        sample_budget=1,
        action_policy='greedy',
        tau=1.0,
        enable_tracer=False,
        profile_rollout=False,
        skip_reasoner_forward=False,
        store_resume_state=False,
        forced_action_indices_by_step=None,
    ):
        episode_runner_start = time.perf_counter() if profile_rollout else None
        
        # 1. Initialize
        h_fused = dynamic_state.get('resume_h_fused', static_ctx['h_fused'])
        x_nav = static_ctx['x_nav']
        x_raw = static_ctx.get('x_raw')
        fused_batch = static_ctx['fused_batch']
        inverse_indices = static_ctx['inverse_indices']
        curr_batch_size = static_ctx['curr_batch_size']
        fused_source_label = static_ctx.get('fused_source_label')
        has_y = static_ctx['has_y']
        graph_source_global_ids = static_ctx.get('graph_source_global_ids')
        allow_constraint_label_fallback = bool(
            static_ctx.get(
                'allow_constraint_label_fallback',
                getattr(self.cfg.life_support, 'allow_constraint_label_fallback', False),
            )
        )
        
        # [Refactor] Explicitly track Raw and Fused states
        x_nav_raw = dynamic_state.get('resume_x_nav_raw', static_ctx['x_nav']) # Start with Raw

        t_sim = dynamic_state['t_sim']
        acc_mask_local = dynamic_state['accumulated_mask'].clone()
        constraint_state = self.state_builder.build_constraint_state(dynamic_state, h_fused.size(0), h_fused.device)
        acc_mask_local = torch.max(acc_mask_local, constraint_state.sampled_mask)
        causal_anchors = dynamic_state['causal_anchors']
        anchor_times = dynamic_state['anchor_times']
        
        hit_before_t = dynamic_state['hit_before_t']
        graph_success = dynamic_state['graph_success']
        reasoner_memory_state = dynamic_state.get('reasoner_memory_state')
        nav_memory_state = dynamic_state.get('nav_memory_state')
        
        self.metrics_flow.start_episode(inference_mode, id(self), acc_mask_local)
        
        clean_nav_mainline = self._clean_nav_mainline_enabled()
        if has_y and fused_source_label is not None:
             pre_hits = (acc_mask_local.view(-1) > 0.5) & (fused_source_label.view(-1) > 0.5)
             pre_hits_graph = scatter_max(pre_hits.float(), fused_batch, dim=0, dim_size=curr_batch_size)[0]
             pre_success = (pre_hits_graph > 0.5)
             if not clean_nav_mainline:
                 hit_before_t = hit_before_t | pre_success
                 graph_success = graph_success | pre_success
        confirmed_source_graph = scatter_max(
            (constraint_state.confirmed_source_mask.view(-1) > 0.5).float(),
            fused_batch,
            dim=0,
            dim_size=curr_batch_size,
        )[0] > 0.5
        if not clean_nav_mainline:
            hit_before_t = hit_before_t | confirmed_source_graph
            graph_success = graph_success | confirmed_source_graph

        trajectory_data = []
        last_logits = None
        compact_training_trajectory = (
            (not inference_mode)
            and (not self.metrics_flow.collect_detailed_step_metrics)
            and (not self._evidence_oracle_enabled())
        )
        last_post_action_evidence_state = None
        last_post_action_valid_mask = None
        
        steps_taken = dynamic_state.get('steps_taken')
        if steps_taken is None:
            steps_taken = torch.zeros(curr_batch_size, device=h_fused.device)
        else:
            steps_taken = steps_taken.detach().clone().to(device=h_fused.device, dtype=torch.float32)
        total_poison_hits_per_graph = torch.zeros(curr_batch_size, device=h_fused.device)
        first_hit_step = torch.full((curr_batch_size,), -1.0, device=h_fused.device)
        
        # Predict Hit Metrics
        predict_hit_at_1 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        predict_hit_at_5 = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        predict_hit_valid = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
        max_hit_prob = torch.zeros(curr_batch_size, device=h_fused.device)
        initial_mainline_semantics = None
        final_mainline_semantics = None
        budget_to_closure = torch.full((curr_batch_size,), -1.0, device=h_fused.device)
        rollout_profile = {
            "observation_s": 0.0,
            "dynamic_topology_s": 0.0,
            "physics_s": 0.0,
            "evidence_s": 0.0,
            "nav_state_build_s": 0.0,
            "navigator_s": 0.0,
            "reasoner_state_build_s": 0.0,
            "reasoner_s": 0.0,
            "metrics_s": 0.0,
            "action_select_s": 0.0,
            "state_update_s": 0.0,
            "reembed_s": 0.0,
            "reward_s": 0.0,
        }
        rollout_profile_steps = 0

        start_episode_index = int(dynamic_state.get('resume_step_index', 0))

        def profile_tic():
            if profile_rollout and h_fused.device.type == "cuda":
                torch.cuda.synchronize(h_fused.device)
            return time.perf_counter()

        def profile_add(name, start_time):
            if not profile_rollout:
                return
            if h_fused.device.type == "cuda":
                torch.cuda.synchronize(h_fused.device)
            rollout_profile[name] += time.perf_counter() - start_time

        # Loop
        for local_step in range(max_episodes):
            step = start_episode_index + int(local_step)
            if profile_rollout:
                rollout_profile_steps += 1
            
            # [Fix] Fuse x_nav_raw -> x_nav (Fused) using Out-of-Place operations
            if x_nav_raw.size(0) != h_fused.size(0):
                 num_fused = h_fused.size(0)
                 
                 # Ch0 (Chlorine): Min (Negative Deviation)
                 # [Fix] Avoid in-place assignment to x_nav_fused[:, 0]
                 val_ch0 = scatter_min(x_nav_raw[:, 0], inverse_indices, dim=0, dim_size=num_fused)[0]
                 
                 # Ch1-End: Max (Binary Flags / Freshness)
                 if x_nav_raw.size(1) > 1:
                     val_ch1_end = scatter_max(x_nav_raw[:, 1:], inverse_indices, dim=0, dim_size=num_fused)[0]
                     x_nav_fused = torch.cat([val_ch0.unsqueeze(1), val_ch1_end], dim=1)
                 else:
                     x_nav_fused = val_ch0.unsqueeze(1)
            else:
                 x_nav_fused = x_nav_raw

            # A. Check Active Status
            active_mask_t = (~hit_before_t).clone()
            force_active_step0 = getattr(self.cfg.life_support, 'enable_pacemaker', True)
            if step == 0 and not inference_mode and force_active_step0:
                active_mask_t = torch.ones_like(hit_before_t, dtype=torch.bool)
            
            if not inference_mode and not active_mask_t.any():
                if getattr(self.cfg.life_support, 'allow_early_stop', False):
                    break
            if inference_mode and not active_mask_t.any():
                break
                
            # [Fix] Out-of-place update to avoid RuntimeError in autograd
            steps_taken = steps_taken + active_mask_t.float()

            # B. Build States
            t_prof = profile_tic()
            curr_edge_index, curr_edge_attr = self.model._update_dynamic_topology(
                causal_anchors, anchor_times, static_ctx['fused_global_ids'], fused_batch, 
                static_ctx['fused_edge_index'], static_ctx['base_edge_attr'], t_sim=t_sim, static_ctx=static_ctx
            )
            profile_add("dynamic_topology_s", t_prof)
            
            # [SSOT Audit] Extract Dynamic STT for Reachability (if available)
            stt_dynamic = None
            if 'stt_dynamic' in static_ctx:
                stt_dynamic = static_ctx['stt_dynamic'] # [E, 1]
                # Match length with curr_edge_index (Fused)
                # static_ctx['stt_dynamic'] is for SUBGRAPH edges (before fusion)
                # In current implementation, fusion is identity for edges (subgraph level).
                # But curr_edge_index might have virtual edges added.
                num_current_edges = curr_edge_index.size(1)
                if stt_dynamic.size(0) < num_current_edges:
                    padding = torch.zeros(num_current_edges - stt_dynamic.size(0), 1, device=stt_dynamic.device)
                    stt_dynamic = torch.cat([stt_dynamic, padding], dim=0)
                elif stt_dynamic.size(0) > num_current_edges:
                    stt_dynamic = stt_dynamic[:num_current_edges]
            
            t_prof = profile_tic()
            physics_in = {
                't_sim': t_sim,
                'valid_mask': acc_mask_local.view(-1),
                'anchor_type': causal_anchors.view(-1),
                'anchor_time': anchor_times.view(-1),
                'edge_index': curr_edge_index,
                'edge_stt': curr_edge_attr[:, 3] if curr_edge_attr.size(1) > 3 else torch.zeros(curr_edge_index.size(1), device=h_fused.device),
                'batch': fused_batch
            }
            physics_ctx = self.model.physics_module(physics_in)
            phys_context = self.state_builder.build_physics_context(
                curr_edge_index, curr_edge_attr, physics_ctx, h_fused.device, num_nodes=h_fused.size(0), batch=fused_batch, stt_dynamic=stt_dynamic
            )
            profile_add("physics_s", t_prof)

            t_prof = profile_tic()
            obs_state = self.state_builder.build_observation_state(x_nav_fused)
            profile_add("observation_s", t_prof)
            t_prof = profile_tic()
            t0_ev = time.time()
            evidence_state = self.state_builder.build_evidence_state(obs_state, phys_context, t_sim)
            evidence_state = self._maybe_refine_evidence_state(
                evidence_state,
                obs_state,
                constraint_state,
                batch_index=fused_batch,
            )
            evidence_oracle_targets = None
            if not inference_mode and self._evidence_oracle_enabled():
                with torch.no_grad():
                    evidence_oracle_targets = self.oracle_label_builder.build(
                        observation_state=obs_state,
                        physics_context=phys_context,
                        t_sim=t_sim,
                        inverse_indices=inverse_indices,
                        x_raw=x_raw,
                        view_batch=static_ctx.get("view_batch"),
                        trigger_time_step=static_ctx.get("trigger_time_step"),
                        step_seconds=static_ctx.get("step_seconds"),
                    )
            t1_ev = time.time()
            profile_add("evidence_s", t_prof)
            
            # [Audit] System v2 Check
            if not inference_mode:
                self.auditor.audit_step(step, {
                    'observation_state': obs_state,
                    'evidence_state': evidence_state,
                    'inverse_indices': inverse_indices,
                    'num_fused_nodes': h_fused.size(0)
                })
            
            # [METRICS] Step 0 Evidence Snapshot (W&B)
            if not inference_mode:
                 # Use Fused Batch for Evidence Stats (Evidence is Fused)
                 # Pass time cost for profiling
                 self.metrics_flow.track_evidence_stats(
                     evidence_state, fused_batch, curr_batch_size, 
                     fused_source_label, step, (t1_ev - t0_ev) * 1000.0
                 )

            # C. FoV & Dispatch Modules
            stats = {'entropy': 2.0, 'race_conflict_mean': 0.0, 'max_prob': 0.0, 'top1_margin': 0.0}
            if step > 0 and last_logits is not None:
                stats = self.metrics_flow.calculate_step_stats(last_logits, fused_batch, energy=physics_ctx.get('race_energy'))
            
            fov_params = {}
            if self.model.fov_controller:
                fov_params = self.model.fov_controller.step(stats)
            
            current_action_k = sample_budget if 'candidate_topM' not in fov_params else int(fov_params['candidate_topM'])
            
            valid_mask_final = ~constraint_state.no_resample_mask.view(-1).bool()
            if 'feasible_mask' in physics_ctx:
                valid_mask_final = valid_mask_final & physics_ctx['feasible_mask'].view(-1).bool()
            if clean_nav_mainline:
                current_mainline_semantics = self.metrics_flow.compute_mainline_semantics(
                    evidence_state=evidence_state,
                    constraint_state=constraint_state,
                    valid_mask=valid_mask_final,
                    batch_index=fused_batch,
                )
                if initial_mainline_semantics is None:
                    initial_mainline_semantics = {
                        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                        for key, value in current_mainline_semantics.items()
                    }
            temp_graph = TempGraph(h_fused, curr_edge_index, curr_edge_attr, fused_batch)
            nav_state_summary = self._build_nav_summary(
                obs_state=obs_state,
                evidence_state=evidence_state,
                constraint_state=constraint_state,
                valid_mask=valid_mask_final,
                fused_batch=fused_batch,
                curr_batch_size=curr_batch_size,
                steps_taken=steps_taken,
                max_episodes=max_episodes,
                step_index=step,
            )
            critic_privileged = None
            if not inference_mode and bool(getattr(self.model.navigator_module, "use_aux_privileged_critic", False)):
                critic_privileged = self._build_critic_privileged(
                    fused_batch=fused_batch,
                    fused_global_ids=static_ctx.get('fused_global_ids'),
                    graph_source_global_ids=graph_source_global_ids,
                    device=h_fused.device,
                )

            t_prof = profile_tic()
            t_state = profile_tic()
            nav_state = self.state_builder.build_nav_state(
                h_fused, x_nav_raw, obs_state, valid_mask_final, nav_memory_state, nav_state_summary[:, :3], nav_state_summary,
                current_action_k, tau, fused_batch, fov_params, action_policy, static_ctx,
                None if action_policy == "nav_only" else last_logits,
                curr_edge_index, curr_edge_attr, evidence_state, constraint_state,
                critic_privileged=critic_privileged,
                emit_action_audit_scalars=((not compact_training_trajectory) or profile_rollout),
                step_index=step,
            )
            profile_add("nav_state_build_s", t_state)
            
            nav_out = self.module_dispatcher.dispatch_navigator(nav_state, temp_graph, physics_ctx)
            nav_logits = nav_out.get('logits')
            if 'updated_memory_state' in nav_out:
                nav_memory_state = nav_out['updated_memory_state']
            profile_add("navigator_s", t_prof)
            
            # Reasoner State
            t_prof = profile_tic()
            t_state = profile_tic()
            reasoner_state = self.state_builder.build_reasoner_state(
                h_fused,
                x_nav_raw,
                obs_state,
                static_ctx,
                inverse_indices,
                causal_anchors,
                acc_mask_local,
                reasoner_memory_state,
                evidence_state,
                constraint_state,
                valid_mask=valid_mask_final,
                nav_state_summary=nav_state_summary,
            )
            vis_reasoner_state = reasoner_state.copy()
            profile_add("reasoner_state_build_s", t_state)
            
            if skip_reasoner_forward:
                logits_fused = h_fused.new_zeros((h_fused.size(0), 1))
            else:
                reasoner_out = self.module_dispatcher.dispatch_reasoner(reasoner_state, temp_graph, physics_ctx)
                if 'updated_memory_state' in reasoner_out:
                    reasoner_memory_state = reasoner_out['updated_memory_state']
                logits_fused = reasoner_out['logits']
            profile_add("reasoner_s", t_prof)

            # Bias Injection
            logits_fused, nav_logits = self.module_dispatcher.apply_biases(
                logits_fused, nav_logits, h_fused, curr_edge_index, curr_edge_attr, t_sim, fused_batch, physics_ctx
            )
            logits_fused = self.state_updater.apply_constraint_masks(logits_fused, constraint_state, fused_batch)
            nav_logits = self.state_updater.apply_constraint_masks(nav_logits, constraint_state, fused_batch)
            last_logits = logits_fused

            # D. Metrics & Bias
            if not inference_mode and has_y:
                t_prof = profile_tic()
                with torch.no_grad():
                    self.metrics_flow.compute_step_metrics(step, logits_fused, fused_batch, fused_source_label, active_mask_t, curr_batch_size)
                    
                    if step == 0:
                        # Hit@1 Snapshot logic (Simplified copy from stepper)
                        _, top1_idx = scatter_max(logits_fused.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                        has_source_in_graph, _ = scatter_max(fused_source_label, fused_batch, dim=0, dim_size=curr_batch_size)
                        predict_hit_valid = (has_source_in_graph.view(-1) > 0.5)
                        valid_idx_mask = (top1_idx >= 0) & (top1_idx < fused_source_label.size(0))
                        hit_check = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                        if valid_idx_mask.any():
                            hit_check[valid_idx_mask] = (fused_source_label[top1_idx[valid_idx_mask]].view(-1) > 0.5)
                        predict_hit_at_1 = hit_check & predict_hit_valid
                        
                        # Hit@5
                        predict_hit_at_5_accum = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                        temp_logits_h5 = logits_fused.view(-1).clone()
                        for _ in range(5):
                            _, top_idx = scatter_max(temp_logits_h5, fused_batch, dim=0, dim_size=curr_batch_size)
                            valid_mask = (top_idx >= 0) & (top_idx < fused_source_label.size(0))
                            current_hit = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
                            if valid_mask.any():
                                 current_hit[valid_mask] = (fused_source_label[top_idx[valid_mask]].view(-1) > 0.5)
                            predict_hit_at_5_accum = predict_hit_at_5_accum | current_hit
                            if valid_mask.any():
                                 temp_logits_h5[top_idx[valid_mask]] = -float('inf')
                        predict_hit_at_5 = predict_hit_at_5_accum & predict_hit_valid
                profile_add("metrics_s", t_prof)
            
            # E. Action Selection
            t_prof = profile_tic()
            resume_state_before_action = None
            if store_resume_state and not compact_training_trajectory:
                resume_state_before_action = self._build_resume_state(
                    step_index=step,
                    h_fused=h_fused,
                    x_nav_raw=x_nav_raw,
                    t_sim=t_sim,
                    acc_mask_local=acc_mask_local,
                    constraint_state=constraint_state,
                    causal_anchors=causal_anchors,
                    anchor_times=anchor_times,
                    hit_before_t=hit_before_t,
                    graph_success=graph_success,
                    reasoner_memory_state=reasoner_memory_state,
                    nav_memory_state=nav_memory_state,
                    steps_taken=(steps_taken - active_mask_t.float()).clamp_min(0.0),
                )
            forced_indices = None
            if forced_action_indices_by_step is not None:
                forced_indices = forced_action_indices_by_step.get(int(step))
            if forced_indices is not None:
                confirmation_mask, final_step_indices, routing_meta = self._route_forced_actions(
                    forced_indices=forced_indices,
                    logits_fused=logits_fused,
                    fused_batch=fused_batch,
                    curr_batch_size=curr_batch_size,
                    current_action_k=current_action_k,
                    valid_mask_final=valid_mask_final,
                )
            else:
                confirmation_mask, final_step_indices, routing_meta = self._route_actions(
                    action_policy=action_policy,
                    logits_fused=logits_fused,
                    nav_logits=nav_logits,
                    nav_out=nav_out,
                    evidence_state=evidence_state,
                    active_mask_t=active_mask_t,
                    fused_batch=fused_batch,
                    curr_batch_size=curr_batch_size,
                    current_action_k=current_action_k,
                    valid_mask_final=valid_mask_final,
                    step_index=step,
                )
            profile_add("action_select_s", t_prof)
            
            # F. State Update
            t_prof = profile_tic()
            constraint_state_before_reward = constraint_state
            acc_mask_local, A_total_mask = self.state_updater.update_mask(acc_mask_local, confirmation_mask)
            verdict_payload = build_runtime_verdict_payload(
                A_total_mask.float(),
                fused_batch=fused_batch,
                fused_global_ids=static_ctx.get('fused_global_ids'),
                graph_source_global_ids=graph_source_global_ids,
            )
            constraint_state, constraint_update = self.state_updater.update_constraint_state(
                constraint_state,
                A_total_mask.float(),
                verdict_payload=verdict_payload,
                fused_source_label=fused_source_label if has_y else None,
                allow_label_fallback=allow_constraint_label_fallback,
            )
            acc_mask_local = torch.max(acc_mask_local, constraint_state.sampled_mask)
            
            x_nav_raw, causal_anchors, anchor_times, poison_at_t = self.state_updater.update_observation_state(
                x_nav_raw, acc_mask_local, inverse_indices, x_raw, static_ctx, t_sim, A_total_mask, causal_anchors, anchor_times, fused_batch
            )
            profile_add("state_update_s", t_prof)
            
            # Metrics: Poison Hits
            if not inference_mode and poison_at_t is not None:
                newly_revealed_raw = A_total_mask[inverse_indices].view(-1).bool()
                poison_hits_mask = (poison_at_t > 0.5) & newly_revealed_raw
                batch_mapping = fused_batch[inverse_indices]
                hits_per_graph = scatter_sum(poison_hits_mask.float(), batch_mapping, dim=0, dim_size=curr_batch_size)
                # [Fix] Out-of-place update
                total_poison_hits_per_graph = total_poison_hits_per_graph + hits_per_graph
            
            # G. Dynamic Re-embedding
            t_prof = profile_tic()
            view_batch = static_ctx['view_batch']
            x_nav_gated, _ = self.model.dynamic_gate(x_nav_raw, view_batch)
            
            call_kwargs = {}
            if self.model.navigator_backbone_accepts_memory_state:
                call_kwargs['memory_state'] = None
            if self.model.navigator_backbone_accepts_batch:
                call_kwargs['batch'] = view_batch
                
            with torch.no_grad():
                h_nav = self.model.navigator_module.backbone(x_nav_gated, curr_edge_index, **call_kwargs)
                if isinstance(h_nav, tuple): h_nav = h_nav[0]
            
            with torch.no_grad():
                h_fused_new, _, _, _, _ = self.model.fusion(
                    h_nav, view_batch, static_ctx['batch_n_id'], static_ctx['batch_scenario_id']
                )
            h_fused = h_fused_new
            profile_add("reembed_s", t_prof)
            
            # H. Rewards & Loss
            nav_gain_loss = None
            nav_reward = None
            nav_reward_bundle = None
            
            reward_step_active = bool(nav_out.get("coupling_active", True))
            if not inference_mode and has_y and self.reward_path_enabled and reward_step_active:
                t_prof = profile_tic()
                # Pre-calc for Reward
                probs_before = gnn_softmax(logits_fused.view(-1), fused_batch)
                prob_before_target = scatter_sum(probs_before * fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
                
                # Run Reasoner on Next State
                # [Fix] Re-build next_evidence_state using next observation and current physics
                # 1. Use updated x_nav_raw from state_updater (which contains the new observation after action)
                # 2. Fuse it to Fused Space to match PhysicsContext
                x_nav_fused_next = self.state_builder.fuse_observation(x_nav_raw, inverse_indices, h_fused.size(0))
                
                # 3. Build Fused Observation State
                obs_state_next = self.state_builder.build_observation_state(x_nav_fused_next)
                
                # 4. Build Evidence State (Fused + Fused)
                evidence_state_next = self.state_builder.build_evidence_state(obs_state_next, phys_context, t_sim)
                evidence_state_next = self._maybe_refine_evidence_state(
                    evidence_state_next,
                    obs_state_next,
                    constraint_state,
                    batch_index=fused_batch,
                )
                valid_mask_after = ~constraint_state.no_resample_mask.view(-1).bool()
                if 'feasible_mask' in physics_ctx:
                    valid_mask_after = valid_mask_after & physics_ctx['feasible_mask'].view(-1).bool()
                
                reasoner_state_next = {
                    'h_fused': h_fused,
                    'x_nav': x_nav_raw, # Pass Raw for Reasoner (it handles fusion internally if needed, or we pass fused?)
                                        # Reasoner usually takes x_nav (which was raw in previous loop).
                                        # Let's keep it consistent with loop start: x_nav_raw
                    'observation_state': obs_state_next, # This is Fused!
                    'inverse_indices': inverse_indices,
                    'causal_anchors': causal_anchors,
                    'accumulated_mask': acc_mask_local,
                    'memory_state': reasoner_memory_state,
                    'evidence_state': evidence_state_next, # [Fix] Use Updated Evidence
                    'constraint_state': constraint_state,
                    'valid_mask': valid_mask_after,
                    'nav_state_summary': nav_state_summary,
                }
                with torch.no_grad():
                    reasoner_out_next = self.module_dispatcher.dispatch_reasoner(reasoner_state_next, temp_graph, physics_ctx)
                    logits_next = reasoner_out_next['logits']
                    logits_next = self.state_updater.apply_constraint_masks(logits_next, constraint_state, fused_batch)
                if post_action_reasoner_logits is None:
                    post_action_reasoner_logits = logits_next
                
                y_action = nav_out.get('y_action')
                budget_before = scatter_sum(acc_mask_local.view(-1).float(), fused_batch, dim=0, dim_size=curr_batch_size) - scatter_sum(
                    A_total_mask.view(-1).float(), fused_batch, dim=0, dim_size=curr_batch_size
                )
                budget_after = scatter_sum(acc_mask_local.view(-1).float(), fused_batch, dim=0, dim_size=curr_batch_size)
                nav_gain_loss, nav_reward, nav_reward_bundle = self.metrics_flow.calculate_rewards(
                    logits_next,
                    fused_batch,
                    fused_source_label,
                    prob_before_target,
                    curr_batch_size,
                    y_action,
                    logits_before=logits_fused,
                    observation_state_before=obs_state,
                    evidence_state_before=evidence_state,
                    evidence_state_next=evidence_state_next,
                    constraint_state_before=constraint_state_before_reward,
                    constraint_state_after=constraint_state,
                    valid_mask_before=valid_mask_final,
                    valid_mask_after=valid_mask_after,
                    selection_mask=A_total_mask,
                    budget_used_before=budget_before,
                    budget_used_after=budget_after,
                )
                self.metrics_flow.record_reward_bundle(step, nav_reward_bundle)
                profile_add("reward_s", t_prof)
            
            # Success Tracking
            payload_source_hit = constraint_update.get('is_source_hit')
            if payload_source_hit is not None:
                new_success = payload_source_hit.to(device=h_fused.device).view(-1).bool()
                just_hit = new_success & (~hit_before_t)
                if just_hit.any():
                     first_hit_step = first_hit_step.clone()
                     first_hit_step[just_hit] = float(step)

                if not clean_nav_mainline:
                    hit_before_t = hit_before_t | new_success
                    graph_success = graph_success | new_success
            elif has_y:
                hits = (A_total_mask.view(-1) > 0.5) & (fused_source_label.view(-1) > 0.5)
                hits_graph = scatter_max(hits.float(), fused_batch, dim=0, dim_size=curr_batch_size)[0]
                new_success = (hits_graph > 0.5)
                
                just_hit = new_success & (~hit_before_t)
                if just_hit.any():
                     # [Fix] Out-of-place update
                     first_hit_step = first_hit_step.clone()
                     first_hit_step[just_hit] = float(step)
                
                if not clean_nav_mainline:
                    hit_before_t = hit_before_t | new_success
                    graph_success = graph_success | new_success
            else:
                new_success = torch.zeros(curr_batch_size, dtype=torch.bool, device=h_fused.device)
            
            t_sim = t_sim + (active_mask_t.float() * 45.0) # Delta T default
            record_post_action_semantics = bool(
                action_policy == "nav_only"
                or _cfg_get(getattr(self.cfg.model, "navigator", {}), "record_post_action_semantics", False)
            )
            post_action_observation_state = None
            post_action_evidence_state = None
            post_action_valid_mask = None
            post_action_mainline_semantics = None
            post_action_reasoner_logits = None
            if record_post_action_semantics:
                x_nav_fused_post = self.state_builder.fuse_observation(
                    x_nav_raw,
                    inverse_indices,
                    h_fused.size(0),
                )
                post_action_observation_state = self.state_builder.build_observation_state(x_nav_fused_post)
                physics_in_post = {
                    't_sim': t_sim,
                    'valid_mask': acc_mask_local.view(-1),
                    'anchor_type': causal_anchors.view(-1),
                    'anchor_time': anchor_times.view(-1),
                    'edge_index': curr_edge_index,
                    'edge_stt': curr_edge_attr[:, 3] if curr_edge_attr.size(1) > 3 else torch.zeros(curr_edge_index.size(1), device=h_fused.device),
                    'batch': fused_batch,
                }
                physics_ctx_post = self.model.physics_module(physics_in_post)
                phys_context_post = self.state_builder.build_physics_context(
                    curr_edge_index,
                    curr_edge_attr,
                    physics_ctx_post,
                    h_fused.device,
                    num_nodes=h_fused.size(0),
                    batch=fused_batch,
                    stt_dynamic=stt_dynamic,
                )
                post_action_evidence_state = self.state_builder.build_evidence_state(
                    post_action_observation_state,
                    phys_context_post,
                    t_sim,
                )
                post_action_evidence_state = self._maybe_refine_evidence_state(
                    post_action_evidence_state,
                    post_action_observation_state,
                    constraint_state,
                    batch_index=fused_batch,
                )
                post_action_valid_mask = ~constraint_state.no_resample_mask.view(-1).bool()
                if phys_context_post.feasible_mask is not None:
                    post_action_valid_mask = post_action_valid_mask & phys_context_post.feasible_mask.view(-1).bool()
                if not skip_reasoner_forward:
                    reasoner_state_post = {
                        'h_fused': h_fused,
                        'x_nav': x_nav_raw,
                        'observation_state': post_action_observation_state,
                        'inverse_indices': inverse_indices,
                        'causal_anchors': causal_anchors,
                        'accumulated_mask': acc_mask_local,
                        'memory_state': reasoner_memory_state,
                        'evidence_state': post_action_evidence_state,
                        'constraint_state': constraint_state,
                        'valid_mask': post_action_valid_mask,
                        'nav_state_summary': nav_state_summary,
                    }
                    with torch.no_grad():
                        reasoner_out_post = self.module_dispatcher.dispatch_reasoner(
                            reasoner_state_post,
                            temp_graph,
                            physics_ctx_post,
                        )
                        post_action_reasoner_logits = reasoner_out_post['logits']
                        post_action_reasoner_logits = self.state_updater.apply_constraint_masks(
                            post_action_reasoner_logits,
                            constraint_state,
                            fused_batch,
                        )
                if clean_nav_mainline:
                    post_action_mainline_semantics = self.metrics_flow.compute_mainline_semantics(
                        evidence_state=post_action_evidence_state,
                        constraint_state=constraint_state,
                        valid_mask=post_action_valid_mask,
                        batch_index=fused_batch,
                    )
                    final_mainline_semantics = {
                        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                        for key, value in post_action_mainline_semantics.items()
                    }
                    closure_now = post_action_mainline_semantics["closure"].view(-1).bool()
                    just_closed = closure_now & (~graph_success)
                    if just_closed.any():
                        current_budget_after = scatter_sum(
                            acc_mask_local.view(-1).float(),
                            fused_batch,
                            dim=0,
                            dim_size=curr_batch_size,
                        )
                        budget_to_closure = budget_to_closure.clone()
                        budget_to_closure[just_closed] = current_budget_after[just_closed]
                    new_success = closure_now
                    hit_before_t = hit_before_t | closure_now
                    graph_success = graph_success | closure_now
            
            # Store Trajectory
            hit_prob_surrogate = torch.zeros(curr_batch_size, device=h_fused.device) # Simplified
            
            selected_indices_tensor = final_step_indices
            if selected_indices_tensor is None:
                selected_indices_tensor = torch.empty(0, dtype=torch.long, device=h_fused.device)
            else:
                selected_indices_tensor = selected_indices_tensor.view(-1).long()

            fused_global_ids = static_ctx.get('fused_global_ids')
            if fused_global_ids is not None and selected_indices_tensor.numel() > 0:
                selected_global_ids = fused_global_ids.view(-1)[selected_indices_tensor]
            else:
                selected_global_ids = torch.empty(0, dtype=torch.long, device=h_fused.device)

            pre_action_constraint_state = None
            if isinstance(vis_reasoner_state, dict):
                pre_action_constraint_state = vis_reasoner_state.get('constraint_state')
            pre_action_confirmed_source = False
            if pre_action_constraint_state is not None:
                pre_action_confirmed_source = bool(
                    (pre_action_constraint_state.confirmed_source_mask.view(-1) > 0.5).any().item()
                )

            last_post_action_evidence_state = post_action_evidence_state
            last_post_action_valid_mask = (
                None if post_action_valid_mask is None else post_action_valid_mask.detach().clone()
            )

            step_record = {
                'reasoner_logits': logits_fused,
                'nav_action': nav_out.get('y_action'),
                'nav_value': nav_out.get('value'),
                'nav_aux_value': nav_out.get('aux_value'),
                'nav_log_prob': nav_out.get('selected_log_prob'),
                'nav_entropy': nav_out.get('policy_entropy'),
                'active_mask': active_mask_t,
                'is_hit': new_success.float(),
                'fused_batch': fused_batch,
                'fused_source_label': fused_source_label,
                'hit_prob_surrogate': hit_prob_surrogate,
                'nav_gain_loss': nav_gain_loss,
                'nav_reward': nav_reward,
                'nav_reward_bundle': nav_reward_bundle,
            }
            if not compact_training_trajectory:
                step_record.update({
                    'nav_logits': nav_logits,
                    'nav_probs': nav_out.get('nav_probs'),
                    'nav_candidates': nav_out.get('selected_indices'),
                    'action_policy_branch': routing_meta.get('action_policy_branch'),
                    'nav_out_consumed': bool(routing_meta.get('nav_out_consumed', False)),
                    'guardrail_mode': routing_meta.get('guardrail_mode', 'none'),
                    'proxy_contract_mode': routing_meta.get('proxy_contract_mode', 'none'),
                    'navigator_proposed_count': int(routing_meta.get('navigator_proposed_count', 0)),
                    'navigator_admitted_count': int(routing_meta.get('navigator_admitted_count', 0)),
                    'navigator_blocked_count': int(routing_meta.get('navigator_blocked_count', 0)),
                    'fallback_to_exploitation_count': int(routing_meta.get('fallback_to_exploitation_count', 0)),
                    'proxy_eligible_graph_count': int(routing_meta.get('proxy_eligible_graph_count', 0)),
                    'h_fused': h_fused,
                    'physics_ctx': physics_ctx,
                    'curr_edge_index': curr_edge_index,
                    'curr_edge_attr': curr_edge_attr,
                    'inverse_indices': inverse_indices,
                    'reasoner_input_state': vis_reasoner_state,
                    'constraint_state': constraint_state,
                    'verdict_payload': verdict_payload,
                    'constraint_update': constraint_update,
                    'selected_indices': selected_indices_tensor,
                    'selected_global_ids': selected_global_ids,
                    'action_executed': bool(selected_indices_tensor.numel() > 0),
                    'pre_action_valid_mask': valid_mask_final.detach().clone(),
                    'pre_action_confirmed_source': pre_action_confirmed_source,
                    'post_action_observation_state': post_action_observation_state,
                    'post_action_evidence_state': post_action_evidence_state,
                    'post_action_valid_mask': last_post_action_valid_mask,
                    'post_action_mainline_semantics': post_action_mainline_semantics,
                    'post_action_reasoner_logits': post_action_reasoner_logits,
                    'post_action_t_sim': t_sim.clone(),
                    'current_action_k': int(current_action_k),
                    'evidence_oracle_targets': evidence_oracle_targets,
                    'dynamic_state': {
                        't_sim': t_sim.clone(),
                        'constraint_state': constraint_state,
                    }
                })
                if resume_state_before_action is not None:
                    step_record['resume_state_before_action'] = resume_state_before_action
            trajectory_data.append(step_record)
            
            if inference_mode and hit_before_t.all():
                break
        
        # Finalize
        budget_used = scatter_sum(acc_mask_local.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
        if (
            trajectory_data
            and not inference_mode
            and has_y
            and self.reward_path_enabled
        ):
            final_step = trajectory_data[-1]
            final_evidence_state = final_step.get('post_action_evidence_state')
            if final_evidence_state is None:
                final_evidence_state = last_post_action_evidence_state
            final_valid_mask = final_step.get('post_action_valid_mask')
            if final_valid_mask is None:
                final_valid_mask = last_post_action_valid_mask
            if final_valid_mask is None:
                final_valid_mask = ~constraint_state.no_resample_mask.view(-1).bool()
            terminal_reward, terminal_bundle = self.metrics_flow.calculate_terminal_rewards(
                fused_batch=fused_batch,
                fused_source_label=fused_source_label,
                evidence_state_final=final_evidence_state,
                constraint_state_final=constraint_state,
                valid_mask_final=final_valid_mask,
                budget_used=budget_used,
                budget_max=float(max(max_episodes * max(sample_budget, 1), 1)),
                final_reasoner_logits=final_step.get('reasoner_logits'),
            )
            if terminal_reward is not None:
                if final_step.get('nav_reward') is None:
                    final_step['nav_reward'] = terminal_reward
                else:
                    final_step['nav_reward'] = final_step['nav_reward'] + terminal_reward
                bundle = final_step.get('nav_reward_bundle') or {}
                bundle.update(terminal_bundle)
                final_step['nav_reward_bundle'] = bundle
                self.metrics_flow.record_reward_bundle("terminal", terminal_bundle)
        if not inference_mode:
            self.metrics_flow.finalize_metrics(graph_success, t_sim, budget_used, total_poison_hits_per_graph)
        official_episode_metrics = self._aggregate_official_episode_metrics(trajectory_data, budget_used)
        if clean_nav_mainline:
            if initial_mainline_semantics is None:
                initial_mainline_semantics = self.metrics_flow.empty_mainline_semantics(curr_batch_size, h_fused.device)
            if final_mainline_semantics is None:
                final_mainline_semantics = self.metrics_flow.empty_mainline_semantics(curr_batch_size, h_fused.device)
            reward_cfg = self.metrics_flow._navigator_reward_cfg()
            decisive_budget_threshold = float(
                reward_cfg.get(
                    "early_closure_budget",
                    reward_cfg.get(
                        "closure_budget_threshold",
                        reward_cfg.get("early_decisive_budget", max(sample_budget, 1)),
                    ),
                )
            )
            budget_used_cpu = budget_used.detach().float().cpu()
            budget_to_closure_cpu = budget_to_closure.detach().float().cpu()
            core_mass_before = initial_mainline_semantics["core_mass"].detach().float().cpu()
            core_mass_after = final_mainline_semantics["core_mass"].detach().float().cpu()
            core_count_before = initial_mainline_semantics["core_count"].detach().float().cpu()
            core_count_after = final_mainline_semantics["core_count"].detach().float().cpu()
            candidate_count_before = initial_mainline_semantics["candidate_count"].detach().float().cpu()
            candidate_count_after = final_mainline_semantics["candidate_count"].detach().float().cpu()
            uncertainty_before = initial_mainline_semantics["uncertainty_core_mass"].detach().float().cpu()
            uncertainty_after = final_mainline_semantics["uncertainty_core_mass"].detach().float().cpu()
            final_closure = final_mainline_semantics["closure"].detach().float().cpu()
            budget_to_closure_masked = budget_to_closure_cpu.clone()
            budget_to_closure_masked[final_closure <= 0.5] = -1.0
            official_episode_metrics.update({
                "raw_closure_success": final_closure,
                "raw_decisive_closure": (
                    final_closure
                    * (budget_to_closure_masked >= 0.0).float()
                    * (budget_to_closure_masked <= decisive_budget_threshold).float()
                ),
                "raw_budget_to_closure": budget_to_closure_masked,
                "raw_core_mass_before": core_mass_before,
                "raw_core_mass_after": core_mass_after,
                "raw_core_mass_delta": core_mass_before - core_mass_after,
                "raw_core_size_before": core_count_before,
                "raw_core_size_after": core_count_after,
                "raw_core_size_delta": core_count_before - core_count_after,
                "raw_candidate_count_before": candidate_count_before,
                "raw_candidate_count_after": candidate_count_after,
                "raw_uncertainty_before": uncertainty_before,
                "raw_uncertainty_after": uncertainty_after,
                "raw_uncertainty_collapse": uncertainty_before - uncertainty_after,
                "raw_budget_efficiency": (core_mass_before - core_mass_after) / budget_used_cpu.clamp_min(1.0),
                "raw_evidence_gain_per_sample": (core_mass_before - core_mass_after) / budget_used_cpu.clamp_min(1.0),
            })
        rollout_profile_total = float(sum(rollout_profile.values()))
        rollout_profile_pct = {}
        rollout_profile_avg_step_ms = {}
        if profile_rollout and rollout_profile_steps > 0:
            rollout_profile_avg_step_ms = {
                key: (value * 1000.0 / float(rollout_profile_steps)) for key, value in rollout_profile.items()
            }
            if rollout_profile_total > 0:
                rollout_profile_pct = {
                    key.replace("_s", "_pct"): (value / rollout_profile_total) for key, value in rollout_profile.items()
                }
        episode_runner_total_s = None
        rollout_profile_other_s = None
        if episode_runner_start is not None:
            episode_runner_total_s = float(time.perf_counter() - episode_runner_start)
            rollout_profile_other_s = max(episode_runner_total_s - rollout_profile_total, 0.0)
            
        return {
            'classification': last_logits[inverse_indices] if last_logits is not None else torch.zeros(curr_batch_size, 2, device=h_fused.device), 
            'trajectory': trajectory_data,
            'final_dynamic_state': {
                'accumulated_mask': acc_mask_local,
                't_sim': t_sim,
                'constraint_state': constraint_state,
                'confirmed_non_source_mask': constraint_state.confirmed_non_source_mask,
                'confirmed_source_mask': constraint_state.confirmed_source_mask,
                'sampled_mask': constraint_state.sampled_mask,
                'no_resample_mask': constraint_state.no_resample_mask,
            },
            # [Cleanup Phase C] Removed debug_sentinel
            'probe_b_metrics': self.metrics_flow.probe_b_metrics,
            'step_metrics': {
                'success': graph_success.float().mean().item(),
                'steps_taken': t_sim.mean().item(), 
                'budget_used': budget_used.mean().item(),
                'raw_success': graph_success.float(),
                'raw_steps': t_sim, 
                'raw_budget': budget_used,
                'raw_predict_hit': predict_hit_at_1.float(),
                'raw_predict_hit_5': predict_hit_at_5.float(),
                'raw_predict_hit_valid': predict_hit_valid.float(),
                'raw_rounds': steps_taken, 
                'raw_max_hit_prob': max_hit_prob,
                'episode_runner_total_s': episode_runner_total_s,
                'rollout_profile_total_s': rollout_profile_total,
                'rollout_profile_other_s': rollout_profile_other_s,
                'rollout_profile_s': rollout_profile,
                'rollout_profile_pct': rollout_profile_pct,
                'rollout_profile_avg_step_ms': rollout_profile_avg_step_ms,
                'rollout_profile_steps': rollout_profile_steps,
                **official_episode_metrics,
            }
        }
