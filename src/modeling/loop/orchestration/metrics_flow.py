import torch
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch_geometric.utils import softmax as gnn_softmax

class MetricsFlow:
    """
    Responsibilities:
    - Calculate step stats (entropy, max prob)
    - Collect probe metrics
    - Calculate losses and rewards
    - Track success and hit rates
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.probe_b_metrics = {}
        self.collect_detailed_step_metrics = bool(
            getattr(getattr(cfg, "training", None), "collect_detailed_step_metrics", True)
        )

    def _cfg_get(self, cfg_obj, key, default):
        if cfg_obj is None:
            return default
        if isinstance(cfg_obj, dict):
            return cfg_obj.get(key, default)
        return getattr(cfg_obj, key, default)

    def _navigator_reward_cfg(self):
        loss_cfg = getattr(self.cfg, "loss", None)
        params_cfg = self._cfg_get(loss_cfg, "params", {})
        reward_cfg = self._cfg_get(params_cfg, "navigator_reward_contract", {})
        if reward_cfg is None:
            reward_cfg = {}
        return reward_cfg

    def _clean_mainline_reward_mode(self, reward_mode=None):
        if reward_mode is None:
            reward_mode = str(self._navigator_reward_cfg().get("mode", ""))
        return reward_mode in {
            "navigator_vnext_evidence_mainline_v1",
            "navigator_official_phi_v1",
            "navigator_vnext_official_phi_v1",
            "navigator_terminal_mrr_v1",
            "navigator_terminal_rank_closure_v1",
        }

    def empty_mainline_semantics(self, graph_count, device):
        zeros = torch.zeros(int(graph_count), device=device, dtype=torch.float32)
        return {
            "core_mass": zeros.clone(),
            "core_count": zeros.clone(),
            "candidate_count": zeros.clone(),
            "uncertainty_core_mass": zeros.clone(),
            "closure": zeros.clone(),
        }

    def compute_mainline_semantics(self, evidence_state, constraint_state, valid_mask, batch_index):
        batch_flat = batch_index.view(-1).long()
        graph_count = int(batch_flat.max().item()) + 1 if batch_flat.numel() > 0 else 0
        device = batch_flat.device
        if graph_count == 0:
            return self.empty_mainline_semantics(0, device)

        valid_mask = valid_mask.view(-1).bool()
        reference = valid_mask.float()
        reward_cfg = self._navigator_reward_cfg()
        support = self._metric_tensor(evidence_state, "support_score", reference, default=0.0).clamp_min(0.0)
        uncertainty = self._metric_tensor(evidence_state, "uncertainty_gap", reference, default=0.0).clamp_min(0.0)
        suspect = self._metric_tensor(evidence_state, "suspect_pool", reference, default=0.0).clamp(0.0, 1.0)
        not_ruled_out = self._metric_tensor(evidence_state, "not_ruled_out_gate", reference, default=1.0).clamp(0.0, 1.0)
        alpha_q_suspect = float(reward_cfg.get("alpha_q_suspect", 0.35))

        confirmed_non = torch.zeros_like(valid_mask)
        confirmed_src = torch.zeros_like(valid_mask)
        if constraint_state is not None:
            confirmed_non = self._metric_tensor(constraint_state, "confirmed_non_source_mask", reference, default=0.0) > 0.5
            confirmed_src = self._metric_tensor(constraint_state, "confirmed_source_mask", reference, default=0.0) > 0.5

        admissible_mask = valid_mask & (~confirmed_non) & (~confirmed_src)
        q = not_ruled_out * (support + alpha_q_suspect * suspect)
        semantics = self._official_core_semantics(
            q=q,
            uncertainty=uncertainty,
            suspect=suspect,
            not_ruled_out=not_ruled_out,
            valid_mask=admissible_mask,
            batch_index=batch_flat,
            curr_batch_size=graph_count,
        )
        return {
            "core_mass": semantics["core_mass"],
            "core_count": semantics["core_count"],
            "candidate_count": semantics["candidate_count"],
            "uncertainty_core_mass": semantics["uncertainty_mass"],
            "closure": semantics["closure"],
        }

    def _single_graph_metrics(self, logits, labels, plausible_delta=1.0):
        logits = logits.detach().float().view(-1)
        labels = labels.detach().float().view(-1)
        finite_mask = torch.isfinite(logits)
        if not bool(finite_mask.any()):
            return {
                "valid_case": False,
                "true_source_rank": None,
                "true_source_prob": 0.0,
                "mrr": 0.0,
                "top1_hit": False,
                "top3_hit": False,
                "top5_hit": False,
                "top1_margin": 0.0,
                "plausible_count": 0.0,
                "finite_candidate_count": 0,
            }

        true_mask = labels > 0.5
        if not bool(true_mask.any()):
            return {
                "valid_case": False,
                "true_source_rank": None,
                "true_source_prob": 0.0,
                "mrr": 0.0,
                "top1_hit": False,
                "top3_hit": False,
                "top5_hit": False,
                "top1_margin": 0.0,
                "plausible_count": 0.0,
                "finite_candidate_count": int(finite_mask.sum().item()),
            }

        safe_logits = logits.clone()
        safe_logits[~finite_mask] = -float("inf")
        probs = torch.zeros_like(safe_logits)
        probs[finite_mask] = torch.softmax(safe_logits[finite_mask], dim=0)

        sorted_idx = torch.argsort(safe_logits, descending=True)
        true_positions = (true_mask[sorted_idx]).nonzero(as_tuple=True)[0]
        rank_true = int(true_positions.min().item() + 1)
        top_vals = torch.topk(safe_logits[finite_mask], k=min(2, int(finite_mask.sum().item()))).values
        top1_margin = 0.0
        if top_vals.numel() >= 2:
            top1_margin = float(top_vals[0].item() - top_vals[1].item())
        plausible_floor = float(top_vals[0].item() - float(plausible_delta))
        plausible_count = float(((safe_logits >= plausible_floor) & finite_mask).float().sum().item())
        true_source_prob = float(probs[true_mask].max().item())
        return {
            "valid_case": True,
            "true_source_rank": rank_true,
            "true_source_prob": true_source_prob,
            "mrr": 1.0 / float(rank_true),
            "top1_hit": bool(rank_true <= 1),
            "top3_hit": bool(rank_true <= 3),
            "top5_hit": bool(rank_true <= 5),
            "top1_margin": top1_margin,
            "plausible_count": plausible_count,
            "finite_candidate_count": int(finite_mask.sum().item()),
        }

    def _single_graph_score_metrics(self, scores, labels, plausible_delta=0.25):
        scores = scores.detach().float().view(-1)
        labels = labels.detach().float().view(-1)
        finite_mask = torch.isfinite(scores)
        if not bool(finite_mask.any()):
            return {
                "valid_case": False,
                "true_source_rank": None,
                "true_source_value": 0.0,
                "top1_hit": False,
                "top3_hit": False,
                "top5_hit": False,
                "top1_margin": 0.0,
                "plausible_count": 0.0,
            }

        true_mask = labels > 0.5
        if not bool(true_mask.any()):
            return {
                "valid_case": False,
                "true_source_rank": None,
                "true_source_value": 0.0,
                "top1_hit": False,
                "top3_hit": False,
                "top5_hit": False,
                "top1_margin": 0.0,
                "plausible_count": 0.0,
            }

        safe_scores = scores.clone()
        safe_scores[~finite_mask] = -float("inf")
        sorted_idx = torch.argsort(safe_scores, descending=True)
        true_positions = (true_mask[sorted_idx]).nonzero(as_tuple=True)[0]
        rank_true = int(true_positions.min().item() + 1)
        top_vals = torch.topk(safe_scores[finite_mask], k=min(2, int(finite_mask.sum().item()))).values
        top1_margin = 0.0
        if top_vals.numel() >= 2:
            top1_margin = float(top_vals[0].item() - top_vals[1].item())
        plausible_floor = float(top_vals[0].item() - float(plausible_delta))
        plausible_count = float(((safe_scores >= plausible_floor) & finite_mask).float().sum().item())
        true_source_value = float(safe_scores[true_mask].max().item())
        return {
            "valid_case": True,
            "true_source_rank": rank_true,
            "true_source_value": true_source_value,
            "top1_hit": bool(rank_true <= 1),
            "top3_hit": bool(rank_true <= 3),
            "top5_hit": bool(rank_true <= 5),
            "top1_margin": top1_margin,
            "plausible_count": plausible_count,
        }

    def _metric_tensor(self, evidence_state, key, reference, default=0.0):
        if evidence_state is None:
            return torch.full_like(reference.view(-1).float(), float(default))
        if hasattr(evidence_state, key):
            value = getattr(evidence_state, key)
        elif isinstance(evidence_state, dict):
            value = evidence_state.get(key)
        else:
            value = None
        if value is None:
            return torch.full_like(reference.view(-1).float(), float(default))
        return value.detach().float().view(-1)

    def _core_mass(self, logits, valid_mask, uncertainty, not_ruled_out):
        logits = logits.detach().float().view(-1)
        valid_mask = valid_mask.detach().view(-1).bool() & torch.isfinite(logits)
        uncertainty = uncertainty.detach().float().view(-1)
        not_ruled_out = not_ruled_out.detach().float().view(-1)
        if not bool(valid_mask.any()):
            return 0.0
        masked_logits = logits.clone()
        masked_logits[~valid_mask] = -float("inf")
        top1_val = float(masked_logits[valid_mask].max().item())
        plausible_floor = top1_val - 1.0
        safe_logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, plausible_floor))
        plausible_weight = ((safe_logits - plausible_floor) / 1.0).clamp(0.0, 1.0)
        core = uncertainty * not_ruled_out * plausible_weight
        core_mass = float(core[valid_mask].mean().item()) if bool(valid_mask.any()) else 0.0
        return core_mass if torch.isfinite(torch.tensor(core_mass)) else 0.0

    def _core_weight_mass(self, q, uncertainty, suspect, not_ruled_out, valid_mask):
        q = q.detach().float().view(-1)
        uncertainty = uncertainty.detach().float().view(-1).clamp_min(0.0)
        suspect = suspect.detach().float().view(-1).clamp(0.0, 1.0)
        not_ruled_out = not_ruled_out.detach().float().view(-1).clamp(0.0, 1.0)
        valid_mask = valid_mask.detach().view(-1).bool() & torch.isfinite(q)
        if not bool(valid_mask.any()):
            return 0.0
        reward_cfg = self._navigator_reward_cfg()
        core_plausible_delta = float(reward_cfg.get("core_plausible_delta", 0.25))
        core_not_ruled_threshold = float(reward_cfg.get("core_not_ruled_threshold", 0.25))
        core_suspect_threshold = float(reward_cfg.get("core_suspect_threshold", 0.25))
        weight_alpha = float(reward_cfg.get("alpha_suspect", 0.5))
        weight_max = float(reward_cfg.get("w_max", 2.0))
        q_valid = q[valid_mask]
        top_q = float(q_valid.max().item()) if q_valid.numel() > 0 else 0.0
        plausible_mask = q >= (top_q - core_plausible_delta)
        core_mask = valid_mask & plausible_mask & (
            (not_ruled_out >= core_not_ruled_threshold)
            | (suspect >= core_suspect_threshold)
        )
        if not bool(core_mask.any()):
            core_mask = valid_mask
        weights = (uncertainty + weight_alpha * suspect).clamp(0.0, weight_max)
        return float(weights[core_mask].sum().item())

    def _official_core_semantics(
        self,
        q,
        uncertainty,
        suspect,
        not_ruled_out,
        valid_mask,
        batch_index,
        curr_batch_size,
    ):
        q = q.detach().float().view(-1)
        uncertainty = uncertainty.detach().float().view(-1).clamp_min(0.0)
        suspect = suspect.detach().float().view(-1).clamp(0.0, 1.0)
        not_ruled_out = not_ruled_out.detach().float().view(-1).clamp(0.0, 1.0)
        valid_mask = valid_mask.detach().view(-1).bool() & torch.isfinite(q)
        batch_index = batch_index.view(-1).long()

        reward_cfg = self._navigator_reward_cfg()
        core_plausible_delta = float(reward_cfg.get("core_plausible_delta", 0.25))
        core_not_ruled_threshold = float(reward_cfg.get("core_not_ruled_threshold", 0.25))
        alpha_core = float(reward_cfg.get("alpha_core", reward_cfg.get("alpha_suspect", 0.5)))
        weight_max = float(reward_cfg.get("w_max", 2.0))
        closure_mass_threshold = float(
            reward_cfg.get("closure_core_mass_threshold", reward_cfg.get("closure_mass_threshold", 0.25))
        )
        closure_size_threshold = float(
            reward_cfg.get("closure_core_size_threshold", reward_cfg.get("closure_size_threshold", 2.0))
        )

        candidate_mask = valid_mask & (not_ruled_out >= core_not_ruled_threshold)
        masked_q = q.masked_fill(~candidate_mask, -float("inf"))
        top_q = scatter_max(masked_q, batch_index, dim=0, dim_size=curr_batch_size)[0]
        top_q = torch.where(torch.isfinite(top_q), top_q, torch.zeros_like(top_q))
        plausible_floor = top_q[batch_index] - core_plausible_delta
        core_mask = candidate_mask & (q >= plausible_floor)
        core_weights = (uncertainty + alpha_core * suspect).clamp(0.0, weight_max) * core_mask.float()

        candidate_count = scatter_sum(candidate_mask.float(), batch_index, dim=0, dim_size=curr_batch_size)
        core_count = scatter_sum(core_mask.float(), batch_index, dim=0, dim_size=curr_batch_size)
        core_mass = scatter_sum(core_weights, batch_index, dim=0, dim_size=curr_batch_size)
        uncertainty_mass = scatter_sum(uncertainty * core_mask.float(), batch_index, dim=0, dim_size=curr_batch_size)
        closure = (
            (core_mass <= closure_mass_threshold)
            | (core_count <= closure_size_threshold)
        ).float()

        return {
            "candidate_mask": candidate_mask,
            "core_mask": core_mask,
            "core_weights": core_weights,
            "candidate_count": candidate_count,
            "core_count": core_count,
            "core_mass": core_mass,
            "uncertainty_mass": uncertainty_mass,
            "closure": closure,
        }

    def _core_weight_mass_per_graph(self, q, uncertainty, suspect, not_ruled_out, valid_mask, batch_index, graph_count):
        q = q.detach().float().view(-1)
        uncertainty = uncertainty.detach().float().view(-1).clamp_min(0.0)
        suspect = suspect.detach().float().view(-1).clamp(0.0, 1.0)
        not_ruled_out = not_ruled_out.detach().float().view(-1).clamp(0.0, 1.0)
        valid_mask = valid_mask.detach().view(-1).bool() & torch.isfinite(q)
        batch_flat = batch_index.view(-1).long()
        if graph_count <= 0 or batch_flat.numel() == 0:
            return torch.zeros(0, device=q.device, dtype=torch.float32)

        reward_cfg = self._navigator_reward_cfg()
        core_plausible_delta = float(reward_cfg.get("core_plausible_delta", 0.25))
        core_not_ruled_threshold = float(reward_cfg.get("core_not_ruled_threshold", 0.25))
        core_suspect_threshold = float(reward_cfg.get("core_suspect_threshold", 0.25))
        weight_alpha = float(reward_cfg.get("alpha_suspect", 0.5))
        weight_max = float(reward_cfg.get("w_max", 2.0))

        masked_q = q.clone()
        masked_q[~valid_mask] = -float("inf")
        top_q = scatter_max(masked_q, batch_flat, dim=0, dim_size=graph_count)[0]
        top_q = torch.where(torch.isfinite(top_q), top_q, torch.zeros_like(top_q))

        plausible_mask = q >= (top_q[batch_flat] - core_plausible_delta)
        core_mask = valid_mask & plausible_mask & (
            (not_ruled_out >= core_not_ruled_threshold)
            | (suspect >= core_suspect_threshold)
        )
        has_core = scatter_sum(core_mask.float(), batch_flat, dim=0, dim_size=graph_count) > 0.5
        effective_core_mask = torch.where(has_core[batch_flat], core_mask, valid_mask)
        weights = (uncertainty + weight_alpha * suspect).clamp(0.0, weight_max)
        return scatter_sum(weights * effective_core_mask.float(), batch_flat, dim=0, dim_size=graph_count)

    def _evidence_contract_terms(
        self,
        q,
        uncertainty,
        suspect,
        not_ruled_out,
        valid_mask,
        label_slice=None,
    ):
        q = q.detach().float().view(-1)
        uncertainty = uncertainty.detach().float().view(-1).clamp_min(0.0)
        suspect = suspect.detach().float().view(-1).clamp(0.0, 1.0)
        not_ruled_out = not_ruled_out.detach().float().view(-1).clamp(0.0, 1.0)
        valid_mask = valid_mask.detach().view(-1).bool() & torch.isfinite(q)
        label_slice = None if label_slice is None else label_slice.detach().float().view(-1)

        reward_cfg = self._navigator_reward_cfg()
        tau_rank = float(reward_cfg.get("tau_rank", 0.25))
        core_plausible_delta = float(reward_cfg.get("core_plausible_delta", 0.25))
        core_not_ruled_threshold = float(reward_cfg.get("core_not_ruled_threshold", 0.25))
        core_suspect_threshold = float(reward_cfg.get("core_suspect_threshold", 0.25))
        weight_alpha = float(reward_cfg.get("alpha_suspect", 0.5))
        weight_max = float(reward_cfg.get("w_max", 2.0))

        if not bool(valid_mask.any()):
            zeros = torch.zeros_like(q)
            return {
                "q": q,
                "candidate_mask": valid_mask,
                "core_mask": valid_mask,
                "core_weights": zeros,
                "uncertainty_core_mean": 0.0,
                "H": 0.0,
                "source_rank": None,
                "candidate_count": 0.0,
                "core_count": 0.0,
                "core_mass": 0.0,
                "source_value": 0.0,
            }

        q_valid = q.clone()
        q_valid[~valid_mask] = -float("inf")
        top_q = float(q_valid[valid_mask].max().item()) if bool(valid_mask.any()) else 0.0
        plausible_mask = q >= (top_q - core_plausible_delta)
        candidate_mask = valid_mask & (not_ruled_out > 1e-6)
        core_mask = candidate_mask & plausible_mask & (
            (not_ruled_out >= core_not_ruled_threshold)
            | (suspect >= core_suspect_threshold)
        )
        if not bool(core_mask.any()):
            core_mask = candidate_mask

        core_weights = (uncertainty + weight_alpha * suspect).clamp(0.0, weight_max) * core_mask.float()
        core_count = float(core_mask.float().sum().item())
        uncertainty_core_mean = 0.0
        if core_count > 0.0:
            uncertainty_core_mean = float((uncertainty * core_mask.float()).sum().item() / core_count)

        H = 0.0
        source_rank = None
        source_value = 0.0
        true_mask = None
        if label_slice is not None:
            true_mask = label_slice > 0.5
            if bool(true_mask.any()):
                source_value = float(q[true_mask].max().item())
                outranking = torch.sigmoid((q - source_value) / max(tau_rank, 1e-6))
                competitor_mask = candidate_mask & (~true_mask)
                H = float(outranking[competitor_mask].sum().item())

                sort_scores = q_valid.clone()
                sorted_idx = torch.argsort(sort_scores, descending=True)
                true_positions = (true_mask[sorted_idx]).nonzero(as_tuple=True)[0]
                source_rank = int(true_positions.min().item() + 1) if true_positions.numel() > 0 else None

        return {
            "q": q,
            "candidate_mask": candidate_mask,
            "core_mask": core_mask,
            "core_weights": core_weights,
            "uncertainty_core_mean": uncertainty_core_mean,
            "H": H,
            "source_rank": source_rank,
            "candidate_count": float(candidate_mask.float().sum().item()),
            "core_count": core_count,
            "core_mass": float(core_weights.sum().item()),
            "source_value": source_value,
        }

    def _evidence_q(self, support, suspect, not_ruled_out):
        reward_cfg = self._navigator_reward_cfg()
        alpha_q_suspect = float(reward_cfg.get("alpha_q_suspect", 0.35))
        support_term = support.detach().float().view(-1).clamp_min(0.0)
        suspect_term = suspect.detach().float().view(-1).clamp(0.0, 1.0)
        gate = not_ruled_out.detach().float().view(-1).clamp(0.0, 1.0)
        return gate * (support_term + alpha_q_suspect * suspect_term)

    def summarize_official_core(
        self,
        *,
        evidence_state,
        constraint_state,
        valid_mask,
        fused_batch,
        curr_batch_size,
        reference,
        label_slice=None,
    ):
        valid_mask = valid_mask.view(-1).bool()
        support = self._metric_tensor(evidence_state, "support_score", reference, default=0.0)
        uncertainty = self._metric_tensor(evidence_state, "uncertainty_gap", reference, default=0.0)
        suspect = self._metric_tensor(evidence_state, "suspect_pool", reference, default=0.0)
        not_ruled_out = self._metric_tensor(evidence_state, "not_ruled_out_gate", reference, default=1.0)
        q = self._evidence_q(support, suspect, not_ruled_out)

        if constraint_state is not None:
            confirmed_non = constraint_state.confirmed_non_source_mask.view(-1) > 0.5
            confirmed_src = constraint_state.confirmed_source_mask.view(-1) > 0.5
            valid_mask = valid_mask & (~confirmed_non) & (~confirmed_src)

        reward_cfg = self._navigator_reward_cfg()
        closure_mass_threshold = float(
            reward_cfg.get("closure_core_mass_threshold", reward_cfg.get("closure_mass_threshold", 0.50))
        )
        closure_size_threshold = float(
            reward_cfg.get("closure_core_size_threshold", reward_cfg.get("closure_size_threshold", 2.0))
        )

        core_mass = []
        core_count = []
        candidate_count = []
        uncertainty_core = []
        closure_reached = []
        source_rank = []
        for graph_idx in range(int(curr_batch_size)):
            node_mask = fused_batch.view(-1) == int(graph_idx)
            label_graph = None
            if label_slice is not None:
                label_graph = label_slice.view(-1)[node_mask]
            terms = self._evidence_contract_terms(
                q[node_mask],
                uncertainty[node_mask],
                suspect[node_mask],
                not_ruled_out[node_mask],
                valid_mask[node_mask],
                label_graph,
            )
            mass = float(terms["core_mass"])
            size = float(terms["core_count"])
            candidates = float(terms["candidate_count"])
            core_mass.append(mass)
            core_count.append(size)
            candidate_count.append(candidates)
            uncertainty_core.append(float(terms["uncertainty_core_mean"]))
            closure_reached.append(float((mass <= closure_mass_threshold) or (size <= closure_size_threshold)))
            source_rank.append(float(-1 if terms["source_rank"] is None else terms["source_rank"]))

        return {
            "q": q,
            "valid_mask": valid_mask,
            "core_mass": torch.tensor(core_mass, dtype=torch.float32, device=reference.device),
            "core_count": torch.tensor(core_count, dtype=torch.float32, device=reference.device),
            "candidate_count": torch.tensor(candidate_count, dtype=torch.float32, device=reference.device),
            "uncertainty_core": torch.tensor(uncertainty_core, dtype=torch.float32, device=reference.device),
            "closure_reached": torch.tensor(closure_reached, dtype=torch.float32, device=reference.device),
            "legacy_source_rank": torch.tensor(source_rank, dtype=torch.float32, device=reference.device),
        }

    def _official_core_state(
        self,
        q,
        uncertainty,
        suspect,
        not_ruled_out,
        valid_mask,
    ):
        q = q.detach().float().view(-1)
        uncertainty = uncertainty.detach().float().view(-1).clamp_min(0.0)
        suspect = suspect.detach().float().view(-1).clamp(0.0, 1.0)
        not_ruled_out = not_ruled_out.detach().float().view(-1).clamp(0.0, 1.0)
        valid_mask = valid_mask.detach().view(-1).bool() & torch.isfinite(q)

        reward_cfg = self._navigator_reward_cfg()
        core_plausible_delta = float(reward_cfg.get("core_plausible_delta", 0.25))
        core_not_ruled_threshold = float(reward_cfg.get("core_not_ruled_threshold", 0.25))
        alpha_core = float(reward_cfg.get("alpha_core", reward_cfg.get("alpha_suspect", 0.5)))
        weight_max = float(reward_cfg.get("w_max", 2.0))

        candidate_mask = valid_mask & (not_ruled_out >= core_not_ruled_threshold)
        if bool(candidate_mask.any()):
            top_q = float(q[candidate_mask].max().item())
            plausible_mask = q >= (top_q - core_plausible_delta)
            core_mask = candidate_mask & plausible_mask
        else:
            core_mask = torch.zeros_like(candidate_mask)

        weights = (uncertainty + alpha_core * suspect).clamp(0.0, weight_max) * core_mask.float()
        uncertainty_mass = float((uncertainty * core_mask.float()).sum().item())
        return {
            "candidate_mask": candidate_mask,
            "core_mask": core_mask,
            "weights": weights,
            "phi": float(weights.sum().item()),
            "core_size": float(core_mask.float().sum().item()),
            "uncertainty_mass": uncertainty_mass,
        }

    def _official_closure_flags(self, core_mass: float, core_size: float):
        reward_cfg = self._navigator_reward_cfg()
        mass_threshold = float(
            reward_cfg.get("closure_mass_threshold", reward_cfg.get("terminal_core_mass_threshold", 1.0))
        )
        size_threshold = float(
            reward_cfg.get("closure_size_threshold", reward_cfg.get("terminal_core_size_threshold", 2.0))
        )
        mass_closed = float(core_mass <= mass_threshold)
        size_closed = float(core_size <= size_threshold)
        closure = float((mass_closed > 0.5) or (size_closed > 0.5))
        return mass_closed, size_closed, closure

    def _official_core_state_batch(
        self,
        q,
        uncertainty,
        suspect,
        not_ruled_out,
        valid_mask,
        batch_flat,
        graph_count,
    ):
        """
        Batched equivalent of _official_core_state.

        This preserves the official phi semantics while removing the per-graph
        Python loop from the hot reward path.
        """
        q = q.detach().float().view(-1)
        uncertainty = uncertainty.detach().float().view(-1).clamp_min(0.0)
        suspect = suspect.detach().float().view(-1).clamp(0.0, 1.0)
        not_ruled_out = not_ruled_out.detach().float().view(-1).clamp(0.0, 1.0)
        valid_mask = valid_mask.detach().view(-1).bool() & torch.isfinite(q)
        batch_flat = batch_flat.view(-1).long()
        graph_count = int(graph_count)

        reward_cfg = self._navigator_reward_cfg()
        core_plausible_delta = float(reward_cfg.get("core_plausible_delta", 0.25))
        core_not_ruled_threshold = float(reward_cfg.get("core_not_ruled_threshold", 0.25))
        alpha_core = float(reward_cfg.get("alpha_core", reward_cfg.get("alpha_suspect", 0.5)))
        weight_max = float(reward_cfg.get("w_max", 2.0))
        closure_mass_threshold = float(
            reward_cfg.get("closure_mass_threshold", reward_cfg.get("terminal_core_mass_threshold", 1.0))
        )
        closure_size_threshold = float(
            reward_cfg.get("closure_size_threshold", reward_cfg.get("terminal_core_size_threshold", 2.0))
        )

        candidate_mask = valid_mask & (not_ruled_out >= core_not_ruled_threshold)
        masked_q = q.clone()
        masked_q[~candidate_mask] = -float("inf")
        top_q = scatter_max(masked_q, batch_flat, dim=0, dim_size=graph_count)[0]
        top_q = torch.where(torch.isfinite(top_q), top_q, torch.zeros_like(top_q))
        core_floor = top_q[batch_flat] - core_plausible_delta
        core_mask = candidate_mask & (q >= core_floor)
        weights = (uncertainty + alpha_core * suspect).clamp(0.0, weight_max) * core_mask.float()

        core_mass = scatter_sum(weights, batch_flat, dim=0, dim_size=graph_count)
        core_size = scatter_sum(core_mask.float(), batch_flat, dim=0, dim_size=graph_count)
        candidate_count = scatter_sum(candidate_mask.float(), batch_flat, dim=0, dim_size=graph_count)
        uncertainty_mass = scatter_sum(uncertainty * core_mask.float(), batch_flat, dim=0, dim_size=graph_count)
        closure = ((core_mass <= closure_mass_threshold) | (core_size <= closure_size_threshold)).float()

        return {
            "candidate_mask": candidate_mask,
            "core_mask": core_mask,
            "weights": weights,
            "phi": core_mass,
            "core_size": core_size,
            "uncertainty_mass": uncertainty_mass,
            "candidate_count": candidate_count,
            "closure": closure,
        }

    def start_episode(self, inference_mode, stepper_id, mask_in):
        if not inference_mode:
             self.probe_b_metrics["meta/stepper_enter"] = 1
             self.probe_b_metrics["meta/stepper_id"] = stepper_id
             
             if mask_in is not None:
                 self.probe_b_metrics["probeB/bridge_mask_sum"] = mask_in.sum().item()

        # [Evidence Bias Metrics]
        # This part requires access to evidence_state which is not passed to start_episode.
        # It's better handled in track_evidence_stats or step-wise logging.
        # Removing from here to avoid variable undefined error.
        pass

    def calculate_step_stats(self, logits, fused_batch, energy=None):
        from torch_geometric.utils import softmax as gnn_softmax
        logit_max_abs = getattr(self.cfg.training, 'logit_max_abs', 20.0)
        eps_prob = getattr(self.cfg.training, 'eps_prob', 1e-8)
        
        logits = torch.clamp(logits, min=-logit_max_abs, max=logit_max_abs)
        probs = gnn_softmax(logits, fused_batch)
        log_p = torch.log(probs + eps_prob)
        entropy_per_node = -probs * log_p
        entropy = scatter_mean(entropy_per_node.sum(dim=-1, keepdim=True), fused_batch, dim=0)
        
        m1, _ = scatter_max(probs.view(-1), fused_batch, dim=0)
        margin = m1 - scatter_mean(probs.view(-1), fused_batch, dim=0)
        
        energy_mean = torch.zeros_like(entropy)
        if energy is not None:
            energy_mean = scatter_mean(energy, fused_batch, dim=0)
            
        return {
            'entropy': entropy.mean().item(),
            'max_prob': m1.mean().item(),
            'top1_margin': margin.mean().item(),
            'race_conflict_mean': energy_mean.mean().item()
        }

    def compute_step_metrics(self, step, logits_fused, fused_batch, fused_source_label, active_mask_t, curr_batch_size):
        """Compute Loss and Hit@K for monitoring"""
        if not self.collect_detailed_step_metrics:
            return 0.0, 0.0

        from torch_geometric.utils import softmax as gnn_softmax
        
        # 1. Step Loss
        step_log_probs = torch.log(gnn_softmax(logits_fused.view(-1), fused_batch) + 1e-9)
        step_loss_per_graph = scatter_sum(-step_log_probs * fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
        step_loss_val = step_loss_per_graph.mean().item()
        
        # 2. Hit@K
        _, top1_idx_step = scatter_max(logits_fused.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
        
        # Hit@1 Check
        has_source_step, _ = scatter_max(fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)
        valid_graphs_step = (has_source_step > 0.5)
        
        hit1_rate = 0.0
        if valid_graphs_step.any():
            hit1_check = torch.zeros(curr_batch_size, dtype=torch.bool, device=logits_fused.device)
            valid_top1 = (top1_idx_step >= 0) & (top1_idx_step < fused_source_label.size(0))
            if valid_top1.any():
                hit1_check[valid_top1] = (fused_source_label[top1_idx_step[valid_top1]].view(-1) > 0.5)
            
            hit1_rate = (hit1_check & valid_graphs_step).float().sum() / valid_graphs_step.float().sum()
            hit1_rate = hit1_rate.item()

        self.probe_b_metrics[f"step_metrics/step_{step}_loss"] = step_loss_val
        self.probe_b_metrics[f"step_metrics/step_{step}_hit1"] = hit1_rate
        
        return step_loss_val, hit1_rate

    def calculate_rewards(
        self,
        logits_next,
        fused_batch,
        fused_source_label,
        prob_before_target,
        curr_batch_size,
        y_action,
        logits_before=None,
        observation_state_before=None,
        evidence_state_before=None,
        evidence_state_next=None,
        constraint_state_before=None,
        constraint_state_after=None,
        valid_mask_before=None,
        valid_mask_after=None,
        selection_mask=None,
        budget_used_before=None,
        budget_used_after=None,
    ):
        """Calculate step reward and an auditable reward bundle."""
        probs_next = gnn_softmax(logits_next.view(-1), fused_batch)
        prob_after_target = scatter_sum(probs_next * fused_source_label.view(-1), fused_batch, dim=0, dim_size=curr_batch_size)

        reward_cfg = self._navigator_reward_cfg()
        reward_mode = str(reward_cfg.get("mode", "prob_gain_v0"))

        step_cost = float(reward_cfg.get("step_penalty", 0.05 if reward_mode == "prob_gain_v0" else 0.01))
        gain_reward = (prob_after_target - prob_before_target).detach() - step_cost
        reward_bundle = {
            "mode": reward_mode,
            "r_total": gain_reward.detach().cpu(),
            "r_prob": (prob_after_target - prob_before_target).detach().cpu(),
            "step_penalty": torch.full_like(prob_after_target.detach().cpu(), step_cost),
        }

        if reward_mode == "navigator_terminal_mrr_v1":
            zero_reward = torch.zeros(int(curr_batch_size), device=logits_next.device, dtype=torch.float32)
            zero_cpu = zero_reward.detach().cpu()
            gain_reward = zero_reward
            reward_bundle = {
                "mode": reward_mode,
                "r_total": zero_cpu,
                "step_penalty": zero_cpu.clone(),
                "terminal_only": True,
            }

        if reward_mode == "navigator_terminal_rank_closure_v1":
            penalty_reward = torch.full(
                (int(curr_batch_size),),
                -float(step_cost),
                device=logits_next.device,
                dtype=torch.float32,
            )
            gain_reward = penalty_reward
            reward_bundle = {
                "mode": reward_mode,
                "r_total": penalty_reward.detach().cpu(),
                "step_penalty": torch.full((int(curr_batch_size),), float(step_cost), dtype=torch.float32),
                "terminal_only": True,
            }

        if reward_mode in {"navigator_official_phi_v1", "navigator_vnext_official_phi_v1", "navigator_bounded_coupling_repair_v1"} and evidence_state_before is not None and evidence_state_next is not None:
            lambda_phi = float(reward_cfg.get("lambda_phi", 1.0))

            if valid_mask_before is None:
                valid_mask_before = torch.isfinite(logits_next.view(-1))
            else:
                valid_mask_before = valid_mask_before.view(-1).bool()
            if valid_mask_after is None:
                valid_mask_after = valid_mask_before.clone()
            else:
                valid_mask_after = valid_mask_after.view(-1).bool()
            if selection_mask is None:
                selection_mask = torch.zeros_like(valid_mask_before).bool()
            else:
                selection_mask = selection_mask.view(-1) > 0.5

            support_before = self._metric_tensor(evidence_state_before, "support_score", logits_next, default=0.0)
            support_next = self._metric_tensor(evidence_state_next, "support_score", logits_next, default=0.0)
            uncertainty_before = self._metric_tensor(evidence_state_before, "uncertainty_gap", logits_next, default=0.0)
            uncertainty_next = self._metric_tensor(evidence_state_next, "uncertainty_gap", logits_next, default=0.0)
            suspect_before = self._metric_tensor(evidence_state_before, "suspect_pool", logits_next, default=0.0)
            suspect_next = self._metric_tensor(evidence_state_next, "suspect_pool", logits_next, default=0.0)
            not_ruled_out_before = self._metric_tensor(evidence_state_before, "not_ruled_out_gate", logits_next, default=1.0)
            not_ruled_out_next = self._metric_tensor(evidence_state_next, "not_ruled_out_gate", logits_next, default=1.0)

            q_before = self._evidence_q(support_before, suspect_before, not_ruled_out_before)
            q_next = self._evidence_q(support_next, suspect_next, not_ruled_out_next)

            if constraint_state_before is not None:
                confirmed_non_before = constraint_state_before.confirmed_non_source_mask.view(-1) > 0.5
                confirmed_src_before = constraint_state_before.confirmed_source_mask.view(-1) > 0.5
            else:
                confirmed_non_before = torch.zeros_like(valid_mask_before)
                confirmed_src_before = torch.zeros_like(valid_mask_before)
            if constraint_state_after is not None:
                confirmed_non_after = constraint_state_after.confirmed_non_source_mask.view(-1) > 0.5
                confirmed_src_after = constraint_state_after.confirmed_source_mask.view(-1) > 0.5
            else:
                confirmed_non_after = torch.zeros_like(valid_mask_after)
                confirmed_src_after = torch.zeros_like(valid_mask_after)

            valid_mask_before = valid_mask_before & (~confirmed_non_before) & (~confirmed_src_before)
            valid_mask_after = valid_mask_after & (~confirmed_non_after) & (~confirmed_src_after)

            batch_flat = fused_batch.view(-1).long()
            graph_count = int(curr_batch_size)
            before_state = self._official_core_state_batch(
                q_before,
                uncertainty_before,
                suspect_before,
                not_ruled_out_before,
                valid_mask_before,
                batch_flat,
                graph_count,
            )
            after_state = self._official_core_state_batch(
                q_next,
                uncertainty_next,
                suspect_next,
                not_ruled_out_next,
                valid_mask_after,
                batch_flat,
                graph_count,
            )

            core_mass_before = before_state["phi"]
            core_mass_after = after_state["phi"]
            core_size_before = before_state["core_size"]
            core_size_after = after_state["core_size"]
            uncertainty_before_mass = before_state["uncertainty_mass"]
            uncertainty_after_mass = after_state["uncertainty_mass"]
            core_mass_delta = core_mass_before - core_mass_after
            core_size_delta = core_size_before - core_size_after
            uncertainty_collapse = uncertainty_before_mass - uncertainty_after_mass
            reward_tensor = (lambda_phi * core_mass_delta).to(dtype=torch.float32)
            if reward_mode == "navigator_bounded_coupling_repair_v1":
                reward_tensor = reward_tensor - step_cost
            selected_float = selection_mask.float()
            selected_count = scatter_sum(selected_float, batch_flat, dim=0, dim_size=graph_count)
            valid_selected = scatter_sum((selection_mask & valid_mask_before).float(), batch_flat, dim=0, dim_size=graph_count)
            invalid_count = selected_count - valid_selected
            invalid_fraction = torch.where(selected_count > 0, invalid_count / selected_count.clamp_min(1.0), torch.zeros_like(selected_count))
            if constraint_state_before is not None:
                no_resample_before = constraint_state_before.no_resample_mask.view(-1) > 0.5
                resample_count = scatter_sum((selection_mask & no_resample_before).float(), batch_flat, dim=0, dim_size=graph_count)
                resample_fraction = torch.where(selected_count > 0, resample_count / selected_count.clamp_min(1.0), torch.zeros_like(selected_count))
            else:
                resample_fraction = torch.zeros_like(selected_count)
            empty_selection = (selected_count <= 0).float()
            harmful_drift = (core_mass_after > (core_mass_before + 1e-6)).float()
            focus_core_delta = core_mass_delta
            wasted_budget_fraction = ((selected_count > 0) & (core_mass_delta.abs() <= 1e-8) & (uncertainty_collapse.abs() <= 1e-8)).float()
            evidence_gain_per_sample = core_mass_delta / valid_selected.clamp_min(1.0)
            closure_after = after_state["closure"]
            metric_rows = None
            if self.collect_detailed_step_metrics:
                metric_rows = [
                    {
                        "reward_total": float(reward_tensor[i].item()),
                        "core_mass_before": float(core_mass_before[i].item()),
                        "core_mass_after": float(core_mass_after[i].item()),
                        "core_size_before": float(core_size_before[i].item()),
                        "core_size_after": float(core_size_after[i].item()),
                        "uncertainty_before": float(uncertainty_before_mass[i].item()),
                        "uncertainty_after": float(uncertainty_after_mass[i].item()),
                    }
                    for i in range(graph_count)
                ]
            reward_bundle = {
                "mode": reward_mode,
                "r_total": reward_tensor.detach().cpu(),
                "step_penalty": torch.full((graph_count,), step_cost, dtype=torch.float32),
                "core_mass_before": core_mass_before.detach().cpu(),
                "core_mass_after": core_mass_after.detach().cpu(),
                "core_mass_delta": core_mass_delta.detach().cpu(),
                "core_size_before": core_size_before.detach().cpu(),
                "core_size_after": core_size_after.detach().cpu(),
                "core_size_delta": core_size_delta.detach().cpu(),
                "uncertainty_core_before": uncertainty_before_mass.detach().cpu(),
                "uncertainty_core_after": uncertainty_after_mass.detach().cpu(),
                "uncertainty_collapse": uncertainty_collapse.detach().cpu(),
                "candidate_count_before": before_state["candidate_count"].detach().cpu(),
                "candidate_count_after": after_state["candidate_count"].detach().cpu(),
                "diag_invalid_fraction": invalid_fraction.detach().cpu(),
                "diag_resample_fraction": resample_fraction.detach().cpu(),
                "diag_empty_selection": empty_selection.detach().cpu(),
                "harmful_drift": harmful_drift.detach().cpu(),
                "focus_core_delta": focus_core_delta.detach().cpu(),
                "wasted_budget_fraction": wasted_budget_fraction.detach().cpu(),
                "evidence_gain_per_sample": evidence_gain_per_sample.detach().cpu(),
                "closure_reached_after": closure_after.detach().cpu(),
                "budget_used_before": None if budget_used_before is None else budget_used_before.detach().cpu(),
                "budget_used_after": None if budget_used_after is None else budget_used_after.detach().cpu(),
                "audit_rows": metric_rows,
            }

        elif reward_mode == "navigator_vnext_evidence_v2" and evidence_state_before is not None and evidence_state_next is not None:
            lambda_rank = float(reward_cfg.get("lambda_rank", 1.0))
            lambda_shrink = float(reward_cfg.get("lambda_shrink", 1.0))

            if valid_mask_before is None:
                valid_mask_before = torch.isfinite(logits_next.view(-1))
            else:
                valid_mask_before = valid_mask_before.view(-1).bool()
            if valid_mask_after is None:
                valid_mask_after = valid_mask_before.clone()
            else:
                valid_mask_after = valid_mask_after.view(-1).bool()
            if selection_mask is None:
                selection_mask = torch.zeros_like(valid_mask_before).bool()
            else:
                selection_mask = selection_mask.view(-1) > 0.5

            support_before = self._metric_tensor(evidence_state_before, "support_score", logits_next, default=0.0)
            support_next = self._metric_tensor(evidence_state_next, "support_score", logits_next, default=0.0)
            uncertainty_before = self._metric_tensor(evidence_state_before, "uncertainty_gap", logits_next, default=0.0)
            uncertainty_next = self._metric_tensor(evidence_state_next, "uncertainty_gap", logits_next, default=0.0)
            suspect_before = self._metric_tensor(evidence_state_before, "suspect_pool", logits_next, default=0.0)
            suspect_next = self._metric_tensor(evidence_state_next, "suspect_pool", logits_next, default=0.0)
            not_ruled_out_before = self._metric_tensor(evidence_state_before, "not_ruled_out_gate", logits_next, default=1.0)
            not_ruled_out_next = self._metric_tensor(evidence_state_next, "not_ruled_out_gate", logits_next, default=1.0)

            q_before = self._evidence_q(
                support_before,
                suspect_before,
                not_ruled_out_before,
            )
            q_next = self._evidence_q(
                support_next,
                suspect_next,
                not_ruled_out_next,
            )

            if constraint_state_before is not None:
                confirmed_non_before = constraint_state_before.confirmed_non_source_mask.view(-1) > 0.5
                confirmed_src_before = constraint_state_before.confirmed_source_mask.view(-1) > 0.5
            else:
                confirmed_non_before = torch.zeros_like(valid_mask_before)
                confirmed_src_before = torch.zeros_like(valid_mask_before)
            if constraint_state_after is not None:
                confirmed_non_after = constraint_state_after.confirmed_non_source_mask.view(-1) > 0.5
                confirmed_src_after = constraint_state_after.confirmed_source_mask.view(-1) > 0.5
            else:
                confirmed_non_after = torch.zeros_like(valid_mask_after)
                confirmed_src_after = torch.zeros_like(valid_mask_after)

            valid_mask_before = valid_mask_before & (~confirmed_non_before) & (~confirmed_src_before)
            valid_mask_after = valid_mask_after & (~confirmed_non_after) & (~confirmed_src_after)

            batch_flat = fused_batch.view(-1).long()
            graph_count = int(curr_batch_size)
            before_state = self._official_core_state_batch(
                q_before,
                uncertainty_before,
                suspect_before,
                not_ruled_out_before,
                valid_mask_before,
                batch_flat,
                graph_count,
            )
            after_state = self._official_core_state_batch(
                q_next,
                uncertainty_next,
                suspect_next,
                not_ruled_out_next,
                valid_mask_after,
                batch_flat,
                graph_count,
            )

            core_mass_before = before_state["phi"]
            core_mass_after = after_state["phi"]
            core_size_before = before_state["core_size"]
            core_size_after = after_state["core_size"]
            uncertainty_before_mass = before_state["uncertainty_mass"]
            uncertainty_after_mass = after_state["uncertainty_mass"]
            core_mass_delta = core_mass_before - core_mass_after
            core_size_delta = core_size_before - core_size_after
            uncertainty_collapse = uncertainty_before_mass - uncertainty_after_mass
            reward_tensor = (lambda_phi * core_mass_delta).to(dtype=torch.float32)
            selected_float = selection_mask.float()
            selected_count = scatter_sum(selected_float, batch_flat, dim=0, dim_size=graph_count)
            valid_selected = scatter_sum((selection_mask & valid_mask_before).float(), batch_flat, dim=0, dim_size=graph_count)
            invalid_count = selected_count - valid_selected
            invalid_fraction = torch.where(selected_count > 0, invalid_count / selected_count.clamp_min(1.0), torch.zeros_like(selected_count))
            if constraint_state_before is not None:
                no_resample_before = constraint_state_before.no_resample_mask.view(-1) > 0.5
                resample_count = scatter_sum((selection_mask & no_resample_before).float(), batch_flat, dim=0, dim_size=graph_count)
                resample_fraction = torch.where(selected_count > 0, resample_count / selected_count.clamp_min(1.0), torch.zeros_like(selected_count))
            else:
                resample_fraction = torch.zeros_like(selected_count)
            empty_selection = (selected_count <= 0).float()
            harmful_drift = (core_mass_after > (core_mass_before + 1e-6)).float()
            focus_core_delta = core_mass_delta
            wasted_budget_fraction = ((selected_count > 0) & (core_mass_delta.abs() <= 1e-8) & (uncertainty_collapse.abs() <= 1e-8)).float()
            evidence_gain_per_sample = core_mass_delta / valid_selected.clamp_min(1.0)
            closure_after = after_state["closure"]
            metric_rows = [
                {
                    "reward_total": float(reward_tensor[i].item()),
                    "core_mass_before": float(core_mass_before[i].item()),
                    "core_mass_after": float(core_mass_after[i].item()),
                    "core_size_before": float(core_size_before[i].item()),
                    "core_size_after": float(core_size_after[i].item()),
                    "uncertainty_before": float(uncertainty_before_mass[i].item()),
                    "uncertainty_after": float(uncertainty_after_mass[i].item()),
                }
                for i in range(graph_count)
            ]
            gain_reward = reward_tensor.detach()
            reward_bundle = {
                "mode": reward_mode,
                "r_total": reward_tensor.detach().cpu(),
                "core_mass_before": core_mass_before.detach().cpu(),
                "core_mass_after": core_mass_after.detach().cpu(),
                "core_mass_delta": core_mass_delta.detach().cpu(),
                "core_size_before": core_size_before.detach().cpu(),
                "core_size_after": core_size_after.detach().cpu(),
                "core_size_delta": core_size_delta.detach().cpu(),
                "uncertainty_core_before": uncertainty_before_mass.detach().cpu(),
                "uncertainty_core_after": uncertainty_after_mass.detach().cpu(),
                "uncertainty_collapse": uncertainty_collapse.detach().cpu(),
                "candidate_count_before": before_state["candidate_count"].detach().cpu(),
                "candidate_count_after": after_state["candidate_count"].detach().cpu(),
                "diag_invalid_fraction": invalid_fraction.detach().cpu(),
                "diag_resample_fraction": resample_fraction.detach().cpu(),
                "diag_empty_selection": empty_selection.detach().cpu(),
                "harmful_drift": harmful_drift.detach().cpu(),
                "focus_core_delta": focus_core_delta.detach().cpu(),
                "wasted_budget_fraction": wasted_budget_fraction.detach().cpu(),
                "evidence_gain_per_sample": evidence_gain_per_sample.detach().cpu(),
                "closure_reached_after": closure_after.detach().cpu(),
                "budget_used_before": None if budget_used_before is None else budget_used_before.detach().cpu(),
                "budget_used_after": None if budget_used_after is None else budget_used_after.detach().cpu(),
                "audit_rows": metric_rows,
            }

        elif reward_mode == "navigator_vnext_evidence_v1" and evidence_state_before is not None and evidence_state_next is not None:
            w_ev_rank = float(reward_cfg.get("w_ev_rank", 0.30))
            w_ev_support = float(reward_cfg.get("w_ev_support", 0.20))
            w_ev_margin = float(reward_cfg.get("w_ev_margin", 0.10))
            w_ev_shrink = float(reward_cfg.get("w_ev_shrink", 0.20))
            w_ev_core = float(reward_cfg.get("w_ev_core", 0.20))
            p_invalid_w = float(reward_cfg.get("p_invalid", 0.50))
            p_resample_w = float(reward_cfg.get("p_resample", 0.25))
            p_noinfo_w = float(reward_cfg.get("p_noinfo", 0.10))
            p_shrinkhack_w = float(reward_cfg.get("p_shrinkhack", 0.15))
            p_harmful_w = float(reward_cfg.get("p_harmful", 0.20))
            p_waste_w = float(reward_cfg.get("p_waste", 0.05))
            support_delta = float(reward_cfg.get("support_plausible_delta", 0.25))

            if valid_mask_before is None:
                valid_mask_before = torch.isfinite(logits_next.view(-1))
            else:
                valid_mask_before = valid_mask_before.view(-1).bool()
            if selection_mask is None:
                selection_mask = torch.zeros_like(valid_mask_before).bool()
            else:
                selection_mask = selection_mask.view(-1) > 0.5

            support_before = self._metric_tensor(evidence_state_before, "support_score", logits_next, default=0.0)
            support_next = self._metric_tensor(evidence_state_next, "support_score", logits_next, default=0.0)
            uncertainty_before = self._metric_tensor(evidence_state_before, "uncertainty_gap", logits_next, default=0.0)
            uncertainty_next = self._metric_tensor(evidence_state_next, "uncertainty_gap", logits_next, default=0.0)
            not_ruled_out_before = self._metric_tensor(evidence_state_before, "not_ruled_out_gate", logits_next, default=1.0)
            not_ruled_out_next = self._metric_tensor(evidence_state_next, "not_ruled_out_gate", logits_next, default=1.0)
            contradiction_before = self._metric_tensor(evidence_state_before, "contradiction_score", logits_next, default=0.0)
            contradiction_next = self._metric_tensor(evidence_state_next, "contradiction_score", logits_next, default=0.0)
            observed_before = self._metric_tensor(observation_state_before, "observed_flag", logits_next, default=0.0)
            no_resample_before = self._metric_tensor(constraint_state_before, "no_resample_mask", logits_next, default=0.0)

            metric_rows = []
            r_total_list = []
            r_ev_rank_list = []
            r_ev_support_list = []
            r_ev_margin_list = []
            r_ev_shrink_list = []
            r_ev_core_list = []
            p_invalid_list = []
            p_resample_list = []
            p_noinfo_list = []
            p_shrinkhack_list = []
            p_harmful_list = []
            p_waste_list = []
            contradiction_drift_list = []
            pre_evidence_rows = []
            post_evidence_rows = []

            for graph_idx in range(int(curr_batch_size)):
                node_mask = fused_batch.view(-1) == int(graph_idx)
                label_slice = fused_source_label.view(-1)[node_mask]
                pre_evidence_metrics = self._single_graph_score_metrics(
                    support_before[node_mask],
                    label_slice,
                    plausible_delta=support_delta,
                )
                post_evidence_metrics = self._single_graph_score_metrics(
                    support_next[node_mask],
                    label_slice,
                    plausible_delta=support_delta,
                )
                pre_evidence_rows.append(pre_evidence_metrics)
                post_evidence_rows.append(post_evidence_metrics)

                selected_graph = selection_mask[node_mask]
                selected_valid = valid_mask_before[node_mask]
                selected_observed = observed_before[node_mask][selected_graph]
                selected_resampled = no_resample_before[node_mask][selected_graph]

                if not bool(pre_evidence_metrics["valid_case"]) or not bool(post_evidence_metrics["valid_case"]):
                    r_ev_rank = 0.0
                    r_ev_support = 0.0
                    r_ev_margin = 0.0
                    r_ev_shrink = 0.0
                    r_ev_core = 0.0
                    contradiction_drift = 0.0
                    p_invalid = 1.0
                    p_resample = 0.0
                    p_noinfo = 1.0
                    p_shrinkhack = 0.0
                    p_harmful = 0.0
                    p_waste = 1.0
                else:
                    rank_gain_raw = float(pre_evidence_metrics["true_source_rank"] - post_evidence_metrics["true_source_rank"])
                    r_ev_rank = max(min(rank_gain_raw, 2.0), -2.0) / 2.0
                    r_ev_support = max(
                        min(
                            float(post_evidence_metrics["true_source_value"] - pre_evidence_metrics["true_source_value"]),
                            1.0,
                        ),
                        -1.0,
                    )
                    r_ev_margin = max(
                        min(float(post_evidence_metrics["top1_margin"] - pre_evidence_metrics["top1_margin"]), 1.0),
                        -1.0,
                    )
                    pre_count = max(float(pre_evidence_metrics["plausible_count"]), 1.0)
                    r_ev_shrink = max(
                        min((float(pre_evidence_metrics["plausible_count"]) - float(post_evidence_metrics["plausible_count"])) / pre_count, 1.0),
                        -1.0,
                    )
                    pre_core_mass = self._core_mass(
                        support_before[node_mask],
                        valid_mask_before[node_mask],
                        uncertainty_before[node_mask],
                        not_ruled_out_before[node_mask],
                    )
                    post_core_mass = self._core_mass(
                        support_next[node_mask],
                        torch.isfinite(support_next[node_mask]),
                        uncertainty_next[node_mask],
                        not_ruled_out_next[node_mask],
                    )
                    r_ev_core = max(
                        min((pre_core_mass - post_core_mass) / max(pre_core_mass, 1e-6), 1.0),
                        -1.0,
                    )

                    true_mask = label_slice > 0.5
                    contradiction_pre_true = float(contradiction_before[node_mask][true_mask].max().item()) if bool(true_mask.any()) else 0.0
                    contradiction_post_true = float(contradiction_next[node_mask][true_mask].max().item()) if bool(true_mask.any()) else 0.0
                    contradiction_drift = max(min(contradiction_post_true - contradiction_pre_true, 1.0), -1.0)

                    p_invalid = float(bool(selected_graph.any()) and not bool((selected_graph & selected_valid).any()))
                    p_resample = float(bool(selected_graph.any()) and bool((selected_resampled > 0.5).any()))
                    no_info = bool(
                        (selected_observed.numel() > 0 and bool((selected_observed > 0.5).all()))
                        or (
                            abs(r_ev_rank) <= 1e-8
                            and abs(r_ev_support) <= 1e-8
                            and abs(r_ev_margin) <= 1e-8
                            and abs(r_ev_shrink) <= 1e-8
                            and abs(r_ev_core) <= 1e-8
                        )
                    )
                    p_noinfo = float(no_info)
                    shrink_hack = bool(r_ev_shrink > 0.0 and r_ev_rank <= 0.0 and r_ev_support <= 0.0 and r_ev_margin <= 0.0)
                    p_shrinkhack = float(shrink_hack)
                    harmful = bool(
                        r_ev_rank < 0.0
                        or r_ev_support < 0.0
                        or r_ev_margin < 0.0
                        or r_ev_core < 0.0
                        or contradiction_drift > 0.0
                    )
                    p_harmful = float(harmful)
                    p_waste = float(bool(selected_graph.any()) and bool(no_info or harmful))

                reward_total = (
                    w_ev_rank * r_ev_rank
                    + w_ev_support * r_ev_support
                    + w_ev_margin * r_ev_margin
                    + w_ev_shrink * r_ev_shrink
                    + w_ev_core * r_ev_core
                    - p_invalid_w * p_invalid
                    - p_resample_w * p_resample
                    - p_noinfo_w * p_noinfo
                    - p_shrinkhack_w * p_shrinkhack
                    - p_harmful_w * p_harmful
                    - p_waste_w * p_waste
                    - step_cost
                )

                metric_rows.append(
                    {
                        "reward_total": float(reward_total),
                        "contradiction_drift": float(contradiction_drift),
                    }
                )
                r_total_list.append(float(reward_total))
                r_ev_rank_list.append(float(r_ev_rank))
                r_ev_support_list.append(float(r_ev_support))
                r_ev_margin_list.append(float(r_ev_margin))
                r_ev_shrink_list.append(float(r_ev_shrink))
                r_ev_core_list.append(float(r_ev_core))
                p_invalid_list.append(float(p_invalid))
                p_resample_list.append(float(p_resample))
                p_noinfo_list.append(float(p_noinfo))
                p_shrinkhack_list.append(float(p_shrinkhack))
                p_harmful_list.append(float(p_harmful))
                p_waste_list.append(float(p_waste))
                contradiction_drift_list.append(float(contradiction_drift))

            reward_tensor = torch.tensor(r_total_list, device=logits_next.device, dtype=torch.float32)
            gain_reward = reward_tensor.detach()
            reward_bundle = {
                "mode": reward_mode,
                "r_total": reward_tensor.detach().cpu(),
                "r_ev_rank": torch.tensor(r_ev_rank_list, dtype=torch.float32),
                "r_ev_support": torch.tensor(r_ev_support_list, dtype=torch.float32),
                "r_ev_margin": torch.tensor(r_ev_margin_list, dtype=torch.float32),
                "r_ev_shrink": torch.tensor(r_ev_shrink_list, dtype=torch.float32),
                "r_ev_core": torch.tensor(r_ev_core_list, dtype=torch.float32),
                "p_invalid": torch.tensor(p_invalid_list, dtype=torch.float32),
                "p_resample": torch.tensor(p_resample_list, dtype=torch.float32),
                "p_noinfo": torch.tensor(p_noinfo_list, dtype=torch.float32),
                "p_shrinkhack": torch.tensor(p_shrinkhack_list, dtype=torch.float32),
                "p_harmful": torch.tensor(p_harmful_list, dtype=torch.float32),
                "p_waste": torch.tensor(p_waste_list, dtype=torch.float32),
                "contradiction_drift": torch.tensor(contradiction_drift_list, dtype=torch.float32),
                "step_penalty": torch.full((int(curr_batch_size),), step_cost, dtype=torch.float32),
                "pre_evidence_metrics": pre_evidence_rows,
                "post_evidence_metrics": post_evidence_rows,
                "budget_used_before": None if budget_used_before is None else budget_used_before.detach().cpu(),
                "budget_used_after": None if budget_used_after is None else budget_used_after.detach().cpu(),
                "audit_rows": metric_rows,
            }

        elif reward_mode == "navigator_capability_v1" and logits_before is not None:
            w_rank = float(reward_cfg.get("w_rank", 0.45))
            w_prob = float(reward_cfg.get("w_prob", 0.30))
            w_margin = float(reward_cfg.get("w_margin", 0.15))
            w_top1 = float(reward_cfg.get("w_top1", 0.05))
            w_ev_rank = float(reward_cfg.get("w_ev_rank", 0.0))
            w_ev_support = float(reward_cfg.get("w_ev_support", 0.0))
            w_ev_margin = float(reward_cfg.get("w_ev_margin", 0.0))
            w_ev_shrink = float(reward_cfg.get("w_ev_shrink", 0.10))
            w_ev_core = float(reward_cfg.get("w_ev_core", 0.05))
            p_invalid_w = float(reward_cfg.get("p_invalid", 0.25))
            p_noinfo_w = float(reward_cfg.get("p_noinfo", 0.10))
            p_shrinkhack_w = float(reward_cfg.get("p_shrinkhack", 0.15))
            p_harmful_w = float(reward_cfg.get("p_harmful", 0.10))

            logits_before = logits_before.view(-1)
            if valid_mask_before is None:
                valid_mask_before = torch.isfinite(logits_before)
            else:
                valid_mask_before = valid_mask_before.view(-1).bool()
            if selection_mask is None:
                selection_mask = torch.zeros_like(logits_before).bool()
            else:
                selection_mask = selection_mask.view(-1) > 0.5

            uncertainty_before = self._metric_tensor(evidence_state_before, "uncertainty_gap", logits_before, default=0.0)
            not_ruled_out_before = self._metric_tensor(evidence_state_before, "not_ruled_out_gate", logits_before, default=1.0)
            uncertainty_next = self._metric_tensor(evidence_state_next, "uncertainty_gap", logits_next, default=0.0)
            not_ruled_out_next = self._metric_tensor(evidence_state_next, "not_ruled_out_gate", logits_next, default=1.0)
            support_before = self._metric_tensor(evidence_state_before, "support_score", logits_before, default=0.0)
            support_next = self._metric_tensor(evidence_state_next, "support_score", logits_next, default=0.0)

            metric_rows = []
            r_total_list = []
            r_rank_list = []
            r_prob_list = []
            r_margin_list = []
            r_top1_list = []
            r_ev_rank_list = []
            r_ev_support_list = []
            r_ev_margin_list = []
            r_ev_shrink_list = []
            r_ev_core_list = []
            p_invalid_list = []
            p_noinfo_list = []
            p_shrinkhack_list = []
            p_harmful_list = []
            pre_rows = []
            post_rows = []
            pre_evidence_rows = []
            post_evidence_rows = []

            for graph_idx in range(int(curr_batch_size)):
                node_mask = fused_batch.view(-1) == int(graph_idx)
                pre_metrics = self._single_graph_metrics(
                    logits_before[node_mask],
                    fused_source_label.view(-1)[node_mask],
                )
                post_metrics = self._single_graph_metrics(
                    logits_next.view(-1)[node_mask],
                    fused_source_label.view(-1)[node_mask],
                )
                pre_rows.append(pre_metrics)
                post_rows.append(post_metrics)
                pre_evidence_metrics = self._single_graph_score_metrics(
                    support_before[node_mask],
                    fused_source_label.view(-1)[node_mask],
                )
                post_evidence_metrics = self._single_graph_score_metrics(
                    support_next[node_mask],
                    fused_source_label.view(-1)[node_mask],
                )
                pre_evidence_rows.append(pre_evidence_metrics)
                post_evidence_rows.append(post_evidence_metrics)

                if not bool(pre_metrics["valid_case"]) or not bool(post_metrics["valid_case"]):
                    r_rank = 0.0
                    r_prob = 0.0
                    r_margin = 0.0
                    r_top1 = 0.0
                    r_ev_rank = 0.0
                    r_ev_support = 0.0
                    r_ev_margin = 0.0
                    r_ev_shrink = 0.0
                    r_ev_core = 0.0
                    p_invalid = 1.0
                    p_noinfo = 1.0
                    p_shrinkhack = 0.0
                    p_harmful = 0.0
                else:
                    rank_gain_raw = float(pre_metrics["true_source_rank"] - post_metrics["true_source_rank"])
                    r_rank = max(min(rank_gain_raw, 2.0), -2.0) / 2.0
                    r_prob = float(post_metrics["true_source_prob"] - pre_metrics["true_source_prob"])
                    r_margin = max(
                        min(float(post_metrics["top1_margin"] - pre_metrics["top1_margin"]), 1.0),
                        -1.0,
                    )
                    r_top1 = float(int(bool(post_metrics["top1_hit"])) - int(bool(pre_metrics["top1_hit"])))
                    if bool(pre_evidence_metrics["valid_case"]) and bool(post_evidence_metrics["valid_case"]):
                        ev_rank_gain_raw = float(pre_evidence_metrics["true_source_rank"] - post_evidence_metrics["true_source_rank"])
                        r_ev_rank = max(min(ev_rank_gain_raw, 2.0), -2.0) / 2.0
                        r_ev_support = max(
                            min(
                                float(post_evidence_metrics["true_source_value"] - pre_evidence_metrics["true_source_value"]),
                                1.0,
                            ),
                            -1.0,
                        )
                        r_ev_margin = max(
                            min(float(post_evidence_metrics["top1_margin"] - pre_evidence_metrics["top1_margin"]), 1.0),
                            -1.0,
                        )
                    else:
                        r_ev_rank = 0.0
                        r_ev_support = 0.0
                        r_ev_margin = 0.0
                    pre_count = max(float(pre_metrics["plausible_count"]), 1.0)
                    r_ev_shrink = max(
                        min((float(pre_metrics["plausible_count"]) - float(post_metrics["plausible_count"])) / pre_count, 1.0),
                        -1.0,
                    )

                    node_uncertainty_before = uncertainty_before[node_mask]
                    node_not_ruled_out_before = not_ruled_out_before[node_mask]
                    node_uncertainty_next = uncertainty_next[node_mask]
                    node_not_ruled_out_next = not_ruled_out_next[node_mask]
                    pre_core_mass = self._core_mass(
                        logits_before[node_mask],
                        valid_mask_before[node_mask],
                        node_uncertainty_before,
                        node_not_ruled_out_before,
                    )
                    post_core_mass = self._core_mass(
                        logits_next.view(-1)[node_mask],
                        torch.isfinite(logits_next.view(-1)[node_mask]),
                        node_uncertainty_next,
                        node_not_ruled_out_next,
                    )
                    r_ev_core = max(
                        min((pre_core_mass - post_core_mass) / max(pre_core_mass, 1e-6), 1.0),
                        -1.0,
                    )
                    selected_graph = selection_mask[node_mask]
                    selected_valid = valid_mask_before[node_mask]
                    p_invalid = float(bool(selected_graph.any()) and not bool((selected_graph & selected_valid).any()))
                    no_info = (
                        abs(r_rank) <= 1e-8
                        and abs(r_prob) <= 1e-8
                        and abs(r_margin) <= 1e-8
                        and abs(r_top1) <= 1e-8
                        and abs(r_ev_shrink) <= 1e-8
                        and abs(r_ev_core) <= 1e-8
                    )
                    p_noinfo = float(no_info)
                    shrink_hack = bool(r_ev_shrink > 0.0 and r_rank <= 0.0 and r_prob <= 0.0 and r_margin <= 0.0 and r_top1 <= 0.0)
                    p_shrinkhack = float(shrink_hack)
                    harmful = bool(r_rank < 0.0 or r_prob < 0.0 or r_margin < 0.0 or r_top1 < 0.0)
                    p_harmful = float(harmful)

                reward_main = (
                    w_rank * r_rank
                    + w_prob * r_prob
                    + w_margin * r_margin
                    + w_top1 * r_top1
                )
                reward_shape = (
                    w_ev_rank * r_ev_rank
                    + w_ev_support * r_ev_support
                    + w_ev_margin * r_ev_margin
                    + w_ev_shrink * r_ev_shrink
                    + w_ev_core * r_ev_core
                )
                reward_penalty = (
                    p_invalid_w * p_invalid
                    + p_noinfo_w * p_noinfo
                    + p_shrinkhack_w * p_shrinkhack
                    + p_harmful_w * p_harmful
                    + step_cost
                )
                reward_total = reward_main + reward_shape - reward_penalty

                metric_rows.append(
                    {
                        "reward_main": float(reward_main),
                        "reward_shape": float(reward_shape),
                        "reward_penalty": float(reward_penalty),
                    }
                )
                r_total_list.append(float(reward_total))
                r_rank_list.append(float(r_rank))
                r_prob_list.append(float(r_prob))
                r_margin_list.append(float(r_margin))
                r_top1_list.append(float(r_top1))
                r_ev_rank_list.append(float(r_ev_rank))
                r_ev_support_list.append(float(r_ev_support))
                r_ev_margin_list.append(float(r_ev_margin))
                r_ev_shrink_list.append(float(r_ev_shrink))
                r_ev_core_list.append(float(r_ev_core))
                p_invalid_list.append(float(p_invalid))
                p_noinfo_list.append(float(p_noinfo))
                p_shrinkhack_list.append(float(p_shrinkhack))
                p_harmful_list.append(float(p_harmful))

            reward_tensor = torch.tensor(r_total_list, device=logits_next.device, dtype=torch.float32)
            gain_reward = reward_tensor.detach()
            reward_bundle = {
                "mode": reward_mode,
                "r_total": reward_tensor.detach().cpu(),
                "r_rank": torch.tensor(r_rank_list, dtype=torch.float32),
                "r_prob": torch.tensor(r_prob_list, dtype=torch.float32),
                "r_margin": torch.tensor(r_margin_list, dtype=torch.float32),
                "r_top1": torch.tensor(r_top1_list, dtype=torch.float32),
                "r_ev_rank": torch.tensor(r_ev_rank_list, dtype=torch.float32),
                "r_ev_support": torch.tensor(r_ev_support_list, dtype=torch.float32),
                "r_ev_margin": torch.tensor(r_ev_margin_list, dtype=torch.float32),
                "r_ev_shrink": torch.tensor(r_ev_shrink_list, dtype=torch.float32),
                "r_ev_core": torch.tensor(r_ev_core_list, dtype=torch.float32),
                "p_invalid": torch.tensor(p_invalid_list, dtype=torch.float32),
                "p_noinfo": torch.tensor(p_noinfo_list, dtype=torch.float32),
                "p_shrinkhack": torch.tensor(p_shrinkhack_list, dtype=torch.float32),
                "p_harmful": torch.tensor(p_harmful_list, dtype=torch.float32),
                "step_penalty": torch.full((int(curr_batch_size),), step_cost, dtype=torch.float32),
                "pre_metrics": pre_rows,
                "post_metrics": post_rows,
                "pre_evidence_metrics": pre_evidence_rows,
                "post_evidence_metrics": post_evidence_rows,
                "budget_used_before": None if budget_used_before is None else budget_used_before.detach().cpu(),
                "budget_used_after": None if budget_used_after is None else budget_used_after.detach().cpu(),
                "audit_rows": metric_rows,
            }

        nav_gain_loss = torch.zeros(curr_batch_size, device=logits_next.device)
        gain_reward_nodes = gain_reward[fused_batch]
        
        if y_action is not None and y_action.size(0) == gain_reward_nodes.size(0):
            action_weights = y_action.view(-1).to(device=logits_next.device, dtype=gain_reward_nodes.dtype)
            step_loss_per_node = -(action_weights * gain_reward_nodes.view(-1))
            nav_gain_loss = scatter_sum(step_loss_per_node, fused_batch, dim=0, dim_size=curr_batch_size)
            
        return nav_gain_loss, gain_reward, reward_bundle

    def calculate_terminal_rewards(
        self,
        fused_batch,
        fused_source_label,
        evidence_state_final,
        constraint_state_final,
        valid_mask_final,
        budget_used,
        budget_max,
        final_reasoner_logits=None,
    ):
        reward_cfg = self._navigator_reward_cfg()
        reward_mode = str(reward_cfg.get("mode", "prob_gain_v0"))
        if reward_mode in {"navigator_terminal_mrr_v1", "navigator_terminal_rank_closure_v1"} and final_reasoner_logits is not None:
            curr_batch_size = int(budget_used.numel()) if isinstance(budget_used, torch.Tensor) else 0
            device = fused_batch.device
            valid_mask_final = valid_mask_final.view(-1).bool()
            logits_final = final_reasoner_logits.view(-1).float()
            if constraint_state_final is not None:
                confirmed_non = constraint_state_final.confirmed_non_source_mask.view(-1) > 0.5
                confirmed_src = constraint_state_final.confirmed_source_mask.view(-1) > 0.5
                valid_mask_final = valid_mask_final & (~confirmed_non) & (~confirmed_src)

            terminal_total = []
            terminal_mrr = []
            terminal_rank = []
            terminal_top1 = []
            terminal_top3 = []
            terminal_top5 = []
            terminal_valid_count = []
            for graph_idx in range(curr_batch_size):
                node_mask = fused_batch.view(-1) == int(graph_idx)
                graph_valid = valid_mask_final[node_mask]
                graph_logits = logits_final[node_mask]
                labels = fused_source_label.view(-1)[node_mask].float()
                finite_mask = graph_valid & torch.isfinite(graph_logits)
                terminal_valid_count.append(float(finite_mask.float().sum().item()))
                if not bool(finite_mask.any()) or not bool((labels > 0.5).any()):
                    terminal_total.append(0.0)
                    terminal_mrr.append(0.0)
                    terminal_rank.append(-1.0)
                    terminal_top1.append(0.0)
                    terminal_top3.append(0.0)
                    terminal_top5.append(0.0)
                    continue

                safe_logits = graph_logits.clone()
                safe_logits[~finite_mask] = -float("inf")
                sorted_idx = torch.argsort(safe_logits, descending=True)
                true_positions = (labels[sorted_idx] > 0.5).nonzero(as_tuple=True)[0]
                if true_positions.numel() == 0:
                    terminal_total.append(0.0)
                    terminal_mrr.append(0.0)
                    terminal_rank.append(-1.0)
                    terminal_top1.append(0.0)
                    terminal_top3.append(0.0)
                    terminal_top5.append(0.0)
                    continue

                rank = int(true_positions.min().item() + 1)
                mrr = 1.0 / float(rank)
                terminal_total.append(mrr)
                terminal_mrr.append(mrr)
                terminal_rank.append(float(rank))
                terminal_top1.append(float(rank <= 1))
                terminal_top3.append(float(rank <= 3))
                terminal_top5.append(float(rank <= 5))

            terminal_tensor = torch.tensor(terminal_total, device=device, dtype=torch.float32)
            terminal_bundle = {
                "terminal_reward_total": terminal_tensor.detach().cpu(),
                "terminal_mrr": torch.tensor(terminal_mrr, dtype=torch.float32),
                "terminal_true_source_rank": torch.tensor(terminal_rank, dtype=torch.float32),
                "terminal_top1_hit": torch.tensor(terminal_top1, dtype=torch.float32),
                "terminal_top3_hit": torch.tensor(terminal_top3, dtype=torch.float32),
                "terminal_top5_hit": torch.tensor(terminal_top5, dtype=torch.float32),
            }
            if reward_mode == "navigator_terminal_rank_closure_v1":
                batch_flat = fused_batch.view(-1).long()
                reward_cfg = self._navigator_reward_cfg()
                lambda_mrr = float(reward_cfg.get("lambda_mrr", 1.0))
                lambda_top5 = float(reward_cfg.get("lambda_top5", 0.0))
                lambda_closure = float(reward_cfg.get("lambda_closure", 0.0))
                lambda_budget_save = float(reward_cfg.get("lambda_budget_save", 0.0))

                if evidence_state_final is not None:
                    reference = valid_mask_final.float()
                    support_final = self._metric_tensor(evidence_state_final, "support_score", reference, default=0.0).clamp_min(0.0)
                    uncertainty_final = self._metric_tensor(evidence_state_final, "uncertainty_gap", reference, default=0.0).clamp_min(0.0)
                    suspect_final = self._metric_tensor(evidence_state_final, "suspect_pool", reference, default=0.0).clamp(0.0, 1.0)
                    not_ruled_out_final = self._metric_tensor(evidence_state_final, "not_ruled_out_gate", reference, default=1.0).clamp(0.0, 1.0)
                    alpha_q_suspect = float(reward_cfg.get("alpha_q_suspect", 0.35))
                    q_final = not_ruled_out_final * (support_final + alpha_q_suspect * suspect_final)
                    state_final = self._official_core_state_batch(
                        q=q_final,
                        uncertainty=uncertainty_final,
                        suspect=suspect_final,
                        not_ruled_out=not_ruled_out_final,
                        valid_mask=valid_mask_final,
                        batch_index=batch_flat,
                        curr_batch_size=curr_batch_size,
                    )
                    closure = state_final["closure"].float()
                    core_mass = state_final["phi"].float()
                    core_size = state_final["core_size"].float()
                    candidate_count = state_final["candidate_count"].float()
                else:
                    closure = torch.zeros(int(curr_batch_size), device=device, dtype=torch.float32)
                    core_mass = torch.zeros(int(curr_batch_size), device=device, dtype=torch.float32)
                    core_size = torch.zeros(int(curr_batch_size), device=device, dtype=torch.float32)
                    candidate_count = torch.zeros(int(curr_batch_size), device=device, dtype=torch.float32)

                budget_used_f = budget_used.detach().float().view(-1)
                budget_den = max(float(budget_max), 1.0)
                budget_bonus = closure * lambda_budget_save * ((budget_den - budget_used_f).clamp_min(0.0) / budget_den)
                rank_reward = (
                    lambda_mrr * torch.tensor(terminal_mrr, device=device, dtype=torch.float32)
                    + lambda_top5 * torch.tensor(terminal_top5, device=device, dtype=torch.float32)
                    + lambda_closure * closure
                    + budget_bonus
                )
                terminal_tensor = rank_reward.to(dtype=torch.float32)
                terminal_bundle.update(
                    {
                        "terminal_reward_total": terminal_tensor.detach().cpu(),
                        "terminal_closure_success": closure.detach().cpu(),
                        "terminal_core_mass_final": core_mass.detach().cpu(),
                        "terminal_core_size_final": core_size.detach().cpu(),
                        "terminal_candidate_count": candidate_count.detach().cpu(),
                        "terminal_budget_bonus": budget_bonus.detach().cpu(),
                        "terminal_valid_candidate_count": torch.tensor(terminal_valid_count, dtype=torch.float32),
                    }
                )
            return terminal_tensor, terminal_bundle
        if reward_mode == "navigator_bounded_coupling_repair_v1" and final_reasoner_logits is not None:
            curr_batch_size = int(budget_used.numel()) if isinstance(budget_used, torch.Tensor) else 0
            device = fused_batch.device
            valid_mask_final = valid_mask_final.view(-1).bool()
            logits_final = final_reasoner_logits.view(-1).float()
            source_hit_graph = torch.zeros(curr_batch_size, device=device, dtype=torch.float32)
            if constraint_state_final is not None:
                confirmed_non = constraint_state_final.confirmed_non_source_mask.view(-1) > 0.5
                confirmed_src = constraint_state_final.confirmed_source_mask.view(-1) > 0.5
                valid_mask_final = valid_mask_final & (~confirmed_non) & (~confirmed_src)
                source_hit_graph = scatter_max(
                    confirmed_src.float(),
                    fused_batch.view(-1).long(),
                    dim=0,
                    dim_size=curr_batch_size,
                )[0].float()

            lambda_mrr = float(reward_cfg.get("lambda_mrr", 1.0))
            lambda_top5 = float(reward_cfg.get("lambda_top5", 0.25))
            lambda_source_hit = float(reward_cfg.get("lambda_source_hit", 0.75))

            terminal_total = []
            terminal_mrr = []
            terminal_rank = []
            terminal_top1 = []
            terminal_top3 = []
            terminal_top5 = []
            for graph_idx in range(curr_batch_size):
                node_mask = fused_batch.view(-1) == int(graph_idx)
                graph_valid = valid_mask_final[node_mask]
                graph_logits = logits_final[node_mask]
                labels = fused_source_label.view(-1)[node_mask].float()
                finite_mask = graph_valid & torch.isfinite(graph_logits)
                source_hit = float(source_hit_graph[graph_idx].item() > 0.5)
                if not bool(finite_mask.any()) or not bool((labels > 0.5).any()):
                    terminal_total.append(lambda_source_hit * source_hit)
                    terminal_mrr.append(0.0)
                    terminal_rank.append(-1.0)
                    terminal_top1.append(0.0)
                    terminal_top3.append(0.0)
                    terminal_top5.append(0.0)
                    continue

                safe_logits = graph_logits.clone()
                safe_logits[~finite_mask] = -float("inf")
                sorted_idx = torch.argsort(safe_logits, descending=True)
                true_positions = (labels[sorted_idx] > 0.5).nonzero(as_tuple=True)[0]
                if true_positions.numel() == 0:
                    terminal_total.append(lambda_source_hit * source_hit)
                    terminal_mrr.append(0.0)
                    terminal_rank.append(-1.0)
                    terminal_top1.append(0.0)
                    terminal_top3.append(0.0)
                    terminal_top5.append(0.0)
                    continue

                rank = int(true_positions.min().item() + 1)
                mrr = 1.0 / float(rank)
                top1 = float(rank <= 1)
                top3 = float(rank <= 3)
                top5 = float(rank <= 5)
                total = (lambda_mrr * mrr) + (lambda_top5 * top5) + (lambda_source_hit * source_hit)
                terminal_total.append(total)
                terminal_mrr.append(mrr)
                terminal_rank.append(float(rank))
                terminal_top1.append(top1)
                terminal_top3.append(top3)
                terminal_top5.append(top5)

            terminal_tensor = torch.tensor(terminal_total, device=device, dtype=torch.float32)
            return terminal_tensor, {
                "terminal_reward_total": terminal_tensor.detach().cpu(),
                "terminal_mrr": torch.tensor(terminal_mrr, dtype=torch.float32),
                "terminal_true_source_rank": torch.tensor(terminal_rank, dtype=torch.float32),
                "terminal_top1_hit": torch.tensor(terminal_top1, dtype=torch.float32),
                "terminal_top3_hit": torch.tensor(terminal_top3, dtype=torch.float32),
                "terminal_top5_hit": torch.tensor(terminal_top5, dtype=torch.float32),
                "terminal_source_hit_bonus": source_hit_graph.detach().cpu() * lambda_source_hit,
                "terminal_source_hit": source_hit_graph.detach().cpu(),
            }
        if reward_mode in {"navigator_official_phi_v1", "navigator_vnext_official_phi_v1"} and evidence_state_final is not None:
            curr_batch_size = int(budget_used.numel()) if isinstance(budget_used, torch.Tensor) else 0
            device = fused_batch.device
            valid_mask_final = valid_mask_final.view(-1).bool()
            support_final = self._metric_tensor(evidence_state_final, "support_score", valid_mask_final, default=0.0)
            suspect_final = self._metric_tensor(evidence_state_final, "suspect_pool", valid_mask_final, default=0.0)
            uncertainty_final = self._metric_tensor(evidence_state_final, "uncertainty_gap", valid_mask_final, default=0.0)
            not_ruled_out_final = self._metric_tensor(evidence_state_final, "not_ruled_out_gate", valid_mask_final, default=1.0)
            q_final = self._evidence_q(support_final, suspect_final, not_ruled_out_final)

            if constraint_state_final is not None:
                confirmed_non = constraint_state_final.confirmed_non_source_mask.view(-1) > 0.5
                confirmed_src = constraint_state_final.confirmed_source_mask.view(-1) > 0.5
                valid_mask_final = valid_mask_final & (~confirmed_non) & (~confirmed_src)

            mu_mass = float(reward_cfg.get("mu_closure_mass", 1.0))
            mu_size = float(reward_cfg.get("mu_closure_size", 0.5))
            mu_early = float(reward_cfg.get("mu_early_closure", reward_cfg.get("mu_early_decisive", 0.25)))
            lambda_budget_save = float(reward_cfg.get("lambda_budget_save", 0.05))
            early_budget = float(
                reward_cfg.get("early_closure_budget", reward_cfg.get("early_decisive_budget", max(float(budget_max) / 2.0, 1.0)))
            )

            batch_flat = fused_batch.view(-1).long()
            state_final = self._official_core_state_batch(
                q_final,
                uncertainty_final,
                suspect_final,
                not_ruled_out_final,
                valid_mask_final,
                batch_flat,
                curr_batch_size,
            )

            core_mass = state_final["phi"]
            core_size = state_final["core_size"]
            uncertainty_mass = state_final["uncertainty_mass"]
            mass_threshold = float(reward_cfg.get("closure_mass_threshold", reward_cfg.get("terminal_core_mass_threshold", 1.0)))
            size_threshold = float(reward_cfg.get("closure_size_threshold", reward_cfg.get("terminal_core_size_threshold", 2.0)))
            mass_closed = (core_mass <= mass_threshold).float()
            size_closed = (core_size <= size_threshold).float()
            closure = (mass_closed > 0.5) | (size_closed > 0.5)
            budget_used_f = budget_used.detach().float().view(-1)
            early_closure = (closure & (budget_used_f <= early_budget)).float()
            budget_bonus = closure.float() * lambda_budget_save * (float(budget_max) - budget_used_f).clamp_min(0.0)
            terminal_tensor = (
                mu_mass * mass_closed
                + mu_size * size_closed
                + mu_early * early_closure
                + budget_bonus
            ).to(dtype=torch.float32)
            return terminal_tensor, {
                "terminal_reward_total": terminal_tensor.detach().cpu(),
                "terminal_budget_bonus": budget_bonus.detach().cpu(),
                "terminal_core_mass_closed": mass_closed.detach().cpu(),
                "terminal_core_size_closed": size_closed.detach().cpu(),
                "terminal_closure_success": closure.float().detach().cpu(),
                "terminal_early_closure": early_closure.detach().cpu(),
                "terminal_core_mass_final": core_mass.detach().cpu(),
                "terminal_core_size_final": core_size.detach().cpu(),
                "terminal_uncertainty_final": uncertainty_mass.detach().cpu(),
            }

        if reward_mode != "navigator_vnext_evidence_v2" or evidence_state_final is None:
            return None, {}

        curr_batch_size = int(budget_used.numel()) if isinstance(budget_used, torch.Tensor) else 0
        device = fused_batch.device
        valid_mask_final = valid_mask_final.view(-1).bool()
        support_final = self._metric_tensor(evidence_state_final, "support_score", valid_mask_final, default=0.0)
        suspect_final = self._metric_tensor(evidence_state_final, "suspect_pool", valid_mask_final, default=0.0)
        uncertainty_final = self._metric_tensor(evidence_state_final, "uncertainty_gap", valid_mask_final, default=0.0)
        not_ruled_out_final = self._metric_tensor(evidence_state_final, "not_ruled_out_gate", valid_mask_final, default=1.0)
        q_final = self._evidence_q(
            support_final,
            suspect_final,
            not_ruled_out_final,
        )

        if constraint_state_final is not None:
            confirmed_non = constraint_state_final.confirmed_non_source_mask.view(-1) > 0.5
            confirmed_src = constraint_state_final.confirmed_source_mask.view(-1) > 0.5
            valid_mask_final = valid_mask_final & (~confirmed_non) & (~confirmed_src)

        mu_rank_hit = float(reward_cfg.get("mu_hit", reward_cfg.get("mu_rank_hit", 1.0)))
        mu_candidate_target = float(reward_cfg.get("mu_target", reward_cfg.get("mu_candidate_target", 0.5)))
        mu_early_decisive = float(reward_cfg.get("mu_early", reward_cfg.get("mu_early_decisive", 0.25)))
        lambda_budget_save = float(reward_cfg.get("lambda_save", reward_cfg.get("lambda_budget_save", 0.05)))
        k_hit = int(reward_cfg.get("K_hit", reward_cfg.get("k_hit", 3)))
        n_target = int(reward_cfg.get("n_target", 3))
        early_decisive_budget = float(reward_cfg.get("early_decisive_budget", max(k_hit, 1)))

        terminal_total = []
        terminal_hit = []
        terminal_collapse = []
        terminal_early = []
        terminal_budget = []
        terminal_candidate = []
        terminal_rank = []
        terminal_success = []
        for graph_idx in range(curr_batch_size):
            node_mask = fused_batch.view(-1) == int(graph_idx)
            label_slice = fused_source_label.view(-1)[node_mask]
            terms = self._evidence_contract_terms(
                q_final[node_mask],
                uncertainty_final[node_mask],
                suspect_final[node_mask],
                not_ruled_out_final[node_mask],
                valid_mask_final[node_mask],
                label_slice,
            )
            source_rank = terms["source_rank"]
            candidate_count = float(terms["candidate_count"])
            rank_hit = float(source_rank is not None and source_rank <= k_hit)
            collapse_hit = float(candidate_count <= float(n_target))
            early_decisive = float(rank_hit > 0.5 and float(budget_used[graph_idx].item()) <= early_decisive_budget)
            success = float(rank_hit > 0.5)
            budget_bonus = success * lambda_budget_save * max(float(budget_max) - float(budget_used[graph_idx].item()), 0.0)
            terminal_val = (
                mu_rank_hit * rank_hit
                + mu_candidate_target * collapse_hit
                + mu_early_decisive * early_decisive
                + budget_bonus
            )
            terminal_total.append(float(terminal_val))
            terminal_hit.append(float(rank_hit))
            terminal_collapse.append(float(collapse_hit))
            terminal_early.append(float(early_decisive))
            terminal_budget.append(float(budget_bonus))
            terminal_candidate.append(float(candidate_count))
            terminal_rank.append(float(-1 if source_rank is None else source_rank))
            terminal_success.append(float(success))

        terminal_tensor = torch.tensor(terminal_total, device=device, dtype=torch.float32)
        return terminal_tensor, {
            "terminal_reward_total": terminal_tensor.detach().cpu(),
            "terminal_rank_hit": torch.tensor(terminal_hit, dtype=torch.float32),
            "terminal_candidate_target": torch.tensor(terminal_collapse, dtype=torch.float32),
            "terminal_early_decisive": torch.tensor(terminal_early, dtype=torch.float32),
            "terminal_budget_bonus": torch.tensor(terminal_budget, dtype=torch.float32),
            "terminal_candidate_count": torch.tensor(terminal_candidate, dtype=torch.float32),
            "terminal_source_rank": torch.tensor(terminal_rank, dtype=torch.float32),
            "terminal_success": torch.tensor(terminal_success, dtype=torch.float32),
        }

    def record_reward_bundle(self, step, reward_bundle):
        if not reward_bundle or not self.collect_detailed_step_metrics:
            return
        for key, value in reward_bundle.items():
            if isinstance(value, torch.Tensor):
                if value.numel() <= 0:
                    continue
                metric_value = float(value.float().mean().item())
                self.probe_b_metrics[f"reward/step_{step}/{key}_mean"] = metric_value
            elif isinstance(value, (int, float)):
                self.probe_b_metrics[f"reward/step_{step}/{key}"] = float(value)

    def track_evidence_stats(self, evidence_state, batch_index, curr_batch_size, fused_source_label=None, step=0, time_cost_ms=0.0):
        """
        Capture Evidence State statistics for W&B monitoring.
        Includes Physical Audit (True vs Negative) if label is provided.
        Extended for Evidence Audit Sprint (Component, Per-Event, Per-Step, Gate, Performance).
        
        Args:
            evidence_state: Dict or Object with evidence tensors (Fused Space)
            batch_index: Tensor mapping nodes to graph index (Fused Space)
            curr_batch_size: Number of graphs
            fused_source_label: Optional [N] Tensor with 1.0 for true source.
            step: Current simulation step
            time_cost_ms: Execution time of EvidenceBuilder
        """
        if not self.collect_detailed_step_metrics:
            return

        if evidence_state is None:
            return

        # Helper for object/dict access
        def get_val(key):
            if hasattr(evidence_state, key):
                return getattr(evidence_state, key)
            elif isinstance(evidence_state, dict):
                return evidence_state.get(key)
            return None

        # [Performance Audit]
        if time_cost_ms > 0:
            self.probe_b_metrics[f'performance/evidence_builder_ms_step{step}'] = time_cost_ms

        # [Helper] Bucketing Logic
        # We need graph sizes to bucket. Count nodes per batch index.
        # This is expensive? scatter_sum of ones.
        ones = torch.ones_like(batch_index, dtype=torch.float)
        graph_sizes = scatter_sum(ones, batch_index, dim=0, dim_size=curr_batch_size)
        
        def get_size_bucket(size):
            if size < 500: return 'small'
            elif size < 2000: return 'medium'
            else: return 'large'
            
        # [Helper] Stage Logic
        stage = 'early'
        if step == 0: stage = 'step0'
        elif step < 3: stage = 'early'
        elif step < 7: stage = 'middle'
        else: stage = 'late'

        # Helper for scalar extraction
        def get_stat(name, tensor):
            if tensor is None: return 0.0, 0.0
            return tensor.float().mean().item(), tensor.float().std().item()

        # Helper for Audit (True vs Neg) with Bucketing
        def audit_field(field_name, tensor):
            if tensor is None: return
            
            # Global Stats
            mean_val = tensor.float().mean().item()
            self.probe_b_metrics[f'evidence/{field_name}_mean'] = mean_val
            
            # True vs Neg
            if fused_source_label is not None and fused_source_label.size(0) == tensor.size(0):
                label_flat = fused_source_label.view(-1)
                source_mask = (label_flat > 0.5)
                neg_mask = (label_flat < 0.5)
                
                true_mean = 0.0
                neg_mean = 0.0
                gap = 0.0
                
                if source_mask.any():
                    true_mean = tensor[source_mask].mean().item()
                    self.probe_b_metrics[f'evidence/{field_name}_true_mean'] = true_mean
                    
                    if neg_mask.any():
                        neg_mean = tensor[neg_mask].mean().item()
                        self.probe_b_metrics[f'evidence/{field_name}_neg_mean'] = neg_mean
                        gap = true_mean - neg_mean
                        self.probe_b_metrics[f'evidence_gap/{field_name}_true_minus_neg_mean'] = gap
                        
                        # [Per-Step Audit]
                        self.probe_b_metrics[f'evidence_step/{stage}/{field_name}_gap'] = gap

                # [Per-Graph / Size Audit]
                # Calculate mean per graph for True and Neg
                # True Mean per Graph: scatter_sum(val * mask) / scatter_sum(mask)
                # Neg Mean per Graph: scatter_sum(val * mask) / scatter_sum(mask)
                
                if source_mask.any():
                    # True Per Graph
                    true_val = tensor * source_mask.float()
                    true_count = scatter_sum(source_mask.float(), batch_index, dim=0, dim_size=curr_batch_size)
                    true_sum = scatter_sum(true_val, batch_index, dim=0, dim_size=curr_batch_size)
                    true_mean_g = true_sum / (true_count + 1e-8)
                    
                    # Neg Per Graph
                    neg_val = tensor * neg_mask.float()
                    neg_count = scatter_sum(neg_mask.float(), batch_index, dim=0, dim_size=curr_batch_size)
                    neg_sum = scatter_sum(neg_val, batch_index, dim=0, dim_size=curr_batch_size)
                    neg_mean_g = neg_sum / (neg_count + 1e-8)
                    
                    # Gap Per Graph
                    gap_g = true_mean_g - neg_mean_g
                    
                    # Log proportion of graphs with correct direction
                    if 'contradiction' in field_name:
                        correct_dir = (gap_g < 0).float()
                    else:
                        correct_dir = (gap_g > 0).float()
                        
                    valid_g = (true_count > 0) & (neg_count > 0)
                    if valid_g.any():
                        correct_ratio = correct_dir[valid_g].mean().item()
                        self.probe_b_metrics[f'evidence_event/{field_name}_correct_direction_ratio'] = correct_ratio
                        
                        # Median and Quantiles of Gap
                        valid_gaps = gap_g[valid_g]
                        self.probe_b_metrics[f'evidence_event/{field_name}_gap_median'] = valid_gaps.median().item()
                        # simple quantile via sorting
                        sorted_gaps, _ = torch.sort(valid_gaps)
                        n_g = sorted_gaps.size(0)
                        self.probe_b_metrics[f'evidence_event/{field_name}_gap_q25'] = sorted_gaps[int(n_g * 0.25)].item()
                        self.probe_b_metrics[f'evidence_event/{field_name}_gap_q75'] = sorted_gaps[int(n_g * 0.75)].item()

                        # [Size Bucketing]
                        # ... (existing code)
                        for b_idx in range(curr_batch_size):
                            if not valid_g[b_idx]: continue
                            size = graph_sizes[b_idx].item()
                            bucket = get_size_bucket(size)
                            # Log one example per bucket? Or aggregate?
                            # Aggregating inside loop is hard.
                            # Just log to a list and aggregate later? 
                            # Or use prefix keys.
                            # Since we can't easily aggregate dynamically here without clutter,
                            # we will just log global ratio for now.
                            # If we want size breakdown, we need scatter logic for buckets.
                            pass

        # 1. Main Fields Audit
        audit_field('support', get_val('support_score'))
        audit_field('contradiction', get_val('contradiction_score'))
        audit_field('reaction_consistency', get_val('reaction_consistency'))
        audit_field('uncertainty_gap', get_val('uncertainty_gap'))
        
        # 2. Sub-terms Audit
        audit_field('support_toxic', get_val('support_toxic_term'))
        audit_field('support_chlorine', get_val('support_chlorine_term'))
        audit_field('contradiction_toxic', get_val('contradiction_toxic_term'))
        audit_field('contradiction_clean', get_val('contradiction_clean_term'))
        audit_field('consistency_positive', get_val('consistency_positive_term'))
        audit_field('consistency_negative', get_val('consistency_negative_penalty'))

        # [Component Contribution Structure]
        # Calculate Ratio of Sub-terms to Total
        def audit_contribution(total_name, sub_names):
            total = get_val(total_name)
            if total is None: return
            total_abs = total.abs() + 1e-6
            
            for sub in sub_names:
                term = get_val(sub)
                if term is not None:
                    ratio = (term.abs() / total_abs).mean().item()
                    self.probe_b_metrics[f'evidence_contrib/{total_name}_by_{sub}'] = ratio

        audit_contribution('support_score', ['support_toxic_term', 'support_chlorine_term'])
        audit_contribution('contradiction_score', ['contradiction_toxic_term', 'contradiction_clean_term'])
        audit_contribution('reaction_consistency', ['consistency_positive_term', 'consistency_negative_penalty'])

        # 4. Gate Audit
        audit_field('compatibility_gate', get_val('compatibility_gate'))
        audit_field('arrival_gate', get_val('arrival_gate'))
        
        # Calculate Active Ratios for Gates
        def log_active_ratio(name, tensor):
            if tensor is not None:
                active = (tensor > 0.01).float()
                if active.sum() > 0:
                    ratio = active.mean().item()
                    self.probe_b_metrics[f'gate/{name}_active_ratio'] = ratio
                    # Per-step
                    self.probe_b_metrics[f'gate_step/{stage}/{name}_active_ratio'] = ratio
                else:
                    self.probe_b_metrics[f'gate/{name}_active_ratio'] = 0.0
        
        log_active_ratio('compatibility', get_val('compatibility_gate'))
        log_active_ratio('arrival', get_val('arrival_gate'))
        
        # 5. Suspect Pool Reduction Audit
        pool = get_val('suspect_pool')
        if pool is not None:
            if pool.size(0) != batch_index.size(0):
                 # Fallback
                 self.probe_b_metrics['suspect_pool/active_ratio'] = pool.float().sum().item() / max(1, curr_batch_size)
            else:
                 # Count per graph
                 pool_size = scatter_sum(pool.float(), batch_index, dim=0, dim_size=curr_batch_size)
                 total_size = graph_sizes
                 
                 # Active Ratio (Global Mean)
                 active_ratio = (pool_size / (total_size + 1e-6)).mean().item()
                 self.probe_b_metrics['suspect_pool/active_ratio'] = active_ratio
                 
                 # Reduction Ratio
                 reduction_ratio = 1.0 - active_ratio
                 self.probe_b_metrics['suspect_pool/reduction_ratio'] = reduction_ratio
                 
                 # Audit Suspect Pool (True Source Inclusion)
                 if fused_source_label is not None and fused_source_label.size(0) == pool.size(0):
                     label_flat = fused_source_label.view(-1)
                     source_mask = (label_flat > 0.5)
                     if source_mask.any():
                         recall = (pool[source_mask] > 0.5).float().mean().item()
                         self.probe_b_metrics['suspect_pool/true_source_recall'] = recall
        else:
            self.probe_b_metrics['suspect_pool/active_ratio'] = 0.0

    def finalize_metrics(self, graph_success, t_sim, budget_used, total_poison_hits):
        self.probe_b_metrics["probeB/hit_rate_final"] = graph_success.float().mean().item()
        self.probe_b_metrics["probeB/budget_used_mean"] = budget_used.mean().item()
        self.probe_b_metrics["rollout/global/avg_poison_hits_per_event"] = total_poison_hits.mean().item()
        return self.probe_b_metrics
