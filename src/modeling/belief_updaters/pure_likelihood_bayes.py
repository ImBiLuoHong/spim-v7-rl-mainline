from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch

from src.modeling.interfaces.belief_updater import BeliefStateBase, BeliefUpdaterBase, BeliefUpdaterCapabilities
from src.modeling.registry import BELIEF_UPDATER_REGISTRY


def _soft_window(delta_t: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, s_early: float, s_late: float) -> torch.Tensor:
    lower = lower.clamp_min(0.0)
    upper = torch.maximum(upper, lower + 1e-6)
    early_gate = torch.sigmoid((delta_t - lower) / max(float(s_early), 1e-6))
    late_gate = torch.sigmoid((upper - delta_t) / max(float(s_late), 1e-6))
    return (early_gate * late_gate).clamp(0.0, 1.0)


class StaticPropagationAssets:
    def __init__(self, graph_path: Path) -> None:
        self.graph_path = Path(graph_path)
        self._loaded = False
        self._n_nodes = 0
        self._rev_rel: Optional[sp.csr_matrix] = None
        self._rev_min: Optional[sp.csr_matrix] = None
        self._rev_med: Optional[sp.csr_matrix] = None
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with np.load(self.graph_path, allow_pickle=True) as f:
            edge_index = f["edge_index"]
            summary = f["edge_attr_summary"]
        u = edge_index[0].astype(np.int64)
        v = edge_index[1].astype(np.int64)
        p_forward = np.clip(summary[:, 0].astype(np.float64), 1e-4, 1.0 - 1e-4)
        q50 = np.maximum(summary[:, 1].astype(np.float64), 1e-4)
        min_stt = np.maximum(summary[:, 3].astype(np.float64), 1e-4)

        # Reverse graph: observation seed -> upstream candidate
        src = np.concatenate([v, u])
        dst = np.concatenate([u, v])
        rel_cost = np.concatenate([-np.log(p_forward), -np.log(1.0 - p_forward)])
        med_cost = np.concatenate([q50, q50])
        min_cost = np.concatenate([min_stt, min_stt])

        n_nodes = int(max(edge_index.max(), src.max(), dst.max())) + 1
        self._n_nodes = n_nodes
        self._rev_rel = sp.coo_matrix((rel_cost, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
        self._rev_med = sp.coo_matrix((med_cost, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
        self._rev_min = sp.coo_matrix((min_cost, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
        self._loaded = True

    def metrics_to_observation(self, obs_global_idx: int, candidate_global_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_loaded()
        obs_global_idx = int(obs_global_idx)
        if obs_global_idx not in self._cache:
            rel_cost = dijkstra(self._rev_rel, directed=True, indices=[obs_global_idx], return_predecessors=False)[0]
            min_cost = dijkstra(self._rev_min, directed=True, indices=[obs_global_idx], return_predecessors=False)[0]
            med_cost = dijkstra(self._rev_med, directed=True, indices=[obs_global_idx], return_predecessors=False)[0]
            rel = np.exp(-rel_cost)
            rel[~np.isfinite(rel_cost)] = 0.0
            min_cost[~np.isfinite(min_cost)] = np.inf
            med_cost[~np.isfinite(med_cost)] = np.inf
            self._cache[obs_global_idx] = (rel.astype(np.float32), min_cost.astype(np.float32), med_cost.astype(np.float32))
        rel_all, min_all, med_all = self._cache[obs_global_idx]
        return rel_all[candidate_global_ids], min_all[candidate_global_ids], med_all[candidate_global_ids]


@dataclass
class PureLikelihoodBeliefState(BeliefStateBase):
    log_joint: Optional[torch.Tensor]
    processed_record_count: int
    onset_grid_min: Optional[torch.Tensor]
    candidate_mask: Optional[torch.Tensor]
    candidate_global_ids: Optional[torch.Tensor]

    def detach(self):
        return PureLikelihoodBeliefState(
            log_joint=None if self.log_joint is None else self.log_joint.detach(),
            processed_record_count=int(self.processed_record_count),
            onset_grid_min=None if self.onset_grid_min is None else self.onset_grid_min.detach(),
            candidate_mask=None if self.candidate_mask is None else self.candidate_mask.detach(),
            candidate_global_ids=None if self.candidate_global_ids is None else self.candidate_global_ids.detach(),
        )

    def to(self, device: torch.device):
        return PureLikelihoodBeliefState(
            log_joint=None if self.log_joint is None else self.log_joint.to(device),
            processed_record_count=int(self.processed_record_count),
            onset_grid_min=None if self.onset_grid_min is None else self.onset_grid_min.to(device),
            candidate_mask=None if self.candidate_mask is None else self.candidate_mask.to(device),
            candidate_global_ids=None if self.candidate_global_ids is None else self.candidate_global_ids.to(device),
        )


@BELIEF_UPDATER_REGISTRY.register("pure_likelihood_bayes")
class PureLikelihoodBayesBelief(BeliefUpdaterBase):
    def __init__(
        self,
        *,
        eps_fp: float = 0.02,
        eps_fn: float = 0.05,
        s_early: float = 15.0,
        s_late: float = 30.0,
        onset_radius_episodes: int = 2,
        mass_cover_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.eps_fp = float(eps_fp)
        self.eps_fn = float(eps_fn)
        self.s_early = float(s_early)
        self.s_late = float(s_late)
        self.onset_radius_episodes = int(max(0, onset_radius_episodes))
        self.mass_cover_threshold = float(mass_cover_threshold)
        self._asset_cache: Dict[str, StaticPropagationAssets] = {}

    def capabilities(self) -> BeliefUpdaterCapabilities:
        return {
            "provides_node_belief": True,
            "provides_global_belief": True,
            "provides_memory_bank": False,
            "output_fields": [
                "belief",
                "joint_posterior",
                "entropy",
                "normalized_entropy",
                "top1_mass",
                "top3_mass",
                "top5_mass",
                "mass_cover_size",
                "mass_cover_size_ratio",
                "mass_cover_mass",
                "hardest_confuser_local",
                "margin_true_vs_hard",
            ],
        }

    def init_state(self, batch_size: int, num_nodes: Optional[int] = None, device: Optional[torch.device] = None) -> PureLikelihoodBeliefState:
        return PureLikelihoodBeliefState(
            log_joint=None,
            processed_record_count=0,
            onset_grid_min=None,
            candidate_mask=None,
            candidate_global_ids=None,
        )

    def _assets_from_topology(self, topology: Any) -> StaticPropagationAssets:
        graph_path = Path(topology.graph_path)
        key = str(graph_path)
        if key not in self._asset_cache:
            self._asset_cache[key] = StaticPropagationAssets(graph_path)
        return self._asset_cache[key]

    def _candidate_mask(self, step_in: Dict[str, Any]) -> torch.Tensor:
        phys_ctx = step_in["physics_ctx"]
        constraint_state = step_in.get("constraint_state")
        feasible = phys_ctx.feasible_mask.view(-1).bool()
        confirmed_non_source = (
            constraint_state.confirmed_non_source_mask.view(-1).bool()
            if constraint_state is not None and getattr(constraint_state, "confirmed_non_source_mask", None) is not None
            else torch.zeros_like(feasible)
        )
        return feasible & (~confirmed_non_source)

    def _build_onset_grid(self, episode_duration_min: float, device: torch.device) -> torch.Tensor:
        step = float(max(episode_duration_min, 1e-6))
        offsets = torch.arange(-self.onset_radius_episodes, self.onset_radius_episodes + 1, device=device, dtype=torch.float32)
        return offsets * step

    def _observation_loglik(
        self,
        *,
        obs_global_idx: int,
        obs_time_min: float,
        is_positive: bool,
        candidate_global_ids: torch.Tensor,
        onset_grid_min: torch.Tensor,
        assets: StaticPropagationAssets,
    ) -> torch.Tensor:
        cand_np = candidate_global_ids.detach().cpu().numpy().astype(np.int64)
        rel_np, min_np, med_np = assets.metrics_to_observation(int(obs_global_idx), cand_np)
        rel = torch.from_numpy(rel_np).to(device=onset_grid_min.device, dtype=torch.float32).unsqueeze(1)
        min_tt = torch.from_numpy(min_np).to(device=onset_grid_min.device, dtype=torch.float32).unsqueeze(1)
        med_tt = torch.from_numpy(med_np).to(device=onset_grid_min.device, dtype=torch.float32).unsqueeze(1)
        delta_t = float(obs_time_min) - onset_grid_min.view(1, -1)
        upper = 2.0 * med_tt - min_tt
        window = _soft_window(delta_t, min_tt, upper, self.s_early, self.s_late)
        p_pos = self.eps_fp + (1.0 - self.eps_fp - self.eps_fn) * rel * window
        p_pos = p_pos.clamp(1e-9, 1.0 - 1e-9)
        return torch.log(p_pos if bool(is_positive) else (1.0 - p_pos))

    def _joint_to_ctx(
        self,
        *,
        log_joint: torch.Tensor,
        candidate_mask: torch.Tensor,
        source_local: Optional[int],
        onset_grid_min: torch.Tensor,
    ) -> Dict[str, Any]:
        joint = torch.softmax(log_joint.view(-1), dim=0).view_as(log_joint)
        belief = joint.sum(dim=1)
        belief = belief / belief.sum().clamp_min(1e-9)
        valid_belief = belief[candidate_mask]
        entropy = float((-(valid_belief.clamp_min(1e-12) * torch.log(valid_belief.clamp_min(1e-12)))).sum().item()) if valid_belief.numel() > 0 else 0.0
        denom = math.log(max(int(valid_belief.numel()), 2))
        normalized_entropy = float(entropy / denom) if denom > 0 else 0.0
        order = torch.argsort(belief, descending=True)
        order = order[candidate_mask[order]]
        top1_mass = float(valid_belief.max().item()) if valid_belief.numel() > 0 else 0.0
        top3_mass = float(valid_belief[: min(3, valid_belief.numel())].sum().item()) if valid_belief.numel() > 0 else 0.0
        top5_mass = float(valid_belief[: min(5, valid_belief.numel())].sum().item()) if valid_belief.numel() > 0 else 0.0
        sorted_vals = belief[order]
        csum = torch.cumsum(sorted_vals, dim=0)
        cover_idx = int((csum >= self.mass_cover_threshold).nonzero(as_tuple=True)[0][0].item()) + 1 if sorted_vals.numel() > 0 else 0
        cover_set = order[:cover_idx]
        mass_cover_mass = float(csum[cover_idx - 1].item()) if cover_idx > 0 else 0.0
        hardest = None
        if source_local is not None:
            for idx in order.tolist():
                if int(idx) != int(source_local):
                    hardest = int(idx)
                    break
        true_mass = float(belief[int(source_local)].item()) if source_local is not None and bool(candidate_mask[int(source_local)].item()) else None
        hard_mass = float(belief[int(hardest)].item()) if hardest is not None else None
        margin = (float(true_mass - hard_mass) if true_mass is not None and hard_mass is not None else None)
        return {
            "belief": belief,
            "joint_posterior": joint,
            "candidate_mask": candidate_mask,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "effective_support": float(math.exp(entropy)),
            "top1_mass": top1_mass,
            "top3_mass": top3_mass,
            "top5_mass": top5_mass,
            "mass_cover_set": cover_set,
            "mass_cover_size": int(cover_idx),
            "mass_cover_size_ratio": float(cover_idx / max(int(candidate_mask.sum().item()), 1)),
            "mass_cover_mass": mass_cover_mass,
            "ordered_candidates": order,
            "source_local": source_local,
            "hardest_confuser_local": hardest,
            "true_mass": true_mass,
            "hard_mass": hard_mass,
            "margin_true_vs_hard": margin,
            "onset_grid_min": onset_grid_min,
        }

    def _source_probs_to_ctx(
        self,
        *,
        belief: torch.Tensor,
        candidate_mask: torch.Tensor,
        source_local: Optional[int],
    ) -> Dict[str, Any]:
        belief = belief.view(-1).float()
        belief = belief / belief.sum().clamp_min(1e-9)
        valid_belief = belief[candidate_mask]
        entropy = float((-(valid_belief.clamp_min(1e-12) * torch.log(valid_belief.clamp_min(1e-12)))).sum().item()) if valid_belief.numel() > 0 else 0.0
        denom = math.log(max(int(valid_belief.numel()), 2))
        normalized_entropy = float(entropy / denom) if denom > 0 else 0.0
        order = torch.argsort(belief, descending=True)
        order = order[candidate_mask[order]]
        sorted_vals = belief[order]
        top1_mass = float(sorted_vals[:1].sum().item()) if sorted_vals.numel() > 0 else 0.0
        top3_mass = float(sorted_vals[: min(3, sorted_vals.numel())].sum().item()) if sorted_vals.numel() > 0 else 0.0
        top5_mass = float(sorted_vals[: min(5, sorted_vals.numel())].sum().item()) if sorted_vals.numel() > 0 else 0.0
        csum = torch.cumsum(sorted_vals, dim=0)
        cover_idx = int((csum >= self.mass_cover_threshold).nonzero(as_tuple=True)[0][0].item()) + 1 if sorted_vals.numel() > 0 else 0
        cover_set = order[:cover_idx]
        mass_cover_mass = float(csum[cover_idx - 1].item()) if cover_idx > 0 else 0.0
        hardest = None
        if source_local is not None:
            for idx in order.tolist():
                if int(idx) != int(source_local):
                    hardest = int(idx)
                    break
        true_mass = float(belief[int(source_local)].item()) if source_local is not None and bool(candidate_mask[int(source_local)].item()) else None
        hard_mass = float(belief[int(hardest)].item()) if hardest is not None else None
        margin = (float(true_mass - hard_mass) if true_mass is not None and hard_mass is not None else None)
        return {
            "belief": belief,
            "candidate_mask": candidate_mask,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "effective_support": float(math.exp(entropy)),
            "top1_mass": top1_mass,
            "top3_mass": top3_mass,
            "top5_mass": top5_mass,
            "mass_cover_set": cover_set,
            "mass_cover_size": int(cover_idx),
            "mass_cover_size_ratio": float(cover_idx / max(int(candidate_mask.sum().item()), 1)),
            "mass_cover_mass": mass_cover_mass,
            "ordered_candidates": order,
            "source_local": source_local,
            "hardest_confuser_local": hardest,
            "true_mass": true_mass,
            "hard_mass": hard_mass,
            "margin_true_vs_hard": margin,
        }

    def _per_source_predictive_likelihood(
        self,
        *,
        state: PureLikelihoodBeliefState,
        step_in: Dict[str, Any],
        action_global_idx: int,
        obs_time_min: float,
        outcome_positive: bool,
    ) -> torch.Tensor:
        if state.log_joint is None:
            raise RuntimeError("predictive likelihood requires initialized belief state.")
        assets = self._assets_from_topology(step_in["topology"])
        loglik = self._observation_loglik(
            obs_global_idx=int(action_global_idx),
            obs_time_min=float(obs_time_min),
            is_positive=bool(outcome_positive),
            candidate_global_ids=state.candidate_global_ids,
            onset_grid_min=state.onset_grid_min,
            assets=assets,
        )
        joint = torch.softmax(state.log_joint.view(-1), dim=0).view_as(state.log_joint)
        source_mass = joint.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cond_tau = joint / source_mass
        per_source = (cond_tau * torch.exp(loglik)).sum(dim=1).clamp(1e-9, 1.0 - 1e-9)
        per_source[~state.candidate_mask] = 0.0
        return per_source

    def branch_update(
        self,
        *,
        state: PureLikelihoodBeliefState,
        step_in: Dict[str, Any],
        action_global_idx: int,
        obs_time_min: float,
        outcome_positive: bool,
        source_local: Optional[int],
    ) -> Dict[str, Any]:
        if state.log_joint is None:
            raise RuntimeError("branch_update requires initialized belief state.")
        assets = self._assets_from_topology(step_in["topology"])
        loglik = self._observation_loglik(
            obs_global_idx=int(action_global_idx),
            obs_time_min=float(obs_time_min),
            is_positive=bool(outcome_positive),
            candidate_global_ids=state.candidate_global_ids,
            onset_grid_min=state.onset_grid_min,
            assets=assets,
        )
        branch_log_joint = state.log_joint + loglik
        return self._joint_to_ctx(
            log_joint=branch_log_joint,
            candidate_mask=state.candidate_mask,
            source_local=source_local,
            onset_grid_min=state.onset_grid_min,
        )

    def predictive_positive_probability(
        self,
        *,
        state: PureLikelihoodBeliefState,
        step_in: Dict[str, Any],
        action_global_idx: int,
        obs_time_min: float,
    ) -> float:
        if state.log_joint is None:
            raise RuntimeError("predictive_positive_probability requires initialized belief state.")
        assets = self._assets_from_topology(step_in["topology"])
        loglik_pos = self._observation_loglik(
            obs_global_idx=int(action_global_idx),
            obs_time_min=float(obs_time_min),
            is_positive=True,
            candidate_global_ids=state.candidate_global_ids,
            onset_grid_min=state.onset_grid_min,
            assets=assets,
        )
        joint = torch.softmax(state.log_joint.view(-1), dim=0).view_as(state.log_joint)
        p_pos = float((joint * torch.exp(loglik_pos)).sum().item())
        return float(min(max(p_pos, 1e-9), 1.0 - 1e-9))

    def hybrid_predictive_positive_probability(
        self,
        *,
        state: PureLikelihoodBeliefState,
        step_in: Dict[str, Any],
        source_prior: torch.Tensor,
        action_global_idx: int,
        obs_time_min: float,
    ) -> float:
        prior = source_prior.view(-1).float()
        prior = prior / prior.sum().clamp_min(1e-9)
        per_source_pos = self._per_source_predictive_likelihood(
            state=state,
            step_in=step_in,
            action_global_idx=int(action_global_idx),
            obs_time_min=float(obs_time_min),
            outcome_positive=True,
        )
        p_pos = float((prior * per_source_pos).sum().item())
        return float(min(max(p_pos, 1e-9), 1.0 - 1e-9))

    def hybrid_branch_update(
        self,
        *,
        state: PureLikelihoodBeliefState,
        step_in: Dict[str, Any],
        source_prior: torch.Tensor,
        action_global_idx: int,
        obs_time_min: float,
        outcome_positive: bool,
        source_local: Optional[int],
    ) -> Dict[str, Any]:
        prior = source_prior.view(-1).float()
        prior = prior / prior.sum().clamp_min(1e-9)
        per_source_lik = self._per_source_predictive_likelihood(
            state=state,
            step_in=step_in,
            action_global_idx=int(action_global_idx),
            obs_time_min=float(obs_time_min),
            outcome_positive=bool(outcome_positive),
        )
        branch = prior * per_source_lik
        branch[~state.candidate_mask] = 0.0
        branch = branch / branch.sum().clamp_min(1e-9)
        return self._source_probs_to_ctx(
            belief=branch,
            candidate_mask=state.candidate_mask,
            source_local=source_local,
        )

    def _step_impl(self, state: PureLikelihoodBeliefState, step_in: Dict[str, Any]) -> Tuple[PureLikelihoodBeliefState, Dict[str, Any]]:
        history = step_in["history"]
        topology = step_in["topology"]
        global_node_ids = step_in["global_node_ids"].view(-1).long()
        device = global_node_ids.device
        episode_duration_min = float(step_in.get("episode_duration_min", 45.0))
        source_local = step_in.get("source_local")

        candidate_mask = self._candidate_mask(step_in).to(device=device)
        candidate_global_ids = global_node_ids
        if state.log_joint is None:
            onset_grid = self._build_onset_grid(episode_duration_min, device=device)
            prior = torch.full((global_node_ids.numel(), onset_grid.numel()), -float("inf"), device=device)
            num_candidates = int(candidate_mask.sum().item())
            if num_candidates > 0:
                prior[candidate_mask] = -math.log(float(num_candidates)) - math.log(float(onset_grid.numel()))
            log_joint = prior
            processed_count = 0
        else:
            onset_grid = state.onset_grid_min.to(device)
            log_joint = state.log_joint.to(device)
            log_joint[~candidate_mask] = -float("inf")
            processed_count = int(state.processed_record_count)

        assets = self._assets_from_topology(topology)
        new_records = history.records[processed_count:]
        for record in new_records:
            loglik = self._observation_loglik(
                obs_global_idx=int(record.node_global_idx),
                obs_time_min=float(record.absolute_time_min),
                is_positive=(str(record.label) == "positive"),
                candidate_global_ids=candidate_global_ids,
                onset_grid_min=onset_grid,
                assets=assets,
            )
            log_joint = log_joint + loglik
        new_state = PureLikelihoodBeliefState(
            log_joint=log_joint.detach(),
            processed_record_count=processed_count + len(new_records),
            onset_grid_min=onset_grid.detach(),
            candidate_mask=candidate_mask.detach(),
            candidate_global_ids=candidate_global_ids.detach(),
        )
        ctx = self._joint_to_ctx(
            log_joint=log_joint,
            candidate_mask=candidate_mask,
            source_local=(None if source_local is None else int(source_local)),
            onset_grid_min=onset_grid,
        )
        ctx["audit"] = {
            "eps_fp": float(self.eps_fp),
            "eps_fn": float(self.eps_fn),
            "s_early": float(self.s_early),
            "s_late": float(self.s_late),
            "onset_radius_episodes": int(self.onset_radius_episodes),
            "processed_record_count": int(new_state.processed_record_count),
            "new_record_count": int(len(new_records)),
            "candidate_count": int(candidate_mask.sum().item()),
        }
        return new_state, ctx
