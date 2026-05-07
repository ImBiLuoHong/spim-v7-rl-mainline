from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import SAGEConv


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.view(-1).float()
    mask = mask.view(-1).bool()
    if not bool(mask.any()):
        return values.new_tensor(0.0)
    return values[mask].mean()


def bound_nonnegative_score(score: torch.Tensor) -> torch.Tensor:
    score = score.float().clamp_min(0.0)
    return score / (1.0 + score)


def derive_two_channel_features(
    support_score: torch.Tensor,
    contradiction_score: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    support_raw = support_score.view(-1).float().clamp_min(0.0)
    contradiction_raw = contradiction_score.view(-1).float().clamp_min(0.0)
    support_bounded = bound_nonnegative_score(support_raw)
    contradiction_bounded = bound_nonnegative_score(contradiction_raw)
    live_plausibility = support_bounded * (1.0 - contradiction_bounded)
    conflict_mass = support_bounded * contradiction_bounded
    ignorance_mass = (1.0 - support_bounded) * (1.0 - contradiction_bounded)
    unresolved_mass = conflict_mass + ignorance_mass
    return {
        "support_raw": support_raw,
        "contradiction_raw": contradiction_raw,
        "support_bounded": support_bounded,
        "contradiction_bounded": contradiction_bounded,
        "live_plausibility": live_plausibility,
        "conflict_mass": conflict_mass,
        "ignorance_mass": ignorance_mass,
        "unresolved_mass": unresolved_mass,
    }


def compute_clean_transition_metrics(
    pre_state: Dict[str, torch.Tensor],
    post_state: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    pre_two = derive_two_channel_features(
        pre_state["support_score"],
        pre_state["contradiction_score"],
    )
    post_two = derive_two_channel_features(
        post_state["support_score"],
        post_state["contradiction_score"],
    )
    pre_valid = pre_state["valid_mask"].view(-1).bool()
    post_valid = post_state["valid_mask"].view(-1).bool()
    pre_pair = pre_state["pair_available"].view(-1).float()
    post_pair = post_state["pair_available"].view(-1).float()

    ignorance_before = float(masked_mean(pre_two["ignorance_mass"], pre_valid).item())
    ignorance_after = float(masked_mean(post_two["ignorance_mass"], post_valid).item())
    conflict_before = float(masked_mean(pre_two["conflict_mass"], pre_valid).item())
    conflict_after = float(masked_mean(post_two["conflict_mass"], post_valid).item())
    live_before = float(masked_mean(pre_two["live_plausibility"], pre_valid).item())
    live_after = float(masked_mean(post_two["live_plausibility"], post_valid).item())
    support_before = float(masked_mean(pre_two["support_bounded"], pre_valid).item())
    support_after = float(masked_mean(post_two["support_bounded"], post_valid).item())
    unresolved_before = float(masked_mean(pre_two["unresolved_mass"], pre_valid).item())
    unresolved_after = float(masked_mean(post_two["unresolved_mass"], post_valid).item())
    pair_before = float(masked_mean(pre_pair, pre_valid).item())
    pair_after = float(masked_mean(post_pair, post_valid).item())

    ignorance_delta = ignorance_before - ignorance_after
    conflict_delta = conflict_before - conflict_after
    live_delta = live_after - live_before
    support_delta = support_after - support_before
    pair_delta = pair_after - pair_before
    unresolved_delta = unresolved_before - unresolved_after
    reward = ignorance_delta + 0.5 * conflict_delta + 0.25 * pair_delta

    return {
        "reward": float(reward),
        "ignorance_before": ignorance_before,
        "ignorance_after": ignorance_after,
        "ignorance_delta": float(ignorance_delta),
        "conflict_before": conflict_before,
        "conflict_after": conflict_after,
        "conflict_delta": float(conflict_delta),
        "live_before": live_before,
        "live_after": live_after,
        "live_delta": float(live_delta),
        "support_before": support_before,
        "support_after": support_after,
        "support_delta": float(support_delta),
        "pair_available_before": pair_before,
        "pair_available_after": pair_after,
        "pair_available_delta": float(pair_delta),
        "unresolved_before": unresolved_before,
        "unresolved_after": unresolved_after,
        "unresolved_delta": float(unresolved_delta),
    }


def pick_topk_valid(scores: torch.Tensor, valid_mask: torch.Tensor, k: int) -> List[int]:
    scores = scores.view(-1).float()
    valid_mask = valid_mask.view(-1).bool()
    valid_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return []
    k = min(int(k), int(valid_idx.numel()))
    ordered = sorted(
        valid_idx.tolist(),
        key=lambda idx: (-float(scores[int(idx)].item()), int(idx)),
    )
    return [int(idx) for idx in ordered[:k]]


def random_valid_pick(valid_mask: torch.Tensor, k: int, generator: torch.Generator) -> List[int]:
    valid_idx = torch.nonzero(valid_mask.view(-1).bool(), as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return []
    perm = torch.randperm(valid_idx.numel(), generator=generator, device="cpu")
    count = min(int(k), int(valid_idx.numel()))
    return valid_idx[perm[:count]].tolist()


def mean_pairwise_distance(selected_indices: Sequence[int], pairwise_distance: torch.Tensor) -> float:
    if len(selected_indices) < 2:
        return 0.0
    vals: List[float] = []
    for left, right in combinations(selected_indices, 2):
        dist = float(pairwise_distance[int(left), int(right)].item())
        if torch.isfinite(pairwise_distance[int(left), int(right)]):
            vals.append(dist)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _normalize_overlap_features(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    features = features.float()
    norms = torch.linalg.norm(features, dim=-1, keepdim=True).clamp_min(float(eps))
    return torch.where(features.abs().sum(dim=-1, keepdim=True) > float(eps), features / norms, torch.zeros_like(features))


def compute_mean_pairwise_overlap(
    selected_indices: Sequence[int],
    overlap_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    if len(selected_indices) < 2:
        return 0.0
    normalized = _normalize_overlap_features(overlap_features, eps=eps)
    vals: List[float] = []
    for left, right in combinations(selected_indices, 2):
        similarity = float((normalized[int(left)] * normalized[int(right)]).sum().item())
        vals.append(max(similarity, 0.0))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def compute_mean_pairwise_jaccard_overlap(
    selected_indices: Sequence[int],
    overlap_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    if len(selected_indices) < 2 or overlap_features.numel() == 0:
        return 0.0
    binary_features = (overlap_features.float() > float(eps)).float()
    vals: List[float] = []
    for left, right in combinations(selected_indices, 2):
        left_vec = binary_features[int(left)]
        right_vec = binary_features[int(right)]
        intersection = float((left_vec * right_vec).sum().item())
        union = float(((left_vec + right_vec) > 0.0).float().sum().item())
        vals.append(intersection / union if union > 0.0 else 0.0)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def compute_role_aware_slot_overlap_penalty(
    slot_idx: int,
    selected_indices: Sequence[int],
    overlap_features: torch.Tensor,
    mode: str = "role_aware_witness_pair_jaccard",
    eps: float = 1e-6,
) -> torch.Tensor:
    overlap_features = overlap_features.float()
    num_nodes = int(overlap_features.size(0))
    if slot_idx <= 0 or len(selected_indices) <= 0 or overlap_features.numel() == 0:
        return overlap_features.new_zeros(num_nodes)
    binary_features = (overlap_features > float(eps)).float()
    if mode == "slot1_frontier_anchor_witness_pair_jaccard":
        if slot_idx != 1:
            return overlap_features.new_zeros(num_nodes)
        reference_slots = [0]
    elif slot_idx == 1:
        reference_slots = [0]
    else:
        reference_slots = list(range(min(int(slot_idx), len(selected_indices))))
    penalties: List[torch.Tensor] = []
    candidate_count = binary_features.sum(dim=1, keepdim=True)
    for ref_slot in reference_slots:
        selected_idx = int(selected_indices[int(ref_slot)])
        selected_vector = binary_features[selected_idx : selected_idx + 1]
        intersections = torch.matmul(binary_features, selected_vector.t())
        selected_count = selected_vector.sum(dim=1, keepdim=True).transpose(0, 1)
        unions = candidate_count + selected_count - intersections
        penalties.append(
            torch.where(
                unions > 0.0,
                intersections / unions.clamp_min(1.0),
                torch.zeros_like(unions),
            ).view(-1)
        )
    if not penalties:
        return overlap_features.new_zeros(num_nodes)
    return torch.stack(penalties, dim=1).mean(dim=1)


class _SlotHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class CleanNavigatorV1(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        graph_feature_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_slots: int = 3,
        greedy_eval: bool = True,
        diversity_mode: str = "none",
        diversity_penalty_weight: float = 0.0,
        complementarity_mode: str = "none",
        complementarity_penalty_weight: float = 0.0,
        role_mode: str = "none",
        role_bias_weight: float = 0.0,
        credit_mode: str = "state_value",
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.graph_feature_dim = int(graph_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.greedy_eval = bool(greedy_eval)
        self.diversity_mode = str(diversity_mode)
        self.diversity_penalty_weight = float(max(diversity_penalty_weight, 0.0))
        self.complementarity_mode = str(complementarity_mode)
        self.complementarity_penalty_weight = float(max(complementarity_penalty_weight, 0.0))
        self.role_mode = str(role_mode)
        self.role_bias_weight = float(max(role_bias_weight, 0.0))
        self.credit_mode = str(credit_mode)

        self.node_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.convs = nn.ModuleList(
            [SAGEConv(self.hidden_dim, self.hidden_dim, aggr="mean") for _ in range(max(int(num_layers), 1))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(max(int(num_layers), 1))])
        self.slot_embeddings = nn.Parameter(torch.randn(self.num_slots, self.hidden_dim))
        actor_in_dim = self.hidden_dim * 3 + self.graph_feature_dim
        self.slot_heads = nn.ModuleList([_SlotHead(actor_in_dim, self.hidden_dim) for _ in range(self.num_slots)])
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.graph_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.set_value_head = (
            nn.Sequential(
                nn.Linear(self.hidden_dim * (self.num_slots + 1) + self.graph_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            if self.credit_mode == "action_set_q"
            else None
        )

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(node_features.float())
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(F.relu(h) + residual)
        return h

    def build_selected_slot_features(
        self,
        node_embeddings: torch.Tensor,
        selected_indices: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        selected_features = node_embeddings.new_zeros(self.num_slots, self.hidden_dim)
        if isinstance(selected_indices, torch.Tensor):
            selected_list = selected_indices.view(-1).tolist()
        else:
            selected_list = list(selected_indices)
        for slot_idx, node_idx in enumerate(selected_list[: self.num_slots]):
            selected_features[slot_idx] = node_embeddings[int(node_idx)]
        return selected_features.reshape(-1)

    def _prepare_policy_context(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: torch.Tensor,
        graph_features: torch.Tensor,
        redundancy_features: torch.Tensor | None,
        complementarity_features: torch.Tensor | None,
        role_potentials: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | None]:
        node_features = node_features.float()
        valid_mask = valid_mask.view(-1).bool()
        graph_features = graph_features.view(-1).float()
        h = self.encode(node_features, edge_index)
        graph_context = h.mean(dim=0)
        value = self.value_head(torch.cat([graph_context, graph_features], dim=0)).view(())
        graph_ctx_expand = graph_context.unsqueeze(0).expand(h.size(0), -1)
        graph_feat_expand = graph_features.unsqueeze(0).expand(h.size(0), -1)

        normalized_redundancy = None
        if redundancy_features is not None and self.diversity_penalty_weight > 0.0:
            if self.diversity_mode == "history_relation_overlap":
                normalized_redundancy = _normalize_overlap_features(redundancy_features.to(h.device))
            elif self.diversity_mode == "witness_pair_jaccard":
                normalized_redundancy = (redundancy_features.to(h.device).float() > 1e-6).float()

        role_aware_overlap_features = None
        if (
            complementarity_features is not None
            and self.complementarity_penalty_weight > 0.0
            and self.complementarity_mode in {
                "role_aware_witness_pair_jaccard",
                "slot1_frontier_anchor_witness_pair_jaccard",
            }
        ):
            role_aware_overlap_features = complementarity_features.to(h.device).float()

        return {
            "node_features": node_features,
            "valid_mask": valid_mask,
            "graph_features": graph_features,
            "h": h,
            "graph_context": graph_context,
            "value": value,
            "graph_ctx_expand": graph_ctx_expand,
            "graph_feat_expand": graph_feat_expand,
            "normalized_redundancy": normalized_redundancy,
            "role_aware_overlap_features": role_aware_overlap_features,
            "role_potentials": None if role_potentials is None else role_potentials.to(h.device).float(),
        }

    def _slot_logits(
        self,
        *,
        slot_idx: int,
        available: torch.Tensor,
        selected: Sequence[int],
        policy_ctx: Dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        h = policy_ctx["h"]
        assert isinstance(h, torch.Tensor)
        graph_ctx_expand = policy_ctx["graph_ctx_expand"]
        graph_feat_expand = policy_ctx["graph_feat_expand"]
        assert isinstance(graph_ctx_expand, torch.Tensor)
        assert isinstance(graph_feat_expand, torch.Tensor)

        slot_embed = self.slot_embeddings[slot_idx].unsqueeze(0).expand(h.size(0), -1)
        slot_input = torch.cat([h, graph_ctx_expand, graph_feat_expand, slot_embed], dim=-1)
        logits = self.slot_heads[slot_idx](slot_input)

        role_potentials = policy_ctx["role_potentials"]
        if (
            role_potentials is not None
            and self.role_mode == "slot_bias"
            and self.role_bias_weight > 0.0
            and slot_idx < int(role_potentials.size(1))
        ):
            logits = logits + self.role_bias_weight * role_potentials[:, slot_idx]

        normalized_redundancy = policy_ctx["normalized_redundancy"]
        if normalized_redundancy is not None and selected:
            selected_tensor = torch.tensor(list(selected), dtype=torch.long, device=h.device)
            selected_vectors = normalized_redundancy[selected_tensor]
            if self.diversity_mode == "witness_pair_jaccard":
                intersections = torch.matmul(normalized_redundancy, selected_vectors.t())
                candidate_count = normalized_redundancy.sum(dim=1, keepdim=True)
                selected_count = selected_vectors.sum(dim=1, keepdim=True).transpose(0, 1)
                unions = candidate_count + selected_count - intersections
                overlap_penalty = torch.where(
                    unions > 0.0,
                    intersections / unions.clamp_min(1.0),
                    torch.zeros_like(unions),
                ).mean(dim=1)
            else:
                overlap_penalty = torch.matmul(normalized_redundancy, selected_vectors.t()).clamp_min(0.0).mean(dim=1)
            logits = logits - self.diversity_penalty_weight * overlap_penalty

        role_aware_overlap_features = policy_ctx["role_aware_overlap_features"]
        if role_aware_overlap_features is not None and selected:
            role_overlap_penalty = compute_role_aware_slot_overlap_penalty(
                slot_idx=slot_idx,
                selected_indices=selected,
                overlap_features=role_aware_overlap_features,
                mode=self.complementarity_mode,
            )
            logits = logits - self.complementarity_penalty_weight * role_overlap_penalty

        return logits.masked_fill(~available, -1e9)

    def _run_policy(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: torch.Tensor,
        graph_features: torch.Tensor,
        selected_indices: Sequence[int] | None = None,
        deterministic: bool = False,
        redundancy_features: torch.Tensor | None = None,
        complementarity_features: torch.Tensor | None = None,
        role_potentials: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        training_sampling_mode: str = "explicit_cpu_multinomial",
        compute_sampling_stats: bool = True,
        slot_logits_limit: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        policy_ctx = self._prepare_policy_context(
            node_features=node_features,
            edge_index=edge_index,
            valid_mask=valid_mask,
            graph_features=graph_features,
            redundancy_features=redundancy_features,
            complementarity_features=complementarity_features,
            role_potentials=role_potentials,
        )
        return self._run_policy_from_context(
            policy_ctx=policy_ctx,
            selected_indices=selected_indices,
            deterministic=deterministic,
            generator=generator,
            training_sampling_mode=training_sampling_mode,
            compute_sampling_stats=compute_sampling_stats,
            slot_logits_limit=slot_logits_limit,
        )

    def _run_policy_from_context(
        self,
        *,
        policy_ctx: Dict[str, torch.Tensor | None],
        selected_indices: Sequence[int] | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        training_sampling_mode: str = "explicit_cpu_multinomial",
        compute_sampling_stats: bool = True,
        slot_logits_limit: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        node_features = policy_ctx["node_features"]
        valid_mask = policy_ctx["valid_mask"]
        graph_features = policy_ctx["graph_features"]
        h = policy_ctx["h"]
        graph_context = policy_ctx["graph_context"]
        value = policy_ctx["value"]
        assert isinstance(node_features, torch.Tensor)
        assert isinstance(valid_mask, torch.Tensor)
        assert isinstance(graph_features, torch.Tensor)
        assert isinstance(h, torch.Tensor)
        assert isinstance(graph_context, torch.Tensor)
        assert isinstance(value, torch.Tensor)

        available = valid_mask.clone()
        selected: List[int] = []
        slot_logits: List[torch.Tensor] = []
        log_prob = node_features.new_tensor(0.0)
        entropy = node_features.new_tensor(0.0)
        static_slot_fast_path = bool(
            deterministic
            and selected_indices is None
            and not compute_sampling_stats
            and policy_ctx["normalized_redundancy"] is None
            and policy_ctx["role_aware_overlap_features"] is None
        )

        if static_slot_fast_path:
            for slot_idx in range(self.num_slots):
                if int(available.sum().item()) <= 0:
                    break
                logits = self._slot_logits(
                    slot_idx=slot_idx,
                    available=available,
                    selected=(),
                    policy_ctx=policy_ctx,
                )
                if slot_logits_limit is None or slot_idx < int(slot_logits_limit):
                    slot_logits.append(logits.detach().clone())
                chosen_idx = int(torch.argmax(logits).item())
                if not bool(torch.isfinite(logits[chosen_idx]).item()):
                    break
                selected.append(chosen_idx)
                available[chosen_idx] = False

            selected_tensor = torch.tensor(selected, dtype=torch.long, device=node_features.device)
            selected_slot_features = self.build_selected_slot_features(h, selected_tensor)
            state_action_feature_vector = torch.cat([graph_context, graph_features, selected_slot_features], dim=0)
            if self.set_value_head is not None:
                set_value = self.set_value_head(state_action_feature_vector).view(())
            else:
                set_value = value
            return {
                "selected_indices": selected_tensor,
                "log_prob": log_prob,
                "entropy": entropy,
                "value": value,
                "set_value": set_value,
                "state_action_feature_vector": state_action_feature_vector,
                "slot_logits": slot_logits,
            }

        for slot_idx in range(self.num_slots):
            if int(available.sum().item()) <= 0:
                break
            dist = None
            logits = self._slot_logits(
                slot_idx=slot_idx,
                available=available,
                selected=selected,
                policy_ctx=policy_ctx,
            )
            if slot_logits_limit is None or slot_idx < int(slot_logits_limit):
                slot_logits.append(logits.detach().clone())

            candidate_idx = torch.nonzero(available, as_tuple=True)[0]
            candidate_logits = logits[candidate_idx]
            if selected_indices is not None and slot_idx < len(selected_indices):
                candidate_probs = torch.softmax(candidate_logits, dim=0) if compute_sampling_stats else None
                dist = Categorical(probs=candidate_probs) if compute_sampling_stats else None
                chosen_idx = int(selected_indices[slot_idx])
                chosen_matches = (candidate_idx == int(chosen_idx)).nonzero(as_tuple=True)[0]
                if chosen_matches.numel() != 1:
                    raise ValueError(
                        f"Chosen index {chosen_idx} is invalid or duplicated for slot {slot_idx}."
                    )
                local_idx = chosen_matches[0].view(())
            elif deterministic or (not self.training and self.greedy_eval):
                local_idx = torch.argmax(candidate_logits)
            elif training_sampling_mode == "legacy_categorical":
                candidate_probs = torch.softmax(candidate_logits, dim=0)
                dist = Categorical(probs=candidate_probs)
                local_idx = dist.sample()
            elif training_sampling_mode == "gumbel_max":
                scaled_logits = candidate_logits
                candidate_probs = torch.softmax(scaled_logits, dim=0)
                dist = Categorical(probs=candidate_probs) if compute_sampling_stats else None
                if generator is None:
                    uniform = torch.rand(candidate_logits.size(0), device=candidate_logits.device)
                else:
                    uniform = torch.rand(
                        candidate_logits.size(0),
                        generator=generator,
                        device="cpu",
                    ).to(candidate_logits.device)
                uniform = uniform.clamp_(1e-6, 1.0 - 1e-6)
                gumbels = -torch.log(-torch.log(uniform))
                local_idx = torch.argmax(scaled_logits + gumbels)
            else:
                candidate_probs = torch.softmax(candidate_logits, dim=0)
                dist = Categorical(probs=candidate_probs)
                sampled = torch.multinomial(
                    candidate_probs.detach().cpu(),
                    num_samples=1,
                    replacement=False,
                    generator=generator,
                )
                local_idx = sampled.view(()).to(candidate_probs.device)
            chosen_idx = int(candidate_idx[local_idx].item())
            selected.append(chosen_idx)
            available[chosen_idx] = False
            if compute_sampling_stats and dist is not None:
                log_prob = log_prob + dist.log_prob(local_idx)
                entropy = entropy + dist.entropy()

        selected_tensor = torch.tensor(selected, dtype=torch.long, device=node_features.device)
        selected_slot_features = self.build_selected_slot_features(h, selected_tensor)
        state_action_feature_vector = torch.cat([graph_context, graph_features, selected_slot_features], dim=0)
        if self.set_value_head is not None:
            set_value = self.set_value_head(state_action_feature_vector).view(())
        else:
            set_value = value
        return {
            "selected_indices": selected_tensor,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "set_value": set_value,
            "state_action_feature_vector": state_action_feature_vector,
            "slot_logits": slot_logits,
        }

    def act(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: torch.Tensor,
        graph_features: torch.Tensor,
        deterministic: bool = False,
        redundancy_features: torch.Tensor | None = None,
        complementarity_features: torch.Tensor | None = None,
        role_potentials: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        training_sampling_mode: str = "explicit_cpu_multinomial",
        compute_sampling_stats: bool = True,
        slot_logits_limit: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self._run_policy(
            node_features=node_features,
            edge_index=edge_index,
            valid_mask=valid_mask,
            graph_features=graph_features,
            selected_indices=None,
            deterministic=deterministic,
            redundancy_features=redundancy_features,
            complementarity_features=complementarity_features,
            role_potentials=role_potentials,
            generator=generator,
            training_sampling_mode=training_sampling_mode,
            compute_sampling_stats=compute_sampling_stats,
            slot_logits_limit=slot_logits_limit,
        )

    def evaluate_action_sequence(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: torch.Tensor,
        graph_features: torch.Tensor,
        selected_indices: Sequence[int],
        redundancy_features: torch.Tensor | None = None,
        complementarity_features: torch.Tensor | None = None,
        role_potentials: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self._run_policy(
            node_features=node_features,
            edge_index=edge_index,
            valid_mask=valid_mask,
            graph_features=graph_features,
            selected_indices=selected_indices,
            deterministic=False,
            redundancy_features=redundancy_features,
            complementarity_features=complementarity_features,
            role_potentials=role_potentials,
            generator=None,
            training_sampling_mode="explicit_cpu_multinomial",
        )
