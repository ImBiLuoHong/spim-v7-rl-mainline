from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_scatter import scatter_max, scatter_mean, scatter_sum


def _cfg_get(cfg_obj: Any, key: str, default: Any) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def tensor_attr(obj: Any, key: str, reference: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if obj is None:
        return torch.full_like(reference.view(-1).float(), float(default))
    if isinstance(obj, dict):
        value = obj.get(key)
    else:
        value = getattr(obj, key, None)
    if value is None:
        return torch.full_like(reference.view(-1).float(), float(default))
    return value.view(-1).to(device=reference.device, dtype=torch.float32)


def default_reward_contract() -> Dict[str, float]:
    return {
        "mode": "navigator_vnext_evidence_v2",
        "tau_rank": 0.25,
        "lambda_rank": 1.0,
        "lambda_shrink": 0.35,
        "support_plausible_delta": 0.25,
        "alpha_suspect": 0.35,
        "w_max": 2.0,
        "not_ruled_out_threshold": 0.5,
        "K_hit": 3,
        "n_target": 5,
        "mu_hit": 1.0,
        "mu_target": 0.35,
        "mu_early": 0.20,
        "lambda_save": 0.05,
    }


def resolve_reward_contract(cfg_like: Any) -> Dict[str, float]:
    base = default_reward_contract()
    if cfg_like is None:
        return base
    reward_cfg = cfg_like
    if not isinstance(reward_cfg, dict):
        reward_cfg = {}
    out = dict(base)
    out.update(reward_cfg)
    return out


def _graph_count(batch: torch.Tensor) -> int:
    batch = batch.view(-1)
    return int(batch.max().item()) + 1 if batch.numel() > 0 else 0


def _masked_graph_max(values: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor, graph_count: int) -> torch.Tensor:
    masked = values.clone()
    masked[~mask] = -float("inf")
    top = scatter_max(masked, batch, dim=0, dim_size=graph_count)[0]
    return torch.where(torch.isfinite(top), top, torch.zeros_like(top))


def _score_metrics(scores: torch.Tensor, labels: torch.Tensor, plausible_delta: float = 0.25) -> Dict[str, Any]:
    scores = scores.detach().float().view(-1)
    labels = labels.detach().float().view(-1)
    finite_mask = torch.isfinite(scores)
    if not bool(finite_mask.any()):
        return {
            "valid_case": False,
            "true_rank": None,
            "true_value": 0.0,
            "top1_hit": False,
            "topk_hit": False,
            "top1_margin": 0.0,
            "plausible_count": 0.0,
        }

    true_mask = labels > 0.5
    if not bool(true_mask.any()):
        return {
            "valid_case": False,
            "true_rank": None,
            "true_value": 0.0,
            "top1_hit": False,
            "topk_hit": False,
            "top1_margin": 0.0,
            "plausible_count": 0.0,
        }

    safe_scores = scores.clone()
    safe_scores[~finite_mask] = -float("inf")
    sorted_idx = torch.argsort(safe_scores, descending=True)
    true_rank = int((true_mask[sorted_idx]).nonzero(as_tuple=True)[0].min().item() + 1)
    top_vals = torch.topk(safe_scores[finite_mask], k=min(2, int(finite_mask.sum().item()))).values
    top1_margin = float(top_vals[0].item() - top_vals[1].item()) if top_vals.numel() >= 2 else 0.0
    plausible_floor = float(top_vals[0].item() - float(plausible_delta))
    plausible_count = float(((safe_scores >= plausible_floor) & finite_mask).float().sum().item())
    return {
        "valid_case": True,
        "true_rank": true_rank,
        "true_value": float(safe_scores[true_mask].max().item()),
        "top1_hit": bool(true_rank <= 1),
        "topk_hit": False,
        "top1_margin": top1_margin,
        "plausible_count": plausible_count,
    }


def build_candidate_semantics(
    evidence_state: Any,
    constraint_state: Any,
    valid_mask: torch.Tensor,
    batch: torch.Tensor,
    contract_cfg: Dict[str, float],
) -> Dict[str, Any]:
    batch = batch.view(-1).long()
    graph_count = _graph_count(batch)
    valid_mask = valid_mask.view(-1).bool()
    reference = valid_mask.float()

    support_score = tensor_attr(evidence_state, "support_score", reference, default=0.0)
    uncertainty_gap = tensor_attr(evidence_state, "uncertainty_gap", reference, default=0.0).clamp(0.0, 1.0)
    suspect_pool = tensor_attr(evidence_state, "suspect_pool", reference, default=0.0).clamp(0.0, 1.0)
    not_ruled_out_gate = tensor_attr(evidence_state, "not_ruled_out_gate", reference, default=1.0).clamp(0.0, 1.0)
    contradiction_score = tensor_attr(evidence_state, "contradiction_score", reference, default=0.0)
    confirmed_non_source = tensor_attr(constraint_state, "confirmed_non_source_mask", reference, default=0.0) > 0.5

    q_score = support_score * not_ruled_out_gate
    candidate_mask = valid_mask & (~confirmed_non_source) & (
        not_ruled_out_gate >= float(contract_cfg["not_ruled_out_threshold"])
    )
    top_q = _masked_graph_max(q_score, candidate_mask, batch, graph_count)
    plausible_floor = top_q[batch] - float(contract_cfg["support_plausible_delta"])
    plausible_mask = candidate_mask & (q_score >= plausible_floor)
    core_mask = plausible_mask

    core_weight = torch.clamp(
        uncertainty_gap + float(contract_cfg["alpha_suspect"]) * suspect_pool,
        min=0.0,
        max=float(contract_cfg["w_max"]),
    ) * core_mask.float()

    candidate_count = scatter_sum(candidate_mask.float(), batch, dim=0, dim_size=graph_count)
    core_count = scatter_sum(core_mask.float(), batch, dim=0, dim_size=graph_count)
    core_mass = scatter_sum(core_weight, batch, dim=0, dim_size=graph_count)
    true_q_placeholder = scatter_sum(torch.zeros_like(q_score), batch, dim=0, dim_size=graph_count)

    return {
        "q_score": q_score,
        "uncertainty_gap": uncertainty_gap,
        "suspect_pool": suspect_pool,
        "not_ruled_out_gate": not_ruled_out_gate,
        "contradiction_score": contradiction_score,
        "candidate_mask": candidate_mask,
        "core_mask": core_mask,
        "core_weight": core_weight,
        "candidate_count": candidate_count,
        "core_count": core_count,
        "core_mass": core_mass,
        "true_q_placeholder": true_q_placeholder,
        "graph_count": graph_count,
    }


def build_deployable_nav_state_summary(
    observation_state: Any,
    evidence_state: Any,
    constraint_state: Any,
    valid_mask: torch.Tensor,
    batch: torch.Tensor,
    budget_remaining: torch.Tensor,
    step_fraction: float,
    contract_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch = batch.view(-1).long()
    graph_count = _graph_count(batch)
    valid_mask = valid_mask.view(-1).bool()
    reference = valid_mask.float()
    reward_cfg = resolve_reward_contract(contract_cfg)

    observed_flag = tensor_attr(observation_state, "observed_flag", reference, default=0.0).clamp(0.0, 1.0)
    toxic_positive_flag = tensor_attr(observation_state, "toxic_positive_flag", reference, default=0.0).clamp(0.0, 1.0)
    candidate_semantics = build_candidate_semantics(
        evidence_state=evidence_state,
        constraint_state=constraint_state,
        valid_mask=valid_mask,
        batch=batch,
        contract_cfg=reward_cfg,
    )
    candidate_ratio = scatter_mean(candidate_semantics["candidate_mask"].float(), batch, dim=0, dim_size=graph_count)
    core_ratio = scatter_mean(candidate_semantics["core_mask"].float(), batch, dim=0, dim_size=graph_count)
    uncertainty_core_mean = scatter_mean(
        candidate_semantics["uncertainty_gap"] * candidate_semantics["core_mask"].float(),
        batch,
        dim=0,
        dim_size=graph_count,
    )
    support_core_mean = scatter_mean(
        candidate_semantics["q_score"] * candidate_semantics["core_mask"].float(),
        batch,
        dim=0,
        dim_size=graph_count,
    )
    observed_ratio = scatter_mean(observed_flag, batch, dim=0, dim_size=graph_count)
    toxic_positive_ratio = scatter_mean(toxic_positive_flag, batch, dim=0, dim_size=graph_count)

    summary = torch.stack(
        [
            observed_ratio,
            toxic_positive_ratio,
            candidate_ratio,
            core_ratio,
            budget_remaining.view(-1).float(),
            torch.full_like(budget_remaining.view(-1).float(), float(step_fraction)),
        ],
        dim=1,
    )
    extras = {
        "candidate_ratio": candidate_ratio,
        "core_ratio": core_ratio,
        "uncertainty_core_mean": uncertainty_core_mean,
        "support_core_mean": support_core_mean,
        "observed_ratio": observed_ratio,
        "toxic_positive_ratio": toxic_positive_ratio,
    }
    return summary, extras


def compute_step_reward_bundle(
    *,
    fused_source_label: torch.Tensor,
    fused_batch: torch.Tensor,
    evidence_state_before: Any,
    evidence_state_after: Any,
    constraint_state_before: Any,
    constraint_state_after: Any,
    valid_mask_before: torch.Tensor,
    valid_mask_after: torch.Tensor,
    selection_mask: torch.Tensor,
    current_action_k: int,
    contract_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    contract = resolve_reward_contract(contract_cfg)
    fused_batch = fused_batch.view(-1).long()
    labels = fused_source_label.view(-1).float()
    graph_count = _graph_count(fused_batch)
    valid_before = valid_mask_before.view(-1).bool()
    valid_after = valid_mask_after.view(-1).bool()
    selection_mask = selection_mask.view(-1).bool()

    before = build_candidate_semantics(
        evidence_state=evidence_state_before,
        constraint_state=constraint_state_before,
        valid_mask=valid_before,
        batch=fused_batch,
        contract_cfg=contract,
    )
    after = build_candidate_semantics(
        evidence_state=evidence_state_after,
        constraint_state=constraint_state_after,
        valid_mask=valid_after,
        batch=fused_batch,
        contract_cfg=contract,
    )

    true_mask = labels > 0.5
    true_q_before = scatter_sum(before["q_score"] * labels, fused_batch, dim=0, dim_size=graph_count)
    true_q_after = scatter_sum(after["q_score"] * labels, fused_batch, dim=0, dim_size=graph_count)

    outrank_before = torch.sigmoid((before["q_score"] - true_q_before[fused_batch]) / float(contract["tau_rank"]))
    outrank_after = torch.sigmoid((after["q_score"] - true_q_after[fused_batch]) / float(contract["tau_rank"]))
    rank_mask_before = before["candidate_mask"].float() * (~true_mask).float()
    rank_mask_after = after["candidate_mask"].float() * (~true_mask).float()
    H_before = scatter_sum(rank_mask_before * outrank_before, fused_batch, dim=0, dim_size=graph_count)
    H_after = scatter_sum(rank_mask_after * outrank_after, fused_batch, dim=0, dim_size=graph_count)
    r_rank = float(contract["lambda_rank"]) * (torch.log1p(H_before) - torch.log1p(H_after))

    removed_mask = before["candidate_mask"] & (~after["candidate_mask"])
    r_shrink = float(contract["lambda_shrink"]) * scatter_sum(
        before["core_weight"] * removed_mask.float(),
        fused_batch,
        dim=0,
        dim_size=graph_count,
    )
    r_step_total = r_rank + r_shrink

    selected_count = scatter_sum(selection_mask.float(), fused_batch, dim=0, dim_size=graph_count)
    selected_valid_count = scatter_sum(
        (selection_mask & valid_before).float(),
        fused_batch,
        dim=0,
        dim_size=graph_count,
    )

    pre_rows: List[Dict[str, Any]] = []
    post_rows: List[Dict[str, Any]] = []
    topk_hits = []
    for graph_idx in range(graph_count):
        node_mask = fused_batch == graph_idx
        pre_metrics = _score_metrics(
            before["q_score"][node_mask],
            labels[node_mask],
            plausible_delta=float(contract["support_plausible_delta"]),
        )
        post_metrics = _score_metrics(
            after["q_score"][node_mask],
            labels[node_mask],
            plausible_delta=float(contract["support_plausible_delta"]),
        )
        if post_metrics["valid_case"]:
            post_metrics["topk_hit"] = bool(post_metrics["true_rank"] <= int(contract["K_hit"]))
        if pre_metrics["valid_case"]:
            pre_metrics["topk_hit"] = bool(pre_metrics["true_rank"] <= int(contract["K_hit"]))
        pre_rows.append(pre_metrics)
        post_rows.append(post_metrics)
        topk_hits.append(float(post_metrics["topk_hit"]) if post_metrics["valid_case"] else 0.0)

    harmful_drift = (
        (r_rank < 0.0)
        | (true_q_after < true_q_before)
        | (
            scatter_sum(after["contradiction_score"] * labels, fused_batch, dim=0, dim_size=graph_count)
            > scatter_sum(before["contradiction_score"] * labels, fused_batch, dim=0, dim_size=graph_count)
        )
    ).float()
    focus_core_delta = after["core_mass"] - before["core_mass"]
    wasted_budget_fraction = (
        torch.clamp(selected_count.new_full(selected_count.shape, float(current_action_k)) - selected_valid_count, min=0.0)
        / max(float(current_action_k), 1.0)
    )
    empty_selection_ratio = (selected_count <= 0.0).float()
    candidate_shrinkage = before["candidate_count"] - after["candidate_count"]
    evidence_true_q_delta = true_q_after - true_q_before
    source_hit_topk = torch.tensor(topk_hits, dtype=torch.float32, device=labels.device)
    budget_efficiency = candidate_shrinkage / torch.clamp(selected_valid_count, min=1.0)

    return {
        "mode": torch.tensor(0.0, device=labels.device),
        "r_rank": r_rank,
        "r_shrink": r_shrink,
        "r_step_total": r_step_total,
        "H_before": H_before,
        "H_after": H_after,
        "candidate_count_before": before["candidate_count"],
        "candidate_count_after": after["candidate_count"],
        "core_mass_before": before["core_mass"],
        "core_mass_after": after["core_mass"],
        "harmful_drift": harmful_drift,
        "focus_core_delta": focus_core_delta,
        "wasted_budget_fraction": wasted_budget_fraction,
        "empty_selection_ratio": empty_selection_ratio,
        "candidate_shrinkage": candidate_shrinkage,
        "evidence_true_q_delta": evidence_true_q_delta,
        "budget_efficiency": budget_efficiency,
        "source_hit_topk": source_hit_topk,
        "pre_metrics": pre_rows,
        "post_metrics": post_rows,
    }


def compute_terminal_reward_bundle(
    trajectory: List[Dict[str, Any]],
    rollout: Dict[str, Any],
    contract_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    if not trajectory:
        zero = torch.zeros(0, dtype=torch.float32)
        return {
            "terminal_topk_hit": zero,
            "terminal_candidate_target": zero,
            "terminal_early_decisive": zero,
            "terminal_budget_bonus": zero,
            "terminal_total": zero,
            "final_candidate_count": zero,
            "budget_used": zero,
            "success": zero,
        }

    contract = resolve_reward_contract(contract_cfg)
    last_step = trajectory[-1]
    batch = last_step["fused_batch"].view(-1).long()
    labels = last_step["fused_source_label"].view(-1).float()
    graph_count = _graph_count(batch)
    post_evidence = last_step.get("post_action_evidence_state")
    if post_evidence is None:
        post_evidence = last_step["reasoner_input_state"]["evidence_state"]
    constraint_state = last_step.get("constraint_state")
    valid_mask = last_step.get("post_action_valid_mask")
    if valid_mask is None:
        valid_mask = last_step.get("pre_action_valid_mask")
    valid_mask = valid_mask.view(-1).bool()

    semantics = build_candidate_semantics(
        evidence_state=post_evidence,
        constraint_state=constraint_state,
        valid_mask=valid_mask,
        batch=batch,
        contract_cfg=contract,
    )
    final_candidate_count = semantics["candidate_count"]
    terminal_topk_hit = torch.zeros(graph_count, dtype=torch.float32, device=labels.device)
    terminal_candidate_target = (final_candidate_count <= float(contract["n_target"])).float()
    terminal_early_decisive = torch.zeros_like(terminal_topk_hit)

    for graph_idx in range(graph_count):
        node_mask = batch == graph_idx
        final_metrics = _score_metrics(
            semantics["q_score"][node_mask],
            labels[node_mask],
            plausible_delta=float(contract["support_plausible_delta"]),
        )
        if final_metrics["valid_case"] and final_metrics["true_rank"] <= int(contract["K_hit"]):
            terminal_topk_hit[graph_idx] = 1.0

        for step in trajectory[:-1]:
            step_post = step.get("post_action_evidence_state")
            step_valid = step.get("post_action_valid_mask")
            if step_post is None or step_valid is None:
                continue
            step_batch = step["fused_batch"].view(-1).long()
            if graph_idx >= _graph_count(step_batch):
                continue
            step_semantics = build_candidate_semantics(
                evidence_state=step_post,
                constraint_state=step.get("constraint_state"),
                valid_mask=step_valid.view(-1).bool(),
                batch=step_batch,
                contract_cfg=contract,
            )
            step_mask = step_batch == graph_idx
            step_metrics = _score_metrics(
                step_semantics["q_score"][step_mask],
                step["fused_source_label"].view(-1).float()[step_mask],
                plausible_delta=float(contract["support_plausible_delta"]),
            )
            if (
                step_metrics["valid_case"]
                and step_metrics["true_rank"] <= int(contract["K_hit"])
                and float(step_semantics["candidate_count"][graph_idx].item()) <= float(contract["n_target"])
            ):
                terminal_early_decisive[graph_idx] = 1.0
                break

    final_dynamic = rollout.get("final_dynamic_state", {})
    sampled_mask = final_dynamic.get("sampled_mask")
    if sampled_mask is None:
        sampled_mask = last_step.get("constraint_state").sampled_mask
    budget_used = scatter_sum(sampled_mask.view(-1).float(), batch, dim=0, dim_size=graph_count)
    success = terminal_topk_hit.clone()
    terminal_budget_bonus = success * float(contract["lambda_save"]) * torch.clamp(
        budget_used.new_full(budget_used.shape, float(len(trajectory))) - budget_used,
        min=0.0,
    )
    terminal_total = (
        float(contract["mu_hit"]) * terminal_topk_hit
        + float(contract["mu_target"]) * terminal_candidate_target
        + float(contract["mu_early"]) * terminal_early_decisive
        + terminal_budget_bonus
    )

    return {
        "terminal_topk_hit": terminal_topk_hit,
        "terminal_candidate_target": terminal_candidate_target,
        "terminal_early_decisive": terminal_early_decisive,
        "terminal_budget_bonus": terminal_budget_bonus,
        "terminal_total": terminal_total,
        "final_candidate_count": final_candidate_count,
        "budget_used": budget_used,
        "success": success,
    }
