from typing import Any, Dict, List, Tuple

import torch
from torch_scatter import scatter_sum


STEP_REWARD_KEYS = [
    "r_rank",
    "r_shrink",
    "diag_harmful_drift",
    "diag_focus_core_delta",
    "diag_wasted_budget_fraction",
    "diag_empty_selection",
    "diag_source_hit",
    "diag_topk_hit",
    "diag_candidate_shrinkage",
    "diag_budget_used",
    "r_total",
]

TERMINAL_REWARD_KEYS = [
    "terminal_rank_hit",
    "terminal_core_hit",
    "terminal_early_decisive",
    "terminal_budget_bonus",
    "r_terminal_total",
]


def reward_contract() -> Dict[str, float]:
    return {
        "tau_rank": 0.25,
        "support_plausible_delta": 0.25,
        "core_gate_threshold": 0.05,
        "lambda_rank": 0.60,
        "lambda_shrink": 0.40,
        "alpha_suspect": 0.25,
        "w_max": 2.0,
        "topk_hit_k": 3,
        "n_target": 3,
        "early_decisive_round": 1,
        "mu_rank_hit": 1.0,
        "mu_core_hit": 0.5,
        "mu_early_decisive": 0.25,
        "lambda_save": 0.05,
        "budget_max": 9.0,
    }


def cfg_get(cfg_obj: Any, key: str, default: Any) -> Any:
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


def _score_metrics(scores: torch.Tensor, labels: torch.Tensor, plausible_delta: float) -> Dict[str, Any]:
    scores = scores.detach().float().view(-1)
    labels = labels.detach().float().view(-1)
    finite_mask = torch.isfinite(scores)
    if not bool(finite_mask.any()):
        return {
            "valid_case": False,
            "true_rank": None,
            "true_value": 0.0,
            "top1_margin": 0.0,
            "plausible_count": 0.0,
        }
    true_mask = labels > 0.5
    if not bool(true_mask.any()):
        return {
            "valid_case": False,
            "true_rank": None,
            "true_value": 0.0,
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
        "top1_margin": top1_margin,
        "plausible_count": plausible_count,
    }


def _post_valid_mask(step: Dict[str, Any], reference: torch.Tensor, default_valid: torch.Tensor) -> torch.Tensor:
    constraint_state = step.get("constraint_state")
    if constraint_state is None:
        return default_valid

    no_resample = tensor_attr(constraint_state, "no_resample_mask", reference, default=0.0) > 0.5
    confirmed_non_source = tensor_attr(constraint_state, "confirmed_non_source_mask", reference, default=0.0) > 0.5
    return default_valid & (~no_resample) & (~confirmed_non_source)


def _core_state(
    score: torch.Tensor,
    uncertainty: torch.Tensor,
    suspect: torch.Tensor,
    not_ruled_out: torch.Tensor,
    valid_mask: torch.Tensor,
    plausible_delta: float,
    alpha_suspect: float,
    w_max: float,
    gate_threshold: float,
) -> Dict[str, Any]:
    score = score.detach().float().view(-1)
    uncertainty = uncertainty.detach().float().view(-1)
    suspect = suspect.detach().float().view(-1)
    not_ruled_out = not_ruled_out.detach().float().view(-1)
    valid_mask = valid_mask.view(-1).bool()
    finite_mask = torch.isfinite(score)
    admissible = valid_mask & finite_mask & (not_ruled_out > float(gate_threshold))
    if not bool(admissible.any()):
        zeros = torch.zeros_like(score)
        return {
            "q": zeros,
            "plausible_weight": zeros,
            "core_mask": torch.zeros_like(valid_mask),
            "weight": zeros,
            "core_weight_sum": 0.0,
            "core_count": 0.0,
        }

    safe_scores = score.clone()
    safe_scores[~admissible] = -float("inf")
    top1 = float(safe_scores[admissible].max().item())
    plausible_floor = top1 - float(plausible_delta)
    bounded_scores = torch.where(torch.isfinite(score), score, torch.full_like(score, plausible_floor))
    plausible_weight = ((bounded_scores - plausible_floor) / max(float(plausible_delta), 1e-6)).clamp(0.0, 1.0)
    q = bounded_scores * not_ruled_out * valid_mask.float()
    core_mask = admissible & (plausible_weight > 0.0)
    weight = core_mask.float() * torch.clamp(uncertainty + float(alpha_suspect) * suspect, min=0.0, max=float(w_max))
    return {
        "q": q,
        "plausible_weight": plausible_weight,
        "core_mask": core_mask,
        "weight": weight,
        "core_weight_sum": float(weight.sum().item()),
        "core_count": float(core_mask.float().sum().item()),
    }


def _soft_outranking_mass(
    q: torch.Tensor,
    labels: torch.Tensor,
    core_mask: torch.Tensor,
    tau_rank: float,
) -> Tuple[float, int]:
    q = q.detach().float().view(-1)
    labels = labels.detach().float().view(-1)
    core_mask = core_mask.view(-1).bool()
    true_mask = labels > 0.5
    if not bool(true_mask.any()):
        return 0.0, 0

    q_true = float(q[true_mask].max().item())
    candidate_mask = core_mask & (~true_mask)
    if not bool(candidate_mask.any()):
        return 0.0, 1

    margins = (q[candidate_mask] - q_true) / max(float(tau_rank), 1e-6)
    outranking = torch.sigmoid(margins).sum()
    true_rank = 1 + int((q[candidate_mask] > q_true).sum().item())
    return float(outranking.item()), true_rank


def compute_step_reward(step: Dict[str, Any], reward_cfg: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    labels = step["fused_source_label"].view(-1).float()
    batch = step["fused_batch"].view(-1).long()
    graph_count = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    pre_evidence = cfg_get(step.get("reasoner_input_state"), "evidence_state", None)
    post_evidence = step.get("post_action_evidence_state")
    pre_valid = step["pre_action_valid_mask"].view(-1).bool()
    post_valid = _post_valid_mask(step, labels, pre_valid)
    selected_indices = step["selected_indices"].view(-1).long()
    current_action_k = max(int(step.get("current_action_k", 1)), 1)

    plausible_delta = float(reward_cfg["support_plausible_delta"])
    tau_rank = float(reward_cfg["tau_rank"])
    alpha_suspect = float(reward_cfg["alpha_suspect"])
    w_max = float(reward_cfg["w_max"])
    gate_threshold = float(reward_cfg["core_gate_threshold"])

    pre_support = tensor_attr(pre_evidence, "support_score", labels)
    post_support = tensor_attr(post_evidence, "support_score", labels)
    pre_uncertainty = tensor_attr(pre_evidence, "uncertainty_gap", labels)
    post_uncertainty = tensor_attr(post_evidence, "uncertainty_gap", labels)
    pre_suspect = tensor_attr(pre_evidence, "suspect_pool", labels)
    post_suspect = tensor_attr(post_evidence, "suspect_pool", labels)
    pre_not_ruled = tensor_attr(pre_evidence, "not_ruled_out_gate", labels, default=1.0)
    post_not_ruled = tensor_attr(post_evidence, "not_ruled_out_gate", labels, default=1.0)

    reward_components: Dict[str, List[float]] = {key: [] for key in STEP_REWARD_KEYS}

    verdict_hit = cfg_get(step.get("constraint_update"), "is_source_hit", None)
    if verdict_hit is None:
        verdict_hit = torch.zeros(graph_count, device=labels.device, dtype=torch.bool)
    else:
        verdict_hit = verdict_hit.view(-1).to(device=labels.device, dtype=torch.bool)

    if selected_indices.numel() > 0:
        selected_batch = batch[selected_indices]
        selected_count = scatter_sum(
            torch.ones_like(selected_batch, dtype=torch.float32),
            selected_batch,
            dim=0,
            dim_size=graph_count,
        )
        selected_valid_count = scatter_sum(
            pre_valid[selected_indices].float(),
            selected_batch,
            dim=0,
            dim_size=graph_count,
        )
    else:
        selected_count = torch.zeros(graph_count, device=labels.device)
        selected_valid_count = torch.zeros(graph_count, device=labels.device)

    for graph_idx in range(graph_count):
        node_mask = batch == graph_idx
        label_slice = labels[node_mask]
        if not bool((label_slice > 0.5).any()):
            for key in STEP_REWARD_KEYS:
                reward_components[key].append(0.0)
            continue

        pre_core = _core_state(
            pre_support[node_mask],
            pre_uncertainty[node_mask],
            pre_suspect[node_mask],
            pre_not_ruled[node_mask],
            pre_valid[node_mask],
            plausible_delta,
            alpha_suspect,
            w_max,
            gate_threshold,
        )
        post_core = _core_state(
            post_support[node_mask],
            post_uncertainty[node_mask],
            post_suspect[node_mask],
            post_not_ruled[node_mask],
            post_valid[node_mask],
            plausible_delta,
            alpha_suspect,
            w_max,
            gate_threshold,
        )

        pre_metrics = _score_metrics(pre_core["q"], label_slice, plausible_delta)
        post_metrics = _score_metrics(post_core["q"], label_slice, plausible_delta)
        if not bool(pre_metrics["valid_case"]) or not bool(post_metrics["valid_case"]):
            for key in STEP_REWARD_KEYS:
                reward_components[key].append(0.0)
            continue

        H_pre, _ = _soft_outranking_mass(pre_core["q"], label_slice, pre_core["core_mask"], tau_rank)
        H_post, post_rank = _soft_outranking_mass(post_core["q"], label_slice, post_core["core_mask"], tau_rank)
        r_rank = float(reward_cfg["lambda_rank"]) * (torch.log1p(torch.tensor(H_pre)) - torch.log1p(torch.tensor(H_post))).item()

        removed = pre_core["core_mask"] & (~post_core["core_mask"]) & (label_slice <= 0.5)
        r_shrink = float(reward_cfg["lambda_shrink"]) * float(pre_core["weight"][removed].sum().item())

        harmful = float(H_post > H_pre + 1e-6 or post_metrics["true_value"] < pre_metrics["true_value"] - 1e-6)
        focus_core_delta = float(post_core["core_weight_sum"] - pre_core["core_weight_sum"])
        wasted_budget_fraction = float(
            max(float(current_action_k) - float(selected_valid_count[graph_idx].item()), 0.0)
            / max(float(current_action_k), 1.0)
        )
        empty_selection = float(selected_count[graph_idx].item() <= 0.0)
        source_hit = float(verdict_hit[graph_idx].item())
        topk_hit = float(post_rank <= int(reward_cfg["topk_hit_k"]))
        candidate_shrinkage = float(pre_core["core_count"] - post_core["core_count"])
        budget_used = float(selected_valid_count[graph_idx].item())
        reward_total = r_rank + r_shrink

        reward_components["r_rank"].append(r_rank)
        reward_components["r_shrink"].append(r_shrink)
        reward_components["diag_harmful_drift"].append(harmful)
        reward_components["diag_focus_core_delta"].append(focus_core_delta)
        reward_components["diag_wasted_budget_fraction"].append(wasted_budget_fraction)
        reward_components["diag_empty_selection"].append(empty_selection)
        reward_components["diag_source_hit"].append(source_hit)
        reward_components["diag_topk_hit"].append(topk_hit)
        reward_components["diag_candidate_shrinkage"].append(candidate_shrinkage)
        reward_components["diag_budget_used"].append(budget_used)
        reward_components["r_total"].append(reward_total)

    reward_bundle = {
        key: torch.tensor(values, dtype=torch.float32, device=labels.device)
        for key, values in reward_components.items()
    }
    return reward_bundle["r_total"], reward_bundle


def compute_terminal_reward(rollout: Dict[str, Any], reward_cfg: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    trajectory = rollout.get("trajectory", [])
    if not trajectory:
        zero = torch.zeros(1, dtype=torch.float32)
        return zero, {key: zero.clone() for key in TERMINAL_REWARD_KEYS}

    last_step = trajectory[-1]
    labels = last_step["fused_source_label"].view(-1).float()
    batch = last_step["fused_batch"].view(-1).long()
    graph_count = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    final_evidence = last_step.get("post_action_evidence_state")
    final_valid = _post_valid_mask(last_step, labels, last_step["pre_action_valid_mask"].view(-1).bool())

    plausible_delta = float(reward_cfg["support_plausible_delta"])
    alpha_suspect = float(reward_cfg["alpha_suspect"])
    w_max = float(reward_cfg["w_max"])
    gate_threshold = float(reward_cfg["core_gate_threshold"])
    topk_hit_k = int(reward_cfg["topk_hit_k"])
    n_target = int(reward_cfg["n_target"])
    early_round_threshold = int(reward_cfg["early_decisive_round"])
    budget_max = float(reward_cfg["budget_max"])

    final_support = tensor_attr(final_evidence, "support_score", labels)
    final_uncertainty = tensor_attr(final_evidence, "uncertainty_gap", labels)
    final_suspect = tensor_attr(final_evidence, "suspect_pool", labels)
    final_not_ruled = tensor_attr(final_evidence, "not_ruled_out_gate", labels, default=1.0)

    raw_budget = cfg_get(rollout.get("step_metrics"), "raw_budget", None)
    if raw_budget is None:
        raw_budget = torch.zeros(graph_count, device=labels.device)
    else:
        raw_budget = raw_budget.view(-1).to(device=labels.device, dtype=torch.float32)

    terminal_components: Dict[str, List[float]] = {key: [] for key in TERMINAL_REWARD_KEYS}

    for graph_idx in range(graph_count):
        node_mask = batch == graph_idx
        label_slice = labels[node_mask]
        if not bool((label_slice > 0.5).any()):
            for key in TERMINAL_REWARD_KEYS:
                terminal_components[key].append(0.0)
            continue

        final_core = _core_state(
            final_support[node_mask],
            final_uncertainty[node_mask],
            final_suspect[node_mask],
            final_not_ruled[node_mask],
            final_valid[node_mask],
            plausible_delta,
            alpha_suspect,
            w_max,
            gate_threshold,
        )
        _, final_rank = _soft_outranking_mass(
            final_core["q"],
            label_slice,
            final_core["core_mask"],
            float(reward_cfg["tau_rank"]),
        )

        decisive_round = None
        for step_idx, step in enumerate(trajectory):
            step_labels = step["fused_source_label"].view(-1).float()[node_mask]
            step_evidence = step.get("post_action_evidence_state")
            step_valid = _post_valid_mask(step, labels, step["pre_action_valid_mask"].view(-1).bool())[node_mask]
            step_core = _core_state(
                tensor_attr(step_evidence, "support_score", labels)[node_mask],
                tensor_attr(step_evidence, "uncertainty_gap", labels)[node_mask],
                tensor_attr(step_evidence, "suspect_pool", labels)[node_mask],
                tensor_attr(step_evidence, "not_ruled_out_gate", labels, default=1.0)[node_mask],
                step_valid,
                plausible_delta,
                alpha_suspect,
                w_max,
                gate_threshold,
            )
            _, step_rank = _soft_outranking_mass(
                step_core["q"],
                step_labels,
                step_core["core_mask"],
                float(reward_cfg["tau_rank"]),
            )
            if step_rank <= topk_hit_k and int(step_core["core_count"]) <= n_target:
                decisive_round = step_idx
                break

        rank_hit = float(final_rank <= topk_hit_k)
        core_hit = float(int(final_core["core_count"]) <= n_target)
        early_decisive = float(decisive_round is not None and decisive_round <= early_round_threshold)
        success = float(rank_hit > 0.5)
        budget_bonus = success * float(reward_cfg["lambda_save"]) * max(budget_max - float(raw_budget[graph_idx].item()), 0.0)
        terminal_total = (
            float(reward_cfg["mu_rank_hit"]) * rank_hit
            + float(reward_cfg["mu_core_hit"]) * core_hit
            + float(reward_cfg["mu_early_decisive"]) * early_decisive
            + budget_bonus
        )

        terminal_components["terminal_rank_hit"].append(rank_hit)
        terminal_components["terminal_core_hit"].append(core_hit)
        terminal_components["terminal_early_decisive"].append(early_decisive)
        terminal_components["terminal_budget_bonus"].append(budget_bonus)
        terminal_components["r_terminal_total"].append(terminal_total)

    terminal_bundle = {
        key: torch.tensor(values, dtype=torch.float32, device=labels.device)
        for key, values in terminal_components.items()
    }
    return terminal_bundle["r_terminal_total"], terminal_bundle
