from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.modeling.navigators.clean_v1 import CleanNavigatorV1, pick_topk_valid
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import (
    compute_degree_norm,
    compute_returns,
    resolve_source_local_idx,
    safe_mean,
)
from src.scripts.diagnostics.run_slot1_counterfactual_leverage_audit import (
    build_namespace_from_control_args,
    load_control_bundle,
)
from src.scripts.run_reasoner_oracle_contrast_injection import build_cfg_with_contrast, load_model_with_contrast
from src.scripts.run_reasoner_oracle_exposure_audit import load_action_plan
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    TempGraph,
    build_state_input,
    load_same_cases,
    make_rollout_state,
    move_payload,
    read_json,
    translate_global_ids,
)


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact136_h3_formal_v1"
DEFAULT_CONTRAST_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_oracle_contrast_injection" / "20260407_exact136_oracle_contrast_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "frozen_belief_navigator_feasibility" / "20260407_exact136_belief_nav_v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
RUNNER_VERSION = "frozen_belief_navigator_feasibility_v1"
PANEL_VERSION = "train_only_exact136_frozen_belief_nav_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-belief navigator feasibility on exact136 train cases.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--belief-temperature", type=float, default=1.0)
    parser.add_argument("--confusion-logit-delta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--periodic-eval-every", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer-step-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--myopic-topm", type=int, default=16)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hash_tensor_payload(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenBeliefInterface:
    def __init__(
        self,
        *,
        reasoner_module: torch.nn.Module,
        device: torch.device,
        temperature: float,
        confusion_logit_delta: float,
    ) -> None:
        self.reasoner_module = reasoner_module
        self.device = device
        self.temperature = float(max(temperature, 1e-6))
        self.confusion_logit_delta = float(max(confusion_logit_delta, 1e-6))

    def belief_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        valid_mask = state["valid_mask"].view(-1).bool()
        graph = TempGraph(state["edge_index"], int(valid_mask.numel()), self.device)
        state_input = move_payload(build_state_input(state), self.device)
        physics_ctx = move_payload(state["phys_ctx"].__dict__, self.device)
        with torch.no_grad():
            out = self.reasoner_module(state_input, graph, physics_ctx=physics_ctx)
        logits = out["logits"].detach().float().view(-1).cpu()
        masked_logits = logits.clone()
        masked_logits[~valid_mask.cpu()] = -float("inf")
        valid_logits = masked_logits[valid_mask.cpu()]
        belief = torch.zeros_like(masked_logits)
        if valid_logits.numel() > 0:
            probs_valid = torch.softmax(valid_logits / self.temperature, dim=0)
            belief[valid_mask.cpu()] = probs_valid
            top_vals, top_idx = torch.topk(valid_logits, k=min(3, int(valid_logits.numel())))
            top1_logit = float(top_vals[0].item())
            valid_indices = torch.nonzero(valid_mask.cpu(), as_tuple=True)[0]
            top1_local = int(valid_indices[int(top_idx[0].item())].item())
            cluster_mask = valid_mask.cpu() & (masked_logits >= top1_logit - self.confusion_logit_delta)
            cluster_mass = float(belief[cluster_mask].sum().item())
            entropy = float(-(probs_valid * torch.log(probs_valid.clamp_min(1e-9))).sum().item())
            top1_mass = float(probs_valid.max().item())
            top3_mass = float(top_vals.new_tensor(probs_valid[: min(3, probs_valid.numel())]).sum().item()) if probs_valid.numel() > 0 else 0.0
            sorted_indices = torch.argsort(masked_logits, descending=True)
            valid_sorted = sorted_indices[torch.isfinite(masked_logits[sorted_indices])]
        else:
            top1_local = None
            cluster_mask = torch.zeros_like(valid_mask.cpu())
            cluster_mass = 0.0
            entropy = 0.0
            top1_mass = 0.0
            top3_mass = 0.0
            valid_sorted = torch.empty(0, dtype=torch.long)

        source_local = resolve_source_local_idx(state["rollout"])
        hardest_confuser = None
        true_rank = None
        if source_local is not None and valid_sorted.numel() > 0:
            positions = (valid_sorted == int(source_local)).nonzero(as_tuple=True)[0]
            if positions.numel() > 0:
                true_rank = int(positions.min().item()) + 1
            for idx in valid_sorted.tolist():
                if int(idx) != int(source_local):
                    hardest_confuser = int(idx)
                    break

        true_mass = float(belief[int(source_local)].item()) if source_local is not None and bool(valid_mask[int(source_local)].item()) else 0.0
        hard_mass = float(belief[int(hardest_confuser)].item()) if hardest_confuser is not None else 0.0
        margin = math.log(max(true_mass, 1e-9)) - math.log(max(hard_mass, 1e-9)) if hardest_confuser is not None else math.log(max(true_mass, 1e-9))

        return {
            "logits": logits,
            "belief": belief,
            "valid_mask": valid_mask.cpu(),
            "entropy": float(entropy),
            "top1_mass": float(top1_mass),
            "top3_mass": float(top3_mass),
            "cluster_mask": cluster_mask,
            "cluster_mass": float(cluster_mass),
            "cluster_count": int(cluster_mask.float().sum().item()),
            "source_local": None if source_local is None else int(source_local),
            "hardest_confuser_local": hardest_confuser,
            "true_rank": true_rank,
            "true_mass": float(true_mass),
            "hard_mass": float(hard_mass),
            "margin_true_vs_hard": float(margin),
        }


def build_belief_nav_state(state: Dict[str, Any], belief_ctx: Dict[str, Any]) -> Dict[str, Any]:
    valid_mask = state["valid_mask"].view(-1).bool().cpu()
    belief = belief_ctx["belief"].view(-1).float()
    max_belief = float(belief[valid_mask].max().item()) if bool(valid_mask.any()) else 0.0
    sampled_mask = state["constraint_state"].sampled_mask.view(-1).float().cpu()
    no_resample_mask = state["constraint_state"].no_resample_mask.view(-1).float().cpu()
    degree_norm = compute_degree_norm(state["edge_index"].cpu(), int(valid_mask.numel()), torch.device("cpu")).view(-1).float()
    hard_idx = belief_ctx["hardest_confuser_local"]
    hard_flag = torch.zeros_like(belief)
    if hard_idx is not None:
        hard_flag[int(hard_idx)] = 1.0
    cluster_flag = belief_ctx["cluster_mask"].view(-1).float()
    gap_to_top1 = max_belief - belief
    node_features = torch.stack(
        [
            belief,
            gap_to_top1,
            cluster_flag,
            hard_flag,
            sampled_mask,
            no_resample_mask,
            valid_mask.float(),
            degree_norm,
        ],
        dim=1,
    ).float()
    graph_features = torch.tensor(
        [
            float(belief_ctx["entropy"]),
            float(belief_ctx["top1_mass"]),
            float(belief_ctx["top3_mass"]),
            float(belief_ctx["cluster_mass"]),
            float(state["info"]["episode"]) / max(float(state["pair_budget"]), 1.0),
            float(sampled_mask.mean().item()),
        ],
        dtype=torch.float32,
    )
    zeros_overlap = torch.zeros((node_features.size(0), 1), dtype=torch.float32)
    role_potentials = torch.stack([belief, cluster_flag, hard_flag], dim=1).float()
    return {
        "node_features": node_features,
        "graph_features": graph_features,
        "edge_index": state["edge_index"].cpu(),
        "valid_mask": valid_mask,
        "redundancy_signature": zeros_overlap,
        "witness_pair_signature": zeros_overlap,
        "role_potentials": role_potentials,
        "belief_ctx": belief_ctx,
        "raw_state": state,
    }


def reward_from_belief_transition(
    pre_belief: Dict[str, Any],
    post_belief: Dict[str, Any],
    *,
    valid_selection: float,
    exact_k: float,
    repeated_penalty: float,
) -> Dict[str, float]:
    eps = 1e-9
    delta_true_log = math.log(max(post_belief["true_mass"], eps)) - math.log(max(pre_belief["true_mass"], eps))
    delta_margin = float(post_belief["margin_true_vs_hard"] - pre_belief["margin_true_vs_hard"])
    delta_entropy = float(pre_belief["entropy"] - post_belief["entropy"])
    terminal_hit_bonus = 1.0 if int(post_belief["true_rank"] or 10**9) <= 1 else 0.0
    invalid_penalty = -0.2 * (1.0 - float(valid_selection))
    repeated_sampling_penalty = -0.05 * float(repeated_penalty)
    shortfall_penalty = -0.05 * (1.0 - float(exact_k))
    reward_total = (
        1.0 * float(delta_true_log)
        + 0.75 * float(delta_margin)
        + 0.25 * float(delta_entropy)
        + float(terminal_hit_bonus)
        + float(invalid_penalty)
        + float(repeated_sampling_penalty)
        + float(shortfall_penalty)
    )
    return {
        "delta_true_log_mass": float(delta_true_log),
        "delta_margin_true_vs_hard": float(delta_margin),
        "delta_entropy": float(delta_entropy),
        "terminal_hit_bonus": float(terminal_hit_bonus),
        "invalid_penalty": float(invalid_penalty),
        "repeated_sampling_penalty": float(repeated_sampling_penalty),
        "shortfall_penalty": float(shortfall_penalty),
        "reward_total": float(reward_total),
    }


def select_action(
    *,
    policy_name: str,
    nav_state: Dict[str, Any],
    model: Optional[CleanNavigatorV1],
    budget: int,
    generator: torch.Generator,
    myopic_topm: int,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    case: Any,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    num_episodes: int,
    action_budget: int,
    frontier_role_mode: str,
    belief_interface: FrozenBeliefInterface,
) -> Tuple[List[int], Optional[Dict[str, torch.Tensor]]]:
    valid_mask = nav_state["valid_mask"]
    if policy_name == "top_support_legacy":
        raw_state = nav_state["raw_state"]
        return pick_topk_valid(raw_state["support_score"], valid_mask, budget), None
    if policy_name == "myopic_belief_entropy":
        belief = nav_state["belief_ctx"]["belief"].view(-1)
        valid_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
        ordered = sorted(valid_idx.tolist(), key=lambda idx: (-float(belief[int(idx)].item()), int(idx)))
        candidate_pool = [int(v) for v in ordered[: max(int(myopic_topm), int(budget))]]
        scores: List[Tuple[float, int]] = []
        for candidate in candidate_pool:
            sim_rollout = deepcopy(rollout)
            sim_history = deepcopy(history)
            sim_rollout.step_with_actions([int(candidate)], sample_types=["myopic_slot"])
            if sim_rollout.history_steps:
                sim_history.append_from_history_step(sim_rollout.history_steps[-1])
            post_state = make_rollout_state(
                case=case,
                rollout=sim_rollout,
                history=sim_history,
                env=env,
                topology=topology,
                num_episodes=num_episodes,
                action_budget=action_budget,
                frontier_role_mode=frontier_role_mode,
            )
            post_belief = belief_interface.belief_from_state(post_state)
            score = float(nav_state["belief_ctx"]["entropy"] - post_belief["entropy"])
            scores.append((score, int(candidate)))
        scores.sort(key=lambda item: (-float(item[0]), int(item[1])))
        return [int(idx) for _, idx in scores[: int(budget)]], None
    if model is None:
        raise ValueError(policy_name)
    model_device = next(model.parameters()).device
    model_out = model.act(
        node_features=nav_state["node_features"].to(model_device),
        edge_index=nav_state["edge_index"].to(model_device),
        valid_mask=nav_state["valid_mask"].to(model_device),
        graph_features=nav_state["graph_features"].to(model_device),
        deterministic=False,
        redundancy_features=nav_state["redundancy_signature"].to(model_device),
        complementarity_features=None,
        role_potentials=nav_state["role_potentials"].to(model_device),
        generator=generator,
        training_sampling_mode="explicit_cpu_multinomial",
    )
    return model_out["selected_indices"].detach().cpu().tolist(), model_out


def run_case_rollout(
    *,
    case: Any,
    policy_name: str,
    model: Optional[CleanNavigatorV1],
    belief_interface: FrozenBeliefInterface,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
    generator: torch.Generator,
    myopic_topm: int,
    deterministic: bool,
    device: torch.device,
) -> Dict[str, Any]:
    rollout = PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=dataset_assets["global_edge_index"],
        stt_dynamic_series=dataset_assets["stt_dynamic_series"],
        num_global_nodes=int(dataset_assets["num_global_nodes"]),
        num_episodes=int(num_episodes),
        samples_per_episode=int(action_budget),
        episode_duration_min=float(episode_duration_min),
    )
    history = ObservationWitnessHistory()
    step_rows: List[Dict[str, Any]] = []
    train_tensors: List[Dict[str, torch.Tensor]] = []
    for episode_idx in range(int(num_episodes)):
        pre_state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=num_episodes,
            action_budget=action_budget,
            frontier_role_mode=frontier_role_mode,
        )
        pre_belief = belief_interface.belief_from_state(pre_state)
        if int(pre_state["valid_mask"].sum().item()) < int(action_budget):
            break
        nav_state = build_belief_nav_state(pre_state, pre_belief)
        if policy_name == "belief_rl" and deterministic and model is not None:
            model_device = next(model.parameters()).device
            model_out = model.act(
                node_features=nav_state["node_features"].to(model_device),
                edge_index=nav_state["edge_index"].to(model_device),
                valid_mask=nav_state["valid_mask"].to(model_device),
                graph_features=nav_state["graph_features"].to(model_device),
                deterministic=True,
                redundancy_features=nav_state["redundancy_signature"].to(model_device),
                complementarity_features=None,
                role_potentials=nav_state["role_potentials"].to(model_device),
                generator=None,
                training_sampling_mode="explicit_cpu_multinomial",
            )
            selected_indices = model_out["selected_indices"].detach().cpu().tolist()
        else:
            selected_indices, model_out = select_action(
                policy_name=policy_name,
                nav_state=nav_state,
                model=model,
                budget=int(action_budget),
                generator=generator,
                myopic_topm=int(myopic_topm),
                rollout=rollout,
                history=history,
                case=case,
                env=env,
                topology=topology,
                num_episodes=num_episodes,
                action_budget=action_budget,
                frontier_role_mode=frontier_role_mode,
                belief_interface=belief_interface,
            )
        selected_indices = [int(v) for v in selected_indices]
        valid_selection = float(all(bool(pre_state["valid_mask"][idx].item()) for idx in selected_indices))
        exact_k = float(len(selected_indices) == int(action_budget))
        repeated_penalty = float(len(set(selected_indices)) != len(selected_indices))
        rollout.step_with_actions(selected_indices, sample_types=[f"{policy_name}_slot_{i}" for i in range(len(selected_indices))])
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])
        post_state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=num_episodes,
            action_budget=action_budget,
            frontier_role_mode=frontier_role_mode,
        )
        post_belief = belief_interface.belief_from_state(post_state)
        reward = reward_from_belief_transition(
            pre_belief,
            post_belief,
            valid_selection=valid_selection,
            exact_k=exact_k,
            repeated_penalty=repeated_penalty,
        )
        step_rows.append(
            {
                "episode": int(episode_idx + 1),
                "reward_total": float(reward["reward_total"]),
                "delta_true_log_mass": float(reward["delta_true_log_mass"]),
                "delta_margin_true_vs_hard": float(reward["delta_margin_true_vs_hard"]),
                "delta_entropy": float(reward["delta_entropy"]),
                "terminal_hit_bonus": float(reward["terminal_hit_bonus"]),
                "invalid_penalty": float(reward["invalid_penalty"]),
                "repeated_sampling_penalty": float(reward["repeated_sampling_penalty"]),
                "shortfall_penalty": float(reward["shortfall_penalty"]),
                "pre_entropy": float(pre_belief["entropy"]),
                "post_entropy": float(post_belief["entropy"]),
                "pre_true_mass": float(pre_belief["true_mass"]),
                "post_true_mass": float(post_belief["true_mass"]),
                "pre_margin_true_vs_hard": float(pre_belief["margin_true_vs_hard"]),
                "post_margin_true_vs_hard": float(post_belief["margin_true_vs_hard"]),
                "pre_cluster_mass": float(pre_belief["cluster_mass"]),
                "post_cluster_mass": float(post_belief["cluster_mass"]),
                "pre_cluster_count": int(pre_belief["cluster_count"]),
                "post_cluster_count": int(post_belief["cluster_count"]),
                "pre_true_rank": None if pre_belief["true_rank"] is None else int(pre_belief["true_rank"]),
                "post_true_rank": None if post_belief["true_rank"] is None else int(post_belief["true_rank"]),
                "valid_selection": float(valid_selection),
                "exact_k": float(exact_k),
                "selected_count": int(len(selected_indices)),
                "selected_indices": list(selected_indices),
            }
        )
        if model_out is not None and policy_name == "belief_rl" and not deterministic:
            train_tensors.append(
                {
                    "log_prob": model_out["log_prob"],
                    "entropy": model_out["entropy"],
                    "value": model_out["value"],
                    "reward": torch.tensor(float(reward["reward_total"]), device=device, dtype=torch.float32),
                }
            )

    final_state = make_rollout_state(
        case=case,
        rollout=rollout,
        history=history,
        env=env,
        topology=topology,
        num_episodes=num_episodes,
        action_budget=action_budget,
        frontier_role_mode=frontier_role_mode,
    )
    final_belief = belief_interface.belief_from_state(final_state)
    denom = max(len(step_rows), 1)
    return {
        "summary": {
            "case_id": case.case_id,
            "reward_total": float(sum(row["reward_total"] for row in step_rows)),
            "reward_mean": float(sum(row["reward_total"] for row in step_rows) / denom),
            "true_log_mass_delta_total": float(sum(row["delta_true_log_mass"] for row in step_rows)),
            "margin_delta_total": float(sum(row["delta_margin_true_vs_hard"] for row in step_rows)),
            "entropy_delta_total": float(sum(row["delta_entropy"] for row in step_rows)),
            "terminal_hit_bonus_total": float(sum(row["terminal_hit_bonus"] for row in step_rows)),
            "action_validity_rate": float(sum(row["valid_selection"] for row in step_rows) / denom),
            "exact_k_rate": float(sum(row["exact_k"] for row in step_rows) / denom),
            "policy_step_count": int(len(step_rows)),
            "budget_used": float(sum(row["selected_count"] for row in step_rows)),
            "budget_used_mean": float(sum(row["selected_count"] for row in step_rows) / denom),
            "final_true_rank": None if final_belief["true_rank"] is None else int(final_belief["true_rank"]),
            "final_top1_hit": float(int(final_belief["true_rank"] or 10**9) <= 1),
            "final_top3_hit": float(int(final_belief["true_rank"] or 10**9) <= 3),
            "final_top5_hit": float(int(final_belief["true_rank"] or 10**9) <= 5),
            "final_mrr": 0.0 if final_belief["true_rank"] is None else float(1.0 / float(final_belief["true_rank"])),
            "final_true_mass": float(final_belief["true_mass"]),
            "final_margin_true_vs_hard": float(final_belief["margin_true_vs_hard"]),
            "final_entropy": float(final_belief["entropy"]),
            "final_cluster_mass": float(final_belief["cluster_mass"]),
            "final_cluster_count": int(final_belief["cluster_count"]),
            "mean_policy_entropy": float(safe_mean([float(t["entropy"].detach().cpu().item()) for t in train_tensors])) if train_tensors else 0.0,
        },
        "steps": step_rows,
        "train_tensors": train_tensors,
    }


def summarise_case_rows(case_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not case_rows:
        return {}
    count = max(len(case_rows), 1)
    keys = [
        "reward_total",
        "reward_mean",
        "true_log_mass_delta_total",
        "margin_delta_total",
        "entropy_delta_total",
        "terminal_hit_bonus_total",
        "action_validity_rate",
        "exact_k_rate",
        "budget_used",
        "budget_used_mean",
        "final_top1_hit",
        "final_top3_hit",
        "final_top5_hit",
        "final_mrr",
        "final_true_mass",
        "final_margin_true_vs_hard",
        "final_entropy",
        "final_cluster_mass",
        "final_cluster_count",
        "mean_policy_entropy",
    ]
    out = {"case_count": float(len(case_rows))}
    for key in keys:
        out[key] = float(sum(float(row[key]) for row in case_rows) / count)
    ranks = [float(row["final_true_rank"]) for row in case_rows if row.get("final_true_rank") is not None]
    out["true_source_rank_mean"] = float(sum(ranks) / max(len(ranks), 1)) if ranks else float("nan")
    out["median_true_source_rank"] = float(pd.Series(ranks).median()) if ranks else float("nan")
    out["success_rate"] = out["final_top1_hit"]
    return out


def summarise_belief_step_rows(step_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not step_rows:
        return {}
    denom = max(len(step_rows), 1)
    return {
        "zero_reward_step_fraction": float(sum(float(abs(float(row["reward_total"])) <= 1e-12) for row in step_rows) / denom),
        "mean_delta_true_log_mass": float(sum(float(row["delta_true_log_mass"]) for row in step_rows) / denom),
        "mean_delta_margin_true_vs_hard": float(sum(float(row["delta_margin_true_vs_hard"]) for row in step_rows) / denom),
        "mean_delta_entropy": float(sum(float(row["delta_entropy"]) for row in step_rows) / denom),
        "mean_terminal_hit_bonus": float(sum(float(row["terminal_hit_bonus"]) for row in step_rows) / denom),
        "mean_pre_entropy": float(sum(float(row["pre_entropy"]) for row in step_rows) / denom),
        "mean_post_entropy": float(sum(float(row["post_entropy"]) for row in step_rows) / denom),
        "mean_pre_cluster_mass": float(sum(float(row["pre_cluster_mass"]) for row in step_rows) / denom),
        "mean_post_cluster_mass": float(sum(float(row["post_cluster_mass"]) for row in step_rows) / denom),
        "mean_pre_cluster_count": float(sum(float(row["pre_cluster_count"]) for row in step_rows) / denom),
        "mean_post_cluster_count": float(sum(float(row["post_cluster_count"]) for row in step_rows) / denom),
        "action_validity_rate": float(sum(float(row["valid_selection"]) for row in step_rows) / denom),
        "exact_k_rate": float(sum(float(row["exact_k"]) for row in step_rows) / denom),
    }


def build_runtime_context(source_root: Path, cache_dir: Path) -> Dict[str, Any]:
    oracle_arm_manifest = read_json(source_root / "arm_b_task_defined_oracle" / "run_manifest.json")
    bridge_package_dir = Path(oracle_arm_manifest["bridge_package_dir"])
    init_checkpoint = Path(oracle_arm_manifest["init_checkpoint"])
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    replayable_df = pd.read_csv(source_root / "same_case_replayable_manifest.csv")
    target_case_ids = replayable_df["case_id"].astype(str).tolist()
    cases, dataset_assets = load_same_cases(cfg_path=cfg_path, cache_dir=cache_dir, target_case_ids=target_case_ids)
    seed_meta = read_json(bridge_package_dir / "seed_metadata.json")
    control_bundle = load_control_bundle(Path(seed_meta["control_dir"]))
    nav_args = build_namespace_from_control_args(control_bundle["args"], "cpu")
    return {
        "bridge_package_dir": bridge_package_dir,
        "init_checkpoint": init_checkpoint,
        "cases": cases,
        "dataset_assets": dataset_assets,
        "num_episodes": int(getattr(nav_args, "num_episodes")),
        "action_budget": int(getattr(nav_args, "action_budget")),
        "episode_duration_min": float(getattr(nav_args, "episode_duration_min")),
        "frontier_role_mode": str(getattr(nav_args, "frontier_role_mode")),
        "source_summary": source_summary,
    }


def train_epoch(
    *,
    model: CleanNavigatorV1,
    optimizer: torch.optim.Optimizer,
    cases: Sequence[Any],
    belief_interface: FrozenBeliefInterface,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
    seed: int,
    myopic_topm: int,
    device: torch.device,
    gamma: float,
    value_coef: float,
    entropy_coef: float,
    grad_clip: float,
    optimizer_step_cases: int,
) -> Dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    step_counter = 0
    optimizer_steps = 0
    for case_idx, case in enumerate(cases):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) * 1000 + int(case_idx))
        out = run_case_rollout(
            case=case,
            policy_name="belief_rl",
            model=model,
            belief_interface=belief_interface,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            num_episodes=num_episodes,
            action_budget=action_budget,
            episode_duration_min=episode_duration_min,
            frontier_role_mode=frontier_role_mode,
            generator=generator,
            myopic_topm=myopic_topm,
            deterministic=False,
            device=device,
        )
        case_rows.append(out["summary"])
        step_rows.extend([{**row, "case_id": case.case_id} for row in out["steps"]])
        tensors = out["train_tensors"]
        if not tensors:
            continue
        rewards = [row["reward"] for row in tensors]
        returns = compute_returns(rewards, gamma=float(gamma), device=device)
        values = torch.stack([row["value"] for row in tensors])
        log_probs = torch.stack([row["log_prob"] for row in tensors])
        entropies = torch.stack([row["entropy"] for row in tensors])
        advantages = returns - values
        value_loss = torch.nn.functional.mse_loss(values, returns)
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_bonus = entropies.mean()
        loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_bonus
        (loss / float(max(optimizer_step_cases, 1))).backward()
        total_loss += float(loss.item())
        total_policy_loss += float(policy_loss.item())
        total_value_loss += float(value_loss.item())
        total_entropy += float(entropy_bonus.item())
        step_counter += 1
        should_step = ((case_idx + 1) % max(int(optimizer_step_cases), 1) == 0) or (case_idx == len(cases) - 1)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    summary = summarise_case_rows(case_rows)
    summary.update(
        {
            "train_loss_mean": float(total_loss / max(step_counter, 1)),
            "policy_loss_mean": float(total_policy_loss / max(step_counter, 1)),
            "value_loss_mean": float(total_value_loss / max(step_counter, 1)),
            "policy_entropy_mean": float(total_entropy / max(step_counter, 1)),
            "optimizer_step_cases": int(optimizer_step_cases),
            "optimizer_step_count": int(optimizer_steps),
        }
    )
    summary.update({f"step_{k}": v for k, v in summarise_belief_step_rows(step_rows).items()})
    return {"summary": summary, "case_rows": case_rows, "step_rows": step_rows}


def evaluate_policy(
    *,
    policy_name: str,
    cases: Sequence[Any],
    model: Optional[CleanNavigatorV1],
    belief_interface: FrozenBeliefInterface,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
    seed: int,
    myopic_topm: int,
    device: torch.device,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) * 1000 + int(case_idx))
        out = run_case_rollout(
            case=case,
            policy_name=policy_name,
            model=model,
            belief_interface=belief_interface,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            num_episodes=num_episodes,
            action_budget=action_budget,
            episode_duration_min=episode_duration_min,
            frontier_role_mode=frontier_role_mode,
            generator=generator,
            myopic_topm=myopic_topm,
            deterministic=True,
            device=device,
        )
        case_rows.append(out["summary"])
        step_rows.extend([{**row, "case_id": case.case_id} for row in out["steps"]])
    return {
        "summary": {**summarise_case_rows(case_rows), **{f"step_{k}": v for k, v in summarise_belief_step_rows(step_rows).items()}},
        "case_rows": case_rows,
        "step_rows": step_rows,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    source_root = Path(args.source_root)
    contrast_root = Path(args.contrast_root)

    runtime = build_runtime_context(source_root, cache_dir)
    contrast_summary = read_json(contrast_root / "summary.json")
    contrast_run_manifest = {
        "checkpoint_path": str(contrast_root / "train_eval" / "epoch_240" / "summary.json"),
        "best_epoch": int(read_json(contrast_root / "train_eval" / "best_checkpoint_summary.json")["best_by_train_mrr"]["epoch"]),
    }

    reasoner_cfg = build_cfg_with_contrast(
        bridge_package_dir=runtime["bridge_package_dir"],
        init_checkpoint=runtime["init_checkpoint"],
        cache_dir=cache_dir,
        epochs=240,
        periodic_every=20,
        batch_size=64,
    )
    frozen_reasoner_ckpt = PROJECT_ROOT / "runs" / "clean_aligned_semidynamic_37116e45862b" / "checkpoints" / "checkpoint_epoch_240.pt"
    reasoner_model = load_model_with_contrast(reasoner_cfg, frozen_reasoner_ckpt, device)
    reasoner_module = getattr(reasoner_model, "reasoner_module", reasoner_model)
    for param in reasoner_module.parameters():
        param.requires_grad = False
    reasoner_hash_before = hash_tensor_payload([param for param in reasoner_module.parameters()])
    belief_interface = FrozenBeliefInterface(
        reasoner_module=reasoner_module,
        device=device,
        temperature=float(args.belief_temperature),
        confusion_logit_delta=float(args.confusion_logit_delta),
    )

    env = CleanTwoChannelEvidenceEnv()
    model = CleanNavigatorV1(
        node_feature_dim=8,
        graph_feature_dim=6,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_slots=int(runtime["action_budget"]),
        greedy_eval=True,
        diversity_mode="none",
        complementarity_mode="none",
        role_mode="none",
        credit_mode="state_value",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    start_time = time.perf_counter()
    history_rows: List[Dict[str, Any]] = []
    checkpoints: List[Tuple[int, Path]] = []

    init_eval_rl = evaluate_policy(
        policy_name="belief_rl",
        cases=runtime["cases"],
        model=model,
        belief_interface=belief_interface,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        dataset_assets=runtime["dataset_assets"],
        num_episodes=runtime["num_episodes"],
        action_budget=runtime["action_budget"],
        episode_duration_min=runtime["episode_duration_min"],
        frontier_role_mode=runtime["frontier_role_mode"],
        seed=int(args.seed),
        myopic_topm=int(args.myopic_topm),
        device=device,
    )
    eval_dir = output_dir / "train_eval" / "init"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(init_eval_rl["case_rows"]).to_csv(eval_dir / "case_rows.csv", index=False)
    pd.DataFrame(init_eval_rl["step_rows"]).to_csv(eval_dir / "step_rows.csv", index=False)
    write_json(eval_dir / "summary.json", init_eval_rl["summary"])

    best_score = float(init_eval_rl["summary"].get("final_mrr", 0.0))
    best_epoch = 0
    best_path = output_dir / "checkpoints" / "checkpoint_epoch_000.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_path)

    for epoch in range(1, int(args.epochs) + 1):
        epoch_seed = int(args.seed) + int(epoch)
        train_out = train_epoch(
            model=model,
            optimizer=optimizer,
            cases=runtime["cases"],
            belief_interface=belief_interface,
            env=env,
            topology=runtime["dataset_assets"]["topology"],
            dataset_assets=runtime["dataset_assets"],
            num_episodes=runtime["num_episodes"],
            action_budget=runtime["action_budget"],
            episode_duration_min=runtime["episode_duration_min"],
            frontier_role_mode=runtime["frontier_role_mode"],
            seed=epoch_seed,
            myopic_topm=int(args.myopic_topm),
            device=device,
            gamma=float(args.gamma),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            grad_clip=float(args.grad_clip),
            optimizer_step_cases=int(args.optimizer_step_cases),
        )
        row = {"epoch": int(epoch), "train_summary": train_out["summary"]}
        history_rows.append(row)
        if epoch % int(args.periodic_eval_every) == 0 or epoch == int(args.epochs):
            ckpt_path = output_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            checkpoints.append((int(epoch), ckpt_path))
            model.eval()
            eval_out = evaluate_policy(
                policy_name="belief_rl",
                cases=runtime["cases"],
                model=model,
                belief_interface=belief_interface,
                env=env,
                topology=runtime["dataset_assets"]["topology"],
                dataset_assets=runtime["dataset_assets"],
                num_episodes=runtime["num_episodes"],
                action_budget=runtime["action_budget"],
                episode_duration_min=runtime["episode_duration_min"],
                frontier_role_mode=runtime["frontier_role_mode"],
                seed=int(args.seed),
                myopic_topm=int(args.myopic_topm),
                device=device,
            )
            epoch_dir = output_dir / "train_eval" / f"epoch_{epoch:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(eval_out["case_rows"]).to_csv(epoch_dir / "case_rows.csv", index=False)
            pd.DataFrame(eval_out["step_rows"]).to_csv(epoch_dir / "step_rows.csv", index=False)
            write_json(epoch_dir / "summary.json", eval_out["summary"])
            score = float(eval_out["summary"].get("final_mrr", 0.0))
            if score > best_score:
                best_score = score
                best_epoch = int(epoch)
                best_path = ckpt_path
        print(json.dumps({"epoch": int(epoch), "train_reward_total": train_out["summary"].get("reward_total", 0.0), "train_success_rate": train_out["summary"].get("success_rate", 0.0), "train_policy_entropy": train_out["summary"].get("policy_entropy_mean", 0.0)}))

    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state)
    final_eval_rl = evaluate_policy(
        policy_name="belief_rl",
        cases=runtime["cases"],
        model=model,
        belief_interface=belief_interface,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        dataset_assets=runtime["dataset_assets"],
        num_episodes=runtime["num_episodes"],
        action_budget=runtime["action_budget"],
        episode_duration_min=runtime["episode_duration_min"],
        frontier_role_mode=runtime["frontier_role_mode"],
        seed=int(args.seed),
        myopic_topm=int(args.myopic_topm),
        device=device,
    )
    final_dir = output_dir / "train_eval" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_eval_rl["case_rows"]).to_csv(final_dir / "case_rows.csv", index=False)
    pd.DataFrame(final_eval_rl["step_rows"]).to_csv(final_dir / "step_rows.csv", index=False)
    write_json(final_dir / "summary.json", final_eval_rl["summary"])

    legacy_top_support = evaluate_policy(
        policy_name="top_support_legacy",
        cases=runtime["cases"],
        model=None,
        belief_interface=belief_interface,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        dataset_assets=runtime["dataset_assets"],
        num_episodes=runtime["num_episodes"],
        action_budget=runtime["action_budget"],
        episode_duration_min=runtime["episode_duration_min"],
        frontier_role_mode=runtime["frontier_role_mode"],
        seed=int(args.seed),
        myopic_topm=int(args.myopic_topm),
        device=device,
    )
    myopic_belief = evaluate_policy(
        policy_name="myopic_belief_entropy",
        cases=runtime["cases"],
        model=None,
        belief_interface=belief_interface,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        dataset_assets=runtime["dataset_assets"],
        num_episodes=runtime["num_episodes"],
        action_budget=runtime["action_budget"],
        episode_duration_min=runtime["episode_duration_min"],
        frontier_role_mode=runtime["frontier_role_mode"],
        seed=int(args.seed),
        myopic_topm=int(args.myopic_topm),
        device=device,
    )

    baseline_rows = []
    for name, payload in [
        ("belief_rl_best", final_eval_rl),
        ("top_support_legacy", legacy_top_support),
        ("myopic_belief_entropy", myopic_belief),
    ]:
        baseline_rows.append({"policy_name": name, **payload["summary"]})
        pd.DataFrame(payload["case_rows"]).to_csv(output_dir / f"{name}_case_rows.csv", index=False)
        pd.DataFrame(payload["step_rows"]).to_csv(output_dir / f"{name}_step_rows.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output_dir / "policy_compare.csv", index=False)

    reward_audit_rows = []
    for policy_name, payload in [("belief_rl_best", final_eval_rl), ("myopic_belief_entropy", myopic_belief)]:
        case_df = pd.DataFrame(payload["case_rows"])[["case_id", "final_top1_hit"]]
        step_df = pd.DataFrame(payload["step_rows"]).merge(case_df, on="case_id", how="left")
        for success_value, sub in step_df.groupby("final_top1_hit"):
            reward_audit_rows.append(
                {
                    "policy_name": policy_name,
                    "final_top1_hit": float(success_value),
                    "step_count": int(len(sub)),
                    "reward_total_mean": float(sub["reward_total"].mean()),
                    "delta_true_log_mass_mean": float(sub["delta_true_log_mass"].mean()),
                    "delta_margin_true_vs_hard_mean": float(sub["delta_margin_true_vs_hard"].mean()),
                    "delta_entropy_mean": float(sub["delta_entropy"].mean()),
                    "terminal_hit_bonus_mean": float(sub["terminal_hit_bonus"].mean()),
                }
            )
    reward_audit_df = pd.DataFrame(reward_audit_rows)
    reward_audit_df.to_csv(output_dir / "reward_audit.csv", index=False)

    reasoner_hash_after = hash_tensor_payload([param for param in reasoner_module.parameters()])
    runtime_s = float(time.perf_counter() - start_time)
    write_json(
        output_dir / "summary.json",
        {
            "runner_version": RUNNER_VERSION,
            "panel_version": PANEL_VERSION,
            "seed": int(args.seed),
            "cases": int(len(runtime["cases"])),
            "epochs": int(args.epochs),
            "periodic_eval_every": int(args.periodic_eval_every),
            "optimizer_step_cases": int(args.optimizer_step_cases),
            "belief_interface": {
                "type": "posterior_like_masked_softmax",
                "temperature": float(args.belief_temperature),
                "confusion_logit_delta": float(args.confusion_logit_delta),
                "outputs": [
                    "candidate_belief_distribution",
                    "entropy",
                    "top1_mass",
                    "top3_mass",
                    "cluster_mass",
                    "cluster_count",
                    "hardest_confuser_local",
                    "margin_true_vs_hard",
                ],
                "note": "Not a true Bayesian posterior; a stable decision-belief contract derived from the frozen contrast-injected reasoner.",
            },
            "reasoner_frozen_audit": {
                "contrast_root": str(contrast_root),
                "frozen_checkpoint": str(frozen_reasoner_ckpt),
                "trainable_param_count": int(sum(int(param.requires_grad) * param.numel() for param in reasoner_module.parameters())),
                "param_hash_before": str(reasoner_hash_before),
                "param_hash_after": str(reasoner_hash_after),
                "hash_stable": bool(reasoner_hash_before == reasoner_hash_after),
            },
            "reward_formula": {
                "reward_total": "1.0*delta_log_true_mass + 0.75*delta_margin_true_vs_hard + 0.25*delta_entropy + terminal_hit_bonus + invalid_penalty + repeated_sampling_penalty + shortfall_penalty",
                "why": {
                    "delta_log_true_mass": "Rewards posterior concentration on the true source.",
                    "delta_margin_true_vs_hard": "Directly rewards separation from the current hardest confuser.",
                    "delta_entropy": "Rewards uncertainty reduction at the belief level.",
                    "terminal_hit_bonus": "Keeps a sparse task-aligned target in the loop.",
                },
            },
            "train_runtime_s": runtime_s,
            "best_epoch": int(best_epoch),
            "best_checkpoint_path": str(best_path),
            "contrast_training_summary": contrast_summary,
            "baseline_compare_path": str(output_dir / "policy_compare.csv"),
            "reward_audit_path": str(output_dir / "reward_audit.csv"),
        },
    )
    write_json(output_dir / "history.json", history_rows)


if __name__ == "__main__":
    main()
