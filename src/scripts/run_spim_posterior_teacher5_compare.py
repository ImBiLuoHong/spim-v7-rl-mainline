from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_audit import load_runtime_context
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    CaseRecord,
    colon_case_id_from_data,
    collect_dataset_assets,
    make_rollout_state,
    read_json,
)
from src.scripts.run_spim_family_sweep import (
    PaperLikeHSRState,
    _belief_metrics,
    _extract_trigger_global,
)
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_PRECHECK_ROOT,
    DEFAULT_SOURCE_ROOT,
    auto_select_teacher,
    compute_teacher_belief,
    summarize_case_metrics,
)


RUNNER_VERSION = "spim_posterior_teacher5_compare_v1"
PANEL_VERSION = "exact136_b30_spim_v3_teacher5_single_sampling_v1"

TEACHER_POLICIES = [
    "posterior_greedy",
    "posterior_entropy_drop",
    "posterior_cover_shrink",
    "posterior_disagreement_split",
    "posterior_thompson_sampling",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified fair compare for 5 posterior-based single-sampling teachers under one SPIM posterior family."
    )
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--precheck-root", type=str, default=str(DEFAULT_PRECHECK_ROOT))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "spim_teacher5_compare" / "20260412_exact136_b30_v1"),
    )
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--split", type=str, default="exact136", choices=["exact136", "val"])
    parser.add_argument("--case-limit", type=int, default=0)

    parser.add_argument(
        "--teacher-family",
        type=str,
        default="auto",
        choices=["auto", "hsr_soft_scenario_posterior_v3", "hsr_paper_topk_ema_v1"],
    )
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--actions-per-round", type=int, default=3)

    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)

    parser.add_argument("--obs-tau-min", type=float, default=30.0)
    parser.add_argument("--obs-eps-fp", type=float, default=0.02)
    parser.add_argument("--obs-eps-fn", type=float, default=0.05)
    parser.add_argument("--source-support-mass-tau", type=float, default=0.95)
    parser.add_argument("--cover-tau", type=float, default=0.7)
    parser.add_argument("--thompson-num-samples", type=int, default=5)
    parser.add_argument("--progress-every-cases", type=int, default=10)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_probs(prob: torch.Tensor) -> torch.Tensor:
    vals = prob.view(-1).float().clone()
    vals = torch.clamp(vals, min=0.0)
    denom = float(vals.sum().item())
    if denom <= 1e-12:
        return torch.full_like(vals, 1.0 / max(int(vals.numel()), 1))
    return vals / denom


def _entropy(prob: torch.Tensor) -> float:
    p = prob.view(-1).float().clamp_min(1e-12)
    return float((-(p * torch.log(p))).sum().item())


def _cover_size(prob: torch.Tensor, tau: float) -> int:
    p = _normalize_probs(prob)
    if int(p.numel()) <= 0:
        return 0
    order = torch.argsort(p, descending=True)
    csum = torch.cumsum(p[order], dim=0)
    hit = (csum >= float(tau)).nonzero(as_tuple=True)[0]
    if hit.numel() <= 0:
        return int(p.numel())
    return int(hit[0].item()) + 1


def _sample_seed(base_seed: int, case_id: str, episode_idx: int, policy_name: str) -> int:
    token = f"{base_seed}:{case_id}:{episode_idx}:{policy_name}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def _select_mass_cover_indices(
    belief: torch.Tensor,
    source_mask: torch.Tensor,
    mass_tau: float,
) -> torch.Tensor:
    mask = source_mask.view(-1).bool()
    idx = torch.nonzero(mask, as_tuple=True)[0]
    if idx.numel() <= 0:
        return idx
    probs = belief[idx].float().clamp_min(0.0)
    if float(probs.sum().item()) <= 1e-12:
        return idx
    probs = probs / probs.sum().clamp_min(1e-12)
    order_local = torch.argsort(probs, descending=True)
    probs_sorted = probs[order_local]
    csum = torch.cumsum(probs_sorted, dim=0)
    keep = int((csum >= float(mass_tau)).nonzero(as_tuple=True)[0][0].item()) + 1
    keep = max(1, min(keep, int(idx.numel())))
    return idx[order_local[:keep]]


def _build_action_source_obs_matrix(
    *,
    state: Dict[str, Any],
    num_nodes: int,
    action_idx: torch.Tensor,
    source_idx: torch.Tensor,
    tau_min: float,
    eps_fp: float,
    eps_fn: float,
) -> torch.Tensor:
    if action_idx.numel() <= 0 or source_idx.numel() <= 0:
        return torch.zeros((int(action_idx.numel()), int(source_idx.numel())), dtype=torch.float32)
    gate = DynamicReachabilityRuleModule()
    seed = source_idx.to(dtype=torch.long, device=state["edge_index"].device)
    dist = gate.compute_distance_matrix(
        seed_indices=seed,
        physics_context=state["phys_ctx"],
        num_nodes=int(num_nodes),
    ).detach().cpu().float()
    arrival = dist[action_idx.long(), :]
    t_obs = float(state["info"]["time_min"])
    p_arrived = torch.sigmoid((float(t_obs) - arrival) / max(float(tau_min), 1e-6))
    p_pos = float(eps_fp) + (1.0 - float(eps_fp) - float(eps_fn)) * p_arrived
    return p_pos.clamp(1e-6, 1.0 - 1e-6)


def _compute_information_scores(
    *,
    belief: torch.Tensor,
    source_idx: torch.Tensor,
    action_idx: torch.Tensor,
    p_pos_a_s: torch.Tensor,
    cover_tau: float,
) -> Dict[str, torch.Tensor]:
    if source_idx.numel() <= 0 or action_idx.numel() <= 0:
        empty = torch.zeros((int(action_idx.numel()),), dtype=torch.float32)
        return {
            "expected_entropy": empty.clone(),
            "cover_shrink": empty.clone(),
            "disagreement": empty.clone(),
            "p_plus": empty.clone(),
        }

    prior = _normalize_probs(belief[source_idx].float())
    p_plus = torch.matmul(p_pos_a_s, prior).clamp(1e-6, 1.0 - 1e-6)
    curr_cover = _cover_size(prior, float(cover_tau))

    exp_entropy: List[float] = []
    exp_cover_shrink: List[float] = []
    for row in range(int(action_idx.numel())):
        like_pos = p_pos_a_s[row, :].clamp(1e-6, 1.0 - 1e-6)
        like_neg = (1.0 - like_pos).clamp(1e-6, 1.0 - 1e-6)
        post_pos = _normalize_probs(prior * like_pos)
        post_neg = _normalize_probs(prior * like_neg)
        h_pos = _entropy(post_pos)
        h_neg = _entropy(post_neg)
        k_pos = _cover_size(post_pos, float(cover_tau))
        k_neg = _cover_size(post_neg, float(cover_tau))
        p = float(p_plus[row].item())
        exp_entropy.append(float(p * h_pos + (1.0 - p) * h_neg))
        exp_cover = float(p * k_pos + (1.0 - p) * k_neg)
        exp_cover_shrink.append(float(curr_cover - exp_cover))

    disagreement = torch.minimum(p_plus, 1.0 - p_plus)
    return {
        "expected_entropy": torch.tensor(exp_entropy, dtype=torch.float32),
        "cover_shrink": torch.tensor(exp_cover_shrink, dtype=torch.float32),
        "disagreement": disagreement.float(),
        "p_plus": p_plus.float(),
    }


def _select_actions_for_teacher(
    *,
    policy_name: str,
    base_seed: int,
    case_id: str,
    episode_idx: int,
    state: Dict[str, Any],
    rollout: PracticalRollout,
    belief_ctx: Dict[str, Any],
    action_budget: int,
    source_support_mass_tau: float,
    cover_tau: float,
    obs_tau_min: float,
    obs_eps_fp: float,
    obs_eps_fn: float,
    thompson_num_samples: int,
) -> Dict[str, Any]:
    belief = belief_ctx["belief"].view(-1).float().cpu()
    candidate_mask = belief_ctx["candidate_mask"].view(-1).bool().cpu()
    sampled_mask = rollout.revealed_mask.view(-1).bool().cpu()
    available_mask = candidate_mask & (~sampled_mask)
    available_idx = torch.nonzero(available_mask, as_tuple=True)[0]
    if available_idx.numel() <= 0:
        return {"actions": [], "diagnostics": {"policy_name": str(policy_name), "empty_available": True}}

    k = min(int(action_budget), int(available_idx.numel()))
    if policy_name == "posterior_greedy":
        score = belief[available_idx]
        order = torch.argsort(score, descending=True)
        return {
            "actions": [int(v) for v in available_idx[order[:k]].tolist()],
            "diagnostics": {"policy_name": str(policy_name)},
        }

    source_idx = _select_mass_cover_indices(
        belief=belief,
        source_mask=available_mask,
        mass_tau=float(source_support_mass_tau),
    )
    if source_idx.numel() <= 0:
        score = belief[available_idx]
        order = torch.argsort(score, descending=True)
        return {
            "actions": [int(v) for v in available_idx[order[:k]].tolist()],
            "diagnostics": {"policy_name": str(policy_name), "fallback": "no_source_support"},
        }

    p_pos_a_s = _build_action_source_obs_matrix(
        state=state,
        num_nodes=int(rollout.num_nodes),
        action_idx=available_idx,
        source_idx=source_idx,
        tau_min=float(obs_tau_min),
        eps_fp=float(obs_eps_fp),
        eps_fn=float(obs_eps_fn),
    )
    info_scores = _compute_information_scores(
        belief=belief,
        source_idx=source_idx,
        action_idx=available_idx,
        p_pos_a_s=p_pos_a_s,
        cover_tau=float(cover_tau),
    )

    if policy_name == "posterior_entropy_drop":
        order = torch.argsort(info_scores["expected_entropy"], descending=False)
        selected = available_idx[order[:k]]
    elif policy_name == "posterior_cover_shrink":
        order = torch.argsort(info_scores["cover_shrink"], descending=True)
        selected = available_idx[order[:k]]
    elif policy_name == "posterior_disagreement_split":
        order = torch.argsort(info_scores["disagreement"], descending=True)
        selected = available_idx[order[:k]]
    elif policy_name == "posterior_thompson_sampling":
        prior = _normalize_probs(belief[source_idx])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_sample_seed(base_seed, case_id, int(episode_idx), str(policy_name)))
        vote = torch.zeros((int(available_idx.numel()),), dtype=torch.float32)
        tie = torch.zeros((int(available_idx.numel()),), dtype=torch.float32)
        n_samples = max(int(thompson_num_samples), 1)
        for _ in range(n_samples):
            draw = torch.multinomial(prior, num_samples=1, replacement=True, generator=gen)
            src_col = int(draw[0].item())
            sampled_score = p_pos_a_s[:, src_col]
            order = torch.argsort(sampled_score, descending=True)
            for rank, pos in enumerate(order[:k]):
                vote[int(pos.item())] += float(k - rank)
                tie[int(pos.item())] += float(sampled_score[int(pos.item())].item())
        score = vote * 1e3 + tie
        order = torch.argsort(score, descending=True)
        selected = available_idx[order[:k]]
    else:
        raise ValueError(f"Unsupported teacher policy: {policy_name}")

    return {
        "actions": [int(v) for v in selected.tolist()],
        "diagnostics": {
            "policy_name": str(policy_name),
            "support_size": int(source_idx.numel()),
            "available_size": int(available_idx.numel()),
            "mean_p_plus_selected": float(info_scores["p_plus"][torch.isin(available_idx, selected)].mean().item())
            if int(selected.numel()) > 0
            else None,
        },
    }


def run_case(
    *,
    case: CaseRecord,
    policy_name: str,
    case_index: int,
    base_seed: int,
    family: str,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
    source_support_mass_tau: float,
    cover_tau: float,
    obs_tau_min: float,
    obs_eps_fp: float,
    obs_eps_fn: float,
    thompson_num_samples: int,
) -> Dict[str, Any]:
    rollout = PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=runtime["dataset_assets"]["global_edge_index"],
        stt_dynamic_series=runtime["dataset_assets"]["stt_dynamic_series"],
        num_global_nodes=int(runtime["dataset_assets"]["num_global_nodes"]),
        num_episodes=int(runtime["num_episodes"]),
        samples_per_episode=int(runtime["action_budget"]),
        episode_duration_min=float(runtime["episode_duration_min"]),
    )
    history = ObservationWitnessHistory()
    trigger_global = _extract_trigger_global(case.data)
    source_local = resolve_source_local_idx(rollout)
    source_global = None if source_local is None else int(rollout.g_ids[int(source_local)].item())
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]
    paper_state = PaperLikeHSRState(source_prior=None)

    step_rows: List[Dict[str, Any]] = []
    hit_round: Optional[int] = None
    hit_sample_index: Optional[int] = None
    budget_used = 0
    termination_reason = "budget_exhausted"

    for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
        state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=runtime["dataset_assets"]["topology"],
            num_episodes=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
            frontier_role_mode=str(runtime["frontier_role_mode"]),
        )
        if int(state["valid_mask"].sum().item()) <= 0:
            termination_reason = "no_valid_nodes"
            break

        belief_ctx = compute_teacher_belief(
            family=family,
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=onset_grid,
            paper_like_alpha=float(paper_like_alpha),
            paper_like_topk_fraction=float(paper_like_topk_fraction),
            paper_like_time_tol_min=float(paper_like_time_tol_min),
            soft_scenario_beta=float(soft_scenario_beta),
        )

        action_out = _select_actions_for_teacher(
            policy_name=str(policy_name),
            base_seed=int(base_seed + case_index * 1009),
            case_id=str(case.case_id),
            episode_idx=int(episode_idx),
            state=state,
            rollout=rollout,
            belief_ctx=belief_ctx,
            action_budget=int(runtime["action_budget"]),
            source_support_mass_tau=float(source_support_mass_tau),
            cover_tau=float(cover_tau),
            obs_tau_min=float(obs_tau_min),
            obs_eps_fp=float(obs_eps_fp),
            obs_eps_fn=float(obs_eps_fn),
            thompson_num_samples=int(thompson_num_samples),
        )
        selected_actions = [int(v) for v in action_out["actions"]]
        if not selected_actions:
            termination_reason = "no_action"
            break

        selected_global_ids = [int(rollout.g_ids[int(v)].item()) for v in selected_actions]
        round_hit = source_local is not None and int(source_local) in set(selected_actions)
        if round_hit and hit_round is None:
            hit_round = int(episode_idx)
            source_slot = selected_actions.index(int(source_local)) + 1
            hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

        rollout.step_with_actions(
            selected_actions,
            sample_types=[f"{policy_name}_slot_{i}" for i in range(len(selected_actions))],
        )
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])
        budget_used += int(len(selected_actions))

        metrics = _belief_metrics(belief_ctx["belief"], belief_ctx["candidate_mask"], source_local, threshold=float(cover_tau))
        step_rows.append(
            {
                "case_id": str(case.case_id),
                "policy_name": str(policy_name),
                "episode_index": int(episode_idx),
                "selected_local_ids": json.dumps([int(v) for v in selected_actions]),
                "selected_global_ids": json.dumps([int(v) for v in selected_global_ids]),
                "selected_count": int(len(selected_actions)),
                "source_hit_in_round": float(bool(round_hit)),
                "hit_sample_index": None if not round_hit else int(hit_sample_index),
                "posterior_entropy": float(metrics["entropy"]),
                "mass_cover_tau_ratio": float(metrics["mass_cover_size_ratio"]),
                "top1_mass": float(metrics["top1_mass"]),
                "top3_mass": float(metrics["top3_mass"]),
                "teacher_diag": json.dumps(action_out["diagnostics"], ensure_ascii=False),
            }
        )
        if round_hit:
            termination_reason = "source_hit"
            break

    final_state = make_rollout_state(
        case=case,
        rollout=rollout,
        history=history,
        env=env,
        topology=runtime["dataset_assets"]["topology"],
        num_episodes=int(runtime["num_episodes"]),
        action_budget=int(runtime["action_budget"]),
        frontier_role_mode=str(runtime["frontier_role_mode"]),
    )
    final_belief_ctx = compute_teacher_belief(
        family=family,
        rollout=rollout,
        state=final_state,
        history=history,
        trigger_global=trigger_global,
        paper_state=paper_state,
        onset_offsets_min=onset_grid,
        paper_like_alpha=float(paper_like_alpha),
        paper_like_topk_fraction=float(paper_like_topk_fraction),
        paper_like_time_tol_min=float(paper_like_time_tol_min),
        soft_scenario_beta=float(soft_scenario_beta),
    )
    final_metrics = _belief_metrics(final_belief_ctx["belief"], final_belief_ctx["candidate_mask"], source_local)
    case_row = {
        "case_id": str(case.case_id),
        "scenario_id": int(case.scenario_id),
        "part_id": int(case.part_id),
        "policy_name": str(policy_name),
        "success_rate": float(hit_round is not None),
        "hit_round": None if hit_round is None else int(hit_round),
        "hit_sample_index": None if hit_sample_index is None else int(hit_sample_index),
        "budget_used": int(budget_used),
        "avg_step_reward": 0.0,
        "teacher_exact_match_rate": 1.0,
        "final_top1_mass": float(final_metrics["top1_mass"]),
        "final_top3_mass": float(final_metrics["top3_mass"]),
        "final_entropy": float(final_metrics["entropy"]),
        "termination_reason": str(termination_reason),
        "source_global_id": source_global,
        "trigger_global_id": trigger_global,
    }
    return {"case_row": case_row, "step_rows": step_rows}


def run_policy(
    *,
    cases: Sequence[CaseRecord],
    policy_name: str,
    family: str,
    runtime: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    base_seed: int,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
    source_support_mass_tau: float,
    cover_tau: float,
    obs_tau_min: float,
    obs_eps_fp: float,
    obs_eps_fn: float,
    thompson_num_samples: int,
    progress_every_cases: int,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for case_idx, case in enumerate(cases, start=1):
        out = run_case(
            case=case,
            policy_name=str(policy_name),
            case_index=int(case_idx - 1),
            base_seed=int(base_seed),
            family=family,
            runtime=runtime,
            env=env,
            paper_like_alpha=float(paper_like_alpha),
            paper_like_topk_fraction=float(paper_like_topk_fraction),
            paper_like_time_tol_min=float(paper_like_time_tol_min),
            soft_scenario_beta=float(soft_scenario_beta),
            source_support_mass_tau=float(source_support_mass_tau),
            cover_tau=float(cover_tau),
            obs_tau_min=float(obs_tau_min),
            obs_eps_fp=float(obs_eps_fp),
            obs_eps_fn=float(obs_eps_fn),
            thompson_num_samples=int(thompson_num_samples),
        )
        case_rows.append(out["case_row"])
        step_rows.extend(out["step_rows"])
        if int(progress_every_cases) > 0 and (case_idx % int(progress_every_cases) == 0):
            print(f"[progress] policy={policy_name} case={case_idx}/{len(cases)}", flush=True)
    return {"case_rows": case_rows, "step_rows": step_rows}


def build_case_level_compare(all_case_rows: List[Dict[str, Any]], policies: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(all_case_rows)
    use_cols = ["case_id", "policy_name", "success_rate", "hit_round", "hit_sample_index", "budget_used", "termination_reason"]
    df = df[use_cols].copy()
    wide = None
    for col in ["success_rate", "hit_round", "hit_sample_index", "budget_used", "termination_reason"]:
        pivot = df.pivot(index="case_id", columns="policy_name", values=col)
        pivot.columns = [f"{p}__{col}" for p in pivot.columns]
        wide = pivot if wide is None else wide.join(pivot, how="outer")
    out = wide.reset_index()
    success_cols = [f"{p}__success_rate" for p in policies if f"{p}__success_rate" in out.columns]
    if success_cols:
        out["num_policy_success"] = out[success_cols].fillna(0.0).sum(axis=1)
    return out


def build_pairwise_complementarity(all_case_rows: List[Dict[str, Any]], policies: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(all_case_rows)[["case_id", "policy_name", "success_rate"]].copy()
    wide = df.pivot(index="case_id", columns="policy_name", values="success_rate").fillna(0.0)
    rows: List[Dict[str, Any]] = []
    for i, p1 in enumerate(policies):
        for p2 in policies[i + 1 :]:
            if p1 not in wide.columns or p2 not in wide.columns:
                continue
            s1 = wide[p1] > 0.5
            s2 = wide[p2] > 0.5
            rows.append(
                {
                    "policy_a": p1,
                    "policy_b": p2,
                    "a_only_success_cases": int((s1 & (~s2)).sum()),
                    "b_only_success_cases": int((s2 & (~s1)).sum()),
                    "both_success_cases": int((s1 & s2).sum()),
                    "either_success_cases": int((s1 | s2).sum()),
                    "jaccard_success": float(((s1 & s2).sum()) / max(int((s1 | s2).sum()), 1)),
                }
            )
    return pd.DataFrame(rows)


def _load_cfg_from_source(source_root: Path) -> Path:
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    return Path(oracle_manifest["config_path"])


def _load_split_cases(
    *,
    cfg_path: Path,
    cache_dir: Path,
    split: str,
    case_limit: int,
) -> tuple[List[CaseRecord], Dict[str, Any], Dict[str, Any]]:
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if isinstance(payload.get("life_support"), dict) and str(payload["life_support"].get("profile")) == "custom_direct_edit":
        payload = dict(payload)
        payload["life_support"] = {k: v for k, v in payload["life_support"].items() if k != "profile"}

    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(payload)
    cfg.training.enable_eval = False
    cfg.training.train_only = False
    cfg.training.enable_wandb = False
    cfg.data.skip_lmdb = False
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.max_samples = None
    cfg.data.rebuild_cache = False
    cfg.paths.cache_dir = str(cache_dir)

    train_loader, val_loader, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        eval_batch_size=1,
        skip_lmdb=False,
        train_only=False,
    )
    _ = train_loader
    if split != "val":
        raise ValueError(f"Unsupported split: {split}")
    if val_loader is None:
        raise RuntimeError("val loader is None")
    dataset = val_loader.dataset

    assets = collect_dataset_assets(dataset)
    if assets.get("topology") is None:
        assets["topology"] = HydraulicTopology(cfg.paths.foundation_path)

    limit = len(dataset) if int(case_limit) <= 0 else min(int(case_limit), len(dataset))
    cases: List[CaseRecord] = []
    for dataset_idx in range(limit):
        data = dataset[dataset_idx]
        case_id, scenario_id, part_id = colon_case_id_from_data(data, str(split), dataset_idx)
        cases.append(
            CaseRecord(
                case_id=str(case_id),
                scenario_id=int(scenario_id),
                part_id=int(part_id),
                dataset_index=int(dataset_idx),
                data=deepcopy(data),
            )
        )
    meta = {
        "cfg_path": str(cfg_path),
        "split_dir": str(getattr(cfg.paths, "split_dir", "")),
        "cache_version": str(getattr(cfg.data, "cache_version", "")),
        "split": str(split),
        "case_limit": int(case_limit),
        "loaded_case_count": int(len(cases)),
        "full_dataset_count": int(len(dataset)),
    }
    return cases, assets, meta


def build_runtime_for_split(
    *,
    source_root: Path,
    cache_dir: Path,
    split: str,
    num_rounds: int,
    actions_per_round: int,
    case_limit: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if split == "exact136":
        runtime = load_runtime_context(source_root, cache_dir)
        runtime["num_episodes"] = int(num_rounds)
        runtime["action_budget"] = int(actions_per_round)
        if int(case_limit) > 0:
            runtime["cases"] = list(runtime["cases"][: int(case_limit)])
        meta = {
            "split": "exact136",
            "cfg_path": None,
            "split_dir": None,
            "cache_version": None,
            "case_limit": int(case_limit),
            "loaded_case_count": int(len(runtime.get("cases", []))),
        }
        return runtime, meta

    cfg_path = _load_cfg_from_source(source_root)
    cases, dataset_assets, split_meta = _load_split_cases(
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        split=split,
        case_limit=int(case_limit),
    )
    exact_runtime = load_runtime_context(source_root, cache_dir)
    runtime = {
        "cases": cases,
        "dataset_assets": dataset_assets,
        "num_episodes": int(num_rounds),
        "action_budget": int(actions_per_round),
        "episode_duration_min": float(exact_runtime["episode_duration_min"]),
        "frontier_role_mode": str(exact_runtime["frontier_role_mode"]),
    }
    return runtime, split_meta


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    precheck_root = Path(args.precheck_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime, split_meta = build_runtime_for_split(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(args.split),
        num_rounds=int(args.num_rounds),
        actions_per_round=int(args.actions_per_round),
        case_limit=int(args.case_limit),
    )

    if str(args.teacher_family) == "auto":
        teacher_decision = auto_select_teacher(precheck_root)
        teacher_family = str(teacher_decision["selected_family"])
    else:
        teacher_family = str(args.teacher_family)
        teacher_decision = {
            "selected_family": str(teacher_family),
            "selection_rule": "manual_override",
            "v3_success_rate": None,
            "v1_success_rate": None,
            "precheck_path": str(precheck_root / "family_summary.csv"),
        }

    env = CleanTwoChannelEvidenceEnv()
    all_case_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []
    round_curve_rows: List[Dict[str, Any]] = []
    budget_curve_rows: List[Dict[str, Any]] = []
    per_policy_summary: Dict[str, Any] = {}

    for policy_name in TEACHER_POLICIES:
        print(f"[run] policy={policy_name}", flush=True)
        out = run_policy(
            cases=runtime["cases"],
            policy_name=str(policy_name),
            family=str(teacher_family),
            runtime=runtime,
            env=env,
            base_seed=int(args.seed),
            paper_like_alpha=float(args.paper_like_alpha),
            paper_like_topk_fraction=float(args.paper_like_topk_fraction),
            paper_like_time_tol_min=float(args.paper_like_time_tol_min),
            soft_scenario_beta=float(args.soft_scenario_beta),
            source_support_mass_tau=float(args.source_support_mass_tau),
            cover_tau=float(args.cover_tau),
            obs_tau_min=float(args.obs_tau_min),
            obs_eps_fp=float(args.obs_eps_fp),
            obs_eps_fn=float(args.obs_eps_fn),
            thompson_num_samples=int(args.thompson_num_samples),
            progress_every_cases=int(args.progress_every_cases),
        )
        case_rows = out["case_rows"]
        step_rows = out["step_rows"]
        all_case_rows.extend(case_rows)
        all_step_rows.extend(step_rows)

        summary = summarize_case_metrics(
            case_rows=case_rows,
            num_rounds=int(args.num_rounds),
            action_budget=int(args.actions_per_round),
        )
        per_policy_summary[str(policy_name)] = summary
        leaderboard_rows.append(
            {
                "policy_name": str(policy_name),
                "success_rate": float(summary["success_rate"]),
                "avg_hit_round_conditional": summary["avg_hit_round_conditional"],
                "budget_used_mean": float(summary["budget_used_mean"]),
                "case_count": int(summary["case_count"]),
            }
        )
        for row in summary.get("round_curve", []):
            round_curve_rows.append({"policy_name": str(policy_name), **row})
        for row in summary.get("budget_curve", []):
            budget_curve_rows.append({"policy_name": str(policy_name), **row})

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by=["success_rate", "avg_hit_round_conditional"],
        ascending=[False, True],
        na_position="last",
    )
    case_compare_df = build_case_level_compare(all_case_rows=all_case_rows, policies=TEACHER_POLICIES)
    pairwise_df = build_pairwise_complementarity(all_case_rows=all_case_rows, policies=TEACHER_POLICIES)

    pd.DataFrame(all_case_rows).to_csv(output_dir / "teacher5_case_rows.csv", index=False)
    pd.DataFrame(all_step_rows).to_csv(output_dir / "teacher5_step_rows.csv", index=False)
    leaderboard_df.to_csv(output_dir / "teacher5_leaderboard.csv", index=False)
    pd.DataFrame(round_curve_rows).to_csv(output_dir / "roundwise_success_curve.csv", index=False)
    pd.DataFrame(budget_curve_rows).to_csv(output_dir / "budget_success_curve.csv", index=False)
    case_compare_df.to_csv(output_dir / "case_level_compare.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_complementarity.csv", index=False)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "protocol": {
            "split": str(args.split),
            "budget_name": "B30",
            "num_rounds": int(args.num_rounds),
            "actions_per_round": int(args.actions_per_round),
            "sample_budget": int(args.num_rounds) * int(args.actions_per_round),
            "action_contract": "each round choose 3 legal unsampled actions",
            "success_definition": "direct source hit within budget",
            "seed": int(args.seed),
        },
        "split_meta": split_meta,
        "posterior_family": {
            "selected_family": str(teacher_family),
            "selection_decision": teacher_decision,
        },
        "teacher_policies": list(TEACHER_POLICIES),
        "teacher_policy_params": {
            "source_support_mass_tau": float(args.source_support_mass_tau),
            "cover_tau": float(args.cover_tau),
            "obs_tau_min": float(args.obs_tau_min),
            "obs_eps_fp": float(args.obs_eps_fp),
            "obs_eps_fn": float(args.obs_eps_fn),
            "thompson_num_samples": int(args.thompson_num_samples),
        },
        "results": {
            "leaderboard": leaderboard_df.to_dict(orient="records"),
            "per_policy_summary": per_policy_summary,
        },
        "artifacts": {
            "case_rows": str(output_dir / "teacher5_case_rows.csv"),
            "step_rows": str(output_dir / "teacher5_step_rows.csv"),
            "leaderboard": str(output_dir / "teacher5_leaderboard.csv"),
            "round_curve": str(output_dir / "roundwise_success_curve.csv"),
            "budget_curve": str(output_dir / "budget_success_curve.csv"),
            "case_level_compare": str(output_dir / "case_level_compare.csv"),
            "pairwise_complementarity": str(output_dir / "pairwise_complementarity.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[done] wrote artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
