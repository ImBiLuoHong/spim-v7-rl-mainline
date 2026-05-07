from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.belief_updaters.evidence_posterior_like import _evidence_contrast_scalar, _masked_zscore
from src.modeling.belief_updaters.pure_likelihood_bayes import PureLikelihoodBayesBelief
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.modeling.loop.navigator_vnext_contract import build_candidate_semantics, default_reward_contract, tensor_attr
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_acceptability_audit import DEFAULT_CACHE_DIR, DEFAULT_CONTRAST_ROOT, DEFAULT_SOURCE_ROOT
from src.scripts.run_posterior_like_belief_audit import load_frozen_reasoner, load_runtime_context, write_json
from src.scripts.run_reasoner_same_case_stronger_source_overfit import TempGraph, build_state_input, make_rollout_state, move_payload


DEFAULT_ACCEPTABILITY_ROOT = PROJECT_ROOT / "artifacts" / "posterior_like_belief_acceptability_audit" / "20260407_exact136_belief_acceptability_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "bayesian_belief_reward_sweep" / "20260409_belief_reward_sweep_v1" / "stage3_greedy"
RUNNER_VERSION = "belief_topk_greedy_compare_v1"
PANEL_VERSION = "exact136_train_only_belief_topk_greedy_b60_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared top-k greedy control compare for belief families.")
    parser.add_argument("--family", type=str, required=True, choices=["calibrated_fused_posterior", "pure_likelihood_bayes", "pure_plus_logits_bayes"])
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--actions-per-round", type=int, default=3)
    parser.add_argument("--fuse-weight-logits", type=float, default=0.5)
    parser.add_argument("--prior-mode", type=str, default="normal", choices=["normal", "shuffled", "uniform"])
    return parser.parse_args()


def _contract_cfg() -> Dict[str, float]:
    cfg = default_reward_contract()
    cfg.update({"support_plausible_delta": 0.25, "not_ruled_out_threshold": 0.5})
    return cfg


def _load_calibrated_fused(acceptability_root: Path) -> tuple[Dict[str, float], float]:
    payload = json.loads((acceptability_root / "summary.json").read_text())
    return dict(payload["head_definitions"]["calibrated_fused_posterior"]), float(payload["head_definitions"]["logits_only_posterior"]["temperature"])


def _safe_softmax(scores: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    out = torch.zeros_like(scores.view(-1).float())
    mask = mask.view(-1).bool()
    if not bool(mask.any()):
        return out
    out[mask] = torch.softmax(scores[mask] / max(float(temperature), 1e-6), dim=0)
    return out


def _normalize_distribution(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(probs.view(-1).float())
    mask = mask.view(-1).bool()
    vals = probs.view(-1).float().clone()
    vals[~mask] = 0.0
    denom = float(vals.sum().item())
    if denom <= 1e-12:
        out[mask] = 1.0 / max(int(mask.sum().item()), 1)
        return out
    out[mask] = vals[mask] / denom
    return out


def _belief_metrics(probs: torch.Tensor, mask: torch.Tensor, source_local: int | None, threshold: float = 0.7) -> Dict[str, Any]:
    probs = _normalize_distribution(probs, mask)
    mask = mask.view(-1).bool()
    valid = probs[mask].clamp_min(1e-12)
    entropy = float((-(valid * torch.log(valid))).sum().item()) if valid.numel() > 0 else 0.0
    eff_support = float(math.exp(entropy))
    order = torch.argsort(probs, descending=True)
    order = order[mask[order]]
    ordered_vals = probs[order]
    csum = torch.cumsum(ordered_vals, dim=0) if ordered_vals.numel() > 0 else torch.tensor([], dtype=torch.float32)
    hits = (csum >= float(threshold)).nonzero(as_tuple=True)[0] if ordered_vals.numel() > 0 else torch.tensor([], dtype=torch.long)
    cover_idx = int(hits[0].item()) + 1 if hits.numel() > 0 else int(ordered_vals.numel())
    rank = None
    true_mass = None
    hard_mass = None
    margin = None
    if source_local is not None and bool(mask[int(source_local)].item()):
        pos = (order == int(source_local)).nonzero(as_tuple=True)[0]
        if pos.numel() > 0:
            rank = int(pos[0].item()) + 1
            true_mass = float(probs[int(source_local)].item())
            for idx in order.tolist():
                if int(idx) != int(source_local):
                    hard_mass = float(probs[int(idx)].item())
                    break
            if hard_mass is not None:
                margin = float(true_mass - hard_mass)
    return {
        "probs": probs,
        "ordered_candidates": [int(v) for v in order.tolist()],
        "entropy": entropy,
        "effective_support": eff_support,
        "top1_mass": float(ordered_vals[:1].sum().item()) if ordered_vals.numel() > 0 else 0.0,
        "top3_mass": float(ordered_vals[: min(3, ordered_vals.numel())].sum().item()) if ordered_vals.numel() > 0 else 0.0,
        "top5_mass": float(ordered_vals[: min(5, ordered_vals.numel())].sum().item()) if ordered_vals.numel() > 0 else 0.0,
        "mass_cover_size_ratio": float(cover_idx / max(int(mask.sum().item()), 1)),
        "mass_cover_size": int(cover_idx),
        "true_rank": rank,
        "true_mass": true_mass,
        "hard_mass": hard_mass,
        "margin_true_vs_hard": margin,
    }


def _compute_components(state: Dict[str, Any], reasoner_module, device: torch.device) -> Dict[str, torch.Tensor]:
    valid_mask = state["valid_mask"].view(-1).bool().cpu()
    graph = TempGraph(state["edge_index"], int(valid_mask.numel()), device)
    state_input = move_payload(build_state_input(state), device)
    physics_ctx = move_payload(state["phys_ctx"].__dict__, device)
    with torch.no_grad():
        out = reasoner_module(state_input, graph, physics_ctx=physics_ctx)
    reasoner_logits = out["logits"].detach().float().view(-1).cpu()
    payload = build_clean_aligned_feature_payload(
        build_state_input(state),
        batch_index=torch.zeros(int(valid_mask.numel()), dtype=torch.long),
        edge_index=state["edge_index"].view(2, -1).long(),
        physics_ctx=state["phys_ctx"].__dict__,
        frontier_mode="unresolved_without_pair",
    )
    semantics = build_candidate_semantics(
        evidence_state=state["evidence_state"],
        constraint_state=state["constraint_state"],
        valid_mask=valid_mask,
        batch=torch.zeros(int(valid_mask.numel()), dtype=torch.long),
        contract_cfg=_contract_cfg(),
    )
    candidate_mask = semantics["candidate_mask"].view(-1).bool().cpu()
    if not bool(candidate_mask.any()):
        candidate_mask = valid_mask.clone()
    q_score = semantics["q_score"].view(-1).float().cpu()
    contradiction = tensor_attr(state["evidence_state"], "contradiction_score", valid_mask.float(), default=0.0).cpu()
    contrast_signal = _evidence_contrast_scalar(payload["node_features"].cpu(), valid_mask)
    return {
        "valid_mask": valid_mask,
        "candidate_mask": candidate_mask,
        "reasoner_logits": reasoner_logits,
        "q_score": q_score,
        "contradiction_score": contradiction,
        "contrast_signal": contrast_signal,
    }


def _family_probs(
    *,
    family: str,
    components: Dict[str, torch.Tensor],
    pure_ctx: Dict[str, Any],
    calibrated_params: Dict[str, float],
    logits_temp: float,
    fuse_weight_logits: float,
    prior_mode: str,
) -> torch.Tensor:
    mask = components["candidate_mask"]
    if family == "calibrated_fused_posterior":
        q_z = _masked_zscore(components["q_score"], mask)
        l_z = _masked_zscore(components["reasoner_logits"], mask)
        c_z = _masked_zscore(components["contrast_signal"], mask)
        d_z = _masked_zscore(components["contradiction_score"], mask)
        energy = (
            float(calibrated_params["lambda_reasoner"]) * l_z
            + float(calibrated_params["lambda_q"]) * q_z
            + float(calibrated_params["lambda_contrast"]) * c_z
            - float(calibrated_params["lambda_contradiction"]) * d_z
        )
        return _safe_softmax(energy, mask, float(calibrated_params["temperature"]))
    if family == "pure_likelihood_bayes":
        return _normalize_distribution(pure_ctx["belief"].cpu(), mask)
    if family == "pure_plus_logits_bayes":
        pure_probs = _normalize_distribution(pure_ctx["belief"].cpu(), mask)
        logits_probs = _safe_softmax(components["reasoner_logits"], mask, logits_temp)
        if str(prior_mode) == "shuffled":
            valid_idx = torch.nonzero(mask, as_tuple=True)[0]
            shuffled = logits_probs.clone()
            if valid_idx.numel() > 1:
                shuffled[valid_idx] = logits_probs[valid_idx].flip(0)
            logits_probs = _normalize_distribution(shuffled, mask)
        elif str(prior_mode) == "uniform":
            logits_probs = _normalize_distribution(torch.ones_like(logits_probs), mask)
        fused = torch.zeros_like(pure_probs)
        eps = 1e-12
        fused[mask] = torch.softmax(torch.log(pure_probs[mask].clamp_min(eps)) + float(fuse_weight_logits) * torch.log(logits_probs[mask].clamp_min(eps)), dim=0)
        return fused
    raise ValueError(f"unsupported family: {family}")


def _pick_topk_unsampled(probs: torch.Tensor, mask: torch.Tensor, rollout: PracticalRollout, k: int) -> List[int]:
    order = torch.argsort(probs.view(-1), descending=True)
    chosen: List[int] = []
    for idx in order.tolist():
        if not bool(mask[int(idx)].item()):
            continue
        if bool(rollout.revealed_mask[int(idx)].item()):
            continue
        chosen.append(int(idx))
        if len(chosen) >= int(k):
            break
    return chosen


def _extract_source_global(rollout: PracticalRollout) -> Optional[int]:
    source_local = resolve_source_local_idx(rollout)
    if source_local is None:
        return None
    return int(rollout.g_ids[int(source_local)].item())


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)
    output_dir = Path(args.output_root) / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    runtime["num_episodes"] = int(args.num_rounds)
    runtime["action_budget"] = int(args.actions_per_round)
    calibrated_params, logits_temp = _load_calibrated_fused(acceptability_root)
    _, _, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    for case in runtime["cases"]:
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
        pure_updater = PureLikelihoodBayesBelief()
        pure_state = pure_updater.init_state(batch_size=1, num_nodes=int(rollout.num_nodes), device=torch.device("cpu"))
        source_global = _extract_source_global(rollout)
        hit_round = None

        for episode_idx in range(1, int(runtime["num_episodes"]) + 1):
            pre_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=runtime["num_episodes"],
                action_budget=runtime["action_budget"],
                frontier_role_mode=runtime["frontier_role_mode"],
            )
            if int(pre_state["valid_mask"].sum().item()) <= 0:
                break
            source_local = resolve_source_local_idx(rollout)
            components = _compute_components(pre_state, reasoner_module, device)
            pure_step_in = {
                "t_sim": torch.tensor([float(rollout.current_time_min)], dtype=torch.float32),
                "valid_mask": pre_state["valid_mask"].view(-1).bool(),
                "constraint_state": pre_state["constraint_state"],
                "physics_ctx": pre_state["phys_ctx"],
                "history": history,
                "global_node_ids": rollout.g_ids.view(-1).long(),
                "topology": topology,
                "episode_duration_min": float(runtime["episode_duration_min"]),
                "source_local": source_local,
            }
            pure_state, pure_ctx = pure_updater.step(pure_state, pure_step_in)
            pre_probs = _family_probs(
                family=args.family,
                components=components,
                pure_ctx=pure_ctx,
                calibrated_params=calibrated_params,
                logits_temp=logits_temp,
                fuse_weight_logits=float(args.fuse_weight_logits),
                prior_mode=str(args.prior_mode),
            )
            pre_metrics = _belief_metrics(pre_probs, components["candidate_mask"], source_local)
            action_local = _pick_topk_unsampled(pre_probs, components["candidate_mask"], rollout, int(runtime["action_budget"]))
            if not action_local:
                break
            action_global = [int(rollout.g_ids[int(idx)].item()) for idx in action_local]
            round_hit = source_global is not None and int(source_global) in set(action_global)
            rollout.step_with_actions(action_local, sample_types=[f"{args.family}_slot_{i}" for i in range(len(action_local))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            post_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=runtime["num_episodes"],
                action_budget=runtime["action_budget"],
                frontier_role_mode=runtime["frontier_role_mode"],
            )
            post_source_local = resolve_source_local_idx(rollout)
            post_components = _compute_components(post_state, reasoner_module, device)
            post_pure_step_in = {
                "t_sim": torch.tensor([float(rollout.current_time_min)], dtype=torch.float32),
                "valid_mask": post_state["valid_mask"].view(-1).bool(),
                "constraint_state": post_state["constraint_state"],
                "physics_ctx": post_state["phys_ctx"],
                "history": history,
                "global_node_ids": rollout.g_ids.view(-1).long(),
                "topology": topology,
                "episode_duration_min": float(runtime["episode_duration_min"]),
                "source_local": post_source_local,
            }
            pure_state, post_pure_ctx = pure_updater.step(pure_state, post_pure_step_in)
            post_probs = _family_probs(
                family=args.family,
                components=post_components,
                pure_ctx=post_pure_ctx,
                calibrated_params=calibrated_params,
                logits_temp=logits_temp,
                fuse_weight_logits=float(args.fuse_weight_logits),
                prior_mode=str(args.prior_mode),
            )
            post_metrics = _belief_metrics(post_probs, post_components["candidate_mask"], post_source_local)

            step_rows.append(
                {
                    "family": args.family,
                    "prior_mode": str(args.prior_mode),
                    "fuse_weight_logits": float(args.fuse_weight_logits),
                    "case_id": case.case_id,
                    "episode_index": int(episode_idx),
                    "selected_global_ids": json.dumps(action_global),
                    "selected_local_ids": json.dumps(action_local),
                    "selected_mass_sum": float(sum(float(pre_probs[int(idx)].item()) for idx in action_local)),
                    "pre_entropy": float(pre_metrics["entropy"]),
                    "post_entropy": float(post_metrics["entropy"]),
                    "delta_entropy": float(pre_metrics["entropy"] - post_metrics["entropy"]),
                    "pre_top1_mass": float(pre_metrics["top1_mass"]),
                    "post_top1_mass": float(post_metrics["top1_mass"]),
                    "delta_top1_mass": float(post_metrics["top1_mass"] - pre_metrics["top1_mass"]),
                    "pre_top3_mass": float(pre_metrics["top3_mass"]),
                    "post_top3_mass": float(post_metrics["top3_mass"]),
                    "delta_top3_mass": float(post_metrics["top3_mass"] - pre_metrics["top3_mass"]),
                    "pre_effective_support": float(pre_metrics["effective_support"]),
                    "post_effective_support": float(post_metrics["effective_support"]),
                    "delta_effective_support_shrink": float(pre_metrics["effective_support"] - post_metrics["effective_support"]),
                    "pre_mass_cover_size_ratio": float(pre_metrics["mass_cover_size_ratio"]),
                    "post_mass_cover_size_ratio": float(post_metrics["mass_cover_size_ratio"]),
                    "delta_mass_cover_shrink": float(pre_metrics["mass_cover_size_ratio"] - post_metrics["mass_cover_size_ratio"]),
                    "pre_true_mass": pre_metrics["true_mass"],
                    "post_true_mass": post_metrics["true_mass"],
                    "delta_true_mass": (float(post_metrics["true_mass"] - pre_metrics["true_mass"]) if pre_metrics["true_mass"] is not None and post_metrics["true_mass"] is not None else None),
                    "pre_margin_true_vs_hard": pre_metrics["margin_true_vs_hard"],
                    "post_margin_true_vs_hard": post_metrics["margin_true_vs_hard"],
                    "delta_margin": (float(post_metrics["margin_true_vs_hard"] - pre_metrics["margin_true_vs_hard"]) if pre_metrics["margin_true_vs_hard"] is not None and post_metrics["margin_true_vs_hard"] is not None else None),
                    "source_hit_in_round": float(round_hit),
                    "source_rank_post": post_metrics["true_rank"],
                    "budget_used": float(rollout.revealed_mask.sum().item()),
                }
            )
            if round_hit and hit_round is None:
                hit_round = int(episode_idx)
                break

        case_rows.append(
            {
                "family": args.family,
                "prior_mode": str(args.prior_mode),
                "fuse_weight_logits": float(args.fuse_weight_logits),
                "case_id": case.case_id,
                "success_rate": float(hit_round is not None),
                "hit_round": hit_round,
                "budget_used": float(rollout.revealed_mask.sum().item()),
            }
        )

    case_df = pd.DataFrame(case_rows)
    step_df = pd.DataFrame(step_rows)
    case_df.to_csv(output_dir / "case_rows.csv", index=False)
    step_df.to_csv(output_dir / "step_rows.csv", index=False)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "family": args.family,
        "prior_mode": str(args.prior_mode),
        "fuse_weight_logits": float(args.fuse_weight_logits),
        "seed": int(args.seed),
        "cache_version": cache_dir.name,
        "evaluated_case_count": int(len(case_df)),
        "success_rate": float(case_df["success_rate"].mean()),
        "hit_round_mean": float(case_df.loc[case_df["success_rate"] > 0.5, "hit_round"].mean()) if (case_df["success_rate"] > 0.5).any() else None,
        "budget_used_mean": float(case_df["budget_used"].mean()),
        "step_delta_entropy_mean": float(step_df["delta_entropy"].mean()) if len(step_df) else None,
        "step_delta_mass_cover_shrink_mean": float(step_df["delta_mass_cover_shrink"].mean()) if len(step_df) else None,
        "step_delta_top1_mass_mean": float(step_df["delta_top1_mass"].mean()) if len(step_df) else None,
        "step_delta_effective_support_shrink_mean": float(step_df["delta_effective_support_shrink"].mean()) if len(step_df) else None,
        "step_source_hit_rate": float(step_df["source_hit_in_round"].mean()) if len(step_df) else None,
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
