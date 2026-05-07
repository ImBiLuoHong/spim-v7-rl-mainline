from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "belief_reward_validity_audit" / "20260409_winner_validity_audit_v1"
RUNNER_VERSION = "belief_reward_validity_audit_v1"
PANEL_VERSION = "exact136_train_only_belief_topk_greedy_b60_validity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validity audit for the current winner belief/reward bridge.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--actions-per-round", type=int, default=3)
    parser.add_argument("--winner-logit-weight", type=float, default=0.5)
    parser.add_argument("--weak-logit-weight", type=float, default=0.1)
    parser.add_argument("--progress-every-cases", type=int, default=10)
    parser.add_argument("--case-subset-csv", type=str, default="")
    parser.add_argument("--families", type=str, default="")
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
    if bool(mask.any()):
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


def _belief_metrics(probs: torch.Tensor, mask: torch.Tensor, source_local: Optional[int], threshold: float = 0.7) -> Dict[str, Any]:
    probs = _normalize_distribution(probs, mask)
    mask = mask.view(-1).bool()
    valid = probs[mask].clamp_min(1e-12)
    entropy = float((-(valid * torch.log(valid))).sum().item()) if valid.numel() else 0.0
    effective_support = float(math.exp(entropy))
    order = torch.argsort(probs, descending=True)
    order = order[mask[order]]
    ordered_vals = probs[order]
    csum = torch.cumsum(ordered_vals, dim=0) if ordered_vals.numel() else torch.tensor([], dtype=torch.float32)
    hits = (csum >= float(threshold)).nonzero(as_tuple=True)[0] if ordered_vals.numel() else torch.tensor([], dtype=torch.long)
    cover_idx = int(hits[0].item()) + 1 if hits.numel() else int(ordered_vals.numel())
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
        "belief": probs,
        "ordered_candidates": [int(v) for v in order.tolist()],
        "entropy": entropy,
        "effective_support": effective_support,
        "top1_mass": float(ordered_vals[:1].sum().item()) if ordered_vals.numel() else 0.0,
        "top3_mass": float(ordered_vals[: min(3, ordered_vals.numel())].sum().item()) if ordered_vals.numel() else 0.0,
        "top5_mass": float(ordered_vals[: min(5, ordered_vals.numel())].sum().item()) if ordered_vals.numel() else 0.0,
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


def _calibrated_fused_probs(components: Dict[str, torch.Tensor], params: Dict[str, float]) -> torch.Tensor:
    mask = components["candidate_mask"]
    q_z = _masked_zscore(components["q_score"], mask)
    l_z = _masked_zscore(components["reasoner_logits"], mask)
    c_z = _masked_zscore(components["contrast_signal"], mask)
    d_z = _masked_zscore(components["contradiction_score"], mask)
    energy = (
        float(params["lambda_reasoner"]) * l_z
        + float(params["lambda_q"]) * q_z
        + float(params["lambda_contrast"]) * c_z
        - float(params["lambda_contradiction"]) * d_z
    )
    return _safe_softmax(energy, mask, float(params["temperature"]))


def _logits_only_probs(components: Dict[str, torch.Tensor], logits_temp: float) -> torch.Tensor:
    return _safe_softmax(components["reasoner_logits"], components["candidate_mask"], logits_temp)


def _fuse_probs(base: torch.Tensor, extra: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    mask = mask.view(-1).bool()
    out = torch.zeros_like(base.view(-1).float())
    eps = 1e-12
    if bool(mask.any()):
        out[mask] = torch.softmax(
            torch.log(base[mask].clamp_min(eps)) + float(weight) * torch.log(extra[mask].clamp_min(eps)),
            dim=0,
        )
    return out


def _shuffle_prior(prior: torch.Tensor, mask: torch.Tensor, case_id: str, episode_index: int, seed: int) -> torch.Tensor:
    prior = prior.view(-1).float().clone()
    mask = mask.view(-1).bool()
    out = torch.zeros_like(prior)
    if not bool(mask.any()):
        return out
    idx = torch.nonzero(mask, as_tuple=True)[0]
    digest = hashlib.sha256(f"{case_id}:{episode_index}:{seed}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:8], 16)
    rng = random.Random(local_seed)
    order = idx.tolist()
    rng.shuffle(order)
    shuffled_vals = prior[torch.tensor(order, dtype=torch.long)]
    out[idx] = shuffled_vals
    return _normalize_distribution(out, mask)


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


def _reward_truth_stats(df: pd.DataFrame, reward_col: str, label: str) -> Dict[str, Any]:
    reward_series = df[reward_col]
    if isinstance(reward_series, pd.DataFrame):
        reward_series = reward_series.iloc[:, 0]
    sub = pd.DataFrame(
        {
            "reward": reward_series,
            "delta_true_mass": df["delta_true_mass"],
            "source_hit_in_round": df["source_hit_in_round"],
            "final_success": df["final_success"],
        }
    ).dropna().copy()
    if sub.empty:
        return {"reward_name": label}
    q_hi = float(sub["reward"].quantile(0.9))
    q_lo = float(sub["reward"].quantile(0.1))
    hi = sub[sub["reward"] >= q_hi].copy()
    lo = sub[sub["reward"] <= q_lo].copy()
    return {
        "reward_name": label,
        "mean": float(sub["reward"].mean()),
        "std": float(sub["reward"].std()),
        "snr": float(sub["reward"].mean() / max(sub["reward"].std(), 1e-9)),
        "corr_delta_true_mass": float(sub[["reward", "delta_true_mass"]].corr(method="spearman").iloc[0, 1]) if len(sub) > 2 else None,
        "corr_final_success": float(sub[["reward", "final_success"]].corr(method="spearman").iloc[0, 1]) if len(sub) > 2 else None,
        "corr_source_hit_in_round": float(sub[["reward", "source_hit_in_round"]].corr(method="spearman").iloc[0, 1]) if len(sub) > 2 else None,
        "top_decile_size": int(len(hi)),
        "top_decile_true_mass_up_rate": float((hi["delta_true_mass"] > 0).mean()) if len(hi) else None,
        "top_decile_true_mass_down_rate": float((hi["delta_true_mass"] < 0).mean()) if len(hi) else None,
        "top_decile_hit_rate": float((hi["source_hit_in_round"] > 0.5).mean()) if len(hi) else None,
        "bottom_decile_size": int(len(lo)),
        "bottom_decile_true_mass_down_rate": float((lo["delta_true_mass"] < 0).mean()) if len(lo) else None,
        "bottom_decile_true_mass_up_rate": float((lo["delta_true_mass"] > 0).mean()) if len(lo) else None,
        "wrong_confidence_count": int(((sub["reward"] > 0) & (sub["delta_true_mass"] < 0)).sum()),
        "wrong_confidence_rate": float(((sub["reward"] > 0) & (sub["delta_true_mass"] < 0)).mean()),
    }


def _extract_source_global(rollout: PracticalRollout) -> Optional[int]:
    source_local = resolve_source_local_idx(rollout)
    if source_local is None:
        return None
    return int(rollout.g_ids[int(source_local)].item())


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    runtime["num_episodes"] = int(args.num_rounds)
    runtime["action_budget"] = int(args.actions_per_round)
    if str(args.case_subset_csv).strip():
        subset_df = pd.read_csv(Path(args.case_subset_csv))
        keep = set(subset_df["case_id"].astype(str).tolist())
        runtime["cases"] = [case for case in runtime["cases"] if str(case.case_id) in keep]
    calibrated_params, logits_temp = _load_calibrated_fused(acceptability_root)
    _, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)

    provenance = {
        "winner_family": "pure_plus_logits_bayes",
        "winner_definition": "pure_likelihood_bayes posterior multiplied by logits-only prior",
        "reasoner_source_root": str(source_root),
        "oracle_arm_manifest": str(source_root / "arm_b_task_defined_oracle" / "run_manifest.json"),
        "same_case_summary": str(source_root / "summary.json"),
        "same_case_manifest": str(source_root / "same_case_manifest.json"),
        "same_case_replayable_manifest": str(source_root / "same_case_replayable_manifest.json"),
        "frozen_reasoner_checkpoint": str(frozen_checkpoint),
        "frozen_reasoner_loader_path": str(PROJECT_ROOT / "src" / "scripts" / "run_posterior_like_belief_audit.py"),
        "train_only_panel": True,
        "same_case_train_only_risk": True,
        "winner_logit_weight": float(args.winner_logit_weight),
        "weak_logit_weight": float(args.weak_logit_weight),
    }
    write_json(output_dir / "provenance_audit.json", provenance)

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    if str(args.families).strip():
        families = [part.strip() for part in str(args.families).split(",") if part.strip()]
    else:
        families = [
            "pure_likelihood_bayes",
            "pure_plus_logits_bayes",
            "pure_plus_logits_bayes_weakprior",
            "pure_plus_logits_bayes_shuffledprior",
            "calibrated_fused_posterior",
        ]
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    for family in families:
        for case_idx, case in enumerate(runtime["cases"], start=1):
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
            source_local = resolve_source_local_idx(rollout)
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
                pure_probs = _normalize_distribution(pure_ctx["belief"].cpu(), components["candidate_mask"])
                logits_probs = _logits_only_probs(components, logits_temp)

                if family == "pure_likelihood_bayes":
                    pre_probs = pure_probs
                elif family == "calibrated_fused_posterior":
                    pre_probs = _calibrated_fused_probs(components, calibrated_params)
                elif family == "pure_plus_logits_bayes":
                    pre_probs = _fuse_probs(pure_probs, logits_probs, components["candidate_mask"], float(args.winner_logit_weight))
                elif family == "pure_plus_logits_bayes_weakprior":
                    pre_probs = _fuse_probs(pure_probs, logits_probs, components["candidate_mask"], float(args.weak_logit_weight))
                elif family == "pure_plus_logits_bayes_shuffledprior":
                    shuffled = _shuffle_prior(logits_probs, components["candidate_mask"], case.case_id, int(episode_idx), int(args.seed))
                    pre_probs = _fuse_probs(pure_probs, shuffled, components["candidate_mask"], float(args.winner_logit_weight))
                else:
                    raise ValueError(family)

                pre_metrics = _belief_metrics(pre_probs, components["candidate_mask"], source_local)
                selected_local = _pick_topk_unsampled(pre_probs, components["candidate_mask"], rollout, int(runtime["action_budget"]))
                if not selected_local:
                    break
                selected_global = [int(rollout.g_ids[int(idx)].item()) for idx in selected_local]
                round_hit = source_global is not None and int(source_global) in set(selected_global)
                rollout.step_with_actions(selected_local, sample_types=[f"{family}_slot_{i}" for i in range(len(selected_local))])
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
                post_pure_probs = _normalize_distribution(post_pure_ctx["belief"].cpu(), post_components["candidate_mask"])
                post_logits_probs = _logits_only_probs(post_components, logits_temp)
                if family == "pure_likelihood_bayes":
                    post_probs = post_pure_probs
                elif family == "calibrated_fused_posterior":
                    post_probs = _calibrated_fused_probs(post_components, calibrated_params)
                elif family == "pure_plus_logits_bayes":
                    post_probs = _fuse_probs(post_pure_probs, post_logits_probs, post_components["candidate_mask"], float(args.winner_logit_weight))
                elif family == "pure_plus_logits_bayes_weakprior":
                    post_probs = _fuse_probs(post_pure_probs, post_logits_probs, post_components["candidate_mask"], float(args.weak_logit_weight))
                else:
                    post_shuffled = _shuffle_prior(post_logits_probs, post_components["candidate_mask"], case.case_id, int(episode_idx) + 1000, int(args.seed))
                    post_probs = _fuse_probs(post_pure_probs, post_shuffled, post_components["candidate_mask"], float(args.winner_logit_weight))
                post_metrics = _belief_metrics(post_probs, post_components["candidate_mask"], post_source_local)

                if round_hit and hit_round is None:
                    hit_round = int(episode_idx)
                sampled_node_mass = float(sum(float(pre_probs[int(idx)].item()) for idx in selected_local))
                step_rows.append(
                    {
                        "family": family,
                        "case_id": case.case_id,
                        "episode_index": int(episode_idx),
                        "selected_global_ids": json.dumps(selected_global),
                        "selected_local_ids": json.dumps(selected_local),
                        "sampled_node_posterior_mass_sum": sampled_node_mass,
                        "pre_entropy": float(pre_metrics["entropy"]),
                        "post_entropy": float(post_metrics["entropy"]),
                        "delta_entropy": float(pre_metrics["entropy"] - post_metrics["entropy"]),
                        "pre_top1_mass": float(pre_metrics["top1_mass"]),
                        "post_top1_mass": float(post_metrics["top1_mass"]),
                        "delta_top1_mass": float(post_metrics["top1_mass"] - pre_metrics["top1_mass"]),
                        "pre_effective_support": float(pre_metrics["effective_support"]),
                        "post_effective_support": float(post_metrics["effective_support"]),
                        "delta_effective_support_shrink": float(pre_metrics["effective_support"] - post_metrics["effective_support"]),
                        "pre_mass_cover_size_ratio": float(pre_metrics["mass_cover_size_ratio"]),
                        "post_mass_cover_size_ratio": float(post_metrics["mass_cover_size_ratio"]),
                        "delta_mass_cover_shrink": float(pre_metrics["mass_cover_size_ratio"] - post_metrics["mass_cover_size_ratio"]),
                        "pre_true_mass": pre_metrics["true_mass"],
                        "post_true_mass": post_metrics["true_mass"],
                        "delta_true_mass": (
                            float(post_metrics["true_mass"] - pre_metrics["true_mass"])
                            if pre_metrics["true_mass"] is not None and post_metrics["true_mass"] is not None
                            else None
                        ),
                        "pre_margin_true_vs_hard": pre_metrics["margin_true_vs_hard"],
                        "post_margin_true_vs_hard": post_metrics["margin_true_vs_hard"],
                        "delta_margin": (
                            float(post_metrics["margin_true_vs_hard"] - pre_metrics["margin_true_vs_hard"])
                            if pre_metrics["margin_true_vs_hard"] is not None and post_metrics["margin_true_vs_hard"] is not None
                            else None
                        ),
                        "source_hit_in_round": float(bool(round_hit)),
                        "source_rank_post": post_metrics["true_rank"],
                        "budget_used": float(rollout.revealed_mask.sum().item()),
                    }
                )

            final_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=runtime["num_episodes"],
                action_budget=runtime["action_budget"],
                frontier_role_mode=runtime["frontier_role_mode"],
            )
            final_source_local = resolve_source_local_idx(rollout)
            final_components = _compute_components(final_state, reasoner_module, device)
            final_pure_step_in = {
                "t_sim": torch.tensor([float(rollout.current_time_min)], dtype=torch.float32),
                "valid_mask": final_state["valid_mask"].view(-1).bool(),
                "constraint_state": final_state["constraint_state"],
                "physics_ctx": final_state["phys_ctx"],
                "history": history,
                "global_node_ids": rollout.g_ids.view(-1).long(),
                "topology": topology,
                "episode_duration_min": float(runtime["episode_duration_min"]),
                "source_local": final_source_local,
            }
            pure_state, final_pure_ctx = pure_updater.step(pure_state, final_pure_step_in)
            final_pure_probs = _normalize_distribution(final_pure_ctx["belief"].cpu(), final_components["candidate_mask"])
            final_logits_probs = _logits_only_probs(final_components, logits_temp)
            if family == "pure_likelihood_bayes":
                final_probs = final_pure_probs
            elif family == "calibrated_fused_posterior":
                final_probs = _calibrated_fused_probs(final_components, calibrated_params)
            elif family == "pure_plus_logits_bayes":
                final_probs = _fuse_probs(final_pure_probs, final_logits_probs, final_components["candidate_mask"], float(args.winner_logit_weight))
            elif family == "pure_plus_logits_bayes_weakprior":
                final_probs = _fuse_probs(final_pure_probs, final_logits_probs, final_components["candidate_mask"], float(args.weak_logit_weight))
            else:
                final_shuffled = _shuffle_prior(final_logits_probs, final_components["candidate_mask"], case.case_id, 9999, int(args.seed))
                final_probs = _fuse_probs(final_pure_probs, final_shuffled, final_components["candidate_mask"], float(args.winner_logit_weight))
            final_metrics = _belief_metrics(final_probs, final_components["candidate_mask"], final_source_local)
            case_rows.append(
                {
                    "family": family,
                    "case_id": case.case_id,
                    "success_rate": float(hit_round is not None),
                    "hit_round": hit_round,
                    "final_rank": final_metrics["true_rank"],
                    "final_top1_hit": float((final_metrics["true_rank"] or 10**9) <= 1),
                    "final_top3_hit": float((final_metrics["true_rank"] or 10**9) <= 3),
                    "final_top5_hit": float((final_metrics["true_rank"] or 10**9) <= 5),
                    "final_mrr": float(1.0 / final_metrics["true_rank"]) if final_metrics["true_rank"] is not None else 0.0,
                    "final_true_mass": final_metrics["true_mass"],
                    "final_entropy": float(final_metrics["entropy"]),
                    "final_mass_cover_size_ratio": float(final_metrics["mass_cover_size_ratio"]),
                    "budget_used": float(rollout.revealed_mask.sum().item()),
                }
            )

            if case_idx % int(args.progress_every_cases) == 0 or case_idx == len(runtime["cases"]):
                pd.DataFrame(case_rows).to_csv(output_dir / "case_rows.partial.csv", index=False)
                pd.DataFrame(step_rows).to_csv(output_dir / "step_rows.partial.csv", index=False)
                print(f"[audit-progress] family={family} case={case_idx}/{len(runtime['cases'])}", flush=True)

    case_df = pd.DataFrame(case_rows)
    step_df = pd.DataFrame(step_rows)
    case_df.to_csv(output_dir / "case_rows.csv", index=False)
    step_df.to_csv(output_dir / "step_rows.csv", index=False)

    compare_rows = []
    for family, sub in case_df.groupby("family"):
        compare_rows.append(
            {
                "family": family,
                "case_count": int(len(sub)),
                "success_rate": float(sub["success_rate"].mean()),
                "hit_round_mean": float(sub["hit_round"].dropna().mean()) if sub["hit_round"].notna().any() else None,
                "budget_used_mean": float(sub["budget_used"].mean()),
                "final_mrr": float(sub["final_mrr"].mean()),
                "final_top5_hit": float(sub["final_top5_hit"].mean()),
                "final_true_mass_mean": float(sub["final_true_mass"].dropna().mean()) if sub["final_true_mass"].notna().any() else None,
                "final_entropy_mean": float(sub["final_entropy"].mean()),
                "final_mass_cover_size_ratio_mean": float(sub["final_mass_cover_size_ratio"].mean()),
            }
        )
    compare_df = pd.DataFrame(compare_rows).sort_values(["success_rate", "final_mrr"], ascending=[False, False])
    compare_df.to_csv(output_dir / "greedy_compare.csv", index=False)

    merged_success = compare_df.set_index("family")["success_rate"].to_dict()
    step_df["final_success"] = step_df["family"].map(lambda f: None)
    final_success_map = {(row["family"], row["case_id"]): float(row["success_rate"]) for _, row in case_df.iterrows()}
    step_df["final_success"] = [final_success_map[(fam, cid)] for fam, cid in zip(step_df["family"], step_df["case_id"])]

    reward_rows = []
    for family, sub in step_df.groupby("family"):
        reward_rows.append(_reward_truth_stats(sub, "delta_mass_cover_shrink", f"{family}:delta_mass_cover_shrink"))
        reward_rows.append(_reward_truth_stats(sub, "delta_entropy", f"{family}:delta_entropy"))
        reward_rows.append(_reward_truth_stats(sub, "sampled_node_posterior_mass_sum", f"{family}:sampled_node_posterior_mass_sum"))
        reward_rows.append(_reward_truth_stats(sub, "source_hit_in_round", f"{family}:terminal_source_hit"))
        combo = sub.copy()
        mc = combo["delta_mass_cover_shrink"]
        z = (mc - mc.mean()) / max(mc.std(), 1e-9)
        combo["reward_terminal_plus_mass_cover"] = combo["source_hit_in_round"] + 0.25 * z
        reward_rows.append(_reward_truth_stats(combo, "reward_terminal_plus_mass_cover", f"{family}:terminal_plus_mass_cover"))
    reward_df = pd.DataFrame(reward_rows)
    reward_df.to_csv(output_dir / "reward_truth_alignment.csv", index=False)

    calibrated = compare_df[compare_df["family"] == "calibrated_fused_posterior"].iloc[0].to_dict()
    write_json(
        output_dir / "summary.json",
        {
            "runner_version": RUNNER_VERSION,
            "panel_version": PANEL_VERSION,
            "seed": int(args.seed),
            "cache_version": cache_dir.name,
            "source_root": str(source_root),
            "reasoner_checkpoint": str(frozen_checkpoint),
            "winner_logit_weight": float(args.winner_logit_weight),
            "weak_logit_weight": float(args.weak_logit_weight),
            "families": families,
            "compare_path": str(output_dir / "greedy_compare.csv"),
            "reward_truth_path": str(output_dir / "reward_truth_alignment.csv"),
            "provenance_path": str(output_dir / "provenance_audit.json"),
            "calibrated_fused_replicated_success_rate": calibrated["success_rate"],
        },
    )


if __name__ == "__main__":
    main()
