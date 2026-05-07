from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

LEGACY_HSR_ROOT = PROJECT_ROOT / "tools" / "legacy" / "src_baselines_archive"
if str(LEGACY_HSR_ROOT) not in sys.path:
    sys.path.append(str(LEGACY_HSR_ROOT))

from hsr_agent import HSRAgent

from src.modeling.belief_updaters.evidence_posterior_like import _evidence_contrast_scalar, _masked_zscore
from src.modeling.belief_updaters.pure_likelihood_bayes import PureLikelihoodBayesBelief
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.modeling.loop.navigator_vnext_contract import build_candidate_semantics, default_reward_contract, tensor_attr
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_authoritative_hsr_baseline import _extract_trigger_global, build_hsr_agent, load_foundation_graph, resolve_foundation_graph_path
from src.scripts.run_posterior_like_belief_acceptability_audit import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONTRAST_ROOT,
    DEFAULT_SOURCE_ROOT,
    fit_logits_temperature,
)
from src.scripts.run_posterior_like_belief_audit import load_frozen_reasoner, load_runtime_context, write_json
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    TempGraph,
    build_state_input,
    make_rollout_state,
    move_payload,
    translate_global_ids,
)


DEFAULT_ACCEPTABILITY_ROOT = PROJECT_ROOT / "artifacts" / "posterior_like_belief_acceptability_audit" / "20260407_exact136_belief_acceptability_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "bayesian_belief_reward_sweep" / "20260409_belief_reward_sweep_v1" / "stage2_screen"
RUNNER_VERSION = "bayesian_belief_family_screen_v1"
PANEL_VERSION = "exact136_train_only_belief_family_screen_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded Bayesian belief-family screen on exact136 replayable histories.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--fuse-weight-logits", type=float, default=0.5)
    parser.add_argument("--fuse-weight-hsr", type=float, default=0.5)
    parser.add_argument("--mass-cover-threshold", type=float, default=0.7)
    return parser.parse_args()


def _contract_cfg() -> Dict[str, float]:
    cfg = default_reward_contract()
    cfg.update({"support_plausible_delta": 0.25, "not_ruled_out_threshold": 0.5})
    return cfg


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


def _rank_metrics(scores: torch.Tensor, mask: torch.Tensor, source_local: int | None) -> Dict[str, Any]:
    scores = scores.view(-1).float()
    mask = mask.view(-1).bool()
    if source_local is None or not bool(mask[int(source_local)].item()):
        return {"rank": None, "top1_hit": 0.0, "top3_hit": 0.0, "top5_hit": 0.0, "mrr": 0.0, "true_mass": None, "hard_mass": None, "margin": None}
    safe = scores.clone()
    safe[~mask] = -float("inf")
    order = torch.argsort(safe, descending=True)
    order = order[torch.isfinite(safe[order])]
    pos = (order == int(source_local)).nonzero(as_tuple=True)[0]
    if pos.numel() <= 0:
        return {"rank": None, "top1_hit": 0.0, "top3_hit": 0.0, "top5_hit": 0.0, "mrr": 0.0, "true_mass": None, "hard_mass": None, "margin": None}
    rank = int(pos.min().item()) + 1
    hard_mass = None
    for idx in order.tolist():
        if int(idx) != int(source_local):
            hard_mass = float(safe[int(idx)].item())
            break
    true_mass = float(safe[int(source_local)].item())
    return {
        "rank": int(rank),
        "top1_hit": float(rank <= 1),
        "top3_hit": float(rank <= 3),
        "top5_hit": float(rank <= 5),
        "mrr": float(1.0 / rank),
        "true_mass": true_mass,
        "hard_mass": hard_mass,
        "margin": (float(true_mass - hard_mass) if hard_mass is not None else None),
    }


def _belief_shape_metrics(probs: torch.Tensor, mask: torch.Tensor, threshold: float) -> Dict[str, Any]:
    probs = probs.view(-1).float()
    mask = mask.view(-1).bool()
    if not bool(mask.any()):
        return {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_support": 0.0,
            "top1_mass": 0.0,
            "top3_mass": 0.0,
            "top5_mass": 0.0,
            "mass_cover_size": 0,
            "mass_cover_size_ratio": 0.0,
            "mass_cover_mass": 0.0,
            "ordered_candidates": [],
        }
    vals = probs[mask].clamp_min(1e-12)
    entropy = float((-(vals * torch.log(vals))).sum().item())
    denom = math.log(max(int(vals.numel()), 2))
    order = torch.argsort(probs, descending=True)
    order = order[mask[order]]
    ordered_vals = probs[order]
    csum = torch.cumsum(ordered_vals, dim=0)
    hits = (csum >= float(threshold)).nonzero(as_tuple=True)[0]
    cover_idx = int(hits[0].item()) + 1 if hits.numel() > 0 else int(ordered_vals.numel())
    return {
        "entropy": entropy,
        "normalized_entropy": float(entropy / denom) if denom > 0 else 0.0,
        "effective_support": float(math.exp(entropy)),
        "top1_mass": float(ordered_vals[:1].sum().item()),
        "top3_mass": float(ordered_vals[: min(3, ordered_vals.numel())].sum().item()),
        "top5_mass": float(ordered_vals[: min(5, ordered_vals.numel())].sum().item()),
        "mass_cover_size": int(cover_idx),
        "mass_cover_size_ratio": float(cover_idx / max(int(mask.sum().item()), 1)),
        "mass_cover_mass": float(csum[cover_idx - 1].item()),
        "ordered_candidates": [int(v) for v in order.tolist()],
    }


def _load_calibrated_fused(acceptability_root: Path) -> Tuple[Dict[str, float], float]:
    payload = json.loads((acceptability_root / "summary.json").read_text())
    params = dict(payload["head_definitions"]["calibrated_fused_posterior"])
    logits_temp = float(payload["head_definitions"]["logits_only_posterior"]["temperature"])
    return params, logits_temp


def _compute_reasoner_state(
    *,
    state: Dict[str, Any],
    reasoner_module,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
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


def _calibrated_fused_probs(
    components: Dict[str, torch.Tensor],
    params: Dict[str, float],
) -> torch.Tensor:
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
    return _safe_softmax(components["reasoner_logits"], components["candidate_mask"], float(logits_temp))


def _hsr_vote_probs(agent: HSRAgent, rollout: PracticalRollout) -> torch.Tensor:
    probs = torch.zeros(int(rollout.num_nodes), dtype=torch.float32)
    vote_counts, candidate_globals = agent._compute_hsr_vote_counts()
    global_to_local = {int(g.item()): int(i) for i, g in enumerate(rollout.g_ids.view(-1))}
    total_votes = float(sum(vote_counts.values()))
    if total_votes > 0.0:
        for gid, count in vote_counts.items():
            local = global_to_local.get(int(gid))
            if local is not None:
                probs[int(local)] = float(count) / total_votes
        return probs
    remaining = list(agent.candidate_set - agent.sampled_nodes)
    if remaining:
        mass = 1.0 / float(len(remaining))
        for gid in remaining:
            local = global_to_local.get(int(gid))
            if local is not None:
                probs[int(local)] = mass
    return probs


def _fuse_probs(base: torch.Tensor, extra: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    mask = mask.view(-1).bool()
    out = torch.zeros_like(base.view(-1).float())
    eps = 1e-12
    if not bool(mask.any()):
        return out
    logp = torch.log(base[mask].clamp_min(eps)) + float(weight) * torch.log(extra[mask].clamp_min(eps))
    out[mask] = torch.softmax(logp, dim=0)
    return out


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    contrast_root = Path(args.contrast_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    _, _, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)
    calibrated_params, logits_temp = _load_calibrated_fused(acceptability_root)

    foundation_graph_path = resolve_foundation_graph_path(source_root)
    graph_data = load_foundation_graph(foundation_graph_path)
    pure_updater = PureLikelihoodBayesBelief()

    design_rows = [
        {"family": "logits_only_posterior", "latent_variables": "source", "observation_model": "reasoner logits only", "bayesian_mechanism": "softmax posterior surrogate", "uses_current_assets": "yes", "status": "screen"},
        {"family": "calibrated_fused_posterior", "latent_variables": "source", "observation_model": "reasoner + q + contrast - contradiction", "bayesian_mechanism": "posterior-like calibrated energy over candidate mask", "uses_current_assets": "yes", "status": "screen"},
        {"family": "pure_likelihood_bayes", "latent_variables": "source + onset", "observation_model": "binary observation likelihood from static propagation assets", "bayesian_mechanism": "analytical recursive Bayes", "uses_current_assets": "yes", "status": "screen"},
        {"family": "hsr_vote_posterior", "latent_variables": "source", "observation_model": "binary consistency via HSR Monte Carlo vote", "bayesian_mechanism": "candidate-set-to-posterior conversion", "uses_current_assets": "yes", "status": "screen"},
        {"family": "pure_plus_logits_bayes", "latent_variables": "source + onset", "observation_model": "physical likelihood with logits prior", "bayesian_mechanism": "log-space fusion of Bayes posterior and logits posterior", "uses_current_assets": "yes", "status": "screen"},
        {"family": "pure_plus_hsr_bayes", "latent_variables": "source + onset", "observation_model": "physical likelihood with HSR soft-consistency factor", "bayesian_mechanism": "log-space fusion of Bayes posterior and HSR vote posterior", "uses_current_assets": "yes", "status": "screen"},
        {"family": "scenario_particle_posterior", "latent_variables": "scenario particles", "observation_model": "scenario simulator library", "bayesian_mechanism": "particle/scenario posterior", "uses_current_assets": "no", "status": "drop_unavailable"},
    ]

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
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
        pure_state = pure_updater.init_state(batch_size=1, num_nodes=int(rollout.num_nodes), device=torch.device("cpu"))
        hsr_agent = build_hsr_agent(
            graph_data=graph_data,
            runtime={
                "episode_duration_min": float(runtime["episode_duration_min"]),
            },
        )
        trigger_global = _extract_trigger_global(case.data)
        initial_candidates = [int(v) for v in rollout.g_ids.detach().cpu().tolist()]
        hsr_agent.reset(candidates=initial_candidates, trigger_node=trigger_global, t_start=0)
        if trigger_global is not None:
            hsr_agent.current_time_step = -1
            hsr_agent.step({int(trigger_global): (1.0, 1)})

        prev_metrics: Dict[str, Dict[str, float]] = {}
        for step in runtime["action_plan"].get(case.case_id, []):
            state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=runtime["num_episodes"],
                action_budget=runtime["action_budget"],
                frontier_role_mode=runtime["frontier_role_mode"],
            )
            source_local = resolve_source_local_idx(rollout)
            components = _compute_reasoner_state(state=state, reasoner_module=reasoner_module, device=device)

            pure_step_in = {
                "t_sim": torch.tensor([float(rollout.current_time_min)], dtype=torch.float32),
                "valid_mask": state["valid_mask"].view(-1).bool(),
                "constraint_state": state["constraint_state"],
                "physics_ctx": state["phys_ctx"],
                "history": history,
                "global_node_ids": rollout.g_ids.view(-1).long(),
                "topology": topology,
                "episode_duration_min": float(runtime["episode_duration_min"]),
                "source_local": source_local,
            }
            pure_state, pure_ctx = pure_updater.step(pure_state, pure_step_in)

            logits_probs = _logits_only_probs(components, logits_temp)
            fused_probs = _calibrated_fused_probs(components, calibrated_params)
            pure_probs = _normalize_distribution(pure_ctx["belief"].cpu(), components["candidate_mask"])
            hsr_probs = _normalize_distribution(_hsr_vote_probs(hsr_agent, rollout), components["candidate_mask"])
            pure_logits_probs = _fuse_probs(pure_probs, logits_probs, components["candidate_mask"], float(args.fuse_weight_logits))
            pure_hsr_probs = _fuse_probs(pure_probs, hsr_probs, components["candidate_mask"], float(args.fuse_weight_hsr))

            family_probs = {
                "logits_only_posterior": logits_probs,
                "calibrated_fused_posterior": fused_probs,
                "pure_likelihood_bayes": pure_probs,
                "hsr_vote_posterior": hsr_probs,
                "pure_plus_logits_bayes": pure_logits_probs,
                "pure_plus_hsr_bayes": pure_hsr_probs,
            }

            for family, probs in family_probs.items():
                rank = _rank_metrics(probs, components["candidate_mask"], source_local)
                shape = _belief_shape_metrics(probs, components["candidate_mask"], float(args.mass_cover_threshold))
                prev = prev_metrics.get(family, {})
                row = {
                    "family": family,
                    "case_id": case.case_id,
                    "episode_index": int(step.round_index) + 1,
                    "candidate_count": int(components["candidate_mask"].sum().item()),
                    **rank,
                    **shape,
                    "delta_entropy": (float(prev.get("entropy", shape["entropy"]) - shape["entropy"]) if family in prev_metrics else None),
                    "delta_true_mass": (float(rank["true_mass"] - prev.get("true_mass")) if family in prev_metrics and rank["true_mass"] is not None and prev.get("true_mass") is not None else None),
                    "delta_margin": (float(rank["margin"] - prev.get("margin")) if family in prev_metrics and rank["margin"] is not None and prev.get("margin") is not None else None),
                    "delta_mass_cover_size_ratio": (float(prev.get("mass_cover_size_ratio", shape["mass_cover_size_ratio"]) - shape["mass_cover_size_ratio"]) if family in prev_metrics else None),
                }
                step_rows.append(row)
                prev_metrics[family] = {
                    "entropy": shape["entropy"],
                    "true_mass": rank["true_mass"],
                    "margin": rank["margin"],
                    "mass_cover_size_ratio": shape["mass_cover_size_ratio"],
                }

            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"oracle_slot_{i}" for i in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            latest_step = rollout.history_steps[-1] if rollout.history_steps else None
            observations: Dict[int, tuple[float, int]] = {}
            if latest_step is not None:
                for sample in latest_step.samples:
                    observations[int(sample.global_idx)] = (
                        float(sample.concentration),
                        int(1 if bool(sample.is_positive) else 0),
                    )
            if observations:
                hsr_agent.step(observations)

    step_df = pd.DataFrame(step_rows)
    step_df.to_csv(output_dir / "belief_family_step_rows.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    for family, sub in step_df.groupby("family"):
        summary_rows.append(
            {
                "family": family,
                "rows": int(len(sub)),
                "top1_hit": float(sub["top1_hit"].mean()),
                "top3_hit": float(sub["top3_hit"].mean()),
                "top5_hit": float(sub["top5_hit"].mean()),
                "mrr": float(sub["mrr"].mean()),
                "true_rank_mean": float(sub["rank"].dropna().mean()),
                "median_rank": float(sub["rank"].dropna().median()),
                "entropy_mean": float(sub["entropy"].mean()),
                "normalized_entropy_mean": float(sub["normalized_entropy"].mean()),
                "effective_support_mean": float(sub["effective_support"].mean()),
                "top1_mass_mean": float(sub["top1_mass"].mean()),
                "top3_mass_mean": float(sub["top3_mass"].mean()),
                "top5_mass_mean": float(sub["top5_mass"].mean()),
                "mass_cover_size_ratio_mean": float(sub["mass_cover_size_ratio"].mean()),
                "delta_entropy_mean": float(sub["delta_entropy"].dropna().mean()),
                "delta_entropy_snr": float(sub["delta_entropy"].dropna().mean() / max(sub["delta_entropy"].dropna().std(ddof=0), 1e-9)),
                "delta_true_mass_mean": float(sub["delta_true_mass"].dropna().mean()),
                "delta_true_mass_snr": float(sub["delta_true_mass"].dropna().mean() / max(sub["delta_true_mass"].dropna().std(ddof=0), 1e-9)),
                "delta_margin_mean": float(sub["delta_margin"].dropna().mean()),
                "delta_margin_snr": float(sub["delta_margin"].dropna().mean() / max(sub["delta_margin"].dropna().std(ddof=0), 1e-9)),
                "delta_mass_cover_size_ratio_mean": float(sub["delta_mass_cover_size_ratio"].dropna().mean()),
                "delta_mass_cover_size_ratio_snr": float(sub["delta_mass_cover_size_ratio"].dropna().mean() / max(sub["delta_mass_cover_size_ratio"].dropna().std(ddof=0), 1e-9)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["top5_hit", "mrr", "mass_cover_size_ratio_mean"], ascending=[False, False, True])
    summary_df.to_csv(output_dir / "belief_family_summary.csv", index=False)

    promoted = summary_df.head(3)["family"].tolist()
    design_df = pd.DataFrame(design_rows)
    design_df["screening_result"] = design_df["family"].map(
        {
            row["family"]: (
                "promote" if row["family"] in promoted else "screened_keepdrop"
            )
            for row in design_rows
        }
    )
    design_df.loc[design_df["status"] == "drop_unavailable", "screening_result"] = "drop_unavailable"
    design_df["promoted"] = design_df["family"].isin(promoted)
    design_df.to_csv(output_dir / "design_map.csv", index=False)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "cache_version": cache_dir.name,
        "source_root": str(source_root),
        "foundation_graph_path": str(foundation_graph_path),
        "case_count": int(len(runtime["cases"])),
        "step_rows": int(len(step_df)),
        "families_screened": [str(v) for v in summary_df["family"].tolist()],
        "promoted_families": [str(v) for v in promoted],
        "belief_family_summary_path": str(output_dir / "belief_family_summary.csv"),
        "design_map_path": str(output_dir / "design_map.csv"),
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
