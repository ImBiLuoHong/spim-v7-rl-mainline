from __future__ import annotations

import argparse
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
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
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory, WitnessRecord
from src.modeling.loop.navigator_vnext_contract import build_candidate_semantics, default_reward_contract, tensor_attr
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_fixed_posterior_action_value_audit import _load_calibrated_params
from src.scripts.run_posterior_like_belief_audit import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONTRAST_ROOT,
    DEFAULT_SOURCE_ROOT,
    load_frozen_reasoner,
    load_runtime_context,
    write_json,
)
from src.scripts.run_reasoner_same_case_stronger_source_overfit import (
    TempGraph,
    build_state_input,
    make_rollout_state,
    move_payload,
    translate_global_ids,
)


DEFAULT_ACCEPTABILITY_ROOT = PROJECT_ROOT / "artifacts" / "posterior_like_belief_acceptability_audit" / "20260407_exact136_belief_acceptability_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "bayesian_belief_reward_sweep" / "20260409_belief_reward_sweep_v1"
RUNNER_VERSION = "bayesian_belief_reward_sweep_v1"
PANEL_VERSION = "exact136_train_only_belief_reward_sweep_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded Bayesian belief-family and reward sweep.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--mode", type=str, default="screen", choices=["screen", "greedy"])
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--actions-per-round", type=int, default=3)
    parser.add_argument("--progress-every-cases", type=int, default=10)
    parser.add_argument("--case-subset-csv", type=str, default="")
    return parser.parse_args()


def _contract_cfg() -> Dict[str, float]:
    cfg = default_reward_contract()
    cfg.update({"support_plausible_delta": 0.25, "not_ruled_out_threshold": 0.5})
    return cfg


def _raw_rank(scores: torch.Tensor, valid_mask: torch.Tensor, source_local: Optional[int]) -> Dict[str, Any]:
    valid_mask = valid_mask.view(-1).bool().cpu()
    scores = scores.view(-1).float().cpu()
    if source_local is None or not bool(valid_mask[int(source_local)].item()):
        return {"rank": None, "top1_hit": 0.0, "top3_hit": 0.0, "top5_hit": 0.0, "mrr": 0.0}
    safe_scores = scores.clone()
    safe_scores[~valid_mask] = -float("inf")
    order = torch.argsort(safe_scores, descending=True)
    valid_order = order[torch.isfinite(safe_scores[order])]
    positions = (valid_order == int(source_local)).nonzero(as_tuple=True)[0]
    if positions.numel() <= 0:
        return {"rank": None, "top1_hit": 0.0, "top3_hit": 0.0, "top5_hit": 0.0, "mrr": 0.0}
    rank = int(positions.min().item()) + 1
    return {
        "rank": rank,
        "top1_hit": float(rank <= 1),
        "top3_hit": float(rank <= 3),
        "top5_hit": float(rank <= 5),
        "mrr": float(1.0 / rank),
    }


def _safe_softmax(scores: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = torch.zeros_like(scores.view(-1).float())
    mask = mask.view(-1).bool()
    if bool(mask.any()):
        probs[mask] = torch.softmax(scores[mask] / max(float(temperature), 1e-6), dim=0)
    return probs


def _compute_candidate_mask(state: Dict[str, Any]) -> torch.Tensor:
    semantics = build_candidate_semantics(
        evidence_state=state["evidence_state"],
        constraint_state=state["constraint_state"],
        valid_mask=state["valid_mask"].view(-1).bool(),
        batch=torch.zeros(int(state["valid_mask"].numel()), dtype=torch.long),
        contract_cfg=_contract_cfg(),
    )
    candidate_mask = semantics["candidate_mask"].view(-1).bool().cpu()
    if not bool(candidate_mask.any()):
        candidate_mask = state["valid_mask"].view(-1).bool().cpu()
    return candidate_mask


def _compute_shared_state_primitives(
    *,
    state: Dict[str, Any],
    reasoner_module,
    device: torch.device,
) -> Dict[str, Any]:
    valid_mask = state["valid_mask"].view(-1).bool().cpu()
    graph = TempGraph(state["edge_index"], int(valid_mask.numel()), device)
    state_input = move_payload(build_state_input(state), device)
    physics_ctx = move_payload(state["phys_ctx"].__dict__, device)
    with torch.no_grad():
        out = reasoner_module(state_input, graph, physics_ctx=physics_ctx)
    logits = out["logits"].detach().float().view(-1).cpu()

    payload = build_clean_aligned_feature_payload(
        build_state_input(state),
        batch_index=torch.zeros(int(state["valid_mask"].numel()), dtype=torch.long),
        edge_index=state["edge_index"].view(2, -1).long(),
        physics_ctx=state["phys_ctx"].__dict__,
        frontier_mode="unresolved_without_pair",
    )
    candidate_mask = _compute_candidate_mask(state)
    q_score = build_candidate_semantics(
        evidence_state=state["evidence_state"],
        constraint_state=state["constraint_state"],
        valid_mask=state["valid_mask"].view(-1).bool(),
        batch=torch.zeros(int(state["valid_mask"].numel()), dtype=torch.long),
        contract_cfg=_contract_cfg(),
    )["q_score"].view(-1).float().cpu()
    contradiction = tensor_attr(state["evidence_state"], "contradiction_score", valid_mask.float(), default=0.0).cpu()
    contrast = _evidence_contrast_scalar(payload["node_features"].cpu(), valid_mask)
    return {
        "valid_mask": valid_mask,
        "candidate_mask": candidate_mask,
        "reasoner_logits": logits,
        "q_score": q_score,
        "contradiction_score": contradiction,
        "contrast_signal": contrast,
    }


def _fused_posterior_from_primitives(shared: Dict[str, Any], params: Dict[str, float]) -> Dict[str, Any]:
    candidate_mask = shared["candidate_mask"]
    logits = shared["reasoner_logits"]
    q_score = shared["q_score"]
    contradiction = shared["contradiction_score"]
    contrast = shared["contrast_signal"]

    q_z = _masked_zscore(q_score, candidate_mask)
    l_z = _masked_zscore(logits, candidate_mask)
    c_z = _masked_zscore(contrast, candidate_mask)
    d_z = _masked_zscore(contradiction, candidate_mask)
    energy = (
        float(params["lambda_reasoner"]) * l_z
        + float(params["lambda_q"]) * q_z
        + float(params["lambda_contrast"]) * c_z
        - float(params["lambda_contradiction"]) * d_z
    )
    probs = _safe_softmax(energy, candidate_mask, float(params["temperature"]))
    return {"belief": probs, "candidate_mask": candidate_mask, "energy": energy}


def _logits_only_from_primitives(shared: Dict[str, Any], temperature: float) -> Dict[str, Any]:
    probs = _safe_softmax(shared["reasoner_logits"], shared["candidate_mask"], float(temperature))
    return {"belief": probs, "candidate_mask": shared["candidate_mask"], "energy": shared["reasoner_logits"].clone()}


def _belief_stats_from_probs(
    *,
    belief: torch.Tensor,
    candidate_mask: torch.Tensor,
    source_local: Optional[int],
    mass_cover_threshold: float = 0.7,
) -> Dict[str, Any]:
    probs = belief.view(-1).float().cpu()
    candidate_mask = candidate_mask.view(-1).bool().cpu()
    probs = probs.clone()
    probs[~candidate_mask] = 0.0
    probs = probs / probs.sum().clamp_min(1e-9)

    order = torch.argsort(probs, descending=True)
    order = order[candidate_mask[order]]
    valid_vals = probs[order]
    entropy = float((-(valid_vals.clamp_min(1e-12) * torch.log(valid_vals.clamp_min(1e-12)))).sum().item()) if valid_vals.numel() else 0.0
    norm_entropy = float(entropy / math.log(max(int(valid_vals.numel()), 2))) if valid_vals.numel() > 1 else 0.0
    eff_support = float(math.exp(entropy))
    csum = torch.cumsum(valid_vals, dim=0)
    cover_idx = int((csum >= float(mass_cover_threshold)).nonzero(as_tuple=True)[0][0].item()) + 1 if valid_vals.numel() else 0
    cover_set = order[:cover_idx]
    hardest = None
    if source_local is not None:
        for idx in order.tolist():
            if int(idx) != int(source_local):
                hardest = int(idx)
                break
    true_mass = float(probs[int(source_local)].item()) if source_local is not None and bool(candidate_mask[int(source_local)].item()) else None
    hard_mass = float(probs[int(hardest)].item()) if hardest is not None else None
    margin = float(true_mass - hard_mass) if true_mass is not None and hard_mass is not None else None
    return {
        "belief": probs,
        "candidate_mask": candidate_mask,
        "ordered_candidates": order,
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "effective_support": eff_support,
        "top1_mass": float(valid_vals[:1].sum().item()) if valid_vals.numel() else 0.0,
        "top3_mass": float(valid_vals[: min(3, valid_vals.numel())].sum().item()) if valid_vals.numel() else 0.0,
        "top5_mass": float(valid_vals[: min(5, valid_vals.numel())].sum().item()) if valid_vals.numel() else 0.0,
        "mass_cover_set": cover_set,
        "mass_cover_size": int(cover_idx),
        "mass_cover_size_ratio": float(cover_idx / max(int(candidate_mask.sum().item()), 1)),
        "mass_cover_mass": float(csum[cover_idx - 1].item()) if cover_idx > 0 else 0.0,
        "source_local": source_local,
        "hardest_confuser_local": hardest,
        "true_mass": true_mass,
        "hard_mass": hard_mass,
        "margin_true_vs_hard": margin,
    }


def _effective_support_ratio(ctx: Dict[str, Any]) -> float:
    candidate_count = max(int(ctx["candidate_mask"].float().sum().item()), 1)
    return float(float(ctx["effective_support"]) / float(candidate_count))


@dataclass
class HSRSoftFactorParams:
    trigger_scale_min: float = 45.0
    pos_slack_min: float = 30.0
    neg_slack_min: float = 30.0
    neg_weight: float = 1.0


def _compute_soft_hsr_factor_belief(
    *,
    history: ObservationWitnessHistory,
    topology: Any,
    global_node_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    trigger_global: Optional[int],
    params: HSRSoftFactorParams,
    device: torch.device,
) -> Dict[str, Any]:
    candidate_mask = candidate_mask.view(-1).bool().cpu()
    cand_global = global_node_ids.view(-1).long().cpu().numpy().astype(np.int64)
    if trigger_global is None or not bool(candidate_mask.any()):
        prior = torch.zeros_like(global_node_ids.view(-1).float().cpu())
        prior[candidate_mask] = 1.0 / max(int(candidate_mask.sum().item()), 1)
        return {"belief": prior, "candidate_mask": candidate_mask}

    from src.modeling.belief_updaters.pure_likelihood_bayes import StaticPropagationAssets

    assets = StaticPropagationAssets(Path(topology.graph_path))
    rel_trig_np, min_trig_np, _ = assets.metrics_to_observation(int(trigger_global), cand_global)
    rel_trig = torch.from_numpy(rel_trig_np).float()
    min_trig = torch.from_numpy(min_trig_np).float()
    log_score = torch.full_like(rel_trig, -float("inf"))
    valid = candidate_mask.clone()
    trigger_term = torch.log(rel_trig.clamp_min(1e-9)) - (min_trig / float(params.trigger_scale_min))
    log_score[valid] = trigger_term[valid]

    pos_records = history.positive_records()
    safe_records = history.safe_records()
    for record in pos_records:
        rel_np, _, med_np = assets.metrics_to_observation(int(record.node_global_idx), cand_global)
        rel = torch.from_numpy(rel_np).float()
        med = torch.from_numpy(med_np).float()
        arrival_diff = med - min_trig
        compat = torch.sigmoid((float(record.absolute_time_min) - arrival_diff) / float(params.pos_slack_min))
        factor = (rel * compat).clamp_min(1e-9)
        log_score[valid] = log_score[valid] + torch.log(factor[valid])
    for record in safe_records:
        _, _, med_np = assets.metrics_to_observation(int(record.node_global_idx), cand_global)
        med = torch.from_numpy(med_np).float()
        arrival_diff = med - min_trig
        compat = torch.sigmoid((arrival_diff - float(record.absolute_time_min)) / float(params.neg_slack_min)).clamp_min(1e-9)
        log_score[valid] = log_score[valid] + float(params.neg_weight) * torch.log(compat[valid])

    belief = _safe_softmax(log_score, candidate_mask, 1.0)
    return {"belief": belief, "candidate_mask": candidate_mask, "energy": log_score}


def _combine_beliefs(base: torch.Tensor, prior: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    base = base.view(-1).float().cpu()
    prior = prior.view(-1).float().cpu()
    candidate_mask = candidate_mask.view(-1).bool().cpu()
    out = torch.zeros_like(base)
    if bool(candidate_mask.any()):
        combined = base[candidate_mask].clamp_min(1e-12) * prior[candidate_mask].clamp_min(1e-12)
        combined = combined / combined.sum().clamp_min(1e-12)
        out[candidate_mask] = combined
    return out


def _load_family_map(acceptability_root: Path) -> Dict[str, Any]:
    accept = json.loads((acceptability_root / "summary.json").read_text())
    return {
        "current_fused": dict(accept["head_definitions"]["current_fused_posterior"]),
        "calibrated_fused": dict(accept["head_definitions"]["calibrated_fused_posterior"]),
        "logits_only_temp": float(accept["head_definitions"]["logits_only_posterior"]["temperature"]),
    }


def _family_list(arg: str) -> List[str]:
    if str(arg).strip():
        return [part.strip() for part in str(arg).split(",") if part.strip()]
    return [
        "logits_only",
        "current_fused",
        "calibrated_fused",
        "pure_likelihood_source_only",
        "pure_likelihood_source_onset",
        "soft_hsr_factor",
        "likelihood_x_fused_prior",
        "likelihood_x_soft_hsr",
    ]


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _screen_mode(args: argparse.Namespace) -> None:
    _seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)

    runtime = load_runtime_context(source_root, cache_dir)
    family_params = _load_family_map(acceptability_root)
    families = _family_list(args.families)
    _, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)

    if str(args.case_subset_csv).strip():
        subset_df = pd.read_csv(Path(args.case_subset_csv))
        keep = set(subset_df["case_id"].astype(str).tolist())
        runtime["cases"] = [case for case in runtime["cases"] if str(case.case_id) in keep]

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    step_rows: List[Dict[str, Any]] = []

    soft_hsr_params = HSRSoftFactorParams()
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
        source_local = resolve_source_local_idx(rollout)
        trigger_global = getattr(case.data, "global_trigger_node", None)
        if isinstance(trigger_global, torch.Tensor):
            trigger_global = int(trigger_global.view(-1)[0].item())
        elif trigger_global is not None:
            trigger_global = int(trigger_global)

        pure_source_only = PureLikelihoodBayesBelief(onset_radius_episodes=0)
        pure_source_onset = PureLikelihoodBayesBelief(onset_radius_episodes=2)
        pure_source_only_state = pure_source_only.init_state(batch_size=1, num_nodes=int(rollout.num_nodes), device=torch.device("cpu"))
        pure_source_onset_state = pure_source_onset.init_state(batch_size=1, num_nodes=int(rollout.num_nodes), device=torch.device("cpu"))

        prev_ctx_by_family: Dict[str, Dict[str, Any]] = {}
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
            shared = _compute_shared_state_primitives(state=state, reasoner_module=reasoner_module, device=device)
            global_ids = rollout.g_ids.detach().cpu().long()
            common_step_in = {
                "history": history,
                "topology": topology,
                "global_node_ids": global_ids,
                "episode_duration_min": float(runtime["episode_duration_min"]),
                "physics_ctx": state["phys_ctx"],
                "constraint_state": state["constraint_state"],
                "source_local": source_local,
            }

            pure_source_only_state, pure_source_only_ctx = pure_source_only.step(pure_source_only_state, common_step_in)
            pure_source_onset_state, pure_source_onset_ctx = pure_source_onset.step(pure_source_onset_state, common_step_in)
            soft_hsr = _compute_soft_hsr_factor_belief(
                history=history,
                topology=topology,
                global_node_ids=global_ids,
                candidate_mask=shared["candidate_mask"],
                trigger_global=trigger_global,
                params=soft_hsr_params,
                device=torch.device("cpu"),
            )
            logits_only = _logits_only_from_primitives(shared, family_params["logits_only_temp"])
            current_fused = _fused_posterior_from_primitives(shared, family_params["current_fused"])
            calibrated_fused = _fused_posterior_from_primitives(shared, family_params["calibrated_fused"])

            family_ctxs: Dict[str, Dict[str, Any]] = {
                "logits_only": _belief_stats_from_probs(belief=logits_only["belief"], candidate_mask=logits_only["candidate_mask"], source_local=source_local),
                "current_fused": _belief_stats_from_probs(belief=current_fused["belief"], candidate_mask=current_fused["candidate_mask"], source_local=source_local),
                "calibrated_fused": _belief_stats_from_probs(belief=calibrated_fused["belief"], candidate_mask=calibrated_fused["candidate_mask"], source_local=source_local),
                "pure_likelihood_source_only": pure_source_only_ctx,
                "pure_likelihood_source_onset": pure_source_onset_ctx,
                "soft_hsr_factor": _belief_stats_from_probs(belief=soft_hsr["belief"], candidate_mask=soft_hsr["candidate_mask"], source_local=source_local),
            }
            family_ctxs["likelihood_x_fused_prior"] = _belief_stats_from_probs(
                belief=_combine_beliefs(
                    pure_source_only_ctx["belief"],
                    calibrated_fused["belief"],
                    pure_source_only_ctx["candidate_mask"],
                ),
                candidate_mask=pure_source_only_ctx["candidate_mask"],
                source_local=source_local,
            )
            family_ctxs["likelihood_x_soft_hsr"] = _belief_stats_from_probs(
                belief=_combine_beliefs(
                    pure_source_only_ctx["belief"],
                    soft_hsr["belief"],
                    pure_source_only_ctx["candidate_mask"],
                ),
                candidate_mask=pure_source_only_ctx["candidate_mask"],
                source_local=source_local,
            )

            for family in families:
                ctx = family_ctxs[family]
                rank = _raw_rank(ctx["belief"], ctx["candidate_mask"], source_local)
                row = {
                    "case_id": case.case_id,
                    "scenario_id": int(case.scenario_id),
                    "part_id": int(case.part_id),
                    "episode_index": int(step.round_index) + 1,
                    "family": family,
                    "candidate_count": int(ctx["candidate_mask"].float().sum().item()),
                    "rank": rank["rank"],
                    "top1_hit": rank["top1_hit"],
                    "top3_hit": rank["top3_hit"],
                    "top5_hit": rank["top5_hit"],
                    "mrr": rank["mrr"],
                    "entropy": float(ctx["entropy"]),
                    "normalized_entropy": float(ctx["normalized_entropy"]),
                    "effective_support": float(ctx["effective_support"]),
                    "effective_support_ratio": _effective_support_ratio(ctx),
                    "top1_mass": float(ctx["top1_mass"]),
                    "top3_mass": float(ctx["top3_mass"]),
                    "top5_mass": float(ctx["top5_mass"]),
                    "mass_cover_size": int(ctx["mass_cover_size"]),
                    "mass_cover_size_ratio": float(ctx["mass_cover_size_ratio"]),
                    "mass_cover_mass": float(ctx["mass_cover_mass"]),
                    "true_mass": ctx["true_mass"],
                    "hard_mass": ctx["hard_mass"],
                    "margin_true_vs_hard": ctx["margin_true_vs_hard"],
                    "source_local": source_local,
                }
                prev = prev_ctx_by_family.get(family)
                if prev is not None:
                    row["delta_true_mass"] = (
                        float(ctx["true_mass"] - prev["true_mass"])
                        if ctx["true_mass"] is not None and prev["true_mass"] is not None
                        else None
                    )
                    row["delta_entropy"] = float(prev["entropy"] - ctx["entropy"])
                    row["delta_mass_cover_shrink"] = float(prev["mass_cover_size_ratio"] - ctx["mass_cover_size_ratio"])
                    row["delta_top1_mass"] = float(ctx["top1_mass"] - prev["top1_mass"])
                    row["delta_effective_support_shrink"] = float(prev["effective_support_ratio"] - _effective_support_ratio(ctx))
                    row["delta_margin"] = (
                        float(ctx["margin_true_vs_hard"] - prev["margin_true_vs_hard"])
                        if ctx["margin_true_vs_hard"] is not None and prev["margin_true_vs_hard"] is not None
                        else None
                    )
                else:
                    row["delta_true_mass"] = None
                    row["delta_entropy"] = None
                    row["delta_mass_cover_shrink"] = None
                    row["delta_top1_mass"] = None
                    row["delta_effective_support_shrink"] = None
                    row["delta_margin"] = None
                step_rows.append(row)
                prev_ctx_by_family[family] = {
                    "true_mass": ctx["true_mass"],
                    "entropy": ctx["entropy"],
                    "mass_cover_size_ratio": ctx["mass_cover_size_ratio"],
                    "top1_mass": ctx["top1_mass"],
                    "effective_support_ratio": row["effective_support_ratio"],
                    "margin_true_vs_hard": ctx["margin_true_vs_hard"],
                }

            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"oracle_slot_{i}" for i in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])

        if case_idx % int(args.progress_every_cases) == 0 or case_idx == len(runtime["cases"]):
            pd.DataFrame(step_rows).to_csv(output_dir / "screen_step_rows.partial.csv", index=False)
            print(f"[screen-progress] case={case_idx}/{len(runtime['cases'])} rows={len(step_rows)}", flush=True)

    step_df = pd.DataFrame(step_rows)
    step_df.to_csv(output_dir / "screen_step_rows.csv", index=False)

    summary_rows = []
    for family, sub in step_df.groupby("family"):
        valid = sub[sub["rank"].notna()].copy()
        summary_rows.append(
            {
                "family": family,
                "step_count": int(len(valid)),
                "top1_hit": float(valid["top1_hit"].mean()),
                "top3_hit": float(valid["top3_hit"].mean()),
                "top5_hit": float(valid["top5_hit"].mean()),
                "mrr": float(valid["mrr"].mean()),
                "true_rank_mean": float(valid["rank"].mean()),
                "median_rank": float(valid["rank"].median()),
                "normalized_entropy_mean": float(valid["normalized_entropy"].mean()),
                "effective_support_ratio_mean": float(valid["effective_support_ratio"].mean()),
                "top1_mass_mean": float(valid["top1_mass"].mean()),
                "top3_mass_mean": float(valid["top3_mass"].mean()),
                "mass_cover_size_ratio_mean": float(valid["mass_cover_size_ratio"].mean()),
                "delta_true_mass_mean": float(valid["delta_true_mass"].dropna().mean()) if valid["delta_true_mass"].notna().any() else None,
                "delta_true_mass_snr": float(valid["delta_true_mass"].dropna().mean() / max(valid["delta_true_mass"].dropna().std(), 1e-9)) if valid["delta_true_mass"].notna().sum() > 1 else None,
                "delta_entropy_mean": float(valid["delta_entropy"].dropna().mean()) if valid["delta_entropy"].notna().any() else None,
                "delta_entropy_snr": float(valid["delta_entropy"].dropna().mean() / max(valid["delta_entropy"].dropna().std(), 1e-9)) if valid["delta_entropy"].notna().sum() > 1 else None,
                "delta_mass_cover_shrink_mean": float(valid["delta_mass_cover_shrink"].dropna().mean()) if valid["delta_mass_cover_shrink"].notna().any() else None,
                "delta_mass_cover_shrink_snr": float(valid["delta_mass_cover_shrink"].dropna().mean() / max(valid["delta_mass_cover_shrink"].dropna().std(), 1e-9)) if valid["delta_mass_cover_shrink"].notna().sum() > 1 else None,
                "delta_top1_mass_mean": float(valid["delta_top1_mass"].dropna().mean()) if valid["delta_top1_mass"].notna().any() else None,
                "delta_top1_mass_snr": float(valid["delta_top1_mass"].dropna().mean() / max(valid["delta_top1_mass"].dropna().std(), 1e-9)) if valid["delta_top1_mass"].notna().sum() > 1 else None,
                "delta_effective_support_shrink_mean": float(valid["delta_effective_support_shrink"].dropna().mean()) if valid["delta_effective_support_shrink"].notna().any() else None,
                "delta_effective_support_shrink_snr": float(valid["delta_effective_support_shrink"].dropna().mean() / max(valid["delta_effective_support_shrink"].dropna().std(), 1e-9)) if valid["delta_effective_support_shrink"].notna().sum() > 1 else None,
                "delta_margin_mean": float(valid["delta_margin"].dropna().mean()) if valid["delta_margin"].notna().any() else None,
                "delta_margin_snr": float(valid["delta_margin"].dropna().mean() / max(valid["delta_margin"].dropna().std(), 1e-9)) if valid["delta_margin"].notna().sum() > 1 else None,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["mrr", "top5_hit"], ascending=[False, False])
    summary_df.to_csv(output_dir / "screen_family_summary.csv", index=False)

    write_json(
        output_dir / "screen_summary.json",
        {
            "runner_version": RUNNER_VERSION,
            "panel_version": PANEL_VERSION,
            "mode": "screen",
            "seed": int(args.seed),
            "family_list": families,
            "source_root": str(source_root),
            "cache_dir": str(cache_dir),
            "reasoner_checkpoint": str(frozen_checkpoint),
            "summary_table": str(output_dir / "screen_family_summary.csv"),
        },
    )


def _belief_family_for_state(
    *,
    family: str,
    state: Dict[str, Any],
    history: ObservationWitnessHistory,
    rollout: PracticalRollout,
    case: Any,
    reasoner_module,
    device: torch.device,
    family_params: Dict[str, Any],
    pure_state_store: Dict[str, Any],
    topology: Any,
) -> Dict[str, Any]:
    shared = _compute_shared_state_primitives(state=state, reasoner_module=reasoner_module, device=device)
    source_local = resolve_source_local_idx(rollout)
    global_ids = rollout.g_ids.detach().cpu().long()
    trigger_global = getattr(case.data, "global_trigger_node", None)
    if isinstance(trigger_global, torch.Tensor):
        trigger_global = int(trigger_global.view(-1)[0].item())
    elif trigger_global is not None:
        trigger_global = int(trigger_global)

    if family == "logits_only":
        return _belief_stats_from_probs(
            belief=_logits_only_from_primitives(shared, family_params["logits_only_temp"])["belief"],
            candidate_mask=shared["candidate_mask"],
            source_local=source_local,
        )
    if family == "current_fused":
        fused = _fused_posterior_from_primitives(shared, family_params["current_fused"])
        return _belief_stats_from_probs(belief=fused["belief"], candidate_mask=fused["candidate_mask"], source_local=source_local)
    if family == "calibrated_fused":
        fused = _fused_posterior_from_primitives(shared, family_params["calibrated_fused"])
        return _belief_stats_from_probs(belief=fused["belief"], candidate_mask=fused["candidate_mask"], source_local=source_local)

    common_step_in = {
        "history": history,
        "topology": topology,
        "global_node_ids": global_ids,
        "episode_duration_min": float(rollout.episode_duration_min),
        "physics_ctx": state["phys_ctx"],
        "constraint_state": state["constraint_state"],
        "source_local": source_local,
    }
    if family == "pure_likelihood_source_only":
        pure_state_store["source_only"], ctx = pure_state_store["source_only_updater"].step(pure_state_store["source_only"], common_step_in)
        return ctx
    if family == "pure_likelihood_source_onset":
        pure_state_store["source_onset"], ctx = pure_state_store["source_onset_updater"].step(pure_state_store["source_onset"], common_step_in)
        return ctx

    source_only_state, source_only_ctx = pure_state_store["source_only_updater"].step(pure_state_store["source_only"], common_step_in)
    pure_state_store["source_only"] = source_only_state
    if family == "soft_hsr_factor":
        soft = _compute_soft_hsr_factor_belief(
            history=history,
            topology=topology,
            global_node_ids=global_ids,
            candidate_mask=shared["candidate_mask"],
            trigger_global=trigger_global,
            params=HSRSoftFactorParams(),
            device=torch.device("cpu"),
        )
        return _belief_stats_from_probs(belief=soft["belief"], candidate_mask=soft["candidate_mask"], source_local=source_local)
    if family == "likelihood_x_fused_prior":
        fused = _fused_posterior_from_primitives(shared, family_params["calibrated_fused"])
        return _belief_stats_from_probs(
            belief=_combine_beliefs(source_only_ctx["belief"], fused["belief"], source_only_ctx["candidate_mask"]),
            candidate_mask=source_only_ctx["candidate_mask"],
            source_local=source_local,
        )
    if family == "likelihood_x_soft_hsr":
        soft = _compute_soft_hsr_factor_belief(
            history=history,
            topology=topology,
            global_node_ids=global_ids,
            candidate_mask=shared["candidate_mask"],
            trigger_global=trigger_global,
            params=HSRSoftFactorParams(),
            device=torch.device("cpu"),
        )
        return _belief_stats_from_probs(
            belief=_combine_beliefs(source_only_ctx["belief"], soft["belief"], source_only_ctx["candidate_mask"]),
            candidate_mask=source_only_ctx["candidate_mask"],
            source_local=source_local,
        )
    raise ValueError(family)


def _greedy_mode(args: argparse.Namespace) -> None:
    _seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)
    runtime = load_runtime_context(source_root, cache_dir)
    runtime["num_episodes"] = int(args.num_rounds)
    runtime["action_budget"] = int(args.actions_per_round)

    family_params = _load_family_map(acceptability_root)
    families = _family_list(args.families)
    _, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)

    if str(args.case_subset_csv).strip():
        subset_df = pd.read_csv(Path(args.case_subset_csv))
        keep = set(subset_df["case_id"].astype(str).tolist())
        runtime["cases"] = [case for case in runtime["cases"] if str(case.case_id) in keep]

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    reward_rows: List[Dict[str, Any]] = []

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
            pure_state_store = {
                "source_only_updater": PureLikelihoodBayesBelief(onset_radius_episodes=0),
                "source_onset_updater": PureLikelihoodBayesBelief(onset_radius_episodes=2),
            }
            pure_state_store["source_only"] = pure_state_store["source_only_updater"].init_state(batch_size=1, num_nodes=int(rollout.num_nodes), device=torch.device("cpu"))
            pure_state_store["source_onset"] = pure_state_store["source_onset_updater"].init_state(batch_size=1, num_nodes=int(rollout.num_nodes), device=torch.device("cpu"))

            prev_ctx = None
            hit_round = None
            source_local = resolve_source_local_idx(rollout)
            for episode_idx in range(int(runtime["num_episodes"])):
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
                if int(state["valid_mask"].sum().item()) <= 0:
                    break
                ctx = _belief_family_for_state(
                    family=family,
                    state=state,
                    history=history,
                    rollout=rollout,
                    case=case,
                    reasoner_module=reasoner_module,
                    device=device,
                    family_params=family_params,
                    pure_state_store=pure_state_store,
                    topology=topology,
                )

                ordered = [int(v) for v in ctx["ordered_candidates"].tolist()]
                selected_local: List[int] = []
                for idx in ordered:
                    if bool(rollout.revealed_mask[int(idx)].item()):
                        continue
                    selected_local.append(int(idx))
                    if len(selected_local) >= int(runtime["action_budget"]):
                        break
                if not selected_local:
                    break
                selected_global = [int(rollout.g_ids[int(idx)].item()) for idx in selected_local]
                round_hit = source_local is not None and int(source_local) in set(selected_local)
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
                post_ctx = _belief_family_for_state(
                    family=family,
                    state=post_state,
                    history=history,
                    rollout=rollout,
                    case=case,
                    reasoner_module=reasoner_module,
                    device=device,
                    family_params=family_params,
                    pure_state_store=pure_state_store,
                    topology=topology,
                )

                if round_hit and hit_round is None:
                    hit_round = int(episode_idx + 1)
                sampled_node_mass = float(ctx["belief"][selected_local[0]].item()) if selected_local else 0.0
                row = {
                    "case_id": case.case_id,
                    "family": family,
                    "episode_index": int(episode_idx + 1),
                    "selected_local_ids": json.dumps(selected_local),
                    "selected_global_ids": json.dumps(selected_global),
                    "sampled_node_posterior_mass": sampled_node_mass,
                    "pre_entropy": float(ctx["entropy"]),
                    "post_entropy": float(post_ctx["entropy"]),
                    "delta_entropy": float(ctx["entropy"] - post_ctx["entropy"]),
                    "pre_top1_mass": float(ctx["top1_mass"]),
                    "post_top1_mass": float(post_ctx["top1_mass"]),
                    "delta_top1_mass": float(post_ctx["top1_mass"] - ctx["top1_mass"]),
                    "pre_mass_cover_size_ratio": float(ctx["mass_cover_size_ratio"]),
                    "post_mass_cover_size_ratio": float(post_ctx["mass_cover_size_ratio"]),
                    "delta_mass_cover_shrink": float(ctx["mass_cover_size_ratio"] - post_ctx["mass_cover_size_ratio"]),
                    "pre_effective_support_ratio": _effective_support_ratio(ctx),
                    "post_effective_support_ratio": _effective_support_ratio(post_ctx),
                    "delta_effective_support_shrink": float(
                        _effective_support_ratio(ctx) - _effective_support_ratio(post_ctx)
                    ),
                    "delta_margin": (
                        float(post_ctx["margin_true_vs_hard"] - ctx["margin_true_vs_hard"])
                        if ctx["margin_true_vs_hard"] is not None and post_ctx["margin_true_vs_hard"] is not None
                        else None
                    ),
                    "delta_true_mass": (
                        float(post_ctx["true_mass"] - ctx["true_mass"])
                        if ctx["true_mass"] is not None and post_ctx["true_mass"] is not None
                        else None
                    ),
                    "source_hit_in_round": float(bool(round_hit)),
                    "budget_used": float(rollout.revealed_mask.sum().item()),
                }
                step_rows.append(row)
                prev_ctx = post_ctx

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
            final_ctx = _belief_family_for_state(
                family=family,
                state=final_state,
                history=history,
                rollout=rollout,
                case=case,
                reasoner_module=reasoner_module,
                device=device,
                family_params=family_params,
                pure_state_store=pure_state_store,
                topology=topology,
            )
            rank = _raw_rank(final_ctx["belief"], final_ctx["candidate_mask"], source_local)
            case_rows.append(
                {
                    "case_id": case.case_id,
                    "family": family,
                    "success_rate": float(hit_round is not None),
                    "hit_round": hit_round,
                    "final_top1_hit": rank["top1_hit"],
                    "final_top3_hit": rank["top3_hit"],
                    "final_top5_hit": rank["top5_hit"],
                    "final_mrr": rank["mrr"],
                    "final_rank": rank["rank"],
                    "final_true_mass": final_ctx["true_mass"],
                    "final_entropy": float(final_ctx["entropy"]),
                    "final_mass_cover_size_ratio": float(final_ctx["mass_cover_size_ratio"]),
                    "budget_used": float(rollout.revealed_mask.sum().item()),
                }
            )
            if case_idx % int(args.progress_every_cases) == 0 or case_idx == len(runtime["cases"]):
                pd.DataFrame(case_rows).to_csv(output_dir / "greedy_case_rows.partial.csv", index=False)
                pd.DataFrame(step_rows).to_csv(output_dir / "greedy_step_rows.partial.csv", index=False)
                print(f"[greedy-progress] family={family} case={case_idx}/{len(runtime['cases'])}", flush=True)

    case_df = pd.DataFrame(case_rows)
    step_df = pd.DataFrame(step_rows)
    case_df.to_csv(output_dir / "greedy_case_rows.csv", index=False)
    step_df.to_csv(output_dir / "greedy_step_rows.csv", index=False)

    compare_rows = []
    for family, sub in case_df.groupby("family"):
        compare_rows.append(
            {
                "family": family,
                "case_count": int(len(sub)),
                "success_rate": float(sub["success_rate"].mean()),
                "final_top1_hit": float(sub["final_top1_hit"].mean()),
                "final_top3_hit": float(sub["final_top3_hit"].mean()),
                "final_top5_hit": float(sub["final_top5_hit"].mean()),
                "final_mrr": float(sub["final_mrr"].mean()),
                "hit_round_mean": float(sub["hit_round"].dropna().mean()) if sub["hit_round"].notna().any() else None,
                "budget_used_mean": float(sub["budget_used"].mean()),
                "final_true_mass_mean": float(sub["final_true_mass"].dropna().mean()) if sub["final_true_mass"].notna().any() else None,
                "final_entropy_mean": float(sub["final_entropy"].mean()),
                "final_mass_cover_size_ratio_mean": float(sub["final_mass_cover_size_ratio"].mean()),
            }
        )
    compare_df = pd.DataFrame(compare_rows).sort_values(["success_rate", "final_mrr"], ascending=[False, False])
    compare_df.to_csv(output_dir / "greedy_family_compare.csv", index=False)

    reward_rows = []
    for family, sub in step_df.groupby("family"):
        final_success_by_case = case_df[case_df["family"] == family].set_index("case_id")["success_rate"].to_dict()
        sub = sub.copy()
        sub["final_success"] = sub["case_id"].map(final_success_by_case).astype(float)
        for reward_name in [
            "delta_entropy",
            "delta_top1_mass",
            "delta_mass_cover_shrink",
            "delta_effective_support_shrink",
            "delta_margin",
            "sampled_node_posterior_mass",
            "delta_true_mass",
            "source_hit_in_round",
        ]:
            vals = sub[reward_name].dropna()
            if len(vals) <= 1:
                continue
            reward_rows.append(
                {
                    "family": family,
                    "reward_name": reward_name,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "snr": float(vals.mean() / max(vals.std(), 1e-9)),
                    "positive_rate": float((vals > 0).mean()),
                    "corr_to_final_success": (
                        float(sub[[reward_name, "final_success"]].corr(method="spearman").iloc[0, 1])
                        if sub[[reward_name, "final_success"]].dropna().shape[0] > 2
                        else None
                    ),
                    "corr_to_source_hit_in_round": (
                        float(sub[[reward_name, "source_hit_in_round"]].corr(method="spearman").iloc[0, 1])
                        if sub[[reward_name, "source_hit_in_round"]].dropna().shape[0] > 2
                        else None
                    ),
                }
            )
    reward_df = pd.DataFrame(reward_rows)
    reward_df.to_csv(output_dir / "reward_family_compare.csv", index=False)

    write_json(
        output_dir / "greedy_summary.json",
        {
            "runner_version": RUNNER_VERSION,
            "panel_version": PANEL_VERSION,
            "mode": "greedy",
            "seed": int(args.seed),
            "families": families,
            "num_rounds": int(args.num_rounds),
            "actions_per_round": int(args.actions_per_round),
            "reasoner_checkpoint": str(frozen_checkpoint),
            "compare_table": str(output_dir / "greedy_family_compare.csv"),
            "reward_table": str(output_dir / "reward_family_compare.csv"),
        },
    )


def main() -> None:
    args = parse_args()
    if args.mode == "screen":
        _screen_mode(args)
    elif args.mode == "greedy":
        _greedy_mode(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
