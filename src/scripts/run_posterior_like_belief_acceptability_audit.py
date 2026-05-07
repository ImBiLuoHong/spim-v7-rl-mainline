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

from src.modeling.belief_updaters.evidence_posterior_like import _evidence_contrast_scalar, _masked_zscore
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.modeling.loop.navigator_vnext_contract import build_candidate_semantics, default_reward_contract, tensor_attr
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "posterior_like_belief_acceptability_audit" / "20260407_exact136_belief_acceptability_v1"
RUNNER_VERSION = "posterior_like_belief_acceptability_audit_v1"
PANEL_VERSION = "exact136_train_only_belief_acceptability_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acceptability audit for posterior-like belief heads on exact136 oracle train cases.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--current-temperature", type=float, default=1.0)
    parser.add_argument("--current-lambda-q", type=float, default=0.75)
    parser.add_argument("--current-lambda-reasoner", type=float, default=1.0)
    parser.add_argument("--current-lambda-contrast", type=float, default=0.5)
    parser.add_argument("--current-lambda-contradiction", type=float, default=0.25)
    parser.add_argument("--support-plausible-delta", type=float, default=0.25)
    parser.add_argument("--not-ruled-out-threshold", type=float, default=0.5)
    parser.add_argument("--confusion-logit-delta", type=float, default=1.0)
    parser.add_argument("--cluster-mass-threshold", type=float, default=0.8)
    parser.add_argument("--material-ranking-preservation-floor", type=float, default=0.9)
    parser.add_argument("--material-cluster-ratio-ceiling", type=float, default=0.25)
    parser.add_argument("--material-hard-delta_snr_floor", type=float, default=0.1)
    return parser.parse_args()


def _safe_softmax(scores: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    scores = scores.view(-1).float()
    mask = mask.view(-1).bool()
    out = torch.zeros_like(scores)
    if not bool(mask.any()):
        return out
    out[mask] = torch.softmax(scores[mask] / float(max(temperature, 1e-6)), dim=0)
    return out


def _topk_mass(probs: torch.Tensor, mask: torch.Tensor, k: int) -> float:
    if not bool(mask.any()):
        return 0.0
    vals = probs[mask]
    k_eff = min(int(k), int(vals.numel()))
    if k_eff <= 0:
        return 0.0
    return float(torch.topk(vals, k=k_eff).values.sum().item())


def _rank_metrics(probs_or_scores: torch.Tensor, mask: torch.Tensor, source_local: int | None) -> Dict[str, Any]:
    scores = probs_or_scores.view(-1).float()
    mask = mask.view(-1).bool()
    if source_local is None or not bool(mask[int(source_local)].item()):
        return {
            "rank": None,
            "top1_hit": 0.0,
            "top3_hit": 0.0,
            "top5_hit": 0.0,
            "mrr": 0.0,
            "margin_true_vs_hard": float("nan"),
            "hardest_confuser_mass": float("nan"),
            "true_mass": float("nan"),
        }
    safe = scores.clone()
    safe[~mask] = -float("inf")
    order = torch.argsort(safe, descending=True)
    order = order[torch.isfinite(safe[order])]
    positions = (order == int(source_local)).nonzero(as_tuple=True)[0]
    if positions.numel() <= 0:
        return {
            "rank": None,
            "top1_hit": 0.0,
            "top3_hit": 0.0,
            "top5_hit": 0.0,
            "mrr": 0.0,
            "margin_true_vs_hard": float("nan"),
            "hardest_confuser_mass": float("nan"),
            "true_mass": float("nan"),
        }
    rank = int(positions.min().item()) + 1
    hardest_confuser = float("nan")
    for idx in order.tolist():
        if int(idx) != int(source_local):
            hardest_confuser = float(safe[int(idx)].item())
            break
    true_mass = float(safe[int(source_local)].item())
    margin = float(true_mass - hardest_confuser) if math.isfinite(hardest_confuser) else float("nan")
    return {
        "rank": rank,
        "top1_hit": float(rank <= 1),
        "top3_hit": float(rank <= 3),
        "top5_hit": float(rank <= 5),
        "mrr": float(1.0 / rank),
        "margin_true_vs_hard": margin,
        "hardest_confuser_mass": hardest_confuser,
        "true_mass": true_mass,
    }


def _mass_cluster_count(probs: torch.Tensor, mask: torch.Tensor, mass_threshold: float) -> Tuple[int, float]:
    mask = mask.view(-1).bool()
    if not bool(mask.any()):
        return 0, 0.0
    vals = probs[mask]
    order = torch.argsort(vals, descending=True)
    sorted_vals = vals[order]
    cumsum = torch.cumsum(sorted_vals, dim=0)
    idx = int((cumsum >= float(mass_threshold)).nonzero(as_tuple=True)[0][0].item()) + 1
    mass = float(cumsum[idx - 1].item())
    return idx, mass


def _normalized_entropy(probs: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    mask = mask.view(-1).bool()
    if not bool(mask.any()):
        return 0.0, 0.0
    vals = probs[mask].clamp_min(1e-12)
    entropy = float((-(vals * torch.log(vals))).sum().item())
    denom = math.log(max(int(vals.numel()), 2))
    return entropy, float(entropy / denom) if denom > 0 else 0.0


def _effective_support(entropy: float) -> float:
    return float(math.exp(float(entropy)))


def _concentration_ratio(topk_mass: float, candidate_count: int, k: int) -> float:
    k_eff = min(int(k), int(candidate_count))
    uniform = float(k_eff) / float(candidate_count) if candidate_count > 0 else 0.0
    if uniform <= 0:
        return 0.0
    return float(topk_mass / uniform)


def _spearman(df: pd.DataFrame, x: str, y: str) -> float | None:
    sub = df[[x, y]].dropna()
    if len(sub) < 5:
        return None
    val = sub[x].corr(sub[y], method="spearman")
    return None if pd.isna(val) else float(val)


def _default_contract(args: argparse.Namespace) -> Dict[str, float]:
    cfg = default_reward_contract()
    cfg.update(
        {
            "support_plausible_delta": float(args.support_plausible_delta),
            "not_ruled_out_threshold": float(args.not_ruled_out_threshold),
        }
    )
    return cfg


def collect_step_payloads(runtime: Dict[str, Any], reasoner_module, args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    contract_cfg = _default_contract(args)
    rows: List[Dict[str, Any]] = []
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
            graph = TempGraph(state["edge_index"], int(state["valid_mask"].numel()), device)
            state_input = move_payload(build_state_input(state), device)
            physics_ctx = move_payload(state["phys_ctx"].__dict__, device)
            with torch.no_grad():
                out = reasoner_module(state_input, graph, physics_ctx=physics_ctx)
            reasoner_logits = out["logits"].detach().float().view(-1).cpu()
            payload = build_clean_aligned_feature_payload(
                build_state_input(state),
                batch_index=torch.zeros(int(state["valid_mask"].numel()), dtype=torch.long),
                edge_index=state["edge_index"].view(2, -1).long(),
                physics_ctx=state["phys_ctx"].__dict__,
                frontier_mode="unresolved_without_pair",
            )
            valid_mask = state["valid_mask"].view(-1).bool().cpu()
            semantics = build_candidate_semantics(
                evidence_state=state["evidence_state"],
                constraint_state=state["constraint_state"],
                valid_mask=valid_mask,
                batch=torch.zeros(int(valid_mask.numel()), dtype=torch.long),
                contract_cfg=contract_cfg,
            )
            candidate_mask = semantics["candidate_mask"].view(-1).bool().cpu()
            if not bool(candidate_mask.any()):
                candidate_mask = valid_mask.clone()
            q_score = semantics["q_score"].view(-1).float().cpu()
            contradiction = tensor_attr(state["evidence_state"], "contradiction_score", valid_mask.float(), default=0.0).cpu()
            contrast_signal = _evidence_contrast_scalar(payload["node_features"].cpu(), valid_mask)
            rows.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "part_id": case.part_id,
                    "episode_index": int(step.round_index) + 1,
                    "source_local": source_local,
                    "valid_mask": valid_mask,
                    "candidate_mask": candidate_mask,
                    "reasoner_logits": reasoner_logits,
                    "q_score": q_score,
                    "contradiction_score": contradiction,
                    "contrast_signal": contrast_signal,
                }
            )
            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"oracle_slot_{i}" for i in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
    return rows


def fit_logits_temperature(step_payloads: List[Dict[str, Any]]) -> float:
    log_temp = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.LBFGS([log_temp], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        temp = torch.nn.functional.softplus(log_temp) + 1e-4
        loss = torch.tensor(0.0)
        count = 0
        for row in step_payloads:
            src = row["source_local"]
            mask = row["candidate_mask"]
            if src is None or not bool(mask[int(src)].item()):
                continue
            logits = row["reasoner_logits"].clone()
            logits[~mask] = -float("inf")
            logp = torch.log_softmax(logits[mask] / temp, dim=0)
            local_idx = int((mask.nonzero(as_tuple=True)[0] == int(src)).nonzero(as_tuple=True)[0][0].item())
            loss = loss - logp[local_idx]
            count += 1
        loss = loss / max(count, 1)
        loss.backward()
        return loss

    opt.step(closure)
    return float((torch.nn.functional.softplus(log_temp) + 1e-4).item())


def fit_calibrated_fused(step_payloads: List[Dict[str, Any]], current: Dict[str, float]) -> Dict[str, float]:
    log_temp = torch.nn.Parameter(torch.tensor(math.log(math.expm1(max(current["temperature"], 1e-3)))))
    raw_q = torch.nn.Parameter(torch.tensor(math.log(math.expm1(max(current["lambda_q"], 1e-3)))))
    raw_c = torch.nn.Parameter(torch.tensor(math.log(math.expm1(max(current["lambda_contrast"], 1e-3)))))
    raw_d = torch.nn.Parameter(torch.tensor(math.log(math.expm1(max(current["lambda_contradiction"], 1e-3)))))
    params = [log_temp, raw_q, raw_c, raw_d]
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        temp = torch.nn.functional.softplus(log_temp) + 1e-4
        w_q = torch.nn.functional.softplus(raw_q)
        w_c = torch.nn.functional.softplus(raw_c)
        w_d = torch.nn.functional.softplus(raw_d)
        loss = torch.tensor(0.0)
        count = 0
        for row in step_payloads:
            src = row["source_local"]
            mask = row["candidate_mask"]
            if src is None or not bool(mask[int(src)].item()):
                continue
            q_z = _masked_zscore(row["q_score"], mask)
            l_z = _masked_zscore(row["reasoner_logits"], mask)
            c_z = _masked_zscore(row["contrast_signal"], mask)
            d_z = _masked_zscore(row["contradiction_score"], mask)
            energy = l_z + w_q * q_z + w_c * c_z - w_d * d_z
            energy[~mask] = -float("inf")
            logp = torch.log_softmax(energy[mask] / temp, dim=0)
            local_idx = int((mask.nonzero(as_tuple=True)[0] == int(src)).nonzero(as_tuple=True)[0][0].item())
            loss = loss - logp[local_idx]
            count += 1
        loss = loss / max(count, 1)
        reg = (
            0.01 * (w_q - current["lambda_q"]) ** 2
            + 0.01 * (w_c - current["lambda_contrast"]) ** 2
            + 0.01 * (w_d - current["lambda_contradiction"]) ** 2
            + 0.01 * (temp - current["temperature"]) ** 2
        )
        total = loss + reg
        total.backward()
        return total

    opt.step(closure)
    return {
        "temperature": float((torch.nn.functional.softplus(log_temp) + 1e-4).item()),
        "lambda_q": float(torch.nn.functional.softplus(raw_q).item()),
        "lambda_reasoner": 1.0,
        "lambda_contrast": float(torch.nn.functional.softplus(raw_c).item()),
        "lambda_contradiction": float(torch.nn.functional.softplus(raw_d).item()),
    }


def evaluate_head(
    head_name: str,
    step_payloads: List[Dict[str, Any]],
    params: Dict[str, float],
    cluster_mass_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prev_by_case: Dict[str, Dict[str, float]] = {}
    for row in step_payloads:
        mask = row["candidate_mask"]
        logits = row["reasoner_logits"]
        if head_name == "logits_only_posterior":
            probs = _safe_softmax(logits, mask, params["temperature"])
            energy = logits.clone()
            energy[~mask] = -float("inf")
        else:
            q_z = _masked_zscore(row["q_score"], mask)
            l_z = _masked_zscore(logits, mask)
            c_z = _masked_zscore(row["contrast_signal"], mask)
            d_z = _masked_zscore(row["contradiction_score"], mask)
            energy = (
                params["lambda_reasoner"] * l_z
                + params["lambda_q"] * q_z
                + params["lambda_contrast"] * c_z
                - params["lambda_contradiction"] * d_z
            )
            probs = _safe_softmax(energy, mask, params["temperature"])
            energy[~mask] = -float("inf")

        rank = _rank_metrics(probs, mask, row["source_local"])
        entropy, norm_entropy = _normalized_entropy(probs, mask)
        candidate_count = int(mask.float().sum().item())
        eff_support = _effective_support(entropy)
        top1_mass = float(probs[mask].max().item()) if bool(mask.any()) else 0.0
        top3_mass = _topk_mass(probs, mask, 3)
        top5_mass = _topk_mass(probs, mask, 5)
        mass80_count, mass80 = _mass_cluster_count(probs, mask, cluster_mass_threshold)
        current_cluster = mask & (energy >= float(energy[mask].max().item()) - 1.0) if bool(mask.any()) else mask
        cluster_count = int(current_cluster.float().sum().item())
        cluster_mass = float(probs[current_cluster].sum().item()) if bool(mask.any()) else 0.0
        out = {
            "head": head_name,
            "case_id": row["case_id"],
            "scenario_id": row["scenario_id"],
            "part_id": row["part_id"],
            "episode_index": row["episode_index"],
            "candidate_count": candidate_count,
            "rank": rank["rank"],
            "top1_hit": rank["top1_hit"],
            "top3_hit": rank["top3_hit"],
            "top5_hit": rank["top5_hit"],
            "mrr": rank["mrr"],
            "margin_true_vs_hard": rank["margin_true_vs_hard"],
            "hardest_confuser_mass": rank["hardest_confuser_mass"],
            "true_mass": rank["true_mass"],
            "entropy": entropy,
            "normalized_entropy": norm_entropy,
            "effective_support": eff_support,
            "effective_support_ratio": float(eff_support / candidate_count) if candidate_count > 0 else 0.0,
            "top1_mass": top1_mass,
            "top3_mass": top3_mass,
            "top5_mass": top5_mass,
            "top1_concentration_ratio": _concentration_ratio(top1_mass, candidate_count, 1),
            "top3_concentration_ratio": _concentration_ratio(top3_mass, candidate_count, 3),
            "top5_concentration_ratio": _concentration_ratio(top5_mass, candidate_count, 5),
            "cluster_count": cluster_count,
            "cluster_count_ratio": float(cluster_count / candidate_count) if candidate_count > 0 else 0.0,
            "cluster_mass": cluster_mass,
            "mass80_cluster_count": mass80_count,
            "mass80_cluster_count_ratio": float(mass80_count / candidate_count) if candidate_count > 0 else 0.0,
            "mass80_cluster_mass": mass80,
            "hardest_confuser_defined": float(math.isfinite(rank["hardest_confuser_mass"])),
            "current_temperature": float(params["temperature"]),
        }
        prev = prev_by_case.get(row["case_id"])
        if prev is None or not math.isfinite(out["true_mass"]):
            out["delta_true_mass"] = None
            out["delta_entropy"] = None
            out["delta_margin"] = None
            out["delta_cluster_count"] = None
        else:
            out["delta_true_mass"] = float(out["true_mass"] - prev["true_mass"])
            out["delta_entropy"] = float(prev["entropy"] - out["entropy"])
            out["delta_margin"] = float(out["margin_true_vs_hard"] - prev["margin_true_vs_hard"])
            out["delta_cluster_count"] = float(prev["cluster_count"] - out["cluster_count"])
        prev_by_case[row["case_id"]] = {
            "true_mass": out["true_mass"],
            "entropy": out["entropy"],
            "margin_true_vs_hard": out["margin_true_vs_hard"],
            "cluster_count": float(out["cluster_count"]),
        }
        rows.append(out)
    df = pd.DataFrame(rows)
    df["next_delta_true_mass"] = df.groupby("case_id")["delta_true_mass"].shift(-1)
    df["next_delta_entropy"] = df.groupby("case_id")["delta_entropy"].shift(-1)
    df["next_delta_margin"] = df.groupby("case_id")["delta_margin"].shift(-1)
    return df


def summarise_head(df: pd.DataFrame, raw_logit_baseline: Dict[str, float]) -> Dict[str, Any]:
    valid = df[df["rank"].notna()].copy()
    deltas = df[df["delta_true_mass"].notna()].copy()
    out: Dict[str, Any] = {
        "valid_case_count": int(len(valid)),
        "top1_hit": float(valid["top1_hit"].mean()),
        "top3_hit": float(valid["top3_hit"].mean()),
        "top5_hit": float(valid["top5_hit"].mean()),
        "mrr": float(valid["mrr"].mean()),
        "true_rank_mean": float(valid["rank"].mean()),
        "median_rank": float(valid["rank"].median()),
        "ranking_preservation_mrr_ratio": float(valid["mrr"].mean() / raw_logit_baseline["mrr"]) if raw_logit_baseline["mrr"] > 0 else None,
        "ranking_preservation_top5_ratio": float(valid["top5_hit"].mean() / raw_logit_baseline["top5_hit"]) if raw_logit_baseline["top5_hit"] > 0 else None,
        "mrr_abs_delta_vs_reasoner": float(valid["mrr"].mean() - raw_logit_baseline["mrr"]),
        "top5_abs_delta_vs_reasoner": float(valid["top5_hit"].mean() - raw_logit_baseline["top5_hit"]),
        "normalized_entropy_mean": float(valid["normalized_entropy"].mean()),
        "normalized_entropy_std": float(valid["normalized_entropy"].std()),
        "effective_support_mean": float(valid["effective_support"].mean()),
        "effective_support_ratio_mean": float(valid["effective_support_ratio"].mean()),
        "top1_mass_mean": float(valid["top1_mass"].mean()),
        "top3_mass_mean": float(valid["top3_mass"].mean()),
        "top5_mass_mean": float(valid["top5_mass"].mean()),
        "top1_concentration_ratio_mean": float(valid["top1_concentration_ratio"].mean()),
        "top3_concentration_ratio_mean": float(valid["top3_concentration_ratio"].mean()),
        "top5_concentration_ratio_mean": float(valid["top5_concentration_ratio"].mean()),
        "cluster_count_mean": float(valid["cluster_count"].mean()),
        "cluster_count_median": float(valid["cluster_count"].median()),
        "cluster_count_ratio_mean": float(valid["cluster_count_ratio"].mean()),
        "mass80_cluster_count_mean": float(valid["mass80_cluster_count"].mean()),
        "mass80_cluster_count_median": float(valid["mass80_cluster_count"].median()),
        "mass80_cluster_count_ratio_mean": float(valid["mass80_cluster_count_ratio"].mean()),
        "hardest_confuser_defined_rate": float(valid["hardest_confuser_defined"].mean()),
    }
    for col in ["delta_true_mass", "delta_entropy", "delta_margin", "delta_cluster_count"]:
        if len(deltas):
            out[f"{col}_mean"] = float(deltas[col].mean())
            out[f"{col}_std"] = float(deltas[col].std())
            out[f"{col}_q25"] = float(deltas[col].quantile(0.25))
            out[f"{col}_q50"] = float(deltas[col].quantile(0.50))
            out[f"{col}_q75"] = float(deltas[col].quantile(0.75))
            out[f"{col}_positive_rate"] = float((deltas[col] > 0).mean())
            std = float(deltas[col].std())
            out[f"{col}_snr"] = float(deltas[col].mean() / std) if std > 1e-9 else None
        else:
            out[f"{col}_mean"] = None
    out["proxy_corr_norm_entropy_to_next_delta_entropy"] = _spearman(valid, "normalized_entropy", "next_delta_entropy")
    out["proxy_corr_norm_entropy_to_next_delta_true_mass"] = _spearman(valid, "normalized_entropy", "next_delta_true_mass")
    out["proxy_corr_one_minus_top3_to_next_delta_margin"] = _spearman(valid.assign(one_minus_top3=1.0 - valid["top3_mass"]), "one_minus_top3", "next_delta_margin")
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    source_root = Path(args.source_root)
    contrast_root = Path(args.contrast_root)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    _, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)
    step_payloads = collect_step_payloads(runtime, reasoner_module, args, device)

    logits_temperature = fit_logits_temperature(step_payloads)
    current_fused = {
        "temperature": float(args.current_temperature),
        "lambda_q": float(args.current_lambda_q),
        "lambda_reasoner": float(args.current_lambda_reasoner),
        "lambda_contrast": float(args.current_lambda_contrast),
        "lambda_contradiction": float(args.current_lambda_contradiction),
    }
    calibrated_fused = fit_calibrated_fused(step_payloads, current_fused)

    heads = {
        "logits_only_posterior": {
            "temperature": float(logits_temperature),
        },
        "current_fused_posterior": current_fused,
        "calibrated_fused_posterior": calibrated_fused,
    }

    head_frames: List[pd.DataFrame] = []
    for head_name, params in heads.items():
        head_frames.append(evaluate_head(head_name, step_payloads, params, float(args.cluster_mass_threshold)))
    step_df = pd.concat(head_frames, ignore_index=True)
    step_df.to_csv(output_dir / "belief_head_step_rows.csv", index=False)

    raw_reasoner_baseline = {
        "mrr": float(
            step_df[step_df["head"] == "logits_only_posterior"]["mrr"].mean()
        ),
        "top5_hit": float(
            step_df[step_df["head"] == "logits_only_posterior"]["top5_hit"].mean()
        ),
    }

    summary_by_head: Dict[str, Any] = {}
    compare_rows: List[Dict[str, Any]] = []
    for head_name, head_df in step_df.groupby("head"):
        head_summary = summarise_head(head_df.copy(), raw_reasoner_baseline)
        summary_by_head[head_name] = head_summary
        compare_rows.append({"head": head_name, **head_summary})
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(output_dir / "belief_head_compare.csv", index=False)

    current = summary_by_head["current_fused_posterior"]
    calibrated = summary_by_head["calibrated_fused_posterior"]
    logits_only = summary_by_head["logits_only_posterior"]

    judgment = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "reasoner_asset": str(frozen_checkpoint),
        "head_definitions": {
            "logits_only_posterior": {
                "distribution": "masked_softmax(reasoner_logits / T)",
                "temperature": float(logits_temperature),
            },
            "current_fused_posterior": current_fused,
            "calibrated_fused_posterior": calibrated_fused,
        },
        "acceptability_criteria": {
            "ranking_preservation_mrr_ratio_floor": float(args.material_ranking_preservation_floor),
            "mass80_cluster_ratio_ceiling": float(args.material_cluster_ratio_ceiling),
            "reward_delta_snr_floor": float(args.material_hard_delta_snr_floor),
            "notes": [
                "Belief head must preserve most of strongest reasoner ranking.",
                "Belief head must expose tighter usable confusion structure than current broad cluster counts.",
                "Delta signals must be stable enough to serve as later reward components.",
            ],
        },
        "current_belief_issues": {
            "mrr_abs_delta_vs_logits": float(current["mrr_abs_delta_vs_reasoner"]),
            "mrr_relative_degradation": float(1.0 - current["ranking_preservation_mrr_ratio"]),
            "top5_abs_delta_vs_logits": float(current["top5_abs_delta_vs_reasoner"]),
            "top5_relative_degradation": float(1.0 - current["ranking_preservation_top5_ratio"]),
            "normalized_entropy_mean": float(current["normalized_entropy_mean"]),
            "mass80_cluster_count_ratio_mean": float(current["mass80_cluster_count_ratio_mean"]),
            "delta_true_mass_snr": float(current["delta_true_mass_snr"]),
            "delta_entropy_snr": float(current["delta_entropy_snr"]),
            "delta_margin_snr": float(current["delta_margin_snr"]),
        },
        "recommended_head": None,
        "acceptability_decision": None,
    }

    if (
        calibrated["ranking_preservation_mrr_ratio"] >= float(args.material_ranking_preservation_floor)
        and calibrated["mass80_cluster_count_ratio_mean"] <= float(args.material_cluster_ratio_ceiling)
        and max(
            float(calibrated["delta_true_mass_snr"] or 0.0),
            float(calibrated["delta_entropy_snr"] or 0.0),
            float(calibrated["delta_margin_snr"] or 0.0),
        ) >= float(args.material_hard_delta_snr_floor)
    ):
        judgment["recommended_head"] = "calibrated_fused_posterior"
        judgment["acceptability_decision"] = "conditionally_acceptable"
    elif (
        logits_only["ranking_preservation_mrr_ratio"] >= float(args.material_ranking_preservation_floor)
        and logits_only["mass80_cluster_count_ratio_mean"] <= float(args.material_cluster_ratio_ceiling)
    ):
        judgment["recommended_head"] = "logits_only_posterior"
        judgment["acceptability_decision"] = "acceptable_but_missing_fused_state_benefits"
    else:
        judgment["recommended_head"] = "none_yet"
        judgment["acceptability_decision"] = "not_yet_acceptable"

    write_json(output_dir / "summary.json", judgment)


if __name__ == "__main__":
    main()
