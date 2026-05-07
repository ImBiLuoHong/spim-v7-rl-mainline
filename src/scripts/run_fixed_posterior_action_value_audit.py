from __future__ import annotations

import argparse
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.belief_updaters.evidence_posterior_like import _evidence_contrast_scalar, _masked_zscore
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.modeling.loop.navigator_vnext_contract import build_candidate_semantics, default_reward_contract, tensor_attr
from src.modeling.navigators.clean_v1 import pick_topk_valid
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


DEFAULT_ACCEPTABILITY_ROOT = PROJECT_ROOT / "artifacts" / "posterior_like_belief_acceptability_audit" / "20260407_exact136_belief_acceptability_v1"
DEFAULT_POLICY_ROOT = PROJECT_ROOT / "artifacts" / "posterior_to_policy_readiness_audit" / "20260407_exact136_policy_readiness_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "fixed_posterior_action_value_audit" / "20260407_exact136_action_value_v1"
RUNNER_VERSION = "fixed_posterior_action_value_audit_v1"
PANEL_VERSION = "exact136_train_only_fixed_posterior_action_value_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded one-step action-value validity audit under fixed calibrated posterior.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--policy-root", type=str, default=str(DEFAULT_POLICY_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--max-round", type=int, default=4)
    parser.add_argument("--states-per-round", type=int, default=24)
    parser.add_argument("--action-pool-cap", type=int, default=16)
    parser.add_argument("--topk-each-source", type=int, default=4)
    return parser.parse_args()


def _safe_softmax(scores: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    out = torch.zeros_like(scores.view(-1).float())
    if not bool(mask.any()):
        return out
    out[mask] = torch.softmax(scores[mask] / max(float(temperature), 1e-6), dim=0)
    return out


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 3:
        return None
    val = sub.iloc[:, 0].corr(sub.iloc[:, 1], method="spearman")
    return None if pd.isna(val) else float(val)


def _load_calibrated_params(acceptability_root: Path) -> Dict[str, float]:
    payload = json.loads((acceptability_root / "summary.json").read_text())
    return dict(payload["head_definitions"]["calibrated_fused_posterior"])


def _load_summary_threshold(policy_root: Path) -> float:
    payload = json.loads((policy_root / "summary.json").read_text())
    return float(payload["summary_families"]["mass_cover"]["threshold"])


def _contract_cfg() -> Dict[str, float]:
    cfg = default_reward_contract()
    cfg.update({"support_plausible_delta": 0.25, "not_ruled_out_threshold": 0.5})
    return cfg


def _compute_posterior_from_state(
    *,
    state: Dict[str, Any],
    reasoner_module,
    params: Dict[str, float],
    mass_cover_threshold: float,
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
    contrast = _evidence_contrast_scalar(payload["node_features"].cpu(), valid_mask)

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

    valid_idx = torch.nonzero(candidate_mask, as_tuple=True)[0]
    vals = probs[valid_idx]
    order = torch.argsort(vals, descending=True)
    sorted_idx = valid_idx[order]
    sorted_vals = vals[order]

    entropy = float((-(sorted_vals.clamp_min(1e-12) * torch.log(sorted_vals.clamp_min(1e-12)))).sum().item()) if len(sorted_vals) else 0.0
    source_local = resolve_source_local_idx(state["rollout"])
    source_rank = None
    source_mass = None
    if source_local is not None and bool(candidate_mask[int(source_local)].item()):
        pos = (sorted_idx == int(source_local)).nonzero(as_tuple=True)[0]
        if pos.numel():
            source_rank = int(pos[0].item()) + 1
            source_mass = float(probs[int(source_local)].item())
    hard_local = None
    hard_mass = None
    for idx in sorted_idx.tolist():
        if source_local is None or int(idx) != int(source_local):
            hard_local = int(idx)
            hard_mass = float(probs[int(idx)].item())
            break
    margin = None
    if source_mass is not None and hard_mass is not None:
        margin = float(source_mass - hard_mass)

    # mass-cover summary
    csum = torch.cumsum(sorted_vals, dim=0)
    mass_idx = int((csum >= float(mass_cover_threshold)).nonzero(as_tuple=True)[0][0].item()) + 1 if len(sorted_vals) else 0
    mass_cover_set = set(sorted_idx[:mass_idx].tolist())
    mass_cover_mass = float(csum[mass_idx - 1].item()) if mass_idx > 0 else 0.0

    return {
        "logits": logits,
        "probs": probs,
        "valid_mask": valid_mask,
        "candidate_mask": candidate_mask,
        "support_score": state["support_score"].view(-1).float().cpu(),
        "q_score": q_score,
        "contradiction_score": contradiction,
        "contrast_signal": contrast,
        "entropy": entropy,
        "source_local": source_local,
        "source_rank": source_rank,
        "source_mass": source_mass,
        "hard_local": hard_local,
        "hard_mass": hard_mass,
        "margin_true_vs_hard": margin,
        "top_order": sorted_idx.tolist(),
        "candidate_count": int(candidate_mask.float().sum().item()),
        "mass_cover_threshold": float(mass_cover_threshold),
        "mass_cover_set": mass_cover_set,
        "mass_cover_size": int(len(mass_cover_set)),
        "mass_cover_size_ratio": float(len(mass_cover_set) / max(int(candidate_mask.float().sum().item()), 1)),
        "mass_cover_mass": mass_cover_mass,
    }


def _state_selection(policy_root: Path, max_round: int, states_per_round: int) -> pd.DataFrame:
    df = pd.read_csv(policy_root / "policy_readiness_step_rows.csv")
    early = df[df["episode_index"] <= int(max_round)].copy()
    # Prefer hard early states: source not yet top1, wide summary, high uncertainty
    early["hard_score"] = (
        (1.0 - early["top1_hit"].fillna(0.0)) * 10.0
        + early["mass_cover_size_ratio"].fillna(0.0) * 5.0
        + early["normalized_entropy"].fillna(0.0) * 2.0
    )
    chosen = []
    for round_idx in range(1, int(max_round) + 1):
        sub = early[early["episode_index"] == round_idx].sort_values(
            ["hard_score", "candidate_count"], ascending=[False, False]
        )
        chosen.append(sub.head(int(states_per_round)))
    out = pd.concat(chosen, ignore_index=True).drop_duplicates(subset=["case_id", "episode_index"])
    return out


def _build_action_pool(
    *,
    posterior: Dict[str, Any],
    topk_each: int,
    cap: int,
    rng: random.Random,
) -> List[int]:
    mask = posterior["candidate_mask"]
    support = posterior["support_score"]
    q_score = posterior["q_score"]
    probs = posterior["probs"]
    ordered_valid = posterior["top_order"]

    pool: List[int] = []

    def add_many(items: List[int]) -> None:
        for item in items:
            if int(item) not in pool and bool(mask[int(item)].item()):
                pool.append(int(item))
                if len(pool) >= int(cap):
                    return

    add_many(pick_topk_valid(support, mask, int(topk_each)))
    add_many(pick_topk_valid(q_score, mask, int(topk_each)))
    add_many([int(v) for v in ordered_valid[: int(topk_each)]])

    # summary-derived: first nodes inside mass-cover set, sorted by posterior
    mass_cover_sorted = [int(v) for v in ordered_valid if int(v) in posterior["mass_cover_set"]]
    add_many(mass_cover_sorted[: int(topk_each)])

    valid_idx = [int(v) for v in torch.nonzero(mask, as_tuple=True)[0].tolist() if int(v) not in pool]
    rng.shuffle(valid_idx)
    add_many(valid_idx[: int(topk_each)])

    # Guarantee hardest confuser if not already there
    if posterior["hard_local"] is not None:
        add_many([int(posterior["hard_local"])])

    return pool[: int(cap)]


def _simulate_one_step(
    *,
    case: Any,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    candidate: int,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    runtime: Dict[str, Any],
    reasoner_module,
    posterior_params: Dict[str, float],
    mass_cover_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    sim_rollout = deepcopy(rollout)
    sim_history = deepcopy(history)
    sim_rollout.step_with_actions([int(candidate)], sample_types=["one_step_audit"])
    if sim_rollout.history_steps:
        sim_history.append_from_history_step(sim_rollout.history_steps[-1])
    post_state = make_rollout_state(
        case=case,
        rollout=sim_rollout,
        history=sim_history,
        env=env,
        topology=topology,
        num_episodes=runtime["num_episodes"],
        action_budget=runtime["action_budget"],
        frontier_role_mode=runtime["frontier_role_mode"],
    )
    return _compute_posterior_from_state(
        state=post_state,
        reasoner_module=reasoner_module,
        params=posterior_params,
        mass_cover_threshold=mass_cover_threshold,
        device=device,
    )


def _fit_grouped_models(action_df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = action_df.dropna(subset=feature_cols + [target_col]).copy()
    groups = df["state_key"].astype(str).values
    X = df[feature_cols].values
    y = df[target_col].values
    if len(set(groups)) < 5:
        raise RuntimeError("Not enough grouped states for bounded learnability check.")
    gkf = GroupKFold(n_splits=5)
    preds_linear = pd.Series(index=df.index, dtype=float)
    preds_mlp = pd.Series(index=df.index, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups):
        linear = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        mlp = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=400, random_state=45))])
        linear.fit(X[train_idx], y[train_idx])
        mlp.fit(X[train_idx], y[train_idx])
        preds_linear.iloc[test_idx] = linear.predict(X[test_idx])
        preds_mlp.iloc[test_idx] = mlp.predict(X[test_idx])
    pred_df = df.copy()
    pred_df["pred_linear"] = preds_linear
    pred_df["pred_mlp"] = preds_mlp

    rows = []
    for name in ["pred_linear", "pred_mlp"]:
        per_state = []
        for _, sub in pred_df.groupby("state_key"):
            corr = _spearman(sub[name], sub[target_col])
            pred_top = sub.sort_values(name, ascending=False).iloc[0]["action_local"]
            true_top = sub.sort_values(target_col, ascending=False).iloc[0]["action_local"]
            per_state.append({"spearman": corr, "top1_match": float(pred_top == true_top)})
        stat = pd.DataFrame(per_state)
        rows.append(
            {
                "model": name.replace("pred_", ""),
                "state_spearman_mean": float(stat["spearman"].dropna().mean()),
                "state_spearman_median": float(stat["spearman"].dropna().median()),
                "state_top1_match_rate": float(stat["top1_match"].mean()),
            }
        )
    return pred_df, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    source_root = Path(args.source_root)
    contrast_root = Path(args.contrast_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)
    policy_root = Path(args.policy_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    _, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)
    posterior_params = _load_calibrated_params(acceptability_root)
    mass_cover_threshold = _load_summary_threshold(policy_root)

    selected_states = _state_selection(policy_root, int(args.max_round), int(args.states_per_round))
    selected_states.to_csv(output_dir / "selected_state_manifest.csv", index=False)
    state_keys = {(str(r.case_id), int(r.episode_index)) for r in selected_states.itertuples(index=False)}

    env = CleanTwoChannelEvidenceEnv()
    topology = runtime["dataset_assets"]["topology"]
    action_rows: List[Dict[str, Any]] = []
    pool_rows: List[Dict[str, Any]] = []

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
            ep = int(step.round_index) + 1
            state_key = (case.case_id, ep)
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
            if state_key in state_keys:
                pre = _compute_posterior_from_state(
                    state=state,
                    reasoner_module=reasoner_module,
                    params=posterior_params,
                    mass_cover_threshold=mass_cover_threshold,
                    device=device,
                )
                rng = random.Random((hash(case.case_id) ^ ep ^ int(args.seed)) & 0xFFFFFFFF)
                pool = _build_action_pool(
                    posterior=pre,
                    topk_each=int(args.topk_each_source),
                    cap=int(args.action_pool_cap),
                    rng=rng,
                )
                pool_rows.append(
                    {
                        "case_id": case.case_id,
                        "episode_index": ep,
                        "candidate_count": pre["candidate_count"],
                        "pool_size": len(pool),
                        "pool_contains_hardest_confuser": float(pre["hard_local"] in set(pool) if pre["hard_local"] is not None else 0.0),
                        "pool_contains_source": float(pre["source_local"] in set(pool) if pre["source_local"] is not None else 0.0),
                    }
                )
                for action_local in pool:
                    post = _simulate_one_step(
                        case=case,
                        rollout=rollout,
                        history=history,
                        candidate=int(action_local),
                        env=env,
                        topology=topology,
                        runtime=runtime,
                        reasoner_module=reasoner_module,
                        posterior_params=posterior_params,
                        mass_cover_threshold=mass_cover_threshold,
                        device=device,
                    )
                    action_rows.append(
                        {
                            "state_key": f"{case.case_id}@{ep}",
                            "case_id": case.case_id,
                            "episode_index": ep,
                            "candidate_count": pre["candidate_count"],
                            "pre_true_mass": pre["source_mass"],
                            "pre_entropy": pre["entropy"],
                            "pre_margin_true_vs_hard": pre["margin_true_vs_hard"],
                            "pre_mass_cover_size_ratio": pre["mass_cover_size_ratio"],
                            "action_local": int(action_local),
                            "action_support_score": float(pre["support_score"][int(action_local)].item()),
                            "action_q_score": float(pre["q_score"][int(action_local)].item()),
                            "action_posterior_mass": float(pre["probs"][int(action_local)].item()),
                            "action_contradiction": float(pre["contradiction_score"][int(action_local)].item()),
                            "action_contrast": float(pre["contrast_signal"][int(action_local)].item()),
                            "action_gap_to_top1": float(pre["probs"][pre["top_order"][0]].item() - pre["probs"][int(action_local)].item()),
                            "in_mass_cover_summary": float(int(action_local) in pre["mass_cover_set"]),
                            "is_hardest_confuser": float(pre["hard_local"] is not None and int(action_local) == int(pre["hard_local"])),
                            "next_true_mass": post["source_mass"],
                            "next_entropy": post["entropy"],
                            "next_margin_true_vs_hard": post["margin_true_vs_hard"],
                            "next_mass_cover_size_ratio": post["mass_cover_size_ratio"],
                            "next_source_rank": post["source_rank"],
                            "next_delta_true_mass": (float(post["source_mass"] - pre["source_mass"]) if pre["source_mass"] is not None and post["source_mass"] is not None else None),
                            "next_delta_entropy": float(pre["entropy"] - post["entropy"]),
                            "next_confusion_shrink": float(pre["mass_cover_size_ratio"] - post["mass_cover_size_ratio"]),
                            "source_hit_next_step": float((post["source_rank"] or 10**9) <= 1),
                        }
                    )
            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"oracle_slot_{i}" for i in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])

    action_df = pd.DataFrame(action_rows)
    pool_df = pd.DataFrame(pool_rows)
    action_df["myopic_utility_target"] = (
        (action_df["next_delta_true_mass"] / (action_df["next_delta_true_mass"].std() + 1e-9))
        + 0.5 * (action_df["next_delta_entropy"] / (action_df["next_delta_entropy"].std() + 1e-9))
        + 0.5 * (action_df["next_confusion_shrink"] / (action_df["next_confusion_shrink"].std() + 1e-9))
    )
    action_df.to_csv(output_dir / "action_value_rows.csv", index=False)
    pool_df.to_csv(output_dir / "action_pool_summary.csv", index=False)

    # proxy separability
    proxy_rows = []
    for proxy in ["action_support_score", "action_q_score", "action_posterior_mass", "in_mass_cover_summary"]:
        per_state = []
        for _, sub in action_df.groupby("state_key"):
            corr = _spearman(sub[proxy], sub["myopic_utility_target"])
            pred_top = sub.sort_values(proxy, ascending=False).iloc[0]["action_local"]
            true_top = sub.sort_values("myopic_utility_target", ascending=False).iloc[0]["action_local"]
            per_state.append({"spearman": corr, "top1_match": float(pred_top == true_top)})
        stat = pd.DataFrame(per_state)
        proxy_rows.append(
            {
                "proxy": proxy,
                "state_spearman_mean": float(stat["spearman"].dropna().mean()),
                "state_spearman_median": float(stat["spearman"].dropna().median()),
                "state_top1_match_rate": float(stat["top1_match"].mean()),
            }
        )
    proxy_df = pd.DataFrame(proxy_rows)
    proxy_df.to_csv(output_dir / "proxy_value_compare.csv", index=False)

    feature_cols = [
        "action_support_score",
        "action_q_score",
        "action_posterior_mass",
        "action_contradiction",
        "action_contrast",
        "action_gap_to_top1",
        "in_mass_cover_summary",
        "is_hardest_confuser",
        "candidate_count",
        "pre_entropy",
        "pre_mass_cover_size_ratio",
    ]
    pred_df, learn_df = _fit_grouped_models(action_df, feature_cols, "myopic_utility_target")
    pred_df.to_csv(output_dir / "action_value_predictions.csv", index=False)
    learn_df.to_csv(output_dir / "learnability_summary.csv", index=False)

    target_summary = []
    for col in ["next_delta_true_mass", "next_delta_entropy", "next_confusion_shrink", "source_hit_next_step", "myopic_utility_target"]:
        series = action_df[col].dropna()
        target_summary.append(
            {
                "target": col,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "q25": float(series.quantile(0.25)),
                "q50": float(series.quantile(0.50)),
                "q75": float(series.quantile(0.75)),
            }
        )
    target_df = pd.DataFrame(target_summary)
    target_df.to_csv(output_dir / "target_summary.csv", index=False)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "fixed_posterior_type": "lightly_calibrated_fused_posterior",
        "fixed_posterior_params": posterior_params,
        "state_subset": {
            "max_round": int(args.max_round),
            "states_per_round": int(args.states_per_round),
            "selected_state_count": int(selected_states.shape[0]),
            "round_counts": selected_states.groupby("episode_index").size().to_dict(),
            "selection_rule": "early rounds only; top hard_score states per round, where hard_score prioritizes non-top1, wide mass-cover summary, and high normalized entropy",
        },
        "action_pool": {
            "cap": int(args.action_pool_cap),
            "topk_each_source": int(args.topk_each_source),
            "sources": [
                "current_sampler_top_support",
                "rule_based_q_score",
                "posterior_topk",
                "posterior_mass_cover_members",
                "random_diversity",
            ],
            "pool_size_mean": float(pool_df["pool_size"].mean()) if len(pool_df) else 0.0,
            "contains_hardest_confuser_rate": float(pool_df["pool_contains_hardest_confuser"].mean()) if len(pool_df) else 0.0,
            "contains_source_rate": float(pool_df["pool_contains_source"].mean()) if len(pool_df) else 0.0,
        },
        "targets": {
            "next_delta_true_mass": "post.true_mass - pre.true_mass",
            "next_delta_entropy": "pre.entropy - post.entropy",
            "next_confusion_shrink": "pre.mass_cover_size_ratio - post.mass_cover_size_ratio",
            "source_hit_next_step": "1 if post.source_rank == 1 else 0",
            "myopic_utility_target": "z(next_delta_true_mass) + 0.5*z(next_delta_entropy) + 0.5*z(next_confusion_shrink)",
        },
        "best_proxy": proxy_df.sort_values(["state_spearman_mean", "state_top1_match_rate"], ascending=[False, False]).iloc[0].to_dict() if len(proxy_df) else None,
        "learnability_best_model": learn_df.sort_values(["state_spearman_mean", "state_top1_match_rate"], ascending=[False, False]).iloc[0].to_dict() if len(learn_df) else None,
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
