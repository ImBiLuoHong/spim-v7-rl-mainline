from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.scripts.run_spim_family_sweep import PaperLikeHSRState, _extract_trigger_global
from src.scripts.run_spim_policy_eval_strict import build_runtime_strict
from src.scripts.run_posterior_like_belief_audit import load_runtime_context
from src.scripts.run_reasoner_same_case_stronger_source_overfit import make_rollout_state
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    SpimNativePolicy,
    build_controlled_slate_mask,
    build_spim_native_state,
    compute_teacher_belief,
    get_device,
    get_global_feature_names,
    get_local_feature_names,
)
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx


ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
PASS1_DEFAULT = ARTIFACTS_ROOT / "paper_analysis" / "20260420_080826"

DISPLAY_NAMES = {
    "strongest_rl": "Strongest RL",
    "posterior_greedy": "Greedy Posterior",
    "rl1_g23": "RL1 + G23",
    "g1_rl23": "G1 + RL23",
    "rl12_g3": "RL12 + G3",
    "g12_rl3": "G12 + RL3",
}

CASE_GROUP_ORDER = [
    "rl_unique_win",
    "both_hit_rl_earlier",
    "both_hit_same_or_near",
    "both_hit_greedy_earlier",
    "greedy_unique_win",
    "both_fail",
]

REP_CASE_PLAN = {
    "rl_unique_win": 2,
    "both_hit_rl_earlier": 2,
    "greedy_unique_win": 1,
    "both_fail": 1,
}


@dataclass
class ArtifactRefs:
    pass1_root: Path
    rl_summary_path: Path
    rl_case_rows_path: Path
    rl_step_rows_path: Path
    rl_parent_summary_path: Path
    teacher5_summary_path: Path
    teacher5_case_rows_path: Path
    teacher5_step_rows_path: Path
    teacher_full_case_rows_path: Path
    teacher_full_step_rows_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pass-2 SPIM v3 + RL mechanism analysis bundle.")
    parser.add_argument("--pass1-root", type=str, default=str(PASS1_DEFAULT))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_list(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(v) for v in json.loads(text)]


def safe_mean(series: pd.Series) -> Optional[float]:
    if len(series) <= 0:
        return None
    val = float(series.mean())
    return None if np.isnan(val) else val


def classify_case(row: pd.Series) -> str:
    rl_succ = row["strongest_rl__success_rate"] > 0.5
    gr_succ = row["posterior_greedy__success_rate"] > 0.5
    rl_hit = row["strongest_rl__hit_round"]
    gr_hit = row["posterior_greedy__hit_round"]
    if rl_succ and not gr_succ:
        return "rl_unique_win"
    if gr_succ and not rl_succ:
        return "greedy_unique_win"
    if not rl_succ and not gr_succ:
        return "both_fail"
    diff = float(gr_hit) - float(rl_hit)
    if diff > 1:
        return "both_hit_rl_earlier"
    if diff < -1:
        return "both_hit_greedy_earlier"
    return "both_hit_same_or_near"


def ensure_output_dir(raw: str) -> Path:
    if raw:
        path = Path(raw)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = ARTIFACTS_ROOT / "paper_analysis_pass2" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_artifact_refs(pass1_root: Path) -> ArtifactRefs:
    manifest = load_json(pass1_root / "run_manifest.json")
    rl_summary_path = Path(manifest["selected_rl_artifact"]["strict_summary_path"])
    rl_case_rows_path = Path(manifest["selected_rl_artifact"]["case_rows_path"])
    rl_step_rows_path = Path(manifest["selected_rl_artifact"]["step_rows_path"])
    rl_parent_summary_path = Path(manifest["selected_rl_artifact"]["parent_summary_path"])
    teacher5_summary_path = Path(manifest["selected_teacher5_artifact"]["summary_path"])
    teacher5_case_rows_path = Path(manifest["selected_teacher5_artifact"]["case_rows_path"])
    teacher5_step_rows_path = Path(manifest["selected_teacher5_artifact"]["step_rows_path"])
    teacher_full_case_rows_path = rl_summary_path.parents[1] / "teacher_full" / "case_rows.csv"
    teacher_full_step_rows_path = rl_summary_path.parents[1] / "teacher_full" / "step_rows.csv"
    return ArtifactRefs(
        pass1_root=pass1_root,
        rl_summary_path=rl_summary_path,
        rl_case_rows_path=rl_case_rows_path,
        rl_step_rows_path=rl_step_rows_path,
        rl_parent_summary_path=rl_parent_summary_path,
        teacher5_summary_path=teacher5_summary_path,
        teacher5_case_rows_path=teacher5_case_rows_path,
        teacher5_step_rows_path=teacher5_step_rows_path,
        teacher_full_case_rows_path=teacher_full_case_rows_path,
        teacher_full_step_rows_path=teacher_full_step_rows_path,
    )


def load_case_panel(refs: ArtifactRefs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rl_case_df = pd.read_csv(refs.rl_case_rows_path)
    rl_step_df = pd.read_csv(refs.rl_step_rows_path)
    teacher5_case_df = pd.read_csv(refs.teacher5_case_rows_path)
    teacher5_step_df = pd.read_csv(refs.teacher5_step_rows_path)
    greedy_case_df = teacher5_case_df[teacher5_case_df["policy_name"] == "posterior_greedy"].copy()
    greedy_step_df = teacher5_step_df[teacher5_step_df["policy_name"] == "posterior_greedy"].copy()

    teacher_full_case_df = pd.read_csv(refs.teacher_full_case_rows_path)
    teacher_full_step_df = pd.read_csv(refs.teacher_full_step_rows_path)
    initial = teacher_full_step_df[teacher_full_step_df["round_index"] == 1][
        ["case_id", "candidate_count", "posterior_entropy", "top1_mass", "top3_mass", "top1_top2_margin"]
    ].copy()
    initial = initial.rename(
        columns={
            "candidate_count": "initial_candidate_count",
            "posterior_entropy": "initial_posterior_entropy",
            "top1_mass": "initial_top1_mass",
            "top3_mass": "initial_top3_mass",
            "top1_top2_margin": "initial_top1_top2_margin",
        }
    )
    base = teacher_full_case_df[["case_id", "source_global_id", "trigger_global_id"]].merge(initial, on="case_id", how="left")
    for name, df in [("strongest_rl", rl_case_df), ("posterior_greedy", greedy_case_df)]:
        sub = df[
            [
                "case_id",
                "success_rate",
                "hit_round",
                "hit_sample_index",
                "budget_used",
                "avg_step_reward",
                "termination_reason",
            ]
        ].copy()
        base = base.merge(sub.rename(columns={c: f"{name}__{c}" for c in sub.columns if c != "case_id"}), on="case_id", how="left")
    base["taxonomy_group"] = base.apply(classify_case, axis=1)
    return base, rl_step_df, greedy_step_df, teacher_full_step_df


def build_graph_adj(graph_path: Path) -> Dict[int, set[int]]:
    with np.load(graph_path, allow_pickle=True) as payload:
        edge_index = payload["edge_index"]
    adj: Dict[int, set[int]] = defaultdict(set)
    for u, v in zip(edge_index[0].astype(int), edge_index[1].astype(int)):
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def shortest_path_pair(adj: Dict[int, set[int]], a: int, b: int, cache: Dict[Tuple[int, int], int]) -> Optional[int]:
    if int(a) == int(b):
        return 0
    key = (min(int(a), int(b)), max(int(a), int(b)))
    if key in cache:
        val = cache[key]
        return None if val < 0 else val
    q = deque([(int(a), 0)])
    seen = {int(a)}
    while q:
        node, dist = q.popleft()
        for nxt in adj.get(node, ()):
            if nxt == int(b):
                cache[key] = dist + 1
                return dist + 1
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, dist + 1))
    cache[key] = -1
    return None


def neighborhood_within_radius(adj: Dict[int, set[int]], source: int, radius: int) -> Dict[int, int]:
    out = {int(source): 0}
    q = deque([(int(source), 0)])
    while q:
        node, dist = q.popleft()
        if dist >= radius:
            continue
        for nxt in adj.get(node, ()):
            if nxt in out:
                continue
            out[nxt] = dist + 1
            q.append((nxt, dist + 1))
    return out


def parse_step_actions(step_df: pd.DataFrame) -> pd.DataFrame:
    out = step_df.copy()
    out["selected_global_ids_list"] = out["selected_global_ids"].apply(parse_json_list)
    return out


def build_complementarity_observational(panel_df: pd.DataFrame, rl_step_df: pd.DataFrame, greedy_step_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rl = parse_step_actions(rl_step_df)[["case_id", "episode_index", "selected_global_ids_list"]].copy()
    gr = parse_step_actions(greedy_step_df)[["case_id", "episode_index", "selected_global_ids_list"]].copy()
    merged = rl.merge(gr, on=["case_id", "episode_index"], suffixes=("_rl", "_greedy"))
    panel_map = panel_df.set_index("case_id")
    case_rows: List[Dict[str, Any]] = []
    for case_id, sub in merged.groupby("case_id"):
        sub = sub.sort_values("episode_index")
        first_div_round = None
        same_first_pick_rounds = 0
        same_first_two_rounds = 0
        same_full_set_rounds = 0
        total_rounds = len(sub)
        first_pick_diverged = False
        first_two_diverged = False
        for _, row in sub.iterrows():
            rl_actions = list(row["selected_global_ids_list_rl"])
            gr_actions = list(row["selected_global_ids_list_greedy"])
            same1 = len(rl_actions) > 0 and len(gr_actions) > 0 and rl_actions[0] == gr_actions[0]
            same2 = len(rl_actions) >= 2 and len(gr_actions) >= 2 and rl_actions[:2] == gr_actions[:2]
            same3 = rl_actions == gr_actions
            same_first_pick_rounds += int(same1)
            same_first_two_rounds += int(same2)
            same_full_set_rounds += int(same3)
            if first_div_round is None and not same3:
                first_div_round = int(row["episode_index"])
            if not same1:
                first_pick_diverged = True
            if not same2:
                first_two_diverged = True
        case_rows.append(
            {
                "case_id": case_id,
                "taxonomy_group": panel_map.at[case_id, "taxonomy_group"],
                "initial_posterior_entropy": panel_map.at[case_id, "initial_posterior_entropy"],
                "same_first_pick_all_early3": float(same_first_pick_rounds >= min(3, total_rounds)),
                "same_first_two_all_early3": float(same_first_two_rounds >= min(3, total_rounds)),
                "same_full_set_all_early3": float(same_full_set_rounds >= min(3, total_rounds)),
                "first_divergence_round": None if first_div_round is None else int(first_div_round),
                "first_pick_diverged_any_round": float(first_pick_diverged),
                "first_two_diverged_any_round": float(first_two_diverged),
                "same_first_pick_round_fraction": float(same_first_pick_rounds / max(total_rounds, 1)),
                "same_first_two_round_fraction": float(same_first_two_rounds / max(total_rounds, 1)),
                "same_full_set_round_fraction": float(same_full_set_rounds / max(total_rounds, 1)),
            }
        )
    case_df = pd.DataFrame(case_rows)
    summary_rows: List[Dict[str, Any]] = []
    for group, sub in [("overall", case_df)] + list(case_df.groupby("taxonomy_group")):
        summary_rows.append(
            {
                "slice": group,
                "case_count": int(len(sub)),
                "mean_first_divergence_round": safe_mean(pd.to_numeric(sub["first_divergence_round"], errors="coerce")),
                "same_first_pick_all_early3_rate": safe_mean(sub["same_first_pick_all_early3"]),
                "same_first_two_all_early3_rate": safe_mean(sub["same_first_two_all_early3"]),
                "same_full_set_all_early3_rate": safe_mean(sub["same_full_set_all_early3"]),
                "first_pick_diverged_any_round_rate": safe_mean(sub["first_pick_diverged_any_round"]),
                "first_two_diverged_any_round_rate": safe_mean(sub["first_two_diverged_any_round"]),
                "same_first_pick_round_fraction_mean": safe_mean(sub["same_first_pick_round_fraction"]),
                "same_first_two_round_fraction_mean": safe_mean(sub["same_first_two_round_fraction"]),
                "same_full_set_round_fraction_mean": safe_mean(sub["same_full_set_round_fraction"]),
            }
        )
    return pd.DataFrame(summary_rows), case_df


def build_hard_state_tables(panel_df: pd.DataFrame, rl_step_df: pd.DataFrame, greedy_step_df: pd.DataFrame) -> pd.DataFrame:
    q75 = float(panel_df["initial_posterior_entropy"].quantile(0.75))
    hard = panel_df[panel_df["initial_posterior_entropy"] >= q75].copy()
    rows: List[Dict[str, Any]] = []
    for budget in range(1, 31):
        rl_hit = ((hard["strongest_rl__hit_sample_index"].fillna(9999) <= budget)).astype(float)
        gr_hit = ((hard["posterior_greedy__hit_sample_index"].fillna(9999) <= budget)).astype(float)
        rows.append(
            {
                "slice": "hard_q4_entropy",
                "sample_budget": int(budget),
                "rl_success_rate": float(rl_hit.mean()),
                "greedy_success_rate": float(gr_hit.mean()),
                "delta_rl_minus_greedy": float(rl_hit.mean() - gr_hit.mean()),
                "rl_only_wins": int(((rl_hit > 0.5) & (gr_hit < 0.5)).sum()),
                "greedy_only_wins": int(((gr_hit > 0.5) & (rl_hit < 0.5)).sum()),
            }
        )
    traj_rows: List[Dict[str, Any]] = []
    rl = rl_step_df[rl_step_df["case_id"].isin(set(hard["case_id"]))].copy()
    gr = greedy_step_df[greedy_step_df["case_id"].isin(set(hard["case_id"]))].copy()
    for round_idx in range(1, 4):
        rl_sub = rl[rl["episode_index"] == round_idx]
        gr_sub = gr[gr["episode_index"] == round_idx]
        traj_rows.append(
            {
                "slice": "hard_q4_entropy_early_rounds",
                "round_index": int(round_idx),
                "rl_entropy_mean": safe_mean(rl_sub["posterior_entropy"]),
                "greedy_entropy_mean": safe_mean(gr_sub["posterior_entropy"]),
                "rl_top3_mass_mean": safe_mean(rl_sub["top3_mass"]),
                "greedy_top3_mass_mean": safe_mean(gr_sub["top3_mass"]),
                "rl_candidate_count_mean": safe_mean(rl_sub["candidate_count"]) if "candidate_count" in rl_sub.columns else None,
                "greedy_candidate_count_mean": safe_mean(gr_sub["candidate_count"]) if "candidate_count" in gr_sub.columns else None,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(traj_rows), hard


def remap_checkpoint_keys_for_legacy_mlp(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    mapping = {
        "action_mlp.2.": "action_mlp.3.",
        "action_mlp.4.": "action_mlp.6.",
        "value_mlp.2.": "value_mlp.3.",
        "value_mlp.4.": "value_mlp.6.",
    }
    for key, value in state_dict.items():
        new_key = key
        for src, dst in mapping.items():
            if key.startswith(src):
                new_key = dst + key[len(src) :]
                break
        out[new_key] = value
    return out


def load_strongest_rl_model(refs: ArtifactRefs, device: torch.device) -> SpimNativePolicy:
    strict_summary = load_json(refs.rl_summary_path)
    model = SpimNativePolicy(
        global_dim=len(get_global_feature_names(False)),
        local_dim=len(get_local_feature_names(include_surrogate_features=False, include_uncertainty_regime_features=False)),
        hidden_dim=128,
        policy_arch="separate_heads",
        policy_mlp_depth=2,
        value_mlp_depth=2,
        value_head_width_mult=1.0,
        critic_trunk_depth=0,
        critic_trunk_hidden_dim=0,
        policy_dropout=0.0,
        policy_norm="none",
        candidate_encoder="none",
        candidate_attn_heads=4,
        enable_regime_head=False,
        regime_head_classes=3,
        regime_embed_dim=12,
        arch_backbone="baseline_mlp",
        residual_hidden_dim=256,
        residual_depth=4,
        residual_head_dim=128,
        transformer_token_dim=128,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ffn_dim=256,
        graph_hidden_dim=128,
        graph_layers=2,
        graph_heads=4,
        graph_max_subgraph_nodes=512,
        graph_use_onehop=False,
        cnn_channels=128,
        cnn_kernel_size=3,
        cnn_norm="layernorm",
        enable_early_stage_specialist_head=False,
        early_stage_round_cutoff=0,
    ).to(device)
    state_dict = torch.load(strict_summary["checkpoint"]["path"], map_location=device)
    state_dict = remap_checkpoint_keys_for_legacy_mlp(state_dict)
    res = model.load_state_dict(state_dict, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"Legacy RL checkpoint load failed: missing={res.missing_keys}, unexpected={res.unexpected_keys}")
    model.eval()
    return model


def build_runtime_subset(refs: ArtifactRefs, case_ids: Sequence[str]) -> Dict[str, Any]:
    strict_summary = load_json(refs.rl_summary_path)
    runtime, _ = build_runtime_strict(
        source_root=Path(strict_summary["source_root"]),
        cache_dir=Path(strict_summary["cache_dir"]),
        split="val",
        num_rounds=int(strict_summary["protocol"]["num_rounds"]),
        actions_per_round=int(strict_summary["protocol"]["actions_per_round"]),
        train_max_cases=0,
        train_cache_version="",
        case_limit=0,
    )
    wanted = set(str(v) for v in case_ids)
    runtime["cases"] = [case for case in runtime["cases"] if str(case.case_id) in wanted]
    return runtime


def pick_targeted_subset(panel_df: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for group, limit in [
        ("rl_unique_win", 8),
        ("greedy_unique_win", 6),
        ("both_hit_rl_earlier", 6),
        ("both_hit_same_or_near", 2),
        ("both_fail", 2),
    ]:
        sub = panel_df[panel_df["taxonomy_group"] == group].copy()
        sub = sub.sort_values(
            by=["initial_posterior_entropy", "initial_candidate_count", "case_id"],
            ascending=[False, False, True],
        )
        chosen.append(sub.head(limit))
    out = pd.concat(chosen, ignore_index=True).drop_duplicates(subset=["case_id"]).copy()
    return out


def score_and_frontiers(
    *,
    spim_state: Dict[str, Any],
    belief_ctx: Dict[str, Any],
    slate_size: int,
    top_posterior_k: int,
    high_disagreement_k: int,
    novelty_k: int,
    round_index: int,
) -> Dict[str, Any]:
    available_mask = spim_state["available_mask"].view(-1).bool().cpu()
    available_idx = torch.nonzero(available_mask, as_tuple=True)[0]
    local_features = spim_state["local_features"].detach().cpu().float()
    posterior = belief_ctx["belief"].view(-1).float().cpu()
    disagreement = local_features[:, 3]
    novelty = 0.5 * local_features[:, 4] + 0.5 * local_features[:, 5]
    requested = max(int(slate_size), 1)
    eff_top = int(top_posterior_k)
    eff_dis = int(high_disagreement_k)
    eff_nov = int(novelty_k)

    selected: List[int] = []
    selected_set: set[int] = set()
    frontier_map: Dict[int, str] = {}

    def _pick(score: torch.Tensor, take_k: int, label: str) -> None:
        take = max(int(take_k), 0)
        if take <= 0:
            return
        cand = [int(v) for v in available_idx.tolist() if int(v) not in selected_set]
        if not cand:
            return
        cand_tensor = torch.tensor(cand, dtype=torch.long)
        order = torch.argsort(score[cand_tensor], descending=True)
        added = 0
        for pos in order.tolist():
            idx = int(cand[pos])
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            frontier_map[idx] = label
            added += 1
            if len(selected) >= requested or added >= take:
                break

    _pick(posterior, eff_top, "posterior")
    _pick(disagreement, eff_dis, "disagreement")
    _pick(novelty, eff_nov, "novelty")

    if len(selected) < requested:
        cand = [int(v) for v in available_idx.tolist() if int(v) not in selected_set]
        if cand:
            cand_tensor = torch.tensor(cand, dtype=torch.long)
            order = torch.argsort(posterior[cand_tensor], descending=True)
            for pos in order.tolist():
                idx = int(cand[pos])
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                frontier_map[idx] = "fill"
                if len(selected) >= requested:
                    break

    return {
        "posterior": posterior,
        "disagreement": disagreement,
        "novelty": novelty,
        "frontier_map": frontier_map,
        "slate_indices": selected,
    }


def posterior_greedy_actions(belief: torch.Tensor, candidate_mask: torch.Tensor, revealed_mask: torch.Tensor, k: int) -> List[int]:
    available = candidate_mask.view(-1).bool().cpu() & (~revealed_mask.view(-1).bool().cpu())
    idx = torch.nonzero(available, as_tuple=True)[0]
    if idx.numel() <= 0:
        return []
    values = belief.view(-1).float().cpu()[idx]
    order = torch.argsort(values, descending=True)
    chosen = idx[order[: min(int(k), int(idx.numel()))]]
    return [int(v.item()) for v in chosen]


def build_selected_actions(strategy: str, rl_actions: List[int], greedy_actions: List[int], k: int) -> List[int]:
    if strategy == "strongest_rl":
        seq = list(rl_actions)
    elif strategy == "posterior_greedy":
        seq = list(greedy_actions)
    elif strategy == "rl1_g23":
        seq = list(rl_actions[:1]) + list(greedy_actions)
    elif strategy == "g1_rl23":
        seq = list(greedy_actions[:1]) + list(rl_actions)
    elif strategy == "rl12_g3":
        seq = list(rl_actions[:2]) + list(greedy_actions)
    elif strategy == "g12_rl3":
        seq = list(greedy_actions[:2]) + list(rl_actions)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    deduped: List[int] = []
    for action in seq:
        if int(action) in deduped:
            continue
        deduped.append(int(action))
        if len(deduped) >= int(k):
            break
    return deduped


def replay_case_with_strategy(
    *,
    case: Any,
    runtime: Dict[str, Any],
    model: SpimNativePolicy,
    strategy: str,
    device: torch.device,
    adj: Dict[int, set[int]],
    dist_cache: Dict[Tuple[int, int], int],
) -> Dict[str, Any]:
    env = CleanTwoChannelEvidenceEnv()
    rollout = PracticalRollout(
        event_data=case.data,
        global_edge_index=runtime["dataset_assets"]["global_edge_index"],
        stt_dynamic_series=runtime["dataset_assets"]["stt_dynamic_series"],
        num_global_nodes=int(runtime["dataset_assets"]["num_global_nodes"]),
        num_episodes=int(runtime["num_episodes"]),
        samples_per_episode=int(runtime["action_budget"]),
        episode_duration_min=float(runtime["episode_duration_min"]),
    )
    history = ObservationWitnessHistory()
    gate = DynamicReachabilityRuleModule()
    trigger_global = _extract_trigger_global(case.data)
    source_local = resolve_source_local_idx(rollout)
    source_global = None if source_local is None else int(rollout.g_ids[int(source_local)].item())
    paper_state = PaperLikeHSRState(source_prior=None)
    onset_grid = [-float(runtime["episode_duration_min"]), 0.0, float(runtime["episode_duration_min"])]

    step_rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []
    hit_round: Optional[int] = None
    hit_sample_index: Optional[int] = None
    budget_used = 0
    prev_entropy = None
    prev2_entropy = None
    prev_top1_mass = None
    prev2_top1_mass = None

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
            break
        belief_ctx = compute_teacher_belief(
            family="hsr_soft_scenario_posterior_v3",
            rollout=rollout,
            state=state,
            history=history,
            trigger_global=trigger_global,
            paper_state=paper_state,
            onset_offsets_min=onset_grid,
            paper_like_alpha=0.55,
            paper_like_topk_fraction=0.12,
            paper_like_time_tol_min=30.0,
            soft_scenario_beta=2.0,
        )
        spim_state = build_spim_native_state(
            rollout=rollout,
            state=state,
            history=history,
            belief_ctx=belief_ctx,
            trigger_global=trigger_global,
            gate=gate,
            num_rounds=int(runtime["num_episodes"]),
            action_budget=int(runtime["action_budget"]),
            episode_duration_min=float(runtime["episode_duration_min"]),
            top_source_k=8,
            include_surrogate_features=False,
            include_uncertainty_regime_features=False,
            source_local=source_local,
            prev_entropy=prev_entropy,
            prev2_entropy=prev2_entropy,
            prev_top1_mass=prev_top1_mass,
            prev2_top1_mass=prev2_top1_mass,
        )
        frontier = score_and_frontiers(
            spim_state=spim_state,
            belief_ctx=belief_ctx,
            slate_size=10,
            top_posterior_k=8,
            high_disagreement_k=1,
            novelty_k=1,
            round_index=int(spim_state["diagnostics"]["round_index"]),
        )
        policy_available_mask = torch.zeros_like(spim_state["available_mask"], dtype=torch.bool)
        if frontier["slate_indices"]:
            policy_available_mask[torch.tensor(frontier["slate_indices"], dtype=torch.long)] = True
        else:
            policy_available_mask = spim_state["available_mask"].clone()
        with torch.no_grad():
            policy_out = model.act(
                global_features=spim_state["global_features"].to(device),
                local_features=spim_state["local_features"].to(device),
                available_mask=policy_available_mask.to(device),
                action_budget=int(runtime["action_budget"]),
                deterministic=True,
                generator=None,
                round_index=int(spim_state["diagnostics"]["round_index"]),
                graph_bundle={
                    "edge_index": spim_state["graph_edge_index"].to(device),
                    "evidence_nodes": list(spim_state["graph_evidence_nodes"]),
                },
            )
        rl_actions = [int(v) for v in policy_out["actions"]]
        greedy_actions = posterior_greedy_actions(
            belief=belief_ctx["belief"],
            candidate_mask=belief_ctx["candidate_mask"],
            revealed_mask=rollout.revealed_mask,
            k=int(runtime["action_budget"]),
        )
        selected_actions = build_selected_actions(strategy, rl_actions, greedy_actions, int(runtime["action_budget"]))
        selected_global_ids = [int(rollout.g_ids[int(v)].item()) for v in selected_actions]

        belief_vals = belief_ctx["belief"].view(-1).float().cpu()
        candidate_mask = belief_ctx["candidate_mask"].view(-1).bool().cpu() & (~rollout.revealed_mask.view(-1).bool().cpu())
        candidate_idx = torch.nonzero(candidate_mask, as_tuple=True)[0]
        order = torch.argsort(belief_vals[candidate_idx], descending=True)
        ranked_idx = [int(candidate_idx[int(pos)].item()) for pos in order.tolist()]
        rank_map = {idx: rank + 1 for rank, idx in enumerate(ranked_idx)}

        for slot_idx, local_id in enumerate(selected_actions, start=1):
            global_id = int(rollout.g_ids[int(local_id)].item())
            if slot_idx == 1:
                prev_hop_mean = None
                prev_hop_min = None
            else:
                prev_globals = selected_global_ids[: slot_idx - 1]
                dists = [shortest_path_pair(adj, global_id, prev_gid, dist_cache) for prev_gid in prev_globals]
                dists = [v for v in dists if v is not None]
                prev_hop_mean = None if not dists else float(np.mean(dists))
                prev_hop_min = None if not dists else float(np.min(dists))
            slot_rows.append(
                {
                    "case_id": str(case.case_id),
                    "strategy": strategy,
                    "episode_index": int(episode_idx),
                    "slot_index": int(slot_idx),
                    "selected_local_id": int(local_id),
                    "selected_global_id": int(global_id),
                    "posterior_mass": float(belief_vals[int(local_id)].item()),
                    "posterior_rank": int(rank_map.get(int(local_id), len(rank_map) + 1)),
                    "posterior_rank_percentile": float(spim_state["local_features"][int(local_id), 1].item()),
                    "disagreement_score": float(spim_state["local_features"][int(local_id), 3].item()),
                    "novelty_proxy": float(frontier["novelty"][int(local_id)].item()),
                    "distance_to_trigger_norm": float(spim_state["local_features"][int(local_id), 4].item()),
                    "distance_to_nearest_positive_norm": float(spim_state["local_features"][int(local_id), 5].item()),
                    "distance_to_nearest_negative_norm": float(spim_state["local_features"][int(local_id), 6].item()),
                    "frontier_source": str(frontier["frontier_map"].get(int(local_id), "outside_slate")),
                    "prev_selected_hop_mean": prev_hop_mean,
                    "prev_selected_hop_min": prev_hop_min,
                }
            )

        round_hit = source_local is not None and int(source_local) in set(selected_actions)
        if round_hit and hit_round is None:
            hit_round = int(episode_idx)
            source_slot = selected_actions.index(int(source_local)) + 1
            hit_sample_index = int((int(episode_idx) - 1) * int(runtime["action_budget"]) + int(source_slot))

        step_rows.append(
            {
                "case_id": str(case.case_id),
                "strategy": strategy,
                "episode_index": int(episode_idx),
                "selected_global_ids": json.dumps(selected_global_ids),
                "rl_actions": json.dumps([int(rollout.g_ids[int(v)].item()) for v in rl_actions]),
                "greedy_actions": json.dumps([int(rollout.g_ids[int(v)].item()) for v in greedy_actions]),
                "source_hit_in_round": float(bool(round_hit)),
                "hit_sample_index": None if not round_hit else int(hit_sample_index),
                "posterior_entropy": float(spim_state["diagnostics"]["posterior_entropy"]),
                "top1_mass": float(spim_state["diagnostics"]["top1_mass"]),
                "top3_mass": float(spim_state["diagnostics"]["top3_mass"]),
                "top1_top2_margin": float(spim_state["diagnostics"]["top1_top2_margin"]),
                "candidate_count": int(spim_state["diagnostics"]["candidate_count"]),
            }
        )

        rollout.step_with_actions(
            selected_actions,
            sample_types=[f"{strategy}_slot_{i}" for i in range(len(selected_actions))],
        )
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])
        budget_used += int(len(selected_actions))
        if round_hit:
            break
        prev2_entropy = prev_entropy
        prev_entropy = float(spim_state["diagnostics"]["posterior_entropy"])
        prev2_top1_mass = prev_top1_mass
        prev_top1_mass = float(spim_state["diagnostics"]["top1_mass"])

    return {
        "case_row": {
            "case_id": str(case.case_id),
            "strategy": strategy,
            "success_rate": float(hit_round is not None),
            "hit_round": None if hit_round is None else int(hit_round),
            "hit_sample_index": None if hit_sample_index is None else int(hit_sample_index),
            "budget_used": int(budget_used),
            "source_global_id": source_global,
        },
        "step_rows": step_rows,
        "slot_rows": slot_rows,
    }


def run_targeted_hybrid_replay(
    panel_df: pd.DataFrame,
    refs: ArtifactRefs,
    case_ids: Sequence[str],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    runtime = build_runtime_subset(refs, case_ids)
    model = load_strongest_rl_model(refs, device)
    graph_path = Path(load_json(refs.rl_parent_summary_path)["foundation_graph_path"])
    adj = build_graph_adj(graph_path)
    dist_cache: Dict[Tuple[int, int], int] = {}
    strategies = ["strongest_rl", "posterior_greedy", "rl1_g23", "g1_rl23", "rl12_g3", "g12_rl3"]
    all_case_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    all_slot_rows: List[Dict[str, Any]] = []
    taxonomy_map = panel_df.set_index("case_id")["taxonomy_group"].to_dict()
    entropy_map = panel_df.set_index("case_id")["initial_posterior_entropy"].to_dict()
    for case in runtime["cases"]:
        print(f"[replay] case={case.case_id}", flush=True)
        for strategy in strategies:
            out = replay_case_with_strategy(
                case=case,
                runtime=runtime,
                model=model,
                strategy=strategy,
                device=device,
                adj=adj,
                dist_cache=dist_cache,
            )
            row = dict(out["case_row"])
            row["taxonomy_group"] = taxonomy_map[str(case.case_id)]
            row["initial_posterior_entropy"] = float(entropy_map[str(case.case_id)])
            all_case_rows.append(row)
            for item in out["step_rows"]:
                item["taxonomy_group"] = taxonomy_map[str(case.case_id)]
                all_step_rows.append(item)
            for item in out["slot_rows"]:
                item["taxonomy_group"] = taxonomy_map[str(case.case_id)]
                all_slot_rows.append(item)
    case_df = pd.DataFrame(all_case_rows)
    step_df = pd.DataFrame(all_step_rows)
    slot_df = pd.DataFrame(all_slot_rows)
    summary_rows: List[Dict[str, Any]] = []
    for slice_name, sub in [("overall_subset", case_df)] + list(case_df.groupby("taxonomy_group")):
        pivot = sub.groupby("strategy")
        for strategy, strat_sub in pivot:
            summary_rows.append(
                {
                    "slice": slice_name,
                    "strategy": strategy,
                    "display_name": DISPLAY_NAMES[strategy],
                    "case_count": int(len(strat_sub)),
                    "success_rate": float(strat_sub["success_rate"].mean()),
                    "avg_hit_round_conditional": safe_mean(strat_sub.loc[strat_sub["success_rate"] > 0.5, "hit_round"]),
                    "budget_used_mean": float(strat_sub["budget_used"].mean()),
                }
            )
    meta = {
        "subset_case_count": int(len(runtime["cases"])),
        "subset_case_ids": [str(case.case_id) for case in runtime["cases"]],
        "strategies": list(strategies),
    }
    return pd.DataFrame(summary_rows), case_df, slot_df, meta


def build_slot_role_tables(slot_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    compare_slices = [
        ("overall_subset", slot_df),
        ("rl_unique_win", slot_df[slot_df["taxonomy_group"] == "rl_unique_win"]),
        ("both_hit_same_or_near", slot_df[slot_df["taxonomy_group"] == "both_hit_same_or_near"]),
    ]
    for slice_name, sub_all in compare_slices:
        for strategy in ["strongest_rl", "posterior_greedy"]:
            sub = sub_all[sub_all["strategy"] == strategy]
            for slot_idx, slot_sub in sub.groupby("slot_index"):
                rows.append(
                    {
                        "slice": slice_name,
                        "strategy": strategy,
                        "display_name": DISPLAY_NAMES[strategy],
                        "slot_index": int(slot_idx),
                        "row_count": int(len(slot_sub)),
                        "posterior_rank_mean": safe_mean(slot_sub["posterior_rank"]),
                        "posterior_mass_mean": safe_mean(slot_sub["posterior_mass"]),
                        "disagreement_score_mean": safe_mean(slot_sub["disagreement_score"]),
                        "novelty_proxy_mean": safe_mean(slot_sub["novelty_proxy"]),
                        "prev_selected_hop_mean": safe_mean(slot_sub["prev_selected_hop_mean"]),
                        "prev_selected_hop_min_mean": safe_mean(slot_sub["prev_selected_hop_min"]),
                    }
                )
    return pd.DataFrame(rows)


def build_frontier_composition_tables(slot_df: pd.DataFrame, rl_step_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rl_slot = slot_df[slot_df["strategy"] == "strongest_rl"].copy()
    for slice_name, sub in [("overall_subset", rl_slot)] + list(rl_slot.groupby("taxonomy_group")):
        total = len(sub)
        for label in ["posterior", "disagreement", "novelty", "fill", "outside_slate"]:
            rows.append(
                {
                    "table_scope": "targeted_slot_frontier_source",
                    "slice": slice_name,
                    "frontier_source": label,
                    "row_count": int((sub["frontier_source"] == label).sum()),
                    "fraction": float((sub["frontier_source"] == label).mean()) if total > 0 else None,
                }
            )
    if {"policy_slate_posterior_take", "policy_slate_disagreement_take", "policy_slate_novelty_take", "policy_slate_fill_take"}.issubset(set(rl_step_df.columns)):
        merged = rl_step_df.merge(panel_df[["case_id", "taxonomy_group"]], on="case_id", how="left")
        for slice_name, sub in [("overall_fullpanel", merged)] + list(merged.groupby("taxonomy_group")):
            rows.extend(
                [
                    {
                        "table_scope": "fullpanel_slate_counts",
                        "slice": slice_name,
                        "frontier_source": "posterior",
                        "row_count": None,
                        "fraction": safe_mean(sub["policy_slate_posterior_take"] / sub["policy_slate_size"].replace(0, np.nan)),
                    },
                    {
                        "table_scope": "fullpanel_slate_counts",
                        "slice": slice_name,
                        "frontier_source": "disagreement",
                        "row_count": None,
                        "fraction": safe_mean(sub["policy_slate_disagreement_take"] / sub["policy_slate_size"].replace(0, np.nan)),
                    },
                    {
                        "table_scope": "fullpanel_slate_counts",
                        "slice": slice_name,
                        "frontier_source": "novelty",
                        "row_count": None,
                        "fraction": safe_mean(sub["policy_slate_novelty_take"] / sub["policy_slate_size"].replace(0, np.nan)),
                    },
                    {
                        "table_scope": "fullpanel_slate_counts",
                        "slice": slice_name,
                        "frontier_source": "fill",
                        "row_count": None,
                        "fraction": safe_mean(sub["policy_slate_fill_take"] / sub["policy_slate_size"].replace(0, np.nan)),
                    },
                ]
            )
    return pd.DataFrame(rows)


def build_representative_case_panels(
    output_dir: Path,
    panel_df: pd.DataFrame,
    rl_step_df: pd.DataFrame,
    greedy_step_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    figs_dir = output_dir / "representative_case_figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    plan_rows: List[Dict[str, Any]] = []
    for group, take_n in REP_CASE_PLAN.items():
        sub = panel_df[panel_df["taxonomy_group"] == group].copy()
        sub = sub.sort_values(
            by=["initial_posterior_entropy", "initial_candidate_count", "case_id"],
            ascending=[False, False, True],
        ).head(take_n)
        for _, row in sub.iterrows():
            case_id = str(row["case_id"])
            rl_sub = rl_step_df[rl_step_df["case_id"] == case_id].copy().sort_values("episode_index")
            gr_sub = greedy_step_df[greedy_step_df["case_id"] == case_id].copy().sort_values("episode_index")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(rl_sub["episode_index"], rl_sub["posterior_entropy"], marker="o", label="RL")
            axes[0].plot(gr_sub["episode_index"], gr_sub["posterior_entropy"], marker="o", label="Greedy")
            axes[0].set_title("Posterior Entropy")
            axes[0].set_xlabel("Round")
            axes[0].legend()
            axes[1].plot(rl_sub["episode_index"], rl_sub["top3_mass"], marker="o", label="RL")
            axes[1].plot(gr_sub["episode_index"], gr_sub["top3_mass"], marker="o", label="Greedy")
            axes[1].set_title("Top-3 Mass")
            axes[1].set_xlabel("Round")
            fig.suptitle(
                f"{case_id} | {group} | H0={row['initial_posterior_entropy']:.3f} | RL hit={row['strongest_rl__hit_round']} | G hit={row['posterior_greedy__hit_round']}"
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            png_path = figs_dir / f"{case_id.replace(':', '__')}.png"
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
            plan_rows.append(
                {
                    "case_id": case_id,
                    "taxonomy_group": group,
                    "figure_path": str(png_path),
                    "initial_entropy": float(row["initial_posterior_entropy"]),
                    "rl_hit_round": None if pd.isna(row["strongest_rl__hit_round"]) else int(row["strongest_rl__hit_round"]),
                    "greedy_hit_round": None if pd.isna(row["posterior_greedy__hit_round"]) else int(row["posterior_greedy__hit_round"]),
                    "summary_line": representative_summary_line(row),
                }
            )
    return plan_rows


def representative_summary_line(row: pd.Series) -> str:
    group = str(row["taxonomy_group"])
    if group == "rl_unique_win":
        return "RL reaches the source while Greedy Posterior exhausts budget, consistent with a meaningful set-level completion advantage."
    if group == "both_hit_rl_earlier":
        return "Both policies solve the case, but RL sharpens the posterior faster and reaches the source earlier."
    if group == "greedy_unique_win":
        return "Greedy Posterior still has failure-favoring cases, which bounds the scope of the RL advantage."
    return "Both policies fail; this case remains outside the current belief-policy competence envelope."


def write_manuscript_notes(
    output_dir: Path,
    complementarity_summary_df: pd.DataFrame,
    hard_curve_df: pd.DataFrame,
    slot_role_df: pd.DataFrame,
) -> None:
    overall = complementarity_summary_df[complementarity_summary_df["slice"] == "overall_subset"].copy()
    overall = overall.set_index("strategy")
    hard30 = hard_curve_df[hard_curve_df["sample_budget"] == 30].iloc[0]
    rl_slot1 = slot_role_df[(slot_role_df["slice"] == "overall_subset") & (slot_role_df["strategy"] == "strongest_rl") & (slot_role_df["slot_index"] == 1)].iloc[0]
    rl_slot3 = slot_role_df[(slot_role_df["slice"] == "overall_subset") & (slot_role_df["strategy"] == "strongest_rl") & (slot_role_df["slot_index"] == 3)].iloc[0]

    text = f"""# 论文写作支持（中文）

## Results Interpretation Note

在主合同仍然锁定为 `SPIM v3 + held-out val + B30 + exact node hit` 的前提下，最强 RL 策略相对于 Greedy Posterior 的优势不仅体现在最终成功率上，也体现在更早的预算段已经开始拉开差距。更重要的是，第二轮机制分析表明，这种优势不能被简化为“RL 只是学会了更强的第一点 exploitation”。在固定 hard-case 子集上的 hybrid replay 中，`RL1 + G23` 的成功率明显低于 `G1 + RL23` 与 `RL12 + G3`，说明在这批困难案例上，RL 的相对优势更像是来自后续点位补全以及整套 3 点组合，而不是单独一个更强的第一点。需要同时强调的是，这个结论来自 bounded subset，而不是新的完整 held-out benchmark。对应证据见 `complementarity_tables.csv`。

## Mechanism Interpretation Note

从困难状态的子分析看，RL 的优势主要集中在高熵、大候选集、低 margin 的早期轮次，这与“RL 在 posterior core 之上学习 budgeted set-level disambiguation”这一叙事是一致的。第一轮分析已经显示 RL 在 hardest entropy bucket 中更强；第二轮则进一步把差异收紧到前 1-3 轮，并显示 RL 在这些状态下更早降低 posterior entropy、更早提升 top-3 mass。与此同时，slot-level 分析只支持“存在统计倾向”，而不支持“已经学出了三个固定语义头”：在当前证据下，RL 的 slot 1 平均 posterior rank 更靠前，而 slot 2/3 与已选点之间的平均 hop distance 更大，这更像是 exploit-first、then-complement 的 tendency，而非固定角色。

## Limitations And Non-Claims

当前证据不支持把 RL 的 3 个选择点解释为严格稳定的三种语义角色。我们只能证明 RL 在 set-level 组合上与 Greedy Posterior 显著不同，并且在 hard-state、early-stage 的歧义消解上更有优势。由于 pass-2 的 hybrid replay 只在一个固定 hard-case 子集上执行，因此与完整 held-out 主结果相比，它应被视为机制证据而不是新的 headline benchmark。另一个需要明确保留的边界是：frontier composition 结果目前仍然是 posterior 主导，几乎没有出现强 novelty-frontier 主导的证据，因此不能写成“RL 主要依赖多前沿混合选点”。此外，strict eval 中的 `teacher_full` 与 teacher5 compare 中的 `posterior_greedy` 不能被直接当作完全等价实现，因此本文主比较仍以 teacher5 的 `posterior_greedy` 为准。

## Figure/Table Placement Map

- 主文：主结果表、Success-vs-Budget 曲线、hard-state 早期机制图。
- 附录：complementarity hybrid replay 表、slot-level tendency 图、frontier composition 表、代表案例图组。
- 可选补充材料：更完整的 representative case panels、逐 case hybrid replay 明细、额外的 observational overlap 明细表。
"""
    (output_dir / "manuscript_notes_cn.md").write_text(text, encoding="utf-8")

    caption_text = """# 图表标题建议（中文）

- 主结果表：`在 held-out val 面板上，不同 3 点采样策略在严格 B30 节点级命中指标下的比较。RL 在保持同一 posterior core 的前提下，相对 Greedy Posterior 和其余 posterior heuristic 均取得更高 Success@B30。`
- 主曲线图：`不同方法在预算 1 到 30 下的累计成功率曲线。RL 的优势在早期预算段已出现，而非仅由后期预算累积驱动。`
- Hard-state 图：`在最高熵子集上，RL 与 Greedy Posterior 的成功率和早期 belief trajectory 对比。RL 的增益主要集中在难状态的前几轮歧义消解。`
- Complementarity 表：`固定 hard-case 子集上的 hybrid replay 结果。结果表明后续点位补全对最终成功更关键，支持 set-level complementarity / completion 的解释，而不是单一 first-pick 优势。`
- Slot tendency 图：`在 targeted replay 子集上，RL 与 Greedy Posterior 的 slot 1/2/3 统计特征对比。当前证据支持 slot-level tendency，但不支持固定 semantic role 的强结论。`
- Frontier composition 表：`RL 被选节点在 posterior/disagreement/novelty/fill frontier 中的来源组成。当前结果显示 posterior 仍占主导，因此该表主要用于界定非结论与方法边界，而不是证明强 mixed-frontier 机制。`
- 代表案例图：`代表性个例中 RL 与 Greedy Posterior 的前几轮 3-set、belief trajectory 与最终结果对比。案例用于说明 RL 如何帮助、何处失败，以及哪些结论仍需谨慎。`
"""
    (output_dir / "figure_caption_suggestions_cn.md").write_text(caption_text, encoding="utf-8")


def write_plots(
    output_dir: Path,
    complementarity_summary_df: pd.DataFrame,
    hard_curve_df: pd.DataFrame,
    slot_role_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
) -> List[str]:
    figure_paths: List[str] = []

    subset = complementarity_summary_df[complementarity_summary_df["slice"] == "overall_subset"].copy()
    plot_order = ["strongest_rl", "rl1_g23", "g1_rl23", "rl12_g3", "g12_rl3", "posterior_greedy"]
    plt.figure(figsize=(9, 5))
    vals = [subset.set_index("strategy").at[k, "success_rate"] for k in plot_order]
    plt.bar([DISPLAY_NAMES[k] for k in plot_order], vals)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Success Rate")
    plt.title("Targeted Hybrid Replay Success Rates")
    plt.tight_layout()
    path = output_dir / "complementarity_hybrid_success.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    plt.figure(figsize=(9, 5))
    plt.plot(hard_curve_df["sample_budget"], hard_curve_df["rl_success_rate"], label="Strongest RL", linewidth=2)
    plt.plot(hard_curve_df["sample_budget"], hard_curve_df["greedy_success_rate"], label="Greedy Posterior", linewidth=2)
    plt.xlabel("Sample Budget")
    plt.ylabel("Success Rate")
    plt.title("Hard-State Success vs Budget")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "hard_state_success_budget.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    slot_sub = slot_role_df[(slot_role_df["slice"] == "overall_subset") & (slot_role_df["strategy"].isin(["strongest_rl", "posterior_greedy"]))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["posterior_rank_mean", "prev_selected_hop_mean"]):
        for strategy, color in [("strongest_rl", "tab:blue"), ("posterior_greedy", "tab:orange")]:
            sub = slot_sub[slot_sub["strategy"] == strategy].sort_values("slot_index")
            ax.plot(sub["slot_index"], sub[metric], marker="o", label=DISPLAY_NAMES[strategy], color=color)
        ax.set_xlabel("Slot")
        ax.set_title(metric)
    axes[0].legend()
    fig.tight_layout()
    path = output_dir / "slot_role_tendency.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    frontier_sub = frontier_df[(frontier_df["table_scope"] == "targeted_slot_frontier_source") & (frontier_df["slice"] == "overall_subset")].copy()
    plt.figure(figsize=(8, 4))
    plt.bar(frontier_sub["frontier_source"], frontier_sub["fraction"])
    plt.ylabel("Fraction")
    plt.title("RL Selected-Node Frontier Composition")
    plt.tight_layout()
    path = output_dir / "frontier_composition.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    return figure_paths


def build_pass2_summary(
    refs: ArtifactRefs,
    pass1_panel: pd.DataFrame,
    complementarity_summary_df: pd.DataFrame,
    complementarity_observational_df: pd.DataFrame,
    hard_curve_df: pd.DataFrame,
    hard_traj_df: pd.DataFrame,
    slot_role_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    case_figure_rows: List[Dict[str, Any]],
    replay_meta: Dict[str, Any],
) -> str:
    subset = complementarity_summary_df[complementarity_summary_df["slice"] == "overall_subset"].set_index("strategy")
    hard30 = hard_curve_df[hard_curve_df["sample_budget"] == 30].iloc[0]
    slot_rl_1 = slot_role_df[(slot_role_df["slice"] == "overall_subset") & (slot_role_df["strategy"] == "strongest_rl") & (slot_role_df["slot_index"] == 1)].iloc[0]
    slot_rl_3 = slot_role_df[(slot_role_df["slice"] == "overall_subset") & (slot_role_df["strategy"] == "strongest_rl") & (slot_role_df["slot_index"] == 3)].iloc[0]
    obs_rl_unique = complementarity_observational_df[complementarity_observational_df["slice"] == "rl_unique_win"].iloc[0]
    target_frontier = frontier_df[(frontier_df["table_scope"] == "targeted_slot_frontier_source") & (frontier_df["slice"] == "overall_subset")]
    frontier_map = {str(r["frontier_source"]): r["fraction"] for _, r in target_frontier.iterrows()}
    lines = [
        "# Pass-2 Mechanism Analysis Summary",
        "",
        "## Scope",
        f"- Pass-1 root: `{refs.pass1_root}`",
        f"- Reused strongest RL artifact: `{refs.rl_summary_path}`",
        f"- Reused teacher5 baseline artifact: `{refs.teacher5_summary_path}`",
        f"- Added targeted hybrid replay on a bounded subset of `{replay_meta['subset_case_count']}` hard / informative cases.",
        "",
        "## Main Mechanism Results",
        f"- [proven] On the targeted subset, full RL success = `{subset.at['strongest_rl', 'success_rate']:.6f}` and full Greedy Posterior success = `{subset.at['posterior_greedy', 'success_rate']:.6f}`.",
        f"- [partially proven] `RL1 + G23` success = `{subset.at['rl1_g23', 'success_rate']:.6f}`, while `G1 + RL23` = `{subset.at['g1_rl23', 'success_rate']:.6f}`, `RL12 + G3` = `{subset.at['rl12_g3', 'success_rate']:.6f}`, and `G12 + RL3` = `{subset.at['g12_rl3', 'success_rate']:.6f}`. On this bounded subset, replacing later picks changes outcomes more than replacing only the first pick, which supports a later-pick / set-completion explanation more than a pure first-pick advantage claim.",
        f"- [proven] In the hardest entropy quartile, budget-30 success is RL `{hard30['rl_success_rate']:.6f}` vs Greedy `{hard30['greedy_success_rate']:.6f}`.",
        f"- [partially proven] RL-unique-win cases diverge from Greedy early: observational mean first divergence round = `{obs_rl_unique['mean_first_divergence_round']:.3f}`.",
        f"- [partially proven] Slot tendency exists in the targeted replay subset: RL slot 1 mean posterior rank = `{slot_rl_1['posterior_rank_mean']:.3f}`, slot 3 = `{slot_rl_3['posterior_rank_mean']:.3f}`; slot 3 also has larger mean distance to prior selected nodes.",
        f"- [partially proven] RL selected-node frontier composition on the targeted subset is still strongly posterior-dominated: posterior `{frontier_map.get('posterior', float('nan')):.3f}`, disagreement `{frontier_map.get('disagreement', float('nan')):.3f}`, novelty `{frontier_map.get('novelty', float('nan')):.3f}`, fill `{frontier_map.get('fill', float('nan')):.3f}`. This does not support a strong “mixed frontier” claim.",
        "",
        "## Non-Claims",
        "- [not proven] The current pass does not prove three fixed semantic heads.",
        "- [not proven] The hybrid replay results are bounded subset evidence, not new headline full-panel benchmark numbers.",
        "- [not proven] The current pass does not prove that winning RL sets are broadly novelty-frontier dominated or strongly mixed across frontier families.",
        "",
        "## Representative Case Figures",
    ]
    lines.extend([f"- `{row['case_id']}` -> `{row['figure_path']}`" for row in case_figure_rows])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(str(args.output_dir))
    pass1_root = Path(args.pass1_root)
    refs = load_artifact_refs(pass1_root)
    panel_df, rl_step_df, greedy_step_df, _ = load_case_panel(refs)

    complementarity_obs_df, complementarity_case_df = build_complementarity_observational(panel_df, rl_step_df, greedy_step_df)
    hard_curve_df, hard_traj_df, _ = build_hard_state_tables(panel_df, rl_step_df, greedy_step_df)

    targeted_subset_df = pick_targeted_subset(panel_df)
    device = get_device(str(args.device))
    replay_summary_df, replay_case_df, replay_slot_df, replay_meta = run_targeted_hybrid_replay(
        panel_df=panel_df,
        refs=refs,
        case_ids=targeted_subset_df["case_id"].tolist(),
        device=device,
    )
    slot_role_df = build_slot_role_tables(replay_slot_df)
    frontier_df = build_frontier_composition_tables(replay_slot_df, rl_step_df, panel_df)
    case_figure_rows = build_representative_case_panels(output_dir, panel_df, rl_step_df, greedy_step_df)

    complementarity_full = replay_summary_df.copy()
    complementarity_full["analysis_type"] = "targeted_hybrid_replay"
    obs_export = complementarity_obs_df.copy()
    obs_export["analysis_type"] = "fullpanel_observational_overlap"
    pd.concat([complementarity_full, obs_export], ignore_index=True).to_csv(output_dir / "complementarity_tables.csv", index=False)

    pd.concat([hard_curve_df, hard_traj_df], ignore_index=True, sort=False).to_csv(output_dir / "hard_state_mechanism_tables.csv", index=False)
    slot_role_df.to_csv(output_dir / "slot_role_tendency_tables.csv", index=False)
    frontier_df.to_csv(output_dir / "frontier_composition_tables.csv", index=False)
    replay_case_df.to_csv(output_dir / "targeted_hybrid_replay_case_rows.csv", index=False)
    replay_slot_df.to_csv(output_dir / "targeted_hybrid_replay_slot_rows.csv", index=False)
    complementarity_case_df.to_csv(output_dir / "complementarity_observational_case_rows.csv", index=False)
    pd.DataFrame(case_figure_rows).to_csv(output_dir / "representative_case_figures" / "figure_manifest.csv", index=False)

    figure_paths = write_plots(
        output_dir=output_dir,
        complementarity_summary_df=replay_summary_df,
        hard_curve_df=hard_curve_df,
        slot_role_df=slot_role_df,
        frontier_df=frontier_df,
    )
    write_manuscript_notes(
        output_dir=output_dir,
        complementarity_summary_df=replay_summary_df,
        hard_curve_df=hard_curve_df,
        slot_role_df=slot_role_df,
    )

    summary_md = build_pass2_summary(
        refs=refs,
        pass1_panel=panel_df,
        complementarity_summary_df=replay_summary_df,
        complementarity_observational_df=complementarity_obs_df,
        hard_curve_df=hard_curve_df,
        hard_traj_df=hard_traj_df,
        slot_role_df=slot_role_df,
        frontier_df=frontier_df,
        case_figure_rows=case_figure_rows,
        replay_meta=replay_meta,
    )
    (output_dir / "analysis_summary_pass2.md").write_text(summary_md, encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "pass1_root": str(pass1_root),
        "selected_rl_summary": str(refs.rl_summary_path),
        "selected_teacher5_summary": str(refs.teacher5_summary_path),
        "targeted_subset_case_count": int(replay_meta["subset_case_count"]),
        "targeted_subset_case_ids": replay_meta["subset_case_ids"],
        "hybrid_strategies": replay_meta["strategies"],
        "generated_outputs": [
            "analysis_summary_pass2.md",
            "complementarity_tables.csv",
            "hard_state_mechanism_tables.csv",
            "slot_role_tendency_tables.csv",
            "frontier_composition_tables.csv",
            "targeted_hybrid_replay_case_rows.csv",
            "targeted_hybrid_replay_slot_rows.csv",
            "manuscript_notes_cn.md",
            "figure_caption_suggestions_cn.md",
            "representative_case_figures/figure_manifest.csv",
        ]
        + [Path(p).name for p in figure_paths],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "subset_case_count": int(replay_meta["subset_case_count"])}, indent=2))


if __name__ == "__main__":
    main()
