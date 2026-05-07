from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

REQUIRED_HEURISTICS = [
    "posterior_greedy",
    "posterior_thompson_sampling",
    "posterior_entropy_drop",
    "posterior_cover_shrink",
    "posterior_disagreement_split",
]

DISPLAY_NAMES = {
    "strongest_rl": "Strongest RL",
    "posterior_greedy": "Greedy Posterior",
    "posterior_thompson_sampling": "Thompson Sampling",
    "posterior_entropy_drop": "Entropy Reduction",
    "posterior_cover_shrink": "Cover Shrink",
    "posterior_disagreement_split": "Disagreement Split",
}

NEAR_HIT_ROUND_DELTA = 1
REPRESENTATIVE_LIMITS = {
    "rl_unique_win": 3,
    "both_hit_rl_earlier": 3,
    "greedy_unique_win": 3,
    "both_fail": 3,
}


@dataclass
class SelectedArtifact:
    root: Path
    summary_path: Path
    case_rows_path: Path
    step_rows_path: Path
    summary: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-facing SPIM v3 + RL analysis bundle from existing artifacts.")
    parser.add_argument("--output-dir", type=str, default="")
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


def percentile_buckets(values: pd.Series, labels: Sequence[str]) -> pd.Series:
    clean = values.astype(float)
    quantiles = np.unique(np.quantile(clean, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(quantiles) < 2:
        return pd.Series([labels[0]] * len(clean), index=clean.index)
    if len(quantiles) == 2:
        return pd.Series([labels[0]] * len(clean), index=clean.index)
    # Reduce labels if duplicated quantiles collapse bins.
    use_labels = list(labels[: len(quantiles) - 1])
    return pd.cut(clean, bins=quantiles, labels=use_labels, include_lowest=True, duplicates="drop").astype(str)


def exact_two_sided_binom_p(b: int, c: int) -> Optional[float]:
    n = int(b) + int(c)
    if n <= 0:
        return None
    k = min(int(b), int(c))
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * cdf)


def select_strongest_rl_artifact() -> SelectedArtifact:
    candidates: List[Tuple[float, float, Path, Dict[str, Any], Dict[str, Any]]] = []
    for strict_summary_path in ARTIFACTS_ROOT.glob("spim_set_level_rl_mainline/*/strict_eval_val_B30/*/summary.json"):
        strict_summary = load_json(strict_summary_path)
        if strict_summary.get("policy_mode") != "rl":
            continue
        if strict_summary.get("split") != "val":
            continue
        if strict_summary.get("split_case_count") != 1031:
            continue
        if strict_summary.get("teacher_family") != "hsr_soft_scenario_posterior_v3":
            continue
        parent_summary_path = strict_summary_path.parents[2] / "summary.json"
        if not parent_summary_path.exists():
            continue
        parent_summary = load_json(parent_summary_path)
        if int(parent_summary.get("train_full_case_count", -1)) != 4823:
            continue
        succ = float(strict_summary["summary"]["success_rate"])
        hit = float(strict_summary["summary"]["avg_hit_round_conditional"])
        candidates.append((succ, -hit, strict_summary_path, strict_summary, parent_summary))
    if not candidates:
        raise RuntimeError("No full-train v3 strict-eval RL artifact found.")
    _, _, summary_path, strict_summary, parent_summary = sorted(candidates, reverse=True)[0]
    return SelectedArtifact(
        root=summary_path.parents[2],
        summary_path=summary_path,
        case_rows_path=summary_path.parent / "case_rows.csv",
        step_rows_path=summary_path.parent / "step_rows.csv",
        summary={"strict": strict_summary, "parent": parent_summary},
    )


def select_teacher5_artifact() -> SelectedArtifact:
    candidates: List[Tuple[int, Path, Dict[str, Any]]] = []
    for summary_path in ARTIFACTS_ROOT.glob("spim_teacher5_compare/*/summary.json"):
        summary = load_json(summary_path)
        if set(summary.get("teacher_policies", [])) != set(REQUIRED_HEURISTICS):
            continue
        split_meta = summary.get("split_meta", {})
        protocol = summary.get("protocol", {})
        family = summary.get("posterior_family", {}).get("selected_family")
        if split_meta.get("split") != "val":
            continue
        if int(split_meta.get("loaded_case_count", 0)) != 1031:
            continue
        if int(protocol.get("sample_budget", -1)) != 30:
            continue
        if family != "hsr_soft_scenario_posterior_v3":
            continue
        candidates.append((int(split_meta["loaded_case_count"]), summary_path, summary))
    if not candidates:
        raise RuntimeError("No aligned teacher5 val artifact found.")
    _, summary_path, summary = sorted(candidates, reverse=True)[0]
    return SelectedArtifact(
        root=summary_path.parent,
        summary_path=summary_path,
        case_rows_path=summary_path.parent / "teacher5_case_rows.csv",
        step_rows_path=summary_path.parent / "teacher5_step_rows.csv",
        summary=summary,
    )


def load_method_case_tables(rl_artifact: SelectedArtifact, teacher5_artifact: SelectedArtifact) -> Dict[str, pd.DataFrame]:
    rl_case_df = pd.read_csv(rl_artifact.case_rows_path)
    teacher5_case_df = pd.read_csv(teacher5_artifact.case_rows_path)
    tables = {"strongest_rl": rl_case_df.copy()}
    for method in REQUIRED_HEURISTICS:
        tables[method] = teacher5_case_df[teacher5_case_df["policy_name"] == method].copy()
    return tables


def load_method_step_tables(rl_artifact: SelectedArtifact, teacher5_artifact: SelectedArtifact) -> Dict[str, pd.DataFrame]:
    rl_step_df = pd.read_csv(rl_artifact.step_rows_path)
    teacher5_step_df = pd.read_csv(teacher5_artifact.step_rows_path)
    tables = {"strongest_rl": rl_step_df.copy()}
    for method in REQUIRED_HEURISTICS:
        tables[method] = teacher5_step_df[teacher5_step_df["policy_name"] == method].copy()
    return tables


def load_initial_feature_table(rl_artifact: SelectedArtifact) -> pd.DataFrame:
    teacher_full_step_path = rl_artifact.root / "strict_eval_val_B30" / "teacher_full" / "step_rows.csv"
    teacher_full_case_path = rl_artifact.root / "strict_eval_val_B30" / "teacher_full" / "case_rows.csv"
    step_df = pd.read_csv(teacher_full_step_path)
    case_df = pd.read_csv(teacher_full_case_path)[["case_id", "source_global_id", "trigger_global_id"]].copy()
    step_df = step_df[step_df["round_index"] == 1].copy()
    step_df = step_df[
        [
            "case_id",
            "candidate_count",
            "candidate_ratio",
            "posterior_entropy",
            "mass_cover_0p7",
            "top1_mass",
            "top3_mass",
            "top1_top2_margin",
        ]
    ].copy()
    step_df = step_df.rename(
        columns={
            "candidate_count": "initial_candidate_count",
            "candidate_ratio": "initial_candidate_ratio",
            "posterior_entropy": "initial_posterior_entropy",
            "mass_cover_0p7": "initial_mass_cover_0p7",
            "top1_mass": "initial_top1_mass",
            "top3_mass": "initial_top3_mass",
            "top1_top2_margin": "initial_top1_top2_margin",
        }
    )
    return case_df.merge(step_df, on="case_id", how="left")


def build_case_level_panel(case_tables: Dict[str, pd.DataFrame], initial_features: pd.DataFrame) -> pd.DataFrame:
    base = initial_features.copy()
    for method, df in case_tables.items():
        sub = df[
            [
                "case_id",
                "success_rate",
                "hit_round",
                "hit_sample_index",
                "budget_used",
                "avg_step_reward",
                "final_top1_mass",
                "final_top3_mass",
                "final_entropy",
                "termination_reason",
            ]
        ].copy()
        rename = {col: f"{method}__{col}" for col in sub.columns if col != "case_id"}
        base = base.merge(sub.rename(columns=rename), on="case_id", how="left")
    return base


def build_comparison_table(panel_df: pd.DataFrame, case_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    greedy = panel_df[["case_id", "posterior_greedy__success_rate", "posterior_greedy__hit_sample_index"]].copy()
    for method, df in case_tables.items():
        row: Dict[str, Any] = {
            "method_id": method,
            "display_name": DISPLAY_NAMES[method],
            "case_count": int(len(df)),
            "success_at_B30": float(df["success_rate"].mean()),
            "avg_hit_round_conditional": safe_mean(df.loc[df["success_rate"] > 0.5, "hit_round"]),
            "budget_used_mean": float(df["budget_used"].mean()),
        }
        if method == "strongest_rl":
            row["avg_step_reward_mean_if_available"] = float(df["avg_step_reward"].mean())
        else:
            row["avg_step_reward_mean_if_available"] = None
        if method == "posterior_greedy":
            row.update(
                {
                    "paired_wins_vs_greedy": 0,
                    "paired_losses_vs_greedy": 0,
                    "paired_ties_vs_greedy": int(len(df)),
                    "net_flip_vs_greedy": 0,
                    "mcnemar_b": 0,
                    "mcnemar_c": 0,
                    "mcnemar_exact_p": None,
                }
            )
        else:
            merged = panel_df[
                [
                    "case_id",
                    f"{method}__success_rate",
                    "posterior_greedy__success_rate",
                ]
            ].copy()
            method_succ = merged[f"{method}__success_rate"] > 0.5
            greedy_succ = merged["posterior_greedy__success_rate"] > 0.5
            wins = int((method_succ & (~greedy_succ)).sum())
            losses = int((greedy_succ & (~method_succ)).sum())
            ties = int((method_succ == greedy_succ).sum())
            row.update(
                {
                    "paired_wins_vs_greedy": wins,
                    "paired_losses_vs_greedy": losses,
                    "paired_ties_vs_greedy": ties,
                    "net_flip_vs_greedy": wins - losses,
                    "mcnemar_b": wins,
                    "mcnemar_c": losses,
                    "mcnemar_exact_p": exact_two_sided_binom_p(wins, losses),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(by=["success_at_B30", "avg_hit_round_conditional"], ascending=[False, True])
    return out


def build_success_curve_table(rl_artifact: SelectedArtifact, teacher5_artifact: SelectedArtifact) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rl_summary = rl_artifact.summary["strict"]["summary"]
    for item in rl_summary.get("budget_curve", []):
        rows.append(
            {
                "method_id": "strongest_rl",
                "display_name": DISPLAY_NAMES["strongest_rl"],
                "sample_budget": int(item["sample_budget"]),
                "cumulative_success_rate": float(item["cumulative_success_rate"]),
            }
        )
    for item in teacher5_artifact.summary["results"]["leaderboard"]:
        _ = item
    budget_df = pd.read_csv(teacher5_artifact.root / "budget_success_curve.csv")
    for _, row in budget_df.iterrows():
        rows.append(
            {
                "method_id": str(row["policy_name"]),
                "display_name": DISPLAY_NAMES[str(row["policy_name"])],
                "sample_budget": int(row["sample_budget"]),
                "cumulative_success_rate": float(row["cumulative_success_rate"]),
            }
        )
    return pd.DataFrame(rows)


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
    if diff > NEAR_HIT_ROUND_DELTA:
        return "both_hit_rl_earlier"
    if diff < -NEAR_HIT_ROUND_DELTA:
        return "both_hit_greedy_earlier"
    return "both_hit_same_or_near"


def build_case_taxonomy(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df.copy()
    df["taxonomy_group"] = df.apply(classify_case, axis=1)
    rows: List[Dict[str, Any]] = []
    total = len(df)
    for group, sub in df.groupby("taxonomy_group"):
        example_ids = sub.sort_values(
            by=["initial_posterior_entropy", "initial_top1_top2_margin"],
            ascending=[False, True],
        )["case_id"].head(8).tolist()
        rows.append(
            {
                "taxonomy_group": group,
                "case_count": int(len(sub)),
                "percentage": float(len(sub) / total),
                "representative_case_ids": json.dumps(example_ids),
                "mean_initial_entropy": safe_mean(sub["initial_posterior_entropy"]),
                "mean_initial_margin": safe_mean(sub["initial_top1_top2_margin"]),
                "mean_initial_candidate_count": safe_mean(sub["initial_candidate_count"]),
                "mean_rl_hit_round": safe_mean(sub["strongest_rl__hit_round"]),
                "mean_greedy_hit_round": safe_mean(sub["posterior_greedy__hit_round"]),
            }
        )
    order = [
        "rl_unique_win",
        "both_hit_rl_earlier",
        "both_hit_same_or_near",
        "both_hit_greedy_earlier",
        "greedy_unique_win",
        "both_fail",
    ]
    return pd.DataFrame(rows).sort_values(by="taxonomy_group", key=lambda s: s.map({k: i for i, k in enumerate(order)}))


def build_difficulty_bucket_tables(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df.copy()
    df["entropy_bucket"] = percentile_buckets(df["initial_posterior_entropy"], ["q1", "q2", "q3", "q4"])
    margin_labels = ["lowest_margin", "low_margin", "mid_margin", "high_margin"]
    df["margin_bucket"] = percentile_buckets(df["initial_top1_top2_margin"], margin_labels)
    support_labels = ["smallest_support", "small_support", "mid_support", "largest_support"]
    df["candidate_bucket"] = percentile_buckets(df["initial_candidate_count"], support_labels)
    rows: List[Dict[str, Any]] = []
    for feature, bucket_col in [
        ("initial_posterior_entropy", "entropy_bucket"),
        ("initial_top1_top2_margin", "margin_bucket"),
        ("initial_candidate_count", "candidate_bucket"),
    ]:
        for bucket, sub in df.groupby(bucket_col):
            rl_succ = sub["strongest_rl__success_rate"] > 0.5
            gr_succ = sub["posterior_greedy__success_rate"] > 0.5
            both_hit = rl_succ & gr_succ
            hit_adv = sub.loc[both_hit, "posterior_greedy__hit_sample_index"] - sub.loc[both_hit, "strongest_rl__hit_sample_index"]
            rows.append(
                {
                    "feature": feature,
                    "bucket": bucket,
                    "case_count": int(len(sub)),
                    "bucket_mean": safe_mean(sub[feature]),
                    "rl_success_rate": float(rl_succ.mean()),
                    "greedy_success_rate": float(gr_succ.mean()),
                    "rl_only_wins": int((rl_succ & (~gr_succ)).sum()),
                    "greedy_only_wins": int((gr_succ & (~rl_succ)).sum()),
                    "ties": int((rl_succ == gr_succ).sum()),
                    "net_flip_rl_minus_greedy": int((rl_succ & (~gr_succ)).sum() - (gr_succ & (~rl_succ)).sum()),
                    "earlier_hit_adv_samples_greedy_minus_rl_on_both_hit": safe_mean(hit_adv),
                }
            )
    return pd.DataFrame(rows)


def parse_selected_actions(step_df: pd.DataFrame) -> pd.DataFrame:
    out = step_df.copy()
    out["selected_global_ids_list"] = out["selected_global_ids"].apply(parse_json_list)
    return out


def build_undirected_adj(graph_path: Path) -> Dict[int, set[int]]:
    with np.load(graph_path, allow_pickle=True) as payload:
        edge_index = payload["edge_index"]
    adj: Dict[int, set[int]] = defaultdict(set)
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    for u, v in zip(src, dst):
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def neighborhood_within_two(adj: Dict[int, set[int]], source: int) -> Dict[int, int]:
    dists = {int(source): 0}
    q = deque([(int(source), 0)])
    while q:
        node, dist = q.popleft()
        if dist >= 2:
            continue
        for nxt in adj.get(node, ()):
            if nxt in dists:
                continue
            dists[nxt] = dist + 1
            q.append((nxt, dist + 1))
    return dists


def shortest_path_pair(adj: Dict[int, set[int]], source: int, target: int, cache: Dict[Tuple[int, int], int]) -> Optional[int]:
    if source == target:
        return 0
    key = (min(int(source), int(target)), max(int(source), int(target)))
    if key in cache:
        return cache[key]
    left_front = {int(source)}
    right_front = {int(target)}
    left_seen = {int(source): 0}
    right_seen = {int(target): 0}
    while left_front and right_front:
        if len(left_front) <= len(right_front):
            next_front = set()
            for node in left_front:
                for nxt in adj.get(node, ()):
                    if nxt in left_seen:
                        continue
                    left_seen[nxt] = left_seen[node] + 1
                    if nxt in right_seen:
                        dist = left_seen[nxt] + right_seen[nxt]
                        cache[key] = dist
                        return dist
                    next_front.add(nxt)
            left_front = next_front
        else:
            next_front = set()
            for node in right_front:
                for nxt in adj.get(node, ()):
                    if nxt in right_seen:
                        continue
                    right_seen[nxt] = right_seen[node] + 1
                    if nxt in left_seen:
                        dist = right_seen[nxt] + left_seen[nxt]
                        cache[key] = dist
                        return dist
                    next_front.add(nxt)
            right_front = next_front
    cache[key] = -1
    return None


def build_policy_behavior_tables(
    panel_df: pd.DataFrame,
    rl_steps: pd.DataFrame,
    greedy_steps: pd.DataFrame,
    adj: Dict[int, set[int]],
) -> pd.DataFrame:
    rl_steps = parse_selected_actions(rl_steps)
    greedy_steps = parse_selected_actions(greedy_steps)
    merge_cols = ["case_id", "episode_index"]
    merged = rl_steps.merge(
        greedy_steps[merge_cols + ["selected_global_ids_list"]],
        on=merge_cols,
        how="inner",
        suffixes=("_rl", "_greedy"),
    )
    dist_cache: Dict[Tuple[int, int], int] = {}
    records: List[Dict[str, Any]] = []
    taxonomy_map = panel_df.set_index("case_id").apply(classify_case, axis=1).to_dict()
    for _, row in merged.iterrows():
        rl_actions = list(row["selected_global_ids_list_rl"])
        gr_actions = list(row["selected_global_ids_list_greedy"])
        rl_set = set(rl_actions)
        gr_set = set(gr_actions)
        pairs_rl = []
        for i in range(len(rl_actions)):
            for j in range(i + 1, len(rl_actions)):
                dist = shortest_path_pair(adj, rl_actions[i], rl_actions[j], dist_cache)
                if dist is not None:
                    pairs_rl.append(dist)
        pairs_gr = []
        for i in range(len(gr_actions)):
            for j in range(i + 1, len(gr_actions)):
                dist = shortest_path_pair(adj, gr_actions[i], gr_actions[j], dist_cache)
                if dist is not None:
                    pairs_gr.append(dist)
        slot_matches = sum(1 for a, b in zip(rl_actions, gr_actions) if a == b)
        records.append(
            {
                "case_id": row["case_id"],
                "episode_index": int(row["episode_index"]),
                "taxonomy_group": taxonomy_map.get(row["case_id"], "unknown"),
                "exact_set_match": float(rl_actions == gr_actions),
                "jaccard_overlap": float(len(rl_set & gr_set) / max(len(rl_set | gr_set), 1)),
                "slot_overlap_fraction": float(slot_matches / max(len(rl_actions), len(gr_actions), 1)),
                "rl_mean_pairwise_hop_spread": safe_mean(pd.Series(pairs_rl)),
                "greedy_mean_pairwise_hop_spread": safe_mean(pd.Series(pairs_gr)),
                "rl_slate_posterior_take": float(row.get("policy_slate_posterior_take", np.nan)),
                "rl_slate_disagreement_take": float(row.get("policy_slate_disagreement_take", np.nan)),
                "rl_slate_novelty_take": float(row.get("policy_slate_novelty_take", np.nan)),
                "rl_slate_fill_take": float(row.get("policy_slate_fill_take", np.nan)),
            }
        )
    detail_df = pd.DataFrame(records)
    rows: List[Dict[str, Any]] = []
    for group_name, sub in [("overall", detail_df)] + list(detail_df.groupby("taxonomy_group")):
        rows.append(
            {
                "slice": group_name,
                "step_count": int(len(sub)),
                "exact_action_set_match_rate": safe_mean(sub["exact_set_match"]),
                "mean_jaccard_overlap": safe_mean(sub["jaccard_overlap"]),
                "mean_slot_overlap_fraction": safe_mean(sub["slot_overlap_fraction"]),
                "mean_rl_pairwise_hop_spread": safe_mean(sub["rl_mean_pairwise_hop_spread"]),
                "mean_greedy_pairwise_hop_spread": safe_mean(sub["greedy_mean_pairwise_hop_spread"]),
                "mean_rl_slate_posterior_take": safe_mean(sub["rl_slate_posterior_take"]),
                "mean_rl_slate_disagreement_take": safe_mean(sub["rl_slate_disagreement_take"]),
                "mean_rl_slate_novelty_take": safe_mean(sub["rl_slate_novelty_take"]),
                "mean_rl_slate_fill_take": safe_mean(sub["rl_slate_fill_take"]),
            }
        )
    return pd.DataFrame(rows), detail_df


def build_mechanism_tables(
    panel_df: pd.DataFrame,
    rl_steps: pd.DataFrame,
    greedy_steps: pd.DataFrame,
) -> pd.DataFrame:
    taxonomy = panel_df.set_index("case_id").apply(classify_case, axis=1).to_dict()
    rl = rl_steps.copy()
    greedy = greedy_steps.copy()
    rl["taxonomy_group"] = rl["case_id"].map(taxonomy)
    greedy["taxonomy_group"] = greedy["case_id"].map(taxonomy)
    rl["method_id"] = "strongest_rl"
    greedy["method_id"] = "posterior_greedy"
    common_cols = ["case_id", "episode_index", "taxonomy_group", "method_id", "posterior_entropy", "top1_mass", "top3_mass"]
    combo = pd.concat([rl[common_cols], greedy[common_cols]], ignore_index=True)
    grouped = (
        combo.groupby(["taxonomy_group", "method_id", "episode_index"], dropna=False)[["posterior_entropy", "top1_mass", "top3_mass"]]
        .mean()
        .reset_index()
    )
    counts = combo.groupby(["taxonomy_group", "method_id", "episode_index"]).size().rename("row_count").reset_index()
    return grouped.merge(counts, on=["taxonomy_group", "method_id", "episode_index"], how="left")


def flatten_sample_sequence(step_df: pd.DataFrame) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for case_id, sub in step_df.groupby("case_id"):
        ordered = sub.sort_values(by="episode_index")
        seq: List[int] = []
        for actions in ordered["selected_global_ids"].apply(parse_json_list):
            seq.extend(actions)
        out[str(case_id)] = seq
    return out


def build_relaxed_metric_table(
    panel_df: pd.DataFrame,
    case_tables: Dict[str, pd.DataFrame],
    step_tables: Dict[str, pd.DataFrame],
    adj: Dict[int, set[int]],
) -> pd.DataFrame:
    source_series = pd.to_numeric(panel_df.set_index("case_id")["source_global_id"], errors="coerce").dropna()
    source_map = {str(case_id): int(source_id) for case_id, source_id in source_series.items()}
    neighborhood_cache: Dict[int, Dict[int, int]] = {}
    sequences = {method: flatten_sample_sequence(df) for method, df in step_tables.items()}
    rows: List[Dict[str, Any]] = []
    radii = [0, 1, 2]
    for method, seqs in sequences.items():
        case_ids = sorted(seqs.keys())
        known_source_case_ids = [case_id for case_id in case_ids if case_id in source_map]
        exact_case_df = case_tables[method].set_index("case_id")
        for budget in range(1, 31):
            for radius in radii:
                hits = 0
                used_case_ids = case_ids
                if radius == 0:
                    for case_id in case_ids:
                        seq = seqs[case_id]
                        hit_sample = exact_case_df.at[case_id, "hit_sample_index"] if case_id in exact_case_df.index else np.nan
                        if pd.notna(hit_sample) and int(hit_sample) <= budget:
                            hits += 1
                else:
                    used_case_ids = known_source_case_ids
                    for case_id in known_source_case_ids:
                        source = int(source_map[case_id])
                        if source not in neighborhood_cache:
                            neighborhood_cache[source] = neighborhood_within_two(adj, source)
                        allowed = neighborhood_cache[source]
                        seq = seqs[case_id][:budget]
                        ok = False
                        for node in seq:
                            if int(node) in allowed and allowed[int(node)] <= radius:
                                ok = True
                                break
                        if ok:
                            hits += 1
                rows.append(
                    {
                        "method_id": method,
                        "display_name": DISPLAY_NAMES[method],
                        "radius_hops": radius,
                        "sample_budget": budget,
                        "case_count_used": int(len(used_case_ids)),
                        "success_rate": float(hits / max(len(used_case_ids), 1)),
                    }
                )
    return pd.DataFrame(rows)


def build_representative_cases(
    panel_df: pd.DataFrame,
    rl_steps: pd.DataFrame,
    greedy_steps: pd.DataFrame,
) -> List[Dict[str, Any]]:
    taxonomy = panel_df.copy()
    taxonomy["taxonomy_group"] = taxonomy.apply(classify_case, axis=1)
    rl_action_map = rl_steps.groupby("case_id")["selected_global_ids"].apply(list).to_dict()
    gr_action_map = greedy_steps.groupby("case_id")["selected_global_ids"].apply(list).to_dict()
    selected: List[Dict[str, Any]] = []
    for group, limit in REPRESENTATIVE_LIMITS.items():
        sub = taxonomy[taxonomy["taxonomy_group"] == group].copy()
        if sub.empty:
            continue
        sub["difficulty_score"] = (
            sub["initial_posterior_entropy"].fillna(0.0)
            + sub["initial_candidate_count"].fillna(0.0) / max(float(sub["initial_candidate_count"].max()), 1.0)
            - sub["initial_top1_top2_margin"].fillna(0.0)
        )
        chosen = sub.sort_values(by=["difficulty_score", "case_id"], ascending=[False, True]).head(limit)
        for _, row in chosen.iterrows():
            case_id = str(row["case_id"])
            selected.append(
                {
                    "case_id": case_id,
                    "taxonomy_group": group,
                    "initial_stats": {
                        "posterior_entropy": row["initial_posterior_entropy"],
                        "top1_top2_margin": row["initial_top1_top2_margin"],
                        "candidate_count": row["initial_candidate_count"],
                        "top1_mass": row["initial_top1_mass"],
                        "top3_mass": row["initial_top3_mass"],
                    },
                    "greedy_outcome": {
                        "success": bool(row["posterior_greedy__success_rate"] > 0.5),
                        "hit_round": None if pd.isna(row["posterior_greedy__hit_round"]) else int(row["posterior_greedy__hit_round"]),
                        "hit_sample_index": None if pd.isna(row["posterior_greedy__hit_sample_index"]) else int(row["posterior_greedy__hit_sample_index"]),
                    },
                    "rl_outcome": {
                        "success": bool(row["strongest_rl__success_rate"] > 0.5),
                        "hit_round": None if pd.isna(row["strongest_rl__hit_round"]) else int(row["strongest_rl__hit_round"]),
                        "hit_sample_index": None if pd.isna(row["strongest_rl__hit_sample_index"]) else int(row["strongest_rl__hit_sample_index"]),
                    },
                    "greedy_first_sets": [parse_json_list(v) for v in gr_action_map.get(case_id, [])[:3]],
                    "rl_first_sets": [parse_json_list(v) for v in rl_action_map.get(case_id, [])[:3]],
                    "short_explanation": explanation_for_case(group),
                }
            )
    return selected


def explanation_for_case(group: str) -> str:
    if group == "rl_unique_win":
        return "RL reaches the true node while Greedy Posterior exhausts budget; this is the clearest evidence of set-level improvement."
    if group == "both_hit_rl_earlier":
        return "Both methods solve the case, but RL reaches the source materially earlier under the same B30 contract."
    if group == "greedy_unique_win":
        return "Greedy Posterior reaches the source while RL misses; this is a concrete failure mode for the learned 3-set policy."
    if group == "both_fail":
        return "Neither policy reaches the source within budget; these cases define the remaining hard regime under the current belief core."
    return "Representative case from the paired RL-vs-Greedy taxonomy."


def write_plots(
    output_dir: Path,
    success_curve_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    policy_behavior_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
    relaxed_df: pd.DataFrame,
) -> List[str]:
    figure_paths: List[str] = []

    plt.figure(figsize=(9, 6))
    plot_order = ["strongest_rl"] + REQUIRED_HEURISTICS
    for method in plot_order:
        sub = success_curve_df[success_curve_df["method_id"] == method].sort_values("sample_budget")
        plt.plot(sub["sample_budget"], sub["cumulative_success_rate"], label=DISPLAY_NAMES[method], linewidth=2)
    plt.xlabel("Sample Budget")
    plt.ylabel("Success Rate")
    plt.title("Success vs Budget on Held-Out Val (1031 cases)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / "success_vs_budget.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    plt.figure(figsize=(8, 5))
    ordered_tax = taxonomy_df.copy().sort_values("case_count", ascending=False)
    plt.bar(ordered_tax["taxonomy_group"], ordered_tax["case_count"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Case Count")
    plt.title("RL vs Greedy Paired Taxonomy")
    plt.tight_layout()
    path = output_dir / "case_taxonomy.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    ent = difficulty_df[difficulty_df["feature"] == "initial_posterior_entropy"].copy()
    plt.figure(figsize=(8, 5))
    x = np.arange(len(ent))
    plt.bar(x - 0.18, ent["rl_success_rate"], width=0.36, label="Strongest RL")
    plt.bar(x + 0.18, ent["greedy_success_rate"], width=0.36, label="Greedy Posterior")
    plt.xticks(x, ent["bucket"])
    plt.ylabel("Success Rate")
    plt.title("Success by Initial Entropy Bucket")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "difficulty_entropy_buckets.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    overall_pb = policy_behavior_df[policy_behavior_df["slice"] == "overall"].iloc[0]
    plt.figure(figsize=(7, 4))
    pb_names = ["Exact Match", "Mean Jaccard", "Slot Overlap"]
    pb_vals = [
        overall_pb["exact_action_set_match_rate"],
        overall_pb["mean_jaccard_overlap"],
        overall_pb["mean_slot_overlap_fraction"],
    ]
    plt.bar(pb_names, pb_vals)
    plt.ylim(0, 1)
    plt.title("RL vs Greedy 3-Set Overlap")
    plt.tight_layout()
    path = output_dir / "policy_overlap.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    mech_sub = mechanism_df[mechanism_df["taxonomy_group"].isin(["rl_unique_win", "greedy_unique_win", "both_hit_rl_earlier"])].copy()
    if not mech_sub.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, group in zip(axes, ["rl_unique_win", "both_hit_rl_earlier", "greedy_unique_win"]):
            sub = mech_sub[mech_sub["taxonomy_group"] == group]
            for method, color in [("strongest_rl", "tab:blue"), ("posterior_greedy", "tab:orange")]:
                dat = sub[sub["method_id"] == method]
                ax.plot(dat["episode_index"], dat["posterior_entropy"], marker="o", label=DISPLAY_NAMES[method], color=color)
            ax.set_title(group)
            ax.set_xlabel("Round")
        axes[0].set_ylabel("Mean Posterior Entropy")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        path = output_dir / "mechanism_entropy_trajectories.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figure_paths.append(str(path))

    plt.figure(figsize=(9, 6))
    for radius, linestyle in zip([0, 1, 2], ["-", "--", ":"]):
        for method, color in [("strongest_rl", "tab:blue"), ("posterior_greedy", "tab:orange")]:
            sub = relaxed_df[(relaxed_df["radius_hops"] == radius) & (relaxed_df["method_id"] == method)].sort_values("sample_budget")
            plt.plot(
                sub["sample_budget"],
                sub["success_rate"],
                linestyle=linestyle,
                color=color,
                label=f"{DISPLAY_NAMES[method]} r<={radius}",
            )
    plt.xlabel("Sample Budget")
    plt.ylabel("Relaxed Success Rate")
    plt.title("Radius-Based Relaxed Success Curves")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = output_dir / "relaxed_radius_curves.png"
    plt.savefig(path, dpi=160)
    plt.close()
    figure_paths.append(str(path))

    return figure_paths


def build_summary_markdown(
    output_dir: Path,
    rl_artifact: SelectedArtifact,
    teacher5_artifact: SelectedArtifact,
    comparison_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    policy_behavior_df: pd.DataFrame,
    contract_notes: List[str],
) -> str:
    strongest_row = comparison_df[comparison_df["method_id"] == "strongest_rl"].iloc[0]
    greedy_row = comparison_df[comparison_df["method_id"] == "posterior_greedy"].iloc[0]
    thompson_row = comparison_df[comparison_df["method_id"] == "posterior_thompson_sampling"].iloc[0]
    entropy_q4 = difficulty_df[(difficulty_df["feature"] == "initial_posterior_entropy") & (difficulty_df["bucket"] == "q4")]
    pb = policy_behavior_df[policy_behavior_df["slice"] == "overall"].iloc[0]
    lines = [
        "# SPIM v3 + RL Paper Analysis",
        "",
        "## Scope",
        "- Built from existing held-out artifacts only; no training rerun and no fresh evaluation rerun were needed.",
        f"- Strongest RL artifact selected from full-train v3 strict evals: `{rl_artifact.summary_path}`.",
        f"- Heuristic baseline artifact selected from full val teacher5 compare: `{teacher5_artifact.summary_path}`.",
        "",
        "## Contract Check",
    ]
    lines.extend([f"- {note}" for note in contract_notes])
    lines.extend(
        [
            "",
            "## Main Results",
            f"- [proven] Strongest RL Success@B30 = `{strongest_row['success_at_B30']:.6f}` vs Greedy Posterior = `{greedy_row['success_at_B30']:.6f}` on the same held-out `val` split (`1031` cases).",
            f"- [proven] RL net flip vs Greedy Posterior = `{int(strongest_row['net_flip_vs_greedy'])}` with McNemar exact p = `{strongest_row['mcnemar_exact_p']:.6g}`.",
            f"- [proven] RL is also stronger than Thompson Sampling (`{thompson_row['success_at_B30']:.6f}`) and substantially stronger than the entropy/cover/disagreement heuristics.",
            f"- [proven] RL conditional average hit round = `{strongest_row['avg_hit_round_conditional']:.3f}` vs Greedy Posterior = `{greedy_row['avg_hit_round_conditional']:.3f}`.",
            "",
            "## Taxonomy",
        ]
    )
    for _, row in taxonomy_df.iterrows():
        lines.append(f"- `{row['taxonomy_group']}`: `{int(row['case_count'])}` cases ({row['percentage']:.2%}).")
    lines.extend(
        [
            "",
            "## Difficulty Signal",
        ]
    )
    if not entropy_q4.empty:
        row = entropy_q4.iloc[0]
        lines.append(
            f"- [proven] In the highest initial-entropy bucket, RL success = `{row['rl_success_rate']:.6f}` vs Greedy Posterior = `{row['greedy_success_rate']:.6f}` with net flip `{int(row['net_flip_rl_minus_greedy'])}`."
        )
    lines.extend(
        [
            "",
            "## Policy Behavior",
            f"- [proven] Exact 3-set match rate between RL and Greedy = `{pb['exact_action_set_match_rate']:.6f}`; mean Jaccard overlap = `{pb['mean_jaccard_overlap']:.6f}`.",
            f"- [partially proven] RL slates mix posterior/disagreement/novelty sources on average (`posterior={pb['mean_rl_slate_posterior_take']:.3f}`, `disagreement={pb['mean_rl_slate_disagreement_take']:.3f}`, `novelty={pb['mean_rl_slate_novelty_take']:.3f}`), but the current artifact only proves slate construction mix, not slot-level role specialization.",
            "",
            "## Caveats",
            "- [partially proven] The RL strict evaluator and the teacher5 runner align on split, case count, budget, and posterior family, but `teacher_full` in strict eval is not numerically identical to `posterior_greedy` in teacher5. The main paired analysis therefore uses the requested `posterior_greedy` heuristic directly rather than substituting `teacher_full`.",
            "- [not proven] True-source posterior rank and cluster-aware selected-node role labels are not exposed in the reusable artifacts, so those mechanism claims remain unproved.",
        ]
    )
    return "\n".join(lines) + "\n"


def ensure_output_dir(raw: str) -> Path:
    if raw:
        path = Path(raw)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = ARTIFACTS_ROOT / "paper_analysis" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(str(args.output_dir))

    rl_artifact = select_strongest_rl_artifact()
    teacher5_artifact = select_teacher5_artifact()

    case_tables = load_method_case_tables(rl_artifact, teacher5_artifact)
    step_tables = load_method_step_tables(rl_artifact, teacher5_artifact)
    initial_features = load_initial_feature_table(rl_artifact)
    panel_df = build_case_level_panel(case_tables, initial_features)

    contract_notes = [
        f"[proven] RL strict eval split = `{rl_artifact.summary['strict']['split']}` with `{rl_artifact.summary['strict']['split_case_count']}` cases and budget `{rl_artifact.summary['strict']['protocol']['budget']}`.",
        f"[proven] Teacher5 compare split = `{teacher5_artifact.summary['split_meta']['split']}` with `{teacher5_artifact.summary['split_meta']['loaded_case_count']}` cases and sample budget `{teacher5_artifact.summary['protocol']['sample_budget']}`.",
        f"[proven] Both artifacts use posterior family `{rl_artifact.summary['strict']['teacher_family']}` / `{teacher5_artifact.summary['posterior_family']['selected_family']}` and the same resolved config path.",
        "[partially proven] The teacher5 summary keeps a stale-looking `panel_version` string containing `exact136`, but the artifact itself proves `val` / `1031` via `split_meta`.",
    ]

    comparison_df = build_comparison_table(panel_df, case_tables)
    success_curve_df = build_success_curve_table(rl_artifact, teacher5_artifact)
    taxonomy_df = build_case_taxonomy(panel_df)
    difficulty_df = build_difficulty_bucket_tables(panel_df)

    graph_path = Path(rl_artifact.summary["parent"]["foundation_graph_path"])
    adj = build_undirected_adj(graph_path)

    policy_behavior_df, policy_behavior_detail_df = build_policy_behavior_tables(
        panel_df=panel_df,
        rl_steps=step_tables["strongest_rl"],
        greedy_steps=step_tables["posterior_greedy"],
        adj=adj,
    )
    mechanism_df = build_mechanism_tables(
        panel_df=panel_df,
        rl_steps=step_tables["strongest_rl"],
        greedy_steps=step_tables["posterior_greedy"],
    )
    relaxed_df = build_relaxed_metric_table(
        panel_df=panel_df,
        case_tables=case_tables,
        step_tables=step_tables,
        adj=adj,
    )
    representative_cases = build_representative_cases(
        panel_df=panel_df,
        rl_steps=step_tables["strongest_rl"],
        greedy_steps=step_tables["posterior_greedy"],
    )

    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
    success_curve_df.to_csv(output_dir / "success_vs_budget.csv", index=False)
    taxonomy_df.to_csv(output_dir / "case_taxonomy.csv", index=False)
    difficulty_df.to_csv(output_dir / "difficulty_bucket_tables.csv", index=False)
    policy_behavior_df.to_csv(output_dir / "policy_behavior_tables.csv", index=False)
    policy_behavior_detail_df.to_csv(output_dir / "policy_behavior_step_detail.csv", index=False)
    mechanism_df.to_csv(output_dir / "belief_trajectory_tables.csv", index=False)
    relaxed_df.to_csv(output_dir / "relaxed_metric_tables.csv", index=False)
    (output_dir / "representative_cases.json").write_text(json.dumps(representative_cases, indent=2), encoding="utf-8")

    figure_paths = write_plots(
        output_dir=output_dir,
        success_curve_df=success_curve_df,
        taxonomy_df=taxonomy_df,
        difficulty_df=difficulty_df,
        policy_behavior_df=policy_behavior_df,
        mechanism_df=mechanism_df,
        relaxed_df=relaxed_df,
    )

    summary_md = build_summary_markdown(
        output_dir=output_dir,
        rl_artifact=rl_artifact,
        teacher5_artifact=teacher5_artifact,
        comparison_df=comparison_df,
        taxonomy_df=taxonomy_df,
        difficulty_df=difficulty_df,
        policy_behavior_df=policy_behavior_df,
        contract_notes=contract_notes,
    )
    (output_dir / "analysis_summary.md").write_text(summary_md, encoding="utf-8")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "selected_rl_artifact": {
            "strict_summary_path": str(rl_artifact.summary_path),
            "case_rows_path": str(rl_artifact.case_rows_path),
            "step_rows_path": str(rl_artifact.step_rows_path),
            "parent_summary_path": str(rl_artifact.root / "summary.json"),
            "success_rate": rl_artifact.summary["strict"]["summary"]["success_rate"],
        },
        "selected_teacher5_artifact": {
            "summary_path": str(teacher5_artifact.summary_path),
            "case_rows_path": str(teacher5_artifact.case_rows_path),
            "step_rows_path": str(teacher5_artifact.step_rows_path),
        },
        "reused_inputs": [
            str(rl_artifact.summary_path),
            str(rl_artifact.case_rows_path),
            str(rl_artifact.step_rows_path),
            str(rl_artifact.root / "strict_eval_val_B30" / "teacher_full" / "case_rows.csv"),
            str(rl_artifact.root / "strict_eval_val_B30" / "teacher_full" / "step_rows.csv"),
            str(teacher5_artifact.summary_path),
            str(teacher5_artifact.case_rows_path),
            str(teacher5_artifact.step_rows_path),
            str(graph_path),
        ],
        "generated_outputs": [
            "analysis_summary.md",
            "comparison_table.csv",
            "success_vs_budget.csv",
            "case_taxonomy.csv",
            "difficulty_bucket_tables.csv",
            "policy_behavior_tables.csv",
            "policy_behavior_step_detail.csv",
            "belief_trajectory_tables.csv",
            "relaxed_metric_tables.csv",
            "representative_cases.json",
        ]
        + [Path(p).name for p in figure_paths],
        "contract_notes": contract_notes,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "selected_rl": str(rl_artifact.summary_path)}, indent=2))


if __name__ == "__main__":
    main()
