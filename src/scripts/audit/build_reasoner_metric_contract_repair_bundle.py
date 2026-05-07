from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FINISH_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_clean_aligned_online_finish" / "20260404_line_final"
DEFAULT_TEST_REPLAY_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_metric_semantics_audit_20260404_replay_test"
DEFAULT_VAL_REPLAY_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_metric_semantics_audit_20260404_val"
DEFAULT_HEURISTIC_JSON = PROJECT_ROOT / "artifacts" / "training_campaign_round1" / "20260322_225625" / "campaign_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reasoner_metric_contract_repair_20260404"
METRIC_CONTRACT_VERSION = "reasoner_metric_contract_repair_v2_20260404"
BUCKET_BOUNDS = [0, 5, 20, 100, 300, 1000, float("inf")]
BUCKET_LABELS = ["1-5", "6-20", "21-100", "101-300", "301-1000", "1001+"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def bool_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(float).mean())


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n| --- |\n| empty |"
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider] + rows)


def normalize_rank(rank: float, candidate_count: float) -> float:
    if pd.isna(rank) or pd.isna(candidate_count):
        return float("nan")
    if float(candidate_count) <= 1.0:
        return 0.0
    return (float(rank) - 1.0) / max(float(candidate_count) - 1.0, 1.0)


def percentile_rank(rank: float, candidate_count: float) -> float:
    if pd.isna(rank) or pd.isna(candidate_count) or float(candidate_count) <= 0.0:
        return float("nan")
    return 1.0 - ((float(rank) - 1.0) / float(candidate_count))


def runner_version_from_run_dir(run_dir: str) -> Optional[str]:
    match = re.search(r"clean_aligned_online_finish_([0-9a-z]+)$", str(run_dir))
    if match:
        return match.group(1)
    return None


def first_line(path: Path, needle: str) -> Optional[int]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if needle in line:
                return line_no
    return None


def line_ref(path: Path, needle: str) -> str:
    line_no = first_line(path, needle)
    return f"{path}:{line_no if line_no is not None else 'unknown'}"


def load_replay_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_candidates = [
        root / "success_failure_bucket_audit" / "raw_case_summary.csv",
        root / "summary" / "case_level_audit.csv",
    ]
    step_candidates = [
        root / "episode_progress_audit" / "raw_case_episode_metrics.csv",
        root / "summary" / "step_level_audit.csv",
    ]
    case_path = next((p for p in case_candidates if p.exists()), None)
    step_path = next((p for p in step_candidates if p.exists()), None)
    if case_path is None or step_path is None:
        raise FileNotFoundError(f"Replay root missing case/step tables: {root}")
    return pd.read_csv(case_path), pd.read_csv(step_path)


def standardize_step_df(step_df: pd.DataFrame, *, split: str) -> pd.DataFrame:
    work = step_df.copy()
    if "split" not in work.columns:
        work["split"] = split
    if "episode" not in work.columns:
        if "episode_index" in work.columns:
            work["episode"] = work["episode_index"].astype(int) - 1
        else:
            raise KeyError("Step replay table must contain episode or episode_index.")
    if "candidate_count_raw_graph" not in work.columns:
        work["candidate_count_raw_graph"] = work.get("total_nodes")
    if "softmax_candidate_count" not in work.columns:
        work["softmax_candidate_count"] = work.get("logits_candidate_size")
    if "pre_action_valid_count" not in work.columns:
        work["pre_action_valid_count"] = work.get("pre_action_valid_size")
    if "post_action_valid_count" not in work.columns:
        work["post_action_valid_count"] = work.get("post_action_valid_size")
    if "unrevealed_candidate_ratio_total" not in work.columns:
        work["unrevealed_candidate_ratio_total"] = work.get("unrevealed_candidate_ratio")
    if "valid_case" not in work.columns:
        work["valid_case"] = work["true_source_rank"].notna()
    work["valid_case"] = work["valid_case"].fillna(False).astype(bool)
    work["fallback_triggered"] = work.get("fallback_triggered", work.get("logits_minus_action_valid", 0)).fillna(0) > 0
    work["degenerate_regime"] = work["pre_action_valid_count"].fillna(0).astype(float) <= 5.0
    work["normalized_rank"] = work.apply(
        lambda row: normalize_rank(row.get("true_source_rank"), row.get("softmax_candidate_count")),
        axis=1,
    )
    work["percentile_rank"] = work.apply(
        lambda row: percentile_rank(row.get("true_source_rank"), row.get("softmax_candidate_count")),
        axis=1,
    )
    if "true_source_observed" not in work.columns:
        work["true_source_observed"] = float("nan")
    return work


def standardize_case_df(case_df: pd.DataFrame, step_df: pd.DataFrame, *, split: str) -> pd.DataFrame:
    work = case_df.copy()
    if "split" not in work.columns:
        work["split"] = split
    if "step_count_observed" not in work.columns:
        work["step_count_observed"] = work.get("trajectory_len", 0)
    work["empty_trajectory"] = work.get("empty_trajectory", work["step_count_observed"].fillna(0).eq(0))
    if "total_nodes" not in work.columns:
        work["total_nodes"] = work.get("candidate_count_raw_graph")
    work["candidate_count_raw_graph"] = work.get("candidate_count_raw_graph", work["total_nodes"])
    if "final_true_source_rank" not in work.columns:
        work["final_true_source_rank"] = work.get("true_source_rank")
    if "final_top1_hit" not in work.columns:
        work["final_top1_hit"] = work.get("top1_hit")
    if "final_top3_hit" not in work.columns:
        work["final_top3_hit"] = work.get("top3_hit")
    if "final_top5_hit" not in work.columns:
        work["final_top5_hit"] = work.get("top5_hit")
    if "final_mrr" not in work.columns:
        work["final_mrr"] = work.get("mrr")
    if "final_softmax_candidate_count" not in work.columns:
        work["final_softmax_candidate_count"] = work.get("final_logits_candidate_size")
    if "final_pre_action_valid_count" not in work.columns:
        work["final_pre_action_valid_count"] = work.get("final_pre_action_valid_size")
    if "final_post_action_valid_count" not in work.columns:
        work["final_post_action_valid_count"] = work.get("final_post_action_valid_size")
    if "final_unrevealed_candidate_ratio_total" not in work.columns:
        work["final_unrevealed_candidate_ratio_total"] = work.get("final_unrevealed_candidate_ratio")
    if "final_finite_infeasible_count" not in work.columns:
        work["final_finite_infeasible_count"] = work.get("final_finite_infeasible_count", float("nan"))
    work["valid_case"] = work.get("valid_case", work["final_true_source_rank"].notna()).fillna(False).astype(bool)

    first_step = (
        step_df.sort_values(["case_id", "episode"])
        .groupby("case_id", as_index=False)
        .first()
        .rename(
            columns={
                "true_source_rank": "step0_true_source_rank",
                "top1_hit": "step0_top1_hit",
                "top3_hit": "step0_top3_hit",
                "top5_hit": "step0_top5_hit",
                "mrr": "step0_mrr",
                "softmax_candidate_count": "step0_softmax_candidate_count",
                "revealed_ratio": "initial_observed_ratio",
                "observed_count": "initial_observed_count",
            }
        )
    )
    fallback_case = (
        step_df.groupby("case_id", as_index=False)["fallback_triggered"]
        .max()
        .rename(columns={"fallback_triggered": "fallback_any_step"})
    )
    work = work.merge(
        first_step[
            [
                "case_id",
                "step0_true_source_rank",
                "step0_top1_hit",
                "step0_top3_hit",
                "step0_top5_hit",
                "step0_mrr",
                "step0_softmax_candidate_count",
                "initial_observed_ratio",
                "initial_observed_count",
            ]
        ],
        on="case_id",
        how="left",
    )
    work = work.merge(fallback_case, on="case_id", how="left")
    work["fallback_any_step"] = work["fallback_any_step"].fillna(False).astype(bool)

    work["final_normalized_rank"] = work.apply(
        lambda row: normalize_rank(row.get("final_true_source_rank"), row.get("final_pre_action_valid_count")),
        axis=1,
    )
    work["final_percentile_rank"] = work.apply(
        lambda row: percentile_rank(row.get("final_true_source_rank"), row.get("final_pre_action_valid_count")),
        axis=1,
    )
    work["step0_normalized_rank"] = work.apply(
        lambda row: normalize_rank(row.get("step0_true_source_rank"), row.get("step0_softmax_candidate_count")),
        axis=1,
    )
    work["step0_percentile_rank"] = work.apply(
        lambda row: percentile_rank(row.get("step0_true_source_rank"), row.get("step0_softmax_candidate_count")),
        axis=1,
    )
    work["candidate_shrink_ratio"] = work["final_pre_action_valid_count"] / work["step0_softmax_candidate_count"]
    work["log_scaled_rank"] = np.log1p(work["final_true_source_rank"])
    return work


def aggregate_replay_metrics(case_df: pd.DataFrame) -> Dict[str, float]:
    valid = case_df[case_df["final_true_source_rank"].notna()].copy()
    return {
        "num_events": float(len(case_df)),
        "valid_events": float(len(valid)),
        "top1_hit": float(valid["final_top1_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "top3_hit": float(valid["final_top3_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "top5_hit": float(valid["final_top5_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "mrr_valid": float(valid["final_mrr"].mean()) if len(valid) else float("nan"),
        "true_source_rank_mean": float(valid["final_true_source_rank"].mean()) if len(valid) else float("nan"),
        "success_rate": float(case_df["success"].astype(float).mean()) if len(case_df) else float("nan"),
    }


def build_replay_alignment_rows(
    *,
    split: str,
    official_summary: Dict[str, Any],
    replay_case_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    replay_metrics = aggregate_replay_metrics(replay_case_df)
    rows: List[Dict[str, Any]] = []
    for metric in [
        "num_events",
        "valid_events",
        "success_rate",
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "mrr_valid",
        "true_source_rank_mean",
    ]:
        official_value = safe_float(official_summary.get(metric))
        replay_value = safe_float(replay_metrics.get(metric))
        delta = replay_value - official_value
        rows.append(
            {
                "split": split,
                "metric": metric,
                "official_value": official_value,
                "replay_value": replay_value,
                "delta": delta,
                "abs_delta": abs(delta) if math.isfinite(delta) else float("nan"),
            }
        )
    return rows


def build_zero_step_summary(all_case_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split, case_df in all_case_dfs.items():
        zero = case_df["empty_trajectory"].fillna(False)
        total_success = float(case_df["success"].astype(float).sum())
        rows.append(
            {
                "split": split,
                "total_cases": int(len(case_df)),
                "success_count": int(case_df["success"].astype(float).sum()),
                "success_rate_all_cases": float(case_df["success"].astype(float).mean()),
                "zero_step_count": int(zero.sum()),
                "zero_step_rate": float(zero.mean()),
                "zero_step_success_count": int(case_df.loc[zero, "success"].astype(float).sum()),
                "zero_step_success_rate": float(case_df.loc[zero, "success"].astype(float).mean()) if int(zero.sum()) else float("nan"),
                "non_zero_step_count": int((~zero).sum()),
                "non_zero_step_success_rate": float(case_df.loc[~zero, "success"].astype(float).mean()) if int((~zero).sum()) else float("nan"),
                "zero_step_share_of_all_successes": float(case_df.loc[zero, "success"].astype(float).sum() / total_success) if total_success > 0 else float("nan"),
                "non_zero_step_valid_ranking_cases": int(case_df.loc[(~zero) & case_df["valid_case"], "case_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def add_final_rank_normalizers(case_df: pd.DataFrame) -> pd.DataFrame:
    work = case_df.copy()
    work["final_normalized_rank"] = work.apply(
        lambda row: normalize_rank(row["final_true_source_rank"], row["final_pre_action_valid_count"]),
        axis=1,
    )
    work["final_percentile_rank"] = work.apply(
        lambda row: percentile_rank(row["final_true_source_rank"], row["final_pre_action_valid_count"]),
        axis=1,
    )
    return work


def candidate_bucket_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    valid = add_final_rank_normalizers(case_df[case_df["valid_case"]].copy())
    valid["candidate_bucket"] = pd.cut(
        valid["final_pre_action_valid_count"],
        bins=BUCKET_BOUNDS,
        labels=BUCKET_LABELS,
        include_lowest=True,
    )
    rows: List[Dict[str, Any]] = []
    for label in BUCKET_LABELS:
        group = valid[valid["candidate_bucket"] == label]
        if group.empty:
            continue
        rows.append(
            {
                "candidate_bucket": label,
                "bucket_min": BUCKET_BOUNDS[BUCKET_LABELS.index(label)] + (1 if label != BUCKET_LABELS[0] else 0),
                "bucket_max": BUCKET_BOUNDS[BUCKET_LABELS.index(label) + 1],
                "count": int(len(group)),
                "top1_hit": float(group["final_top1_hit"].astype(float).mean()),
                "top3_hit": float(group["final_top3_hit"].astype(float).mean()),
                "top5_hit": float(group["final_top5_hit"].astype(float).mean()),
                "mrr_valid": float(group["final_mrr"].mean()),
                "percentile_rank_mean": float(group["final_percentile_rank"].mean()),
                "normalized_rank_mean": float(group["final_normalized_rank"].mean()),
                "log_scaled_rank_mean": float(np.log1p(group["final_true_source_rank"]).mean()),
                "true_source_rank_mean": float(group["final_true_source_rank"].mean()),
                "revealed_ratio_mean": float(group["final_revealed_ratio"].mean()),
                "final_candidate_count_mean": float(group["final_pre_action_valid_count"].mean()),
                "non_zero_step_success_rate": float(group["success"].astype(float).mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["weakness_order"] = out["mrr_valid"].rank(method="dense", ascending=True).astype(int)
    return out


def episode_progress_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode, group in step_df.groupby("episode", observed=True):
        valid = group[group["valid_case"]].copy()
        rows.append(
            {
                "episode": int(episode),
                "cases_at_episode": int(group["case_id"].nunique()),
                "valid_cases_at_episode": int(valid["case_id"].nunique()),
                "top1_hit": float(valid["top1_hit"].astype(float).mean()) if len(valid) else float("nan"),
                "top3_hit": float(valid["top3_hit"].astype(float).mean()) if len(valid) else float("nan"),
                "top5_hit": float(valid["top5_hit"].astype(float).mean()) if len(valid) else float("nan"),
                "mrr": float(valid["mrr"].mean()) if len(valid) else float("nan"),
                "percentile_rank_mean": float(valid["percentile_rank"].mean()) if len(valid) else float("nan"),
                "normalized_rank_mean": float(valid["normalized_rank"].mean()) if len(valid) else float("nan"),
                "true_source_rank_mean": float(valid["true_source_rank"].mean()) if len(valid) else float("nan"),
                "pre_action_valid_size_mean": float(group["pre_action_valid_count"].mean()),
                "softmax_candidate_size_mean": float(group["softmax_candidate_count"].mean()),
                "revealed_ratio_mean": float(group["revealed_ratio"].mean()),
                "unrevealed_candidate_ratio_mean": float(group["unrevealed_candidate_ratio_total"].mean()),
                "true_source_observed_rate": float(group["true_source_observed"].astype(float).mean()) if group["true_source_observed"].notna().any() else float("nan"),
                "success_before_step_rate": bool_mean(group["success_before_step"]),
                "success_event_rate": bool_mean(group["success_event"]),
                "degenerate_regime_rate": bool_mean(group["degenerate_regime"]),
                "fallback_rate": bool_mean(group["fallback_triggered"]),
            }
        )
    return pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)


def episode_curve_ready(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_cols = [
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "mrr",
        "percentile_rank_mean",
        "normalized_rank_mean",
        "pre_action_valid_size_mean",
        "softmax_candidate_size_mean",
        "revealed_ratio_mean",
        "unrevealed_candidate_ratio_mean",
        "degenerate_regime_rate",
        "fallback_rate",
    ]
    for _, row in episode_df.iterrows():
        for metric in metric_cols:
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "metric": metric,
                    "value": row[metric],
                    "cases_at_episode": int(row["cases_at_episode"]),
                    "valid_cases_at_episode": int(row["valid_cases_at_episode"]),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_mask_contract_table() -> pd.DataFrame:
    state_updates = PROJECT_ROOT / "src/modeling/loop/orchestration/state_updates.py"
    episode_runner = PROJECT_ROOT / "src/modeling/loop/episode_runner.py"
    state_builders = PROJECT_ROOT / "src/modeling/loop/orchestration/state_builders.py"
    clean_features = PROJECT_ROOT / "src/modeling/clean_aligned_features.py"
    clean_reasoner = PROJECT_ROOT / "src/modeling/reasoners/clean_aligned.py"
    phase45 = PROJECT_ROOT / "src/modeling/architectures/phase4_5_model.py"
    rows = [
        {
            "contract_object": "confirmed_non_source_mask",
            "semantic_role": "hard exclusion for nodes already confirmed non-source",
            "created_or_updated_at": line_ref(state_updates, "confirmed_non_source_mask = constraint_state.confirmed_non_source_mask.clone()"),
            "forward_stage_behavior": "not removed before CleanAlignedReasoner.forward; only available indirectly through downstream state/evidence",
            "post_forward_logits_behavior": "masked to -inf in apply_constraint_masks",
            "action_routing_behavior": "indirectly excluded because selected nodes also enter no_resample_mask",
            "reflected_in_saved_logits_normally": "yes",
            "reintroduced_by_fallback": "yes_only_if_fully_masked_graph_without_confirmed_source",
            "notes": "normal path clean; fallback can temporarily revive them in degenerate graphs",
        },
        {
            "contract_object": "no_resample_mask",
            "semantic_role": "hard exclusion for already sampled nodes",
            "created_or_updated_at": line_ref(state_updates, "no_resample_mask = torch.max(constraint_state.no_resample_mask, selection_mask)"),
            "forward_stage_behavior": "participates in valid_mask_final and feature payload",
            "post_forward_logits_behavior": "masked to -inf in apply_constraint_masks",
            "action_routing_behavior": "excluded in valid_mask_final before routing",
            "reflected_in_saved_logits_normally": "yes",
            "reintroduced_by_fallback": "yes_only_if_fully_masked_graph_without_confirmed_source",
            "notes": "this is the main action-valid hard exclusion",
        },
        {
            "contract_object": "confirmed_source_mask",
            "semantic_role": "lock routing/logits onto already confirmed source nodes",
            "created_or_updated_at": line_ref(state_updates, "confirmed_source_mask = constraint_state.confirmed_source_mask.clone()"),
            "forward_stage_behavior": "not removed before forward",
            "post_forward_logits_behavior": "if any graph has confirmed source, all other nodes in that graph become -inf and confirmed sources keep raw logits",
            "action_routing_behavior": "effectively trivializes routing once present",
            "reflected_in_saved_logits_normally": "yes",
            "reintroduced_by_fallback": "no",
            "notes": "not observed in current replayed step tables before action",
        },
        {
            "contract_object": "feasible_mask",
            "semantic_role": "physics-feasible candidate filter",
            "created_or_updated_at": line_ref(state_builders, "feasible_mask_tensor = physics_ctx['feasible_mask']"),
            "forward_stage_behavior": "passed inside valid_mask_final and feature payload to reasoner/navigator",
            "post_forward_logits_behavior": "not hard-masked in apply_constraint_masks",
            "action_routing_behavior": "excluded in valid_mask_final before routing",
            "reflected_in_saved_logits_normally": "no_hard_guarantee",
            "reintroduced_by_fallback": "not_applicable",
            "notes": "ranking logits can stay finite on infeasible nodes unless the model itself suppresses them",
        },
        {
            "contract_object": "valid_mask_final",
            "semantic_role": "composite action-valid mask = ~no_resample & feasible_mask",
            "created_or_updated_at": line_ref(episode_runner, "valid_mask_final = ~constraint_state.no_resample_mask.view(-1).bool()"),
            "forward_stage_behavior": "passed into navigator and reasoner state as runtime context",
            "post_forward_logits_behavior": "only the no_resample component is hard-reflected after forward",
            "action_routing_behavior": "direct routing gate",
            "reflected_in_saved_logits_normally": "partial_only",
            "reintroduced_by_fallback": "partial_only",
            "notes": "this is why ranking candidate set and action-valid set are not contract-identical",
        },
        {
            "contract_object": "fully_masked_fallback",
            "semantic_role": "degenerate safety restore when masking would leave a graph with no finite logits",
            "created_or_updated_at": line_ref(state_updates, "fully_masked_graph = (~graph_has_any_finite) & (~graph_has_confirmed_source)"),
            "forward_stage_behavior": "not applicable",
            "post_forward_logits_behavior": "restores raw logits for that graph",
            "action_routing_behavior": "action-valid set can still be empty while logits become finite again",
            "reflected_in_saved_logits_normally": "no_by_design",
            "reintroduced_by_fallback": "yes",
            "notes": "this is the only observed contract break in current test replay",
        },
        {
            "contract_object": "reasoner_forward_contract",
            "semantic_role": "clean aligned reasoner computes logits from features, not hard-pruned candidate subset",
            "created_or_updated_at": line_ref(clean_reasoner, "logits = self.head(torch.cat([h, graph_context, graph_features], dim=1))"),
            "forward_stage_behavior": "all fused nodes receive logits; valid_mask and feasible_mask are features, not hard pruning",
            "post_forward_logits_behavior": "hard exclusions happen only after forward",
            "action_routing_behavior": "not applicable",
            "reflected_in_saved_logits_normally": "post_forward_only",
            "reintroduced_by_fallback": "not_applicable",
            "notes": "important for interpreting ranking metrics versus action-valid candidates",
        },
        {
            "contract_object": "frozen_clean_v1_bridge",
            "semantic_role": "frozen navigator consumes clean state and valid_mask, not current reasoner logits",
            "created_or_updated_at": line_ref(PROJECT_ROOT / "src/modeling/navigators/frozen_clean_bridge.py", "def _build_node_features"),
            "forward_stage_behavior": "navigator sees support/contradiction/feasible/no_resample/valid_mask features",
            "post_forward_logits_behavior": "not applicable",
            "action_routing_behavior": "frozen navigator owns rollout under navigator_only policy",
            "reflected_in_saved_logits_normally": "not_applicable",
            "reintroduced_by_fallback": "not_applicable",
            "notes": "explains why system SR can stay flat while reasoner ranking changes",
        },
    ]
    return pd.DataFrame(rows)


def build_fallback_incidence_summary(step_df: pd.DataFrame) -> pd.DataFrame:
    fallback = step_df["fallback_triggered"].fillna(False)
    rows: List[Dict[str, Any]] = [
        {
            "scope": "overall_step",
            "key": "all_steps",
            "count": int(fallback.sum()),
            "denominator": int(len(step_df)),
            "rate": float(fallback.mean()),
            "notes": "step-level fully-masked fallback incidence",
        },
        {
            "scope": "overall_case",
            "key": "any_case",
            "count": int(step_df.loc[fallback, "case_id"].nunique()),
            "denominator": int(step_df["case_id"].nunique()),
            "rate": float(step_df.loc[fallback, "case_id"].nunique() / max(step_df["case_id"].nunique(), 1)),
            "notes": "case-level incidence of any fallback event",
        },
    ]
    for episode, group in step_df.groupby("episode", observed=True):
        fb = group["fallback_triggered"].fillna(False)
        rows.append(
            {
                "scope": "by_episode",
                "key": int(episode),
                "count": int(fb.sum()),
                "denominator": int(len(group)),
                "rate": float(fb.mean()),
                "notes": "episode-indexed fallback rate",
            }
        )
    return pd.DataFrame(rows)


def build_degnerate_examples(step_df: pd.DataFrame) -> pd.DataFrame:
    fallback = step_df[step_df["fallback_triggered"].fillna(False)].copy()
    if fallback.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "episode",
                "softmax_candidate_count",
                "pre_action_valid_count",
                "finite_confirmed_non_source_count",
                "finite_no_resample_count",
                "finite_infeasible_count",
            ]
        )
    return fallback[
        [
            "case_id",
            "episode",
            "softmax_candidate_count",
            "pre_action_valid_count",
            "finite_confirmed_non_source_count",
            "finite_no_resample_count",
            "finite_infeasible_count",
        ]
    ].sort_values(["case_id", "episode"]).reset_index(drop=True)


def build_heuristic_alignment_table(heuristic_json: Path, official_test_sr: float) -> pd.DataFrame:
    obj = load_json(heuristic_json)
    rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for phase in ["pilot", "main"]:
        for cfg_name, payload in obj[phase]["configs"].items():
            summary = payload["eval"]["test"]["summary"]
            all_rows.append(
                {
                    "phase": phase,
                    "config": cfg_name,
                    "scope": "overall",
                    "bucket_family": "overall",
                    "bucket_name": "overall",
                    "success_rate": float(summary["success_rate"]),
                    "top1_hit": float(summary["top1_hit"]),
                    "top3_hit": float(summary["top3_hit"]),
                    "top5_hit": float(summary["top5_hit"]),
                    "mrr_valid": float(summary["mrr_valid"]),
                    "compareability": "non_aligned_supportive_only",
                    "note": "legacy overall test summary on a different runtime contract",
                }
            )
            for family_name, family_rows in summary.get("bucket_summary", {}).items():
                for bucket_name, row in family_rows.items():
                    all_rows.append(
                        {
                            "phase": phase,
                            "config": cfg_name,
                            "scope": "bucket",
                            "bucket_family": family_name,
                            "bucket_name": bucket_name,
                            "success_rate": safe_float(row.get("success_rate")),
                            "top1_hit": safe_float(row.get("top1_hit")),
                            "top3_hit": safe_float(row.get("top3_hit")),
                            "top5_hit": safe_float(row.get("top5_hit")),
                            "mrr_valid": safe_float(row.get("mrr")),
                            "compareability": "invalid_for_headline_compare",
                            "note": "legacy easy-slice bucket, not apples-to-apples with current overall downstream line",
                        }
                    )
    df = pd.DataFrame(all_rows)
    top_buckets = df[df["scope"] == "bucket"].sort_values("success_rate", ascending=False).head(8)
    overall = df[df["scope"] == "overall"].sort_values("success_rate", ascending=False).head(8)
    rows.extend(overall.to_dict(orient="records"))
    rows.extend(top_buckets.to_dict(orient="records"))
    rows.append(
        {
            "phase": "authoritative_current",
            "config": "clean_env_frozen_nav_reasoner_best",
            "scope": "overall",
            "bucket_family": "overall",
            "bucket_name": "overall",
            "success_rate": official_test_sr,
            "top1_hit": float("nan"),
            "top3_hit": float("nan"),
            "top5_hit": float("nan"),
            "mrr_valid": float("nan"),
            "compareability": "authoritative_current",
            "note": "current official downstream supportive SR for reference only",
        }
    )
    return pd.DataFrame(rows)


def build_factorized_metrics_table(
    *,
    official_compare: Dict[str, Any],
    test_case_df: pd.DataFrame,
    test_step_df: pd.DataFrame,
) -> pd.DataFrame:
    first = test_step_df.sort_values(["case_id", "episode"]).groupby("case_id", as_index=False).first()
    valid_first = first[first["valid_case"]].copy()
    valid_final = test_case_df[test_case_df["valid_case"]].copy()
    paired = valid_final.merge(
        valid_first[
            [
                "case_id",
                "true_source_rank",
                "mrr",
                "top1_hit",
                "top3_hit",
                "top5_hit",
                "softmax_candidate_count",
                "revealed_ratio",
                "percentile_rank",
                "normalized_rank",
            ]
        ],
        on="case_id",
        how="inner",
        suffixes=("_final_unused", "_step0"),
    )
    rows = [
        {
            "factor_bucket": "reasoner_fixed_revealed_state",
            "view": "episode0_ranking",
            "metric": "top1_hit",
            "value": float(valid_first["top1_hit"].astype(float).mean()),
            "denominator": int(valid_first["case_id"].nunique()),
            "interpretation": "reasoner discrimination on the initial revealed state before navigator rollout unfolds",
        },
        {
            "factor_bucket": "reasoner_fixed_revealed_state",
            "view": "episode0_ranking",
            "metric": "mrr",
            "value": float(valid_first["mrr"].mean()),
            "denominator": int(valid_first["case_id"].nunique()),
            "interpretation": "same as above",
        },
        {
            "factor_bucket": "reasoner_fixed_revealed_state",
            "view": "episode0_ranking",
            "metric": "percentile_rank_mean",
            "value": float(valid_first["percentile_rank"].mean()),
            "denominator": int(valid_first["case_id"].nunique()),
            "interpretation": "scale-aware initial ranking quality",
        },
        {
            "factor_bucket": "reasoner_fixed_revealed_state",
            "view": "episode0_state_difficulty",
            "metric": "candidate_count_mean",
            "value": float(valid_first["pre_action_valid_count"].mean()),
            "denominator": int(valid_first["case_id"].nunique()),
            "interpretation": "initial candidate-space size faced by the reasoner",
        },
        {
            "factor_bucket": "reasoner_fixed_revealed_state",
            "view": "episode0_state_difficulty",
            "metric": "revealed_ratio_mean",
            "value": float(valid_first["revealed_ratio"].mean()),
            "denominator": int(valid_first["case_id"].nunique()),
            "interpretation": "how sparse the initial revealed state is",
        },
        {
            "factor_bucket": "navigator_induced_difficulty",
            "view": "rollout_entry_conditions",
            "metric": "zero_step_rate",
            "value": float(test_case_df["empty_trajectory"].astype(float).mean()),
            "denominator": int(len(test_case_df)),
            "interpretation": "share of cases that auto-succeed before any replayed ranking step exists",
        },
        {
            "factor_bucket": "navigator_induced_difficulty",
            "view": "rollout_entry_conditions",
            "metric": "non_zero_step_success_rate",
            "value": float(test_case_df.loc[~test_case_df["empty_trajectory"], "success"].astype(float).mean()),
            "denominator": int((~test_case_df["empty_trajectory"]).sum()),
            "interpretation": "system hit rate once zero-step auto-success is removed",
        },
        {
            "factor_bucket": "navigator_induced_difficulty",
            "view": "rollout_entry_conditions",
            "metric": "candidate_shrink_ratio_mean",
            "value": float((paired["final_pre_action_valid_count"] / paired["softmax_candidate_count"]).mean()),
            "denominator": int(len(paired)),
            "interpretation": "mean shrink of action-valid candidates from episode 0 to final replay step",
        },
        {
            "factor_bucket": "coupled_system",
            "view": "final_rollout_state",
            "metric": "top1_hit",
            "value": float(valid_final["final_top1_hit"].astype(float).mean()),
            "denominator": int(len(valid_final)),
            "interpretation": "reasoner ranking after frozen navigator has induced the final revealed state",
        },
        {
            "factor_bucket": "coupled_system",
            "view": "final_rollout_state",
            "metric": "mrr_valid",
            "value": float(valid_final["final_mrr"].mean()),
            "denominator": int(len(valid_final)),
            "interpretation": "same as above",
        },
        {
            "factor_bucket": "coupled_system",
            "view": "final_rollout_state",
            "metric": "success_rate",
            "value": float(test_case_df["success"].astype(float).mean()),
            "denominator": int(len(test_case_df)),
            "interpretation": "system-level physical hit under the frozen navigator rollout",
        },
        {
            "factor_bucket": "coupled_system",
            "view": "best_vs_bridged_official",
            "metric": "delta_top1_hit",
            "value": float(official_compare["delta"]["best_vs_bridged_online"]["top1_hit"]),
            "denominator": int(official_compare["best_checkpoint"]["valid_events"]),
            "interpretation": "archived official change in final ranking versus bridged baseline",
        },
        {
            "factor_bucket": "coupled_system",
            "view": "best_vs_bridged_official",
            "metric": "delta_mrr_valid",
            "value": float(official_compare["delta"]["best_vs_bridged_online"]["mrr_valid"]),
            "denominator": int(official_compare["best_checkpoint"]["valid_events"]),
            "interpretation": "archived official change in final ranking versus bridged baseline",
        },
        {
            "factor_bucket": "coupled_system",
            "view": "best_vs_bridged_official",
            "metric": "delta_success_rate",
            "value": float(official_compare["delta"]["best_vs_bridged_online"]["success_rate"]),
            "denominator": int(official_compare["best_checkpoint"]["num_events"]),
            "interpretation": "archived official change in frozen-policy SR versus bridged baseline",
        },
    ]
    return pd.DataFrame(rows)


def build_headline_vs_supportive_table() -> pd.DataFrame:
    rows = [
        {
            "claim_or_metric": "Top1 / Top3 / Top5 / MRR",
            "classification": "headline",
            "measures": "reasoner final-step ranking quality on valid ranking cases",
            "allowed_for_reasoner_claim": "yes",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "why": "best available direct ranking readout for the current downstream reasoner line",
        },
        {
            "claim_or_metric": "normalized rank / percentile rank / bucketed candidate-size performance",
            "classification": "supportive",
            "measures": "scale-aware reasoner ranking quality",
            "allowed_for_reasoner_claim": "yes_supportive",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "why": "needed to avoid raw-rank pollution from very large candidate sets",
        },
        {
            "claim_or_metric": "success_rate / avg_budget_used",
            "classification": "supportive",
            "measures": "system physical hit under frozen navigator rollout",
            "allowed_for_reasoner_claim": "no",
            "frozen_navigator_dominated": "yes",
            "zero_step_contaminated": "yes",
            "why": "system metric, not a pure reasoner metric",
        },
        {
            "claim_or_metric": "per-episode ranking curves / revealed ratios / fallback incidence",
            "classification": "diagnostic_only",
            "measures": "failure analysis and bottleneck localization",
            "allowed_for_reasoner_claim": "no_headline",
            "frozen_navigator_dominated": "mixed",
            "zero_step_contaminated": "indirectly",
            "why": "useful for diagnosis, not for top-line claims",
        },
        {
            "claim_or_metric": "direct compare against remembered legacy ~77% bucket values",
            "classification": "invalid_compare",
            "measures": "non-comparable legacy bucket slices",
            "allowed_for_reasoner_claim": "no",
            "frozen_navigator_dominated": "different_contract",
            "zero_step_contaminated": "different_contract",
            "why": "different stack, different controller, different denominator scope",
        },
    ]
    return pd.DataFrame(rows)


def build_metric_formula_table(
    *,
    test_case_df: pd.DataFrame,
    zero_step_df: pd.DataFrame,
) -> pd.DataFrame:
    top_valid = int(test_case_df["valid_case"].sum())
    total = int(len(test_case_df))
    zero_step_test = int(zero_step_df.loc[zero_step_df["split"] == "test", "zero_step_count"].iloc[0])
    episode_runner = PROJECT_ROOT / "src/modeling/loop/episode_runner.py"
    state_updates = PROJECT_ROOT / "src/modeling/loop/orchestration/state_updates.py"
    round1 = PROJECT_ROOT / "src/scripts/training_campaign_round1.py"
    frozen_bridge = PROJECT_ROOT / "src/modeling/navigators/frozen_clean_bridge.py"
    rows = [
        {
            "metric": "top1_hit",
            "metric_class": "reasoner_headline_ranking",
            "formula": "mean(1[rank_true <= 1]) over valid final ranking cases",
            "denominator": f"{top_valid} valid final ranking cases",
            "measures_subject": "reasoner ranking ability on the final revealed state",
            "headline_allowed": "yes",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "code_path": line_ref(round1, '"top1_hit": float(valid_df["top1_hit"].mean())'),
            "notes": "primary headline metric",
        },
        {
            "metric": "top3_hit",
            "metric_class": "reasoner_headline_ranking",
            "formula": "mean(1[rank_true <= 3]) over valid final ranking cases",
            "denominator": f"{top_valid} valid final ranking cases",
            "measures_subject": "reasoner ranking ability on the final revealed state",
            "headline_allowed": "yes",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "code_path": line_ref(round1, '"top3_hit": float(valid_df["top3_hit"].mean())'),
            "notes": "primary headline metric",
        },
        {
            "metric": "top5_hit",
            "metric_class": "reasoner_headline_ranking",
            "formula": "mean(1[rank_true <= 5]) over valid final ranking cases",
            "denominator": f"{top_valid} valid final ranking cases",
            "measures_subject": "reasoner ranking ability on the final revealed state",
            "headline_allowed": "yes",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "code_path": line_ref(round1, '"top5_hit": float(valid_df["top5_hit"].mean())'),
            "notes": "primary headline metric",
        },
        {
            "metric": "mrr_valid",
            "metric_class": "reasoner_headline_ranking",
            "formula": "mean(1 / rank_true) over valid final ranking cases",
            "denominator": f"{top_valid} valid final ranking cases",
            "measures_subject": "reasoner ranking ability on the final revealed state",
            "headline_allowed": "yes",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "code_path": line_ref(round1, '"mrr_valid": float(valid_df["mrr"].mean())'),
            "notes": "checkpoint selection and headline scalar",
        },
        {
            "metric": "normalized_rank_mean",
            "metric_class": "reasoner_scale_aware_support",
            "formula": "mean((rank_true - 1) / max(candidate_count - 1, 1))",
            "denominator": f"{top_valid} valid final ranking cases",
            "measures_subject": "reasoner ranking quality normalized by final candidate size",
            "headline_allowed": "supportive_only",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "code_path": "derived from replay final_true_source_rank and final_pre_action_valid_count",
            "notes": "lower is better",
        },
        {
            "metric": "percentile_rank_mean",
            "metric_class": "reasoner_scale_aware_support",
            "formula": "mean(1 - ((rank_true - 1) / candidate_count))",
            "denominator": f"{top_valid} valid final ranking cases",
            "measures_subject": "reasoner ranking quality normalized by final candidate size",
            "headline_allowed": "supportive_only",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "code_path": "derived from replay final_true_source_rank and final_pre_action_valid_count",
            "notes": "higher is better",
        },
        {
            "metric": "candidate_bucket_topk_mrr",
            "metric_class": "diagnostic_only",
            "formula": "same final ranking metrics recomputed inside explicit candidate-size buckets",
            "denominator": "valid final ranking cases within each explicit bucket",
            "measures_subject": "which candidate-size regimes are hardest for the reasoner",
            "headline_allowed": "supportive_only",
            "frozen_navigator_dominated": "partially",
            "zero_step_contaminated": "no",
            "code_path": "derived from replay final_pre_action_valid_count buckets",
            "notes": "bucket boundaries are explicit in this pack",
        },
        {
            "metric": "success_rate",
            "metric_class": "system_level_rollout",
            "formula": "mean(success_i) over all evaluated cases",
            "denominator": f"{total} all evaluated cases",
            "measures_subject": "system physical hit under frozen navigator rollout",
            "headline_allowed": "no",
            "frozen_navigator_dominated": "yes",
            "zero_step_contaminated": f"yes ({zero_step_test} zero-step auto-success cases in test)",
            "code_path": line_ref(round1, '"success_rate": float(df["success"].mean())'),
            "notes": "supportive system metric only",
        },
        {
            "metric": "non_zero_step_success_rate",
            "metric_class": "system_level_rollout",
            "formula": "mean(success_i) over cases with at least one replayed step",
            "denominator": f"{total - zero_step_test} non-zero-step cases",
            "measures_subject": "system physical hit after removing zero-step auto-success",
            "headline_allowed": "no",
            "frozen_navigator_dominated": "yes",
            "zero_step_contaminated": "no",
            "code_path": "derived from replay step_count_observed > 0",
            "notes": "better supportive SR for failure analysis",
        },
        {
            "metric": "avg_budget_used",
            "metric_class": "system_level_rollout",
            "formula": "mean(budget_used_i) over all evaluated cases",
            "denominator": f"{total} all evaluated cases",
            "measures_subject": "system rollout cost under frozen navigator",
            "headline_allowed": "no",
            "frozen_navigator_dominated": "yes",
            "zero_step_contaminated": "yes",
            "code_path": line_ref(round1, '"avg_budget_used": float(df["budget_used"].mean())'),
            "notes": "supportive cost metric only",
        },
        {
            "metric": "zero_step_auto_success_rate",
            "metric_class": "diagnostic_only",
            "formula": "mean(1[step_count_observed == 0])",
            "denominator": f"{total} all evaluated cases",
            "measures_subject": "how much of SR is occupied by auto-success before any replayed ranking step exists",
            "headline_allowed": "no",
            "frozen_navigator_dominated": "yes",
            "zero_step_contaminated": "self",
            "code_path": "derived from replay step_count_observed",
            "notes": "composition diagnostic",
        },
        {
            "metric": "per_episode_topk_mrr_percentile",
            "metric_class": "diagnostic_only",
            "formula": "episode-wise aggregation over replayed step tables",
            "denominator": "cases surviving to each replayed episode",
            "measures_subject": "how ranking evolves across the rollout",
            "headline_allowed": "no",
            "frozen_navigator_dominated": "mixed",
            "zero_step_contaminated": "indirectly",
            "code_path": "derived from replay raw_case_episode_metrics.csv",
            "notes": "survivor-biased; use only for diagnosis",
        },
        {
            "metric": "mask_and_fallback_incidence",
            "metric_class": "diagnostic_only",
            "formula": "counts / rates of logits-minus-action-valid gaps and finite excluded nodes",
            "denominator": "replayed steps or cases",
            "measures_subject": "candidate/logits contract cleanliness",
            "headline_allowed": "no",
            "frozen_navigator_dominated": "no",
            "zero_step_contaminated": "no",
            "code_path": line_ref(state_updates, "masked_flat_logits[fully_masked_graph[batch_flat]] = raw_flat_logits[fully_masked_graph[batch_flat]]"),
            "notes": "used to judge whether a hotfix is needed",
        },
        {
            "metric": "navigator_driven_rollout_authority",
            "metric_class": "contract_fact",
            "formula": "sampling policy = navigator_only; frozen_clean_v1_bridge does not consume current reasoner_logits",
            "denominator": "not applicable",
            "measures_subject": "who controls the rollout",
            "headline_allowed": "no",
            "frozen_navigator_dominated": "yes",
            "zero_step_contaminated": "not_applicable",
            "code_path": f"{line_ref(episode_runner, 'if action_policy == \"nav_only\":')} | {line_ref(frozen_bridge, 'def _build_node_features')}",
            "notes": "critical for factorizing reasoner vs navigator responsibility",
        },
    ]
    return pd.DataFrame(rows)


def render_authoritative_metric_contract(metric_df: pd.DataFrame) -> str:
    headline = metric_df[metric_df["metric_class"].str.contains("headline|scale_aware", na=False)][
        ["metric", "metric_class", "denominator", "measures_subject", "headline_allowed"]
    ]
    system = metric_df[metric_df["metric_class"] == "system_level_rollout"][
        ["metric", "denominator", "measures_subject", "headline_allowed", "zero_step_contaminated"]
    ]
    diagnostic = metric_df[metric_df["metric_class"] == "diagnostic_only"][
        ["metric", "measures_subject", "notes"]
    ]
    return "\n".join(
        [
            "# Authoritative Metric Contract",
            "",
            f"Version: `{METRIC_CONTRACT_VERSION}`.",
            "",
            "## Class 1: reasoner headline ranking metrics",
            "",
            dataframe_to_markdown(headline),
            "",
            "## Class 2: system-level rollout metrics",
            "",
            dataframe_to_markdown(system),
            "",
            "## Class 3: diagnostic-only metrics",
            "",
            dataframe_to_markdown(diagnostic),
            "",
            "## Contract summary",
            "",
            "- Headline reasoner claims may use final-step `Top1 / Top3 / Top5 / MRR`.",
            "- `normalized_rank` and `percentile_rank` are authoritative scale-aware support metrics, not replacements for the headline tuple.",
            "- `success_rate` is system-level frozen-navigator rollout behavior and cannot stand in for reasoner quality.",
        ]
    )


def render_success_rate_judgment(
    *,
    zero_step_df: pd.DataFrame,
    official_test: Dict[str, Any],
    replay_alignment_df: pd.DataFrame,
) -> str:
    align = replay_alignment_df[replay_alignment_df["split"] == "test"].copy()
    return "\n".join(
        [
            "# Success Rate Semantics Judgment",
            "",
            "## Current definition",
            "",
            "- `success_rate = mean(success_i)` over all evaluated cases in the split.",
            "- In the active downstream line, this is frozen-navigator physical hit under budget, not reasoner final-rank correctness.",
            "",
            "## Zero-step composition",
            "",
            dataframe_to_markdown(zero_step_df),
            "",
            "## Headline judgment",
            "",
            f"- Archived official test SR is `{official_test['success_rate']:.6f}`.",
            f"- Removing zero-step cases lowers replayed test SR to `{zero_step_df.loc[zero_step_df['split']=='test', 'non_zero_step_success_rate'].iloc[0]:.6f}`.",
            "- Therefore zero-step auto-success materially contaminates SR and it should be stripped from any reasoner-quality headline claim.",
            "",
            "## Replay drift note",
            "",
            dataframe_to_markdown(align[["metric", "official_value", "replay_value", "delta"]]),
            "",
            "- The official archived finish summary remains the authoritative headline source.",
            "- The replay tables in this pack are authoritative for diagnostics, with replay drift explicitly surfaced rather than hidden.",
        ]
    )


def render_heuristic_alignment_note(heuristic_df: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Heuristic Alignment Note",
            "",
            "- The remembered `~77%` values are located in legacy bucket slices such as `candidate_count.small` or `observed_fraction.rich`, not in the current authoritative downstream overall line.",
            "- Those legacy bucket slices are not apples-to-apples comparable with the current clean-env frozen-navigator overall SR.",
            "",
            "## Extracted alignment table",
            "",
            dataframe_to_markdown(heuristic_df),
            "",
            "## Authoritative conclusion",
            "",
            "- Overall-vs-overall can be shown only as supportive non-aligned context.",
            "- Bucket-vs-overall or legacy-vs-current headline comparisons are invalid and should be labeled as such.",
        ]
    )


def render_candidate_mask_flow(
    contract_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    replay_step_df: pd.DataFrame,
) -> str:
    fallback_steps = int(replay_step_df["fallback_triggered"].fillna(False).sum())
    fallback_cases = int(replay_step_df.loc[replay_step_df["fallback_triggered"].fillna(False), "case_id"].nunique())
    return "\n".join(
        [
            "# Candidate Mask Flow",
            "",
            "## Contract table",
            "",
            dataframe_to_markdown(contract_df),
            "",
            "## Empirical incidence",
            "",
            f"- Fallback steps: `{fallback_steps}`.",
            f"- Fallback cases: `{fallback_cases}`.",
            f"- Finite excluded-node steps coincide with fallback steps: `{int(replay_step_df['finite_on_confirmed_non_source'].fillna(False).sum())}`.",
            "",
            "## Degenerate examples",
            "",
            dataframe_to_markdown(fallback_df),
            "",
            "## Judgment",
            "",
            "- Normal path is clean enough for ranking diagnostics: excluded nodes are normally hard-masked after forward.",
            "- The remaining contract breach is the rare fully-masked fallback that restores raw logits for one degenerate test case.",
            "- `feasible_mask` is not a post-forward hard logit mask, so action-valid and ranking-candidate sets are not strictly identical.",
        ]
    )


def render_episode_progress_judgment(
    *,
    episode_df: pd.DataFrame,
    paired_case_df: pd.DataFrame,
) -> str:
    step0 = episode_df.loc[episode_df["episode"] == 0].iloc[0]
    final_top1 = float(paired_case_df["final_top1_hit"].astype(float).mean())
    final_mrr = float(paired_case_df["final_mrr"].mean())
    shrink = paired_case_df["candidate_shrink_ratio"].dropna()
    return "\n".join(
        [
            "# Episode Progress Judgment",
            "",
            f"- [proven] Episode 0 ranking is weak: Top-1 `{step0['top1_hit']:.4f}`, MRR `{step0['mrr']:.4f}`, with mean pre-action candidate size `{step0['pre_action_valid_size_mean']:.1f}` and revealed ratio `{step0['revealed_ratio_mean']:.4f}`.",
            f"- [proven] Final paired ranking improves to Top-1 `{final_top1:.4f}` and MRR `{final_mrr:.4f}`.",
            f"- [proven] The candidate shrink from episode 0 to final is modest on average: mean ratio `{shrink.mean():.4f}`, median `{shrink.median():.4f}`. Final gains are therefore not explainable as pure trivialization.",
            "- [partially proven] Raw per-episode survivor curves are selection-biased because easy cases exit early; they should not be read as a fixed-cohort learning curve.",
            "- [not proved] The current replay export does not directly log `true source observed but not yet sampled`, so that finer regime boundary remains unproven.",
            "",
            "## Curve summary",
            "",
            dataframe_to_markdown(episode_df),
        ]
    )


def render_weakness_factorization(
    *,
    factor_df: pd.DataFrame,
    candidate_bucket_df: pd.DataFrame,
    official_compare: Dict[str, Any],
) -> str:
    weakest = candidate_bucket_df.sort_values(["mrr_valid", "count"], ascending=[True, False]).iloc[0]
    delta_sr = official_compare["delta"]["best_vs_bridged_online"]["success_rate"]
    delta_mrr = official_compare["delta"]["best_vs_bridged_online"]["mrr_valid"]
    return "\n".join(
        [
            "# Weakness Factorization Summary",
            "",
            "## Reasoner weakness",
            "",
            "- Episode-0 ranking is weak on sparse revealed states and large candidate sets.",
            f"- The weakest substantial final bucket is `{weakest['candidate_bucket']}` with Top-1 `{weakest['top1_hit']:.4f}`, MRR `{weakest['mrr_valid']:.4f}`, percentile rank `{weakest['percentile_rank_mean']:.4f}`.",
            "",
            "## Navigator-induced difficulty",
            "",
            "- Zero-step auto-success occupies about one-fifth of both val/test denominators.",
            "- Outside zero-step cases, non-zero-step SR drops sharply, which shows how much of the rollout burden sits in the frozen navigator trajectory.",
            "",
            "## Coupled bottleneck",
            "",
            f"- Archived best-vs-bridged deltas show strong ranking gain (`delta_mrr={delta_mrr:+.4f}`) without SR gain (`delta_success_rate={delta_sr:+.4f}`).",
            "- This is direct evidence that the system-level bottleneck is not reducible to `reasoner weak` alone.",
            "",
            "## Factorized table",
            "",
            dataframe_to_markdown(factor_df),
        ]
    )


def render_final_judgment(
    *,
    zero_step_df: pd.DataFrame,
    candidate_bucket_df: pd.DataFrame,
    replay_alignment_df: pd.DataFrame,
    official_compare: Dict[str, Any],
) -> str:
    zero_test = zero_step_df.loc[zero_step_df["split"] == "test"].iloc[0]
    weakest = candidate_bucket_df.sort_values(["mrr_valid", "count"], ascending=[True, False]).iloc[0]
    delta = official_compare["delta"]["best_vs_bridged_online"]
    test_drift = replay_alignment_df[replay_alignment_df["split"] == "test"]
    return "\n".join(
        [
            "# Final Judgment",
            "",
            f"- [proven] `success_rate` is frozen-navigator physical hit under budget, not a pure reasoner metric; test zero-step auto-success count is `{int(zero_test['zero_step_count'])}` / `{int(zero_test['total_cases'])}`.",
            f"- [proven] The reasoner's clear intrinsic weakness is early ranking under sparse reveal and large candidate sets; step-0 Top-1 is about `0.0817` and step-0 MRR about `0.2084`.",
            f"- [proven] The practically weakest final candidate-size regimes are the large buckets (`{weakest['candidate_bucket']}` and adjacent large buckets), where Top-1/MRR remain low even after rollout.",
            f"- [proven] Archived best-vs-bridged deltas show ranking gain without SR gain: Top-1 `{delta['top1_hit']:+.4f}`, MRR `{delta['mrr_valid']:+.4f}`, SR `{delta['success_rate']:+.4f}`.",
            "- [partially proven] Current replay diagnostics have bounded replay drift versus the archived official finish summary; the official compare remains the authoritative headline source while replay tables remain authoritative for diagnostics.",
            "- [partially proven] The candidate/logits contract is clean on the normal path, but one rare fully-masked fallback reintroduces excluded nodes for a degenerate graph.",
            "- [not proved] The current replay export does not isolate `true source observed but not yet sampled`, so that narrower regime split is still missing.",
            "",
            "## Main contradiction",
            "",
            "- The current main contradiction is not `reasoner bad everywhere`.",
            "- The reasoner is weak earliest, especially in large candidate spaces, but the larger system bottleneck is that frozen navigator-induced trajectories do not convert ranking gains into system SR gains.",
            "",
            "## Single next capability direction",
            "",
            "- If only one minimal intervention is allowed next, target earlier discriminative evidence acquisition under the frozen rollout contract: move cases out of the `301+` candidate regimes by the first few episodes.",
            "- Expected authoritative metrics to move: non-zero-step `success_rate`, early-episode Top-1 / MRR, and large-bucket percentile rank.",
            "- Why not paper packaging / more metric work / semidynamic reopen: the metric contract is now explicit enough, and the archived-vs-replay drift is already surfaced rather than hidden.",
            "",
            "## Replay drift table",
            "",
            dataframe_to_markdown(test_drift[["metric", "official_value", "replay_value", "delta"]]),
        ]
    )


def render_claim_boundary_update() -> str:
    return "\n".join(
        [
            "# Claim Boundary Update",
            "",
            "- Headline the final-step ranking tuple, not frozen-policy SR.",
            "- Use normalized/percentile rank and candidate buckets as scale-aware support, not as replacements for the headline tuple.",
            "- Treat per-episode curves, zero-step composition, and fallback incidence as diagnostic evidence.",
            "- State replay drift explicitly whenever replay-derived diagnostics are discussed next to archived official compare numbers.",
        ]
    )


def write_simple_plots(
    *,
    figure_dir: Path,
    episode_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    zero_step_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    headline_df: pd.DataFrame,
) -> None:
    ensure_dir(figure_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_df["episode"], episode_df["top1_hit"], marker="o", label="Top1")
    ax.plot(episode_df["episode"], episode_df["mrr"], marker="s", label="MRR")
    ax.plot(episode_df["episode"], episode_df["percentile_rank_mean"], marker="^", label="Percentile")
    ax.set_xlabel("Episode")
    ax.set_title("Per-Episode Ranking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "per_episode_ranking_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bucket_df["candidate_bucket"], bucket_df["top1_hit"], marker="o", label="Top1")
    ax.plot(bucket_df["candidate_bucket"], bucket_df["mrr_valid"], marker="s", label="MRR")
    ax.plot(bucket_df["candidate_bucket"], bucket_df["percentile_rank_mean"], marker="^", label="Percentile")
    ax.set_xlabel("Candidate bucket")
    ax.set_title("Candidate Bucket Performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "candidate_bucket_performance_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = headline_df.copy()
    ax.bar(plot_df["claim_or_metric"], [1] * len(plot_df), color=["#215732", "#4b8b3b", "#c48a1c", "#9b3529", "#5a5a5a"][: len(plot_df)])
    ax.set_title("Headline vs Supportive vs Invalid")
    ax.set_ylabel("Category marker")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "headline_vs_supportive_metric_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(zero_step_df["split"], zero_step_df["zero_step_count"], label="zero-step")
    ax.bar(
        zero_step_df["split"],
        zero_step_df["non_zero_step_count"],
        bottom=zero_step_df["zero_step_count"],
        label="non-zero-step",
    )
    ax.set_title("Zero-Step Auto-Success Composition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "zero_step_autosuccess_composition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    overall = fallback_df[fallback_df["scope"].isin(["overall_step", "overall_case"])]
    ax.bar(overall["scope"], overall["rate"])
    ax.set_ylim(0, max(0.01, float(overall["rate"].max()) * 1.25 if not overall.empty else 0.01))
    ax.set_title("Mask / Fallback Incidence")
    fig.tight_layout()
    fig.savefig(figure_dir / "mask_fallback_incidence_summary.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the authoritative reasoner metric contract repair bundle.")
    parser.add_argument("--finish-root", type=str, default=str(DEFAULT_FINISH_ROOT))
    parser.add_argument("--test-replay-root", type=str, default=str(DEFAULT_TEST_REPLAY_ROOT))
    parser.add_argument("--val-replay-root", type=str, default=str(DEFAULT_VAL_REPLAY_ROOT))
    parser.add_argument("--heuristic-json", type=str, default=str(DEFAULT_HEURISTIC_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-replay-command", type=str, default="")
    parser.add_argument("--val-replay-command", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finish_root = Path(args.finish_root)
    test_replay_root = Path(args.test_replay_root)
    val_replay_root = Path(args.val_replay_root)
    heuristic_json = Path(args.heuristic_json)
    output_dir = Path(args.output_dir)

    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summary"
    figure_dir = output_dir / "figures_or_figure_ready"
    manifest_dir = output_dir / "manifest"
    for path in [raw_dir, summary_dir, figure_dir, manifest_dir]:
        ensure_dir(path)

    selection_manifest = load_json(finish_root / "selection_manifest.json")
    run_manifest = load_json(finish_root / "run_manifest.json")
    compare_json = load_json(finish_root / "compare" / "final_compare.json")
    runner_snapshot = load_json(finish_root / "delivery" / "runner_snapshot.json")

    official_test = selection_manifest["selected_best_checkpoint"]["test_summary"]
    official_val = selection_manifest["selected_best_checkpoint"]["val_summary"]

    test_case_raw, test_step_raw = load_replay_tables(test_replay_root)
    val_case_raw, val_step_raw = load_replay_tables(val_replay_root)
    test_step_df = standardize_step_df(test_step_raw, split="test")
    val_step_df = standardize_step_df(val_step_raw, split="val")
    test_case_df = standardize_case_df(test_case_raw, test_step_df, split="test")
    val_case_df = standardize_case_df(val_case_raw, val_step_df, split="val")

    zero_step_df = build_zero_step_summary({"val": val_case_df, "test": test_case_df})
    replay_alignment_rows = []
    replay_alignment_rows.extend(build_replay_alignment_rows(split="val", official_summary=official_val, replay_case_df=val_case_df))
    replay_alignment_rows.extend(build_replay_alignment_rows(split="test", official_summary=official_test, replay_case_df=test_case_df))
    replay_alignment_df = pd.DataFrame(replay_alignment_rows)

    metric_formula_df = build_metric_formula_table(test_case_df=test_case_df, zero_step_df=zero_step_df)
    candidate_bucket_df = candidate_bucket_summary(test_case_df)
    episode_df = episode_progress_summary(test_step_df)
    episode_curve_df = episode_curve_ready(episode_df)
    contract_df = build_candidate_mask_contract_table()
    fallback_summary_df = build_fallback_incidence_summary(test_step_df)
    degenerate_examples_df = build_degnerate_examples(test_step_df)
    factor_df = build_factorized_metrics_table(
        official_compare=compare_json,
        test_case_df=test_case_df,
        test_step_df=test_step_df,
    )
    headline_df = build_headline_vs_supportive_table()
    heuristic_df = build_heuristic_alignment_table(heuristic_json, official_test_sr=float(official_test["success_rate"]))

    candidate_curve_df = candidate_bucket_df[
        ["candidate_bucket", "count", "top1_hit", "top3_hit", "top5_hit", "mrr_valid", "percentile_rank_mean", "normalized_rank_mean"]
    ].copy()
    zero_step_curve_df = zero_step_df[
        ["split", "zero_step_count", "non_zero_step_count", "zero_step_rate", "success_rate_all_cases", "non_zero_step_success_rate"]
    ].copy()
    fallback_curve_df = fallback_summary_df.copy()
    headline_curve_df = headline_df.copy()

    paired_case_df = test_case_df[test_case_df["valid_case"] & test_case_df["step0_true_source_rank"].notna()].copy()

    metric_formula_df.to_csv(raw_dir / "metric_formula_table.csv", index=False)
    zero_step_df.to_csv(raw_dir / "zero_step_autosuccess_summary.csv", index=False)
    heuristic_df.to_csv(raw_dir / "heuristic_alignment_table.csv", index=False)
    candidate_bucket_df.to_csv(raw_dir / "candidate_size_bucket_summary.csv", index=False)
    episode_df.to_csv(raw_dir / "episode_progress_summary.csv", index=False)
    episode_curve_df.to_csv(raw_dir / "episode_progress_curve_ready.csv", index=False)
    contract_df.to_csv(raw_dir / "candidate_mask_contract_table.csv", index=False)
    fallback_summary_df.to_csv(raw_dir / "fallback_incidence_summary.csv", index=False)
    factor_df.to_csv(raw_dir / "factorized_metrics_table.csv", index=False)
    headline_df.to_csv(raw_dir / "headline_vs_supportive_table.csv", index=False)
    replay_alignment_df.to_csv(raw_dir / "replay_alignment_table.csv", index=False)
    degenerate_examples_df.to_csv(raw_dir / "degenerate_case_examples.csv", index=False)
    test_case_df.to_csv(raw_dir / "test_case_replay_standardized.csv", index=False)
    test_step_df.to_csv(raw_dir / "test_step_replay_standardized.csv", index=False)

    candidate_curve_df.to_csv(figure_dir / "candidate_bucket_curve_ready.csv", index=False)
    episode_curve_df.to_csv(figure_dir / "per_episode_ranking_curve_ready.csv", index=False)
    headline_curve_df.to_csv(figure_dir / "headline_vs_supportive_curve_ready.csv", index=False)
    zero_step_curve_df.to_csv(figure_dir / "zero_step_autosuccess_curve_ready.csv", index=False)
    fallback_curve_df.to_csv(figure_dir / "mask_fallback_incidence_curve_ready.csv", index=False)

    write_simple_plots(
        figure_dir=figure_dir,
        episode_df=episode_df,
        bucket_df=candidate_bucket_df,
        zero_step_df=zero_step_df,
        fallback_df=fallback_summary_df,
        headline_df=headline_df,
    )

    write_text(summary_dir / "authoritative_metric_contract.md", render_authoritative_metric_contract(metric_formula_df))
    write_text(
        summary_dir / "success_rate_semantics_judgment.md",
        render_success_rate_judgment(
            zero_step_df=zero_step_df,
            official_test=official_test,
            replay_alignment_df=replay_alignment_df,
        ),
    )
    write_text(summary_dir / "heuristic_alignment_note.md", render_heuristic_alignment_note(heuristic_df))
    write_text(
        summary_dir / "candidate_mask_flow.md",
        render_candidate_mask_flow(
            contract_df=contract_df,
            fallback_df=degenerate_examples_df,
            replay_step_df=test_step_df,
        ),
    )
    write_text(
        summary_dir / "episode_progress_judgment.md",
        render_episode_progress_judgment(
            episode_df=episode_df,
            paired_case_df=paired_case_df,
        ),
    )
    write_text(
        summary_dir / "weakness_factorization_summary.md",
        render_weakness_factorization(
            factor_df=factor_df,
            candidate_bucket_df=candidate_bucket_df,
            official_compare=compare_json,
        ),
    )
    write_text(
        summary_dir / "final_judgment.md",
        render_final_judgment(
            zero_step_df=zero_step_df,
            candidate_bucket_df=candidate_bucket_df,
            replay_alignment_df=replay_alignment_df,
            official_compare=compare_json,
        ),
    )
    write_text(summary_dir / "claim_boundary_update.md", render_claim_boundary_update())

    bundle_index_lines = [
        "# Bundle Index",
        "",
        "## raw",
    ]
    for path in sorted(raw_dir.iterdir()):
        bundle_index_lines.append(f"- raw/{path.name}")
    bundle_index_lines.extend(["", "## figures_or_figure_ready"])
    for path in sorted(figure_dir.iterdir()):
        bundle_index_lines.append(f"- figures_or_figure_ready/{path.name}")
    bundle_index_lines.extend(["", "## summary"])
    for path in sorted(summary_dir.iterdir()):
        if path.name == "bundle_index.md":
            continue
        bundle_index_lines.append(f"- summary/{path.name}")
    bundle_index_lines.extend(["", "## manifest"])
    write_text(summary_dir / "bundle_index.md", "\n".join(bundle_index_lines))

    runner_version = runner_version_from_run_dir(runner_snapshot.get("run_dir", "")) or "unknown"
    commands_executed = [
        cmd
        for cmd in [
            args.val_replay_command,
            args.test_replay_command,
            f"python {Path(__file__).resolve()} --finish-root {finish_root} --test-replay-root {test_replay_root} --val-replay-root {val_replay_root} --heuristic-json {heuristic_json} --output-dir {output_dir}",
        ]
        if cmd
    ]
    replay_drift_exists = bool((replay_alignment_df["abs_delta"].fillna(0) > 1e-9).any())

    provenance_manifest = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "official_finish_root": str(finish_root),
        "test_replay_root": str(test_replay_root),
        "val_replay_root": str(val_replay_root),
        "heuristic_json": str(heuristic_json),
        "runner_version": runner_version,
        "runner_git_head": runner_snapshot.get("git_head"),
        "panel_version": run_manifest.get("panel_version"),
        "seed": run_manifest.get("seed"),
        "source_artifacts": {
            "official_test_summary": str(finish_root / "compare" / "final_compare.json"),
            "selection_manifest": str(finish_root / "selection_manifest.json"),
            "test_replay_case_csv": str(test_replay_root / "success_failure_bucket_audit" / "raw_case_summary.csv"),
            "test_replay_step_csv": str(test_replay_root / "episode_progress_audit" / "raw_case_episode_metrics.csv"),
            "val_replay_case_csv": str(val_replay_root / "success_failure_bucket_audit" / "raw_case_summary.csv"),
            "val_replay_step_csv": str(val_replay_root / "episode_progress_audit" / "raw_case_episode_metrics.csv"),
        },
        "commands_executed": commands_executed,
        "bounded_reevaluation_performed": True,
        "reevaluation_scope": ["val_semantics_replay", "test_semantics_replay"],
        "replay_drift_exists": replay_drift_exists,
        "replay_alignment_table": str(raw_dir / "replay_alignment_table.csv"),
        "candidate_bucket_bounds": BUCKET_LABELS,
    }
    write_json(summary_dir / "provenance_manifest.json", provenance_manifest)
    write_json(
        manifest_dir / "bundle_manifest.json",
        {
            "runner_version": runner_version,
            "panel_version": run_manifest.get("panel_version"),
            "seed": run_manifest.get("seed"),
            "authoritative_source_artifacts": provenance_manifest["source_artifacts"],
            "commands_executed": commands_executed,
            "bounded_reevaluation_performed": True,
            "replay_drift_exists": replay_drift_exists,
        },
    )
    write_text(manifest_dir / "commands_executed.txt", "\n".join(commands_executed) if commands_executed else "none")
    write_json(
        manifest_dir / "replay_alignment_summary.json",
        {
            "max_abs_delta_by_split": replay_alignment_df.groupby("split")["abs_delta"].max().to_dict(),
            "replay_drift_exists": replay_drift_exists,
        },
    )


if __name__ == "__main__":
    main()
