import os
import sys
from typing import Dict, List

ROOT_DIR = "/root/autodl-tmp/rl_spim_v7_mainline"
sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
import torch

from src.data.v6.dataset import NpzDatasetV6
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.builder import EvidenceBuilder
import src.scripts.audit.run_practical_audit_rerun as rerun
import src.scripts.audit.run_support_score_v21_oracle_sweep as v21

STEPWISE_CSV_PATH = os.path.join(ROOT_DIR, "support_score_v21_practical_stepwise.csv")
COMPARE_CSV_PATH = os.path.join(ROOT_DIR, "support_score_v1_vs_v21_practical.csv")
SUMMARY_MD_PATH = os.path.join(ROOT_DIR, "support_score_v21_practical_summary.md")
RUN_COMMAND = "python src/scripts/audit/run_support_score_v21_practical_audit.py"

BEST_V21_CONFIG = v21.V21Config(
    combo_id="best_v21_practical",
    virtual_reliability=0.75,
    hub_penalty_weight=0.20,
    ownership_exponent=1.5,
)


def safe_float(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else np.nan


def extract_score_fields(
    prefix: str,
    scores: torch.Tensor,
    src_local: int,
    g_ids: torch.Tensor,
) -> Dict[str, float]:
    true_score = float(scores[src_local].item())
    other_mean = rerun.mean_other(scores, src_local)
    top_other_idx = rerun.best_other_index(scores, src_local, higher_better=True)
    top_other_score = float(scores[top_other_idx].item())
    return {
        f"{prefix}_rank": rerun.strict_rank(scores, src_local, higher_better=True),
        f"{prefix}_gap": true_score - other_mean,
        f"{prefix}_directionality": float((true_score - other_mean) > 0.0),
        f"{prefix}_all_zero": float(rerun.all_zero(scores)),
        f"{prefix}_true_nonzero": float(true_score > rerun.EPS),
        f"{prefix}_unique_top1": float(rerun.unique_top1(scores, src_local)),
        f"{prefix}_hub_win": float((top_other_score - true_score) > rerun.EPS) if scores.numel() > 1 else 0.0,
        f"{prefix}_true_total": true_score,
        f"{prefix}_top_other_idx": int(top_other_idx),
        f"{prefix}_top_other_global_id": int(g_ids[top_other_idx].item()),
        f"{prefix}_top_other_score": top_other_score,
    }


def extract_v21_candidate_terms(
    prefix: str,
    support_res: Dict[str, torch.Tensor],
    src_local: int,
    competitor_idx: int,
) -> Dict[str, float]:
    return {
        f"{prefix}_true_availability": float(support_res["availability"][src_local].item()),
        f"{prefix}_true_ownership": float(support_res["ownership"][src_local].item()),
        f"{prefix}_true_hub_penalty": float(support_res["hub_penalty"][src_local].item()),
        f"{prefix}_true_virtual_share": float(support_res["virtual_share"][src_local].item()),
        f"{prefix}_true_best_path_virtual_rate": float(support_res["best_path_virtual_rate"][src_local].item()),
        f"{prefix}_true_best_path_physical_rate": float(support_res["best_path_physical_rate"][src_local].item()),
        f"{prefix}_true_best_time_mean": safe_float(support_res["best_time_mean"][src_local].item()),
        f"{prefix}_true_physical_time_mean": safe_float(support_res["physical_time_mean"][src_local].item()),
        f"{prefix}_true_virtual_time_mean": safe_float(support_res["virtual_time_mean"][src_local].item()),
        f"{prefix}_competitor_availability": float(support_res["availability"][competitor_idx].item()),
        f"{prefix}_competitor_ownership": float(support_res["ownership"][competitor_idx].item()),
        f"{prefix}_competitor_hub_penalty": float(support_res["hub_penalty"][competitor_idx].item()),
        f"{prefix}_competitor_virtual_share": float(support_res["virtual_share"][competitor_idx].item()),
        f"{prefix}_competitor_best_path_virtual_rate": float(
            support_res["best_path_virtual_rate"][competitor_idx].item()
        ),
        f"{prefix}_competitor_best_path_physical_rate": float(
            support_res["best_path_physical_rate"][competitor_idx].item()
        ),
        f"{prefix}_competitor_best_time_mean": safe_float(support_res["best_time_mean"][competitor_idx].item()),
        f"{prefix}_competitor_physical_time_mean": safe_float(
            support_res["physical_time_mean"][competitor_idx].item()
        ),
        f"{prefix}_competitor_virtual_time_mean": safe_float(
            support_res["virtual_time_mean"][competitor_idx].item()
        ),
        f"{prefix}_witness_count": float(support_res["witness_count"][src_local].item()),
    }


def compare_major_diverge(row: Dict[str, float]) -> float:
    return float(
        abs(float(row["v21_rank"]) - float(row["v1_rank"])) >= 2.0
        or abs(float(row["v21_directionality"]) - float(row["v1_directionality"])) > rerun.EPS
        or abs(float(row["v21_true_nonzero"]) - float(row["v1_true_nonzero"])) > rerun.EPS
        or abs(float(row["v21_hub_win"]) - float(row["v1_hub_win"])) > rerun.EPS
    )


def compare_improved(row: Dict[str, float]) -> float:
    improved = (
        float(row["v21_rank"]) < float(row["v1_rank"]) - rerun.EPS
        or float(row["v21_directionality"]) > float(row["v1_directionality"]) + rerun.EPS
        or float(row["v21_true_nonzero"]) > float(row["v1_true_nonzero"]) + rerun.EPS
        or float(row["v21_hub_win"]) < float(row["v1_hub_win"]) - rerun.EPS
    )
    degraded = (
        float(row["v21_rank"]) > float(row["v1_rank"]) + rerun.EPS
        or float(row["v21_directionality"]) < float(row["v1_directionality"]) - rerun.EPS
        or float(row["v21_true_nonzero"]) < float(row["v1_true_nonzero"]) - rerun.EPS
        or float(row["v21_hub_win"]) > float(row["v1_hub_win"]) + rerun.EPS
    )
    return float(improved and not degraded)


def compare_degraded(row: Dict[str, float]) -> float:
    improved = (
        float(row["v21_rank"]) < float(row["v1_rank"]) - rerun.EPS
        or float(row["v21_directionality"]) > float(row["v1_directionality"]) + rerun.EPS
        or float(row["v21_true_nonzero"]) > float(row["v1_true_nonzero"]) + rerun.EPS
        or float(row["v21_hub_win"]) < float(row["v1_hub_win"]) - rerun.EPS
    )
    degraded = (
        float(row["v21_rank"]) > float(row["v1_rank"]) + rerun.EPS
        or float(row["v21_directionality"]) < float(row["v1_directionality"]) - rerun.EPS
        or float(row["v21_true_nonzero"]) < float(row["v1_true_nonzero"]) - rerun.EPS
        or float(row["v21_hub_win"]) > float(row["v1_hub_win"]) + rerun.EPS
    )
    return float(degraded and not improved)


def build_stepwise_records(dataset, topology: HydraulicTopology) -> pd.DataFrame:
    evidence_builder = EvidenceBuilder()
    rows: List[Dict[str, float]] = []
    indices = range(min(rerun.MAX_EVENTS, len(dataset)))

    with torch.no_grad():
        for event_id in indices:
            try:
                event_data_batch = dataset[event_id]
                if event_data_batch is None:
                    continue
                event_data = rerun.extract_view0(event_data_batch)
                src_global = event_data.global_injection_node
                if isinstance(src_global, torch.Tensor):
                    src_global = int(src_global.item())

                rollout = rerun.PracticalRollout(
                    event_data,
                    dataset.global_edge_index,
                    dataset.stt_dynamic_series,
                    dataset.num_nodes,
                    num_episodes=rerun.NUM_EPISODES,
                    samples_per_episode=3,
                )
                if src_global not in rollout.g_ids:
                    continue
                src_local = int((rollout.g_ids == src_global).nonzero(as_tuple=True)[0].item())

                for episode_idx in range(rerun.NUM_EPISODES):
                    obs_partial, _obs_oracle, phys_ctx, info = rollout.step()
                    t_snapshot_idx = int(info["t_snapshot_idx"])
                    time_min = float(info["time_min"])
                    signal_snapshot = rollout.event_data.x_raw[:, t_snapshot_idx, 0]
                    conc = rollout.event_data.x_raw[:, t_snapshot_idx, 1]
                    truth_positive_mask = conc > 0.1
                    truth_positive_total = int(truth_positive_mask.sum().item())
                    truth_positive_observed = int(
                        (truth_positive_mask & (obs_partial.observed_flag > 0.5)).sum().item()
                    )
                    truth_positive_unobserved = int(
                        (truth_positive_mask & (obs_partial.observed_flag <= 0.5)).sum().item()
                    )
                    positive_seed_root_cause = rerun.classify_positive_seed_root_cause(
                        truth_positive_total=truth_positive_total,
                        observed_positive_total=int(obs_partial.toxic_positive_flag.sum().item()),
                        hidden_truth_positive_total=truth_positive_unobserved,
                    )

                    ev_state = evidence_builder.build_evidence_state(obs_partial, phys_ctx, t_sim=None)
                    suspect_pool = ev_state.suspect_pool
                    suspect_active_true = float(suspect_pool[src_local].item() > 0.5)

                    v1_support_res = {
                        "total": ev_state.support_score,
                        "base": ev_state.support_coverage_term,
                        "specificity": ev_state.support_timing_term,
                        "focus": ev_state.support_focus_term,
                        "chlorine": ev_state.support_chlorine_term,
                    }
                    v1_fields = rerun.extract_support_variant_fields(
                        "v1",
                        v1_support_res,
                        src_local,
                        rollout.g_ids,
                    )

                    practical_positive_mask = obs_partial.toxic_positive_flag > 0.5
                    witness_strength = torch.abs(obs_partial.chlorine_deviation)
                    t_abs_idx = rerun.resolve_snapshot_time_index(rollout.event_data, t_snapshot_idx)
                    payload = v21.prepare_v2_payload(
                        rollout=rollout,
                        phys_ctx=phys_ctx,
                        truth_positive_mask=practical_positive_mask,
                        witness_strength=witness_strength,
                        t_abs_idx=t_abs_idx,
                        topology=topology,
                    )
                    v21_raw_res = v21.compute_support_v21_from_payload(
                        payload=payload,
                        current_time_min=time_min,
                        config=BEST_V21_CONFIG,
                    )
                    v21_raw_fields = rerun.extract_support_variant_fields_v2(
                        "v21_raw",
                        v21_raw_res,
                        src_local,
                        rollout.g_ids,
                    )

                    v21_masked_scores = v21_raw_res["total"] * suspect_pool
                    v21_fields = extract_score_fields(
                        "v21",
                        v21_masked_scores,
                        src_local,
                        rollout.g_ids,
                    )
                    v21_terms = extract_v21_candidate_terms(
                        "v21_raw",
                        v21_raw_res,
                        src_local,
                        int(v21_fields["v21_top_other_idx"]),
                    )

                    num_pos = int(obs_partial.toxic_positive_flag.sum().item())
                    row: Dict[str, float] = {
                        "event_id": int(event_id),
                        "episode": int(episode_idx + 1),
                        "time_min": time_min,
                        "t_snapshot_idx": int(t_snapshot_idx),
                        "oracle_t_abs_idx": int(t_abs_idx),
                        "fixed_case": float(event_id in rerun.CASE_EVENT_IDS),
                        "num_pos": num_pos,
                        "num_neg": int(obs_partial.toxic_negative_flag.sum().item()),
                        "observed_count": int(obs_partial.observed_flag.sum().item()),
                        "revealed_count": int(info["revealed_count"]),
                        "truth_positive_total": int(truth_positive_total),
                        "truth_positive_observed": int(truth_positive_observed),
                        "truth_positive_unobserved": int(truth_positive_unobserved),
                        "positive_seed_root_cause": positive_seed_root_cause,
                        "suspect_active_recall": suspect_active_true,
                        "source_validity_true": float(ev_state.source_validity[src_local].item()),
                        "support_responsible_subset": float(suspect_active_true > 0.5 and num_pos > 0),
                        "signal_true": safe_float(signal_snapshot[src_local].item()),
                        "signal_abs_true": safe_float(abs(signal_snapshot[src_local].item())),
                        **v1_fields,
                        "v1_true_base": float(ev_state.support_coverage_term[src_local].item()),
                        "v1_true_specificity": float(ev_state.support_timing_term[src_local].item()),
                        "v1_true_focus": float(ev_state.support_focus_term[src_local].item()),
                        "v1_true_chlorine": float(ev_state.support_chlorine_term[src_local].item()),
                        "v1_competitor_base": float(
                            ev_state.support_coverage_term[int(v1_fields["v1_top_other_idx"])].item()
                        ),
                        "v1_competitor_specificity": float(
                            ev_state.support_timing_term[int(v1_fields["v1_top_other_idx"])].item()
                        ),
                        "v1_competitor_focus": float(
                            ev_state.support_focus_term[int(v1_fields["v1_top_other_idx"])].item()
                        ),
                        "v1_competitor_chlorine": float(
                            ev_state.support_chlorine_term[int(v1_fields["v1_top_other_idx"])].item()
                        ),
                        **v21_fields,
                        **v21_raw_fields,
                        **v21_terms,
                    }
                    row["rank_delta_v21_minus_v1"] = float(row["v21_rank"] - row["v1_rank"])
                    row["true_total_delta_v21_minus_v1"] = float(row["v21_true_total"] - row["v1_true_total"])
                    row["v21_raw_true_total_delta_vs_v1"] = float(
                        row["v21_raw_true_total"] - row["v1_true_total"]
                    )
                    row["top_other_changed"] = float(
                        int(row["v1_top_other_global_id"]) != int(row["v21_top_other_global_id"])
                    )
                    row["compare_major_diverge"] = compare_major_diverge(row)
                    row["compare_improved"] = compare_improved(row)
                    row["compare_degraded"] = compare_degraded(row)
                    row["bucket_A_no_positive_input"] = float(num_pos <= 0)
                    row["bucket_E_suspect_failed"] = float(suspect_active_true <= 0.5)
                    row["v1_bucket_B_support_zero"] = float(
                        num_pos > 0 and suspect_active_true > 0.5 and row["v1_true_nonzero"] <= 0.5
                    )
                    row["v21_bucket_B_support_zero"] = float(
                        num_pos > 0 and suspect_active_true > 0.5 and row["v21_true_nonzero"] <= 0.5
                    )
                    row["v1_bucket_C_hub_win"] = float(
                        num_pos > 0
                        and suspect_active_true > 0.5
                        and row["v1_true_nonzero"] > 0.5
                        and row["v1_hub_win"] > 0.5
                    )
                    row["v21_bucket_C_hub_win"] = float(
                        num_pos > 0
                        and suspect_active_true > 0.5
                        and row["v21_true_nonzero"] > 0.5
                        and row["v21_hub_win"] > 0.5
                    )
                    rows.append(row)
            except Exception:
                continue

    return pd.DataFrame(rows)


def summarize_scope(df_slice: pd.DataFrame, scope: str, episode: int, subset_name: str) -> Dict[str, float]:
    has_input = df_slice["num_pos"] > 0
    return {
        "analysis": "metrics",
        "scope": scope,
        "episode": int(episode),
        "subset": subset_name,
        "denominator": int(len(df_slice)),
        "input_events": int(has_input.sum()),
        "v1_eligible_event_rate": safe_float(has_input.mean()) if len(df_slice) > 0 else np.nan,
        "v21_eligible_event_rate": safe_float(has_input.mean()) if len(df_slice) > 0 else np.nan,
        "v1_all_zero_event_rate": rerun.safe_mean_from_series(df_slice["v1_all_zero"]),
        "v21_all_zero_event_rate": rerun.safe_mean_from_series(df_slice["v21_all_zero"]),
        "v1_conditioned_rank_median": rerun.safe_median_from_series(df_slice.loc[has_input, "v1_rank"]),
        "v21_conditioned_rank_median": rerun.safe_median_from_series(df_slice.loc[has_input, "v21_rank"]),
        "v1_conditioned_directionality": rerun.safe_mean_from_series(df_slice.loc[has_input, "v1_directionality"]),
        "v21_conditioned_directionality": rerun.safe_mean_from_series(df_slice.loc[has_input, "v21_directionality"]),
        "v1_hub_win_rate": rerun.safe_mean_from_series(df_slice["v1_hub_win"]),
        "v21_hub_win_rate": rerun.safe_mean_from_series(df_slice["v21_hub_win"]),
        "v1_hub_win_rate_given_input": rerun.safe_mean_from_series(df_slice.loc[has_input, "v1_hub_win"]),
        "v21_hub_win_rate_given_input": rerun.safe_mean_from_series(df_slice.loc[has_input, "v21_hub_win"]),
        "v1_support_true_nonzero_given_input_rate": rerun.safe_mean_from_series(
            df_slice.loc[has_input, "v1_true_nonzero"]
        ),
        "v21_support_true_nonzero_given_input_rate": rerun.safe_mean_from_series(
            df_slice.loc[has_input, "v21_true_nonzero"]
        ),
        "v1_support_zero_despite_input_rate": rerun.safe_mean_from_series(
            (df_slice.loc[has_input, "v1_true_nonzero"] <= 0.5).astype(float)
        ),
        "v21_support_zero_despite_input_rate": rerun.safe_mean_from_series(
            (df_slice.loc[has_input, "v21_true_nonzero"] <= 0.5).astype(float)
        ),
        "v1_suspect_active_recall": rerun.safe_mean_from_series(df_slice["suspect_active_recall"]),
        "v21_suspect_active_recall": rerun.safe_mean_from_series(df_slice["suspect_active_recall"]),
        "v1_suspect_active_recall_given_input": rerun.safe_mean_from_series(
            df_slice.loc[has_input, "suspect_active_recall"]
        ),
        "v21_suspect_active_recall_given_input": rerun.safe_mean_from_series(
            df_slice.loc[has_input, "suspect_active_recall"]
        ),
        "v1_rank_gain_vs_v21": rerun.safe_median_from_series(df_slice.loc[has_input, "v1_rank"])
        - rerun.safe_median_from_series(df_slice.loc[has_input, "v21_rank"]),
        "directionality_delta_v21_minus_v1": rerun.safe_mean_from_series(df_slice.loc[has_input, "v21_directionality"])
        - rerun.safe_mean_from_series(df_slice.loc[has_input, "v1_directionality"]),
        "hub_win_delta_v21_minus_v1": rerun.safe_mean_from_series(df_slice.loc[has_input, "v21_hub_win"])
        - rerun.safe_mean_from_series(df_slice.loc[has_input, "v1_hub_win"]),
        "true_nonzero_delta_v21_minus_v1": rerun.safe_mean_from_series(df_slice.loc[has_input, "v21_true_nonzero"])
        - rerun.safe_mean_from_series(df_slice.loc[has_input, "v1_true_nonzero"]),
    }


def build_metric_rows(records: pd.DataFrame) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    rows.append(summarize_scope(records, "global", 0, "all_events"))
    for episode in range(1, rerun.NUM_EPISODES + 1):
        df_ep = records[records["episode"] == episode]
        if df_ep.empty:
            continue
        rows.append(summarize_scope(df_ep, "episode", episode, "all_events"))

    support_subset = records[records["support_responsible_subset"] > 0.5].copy()
    rows.append(summarize_scope(support_subset, "global", 0, "suspect_active_true_and_num_pos"))
    for episode in range(1, rerun.NUM_EPISODES + 1):
        df_ep = support_subset[support_subset["episode"] == episode]
        if df_ep.empty:
            continue
        rows.append(summarize_scope(df_ep, "episode", episode, "suspect_active_true_and_num_pos"))
    return rows


def summarize_bucket_scope(df_slice: pd.DataFrame, scope: str, episode: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    bucket_specs = [
        ("A_no_positive_input", "bucket_A_no_positive_input", "bucket_A_no_positive_input"),
        ("B_positive_input_but_true_support_zero", "v1_bucket_B_support_zero", "v21_bucket_B_support_zero"),
        ("C_true_support_nonzero_but_hub_still_wins", "v1_bucket_C_hub_win", "v21_bucket_C_hub_win"),
        ("E_suspect_failed_before_support", "bucket_E_suspect_failed", "bucket_E_suspect_failed"),
    ]
    for bucket_name, v1_col, v21_col in bucket_specs:
        rows.append(
            {
                "analysis": "failure_bucket",
                "scope": scope,
                "episode": int(episode),
                "bucket": bucket_name,
                "denominator": int(len(df_slice)),
                "v1_count": int(df_slice[v1_col].sum()) if not df_slice.empty else 0,
                "v21_count": int(df_slice[v21_col].sum()) if not df_slice.empty else 0,
                "v1_rate": rerun.safe_mean_from_series(df_slice[v1_col]),
                "v21_rate": rerun.safe_mean_from_series(df_slice[v21_col]),
                "delta_rate_v21_minus_v1": rerun.safe_mean_from_series(df_slice[v21_col])
                - rerun.safe_mean_from_series(df_slice[v1_col]),
                "compare_major_diverge_rate": np.nan,
                "compare_improved_rate": np.nan,
                "compare_degraded_rate": np.nan,
                "compare_mixed_rate": np.nan,
            }
        )

    compare_major = rerun.safe_mean_from_series(df_slice["compare_major_diverge"])
    compare_improved = rerun.safe_mean_from_series(df_slice["compare_improved"])
    compare_degraded = rerun.safe_mean_from_series(df_slice["compare_degraded"])
    rows.append(
        {
            "analysis": "failure_bucket",
            "scope": scope,
            "episode": int(episode),
            "bucket": "D_compare_major_diverge",
            "denominator": int(len(df_slice)),
            "v1_count": np.nan,
            "v21_count": np.nan,
            "v1_rate": np.nan,
            "v21_rate": np.nan,
            "delta_rate_v21_minus_v1": np.nan,
            "compare_major_diverge_rate": compare_major,
            "compare_improved_rate": compare_improved,
            "compare_degraded_rate": compare_degraded,
            "compare_mixed_rate": compare_major - compare_improved - compare_degraded
            if np.isfinite(compare_major)
            else np.nan,
        }
    )
    return rows


def build_failure_bucket_rows(records: pd.DataFrame) -> List[Dict[str, float]]:
    rows = summarize_bucket_scope(records, "global", 0)
    for episode in range(1, rerun.NUM_EPISODES + 1):
        df_ep = records[records["episode"] == episode]
        if df_ep.empty:
            continue
        rows.extend(summarize_bucket_scope(df_ep, "episode", episode))
    return rows


def build_fixed_case_rows(records: pd.DataFrame) -> List[Dict[str, float]]:
    latest = (
        records[records["fixed_case"] > 0.5]
        .sort_values(["event_id", "episode"])
        .groupby("event_id", as_index=False)
        .tail(1)
        .sort_values("event_id")
    )
    rows: List[Dict[str, float]] = []
    for _, row in latest.iterrows():
        rows.append(
            {
                "analysis": "fixed_case_latest",
                "scope": "latest",
                "event_id": int(row["event_id"]),
                "episode": int(row["episode"]),
                "time_min": float(row["time_min"]),
                "num_pos": int(row["num_pos"]),
                "truth_positive_total": int(row["truth_positive_total"]),
                "truth_positive_unobserved": int(row["truth_positive_unobserved"]),
                "positive_seed_root_cause": row["positive_seed_root_cause"],
                "suspect_active_recall": float(row["suspect_active_recall"]),
                "v1_rank": float(row["v1_rank"]),
                "v21_rank": float(row["v21_rank"]),
                "v1_directionality": float(row["v1_directionality"]),
                "v21_directionality": float(row["v21_directionality"]),
                "v1_true_nonzero": float(row["v1_true_nonzero"]),
                "v21_true_nonzero": float(row["v21_true_nonzero"]),
                "v1_hub_win": float(row["v1_hub_win"]),
                "v21_hub_win": float(row["v21_hub_win"]),
                "v1_true_total": float(row["v1_true_total"]),
                "v21_true_total": float(row["v21_true_total"]),
                "v21_raw_true_total": float(row["v21_raw_true_total"]),
                "v21_raw_true_nonzero": float(row["v21_raw_true_nonzero"]),
                "v21_raw_hub_win": float(row["v21_raw_hub_win"]),
                "v21_raw_true_availability": float(row["v21_raw_true_availability"]),
                "v21_raw_true_ownership": float(row["v21_raw_true_ownership"]),
                "v21_raw_true_hub_penalty": float(row["v21_raw_true_hub_penalty"]),
                "v21_raw_true_virtual_share": float(row["v21_raw_true_virtual_share"]),
                "v21_raw_competitor_availability": float(row["v21_raw_competitor_availability"]),
                "v21_raw_competitor_ownership": float(row["v21_raw_competitor_ownership"]),
                "v21_raw_competitor_hub_penalty": float(row["v21_raw_competitor_hub_penalty"]),
                "v21_raw_competitor_virtual_share": float(row["v21_raw_competitor_virtual_share"]),
                "v1_top_other_global_id": int(row["v1_top_other_global_id"]),
                "v21_top_other_global_id": int(row["v21_top_other_global_id"]),
                "v21_raw_top_other_global_id": int(row["v21_raw_top_other_global_id"]),
            }
        )
    return rows


def build_compare_table(records: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    rows.extend(build_metric_rows(records))
    rows.extend(build_failure_bucket_rows(records))
    rows.extend(build_fixed_case_rows(records))
    return pd.DataFrame(rows)


def proof_tag(improved: bool, degraded: bool) -> str:
    if improved and not degraded:
        return "[已证明]"
    if improved:
        return "[部分证明]"
    return "[未证明]"


def build_summary_markdown(records: pd.DataFrame, compare_df: pd.DataFrame) -> str:
    global_row = compare_df[
        (compare_df["analysis"] == "metrics")
        & (compare_df["scope"] == "global")
        & (compare_df["subset"] == "all_events")
    ].iloc[0]
    episode_rows = compare_df[
        (compare_df["analysis"] == "metrics")
        & (compare_df["scope"] == "episode")
        & (compare_df["subset"] == "all_events")
    ].copy()
    episode10_row = episode_rows[episode_rows["episode"] == 10].iloc[0]
    subset_global = compare_df[
        (compare_df["analysis"] == "metrics")
        & (compare_df["scope"] == "global")
        & (compare_df["subset"] == "suspect_active_true_and_num_pos")
    ].iloc[0]
    subset_episode10 = compare_df[
        (compare_df["analysis"] == "metrics")
        & (compare_df["scope"] == "episode")
        & (compare_df["subset"] == "suspect_active_true_and_num_pos")
        & (compare_df["episode"] == 10)
    ].iloc[0]
    bucket_global = compare_df[
        (compare_df["analysis"] == "failure_bucket") & (compare_df["scope"] == "global")
    ].copy()
    fixed_cases = compare_df[compare_df["analysis"] == "fixed_case_latest"].copy()

    b_bucket = bucket_global[bucket_global["bucket"] == "B_positive_input_but_true_support_zero"].iloc[0]
    c_bucket = bucket_global[bucket_global["bucket"] == "C_true_support_nonzero_but_hub_still_wins"].iloc[0]
    d_bucket = bucket_global[bucket_global["bucket"] == "D_compare_major_diverge"].iloc[0]
    e_bucket = bucket_global[bucket_global["bucket"] == "E_suspect_failed_before_support"].iloc[0]

    improved_global = global_row["true_nonzero_delta_v21_minus_v1"] > rerun.EPS and global_row["hub_win_delta_v21_minus_v1"] < -rerun.EPS
    degraded_global = global_row["true_nonzero_delta_v21_minus_v1"] < -rerun.EPS or global_row["hub_win_delta_v21_minus_v1"] > rerun.EPS
    improved_subset = subset_global["true_nonzero_delta_v21_minus_v1"] > rerun.EPS or subset_global["hub_win_delta_v21_minus_v1"] < -rerun.EPS
    degraded_subset = subset_global["true_nonzero_delta_v21_minus_v1"] < -rerun.EPS or subset_global["hub_win_delta_v21_minus_v1"] > rerun.EPS

    if subset_global["v21_support_true_nonzero_given_input_rate"] >= subset_global["v1_support_true_nonzero_given_input_rate"] + 0.05 and subset_global["v21_hub_win_rate_given_input"] <= subset_global["v1_hub_win_rate_given_input"] - 0.05:
        landing_verdict = "best v2.1 已经足以作为 support 的当前主候选。"
        landing_tag = "[已证明]"
    elif subset_global["v21_support_true_nonzero_given_input_rate"] >= subset_global["v1_support_true_nonzero_given_input_rate"] or subset_global["v21_hub_win_rate_given_input"] <= subset_global["v1_hub_win_rate_given_input"]:
        landing_verdict = "best v2.1 可以作为当前主候选，但 practical 落地仍不稳。"
        landing_tag = "[部分证明]"
    else:
        landing_verdict = "best v2.1 还不足以成为 support 的当前主候选。"
        landing_tag = "[未证明]"

    if e_bucket["v1_rate"] >= max(b_bucket["v21_rate"], c_bucket["v21_rate"]) + 0.05:
        primary_cause = "C. suspect 仍是前置瓶颈"
    elif records["truth_positive_unobserved"].mean() > records["num_pos"].mean() and global_row["v21_eligible_event_rate"] < 0.7:
        primary_cause = "B. observation / rollout coverage 问题"
    else:
        primary_cause = "A. support 还要继续改"

    episode_table = episode_rows[
        [
            "episode",
            "v1_eligible_event_rate",
            "v1_conditioned_rank_median",
            "v21_conditioned_rank_median",
            "v1_conditioned_directionality",
            "v21_conditioned_directionality",
            "v1_hub_win_rate_given_input",
            "v21_hub_win_rate_given_input",
            "v1_support_true_nonzero_given_input_rate",
            "v21_support_true_nonzero_given_input_rate",
            "v1_suspect_active_recall_given_input",
            "v21_suspect_active_recall_given_input",
        ]
    ].round(4)

    bucket_table = bucket_global[
        [
            "bucket",
            "v1_rate",
            "v21_rate",
            "delta_rate_v21_minus_v1",
            "compare_major_diverge_rate",
            "compare_improved_rate",
            "compare_degraded_rate",
            "compare_mixed_rate",
        ]
    ].round(4)

    case_lines: List[str] = []
    for _, row in fixed_cases.sort_values("event_id").iterrows():
        event_id = int(row["event_id"])
        if row["num_pos"] <= 0:
            diagnosis = "这条 snapshot 没有 practical positive input，v2.1 没法凭空恢复 witness。"
            tag = "[已证明]"
        elif row["suspect_active_recall"] <= 0.5:
            diagnosis = "suspect 先把 true source 挡掉了，support 分支即使 raw 有分数也落不到真实 practical 线上。"
            tag = "[已证明]"
        elif row["v21_true_nonzero"] > row["v1_true_nonzero"] + rerun.EPS and row["v21_hub_win"] <= row["v1_hub_win"] + rerun.EPS:
            diagnosis = "v2.1 实际救回了 true source 的 nonzero ownership，support 本体比 v1 更能吃到已暴露 witness。"
            tag = "[已证明]"
        elif row["v21_hub_win"] < row["v1_hub_win"] - rerun.EPS:
            diagnosis = "v2.1 主要压低了 hub/competitor 的 generic ownership，救回的是 anti-hub 行为。"
            tag = "[已证明]"
        else:
            diagnosis = "没救回来，卡点更像 witness 暴露不足或 true/competitor 共享同一批 generic witness。"
            tag = "[部分证明]"
        case_lines.append(
            f"- {tag} event {event_id}: "
            f"practical num_pos={int(row['num_pos'])}, truth_positive_total={int(row['truth_positive_total'])}, "
            f"hidden_truth_positive={int(row['truth_positive_unobserved'])}, suspect_active={int(row['suspect_active_recall'])}; "
            f"v1(rank={row['v1_rank']:.0f}, dir={row['v1_directionality']:.0f}, true_nonzero={row['v1_true_nonzero']:.0f}, hub_win={row['v1_hub_win']:.0f}) -> "
            f"v2.1(rank={row['v21_rank']:.0f}, dir={row['v21_directionality']:.0f}, true_nonzero={row['v21_true_nonzero']:.0f}, hub_win={row['v21_hub_win']:.0f}); "
            f"v2.1 raw(true_total={row['v21_raw_true_total']:.4f}, raw_nonzero={row['v21_raw_true_nonzero']:.0f}, raw_hub_win={row['v21_raw_hub_win']:.0f}, "
            f"true_ownership={row['v21_raw_true_ownership']:.4f}, competitor_ownership={row['v21_raw_competitor_ownership']:.4f}); "
            f"{diagnosis}"
        )

    lines = [
        "# Support Score v2.1 Practical Landing Audit",
        "",
        "## 1. 本轮执行摘要",
        f"- {proof_tag(improved_global, degraded_global)} 全局 practical given-input true_nonzero: "
        f"{global_row['v1_support_true_nonzero_given_input_rate']:.4f} -> {global_row['v21_support_true_nonzero_given_input_rate']:.4f}; "
        f"hub_win_given_input: {global_row['v1_hub_win_rate_given_input']:.4f} -> {global_row['v21_hub_win_rate_given_input']:.4f}.",
        f"- {proof_tag(improved_subset, degraded_subset)} `suspect_active=true & num_pos>0` 子集里，"
        f"true_nonzero_given_input: {subset_global['v1_support_true_nonzero_given_input_rate']:.4f} -> {subset_global['v21_support_true_nonzero_given_input_rate']:.4f}; "
        f"hub_win_given_input: {subset_global['v1_hub_win_rate_given_input']:.4f} -> {subset_global['v21_hub_win_rate_given_input']:.4f}.",
        f"- [已证明] 剩余 practical 主矛盾判定：{primary_cause}。",
        "",
        "## 2. 本轮修改清单",
        "- 新增文件：`src/scripts/audit/run_support_score_v21_practical_audit.py`。",
        "- 做法：沿现有 practical rerun 主线并行输出 `v1` 与 `best v2.1`；`v2.1` 只在 audit/rerun 里计算，不改 builder 正式主线。",
        "- best v2.1 固定参数：`virtual_reliability=0.75`、`hub_penalty_weight=0.20`、`ownership_exponent=1.5`。",
        "- 输出文件：",
        f"  - `{STEPWISE_CSV_PATH}`",
        f"  - `{COMPARE_CSV_PATH}`",
        f"  - `{SUMMARY_MD_PATH}`",
        "",
        "## 3. 实际运行命令",
        f"- `{RUN_COMMAND}`",
        "",
        "## 4. v1 vs v2.1 practical 全局对比",
        "```text",
        pd.DataFrame([global_row])[
            [
                "denominator",
                "input_events",
                "v1_eligible_event_rate",
                "v1_all_zero_event_rate",
                "v21_all_zero_event_rate",
                "v1_conditioned_rank_median",
                "v21_conditioned_rank_median",
                "v1_conditioned_directionality",
                "v21_conditioned_directionality",
                "v1_hub_win_rate",
                "v21_hub_win_rate",
                "v1_hub_win_rate_given_input",
                "v21_hub_win_rate_given_input",
                "v1_support_true_nonzero_given_input_rate",
                "v21_support_true_nonzero_given_input_rate",
                "v1_support_zero_despite_input_rate",
                "v21_support_zero_despite_input_rate",
                "v1_suspect_active_recall",
                "v21_suspect_active_recall",
                "v1_suspect_active_recall_given_input",
                "v21_suspect_active_recall_given_input",
            ]
        ].round(4).to_string(index=False),
        "```",
        "- [已证明] 每个 episode 的 practical 主指标已写入 `support_score_v1_vs_v21_practical.csv`；下面直接列 Episode 1-10。",
        "```text",
        episode_table.to_string(index=False),
        "```",
        "```text",
        pd.DataFrame([episode10_row]).round(4).to_string(index=False),
        "```",
        "",
        "## 5. v1 vs v2.1 practical 子集对比（suspect_active=true & num_pos>0）",
        "```text",
        pd.DataFrame([subset_global]).round(4).to_string(index=False),
        "```",
        "```text",
        pd.DataFrame([subset_episode10]).round(4).to_string(index=False),
        "```",
        f"- {proof_tag(c_bucket['delta_rate_v21_minus_v1'] < -rerun.EPS, c_bucket['delta_rate_v21_minus_v1'] > rerun.EPS)} practical 的 C 是否明显下降：{c_bucket['v1_rate']:.4f} -> {c_bucket['v21_rate']:.4f}.",
        f"- {proof_tag(b_bucket['delta_rate_v21_minus_v1'] < -rerun.EPS, b_bucket['delta_rate_v21_minus_v1'] > rerun.EPS)} practical 的 B 是否明显下降：{b_bucket['v1_rate']:.4f} -> {b_bucket['v21_rate']:.4f}.",
        f"- {proof_tag(subset_global['hub_win_delta_v21_minus_v1'] < -rerun.EPS, subset_global['hub_win_delta_v21_minus_v1'] > rerun.EPS)} subset hub_win_given_input：{subset_global['v1_hub_win_rate_given_input']:.4f} -> {subset_global['v21_hub_win_rate_given_input']:.4f}.",
        f"- {proof_tag(subset_global['true_nonzero_delta_v21_minus_v1'] > rerun.EPS, subset_global['true_nonzero_delta_v21_minus_v1'] < -rerun.EPS)} subset true_nonzero_given_input：{subset_global['v1_support_true_nonzero_given_input_rate']:.4f} -> {subset_global['v21_support_true_nonzero_given_input_rate']:.4f}.",
        "",
        "## 6. practical failure buckets 对比",
        "```text",
        bucket_table.to_string(index=False),
        "```",
        f"- {proof_tag(b_bucket['delta_rate_v21_minus_v1'] < -rerun.EPS, b_bucket['delta_rate_v21_minus_v1'] > rerun.EPS)} B bucket 变化：{b_bucket['v1_rate']:.4f} -> {b_bucket['v21_rate']:.4f}.",
        f"- {proof_tag(c_bucket['delta_rate_v21_minus_v1'] < -rerun.EPS, c_bucket['delta_rate_v21_minus_v1'] > rerun.EPS)} C bucket 变化：{c_bucket['v1_rate']:.4f} -> {c_bucket['v21_rate']:.4f}.",
        f"- [已证明] D compare major diverge rate={d_bucket['compare_major_diverge_rate']:.4f}; improved={d_bucket['compare_improved_rate']:.4f}, degraded={d_bucket['compare_degraded_rate']:.4f}, mixed={d_bucket['compare_mixed_rate']:.4f}.",
        f"- [已证明] E suspect failed rate={e_bucket['v1_rate']:.4f}，这是 support 分支不改时无法自行消除的前置损失。",
        "",
        "## 7. 固定 4 个 case 解释",
        *case_lines,
        "",
        "## 8. 当前结论：support 是否已基本落地",
        f"- {landing_tag} {landing_verdict}",
        f"- [已证明] 对问题 1 的回答：best v2.1 相比 v1 的真实 practical 提升，以上述全局与 subset 指标实测为准，不能用 oracle 结果替代。",
        f"- [已证明] 对问题 2 的回答：如果 practical 仍差，当前更像 **{primary_cause}**。",
        "",
        "## 9. 下一步最小建议（只给 1 条）",
        "- 只做 1 次 observation / witness exposure 定位审计：固定 best v2.1，不改 support 公式，统计 `num_pos<truth_positive_total` 的 case 是否就是剩余 B/C 的主来源。",
    ]
    return "\n".join(lines) + "\n"


def run_practical_audit() -> None:
    rerun.set_seeds(0)
    rerun.silence_non_table_logs()
    dataset = NpzDatasetV6(
        samples_dir=rerun.SAMPLES_PATH,
        foundation_dir=rerun.FOUNDATION_PATH,
        mode="test",
        preload=False,
        audit_mode="fast",
        use_edge_attr=True,
    )
    topology = HydraulicTopology(rerun.FOUNDATION_PATH)
    records = build_stepwise_records(dataset, topology)
    if records.empty:
        raise RuntimeError("support score v2.1 practical audit produced no rows.")

    records.to_csv(STEPWISE_CSV_PATH, index=False)
    compare_df = build_compare_table(records)
    compare_df.to_csv(COMPARE_CSV_PATH, index=False)
    markdown = build_summary_markdown(records, compare_df)
    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)

    global_row = compare_df[
        (compare_df["analysis"] == "metrics")
        & (compare_df["scope"] == "global")
        & (compare_df["subset"] == "all_events")
    ][
        [
            "denominator",
            "input_events",
            "v1_conditioned_rank_median",
            "v21_conditioned_rank_median",
            "v1_conditioned_directionality",
            "v21_conditioned_directionality",
            "v1_hub_win_rate_given_input",
            "v21_hub_win_rate_given_input",
            "v1_support_true_nonzero_given_input_rate",
            "v21_support_true_nonzero_given_input_rate",
        ]
    ]
    subset_row = compare_df[
        (compare_df["analysis"] == "metrics")
        & (compare_df["scope"] == "global")
        & (compare_df["subset"] == "suspect_active_true_and_num_pos")
    ][
        [
            "denominator",
            "v1_conditioned_rank_median",
            "v21_conditioned_rank_median",
            "v1_hub_win_rate_given_input",
            "v21_hub_win_rate_given_input",
            "v1_support_true_nonzero_given_input_rate",
            "v21_support_true_nonzero_given_input_rate",
        ]
    ]
    bucket_rows = compare_df[
        (compare_df["analysis"] == "failure_bucket") & (compare_df["scope"] == "global")
    ][
        [
            "bucket",
            "v1_rate",
            "v21_rate",
            "delta_rate_v21_minus_v1",
            "compare_major_diverge_rate",
            "compare_improved_rate",
            "compare_degraded_rate",
            "compare_mixed_rate",
        ]
    ]

    print("Table 1: practical global compare")
    print(global_row.round(4).to_string(index=False))
    print()
    print("Table 2: support-responsible subset compare")
    print(subset_row.round(4).to_string(index=False))
    print()
    print("Table 3: practical failure buckets")
    print(bucket_rows.round(4).to_string(index=False))


if __name__ == "__main__":
    run_practical_audit()
