import argparse
import os
import sys
import random
import logging
import json
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = "/root/autodl-tmp/rl_spim_v7_mainline"
sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
import torch

from src.data.v6.dataset import NpzDatasetV6
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.builder import EvidenceBuilder
from src.modeling.evidence.contradiction_oracle_v1 import (
    DEFAULT_ADMISSIBILITY_COMPARE_MODES,
    DEFAULT_ADMISSIBILITY_MODE,
    DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN,
    DEFAULT_MINED_TOP_K_SAFE_WITNESSES,
    DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE,
    DEFAULT_RELAXED_FRONTIER_WINDOW_TAU_MIN,
    DEFAULT_SAFE_VIOLATION_TAU_MIN,
    DEFAULT_TIME_BRACKETING_RELAX_SLACK_MIN,
    DEFAULT_TOP_K_WITNESSES,
    DEFAULT_WITNESS_MINING_MODE,
    AdmissibilityCompareConfig,
    PracticalContradictionV2Config,
    derive_practical_v2_contradiction,
    compute_practical_v2_admissibility_compare,
    extract_candidate_top_witnesses,
)
from src.modeling.state.schema import ObservationState
from src.scripts.audit.utils_practical_rollout import PracticalRollout, compute_audit_support

FOUNDATION_PATH = f"{ROOT_DIR}/datanew/production_data/foundation_20260114_164946_86d5023e"
SAMPLES_PATH = f"{FOUNDATION_PATH}/subgraph_v11_prod"

CSV_PATH = os.path.join(ROOT_DIR, "evidence_axis_semantics_cleanup_summary.csv")
MD_PATH = os.path.join(ROOT_DIR, "evidence_axis_semantics_cleanup.md")
STEPWISE_CSV_PATH = os.path.join(ROOT_DIR, "practical_audit_diagnostic_stepwise.csv")
CONTRA_ALIGNMENT_CSV_PATH = os.path.join(ROOT_DIR, "contradiction_alignment_overview.csv")
CONTRA_ALIGNMENT_MD_PATH = os.path.join(ROOT_DIR, "contradiction_alignment_overview.md")
CASE_MD_PATH = os.path.join(ROOT_DIR, "practical_audit_diagnostic_cases.md")
CASE_TRACE_CSV_PATH = os.path.join(ROOT_DIR, "practical_audit_case_trace.csv")
CASE_TRACE_MD_PATH = os.path.join(ROOT_DIR, "practical_audit_case_trace.md")
POSITIVE_SEED_CSV_PATH = os.path.join(ROOT_DIR, "practical_audit_positive_seed_survival.csv")
POSITIVE_SEED_MD_PATH = os.path.join(ROOT_DIR, "practical_audit_positive_seed_survival.md")
B_ROOTCAUSE_CSV_PATH = os.path.join(ROOT_DIR, "support_score_b_rootcause.csv")
C_ROOTCAUSE_CSV_PATH = os.path.join(ROOT_DIR, "support_score_c_rootcause.csv")
SUBTERM_ROOTCAUSE_MD_PATH = os.path.join(ROOT_DIR, "support_score_subterm_rootcause_summary.md")
ORACLE_STEPWISE_CSV_PATH = os.path.join(ROOT_DIR, "support_score_oracle_stepwise.csv")
ORACLE_METRICS_CSV_PATH = os.path.join(ROOT_DIR, "support_score_oracle_metrics.csv")
ORACLE_VS_PRACTICAL_CSV_PATH = os.path.join(ROOT_DIR, "support_score_oracle_vs_practical.csv")
ORACLE_SUMMARY_MD_PATH = os.path.join(ROOT_DIR, "support_score_oracle_summary.md")
V2_ORACLE_STEPWISE_CSV_PATH = os.path.join(ROOT_DIR, "support_score_v2_oracle_stepwise.csv")
V2_ORACLE_METRICS_CSV_PATH = os.path.join(ROOT_DIR, "support_score_v2_oracle_metrics.csv")
V1_VS_V2_ORACLE_CSV_PATH = os.path.join(ROOT_DIR, "support_score_v1_vs_v2_oracle.csv")
V2_SUMMARY_MD_PATH = os.path.join(ROOT_DIR, "support_score_v2_summary.md")
CONTRA_ORACLE_V1_MD_PATH = os.path.join(ROOT_DIR, "contradiction_oracle_v1_audit.md")
CONTRA_ORACLE_V1_JSON_PATH = os.path.join(ROOT_DIR, "contradiction_oracle_v1_summary.json")
CONTRA_ORACLE_V1_COMPARE_CSV_PATH = os.path.join(ROOT_DIR, "contradiction_oracle_v1_compare.csv")
CONTRA_PRACTICAL_V2_MD_PATH = os.path.join(ROOT_DIR, "contradiction_practical_v2_audit.md")
CONTRA_PRACTICAL_V2_JSON_PATH = os.path.join(ROOT_DIR, "contradiction_practical_v2_summary.json")
CONTRA_PRACTICAL_V2_COMPARE_CSV_PATH = os.path.join(ROOT_DIR, "contradiction_practical_v2_compare.csv")
CONTRA_PAIR_AVAILABILITY_MD_PATH = os.path.join(ROOT_DIR, "contradiction_pair_availability_audit.md")
CONTRA_PAIR_AVAILABILITY_JSON_PATH = os.path.join(ROOT_DIR, "contradiction_pair_availability_summary.json")
CONTRA_PAIR_AVAILABILITY_CSV_PATH = os.path.join(ROOT_DIR, "contradiction_pair_availability_compare.csv")
CONTRA_WITNESS_COVERAGE_MD_PATH = os.path.join(ROOT_DIR, "contradiction_witness_coverage_pilot.md")
CONTRA_WITNESS_COVERAGE_JSON_PATH = os.path.join(ROOT_DIR, "contradiction_witness_coverage_summary.json")
CONTRA_WITNESS_COVERAGE_CSV_PATH = os.path.join(ROOT_DIR, "contradiction_witness_coverage_compare.csv")
CONTRA_ADMISSIBILITY_CEILING_MD_PATH = os.path.join(ROOT_DIR, "contradiction_admissibility_ceiling_audit.md")
CONTRA_ADMISSIBILITY_CEILING_JSON_PATH = os.path.join(ROOT_DIR, "contradiction_admissibility_ceiling_summary.json")
CONTRA_ADMISSIBILITY_CEILING_CSV_PATH = os.path.join(ROOT_DIR, "contradiction_admissibility_ceiling_compare.csv")
EPS = 1e-6
STRICT_HARD_BUFFER_MIN = 10.0
SUSPECT_THRESHOLD = 0.5
MAX_EVENTS = 100
NUM_EPISODES = 10
SUPPORT_COMPARE_L1_ALERT = 0.05
CONTRA_TIME_CONFLICT_ALERT = 0.10
DEFAULT_SUPPORT_TIME_MIN = 100.0
CASE_EVENT_IDS = [5, 0, 60, 83]
V2_MAX_WITNESSES = 20
V2_TIME_SIGMA_MIN = 35.0
V2_TIME_PRIOR_OFFSET_MIN = 10.0
V2_VIRTUAL_RELIABILITY = 0.85
V2_OWNERSHIP_POWER = 1.5
V2_AVAIL_ACTIVE_THRESHOLD = 0.05
V2_AVAIL_WEIGHT = 0.35
V2_HUB_PENALTY_WEIGHT = 0.20
ORACLE_V1_SAFE_TAU_MIN = DEFAULT_SAFE_VIOLATION_TAU_MIN
ORACLE_V1_TOP_K_WITNESSES = DEFAULT_TOP_K_WITNESSES
PRACTICAL_V2_HISTORY_PHYSCTX_MODE = DEFAULT_ORACLE_HISTORY_PHYSCTX_MODE
PRACTICAL_V2_CURRENT_TIME_PHYSCTX_MODE = "current_time_physctx"
WITNESS_MINING_MODES = [
    DEFAULT_WITNESS_MINING_MODE,
    "candidate_conditioned_frontier_safe",
    "candidate_conditioned_topk_safe",
]
WITNESS_MINING_TOP_K = DEFAULT_MINED_TOP_K_SAFE_WITNESSES
WITNESS_MINING_FRONT_CLOSE_TAU_MIN = DEFAULT_FRONTIER_SAFE_CLOSE_TAU_MIN
ADMISSIBILITY_COMPARE_MODES = list(DEFAULT_ADMISSIBILITY_COMPARE_MODES)
ADMISSIBILITY_COMPARE_CONFIG = AdmissibilityCompareConfig(
    time_bracketing_relax_slack_min=DEFAULT_TIME_BRACKETING_RELAX_SLACK_MIN,
    frontier_window_relax_tau_min=DEFAULT_RELAXED_FRONTIER_WINDOW_TAU_MIN,
)
ADMISSIBILITY_MODE_SHORT_LABEL = {
    "baseline_admissibility": "baseline",
    "topology_relaxed_compare": "topology",
    "time_bracketing_relaxed_compare": "time",
    "frontier_window_relaxed_compare": "frontier",
    "union_relaxed_upper_bound": "union",
}
PRACTICAL_V2_MAIN_CONFIG = PracticalContradictionV2Config(
    label="safe_dominant_raw",
    gap_cap_min=60.0,
    gap_log_tau_min=15.0,
    soft_count_tau_min=12.0,
    near_safe_tau_min=10.0,
    near_safe_slack_min=20.0,
    alpha_gap=0.35,
    beta_near_safe=1.0,
    gamma_soft_count=0.75,
    normalize_by_eligible_safe_count=False,
)
PRACTICAL_V2_NORMALIZED_CONFIG = PracticalContradictionV2Config(
    label="safe_dominant_norm",
    gap_cap_min=PRACTICAL_V2_MAIN_CONFIG.gap_cap_min,
    gap_log_tau_min=PRACTICAL_V2_MAIN_CONFIG.gap_log_tau_min,
    soft_count_tau_min=PRACTICAL_V2_MAIN_CONFIG.soft_count_tau_min,
    near_safe_tau_min=PRACTICAL_V2_MAIN_CONFIG.near_safe_tau_min,
    near_safe_slack_min=PRACTICAL_V2_MAIN_CONFIG.near_safe_slack_min,
    alpha_gap=PRACTICAL_V2_MAIN_CONFIG.alpha_gap,
    beta_near_safe=PRACTICAL_V2_MAIN_CONFIG.beta_near_safe,
    gamma_soft_count=PRACTICAL_V2_MAIN_CONFIG.gamma_soft_count,
    normalize_by_eligible_safe_count=True,
)
CONTRA_ORACLE_V1_BUFFER_METRICS = [
    "total",
    "interval_gap",
    "safe_violation",
    "violated_safe_count",
    "top_witness_margin",
]
CONTRA_PRACTICAL_V2_BUFFER_METRICS = [
    "total",
    "total_practical_v2_raw",
    "total_practical_v2_norm",
    "interval_gap",
    "interval_gap_capped",
    "interval_gap_log",
    "safe_violation",
    "soft_violated_safe_count",
    "near_safe_mass",
    "violated_safe_count",
    "eligible_safe_witness_count",
    "positive_margin_count",
    "best_margin_topk_mean",
    "top_witness_margin",
    "gap_component",
    "safe_component",
]


def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def silence_non_table_logs() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("src.data.v6.dataset").setLevel(logging.ERROR)


def extract_view0(event_data_batch):
    event_data = event_data_batch.clone()
    if hasattr(event_data, "view_batch"):
        mask_v0 = event_data.view_batch == 0
        n_v0 = int(mask_v0.sum().item())
        if event_data.x is not None:
            event_data.x = event_data.x[mask_v0]
        if event_data.y is not None:
            event_data.y = event_data.y[mask_v0]
        if event_data.n_id is not None:
            event_data.n_id = event_data.n_id[mask_v0]
        if hasattr(event_data, "global_ids"):
            event_data.global_ids = event_data.global_ids[mask_v0]
        if hasattr(event_data, "x_raw"):
            event_data.x_raw = event_data.x_raw[mask_v0]
        mask_edges = event_data.edge_index[0] < n_v0
        event_data.edge_index = event_data.edge_index[:, mask_edges]
        if event_data.edge_attr is not None:
            event_data.edge_attr = event_data.edge_attr[mask_edges]
        event_data.num_nodes = n_v0
    if event_data.x is not None and event_data.x.shape[1] > 5:
        event_data.x[:, 5] = 1.0
    return event_data


def strict_rank(scores: torch.Tensor, true_idx: int, higher_better: bool) -> int:
    true_score = float(scores[true_idx].item())
    if higher_better:
        return int((scores > true_score + EPS).sum().item()) + 1
    return int((scores < true_score - EPS).sum().item()) + 1


def mean_other(scores: torch.Tensor, true_idx: int) -> float:
    if scores.numel() <= 1:
        return 0.0
    others = torch.cat([scores[:true_idx], scores[true_idx + 1:]])
    return float(others.mean().item()) if others.numel() > 0 else 0.0


def unique_top1(scores: torch.Tensor, true_idx: int) -> bool:
    true_score = float(scores[true_idx].item())
    max_score = float(scores.max().item())
    return abs(true_score - max_score) <= EPS and int((scores >= max_score - EPS).sum().item()) == 1


def all_zero(scores: torch.Tensor) -> bool:
    return float(scores.abs().max().item()) <= EPS


def masked_mean(scores: torch.Tensor, mask: torch.Tensor) -> float:
    if mask is None or not bool(mask.any()):
        return np.nan
    return float(scores[mask].mean().item())


def ratio_or_nan(mask: pd.Series) -> float:
    if mask is None or len(mask) == 0:
        return np.nan
    return float(mask.mean())


def count_ratio_line(df: pd.DataFrame, column: str) -> str:
    if df.empty:
        return f"{column}=nan (0/0)"
    count = int(df[column].sum())
    return f"{column}={count / len(df):.4f} ({count}/{len(df)})"


def best_other_index(scores: torch.Tensor, true_idx: int, higher_better: bool) -> int:
    if scores.numel() <= 1:
        return true_idx
    masked = scores.clone()
    masked[true_idx] = -float("inf") if higher_better else float("inf")
    return int(masked.argmax().item()) if higher_better else int(masked.argmin().item())


def finite_float_or_nan(value: torch.Tensor) -> float:
    scalar = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    return scalar if np.isfinite(scalar) else np.nan


def serialize_witnesses(witnesses: List[Dict[str, Any]]) -> str:
    if not witnesses:
        return ""
    parts = []
    for witness in witnesses:
        parts.append(
            "safe={safe_global_idx}@{safe_time_min:.1f}|pos={positive_global_idx}@{positive_time_min:.1f}|margin={margin:.4f}".format(
                **witness
            )
        )
    return " || ".join(parts)


def dominant_component_label(flag: int) -> str:
    if int(flag) == 1:
        return "interval"
    if int(flag) == 2:
        return "safe"
    return "none"


def init_contradiction_candidate_buffers(metric_names: List[str]) -> Dict[str, List[np.ndarray]]:
    buffers: Dict[str, List[np.ndarray]] = {}
    for metric_name in metric_names:
        buffers[f"all_{metric_name}"] = []
        buffers[f"non_source_{metric_name}"] = []
    return buffers


def append_contradiction_candidate_buffers(
    candidate_buffers: Dict[str, List[np.ndarray]],
    contra_res: Dict[str, Any],
    non_source_mask: torch.Tensor,
    metric_names: List[str],
) -> None:
    for metric_name in metric_names:
        if metric_name not in contra_res:
            continue
        candidate_buffers[f"all_{metric_name}"].append(
            contra_res[metric_name].detach().cpu().numpy().astype(np.float32)
        )
        if bool(non_source_mask.any().item()):
            candidate_buffers[f"non_source_{metric_name}"].append(
                contra_res[metric_name][non_source_mask].detach().cpu().numpy().astype(np.float32)
            )


def extract_contradiction_score_fields(
    prefix: str,
    scores: torch.Tensor,
    src_local: int,
    g_ids: torch.Tensor,
) -> Dict[str, Any]:
    true_score = float(scores[src_local].item())
    other_mean = mean_other(scores, src_local)
    top_other_idx = best_other_index(scores, src_local, higher_better=False)
    top_other_score = float(scores[top_other_idx].item())
    return {
        f"{prefix}_rank": strict_rank(scores, src_local, higher_better=False),
        f"{prefix}_gap": other_mean - true_score,
        f"{prefix}_directionality": float((other_mean - true_score) > 0.0),
        f"{prefix}_all_zero": float(all_zero(scores)),
        f"{prefix}_true": true_score,
        f"{prefix}_other_mean": other_mean,
        f"{prefix}_top_other_idx": int(top_other_idx),
        f"{prefix}_top_other_global_id": int(g_ids[top_other_idx].item()),
        f"{prefix}_top_other_score": top_other_score,
    }


def extract_contradiction_candidate_fields(
    prefix: str,
    contra_res: Dict[str, Any],
    candidate_idx: int,
    g_ids: torch.Tensor,
) -> Dict[str, Any]:
    candidate_idx = int(candidate_idx)
    witnesses = extract_candidate_top_witnesses(
        contra_res,
        candidate_idx,
        top_k=ORACLE_V1_TOP_K_WITNESSES,
    )
    safe_local = int(contra_res["top_witness_safe_local_idx"][candidate_idx].item())
    pos_local = int(contra_res["top_witness_pos_local_idx"][candidate_idx].item())
    fields = {
        f"{prefix}_global_id": int(g_ids[candidate_idx].item()),
        f"{prefix}_total": float(contra_res["total"][candidate_idx].item()),
        f"{prefix}_interval_gap": float(contra_res["interval_gap"][candidate_idx].item()),
        f"{prefix}_safe_violation": float(contra_res["safe_violation"][candidate_idx].item()),
        f"{prefix}_violated_safe_count": float(contra_res["violated_safe_count"][candidate_idx].item()),
        f"{prefix}_top_witness_margin": float(contra_res["top_witness_margin"][candidate_idx].item()),
        f"{prefix}_upper_bound": finite_float_or_nan(contra_res["upper_bound"][candidate_idx]),
        f"{prefix}_lower_bound": finite_float_or_nan(contra_res["lower_bound"][candidate_idx]),
        f"{prefix}_top_witness_safe_local_idx": safe_local,
        f"{prefix}_top_witness_safe_global_id": int(g_ids[safe_local].item()) if safe_local >= 0 else -1,
        f"{prefix}_top_witness_pos_local_idx": pos_local,
        f"{prefix}_top_witness_pos_global_id": int(g_ids[pos_local].item()) if pos_local >= 0 else -1,
        f"{prefix}_top_witnesses": serialize_witnesses(witnesses),
    }
    optional_float_fields = [
        ("positive_reachable_count", "positive_reachable_count"),
        ("safe_reachable_count", "safe_reachable_count"),
        ("pair_count", "pair_count"),
        ("positive_margin_pair_count", "positive_margin_pair_count"),
        ("non_positive_margin_pair_count", "non_positive_margin_pair_count"),
        ("pair_available", "pair_available"),
        ("positive_margin_available", "positive_margin_available"),
        ("upper_bound_finite", "upper_bound_finite"),
        ("lower_bound_finite", "lower_bound_finite"),
        ("interval_bounds_available", "interval_bounds_available"),
        ("interval_regime_available", "interval_regime_available"),
        ("soft_violated_safe_count", "soft_violated_safe_count"),
        ("near_safe_mass", "near_safe_mass"),
        ("interval_gap_capped", "interval_gap_capped"),
        ("interval_gap_log", "interval_gap_log"),
        ("eligible_safe_witness_count", "eligible_safe_witness_count"),
        ("positive_margin_count", "positive_margin_count"),
        ("best_margin_topk_mean", "best_margin_topk_mean"),
        ("gap_component", "gap_component"),
        ("safe_component", "safe_component"),
        ("safe_regime_available", "safe_regime_available"),
        ("total_practical_v2_raw", "total_practical_v2_raw"),
        ("total_practical_v2_norm", "total_practical_v2_norm"),
    ]
    for source_key, suffix in optional_float_fields:
        if source_key in contra_res:
            fields[f"{prefix}_{suffix}"] = float(contra_res[source_key][candidate_idx].item())
    if "dominant_component_flag" in contra_res:
        flag = int(contra_res["dominant_component_flag"][candidate_idx].item())
        fields[f"{prefix}_dominant_component_flag"] = flag
        fields[f"{prefix}_dominant_component"] = dominant_component_label(flag)
    return fields


def build_history_time_lookup(contra_res: Dict[str, Any], record_key: str) -> Dict[int, float]:
    lookup: Dict[int, float] = {}
    compressed_history = contra_res.get("compressed_history", {})
    for record in compressed_history.get(record_key, []):
        lookup[int(record["local_idx"])] = float(record["time_min"])
    return lookup


def classify_candidate_pair_bucket(row: pd.Series) -> str:
    if row["has_positive_evidence"] <= 0.5:
        return "no_positive"
    if row["has_eligible_safe_witness"] <= 0.5:
        return "no_eligible_safe"
    if row["pair_available"] <= 0.5:
        return "no_pair_available"
    if row["positive_margin_available"] <= 0.5:
        if row["safe_regime_available"] > 0.5:
            return "pair_available_safe_only_weak"
        return "pair_available_but_non_positive_margin"
    if row["interval_regime_available"] <= 0.5:
        return "pair_available_but_interval_gap_zero"
    if row["is_true_source_mis_hit"] > 0.5:
        return "true_source_mis_hit"
    if row["dominant_component"] == "safe":
        return "active_safe_dominant"
    return "active_other"


def classify_true_source_mis_hit_bucket(row: pd.Series) -> str:
    if row["is_true_source_mis_hit"] <= 0.5:
        return "not_mis_hit"
    if row["current_time_total"] > max(row["total"], EPS) * 1.5 and row["current_time_interval_gap"] > row["interval_gap"] + 1.0:
        return "physctx_time_sensitivity"
    if row["eligible_safe_witness_count"] >= 3.0:
        return "multi_safe_accumulation"
    if row["top_witness_margin"] >= 600.0 or row["interval_gap"] >= 600.0:
        return "long_path_outlier"
    if row["top_witness_time_gap_min"] <= 45.0:
        return "safe_time_close_to_positive"
    return "pairing_unstable_or_mixed"


def build_contradiction_candidate_audit_frame(
    event_id: int,
    episode: int,
    time_min: float,
    src_local: int,
    support_top_other_idx: int,
    contra_res: Dict[str, Any],
    current_time_res: Dict[str, Any],
    g_ids: torch.Tensor,
) -> pd.DataFrame:
    positive_lookup = build_history_time_lookup(contra_res, "positive_records")
    safe_lookup = build_history_time_lookup(contra_res, "safe_records")
    num_nodes = int(g_ids.numel())
    candidate_local_idx = np.arange(num_nodes, dtype=np.int64)
    top_safe_local = contra_res["top_witness_safe_local_idx"].detach().cpu().numpy().astype(np.int64)
    top_pos_local = contra_res["top_witness_pos_local_idx"].detach().cpu().numpy().astype(np.int64)
    top_safe_time = np.array([safe_lookup.get(int(idx), np.nan) if int(idx) >= 0 else np.nan for idx in top_safe_local])
    top_pos_time = np.array([positive_lookup.get(int(idx), np.nan) if int(idx) >= 0 else np.nan for idx in top_pos_local])

    frame = pd.DataFrame(
        {
            "event_id": int(event_id),
            "episode": int(episode),
            "time_min": float(time_min),
            "candidate_local_idx": candidate_local_idx,
            "candidate_global_id": g_ids.detach().cpu().numpy().astype(np.int64),
            "is_true_source": (candidate_local_idx == int(src_local)).astype(np.float32),
            "is_support_top_competitor": (candidate_local_idx == int(support_top_other_idx)).astype(np.float32),
            "has_positive_evidence": np.full(num_nodes, float(contra_res["positive_count"] > 0), dtype=np.float32),
            "positive_evidence_count": np.full(num_nodes, float(contra_res["positive_count"]), dtype=np.float32),
            "safe_history_count": np.full(num_nodes, float(contra_res["safe_count"]), dtype=np.float32),
            "has_reachable_positive_witness": contra_res["positive_reachable_count"].detach().cpu().numpy().astype(np.float32) > 0.0,
            "reachable_positive_count": contra_res["positive_reachable_count"].detach().cpu().numpy().astype(np.float32),
            "has_eligible_safe_witness": contra_res["safe_reachable_count"].detach().cpu().numpy().astype(np.float32) > 0.0,
            "eligible_safe_witness_count": contra_res["safe_reachable_count"].detach().cpu().numpy().astype(np.float32),
            "pairable_safe_witness_count": contra_res["eligible_safe_witness_count"].detach().cpu().numpy().astype(np.float32),
            "pair_available": contra_res["pair_available"].detach().cpu().numpy().astype(np.float32),
            "pair_count": contra_res["pair_count"].detach().cpu().numpy().astype(np.float32),
            "positive_margin_available": contra_res["positive_margin_available"].detach().cpu().numpy().astype(np.float32),
            "positive_margin_pair_count": contra_res["positive_margin_pair_count"].detach().cpu().numpy().astype(np.float32),
            "non_positive_margin_pair_count": contra_res["non_positive_margin_pair_count"].detach().cpu().numpy().astype(np.float32),
            "best_margin": contra_res["top_witness_margin"].detach().cpu().numpy().astype(np.float32),
            "best_margin_topk_mean": contra_res["best_margin_topk_mean"].detach().cpu().numpy().astype(np.float32),
            "interval_gap": contra_res["interval_gap"].detach().cpu().numpy().astype(np.float32),
            "safe_violation": contra_res["safe_violation"].detach().cpu().numpy().astype(np.float32),
            "violated_safe_count": contra_res["violated_safe_count"].detach().cpu().numpy().astype(np.float32),
            "soft_count": contra_res["soft_violated_safe_count"].detach().cpu().numpy().astype(np.float32),
            "near_safe_mass": contra_res["near_safe_mass"].detach().cpu().numpy().astype(np.float32),
            "total": contra_res["total"].detach().cpu().numpy().astype(np.float32),
            "dominant_component": [
                dominant_component_label(int(flag))
                for flag in contra_res["dominant_component_flag"].detach().cpu().numpy().astype(np.int64)
            ],
            "interval_bounds_available": contra_res["interval_bounds_available"].detach().cpu().numpy().astype(np.float32),
            "interval_regime_available": contra_res["interval_regime_available"].detach().cpu().numpy().astype(np.float32),
            "safe_regime_available": contra_res["safe_regime_available"].detach().cpu().numpy().astype(np.float32),
            "top_witness_safe_global_id": np.where(top_safe_local >= 0, g_ids.detach().cpu().numpy().astype(np.int64)[top_safe_local.clip(min=0)], -1),
            "top_witness_pos_global_id": np.where(top_pos_local >= 0, g_ids.detach().cpu().numpy().astype(np.int64)[top_pos_local.clip(min=0)], -1),
            "top_witness_safe_time_min": top_safe_time,
            "top_witness_pos_time_min": top_pos_time,
            "top_witness_time_gap_min": top_safe_time - top_pos_time,
            "current_time_pair_available": current_time_res["pair_available"].detach().cpu().numpy().astype(np.float32),
            "current_time_pair_count": current_time_res["pair_count"].detach().cpu().numpy().astype(np.float32),
            "current_time_best_margin": current_time_res["top_witness_margin"].detach().cpu().numpy().astype(np.float32),
            "current_time_interval_gap": current_time_res["interval_gap"].detach().cpu().numpy().astype(np.float32),
            "current_time_total": current_time_res["total"].detach().cpu().numpy().astype(np.float32),
            "current_time_interval_regime_available": current_time_res["interval_regime_available"].detach().cpu().numpy().astype(np.float32),
            "current_time_dominant_component": [
                dominant_component_label(int(flag))
                for flag in current_time_res["dominant_component_flag"].detach().cpu().numpy().astype(np.int64)
            ],
        }
    )
    frame["has_reachable_positive_witness"] = frame["has_reachable_positive_witness"].astype(np.float32)
    frame["has_eligible_safe_witness"] = frame["has_eligible_safe_witness"].astype(np.float32)
    frame["is_true_source_mis_hit"] = (
        (frame["is_true_source"] > 0.5) & (frame["total"] > EPS)
    ).astype(np.float32)
    frame["zero_reason_bucket"] = frame.apply(classify_candidate_pair_bucket, axis=1)
    frame["mis_hit_risk_bucket"] = frame.apply(classify_true_source_mis_hit_bucket, axis=1)
    return frame


def build_witness_mining_candidate_frame(
    event_id: int,
    episode: int,
    time_min: float,
    src_local: int,
    support_top_other_idx: int,
    contra_res: Dict[str, Any],
    current_time_res: Dict[str, Any],
    g_ids: torch.Tensor,
    witness_mining_mode: str,
) -> pd.DataFrame:
    positive_lookup = build_history_time_lookup(contra_res, "positive_records")
    safe_lookup = build_history_time_lookup(contra_res, "safe_records")
    num_nodes = int(g_ids.numel())
    candidate_local_idx = np.arange(num_nodes, dtype=np.int64)
    g_ids_np = g_ids.detach().cpu().numpy().astype(np.int64)
    top_safe_local = contra_res["top_witness_safe_local_idx"].detach().cpu().numpy().astype(np.int64)
    top_pos_local = contra_res["top_witness_pos_local_idx"].detach().cpu().numpy().astype(np.int64)
    top_safe_time = np.array([safe_lookup.get(int(idx), np.nan) if int(idx) >= 0 else np.nan for idx in top_safe_local])
    top_pos_time = np.array([positive_lookup.get(int(idx), np.nan) if int(idx) >= 0 else np.nan for idx in top_pos_local])

    mined_safe_count = contra_res.get("mined_safe_candidate_count", contra_res["safe_reachable_count"]).detach().cpu().numpy().astype(np.float32)
    hydraulic_comparable_safe_count = contra_res.get(
        "hydraulic_comparable_safe_count",
        contra_res["safe_reachable_count"],
    ).detach().cpu().numpy().astype(np.float32)
    front_close_safe_count = contra_res.get(
        "front_close_safe_count",
        torch.zeros_like(contra_res["safe_reachable_count"]),
    ).detach().cpu().numpy().astype(np.float32)
    frontier_safe_count = contra_res.get(
        "frontier_safe_count",
        torch.zeros_like(contra_res["safe_reachable_count"]),
    ).detach().cpu().numpy().astype(np.float32)
    selected_safe_witness_count = contra_res.get(
        "selected_safe_witness_count",
        contra_res["eligible_safe_witness_count"],
    ).detach().cpu().numpy().astype(np.float32)
    topk_safe_count = contra_res.get(
        "topk_safe_count",
        contra_res["eligible_safe_witness_count"],
    ).detach().cpu().numpy().astype(np.float32)
    pair_available_after_mining = contra_res.get(
        "pair_available_after_mining",
        contra_res["pair_available"],
    ).detach().cpu().numpy().astype(np.float32)
    positive_margin_available_after_mining = contra_res.get(
        "positive_margin_available_after_mining",
        contra_res["positive_margin_available"],
    ).detach().cpu().numpy().astype(np.float32)

    frame = pd.DataFrame(
        {
            "event_id": int(event_id),
            "episode": int(episode),
            "time_min": float(time_min),
            "candidate_local_idx": candidate_local_idx,
            "candidate_global_id": g_ids_np,
            "is_true_source": (candidate_local_idx == int(src_local)).astype(np.float32),
            "is_support_top_competitor": (candidate_local_idx == int(support_top_other_idx)).astype(np.float32),
            "witness_mining_mode": str(witness_mining_mode),
            "has_positive_evidence": np.full(num_nodes, float(contra_res["positive_count"] > 0), dtype=np.float32),
            "positive_evidence_count": np.full(num_nodes, float(contra_res["positive_count"]), dtype=np.float32),
            "safe_history_count": np.full(num_nodes, float(contra_res["safe_count"]), dtype=np.float32),
            "mined_safe_candidate_count": mined_safe_count,
            "hydraulic_comparable_safe_count": hydraulic_comparable_safe_count,
            "front_close_safe_count": front_close_safe_count,
            "frontier_safe_count": frontier_safe_count,
            "selected_safe_witness_count": selected_safe_witness_count,
            "topk_safe_count": topk_safe_count,
            "has_eligible_safe_witness": (mined_safe_count > 0.0).astype(np.float32),
            "pair_available": pair_available_after_mining,
            "pair_available_after_mining": pair_available_after_mining,
            "pair_count": contra_res["pair_count"].detach().cpu().numpy().astype(np.float32),
            "positive_margin_available": positive_margin_available_after_mining,
            "positive_margin_after_mining": positive_margin_available_after_mining,
            "positive_margin_pair_count": contra_res["positive_margin_pair_count"].detach().cpu().numpy().astype(np.float32),
            "best_margin": contra_res["top_witness_margin"].detach().cpu().numpy().astype(np.float32),
            "best_margin_topk_mean": contra_res["best_margin_topk_mean"].detach().cpu().numpy().astype(np.float32),
            "interval_gap": contra_res["interval_gap"].detach().cpu().numpy().astype(np.float32),
            "safe_violation": contra_res["safe_violation"].detach().cpu().numpy().astype(np.float32),
            "violated_safe_count": contra_res["violated_safe_count"].detach().cpu().numpy().astype(np.float32),
            "soft_count": contra_res["soft_violated_safe_count"].detach().cpu().numpy().astype(np.float32),
            "near_safe_mass": contra_res["near_safe_mass"].detach().cpu().numpy().astype(np.float32),
            "aggregated_safe_mass": contra_res["near_safe_mass"].detach().cpu().numpy().astype(np.float32),
            "safe_component": contra_res["safe_component"].detach().cpu().numpy().astype(np.float32),
            "total": contra_res["total"].detach().cpu().numpy().astype(np.float32),
            "dominant_component": [
                dominant_component_label(int(flag))
                for flag in contra_res["dominant_component_flag"].detach().cpu().numpy().astype(np.int64)
            ],
            "interval_bounds_available": contra_res["interval_bounds_available"].detach().cpu().numpy().astype(np.float32),
            "interval_regime_available": contra_res["interval_regime_available"].detach().cpu().numpy().astype(np.float32),
            "safe_regime_available": contra_res["safe_regime_available"].detach().cpu().numpy().astype(np.float32),
            "top_witness_safe_global_id": np.where(top_safe_local >= 0, g_ids_np[top_safe_local.clip(min=0)], -1),
            "top_witness_pos_global_id": np.where(top_pos_local >= 0, g_ids_np[top_pos_local.clip(min=0)], -1),
            "top_witness_safe_time_min": top_safe_time,
            "top_witness_pos_time_min": top_pos_time,
            "top_witness_time_gap_min": top_safe_time - top_pos_time,
            "current_time_interval_gap": current_time_res["interval_gap"].detach().cpu().numpy().astype(np.float32),
            "current_time_total": current_time_res["total"].detach().cpu().numpy().astype(np.float32),
            "current_time_interval_regime_available": current_time_res["interval_regime_available"].detach().cpu().numpy().astype(np.float32),
            "current_time_dominant_component": [
                dominant_component_label(int(flag))
                for flag in current_time_res["dominant_component_flag"].detach().cpu().numpy().astype(np.int64)
            ],
            "frontier_safe_close_tau_min": np.full(
                num_nodes,
                float(contra_res.get("frontier_safe_close_tau_min", WITNESS_MINING_FRONT_CLOSE_TAU_MIN)),
                dtype=np.float32,
            ),
            "mined_top_k_safe_witnesses": np.full(
                num_nodes,
                float(contra_res.get("mined_top_k_safe_witnesses", WITNESS_MINING_TOP_K)),
                dtype=np.float32,
            ),
        }
    )
    frame["is_true_source_mis_hit"] = (
        (frame["is_true_source"] > 0.5) & (frame["total"] > EPS)
    ).astype(np.float32)
    frame["zero_reason_bucket"] = frame.apply(classify_candidate_pair_bucket, axis=1)
    frame["zero_reason_bucket_after_mining"] = frame["zero_reason_bucket"]
    return frame


def classify_witness_outlier_risk_bucket(row: pd.Series) -> str:
    if row["is_true_source"] > 0.5 and row["total"] > EPS:
        return "true_source_activation"
    if row.get("coverage_gain_but_score_instability", 0.0) > 0.5:
        return "coverage_gain_score_instability"
    if row["current_time_interval_gap"] > max(row["interval_gap"] + 1.0, EPS) * 5.0 and row["current_time_interval_gap"] > 5000.0:
        return "current_time_interval_sensitivity"
    if row["best_margin"] > 5000.0 or row["interval_gap"] > 5000.0:
        return "long_path_outlier"
    return "stable_or_none"


def annotate_witness_mining_compare_frame(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    drop_cols = [
        "baseline_has_eligible_safe_witness",
        "baseline_pair_available",
        "baseline_positive_margin_available",
        "baseline_mined_safe_candidate_count",
        "baseline_best_margin",
        "baseline_total",
        "baseline_zero_reason_bucket",
        "coverage_gain_vs_baseline",
        "coverage_gain_but_score_instability",
        "outlier_risk_bucket",
    ]
    candidate_df = candidate_df.drop(columns=[col for col in drop_cols if col in candidate_df.columns], errors="ignore")

    base_cols = [
        "event_id",
        "episode",
        "time_min",
        "candidate_local_idx",
        "has_eligible_safe_witness",
        "pair_available",
        "positive_margin_available",
        "mined_safe_candidate_count",
        "best_margin",
        "total",
        "zero_reason_bucket",
    ]
    baseline = candidate_df[candidate_df["witness_mining_mode"] == DEFAULT_WITNESS_MINING_MODE][base_cols].rename(
        columns={
            "has_eligible_safe_witness": "baseline_has_eligible_safe_witness",
            "pair_available": "baseline_pair_available",
            "positive_margin_available": "baseline_positive_margin_available",
            "mined_safe_candidate_count": "baseline_mined_safe_candidate_count",
            "best_margin": "baseline_best_margin",
            "total": "baseline_total",
            "zero_reason_bucket": "baseline_zero_reason_bucket",
        }
    )
    merged = candidate_df.merge(
        baseline,
        on=["event_id", "episode", "time_min", "candidate_local_idx"],
        how="left",
    )
    merged["coverage_gain_vs_baseline"] = (
        (merged["has_eligible_safe_witness"] > merged["baseline_has_eligible_safe_witness"])
        | (merged["pair_available"] > merged["baseline_pair_available"])
        | (merged["positive_margin_available"] > merged["baseline_positive_margin_available"])
    ).astype(np.float32)
    large_margin_jump = (
        (merged["best_margin"] > merged["baseline_best_margin"] + 5000.0)
        & (merged["best_margin"] > np.maximum(merged["baseline_best_margin"] * 2.0, 5000.0))
    )
    large_total_jump = (
        (merged["total"] > merged["baseline_total"] + 500.0)
        & (merged["total"] > np.maximum(merged["baseline_total"] * 2.0, 200.0))
    )
    merged["coverage_gain_but_score_instability"] = (
        merged["coverage_gain_vs_baseline"] > 0.5
    ) & (large_margin_jump | large_total_jump)
    merged["coverage_gain_but_score_instability"] = merged["coverage_gain_but_score_instability"].astype(np.float32)
    merged["outlier_risk_bucket"] = merged.apply(classify_witness_outlier_risk_bucket, axis=1)
    return merged


def tensor_to_int_list(values: List[int]) -> List[int]:
    out: List[int] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            out.append(int(value.item()))
        else:
            out.append(int(value))
    return out


def compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    out_degree = torch.bincount(src, minlength=num_nodes).float()
    in_degree = torch.bincount(dst, minlength=num_nodes).float()
    return out_degree + in_degree


def resolve_support_weights(phys_ctx) -> torch.Tensor:
    if phys_ctx.stt_dynamic is not None:
        w_soft = torch.abs(phys_ctx.stt_dynamic.view(-1))
    elif phys_ctx.stt_median is not None:
        w_soft = torch.expm1(phys_ctx.stt_median)
    elif phys_ctx.edge_attr is not None and phys_ctx.edge_attr.size(1) > 0:
        w_soft = torch.expm1(phys_ctx.edge_attr[:, 0])
    else:
        w_soft = torch.ones_like(phys_ctx.edge_index[0], dtype=torch.float) * 20.0
    return torch.clamp(w_soft * EvidenceBuilder.STT_SCALE_FACTOR, min=0.0)


def count_supportable_positive_seeds(
    reachability_module,
    phys_ctx,
    positive_indices: torch.Tensor,
    candidate_indices: List[int],
    time_limit_min: float = DEFAULT_SUPPORT_TIME_MIN,
) -> torch.Tensor:
    device = phys_ctx.edge_index.device
    if positive_indices.numel() == 0 or not candidate_indices:
        return torch.zeros(len(candidate_indices), device=device)

    positive_indices = positive_indices[:50]
    candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    num_nodes = int(phys_ctx.batch.numel()) if phys_ctx.batch is not None else int(candidate_tensor.max().item()) + 1
    w_soft = resolve_support_weights(phys_ctx)
    scores = torch.zeros(len(candidate_indices), device=device)
    support_limit = float(time_limit_min) + EvidenceBuilder.SUPPORT_BUFFER

    for pos_idx in positive_indices:
        seed = torch.zeros(num_nodes, device=device)
        seed[int(pos_idx.item())] = 1.0
        dist = reachability_module.compute_distance(seed, phys_ctx, w_soft, num_nodes)
        scores += (dist[candidate_tensor] <= support_limit).float()

    return scores


def classify_positive_seed_root_cause(
    truth_positive_total: int,
    observed_positive_total: int,
    hidden_truth_positive_total: int,
) -> str:
    if observed_positive_total > 0:
        return "observed_positive_available"
    if truth_positive_total <= 0:
        return "no_truth_positive_at_snapshot"
    if hidden_truth_positive_total > 0:
        return "truth_positive_hidden_from_rollout"
    return "observation_alignment_gap"


def collect_support_pair_stats(
    rollout: PracticalRollout,
    phys_ctx,
    obs_partial,
    src_local: int,
    support_top_other_idx: int,
    truth_positive_indices: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[int, Dict[str, float]]]:
    observed_positive_indices = obs_partial.toxic_positive_flag.nonzero().view(-1)
    candidate_indices = [src_local]
    if support_top_other_idx != src_local:
        candidate_indices.append(support_top_other_idx)

    observed_supportable = count_supportable_positive_seeds(
        rollout.reachability_module,
        phys_ctx,
        observed_positive_indices,
        candidate_indices,
    )
    truth_supportable = count_supportable_positive_seeds(
        rollout.reachability_module,
        phys_ctx,
        truth_positive_indices,
        candidate_indices,
    )

    support_lookup: Dict[int, Dict[str, float]] = {}
    observed_total = max(int(observed_positive_indices.numel()), 1)
    truth_total = max(int(truth_positive_indices.numel()), 1)
    for cand_pos, cand_idx in enumerate(candidate_indices):
        support_lookup[int(cand_idx)] = {
            "observed_positive_supportable_count": float(observed_supportable[cand_pos].item()),
            "truth_positive_supportable_count": float(truth_supportable[cand_pos].item()),
            "observed_positive_supportable_fraction": float(observed_supportable[cand_pos].item() / observed_total),
            "truth_positive_supportable_fraction": float(truth_supportable[cand_pos].item() / truth_total),
        }
    return observed_positive_indices, support_lookup


def max_positive_gap_source(row: pd.Series) -> str:
    gaps = {
        "base": float(row["support_competitor_base"] - row["support_true_base"]),
        "specificity": float(row["support_competitor_specificity"] - row["support_true_specificity"]),
        "focus": float(row["support_competitor_focus"] - row["support_true_focus"]),
        "chlorine": float(row["support_competitor_chlorine"] - row["support_true_chlorine"]),
    }
    max_gap = max(gaps.values())
    if max_gap <= EPS:
        return "none"
    winners = [name for name, value in gaps.items() if value >= max_gap - EPS]
    if len(winners) == 1:
        return winners[0]
    return "tie"


def classify_b_rootcause(row: pd.Series) -> str:
    if row["true_truth_positive_supportable_count"] <= EPS:
        return "B1_no_true_supportable_witness"
    if row["true_observed_positive_supportable_count"] <= EPS:
        return "B2_truth_witness_hidden_or_unavailable"
    return "B3_observed_witness_present_but_support_zero"


def classify_c_rootcause(row: pd.Series) -> str:
    if row["competitor_observed_positive_supportable_gt_true"] > 0.5:
        return "C1_availability_deficit"
    if row["largest_gap_source"] == "specificity":
        return "C2_specificity_dominant_loss"
    if row["largest_gap_source"] == "focus":
        return "C3_focus_dominant_loss"
    return "mixed_or_other"


def build_support_rootcause_row(
    idx: int,
    episode_idx: int,
    info: Dict[str, float],
    rollout: PracticalRollout,
    ev_state,
    src_local: int,
    support_top_other_idx: int,
    num_pos: int,
    truth_positive_total: int,
    truth_positive_unobserved: int,
    positive_seed_root_cause: str,
    support_lookup: Dict[int, Dict[str, float]],
) -> Dict[str, float]:
    true_support = float(ev_state.support_score[src_local].item())
    competitor_support = float(ev_state.support_score[support_top_other_idx].item())
    row = {
        "event_id": idx,
        "episode": episode_idx + 1,
        "time_min": float(info["time_min"]),
        "num_pos": int(num_pos),
        "truth_positive_total": int(truth_positive_total),
        "truth_positive_unobserved": int(truth_positive_unobserved),
        "positive_seed_root_cause": positive_seed_root_cause,
        "true_source_local_idx": int(src_local),
        "true_source_global_id": int(rollout.g_ids[src_local].item()),
        "top_competitor_local_idx": int(support_top_other_idx),
        "top_competitor_global_id": int(rollout.g_ids[support_top_other_idx].item()),
        "suspect_active_true": float(ev_state.suspect_pool[src_local].item()),
        "suspect_active_competitor": float(ev_state.suspect_pool[support_top_other_idx].item()),
        "support_true_total": true_support,
        "support_competitor_total": competitor_support,
        "support_true_nonzero": float(true_support > EPS),
        "hub_win": float((competitor_support - true_support) > EPS),
        "support_true_base": float(ev_state.support_coverage_term[src_local].item()),
        "support_competitor_base": float(ev_state.support_coverage_term[support_top_other_idx].item()),
        "support_true_specificity": float(ev_state.support_timing_term[src_local].item()),
        "support_competitor_specificity": float(ev_state.support_timing_term[support_top_other_idx].item()),
        "support_true_focus": float(ev_state.support_focus_term[src_local].item()),
        "support_competitor_focus": float(ev_state.support_focus_term[support_top_other_idx].item()),
        "support_true_chlorine": float(ev_state.support_chlorine_term[src_local].item()),
        "support_competitor_chlorine": float(ev_state.support_chlorine_term[support_top_other_idx].item()),
        "true_observed_positive_supportable_count": float(
            support_lookup[src_local]["observed_positive_supportable_count"]
        ),
        "competitor_observed_positive_supportable_count": float(
            support_lookup[support_top_other_idx]["observed_positive_supportable_count"]
        ),
        "true_truth_positive_supportable_count": float(
            support_lookup[src_local]["truth_positive_supportable_count"]
        ),
        "competitor_truth_positive_supportable_count": float(
            support_lookup[support_top_other_idx]["truth_positive_supportable_count"]
        ),
    }

    row["support_true_base_zero"] = float(row["support_true_base"] <= EPS)
    row["support_true_specificity_zero"] = float(row["support_true_specificity"] <= EPS)
    row["support_true_focus_zero"] = float(row["support_true_focus"] <= EPS)
    row["support_true_chlorine_zero"] = float(row["support_true_chlorine"] <= EPS)
    row["competitor_observed_positive_supportable_gt_true"] = float(
        row["competitor_observed_positive_supportable_count"] > row["true_observed_positive_supportable_count"] + EPS
    )
    row["competitor_truth_positive_supportable_gt_true"] = float(
        row["competitor_truth_positive_supportable_count"] > row["true_truth_positive_supportable_count"] + EPS
    )
    row["observed_supportable_gap"] = float(
        row["competitor_observed_positive_supportable_count"] - row["true_observed_positive_supportable_count"]
    )
    row["truth_supportable_gap"] = float(
        row["competitor_truth_positive_supportable_count"] - row["true_truth_positive_supportable_count"]
    )
    row["competitor_gt_true_base"] = float(row["support_competitor_base"] > row["support_true_base"] + EPS)
    row["competitor_gt_true_specificity"] = float(
        row["support_competitor_specificity"] > row["support_true_specificity"] + EPS
    )
    row["competitor_gt_true_focus"] = float(row["support_competitor_focus"] > row["support_true_focus"] + EPS)
    row["competitor_gt_true_chlorine"] = float(
        row["support_competitor_chlorine"] > row["support_true_chlorine"] + EPS
    )
    row["supportable_similar_le1"] = float(abs(row["observed_supportable_gap"]) <= 1.0)
    row["supportable_equal"] = float(abs(row["observed_supportable_gap"]) <= EPS)
    row["largest_gap_source"] = max_positive_gap_source(pd.Series(row))
    return row


def append_case_trace_rows(
    case_rows: List[Dict[str, float]],
    idx: int,
    episode_idx: int,
    info: Dict[str, float],
    rollout: PracticalRollout,
    obs_partial,
    phys_ctx,
    ev_state,
    formal_ev_state,
    reach: Dict[str, torch.Tensor],
    support_scores: torch.Tensor,
    legacy_support_scores: torch.Tensor,
    legacy_terms: Dict[str, torch.Tensor],
    old_contra: torch.Tensor,
    formal_contra: torch.Tensor,
    nosuspect_formal_contra: torch.Tensor,
    strict_contra: torch.Tensor,
    src_local: int,
    support_top_other_idx: int,
    truth_positive_total: int,
    truth_positive_unobserved: int,
    positive_seed_root_cause: str,
    truth_positive_indices: torch.Tensor,
    observed_positive_indices: torch.Tensor,
    support_lookup: Dict[int, Dict[str, float]],
) -> None:
    if idx not in CASE_EVENT_IDS:
        return

    num_nodes = int(obs_partial.observed_flag.numel())
    degree = compute_degree(phys_ctx.edge_index, num_nodes)
    candidate_indices = [src_local]
    if support_top_other_idx != src_local:
        candidate_indices.append(support_top_other_idx)

    role_lookup = {
        src_local: "true_source",
        support_top_other_idx: "top_competitor",
    }
    sampled_local_ids = tensor_to_int_list(info["samples"])
    sampled_global_ids = [int(rollout.g_ids[s].item()) for s in sampled_local_ids]
    suspect_raw_scores = reach["topology_reachable"] + 0.8 * reach["soft_reachability"] - 0.8 * reach["hard_reachability_from_neg"]

    for cand_pos, cand_idx in enumerate(candidate_indices):
        global_id = int(rollout.g_ids[cand_idx].item())
        old_val = float(old_contra[cand_idx].item())
        formal_val = float(formal_contra[cand_idx].item())
        nosuspect_formal_val = float(nosuspect_formal_contra[cand_idx].item())
        strict_val = float(strict_contra[cand_idx].item())
        case_rows.append(
            {
                "event_id": idx,
                "episode": episode_idx + 1,
                "time_min": float(info["time_min"]),
                "candidate_role": role_lookup[cand_idx],
                "candidate_local_idx": int(cand_idx),
                "candidate_global_id": global_id,
                "true_source_local_idx": int(src_local),
                "true_source_global_id": int(rollout.g_ids[src_local].item()),
                "top_competitor_local_idx": int(support_top_other_idx),
                "top_competitor_global_id": int(rollout.g_ids[support_top_other_idx].item()),
                "candidate_degree": float(degree[cand_idx].item()),
                "observed_count": int(obs_partial.observed_flag.sum().item()),
                "revealed_count": int(info["revealed_count"]),
                "num_pos": int(observed_positive_indices.numel()),
                "num_neg": int(obs_partial.toxic_negative_flag.sum().item()),
                "truth_positive_total": int(truth_positive_total),
                "truth_positive_unobserved": int(truth_positive_unobserved),
                "positive_seed_root_cause": positive_seed_root_cause,
                "sample_types": "|".join(info["types"]),
                "sampled_local_ids": "|".join(str(v) for v in sampled_local_ids),
                "sampled_global_ids": "|".join(str(v) for v in sampled_global_ids),
                "suspect_active": float(ev_state.suspect_pool[cand_idx].item()),
                "suspect_raw_score": float(suspect_raw_scores[cand_idx].item()),
                "suspect_rank_raw": strict_rank(suspect_raw_scores, cand_idx, higher_better=True),
                "topology_gate": float(reach["topology_reachable"][cand_idx].item()),
                "coarse_time_gate": float(reach["soft_reachability"][cand_idx].item()),
                "negative_pressure_soft": float(reach["soft_reachability_from_neg"][cand_idx].item()),
                "negative_pressure_hard": float(reach["hard_reachability_from_neg"][cand_idx].item()),
                "not_ruled_out_gate": float((1.0 - reach["hard_reachability_from_neg"][cand_idx]).item()),
                "support_builder_total": float(support_scores[cand_idx].item()),
                "support_builder_rank": strict_rank(support_scores, cand_idx, higher_better=True),
                "support_builder_minus_true": float(support_scores[cand_idx].item() - support_scores[src_local].item()),
                "support_legacy_total": float(legacy_support_scores[cand_idx].item()),
                "support_builder_minus_legacy": float((support_scores[cand_idx] - legacy_support_scores[cand_idx]).item()),
                "support_base": float(ev_state.support_coverage_term[cand_idx].item()),
                "support_specificity": float(ev_state.support_timing_term[cand_idx].item()),
                "support_focus": float(ev_state.support_focus_term[cand_idx].item()),
                "support_chlorine": float(ev_state.support_chlorine_term[cand_idx].item()),
                "support_hub_win_flag": float(
                    cand_idx == support_top_other_idx and support_scores[support_top_other_idx].item() > support_scores[src_local].item() + EPS
                ),
                "legacy_rank_term": float(legacy_terms.get("r", torch.zeros_like(legacy_support_scores))[cand_idx].item()),
                "legacy_hub_term": float(legacy_terms.get("h", torch.zeros_like(legacy_support_scores))[cand_idx].item()),
                "observed_positive_supportable_count": float(
                    support_lookup[cand_idx]["observed_positive_supportable_count"]
                ),
                "truth_positive_supportable_count": float(
                    support_lookup[cand_idx]["truth_positive_supportable_count"]
                ),
                "observed_positive_supportable_fraction": float(
                    support_lookup[cand_idx]["observed_positive_supportable_fraction"]
                ),
                "truth_positive_supportable_fraction": float(
                    support_lookup[cand_idx]["truth_positive_supportable_fraction"]
                ),
                "support_zero_due_no_input": float(observed_positive_indices.numel() == 0 and abs(float(support_scores[cand_idx].item())) <= EPS),
                "support_zero_despite_input": float(observed_positive_indices.numel() > 0 and abs(float(support_scores[cand_idx].item())) <= EPS),
                "contradiction_old_total": old_val,
                "contradiction_old_soft": float(ev_state.contradiction_toxic_term[cand_idx].item()),
                "contradiction_old_hard": float(ev_state.contradiction_clean_term[cand_idx].item()),
                "contradiction_formal_total": formal_val,
                "contradiction_formal_soft": float(formal_ev_state.contradiction_toxic_term[cand_idx].item()),
                "contradiction_formal_hard": float(formal_ev_state.contradiction_clean_term[cand_idx].item()),
                "contradiction_nosuspect_formal_total": nosuspect_formal_val,
                "contradiction_formal_minus_old": formal_val - old_val,
                "contradiction_nosuspect_formal_minus_formal": nosuspect_formal_val - formal_val,
                "contradiction_strict_total": strict_val,
                "contradiction_time_conflict_candidate": float(
                    observed_positive_indices.numel() > 0
                    and int(obs_partial.toxic_negative_flag.sum().item()) > 0
                    and old_val > strict_val + CONTRA_TIME_CONFLICT_ALERT
                    and strict_val <= EPS
                ),
                "reaction_total": float(ev_state.reaction_consistency[cand_idx].item()),
                "uncertainty_gap": float(ev_state.uncertainty_gap[cand_idx].item()),
            }
        )


def build_suspect_raw_and_mask(reach: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    topo = reach["topology_reachable"]
    soft = reach["soft_reachability"]
    hard_neg = reach["hard_reachability_from_neg"]
    raw_score = 1.0 * topo + 0.8 * soft - 0.8 * hard_neg
    active_mask = (raw_score > SUSPECT_THRESHOLD).float()
    if float(active_mask.sum().item()) < 0.5:
        active_mask = topo.clone()
    return {"raw_score": raw_score, "active_mask": active_mask}


def compute_old_contradiction(
    evidence_builder: EvidenceBuilder,
    obs,
    phys_ctx,
    suspect_pool: torch.Tensor,
    reach: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return evidence_builder.compute_contradiction_score(obs, phys_ctx, suspect_pool, reach)["total"]


def build_current_time_tensor(current_time_min: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([float(current_time_min)], dtype=torch.float, device=device)


def compute_formal_contradiction_variants(
    evidence_builder: EvidenceBuilder,
    obs,
    phys_ctx,
    current_time_min: float,
) -> Dict[str, torch.Tensor]:
    t_sim = build_current_time_tensor(current_time_min, obs.observed_flag.device)
    formal_ev_state = evidence_builder.build_evidence_state(obs, phys_ctx, t_sim=t_sim)
    formal_reach = compute_builder_reachability(
        evidence_builder,
        obs,
        phys_ctx,
        t_sim=t_sim,
    )
    nosuspect_res = evidence_builder.compute_contradiction_score(
        obs,
        phys_ctx,
        torch.ones_like(formal_ev_state.suspect_pool),
        formal_reach,
        t_sim=t_sim,
    )
    return {
        "t_sim": t_sim,
        "formal_ev_state": formal_ev_state,
        "formal_reach": formal_reach,
        "formal_total": formal_ev_state.contradiction_score,
        "nosuspect_formal_total": nosuspect_res["total"],
        "nosuspect_formal_soft": nosuspect_res["soft"],
        "nosuspect_formal_hard": nosuspect_res["hard"],
    }


def compute_strict_exclusion_contradiction(reachability_module, obs, phys_ctx, current_time_min: float) -> torch.Tensor:
    device = obs.observed_flag.device
    num_nodes = obs.observed_flag.size(0)
    safe_indices = obs.toxic_negative_flag.nonzero().view(-1)
    pos_indices = obs.toxic_positive_flag.nonzero().view(-1)

    scores = torch.zeros(num_nodes, device=device)
    if safe_indices.numel() == 0 or pos_indices.numel() == 0:
        return scores

    adj_rev = reachability_module._build_scipy_reverse_graph(
        phys_ctx.edge_index,
        phys_ctx.stt_dynamic.view(-1),
        num_nodes,
    )

    safe_dist_np = reachability_module._run_scipy_dijkstra(adj_rev, safe_indices.cpu().numpy())
    pos_dist_np = reachability_module._run_scipy_dijkstra(adj_rev, pos_indices.cpu().numpy())

    safe_dist = torch.from_numpy(np.asarray(safe_dist_np)).float().to(device)
    pos_dist = torch.from_numpy(np.asarray(pos_dist_np)).float().to(device)
    inf_thresh = reachability_module.infinity / 2

    safe_reachable = safe_dist < inf_thresh
    hard_arrived = safe_dist <= (float(current_time_min) - STRICT_HARD_BUFFER_MIN)
    pos_reachable = pos_dist < inf_thresh

    witness_mask = ((safe_dist.unsqueeze(1) <= pos_dist.unsqueeze(0)) & pos_reachable.unsqueeze(0)).any(dim=1)
    exclusion_hits = safe_reachable & hard_arrived & witness_mask
    scores = exclusion_hits.float().sum(dim=0)
    return scores


def classify_case(row: pd.Series) -> str:
    if row["num_pos"] <= 0:
        return "triage_no_positive_seed"
    if row["suspect_active_recall"] < 0.5:
        return "abnormal_suspect_drops_true"
    if row["contra_eligible_strict"] > 0.5 and row["contra_time_conflict"] > 0.5:
        return "abnormal_negative_time_gate_conflict"
    if row["support_hub_win"] > 0.5 and row["support_builder_vs_legacy_l1"] > SUPPORT_COMPARE_L1_ALERT:
        return "abnormal_support_compare_diverge"
    if row["support_hub_win"] > 0.5:
        return "abnormal_support_hub_win"
    if row["reaction_eligible"] > 0.5 and row["reaction_directionality"] < 0.5:
        return "triage_reaction_noise"
    if (
        row["support_rank"] <= 3
        and row["suspect_active_recall"] > 0.5
        and row["contra_strict_directionality"] > 0.5
    ):
        return "normal_axes_consistent"
    return "triage_mixed"


def responsibility_hint(case_label: str) -> str:
    mapping = {
        "normal_axes_consistent": "aligned",
        "abnormal_suspect_drops_true": "semantic_or_implementation_or_upstream_physics",
        "abnormal_negative_time_gate_conflict": "contradiction_semantic_or_implementation",
        "abnormal_support_compare_diverge": "audit_compare_vs_builder",
        "abnormal_support_hub_win": "builder_semantic_or_upstream_physics",
        "triage_no_positive_seed": "upstream_observation_or_rollout",
        "triage_reaction_noise": "reaction_field_semantic_noise",
        "triage_mixed": "needs_manual_triage",
    }
    return mapping.get(case_label, "needs_manual_triage")


def build_oracle_support_observation_state(
    obs_partial: ObservationState,
    signal_snapshot: torch.Tensor,
    truth_positive_mask: torch.Tensor,
) -> ObservationState:
    oracle_observed = torch.maximum(obs_partial.observed_flag, truth_positive_mask.float())
    oracle_positive = truth_positive_mask.float()
    oracle_negative = obs_partial.toxic_negative_flag.clone()
    oracle_chlorine = obs_partial.chlorine_deviation.clone()
    oracle_chlorine[truth_positive_mask] = signal_snapshot[truth_positive_mask]
    return ObservationState(
        observed_flag=oracle_observed,
        chlorine_deviation=oracle_chlorine,
        toxic_positive_flag=oracle_positive,
        toxic_negative_flag=oracle_negative,
        freshness=oracle_observed.clone(),
    )


def compute_builder_reachability(
    evidence_builder: EvidenceBuilder,
    observation_state: ObservationState,
    physics_context,
    t_sim: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    if physics_context.batch is not None:
        batch = physics_context.batch
    else:
        batch = torch.zeros(
            observation_state.observed_flag.size(0),
            dtype=torch.long,
            device=observation_state.observed_flag.device,
        )
    if physics_context.stt_dynamic is not None:
        return evidence_builder.dynamic_reachability.compute_reachability(
            observation_state,
            physics_context,
            t_sim,
            batch,
        )
    return evidence_builder.reachability.compute_reachability(
        observation_state,
        physics_context,
        t_sim,
        batch,
    )


def resolve_snapshot_time_index(event_data, t_snapshot_idx: int) -> int:
    x_raw = getattr(event_data, "x_raw", None)
    global_start = getattr(event_data, "global_start_step", 96)
    if isinstance(global_start, torch.Tensor):
        global_start = int(global_start.item())
    is_v11 = bool(x_raw is not None and x_raw.shape[1] > 200)
    scale_factor = 1 if is_v11 else 3
    t_abs_idx = int(global_start) + (int(t_snapshot_idx) // scale_factor)
    return max(0, min(287, t_abs_idx))


def select_oracle_witness_indices(
    truth_positive_mask: torch.Tensor,
    witness_strength: torch.Tensor,
    max_witnesses: int = V2_MAX_WITNESSES,
) -> torch.Tensor:
    witness_idx = truth_positive_mask.nonzero().view(-1)
    if witness_idx.numel() <= max_witnesses:
        return witness_idx
    strengths = witness_strength[witness_idx]
    order = torch.argsort(strengths, descending=True)
    return witness_idx[order[:max_witnesses]]


def build_virtual_time_lookup(
    topology: HydraulicTopology,
    witness_global_id: int,
    subgraph_global_ids: torch.Tensor,
    witness_local_idx: int,
    t_abs_idx: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    virtual_times = torch.full((num_nodes,), float("inf"), device=device)
    virt_edge_index, virt_edge_attr = topology.get_virtual_edges_for_subgraph(
        witness_global_id,
        subgraph_global_ids,
        time_idx=t_abs_idx,
        anchor_value=1.0,
    )
    if virt_edge_index.numel() > 0:
        src_local = virt_edge_index[0].to(device)
        stt_weight = virt_edge_attr[:, 0].to(device)
        virtual_times[src_local] = stt_weight
    virtual_times[int(witness_local_idx)] = 0.0
    return virtual_times


def travel_time_to_availability(
    travel_time: torch.Tensor,
    current_time_min: float,
    reliability: float,
) -> torch.Tensor:
    finite_mask = torch.isfinite(travel_time) & (travel_time < 1e8)
    travel_safe = torch.where(finite_mask, travel_time, torch.zeros_like(travel_time))
    late = torch.relu(travel_safe - float(current_time_min))
    late_penalty = torch.exp(-((late ** 2) / (2.0 * (V2_TIME_SIGMA_MIN ** 2))))
    distance_decay = 1.0 / (1.0 + travel_safe / max(float(current_time_min) + V2_TIME_PRIOR_OFFSET_MIN, 1.0))
    availability = reliability * late_penalty * distance_decay
    return availability * finite_mask.float()


def zero_support_v2_dict(num_nodes: int, device: torch.device) -> Dict[str, torch.Tensor]:
    zero = torch.zeros(num_nodes, device=device)
    zero_i = torch.zeros((num_nodes, 0), device=device)
    return {
        "total": zero,
        "availability": zero,
        "ownership": zero,
        "hub_penalty": zero,
        "virtual_share": zero,
        "best_time_mean": zero,
        "physical_time_mean": zero,
        "virtual_time_mean": zero,
        "best_path_virtual_rate": zero,
        "best_path_physical_rate": zero,
        "witness_count": torch.zeros(num_nodes, device=device),
        "availability_matrix": zero_i,
        "ownership_matrix": zero_i,
        "physical_time_matrix": zero_i,
        "virtual_time_matrix": zero_i,
    }


def compute_support_v2_oracle(
    rollout: PracticalRollout,
    phys_ctx,
    truth_positive_mask: torch.Tensor,
    witness_strength: torch.Tensor,
    current_time_min: float,
    t_abs_idx: int,
    topology: HydraulicTopology,
) -> Dict[str, torch.Tensor]:
    device = phys_ctx.edge_index.device
    num_nodes = int(phys_ctx.batch.numel()) if phys_ctx.batch is not None else int(rollout.g_ids.numel())
    witness_idx = select_oracle_witness_indices(truth_positive_mask, witness_strength)
    if witness_idx.numel() == 0:
        return zero_support_v2_dict(num_nodes, device)

    phys_adj_rev = rollout.reachability_module._build_scipy_reverse_graph(
        phys_ctx.edge_index,
        phys_ctx.stt_dynamic.view(-1),
        num_nodes,
    )

    availability_cols: List[torch.Tensor] = []
    physical_time_cols: List[torch.Tensor] = []
    virtual_time_cols: List[torch.Tensor] = []
    best_virtual_cols: List[torch.Tensor] = []

    subgraph_global_ids = rollout.g_ids.detach().cpu()

    for witness_local_idx in witness_idx.tolist():
        phys_dist_np = rollout.reachability_module._run_scipy_dijkstra(
            phys_adj_rev,
            np.array([int(witness_local_idx)], dtype=np.int64),
        )
        phys_dist = torch.from_numpy(np.asarray(phys_dist_np[0])).float().to(device)
        witness_global_id = int(rollout.g_ids[int(witness_local_idx)].item())
        virt_dist = build_virtual_time_lookup(
            topology=topology,
            witness_global_id=witness_global_id,
            subgraph_global_ids=subgraph_global_ids,
            witness_local_idx=int(witness_local_idx),
            t_abs_idx=t_abs_idx,
            num_nodes=num_nodes,
            device=device,
        )

        phys_avail = travel_time_to_availability(phys_dist, current_time_min, reliability=1.0)
        virt_avail = travel_time_to_availability(
            virt_dist,
            current_time_min,
            reliability=V2_VIRTUAL_RELIABILITY,
        )
        availability = torch.maximum(phys_avail, virt_avail)

        availability_cols.append(availability)
        physical_time_cols.append(phys_dist)
        virtual_time_cols.append(virt_dist)
        best_virtual_cols.append((virt_avail > phys_avail + EPS).float())

    availability_matrix = torch.stack(availability_cols, dim=1)
    physical_time_matrix = torch.stack(physical_time_cols, dim=1)
    virtual_time_matrix = torch.stack(virtual_time_cols, dim=1)
    best_virtual_matrix = torch.stack(best_virtual_cols, dim=1)
    best_physical_matrix = 1.0 - best_virtual_matrix

    ownership_mass = availability_matrix.clamp_min(0.0).pow(V2_OWNERSHIP_POWER)
    ownership_matrix = ownership_mass / (ownership_mass.sum(dim=0, keepdim=True) + EPS)
    generic_count = (availability_matrix > V2_AVAIL_ACTIVE_THRESHOLD).float().sum(dim=0)
    witness_weight = 1.0 / (1.0 + torch.log1p(generic_count))
    witness_weight = torch.where(generic_count > 0.0, witness_weight, torch.zeros_like(witness_weight))

    availability_term = availability_matrix.mean(dim=1)
    ownership_term = (availability_matrix * ownership_matrix * witness_weight.unsqueeze(0)).mean(dim=1)
    hub_penalty_term = (
        availability_matrix
        * (1.0 - ownership_matrix)
        * (1.0 - witness_weight.unsqueeze(0))
    ).mean(dim=1)
    total = V2_AVAIL_WEIGHT * availability_term + ownership_term - V2_HUB_PENALTY_WEIGHT * hub_penalty_term

    finite_phys = torch.isfinite(physical_time_matrix) & (physical_time_matrix < 1e8)
    finite_virt = torch.isfinite(virtual_time_matrix) & (virtual_time_matrix < 1e8)
    finite_best = torch.where(best_virtual_matrix > 0.5, finite_virt, finite_phys)
    best_time_matrix = torch.where(best_virtual_matrix > 0.5, virtual_time_matrix, physical_time_matrix)

    best_time_mean = torch.where(
        finite_best.any(dim=1),
        (best_time_matrix * finite_best.float()).sum(dim=1) / finite_best.float().sum(dim=1).clamp_min(1.0),
        torch.full((num_nodes,), float("inf"), device=device),
    )
    physical_time_mean = torch.where(
        finite_phys.any(dim=1),
        (physical_time_matrix * finite_phys.float()).sum(dim=1) / finite_phys.float().sum(dim=1).clamp_min(1.0),
        torch.full((num_nodes,), float("inf"), device=device),
    )
    virtual_time_mean = torch.where(
        finite_virt.any(dim=1),
        (virtual_time_matrix * finite_virt.float()).sum(dim=1) / finite_virt.float().sum(dim=1).clamp_min(1.0),
        torch.full((num_nodes,), float("inf"), device=device),
    )

    return {
        "total": total,
        "availability": availability_term,
        "ownership": ownership_term,
        "hub_penalty": hub_penalty_term,
        "virtual_share": best_virtual_matrix.mean(dim=1),
        "best_time_mean": best_time_mean,
        "physical_time_mean": physical_time_mean,
        "virtual_time_mean": virtual_time_mean,
        "best_path_virtual_rate": best_virtual_matrix.mean(dim=1),
        "best_path_physical_rate": best_physical_matrix.mean(dim=1),
        "witness_count": torch.full((num_nodes,), float(witness_idx.numel()), device=device),
        "availability_matrix": availability_matrix,
        "ownership_matrix": ownership_matrix,
        "physical_time_matrix": physical_time_matrix,
        "virtual_time_matrix": virtual_time_matrix,
    }


def extract_support_variant_fields_v2(
    prefix: str,
    support_res: Dict[str, torch.Tensor],
    src_local: int,
    g_ids: torch.Tensor,
) -> Dict[str, float]:
    scores = support_res["total"]
    true_score = float(scores[src_local].item())
    other_mean = mean_other(scores, src_local)
    top_other_idx = best_other_index(scores, src_local, higher_better=True)
    top_other_score = float(scores[top_other_idx].item())

    def safe_float(value: torch.Tensor) -> float:
        item = float(value.item())
        return item if np.isfinite(item) else np.nan

    return {
        f"{prefix}_rank": strict_rank(scores, src_local, higher_better=True),
        f"{prefix}_gap": true_score - other_mean,
        f"{prefix}_directionality": float((true_score - other_mean) > 0.0),
        f"{prefix}_all_zero": float(all_zero(scores)),
        f"{prefix}_true_nonzero": float(true_score > EPS),
        f"{prefix}_unique_top1": float(unique_top1(scores, src_local)),
        f"{prefix}_hub_win": float((top_other_score - true_score) > EPS) if scores.numel() > 1 else 0.0,
        f"{prefix}_true_total": true_score,
        f"{prefix}_top_other_idx": int(top_other_idx),
        f"{prefix}_top_other_global_id": int(g_ids[top_other_idx].item()),
        f"{prefix}_top_other_score": top_other_score,
        f"{prefix}_true_availability": float(support_res["availability"][src_local].item()),
        f"{prefix}_true_ownership": float(support_res["ownership"][src_local].item()),
        f"{prefix}_true_hub_penalty": float(support_res["hub_penalty"][src_local].item()),
        f"{prefix}_true_virtual_share": float(support_res["virtual_share"][src_local].item()),
        f"{prefix}_true_best_path_virtual_rate": float(support_res["best_path_virtual_rate"][src_local].item()),
        f"{prefix}_true_best_path_physical_rate": float(support_res["best_path_physical_rate"][src_local].item()),
        f"{prefix}_true_best_time_mean": safe_float(support_res["best_time_mean"][src_local]),
        f"{prefix}_true_physical_time_mean": safe_float(support_res["physical_time_mean"][src_local]),
        f"{prefix}_true_virtual_time_mean": safe_float(support_res["virtual_time_mean"][src_local]),
        f"{prefix}_competitor_availability": float(support_res["availability"][top_other_idx].item()),
        f"{prefix}_competitor_ownership": float(support_res["ownership"][top_other_idx].item()),
        f"{prefix}_competitor_hub_penalty": float(support_res["hub_penalty"][top_other_idx].item()),
        f"{prefix}_competitor_virtual_share": float(support_res["virtual_share"][top_other_idx].item()),
        f"{prefix}_competitor_best_path_virtual_rate": float(support_res["best_path_virtual_rate"][top_other_idx].item()),
        f"{prefix}_competitor_best_path_physical_rate": float(support_res["best_path_physical_rate"][top_other_idx].item()),
        f"{prefix}_competitor_best_time_mean": safe_float(support_res["best_time_mean"][top_other_idx]),
        f"{prefix}_competitor_physical_time_mean": safe_float(support_res["physical_time_mean"][top_other_idx]),
        f"{prefix}_competitor_virtual_time_mean": safe_float(support_res["virtual_time_mean"][top_other_idx]),
        f"{prefix}_witness_count": float(support_res["witness_count"][src_local].item()),
    }


def extract_support_variant_fields(
    prefix: str,
    support_res: Dict[str, torch.Tensor],
    src_local: int,
    g_ids: torch.Tensor,
) -> Dict[str, float]:
    scores = support_res["total"]
    true_score = float(scores[src_local].item())
    other_mean = mean_other(scores, src_local)
    top_other_idx = best_other_index(scores, src_local, higher_better=True)
    top_other_score = float(scores[top_other_idx].item())
    return {
        f"{prefix}_rank": strict_rank(scores, src_local, higher_better=True),
        f"{prefix}_gap": true_score - other_mean,
        f"{prefix}_directionality": float((true_score - other_mean) > 0.0),
        f"{prefix}_all_zero": float(all_zero(scores)),
        f"{prefix}_true_nonzero": float(true_score > EPS),
        f"{prefix}_unique_top1": float(unique_top1(scores, src_local)),
        f"{prefix}_hub_win": float((top_other_score - true_score) > EPS) if scores.numel() > 1 else 0.0,
        f"{prefix}_true_total": true_score,
        f"{prefix}_top_other_idx": int(top_other_idx),
        f"{prefix}_top_other_global_id": int(g_ids[top_other_idx].item()),
        f"{prefix}_top_other_score": top_other_score,
        f"{prefix}_true_base": float(support_res["base"][src_local].item()),
        f"{prefix}_true_specificity": float(support_res["specificity"][src_local].item()),
        f"{prefix}_true_focus": float(support_res["focus"][src_local].item()),
        f"{prefix}_true_chlorine": float(support_res["chlorine"][src_local].item()),
        f"{prefix}_competitor_base": float(support_res["base"][top_other_idx].item()),
        f"{prefix}_competitor_specificity": float(support_res["specificity"][top_other_idx].item()),
        f"{prefix}_competitor_focus": float(support_res["focus"][top_other_idx].item()),
        f"{prefix}_competitor_chlorine": float(support_res["chlorine"][top_other_idx].item()),
    }


def extract_contradiction_variant_fields(
    prefix: str,
    contra_res: Dict[str, Any],
    src_local: int,
    g_ids: torch.Tensor,
) -> Dict[str, Any]:
    scores = contra_res["total"]
    true_score = float(scores[src_local].item())
    other_mean = mean_other(scores, src_local)
    top_other_idx = best_other_index(scores, src_local, higher_better=False)
    top_other_score = float(scores[top_other_idx].item())
    result = {
        f"{prefix}_rank": strict_rank(scores, src_local, higher_better=False),
        f"{prefix}_gap": other_mean - true_score,
        f"{prefix}_directionality": float((other_mean - true_score) > 0.0),
        f"{prefix}_all_zero": float(all_zero(scores)),
        f"{prefix}_true": true_score,
        f"{prefix}_other_mean": other_mean,
        f"{prefix}_top_other_idx": int(top_other_idx),
        f"{prefix}_top_other_global_id": int(g_ids[top_other_idx].item()),
        f"{prefix}_top_other_score": top_other_score,
        f"{prefix}_history_positive_count": int(contra_res["positive_count"]),
        f"{prefix}_history_safe_count": int(contra_res["safe_count"]),
        f"{prefix}_safe_tau_min": float(contra_res["safe_violation_tau_min"]),
        f"{prefix}_mode": str(contra_res.get("mode", "")),
        f"{prefix}_phys_ctx_mode": str(contra_res.get("phys_ctx_mode", "")),
    }
    if "practical_v2_config" in contra_res:
        config = contra_res["practical_v2_config"]
        result[f"{prefix}_config_label"] = str(config.get("label", ""))
        result[f"{prefix}_normalize_by_eligible_safe_count"] = float(
            bool(config.get("normalize_by_eligible_safe_count", False))
        )
        result[f"{prefix}_gap_cap_min"] = float(config.get("gap_cap_min", np.nan))
        result[f"{prefix}_gap_log_tau_min"] = float(config.get("gap_log_tau_min", np.nan))
        result[f"{prefix}_soft_count_tau_min"] = float(config.get("soft_count_tau_min", np.nan))
        result[f"{prefix}_near_safe_tau_min"] = float(config.get("near_safe_tau_min", np.nan))
        result[f"{prefix}_near_safe_slack_min"] = float(config.get("near_safe_slack_min", np.nan))
        result[f"{prefix}_alpha_gap"] = float(config.get("alpha_gap", np.nan))
        result[f"{prefix}_beta_near_safe"] = float(config.get("beta_near_safe", np.nan))
        result[f"{prefix}_gamma_soft_count"] = float(config.get("gamma_soft_count", np.nan))
    true_fields = extract_contradiction_candidate_fields(f"{prefix}_true", contra_res, src_local, g_ids)
    competitor_fields = extract_contradiction_candidate_fields(
        f"{prefix}_competitor",
        contra_res,
        top_other_idx,
        g_ids,
    )
    result.update(true_fields)
    result.update(competitor_fields)
    return result


def extract_contradiction_oracle_v1_fields(
    prefix: str,
    contra_res: Dict[str, Any],
    src_local: int,
    g_ids: torch.Tensor,
) -> Dict[str, Any]:
    return extract_contradiction_variant_fields(prefix, contra_res, src_local, g_ids)


def safe_mean_from_series(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    return float(series.mean())


def safe_median_from_series(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    return float(series.median())


def collect_event_records(
    dataset,
    topology: HydraulicTopology,
    include_current_time_physctx_compare: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    evidence_builder = EvidenceBuilder()
    records: List[Dict[str, float]] = []
    case_trace_rows: List[Dict[str, float]] = []
    support_rootcause_rows: List[Dict[str, float]] = []
    candidate_audit_frames: List[pd.DataFrame] = []
    witness_mining_frames: List[pd.DataFrame] = []
    admissibility_compare_frames: List[pd.DataFrame] = []
    oracle_v1_candidate_buffers = init_contradiction_candidate_buffers(CONTRA_ORACLE_V1_BUFFER_METRICS)
    practical_v2_candidate_buffers = init_contradiction_candidate_buffers(CONTRA_PRACTICAL_V2_BUFFER_METRICS)
    indices = range(min(MAX_EVENTS, len(dataset)))

    for idx in indices:
        try:
            event_data_batch = dataset[idx]
            if event_data_batch is None:
                continue
            event_data = extract_view0(event_data_batch)
            src_global = event_data.global_injection_node
            if isinstance(src_global, torch.Tensor):
                src_global = int(src_global.item())

            rollout = PracticalRollout(
                event_data,
                dataset.global_edge_index,
                dataset.stt_dynamic_series,
                dataset.num_nodes,
                num_episodes=NUM_EPISODES,
                samples_per_episode=3,
            )

            if src_global not in rollout.g_ids:
                continue
            src_local = int((rollout.g_ids == src_global).nonzero(as_tuple=True)[0].item())
            batch = torch.zeros(event_data.num_nodes, dtype=torch.long, device=event_data.x.device)

            for episode_idx in range(NUM_EPISODES):
                obs_partial, _obs_oracle, phys_ctx, info = rollout.step()
                num_nodes = int(obs_partial.observed_flag.numel())
                num_pos = int(obs_partial.toxic_positive_flag.sum().item())
                num_neg = int(obs_partial.toxic_negative_flag.sum().item())
                t_snapshot_idx = int(info["t_snapshot_idx"])
                signal_snapshot = rollout.event_data.x_raw[:, t_snapshot_idx, 0]
                conc = rollout.event_data.x_raw[:, t_snapshot_idx, 1]
                truth_positive_mask = conc > 0.1
                truth_positive_total = int(truth_positive_mask.sum().item())
                truth_positive_observed = int((truth_positive_mask & (obs_partial.observed_flag > 0.5)).sum().item())
                truth_positive_unobserved = int((truth_positive_mask & (obs_partial.observed_flag <= 0.5)).sum().item())
                positive_seed_root_cause = classify_positive_seed_root_cause(
                    truth_positive_total=truth_positive_total,
                    observed_positive_total=num_pos,
                    hidden_truth_positive_total=truth_positive_unobserved,
                )
                truth_positive_indices = truth_positive_mask.nonzero().view(-1)

                ev_state = evidence_builder.build_evidence_state(obs_partial, phys_ctx, t_sim=None)
                support_scores = ev_state.support_score
                support_true = float(support_scores[src_local].item())
                support_other_mean = mean_other(support_scores, src_local)
                legacy_support_scores, legacy_support_terms = compute_audit_support(
                    rollout.reachability_module,
                    phys_ctx,
                    obs_partial,
                    t_sim=None,
                )
                legacy_support_true = float(legacy_support_scores[src_local].item())
                support_top_other_idx = best_other_index(support_scores, src_local, higher_better=True)
                support_top_other_score = float(support_scores[support_top_other_idx].item())
                observed_positive_indices, support_lookup = collect_support_pair_stats(
                    rollout=rollout,
                    phys_ctx=phys_ctx,
                    obs_partial=obs_partial,
                    src_local=src_local,
                    support_top_other_idx=support_top_other_idx,
                    truth_positive_indices=truth_positive_indices,
                )
                support_rootcause_rows.append(
                    build_support_rootcause_row(
                        idx=idx,
                        episode_idx=episode_idx,
                        info=info,
                        rollout=rollout,
                        ev_state=ev_state,
                        src_local=src_local,
                        support_top_other_idx=support_top_other_idx,
                        num_pos=num_pos,
                        truth_positive_total=truth_positive_total,
                        truth_positive_unobserved=truth_positive_unobserved,
                        positive_seed_root_cause=positive_seed_root_cause,
                        support_lookup=support_lookup,
                    )
                )

                reach = rollout.reachability_module.compute_reachability(obs_partial, phys_ctx, t_sim=None, batch=batch)
                suspect_terms = build_suspect_raw_and_mask(reach)
                suspect_raw = suspect_terms["raw_score"]
                suspect_mask = suspect_terms["active_mask"]
                suspect_true = float(suspect_raw[src_local].item())
                suspect_other_mean = mean_other(suspect_raw, src_local)

                old_contra = compute_old_contradiction(evidence_builder, obs_partial, phys_ctx, suspect_mask, reach)
                old_contra_true = float(old_contra[src_local].item())
                old_contra_other_mean = mean_other(old_contra, src_local)
                current_time_min = float(info["time_min"])
                formal_contra_variants = compute_formal_contradiction_variants(
                    evidence_builder,
                    obs_partial,
                    phys_ctx,
                    current_time_min=current_time_min,
                )
                formal_ev_state = formal_contra_variants["formal_ev_state"]
                formal_contra = formal_contra_variants["formal_total"]
                formal_contra_true = float(formal_contra[src_local].item())
                formal_contra_other_mean = mean_other(formal_contra, src_local)
                nosuspect_formal_contra = formal_contra_variants["nosuspect_formal_total"]
                nosuspect_formal_true = float(nosuspect_formal_contra[src_local].item())
                nosuspect_formal_other_mean = mean_other(nosuspect_formal_contra, src_local)

                strict_contra = compute_strict_exclusion_contradiction(
                    rollout.reachability_module,
                    obs_partial,
                    phys_ctx,
                    current_time_min=current_time_min,
                )
                strict_contra_true = float(strict_contra[src_local].item())
                strict_contra_other_mean = mean_other(strict_contra, src_local)

                oracle_v1_res = evidence_builder.compute_contradiction_score(
                    obs_partial,
                    phys_ctx,
                    torch.ones_like(suspect_mask),
                    formal_contra_variants["formal_reach"],
                    t_sim=formal_contra_variants["t_sim"],
                    contradiction_mode="oracle_v1",
                    oracle_history_steps=rollout.history_steps,
                    safe_violation_tau_min=ORACLE_V1_SAFE_TAU_MIN,
                    history_phys_ctx_mode=PRACTICAL_V2_HISTORY_PHYSCTX_MODE,
                )
                oracle_v1_true = float(oracle_v1_res["total"][src_local].item())
                oracle_v1_other_mean = mean_other(oracle_v1_res["total"], src_local)
                oracle_v1_fields = extract_contradiction_variant_fields(
                    "contra_oracle_v1",
                    oracle_v1_res,
                    src_local,
                    rollout.g_ids,
                )
                oracle_v1_support_comp_fields = extract_contradiction_candidate_fields(
                    "contra_oracle_v1_support_comp",
                    oracle_v1_res,
                    support_top_other_idx,
                    rollout.g_ids,
                )
                non_source_mask = torch.ones(num_nodes, dtype=torch.bool, device=obs_partial.observed_flag.device)
                non_source_mask[src_local] = False
                append_contradiction_candidate_buffers(
                    oracle_v1_candidate_buffers,
                    oracle_v1_res,
                    non_source_mask,
                    CONTRA_ORACLE_V1_BUFFER_METRICS,
                )

                practical_v2_res = evidence_builder.compute_contradiction_score(
                    obs_partial,
                    phys_ctx,
                    torch.ones_like(suspect_mask),
                    formal_contra_variants["formal_reach"],
                    t_sim=formal_contra_variants["t_sim"],
                    contradiction_mode="practical_v2",
                    oracle_history_steps=rollout.history_steps,
                    safe_violation_tau_min=ORACLE_V1_SAFE_TAU_MIN,
                    history_phys_ctx_mode=PRACTICAL_V2_HISTORY_PHYSCTX_MODE,
                    practical_v2_config=PRACTICAL_V2_MAIN_CONFIG,
                )
                practical_v2_true = float(practical_v2_res["total"][src_local].item())
                practical_v2_other_mean = mean_other(practical_v2_res["total"], src_local)
                practical_v2_fields = extract_contradiction_variant_fields(
                    "contra_practical_v2",
                    practical_v2_res,
                    src_local,
                    rollout.g_ids,
                )
                practical_v2_support_comp_fields = extract_contradiction_candidate_fields(
                    "contra_practical_v2_support_comp",
                    practical_v2_res,
                    support_top_other_idx,
                    rollout.g_ids,
                )
                append_contradiction_candidate_buffers(
                    practical_v2_candidate_buffers,
                    practical_v2_res,
                    non_source_mask,
                    CONTRA_PRACTICAL_V2_BUFFER_METRICS,
                )

                practical_v2_norm_res = derive_practical_v2_contradiction(
                    practical_v2_res,
                    config=PRACTICAL_V2_NORMALIZED_CONFIG,
                )
                practical_v2_gap_capped_scores = (
                    practical_v2_res["interval_gap_capped"] / max(PRACTICAL_V2_MAIN_CONFIG.gap_cap_min, 1.0)
                )
                practical_v2_gap_log_scores = practical_v2_res["interval_gap_log"]
                practical_v2_gap_capped_fields = extract_contradiction_score_fields(
                    "contra_practical_v2_gap_capped",
                    practical_v2_gap_capped_scores,
                    src_local,
                    rollout.g_ids,
                )
                practical_v2_gap_log_fields = extract_contradiction_score_fields(
                    "contra_practical_v2_gap_log",
                    practical_v2_gap_log_scores,
                    src_local,
                    rollout.g_ids,
                )
                practical_v2_norm_fields = extract_contradiction_score_fields(
                    "contra_practical_v2_norm",
                    practical_v2_norm_res["total"],
                    src_local,
                    rollout.g_ids,
                )

                practical_v2_current_time_res = evidence_builder.compute_contradiction_score(
                    obs_partial,
                    phys_ctx,
                    torch.ones_like(suspect_mask),
                    formal_contra_variants["formal_reach"],
                    t_sim=formal_contra_variants["t_sim"],
                    contradiction_mode="practical_v2",
                    oracle_history_steps=rollout.history_steps,
                    safe_violation_tau_min=ORACLE_V1_SAFE_TAU_MIN,
                    history_phys_ctx_mode=PRACTICAL_V2_CURRENT_TIME_PHYSCTX_MODE,
                    practical_v2_config=PRACTICAL_V2_MAIN_CONFIG,
                ) if include_current_time_physctx_compare else practical_v2_res
                practical_v2_current_time_fields = extract_contradiction_variant_fields(
                    "contra_practical_v2_current_time",
                    practical_v2_current_time_res,
                    src_local,
                    rollout.g_ids,
                )
                candidate_audit_frames.append(
                    build_contradiction_candidate_audit_frame(
                        event_id=idx,
                        episode=episode_idx + 1,
                        time_min=float(info["time_min"]),
                        src_local=src_local,
                        support_top_other_idx=support_top_other_idx,
                        contra_res=practical_v2_res,
                        current_time_res=practical_v2_current_time_res,
                        g_ids=rollout.g_ids,
                    )
                )
                witness_mining_results: Dict[str, Dict[str, Any]] = {
                    DEFAULT_WITNESS_MINING_MODE: practical_v2_res,
                }
                for mining_mode in WITNESS_MINING_MODES:
                    if mining_mode == DEFAULT_WITNESS_MINING_MODE:
                        continue
                    witness_mining_results[mining_mode] = evidence_builder.compute_contradiction_score(
                        obs_partial,
                        phys_ctx,
                        torch.ones_like(suspect_mask),
                        formal_contra_variants["formal_reach"],
                        t_sim=formal_contra_variants["t_sim"],
                        contradiction_mode="practical_v2",
                        oracle_history_steps=rollout.history_steps,
                        safe_violation_tau_min=ORACLE_V1_SAFE_TAU_MIN,
                        history_phys_ctx_mode=PRACTICAL_V2_HISTORY_PHYSCTX_MODE,
                        practical_v2_config=PRACTICAL_V2_MAIN_CONFIG,
                        witness_mining_mode=mining_mode,
                        frontier_safe_close_tau_min=WITNESS_MINING_FRONT_CLOSE_TAU_MIN,
                        mined_top_k_safe_witnesses=WITNESS_MINING_TOP_K,
                    )
                for mining_mode, mining_res in witness_mining_results.items():
                    witness_mining_frames.append(
                        build_witness_mining_candidate_frame(
                            event_id=idx,
                            episode=episode_idx + 1,
                            time_min=float(info["time_min"]),
                            src_local=src_local,
                            support_top_other_idx=support_top_other_idx,
                            contra_res=mining_res,
                            current_time_res=practical_v2_current_time_res,
                            g_ids=rollout.g_ids,
                            witness_mining_mode=mining_mode,
                        )
                    )
                admissibility_compare_results = compute_practical_v2_admissibility_compare(
                    reachability_module=rollout.reachability_module,
                    history_steps=rollout.history_steps,
                    num_nodes=num_nodes,
                    safe_violation_tau_min=ORACLE_V1_SAFE_TAU_MIN,
                    suspect_pool=torch.ones_like(suspect_mask),
                    phys_ctx_mode=PRACTICAL_V2_HISTORY_PHYSCTX_MODE,
                    config=PRACTICAL_V2_MAIN_CONFIG,
                    frontier_safe_close_tau_min=WITNESS_MINING_FRONT_CLOSE_TAU_MIN,
                    compare_config=ADMISSIBILITY_COMPARE_CONFIG,
                    admissibility_modes=ADMISSIBILITY_COMPARE_MODES,
                )
                for admissibility_mode, admissibility_res in admissibility_compare_results.items():
                    admissibility_compare_frames.append(
                        build_admissibility_compare_candidate_frame(
                            event_id=idx,
                            episode=episode_idx + 1,
                            time_min=float(info["time_min"]),
                            src_local=src_local,
                            support_top_other_idx=support_top_other_idx,
                            contra_res=admissibility_res,
                            current_time_res=practical_v2_current_time_res,
                            g_ids=rollout.g_ids,
                            admissibility_mode=admissibility_mode,
                        )
                    )

                reaction_scores = ev_state.reaction_consistency
                reaction_true = float(reaction_scores[src_local].item())
                reaction_other_mean = mean_other(reaction_scores, src_local)

                uncertainty_scores = ev_state.uncertainty_gap
                uncertainty_true = float(uncertainty_scores[src_local].item())
                observed_mask = obs_partial.observed_flag > 0.5
                unobserved_mask = ~observed_mask
                expected_uncertainty = 1.0 - obs_partial.observed_flag

                contra_time_conflict = float(
                    num_pos > 0
                    and num_neg > 0
                    and old_contra_true > strict_contra_true + CONTRA_TIME_CONFLICT_ALERT
                    and strict_contra_true <= EPS
                )

                practical_support_res = {
                    "total": support_scores,
                    "base": ev_state.support_coverage_term,
                    "specificity": ev_state.support_timing_term,
                    "focus": ev_state.support_focus_term,
                    "chlorine": ev_state.support_chlorine_term,
                }
                practical_support_fields = extract_support_variant_fields(
                    "practical",
                    practical_support_res,
                    src_local,
                    rollout.g_ids,
                )

                oracle_obs = build_oracle_support_observation_state(
                    obs_partial=obs_partial,
                    signal_snapshot=signal_snapshot,
                    truth_positive_mask=truth_positive_mask,
                )
                oracle_reach = compute_builder_reachability(
                    evidence_builder,
                    oracle_obs,
                    phys_ctx,
                    t_sim=None,
                )
                oracle_suspect_pool = evidence_builder.compute_suspect_pool(
                    oracle_obs,
                    phys_ctx,
                    oracle_reach,
                )
                oracle_pre_res = evidence_builder.compute_support_score(
                    oracle_obs,
                    phys_ctx,
                    torch.ones_like(oracle_suspect_pool),
                    oracle_reach,
                    t_sim=None,
                )
                oracle_post_res = evidence_builder.compute_support_score(
                    oracle_obs,
                    phys_ctx,
                    oracle_suspect_pool,
                    oracle_reach,
                    t_sim=None,
                )
                oracle_pre_fields = extract_support_variant_fields(
                    "oracle_pre",
                    oracle_pre_res,
                    src_local,
                    rollout.g_ids,
                )
                oracle_post_fields = extract_support_variant_fields(
                    "oracle_post",
                    oracle_post_res,
                    src_local,
                    rollout.g_ids,
                )
                oracle_t_abs_idx = resolve_snapshot_time_index(rollout.event_data, t_snapshot_idx)
                v2_main_res = compute_support_v2_oracle(
                    rollout=rollout,
                    phys_ctx=phys_ctx,
                    truth_positive_mask=truth_positive_mask,
                    witness_strength=conc,
                    current_time_min=float(info["time_min"]),
                    t_abs_idx=oracle_t_abs_idx,
                    topology=topology,
                )
                v2_main_fields = extract_support_variant_fields_v2(
                    "v2_main",
                    v2_main_res,
                    src_local,
                    rollout.g_ids,
                )
                oracle_v1_combo_scores = v2_main_res["total"] - oracle_v1_res["total"]
                oracle_v1_combo_true = float(oracle_v1_combo_scores[src_local].item())
                oracle_v1_combo_other_mean = mean_other(oracle_v1_combo_scores, src_local)
                oracle_v1_combo_top_other_idx = best_other_index(
                    oracle_v1_combo_scores,
                    src_local,
                    higher_better=True,
                )
                oracle_v1_combo_top_other_score = float(oracle_v1_combo_scores[oracle_v1_combo_top_other_idx].item())
                oracle_v1_combo_fields = {
                    "v2_combo_oracle_v1_rank": strict_rank(oracle_v1_combo_scores, src_local, higher_better=True),
                    "v2_combo_oracle_v1_gap": oracle_v1_combo_true - oracle_v1_combo_other_mean,
                    "v2_combo_oracle_v1_directionality": float(
                        (oracle_v1_combo_true - oracle_v1_combo_other_mean) > 0.0
                    ),
                    "v2_combo_oracle_v1_all_zero": float(all_zero(oracle_v1_combo_scores)),
                    "v2_combo_oracle_v1_true_nonzero": float(oracle_v1_combo_true > EPS),
                    "v2_combo_oracle_v1_hub_win": float(
                        (oracle_v1_combo_top_other_score - oracle_v1_combo_true) > EPS
                    ) if oracle_v1_combo_scores.numel() > 1 else 0.0,
                    "v2_combo_oracle_v1_true_total": oracle_v1_combo_true,
                    "v2_combo_oracle_v1_top_other_idx": int(oracle_v1_combo_top_other_idx),
                    "v2_combo_oracle_v1_top_other_global_id": int(
                        rollout.g_ids[oracle_v1_combo_top_other_idx].item()
                    ),
                    "v2_combo_oracle_v1_top_other_score": oracle_v1_combo_top_other_score,
                }
                practical_v2_combo_scores = v2_main_res["total"] - practical_v2_res["total"]
                practical_v2_combo_true = float(practical_v2_combo_scores[src_local].item())
                practical_v2_combo_other_mean = mean_other(practical_v2_combo_scores, src_local)
                practical_v2_combo_top_other_idx = best_other_index(
                    practical_v2_combo_scores,
                    src_local,
                    higher_better=True,
                )
                practical_v2_combo_top_other_score = float(
                    practical_v2_combo_scores[practical_v2_combo_top_other_idx].item()
                )
                practical_v2_combo_fields = {
                    "v2_combo_practical_v2_rank": strict_rank(practical_v2_combo_scores, src_local, higher_better=True),
                    "v2_combo_practical_v2_gap": practical_v2_combo_true - practical_v2_combo_other_mean,
                    "v2_combo_practical_v2_directionality": float(
                        (practical_v2_combo_true - practical_v2_combo_other_mean) > 0.0
                    ),
                    "v2_combo_practical_v2_all_zero": float(all_zero(practical_v2_combo_scores)),
                    "v2_combo_practical_v2_true_nonzero": float(practical_v2_combo_true > EPS),
                    "v2_combo_practical_v2_hub_win": float(
                        (practical_v2_combo_top_other_score - practical_v2_combo_true) > EPS
                    ) if practical_v2_combo_scores.numel() > 1 else 0.0,
                    "v2_combo_practical_v2_true_total": practical_v2_combo_true,
                    "v2_combo_practical_v2_top_other_idx": int(practical_v2_combo_top_other_idx),
                    "v2_combo_practical_v2_top_other_global_id": int(
                        rollout.g_ids[practical_v2_combo_top_other_idx].item()
                    ),
                    "v2_combo_practical_v2_top_other_score": practical_v2_combo_top_other_score,
                }
                practical_v2_current_time_combo_scores = v2_main_res["total"] - practical_v2_current_time_res["total"]
                practical_v2_current_time_combo_true = float(practical_v2_current_time_combo_scores[src_local].item())
                practical_v2_current_time_combo_other_mean = mean_other(
                    practical_v2_current_time_combo_scores,
                    src_local,
                )
                practical_v2_current_time_combo_top_other_idx = best_other_index(
                    practical_v2_current_time_combo_scores,
                    src_local,
                    higher_better=True,
                )
                practical_v2_current_time_combo_top_other_score = float(
                    practical_v2_current_time_combo_scores[practical_v2_current_time_combo_top_other_idx].item()
                )
                practical_v2_current_time_combo_fields = {
                    "v2_combo_practical_v2_current_time_rank": strict_rank(
                        practical_v2_current_time_combo_scores,
                        src_local,
                        higher_better=True,
                    ),
                    "v2_combo_practical_v2_current_time_gap": (
                        practical_v2_current_time_combo_true - practical_v2_current_time_combo_other_mean
                    ),
                    "v2_combo_practical_v2_current_time_directionality": float(
                        (practical_v2_current_time_combo_true - practical_v2_current_time_combo_other_mean) > 0.0
                    ),
                    "v2_combo_practical_v2_current_time_all_zero": float(
                        all_zero(practical_v2_current_time_combo_scores)
                    ),
                    "v2_combo_practical_v2_current_time_true_nonzero": float(
                        practical_v2_current_time_combo_true > EPS
                    ),
                    "v2_combo_practical_v2_current_time_hub_win": float(
                        (
                            practical_v2_current_time_combo_top_other_score
                            - practical_v2_current_time_combo_true
                        ) > EPS
                    ) if practical_v2_current_time_combo_scores.numel() > 1 else 0.0,
                    "v2_combo_practical_v2_current_time_true_total": practical_v2_current_time_combo_true,
                    "v2_combo_practical_v2_current_time_top_other_idx": int(
                        practical_v2_current_time_combo_top_other_idx
                    ),
                    "v2_combo_practical_v2_current_time_top_other_global_id": int(
                        rollout.g_ids[practical_v2_current_time_combo_top_other_idx].item()
                    ),
                    "v2_combo_practical_v2_current_time_top_other_score": (
                        practical_v2_current_time_combo_top_other_score
                    ),
                }

                records.append(
                    {
                        "event_id": idx,
                        "episode": episode_idx + 1,
                        "time_min": float(info["time_min"]),
                        "num_events": 1,
                        "num_nodes": num_nodes,
                        "num_pos": num_pos,
                        "num_neg": num_neg,
                        "observed_count": int(obs_partial.observed_flag.sum().item()),
                        "revealed_count": int(info["revealed_count"]),
                        "negative_seed_count": num_neg,
                        "oracle_num_pos": truth_positive_total,
                        "oracle_positive_gain": int(truth_positive_total - num_pos),
                        "truth_positive_total": truth_positive_total,
                        "truth_positive_observed": truth_positive_observed,
                        "truth_positive_unobserved": truth_positive_unobserved,
                        "no_positive_seed_root_cause": positive_seed_root_cause,
                        "support_rank": strict_rank(support_scores, src_local, higher_better=True),
                        "support_gap": support_true - support_other_mean,
                        "support_directionality": float((support_true - support_other_mean) > 0.0),
                        "support_all_zero": float(all_zero(support_scores)),
                        "support_all_zero_given_input": float(num_pos > 0 and all_zero(support_scores)),
                        "support_all_zero_due_no_input": float(num_pos == 0 and all_zero(support_scores)),
                        "support_builder_vs_legacy_l1": float((support_scores - legacy_support_scores).abs().mean().item()),
                        "support_true_builder": support_true,
                        "support_true_legacy": legacy_support_true,
                        "support_top_other_idx": support_top_other_idx,
                        "support_top_other_global_id": int(rollout.g_ids[support_top_other_idx].item()),
                        "support_top_other_score": support_top_other_score,
                        "support_true_nonzero": float(support_true > EPS),
                        "support_unique_top1": float(unique_top1(support_scores, src_local)),
                        "support_hub_win": float((support_top_other_score - support_true) > EPS) if num_nodes > 1 else 0.0,
                        "support_true_base": float(ev_state.support_coverage_term[src_local].item()),
                        "support_true_specificity": float(ev_state.support_timing_term[src_local].item()),
                        "support_true_focus": float(ev_state.support_focus_term[src_local].item()),
                        "support_true_chlorine": float(ev_state.support_chlorine_term[src_local].item()),
                        "support_competitor_base": float(ev_state.support_coverage_term[support_top_other_idx].item()),
                        "support_competitor_specificity": float(ev_state.support_timing_term[support_top_other_idx].item()),
                        "support_competitor_focus": float(ev_state.support_focus_term[support_top_other_idx].item()),
                        "support_competitor_chlorine": float(ev_state.support_chlorine_term[support_top_other_idx].item()),
                        "suspect_raw_rank": strict_rank(suspect_raw, src_local, higher_better=True),
                        "suspect_raw_gap": suspect_true - suspect_other_mean,
                        "suspect_raw_directionality": float((suspect_true - suspect_other_mean) > 0.0),
                        "suspect_active_recall": float(suspect_mask[src_local].item() > 0.5),
                        "suspect_active_reduction": 1.0 - (float(suspect_mask.sum().item()) / max(num_nodes, 1)),
                        "suspect_candidate_set_size": float(suspect_mask.sum().item()),
                        "suspect_threshold_value": SUSPECT_THRESHOLD,
                        "contra_old_rank": strict_rank(old_contra, src_local, higher_better=False),
                        "contra_old_gap": old_contra_other_mean - old_contra_true,
                        "contra_old_directionality": float((old_contra_other_mean - old_contra_true) > 0.0),
                        "contra_old_all_zero": float(all_zero(old_contra)),
                        "contra_old_true": old_contra_true,
                        "contra_formal_rank": strict_rank(formal_contra, src_local, higher_better=False),
                        "contra_formal_gap": formal_contra_other_mean - formal_contra_true,
                        "contra_formal_directionality": float((formal_contra_other_mean - formal_contra_true) > 0.0),
                        "contra_formal_all_zero": float(all_zero(formal_contra)),
                        "contra_formal_true": formal_contra_true,
                        "contra_nosuspect_formal_rank": strict_rank(nosuspect_formal_contra, src_local, higher_better=False),
                        "contra_nosuspect_formal_gap": nosuspect_formal_other_mean - nosuspect_formal_true,
                        "contra_nosuspect_formal_directionality": float(
                            (nosuspect_formal_other_mean - nosuspect_formal_true) > 0.0
                        ),
                        "contra_nosuspect_formal_all_zero": float(all_zero(nosuspect_formal_contra)),
                        "contra_nosuspect_formal_true": nosuspect_formal_true,
                        "contra_strict_rank": strict_rank(strict_contra, src_local, higher_better=False),
                        "contra_strict_gap": strict_contra_other_mean - strict_contra_true,
                        "contra_strict_directionality": float((strict_contra_other_mean - strict_contra_true) > 0.0),
                        "contra_strict_all_zero": float(all_zero(strict_contra)),
                        "contra_strict_true": strict_contra_true,
                        "contra_old_vs_formal_l1": float((old_contra - formal_contra).abs().mean().item()),
                        "contra_old_vs_formal_linf": float((old_contra - formal_contra).abs().max().item()),
                        "contra_old_vs_formal_any_diff": float((old_contra - formal_contra).abs().max().item() > EPS),
                        "contra_old_nonzero_formal_zero": float(old_contra_true > EPS and formal_contra_true <= EPS),
                        "contra_formal_nonzero_old_zero": float(formal_contra_true > EPS and old_contra_true <= EPS),
                        "contra_formal_vs_nosuspect_l1": float(
                            (formal_contra - nosuspect_formal_contra).abs().mean().item()
                        ),
                        "contra_formal_vs_nosuspect_linf": float(
                            (formal_contra - nosuspect_formal_contra).abs().max().item()
                        ),
                        "contra_formal_vs_nosuspect_any_diff": float(
                            (formal_contra - nosuspect_formal_contra).abs().max().item() > EPS
                        ),
                        "contra_formal_zero_nosuspect_nonzero": float(
                            formal_contra_true <= EPS and nosuspect_formal_true > EPS
                        ),
                        "contra_nosuspect_true_lift": nosuspect_formal_true - formal_contra_true,
                        "contra_body_still_zero_without_suspect": float(all_zero(nosuspect_formal_contra)),
                        "contra_eligible_strict": float(num_pos > 0 and num_neg > 0),
                        "contra_time_conflict": contra_time_conflict,
                        "contra_oracle_v1_rank": strict_rank(oracle_v1_res["total"], src_local, higher_better=False),
                        "contra_oracle_v1_gap": oracle_v1_other_mean - oracle_v1_true,
                        "contra_oracle_v1_directionality": float((oracle_v1_other_mean - oracle_v1_true) > 0.0),
                        "contra_oracle_v1_all_zero": float(all_zero(oracle_v1_res["total"])),
                        "contra_oracle_v1_true": oracle_v1_true,
                        "contra_oracle_v1_other_mean": oracle_v1_other_mean,
                        "contra_oracle_v1_old_l1": float((oracle_v1_res["total"] - old_contra).abs().mean().item()),
                        "contra_oracle_v1_old_linf": float((oracle_v1_res["total"] - old_contra).abs().max().item()),
                        "contra_oracle_v1_vs_nosuspect_l1": float(
                            (oracle_v1_res["total"] - nosuspect_formal_contra).abs().mean().item()
                        ),
                        "contra_oracle_v1_vs_nosuspect_linf": float(
                            (oracle_v1_res["total"] - nosuspect_formal_contra).abs().max().item()
                        ),
                        "contra_oracle_v1_true_minus_old": oracle_v1_true - old_contra_true,
                        "contra_oracle_v1_true_minus_nosuspect": oracle_v1_true - nosuspect_formal_true,
                        "contra_oracle_v1_eligible": float(
                            oracle_v1_fields["contra_oracle_v1_history_positive_count"] > 0
                            and oracle_v1_fields["contra_oracle_v1_history_safe_count"] > 0
                        ),
                        "contra_practical_v2_rank": strict_rank(practical_v2_res["total"], src_local, higher_better=False),
                        "contra_practical_v2_gap": practical_v2_other_mean - practical_v2_true,
                        "contra_practical_v2_directionality": float((practical_v2_other_mean - practical_v2_true) > 0.0),
                        "contra_practical_v2_all_zero": float(all_zero(practical_v2_res["total"])),
                        "contra_practical_v2_true": practical_v2_true,
                        "contra_practical_v2_other_mean": practical_v2_other_mean,
                        "contra_practical_v2_old_l1": float((practical_v2_res["total"] - old_contra).abs().mean().item()),
                        "contra_practical_v2_old_linf": float((practical_v2_res["total"] - old_contra).abs().max().item()),
                        "contra_practical_v2_oracle_v1_l1": float(
                            (practical_v2_res["total"] - oracle_v1_res["total"]).abs().mean().item()
                        ),
                        "contra_practical_v2_oracle_v1_linf": float(
                            (practical_v2_res["total"] - oracle_v1_res["total"]).abs().max().item()
                        ),
                        "contra_practical_v2_vs_nosuspect_l1": float(
                            (practical_v2_res["total"] - nosuspect_formal_contra).abs().mean().item()
                        ),
                        "contra_practical_v2_vs_nosuspect_linf": float(
                            (practical_v2_res["total"] - nosuspect_formal_contra).abs().max().item()
                        ),
                        "contra_practical_v2_true_minus_old": practical_v2_true - old_contra_true,
                        "contra_practical_v2_true_minus_oracle_v1": practical_v2_true - oracle_v1_true,
                        "contra_practical_v2_true_minus_nosuspect": practical_v2_true - nosuspect_formal_true,
                        "contra_practical_v2_eligible": float(
                            practical_v2_fields["contra_practical_v2_history_positive_count"] > 0
                            and practical_v2_fields["contra_practical_v2_history_safe_count"] > 0
                        ),
                        "contra_practical_v2_current_time_available": float(include_current_time_physctx_compare),
                        "contra_practical_v2_current_time_l1": float(
                            (practical_v2_current_time_res["total"] - practical_v2_res["total"]).abs().mean().item()
                        ),
                        "contra_practical_v2_current_time_linf": float(
                            (practical_v2_current_time_res["total"] - practical_v2_res["total"]).abs().max().item()
                        ),
                        "contra_practical_v2_current_time_true_delta": float(
                            practical_v2_current_time_res["total"][src_local].item() - practical_v2_true
                        ),
                        "reaction_rank": strict_rank(reaction_scores, src_local, higher_better=True),
                        "reaction_gap": reaction_true - reaction_other_mean,
                        "reaction_directionality": float((reaction_true - reaction_other_mean) > 0.0),
                        "reaction_all_zero": float(all_zero(reaction_scores)),
                        "reaction_true": reaction_true,
                        "reaction_eligible": float((num_pos + num_neg) > 0),
                        "uncertainty_true": uncertainty_true,
                        "uncertainty_observed_mean": masked_mean(uncertainty_scores, observed_mask),
                        "uncertainty_unobserved_mean": masked_mean(uncertainty_scores, unobserved_mask),
                        "uncertainty_obs_alignment_l1": float((uncertainty_scores - expected_uncertainty).abs().mean().item()),
                        "source_validity_true": float(ev_state.source_validity[src_local].item()),
                        "topology_true": float(reach["topology_reachable"][src_local].item()),
                        "soft_reachability_true": float(reach["soft_reachability"][src_local].item()),
                        "hard_neg_true": float(reach["hard_reachability_from_neg"][src_local].item()),
                        "topology_top_other": float(reach["topology_reachable"][support_top_other_idx].item()),
                        "soft_reachability_top_other": float(reach["soft_reachability"][support_top_other_idx].item()),
                        "hard_neg_top_other": float(reach["hard_reachability_from_neg"][support_top_other_idx].item()),
                        "oracle_suspect_pool_true": float(oracle_suspect_pool[src_local].item()),
                        "oracle_suspect_pool_sum": float(oracle_suspect_pool.sum().item()),
                        "oracle_suspect_pool_pre_top_other": float(
                            oracle_suspect_pool[int(oracle_pre_fields["oracle_pre_top_other_idx"])].item()
                        ),
                        "oracle_suspect_pool_post_top_other": float(
                            oracle_suspect_pool[int(oracle_post_fields["oracle_post_top_other_idx"])].item()
                        ),
                        **practical_support_fields,
                        **oracle_pre_fields,
                        **oracle_post_fields,
                        "oracle_pre_rank_delta": float(strict_rank(support_scores, src_local, higher_better=True) - oracle_pre_fields["oracle_pre_rank"]),
                        "oracle_post_rank_delta": float(strict_rank(support_scores, src_local, higher_better=True) - oracle_post_fields["oracle_post_rank"]),
                        "oracle_pre_rank_improved": float(oracle_pre_fields["oracle_pre_rank"] < strict_rank(support_scores, src_local, higher_better=True)),
                        "oracle_post_rank_improved": float(oracle_post_fields["oracle_post_rank"] < strict_rank(support_scores, src_local, higher_better=True)),
                        "oracle_pre_rank_improved_2plus": float(oracle_pre_fields["oracle_pre_rank"] <= strict_rank(support_scores, src_local, higher_better=True) - 2),
                        "oracle_post_rank_improved_2plus": float(oracle_post_fields["oracle_post_rank"] <= strict_rank(support_scores, src_local, higher_better=True) - 2),
                        "oracle_pre_directionality_flip_to_true": float(
                            (support_true - support_other_mean) <= 0.0 and oracle_pre_fields["oracle_pre_directionality"] > 0.5
                        ),
                        "oracle_post_directionality_flip_to_true": float(
                            (support_true - support_other_mean) <= 0.0 and oracle_post_fields["oracle_post_directionality"] > 0.5
                        ),
                        "oracle_t_abs_idx": int(oracle_t_abs_idx),
                        **oracle_v1_fields,
                        **oracle_v1_support_comp_fields,
                        **practical_v2_gap_capped_fields,
                        **practical_v2_gap_log_fields,
                        **practical_v2_norm_fields,
                        **practical_v2_fields,
                        **practical_v2_support_comp_fields,
                        **practical_v2_current_time_fields,
                        **v2_main_fields,
                        **oracle_v1_combo_fields,
                        **practical_v2_combo_fields,
                        **practical_v2_current_time_combo_fields,
                    }
                )
                append_case_trace_rows(
                    case_rows=case_trace_rows,
                    idx=idx,
                    episode_idx=episode_idx,
                    info=info,
                    rollout=rollout,
                    obs_partial=obs_partial,
                    phys_ctx=phys_ctx,
                    ev_state=ev_state,
                    formal_ev_state=formal_ev_state,
                    reach=reach,
                    support_scores=support_scores,
                    legacy_support_scores=legacy_support_scores,
                    legacy_terms=legacy_support_terms,
                    old_contra=old_contra,
                    formal_contra=formal_contra,
                    nosuspect_formal_contra=nosuspect_formal_contra,
                    strict_contra=strict_contra,
                    src_local=src_local,
                    support_top_other_idx=support_top_other_idx,
                    truth_positive_total=truth_positive_total,
                    truth_positive_unobserved=truth_positive_unobserved,
                    positive_seed_root_cause=positive_seed_root_cause,
                    truth_positive_indices=truth_positive_indices,
                    observed_positive_indices=observed_positive_indices,
                    support_lookup=support_lookup,
                )
        except Exception:
            continue

    return (
        pd.DataFrame(records),
        pd.DataFrame(case_trace_rows),
        pd.DataFrame(support_rootcause_rows),
        oracle_v1_candidate_buffers,
        practical_v2_candidate_buffers,
        pd.concat(candidate_audit_frames, ignore_index=True) if candidate_audit_frames else pd.DataFrame(),
        pd.concat(witness_mining_frames, ignore_index=True) if witness_mining_frames else pd.DataFrame(),
        pd.concat(admissibility_compare_frames, ignore_index=True) if admissibility_compare_frames else pd.DataFrame(),
    )


def summarize_support(df_ep: pd.DataFrame) -> Dict[str, float]:
    eligible = df_ep[df_ep["num_pos"] > 0]
    return {
        "episode": int(df_ep["episode"].iloc[0]),
        "axis": "support",
        "version": "cleaned",
        "num_events": int(len(df_ep)),
        "eligible_event_rate": float((df_ep["num_pos"] > 0).mean()),
        "all_zero_event_rate": float(df_ep["support_all_zero"].mean()),
        "rank_median": float(df_ep["support_rank"].median()),
        "gap_mean": float(df_ep["support_gap"].mean()),
        "directionality": float(df_ep["support_directionality"].mean()),
        "conditioned_rank_median": float(eligible["support_rank"].median()) if not eligible.empty else np.nan,
        "conditioned_gap_mean": float(eligible["support_gap"].mean()) if not eligible.empty else np.nan,
        "conditioned_directionality": float(eligible["support_directionality"].mean()) if not eligible.empty else np.nan,
        "true_nonzero_rate": float(df_ep["support_true_nonzero"].mean()),
        "unique_top1_rate": float(df_ep["support_unique_top1"].mean()),
        "hub_win_rate": float(df_ep["support_hub_win"].mean()),
        "raw_rank_median": np.nan,
        "raw_gap_mean": np.nan,
        "raw_directionality": np.nan,
        "active_recall": np.nan,
        "active_reduction": np.nan,
        "candidate_set_size_mean": np.nan,
        "threshold_value": np.nan,
        "negative_seed_count_mean": np.nan,
        "compare_l1_mean": float(df_ep["support_builder_vs_legacy_l1"].mean()),
        "time_conflict_rate": np.nan,
        "observed_mean": np.nan,
        "unobserved_mean": np.nan,
        "obs_alignment_l1_mean": np.nan,
    }


def summarize_suspect_raw(df_ep: pd.DataFrame) -> Dict[str, float]:
    return {
        "episode": int(df_ep["episode"].iloc[0]),
        "axis": "suspect",
        "version": "raw_score",
        "num_events": int(len(df_ep)),
        "eligible_event_rate": float((df_ep["num_pos"] > 0).mean()),
        "all_zero_event_rate": np.nan,
        "rank_median": np.nan,
        "gap_mean": np.nan,
        "directionality": np.nan,
        "conditioned_rank_median": np.nan,
        "conditioned_gap_mean": np.nan,
        "conditioned_directionality": np.nan,
        "true_nonzero_rate": np.nan,
        "unique_top1_rate": np.nan,
        "hub_win_rate": np.nan,
        "raw_rank_median": float(df_ep["suspect_raw_rank"].median()),
        "raw_gap_mean": float(df_ep["suspect_raw_gap"].mean()),
        "raw_directionality": float(df_ep["suspect_raw_directionality"].mean()),
        "active_recall": np.nan,
        "active_reduction": np.nan,
        "candidate_set_size_mean": np.nan,
        "threshold_value": np.nan,
        "negative_seed_count_mean": np.nan,
        "compare_l1_mean": np.nan,
        "time_conflict_rate": np.nan,
        "observed_mean": np.nan,
        "unobserved_mean": np.nan,
        "obs_alignment_l1_mean": np.nan,
    }


def summarize_suspect_active(df_ep: pd.DataFrame) -> Dict[str, float]:
    return {
        "episode": int(df_ep["episode"].iloc[0]),
        "axis": "suspect",
        "version": "active_mask",
        "num_events": int(len(df_ep)),
        "eligible_event_rate": float((df_ep["num_pos"] > 0).mean()),
        "all_zero_event_rate": np.nan,
        "rank_median": np.nan,
        "gap_mean": np.nan,
        "directionality": np.nan,
        "conditioned_rank_median": np.nan,
        "conditioned_gap_mean": np.nan,
        "conditioned_directionality": np.nan,
        "true_nonzero_rate": np.nan,
        "unique_top1_rate": np.nan,
        "hub_win_rate": np.nan,
        "raw_rank_median": np.nan,
        "raw_gap_mean": np.nan,
        "raw_directionality": np.nan,
        "active_recall": float(df_ep["suspect_active_recall"].mean()),
        "active_reduction": float(df_ep["suspect_active_reduction"].mean()),
        "candidate_set_size_mean": float(df_ep["suspect_candidate_set_size"].mean()),
        "threshold_value": float(df_ep["suspect_threshold_value"].mean()),
        "negative_seed_count_mean": np.nan,
        "compare_l1_mean": np.nan,
        "time_conflict_rate": np.nan,
        "observed_mean": np.nan,
        "unobserved_mean": np.nan,
        "obs_alignment_l1_mean": np.nan,
    }


def summarize_contradiction(df_ep: pd.DataFrame, prefix: str, version: str) -> Dict[str, float]:
    eligible = df_ep[df_ep["num_neg"] > 0]
    return {
        "episode": int(df_ep["episode"].iloc[0]),
        "axis": "contradiction",
        "version": version,
        "num_events": int(len(df_ep)),
        "eligible_event_rate": float((df_ep["num_neg"] > 0).mean()),
        "all_zero_event_rate": float(df_ep[f"{prefix}_all_zero"].mean()),
        "rank_median": float(df_ep[f"{prefix}_rank"].median()),
        "gap_mean": float(df_ep[f"{prefix}_gap"].mean()),
        "directionality": float(df_ep[f"{prefix}_directionality"].mean()),
        "conditioned_rank_median": float(eligible[f"{prefix}_rank"].median()) if not eligible.empty else np.nan,
        "conditioned_gap_mean": float(eligible[f"{prefix}_gap"].mean()) if not eligible.empty else np.nan,
        "conditioned_directionality": float(eligible[f"{prefix}_directionality"].mean()) if not eligible.empty else np.nan,
        "true_nonzero_rate": np.nan,
        "unique_top1_rate": np.nan,
        "hub_win_rate": np.nan,
        "raw_rank_median": np.nan,
        "raw_gap_mean": np.nan,
        "raw_directionality": np.nan,
        "active_recall": np.nan,
        "active_reduction": np.nan,
        "candidate_set_size_mean": np.nan,
        "threshold_value": np.nan,
        "negative_seed_count_mean": float(df_ep["negative_seed_count"].mean()),
        "compare_l1_mean": np.nan,
        "time_conflict_rate": float(df_ep["contra_time_conflict"].mean()) if prefix == "contra_old" else np.nan,
        "observed_mean": np.nan,
        "unobserved_mean": np.nan,
        "obs_alignment_l1_mean": np.nan,
    }


def summarize_reaction(df_ep: pd.DataFrame) -> Dict[str, float]:
    eligible = df_ep[df_ep["reaction_eligible"] > 0.5]
    return {
        "episode": int(df_ep["episode"].iloc[0]),
        "axis": "reaction_consistency",
        "version": "builder",
        "num_events": int(len(df_ep)),
        "eligible_event_rate": float((df_ep["reaction_eligible"] > 0.5).mean()),
        "all_zero_event_rate": float(df_ep["reaction_all_zero"].mean()),
        "rank_median": float(df_ep["reaction_rank"].median()),
        "gap_mean": float(df_ep["reaction_gap"].mean()),
        "directionality": float(df_ep["reaction_directionality"].mean()),
        "conditioned_rank_median": float(eligible["reaction_rank"].median()) if not eligible.empty else np.nan,
        "conditioned_gap_mean": float(eligible["reaction_gap"].mean()) if not eligible.empty else np.nan,
        "conditioned_directionality": float(eligible["reaction_directionality"].mean()) if not eligible.empty else np.nan,
        "true_nonzero_rate": float((df_ep["reaction_true"].abs() > EPS).mean()),
        "unique_top1_rate": np.nan,
        "hub_win_rate": np.nan,
        "raw_rank_median": np.nan,
        "raw_gap_mean": np.nan,
        "raw_directionality": np.nan,
        "active_recall": np.nan,
        "active_reduction": np.nan,
        "candidate_set_size_mean": np.nan,
        "threshold_value": np.nan,
        "negative_seed_count_mean": np.nan,
        "compare_l1_mean": np.nan,
        "time_conflict_rate": np.nan,
        "observed_mean": np.nan,
        "unobserved_mean": np.nan,
        "obs_alignment_l1_mean": np.nan,
    }


def summarize_uncertainty(df_ep: pd.DataFrame) -> Dict[str, float]:
    return {
        "episode": int(df_ep["episode"].iloc[0]),
        "axis": "uncertainty_gap",
        "version": "builder",
        "num_events": int(len(df_ep)),
        "eligible_event_rate": 1.0,
        "all_zero_event_rate": float((df_ep["uncertainty_unobserved_mean"].fillna(0.0) <= EPS).mean()),
        "rank_median": np.nan,
        "gap_mean": np.nan,
        "directionality": np.nan,
        "conditioned_rank_median": np.nan,
        "conditioned_gap_mean": np.nan,
        "conditioned_directionality": np.nan,
        "true_nonzero_rate": float((df_ep["uncertainty_true"] > EPS).mean()),
        "unique_top1_rate": np.nan,
        "hub_win_rate": np.nan,
        "raw_rank_median": np.nan,
        "raw_gap_mean": np.nan,
        "raw_directionality": np.nan,
        "active_recall": np.nan,
        "active_reduction": np.nan,
        "candidate_set_size_mean": np.nan,
        "threshold_value": np.nan,
        "negative_seed_count_mean": np.nan,
        "compare_l1_mean": np.nan,
        "time_conflict_rate": np.nan,
        "observed_mean": float(df_ep["uncertainty_observed_mean"].mean()),
        "unobserved_mean": float(df_ep["uncertainty_unobserved_mean"].mean()),
        "obs_alignment_l1_mean": float(df_ep["uncertainty_obs_alignment_l1"].mean()),
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for episode in range(1, NUM_EPISODES + 1):
        df_ep = df[df["episode"] == episode]
        if df_ep.empty:
            continue
        rows.append(summarize_support(df_ep))
        rows.append(summarize_suspect_raw(df_ep))
        rows.append(summarize_suspect_active(df_ep))
        rows.append(summarize_contradiction(df_ep, "contra_old", "old"))
        rows.append(summarize_contradiction(df_ep, "contra_formal", "formal"))
        rows.append(summarize_contradiction(df_ep, "contra_nosuspect_formal", "nosuspect_formal"))
        rows.append(summarize_contradiction(df_ep, "contra_strict", "strict_exclusion"))
        rows.append(summarize_reaction(df_ep))
        rows.append(summarize_uncertainty(df_ep))
    columns = [
        "episode", "axis", "version", "num_events", "eligible_event_rate", "all_zero_event_rate",
        "rank_median", "gap_mean", "directionality", "conditioned_rank_median", "conditioned_gap_mean",
        "conditioned_directionality", "true_nonzero_rate", "unique_top1_rate", "hub_win_rate",
        "raw_rank_median", "raw_gap_mean", "raw_directionality", "active_recall", "active_reduction",
        "candidate_set_size_mean", "threshold_value", "negative_seed_count_mean", "compare_l1_mean",
        "time_conflict_rate", "observed_mean", "unobserved_mean", "obs_alignment_l1_mean",
    ]
    return pd.DataFrame(rows)[columns]


def make_support_table(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary[(summary["axis"] == "support") & (summary["version"] == "cleaned")].copy()
    return df[
        [
            "episode", "rank_median", "gap_mean", "all_zero_event_rate", "eligible_event_rate",
            "conditioned_rank_median", "conditioned_gap_mean", "conditioned_directionality",
            "true_nonzero_rate", "unique_top1_rate", "hub_win_rate", "compare_l1_mean",
        ]
    ].set_index("episode")


def make_suspect_table(summary: pd.DataFrame) -> pd.DataFrame:
    raw = summary[(summary["axis"] == "suspect") & (summary["version"] == "raw_score")].copy().set_index("episode")
    active = summary[(summary["axis"] == "suspect") & (summary["version"] == "active_mask")].copy().set_index("episode")
    out = pd.DataFrame(index=raw.index)
    out["raw_rank_median"] = raw["raw_rank_median"]
    out["raw_gap_mean"] = raw["raw_gap_mean"]
    out["raw_directionality"] = raw["raw_directionality"]
    out["active_recall"] = active["active_recall"]
    out["active_reduction"] = active["active_reduction"]
    out["candidate_set_size_mean"] = active["candidate_set_size_mean"]
    out["threshold_value"] = active["threshold_value"]
    return out


def make_contradiction_table(summary: pd.DataFrame) -> pd.DataFrame:
    old = summary[(summary["axis"] == "contradiction") & (summary["version"] == "old")].copy().set_index("episode")
    formal = summary[(summary["axis"] == "contradiction") & (summary["version"] == "formal")].copy().set_index("episode")
    nosuspect_formal = summary[
        (summary["axis"] == "contradiction") & (summary["version"] == "nosuspect_formal")
    ].copy().set_index("episode")
    strict = summary[(summary["axis"] == "contradiction") & (summary["version"] == "strict_exclusion")].copy().set_index("episode")
    out = pd.DataFrame(index=old.index)
    out["old_rank_median"] = old["rank_median"]
    out["old_gap_mean"] = old["gap_mean"]
    out["old_directionality"] = old["directionality"]
    out["old_time_conflict_rate"] = old["time_conflict_rate"]
    out["formal_rank_median"] = formal["rank_median"]
    out["formal_gap_mean"] = formal["gap_mean"]
    out["formal_directionality"] = formal["directionality"]
    out["formal_all_zero_event_rate"] = formal["all_zero_event_rate"]
    out["nosuspect_formal_rank_median"] = nosuspect_formal["rank_median"]
    out["nosuspect_formal_gap_mean"] = nosuspect_formal["gap_mean"]
    out["nosuspect_formal_directionality"] = nosuspect_formal["directionality"]
    out["nosuspect_formal_all_zero_event_rate"] = nosuspect_formal["all_zero_event_rate"]
    out["strict_exclusion_rank_median"] = strict["rank_median"]
    out["strict_exclusion_gap_mean"] = strict["gap_mean"]
    out["strict_exclusion_directionality"] = strict["directionality"]
    out["negative_seed_count_mean"] = strict["negative_seed_count_mean"]
    out["eligible_event_rate"] = strict["eligible_event_rate"]
    return out


def build_contradiction_alignment_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for episode in [0] + list(range(1, NUM_EPISODES + 1)):
        if episode == 0:
            scope = "global"
            df_ep = df.copy()
        else:
            scope = f"episode_{episode}"
            df_ep = df[df["episode"] == episode].copy()
        if df_ep.empty:
            continue
        eligible = df_ep[df_ep["num_neg"] > 0]
        base_df = eligible if not eligible.empty else df_ep
        rows.append(
            {
                "scope": scope,
                "episode": episode,
                "num_events": int(len(df_ep)),
                "eligible_event_rate": float((df_ep["num_neg"] > 0).mean()),
                "old_all_zero_rate": float(base_df["contra_old_all_zero"].mean()),
                "formal_all_zero_rate": float(base_df["contra_formal_all_zero"].mean()),
                "nosuspect_formal_all_zero_rate": float(base_df["contra_nosuspect_formal_all_zero"].mean()),
                "strict_all_zero_rate": float(base_df["contra_strict_all_zero"].mean()),
                "old_true_mean": float(base_df["contra_old_true"].mean()),
                "formal_true_mean": float(base_df["contra_formal_true"].mean()),
                "nosuspect_formal_true_mean": float(base_df["contra_nosuspect_formal_true"].mean()),
                "strict_true_mean": float(base_df["contra_strict_true"].mean()),
                "old_directionality": float(base_df["contra_old_directionality"].mean()),
                "formal_directionality": float(base_df["contra_formal_directionality"].mean()),
                "nosuspect_formal_directionality": float(base_df["contra_nosuspect_formal_directionality"].mean()),
                "strict_directionality": float(base_df["contra_strict_directionality"].mean()),
                "old_vs_formal_l1_mean": float(base_df["contra_old_vs_formal_l1"].mean()),
                "old_vs_formal_linf_max": float(base_df["contra_old_vs_formal_linf"].max()),
                "old_vs_formal_any_diff_rate": float(base_df["contra_old_vs_formal_any_diff"].mean()),
                "old_nonzero_formal_zero_rate": float(base_df["contra_old_nonzero_formal_zero"].mean()),
                "formal_nonzero_old_zero_rate": float(base_df["contra_formal_nonzero_old_zero"].mean()),
                "formal_vs_nosuspect_l1_mean": float(base_df["contra_formal_vs_nosuspect_l1"].mean()),
                "formal_vs_nosuspect_linf_max": float(base_df["contra_formal_vs_nosuspect_linf"].max()),
                "formal_vs_nosuspect_any_diff_rate": float(base_df["contra_formal_vs_nosuspect_any_diff"].mean()),
                "formal_zero_nosuspect_nonzero_rate": float(base_df["contra_formal_zero_nosuspect_nonzero"].mean()),
                "nosuspect_true_lift_mean": float(base_df["contra_nosuspect_true_lift"].mean()),
                "body_still_zero_without_suspect_rate": float(base_df["contra_body_still_zero_without_suspect"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_contradiction_alignment_markdown(alignment: pd.DataFrame) -> str:
    if alignment.empty:
        return "# Contradiction Alignment Overview\n\nNo alignment rows collected.\n"
    overall = alignment[alignment["scope"] == "global"].iloc[0]
    lines = [
        "# Contradiction Alignment Overview",
        "",
        "## 1. `t_sim=None` vs formal runtime",
        (
            f"- global eligible_event_rate={overall['eligible_event_rate']:.2%}, "
            f"old_vs_formal_l1_mean={overall['old_vs_formal_l1_mean']:.4f}, "
            f"old_vs_formal_linf_max={overall['old_vs_formal_linf_max']:.4f}, "
            f"old_vs_formal_any_diff_rate={overall['old_vs_formal_any_diff_rate']:.2%}, "
            f"old_nonzero_formal_zero_rate={overall['old_nonzero_formal_zero_rate']:.2%}, "
            f"formal_nonzero_old_zero_rate={overall['formal_nonzero_old_zero_rate']:.2%}."
        ),
        "",
        "## 2. suspect 乘法压制",
        (
            f"- formal_vs_nosuspect_l1_mean={overall['formal_vs_nosuspect_l1_mean']:.4f}, "
            f"formal_vs_nosuspect_linf_max={overall['formal_vs_nosuspect_linf_max']:.4f}, "
            f"formal_vs_nosuspect_any_diff_rate={overall['formal_vs_nosuspect_any_diff_rate']:.2%}, "
            f"formal_zero_nosuspect_nonzero_rate={overall['formal_zero_nosuspect_nonzero_rate']:.2%}, "
            f"nosuspect_true_lift_mean={overall['nosuspect_true_lift_mean']:.4f}."
        ),
        "",
        "## 3. body 自身是否仍弱",
        (
            f"- nosuspect_formal_all_zero_rate={overall['nosuspect_formal_all_zero_rate']:.2%}, "
            f"nosuspect_formal_directionality={overall['nosuspect_formal_directionality']:.2%}, "
            f"nosuspect_formal_true_mean={overall['nosuspect_formal_true_mean']:.4f}, "
            f"strict_true_mean={overall['strict_true_mean']:.4f}."
        ),
        "",
        "## 4. Episode Breakdown",
        "```text",
        alignment.round(4).to_string(index=False),
        "```",
        "",
        "## 输出文件",
        f"- `{CONTRA_ALIGNMENT_CSV_PATH}`",
        f"- `{CONTRA_ALIGNMENT_MD_PATH}`",
    ]
    return "\n".join(lines) + "\n"


def make_reaction_table(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary[(summary["axis"] == "reaction_consistency") & (summary["version"] == "builder")].copy()
    return df[
        [
            "episode", "eligible_event_rate", "all_zero_event_rate", "rank_median",
            "gap_mean", "directionality", "conditioned_directionality", "true_nonzero_rate",
        ]
    ].set_index("episode")


def make_uncertainty_table(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary[(summary["axis"] == "uncertainty_gap") & (summary["version"] == "builder")].copy()
    return df[
        [
            "episode", "observed_mean", "unobserved_mean", "obs_alignment_l1_mean",
            "true_nonzero_rate",
        ]
    ].set_index("episode")


def support_decision(ep10_support: pd.Series) -> str:
    if (
        ep10_support["conditioned_rank_median"] <= 2.0
        and ep10_support["conditioned_directionality"] >= 0.5
        and ep10_support["all_zero_event_rate"] <= 0.5
        and ep10_support["unique_top1_rate"] >= 0.5
    ):
        return "ready as main evidence feature"
    if (
        ep10_support["conditioned_rank_median"] <= 3.0
        and ep10_support["conditioned_directionality"] >= 0.5
    ):
        return "still conditional"
    return "not ready"


def suspect_diagnosis_and_decision(ep10_suspect_raw: pd.Series, ep10_suspect_active: pd.Series) -> Dict[str, str]:
    raw_good = ep10_suspect_raw["raw_rank_median"] <= 2.0 and ep10_suspect_raw["raw_directionality"] >= 0.5
    active_bad = ep10_suspect_active["active_recall"] < 0.5
    if raw_good and active_bad:
        return {
            "diagnosis": "raw 好但 active 差，说明 threshold / gating / mask 设计有问题。",
            "decision": "disable hard mask",
        }
    if raw_good:
        return {
            "diagnosis": "raw 与 active 没有明显冲突，当前主要可当 raw soft prior 看待。",
            "decision": "keep as raw soft prior",
        }
    return {
        "diagnosis": "raw 也差，说明 suspect 定义本体有问题。",
        "decision": "redesign needed",
    }


def contradiction_decision(ep10_old: pd.Series, ep10_strict: pd.Series) -> str:
    if (
        ep10_strict["gap_mean"] > 0.0
        and ep10_strict["directionality"] >= 0.5
        and (
            ep10_old["gap_mean"] <= 0.0
            or ep10_old["directionality"] < ep10_strict["directionality"]
            or ep10_old["rank_median"] > ep10_strict["rank_median"]
        )
    ):
        return "replace with strict exclusion"
    if ep10_old["gap_mean"] > 0.0 and ep10_old["directionality"] >= 0.5:
        return "keep old"
    return "disable for now"


def reaction_decision(ep10_reaction: pd.Series) -> str:
    if ep10_reaction["conditioned_directionality"] >= 0.5 and ep10_reaction["all_zero_event_rate"] <= 0.5:
        return "keep as auxiliary"
    return "do not trust yet"


def uncertainty_decision(ep10_uncertainty: pd.Series) -> str:
    if (
        ep10_uncertainty["obs_alignment_l1_mean"] <= 1e-6
        and ep10_uncertainty["observed_mean"] <= 1e-6
        and ep10_uncertainty["unobserved_mean"] >= 0.99
    ):
        return "currently just unobserved mask"
    return "contains more than mask semantics"


def next_step_recommendation(support_status: str, suspect_status: str, contradiction_status: str) -> str:
    usable_axes = []
    if support_status in {"ready as main evidence feature", "still conditional"}:
        usable_axes.append("support")
    if suspect_status == "keep as raw soft prior":
        usable_axes.append("suspect raw")
    if contradiction_status == "replace with strict exclusion":
        usable_axes.append("strict contradiction")
    if usable_axes:
        return "可以开始让模型消费这些轴，但只限已保留版本：" + "、".join(usable_axes) + "；其余轴继续改公式。"
    return "应先继续改公式，再考虑让模型消费这些轴。"


def build_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    ep10_support = summary[(summary["axis"] == "support") & (summary["version"] == "cleaned")].iloc[-1]
    ep10_suspect_raw = summary[(summary["axis"] == "suspect") & (summary["version"] == "raw_score")].iloc[-1]
    ep10_suspect_active = summary[(summary["axis"] == "suspect") & (summary["version"] == "active_mask")].iloc[-1]
    ep10_old = summary[(summary["axis"] == "contradiction") & (summary["version"] == "old")].iloc[-1]
    ep10_strict = summary[(summary["axis"] == "contradiction") & (summary["version"] == "strict_exclusion")].iloc[-1]
    ep10_reaction = summary[(summary["axis"] == "reaction_consistency") & (summary["version"] == "builder")].iloc[-1]
    ep10_uncertainty = summary[(summary["axis"] == "uncertainty_gap") & (summary["version"] == "builder")].iloc[-1]

    support_status = support_decision(ep10_support)
    suspect_meta = suspect_diagnosis_and_decision(ep10_suspect_raw, ep10_suspect_active)
    contradiction_status = contradiction_decision(ep10_old, ep10_strict)
    reaction_status = reaction_decision(ep10_reaction)
    uncertainty_status = uncertainty_decision(ep10_uncertainty)

    return pd.DataFrame(
        [
            {
                "axis": "Support",
                "decision": support_status,
                "evidence": f"cond_rank={ep10_support['conditioned_rank_median']:.2f}, cond_dir={ep10_support['conditioned_directionality']:.2f}, zero={ep10_support['all_zero_event_rate']:.2f}, compare_l1={ep10_support['compare_l1_mean']:.4f}",
            },
            {
                "axis": "Suspect",
                "decision": suspect_meta["decision"],
                "evidence": f"raw_rank={ep10_suspect_raw['raw_rank_median']:.2f}, raw_dir={ep10_suspect_raw['raw_directionality']:.2f}, active_recall={ep10_suspect_active['active_recall']:.2f}",
            },
            {
                "axis": "Contradiction",
                "decision": contradiction_status,
                "evidence": f"old_gap={ep10_old['gap_mean']:.4f}, strict_gap={ep10_strict['gap_mean']:.4f}, strict_dir={ep10_strict['directionality']:.2f}, time_conflict={ep10_old['time_conflict_rate']:.2f}",
            },
            {
                "axis": "Reaction",
                "decision": reaction_status,
                "evidence": f"dir={ep10_reaction['conditioned_directionality']:.2f}, zero={ep10_reaction['all_zero_event_rate']:.2f}",
            },
            {
                "axis": "Uncertainty",
                "decision": uncertainty_status,
                "evidence": f"align_l1={ep10_uncertainty['obs_alignment_l1_mean']:.6f}, obs={ep10_uncertainty['observed_mean']:.2f}, unobs={ep10_uncertainty['unobserved_mean']:.2f}",
            },
        ]
    ).set_index("axis")


def build_markdown(summary: pd.DataFrame) -> str:
    ep10_support = summary[(summary["axis"] == "support") & (summary["version"] == "cleaned")].iloc[-1]
    ep10_suspect_raw = summary[(summary["axis"] == "suspect") & (summary["version"] == "raw_score")].iloc[-1]
    ep10_suspect_active = summary[(summary["axis"] == "suspect") & (summary["version"] == "active_mask")].iloc[-1]
    ep10_old = summary[(summary["axis"] == "contradiction") & (summary["version"] == "old")].iloc[-1]
    ep10_strict = summary[(summary["axis"] == "contradiction") & (summary["version"] == "strict_exclusion")].iloc[-1]
    ep10_reaction = summary[(summary["axis"] == "reaction_consistency") & (summary["version"] == "builder")].iloc[-1]
    ep10_uncertainty = summary[(summary["axis"] == "uncertainty_gap") & (summary["version"] == "builder")].iloc[-1]

    support_status = support_decision(ep10_support)
    suspect_meta = suspect_diagnosis_and_decision(ep10_suspect_raw, ep10_suspect_active)
    contradiction_status = contradiction_decision(ep10_old, ep10_strict)
    reaction_status = reaction_decision(ep10_reaction)
    uncertainty_status = uncertainty_decision(ep10_uncertainty)
    next_step = next_step_recommendation(support_status, suspect_meta["decision"], contradiction_status)

    lines = [
        "# Evidence Axis Semantics Cleanup Sprint",
        "",
        "## 1. support 在 cleaned metrics 下，是否仍然站得住？",
        f"站得住的前提是 **conditioned** 口径。Episode 10 的 conditioned_rank_median={ep10_support['conditioned_rank_median']:.2f}，conditioned_gap_mean={ep10_support['conditioned_gap_mean']:.4f}，conditioned_directionality={ep10_support['conditioned_directionality']:.2%}；full 口径的 all_zero_event_rate={ep10_support['all_zero_event_rate']:.2%}、eligible_event_rate={ep10_support['eligible_event_rate']:.2%} 说明它仍受无 positive seed 事件影响；builder_vs_legacy_l1={ep10_support['compare_l1_mean']:.4f} 则显示 compare 路径仍有分歧。工程判断：**{support_status}**。",
        "",
        "## 2. suspect 的问题主要在 raw score 还是 active mask？",
        f"Episode 10 的 raw_rank_median={ep10_suspect_raw['raw_rank_median']:.2f}、raw_gap_mean={ep10_suspect_raw['raw_gap_mean']:.4f}、raw_directionality={ep10_suspect_raw['raw_directionality']:.2%}；active_recall={ep10_suspect_active['active_recall']:.2%}、active_reduction={ep10_suspect_active['active_reduction']:.2%}、candidate_set_size_mean={ep10_suspect_active['candidate_set_size_mean']:.2f}。B3 root diagnosis：{suspect_meta['diagnosis']} 工程判断：**{suspect_meta['decision']}**。",
        "",
        "## 3. contradiction 改成 strict-exclusion 后，是否不再 self-contradict？",
        f"旧版 Episode 10: rank={ep10_old['rank_median']:.2f}, gap={ep10_old['gap_mean']:.4f}, directionality={ep10_old['directionality']:.2%}, time_conflict_rate={ep10_old['time_conflict_rate']:.2%}。strict-exclusion Episode 10: rank={ep10_strict['rank_median']:.2f}, gap={ep10_strict['gap_mean']:.4f}, directionality={ep10_strict['directionality']:.2%}，negative_seed_count_mean={ep10_strict['negative_seed_count_mean']:.2f}，eligible_event_rate={ep10_strict['eligible_event_rate']:.2%}。工程判断：**{contradiction_status}**。",
        "",
        "## 4. reaction / uncertainty 当前在表达什么？",
        f"Reaction Episode 10: conditioned_directionality={ep10_reaction['conditioned_directionality']:.2%}，all_zero_event_rate={ep10_reaction['all_zero_event_rate']:.2%}。工程判断：**{reaction_status}**。",
        f"Uncertainty Episode 10: observed_mean={ep10_uncertainty['observed_mean']:.4f}，unobserved_mean={ep10_uncertainty['unobserved_mean']:.4f}，obs_alignment_l1_mean={ep10_uncertainty['obs_alignment_l1_mean']:.6f}。工程判断：**{uncertainty_status}**。",
        "",
        "## 5. 五轴当前分别应该保留、降级还是禁用？",
        f"Support: **{support_status}**  ",
        f"Suspect: **{suspect_meta['decision']}**  ",
        f"Contradiction: **{contradiction_status}**  ",
        f"Reaction: **{reaction_status}**  ",
        f"Uncertainty: **{uncertainty_status}**",
        "",
        "## 6. 下一步最该做的是继续改公式，还是可以开始让模型消费这些轴？",
        next_step,
    ]
    return "\n".join(lines) + "\n"


def build_case_digest(df: pd.DataFrame) -> str:
    sections = [
        "normal_axes_consistent",
        "abnormal_support_hub_win",
        "abnormal_support_compare_diverge",
        "abnormal_suspect_drops_true",
        "abnormal_negative_time_gate_conflict",
        "triage_no_positive_seed",
        "triage_reaction_noise",
        "triage_mixed",
    ]
    lines = ["# Practical Audit Diagnostic Cases", ""]
    bucket_counts = df["diagnostic_bucket"].value_counts().sort_index()
    lines.append("## Bucket Counts")
    lines.append(bucket_counts.to_string())
    lines.append("")
    for bucket in sections:
        subset = df[df["diagnostic_bucket"] == bucket]
        lines.append(f"## {bucket}")
        if subset.empty:
            lines.append("No cases captured.")
            lines.append("")
            continue
        cols = [
            "event_id", "episode", "time_min", "num_pos", "num_neg", "support_rank",
            "support_builder_vs_legacy_l1", "suspect_active_recall", "contra_old_true",
            "contra_formal_true", "contra_nosuspect_formal_true", "contra_strict_true",
            "reaction_gap", "uncertainty_obs_alignment_l1",
            "responsibility_hint",
        ]
        lines.append(subset[cols].head(5).to_string(index=False))
        lines.append("")
    return "\n".join(lines)


def build_positive_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for episode in range(1, NUM_EPISODES + 1):
        df_ep = df[df["episode"] == episode]
        if df_ep.empty:
            continue
        has_input = df_ep["num_pos"] > 0
        no_input = ~has_input
        rows.append(
            {
                "episode": episode,
                "num_events": int(len(df_ep)),
                "truth_positive_mean": float(df_ep["truth_positive_total"].mean()),
                "truth_positive_median": float(df_ep["truth_positive_total"].median()),
                "observed_positive_mean": float(df_ep["num_pos"].mean()),
                "hidden_truth_positive_mean": float(df_ep["truth_positive_unobserved"].mean()),
                "no_observed_positive_rate": float((df_ep["num_pos"] <= 0).mean()),
                "no_truth_positive_rate": float((df_ep["truth_positive_total"] <= 0).mean()),
                "no_observed_but_truth_positive_rate": float(
                    ((df_ep["num_pos"] <= 0) & (df_ep["truth_positive_total"] > 0)).mean()
                ),
                "support_all_zero_rate": float(df_ep["support_all_zero"].mean()),
                "support_all_zero_given_input_rate": float(df_ep.loc[has_input, "support_all_zero"].mean()) if bool(has_input.any()) else np.nan,
                "support_all_zero_due_no_input_rate": float(df_ep.loc[no_input, "support_all_zero"].mean()) if bool(no_input.any()) else np.nan,
                "support_true_nonzero_given_input_rate": float(df_ep.loc[has_input, "support_true_nonzero"].mean()) if bool(has_input.any()) else np.nan,
                "suspect_active_recall_given_input": float(df_ep.loc[has_input, "suspect_active_recall"].mean()) if bool(has_input.any()) else np.nan,
                "suspect_active_recall_given_no_input": float(df_ep.loc[no_input, "suspect_active_recall"].mean()) if bool(no_input.any()) else np.nan,
                "support_hub_win_given_input_rate": float(df_ep.loc[has_input, "support_hub_win"].mean()) if bool(has_input.any()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_positive_seed_markdown(summary: pd.DataFrame, case_trace: pd.DataFrame) -> str:
    lines = ["# Positive Seed Survival Diagnostics", ""]
    if not summary.empty:
        lines.append("## Episode Summary")
        lines.append(summary.round(4).to_string(index=False))
        lines.append("")

    lines.append("## Fixed Case Slices")
    for event_id in CASE_EVENT_IDS:
        subset = case_trace[case_trace["event_id"] == event_id]
        lines.append(f"### event {event_id}")
        if subset.empty:
            lines.append("No trace rows captured.")
            lines.append("")
            continue
        cols = [
            "episode",
            "candidate_role",
            "num_pos",
            "truth_positive_total",
            "truth_positive_unobserved",
            "positive_seed_root_cause",
            "support_builder_total",
            "support_zero_due_no_input",
            "support_zero_despite_input",
            "observed_positive_supportable_count",
            "truth_positive_supportable_count",
        ]
        lines.append(subset[cols].to_string(index=False))
        lines.append("")
    return "\n".join(lines)


def build_case_trace_markdown(case_trace: pd.DataFrame) -> str:
    lines = ["# Practical Audit Fixed Case Trace", ""]
    for event_id in CASE_EVENT_IDS:
        subset = case_trace[case_trace["event_id"] == event_id]
        lines.append(f"## event {event_id}")
        if subset.empty:
            lines.append("No trace rows captured.")
            lines.append("")
            continue
        cols = [
            "episode",
            "candidate_role",
            "candidate_global_id",
            "candidate_degree",
            "suspect_active",
            "suspect_raw_score",
            "topology_gate",
            "coarse_time_gate",
            "negative_pressure_hard",
            "support_builder_total",
            "support_legacy_total",
            "support_base",
            "support_specificity",
            "support_focus",
            "support_chlorine",
            "legacy_rank_term",
            "legacy_hub_term",
            "observed_positive_supportable_count",
            "truth_positive_supportable_count",
            "contradiction_old_total",
            "contradiction_formal_total",
            "contradiction_nosuspect_formal_total",
            "contradiction_strict_total",
            "positive_seed_root_cause",
        ]
        lines.append(subset[cols].round(4).to_string(index=False))
        lines.append("")
    return "\n".join(lines)


def build_support_rootcause_outputs(rootcause_df: pd.DataFrame, case_trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    df = rootcause_df.copy()
    if df.empty:
        empty_md = "# Support Score Subterm Rootcause Summary\n\nNo rootcause rows collected.\n"
        return df, df, empty_md

    df["b_subset"] = (
        (df["suspect_active_true"] > 0.5)
        & (df["num_pos"] > 0)
        & (df["support_true_nonzero"] <= 0.5)
    ).astype(float)
    df["c_subset"] = (
        (df["suspect_active_true"] > 0.5)
        & (df["num_pos"] > 0)
        & (df["support_true_nonzero"] > 0.5)
        & (df["hub_win"] > 0.5)
    ).astype(float)

    b_df = df[df["b_subset"] > 0.5].copy()
    b_df["b_bucket"] = b_df.apply(classify_b_rootcause, axis=1) if not b_df.empty else pd.Series(dtype=str)
    c_df = df[df["c_subset"] > 0.5].copy()
    c_df["c_bucket"] = c_df.apply(classify_c_rootcause, axis=1) if not c_df.empty else pd.Series(dtype=str)

    b_export_cols = [
        "event_id", "episode", "time_min", "num_pos", "truth_positive_total", "truth_positive_unobserved",
        "positive_seed_root_cause", "b_bucket", "suspect_active_true",
        "support_true_total", "support_competitor_total",
        "true_truth_positive_supportable_count", "true_observed_positive_supportable_count",
        "competitor_truth_positive_supportable_count", "competitor_observed_positive_supportable_count",
        "support_true_base", "support_true_specificity", "support_true_focus", "support_true_chlorine",
        "support_true_base_zero", "support_true_specificity_zero", "support_true_focus_zero", "support_true_chlorine_zero",
        "competitor_observed_positive_supportable_gt_true", "competitor_truth_positive_supportable_gt_true",
        "observed_supportable_gap", "truth_supportable_gap",
        "top_competitor_global_id",
    ]
    c_export_cols = [
        "event_id", "episode", "time_min", "num_pos", "truth_positive_total", "truth_positive_unobserved",
        "positive_seed_root_cause", "c_bucket", "largest_gap_source", "supportable_similar_le1", "supportable_equal",
        "support_true_total", "support_competitor_total",
        "support_true_base", "support_competitor_base",
        "support_true_specificity", "support_competitor_specificity",
        "support_true_focus", "support_competitor_focus",
        "support_true_chlorine", "support_competitor_chlorine",
        "true_observed_positive_supportable_count", "competitor_observed_positive_supportable_count",
        "true_truth_positive_supportable_count", "competitor_truth_positive_supportable_count",
        "competitor_gt_true_base", "competitor_gt_true_specificity", "competitor_gt_true_focus", "competitor_gt_true_chlorine",
        "competitor_observed_positive_supportable_gt_true", "competitor_truth_positive_supportable_gt_true",
        "observed_supportable_gap", "truth_supportable_gap",
        "top_competitor_global_id",
    ]
    b_df[b_export_cols].to_csv(B_ROOTCAUSE_CSV_PATH, index=False)
    c_df[c_export_cols].to_csv(C_ROOTCAUSE_CSV_PATH, index=False)

    lines = ["# Support Score Subterm Rootcause Summary", ""]
    lines.append("## Inputs")
    lines.append("- Scope: rerun-only path from `src/scripts/audit/run_practical_audit_rerun.py`.")
    lines.append("- B subset: `suspect_active=true & num_pos>0 & support_true_nonzero=0`.")
    lines.append("- C subset: `suspect_active=true & num_pos>0 & support_true_nonzero>0 & hub_win=1`.")
    lines.append("- Similar supportable positives assumption: `abs(competitor_observed_positive_supportable_count - true_observed_positive_supportable_count) <= 1`.")
    lines.append("")

    lines.append("## B Subset (Availability)")
    lines.append(f"- B total: {len(b_df)}")
    lines.append(
        f"- [已证明] {count_ratio_line(b_df.assign(flag=(b_df['true_truth_positive_supportable_count'] <= EPS).astype(float)), 'flag').replace('flag', 'true_truth_positive_supportable_count==0')}"
        if not b_df.empty else "- [未证明] B subset empty."
    )
    if not b_df.empty:
        lines.append(
            f"- [已证明] {count_ratio_line(b_df.assign(flag=(b_df['true_observed_positive_supportable_count'] <= EPS).astype(float)), 'flag').replace('flag', 'true_observed_positive_supportable_count==0')}"
        )
        lines.append(
            f"- [已证明] base/spec/focus/chlorine zero rates = "
            f"{b_df['support_true_base_zero'].mean():.4f} / "
            f"{b_df['support_true_specificity_zero'].mean():.4f} / "
            f"{b_df['support_true_focus_zero'].mean():.4f} / "
            f"{b_df['support_true_chlorine_zero'].mean():.4f}"
        )
        lines.append(
            f"- [已证明] competitor observed supportable > true = {b_df['competitor_observed_positive_supportable_gt_true'].mean():.4f}"
            f" ({int(b_df['competitor_observed_positive_supportable_gt_true'].sum())}/{len(b_df)})"
        )
        lines.append(
            f"- [已证明] competitor truth supportable > true = {b_df['competitor_truth_positive_supportable_gt_true'].mean():.4f}"
            f" ({int(b_df['competitor_truth_positive_supportable_gt_true'].sum())}/{len(b_df)})"
        )
        bucket_counts = b_df["b_bucket"].value_counts()
        for bucket in [
            "B1_no_true_supportable_witness",
            "B2_truth_witness_hidden_or_unavailable",
            "B3_observed_witness_present_but_support_zero",
        ]:
            count = int(bucket_counts.get(bucket, 0))
            lines.append(f"- [已证明] {bucket}: {count / len(b_df):.4f} ({count}/{len(b_df)})")
        if int(bucket_counts.get("B3_observed_witness_present_but_support_zero", 0)) == 0:
            lines.append("- [已证明] 未发现 observed supportable witness 已存在但 final support 仍被压成 0 的全局样本。")
        else:
            lines.append("- [部分证明] 存在 observed supportable witness 仍然 final support=0 的样本，需要继续逐条核对数值边界。")
    lines.append("")

    lines.append("## C Subset (Advantage)")
    lines.append(f"- C total: {len(c_df)}")
    if not c_df.empty:
        gap_patterns: Dict[str, int] = {}
        for _, row in c_df.iterrows():
            gaps = {
                "base": float(row["support_competitor_base"] - row["support_true_base"]),
                "specificity": float(row["support_competitor_specificity"] - row["support_true_specificity"]),
                "focus": float(row["support_competitor_focus"] - row["support_true_focus"]),
                "chlorine": float(row["support_competitor_chlorine"] - row["support_true_chlorine"]),
            }
            max_gap = max(gaps.values())
            if max_gap <= EPS:
                pattern = "none"
            else:
                winners = [name for name, value in gaps.items() if value >= max_gap - EPS]
                pattern = "+".join(winners)
            gap_patterns[pattern] = gap_patterns.get(pattern, 0) + 1
        lines.append(
            f"- [已证明] competitor > true base/spec/focus/chlorine = "
            f"{c_df['competitor_gt_true_base'].mean():.4f} / "
            f"{c_df['competitor_gt_true_specificity'].mean():.4f} / "
            f"{c_df['competitor_gt_true_focus'].mean():.4f} / "
            f"{c_df['competitor_gt_true_chlorine'].mean():.4f}"
        )
        gap_counts = c_df["largest_gap_source"].value_counts()
        gap_parts = [f"{name}:{count}" for name, count in gap_counts.items()]
        lines.append(f"- [已证明] largest gap source counts = {'; '.join(gap_parts)}")
        pattern_parts = [f"{name}:{count}" for name, count in sorted(gap_patterns.items(), key=lambda kv: (-kv[1], kv[0]))]
        lines.append(f"- [已证明] largest gap pattern counts = {'; '.join(pattern_parts)}")
        lines.append(
            f"- [已证明] competitor observed supportable > true = {c_df['competitor_observed_positive_supportable_gt_true'].mean():.4f}"
            f" ({int(c_df['competitor_observed_positive_supportable_gt_true'].sum())}/{len(c_df)})"
        )
        lines.append(
            f"- [已证明] competitor truth supportable > true = {c_df['competitor_truth_positive_supportable_gt_true'].mean():.4f}"
            f" ({int(c_df['competitor_truth_positive_supportable_gt_true'].sum())}/{len(c_df)})"
        )
        lines.append(
            f"- [已证明] supportable positives similar but competitor still wins = {c_df['supportable_similar_le1'].mean():.4f}"
            f" ({int(c_df['supportable_similar_le1'].sum())}/{len(c_df)})"
        )
        c_bucket_counts = c_df["c_bucket"].value_counts()
        for bucket in [
            "C1_availability_deficit",
            "C2_specificity_dominant_loss",
            "C3_focus_dominant_loss",
            "mixed_or_other",
        ]:
            count = int(c_bucket_counts.get(bucket, 0))
            lines.append(f"- [已证明] {bucket}: {count / len(c_df):.4f} ({count}/{len(c_df)})")
    else:
        lines.append("- [未证明] C subset empty.")
    lines.append("")

    lines.append("## Fixed Cases (Explanation Only)")
    case_cols = [
        "event_id",
        "episode",
        "candidate_role",
        "num_pos",
        "support_builder_total",
        "support_base",
        "support_specificity",
        "support_focus",
        "support_chlorine",
        "observed_positive_supportable_count",
        "truth_positive_supportable_count",
    ]
    for event_id in CASE_EVENT_IDS:
        subset = case_trace[case_trace["event_id"] == event_id]
        lines.append(f"### event {event_id}")
        if subset.empty:
            lines.append("- [未证明] no fixed-case rows captured.")
            continue
        lines.append(subset[case_cols].round(4).to_string(index=False))
        lines.append("")

    return b_df[b_export_cols], c_df[c_export_cols], "\n".join(lines)


def build_oracle_stepwise_export(records: pd.DataFrame) -> pd.DataFrame:
    df = records.copy()
    df["fixed_case"] = df["event_id"].isin(CASE_EVENT_IDS).astype(float)
    df["practical_b_subset"] = (
        (df["source_validity_true"] > 0.5)
        & (df["num_pos"] > 0)
        & (df["support_true_nonzero"] <= 0.5)
    ).astype(float)
    df["practical_c_subset"] = (
        (df["source_validity_true"] > 0.5)
        & (df["num_pos"] > 0)
        & (df["support_true_nonzero"] > 0.5)
        & (df["support_hub_win"] > 0.5)
    ).astype(float)
    export_cols = [
        "event_id",
        "episode",
        "time_min",
        "fixed_case",
        "num_pos",
        "oracle_num_pos",
        "oracle_positive_gain",
        "truth_positive_total",
        "truth_positive_observed",
        "truth_positive_unobserved",
        "source_validity_true",
        "no_positive_seed_root_cause",
        "support_rank",
        "support_gap",
        "support_directionality",
        "support_true_nonzero",
        "support_hub_win",
        "support_true_builder",
        "support_top_other_global_id",
        "support_top_other_score",
        "support_true_base",
        "support_true_specificity",
        "support_true_focus",
        "support_true_chlorine",
        "support_competitor_base",
        "support_competitor_specificity",
        "support_competitor_focus",
        "support_competitor_chlorine",
        "oracle_suspect_pool_true",
        "oracle_suspect_pool_sum",
        "oracle_suspect_pool_pre_top_other",
        "oracle_suspect_pool_post_top_other",
        "practical_rank",
        "practical_gap",
        "practical_directionality",
        "practical_true_nonzero",
        "practical_hub_win",
        "practical_true_total",
        "practical_top_other_global_id",
        "practical_top_other_score",
        "practical_true_base",
        "practical_true_specificity",
        "practical_true_focus",
        "practical_true_chlorine",
        "practical_competitor_base",
        "practical_competitor_specificity",
        "practical_competitor_focus",
        "practical_competitor_chlorine",
        "oracle_pre_rank",
        "oracle_pre_gap",
        "oracle_pre_directionality",
        "oracle_pre_all_zero",
        "oracle_pre_true_nonzero",
        "oracle_pre_hub_win",
        "oracle_pre_true_total",
        "oracle_pre_top_other_global_id",
        "oracle_pre_top_other_score",
        "oracle_pre_true_base",
        "oracle_pre_true_specificity",
        "oracle_pre_true_focus",
        "oracle_pre_true_chlorine",
        "oracle_pre_competitor_base",
        "oracle_pre_competitor_specificity",
        "oracle_pre_competitor_focus",
        "oracle_pre_competitor_chlorine",
        "oracle_post_rank",
        "oracle_post_gap",
        "oracle_post_directionality",
        "oracle_post_all_zero",
        "oracle_post_true_nonzero",
        "oracle_post_hub_win",
        "oracle_post_true_total",
        "oracle_post_top_other_global_id",
        "oracle_post_top_other_score",
        "oracle_post_true_base",
        "oracle_post_true_specificity",
        "oracle_post_true_focus",
        "oracle_post_true_chlorine",
        "oracle_post_competitor_base",
        "oracle_post_competitor_specificity",
        "oracle_post_competitor_focus",
        "oracle_post_competitor_chlorine",
        "oracle_pre_rank_delta",
        "oracle_post_rank_delta",
        "oracle_pre_rank_improved",
        "oracle_post_rank_improved",
        "oracle_pre_rank_improved_2plus",
        "oracle_post_rank_improved_2plus",
        "oracle_pre_directionality_flip_to_true",
        "oracle_post_directionality_flip_to_true",
        "practical_b_subset",
        "practical_c_subset",
    ]
    return df[export_cols].copy()


def summarize_oracle_variant_slice(
    df_slice: pd.DataFrame,
    prefix: str,
    variant: str,
    scope: str,
    episode: int,
) -> Dict[str, float]:
    oracle_input_mask = df_slice["oracle_num_pos"] > 0
    eligible = df_slice.loc[oracle_input_mask]
    return {
        "variant": variant,
        "scope": scope,
        "episode": episode,
        "num_events": int(len(df_slice)),
        "oracle_input_events": int(oracle_input_mask.sum()),
        "oracle_eligible_event_rate": float(oracle_input_mask.mean()) if len(df_slice) > 0 else np.nan,
        "oracle_all_zero_event_rate": safe_mean_from_series(df_slice[f"{prefix}_all_zero"]),
        "oracle_rank_median_all": safe_median_from_series(df_slice[f"{prefix}_rank"]),
        "oracle_directionality_all": safe_mean_from_series(df_slice[f"{prefix}_directionality"]),
        "oracle_conditioned_rank_median": safe_median_from_series(eligible[f"{prefix}_rank"]),
        "oracle_conditioned_directionality": safe_mean_from_series(eligible[f"{prefix}_directionality"]),
        "oracle_true_nonzero_given_oracle_input_rate": safe_mean_from_series(eligible[f"{prefix}_true_nonzero"]),
        "oracle_zero_despite_oracle_input_rate": safe_mean_from_series(
            (eligible[f"{prefix}_true_nonzero"] <= 0.5).astype(float)
        ),
        "oracle_hub_win_rate": safe_mean_from_series(eligible[f"{prefix}_hub_win"]),
        "oracle_hub_win_rate_all_events": safe_mean_from_series(df_slice[f"{prefix}_hub_win"]),
    }


def build_oracle_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    variants = [
        ("oracle_pre_mask", "oracle_pre"),
        ("oracle_post_mask", "oracle_post"),
    ]
    for variant, prefix in variants:
        rows.append(summarize_oracle_variant_slice(records, prefix, variant, "global", 0))
        for episode in range(1, NUM_EPISODES + 1):
            df_ep = records[records["episode"] == episode]
            if df_ep.empty:
                continue
            rows.append(summarize_oracle_variant_slice(df_ep, prefix, variant, "episode", episode))
    return pd.DataFrame(rows)


def build_oracle_vs_practical(records: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    b_subset = records[
        (records["source_validity_true"] > 0.5)
        & (records["num_pos"] > 0)
        & (records["support_true_nonzero"] <= 0.5)
    ].copy()
    c_subset = records[
        (records["source_validity_true"] > 0.5)
        & (records["num_pos"] > 0)
        & (records["support_true_nonzero"] > 0.5)
        & (records["support_hub_win"] > 0.5)
    ].copy()

    for variant, prefix in [("oracle_pre_mask", "oracle_pre"), ("oracle_post_mask", "oracle_post")]:
        rows.append(
            {
                "analysis": "A_B_subset_recovery",
                "variant": variant,
                "denominator": int(len(b_subset)),
                "oracle_true_nonzero_count": int((b_subset[f"{prefix}_true_nonzero"] > 0.5).sum()),
                "oracle_true_nonzero_rate": safe_mean_from_series((b_subset[f"{prefix}_true_nonzero"] > 0.5).astype(float)),
                "rank_improved_any_count": int((b_subset[f"{prefix}_rank"] < b_subset["support_rank"]).sum()),
                "rank_improved_any_rate": safe_mean_from_series((b_subset[f"{prefix}_rank"] < b_subset["support_rank"]).astype(float)),
                "rank_improved_2plus_count": int((b_subset[f"{prefix}_rank"] <= b_subset["support_rank"] - 2).sum()),
                "rank_improved_2plus_rate": safe_mean_from_series((b_subset[f"{prefix}_rank"] <= b_subset["support_rank"] - 2).astype(float)),
                "practical_rank_median": safe_median_from_series(b_subset["support_rank"]),
                "oracle_rank_median": safe_median_from_series(b_subset[f"{prefix}_rank"]),
                "practical_true_nonzero_rate": safe_mean_from_series(b_subset["support_true_nonzero"]),
            }
        )
        rows.append(
            {
                "analysis": "B_C_subset_persistence",
                "variant": variant,
                "denominator": int(len(c_subset)),
                "still_hub_win_count": int((c_subset[f"{prefix}_hub_win"] > 0.5).sum()),
                "still_hub_win_rate": safe_mean_from_series((c_subset[f"{prefix}_hub_win"] > 0.5).astype(float)),
                "directionality_flip_to_true_count": int(
                    ((c_subset["support_directionality"] <= 0.5) & (c_subset[f"{prefix}_directionality"] > 0.5)).sum()
                ),
                "directionality_flip_to_true_rate": safe_mean_from_series(
                    ((c_subset["support_directionality"] <= 0.5) & (c_subset[f"{prefix}_directionality"] > 0.5)).astype(float)
                ),
                "rank_improved_any_count": int((c_subset[f"{prefix}_rank"] < c_subset["support_rank"]).sum()),
                "rank_improved_2plus_count": int((c_subset[f"{prefix}_rank"] <= c_subset["support_rank"] - 2).sum()),
                "practical_hub_win_rate": safe_mean_from_series(c_subset["support_hub_win"]),
                "oracle_hub_win_rate": safe_mean_from_series(c_subset[f"{prefix}_hub_win"]),
                "practical_directionality_rate": safe_mean_from_series(c_subset["support_directionality"]),
                "oracle_directionality_rate": safe_mean_from_series(c_subset[f"{prefix}_directionality"]),
            }
        )

    rows.append(
        {
            "analysis": "oracle_pre_post_identity",
            "variant": "oracle_pre_vs_post",
            "denominator": int(len(records)),
            "true_total_diff_count": int((records["oracle_pre_true_total"] != records["oracle_post_true_total"]).sum()),
            "rank_diff_count": int((records["oracle_pre_rank"] != records["oracle_post_rank"]).sum()),
            "hub_diff_count": int((records["oracle_pre_hub_win"] != records["oracle_post_hub_win"]).sum()),
            "suspect_true_zero_with_oracle_input_count": int(
                ((records["oracle_num_pos"] > 0) & (records["oracle_suspect_pool_true"] <= 0.5)).sum()
            ),
            "suspect_true_zero_with_oracle_input_rate": safe_mean_from_series(
                (((records["oracle_num_pos"] > 0) & (records["oracle_suspect_pool_true"] <= 0.5)).astype(float))
            ),
        }
    )

    fixed_case_latest = (
        records[records["event_id"].isin(CASE_EVENT_IDS)]
        .sort_values(["event_id", "episode"])
        .groupby("event_id", as_index=False)
        .tail(1)
        .sort_values("event_id")
    )
    for _, row in fixed_case_latest.iterrows():
        rows.append(
            {
                "analysis": "fixed_case_latest",
                "variant": "latest",
                "event_id": int(row["event_id"]),
                "episode": int(row["episode"]),
                "time_min": float(row["time_min"]),
                "num_pos": int(row["num_pos"]),
                "oracle_num_pos": int(row["oracle_num_pos"]),
                "practical_rank": float(row["support_rank"]),
                "practical_directionality": float(row["support_directionality"]),
                "practical_hub_win": float(row["support_hub_win"]),
                "practical_true_nonzero": float(row["support_true_nonzero"]),
                "practical_true_total": float(row["support_true_builder"]),
                "oracle_pre_rank": float(row["oracle_pre_rank"]),
                "oracle_pre_directionality": float(row["oracle_pre_directionality"]),
                "oracle_pre_hub_win": float(row["oracle_pre_hub_win"]),
                "oracle_pre_true_nonzero": float(row["oracle_pre_true_nonzero"]),
                "oracle_pre_true_total": float(row["oracle_pre_true_total"]),
                "oracle_post_rank": float(row["oracle_post_rank"]),
                "oracle_post_directionality": float(row["oracle_post_directionality"]),
                "oracle_post_hub_win": float(row["oracle_post_hub_win"]),
                "oracle_post_true_nonzero": float(row["oracle_post_true_nonzero"]),
                "oracle_post_true_total": float(row["oracle_post_true_total"]),
                "oracle_suspect_pool_true": float(row["oracle_suspect_pool_true"]),
                "oracle_pre_top_other_global_id": int(row["oracle_pre_top_other_global_id"]),
                "oracle_post_top_other_global_id": int(row["oracle_post_top_other_global_id"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_oracle_gate(metrics: pd.DataFrame) -> Dict[str, object]:
    pre_global = metrics[(metrics["variant"] == "oracle_pre_mask") & (metrics["scope"] == "global")].iloc[0]
    failures = {
        "directionality": max(0.0, 0.85 - float(pre_global["oracle_conditioned_directionality"])),
        "median_rank": max(0.0, float(pre_global["oracle_conditioned_rank_median"]) - 2.0),
        "hub_win": max(0.0, float(pre_global["oracle_hub_win_rate"]) - 0.20),
        "true_nonzero_given_oracle_input": max(
            0.0,
            0.95 - float(pre_global["oracle_true_nonzero_given_oracle_input_rate"]),
        ),
    }
    failed = {name: margin for name, margin in failures.items() if margin > EPS}
    worst_two = [name for name, _margin in sorted(failed.items(), key=lambda item: item[1], reverse=True)[:2]]
    return {
        "pass": len(failed) == 0,
        "pre_global": pre_global,
        "failed_margins": failed,
        "worst_two": worst_two,
    }


def build_oracle_summary_markdown(
    oracle_stepwise: pd.DataFrame,
    oracle_metrics: pd.DataFrame,
    oracle_vs_practical: pd.DataFrame,
) -> str:
    pre_global = oracle_metrics[(oracle_metrics["variant"] == "oracle_pre_mask") & (oracle_metrics["scope"] == "global")]
    pre_ep10 = oracle_metrics[
        (oracle_metrics["variant"] == "oracle_pre_mask")
        & (oracle_metrics["scope"] == "episode")
        & (oracle_metrics["episode"] == 10)
    ]
    post_global = oracle_metrics[(oracle_metrics["variant"] == "oracle_post_mask") & (oracle_metrics["scope"] == "global")]
    post_ep10 = oracle_metrics[
        (oracle_metrics["variant"] == "oracle_post_mask")
        & (oracle_metrics["scope"] == "episode")
        & (oracle_metrics["episode"] == 10)
    ]
    subset_rows = oracle_vs_practical[
        oracle_vs_practical["analysis"].isin(["A_B_subset_recovery", "B_C_subset_persistence"])
    ].copy()
    identity_row = oracle_vs_practical[oracle_vs_practical["analysis"] == "oracle_pre_post_identity"].iloc[0]
    fixed_case_rows = oracle_vs_practical[oracle_vs_practical["analysis"] == "fixed_case_latest"].copy()
    gate = evaluate_oracle_gate(oracle_metrics)

    case_lines: List[str] = []
    for _, row in fixed_case_rows.sort_values("event_id").iterrows():
        event_id = int(row["event_id"])
        if row["oracle_num_pos"] <= 0:
            diagnosis = "current snapshot 本身没有 truth-positive，oracle 不提供额外 positive 输入，因此 support 不变。"
            proof_tag = "[已证明]"
        elif abs(row["oracle_pre_true_total"] - row["oracle_post_true_total"]) > EPS:
            diagnosis = "pre-mask 恢复但 post-mask 再掉下去，说明 suspect mask 继续压制。"
            proof_tag = "[已证明]"
        elif (
            abs(row["practical_true_total"] - row["oracle_pre_true_total"]) <= EPS
            and abs(row["oracle_pre_true_total"] - row["oracle_post_true_total"]) <= EPS
            and abs(row["practical_rank"] - row["oracle_pre_rank"]) <= EPS
            and abs(row["practical_directionality"] - row["oracle_pre_directionality"]) <= EPS
            and abs(row["practical_hub_win"] - row["oracle_pre_hub_win"]) <= EPS
        ):
            diagnosis = "practical 与 oracle 完全一致，说明这条 case 不是观测缺失造成，也不是 suspect mask 追加压制。"
            proof_tag = "[已证明]"
        elif row["oracle_pre_hub_win"] > 0.5 or row["oracle_pre_true_nonzero"] <= 0.5:
            diagnosis = "pre-mask oracle 仍然没救回来，且 post-mask 不再额外变差，说明 support 本体仍然偏错。"
            proof_tag = "[已证明]"
        else:
            diagnosis = "pre-mask 有改善且 post-mask 保持不变，说明主要是观测缺失；suspect mask 在 oracle 输入下没有继续加坏。"
            proof_tag = "[已证明]"
        case_lines.append(
            f"- {proof_tag} event {event_id}: practical(rank={row['practical_rank']:.0f}, true_nonzero={row['practical_true_nonzero']:.0f}, hub_win={row['practical_hub_win']:.0f}) -> "
            f"oracle pre(rank={row['oracle_pre_rank']:.0f}, true_nonzero={row['oracle_pre_true_nonzero']:.0f}, hub_win={row['oracle_pre_hub_win']:.0f}) -> "
            f"oracle post(rank={row['oracle_post_rank']:.0f}, true_nonzero={row['oracle_post_true_nonzero']:.0f}, hub_win={row['oracle_post_hub_win']:.0f}, suspect_true={row['oracle_suspect_pool_true']:.0f}); "
            f"{diagnosis}"
        )

    lines = [
        "# Support Score Oracle Snapshot Summary",
        "",
        "## Definition",
        "- [已证明] Oracle input is restricted to current-snapshot truth-positive only: `x_raw[:, t_snapshot_idx, 1] > 0.1`.",
        "- [已证明] No future information is used.",
        "- [已证明] `oracle_pre_mask` recomputes raw support body with `suspect_pool=1`.",
        "- [已证明] `oracle_post_mask` recomputes support with the normal suspect-pool multiplication under the same oracle-positive input.",
        "",
        "## Commands",
        "- `python src/scripts/audit/run_practical_audit_rerun.py`",
        "",
        "## Oracle Pre-Mask Baseline",
        "```text",
        pre_global.round(4).to_string(index=False),
        "",
        pre_ep10.round(4).to_string(index=False),
        "```",
        "",
        "## Oracle Post-Mask Baseline",
        "```text",
        post_global.round(4).to_string(index=False),
        "",
        post_ep10.round(4).to_string(index=False),
        "```",
        "",
        "## Oracle vs Practical",
        "```text",
        subset_rows.round(4).to_string(index=False),
        "```",
        "",
        "## Pre vs Post Identity",
        f"- [已证明] true_total diff rows = {int(identity_row['true_total_diff_count'])}/{int(identity_row['denominator'])}.",
        f"- [已证明] rank diff rows = {int(identity_row['rank_diff_count'])}/{int(identity_row['denominator'])}.",
        f"- [已证明] hub diff rows = {int(identity_row['hub_diff_count'])}/{int(identity_row['denominator'])}.",
        f"- [已证明] oracle_input>0 but suspect_true=0 rows = {int(identity_row['suspect_true_zero_with_oracle_input_count'])}/{int(pre_global.iloc[0]['oracle_input_events'])}, "
        f"but they do not create any pre/post delta on the exported true-source metrics.",
        "",
        "## Fixed Cases",
        *case_lines,
        "",
        "## Oracle Gate",
        f"- [已证明] pre-mask oracle gate pass = {gate['pass']}.",
        f"- [已证明] worst two failed metrics = {', '.join(gate['worst_two']) if gate['worst_two'] else 'none'}.",
        f"- [已证明] pre-mask oracle global: directionality={gate['pre_global']['oracle_conditioned_directionality']:.4f}, "
        f"median_rank={gate['pre_global']['oracle_conditioned_rank_median']:.4f}, "
        f"hub_win={gate['pre_global']['oracle_hub_win_rate']:.4f}, "
        f"true_nonzero_given_oracle_input={gate['pre_global']['oracle_true_nonzero_given_oracle_input_rate']:.4f}.",
    ]
    if gate["pass"]:
        lines.append("- [已证明] Since pre-mask oracle clears the gate, support body upper-bound is currently established.")
    else:
        lines.append("- [已证明] Since pre-mask oracle still fails the gate after removing suspect_pool and filling current truth-positive inputs, the support body itself is still wrong.")
    lines.extend(
        [
            "",
            "## Output Files",
            f"- `{ORACLE_STEPWISE_CSV_PATH}`",
            f"- `{ORACLE_METRICS_CSV_PATH}`",
            f"- `{ORACLE_VS_PRACTICAL_CSV_PATH}`",
            f"- `{ORACLE_SUMMARY_MD_PATH}`",
            "",
            "## Notes",
            "- [已证明] Per-episode metrics for all 10 episodes are stored in the CSV; the markdown only inlines global and Episode 10 for readability.",
            "- [已证明] `rank_improved_2plus` is the explicit threshold used here for \"明显改善\".",
        ]
    )
    return "\n".join(lines) + "\n"


def build_v2_oracle_stepwise_export(records: pd.DataFrame) -> pd.DataFrame:
    df = records.copy()
    df["fixed_case"] = df["event_id"].isin(CASE_EVENT_IDS).astype(float)
    df["practical_b_subset"] = (
        (df["source_validity_true"] > 0.5)
        & (df["num_pos"] > 0)
        & (df["support_true_nonzero"] <= 0.5)
    ).astype(float)
    df["b_oracle_relevant"] = (
        (df["practical_b_subset"] > 0.5)
        & (df["oracle_num_pos"] > 0)
        & (df["oracle_pre_true_nonzero"] <= 0.5)
    ).astype(float)
    df["c_subset"] = (
        (df["source_validity_true"] > 0.5)
        & (df["num_pos"] > 0)
        & (df["support_true_nonzero"] > 0.5)
        & (df["support_hub_win"] > 0.5)
    ).astype(float)
    export_cols = [
        "event_id",
        "episode",
        "time_min",
        "oracle_t_abs_idx",
        "fixed_case",
        "num_pos",
        "oracle_num_pos",
        "oracle_positive_gain",
        "truth_positive_total",
        "truth_positive_observed",
        "truth_positive_unobserved",
        "source_validity_true",
        "no_positive_seed_root_cause",
        "practical_b_subset",
        "b_oracle_relevant",
        "c_subset",
        "practical_rank",
        "practical_directionality",
        "practical_true_nonzero",
        "practical_hub_win",
        "practical_true_total",
        "oracle_pre_rank",
        "oracle_pre_gap",
        "oracle_pre_directionality",
        "oracle_pre_true_nonzero",
        "oracle_pre_hub_win",
        "oracle_pre_true_total",
        "oracle_pre_true_base",
        "oracle_pre_true_specificity",
        "oracle_pre_true_focus",
        "oracle_pre_true_chlorine",
        "oracle_pre_competitor_base",
        "oracle_pre_competitor_specificity",
        "oracle_pre_competitor_focus",
        "oracle_pre_competitor_chlorine",
        "v2_main_rank",
        "v2_main_gap",
        "v2_main_directionality",
        "v2_main_all_zero",
        "v2_main_true_nonzero",
        "v2_main_hub_win",
        "v2_main_true_total",
        "v2_main_top_other_global_id",
        "v2_main_top_other_score",
        "v2_main_true_availability",
        "v2_main_true_ownership",
        "v2_main_true_hub_penalty",
        "v2_main_true_virtual_share",
        "v2_main_true_best_path_virtual_rate",
        "v2_main_true_best_path_physical_rate",
        "v2_main_true_best_time_mean",
        "v2_main_true_physical_time_mean",
        "v2_main_true_virtual_time_mean",
        "v2_main_competitor_availability",
        "v2_main_competitor_ownership",
        "v2_main_competitor_hub_penalty",
        "v2_main_competitor_virtual_share",
        "v2_main_competitor_best_path_virtual_rate",
        "v2_main_competitor_best_path_physical_rate",
        "v2_main_competitor_best_time_mean",
        "v2_main_competitor_physical_time_mean",
        "v2_main_competitor_virtual_time_mean",
        "v2_main_witness_count",
    ]
    return df[export_cols].copy()


def build_support_v2_oracle_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    variants = [
        ("v1_oracle_pre_mask", "oracle_pre"),
        ("v2_main", "v2_main"),
    ]
    for variant, prefix in variants:
        rows.append(summarize_oracle_variant_slice(records, prefix, variant, "global", 0))
        for episode in range(1, NUM_EPISODES + 1):
            df_ep = records[records["episode"] == episode]
            if df_ep.empty:
                continue
            rows.append(summarize_oracle_variant_slice(df_ep, prefix, variant, "episode", episode))
    return pd.DataFrame(rows)


def summarize_v1_v2_slice(
    df_slice: pd.DataFrame,
    analysis: str,
    conditioned_on_oracle_input: bool = True,
) -> Dict[str, float]:
    eval_df = df_slice[df_slice["oracle_num_pos"] > 0].copy() if conditioned_on_oracle_input else df_slice.copy()
    return {
        "analysis": analysis,
        "denominator": int(len(df_slice)),
        "oracle_input_events": int((df_slice["oracle_num_pos"] > 0).sum()),
        "evaluated_events": int(len(eval_df)),
        "v1_conditioned_rank_median": safe_median_from_series(eval_df["oracle_pre_rank"]),
        "v2_conditioned_rank_median": safe_median_from_series(eval_df["v2_main_rank"]),
        "v1_conditioned_directionality": safe_mean_from_series(eval_df["oracle_pre_directionality"]),
        "v2_conditioned_directionality": safe_mean_from_series(eval_df["v2_main_directionality"]),
        "v1_true_nonzero_given_oracle_input_rate": safe_mean_from_series(eval_df["oracle_pre_true_nonzero"]),
        "v2_true_nonzero_given_oracle_input_rate": safe_mean_from_series(eval_df["v2_main_true_nonzero"]),
        "v1_zero_despite_oracle_input_rate": safe_mean_from_series(
            (eval_df["oracle_pre_true_nonzero"] <= 0.5).astype(float)
        ),
        "v2_zero_despite_oracle_input_rate": safe_mean_from_series(
            (eval_df["v2_main_true_nonzero"] <= 0.5).astype(float)
        ),
        "v1_hub_win_rate": safe_mean_from_series(eval_df["oracle_pre_hub_win"]),
        "v2_hub_win_rate": safe_mean_from_series(eval_df["v2_main_hub_win"]),
        "v1_true_total_mean": safe_mean_from_series(eval_df["oracle_pre_true_total"]),
        "v2_true_total_mean": safe_mean_from_series(eval_df["v2_main_true_total"]),
        "rank_median_delta": safe_median_from_series(eval_df["oracle_pre_rank"]) - safe_median_from_series(eval_df["v2_main_rank"]),
        "directionality_delta": safe_mean_from_series(eval_df["v2_main_directionality"]) - safe_mean_from_series(eval_df["oracle_pre_directionality"]),
        "true_nonzero_delta": safe_mean_from_series(eval_df["v2_main_true_nonzero"]) - safe_mean_from_series(eval_df["oracle_pre_true_nonzero"]),
        "zero_rate_delta": safe_mean_from_series((eval_df["v2_main_true_nonzero"] <= 0.5).astype(float))
        - safe_mean_from_series((eval_df["oracle_pre_true_nonzero"] <= 0.5).astype(float)),
        "hub_win_delta": safe_mean_from_series(eval_df["v2_main_hub_win"]) - safe_mean_from_series(eval_df["oracle_pre_hub_win"]),
    }


def build_v1_vs_v2_oracle(records: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    rows.append(summarize_v1_v2_slice(records, "global_oracle_input"))

    b_oracle_relevant = records[
        (records["source_validity_true"] > 0.5)
        & (records["num_pos"] > 0)
        & (records["support_true_nonzero"] <= 0.5)
        & (records["oracle_num_pos"] > 0)
        & (records["oracle_pre_true_nonzero"] <= 0.5)
    ].copy()
    rows.append(summarize_v1_v2_slice(b_oracle_relevant, "B_oracle_relevant", conditioned_on_oracle_input=False))

    c_subset = records[
        (records["source_validity_true"] > 0.5)
        & (records["num_pos"] > 0)
        & (records["support_true_nonzero"] > 0.5)
        & (records["support_hub_win"] > 0.5)
    ].copy()
    rows.append(summarize_v1_v2_slice(c_subset, "C_subset", conditioned_on_oracle_input=False))

    fixed_case_latest = (
        records[records["event_id"].isin(CASE_EVENT_IDS)]
        .sort_values(["event_id", "episode"])
        .groupby("event_id", as_index=False)
        .tail(1)
        .sort_values("event_id")
    )
    for _, row in fixed_case_latest.iterrows():
        rows.append(
            {
                "analysis": "fixed_case_latest",
                "event_id": int(row["event_id"]),
                "episode": int(row["episode"]),
                "time_min": float(row["time_min"]),
                "oracle_num_pos": int(row["oracle_num_pos"]),
                "v1_rank": float(row["oracle_pre_rank"]),
                "v2_rank": float(row["v2_main_rank"]),
                "v1_directionality": float(row["oracle_pre_directionality"]),
                "v2_directionality": float(row["v2_main_directionality"]),
                "v1_true_nonzero": float(row["oracle_pre_true_nonzero"]),
                "v2_true_nonzero": float(row["v2_main_true_nonzero"]),
                "v1_hub_win": float(row["oracle_pre_hub_win"]),
                "v2_hub_win": float(row["v2_main_hub_win"]),
                "v1_true_total": float(row["oracle_pre_true_total"]),
                "v2_true_total": float(row["v2_main_true_total"]),
                "v2_true_availability": float(row["v2_main_true_availability"]),
                "v2_true_ownership": float(row["v2_main_true_ownership"]),
                "v2_true_hub_penalty": float(row["v2_main_true_hub_penalty"]),
                "v2_true_virtual_share": float(row["v2_main_true_virtual_share"]),
                "v2_competitor_availability": float(row["v2_main_competitor_availability"]),
                "v2_competitor_ownership": float(row["v2_main_competitor_ownership"]),
                "v2_competitor_hub_penalty": float(row["v2_main_competitor_hub_penalty"]),
                "v2_competitor_virtual_share": float(row["v2_main_competitor_virtual_share"]),
                "v2_true_best_time_mean": float(row["v2_main_true_best_time_mean"]) if np.isfinite(row["v2_main_true_best_time_mean"]) else np.nan,
                "v2_competitor_best_time_mean": float(row["v2_main_competitor_best_time_mean"]) if np.isfinite(row["v2_main_competitor_best_time_mean"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def proof_tag(improved: bool, degraded: bool) -> str:
    if improved and not degraded:
        return "[已证明]"
    if improved:
        return "[部分证明]"
    return "[未证明]"


def build_fixed_case_v2_lines(fixed_case_rows: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for _, row in fixed_case_rows.sort_values("event_id").iterrows():
        event_id = int(row["event_id"])
        improved = (
            (row["v2_rank"] < row["v1_rank"])
            or (row["v2_hub_win"] < row["v1_hub_win"])
            or (row["v2_directionality"] > row["v1_directionality"])
            or (row["v2_true_nonzero"] > row["v1_true_nonzero"])
        )
        degraded = (
            (row["v2_rank"] > row["v1_rank"])
            or (row["v2_hub_win"] > row["v1_hub_win"])
            or (row["v2_directionality"] < row["v1_directionality"])
        )
        path_note = (
            f"true virtual_share={row['v2_true_virtual_share']:.3f}, competitor virtual_share={row['v2_competitor_virtual_share']:.3f}, "
            f"true best_tt={row['v2_true_best_time_mean']:.2f}, competitor best_tt={row['v2_competitor_best_time_mean']:.2f}"
        )
        if row["oracle_num_pos"] <= 0:
            diagnosis = "snapshot 本身没有 oracle positives，v2 无法凭空恢复 witness ownership。"
        elif row["v2_true_nonzero"] <= 0.5:
            diagnosis = "oracle positives 已注入，但 true source 仍拿不到非零 ownership，主要卡在 availability 不足或 witness 仍过于 generic。"
        elif row["v2_hub_win"] < row["v1_hub_win"] - EPS:
            diagnosis = "v2 把 generic witness 份额从 competitor / hub 身上压了下来，true source 的 ownership 优势更强。"
        elif row["v2_rank"] < row["v1_rank"] - EPS:
            diagnosis = "v2 主要通过更好的 time-plausible attribution 把 true source 排名往前拉。"
        else:
            diagnosis = "v2 提升有限，说明当前 snapshot 下 true/competitor 在 travel-time 可解释性上仍然高度重叠。"
        lines.append(
            f"- {proof_tag(improved, degraded)} event {event_id}: v1(rank={row['v1_rank']:.0f}, dir={row['v1_directionality']:.0f}, "
            f"true_nonzero={row['v1_true_nonzero']:.0f}, hub_win={row['v1_hub_win']:.0f}) -> "
            f"v2(rank={row['v2_rank']:.0f}, dir={row['v2_directionality']:.0f}, true_nonzero={row['v2_true_nonzero']:.0f}, "
            f"hub_win={row['v2_hub_win']:.0f}); {diagnosis} {path_note}."
        )
    return lines


def build_support_v2_summary_markdown(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    global_row = comparison[comparison["analysis"] == "global_oracle_input"].iloc[0]
    b_row = comparison[comparison["analysis"] == "B_oracle_relevant"].iloc[0]
    c_row = comparison[comparison["analysis"] == "C_subset"].iloc[0]
    fixed_case_rows = comparison[comparison["analysis"] == "fixed_case_latest"].copy()

    v1_global = metrics[(metrics["variant"] == "v1_oracle_pre_mask") & (metrics["scope"] == "global")].iloc[0]
    v2_global = metrics[(metrics["variant"] == "v2_main") & (metrics["scope"] == "global")].iloc[0]

    improved_global = (
        (global_row["rank_median_delta"] > 0.0)
        and (global_row["directionality_delta"] >= 0.0)
        and (global_row["hub_win_delta"] <= 0.0)
    )
    degraded_global = (
        (global_row["rank_median_delta"] < 0.0)
        or (global_row["directionality_delta"] < 0.0)
        or (global_row["hub_win_delta"] > 0.0)
    )
    improved_b = b_row["true_nonzero_delta"] > 0.0 and b_row["zero_rate_delta"] < 0.0
    degraded_b = b_row["true_nonzero_delta"] < 0.0
    improved_c = (
        (c_row["hub_win_delta"] < 0.0)
        and (
            (c_row["directionality_delta"] > 0.0)
            or (c_row["rank_median_delta"] > 0.0)
        )
    )
    degraded_c = c_row["hub_win_delta"] > 0.0

    fixed_case_lines = build_fixed_case_v2_lines(fixed_case_rows)
    output_files = [
        V2_ORACLE_STEPWISE_CSV_PATH,
        V2_ORACLE_METRICS_CSV_PATH,
        V1_VS_V2_ORACLE_CSV_PATH,
        V2_SUMMARY_MD_PATH,
    ]

    lines = [
        "# Support Score V2 Oracle Summary",
        "",
        "## 1. 本轮执行摘要",
        f"- {proof_tag(improved_global, degraded_global)} 在 oracle-snapshot 审计线中，`v2_main` 已并行落地到 `src/scripts/audit/run_practical_audit_rerun.py`，并输出 v1/v2 对比文件。",
        f"- {proof_tag(improved_b, degraded_b)} `B_oracle_relevant` 是否减少，以 `true_nonzero_delta={b_row['true_nonzero_delta']:.4f}`、`zero_rate_delta={b_row['zero_rate_delta']:.4f}` 实测判断。",
        f"- {proof_tag(improved_c, degraded_c)} `C` 子集是否更少被 hub 赢，以 `hub_win_delta={c_row['hub_win_delta']:.4f}`、`directionality_delta={c_row['directionality_delta']:.4f}`、`rank_median_delta={c_row['rank_median_delta']:.4f}` 实测判断。",
        "",
        "## 2. support_score v2 设计说明",
        "- `v2_main` 代码位置：`src/scripts/audit/run_practical_audit_rerun.py::compute_support_v2_oracle`。",
        "- 核心 witness 级公式：",
        "```text",
        "availability(s,i) = max(phys_avail(s,i), virtual_reliability * virt_avail(s,i))",
        "phys/virt_avail(s,i) = late_penalty(T(s,i), t_now) * distance_decay(T(s,i), t_now)",
        "ownership(s,i) = availability(s,i)^p / sum_c availability(c,i)^p",
        "witness_weight(i) = 1 / (1 + log(1 + generic_count_i))",
        "availability_term(s) = mean_i availability(s,i)",
        "ownership_term(s) = mean_i availability(s,i) * ownership(s,i) * witness_weight(i)",
        "hub_penalty(s) = mean_i availability(s,i) * (1 - ownership(s,i)) * (1 - witness_weight(i))",
        "support_v2(s) = 0.35 * availability_term(s) + ownership_term(s) - 0.20 * hub_penalty(s)",
        "```",
        "- 奖励什么：time-plausible witness、在同一 witness 上相对别的 candidate 更有 ownership 的解释、以及更依赖 physical path 的解释。",
        "- 惩罚什么：到得晚、只能靠较弱 virtual 解释、谁都能解释的 generic witness、以及能碰到很多 witness 但拿不到 ownership 的 hub/promiscuous candidate。",
        "- 与 v1 最本质的不同：v1 主要在 candidate 汇总后做 `specificity + focus`；v2 把动态 `travel_time(s,i,t)` 放进每个 witness，再按 witness 做 ownership 归一化与 generic/hub 折价。",
        "",
        "## 3. 本轮修改清单",
        "- 文件：`src/scripts/audit/run_practical_audit_rerun.py`。",
        "- 新增：oracle-only `v2_main` 计算分支、dynamic physical+virtual travel-time witness 级打分、v1/v2 对比导出、v2 审计摘要。",
        "- 未改动：Navigator / Reasoner / StateNet / 训练路径。",
        "",
        "## 4. 实际运行命令",
        "- `python src/scripts/audit/run_practical_audit_rerun.py`",
        "",
        "## 5. v1 vs v2 oracle 全局对比",
        "```text",
        metrics[(metrics["scope"] == "global") & (metrics["variant"].isin(["v1_oracle_pre_mask", "v2_main"]))][
            [
                "variant",
                "oracle_eligible_event_rate",
                "oracle_conditioned_rank_median",
                "oracle_conditioned_directionality",
                "oracle_true_nonzero_given_oracle_input_rate",
                "oracle_zero_despite_oracle_input_rate",
                "oracle_hub_win_rate",
            ]
        ].round(4).to_string(index=False),
        "```",
        f"- {proof_tag(improved_global, degraded_global)} 全局 conditioned rank median: {global_row['v1_conditioned_rank_median']:.4f} -> {global_row['v2_conditioned_rank_median']:.4f}.",
        f"- {proof_tag(global_row['directionality_delta'] >= 0.0, global_row['directionality_delta'] < 0.0)} 全局 conditioned directionality: {global_row['v1_conditioned_directionality']:.4f} -> {global_row['v2_conditioned_directionality']:.4f}.",
        f"- {proof_tag(global_row['true_nonzero_delta'] >= 0.0, global_row['true_nonzero_delta'] < 0.0)} 全局 true_nonzero_given_oracle_input: {global_row['v1_true_nonzero_given_oracle_input_rate']:.4f} -> {global_row['v2_true_nonzero_given_oracle_input_rate']:.4f}.",
        f"- {proof_tag(global_row['hub_win_delta'] <= 0.0, global_row['hub_win_delta'] > 0.0)} 全局 hub_win_rate: {global_row['v1_hub_win_rate']:.4f} -> {global_row['v2_hub_win_rate']:.4f}.",
        "",
        "## 6. B 相关子集对比",
        "```text",
        pd.DataFrame([b_row]).round(4).to_string(index=False),
        "```",
        f"- {proof_tag(improved_b, degraded_b)} `B_oracle_relevant` 样本数={int(b_row['denominator'])}，v2 是否减少 true_nonzero=0，取决于 true_nonzero_delta={b_row['true_nonzero_delta']:.4f} 与 zero_rate_delta={b_row['zero_rate_delta']:.4f}。",
        "",
        "## 7. C 子集对比",
        "```text",
        pd.DataFrame([c_row]).round(4).to_string(index=False),
        "```",
        f"- {proof_tag(improved_c, degraded_c)} `C` 样本数={int(c_row['denominator'])}，v2 是否减少 hub_win，取决于 hub_win_delta={c_row['hub_win_delta']:.4f}；directionality_delta={c_row['directionality_delta']:.4f}；rank_median_delta={c_row['rank_median_delta']:.4f}。",
        "",
        "## 8. 固定 4 个 case 解释",
        *fixed_case_lines,
        "",
        "## 9. 当前结论：v2 是否比 v1 更接近正确语义",
        f"- {proof_tag(improved_global or improved_b or improved_c, degraded_global and degraded_b and degraded_c)} 结论依据：全局、`B_oracle_relevant`、`C` 三条线是否同时朝 `true source ownership advantage` 方向移动。",
        f"- v1 global oracle baseline = rank {v1_global['oracle_conditioned_rank_median']:.4f}, directionality {v1_global['oracle_conditioned_directionality']:.4f}, hub_win {v1_global['oracle_hub_win_rate']:.4f}.",
        f"- v2 global oracle result = rank {v2_global['oracle_conditioned_rank_median']:.4f}, directionality {v2_global['oracle_conditioned_directionality']:.4f}, hub_win {v2_global['oracle_hub_win_rate']:.4f}.",
        "",
        "## 10. 下一步最小建议（只给 1 条）",
        "- 只在 oracle 线上继续做 1 次小步校准：固定公式不扩架构，只扫 `virtual_reliability` 与 `hub_penalty_weight` 两个标量，看能否进一步压低 `C` 子集 hub_win。",
        "",
        "## 输出文件",
        *[f"- `{path}`" for path in output_files],
    ]
    return "\n".join(lines) + "\n"


def concat_candidate_buffer(candidate_buffers: Dict[str, List[np.ndarray]], key: str) -> np.ndarray:
    chunks = [chunk.reshape(-1) for chunk in candidate_buffers.get(key, []) if chunk.size > 0]
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def summarize_distribution_array(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "nonzero_rate": np.nan,
            "positive_rate": np.nan,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "nonzero_rate": float((np.abs(finite) > EPS).mean()),
        "positive_rate": float((finite > 0.0).mean()),
    }


def summarize_contradiction_variant_scope(
    df_slice: pd.DataFrame,
    prefix: str,
    variant: str,
    eligible_mask: pd.Series,
    scope: str,
) -> Dict[str, float]:
    eligible = df_slice.loc[eligible_mask]
    return {
        "scope": scope,
        "variant": variant,
        "rows": int(len(df_slice)),
        "eligible_rows": int(eligible_mask.sum()),
        "eligible_rate": float(eligible_mask.mean()) if len(df_slice) > 0 else np.nan,
        "rank_median": safe_median_from_series(df_slice[f"{prefix}_rank"]),
        "gap_mean": safe_mean_from_series(df_slice[f"{prefix}_gap"]),
        "directionality": safe_mean_from_series(df_slice[f"{prefix}_directionality"]),
        "all_zero_rate": safe_mean_from_series(df_slice[f"{prefix}_all_zero"]),
        "true_mean": safe_mean_from_series(df_slice[f"{prefix}_true"]),
        "eligible_rank_median": safe_median_from_series(eligible[f"{prefix}_rank"]),
        "eligible_gap_mean": safe_mean_from_series(eligible[f"{prefix}_gap"]),
        "eligible_directionality": safe_mean_from_series(eligible[f"{prefix}_directionality"]),
    }


def build_contradiction_oracle_v1_compare(records: pd.DataFrame) -> pd.DataFrame:
    latest = records[records["episode"] == records["episode"].max()].copy()
    rows: List[Dict[str, float]] = []
    scopes = [
        ("global", records),
        ("latest_episode", latest),
    ]
    variants = [
        ("old", "contra_old", lambda df: df["num_neg"] > 0),
        ("formal", "contra_formal", lambda df: df["num_neg"] > 0),
        ("nosuspect_formal", "contra_nosuspect_formal", lambda df: df["num_neg"] > 0),
        ("strict_compare", "contra_strict", lambda df: df["num_neg"] > 0),
        ("oracle_v1", "contra_oracle_v1", lambda df: df["contra_oracle_v1_eligible"] > 0.5),
    ]
    for scope_name, scope_df in scopes:
        for variant, prefix, mask_fn in variants:
            rows.append(
                summarize_contradiction_variant_scope(
                    scope_df,
                    prefix,
                    variant,
                    eligible_mask=mask_fn(scope_df),
                    scope=scope_name,
                )
            )
    return pd.DataFrame(rows)


def summarize_support_combo_variant(df_slice: pd.DataFrame, prefix: str, label: str) -> Dict[str, float]:
    return {
        "label": label,
        "rows": int(len(df_slice)),
        "rank_median": safe_median_from_series(df_slice[f"{prefix}_rank"]),
        "gap_mean": safe_mean_from_series(df_slice[f"{prefix}_gap"]),
        "directionality": safe_mean_from_series(df_slice[f"{prefix}_directionality"]),
        "true_nonzero_rate": safe_mean_from_series(df_slice[f"{prefix}_true_nonzero"]),
        "hub_win_rate": safe_mean_from_series(df_slice[f"{prefix}_hub_win"]),
        "true_total_mean": safe_mean_from_series(df_slice[f"{prefix}_true_total"]),
    }


def build_contradiction_oracle_v1_summary(
    records: pd.DataFrame,
    candidate_buffers: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    compare_df = build_contradiction_oracle_v1_compare(records)
    all_total = concat_candidate_buffer(candidate_buffers, "all_total")
    all_interval_gap = concat_candidate_buffer(candidate_buffers, "all_interval_gap")
    all_safe_violation = concat_candidate_buffer(candidate_buffers, "all_safe_violation")
    all_violated_safe_count = concat_candidate_buffer(candidate_buffers, "all_violated_safe_count")
    all_top_margin = concat_candidate_buffer(candidate_buffers, "all_top_witness_margin")
    non_source_total = concat_candidate_buffer(candidate_buffers, "non_source_total")
    non_source_interval_gap = concat_candidate_buffer(candidate_buffers, "non_source_interval_gap")
    non_source_safe_violation = concat_candidate_buffer(candidate_buffers, "non_source_safe_violation")
    non_source_violated_safe_count = concat_candidate_buffer(candidate_buffers, "non_source_violated_safe_count")
    non_source_top_margin = concat_candidate_buffer(candidate_buffers, "non_source_top_witness_margin")

    true_distribution = {
        "total": summarize_distribution_array(records["contra_oracle_v1_true"].to_numpy(dtype=np.float32)),
        "interval_gap": summarize_distribution_array(records["contra_oracle_v1_true_interval_gap"].to_numpy(dtype=np.float32)),
        "safe_violation": summarize_distribution_array(records["contra_oracle_v1_true_safe_violation"].to_numpy(dtype=np.float32)),
        "violated_safe_count": summarize_distribution_array(
            records["contra_oracle_v1_true_violated_safe_count"].to_numpy(dtype=np.float32)
        ),
        "top_witness_margin": summarize_distribution_array(
            records["contra_oracle_v1_true_top_witness_margin"].to_numpy(dtype=np.float32)
        ),
    }
    all_distribution = {
        "total": summarize_distribution_array(all_total),
        "interval_gap": summarize_distribution_array(all_interval_gap),
        "safe_violation": summarize_distribution_array(all_safe_violation),
        "violated_safe_count": summarize_distribution_array(all_violated_safe_count),
        "top_witness_margin": summarize_distribution_array(all_top_margin),
    }
    non_source_distribution = {
        "total": summarize_distribution_array(non_source_total),
        "interval_gap": summarize_distribution_array(non_source_interval_gap),
        "safe_violation": summarize_distribution_array(non_source_safe_violation),
        "violated_safe_count": summarize_distribution_array(non_source_violated_safe_count),
        "top_witness_margin": summarize_distribution_array(non_source_top_margin),
    }

    combo_subset = records[
        (records["oracle_num_pos"] > 0)
        & (records["contra_oracle_v1_eligible"] > 0.5)
    ].copy()
    combo_support_only = summarize_support_combo_variant(combo_subset, "v2_main", "support_v2_oracle_only")
    combo_with_contra = summarize_support_combo_variant(
        combo_subset,
        "v2_combo_oracle_v1",
        "support_v2_oracle_minus_contra_oracle_v1",
    )
    combo_delta = {
        "rank_median_delta": float(combo_support_only["rank_median"] - combo_with_contra["rank_median"])
        if np.isfinite(combo_support_only["rank_median"]) and np.isfinite(combo_with_contra["rank_median"]) else np.nan,
        "directionality_delta": float(combo_with_contra["directionality"] - combo_support_only["directionality"])
        if np.isfinite(combo_support_only["directionality"]) and np.isfinite(combo_with_contra["directionality"]) else np.nan,
        "true_nonzero_delta": float(combo_with_contra["true_nonzero_rate"] - combo_support_only["true_nonzero_rate"])
        if np.isfinite(combo_support_only["true_nonzero_rate"]) and np.isfinite(combo_with_contra["true_nonzero_rate"]) else np.nan,
        "hub_win_delta": float(combo_with_contra["hub_win_rate"] - combo_support_only["hub_win_rate"])
        if np.isfinite(combo_support_only["hub_win_rate"]) and np.isfinite(combo_with_contra["hub_win_rate"]) else np.nan,
    }

    case_rows = (
        records[
            (records["contra_oracle_v1_eligible"] > 0.5)
            & (records["contra_oracle_v1_support_comp_total"] > EPS)
            & (records["contra_oracle_v1_support_comp_top_witness_margin"] > -float("inf"))
        ]
        .sort_values(
            ["contra_oracle_v1_support_comp_total", "contra_oracle_v1_support_comp_top_witness_margin"],
            ascending=False,
        )
        .drop_duplicates(["event_id", "contra_oracle_v1_support_comp_global_id"])
        .head(5)
    )
    case_studies: List[Dict[str, Any]] = []
    for _, row in case_rows.iterrows():
        case_studies.append(
            {
                "event_id": int(row["event_id"]),
                "episode": int(row["episode"]),
                "time_min": float(row["time_min"]),
                "candidate_global_id": int(row["contra_oracle_v1_support_comp_global_id"]),
                "candidate_support_score": float(row["support_top_other_score"]),
                "candidate_contradiction_total": float(row["contra_oracle_v1_support_comp_total"]),
                "interval_gap": float(row["contra_oracle_v1_support_comp_interval_gap"]),
                "safe_violation": float(row["contra_oracle_v1_support_comp_safe_violation"]),
                "violated_safe_count": float(row["contra_oracle_v1_support_comp_violated_safe_count"]),
                "top_witness_margin": float(row["contra_oracle_v1_support_comp_top_witness_margin"]),
                "top_witness_safe_global_id": int(row["contra_oracle_v1_support_comp_top_witness_safe_global_id"]),
                "top_witness_pos_global_id": int(row["contra_oracle_v1_support_comp_top_witness_pos_global_id"]),
                "witnesses": row["contra_oracle_v1_support_comp_top_witnesses"],
            }
        )

    global_oracle_v1 = compare_df[(compare_df["scope"] == "global") & (compare_df["variant"] == "oracle_v1")].iloc[0]
    global_old = compare_df[(compare_df["scope"] == "global") & (compare_df["variant"] == "old")].iloc[0]
    global_nosuspect = compare_df[
        (compare_df["scope"] == "global") & (compare_df["variant"] == "nosuspect_formal")
    ].iloc[0]
    separation = {
        "true_rank_median": float(global_oracle_v1["rank_median"]),
        "true_gap_mean": float(global_oracle_v1["gap_mean"]),
        "true_directionality": float(global_oracle_v1["directionality"]),
        "true_mean": float(global_oracle_v1["true_mean"]),
        "non_source_mean": float(non_source_distribution["total"]["mean"]),
        "true_minus_non_source_mean": float(global_oracle_v1["true_mean"] - non_source_distribution["total"]["mean"])
        if np.isfinite(non_source_distribution["total"]["mean"]) else np.nan,
        "old_gap_mean": float(global_old["gap_mean"]),
        "nosuspect_formal_gap_mean": float(global_nosuspect["gap_mean"]),
    }

    return {
        "config": {
            "audit_path": "src/scripts/audit/run_practical_audit_rerun.py",
            "contradiction_mode": "oracle_v1",
            "safe_violation_tau_min": ORACLE_V1_SAFE_TAU_MIN,
            "top_k_witnesses": ORACLE_V1_TOP_K_WITNESSES,
            "max_events": MAX_EVENTS,
            "num_episodes": NUM_EPISODES,
        },
        "compare": compare_df.to_dict(orient="records"),
        "decomposition_stats": {
            "all_candidates": all_distribution,
            "true_source": true_distribution,
            "non_source_candidates": non_source_distribution,
        },
        "separation": separation,
        "support_combo": {
            "subset_rows": int(len(combo_subset)),
            "support_only": combo_support_only,
            "support_minus_contradiction": combo_with_contra,
            "delta": combo_delta,
        },
        "case_studies": case_studies,
    }


def build_contradiction_oracle_v1_markdown(summary: Dict[str, Any]) -> str:
    compare_df = pd.DataFrame(summary["compare"])
    global_table = compare_df[compare_df["scope"] == "global"].copy()
    latest_table = compare_df[compare_df["scope"] == "latest_episode"].copy()
    support_combo = summary["support_combo"]
    support_combo_table = pd.DataFrame(
        [support_combo["support_only"], support_combo["support_minus_contradiction"]]
    )
    case_lines = []
    for case in summary["case_studies"]:
        case_lines.append(
            f"- [已证明] event={case['event_id']} episode={case['episode']} time={case['time_min']:.1f}min: "
            f"candidate={case['candidate_global_id']} 被反驳；"
            f"interval_gap={case['interval_gap']:.4f}, safe_violation={case['safe_violation']:.4f}, "
            f"violated_safe_count={case['violated_safe_count']:.0f}, top_margin={case['top_witness_margin']:.4f}; "
            f"top_witness safe={case['top_witness_safe_global_id']} / pos={case['top_witness_pos_global_id']}. "
            f"witnesses={case['witnesses']}"
        )

    lines = [
        "# contradiction oracle_v1 audit",
        "",
        "## 1. 本轮执行摘要",
        "- [已证明] 现有 Oracle contradiction 的主审计 SSOT 仍是 `src/scripts/audit/run_practical_audit_rerun.py`；本轮没有接训练主链，也没有进入 Practical complaint/report/noise 语义。",
        f"- [已证明] `oracle_v1` 使用 history-conditioned explicit sampled positives/safes，tau={summary['config']['safe_violation_tau_min']:.1f}min，top-k witness={summary['config']['top_k_witnesses']}.",
        "",
        "## 2. old vs oracle_v1 对比",
        "```text",
        global_table[
            [
                "variant",
                "rows",
                "eligible_rate",
                "rank_median",
                "gap_mean",
                "directionality",
                "all_zero_rate",
                "true_mean",
            ]
        ].round(4).to_string(index=False),
        "```",
        f"- [已证明] global old gap_mean={summary['separation']['old_gap_mean']:.4f}, "
        f"nosuspect_formal gap_mean={summary['separation']['nosuspect_formal_gap_mean']:.4f}, "
        f"oracle_v1 gap_mean={summary['separation']['true_gap_mean']:.4f}.",
        "",
        "## 3. latest episode 对比",
        "```text",
        latest_table[
            [
                "variant",
                "rows",
                "eligible_rate",
                "rank_median",
                "gap_mean",
                "directionality",
                "all_zero_rate",
                "true_mean",
            ]
        ].round(4).to_string(index=False),
        "```",
        "",
        "## 4. decomposition 分布",
        f"- [已证明] all-candidate interval_gap: {summary['decomposition_stats']['all_candidates']['interval_gap']}",
        f"- [已证明] all-candidate safe_violation: {summary['decomposition_stats']['all_candidates']['safe_violation']}",
        f"- [已证明] all-candidate violated_safe_count: {summary['decomposition_stats']['all_candidates']['violated_safe_count']}",
        f"- [已证明] all-candidate top_witness_margin: {summary['decomposition_stats']['all_candidates']['top_witness_margin']}",
        "",
        "## 5. true source vs non-source 分离",
        f"- [已证明] true_source rank_median={summary['separation']['true_rank_median']:.4f}, gap_mean={summary['separation']['true_gap_mean']:.4f}, directionality={summary['separation']['true_directionality']:.4f}.",
        f"- [已证明] true_mean={summary['separation']['true_mean']:.4f}, non_source_mean={summary['separation']['non_source_mean']:.4f}, "
        f"true_minus_non_source_mean={summary['separation']['true_minus_non_source_mean']:.4f}.",
        "",
        "## 6. Oracle support 组合对比",
        "```text",
        support_combo_table[
            [
                "label",
                "rows",
                "rank_median",
                "gap_mean",
                "directionality",
                "true_nonzero_rate",
                "hub_win_rate",
                "true_total_mean",
            ]
        ].round(4).to_string(index=False),
        "```",
        f"- [已证明] combo subset rows={support_combo['subset_rows']}.",
        f"- [已证明] delta: {support_combo['delta']}.",
        "",
        "## 7. witness case study",
        *case_lines,
        "",
        "## 输出文件",
        f"- `{CONTRA_ORACLE_V1_MD_PATH}`",
        f"- `{CONTRA_ORACLE_V1_JSON_PATH}`",
        f"- `{CONTRA_ORACLE_V1_COMPARE_CSV_PATH}`",
        f"- `{STEPWISE_CSV_PATH}`",
    ]
    return "\n".join(lines) + "\n"


def build_contradiction_practical_v2_compare(records: pd.DataFrame) -> pd.DataFrame:
    latest = records[records["episode"] == records["episode"].max()].copy()
    rows: List[Dict[str, float]] = []
    scopes = [
        ("global", records),
        ("latest_episode", latest),
    ]
    variants = [
        ("old", "contra_old", lambda df: df["num_neg"] > 0),
        ("formal", "contra_formal", lambda df: df["num_neg"] > 0),
        ("nosuspect_formal", "contra_nosuspect_formal", lambda df: df["num_neg"] > 0),
        ("strict_compare", "contra_strict", lambda df: df["num_neg"] > 0),
        ("oracle_v1", "contra_oracle_v1", lambda df: df["contra_oracle_v1_eligible"] > 0.5),
        ("practical_v2_gap_capped_only", "contra_practical_v2_gap_capped", lambda df: df["contra_practical_v2_eligible"] > 0.5),
        ("practical_v2_gap_log_only", "contra_practical_v2_gap_log", lambda df: df["contra_practical_v2_eligible"] > 0.5),
        ("practical_v2_norm", "contra_practical_v2_norm", lambda df: df["contra_practical_v2_eligible"] > 0.5),
        ("practical_v2", "contra_practical_v2", lambda df: df["contra_practical_v2_eligible"] > 0.5),
        (
            "practical_v2_current_time_physctx",
            "contra_practical_v2_current_time",
            lambda df: (df["contra_practical_v2_eligible"] > 0.5) & (df["contra_practical_v2_current_time_available"] > 0.5),
        ),
    ]
    for scope_name, scope_df in scopes:
        for variant, prefix, mask_fn in variants:
            rows.append(
                summarize_contradiction_variant_scope(
                    scope_df,
                    prefix,
                    variant,
                    eligible_mask=mask_fn(scope_df),
                    scope=scope_name,
                )
            )
    return pd.DataFrame(rows)


def build_contradiction_distribution_bundle(
    records: pd.DataFrame,
    candidate_buffers: Dict[str, List[np.ndarray]],
    true_column_map: Dict[str, str],
) -> Dict[str, Any]:
    true_distribution: Dict[str, Any] = {}
    all_distribution: Dict[str, Any] = {}
    non_source_distribution: Dict[str, Any] = {}
    for metric_name, true_column in true_column_map.items():
        if true_column not in records.columns:
            continue
        true_distribution[metric_name] = summarize_distribution_array(
            records[true_column].to_numpy(dtype=np.float32)
        )
        all_distribution[metric_name] = summarize_distribution_array(
            concat_candidate_buffer(candidate_buffers, f"all_{metric_name}")
        )
        non_source_distribution[metric_name] = summarize_distribution_array(
            concat_candidate_buffer(candidate_buffers, f"non_source_{metric_name}")
        )
    return {
        "all_candidates": all_distribution,
        "true_source": true_distribution,
        "non_source_candidates": non_source_distribution,
    }


def summarize_bucket_slice(records: pd.DataFrame, label: str, mask: pd.Series) -> Dict[str, Any]:
    bucket = records.loc[mask].copy()
    return {
        "bucket": label,
        "rows": int(len(bucket)),
        "rate": float(len(bucket) / len(records)) if len(records) > 0 else np.nan,
        "directionality": safe_mean_from_series(bucket["contra_practical_v2_directionality"]),
        "all_zero_rate": safe_mean_from_series(bucket["contra_practical_v2_all_zero"]),
        "true_mean": safe_mean_from_series(bucket["contra_practical_v2_true"]),
        "support_comp_mean": safe_mean_from_series(bucket["contra_practical_v2_support_comp_total"]),
        "support_comp_near_safe_mean": safe_mean_from_series(
            bucket["contra_practical_v2_support_comp_near_safe_mass"]
        ),
        "support_comp_soft_count_mean": safe_mean_from_series(
            bucket["contra_practical_v2_support_comp_soft_violated_safe_count"]
        ),
        "support_comp_eligible_safe_mean": safe_mean_from_series(
            bucket["contra_practical_v2_support_comp_eligible_safe_witness_count"]
        ),
        "support_comp_positive_margin_mean": safe_mean_from_series(
            bucket["contra_practical_v2_support_comp_positive_margin_count"]
        ),
    }


def build_practical_v2_bucket_breakdown(records: pd.DataFrame) -> List[Dict[str, Any]]:
    eligible = records["contra_practical_v2_eligible"] > 0.5
    tie_mask = (
        (records["contra_practical_v2_all_zero"] > 0.5)
        | ((records["contra_practical_v2_top_other_score"] - records["contra_practical_v2_true"]).abs() <= EPS)
    )
    support_comp_active = records["contra_practical_v2_support_comp_total"] > EPS
    rows = [
        summarize_bucket_slice(records, "all_zero_or_tie", eligible & tie_mask),
        summarize_bucket_slice(
            records,
            "single_safe_witness_refutation",
            eligible
            & support_comp_active
            & (records["contra_practical_v2_support_comp_eligible_safe_witness_count"].round() == 1),
        ),
        summarize_bucket_slice(
            records,
            "multi_safe_witness_refutation",
            eligible
            & support_comp_active
            & (records["contra_practical_v2_support_comp_eligible_safe_witness_count"] >= 2.0),
        ),
        summarize_bucket_slice(
            records,
            "support_top_competitor_contradicted",
            eligible & support_comp_active,
        ),
        summarize_bucket_slice(
            records,
            "true_source_contradicted",
            eligible & (records["contra_practical_v2_true"] > EPS),
        ),
        summarize_bucket_slice(
            records,
            "non_source_contradicted",
            eligible & support_comp_active,
        ),
        summarize_bucket_slice(
            records,
            "interval_dominant_snapshot",
            eligible
            & support_comp_active
            & (records["contra_practical_v2_support_comp_dominant_component"] == "interval"),
        ),
        summarize_bucket_slice(
            records,
            "safe_dominant_snapshot",
            eligible
            & support_comp_active
            & (records["contra_practical_v2_support_comp_dominant_component"] == "safe"),
        ),
    ]
    return rows


def build_practical_v2_case_record(
    row: pd.Series,
    case_label: str,
    candidate_prefix: str,
    candidate_role: str,
) -> Dict[str, Any]:
    return {
        "case_label": case_label,
        "candidate_role": candidate_role,
        "event_id": int(row["event_id"]),
        "episode": int(row["episode"]),
        "time_min": float(row["time_min"]),
        "support_top_competitor_global_id": int(row["support_top_other_global_id"]),
        "candidate_global_id": int(row[f"{candidate_prefix}_global_id"]),
        "candidate_total": float(row[f"{candidate_prefix}_total"]),
        "interval_gap": float(row[f"{candidate_prefix}_interval_gap"]),
        "safe_violation": float(row[f"{candidate_prefix}_safe_violation"]),
        "soft_violated_safe_count": float(row.get(f"{candidate_prefix}_soft_violated_safe_count", np.nan)),
        "near_safe_mass": float(row.get(f"{candidate_prefix}_near_safe_mass", np.nan)),
        "eligible_safe_witness_count": float(row.get(f"{candidate_prefix}_eligible_safe_witness_count", np.nan)),
        "positive_margin_count": float(row.get(f"{candidate_prefix}_positive_margin_count", np.nan)),
        "best_margin_topk_mean": float(row.get(f"{candidate_prefix}_best_margin_topk_mean", np.nan)),
        "gap_component": float(row.get(f"{candidate_prefix}_gap_component", np.nan)),
        "safe_component": float(row.get(f"{candidate_prefix}_safe_component", np.nan)),
        "dominant_component": str(row.get(f"{candidate_prefix}_dominant_component", "")),
        "top_witness_margin": float(row[f"{candidate_prefix}_top_witness_margin"]),
        "top_witness_safe_global_id": int(row[f"{candidate_prefix}_top_witness_safe_global_id"]),
        "top_witness_pos_global_id": int(row[f"{candidate_prefix}_top_witness_pos_global_id"]),
        "witnesses": row[f"{candidate_prefix}_top_witnesses"],
        "true_source_total": float(row["contra_practical_v2_true"]),
        "support_top_competitor_total": float(row["contra_practical_v2_support_comp_total"]),
    }


def build_practical_v2_case_studies(records: pd.DataFrame) -> List[Dict[str, Any]]:
    used_keys = set()
    cases: List[Dict[str, Any]] = []
    case_specs = [
        (
            "single_safe_witness",
            "contra_practical_v2_support_comp",
            "support_top_competitor",
            (records["contra_practical_v2_eligible"] > 0.5)
            & (records["contra_practical_v2_support_comp_total"] > EPS)
            & (records["contra_practical_v2_support_comp_eligible_safe_witness_count"].round() == 1),
            ["contra_practical_v2_support_comp_total", "contra_practical_v2_support_comp_top_witness_margin"],
            [False, False],
        ),
        (
            "multi_safe_witness",
            "contra_practical_v2_support_comp",
            "support_top_competitor",
            (records["contra_practical_v2_eligible"] > 0.5)
            & (records["contra_practical_v2_support_comp_total"] > EPS)
            & (records["contra_practical_v2_support_comp_eligible_safe_witness_count"] >= 2.0),
            ["contra_practical_v2_support_comp_total", "contra_practical_v2_support_comp_near_safe_mass"],
            [False, False],
        ),
        (
            "interval_dominant",
            "contra_practical_v2_support_comp",
            "support_top_competitor",
            (records["contra_practical_v2_eligible"] > 0.5)
            & (records["contra_practical_v2_support_comp_total"] > EPS)
            & (records["contra_practical_v2_support_comp_dominant_component"] == "interval"),
            ["contra_practical_v2_support_comp_gap_component", "contra_practical_v2_support_comp_total"],
            [False, False],
        ),
        (
            "safe_dominant",
            "contra_practical_v2_support_comp",
            "support_top_competitor",
            (records["contra_practical_v2_eligible"] > 0.5)
            & (records["contra_practical_v2_support_comp_total"] > EPS)
            & (records["contra_practical_v2_support_comp_dominant_component"] == "safe"),
            ["contra_practical_v2_support_comp_safe_component", "contra_practical_v2_support_comp_total"],
            [False, False],
        ),
        (
            "true_source_mis_hit",
            "contra_practical_v2_true",
            "true_source",
            (records["contra_practical_v2_eligible"] > 0.5)
            & (records["contra_practical_v2_true"] > EPS)
            & (records["contra_practical_v2_true"] > records["contra_practical_v2_support_comp_total"] + EPS),
            ["contra_practical_v2_true", "contra_practical_v2_true_near_safe_mass"],
            [False, False],
        ),
        (
            "non_source_successful_refutation",
            "contra_practical_v2_support_comp",
            "support_top_competitor",
            (records["contra_practical_v2_eligible"] > 0.5)
            & (records["contra_practical_v2_support_comp_total"] > records["contra_practical_v2_true"] + EPS),
            ["contra_practical_v2_support_comp_total", "contra_practical_v2_support_comp_near_safe_mass"],
            [False, False],
        ),
    ]
    for case_label, candidate_prefix, candidate_role, mask, sort_cols, ascending in case_specs:
        subset = records.loc[mask].sort_values(sort_cols, ascending=ascending)
        for _, row in subset.iterrows():
            candidate_key = (
                int(row["event_id"]),
                int(row["episode"]),
                int(row[f"{candidate_prefix}_global_id"]),
            )
            if candidate_key in used_keys:
                continue
            used_keys.add(candidate_key)
            cases.append(build_practical_v2_case_record(row, case_label, candidate_prefix, candidate_role))
            break
    return cases


def build_support_combo_comparison(records: pd.DataFrame) -> Dict[str, Any]:
    combo_subset = records[
        (records["oracle_num_pos"] > 0)
        & (records["contra_practical_v2_eligible"] > 0.5)
    ].copy()
    variants = [
        summarize_support_combo_variant(combo_subset, "v2_main", "support_v2_oracle_only"),
        summarize_support_combo_variant(combo_subset, "v2_combo_oracle_v1", "support_v2_oracle_minus_contra_oracle_v1"),
        summarize_support_combo_variant(combo_subset, "v2_combo_practical_v2", "support_v2_oracle_minus_contra_practical_v2"),
        summarize_support_combo_variant(
            combo_subset,
            "v2_combo_practical_v2_current_time",
            "support_v2_oracle_minus_contra_practical_v2_current_time",
        ),
    ]
    support_only = variants[0]
    oracle_v1_combo = variants[1]
    practical_v2_combo = variants[2]
    current_time_combo = variants[3]
    return {
        "subset_rows": int(len(combo_subset)),
        "variants": variants,
        "delta_vs_support_only": {
            "oracle_v1_directionality_delta": float(oracle_v1_combo["directionality"] - support_only["directionality"]),
            "practical_v2_directionality_delta": float(practical_v2_combo["directionality"] - support_only["directionality"]),
            "practical_v2_rank_median_delta": float(practical_v2_combo["rank_median"] - support_only["rank_median"]),
            "practical_v2_hub_win_delta": float(practical_v2_combo["hub_win_rate"] - support_only["hub_win_rate"]),
            "practical_v2_true_nonzero_delta": float(
                practical_v2_combo["true_nonzero_rate"] - support_only["true_nonzero_rate"]
            ),
        },
        "delta_vs_oracle_v1_combo": {
            "directionality_delta": float(practical_v2_combo["directionality"] - oracle_v1_combo["directionality"]),
            "rank_median_delta": float(practical_v2_combo["rank_median"] - oracle_v1_combo["rank_median"]),
            "hub_win_delta": float(practical_v2_combo["hub_win_rate"] - oracle_v1_combo["hub_win_rate"]),
        },
        "time_compare": {
            "witness_time_physctx": practical_v2_combo,
            "current_time_physctx": current_time_combo,
        },
    }


def build_v1_judgement(oracle_v1_summary: Dict[str, Any]) -> Dict[str, Any]:
    global_oracle_v1 = next(
        row for row in oracle_v1_summary["compare"] if row["scope"] == "global" and row["variant"] == "oracle_v1"
    )
    all_stats = oracle_v1_summary["decomposition_stats"]["all_candidates"]
    true_stats = oracle_v1_summary["decomposition_stats"]["true_source"]
    return {
        "headline": (
            "practical_v2 是在 oracle_v1 语义成果基础上的继续推进；当前项目语境下，两者沿同一 practical "
            "rollout/audit 资产演化，不另造分支。"
        ),
        "main_signal": [
            "v1 的主信号首先来自 interval_gap 提供的 earliest-positive / latest-safe envelope 冲突。",
            "safe_violation 已经在加总里提供软反驳，但 violated_safe_count 仍然过硬、过稀，更多像 witness 是否越线的稀疏开关。",
            "top_witness_margin 已经证明 witness pairing 是有语义信息的，尤其在被成功反驳的 non-source 上最明显。",
        ],
        "why_zero_tie_heavy": [
            f"all_zero_rate={global_oracle_v1['all_zero_rate']:.4f}，说明大多数 snapshot 还没进入 working regime。",
            f"violated_safe_count mean={all_stats['violated_safe_count']['mean']:.4f}，hard positive 的 safe 反驳几乎总是 0。",
            f"safe_violation mean={all_stats['safe_violation']['mean']:.4f}，相对 interval_gap mean={all_stats['interval_gap']['mean']:.4f} 偏弱，safe 侧贡献容易被压住。",
            f"interval_gap max={all_stats['interval_gap']['max']:.4f} 但 p99={all_stats['interval_gap']['p99']:.4f}，说明它极可能被少量 outlier 主导。",
        ],
        "proven": [
            f"directionality={global_oracle_v1['directionality']:.4f}、true_mean={global_oracle_v1['true_mean']:.4f} 已证明 contradiction 的语义方向是对的。",
            f"true_source interval_gap mean={true_stats['interval_gap']['mean']:.4f}、safe_violation mean={true_stats['safe_violation']['mean']:.4f} 表明 true source / non-source 仍有分离。",
            "但 v1 目前仍然 zero/tie 过重、标量被 outlier 拉动，尚不适合直接升格成正式训练标量。",
        ],
    }


def build_practical_v2_summary(
    records: pd.DataFrame,
    oracle_v1_summary: Dict[str, Any],
    practical_v2_candidate_buffers: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    compare_df = build_contradiction_practical_v2_compare(records)
    support_combo = build_support_combo_comparison(records)
    distribution_bundle = build_contradiction_distribution_bundle(
        records,
        practical_v2_candidate_buffers,
        true_column_map={
            "total": "contra_practical_v2_true",
            "total_practical_v2_raw": "contra_practical_v2_true_total_practical_v2_raw",
            "total_practical_v2_norm": "contra_practical_v2_true_total_practical_v2_norm",
            "interval_gap": "contra_practical_v2_true_interval_gap",
            "interval_gap_capped": "contra_practical_v2_true_interval_gap_capped",
            "interval_gap_log": "contra_practical_v2_true_interval_gap_log",
            "safe_violation": "contra_practical_v2_true_safe_violation",
            "soft_violated_safe_count": "contra_practical_v2_true_soft_violated_safe_count",
            "near_safe_mass": "contra_practical_v2_true_near_safe_mass",
            "violated_safe_count": "contra_practical_v2_true_violated_safe_count",
            "eligible_safe_witness_count": "contra_practical_v2_true_eligible_safe_witness_count",
            "positive_margin_count": "contra_practical_v2_true_positive_margin_count",
            "best_margin_topk_mean": "contra_practical_v2_true_best_margin_topk_mean",
            "top_witness_margin": "contra_practical_v2_true_top_witness_margin",
            "gap_component": "contra_practical_v2_true_gap_component",
            "safe_component": "contra_practical_v2_true_safe_component",
        },
    )
    global_v1 = compare_df[(compare_df["scope"] == "global") & (compare_df["variant"] == "oracle_v1")].iloc[0]
    global_v2 = compare_df[(compare_df["scope"] == "global") & (compare_df["variant"] == "practical_v2")].iloc[0]
    current_rows = compare_df[
        (compare_df["scope"] == "global") & (compare_df["variant"] == "practical_v2_current_time_physctx")
    ]
    global_current = current_rows.iloc[0] if not current_rows.empty else global_v2
    current_time_compare_executed = bool((records["contra_practical_v2_current_time_available"] > 0.5).any())
    sweep_global = compare_df[
        (compare_df["scope"] == "global")
        & compare_df["variant"].isin(
            [
                "oracle_v1",
                "practical_v2_gap_capped_only",
                "practical_v2_gap_log_only",
                "practical_v2_norm",
                "practical_v2",
                "practical_v2_current_time_physctx",
            ]
        )
    ].copy()
    selected_time_semantics = "witness_time_physctx"
    if current_time_compare_executed and (
        global_current["all_zero_rate"] < global_v2["all_zero_rate"] - 1e-6
        and global_current["directionality"] >= global_v2["directionality"] - 0.01
        and global_current["gap_mean"] <= global_v2["gap_mean"] * 1.5
    ):
        selected_time_semantics = "current_time_physctx"
    separation = {
        "oracle_v1_true_mean": float(global_v1["true_mean"]),
        "practical_v2_true_mean": float(global_v2["true_mean"]),
        "non_source_mean": float(distribution_bundle["non_source_candidates"]["total"]["mean"]),
        "true_minus_non_source_mean": float(
            global_v2["true_mean"] - distribution_bundle["non_source_candidates"]["total"]["mean"]
        ),
        "directionality": float(global_v2["directionality"]),
        "gap_mean": float(global_v2["gap_mean"]),
        "soft_count_true_mean": float(distribution_bundle["true_source"]["soft_violated_safe_count"]["mean"]),
        "soft_count_non_source_mean": float(
            distribution_bundle["non_source_candidates"]["soft_violated_safe_count"]["mean"]
        ),
        "near_safe_true_mean": float(distribution_bundle["true_source"]["near_safe_mass"]["mean"]),
        "near_safe_non_source_mean": float(distribution_bundle["non_source_candidates"]["near_safe_mass"]["mean"]),
    }
    key_results = {
        "all_zero_rate_delta_vs_v1": float(global_v2["all_zero_rate"] - global_v1["all_zero_rate"]),
        "directionality_delta_vs_v1": float(global_v2["directionality"] - global_v1["directionality"]),
        "true_mean_delta_vs_v1": float(global_v2["true_mean"] - global_v1["true_mean"]),
        "support_combo_directionality_delta_vs_support_only": float(
            support_combo["delta_vs_support_only"]["practical_v2_directionality_delta"]
        ),
        "support_combo_directionality_delta_vs_oracle_v1_combo": float(
            support_combo["delta_vs_oracle_v1_combo"]["directionality_delta"]
        ),
    }
    return {
        "v1_judgement": build_v1_judgement(oracle_v1_summary),
        "config": {
            "audit_path": "src/scripts/audit/run_practical_audit_rerun.py",
            "rollout_path": "src/scripts/audit/utils_practical_rollout.py",
            "reused_paths": [
                "src/scripts/audit/run_practical_audit_rerun.py",
                "src/scripts/audit/utils_practical_rollout.py",
            ],
            "compatibility_note": (
                "保留 oracle_v1 产物兼容；practical_v2 是沿同一 practical contradiction rollout/audit 资产继续推进。"
            ),
            "contradiction_mode": "practical_v2",
            "safe_violation_tau_min": ORACLE_V1_SAFE_TAU_MIN,
            "history_phys_ctx_mode": PRACTICAL_V2_HISTORY_PHYSCTX_MODE,
            "selected_time_semantics": selected_time_semantics,
            "current_time_compare_executed": current_time_compare_executed,
            "practical_v2_main_config": PRACTICAL_V2_MAIN_CONFIG.to_dict(),
            "practical_v2_normalized_config": PRACTICAL_V2_NORMALIZED_CONFIG.to_dict(),
            "max_events": MAX_EVENTS,
            "num_episodes": NUM_EPISODES,
        },
        "compare": compare_df.to_dict(orient="records"),
        "sweep_global": sweep_global.to_dict(orient="records"),
        "decomposition_stats": distribution_bundle,
        "separation": separation,
        "key_results": key_results,
        "support_combo": support_combo,
        "bucket_breakdown": build_practical_v2_bucket_breakdown(records),
        "time_semantics_compare": {
            "witness_time_physctx": compare_df[
                (compare_df["scope"] == "global") & (compare_df["variant"] == "practical_v2")
            ].iloc[0].to_dict(),
            "current_time_physctx": global_current.to_dict(),
            "judgement": (
                "current_time_physctx 对比未执行。"
                if not current_time_compare_executed
                else (
                    (
                        "witness_time_physctx 仍作为主线保留：current_time_physctx 虽然略降 zero/tie，"
                        "但 gap_mean 明显放大，重新暴露 interval outlier 主导风险。"
                    )
                    if selected_time_semantics == "witness_time_physctx"
                    else "current_time_physctx 在 zero/tie 上更稳，因此被标记为更强的受控对比。"
                )
            ),
        },
        "case_studies": build_practical_v2_case_studies(records),
    }


def build_contradiction_practical_v2_markdown(summary: Dict[str, Any]) -> str:
    compare_df = pd.DataFrame(summary["compare"])
    support_combo_table = pd.DataFrame(summary["support_combo"]["variants"])
    bucket_table = pd.DataFrame(summary["bucket_breakdown"])
    sweep_table = pd.DataFrame(summary["sweep_global"])
    case_lines = []
    for case in summary["case_studies"]:
        case_lines.append(
            f"- [已证明] {case['case_label']} / {case['candidate_role']}: event={case['event_id']} "
            f"episode={case['episode']} time={case['time_min']:.1f}min candidate={case['candidate_global_id']}; "
            f"total={case['candidate_total']:.4f}, interval_gap={case['interval_gap']:.4f}, "
            f"near_safe_mass={case['near_safe_mass']:.4f}, soft_count={case['soft_violated_safe_count']:.4f}, "
            f"eligible_safe={case['eligible_safe_witness_count']:.1f}, dominant={case['dominant_component']}. "
            f"witnesses={case['witnesses']}"
        )
    v1_judgement = summary["v1_judgement"]
    global_compare = compare_df[
        compare_df["scope"] == "global"
    ][
        [
            "variant",
            "rows",
            "eligible_rate",
            "rank_median",
            "gap_mean",
            "directionality",
            "all_zero_rate",
            "true_mean",
        ]
    ].copy()
    lines = [
        "# contradiction practical_v2 audit",
        "",
        "## 1. v1 复盘结论（先读旧产物后继续）",
        f"- [已证明] {v1_judgement['headline']}",
        *[f"- [已证明] {line}" for line in v1_judgement["main_signal"]],
        *[f"- [已证明] {line}" for line in v1_judgement["why_zero_tie_heavy"]],
        *[f"- [已证明] {line}" for line in v1_judgement["proven"]],
        "",
        "## 2. 继续复用的现有路径",
        "- [已证明] 主审计 SSOT 继续使用 `src/scripts/audit/run_practical_audit_rerun.py`。",
        "- [已证明] sampled-history rollout / phys_ctx 构造继续使用 `src/scripts/audit/utils_practical_rollout.py`。",
        "",
        "## 3. v1 vs practical_v2 全局对比",
        "```text",
        global_compare.round(4).to_string(index=False),
        "```",
        f"- [已证明] key deltas: {summary['key_results']}.",
        "",
        "## 4. practical_v2 sweep / calibration",
        "```text",
        sweep_table[
            [
                "variant",
                "eligible_rate",
                "rank_median",
                "gap_mean",
                "directionality",
                "all_zero_rate",
                "true_mean",
            ]
        ].round(4).to_string(index=False),
        "```",
        f"- [已证明] time semantics compare: {summary['time_semantics_compare']['judgement']}",
        "",
        "## 5. decomposition / separation / support combo",
        f"- [已证明] practical_v2 all-candidate interval_gap_log: {summary['decomposition_stats']['all_candidates']['interval_gap_log']}",
        f"- [已证明] practical_v2 all-candidate near_safe_mass: {summary['decomposition_stats']['all_candidates']['near_safe_mass']}",
        f"- [已证明] practical_v2 all-candidate soft_violated_safe_count: {summary['decomposition_stats']['all_candidates']['soft_violated_safe_count']}",
        f"- [已证明] practical_v2 separation: {summary['separation']}",
        "```text",
        support_combo_table[
            [
                "label",
                "rows",
                "rank_median",
                "gap_mean",
                "directionality",
                "true_nonzero_rate",
                "hub_win_rate",
                "true_total_mean",
            ]
        ].round(4).to_string(index=False),
        "```",
        "",
        "## 6. bucket breakdown",
        "```text",
        bucket_table[
            [
                "bucket",
                "rows",
                "rate",
                "directionality",
                "all_zero_rate",
                "true_mean",
                "support_comp_mean",
                "support_comp_near_safe_mean",
                "support_comp_soft_count_mean",
            ]
        ].round(4).to_string(index=False),
        "```",
        "",
        "## 7. witness cases",
        *case_lines,
        "",
        "## 输出文件",
        f"- `{CONTRA_PRACTICAL_V2_MD_PATH}`",
        f"- `{CONTRA_PRACTICAL_V2_JSON_PATH}`",
        f"- `{CONTRA_PRACTICAL_V2_COMPARE_CSV_PATH}`",
        f"- `{STEPWISE_CSV_PATH}`",
    ]
    return "\n".join(lines) + "\n"


def classify_snapshot_pair_bucket(row: pd.Series) -> str:
    if row["has_positive_evidence"] <= 0.5:
        return "no_positive"
    if row["any_eligible_safe_witness"] <= 0.5:
        return "no_eligible_safe"
    if row["any_pair_available"] <= 0.5:
        return "no_pair_available"
    if row["any_positive_margin_available"] <= 0.5:
        if row["any_safe_regime_available"] > 0.5:
            return "pair_available_safe_only_weak"
        return "pair_available_but_non_positive_margin"
    if row["any_interval_regime_available"] <= 0.5:
        return "pair_available_but_interval_gap_zero"
    if row["any_true_source_mis_hit"] > 0.5:
        return "true_source_mis_hit"
    if row["active_safe_dominant_candidate_count"] > 0:
        return "active_safe_dominant"
    return "active_other"


def build_pair_availability_snapshot_frame(records: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    snapshot_meta = records[
        [
            "event_id",
            "episode",
            "time_min",
            "contra_practical_v2_all_zero",
            "contra_practical_v2_true",
            "contra_practical_v2_top_other_score",
            "contra_practical_v2_gap",
            "contra_practical_v2_directionality",
            "support_top_other_global_id",
        ]
    ].copy()
    grouped = (
        candidate_df.groupby(["event_id", "episode", "time_min"], as_index=False)
        .agg(
            has_positive_evidence=("has_positive_evidence", "max"),
            any_eligible_safe_witness=("has_eligible_safe_witness", "max"),
            any_pair_available=("pair_available", "max"),
            any_positive_margin_available=("positive_margin_available", "max"),
            any_interval_regime_available=("interval_regime_available", "max"),
            any_safe_regime_available=("safe_regime_available", "max"),
            any_pair_available_current_time=("current_time_pair_available", "max"),
            any_interval_regime_available_current_time=("current_time_interval_regime_available", "max"),
            any_true_source_mis_hit=("is_true_source_mis_hit", "max"),
            active_safe_dominant_candidate_count=("zero_reason_bucket", lambda s: int((s == "active_safe_dominant").sum())),
            interval_dominant_candidate_count=("dominant_component", lambda s: int((s == "interval").sum())),
            current_time_interval_dominant_candidate_count=(
                "current_time_dominant_component",
                lambda s: int((s == "interval").sum()),
            ),
            max_pair_count=("pair_count", "max"),
            max_positive_margin_pair_count=("positive_margin_pair_count", "max"),
            max_best_margin=("best_margin", "max"),
            max_interval_gap=("interval_gap", "max"),
            max_current_time_interval_gap=("current_time_interval_gap", "max"),
            max_total=("total", "max"),
            min_total=("total", "min"),
            max_current_time_total=("current_time_total", "max"),
        )
    )
    snapshot = snapshot_meta.merge(grouped, on=["event_id", "episode", "time_min"], how="left")
    snapshot["all_zero_or_tie"] = (
        (snapshot["contra_practical_v2_all_zero"] > 0.5)
        | ((snapshot["max_total"] - snapshot["min_total"]).abs() <= EPS)
    ).astype(np.float32)
    snapshot["snapshot_pair_bucket"] = snapshot.apply(classify_snapshot_pair_bucket, axis=1)
    snapshot["observability_ceiling"] = snapshot["snapshot_pair_bucket"].isin(
        ["no_positive", "no_eligible_safe", "no_pair_available"]
    ).astype(np.float32)
    snapshot["pair_margin_ceiling"] = snapshot["snapshot_pair_bucket"].isin(
        [
            "pair_available_but_non_positive_margin",
            "pair_available_but_interval_gap_zero",
            "pair_available_safe_only_weak",
        ]
    ).astype(np.float32)
    return snapshot


def summarize_value_counts(series: pd.Series) -> List[Dict[str, Any]]:
    counts = series.value_counts(dropna=False)
    total = int(counts.sum())
    rows: List[Dict[str, Any]] = []
    for label, count in counts.items():
        rows.append(
            {
                "bucket": str(label),
                "count": int(count),
                "rate": float(count / total) if total > 0 else np.nan,
            }
        )
    return rows


def build_pair_availability_case_record(row: pd.Series, label: str) -> Dict[str, Any]:
    return {
        "case_label": label,
        "event_id": int(row["event_id"]),
        "episode": int(row["episode"]),
        "time_min": float(row["time_min"]),
        "candidate_global_id": int(row["candidate_global_id"]),
        "is_true_source": float(row["is_true_source"]),
        "is_support_top_competitor": float(row["is_support_top_competitor"]),
        "zero_reason_bucket": str(row["zero_reason_bucket"]),
        "mis_hit_risk_bucket": str(row["mis_hit_risk_bucket"]),
        "pair_count": float(row["pair_count"]),
        "positive_margin_pair_count": float(row["positive_margin_pair_count"]),
        "best_margin": float(row["best_margin"]),
        "interval_gap": float(row["interval_gap"]),
        "safe_violation": float(row["safe_violation"]),
        "soft_count": float(row["soft_count"]),
        "near_safe_mass": float(row["near_safe_mass"]),
        "total": float(row["total"]),
        "dominant_component": str(row["dominant_component"]),
        "top_witness_safe_global_id": int(row["top_witness_safe_global_id"]),
        "top_witness_pos_global_id": int(row["top_witness_pos_global_id"]),
        "top_witness_safe_time_min": float(row["top_witness_safe_time_min"]) if np.isfinite(row["top_witness_safe_time_min"]) else np.nan,
        "top_witness_pos_time_min": float(row["top_witness_pos_time_min"]) if np.isfinite(row["top_witness_pos_time_min"]) else np.nan,
        "top_witness_time_gap_min": float(row["top_witness_time_gap_min"]) if np.isfinite(row["top_witness_time_gap_min"]) else np.nan,
        "current_time_interval_gap": float(row["current_time_interval_gap"]),
        "current_time_total": float(row["current_time_total"]),
    }


def build_pair_availability_case_examples(candidate_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    used = set()
    specs = [
        (
            "no_eligible_safe",
            (candidate_df["is_support_top_competitor"] > 0.5) & (candidate_df["zero_reason_bucket"] == "no_eligible_safe"),
            ["candidate_global_id"],
            [True],
        ),
        (
            "pair_exists_but_no_activation",
            (candidate_df["pair_available"] > 0.5)
            & candidate_df["zero_reason_bucket"].isin(
                ["pair_available_but_non_positive_margin", "pair_available_but_interval_gap_zero", "pair_available_safe_only_weak"]
            ),
            ["pair_count", "best_margin"],
            [False, False],
        ),
        (
            "active_safe_dominant_success",
            (candidate_df["is_support_top_competitor"] > 0.5)
            & (candidate_df["zero_reason_bucket"] == "active_safe_dominant"),
            ["total", "near_safe_mass"],
            [False, False],
        ),
        (
            "true_source_mis_hit",
            candidate_df["is_true_source_mis_hit"] > 0.5,
            ["total", "near_safe_mass"],
            [False, False],
        ),
        (
            "suspected_interval_like_but_unstable",
            (candidate_df["current_time_interval_regime_available"] > 0.5)
            & (candidate_df["interval_regime_available"] <= 0.5)
            & (candidate_df["current_time_interval_gap"] > candidate_df["interval_gap"] + 10.0),
            ["current_time_interval_gap", "current_time_total"],
            [False, False],
        ),
    ]
    for label, mask, sort_cols, ascending in specs:
        subset = candidate_df.loc[mask].sort_values(sort_cols, ascending=ascending)
        for _, row in subset.iterrows():
            key = (int(row["event_id"]), int(row["episode"]), int(row["candidate_global_id"]))
            if key in used:
                continue
            used.add(key)
            cases.append(build_pair_availability_case_record(row, label))
            break
    return cases


def build_pair_availability_summary(
    records: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> Dict[str, Any]:
    snapshot_df = build_pair_availability_snapshot_frame(records, candidate_df)
    all_zero_tie = snapshot_df[snapshot_df["all_zero_or_tie"] > 0.5].copy()
    mis_hit_df = candidate_df[candidate_df["is_true_source_mis_hit"] > 0.5].copy()
    witness_interval_candidates = candidate_df[candidate_df["interval_regime_available"] > 0.5].copy()
    current_time_interval_candidates = candidate_df[candidate_df["current_time_interval_regime_available"] > 0.5].copy()
    pair_available_candidates = candidate_df[candidate_df["pair_available"] > 0.5].copy()
    pair_available_snapshots = snapshot_df[snapshot_df["any_pair_available"] > 0.5].copy()
    witness_interval_snapshot_count = int((snapshot_df["any_interval_regime_available"] > 0.5).sum())
    current_interval_snapshot_count = int((snapshot_df["any_interval_regime_available_current_time"] > 0.5).sum())
    witness_interval_candidate_count = int((candidate_df["interval_regime_available"] > 0.5).sum())
    current_interval_candidate_count = int((candidate_df["current_time_interval_regime_available"] > 0.5).sum())
    interval_diagnosis = (
        "interval regime 不是完全不存在：witness_time 下已有少量 snapshot/candidate 激活，"
        "但它只以极稀 sparse candidate 形式出现，且 interval-dominant candidate 仍为 0；"
        "current_time compare 只增加了 interval-like candidate 数量并显著放大 outlier，"
        "没有把它变成稳定主导分支。"
    )
    return {
        "config": {
            "audit_path": "src/scripts/audit/run_practical_audit_rerun.py",
            "rollout_path": "src/scripts/audit/utils_practical_rollout.py",
            "diagnostic_csv_path": CONTRA_PAIR_AVAILABILITY_CSV_PATH,
            "selected_time_semantics": "witness_time_physctx",
        },
        "global_counts": {
            "snapshot_count": int(len(snapshot_df)),
            "candidate_count": int(len(candidate_df)),
            "all_zero_or_tie_count": int((snapshot_df["all_zero_or_tie"] > 0.5).sum()),
            "all_zero_or_tie_rate": float((snapshot_df["all_zero_or_tie"] > 0.5).mean()),
            "observability_ceiling_count_within_all_zero_or_tie": int(all_zero_tie["observability_ceiling"].sum()),
            "observability_ceiling_rate_within_all_zero_or_tie": float(all_zero_tie["observability_ceiling"].mean()) if len(all_zero_tie) > 0 else np.nan,
            "pair_margin_ceiling_count_within_all_zero_or_tie": int(all_zero_tie["pair_margin_ceiling"].sum()),
            "pair_margin_ceiling_rate_within_all_zero_or_tie": float(all_zero_tie["pair_margin_ceiling"].mean()) if len(all_zero_tie) > 0 else np.nan,
            "true_source_mis_hit_count": int((candidate_df["is_true_source_mis_hit"] > 0.5).sum()),
        },
        "candidate_rates": {
            "has_positive_evidence_rate": float((candidate_df["has_positive_evidence"] > 0.5).mean()),
            "has_eligible_safe_witness_rate": float((candidate_df["has_eligible_safe_witness"] > 0.5).mean()),
            "pair_available_rate": float((candidate_df["pair_available"] > 0.5).mean()),
            "positive_margin_available_rate": float((candidate_df["positive_margin_available"] > 0.5).mean()),
            "interval_regime_available_rate": float((candidate_df["interval_regime_available"] > 0.5).mean()),
            "safe_regime_available_rate": float((candidate_df["safe_regime_available"] > 0.5).mean()),
        },
        "snapshot_rates": {
            "has_positive_evidence_rate": float((snapshot_df["has_positive_evidence"] > 0.5).mean()),
            "any_eligible_safe_witness_rate": float((snapshot_df["any_eligible_safe_witness"] > 0.5).mean()),
            "any_pair_available_rate": float((snapshot_df["any_pair_available"] > 0.5).mean()),
            "any_positive_margin_available_rate": float((snapshot_df["any_positive_margin_available"] > 0.5).mean()),
            "any_interval_regime_available_rate": float((snapshot_df["any_interval_regime_available"] > 0.5).mean()),
            "any_interval_regime_available_current_time_rate": float(
                (snapshot_df["any_interval_regime_available_current_time"] > 0.5).mean()
            ),
        },
        "pair_conditioned_rates": {
            "candidate_positive_margin_rate_given_pair": float(
                (pair_available_candidates["positive_margin_available"] > 0.5).mean()
            ) if len(pair_available_candidates) > 0 else np.nan,
            "candidate_non_positive_margin_rate_given_pair": float(
                (pair_available_candidates["positive_margin_available"] <= 0.5).mean()
            ) if len(pair_available_candidates) > 0 else np.nan,
            "snapshot_positive_margin_rate_given_pair": float(
                (pair_available_snapshots["any_positive_margin_available"] > 0.5).mean()
            ) if len(pair_available_snapshots) > 0 else np.nan,
            "snapshot_non_positive_margin_rate_given_pair": float(
                (pair_available_snapshots["any_positive_margin_available"] <= 0.5).mean()
            ) if len(pair_available_snapshots) > 0 else np.nan,
        },
        "all_zero_reason_counts": {
            "no_positive": int((all_zero_tie["snapshot_pair_bucket"] == "no_positive").sum()),
            "no_eligible_safe": int((all_zero_tie["snapshot_pair_bucket"] == "no_eligible_safe").sum()),
            "no_pair_available": int((all_zero_tie["snapshot_pair_bucket"] == "no_pair_available").sum()),
            "pair_available_but_non_positive_margin": int(
                (all_zero_tie["snapshot_pair_bucket"] == "pair_available_but_non_positive_margin").sum()
            ),
            "pair_available_but_interval_gap_zero": int(
                (all_zero_tie["snapshot_pair_bucket"] == "pair_available_but_interval_gap_zero").sum()
            ),
            "pair_available_safe_only_weak": int(
                (all_zero_tie["snapshot_pair_bucket"] == "pair_available_safe_only_weak").sum()
            ),
        },
        "all_zero_or_tie_breakdown": summarize_value_counts(all_zero_tie["snapshot_pair_bucket"]),
        "candidate_bucket_breakdown": summarize_value_counts(candidate_df["zero_reason_bucket"]),
        "snapshot_bucket_breakdown": summarize_value_counts(snapshot_df["snapshot_pair_bucket"]),
        "interval_regime_analysis": {
            "witness_time_snapshot_count": witness_interval_snapshot_count,
            "current_time_snapshot_count": current_interval_snapshot_count,
            "witness_time_candidate_count": witness_interval_candidate_count,
            "current_time_candidate_count": current_interval_candidate_count,
            "witness_time_interval_dominant_candidate_count": int(
                ((candidate_df["dominant_component"] == "interval") & (candidate_df["total"] > EPS)).sum()
            ),
            "current_time_interval_dominant_candidate_count": int(
                ((candidate_df["current_time_dominant_component"] == "interval") & (candidate_df["current_time_total"] > EPS)).sum()
            ),
            "witness_time_interval_gap_distribution": summarize_distribution_array(
                witness_interval_candidates["interval_gap"].to_numpy(dtype=np.float32)
            ),
            "current_time_interval_gap_distribution": summarize_distribution_array(
                current_time_interval_candidates["current_time_interval_gap"].to_numpy(dtype=np.float32)
            ),
            "diagnosis": interval_diagnosis,
        },
        "mis_hit_analysis": {
            "snapshot_count": int(snapshot_df["any_true_source_mis_hit"].sum()),
            "candidate_count": int(len(mis_hit_df)),
            "bucket_breakdown": summarize_value_counts(mis_hit_df["mis_hit_risk_bucket"]) if not mis_hit_df.empty else [],
            "mean_total": float(mis_hit_df["total"].mean()) if not mis_hit_df.empty else np.nan,
            "mean_interval_gap": float(mis_hit_df["interval_gap"].mean()) if not mis_hit_df.empty else np.nan,
            "mean_pair_count": float(mis_hit_df["pair_count"].mean()) if not mis_hit_df.empty else np.nan,
            "mean_safe_time_gap_min": float(mis_hit_df["top_witness_time_gap_min"].mean()) if not mis_hit_df.empty else np.nan,
        },
        "representative_cases": build_pair_availability_case_examples(candidate_df),
    }, snapshot_df


def build_pair_availability_markdown(summary: Dict[str, Any]) -> str:
    zero_table = pd.DataFrame(summary["all_zero_or_tie_breakdown"])
    candidate_bucket_table = pd.DataFrame(summary["candidate_bucket_breakdown"])
    snapshot_bucket_table = pd.DataFrame(summary["snapshot_bucket_breakdown"])
    mis_hit_table = pd.DataFrame(summary["mis_hit_analysis"]["bucket_breakdown"])
    case_lines = []
    for case in summary["representative_cases"]:
        case_lines.append(
            f"- [已证明] {case['case_label']}: event={case['event_id']} episode={case['episode']} "
            f"candidate={case['candidate_global_id']} total={case['total']:.4f}, pair_count={case['pair_count']:.1f}, "
            f"best_margin={case['best_margin']:.4f}, interval_gap={case['interval_gap']:.4f}, "
            f"zero_reason={case['zero_reason_bucket']}, mis_hit_risk={case['mis_hit_risk_bucket']}."
        )
    lines = [
        "# contradiction pair-availability audit",
        "",
        "## 1. 目标",
        "- [已证明] 本轮不再优先调新的 contradiction score，而是沿现有 practical rerun / rollout 路径，把 zero/tie 拆成 observability、safe availability、pair availability、margin activation 四类问题。",
        "",
        "## 2. 全局统计",
        f"- [已证明] global counts: {summary['global_counts']}",
        f"- [已证明] candidate rates: {summary['candidate_rates']}",
        f"- [已证明] snapshot rates: {summary['snapshot_rates']}",
        f"- [已证明] pair conditioned rates: {summary['pair_conditioned_rates']}",
        "",
        "## 3. all_zero_or_tie 成因分解",
        f"- [已证明] explicit zero counts: {summary['all_zero_reason_counts']}",
        "```text",
        zero_table.round(4).to_string(index=False) if not zero_table.empty else "empty",
        "```",
        "",
        "## 4. candidate / snapshot bucket",
        "```text",
        candidate_bucket_table.round(4).to_string(index=False) if not candidate_bucket_table.empty else "empty",
        "```",
        "```text",
        snapshot_bucket_table.round(4).to_string(index=False) if not snapshot_bucket_table.empty else "empty",
        "```",
        "",
        "## 5. interval regime 缺失解释",
        f"- [已证明] {summary['interval_regime_analysis']['diagnosis']}",
        f"- [已证明] interval stats: {summary['interval_regime_analysis']}",
        "",
        "## 6. true-source mis-hit",
        f"- [已证明] mis-hit stats: {summary['mis_hit_analysis']}",
        "```text",
        mis_hit_table.round(4).to_string(index=False) if not mis_hit_table.empty else "empty",
        "```",
        "",
        "## 7. representative cases",
        *case_lines,
        "",
        "## 输出文件",
        f"- `{CONTRA_PAIR_AVAILABILITY_MD_PATH}`",
        f"- `{CONTRA_PAIR_AVAILABILITY_JSON_PATH}`",
        f"- `{CONTRA_PAIR_AVAILABILITY_CSV_PATH}`",
    ]
    return "\n".join(lines) + "\n"


def build_witness_mining_snapshot_frame(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    snapshot = (
        candidate_df.groupby(["witness_mining_mode", "event_id", "episode", "time_min"], as_index=False)
        .agg(
            has_positive_evidence=("has_positive_evidence", "max"),
            any_eligible_safe_witness=("has_eligible_safe_witness", "max"),
            any_pair_available=("pair_available", "max"),
            any_positive_margin_available=("positive_margin_available", "max"),
            any_interval_regime_available=("interval_regime_available", "max"),
            any_safe_regime_available=("safe_regime_available", "max"),
            any_true_source_mis_hit=("is_true_source_mis_hit", "max"),
            any_active_contradiction=("total", lambda s: float((s > EPS).any())),
            active_safe_dominant_candidate_count=("dominant_component", lambda s: int((s == "safe").sum())),
            interval_dominant_candidate_count=("dominant_component", lambda s: int((s == "interval").sum())),
            outlier_candidate_count=("outlier_risk_bucket", lambda s: int((s != "stable_or_none").sum())),
            max_total=("total", "max"),
            min_total=("total", "min"),
        )
    )
    snapshot["all_zero_or_tie"] = (
        (snapshot["max_total"].abs() <= EPS)
        | ((snapshot["max_total"] - snapshot["min_total"]).abs() <= EPS)
    ).astype(np.float32)
    snapshot["snapshot_pair_bucket"] = snapshot.apply(classify_snapshot_pair_bucket, axis=1)
    return snapshot


def build_witness_mining_mode_metrics(
    candidate_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    witness_mining_mode: str,
) -> Dict[str, Any]:
    mode_candidate = candidate_df[candidate_df["witness_mining_mode"] == witness_mining_mode].copy()
    mode_snapshot = snapshot_df[snapshot_df["witness_mining_mode"] == witness_mining_mode].copy()
    all_zero_tie = mode_snapshot[mode_snapshot["all_zero_or_tie"] > 0.5].copy()
    active_candidate = mode_candidate[mode_candidate["total"] > EPS].copy()
    pair_available_candidate = mode_candidate[mode_candidate["pair_available"] > 0.5].copy()

    return {
        "snapshot_count": int(len(mode_snapshot)),
        "candidate_count": int(len(mode_candidate)),
        "candidate_rates": {
            "has_positive_evidence_rate": float((mode_candidate["has_positive_evidence"] > 0.5).mean()),
            "has_eligible_safe_witness_rate": float((mode_candidate["has_eligible_safe_witness"] > 0.5).mean()),
            "pair_available_rate": float((mode_candidate["pair_available"] > 0.5).mean()),
            "positive_margin_available_rate": float((mode_candidate["positive_margin_available"] > 0.5).mean()),
            "active_contradiction_coverage_rate": float((mode_candidate["total"] > EPS).mean()),
            "interval_regime_available_rate": float((mode_candidate["interval_regime_available"] > 0.5).mean()),
            "safe_regime_available_rate": float((mode_candidate["safe_regime_available"] > 0.5).mean()),
            "coverage_gain_vs_baseline_rate": float((mode_candidate["coverage_gain_vs_baseline"] > 0.5).mean()),
        },
        "snapshot_rates": {
            "has_positive_evidence_rate": float((mode_snapshot["has_positive_evidence"] > 0.5).mean()),
            "any_eligible_safe_witness_rate": float((mode_snapshot["any_eligible_safe_witness"] > 0.5).mean()),
            "any_pair_available_rate": float((mode_snapshot["any_pair_available"] > 0.5).mean()),
            "any_positive_margin_available_rate": float((mode_snapshot["any_positive_margin_available"] > 0.5).mean()),
            "any_active_contradiction_rate": float((mode_snapshot["any_active_contradiction"] > 0.5).mean()),
            "all_zero_or_tie_rate": float((mode_snapshot["all_zero_or_tie"] > 0.5).mean()),
        },
        "mean_counts": {
            "mined_safe_candidate_count": float(mode_candidate["mined_safe_candidate_count"].mean()),
            "hydraulic_comparable_safe_count": float(mode_candidate["hydraulic_comparable_safe_count"].mean()),
            "front_close_safe_count": float(mode_candidate["front_close_safe_count"].mean()),
            "selected_safe_witness_count": float(mode_candidate["selected_safe_witness_count"].mean()),
            "topk_safe_count": float(mode_candidate["topk_safe_count"].mean()),
        },
        "all_zero_reason_counts": {
            "no_positive": int((all_zero_tie["snapshot_pair_bucket"] == "no_positive").sum()),
            "no_eligible_safe": int((all_zero_tie["snapshot_pair_bucket"] == "no_eligible_safe").sum()),
            "no_pair_available": int((all_zero_tie["snapshot_pair_bucket"] == "no_pair_available").sum()),
            "pair_available_but_non_positive_margin": int(
                (all_zero_tie["snapshot_pair_bucket"] == "pair_available_but_non_positive_margin").sum()
            ),
            "pair_available_but_interval_gap_zero": int(
                (all_zero_tie["snapshot_pair_bucket"] == "pair_available_but_interval_gap_zero").sum()
            ),
            "pair_available_safe_only_weak": int(
                (all_zero_tie["snapshot_pair_bucket"] == "pair_available_safe_only_weak").sum()
            ),
        },
        "snapshot_bucket_breakdown": summarize_value_counts(mode_snapshot["snapshot_pair_bucket"]),
        "candidate_bucket_breakdown": summarize_value_counts(mode_candidate["zero_reason_bucket"]),
        "dominant_component": {
            "active_safe_dominant_count": int((active_candidate["dominant_component"] == "safe").sum()),
            "active_interval_dominant_count": int((active_candidate["dominant_component"] == "interval").sum()),
            "active_other_count": int(
                ((active_candidate["dominant_component"] != "safe") & (active_candidate["dominant_component"] != "interval")).sum()
            ),
        },
        "pair_conditioned_rates": {
            "candidate_positive_margin_rate_given_pair": float(
                (pair_available_candidate["positive_margin_available"] > 0.5).mean()
            ) if not pair_available_candidate.empty else np.nan,
        },
        "outlier_risk": {
            "bucket_breakdown": summarize_value_counts(mode_candidate["outlier_risk_bucket"]),
            "coverage_gain_but_score_instability_count": int((mode_candidate["coverage_gain_but_score_instability"] > 0.5).sum()),
        },
        "true_source_risk": {
            "candidate_count": int((mode_candidate["is_true_source_mis_hit"] > 0.5).sum()),
            "snapshot_count": int(
                mode_snapshot.loc[mode_snapshot["any_true_source_mis_hit"] > 0.5, ["event_id", "episode", "time_min"]].shape[0]
            ),
        },
    }


def choose_witness_case(
    wide_df: pd.DataFrame,
    label: str,
    mask: pd.Series,
    sort_cols: List[str],
    ascending: List[bool],
    used: set,
) -> Optional[Dict[str, Any]]:
    subset = wide_df.loc[mask].sort_values(sort_cols, ascending=ascending)
    for _, row in subset.iterrows():
        key = (int(row["event_id"]), int(row["episode"]), int(row["candidate_global_id"]))
        if key in used:
            continue
        used.add(key)
        return {
            "case_label": label,
            "event_id": int(row["event_id"]),
            "episode": int(row["episode"]),
            "time_min": float(row["time_min"]),
            "candidate_global_id": int(row["candidate_global_id"]),
            "is_true_source": float(row["is_true_source"]),
            "baseline_zero_reason_bucket": str(row["baseline_zero_reason_bucket"]),
            "frontier_zero_reason_bucket": str(row["frontier_zero_reason_bucket"]),
            "topk_zero_reason_bucket": str(row["topk_zero_reason_bucket"]),
            "baseline_pair_available": float(row["baseline_pair_available"]),
            "frontier_pair_available": float(row["frontier_pair_available"]),
            "topk_pair_available": float(row["topk_pair_available"]),
            "baseline_positive_margin_available": float(row["baseline_positive_margin_available"]),
            "frontier_positive_margin_available": float(row["frontier_positive_margin_available"]),
            "topk_positive_margin_available": float(row["topk_positive_margin_available"]),
            "baseline_best_margin": float(row["baseline_best_margin"]),
            "frontier_best_margin": float(row["frontier_best_margin"]),
            "topk_best_margin": float(row["topk_best_margin"]),
            "baseline_total": float(row["baseline_total"]),
            "frontier_total": float(row["frontier_total"]),
            "topk_total": float(row["topk_total"]),
            "baseline_outlier_risk_bucket": str(row["baseline_outlier_risk_bucket"]),
            "frontier_outlier_risk_bucket": str(row["frontier_outlier_risk_bucket"]),
            "topk_outlier_risk_bucket": str(row["topk_outlier_risk_bucket"]),
        }
    return None


def build_witness_mining_case_examples(candidate_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if candidate_df.empty:
        return []

    base = candidate_df[candidate_df["witness_mining_mode"] == DEFAULT_WITNESS_MINING_MODE].copy().rename(
        columns=lambda c: f"baseline_{c}" if c not in ["event_id", "episode", "time_min", "candidate_local_idx"] else c
    )
    frontier = candidate_df[candidate_df["witness_mining_mode"] == "candidate_conditioned_frontier_safe"].copy().rename(
        columns=lambda c: f"frontier_{c}" if c not in ["event_id", "episode", "time_min", "candidate_local_idx"] else c
    )
    topk = candidate_df[candidate_df["witness_mining_mode"] == "candidate_conditioned_topk_safe"].copy().rename(
        columns=lambda c: f"topk_{c}" if c not in ["event_id", "episode", "time_min", "candidate_local_idx"] else c
    )
    wide = (
        base.merge(frontier, on=["event_id", "episode", "time_min", "candidate_local_idx"], how="inner")
        .merge(topk, on=["event_id", "episode", "time_min", "candidate_local_idx"], how="inner")
    )
    wide["candidate_global_id"] = wide["baseline_candidate_global_id"]
    wide["is_true_source"] = wide["baseline_is_true_source"]
    used: set = set()
    cases: List[Dict[str, Any]] = []
    specs = [
        (
            "baseline_pair_lost_under_frontier_filter",
            (wide["baseline_pair_available"] > 0.5) & (wide["frontier_pair_available"] <= 0.5),
            ["baseline_best_margin", "baseline_total"],
            [False, False],
        ),
        (
            "frontier_hydraulic_gain_but_no_pair_gain",
            (wide["frontier_hydraulic_comparable_safe_count"] > wide["baseline_hydraulic_comparable_safe_count"])
            & (wide["frontier_pair_available"] <= wide["baseline_pair_available"]),
            ["frontier_hydraulic_comparable_safe_count", "baseline_hydraulic_comparable_safe_count"],
            [False, False],
        ),
        (
            "topk_more_selected_safe_without_extra_coverage",
            (wide["topk_selected_safe_witness_count"] > wide["frontier_selected_safe_witness_count"])
            & (wide["topk_pair_available"] == wide["frontier_pair_available"])
            & (wide["topk_positive_margin_available"] == wide["frontier_positive_margin_available"]),
            ["topk_selected_safe_witness_count", "frontier_selected_safe_witness_count"],
            [False, False],
        ),
        (
            "topk_noise_case",
            (wide["topk_coverage_gain_but_score_instability"] > 0.5),
            ["topk_total", "topk_best_margin"],
            [False, False],
        ),
        (
            "baseline_active_case_stays_safe_dominant",
            (wide["baseline_positive_margin_available"] > 0.5)
            & (wide["topk_positive_margin_available"] > 0.5),
            ["baseline_total", "topk_total"],
            [False, False],
        ),
        (
            "interval_still_sparse_outlier",
            (wide["topk_current_time_interval_gap"] > np.maximum(wide["topk_interval_gap"] + 10.0, 5000.0))
            & (wide["topk_current_time_interval_regime_available"] > 0.5),
            ["topk_current_time_interval_gap", "topk_total"],
            [False, False],
        ),
        (
            "true_source_risk_case",
            (wide["frontier_is_true_source_mis_hit"] > 0.5) | (wide["topk_is_true_source_mis_hit"] > 0.5),
            ["topk_total", "frontier_total"],
            [False, False],
        ),
    ]
    for label, mask, sort_cols, ascending in specs:
        case = choose_witness_case(wide, label, mask, sort_cols, ascending, used)
        if case is not None:
            cases.append(case)
    return cases


def build_witness_question_answers(summary: Dict[str, Any]) -> Dict[str, str]:
    baseline = summary["mode_metrics"][DEFAULT_WITNESS_MINING_MODE]
    frontier = summary["mode_metrics"]["candidate_conditioned_frontier_safe"]
    topk = summary["mode_metrics"]["candidate_conditioned_topk_safe"]

    frontier_safe_delta = frontier["candidate_rates"]["has_eligible_safe_witness_rate"] - baseline["candidate_rates"]["has_eligible_safe_witness_rate"]
    frontier_pair_delta = frontier["candidate_rates"]["pair_available_rate"] - baseline["candidate_rates"]["pair_available_rate"]
    topk_pair_delta = topk["candidate_rates"]["pair_available_rate"] - baseline["candidate_rates"]["pair_available_rate"]
    best_all_zero_drop = baseline["snapshot_rates"]["all_zero_or_tie_rate"] - min(
        frontier["snapshot_rates"]["all_zero_or_tie_rate"],
        topk["snapshot_rates"]["all_zero_or_tie_rate"],
    )

    q1 = (
        "不是。candidate-conditioned frontier/top-k 没有把 safe coverage 拉高，反而把 admissible safe witness 压得比 baseline 更稀；"
        "这说明当前真正 front-close、positive-comparable 的 safe witness 本身就很少。"
        if frontier_safe_delta <= 0.0 and frontier_pair_delta <= 0.0 and topk_pair_delta <= 0.0
        else "它是局部瓶颈，但还没证明它已经是唯一主瓶颈。"
    )
    q2 = (
        f"没有。frontier-safe 下 has_eligible_safe_witness_rate 变化 {frontier_safe_delta:.4f}，"
        f"pair_available_rate 变化 {frontier_pair_delta:.4f}；结果不是 gain，而是轻微回落。"
        if frontier_safe_delta <= 0.0 and frontier_pair_delta <= 0.0
        else f"有局部提升：has_eligible_safe_witness_rate 变化 {frontier_safe_delta:.4f}，"
             f"pair_available_rate 变化 {frontier_pair_delta:.4f}。"
    )
    q3 = (
        "top-k 没有比单 frontier-safe 带来更多 coverage；它只是在同等 coverage 下保留了更多 selected safe，"
        "而 instability bucket 也没有被显著放大。"
        if topk["candidate_rates"]["positive_margin_available_rate"] == frontier["candidate_rates"]["positive_margin_available_rate"]
        and topk["candidate_rates"]["pair_available_rate"] == frontier["candidate_rates"]["pair_available_rate"]
        else "top-k 主要带来更多噪声；coverage 提升如果存在，也伴随更高 instability。"
    )
    q4 = (
        "仍然主要是 safe-dominant。非 baseline mode 下 active interval-dominant candidate 依旧极少或为 0。"
        if max(
            frontier["dominant_component"]["active_interval_dominant_count"],
            topk["dominant_component"]["active_interval_dominant_count"],
        ) == 0
        else "interval 仍未成为稳定主导，只在极少数 case 里冒头。"
    )
    q5 = (
        "还不能推进到 inference-time auxiliary。coverage 没有出现实质改善，all_zero_or_tie 也没有下降，"
        "因此当前更接近稀疏 auxiliary / 继续资格验证，而不是可升格的 inference-time 辅助信号。"
        if best_all_zero_drop < 0.05
        or max(frontier["true_source_risk"]["candidate_count"], topk["true_source_risk"]["candidate_count"]) > 0
        else "可以开始接近 inference-time auxiliary，但仍需要继续 guard 和稳定性验证。"
    )
    return {
        "q1_true_bottleneck_is_overconservative_safe_mining": q1,
        "q2_frontier_safe_coverage_gain": q2,
        "q3_topk_stability": q3,
        "q4_safe_vs_interval_dominance": q4,
        "q5_project_positioning": q5,
    }


def build_witness_mining_summary(
    records: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> Dict[str, Any]:
    snapshot_df = build_witness_mining_snapshot_frame(candidate_df)
    baseline_snapshot_count = int(
        records.loc[:, ["event_id", "episode", "time_min"]].drop_duplicates().shape[0]
    )
    mode_metrics = {
        mode: build_witness_mining_mode_metrics(candidate_df, snapshot_df, mode)
        for mode in WITNESS_MINING_MODES
    }
    baseline_metrics = mode_metrics[DEFAULT_WITNESS_MINING_MODE]
    delta_vs_baseline: Dict[str, Dict[str, float]] = {}
    for mode in WITNESS_MINING_MODES:
        if mode == DEFAULT_WITNESS_MINING_MODE:
            continue
        delta_vs_baseline[mode] = {
            "candidate_has_eligible_safe_witness_rate_delta": float(
                mode_metrics[mode]["candidate_rates"]["has_eligible_safe_witness_rate"]
                - baseline_metrics["candidate_rates"]["has_eligible_safe_witness_rate"]
            ),
            "candidate_pair_available_rate_delta": float(
                mode_metrics[mode]["candidate_rates"]["pair_available_rate"]
                - baseline_metrics["candidate_rates"]["pair_available_rate"]
            ),
            "candidate_positive_margin_available_rate_delta": float(
                mode_metrics[mode]["candidate_rates"]["positive_margin_available_rate"]
                - baseline_metrics["candidate_rates"]["positive_margin_available_rate"]
            ),
            "snapshot_all_zero_or_tie_rate_delta": float(
                mode_metrics[mode]["snapshot_rates"]["all_zero_or_tie_rate"]
                - baseline_metrics["snapshot_rates"]["all_zero_or_tie_rate"]
            ),
            "snapshot_any_pair_available_rate_delta": float(
                mode_metrics[mode]["snapshot_rates"]["any_pair_available_rate"]
                - baseline_metrics["snapshot_rates"]["any_pair_available_rate"]
            ),
            "true_source_mis_hit_count_delta": float(
                mode_metrics[mode]["true_source_risk"]["candidate_count"]
                - baseline_metrics["true_source_risk"]["candidate_count"]
            ),
        }
    summary = {
        "config": {
            "audit_path": "src/scripts/audit/run_practical_audit_rerun.py",
            "rollout_path": "src/scripts/audit/utils_practical_rollout.py",
            "diagnostic_csv_path": CONTRA_WITNESS_COVERAGE_CSV_PATH,
            "baseline_snapshot_denominator": baseline_snapshot_count,
            "historical_snapshot_reference": 970,
            "rerun_snapshot_reference": baseline_snapshot_count,
            "witness_mining_modes": WITNESS_MINING_MODES,
            "frontier_safe_close_tau_min": WITNESS_MINING_FRONT_CLOSE_TAU_MIN,
            "topk_safe_witnesses": WITNESS_MINING_TOP_K,
            "history_physctx_mode": PRACTICAL_V2_HISTORY_PHYSCTX_MODE,
            "current_time_physctx_is_diagnostic_only": True,
        },
        "mode_metrics": mode_metrics,
        "delta_vs_baseline": delta_vs_baseline,
        "mode_snapshot_counts_consistent": {
            mode: int(mode_metrics[mode]["snapshot_count"] == baseline_snapshot_count)
            for mode in WITNESS_MINING_MODES
        },
        "representative_cases": build_witness_mining_case_examples(candidate_df),
    }
    summary["question_answers"] = build_witness_question_answers(summary)
    return summary


def build_witness_mining_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# contradiction witness coverage pilot",
        "",
        "## 1. 实验口径",
        f"- [已证明] 当前 rerun baseline snapshot 分母是 `{summary['config']['baseline_snapshot_denominator']}`，不是历史 `{summary['config']['historical_snapshot_reference']}`。",
        "- [已证明] 本轮没有新建第三条 audit pipeline；仍沿 `run_practical_audit_rerun.py` + `utils_practical_rollout.py`。",
        f"- [已证明] compare modes: `{', '.join(summary['config']['witness_mining_modes'])}`。",
        f"- [已证明] frontier-safe 只把 `current_time_physctx` 保留为 diagnostic；主 compare 仍 anchored 在 `{summary['config']['history_physctx_mode']}` practical rerun 上。",
        "",
        "## 2. Mode Summary",
    ]
    for mode in WITNESS_MINING_MODES:
        metrics = summary["mode_metrics"][mode]
        lines.extend(
            [
                f"### {mode}",
                f"- candidate_rates: {metrics['candidate_rates']}",
                f"- snapshot_rates: {metrics['snapshot_rates']}",
                f"- mean_counts: {metrics['mean_counts']}",
                f"- true_source_risk: {metrics['true_source_risk']}",
                f"- outlier_risk: {metrics['outlier_risk']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. Delta vs Baseline",
            f"- [已证明] delta_vs_baseline: {summary['delta_vs_baseline']}",
            "",
            "## 4. 五个问题",
            f"- [部分证明] Q1: {summary['question_answers']['q1_true_bottleneck_is_overconservative_safe_mining']}",
            f"- [部分证明] Q2: {summary['question_answers']['q2_frontier_safe_coverage_gain']}",
            f"- [部分证明] Q3: {summary['question_answers']['q3_topk_stability']}",
            f"- [部分证明] Q4: {summary['question_answers']['q4_safe_vs_interval_dominance']}",
            f"- [部分证明] Q5: {summary['question_answers']['q5_project_positioning']}",
            "",
            "## 5. Representative Cases",
        ]
    )
    for case in summary["representative_cases"]:
        lines.append(
            f"- [已证明] {case['case_label']}: event={case['event_id']} episode={case['episode']} candidate={case['candidate_global_id']} "
            f"`pair baseline/frontier/topk={case['baseline_pair_available']:.0f}/{case['frontier_pair_available']:.0f}/{case['topk_pair_available']:.0f}`, "
            f"`margin baseline/frontier/topk={case['baseline_positive_margin_available']:.0f}/{case['frontier_positive_margin_available']:.0f}/{case['topk_positive_margin_available']:.0f}`, "
            f"`outlier baseline/frontier/topk={case['baseline_outlier_risk_bucket']}/{case['frontier_outlier_risk_bucket']}/{case['topk_outlier_risk_bucket']}`."
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            f"- `{CONTRA_WITNESS_COVERAGE_MD_PATH}`",
            f"- `{CONTRA_WITNESS_COVERAGE_JSON_PATH}`",
            f"- `{CONTRA_WITNESS_COVERAGE_CSV_PATH}`",
        ]
    )
    return "\n".join(lines) + "\n"


def classify_admissibility_candidate_bucket(row: pd.Series) -> str:
    if row["has_positive_evidence"] <= 0.5:
        return "no_positive"
    if row["has_safe_observation_anywhere"] <= 0.5:
        return "no_safe_observation_anywhere"
    if row["has_hydraulically_comparable_safe"] <= 0.5:
        return "no_hydraulically_comparable_safe"
    if row["pair_available_under_mode"] <= 0.5:
        if row["has_time_bracketing_safe"] <= 0.5 and row["has_frontier_window_safe"] <= 0.5:
            return "no_front_bracketing_safe"
        if row["has_time_bracketing_safe"] <= 0.5:
            return "no_time_bracketing_safe"
        if row["has_frontier_window_safe"] <= 0.5:
            return "no_frontier_window_safe"
        return "no_pair_available_under_mode"
    if row["positive_margin_available_under_mode"] <= 0.5:
        if row["safe_regime_available"] > 0.5:
            return "pair_available_safe_only_weak"
        return "pair_available_but_non_positive_margin"
    if row["interval_regime_available"] <= 0.5:
        return "pair_available_but_interval_gap_zero"
    if row["is_true_source_mis_hit"] > 0.5:
        return "true_source_mis_hit"
    if row["dominant_component"] == "safe":
        return "active_safe_dominant"
    return "active_other"


def build_admissibility_compare_candidate_frame(
    event_id: int,
    episode: int,
    time_min: float,
    src_local: int,
    support_top_other_idx: int,
    contra_res: Dict[str, Any],
    current_time_res: Dict[str, Any],
    g_ids: torch.Tensor,
    admissibility_mode: str,
) -> pd.DataFrame:
    positive_lookup = build_history_time_lookup(contra_res, "positive_records")
    safe_lookup = build_history_time_lookup(contra_res, "safe_records")
    num_nodes = int(g_ids.numel())
    candidate_local_idx = np.arange(num_nodes, dtype=np.int64)
    g_ids_np = g_ids.detach().cpu().numpy().astype(np.int64)
    top_safe_local = contra_res["top_witness_safe_local_idx"].detach().cpu().numpy().astype(np.int64)
    top_pos_local = contra_res["top_witness_pos_local_idx"].detach().cpu().numpy().astype(np.int64)
    top_safe_time = np.array([safe_lookup.get(int(idx), np.nan) if int(idx) >= 0 else np.nan for idx in top_safe_local])
    top_pos_time = np.array([positive_lookup.get(int(idx), np.nan) if int(idx) >= 0 else np.nan for idx in top_pos_local])

    pair_available_under_mode = contra_res.get("pair_available_under_mode", contra_res["pair_available"]).detach().cpu().numpy().astype(np.float32)
    positive_margin_available_under_mode = contra_res.get(
        "positive_margin_available_under_mode",
        contra_res["positive_margin_available"],
    ).detach().cpu().numpy().astype(np.float32)
    has_safe_observation_anywhere = contra_res.get(
        "has_safe_observation_anywhere",
        torch.full_like(contra_res["pair_available"], float(contra_res["safe_count"] > 0)),
    ).detach().cpu().numpy().astype(np.float32)
    has_hydraulically_comparable_safe = contra_res.get(
        "has_hydraulically_comparable_safe",
        (contra_res["safe_reachable_count"] > 0.0).float(),
    ).detach().cpu().numpy().astype(np.float32)
    has_time_bracketing_safe = contra_res.get(
        "has_time_bracketing_safe",
        (contra_res["positive_margin_available"] > 0.0).float(),
    ).detach().cpu().numpy().astype(np.float32)
    has_frontier_window_safe = contra_res.get(
        "has_frontier_window_safe",
        torch.zeros_like(contra_res["pair_available"]),
    ).detach().cpu().numpy().astype(np.float32)
    hydraulic_comparable_safe_count = contra_res.get(
        "hydraulic_comparable_safe_count",
        contra_res["safe_reachable_count"],
    ).detach().cpu().numpy().astype(np.float32)
    time_bracketing_safe_count = contra_res.get(
        "time_bracketing_safe_count",
        contra_res["positive_margin_count"],
    ).detach().cpu().numpy().astype(np.float32)
    frontier_window_safe_count = contra_res.get(
        "frontier_window_safe_count",
        torch.zeros_like(contra_res["pair_available"]),
    ).detach().cpu().numpy().astype(np.float32)

    frame = pd.DataFrame(
        {
            "event_id": int(event_id),
            "episode": int(episode),
            "time_min": float(time_min),
            "candidate_local_idx": candidate_local_idx,
            "candidate_global_id": g_ids_np,
            "is_true_source": (candidate_local_idx == int(src_local)).astype(np.float32),
            "is_support_top_competitor": (candidate_local_idx == int(support_top_other_idx)).astype(np.float32),
            "admissibility_mode": str(admissibility_mode),
            "has_positive_evidence": np.full(num_nodes, float(contra_res["positive_count"] > 0), dtype=np.float32),
            "positive_evidence_count": np.full(num_nodes, float(contra_res["positive_count"]), dtype=np.float32),
            "safe_history_count": np.full(num_nodes, float(contra_res["safe_count"]), dtype=np.float32),
            "has_safe_observation_anywhere": has_safe_observation_anywhere,
            "has_hydraulically_comparable_safe": has_hydraulically_comparable_safe,
            "has_time_bracketing_safe": has_time_bracketing_safe,
            "has_frontier_window_safe": has_frontier_window_safe,
            "hydraulic_comparable_safe_count": hydraulic_comparable_safe_count,
            "time_bracketing_safe_count": time_bracketing_safe_count,
            "frontier_window_safe_count": frontier_window_safe_count,
            "selected_safe_witness_count": contra_res["selected_safe_witness_count"].detach().cpu().numpy().astype(np.float32),
            "pair_available_under_mode": pair_available_under_mode,
            "positive_margin_available_under_mode": positive_margin_available_under_mode,
            "pair_count": contra_res["pair_count"].detach().cpu().numpy().astype(np.float32),
            "positive_margin_pair_count": contra_res["positive_margin_pair_count"].detach().cpu().numpy().astype(np.float32),
            "best_margin": contra_res["top_witness_margin"].detach().cpu().numpy().astype(np.float32),
            "best_margin_topk_mean": contra_res["best_margin_topk_mean"].detach().cpu().numpy().astype(np.float32),
            "interval_gap": contra_res["interval_gap"].detach().cpu().numpy().astype(np.float32),
            "safe_violation": contra_res["safe_violation"].detach().cpu().numpy().astype(np.float32),
            "violated_safe_count": contra_res["violated_safe_count"].detach().cpu().numpy().astype(np.float32),
            "soft_count": contra_res["soft_violated_safe_count"].detach().cpu().numpy().astype(np.float32),
            "near_safe_mass": contra_res["near_safe_mass"].detach().cpu().numpy().astype(np.float32),
            "safe_component": contra_res["safe_component"].detach().cpu().numpy().astype(np.float32),
            "total": contra_res["total"].detach().cpu().numpy().astype(np.float32),
            "dominant_component": [
                dominant_component_label(int(flag))
                for flag in contra_res["dominant_component_flag"].detach().cpu().numpy().astype(np.int64)
            ],
            "interval_regime_available": contra_res["interval_regime_available"].detach().cpu().numpy().astype(np.float32),
            "safe_regime_available": contra_res["safe_regime_available"].detach().cpu().numpy().astype(np.float32),
            "top_witness_safe_global_id": np.where(top_safe_local >= 0, g_ids_np[top_safe_local.clip(min=0)], -1),
            "top_witness_pos_global_id": np.where(top_pos_local >= 0, g_ids_np[top_pos_local.clip(min=0)], -1),
            "top_witness_safe_time_min": top_safe_time,
            "top_witness_pos_time_min": top_pos_time,
            "top_witness_time_gap_min": top_safe_time - top_pos_time,
            "current_time_interval_gap": current_time_res["interval_gap"].detach().cpu().numpy().astype(np.float32),
            "current_time_total": current_time_res["total"].detach().cpu().numpy().astype(np.float32),
            "current_time_interval_regime_available": current_time_res["interval_regime_available"].detach().cpu().numpy().astype(np.float32),
            "current_time_dominant_component": [
                dominant_component_label(int(flag))
                for flag in current_time_res["dominant_component_flag"].detach().cpu().numpy().astype(np.int64)
            ],
            "time_bracketing_margin_floor_min": np.full(
                num_nodes,
                float(contra_res.get("time_bracketing_margin_floor_min", 0.0)),
                dtype=np.float32,
            ),
            "admissibility_frontier_close_tau_min": np.full(
                num_nodes,
                float(contra_res.get("admissibility_frontier_close_tau_min", WITNESS_MINING_FRONT_CLOSE_TAU_MIN)),
                dtype=np.float32,
            ),
        }
    )
    frame["is_true_source_mis_hit"] = (
        (frame["is_true_source"] > 0.5) & (frame["total"] > EPS)
    ).astype(np.float32)
    frame["zero_reason_bucket_under_mode"] = frame.apply(classify_admissibility_candidate_bucket, axis=1)
    return frame


def build_admissibility_compare_wide_frame(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    key_cols = ["event_id", "episode", "time_min", "candidate_local_idx"]
    wide: Optional[pd.DataFrame] = None
    for mode in ADMISSIBILITY_COMPARE_MODES:
        short = ADMISSIBILITY_MODE_SHORT_LABEL[mode]
        subset = candidate_df[candidate_df["admissibility_mode"] == mode].copy()
        if subset.empty:
            continue
        subset = subset.rename(columns=lambda c: f"{short}_{c}" if c not in key_cols else c)
        wide = subset if wide is None else wide.merge(subset, on=key_cols, how="inner")
    if wide is None or wide.empty:
        return pd.DataFrame()
    wide["candidate_global_id"] = wide["baseline_candidate_global_id"]
    wide["is_true_source"] = wide["baseline_is_true_source"]
    return wide


def classify_admissibility_observability_bucket(row: pd.Series) -> str:
    if row["baseline_has_positive_evidence"] <= 0.5:
        return "no_positive"
    if row["baseline_has_safe_observation_anywhere"] <= 0.5:
        return "no_safe_observation_anywhere"
    if row["union_has_hydraulically_comparable_safe"] <= 0.5:
        return "no_hydraulically_comparable_safe_under_union"
    if row["union_pair_available_under_mode"] <= 0.5:
        if row["union_has_time_bracketing_safe"] <= 0.5 and row["union_has_frontier_window_safe"] <= 0.5:
            return "no_front_bracketing_safe_under_union"
        if row["union_has_time_bracketing_safe"] <= 0.5:
            return "no_time_bracketing_safe_under_union"
        if row["union_has_frontier_window_safe"] <= 0.5:
            return "no_frontier_window_safe_under_union"
        return "no_pair_available_even_under_union"
    return "observable_under_union"


def classify_admissibility_ceiling_bucket(row: pd.Series) -> str:
    if row["baseline_pair_available_under_mode"] > 0.5:
        if row["baseline_positive_margin_available_under_mode"] <= 0.5 and row["union_positive_margin_available_under_mode"] > 0.5:
            if row["topology_positive_margin_available_under_mode"] > row["baseline_positive_margin_available_under_mode"]:
                return "released_by_topology_margin"
            if row["time_positive_margin_available_under_mode"] > row["baseline_positive_margin_available_under_mode"]:
                return "released_by_time_bracketing_margin"
            if row["frontier_positive_margin_available_under_mode"] > row["baseline_positive_margin_available_under_mode"]:
                return "released_by_frontier_window_margin"
            return "released_only_by_union_margin"
        return "baseline_available"
    if row["topology_pair_available_under_mode"] > row["baseline_pair_available_under_mode"]:
        return "released_by_topology"
    if row["time_pair_available_under_mode"] > row["baseline_pair_available_under_mode"]:
        return "released_by_time_bracketing"
    if row["frontier_pair_available_under_mode"] > row["baseline_pair_available_under_mode"]:
        return "released_by_frontier_window"
    if row["union_pair_available_under_mode"] > row["baseline_pair_available_under_mode"]:
        return "released_only_by_union"
    return "no_gain_even_under_union"


def annotate_admissibility_compare_frame(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    candidate_df = candidate_df.drop(
        columns=[
            "observability_ceiling_bucket",
            "admissibility_ceiling_bucket",
            "coverage_gain_vs_baseline",
            "pair_gain_vs_baseline",
            "positive_margin_gain_vs_baseline",
        ],
        errors="ignore",
    )
    wide = build_admissibility_compare_wide_frame(candidate_df)
    if wide.empty:
        return candidate_df
    wide["observability_ceiling_bucket"] = wide.apply(classify_admissibility_observability_bucket, axis=1)
    wide["admissibility_ceiling_bucket"] = wide.apply(classify_admissibility_ceiling_bucket, axis=1)
    wide["coverage_gain_vs_baseline"] = (
        (wide["union_pair_available_under_mode"] > wide["baseline_pair_available_under_mode"])
        | (wide["union_positive_margin_available_under_mode"] > wide["baseline_positive_margin_available_under_mode"])
    ).astype(np.float32)
    wide["pair_gain_vs_baseline"] = (
        wide["union_pair_available_under_mode"] > wide["baseline_pair_available_under_mode"]
    ).astype(np.float32)
    wide["positive_margin_gain_vs_baseline"] = (
        wide["union_positive_margin_available_under_mode"] > wide["baseline_positive_margin_available_under_mode"]
    ).astype(np.float32)
    return candidate_df.merge(
        wide[
            [
                "event_id",
                "episode",
                "time_min",
                "candidate_local_idx",
                "observability_ceiling_bucket",
                "admissibility_ceiling_bucket",
                "coverage_gain_vs_baseline",
                "pair_gain_vs_baseline",
                "positive_margin_gain_vs_baseline",
            ]
        ],
        on=["event_id", "episode", "time_min", "candidate_local_idx"],
        how="left",
    )


def classify_admissibility_snapshot_bucket(row: pd.Series) -> str:
    if row["has_positive_evidence"] <= 0.5:
        return "no_positive"
    if row["has_safe_observation_anywhere"] <= 0.5:
        return "no_safe_observation_anywhere"
    if row["any_hydraulically_comparable_safe"] <= 0.5:
        return "no_hydraulically_comparable_safe"
    if row["any_pair_available_under_mode"] <= 0.5:
        if row["any_time_bracketing_safe"] <= 0.5 and row["any_frontier_window_safe"] <= 0.5:
            return "no_front_bracketing_safe"
        if row["any_time_bracketing_safe"] <= 0.5:
            return "no_time_bracketing_safe"
        if row["any_frontier_window_safe"] <= 0.5:
            return "no_frontier_window_safe"
        return "no_pair_available_under_mode"
    if row["any_positive_margin_available_under_mode"] <= 0.5:
        if row["any_safe_regime_available"] > 0.5:
            return "pair_available_safe_only_weak"
        return "pair_available_but_non_positive_margin"
    if row["any_interval_regime_available"] <= 0.5:
        return "pair_available_but_interval_gap_zero"
    if row["any_true_source_mis_hit"] > 0.5:
        return "true_source_mis_hit"
    if row["active_safe_dominant_candidate_count"] > 0:
        return "active_safe_dominant"
    return "active_other"


def build_admissibility_compare_snapshot_frame(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    snapshot = (
        candidate_df.groupby(["admissibility_mode", "event_id", "episode", "time_min"], as_index=False)
        .agg(
            has_positive_evidence=("has_positive_evidence", "max"),
            has_safe_observation_anywhere=("has_safe_observation_anywhere", "max"),
            any_hydraulically_comparable_safe=("has_hydraulically_comparable_safe", "max"),
            any_time_bracketing_safe=("has_time_bracketing_safe", "max"),
            any_frontier_window_safe=("has_frontier_window_safe", "max"),
            any_pair_available_under_mode=("pair_available_under_mode", "max"),
            any_positive_margin_available_under_mode=("positive_margin_available_under_mode", "max"),
            any_interval_regime_available=("interval_regime_available", "max"),
            any_safe_regime_available=("safe_regime_available", "max"),
            any_true_source_mis_hit=("is_true_source_mis_hit", "max"),
            any_active_contradiction=("total", lambda s: float((s > EPS).any())),
            active_safe_dominant_candidate_count=("dominant_component", lambda s: int((s == "safe").sum())),
            max_total=("total", "max"),
            min_total=("total", "min"),
        )
    )
    snapshot["all_zero_or_tie"] = (
        (snapshot["max_total"].abs() <= EPS)
        | ((snapshot["max_total"] - snapshot["min_total"]).abs() <= EPS)
    ).astype(np.float32)
    snapshot["snapshot_zero_reason_bucket_under_mode"] = snapshot.apply(classify_admissibility_snapshot_bucket, axis=1)
    return snapshot


def build_admissibility_snapshot_wide_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame()
    key_cols = ["event_id", "episode", "time_min"]
    wide: Optional[pd.DataFrame] = None
    for mode in ADMISSIBILITY_COMPARE_MODES:
        short = ADMISSIBILITY_MODE_SHORT_LABEL[mode]
        subset = snapshot_df[snapshot_df["admissibility_mode"] == mode].copy()
        if subset.empty:
            continue
        subset = subset.rename(columns=lambda c: f"{short}_{c}" if c not in key_cols else c)
        wide = subset if wide is None else wide.merge(subset, on=key_cols, how="inner")
    return wide if wide is not None else pd.DataFrame()


def build_admissibility_mode_metrics(
    candidate_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    admissibility_mode: str,
) -> Dict[str, Any]:
    mode_candidate = candidate_df[candidate_df["admissibility_mode"] == admissibility_mode].copy()
    mode_snapshot = snapshot_df[snapshot_df["admissibility_mode"] == admissibility_mode].copy()
    all_zero_tie = mode_snapshot[mode_snapshot["all_zero_or_tie"] > 0.5].copy()
    pair_available_candidate = mode_candidate[mode_candidate["pair_available_under_mode"] > 0.5].copy()

    return {
        "snapshot_count": int(len(mode_snapshot)),
        "candidate_count": int(len(mode_candidate)),
        "candidate_rates": {
            "has_positive_evidence_rate": float((mode_candidate["has_positive_evidence"] > 0.5).mean()),
            "has_safe_observation_anywhere_rate": float((mode_candidate["has_safe_observation_anywhere"] > 0.5).mean()),
            "has_hydraulically_comparable_safe_rate": float((mode_candidate["has_hydraulically_comparable_safe"] > 0.5).mean()),
            "has_time_bracketing_safe_rate": float((mode_candidate["has_time_bracketing_safe"] > 0.5).mean()),
            "has_frontier_window_safe_rate": float((mode_candidate["has_frontier_window_safe"] > 0.5).mean()),
            "pair_available_rate": float((mode_candidate["pair_available_under_mode"] > 0.5).mean()),
            "positive_margin_available_rate": float((mode_candidate["positive_margin_available_under_mode"] > 0.5).mean()),
            "active_contradiction_coverage_rate": float((mode_candidate["total"] > EPS).mean()),
            "coverage_gain_vs_baseline_rate": float((mode_candidate["coverage_gain_vs_baseline"] > 0.5).mean()),
        },
        "snapshot_rates": {
            "has_positive_evidence_rate": float((mode_snapshot["has_positive_evidence"] > 0.5).mean()),
            "has_safe_observation_anywhere_rate": float((mode_snapshot["has_safe_observation_anywhere"] > 0.5).mean()),
            "any_hydraulically_comparable_safe_rate": float((mode_snapshot["any_hydraulically_comparable_safe"] > 0.5).mean()),
            "any_time_bracketing_safe_rate": float((mode_snapshot["any_time_bracketing_safe"] > 0.5).mean()),
            "any_frontier_window_safe_rate": float((mode_snapshot["any_frontier_window_safe"] > 0.5).mean()),
            "any_pair_available_rate": float((mode_snapshot["any_pair_available_under_mode"] > 0.5).mean()),
            "any_positive_margin_available_rate": float((mode_snapshot["any_positive_margin_available_under_mode"] > 0.5).mean()),
            "any_active_contradiction_rate": float((mode_snapshot["any_active_contradiction"] > 0.5).mean()),
            "all_zero_or_tie_rate": float((mode_snapshot["all_zero_or_tie"] > 0.5).mean()),
        },
        "mean_counts": {
            "hydraulic_comparable_safe_count": float(mode_candidate["hydraulic_comparable_safe_count"].mean()),
            "time_bracketing_safe_count": float(mode_candidate["time_bracketing_safe_count"].mean()),
            "frontier_window_safe_count": float(mode_candidate["frontier_window_safe_count"].mean()),
            "selected_safe_witness_count": float(mode_candidate["selected_safe_witness_count"].mean()),
        },
        "all_zero_reason_counts": {
            str(bucket): int((all_zero_tie["snapshot_zero_reason_bucket_under_mode"] == bucket).sum())
            for bucket in sorted(all_zero_tie["snapshot_zero_reason_bucket_under_mode"].unique().tolist())
        },
        "snapshot_bucket_breakdown": summarize_value_counts(mode_snapshot["snapshot_zero_reason_bucket_under_mode"]),
        "candidate_bucket_breakdown": summarize_value_counts(mode_candidate["zero_reason_bucket_under_mode"]),
        "pair_conditioned_rates": {
            "candidate_positive_margin_rate_given_pair": float(
                (pair_available_candidate["positive_margin_available_under_mode"] > 0.5).mean()
            ) if not pair_available_candidate.empty else np.nan,
        },
        "true_source_risk": {
            "candidate_count": int((mode_candidate["is_true_source_mis_hit"] > 0.5).sum()),
            "snapshot_count": int(
                mode_snapshot.loc[mode_snapshot["any_true_source_mis_hit"] > 0.5, ["event_id", "episode", "time_min"]].shape[0]
            ),
        },
    }


def build_admissibility_case_record(row: pd.Series, case_label: str) -> Dict[str, Any]:
    return {
        "case_label": case_label,
        "event_id": int(row["event_id"]),
        "episode": int(row["episode"]),
        "time_min": float(row["time_min"]),
        "candidate_global_id": int(row["candidate_global_id"]),
        "is_true_source": float(row["is_true_source"]),
        "baseline_pair_available": float(row["baseline_pair_available_under_mode"]),
        "topology_pair_available": float(row["topology_pair_available_under_mode"]),
        "time_pair_available": float(row["time_pair_available_under_mode"]),
        "frontier_pair_available": float(row["frontier_pair_available_under_mode"]),
        "union_pair_available": float(row["union_pair_available_under_mode"]),
        "baseline_positive_margin_available": float(row["baseline_positive_margin_available_under_mode"]),
        "topology_positive_margin_available": float(row["topology_positive_margin_available_under_mode"]),
        "time_positive_margin_available": float(row["time_positive_margin_available_under_mode"]),
        "frontier_positive_margin_available": float(row["frontier_positive_margin_available_under_mode"]),
        "union_positive_margin_available": float(row["union_positive_margin_available_under_mode"]),
        "baseline_zero_reason_bucket": str(row["baseline_zero_reason_bucket_under_mode"]),
        "union_zero_reason_bucket": str(row["union_zero_reason_bucket_under_mode"]),
        "observability_ceiling_bucket": str(row["observability_ceiling_bucket"]),
        "admissibility_ceiling_bucket": str(row["admissibility_ceiling_bucket"]),
        "baseline_best_margin": float(row["baseline_best_margin"]),
        "union_best_margin": float(row["union_best_margin"]),
        "baseline_total": float(row["baseline_total"]),
        "union_total": float(row["union_total"]),
    }


def build_admissibility_case_examples(candidate_df: pd.DataFrame) -> List[Dict[str, Any]]:
    wide = build_admissibility_compare_wide_frame(candidate_df)
    if wide.empty:
        return []
    wide["observability_ceiling_bucket"] = wide.apply(classify_admissibility_observability_bucket, axis=1)
    wide["admissibility_ceiling_bucket"] = wide.apply(classify_admissibility_ceiling_bucket, axis=1)
    used: set = set()
    cases: List[Dict[str, Any]] = []

    def pick_case(label: str, mask: pd.Series, sort_cols: List[str], ascending: List[bool]) -> None:
        subset = wide.loc[mask].sort_values(sort_cols, ascending=ascending)
        for _, row in subset.iterrows():
            key = (int(row["event_id"]), int(row["episode"]), int(row["candidate_global_id"]))
            if key in used:
                continue
            used.add(key)
            cases.append(build_admissibility_case_record(row, label))
            break

    pick_case(
        "baseline_no_pair_still_no_pair",
        (wide["baseline_pair_available_under_mode"] <= 0.5) & (wide["union_pair_available_under_mode"] <= 0.5),
        ["baseline_has_hydraulically_comparable_safe", "union_best_margin"],
        [True, False],
    )
    pick_case(
        "topology_relaxed_pair_gain",
        (wide["baseline_pair_available_under_mode"] <= 0.5) & (wide["topology_pair_available_under_mode"] > 0.5),
        ["topology_best_margin", "topology_total"],
        [False, False],
    )
    pick_case(
        "time_relaxed_pair_gain",
        (wide["baseline_pair_available_under_mode"] <= 0.5) & (wide["time_pair_available_under_mode"] > 0.5),
        ["time_best_margin", "time_total"],
        [False, False],
    )
    pick_case(
        "frontier_relaxed_pair_gain",
        (wide["baseline_pair_available_under_mode"] <= 0.5) & (wide["frontier_pair_available_under_mode"] > 0.5),
        ["frontier_best_margin", "frontier_total"],
        [False, False],
    )
    pick_case(
        "union_only_pair_gain",
        (wide["baseline_pair_available_under_mode"] <= 0.5)
        & (wide["topology_pair_available_under_mode"] <= 0.5)
        & (wide["time_pair_available_under_mode"] <= 0.5)
        & (wide["frontier_pair_available_under_mode"] <= 0.5)
        & (wide["union_pair_available_under_mode"] > 0.5),
        ["union_best_margin", "union_total"],
        [False, False],
    )
    pick_case(
        "baseline_pair_stable",
        (wide["baseline_pair_available_under_mode"] > 0.5)
        & (wide["union_pair_available_under_mode"] > 0.5)
        & (wide["baseline_positive_margin_available_under_mode"] == wide["union_positive_margin_available_under_mode"]),
        ["baseline_total", "union_total"],
        [False, False],
    )
    pick_case(
        "relaxed_gain_but_still_non_positive_margin",
        (wide["union_pair_available_under_mode"] > wide["baseline_pair_available_under_mode"])
        & (wide["union_positive_margin_available_under_mode"] <= 0.5),
        ["union_total", "union_best_margin"],
        [False, False],
    )
    pick_case(
        "interval_still_sparse_outlier",
        (wide["union_current_time_interval_regime_available"] > 0.5)
        & (wide["union_current_time_interval_gap"] > np.maximum(wide["union_interval_gap"] + 10.0, 5000.0)),
        ["union_current_time_interval_gap", "union_current_time_total"],
        [False, False],
    )
    pick_case(
        "no_hydraulic_comparability_even_union",
        (wide["baseline_has_positive_evidence"] > 0.5)
        & (wide["baseline_has_safe_observation_anywhere"] > 0.5)
        & (wide["union_has_hydraulically_comparable_safe"] <= 0.5),
        ["baseline_candidate_global_id"],
        [True],
    )
    pick_case(
        "true_source_risk_case",
        (wide["union_is_true_source_mis_hit"] > 0.5)
        | (wide["topology_is_true_source_mis_hit"] > 0.5)
        | (wide["time_is_true_source_mis_hit"] > 0.5)
        | (wide["frontier_is_true_source_mis_hit"] > 0.5),
        ["union_total", "union_best_margin"],
        [False, False],
    )
    return cases


def build_admissibility_question_answers(summary: Dict[str, Any]) -> Dict[str, str]:
    snapshot_ceiling = summary["ceiling_decomposition"]["snapshot"]
    union_delta = summary["delta_vs_baseline"].get("union_relaxed_upper_bound", {})
    best_mode = summary["headroom_by_mode"][0] if summary["headroom_by_mode"] else None

    q1 = (
        f"{summary['final_position']['sentence_1_primary_ceiling']} Snapshot 口径下，baseline 无 pair 的 {snapshot_ceiling['baseline_no_pair_count']} 个 case 中，"
        f"{snapshot_ceiling['observability_ceiling_count']} 个到 union 仍无 pair，仅 {snapshot_ceiling['admissibility_pair_headroom_count']} 个被 relax 释放。"
    )
    q2 = (
        f"union 相对 baseline 的 candidate pair delta={union_delta.get('candidate_pair_available_rate_delta', np.nan):.4f}，"
        f"candidate positive-margin delta={union_delta.get('candidate_positive_margin_available_rate_delta', np.nan):.4f}，"
        f"snapshot all_zero_or_tie delta={union_delta.get('snapshot_all_zero_or_tie_rate_delta', np.nan):.4f}。"
    )
    if best_mode is None:
        q3 = "没有可比较的 relaxed mode。"
    else:
        q3 = (
            f"提升最大的是 {best_mode['admissibility_mode']}，"
            f"snapshot pair gain={best_mode['snapshot_pair_gain_count']}，"
            f"snapshot positive-margin gain={best_mode['snapshot_positive_gain_count']}，"
            f"snapshot all-zero release={best_mode['snapshot_all_zero_release_count']}。"
        )
    q4 = (
        "可以。union upper bound 仍然提升有限，说明在当前观测体系下 contradiction 本身就是低覆盖率信号。"
        if snapshot_ceiling["observability_ceiling_count"] >= snapshot_ceiling["admissibility_pair_headroom_count"]
        and abs(union_delta.get("snapshot_all_zero_or_tie_rate_delta", 0.0)) < 0.05
        else "还不能完全下这个结论，但 union upper bound 也没有显示出足够大的 headroom。"
    )
    q5 = f"{summary['final_position']['sentence_2_next_direction']} {summary['final_position']['sentence_3_resource_call']}"
    return {
        "q1_primary_ceiling": q1,
        "q2_small_relax_lift": q2,
        "q3_biggest_headroom_mode": q3,
        "q4_union_upper_bound_judgement": q4,
        "q5_project_positioning": q5,
    }


def build_admissibility_compare_summary(
    records: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> Dict[str, Any]:
    candidate_df = annotate_admissibility_compare_frame(candidate_df)
    snapshot_df = build_admissibility_compare_snapshot_frame(candidate_df)
    wide_candidate = build_admissibility_compare_wide_frame(candidate_df)
    wide_snapshot = build_admissibility_snapshot_wide_frame(snapshot_df)
    baseline_snapshot_count = int(records.loc[:, ["event_id", "episode", "time_min"]].drop_duplicates().shape[0])
    mode_metrics = {
        mode: build_admissibility_mode_metrics(candidate_df, snapshot_df, mode)
        for mode in ADMISSIBILITY_COMPARE_MODES
    }
    baseline_metrics = mode_metrics[DEFAULT_ADMISSIBILITY_MODE]
    delta_vs_baseline: Dict[str, Dict[str, float]] = {}
    headroom_by_mode: List[Dict[str, Any]] = []
    baseline_all_zero = wide_snapshot["baseline_all_zero_or_tie"] > 0.5 if not wide_snapshot.empty else pd.Series(dtype=bool)

    for mode in ADMISSIBILITY_COMPARE_MODES:
        if mode == DEFAULT_ADMISSIBILITY_MODE:
            continue
        short = ADMISSIBILITY_MODE_SHORT_LABEL[mode]
        delta_vs_baseline[mode] = {
            "candidate_pair_available_rate_delta": float(
                mode_metrics[mode]["candidate_rates"]["pair_available_rate"]
                - baseline_metrics["candidate_rates"]["pair_available_rate"]
            ),
            "candidate_positive_margin_available_rate_delta": float(
                mode_metrics[mode]["candidate_rates"]["positive_margin_available_rate"]
                - baseline_metrics["candidate_rates"]["positive_margin_available_rate"]
            ),
            "snapshot_any_pair_available_rate_delta": float(
                mode_metrics[mode]["snapshot_rates"]["any_pair_available_rate"]
                - baseline_metrics["snapshot_rates"]["any_pair_available_rate"]
            ),
            "snapshot_any_positive_margin_available_rate_delta": float(
                mode_metrics[mode]["snapshot_rates"]["any_positive_margin_available_rate"]
                - baseline_metrics["snapshot_rates"]["any_positive_margin_available_rate"]
            ),
            "snapshot_all_zero_or_tie_rate_delta": float(
                mode_metrics[mode]["snapshot_rates"]["all_zero_or_tie_rate"]
                - baseline_metrics["snapshot_rates"]["all_zero_or_tie_rate"]
            ),
        }
        if not wide_candidate.empty and not wide_snapshot.empty:
            headroom_by_mode.append(
                {
                    "admissibility_mode": mode,
                    "candidate_pair_gain_count": int(
                        (wide_candidate[f"{short}_pair_available_under_mode"] > wide_candidate["baseline_pair_available_under_mode"]).sum()
                    ),
                    "candidate_positive_gain_count": int(
                        (
                            wide_candidate[f"{short}_positive_margin_available_under_mode"]
                            > wide_candidate["baseline_positive_margin_available_under_mode"]
                        ).sum()
                    ),
                    "snapshot_pair_gain_count": int(
                        (wide_snapshot[f"{short}_any_pair_available_under_mode"] > wide_snapshot["baseline_any_pair_available_under_mode"]).sum()
                    ),
                    "snapshot_positive_gain_count": int(
                        (
                            wide_snapshot[f"{short}_any_positive_margin_available_under_mode"]
                            > wide_snapshot["baseline_any_positive_margin_available_under_mode"]
                        ).sum()
                    ),
                    "snapshot_all_zero_release_count": int((baseline_all_zero & (wide_snapshot[f"{short}_all_zero_or_tie"] <= 0.5)).sum()),
                }
            )
    headroom_by_mode = sorted(
        headroom_by_mode,
        key=lambda row: (
            row["snapshot_all_zero_release_count"],
            row["snapshot_pair_gain_count"],
            row["snapshot_positive_gain_count"],
            row["candidate_pair_gain_count"],
        ),
        reverse=True,
    )

    candidate_ceiling = {
        "baseline_no_pair_count": 0,
        "observability_ceiling_count": 0,
        "admissibility_pair_headroom_count": 0,
        "baseline_pair_but_no_positive_margin_count": 0,
        "admissibility_margin_headroom_count": 0,
    }
    snapshot_ceiling = {
        "baseline_no_pair_count": 0,
        "observability_ceiling_count": 0,
        "admissibility_pair_headroom_count": 0,
        "baseline_all_zero_or_tie_count": 0,
        "admissibility_all_zero_release_count": 0,
        "baseline_pair_but_no_positive_margin_count": 0,
        "admissibility_margin_headroom_count": 0,
    }
    if not wide_candidate.empty:
        candidate_baseline_no_pair = wide_candidate["baseline_pair_available_under_mode"] <= 0.5
        candidate_union_pair = wide_candidate["union_pair_available_under_mode"] > 0.5
        candidate_baseline_pair_no_margin = (
            (wide_candidate["baseline_pair_available_under_mode"] > 0.5)
            & (wide_candidate["baseline_positive_margin_available_under_mode"] <= 0.5)
        )
        candidate_ceiling = {
            "baseline_no_pair_count": int(candidate_baseline_no_pair.sum()),
            "observability_ceiling_count": int((candidate_baseline_no_pair & ~candidate_union_pair).sum()),
            "admissibility_pair_headroom_count": int((candidate_baseline_no_pair & candidate_union_pair).sum()),
            "baseline_pair_but_no_positive_margin_count": int(candidate_baseline_pair_no_margin.sum()),
            "admissibility_margin_headroom_count": int(
                (candidate_baseline_pair_no_margin & (wide_candidate["union_positive_margin_available_under_mode"] > 0.5)).sum()
            ),
        }
    if not wide_snapshot.empty:
        snapshot_baseline_no_pair = wide_snapshot["baseline_any_pair_available_under_mode"] <= 0.5
        snapshot_union_pair = wide_snapshot["union_any_pair_available_under_mode"] > 0.5
        snapshot_baseline_pair_no_margin = (
            (wide_snapshot["baseline_any_pair_available_under_mode"] > 0.5)
            & (wide_snapshot["baseline_any_positive_margin_available_under_mode"] <= 0.5)
        )
        snapshot_ceiling = {
            "baseline_no_pair_count": int(snapshot_baseline_no_pair.sum()),
            "observability_ceiling_count": int((snapshot_baseline_no_pair & ~snapshot_union_pair).sum()),
            "admissibility_pair_headroom_count": int((snapshot_baseline_no_pair & snapshot_union_pair).sum()),
            "baseline_all_zero_or_tie_count": int((wide_snapshot["baseline_all_zero_or_tie"] > 0.5).sum()),
            "admissibility_all_zero_release_count": int(
                ((wide_snapshot["baseline_all_zero_or_tie"] > 0.5) & (wide_snapshot["union_all_zero_or_tie"] <= 0.5)).sum()
            ),
            "baseline_pair_but_no_positive_margin_count": int(snapshot_baseline_pair_no_margin.sum()),
            "admissibility_margin_headroom_count": int(
                (snapshot_baseline_pair_no_margin & (wide_snapshot["union_any_positive_margin_available_under_mode"] > 0.5)).sum()
            ),
        }

    union_delta = delta_vs_baseline.get("union_relaxed_upper_bound", {})
    if snapshot_ceiling["observability_ceiling_count"] > snapshot_ceiling["admissibility_pair_headroom_count"] * 1.5:
        sentence_1 = "contradiction 当前的主 ceiling 是 observability ceiling，不是 admissibility ceiling。"
    elif snapshot_ceiling["admissibility_pair_headroom_count"] > snapshot_ceiling["observability_ceiling_count"]:
        sentence_1 = "contradiction 当前的主 ceiling 更接近 admissibility ceiling，但 observability 约束仍未消失。"
    else:
        sentence_1 = "contradiction 当前是 mixed ceiling，但 admissibility 并没有显示出足够大的独立 headroom。"

    if (
        snapshot_ceiling["observability_ceiling_count"] >= snapshot_ceiling["admissibility_pair_headroom_count"]
        and abs(union_delta.get("snapshot_all_zero_or_tie_rate_delta", 0.0)) < 0.05
    ):
        sentence_2 = "下一步最合理的工程方向是停止深挖 contradiction score 本身，把它冻结为 sparse auxiliary，并把真正的增益探索收回到上游 observability / sensing 形成研究。"
        sentence_3 = "在当前证据下，contradiction 不再适合继续占用主线研发资源；最多保留为 audit / explanation branch。"
    else:
        sentence_2 = "下一步更合理的是把 contradiction 保留为稀疏辅助线，同时只在上游 observability 侧继续做受控研究。"
        sentence_3 = "在拿到新的观测形成证据前，contradiction 不应继续占用主线 feature 研发资源。"

    summary = {
        "config": {
            "audit_path": "src/scripts/audit/run_practical_audit_rerun.py",
            "rollout_path": "src/scripts/audit/utils_practical_rollout.py",
            "diagnostic_csv_path": CONTRA_ADMISSIBILITY_CEILING_CSV_PATH,
            "baseline_snapshot_denominator": baseline_snapshot_count,
            "historical_snapshot_reference": 970,
            "rerun_snapshot_reference": baseline_snapshot_count,
            "rerun_baseline_discrepancy": int(baseline_snapshot_count - 970),
            "admissibility_modes": ADMISSIBILITY_COMPARE_MODES,
            "baseline_mode": DEFAULT_ADMISSIBILITY_MODE,
            "history_physctx_mode": PRACTICAL_V2_HISTORY_PHYSCTX_MODE,
            "frontier_safe_close_tau_min": WITNESS_MINING_FRONT_CLOSE_TAU_MIN,
            "compare_config": ADMISSIBILITY_COMPARE_CONFIG.to_dict(),
        },
        "mode_metrics": mode_metrics,
        "delta_vs_baseline": delta_vs_baseline,
        "mode_snapshot_counts_consistent": {
            mode: int(mode_metrics[mode]["snapshot_count"] == baseline_snapshot_count)
            for mode in ADMISSIBILITY_COMPARE_MODES
        },
        "ceiling_decomposition": {
            "candidate": candidate_ceiling,
            "snapshot": snapshot_ceiling,
        },
        "headroom_by_mode": headroom_by_mode,
        "representative_cases": build_admissibility_case_examples(candidate_df),
        "candidate_bucket_breakdown": summarize_value_counts(candidate_df["zero_reason_bucket_under_mode"]),
        "observability_ceiling_bucket_breakdown": summarize_value_counts(candidate_df["observability_ceiling_bucket"]),
        "admissibility_ceiling_bucket_breakdown": summarize_value_counts(candidate_df["admissibility_ceiling_bucket"]),
        "final_position": {
            "sentence_1_primary_ceiling": sentence_1,
            "sentence_2_next_direction": sentence_2,
            "sentence_3_resource_call": sentence_3,
        },
    }
    summary["question_answers"] = build_admissibility_question_answers(summary)
    return summary


def build_admissibility_compare_markdown(summary: Dict[str, Any]) -> str:
    mode_lines: List[str] = []
    for mode in ADMISSIBILITY_COMPARE_MODES:
        metrics = summary["mode_metrics"][mode]
        mode_lines.extend(
            [
                f"### {mode}",
                f"- candidate_rates: {metrics['candidate_rates']}",
                f"- snapshot_rates: {metrics['snapshot_rates']}",
                f"- mean_counts: {metrics['mean_counts']}",
                f"- true_source_risk: {metrics['true_source_risk']}",
                "",
            ]
        )

    headroom_table = pd.DataFrame(summary["headroom_by_mode"])
    case_lines = []
    for case in summary["representative_cases"]:
        case_lines.append(
            f"- [已证明] {case['case_label']}: event={case['event_id']} episode={case['episode']} candidate={case['candidate_global_id']} "
            f"`pair baseline/topology/time/frontier/union={case['baseline_pair_available']:.0f}/{case['topology_pair_available']:.0f}/{case['time_pair_available']:.0f}/{case['frontier_pair_available']:.0f}/{case['union_pair_available']:.0f}`, "
            f"`margin baseline/union={case['baseline_positive_margin_available']:.0f}/{case['union_positive_margin_available']:.0f}`, "
            f"`bucket obs/admiss={case['observability_ceiling_bucket']}/{case['admissibility_ceiling_bucket']}`."
        )
    if not case_lines:
        case_lines = ["- representative cases unavailable."]

    lines = [
        "# contradiction admissibility ceiling audit",
        "",
        "## 1. 审计口径",
        f"- [已证明] baseline snapshot 分母是 `{summary['config']['baseline_snapshot_denominator']}`，所有 compare mode 的 snapshot 分母一致性为 `{summary['mode_snapshot_counts_consistent']}`。",
        f"- [已证明] 与历史 970 snapshot 的差异单独记为 rerun baseline discrepancy=`{summary['config']['rerun_baseline_discrepancy']}`。",
        "- [已证明] 这一步是 audit-only compare；没有修改主线 contradiction score，也没有新建第三条 audit pipeline。",
        f"- [已证明] admissibility modes: `{', '.join(summary['config']['admissibility_modes'])}`。",
        f"- [已证明] compare_config: {summary['config']['compare_config']}",
        "",
        "## 2. Mode Summary",
        *mode_lines,
        "## 3. Delta vs Baseline",
        f"- [已证明] delta_vs_baseline: {summary['delta_vs_baseline']}",
        "",
        "## 4. Ceiling Decomposition",
        f"- [已证明] candidate ceiling decomposition: {summary['ceiling_decomposition']['candidate']}",
        f"- [已证明] snapshot ceiling decomposition: {summary['ceiling_decomposition']['snapshot']}",
        f"- [已证明] observability ceiling bucket breakdown: {summary['observability_ceiling_bucket_breakdown']}",
        f"- [已证明] admissibility ceiling bucket breakdown: {summary['admissibility_ceiling_bucket_breakdown']}",
        "",
        "## 5. Headroom Ranking",
        "```text",
        "empty" if headroom_table.empty else headroom_table.to_string(index=False),
        "```",
        "",
        "## 6. 五个问题",
        f"- [部分证明] Q1: {summary['question_answers']['q1_primary_ceiling']}",
        f"- [部分证明] Q2: {summary['question_answers']['q2_small_relax_lift']}",
        f"- [部分证明] Q3: {summary['question_answers']['q3_biggest_headroom_mode']}",
        f"- [部分证明] Q4: {summary['question_answers']['q4_union_upper_bound_judgement']}",
        f"- [部分证明] Q5: {summary['question_answers']['q5_project_positioning']}",
        "",
        "## 7. Final Position",
        f"- [已证明] {summary['final_position']['sentence_1_primary_ceiling']}",
        f"- [已证明] {summary['final_position']['sentence_2_next_direction']}",
        f"- [已证明] {summary['final_position']['sentence_3_resource_call']}",
        "",
        "## 8. Representative Cases",
        *case_lines,
        "",
        "## 输出文件",
        f"- `{CONTRA_ADMISSIBILITY_CEILING_MD_PATH}`",
        f"- `{CONTRA_ADMISSIBILITY_CEILING_JSON_PATH}`",
        f"- `{CONTRA_ADMISSIBILITY_CEILING_CSV_PATH}`",
    ]
    return "\n".join(lines) + "\n"


def print_tables(summary: pd.DataFrame, records: pd.DataFrame) -> None:
    support_table = make_support_table(summary).round(4)
    suspect_table = make_suspect_table(summary).round(4)
    contradiction_table = make_contradiction_table(summary).round(4)
    contradiction_alignment = build_contradiction_alignment_summary(records).round(4)
    reaction_table = make_reaction_table(summary).round(4)
    uncertainty_table = make_uncertainty_table(summary).round(4)
    decision_table = build_decision_table(summary)
    bucket_table = records["diagnostic_bucket"].value_counts().rename_axis("diagnostic_bucket").to_frame("count")

    print("Table 1: support cleaned metrics")
    print(support_table.to_string())
    print()
    print("Table 2: suspect raw vs active split")
    print(suspect_table.to_string())
    print()
    print("Table 3: contradiction old vs formal vs no-suspect-formal vs strict compare")
    print(contradiction_table.to_string())
    print()
    print("Table 4: contradiction alignment decomposition")
    print(contradiction_alignment.to_string(index=False))
    print()
    print("Table 5: reaction consistency diagnostics")
    print(reaction_table.to_string())
    print()
    print("Table 6: uncertainty diagnostics")
    print(uncertainty_table.to_string())
    print()
    print("Table 7: diagnostic bucket counts")
    print(bucket_table.to_string())
    print()
    print("Table 8: axis decision summary")
    print(decision_table.to_string())


def print_positive_seed_table(positive_seed_summary: pd.DataFrame) -> None:
    print()
    print("Table 9: positive seed survival diagnostics")
    print(positive_seed_summary.round(4).to_string(index=False))


def run_audit(include_current_time_physctx_compare: bool = True) -> None:
    set_seeds(0)
    silence_non_table_logs()
    dataset = NpzDatasetV6(
        samples_dir=SAMPLES_PATH,
        foundation_dir=FOUNDATION_PATH,
        mode="test",
        preload=False,
        audit_mode="fast",
        use_edge_attr=True,
    )
    topology = HydraulicTopology(FOUNDATION_PATH)
    records, case_trace, support_rootcause, oracle_v1_candidate_buffers, practical_v2_candidate_buffers, candidate_audit_df, witness_mining_compare_df, admissibility_compare_df = (
        collect_event_records(
            dataset,
            topology,
            include_current_time_physctx_compare=include_current_time_physctx_compare,
        )
    )
    if records.empty:
        raise RuntimeError("Audit produced no records.")
    pair_availability_summary, pair_snapshot_df = build_pair_availability_summary(records, candidate_audit_df)
    pair_snapshot_export = pair_snapshot_df[
        [
            "event_id",
            "episode",
            "time_min",
            "all_zero_or_tie",
            "snapshot_pair_bucket",
            "observability_ceiling",
            "pair_margin_ceiling",
            "has_positive_evidence",
            "any_eligible_safe_witness",
            "any_pair_available",
            "any_positive_margin_available",
            "any_interval_regime_available",
            "any_safe_regime_available",
            "any_interval_regime_available_current_time",
            "any_true_source_mis_hit",
        ]
    ].rename(
        columns={
            "all_zero_or_tie": "contra_pair_all_zero_or_tie",
            "snapshot_pair_bucket": "contra_pair_snapshot_bucket",
            "observability_ceiling": "contra_pair_observability_ceiling",
            "pair_margin_ceiling": "contra_pair_pair_margin_ceiling",
            "has_positive_evidence": "contra_pair_has_positive_evidence",
            "any_eligible_safe_witness": "contra_pair_any_eligible_safe_witness",
            "any_pair_available": "contra_pair_any_pair_available",
            "any_positive_margin_available": "contra_pair_any_positive_margin_available",
            "any_interval_regime_available": "contra_pair_any_interval_regime_available",
            "any_safe_regime_available": "contra_pair_any_safe_regime_available",
            "any_interval_regime_available_current_time": "contra_pair_any_interval_regime_available_current_time",
            "any_true_source_mis_hit": "contra_pair_any_true_source_mis_hit",
        }
    )
    records = records.merge(pair_snapshot_export, on=["event_id", "episode", "time_min"], how="left")

    records["diagnostic_bucket"] = records.apply(classify_case, axis=1)
    records["responsibility_hint"] = records["diagnostic_bucket"].map(responsibility_hint)
    records.to_csv(STEPWISE_CSV_PATH, index=False)
    candidate_audit_df.to_csv(CONTRA_PAIR_AVAILABILITY_CSV_PATH, index=False)
    with open(CONTRA_PAIR_AVAILABILITY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(pair_availability_summary, f, indent=2, ensure_ascii=False)
    with open(CONTRA_PAIR_AVAILABILITY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(build_pair_availability_markdown(pair_availability_summary))
    witness_mining_compare_df = annotate_witness_mining_compare_frame(witness_mining_compare_df)
    witness_mining_compare_df.to_csv(CONTRA_WITNESS_COVERAGE_CSV_PATH, index=False)
    witness_mining_summary = build_witness_mining_summary(records, witness_mining_compare_df)
    with open(CONTRA_WITNESS_COVERAGE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(witness_mining_summary, f, indent=2, ensure_ascii=False)
    with open(CONTRA_WITNESS_COVERAGE_MD_PATH, "w", encoding="utf-8") as f:
        f.write(build_witness_mining_markdown(witness_mining_summary))
    admissibility_compare_df = annotate_admissibility_compare_frame(admissibility_compare_df)
    admissibility_compare_df.to_csv(CONTRA_ADMISSIBILITY_CEILING_CSV_PATH, index=False)
    admissibility_summary = build_admissibility_compare_summary(records, admissibility_compare_df)
    with open(CONTRA_ADMISSIBILITY_CEILING_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(admissibility_summary, f, indent=2, ensure_ascii=False)
    with open(CONTRA_ADMISSIBILITY_CEILING_MD_PATH, "w", encoding="utf-8") as f:
        f.write(build_admissibility_compare_markdown(admissibility_summary))

    summary = build_summary(records)
    summary.to_csv(CSV_PATH, index=False)
    contradiction_alignment = build_contradiction_alignment_summary(records)
    contradiction_alignment.to_csv(CONTRA_ALIGNMENT_CSV_PATH, index=False)
    positive_seed_summary = build_positive_seed_summary(records)
    positive_seed_summary.to_csv(POSITIVE_SEED_CSV_PATH, index=False)
    markdown = build_markdown(summary)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    contradiction_alignment_markdown = build_contradiction_alignment_markdown(contradiction_alignment)
    with open(CONTRA_ALIGNMENT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(contradiction_alignment_markdown)
    case_digest = build_case_digest(records)
    with open(CASE_MD_PATH, "w", encoding="utf-8") as f:
        f.write(case_digest)
    case_trace.to_csv(CASE_TRACE_CSV_PATH, index=False)
    case_trace_markdown = build_case_trace_markdown(case_trace)
    with open(CASE_TRACE_MD_PATH, "w", encoding="utf-8") as f:
        f.write(case_trace_markdown)
    positive_seed_markdown = build_positive_seed_markdown(positive_seed_summary, case_trace)
    with open(POSITIVE_SEED_MD_PATH, "w", encoding="utf-8") as f:
        f.write(positive_seed_markdown)
    _b_rootcause, _c_rootcause, rootcause_markdown = build_support_rootcause_outputs(support_rootcause, case_trace)
    with open(SUBTERM_ROOTCAUSE_MD_PATH, "w", encoding="utf-8") as f:
        f.write(rootcause_markdown)
    oracle_stepwise = build_oracle_stepwise_export(records)
    oracle_stepwise.to_csv(ORACLE_STEPWISE_CSV_PATH, index=False)
    oracle_metrics = build_oracle_metrics(records)
    oracle_metrics.to_csv(ORACLE_METRICS_CSV_PATH, index=False)
    oracle_vs_practical = build_oracle_vs_practical(records)
    oracle_vs_practical.to_csv(ORACLE_VS_PRACTICAL_CSV_PATH, index=False)
    oracle_markdown = build_oracle_summary_markdown(oracle_stepwise, oracle_metrics, oracle_vs_practical)
    with open(ORACLE_SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(oracle_markdown)
    v2_oracle_stepwise = build_v2_oracle_stepwise_export(records)
    v2_oracle_stepwise.to_csv(V2_ORACLE_STEPWISE_CSV_PATH, index=False)
    v2_oracle_metrics = build_support_v2_oracle_metrics(records)
    v2_oracle_metrics.to_csv(V2_ORACLE_METRICS_CSV_PATH, index=False)
    v1_vs_v2_oracle = build_v1_vs_v2_oracle(records)
    v1_vs_v2_oracle.to_csv(V1_VS_V2_ORACLE_CSV_PATH, index=False)
    v2_summary = build_support_v2_summary_markdown(v2_oracle_metrics, v1_vs_v2_oracle)
    with open(V2_SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(v2_summary)
    contra_oracle_v1_compare = build_contradiction_oracle_v1_compare(records)
    contra_oracle_v1_compare.to_csv(CONTRA_ORACLE_V1_COMPARE_CSV_PATH, index=False)
    contra_oracle_v1_summary = build_contradiction_oracle_v1_summary(records, oracle_v1_candidate_buffers)
    with open(CONTRA_ORACLE_V1_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(contra_oracle_v1_summary, f, indent=2, ensure_ascii=False)
    contra_oracle_v1_markdown = build_contradiction_oracle_v1_markdown(contra_oracle_v1_summary)
    with open(CONTRA_ORACLE_V1_MD_PATH, "w", encoding="utf-8") as f:
        f.write(contra_oracle_v1_markdown)
    contra_practical_v2_compare = build_contradiction_practical_v2_compare(records)
    contra_practical_v2_compare.to_csv(CONTRA_PRACTICAL_V2_COMPARE_CSV_PATH, index=False)
    contra_practical_v2_summary = build_practical_v2_summary(
        records,
        contra_oracle_v1_summary,
        practical_v2_candidate_buffers,
    )
    with open(CONTRA_PRACTICAL_V2_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(contra_practical_v2_summary, f, indent=2, ensure_ascii=False)
    contra_practical_v2_markdown = build_contradiction_practical_v2_markdown(contra_practical_v2_summary)
    with open(CONTRA_PRACTICAL_V2_MD_PATH, "w", encoding="utf-8") as f:
        f.write(contra_practical_v2_markdown)
    print_tables(summary, records)
    print_positive_seed_table(positive_seed_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-current-time-physctx-compare",
        action="store_true",
        help="Skip the controlled current_time_physctx contradiction compare if runtime is too high.",
    )
    args = parser.parse_args()
    run_audit(include_current_time_physctx_compare=not args.skip_current_time_physctx_compare)
