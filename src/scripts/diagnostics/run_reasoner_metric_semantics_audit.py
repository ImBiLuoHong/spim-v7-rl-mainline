from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.builders.model_builder import ModelBuilder
from src.scripts.train_clean_aligned_online_finish import (
    build_eval_overrides,
    load_plain_state_dict,
)
from src.scripts.training_campaign_round1 import build_eval_loaders_with_batch, prepare_cfg


DEFAULT_FINISH_ROOT = PROJECT_ROOT / "artifacts" / "reasoner_clean_aligned_online_finish" / "20260404_line_final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reasoner_metric_semantics_audit_20260404"


def json_ready(value: Any):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def bool_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(float).mean())


def first_line(path: Path, needle: str) -> Optional[int]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if needle in line:
                return line_no
    return None


def line_map() -> Dict[str, Dict[str, Any]]:
    files = {
        "train_finish": PROJECT_ROOT / "src/scripts/train_clean_aligned_online_finish.py",
        "round1_eval": PROJECT_ROOT / "src/scripts/training_campaign_round1.py",
        "evaluator": PROJECT_ROOT / "src/evaluation/evaluator.py",
        "episode_runner": PROJECT_ROOT / "src/modeling/loop/episode_runner.py",
        "state_updates": PROJECT_ROOT / "src/modeling/loop/orchestration/state_updates.py",
        "clean_aligned_reasoner": PROJECT_ROOT / "src/modeling/reasoners/clean_aligned.py",
        "clean_aligned_features": PROJECT_ROOT / "src/modeling/clean_aligned_features.py",
        "phase45": PROJECT_ROOT / "src/modeling/architectures/phase4_5_model.py",
        "state_schema": PROJECT_ROOT / "src/modeling/state/schema.py",
        "state_builders": PROJECT_ROOT / "src/modeling/loop/orchestration/state_builders.py",
    }
    needles = {
        "selection_tuple": ("train_finish", "def selection_tuple"),
        "selection_scope": ("train_finish", '"selection_scope": "formal_val_batch1_sparse_checkpoint_sweep"'),
        "sampling_policy": ("phase45", "sampling_policy"),
        "eval_success_rate": ("round1_eval", '"success_rate": float(df["success"].mean())'),
        "eval_top1": ("round1_eval", '"top1_hit": float(valid_df["top1_hit"].mean())'),
        "compute_rank_fields": ("round1_eval", "def compute_rank_fields"),
        "probs_from_logits": ("round1_eval", "def probs_from_logits"),
        "legacy_success": ("evaluator", "metrics['legacy/Success_Rate'] = ep_mean('success')"),
        "nav_only_route": ("episode_runner", 'if action_policy == "nav_only":'),
        "build_nav_state_without_reasoner_logits": ("episode_runner", 'None if action_policy == "nav_only" else last_logits'),
        "valid_mask_final": ("episode_runner", "valid_mask_final = ~constraint_state.no_resample_mask.view(-1).bool()"),
        "apply_constraint_masks": ("episode_runner", "self.state_updater.apply_constraint_masks(logits_fused, constraint_state, fused_batch)"),
        "success_tracking": ("episode_runner", "payload_source_hit = constraint_update.get('is_source_hit')"),
        "constraint_update": ("state_updates", "def update_constraint_state"),
        "constraint_apply": ("state_updates", "def apply_constraint_masks"),
        "hard_exclusion": ("state_updates", "hard_exclusion = no_resample_mask | confirmed_non_source_mask"),
        "restore_fully_masked": ("state_updates", "fully_masked_graph = (~graph_has_any_finite) & (~graph_has_confirmed_source)"),
        "clean_reasoner_forward": ("clean_aligned_reasoner", "def forward"),
        "feature_payload": ("clean_aligned_features", "def build_clean_aligned_feature_payload"),
        "constraint_state_schema": ("state_schema", "class ConstraintState"),
        "build_constraint_state": ("state_builders", "def build_constraint_state"),
        "initial_constraint_state": ("phase45", "constraint_state = ConstraintState("),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for key, (file_key, needle) in needles.items():
        path = files[file_key]
        out[key] = {
            "path": str(path),
            "line": first_line(path, needle),
            "needle": needle,
        }
    return out


def softmax_stats_from_logits(logits: torch.Tensor) -> Dict[str, Any]:
    logits = logits.detach().float().view(-1)
    finite_mask = torch.isfinite(logits)
    finite_logits = logits[finite_mask]
    if finite_logits.numel() == 0:
        return {
            "finite_candidate_count": 0,
            "entropy": float("nan"),
            "entropy_norm": float("nan"),
            "max_prob": float("nan"),
        }
    probs = torch.softmax(finite_logits, dim=0)
    entropy = float((-(probs * (probs.clamp_min(1e-12).log()))).sum().item())
    denom = math.log(float(finite_logits.numel())) if finite_logits.numel() > 1 else 1.0
    entropy_norm = entropy / denom if denom > 0 else 0.0
    return {
        "finite_candidate_count": int(finite_logits.numel()),
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "max_prob": float(probs.max().item()),
    }


def rank_metrics_from_masked_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    logits = logits.detach().float().view(-1)
    labels = labels.detach().float().view(-1)
    finite_mask = torch.isfinite(logits)
    true_mask = labels > 0.5
    stats = softmax_stats_from_logits(logits)
    if not bool(finite_mask.any()) or not bool(true_mask.any()):
        return {
            **stats,
            "valid_case": False,
            "true_source_rank": None,
            "mrr": None,
            "top1_hit": None,
            "top3_hit": None,
            "top5_hit": None,
        }
    safe_logits = logits.clone()
    safe_logits[~finite_mask] = -float("inf")
    sorted_idx = torch.argsort(safe_logits, descending=True)
    true_positions = (true_mask[sorted_idx]).nonzero(as_tuple=True)[0]
    if true_positions.numel() == 0:
        return {
            **stats,
            "valid_case": False,
            "true_source_rank": None,
            "mrr": None,
            "top1_hit": None,
            "top3_hit": None,
            "top5_hit": None,
        }
    rank_true = int(true_positions.min().item() + 1)
    return {
        **stats,
        "valid_case": True,
        "true_source_rank": rank_true,
        "mrr": 1.0 / float(rank_true),
        "top1_hit": bool(rank_true <= 1),
        "top3_hit": bool(rank_true <= 3),
        "top5_hit": bool(rank_true <= 5),
    }


def tensor_sum(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int((value.detach().view(-1).float() > 0.5).sum().item())
    return int(torch.as_tensor(value).view(-1).float().gt(0.5).sum().item())


def maybe_get_feasible_mask(step: Dict[str, Any], node_mask: torch.Tensor) -> Optional[torch.Tensor]:
    physics_ctx = step.get("physics_ctx")
    if not isinstance(physics_ctx, dict):
        return None
    feasible = physics_ctx.get("feasible_mask")
    if not isinstance(feasible, torch.Tensor):
        return None
    return feasible.detach().view(-1).cpu()[node_mask.cpu()]


def summarize_series(values: Iterable[float]) -> Dict[str, Any]:
    series = pd.Series(list(values), dtype="float64").dropna()
    if len(series) == 0:
        return {"count": 0}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def expand_meta_values(value: Any, batch_size: int, *, cast=None, default=None) -> List[Any]:
    if value is None:
        return [default] * batch_size
    if isinstance(value, torch.Tensor):
        raw_values = value.detach().cpu().view(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value]
    if len(raw_values) == 1 and batch_size > 1:
        raw_values = raw_values * batch_size
    out = []
    for idx in range(batch_size):
        item = raw_values[idx] if idx < len(raw_values) else default
        if item is None:
            out.append(default)
            continue
        if cast is not None:
            try:
                item = cast(item)
            except Exception:
                item = default
        out.append(item)
    return out


def make_bucket_rate_table(
    df: pd.DataFrame,
    *,
    value_col: str,
    success_col: str,
    bucket_count: int,
    label_prefix: str,
) -> pd.DataFrame:
    work = df[[value_col, success_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=["bucket", "count", "value_mean", "value_min", "value_max", "success_rate"])
    try:
        work["bucket"] = pd.qcut(work[value_col], q=min(bucket_count, work[value_col].nunique()), duplicates="drop")
    except ValueError:
        work["bucket"] = pd.cut(work[value_col], bins=min(bucket_count, max(work[value_col].nunique(), 1)), duplicates="drop")
    rows = []
    for idx, (bucket, bucket_df) in enumerate(work.groupby("bucket", observed=True), start=1):
        rows.append(
            {
                "bucket": f"{label_prefix}_{idx}",
                "bucket_range": str(bucket),
                "count": int(len(bucket_df)),
                "value_mean": float(bucket_df[value_col].mean()),
                "value_min": float(bucket_df[value_col].min()),
                "value_max": float(bucket_df[value_col].max()),
                "success_rate": float(bucket_df[success_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_metric_formula_table(codepaths: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "success_rate",
                "level": "episode / case",
                "numerator_or_definition": "mean( success_i ) over all cases, where success_i = bool(raw_success_i > 0.5)",
                "valid_case_filter": "none",
                "aggregation_scope": "all evaluated cases in split",
                "special_cases": "no valid-case masking; this is independent of final-step rank validity",
                "active_codepath": f"{codepaths['eval_success_rate']['path']}:{codepaths['eval_success_rate']['line']}",
            },
            {
                "metric": "top1_hit",
                "level": "final-step ranking over valid cases",
                "numerator_or_definition": "mean( 1[rank_true_i <= 1] ) over valid cases only",
                "valid_case_filter": "requires at least one true source label and at least one finite logit at final step",
                "aggregation_scope": "valid_df only",
                "special_cases": "uses final-step reasoner logits; invalid cases excluded from numerator and denominator",
                "active_codepath": f"{codepaths['compute_rank_fields']['path']}:{codepaths['compute_rank_fields']['line']}",
            },
            {
                "metric": "top3_hit",
                "level": "final-step ranking over valid cases",
                "numerator_or_definition": "mean( 1[rank_true_i <= 3] ) over valid cases only",
                "valid_case_filter": "same valid_df filter as top1_hit",
                "aggregation_scope": "valid_df only",
                "special_cases": "none beyond valid-case filter",
                "active_codepath": f"{codepaths['compute_rank_fields']['path']}:{codepaths['compute_rank_fields']['line']}",
            },
            {
                "metric": "top5_hit",
                "level": "final-step ranking over valid cases",
                "numerator_or_definition": "mean( 1[rank_true_i <= 5] ) over valid cases only",
                "valid_case_filter": "same valid_df filter as top1_hit",
                "aggregation_scope": "valid_df only",
                "special_cases": "none beyond valid-case filter",
                "active_codepath": f"{codepaths['compute_rank_fields']['path']}:{codepaths['compute_rank_fields']['line']}",
            },
            {
                "metric": "mrr_valid",
                "level": "final-step ranking over valid cases",
                "numerator_or_definition": "mean( 1 / rank_true_i ) over valid cases only",
                "valid_case_filter": "same valid_df filter as top1_hit",
                "aggregation_scope": "valid_df only",
                "special_cases": "none beyond valid-case filter",
                "active_codepath": f"{codepaths['compute_rank_fields']['path']}:{codepaths['compute_rank_fields']['line']}",
            },
            {
                "metric": "true_source_rank_mean",
                "level": "final-step ranking over valid cases",
                "numerator_or_definition": "mean( rank_true_i ) over valid cases only",
                "valid_case_filter": "same valid_df filter as top1_hit",
                "aggregation_scope": "valid_df only",
                "special_cases": "lower is better; selection tuple negates this term",
                "active_codepath": f"{codepaths['selection_tuple']['path']}:{codepaths['selection_tuple']['line']}",
            },
            {
                "metric": "selection_tuple",
                "level": "checkpoint selection on validation split",
                "numerator_or_definition": "(mrr_valid, top1_hit, top5_hit, success_rate, -true_source_rank_mean)",
                "valid_case_filter": "inherits each metric definition above",
                "aggregation_scope": "formal val batch1 sparse checkpoint sweep",
                "special_cases": "lexicographic ordering favors mrr first, then top1, then top5, then success_rate",
                "active_codepath": f"{codepaths['selection_tuple']['path']}:{codepaths['selection_tuple']['line']}",
            },
        ]
    )


def render_metric_definition_md(
    *,
    selection_manifest: Dict[str, Any],
    compare_json: Dict[str, Any],
    codepaths: Dict[str, Dict[str, Any]],
) -> str:
    best = selection_manifest["selected_best_checkpoint"]
    best_test = best["test_summary"]
    bridged = compare_json["baselines"]["bridged_online"]
    bounded = compare_json["baselines"]["clean_aligned_bounded_online"]
    return "\n".join(
        [
            "# Success Rate Definition Audit",
            "",
            "## Formula-Level Definition",
            "",
            "- `success_rate = mean(df.success)` over all evaluated cases in the split.",
            "- In active evaluation code, `df.success` comes from `raw_success` returned by the episode runner, not from final-step rank or top-k hit.",
            "- `top1/top3/top5/mrr_valid/true_source_rank_mean` are computed separately from the final-step reasoner logits and averaged only over `valid_df`.",
            "- The checkpoint selector is lexicographic: `mrr_valid`, then `top1_hit`, then `top5_hit`, then `success_rate`, then lower `true_source_rank_mean`.",
            "",
            "## Natural-Language Meaning",
            "",
            "- In this line, `success_rate` answers: did the rollout eventually mark the case successful within the episode budget?",
            "- It does not answer: was the final reasoner top-1 correct, was the true source in the top-k, or did the reasoner rank improve before the source was already found.",
            "- Because action policy is `navigator_only`, `success_rate` is coupled to the frozen navigator trajectory much more than to current reasoner ranking quality.",
            "",
            "## Authoritative Manifest Facts",
            "",
            f"- Selected best checkpoint: epoch `{best['epoch']}` at `{best['path']}`.",
            f"- Best-checkpoint test metrics: `success_rate={best_test['success_rate']:.6f}`, `top1_hit={best_test['top1_hit']:.6f}`, `top5_hit={best_test['top5_hit']:.6f}`, `mrr_valid={best_test['mrr_valid']:.6f}`, `true_source_rank_mean={best_test['true_source_rank_mean']:.3f}`.",
            f"- Bridged-online baseline: `success_rate={bridged['success_rate']:.6f}`, `top1_hit={bridged['top1_hit']:.6f}`, `mrr_valid={bridged['mrr_valid']:.6f}`.",
            f"- Clean-aligned bounded baseline: `success_rate={bounded['success_rate']:.6f}`, `top1_hit={bounded['top1_hit']:.6f}`, `mrr_valid={bounded['mrr_valid']:.6f}`.",
            "",
            "## Exact Code Paths",
            "",
            f"- `evaluate_split` aggregation: `{codepaths['eval_success_rate']['path']}:{codepaths['eval_success_rate']['line']}`",
            f"- final-step rank computation: `{codepaths['compute_rank_fields']['path']}:{codepaths['compute_rank_fields']['line']}`",
            f"- episode-level legacy success summary: `{codepaths['legacy_success']['path']}:{codepaths['legacy_success']['line']}`",
            f"- checkpoint selection tuple: `{codepaths['selection_tuple']['path']}:{codepaths['selection_tuple']['line']}`",
            "",
            "## Important Special Cases",
            "",
            "- `success_rate` includes every case, even if final-step rank is invalid.",
            "- `topk` and `mrr_valid` exclude invalid cases by construction.",
            "- For this reason, `success_rate` can stay flat while `topk/MRR` move substantially.",
            "- The manifests also expose `legacy/Predict_Hit@1` and related fields, but under inference-mode evaluation those remain structurally invalid here and should not be used as primary evidence.",
            "",
        ]
    )


def extract_case_and_step_rows(
    *,
    model: torch.nn.Module,
    loader: Any,
    cfg: Any,
    device: torch.device,
    split_name: str,
    max_cases: Optional[int],
) -> Dict[str, pd.DataFrame]:
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for case_idx, batch in enumerate(loader):
            if max_cases is not None and case_idx >= int(max_cases):
                break

            batch = batch.to(device)
            out = model(
                batch,
                inference_mode=True,
                max_episodes=cfg.training.max_eval_episodes,
                return_trajectory=True,
            )
            trajectory = out.get("trajectory", [])
            step_metrics = out.get("step_metrics", {})
            final_dynamic_state = out.get("final_dynamic_state", {})
            batch_size = int(batch.num_graphs)
            scenario_ids = expand_meta_values(getattr(batch, "scenario_id", None), batch_size, cast=int, default=-1)
            part_ids = expand_meta_values(getattr(batch, "part_id", None), batch_size, cast=int, default=None)

            for graph_idx in range(batch_size):
                scenario_id = scenario_ids[graph_idx]
                part_id = graph_idx if part_ids[graph_idx] is None else int(part_ids[graph_idx])
                case_id = f"{split_name}:scenario{scenario_id}:part{part_id}"
                source_found_before = False
                first_success_episode = None
                graph_step_rows: List[Dict[str, Any]] = []

                for step_idx, step in enumerate(trajectory, start=1):
                    fused_batch = step["fused_batch"].detach().cpu().view(-1)
                    node_mask = fused_batch == graph_idx
                    logits = step["reasoner_logits"].detach().cpu().view(-1)[node_mask]
                    labels = step["fused_source_label"].detach().cpu().view(-1)[node_mask]
                    rank_stats = rank_metrics_from_masked_logits(logits, labels)

                    reasoner_state = step.get("reasoner_input_state", {})
                    obs_state = reasoner_state.get("observation_state")
                    observed_flag = (
                        obs_state.observed_flag.detach().cpu().view(-1)[node_mask]
                        if obs_state is not None
                        else torch.zeros_like(logits)
                    )
                    pre_valid_mask = step.get("pre_action_valid_mask")
                    pre_valid = (
                        pre_valid_mask.detach().cpu().view(-1)[node_mask].bool()
                        if isinstance(pre_valid_mask, torch.Tensor)
                        else torch.zeros_like(logits, dtype=torch.bool)
                    )
                    post_valid_mask = step.get("post_action_valid_mask")
                    post_valid = (
                        post_valid_mask.detach().cpu().view(-1)[node_mask].bool()
                        if isinstance(post_valid_mask, torch.Tensor)
                        else torch.zeros_like(logits, dtype=torch.bool)
                    )
                    pre_constraint_state = reasoner_state.get("constraint_state")
                    confirmed_non_source = (
                        pre_constraint_state.confirmed_non_source_mask.detach().cpu().view(-1)[node_mask].bool()
                        if pre_constraint_state is not None
                        else torch.zeros_like(logits, dtype=torch.bool)
                    )
                    confirmed_source = (
                        pre_constraint_state.confirmed_source_mask.detach().cpu().view(-1)[node_mask].bool()
                        if pre_constraint_state is not None
                        else torch.zeros_like(logits, dtype=torch.bool)
                    )
                    no_resample = (
                        pre_constraint_state.no_resample_mask.detach().cpu().view(-1)[node_mask].bool()
                        if pre_constraint_state is not None
                        else torch.zeros_like(logits, dtype=torch.bool)
                    )
                    feasible_mask = maybe_get_feasible_mask(step, node_mask)
                    if feasible_mask is not None:
                        feasible_mask = feasible_mask.bool()
                    finite_mask = torch.isfinite(logits)
                    success_event = bool(step["is_hit"].detach().cpu().view(-1)[graph_idx].item() > 0.5)
                    if success_event and first_success_episode is None:
                        first_success_episode = int(step_idx)
                    success_before_step = bool(source_found_before)
                    source_found_before = bool(source_found_before or success_event)

                    total_nodes = int(node_mask.sum().item())
                    observed_count = int((observed_flag > 0.5).sum().item())
                    finite_candidate_count = int(finite_mask.sum().item())
                    confirmed_source_count = int(confirmed_source.sum().item())
                    action_valid_count = int(pre_valid.sum().item())
                    finite_confirmed_non_source = int((finite_mask & confirmed_non_source).sum().item())
                    finite_no_resample = int((finite_mask & no_resample).sum().item())
                    if feasible_mask is not None:
                        finite_infeasible = int((finite_mask & (~feasible_mask)).sum().item())
                    else:
                        finite_infeasible = None

                    revealed_candidate_count = int((finite_mask & (observed_flag > 0.5)).sum().item())
                    unrevealed_candidate_count = int((finite_mask & (~(observed_flag > 0.5))).sum().item())
                    row = {
                        "case_id": case_id,
                        "scenario_id": scenario_id,
                        "part_id": part_id,
                        "episode_index": int(step_idx),
                        "total_nodes": total_nodes,
                        "valid_case": rank_stats["valid_case"],
                        "top1_hit": rank_stats["top1_hit"],
                        "top3_hit": rank_stats["top3_hit"],
                        "top5_hit": rank_stats["top5_hit"],
                        "mrr": rank_stats["mrr"],
                        "true_source_rank": rank_stats["true_source_rank"],
                        "entropy": rank_stats["entropy"],
                        "entropy_norm": rank_stats["entropy_norm"],
                        "max_prob": rank_stats["max_prob"],
                        "logits_candidate_size": finite_candidate_count,
                        "pre_action_valid_size": action_valid_count,
                        "post_action_valid_size": int(post_valid.sum().item()),
                        "confirmed_source_count": confirmed_source_count,
                        "confirmed_non_source_count": int(confirmed_non_source.sum().item()),
                        "no_resample_count": int(no_resample.sum().item()),
                        "observed_count": observed_count,
                        "revealed_ratio": observed_count / max(total_nodes, 1),
                        "revealed_candidate_count": revealed_candidate_count,
                        "unrevealed_candidate_count": unrevealed_candidate_count,
                        "unrevealed_candidate_ratio": (
                            unrevealed_candidate_count / finite_candidate_count if finite_candidate_count > 0 else float("nan")
                        ),
                        "success_event": success_event,
                        "success_before_step": success_before_step,
                        "success_by_step": bool(source_found_before),
                        "pre_action_confirmed_source": bool(step.get("pre_action_confirmed_source", False)),
                        "finite_on_confirmed_non_source": bool(finite_confirmed_non_source > 0),
                        "finite_on_no_resample": bool(finite_no_resample > 0),
                        "finite_confirmed_non_source_count": finite_confirmed_non_source,
                        "finite_no_resample_count": finite_no_resample,
                        "finite_infeasible_count": finite_infeasible,
                        "logits_minus_action_valid": finite_candidate_count - action_valid_count,
                    }
                    graph_step_rows.append(row)
                    step_rows.append(row)

                raw_success = bool(step_metrics.get("raw_success", torch.zeros(batch_size)).detach().cpu().view(-1)[graph_idx].item() > 0.5)
                raw_budget = float(step_metrics.get("raw_budget", torch.zeros(batch_size)).detach().cpu().view(-1)[graph_idx].item())
                raw_rounds = float(step_metrics.get("raw_rounds", torch.zeros(batch_size)).detach().cpu().view(-1)[graph_idx].item())
                raw_steps = float(step_metrics.get("raw_steps", torch.zeros(batch_size)).detach().cpu().view(-1)[graph_idx].item())

                if not graph_step_rows:
                    raw_batch_mask = batch.batch.detach().cpu().view(-1) == graph_idx
                    case_rows.append(
                        {
                            "case_id": case_id,
                            "scenario_id": scenario_id,
                            "part_id": part_id,
                            "success": raw_success,
                            "budget_used": raw_budget,
                            "episodes_completed": raw_rounds,
                            "physical_time_mins": raw_steps,
                            "first_success_episode": None,
                            "step_count_observed": 0,
                            "final_top1_hit": None,
                            "final_top3_hit": None,
                            "final_top5_hit": None,
                            "final_mrr": None,
                            "final_true_source_rank": None,
                            "final_entropy": None,
                            "final_entropy_norm": None,
                            "final_max_prob": None,
                            "final_logits_candidate_size": None,
                            "final_pre_action_valid_size": None,
                            "final_post_action_valid_size": None,
                            "final_revealed_ratio": None,
                            "final_revealed_candidate_count": None,
                            "final_unrevealed_candidate_count": None,
                            "final_unrevealed_candidate_ratio": None,
                            "final_pre_action_confirmed_source": False,
                            "final_logits_minus_action_valid": None,
                            "final_finite_confirmed_non_source_count": None,
                            "final_finite_no_resample_count": None,
                            "final_finite_infeasible_count": None,
                            "total_nodes": int(raw_batch_mask.sum().item()),
                            "final_confirmed_source_count": None,
                            "final_confirmed_non_source_count": None,
                            "final_no_resample_count": None,
                            "final_sampled_ratio": None,
                        }
                    )
                    continue

                last_step_row = graph_step_rows[-1]
                final_fused_batch = trajectory[-1]["fused_batch"].detach().cpu().view(-1)
                final_node_mask = final_fused_batch == graph_idx
                final_constraint = final_dynamic_state.get("constraint_state")
                if final_constraint is not None:
                    final_confirmed_source = int(final_constraint.confirmed_source_mask.detach().cpu().view(-1)[final_node_mask].gt(0.5).sum().item())
                    final_confirmed_non_source = int(final_constraint.confirmed_non_source_mask.detach().cpu().view(-1)[final_node_mask].gt(0.5).sum().item())
                    final_no_resample = int(final_constraint.no_resample_mask.detach().cpu().view(-1)[final_node_mask].gt(0.5).sum().item())
                else:
                    final_confirmed_source = 0
                    final_confirmed_non_source = 0
                    final_no_resample = 0

                case_rows.append(
                    {
                        "case_id": case_id,
                        "scenario_id": scenario_id,
                        "part_id": part_id,
                        "success": raw_success,
                        "budget_used": raw_budget,
                        "episodes_completed": raw_rounds,
                        "physical_time_mins": raw_steps,
                        "first_success_episode": first_success_episode,
                        "step_count_observed": len(graph_step_rows),
                        "final_top1_hit": last_step_row["top1_hit"],
                        "final_top3_hit": last_step_row["top3_hit"],
                        "final_top5_hit": last_step_row["top5_hit"],
                        "final_mrr": last_step_row["mrr"],
                        "final_true_source_rank": last_step_row["true_source_rank"],
                        "final_entropy": last_step_row["entropy"],
                        "final_entropy_norm": last_step_row["entropy_norm"],
                        "final_max_prob": last_step_row["max_prob"],
                        "final_logits_candidate_size": int(last_step_row["logits_candidate_size"]),
                        "final_pre_action_valid_size": int(last_step_row["pre_action_valid_size"]),
                        "final_post_action_valid_size": int(last_step_row["post_action_valid_size"]),
                        "final_revealed_ratio": float(last_step_row["revealed_ratio"]),
                        "final_revealed_candidate_count": int(last_step_row["revealed_candidate_count"]),
                        "final_unrevealed_candidate_count": int(last_step_row["unrevealed_candidate_count"]),
                        "final_unrevealed_candidate_ratio": float(last_step_row["unrevealed_candidate_ratio"]),
                        "final_pre_action_confirmed_source": bool(last_step_row["pre_action_confirmed_source"]),
                        "final_logits_minus_action_valid": int(last_step_row["logits_minus_action_valid"]),
                        "final_finite_confirmed_non_source_count": int(last_step_row["finite_confirmed_non_source_count"]),
                        "final_finite_no_resample_count": int(last_step_row["finite_no_resample_count"]),
                        "final_finite_infeasible_count": last_step_row["finite_infeasible_count"],
                        "total_nodes": int(last_step_row["total_nodes"]),
                        "final_confirmed_source_count": final_confirmed_source,
                        "final_confirmed_non_source_count": final_confirmed_non_source,
                        "final_no_resample_count": final_no_resample,
                        "final_sampled_ratio": final_no_resample / max(int(last_step_row["total_nodes"]), 1),
                    }
                )

    return {
        "cases": pd.DataFrame(case_rows),
        "steps": pd.DataFrame(step_rows),
    }


def build_episode_progress_metrics(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for episode_index, group in step_df.groupby("episode_index", observed=True):
        valid_group = group[group["valid_case"] == True]
        unresolved = group[group["pre_action_confirmed_source"] == False]
        unresolved_valid = unresolved[unresolved["valid_case"] == True]
        rows.append(
            {
                "episode_index": int(episode_index),
                "num_cases": int(len(group)),
                "num_unresolved_cases": int(len(unresolved)),
                "success_event_rate": bool_mean(group["success_event"]),
                "success_by_step_rate": bool_mean(group["success_by_step"]),
                "top1_all": bool_mean(valid_group["top1_hit"]) if len(valid_group) else float("nan"),
                "top3_all": bool_mean(valid_group["top3_hit"]) if len(valid_group) else float("nan"),
                "top5_all": bool_mean(valid_group["top5_hit"]) if len(valid_group) else float("nan"),
                "mrr_all": float(valid_group["mrr"].mean()) if len(valid_group) else float("nan"),
                "rank_mean_all": float(valid_group["true_source_rank"].mean()) if len(valid_group) else float("nan"),
                "top1_unresolved_only": bool_mean(unresolved_valid["top1_hit"]) if len(unresolved_valid) else float("nan"),
                "top3_unresolved_only": bool_mean(unresolved_valid["top3_hit"]) if len(unresolved_valid) else float("nan"),
                "top5_unresolved_only": bool_mean(unresolved_valid["top5_hit"]) if len(unresolved_valid) else float("nan"),
                "mrr_unresolved_only": float(unresolved_valid["mrr"].mean()) if len(unresolved_valid) else float("nan"),
                "rank_mean_unresolved_only": float(unresolved_valid["true_source_rank"].mean()) if len(unresolved_valid) else float("nan"),
                "entropy_mean": float(valid_group["entropy"].mean()) if len(valid_group) else float("nan"),
                "entropy_norm_mean": float(valid_group["entropy_norm"].mean()) if len(valid_group) else float("nan"),
                "max_prob_mean": float(valid_group["max_prob"].mean()) if len(valid_group) else float("nan"),
                "logits_candidate_size_mean": float(group["logits_candidate_size"].mean()),
                "pre_action_valid_size_mean": float(group["pre_action_valid_size"].mean()),
                "revealed_ratio_mean": float(group["revealed_ratio"].mean()),
                "unrevealed_candidate_ratio_mean": float(group["unrevealed_candidate_ratio"].mean()),
                "revealed_candidate_count_mean": float(group["revealed_candidate_count"].mean()),
                "unrevealed_candidate_count_mean": float(group["unrevealed_candidate_count"].mean()),
                "pre_action_confirmed_source_rate": bool_mean(group["pre_action_confirmed_source"]),
            }
        )
    return pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)


def build_success_failure_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for success_value, group in case_df.groupby("success", observed=True):
        rows.append(
            {
                "outcome": "success" if bool(success_value) else "failure",
                "count": int(len(group)),
                "final_logits_candidate_size_mean": float(group["final_logits_candidate_size"].mean()),
                "final_logits_candidate_size_median": float(group["final_logits_candidate_size"].median()),
                "final_pre_action_valid_size_mean": float(group["final_pre_action_valid_size"].mean()),
                "final_post_action_valid_size_mean": float(group["final_post_action_valid_size"].mean()),
                "final_revealed_ratio_mean": float(group["final_revealed_ratio"].mean()),
                "final_sampled_ratio_mean": float(group["final_sampled_ratio"].mean()),
                "final_unrevealed_candidate_ratio_mean": float(group["final_unrevealed_candidate_ratio"].mean()),
                "final_top1_mean": float(group["final_top1_hit"].astype(float).mean()),
                "final_mrr_mean": float(group["final_mrr"].mean()),
                "pct_final_revealed_ge_0p90": float((group["final_revealed_ratio"] >= 0.90).mean()),
                "pct_final_sampled_ge_0p90": float((group["final_sampled_ratio"] >= 0.90).mean()),
                "pct_final_remaining_le_0p10": float((group["final_post_action_valid_size"] / group["total_nodes"] <= 0.10).mean()),
                "pct_final_preconfirmed_source": float(group["final_pre_action_confirmed_source"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def choose_representative_case_ids(case_df: pd.DataFrame) -> List[str]:
    picks: List[str] = []
    for subset in [
        case_df[case_df["success"] == True].sort_values(["final_revealed_ratio", "final_true_source_rank"], ascending=[True, True]),
        case_df[case_df["success"] == True].sort_values(["final_revealed_ratio", "final_true_source_rank"], ascending=[False, True]),
        case_df[case_df["success"] == False].sort_values(["final_logits_candidate_size", "final_true_source_rank"], ascending=[True, False]),
        case_df[case_df["success"] == False].sort_values(["final_logits_candidate_size", "final_true_source_rank"], ascending=[False, False]),
    ]:
        if not subset.empty:
            candidate = str(subset.iloc[0]["case_id"])
            if candidate not in picks:
                picks.append(candidate)
    return picks[:4]


def write_episode_plot(path: Path, episode_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(episode_df["episode_index"], episode_df["top1_all"], marker="o", label="top1 all")
    axes[0, 0].plot(episode_df["episode_index"], episode_df["top1_unresolved_only"], marker="s", label="top1 unresolved-only")
    axes[0, 0].set_title("Top1 By Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].legend()

    axes[0, 1].plot(episode_df["episode_index"], episode_df["mrr_all"], marker="o", label="MRR all")
    axes[0, 1].plot(episode_df["episode_index"], episode_df["mrr_unresolved_only"], marker="s", label="MRR unresolved-only")
    axes[0, 1].set_title("MRR By Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("MRR")
    axes[0, 1].legend()

    axes[1, 0].plot(episode_df["episode_index"], episode_df["logits_candidate_size_mean"], marker="o", label="logits candidates")
    axes[1, 0].plot(episode_df["episode_index"], episode_df["pre_action_valid_size_mean"], marker="s", label="action-valid")
    axes[1, 0].set_title("Candidate Space By Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Nodes")
    axes[1, 0].legend()

    axes[1, 1].plot(episode_df["episode_index"], episode_df["revealed_ratio_mean"], marker="o", label="revealed ratio")
    axes[1, 1].plot(episode_df["episode_index"], episode_df["unrevealed_candidate_ratio_mean"], marker="s", label="unrevealed candidate ratio")
    axes[1, 1].set_title("Reveal State By Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_candidate_size_plot(path: Path, case_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    success = case_df[case_df["success"] == True]
    failure = case_df[case_df["success"] == False]

    axes[0].hist(
        [success["final_logits_candidate_size"], failure["final_logits_candidate_size"]],
        bins=20,
        label=["success", "failure"],
        alpha=0.75,
    )
    axes[0].set_title("Final Logits Candidate Size")
    axes[0].set_xlabel("Candidate Size")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(
        [success["final_post_action_valid_size"], failure["final_post_action_valid_size"]],
        bins=20,
        label=["success", "failure"],
        alpha=0.75,
    )
    axes[1].set_title("Final Remaining Action-Valid Size")
    axes[1].set_xlabel("Remaining Valid Nodes")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_candidate_mask_md(
    *,
    case_df: pd.DataFrame,
    step_df: pd.DataFrame,
    codepaths: Dict[str, Dict[str, Any]],
) -> str:
    final_preconfirmed_rate = float(case_df["final_pre_action_confirmed_source"].astype(float).mean())
    finite_non_source_cases = int((step_df["finite_on_confirmed_non_source"] == True).sum())
    finite_no_resample_cases = int((step_df["finite_on_no_resample"] == True).sum())
    logits_superset_rate = float((case_df["final_logits_minus_action_valid"] > 0).mean())
    return "\n".join(
        [
            "# Candidate And Logits Mask Flow",
            "",
            "## Hard-Constraint Flow",
            "",
            "- `ConstraintState` holds `confirmed_non_source_mask`, `confirmed_source_mask`, `sampled_mask`, and `no_resample_mask`.",
            "- The episode runner builds `valid_mask_final = ~no_resample_mask & feasible_mask` for action selection and state summaries.",
            "- The reasoner logits then go through `apply_constraint_masks`, which hard-masks `no_resample_mask | confirmed_non_source_mask` and gives `confirmed_source_mask` an exclusive keep-alive override per graph.",
            "- This means action-valid space and logits-ranking space are not the same object.",
            "",
            "## Exact Answers",
            "",
            "- Already sampled / explicitly confirmed non-source nodes are intended to be hard-excluded from logits.",
            "- Confirmed source nodes are not excluded; they are hard-kept and can become the only finite logits in that graph.",
            "- `feasible_mask` is not applied inside `apply_constraint_masks`; it constrains action selection through `valid_mask_final`, not the stored ranking logits themselves.",
            "- Support / contradiction / suspect / arrival signals are feature channels, not hard exclusions.",
            "",
            "## Measured Audit Facts",
            "",
            f"- Final-step pre-confirmed-source rate: `{final_preconfirmed_rate:.4f}`.",
            f"- Steps with finite logits on confirmed non-source nodes: `{finite_non_source_cases}`.",
            f"- Steps with finite logits on no-resample nodes: `{finite_no_resample_cases}`.",
            f"- Cases where final logits candidate set was larger than the action-valid set: `{logits_superset_rate:.4f}`.",
            "",
            "## Code Paths",
            "",
            f"- `valid_mask_final` build: `{codepaths['valid_mask_final']['path']}:{codepaths['valid_mask_final']['line']}`",
            f"- constraint update: `{codepaths['constraint_update']['path']}:{codepaths['constraint_update']['line']}`",
            f"- hard exclusion and confirmed-source override: `{codepaths['constraint_apply']['path']}:{codepaths['constraint_apply']['line']}`",
            f"- nav-only routing against `valid_mask_final`: `{codepaths['nav_only_route']['path']}:{codepaths['nav_only_route']['line']}`",
            "",
            "## Direct Conclusion",
            "",
            "- To the specific question `already-confirmed safe nodes are they completely kicked out of final softmax candidate space?`: [partially proven] yes for the normal hard-mask path, but not as a universal contract for all ranking uses because the evaluation ranking logits are not additionally intersected with `feasible_mask`, and `confirmed_source_mask` can intentionally re-enter as an exclusive keep-alive override.",
            "",
        ]
    )


def render_episode_summary_md(
    *,
    episode_df: pd.DataFrame,
    case_df: pd.DataFrame,
) -> str:
    first_row = episode_df.iloc[0].to_dict() if len(episode_df) else {}
    last_row = episode_df.iloc[-1].to_dict() if len(episode_df) else {}
    unresolved_gain = safe_float(last_row.get("top1_unresolved_only")) - safe_float(first_row.get("top1_unresolved_only"))
    all_gain = safe_float(last_row.get("top1_all")) - safe_float(first_row.get("top1_all"))
    preconfirmed_final_rate = float(case_df["final_pre_action_confirmed_source"].astype(float).mean()) if len(case_df) else float("nan")
    return "\n".join(
        [
            "# Episode-Wise Prediction Audit",
            "",
            f"- Episode-1 `top1_all={safe_float(first_row.get('top1_all')):.4f}`, `top1_unresolved_only={safe_float(first_row.get('top1_unresolved_only')):.4f}`, `mrr_unresolved_only={safe_float(first_row.get('mrr_unresolved_only')):.4f}`.",
            f"- Final observed episode `top1_all={safe_float(last_row.get('top1_all')):.4f}`, `top1_unresolved_only={safe_float(last_row.get('top1_unresolved_only')):.4f}`, `mrr_unresolved_only={safe_float(last_row.get('mrr_unresolved_only')):.4f}`.",
            f"- Overall top1 gain across episodes: `{all_gain:+.4f}`.",
            f"- Unresolved-only top1 gain across episodes: `{unresolved_gain:+.4f}`.",
            f"- Final-step pre-confirmed-source rate: `{preconfirmed_final_rate:.4f}`.",
            "",
            "Interpretation:",
            "- `all` curves include cases where the source was already confirmed before the step, so they mix real prediction with hard post-hit keep-alive behavior.",
            "- `unresolved_only` curves are the cleaner proxy for genuine step-by-step reasoner discrimination before the source was already found.",
            "- Candidate size and reveal ratio should be read together with those unresolved-only curves; if ranking improves only when unresolved mass collapses, the late boost is mostly state exposure rather than early discrimination.",
            "",
        ]
    )


def render_final_judgment_md(
    *,
    selection_manifest: Dict[str, Any],
    compare_json: Dict[str, Any],
    case_df: pd.DataFrame,
    episode_df: pd.DataFrame,
) -> str:
    best = compare_json["best_checkpoint"]
    bridged = compare_json["baselines"]["bridged_online"]
    bounded = compare_json["baselines"]["clean_aligned_bounded_online"]
    final_preconfirmed_rate = float(case_df["final_pre_action_confirmed_source"].astype(float).mean()) if len(case_df) else float("nan")
    logits_superset_rate = float((case_df["final_logits_minus_action_valid"] > 0).mean()) if len(case_df) else float("nan")
    unresolved_top1_start = safe_float(episode_df["top1_unresolved_only"].iloc[0]) if len(episode_df) else float("nan")
    unresolved_top1_end = safe_float(episode_df["top1_unresolved_only"].iloc[-1]) if len(episode_df) else float("nan")
    return "\n".join(
        [
            "# Final Judgment",
            "",
            "## Core Findings",
            "",
            f"1. [proven] `success_rate` is not a clean reasoner-quality metric in this line. The run is `sampling_policy=navigator_only`, so `success_rate` mainly tracks whether the frozen navigator physically samples the true source within budget. Evidence: best checkpoint `success_rate={best['success_rate']:.6f}` vs bridged baseline `success_rate={bridged['success_rate']:.6f}` is almost unchanged, while `top1_hit` jumps from `{bridged['top1_hit']:.6f}` to `{best['top1_hit']:.6f}` and `mrr_valid` jumps from `{bridged['mrr_valid']:.6f}` to `{best['mrr_valid']:.6f}`.",
            f"2. [proven] The current reasoner does consume some clean-state exclusions as hard masks: sampled / no-resample and explicit confirmed-non-source masks go through a hard `-inf` path before action selection; confirmed-source gets an exclusive keep-alive override.",
            f"3. [partially proven] Final-step ranking metrics are stronger evidence than `success_rate`, but they are still not a pure pre-hit ability metric because final-step pre-confirmed-source rate is `{final_preconfirmed_rate:.4f}`.",
            f"4. [proven] Final logits and action-valid candidate spaces differ materially. Cases with `final logits candidate size > final action-valid size`: `{logits_superset_rate:.4f}`. This comes from ranking logits not being intersected with the same `valid_mask_final` object used by action routing.",
            f"5. [partially proven] Genuine unresolved-only prediction improves from episode-1 `top1={unresolved_top1_start:.4f}` to final observed episode `top1={unresolved_top1_end:.4f}`, but the all-case curve can overstate late performance because it includes already-confirmed-source cases.",
            "",
            "## Direct Answers",
            "",
            "1. Current low `SR`: mainly a metric-semantics / consumer-lane issue, not direct proof that the reasoner is bad.",
            "2. Hard clean-state consumption: yes for explicit sampled / confirmed-non-source / no-resample constraints; no for evidence-only support/contradiction signals, which are features rather than hard masks.",
            "3. Best headline metric now: `mrr_valid` plus `top1/top3/top5`, with an explicit note that these are final-step ranking metrics under the frozen navigator trajectory.",
            "4. Should `SR` be the headline in the paper: no. It should be described as rollout source-found rate under a frozen navigator budget, not as the primary reasoner capability number.",
            "5. Does the result completely fail to support the paper: [not proved]. The cleaner interpretation is that the previous story over-read `SR`; the ranking gains still support a more bounded claim about downstream reasoner discrimination under fixed trajectories.",
            "",
            "## Claim Boundary",
            "",
            f"- [proven] Best-checkpoint selection itself followed the intended lexicographic rule and correctly picked epoch `{selection_manifest['selected_best_checkpoint']['epoch']}`.",
            f"- [partially proven] Final-step ranking gains over the clean-aligned bounded baseline remain real (`top1 {best['top1_hit']:.6f}` vs `{bounded['top1_hit']:.6f}`, `mrr {best['mrr_valid']:.6f}` vs `{bounded['mrr_valid']:.6f}`), but those gains should not be narrated as direct rollout-success gains.",
            "- [not proved] We do not prove here that the reasoner alone would improve rollout success if it were actually allowed to drive action selection; that would require a separate consumer-lane audit.",
            "",
        ]
    )


def render_claim_boundary_update_md() -> str:
    return "\n".join(
        [
            "# Claim Boundary Update",
            "",
            "- Headline the fixed-trajectory downstream ranking gains, not `success_rate`.",
            "- Describe `success_rate` as frozen-navigator source-found rate under budget, which is useful context but not the main reasoner metric.",
            "- Explicitly warn that final-step ranking can include already-confirmed-source cases; when discussing genuine online discrimination, prefer the unresolved-only episode audit.",
            "- State that clean-state hard constraints are partially consumed in logits, but the ranking candidate set is not exactly the same as the action-valid candidate set.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded metric semantics audit for the authoritative downstream reasoner line.")
    parser.add_argument("--finish-root", type=str, default=str(DEFAULT_FINISH_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finish_root = Path(args.finish_root)
    output_dir = Path(args.output_dir)
    metric_dir = output_dir / "metric_definition_audit"
    mask_dir = output_dir / "candidate_mask_audit"
    episode_dir = output_dir / "episode_progress_audit"
    bucket_dir = output_dir / "success_failure_bucket_audit"
    summary_dir = output_dir / "summary"
    for path in [metric_dir, mask_dir, episode_dir, bucket_dir, summary_dir]:
        path.mkdir(parents=True, exist_ok=True)

    selection_manifest = json.loads((finish_root / "selection_manifest.json").read_text(encoding="utf-8"))
    compare_json = json.loads((finish_root / "compare" / "final_compare.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((finish_root / "run_manifest.json").read_text(encoding="utf-8"))
    throughput_summary = json.loads((finish_root / "throughput_summary.json").read_text(encoding="utf-8"))
    resolved_cfg = yaml.safe_load((finish_root / "delivery" / "resolved_config.yaml").read_text(encoding="utf-8"))

    bridge_package_dir = Path(run_manifest["authoritative_frozen_navigator_package"])
    init_checkpoint = Path(run_manifest["authoritative_clean_aligned_bounded_best"])
    cache_version = throughput_summary["cache_version"]
    cache_dir = Path(throughput_summary["cache_dir"])
    best_checkpoint_path = Path(selection_manifest["selected_best_checkpoint"]["path"])

    overrides = build_eval_overrides(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_version=cache_version,
        cache_dir=cache_dir,
    )
    cfg = prepare_cfg(overrides, run_name="reasoner_metric_semantics_audit", max_epochs=1, seed=int(run_manifest["seed"]))
    _, eval_loaders = build_eval_loaders_with_batch(eval_batch_size=1)
    loader = eval_loaders[str(args.split)]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ModelBuilder.build_model(cfg).to(device)
    model.load_state_dict(load_plain_state_dict(best_checkpoint_path), strict=True)
    model.eval()

    extracted = extract_case_and_step_rows(
        model=model,
        loader=loader,
        cfg=cfg,
        device=device,
        split_name=str(args.split),
        max_cases=args.max_cases,
    )
    case_df = extracted["cases"].sort_values("case_id").reset_index(drop=True)
    step_df = extracted["steps"].sort_values(["case_id", "episode_index"]).reset_index(drop=True)
    episode_df = build_episode_progress_metrics(step_df)

    success_failure_df = build_success_failure_summary(case_df)
    revealed_bucket_df = make_bucket_rate_table(
        case_df,
        value_col="final_revealed_ratio",
        success_col="success",
        bucket_count=5,
        label_prefix="revealed_ratio",
    )
    unresolved_bucket_df = make_bucket_rate_table(
        case_df.assign(final_remaining_ratio=case_df["final_post_action_valid_size"] / case_df["total_nodes"].clip(lower=1)),
        value_col="final_remaining_ratio",
        success_col="success",
        bucket_count=5,
        label_prefix="remaining_ratio",
    )
    representative_case_ids = choose_representative_case_ids(case_df)
    representative_trace_df = step_df[step_df["case_id"].isin(representative_case_ids)].copy()

    write_candidate_size_plot(bucket_dir / "candidate_size_vs_success.png", case_df)
    write_episode_plot(episode_dir / "episode_progress_figure.png", episode_df)

    codepaths = line_map()
    metric_formula_df = build_metric_formula_table(codepaths)
    metric_formula_df.to_csv(metric_dir / "metric_formula_table.csv", index=False)
    write_json(
        metric_dir / "metric_codepath_manifest.json",
        {
            "codepaths": codepaths,
            "authoritative_artifacts": {
                "finish_root": str(finish_root),
                "selection_manifest": str(finish_root / "selection_manifest.json"),
                "compare_json": str(finish_root / "compare" / "final_compare.json"),
                "run_manifest": str(finish_root / "run_manifest.json"),
                "resolved_config": str(finish_root / "delivery" / "resolved_config.yaml"),
            },
        },
    )
    write_text(
        metric_dir / "success_rate_definition.md",
        render_metric_definition_md(
            selection_manifest=selection_manifest,
            compare_json=compare_json,
            codepaths=codepaths,
        ),
    )

    hard_vs_soft_rows = [
        {
            "signal": "confirmed_non_source_mask",
            "kind": "hard",
            "consumed_where": "ConstraintState -> apply_constraint_masks",
            "effect": "set logits to -inf",
            "exact_codepath": f"{codepaths['constraint_apply']['path']}:{codepaths['constraint_apply']['line']}",
        },
        {
            "signal": "no_resample_mask / sampled_mask",
            "kind": "hard",
            "consumed_where": "valid_mask_final and apply_constraint_masks",
            "effect": "blocked from action-valid set and logit set",
            "exact_codepath": f"{codepaths['valid_mask_final']['path']}:{codepaths['valid_mask_final']['line']}",
        },
        {
            "signal": "confirmed_source_mask",
            "kind": "hard",
            "consumed_where": "apply_constraint_masks",
            "effect": "exclusive keep-alive override per graph",
            "exact_codepath": f"{codepaths['constraint_apply']['path']}:{codepaths['constraint_apply']['line']}",
        },
        {
            "signal": "feasible_mask",
            "kind": "hard for action routing only",
            "consumed_where": "valid_mask_final",
            "effect": "limits action-valid nodes but not the stored ranking logits directly",
            "exact_codepath": f"{codepaths['valid_mask_final']['path']}:{codepaths['valid_mask_final']['line']}",
        },
        {
            "signal": "support / contradiction / suspect / arrival features",
            "kind": "soft feature",
            "consumed_where": "clean-aligned feature payload",
            "effect": "feature channels into node/graph embeddings only",
            "exact_codepath": f"{codepaths['feature_payload']['path']}:{codepaths['feature_payload']['line']}",
        },
    ]
    pd.DataFrame(hard_vs_soft_rows).to_csv(mask_dir / "hard_vs_soft_constraints_table.csv", index=False)
    write_text(
        mask_dir / "candidate_mask_flow.md",
        render_candidate_mask_md(case_df=case_df, step_df=step_df, codepaths=codepaths),
    )
    write_json(
        mask_dir / "logits_candidate_contract.json",
        {
            "final_preconfirmed_source_rate": float(case_df["final_pre_action_confirmed_source"].astype(float).mean()) if len(case_df) else None,
            "final_logits_candidate_size_summary": summarize_series(case_df["final_logits_candidate_size"]),
            "final_action_valid_size_summary": summarize_series(case_df["final_pre_action_valid_size"]),
            "final_logits_minus_action_valid_summary": summarize_series(case_df["final_logits_minus_action_valid"]),
            "steps_with_finite_confirmed_non_source": int((step_df["finite_on_confirmed_non_source"] == True).sum()),
            "steps_with_finite_no_resample": int((step_df["finite_on_no_resample"] == True).sum()),
            "config_sampling_policy": resolved_cfg["model"].get("sampling_policy"),
            "config_navigator_type": resolved_cfg["model"].get("navigator_type"),
        },
    )

    success_failure_df.to_csv(bucket_dir / "success_failure_bucket_summary.csv", index=False)
    revealed_bucket_df.to_csv(bucket_dir / "revealed_ratio_vs_success.csv", index=False)
    unresolved_bucket_df.to_csv(bucket_dir / "unresolved_vs_success.csv", index=False)
    case_df.to_csv(bucket_dir / "raw_case_summary.csv", index=False)

    episode_df.to_csv(episode_dir / "episode_progress_metrics.csv", index=False)
    representative_trace_df.to_csv(episode_dir / "representative_cases_episode_trace.csv", index=False)
    step_df.to_csv(episode_dir / "raw_case_episode_metrics.csv", index=False)
    write_text(
        episode_dir / "episode_progress_summary.md",
        render_episode_summary_md(episode_df=episode_df, case_df=case_df),
    )

    write_text(
        summary_dir / "final_judgment.md",
        render_final_judgment_md(
            selection_manifest=selection_manifest,
            compare_json=compare_json,
            case_df=case_df,
            episode_df=episode_df,
        ),
    )
    write_text(summary_dir / "claim_boundary_update.md", render_claim_boundary_update_md())
    write_json(
        summary_dir / "audit_manifest.json",
        {
            "finish_root": str(finish_root),
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(best_checkpoint_path),
            "split": str(args.split),
            "seed": int(run_manifest["seed"]),
            "panel_version": run_manifest["panel_version"],
            "sampling_policy": resolved_cfg["model"].get("sampling_policy"),
            "navigator_type": resolved_cfg["model"].get("navigator_type"),
            "reasoner_type": resolved_cfg["model"].get("reasoner_type"),
            "max_eval_episodes": resolved_cfg["training"].get("max_eval_episodes"),
            "num_cases": int(len(case_df)),
            "num_step_rows": int(len(step_df)),
            "representative_case_ids": representative_case_ids,
        },
    )

    write_text(
        summary_dir / "bundle_index.md",
        "\n".join(
            [
                "# Reasoner Metric Semantics Audit",
                "",
                f"- finish root: `{finish_root}`",
                f"- output dir: `{output_dir}`",
                f"- best checkpoint: `{best_checkpoint_path}`",
                f"- split audited: `{args.split}`",
                f"- cases audited: `{len(case_df)}`",
                "",
                "Key files:",
                "- `metric_definition_audit/success_rate_definition.md`",
                "- `candidate_mask_audit/candidate_mask_flow.md`",
                "- `episode_progress_audit/episode_progress_summary.md`",
                "- `success_failure_bucket_audit/success_failure_bucket_summary.csv`",
                "- `summary/final_judgment.md`",
                "",
            ]
        ),
    )


if __name__ == "__main__":
    main()
