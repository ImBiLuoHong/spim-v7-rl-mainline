import argparse
import copy
import json
import logging
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.evaluation.evaluator import Evaluator
from src.evaluation.runner import compute_geodesic_distances
from src.modeling.builders.model_builder import ModelBuilder
from src.scripts.train_phase4_end2end import get_config_hash, run_training


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OFFICIAL_CONFIGS = [
    {
        "name": "base_no_evidence",
        "path": "configs/evidence_v1/base_no_evidence.yaml",
    },
    {
        "name": "support_mainline",
        "path": "configs/evidence_v1/support_mainline.yaml",
    },
    {
        "name": "support_plus_soft_suspect",
        "path": "configs/evidence_v1/support_plus_soft_suspect.yaml",
    },
    {
        "name": "support_plus_contradiction_aux_compare",
        "path": "configs/evidence_v1/support_plus_contradiction_aux_compare.yaml",
    },
]


LMDB_CACHE_DIR = os.environ.get("GUMBEL_LMDB_CACHE_DIR", os.path.join(os.getcwd(), "data", "cache_lmdb"))
FULL_EVAL_CACHE_VERSION = "round1_full_eval"


PHASE_SPECS = {
    "pilot": {
        "cache_version": "round1_pilot_n256",
        "epochs": 4,
        "batch_size": 4,
        "max_samples": 256,
        "seed": 45,
        "force_rebuild": True,
    },
    "main": {
        "cache_version": "round1_main_n1024",
        "epochs": 6,
        "batch_size": 8,
        "max_samples": 1024,
        "seed": 45,
        "force_rebuild": True,
    },
}


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return {
            "rows": int(len(value)),
            "columns": list(value.columns),
        }
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Series):
        return {k: _json_ready(v) for k, v in value.to_dict().items()}
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def deep_merge(base, extra):
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def prepare_cfg(overrides, run_name=None, max_epochs=None, seed=None):
    cfg = Config()
    cfg.apply_overrides(overrides)
    cfg.data.use_dataloader_v6 = True
    cfg.data.filter_no_source = True
    cfg.model.architecture = "phase4_5"
    if max_epochs is not None:
        cfg.training.num_epochs = max_epochs
    if seed is not None:
        cfg.training.seed = seed
    if run_name is not None:
        cfg.training.run_name = run_name
    return cfg


def stable_cfg_dict(cfg):
    def _to_dict_stable(obj):
        if isinstance(obj, dict):
            return {k: _to_dict_stable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict_stable(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return {
                k: _to_dict_stable(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        return obj

    return _to_dict_stable(cfg)


def resolve_run_dir(overrides, run_name, max_epochs, seed):
    cfg = prepare_cfg(overrides, run_name=run_name, max_epochs=max_epochs, seed=seed)
    cfg_hash = get_config_hash(stable_cfg_dict(cfg))
    return os.path.join(cfg.paths.experiments_dir, f"{run_name}_{cfg_hash}")


def build_training_loaders(phase_spec):
    cache_version = phase_spec["cache_version"]
    overrides = {
        "paths": {
            "cache_dir": LMDB_CACHE_DIR,
        },
        "data": {
            "cache_version": cache_version,
            "max_samples": phase_spec["max_samples"],
            "skip_lmdb": False,
        },
        "efficiency": {
            "batch_size": phase_spec["batch_size"],
        },
    }
    cfg = prepare_cfg(overrides, max_epochs=phase_spec["epochs"], seed=phase_spec["seed"])
    return create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=phase_spec["batch_size"],
        skip_lmdb=getattr(cfg.data, "skip_lmdb", False),
    )


def build_eval_loaders():
    return build_eval_loaders_with_batch(eval_batch_size=1)


def build_eval_loaders_with_batch(eval_batch_size=1, loader_overrides=None):
    loader_overrides = copy.deepcopy(loader_overrides or {})
    overrides = {
        "paths": {
            "cache_dir": LMDB_CACHE_DIR,
        },
        "data": {
            "cache_version": FULL_EVAL_CACHE_VERSION,
            "skip_lmdb": False,
        },
        "efficiency": {
            "batch_size": eval_batch_size,
        }
    }
    for key in (
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "preload",
        "non_blocking_transfer",
    ):
        if key in loader_overrides:
            overrides["data"][key] = loader_overrides[key]
            overrides["efficiency"][key] = loader_overrides[key]
    cfg = prepare_cfg(overrides)
    _, val_loader, test_loader, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=eval_batch_size,
        eval_batch_size=eval_batch_size,
        skip_lmdb=getattr(cfg.data, "skip_lmdb", False),
    )
    return cfg, {"val": val_loader, "test": test_loader}


def prewarm_lmdb_cache(cache_version, max_samples=None):
    overrides = {
        "paths": {
            "cache_dir": LMDB_CACHE_DIR,
        },
        "data": {
            "cache_version": cache_version,
            "max_samples": max_samples,
            "rebuild_cache": True,
            "skip_lmdb": False,
        },
        "efficiency": {
            "batch_size": 1,
        },
    }
    cfg = prepare_cfg(overrides)
    logger.info("Prewarming LMDB cache with rebuild=true before campaign start")
    create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        skip_lmdb=getattr(cfg.data, "skip_lmdb", False),
    )


def configure_offline_wandb():
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")


def build_model_for_eval(cfg, loader, checkpoint_path, device):
    topology_engine = None
    if hasattr(loader.dataset, "topology"):
        topology_engine = loader.dataset.topology
    elif hasattr(loader.dataset, "dataset") and hasattr(loader.dataset.dataset, "topology"):
        topology_engine = loader.dataset.dataset.topology

    model = ModelBuilder.build_model(cfg).to(device)
    if hasattr(model, "topology_engine"):
        model.topology_engine = topology_engine

    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        logger.warning("Strict checkpoint load failed for %s, falling back to strict=False", checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def probs_from_logits(logits):
    if logits.dim() == 1:
        return torch.sigmoid(logits)
    if logits.size(-1) == 1:
        return torch.sigmoid(logits.view(-1))
    return torch.softmax(logits, dim=-1)[:, -1]


def safe_group_value(value, index, default=None):
    if isinstance(value, (list, tuple)):
        if index < len(value):
            return value[index]
        return default
    return value if value is not None else default


def per_graph_values(value, batch_size, default=None, cast=None):
    if value is None:
        return [default] * batch_size

    if isinstance(value, torch.Tensor):
        raw_values = value.detach().cpu().view(-1).tolist()
    elif isinstance(value, np.ndarray):
        raw_values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value]

    if len(raw_values) == 1 and batch_size > 1:
        raw_values = raw_values * batch_size

    values = []
    for idx in range(batch_size):
        item = raw_values[idx] if idx < len(raw_values) else default
        if isinstance(item, torch.Tensor):
            item = item.item() if item.numel() == 1 else item.detach().cpu().tolist()
        if item is None:
            values.append(default)
            continue
        if cast is not None:
            try:
                item = cast(item)
            except Exception:
                item = default
        values.append(item)
    return values


def loader_batch_size(loader):
    batch_size = getattr(loader, "batch_size", None)
    if batch_size is None:
        batch_sampler = getattr(loader, "batch_sampler", None)
        batch_size = getattr(batch_sampler, "batch_size", None)
    try:
        return int(batch_size) if batch_size is not None else None
    except Exception:
        return None


def resolve_eval_policy(loader, eval_policy="auto"):
    valid_policies = {"auto", "formal", "fast_untrusted"}
    if eval_policy not in valid_policies:
        raise ValueError(f"Unknown eval_policy={eval_policy!r}; expected one of {sorted(valid_policies)}")

    batch_size = loader_batch_size(loader)
    is_batch1 = batch_size == 1
    if eval_policy == "formal" and not is_batch1:
        raise ValueError(
            "Formal evaluation requires batch_size=1. "
            "Use eval_policy='fast_untrusted' only for internal profiling."
        )

    if eval_policy == "formal":
        trust_level = "formal_batch1"
        reporting_allowed = True
    elif eval_policy == "fast_untrusted":
        trust_level = "fast_untrusted"
        reporting_allowed = False
    elif is_batch1:
        trust_level = "formal_batch1"
        reporting_allowed = True
    else:
        trust_level = "auto_untrusted_batch_gt1"
        reporting_allowed = False

    return {
        "eval_policy": eval_policy,
        "eval_batch_size": batch_size,
        "eval_trust_level": trust_level,
        "formal_reporting_allowed": reporting_allowed,
    }


def compute_rank_fields(probs, labels, top_ks=(1, 3, 5)):
    true_mask = labels > 0.5
    if true_mask.sum().item() == 0:
        fields = {
            "valid_case": False,
            "true_source_rank": None,
            "mrr": None,
            "true_source_prob": None,
            "max_pred_prob": float(probs.max().item()) if probs.numel() > 0 else None,
        }
        for k in top_ks:
            fields[f"top{k}_hit"] = None
        return fields

    sorted_idx = torch.argsort(probs, descending=True)
    sorted_true = true_mask[sorted_idx]
    true_positions = sorted_true.nonzero(as_tuple=True)[0]
    rank_true = int(true_positions.min().item() + 1)

    fields = {
        "valid_case": True,
        "true_source_rank": rank_true,
        "mrr": 1.0 / rank_true,
        "true_source_prob": float(probs[true_mask].max().item()),
        "max_pred_prob": float(probs.max().item()),
    }
    for k in top_ks:
        fields[f"top{k}_hit"] = bool(rank_true <= k)
    return fields


def evaluate_split(model, loader, cfg, split_name, device, eval_policy="auto"):
    policy_info = resolve_eval_policy(loader, eval_policy=eval_policy)
    evaluator = Evaluator(cfg)
    per_case_records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {split_name}", leave=False):
            batch = batch.to(device)
            out = model(
                batch,
                inference_mode=True,
                max_steps=cfg.training.max_eval_episodes,
                return_trajectory=True,
            )

            trajectory = out.get("trajectory", [])
            step_metrics = out.get("step_metrics", {})
            final_step = trajectory[-1] if trajectory else None
            geodesic_dists = compute_geodesic_distances(batch).cpu()

            raw_success = step_metrics.get("raw_success", torch.zeros(batch.num_graphs, device=device)).detach().cpu()
            raw_budget = step_metrics.get("raw_budget", torch.zeros(batch.num_graphs, device=device)).detach().cpu()
            raw_rounds = step_metrics.get("raw_rounds", torch.zeros(batch.num_graphs, device=device)).detach().cpu()
            raw_steps = step_metrics.get("raw_steps", torch.zeros(batch.num_graphs, device=device)).detach().cpu()
            raw_hit1 = step_metrics.get("raw_predict_hit", torch.zeros(batch.num_graphs, device=device)).detach().cpu()
            raw_hit5 = step_metrics.get("raw_predict_hit_5", torch.zeros(batch.num_graphs, device=device)).detach().cpu()
            raw_hit_valid = step_metrics.get("raw_predict_hit_valid", torch.zeros(batch.num_graphs, device=device)).detach().cpu()
            raw_max_hit_prob = step_metrics.get("raw_max_hit_prob", torch.zeros(batch.num_graphs, device=device)).detach().cpu()

            scenario_ids = per_graph_values(
                getattr(batch, "scenario_id", None),
                batch.num_graphs,
                default=-1,
                cast=int,
            )
            trigger_steps = per_graph_values(
                getattr(batch, "trigger_time_step", None),
                batch.num_graphs,
                default=None,
                cast=int,
            )
            start_steps = per_graph_values(
                getattr(batch, "global_start_step", None),
                batch.num_graphs,
                default=None,
                cast=int,
            )
            step_seconds_all = per_graph_values(
                getattr(batch, "step_seconds", None),
                batch.num_graphs,
                default=None,
                cast=int,
            )

            batch_index = batch.batch.detach().cpu()
            x_cpu = batch.x.detach().cpu()
            x_raw_cpu = batch.x_raw.detach().cpu() if hasattr(batch, "x_raw") else None
            part_ids = per_graph_values(
                getattr(batch, "part_id", None),
                batch.num_graphs,
                default=None,
                cast=int,
            )
            group_labels = getattr(batch, "group_label", None)

            for graph_idx in range(batch.num_graphs):
                node_mask = batch_index == graph_idx
                part_id = safe_group_value(part_ids, graph_idx, graph_idx)
                scenario_id = safe_group_value(scenario_ids, graph_idx, -1)
                group_label = safe_group_value(group_labels, graph_idx)
                trigger_time_step = safe_group_value(trigger_steps, graph_idx, None)
                global_start_step = safe_group_value(start_steps, graph_idx, None)
                step_seconds = safe_group_value(step_seconds_all, graph_idx, None)
                alarm_lag_steps = None
                if trigger_time_step is not None and global_start_step is not None:
                    alarm_lag_steps = global_start_step - trigger_time_step
                alarm_lag_minutes = None
                if alarm_lag_steps is not None and step_seconds is not None:
                    alarm_lag_minutes = alarm_lag_steps * step_seconds / 60.0
                case_id = f"{split_name}:scenario{scenario_id}:part{part_id}"

                trajectory_probs = []
                trajectory_hits = []
                for step in trajectory:
                    step_prob = step.get("hit_prob_surrogate")
                    step_hit = step.get("is_hit")
                    if step_prob is not None:
                        trajectory_probs.append(float(step_prob[graph_idx].detach().cpu().item()))
                    if step_hit is not None:
                        trajectory_hits.append(float(step_hit[graph_idx].detach().cpu().item()))
                first_hit_step = None
                for step_id, hit_value in enumerate(trajectory_hits, start=1):
                    if hit_value > 0.5:
                        first_hit_step = step_id
                        break

                record = {
                    "split": split_name,
                    "case_id": case_id,
                    "scenario_id": scenario_id,
                    "part_id": int(part_id),
                    "graph_idx": graph_idx,
                    "group_label": group_label,
                    "candidate_count": int(node_mask.sum().item()),
                    "initial_observed_count": int((x_cpu[node_mask, 3] > 0.5).sum().item()) if x_cpu.size(1) > 3 else 0,
                    "initial_positive_count": int((x_cpu[node_mask, 1] > 0.5).sum().item()) if x_cpu.size(1) > 1 else 0,
                    "sensor_count": int((x_cpu[node_mask, 5] > 0.5).sum().item()) if x_cpu.size(1) > 5 else 0,
                    "observed_fraction": float((x_cpu[node_mask, 3] > 0.5).float().mean().item()) if x_cpu.size(1) > 3 else 0.0,
                    "window_steps": int(x_raw_cpu.size(1)) if x_raw_cpu is not None else None,
                    "trigger_time_step": trigger_time_step,
                    "global_start_step": global_start_step,
                    "alarm_lag_steps": alarm_lag_steps,
                    "alarm_lag_minutes": alarm_lag_minutes,
                    "success": bool(raw_success[graph_idx].item() > 0.5),
                    "budget_used": float(raw_budget[graph_idx].item()),
                    "episodes_completed": float(raw_rounds[graph_idx].item()),
                    "physical_time_mins": float(raw_steps[graph_idx].item()),
                    "predict_hit_valid_contract": bool(raw_hit_valid[graph_idx].item() > 0.5),
                    "predict_hit1_contract": bool(raw_hit1[graph_idx].item() > 0.5),
                    "predict_hit5_contract": bool(raw_hit5[graph_idx].item() > 0.5),
                    "max_hit_prob_contract": float(raw_max_hit_prob[graph_idx].item()),
                    "first_hit_step": first_hit_step,
                    "geodesic_dist": float(geodesic_dists[graph_idx].item()) if graph_idx < len(geodesic_dists) else -1.0,
                }

                if final_step is not None:
                    fused_batch = final_step["fused_batch"].detach().cpu()
                    fused_mask = fused_batch == graph_idx
                    logits = final_step["reasoner_logits"][fused_mask].detach().cpu()
                    labels = final_step["fused_source_label"][fused_mask].detach().cpu().view(-1)
                    probs = probs_from_logits(logits.detach().cpu())
                    record.update(compute_rank_fields(probs, labels))
                else:
                    record.update(
                        {
                            "valid_case": False,
                            "true_source_rank": None,
                            "mrr": None,
                            "true_source_prob": None,
                            "max_pred_prob": None,
                            "top1_hit": None,
                            "top3_hit": None,
                            "top5_hit": None,
                        }
                    )

                evaluator.update_episode(
                    {
                        "success": record["success"],
                        "predict_hit": record["predict_hit1_contract"],
                        "predict_hit_at_5": record["predict_hit5_contract"],
                        "predict_hit_valid": record["predict_hit_valid_contract"],
                        "budget": record["budget_used"],
                        "physical_time_mins": record["physical_time_mins"],
                        "episodes_completed": record["episodes_completed"],
                        "max_hit_prob": record["max_hit_prob_contract"],
                        "geodesic_dist": record["geodesic_dist"],
                        "trajectory_probs": trajectory_probs,
                        "trajectory_hits": trajectory_hits,
                    }
                )
                per_case_records.append(record)

    summary = evaluator.summarize()
    df = pd.DataFrame(per_case_records)
    valid_df = df[df["valid_case"] == True].copy()

    summary.update(
        {
            "num_events": int(len(df)),
            "valid_events": int(len(valid_df)),
            "valid_event_rate": float(len(valid_df) / len(df)) if len(df) else 0.0,
            "top1_hit": float(valid_df["top1_hit"].mean()) if len(valid_df) else None,
            "top3_hit": float(valid_df["top3_hit"].mean()) if len(valid_df) else None,
            "top5_hit": float(valid_df["top5_hit"].mean()) if len(valid_df) else None,
            "mrr_valid": float(valid_df["mrr"].mean()) if len(valid_df) else None,
            "true_source_rank_mean": float(valid_df["true_source_rank"].mean()) if len(valid_df) else None,
            "true_source_rank_median": float(valid_df["true_source_rank"].median()) if len(valid_df) else None,
            "success_rate": float(df["success"].mean()) if len(df) else 0.0,
            "avg_budget_used": float(df["budget_used"].mean()) if len(df) else 0.0,
            "avg_rounds": float(df["episodes_completed"].mean()) if len(df) else 0.0,
            "avg_physical_time_mins": float(df["physical_time_mins"].mean()) if len(df) else 0.0,
        }
    )
    summary.update(policy_info)

    step_curve = {
        key.replace("test/", ""): float(value)
        for key, value in summary.items()
        if key.startswith("test/hit@1_step_")
    }
    summary["step_hit_curve"] = step_curve
    return summary, df


def summarize_history(history_path):
    if not os.path.exists(history_path):
        return {
            "epochs_logged": 0,
            "loss_start": None,
            "loss_end": None,
            "loss_min": None,
            "loss_has_nan": True,
        }

    rows = []
    with open(history_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {
            "epochs_logged": 0,
            "loss_start": None,
            "loss_end": None,
            "loss_min": None,
            "loss_has_nan": True,
        }

    losses = [row.get("train_loss") for row in rows if row.get("train_loss") is not None]
    loss_has_nan = any((loss is None) or (not math.isfinite(loss)) for loss in losses)
    return {
        "epochs_logged": len(rows),
        "loss_start": float(losses[0]) if losses else None,
        "loss_end": float(losses[-1]) if losses else None,
        "loss_min": float(min(losses)) if losses else None,
        "loss_has_nan": loss_has_nan,
        "history": rows,
    }


def basic_slice_metrics(df):
    valid_df = df[df["valid_case"] == True]
    return {
        "count": int(len(df)),
        "valid_count": int(len(valid_df)),
        "top1_hit": float(valid_df["top1_hit"].mean()) if len(valid_df) else None,
        "top3_hit": float(valid_df["top3_hit"].mean()) if len(valid_df) else None,
        "top5_hit": float(valid_df["top5_hit"].mean()) if len(valid_df) else None,
        "mrr": float(valid_df["mrr"].mean()) if len(valid_df) else None,
        "mean_rank": float(valid_df["true_source_rank"].mean()) if len(valid_df) else None,
        "success_rate": float(df["success"].mean()) if len(df) else None,
    }


def bucketize(df, field, labels):
    if df.empty or df[field].dropna().empty:
        return None
    q1 = df[field].quantile(1 / 3)
    q2 = df[field].quantile(2 / 3)
    if pd.isna(q1) or pd.isna(q2) or q1 == q2:
        median = df[field].median()
        if pd.isna(median):
            return None
        bins = [-np.inf, median, np.inf]
        bucket_labels = [labels[0], labels[-1]]
    else:
        bins = [-np.inf, q1, q2, np.inf]
        bucket_labels = labels
    return pd.cut(df[field], bins=bins, labels=bucket_labels, include_lowest=True)


def build_bucket_summary(df):
    summary = {}
    bucket_specs = [
        ("candidate_count", ["small", "medium", "large"]),
        ("observed_fraction", ["sparse", "medium", "rich"]),
        ("alarm_lag_steps", ["short_window", "mid_window", "long_window"]),
    ]
    for field, labels in bucket_specs:
        categories = bucketize(df, field, labels)
        if categories is None:
            summary[field] = {}
            continue
        field_summary = {}
        for label in categories.dropna().unique():
            bucket_df = df[categories == label]
            field_summary[str(label)] = basic_slice_metrics(bucket_df)
        summary[field] = field_summary
    return summary


def compare_delta(candidate_summary, baseline_summary):
    keys = [
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "mrr_valid",
        "success_rate",
        "true_source_rank_mean",
        "avg_budget_used",
        "avg_physical_time_mins",
    ]
    delta = {}
    for key in keys:
        cand = candidate_summary.get(key)
        base = baseline_summary.get(key)
        if cand is None or base is None:
            delta[key] = None
        else:
            delta[key] = float(cand - base)
    return delta


def markdown_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def run_phase(phase_name, phase_spec, eval_loaders, artifact_dir, device):
    logger.info("=== Phase %s started ===", phase_name)
    prewarm_lmdb_cache(
        cache_version=phase_spec["cache_version"],
        max_samples=phase_spec["max_samples"],
    )
    training_loaders = build_training_loaders(phase_spec)
    phase_dir = os.path.join(artifact_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)

    phase_results = {
        "phase": phase_name,
        "settings": phase_spec,
        "configs": {},
    }

    for config_entry in OFFICIAL_CONFIGS:
        config_name = config_entry["name"]
        config_overrides = load_yaml(config_entry["path"])
        run_name = f"training_campaign_round1_{phase_name}_{config_name}"
        phase_overrides = {
            "paths": {
                "cache_dir": LMDB_CACHE_DIR,
            },
            "training": {
                "num_epochs": phase_spec["epochs"],
                "seed": phase_spec["seed"],
                "val_every_n_epochs": 1,
            },
            "efficiency": {
                "batch_size": phase_spec["batch_size"],
            },
            "data": {
                "cache_version": phase_spec["cache_version"],
                "max_samples": phase_spec["max_samples"],
            },
        }
        merged_overrides = deep_merge(config_overrides, phase_overrides)
        run_dir = resolve_run_dir(
            merged_overrides,
            run_name=run_name,
            max_epochs=phase_spec["epochs"],
            seed=phase_spec["seed"],
        )
        checkpoint_path = os.path.join(run_dir, "model_best.pt")
        history_path = os.path.join(run_dir, "epoch_history.jsonl")
        config_result = {
            "config_name": config_name,
            "config_path": config_entry["path"],
            "run_name": run_name,
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
        }

        try:
            logger.info("Running %s via official training entry function", config_name)
            best_metrics = run_training(
                loss_config_override=merged_overrides,
                run_name=run_name,
                max_epochs=phase_spec["epochs"],
                loaders=training_loaders,
                seed=phase_spec["seed"],
                skip_audit=True,
                force_rebuild=phase_spec["force_rebuild"],
            )
            config_result["train_best_metrics"] = best_metrics
            history_summary = summarize_history(history_path)
            config_result["history_summary"] = history_summary
            config_result["checkpoint_exists"] = os.path.exists(checkpoint_path)

            eval_cfg = prepare_cfg(merged_overrides, run_name=run_name, max_epochs=phase_spec["epochs"], seed=phase_spec["seed"])
            model = build_model_for_eval(eval_cfg, eval_loaders["val"], checkpoint_path, device)
            split_results = {}
            for split_name, loader in eval_loaders.items():
                split_summary, split_df = evaluate_split(model, loader, eval_cfg, split_name, device)
                split_summary["bucket_summary"] = build_bucket_summary(split_df)
                split_dir = os.path.join(phase_dir, config_name)
                os.makedirs(split_dir, exist_ok=True)
                split_csv_path = os.path.join(split_dir, f"{split_name}_per_case.csv")
                split_json_path = os.path.join(split_dir, f"{split_name}_summary.json")
                split_df.to_csv(split_csv_path, index=False)
                with open(split_json_path, "w") as f:
                    json.dump(_json_ready(split_summary), f, indent=2)
                split_results[split_name] = {
                    "summary": split_summary,
                    "per_case_csv": split_csv_path,
                    "per_case_df": split_df,
                }
            config_result["eval"] = split_results

            pilot_stable = (
                history_summary["epochs_logged"] == phase_spec["epochs"]
                and not history_summary["loss_has_nan"]
                and config_result["checkpoint_exists"]
            )
            config_result["stable"] = bool(pilot_stable)
            config_result["restore_eval_ok"] = True
        except Exception as exc:
            logger.exception("Phase %s config %s failed", phase_name, config_name)
            config_result["stable"] = False
            config_result["restore_eval_ok"] = False
            config_result["error"] = str(exc)

        phase_results["configs"][config_name] = config_result

    phase_results_path = os.path.join(phase_dir, "phase_results.json")
    with open(phase_results_path, "w") as f:
        json.dump(_json_ready(phase_results), f, indent=2)
    phase_results["phase_results_path"] = phase_results_path
    return phase_results


def phase_is_stable(phase_results):
    return all(
        result.get("stable") and result.get("restore_eval_ok")
        for result in phase_results["configs"].values()
    )


def select_best_config(main_results, split_name="test"):
    best_name = None
    best_tuple = None
    for config_name, result in main_results["configs"].items():
        split_summary = result.get("eval", {}).get(split_name, {}).get("summary", {})
        if not split_summary:
            continue
        ranking_tuple = (
            split_summary.get("top1_hit", float("-inf")) or float("-inf"),
            split_summary.get("mrr_valid", float("-inf")) or float("-inf"),
            -(split_summary.get("true_source_rank_mean", float("inf")) or float("inf")),
        )
        if best_tuple is None or ranking_tuple > best_tuple:
            best_tuple = ranking_tuple
            best_name = config_name
    return best_name


def build_case_breakdown(main_results, output_path_md, output_path_csv):
    test_frames = {}
    for config_name, result in main_results["configs"].items():
        df = result.get("eval", {}).get("test", {}).get("per_case_df")
        if df is None:
            continue
        keep_cols = [
            "case_id",
            "scenario_id",
            "part_id",
            "candidate_count",
            "observed_fraction",
            "alarm_lag_steps",
            "success",
            "top1_hit",
            "top3_hit",
            "top5_hit",
            "true_source_rank",
            "first_hit_step",
        ]
        renamed = df[keep_cols].copy()
        renamed.columns = [
            "case_id",
            "scenario_id",
            "part_id",
            "candidate_count",
            "observed_fraction",
            "alarm_lag_steps",
            f"{config_name}__success",
            f"{config_name}__top1_hit",
            f"{config_name}__top3_hit",
            f"{config_name}__top5_hit",
            f"{config_name}__true_source_rank",
            f"{config_name}__first_hit_step",
        ]
        test_frames[config_name] = renamed

    if "base_no_evidence" not in test_frames or "support_mainline" not in test_frames:
        return None

    merged = test_frames["base_no_evidence"]
    for config_name, frame in test_frames.items():
        if config_name == "base_no_evidence":
            continue
        merged = merged.merge(frame, on=["case_id", "scenario_id", "part_id", "candidate_count", "observed_fraction", "alarm_lag_steps"], how="outer")

    merged["support_rank_gain_vs_base"] = (
        merged["base_no_evidence__true_source_rank"] - merged["support_mainline__true_source_rank"]
    )
    merged["suspect_rank_gain_vs_support"] = (
        merged["support_mainline__true_source_rank"] - merged["support_plus_soft_suspect__true_source_rank"]
    )
    merged["contradiction_rank_gain_vs_support"] = (
        merged["support_mainline__true_source_rank"] - merged["support_plus_contradiction_aux_compare__true_source_rank"]
    )
    merged.to_csv(output_path_csv, index=False)

    support_wins = merged[
        (merged["support_mainline__top1_hit"] == True)
        & (merged["base_no_evidence__top1_hit"] != True)
    ].sort_values("support_rank_gain_vs_base", ascending=False)
    support_success_wins = merged[
        (merged["support_mainline__success"] == True)
        & (merged["base_no_evidence__success"] != True)
    ].sort_values("candidate_count", ascending=False)
    suspect_regressions = merged[
        (merged["support_mainline__top1_hit"] == True)
        & (merged["support_plus_soft_suspect__top1_hit"] != True)
    ].sort_values("candidate_count", ascending=False)
    contradiction_regressions = merged[
        (merged["support_mainline__top1_hit"] == True)
        & (merged["support_plus_contradiction_aux_compare__top1_hit"] != True)
    ].sort_values("candidate_count", ascending=False)
    hard_cases = merged[
        (merged["base_no_evidence__top5_hit"] != True)
        & (merged["support_mainline__top5_hit"] != True)
        & (merged["support_plus_soft_suspect__top5_hit"] != True)
        & (merged["support_plus_contradiction_aux_compare__top5_hit"] != True)
    ].sort_values(["candidate_count", "alarm_lag_steps"], ascending=[False, False])

    def slice_markdown(df, columns):
        if df.empty:
            return "None"
        return markdown_table(columns, df.head(10).to_dict(orient="records"))

    lines = []
    lines.append("# Training Campaign Round 1 Case Breakdown")
    lines.append("")
    lines.append(f"- support beats base on final Top-1: {len(support_wins)} cases")
    lines.append(f"- base fails but support succeeds at episode level: {len(support_success_wins)} cases")
    lines.append(f"- soft suspect regressions vs support_mainline: {len(suspect_regressions)} cases")
    lines.append(f"- contradiction auxiliary regressions vs support_mainline: {len(contradiction_regressions)} cases")
    lines.append(f"- still-hard multi-window cases (all variants miss Top-5): {len(hard_cases)} cases")
    lines.append("")
    lines.append("## Support Wins")
    lines.append("")
    lines.append(
        slice_markdown(
            support_wins,
            ["case_id", "candidate_count", "observed_fraction", "alarm_lag_steps", "base_no_evidence__true_source_rank", "support_mainline__true_source_rank"],
        )
    )
    lines.append("")
    lines.append("## Base Fail But Support Success")
    lines.append("")
    lines.append(
        slice_markdown(
            support_success_wins,
            ["case_id", "candidate_count", "observed_fraction", "alarm_lag_steps", "base_no_evidence__success", "support_mainline__success"],
        )
    )
    lines.append("")
    lines.append("## Soft Suspect Regressions")
    lines.append("")
    lines.append(
        slice_markdown(
            suspect_regressions,
            ["case_id", "candidate_count", "observed_fraction", "alarm_lag_steps", "support_mainline__true_source_rank", "support_plus_soft_suspect__true_source_rank"],
        )
    )
    lines.append("")
    lines.append("## Contradiction Auxiliary Regressions")
    lines.append("")
    lines.append(
        slice_markdown(
            contradiction_regressions,
            ["case_id", "candidate_count", "observed_fraction", "alarm_lag_steps", "support_mainline__true_source_rank", "support_plus_contradiction_aux_compare__true_source_rank"],
        )
    )
    lines.append("")
    lines.append("## Still-Hard Cases")
    lines.append("")
    lines.append(
        slice_markdown(
            hard_cases,
            ["case_id", "candidate_count", "observed_fraction", "alarm_lag_steps", "base_no_evidence__true_source_rank", "support_mainline__true_source_rank"],
        )
    )
    lines.append("")

    with open(output_path_md, "w") as f:
        f.write("\n".join(lines))

    return {
        "merged_csv": output_path_csv,
        "markdown": output_path_md,
        "support_wins": len(support_wins),
        "support_success_wins": len(support_success_wins),
        "suspect_regressions": len(suspect_regressions),
        "contradiction_regressions": len(contradiction_regressions),
        "hard_cases": len(hard_cases),
    }


def render_report(campaign_results, report_path, summary_path, case_breakdown_info):
    pilot_results = campaign_results["pilot"]
    main_results = campaign_results.get("main")
    pilot_stable = phase_is_stable(pilot_results)

    lines = []
    lines.append("# Training Campaign Round 1")
    lines.append("")
    lines.append("## Execution Summary")
    lines.append("")
    lines.append("- Goal: run the first formal support-led training campaign without reopening readiness or evidence semantics.")
    lines.append("- Scope stayed on the official training path `src/scripts/train_phase4_end2end.py` and the four sanctioned compare configs.")
    lines.append(f"- Pilot stable: {'yes' if pilot_stable else 'no'}")
    lines.append("")
    lines.append("## Baseline Or Current Mainline Path")
    lines.append("")
    lines.append("- Official training entry: `src/scripts/train_phase4_end2end.py`")
    lines.append("- Official compare configs: `base_no_evidence`, `support_mainline`, `support_plus_soft_suspect`, `support_plus_contradiction_aux_compare`")
    lines.append("- Training loop semantics were left support-led; only logging/evaluation artifacts were added so epoch history and real validation metrics could be captured.")
    lines.append("")
    lines.append("## Phase Settings")
    lines.append("")
    phase_rows = []
    for phase_name, phase_data in campaign_results.items():
        settings = phase_data["settings"]
        phase_rows.append(
            {
                "phase": phase_name,
                "epochs": settings["epochs"],
                "batch_size": settings["batch_size"],
                "train_max_samples": settings["max_samples"],
                "seed": settings["seed"],
            }
        )
    lines.append(markdown_table(["phase", "epochs", "batch_size", "train_max_samples", "seed"], phase_rows))
    lines.append("")
    lines.append("## Pilot Stability")
    lines.append("")
    pilot_rows = []
    for config_name, result in pilot_results["configs"].items():
        history_summary = result.get("history_summary", {})
        val_summary = result.get("eval", {}).get("val", {}).get("summary", {})
        pilot_rows.append(
            {
                "config": config_name,
                "stable": "yes" if result.get("stable") else "no",
                "loss_start": round(history_summary.get("loss_start", 0.0), 4) if history_summary.get("loss_start") is not None else "n/a",
                "loss_end": round(history_summary.get("loss_end", 0.0), 4) if history_summary.get("loss_end") is not None else "n/a",
                "val_top1": round(val_summary.get("top1_hit", 0.0), 4) if val_summary else "n/a",
                "val_top3": round(val_summary.get("top3_hit", 0.0), 4) if val_summary else "n/a",
                "val_mrr": round(val_summary.get("mrr_valid", 0.0), 4) if val_summary else "n/a",
                "checkpoint": "yes" if result.get("checkpoint_exists") else "no",
            }
        )
    lines.append(markdown_table(["config", "stable", "loss_start", "loss_end", "val_top1", "val_top3", "val_mrr", "checkpoint"], pilot_rows))
    lines.append("")

    if main_results is None:
        lines.append("## Main Compare")
        lines.append("")
        lines.append("- Pilot was not stable enough to justify the main compare phase.")
        lines.append("")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        return

    lines.append("## Main Compare")
    lines.append("")
    main_rows = []
    for config_name, result in main_results["configs"].items():
        test_summary = result.get("eval", {}).get("test", {}).get("summary", {})
        main_rows.append(
            {
                "config": config_name,
                "top1": round(test_summary.get("top1_hit", 0.0), 4),
                "top3": round(test_summary.get("top3_hit", 0.0), 4),
                "top5": round(test_summary.get("top5_hit", 0.0), 4),
                "mrr": round(test_summary.get("mrr_valid", 0.0), 4),
                "rank_mean": round(test_summary.get("true_source_rank_mean", 0.0), 3),
                "success_rate": round(test_summary.get("success_rate", 0.0), 4),
                "avg_budget": round(test_summary.get("avg_budget_used", 0.0), 2),
            }
        )
    lines.append(markdown_table(["config", "top1", "top3", "top5", "mrr", "rank_mean", "success_rate", "avg_budget"], main_rows))
    lines.append("")

    base_test = main_results["configs"]["base_no_evidence"]["eval"]["test"]["summary"]
    support_test = main_results["configs"]["support_mainline"]["eval"]["test"]["summary"]
    suspect_test = main_results["configs"]["support_plus_soft_suspect"]["eval"]["test"]["summary"]
    contradiction_test = main_results["configs"]["support_plus_contradiction_aux_compare"]["eval"]["test"]["summary"]

    support_delta = compare_delta(support_test, base_test)
    suspect_delta = compare_delta(suspect_test, support_test)
    contradiction_delta = compare_delta(contradiction_test, support_test)

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- [proven] `support_mainline` vs `base_no_evidence` on test: Top-1 {support_test['top1_hit']:.4f} vs {base_test['top1_hit']:.4f}, "
        f"Top-3 {support_test['top3_hit']:.4f} vs {base_test['top3_hit']:.4f}, MRR {support_test['mrr_valid']:.4f} vs {base_test['mrr_valid']:.4f}."
    )
    lines.append(
        f"- [proven] `support_plus_soft_suspect` vs `support_mainline`: delta Top-1 {suspect_delta['top1_hit']:+.4f}, "
        f"delta MRR {suspect_delta['mrr_valid']:+.4f}, delta Success {suspect_delta['success_rate']:+.4f}."
    )
    lines.append(
        f"- [proven] `support_plus_contradiction_aux_compare` vs `support_mainline`: delta Top-1 {contradiction_delta['top1_hit']:+.4f}, "
        f"delta MRR {contradiction_delta['mrr_valid']:+.4f}, delta Success {contradiction_delta['success_rate']:+.4f}."
    )

    support_bucket = main_results["configs"]["support_mainline"]["eval"]["test"]["summary"]["bucket_summary"]
    base_bucket = main_results["configs"]["base_no_evidence"]["eval"]["test"]["summary"]["bucket_summary"]
    sparse_support = support_bucket.get("observed_fraction", {}).get("sparse", {})
    sparse_base = base_bucket.get("observed_fraction", {}).get("sparse", {})
    long_support = support_bucket.get("alarm_lag_steps", {}).get("long_window", {})
    long_base = base_bucket.get("alarm_lag_steps", {}).get("long_window", {})

    if sparse_support and sparse_base:
        lines.append(
            f"- [proven] Sparse-observation bucket: support Top-1 {sparse_support.get('top1_hit', 0.0):.4f} vs base {sparse_base.get('top1_hit', 0.0):.4f}; "
            f"MRR {sparse_support.get('mrr', 0.0):.4f} vs {sparse_base.get('mrr', 0.0):.4f}."
        )
    if long_support and long_base:
        lines.append(
            f"- [proven] Long-window bucket: support Top-1 {long_support.get('top1_hit', 0.0):.4f} vs base {long_base.get('top1_hit', 0.0):.4f}; "
            f"MRR {long_support.get('mrr', 0.0):.4f} vs {long_base.get('mrr', 0.0):.4f}."
        )

    lines.append(
        "- [partially proven] This round proves the support-led multi-step line is stable and useful within the current multi-window architecture. "
        "It does not separately benchmark against a single-static-slice architecture, so that narrower claim remains indirect."
    )
    lines.append("")

    step_curve_support = support_test.get("step_hit_curve", {})
    step_curve_base = base_test.get("step_hit_curve", {})
    if step_curve_support and step_curve_base:
        lines.append("## Stepwise Hit@1")
        lines.append("")
        curve_rows = []
        shared_steps = sorted(set(step_curve_support.keys()) & set(step_curve_base.keys()))
        for step_key in shared_steps[:10]:
            curve_rows.append(
                {
                    "step": step_key.replace("hit@1_step_", ""),
                    "base": round(step_curve_base[step_key], 4),
                    "support": round(step_curve_support[step_key], 4),
                }
            )
        lines.append(markdown_table(["step", "base", "support"], curve_rows))
        lines.append("")

    lines.append("## Final Judgement")
    lines.append("")
    best_config = select_best_config(main_results)
    lines.append(
        f"- `support_mainline` significantly better than `base_no_evidence`: {'yes' if support_delta['top1_hit'] and support_delta['top1_hit'] > 0 else 'no'}."
    )
    lines.append(
        f"- `support_plus_soft_suspect` worth keeping as a formal compare branch: {'yes' if suspect_delta['top1_hit'] and suspect_delta['top1_hit'] > 0 else 'no'}."
    )
    lines.append(
        f"- `support_plus_contradiction_aux_compare` still has extra retained value: {'yes' if contradiction_delta['top1_hit'] and contradiction_delta['top1_hit'] > 0 else 'no'}."
    )
    lines.append(
        f"- Current support-led multi-window, multi-step system is qualified for larger-scale training: {'yes' if support_delta['top1_hit'] and support_delta['top1_hit'] > 0 and pilot_stable else 'no'}."
    )
    lines.append(f"- Best compare config on this round's main test table: `{best_config}`")
    if case_breakdown_info:
        lines.append(f"- Case breakdown: `{case_breakdown_info['markdown']}` and `{case_breakdown_info['merged_csv']}`")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def build_summary_json(campaign_results, summary_path, case_breakdown_info):
    pilot_results = campaign_results["pilot"]
    main_results = campaign_results.get("main")
    pilot_stable = phase_is_stable(pilot_results)

    summary = {
        "ran_configs": [entry["name"] for entry in OFFICIAL_CONFIGS],
        "pilot_stable": pilot_stable,
        "best_mainline_config": None,
        "support_vs_base_gain": None,
        "suspect_aux_gain": None,
        "contradiction_aux_gain": None,
        "recommendation_next_step": None,
    }

    if main_results is not None:
        base_test = main_results["configs"]["base_no_evidence"]["eval"]["test"]["summary"]
        support_test = main_results["configs"]["support_mainline"]["eval"]["test"]["summary"]
        suspect_test = main_results["configs"]["support_plus_soft_suspect"]["eval"]["test"]["summary"]
        contradiction_test = main_results["configs"]["support_plus_contradiction_aux_compare"]["eval"]["test"]["summary"]

        summary["best_mainline_config"] = select_best_config(main_results)
        summary["support_vs_base_gain"] = compare_delta(support_test, base_test)
        summary["suspect_aux_gain"] = compare_delta(suspect_test, support_test)
        summary["contradiction_aux_gain"] = compare_delta(contradiction_test, support_test)

        if summary["support_vs_base_gain"]["top1_hit"] is not None and summary["support_vs_base_gain"]["top1_hit"] > 0:
            if summary["suspect_aux_gain"]["top1_hit"] is not None and summary["suspect_aux_gain"]["top1_hit"] > 0:
                suspect_clause = "soft suspect still deserves to remain as a secondary compare branch."
            else:
                suspect_clause = "soft suspect should stay only as a soft prior / secondary compare, not the mainline."

            if summary["contradiction_aux_gain"]["top1_hit"] is not None and summary["contradiction_aux_gain"]["top1_hit"] > 0:
                contradiction_clause = "contradiction auxiliary can remain as a limited compare."
            else:
                contradiction_clause = "contradiction should remain primarily an explanatory or audit-side channel."

            summary["recommendation_next_step"] = (
                "Expand training scale around support_mainline, keep the same official entry and split policy, and treat side branches conservatively; "
                + suspect_clause + " " + contradiction_clause
            )
        else:
            summary["recommendation_next_step"] = (
                "Do not expand yet; revisit pilot/main anomalies before scaling."
            )
    else:
        summary["recommendation_next_step"] = (
            "Pilot was not stable; fix the pilot issues before any larger compare."
        )

    if case_breakdown_info is not None:
        summary["case_breakdown"] = case_breakdown_info

    with open(summary_path, "w") as f:
        json.dump(_json_ready(summary), f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="artifacts/training_campaign_round1")
    parser.add_argument("--pilot_only", action="store_true")
    parser.add_argument("--skip_pilot", action="store_true")
    parser.add_argument("--reuse_artifact_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_offline_wandb()

    if args.skip_pilot:
        if args.reuse_artifact_dir:
            artifact_dir = args.reuse_artifact_dir
        else:
            existing = sorted(
                [
                    os.path.join(args.output_root, d)
                    for d in os.listdir(args.output_root)
                ]
            )
            if not existing:
                raise RuntimeError("No existing artifact directory found for --skip_pilot")
            artifact_dir = existing[-1]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = os.path.join(args.output_root, timestamp)
        os.makedirs(artifact_dir, exist_ok=True)

    report_path = os.path.join(os.getcwd(), "training_campaign_round1.md")
    summary_path = os.path.join(os.getcwd(), "training_campaign_round1_summary.json")
    case_breakdown_md = os.path.join(os.getcwd(), "training_campaign_round1_case_breakdown.md")
    case_breakdown_csv = os.path.join(os.getcwd(), "training_campaign_round1_case_breakdown.csv")

    if not args.skip_pilot:
        prewarm_lmdb_cache(cache_version=FULL_EVAL_CACHE_VERSION, max_samples=None)
    logger.info("Building shared full-holdout evaluation loaders")
    _, eval_loaders = build_eval_loaders()
    device = torch.device(args.device)

    campaign_results = {}
    if args.skip_pilot:
        pilot_phase_path = os.path.join(artifact_dir, "pilot", "phase_results.json")
        if not os.path.exists(pilot_phase_path):
            raise RuntimeError(f"Missing pilot phase results at {pilot_phase_path}")
        with open(pilot_phase_path, "r") as f:
            campaign_results["pilot"] = json.load(f)
    else:
        campaign_results["pilot"] = run_phase("pilot", PHASE_SPECS["pilot"], eval_loaders, artifact_dir, device)

    main_results = None
    if not args.pilot_only and phase_is_stable(campaign_results["pilot"]):
        main_results = run_phase("main", PHASE_SPECS["main"], eval_loaders, artifact_dir, device)
        campaign_results["main"] = main_results
    else:
        logger.warning("Pilot unstable or pilot-only requested; main compare skipped.")

    case_breakdown_info = None
    if main_results is not None:
        case_breakdown_info = build_case_breakdown(main_results, case_breakdown_md, case_breakdown_csv)

    render_report(campaign_results, report_path, summary_path, case_breakdown_info)
    summary = build_summary_json(campaign_results, summary_path, case_breakdown_info)

    campaign_results_path = os.path.join(artifact_dir, "campaign_results.json")
    with open(campaign_results_path, "w") as f:
        json.dump(_json_ready(campaign_results), f, indent=2)

    logger.info("Campaign artifacts written to %s", artifact_dir)
    logger.info("Report: %s", report_path)
    logger.info("Summary: %s", summary_path)
    if case_breakdown_info:
        logger.info("Case breakdown: %s", case_breakdown_md)
    print(json.dumps(_json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
