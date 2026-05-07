from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.semi_dynamic_bank import (
    SemiDynamicBankStats,
    SemiDynamicTrajectoryBankWriter,
    build_bank_sample,
    path_size_bytes,
)
from src.data.v6.topology import HydraulicTopology
from src.data.v6.loader import create_dataloaders
from src.modeling.builders.model_builder import ModelBuilder
from src.modeling.clean_aligned_features import build_clean_aligned_feature_payload
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.build_reasoner_metric_contract_repair_bundle import (
    standardize_case_df,
    standardize_step_df,
)
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_aligned_reasoner_mainline import prepare_cfg
from src.scripts.diagnostics.run_clean_navigator_v1 import build_state_bundle, resolve_source_local_idx
from src.scripts.diagnostics.run_slot1_counterfactual_leverage_audit import (
    build_namespace_from_control_args,
    load_control_bundle,
)
from src.scripts.run_reasoner_train_only_overfit_500 import (
    build_overfit_overrides,
    check_gpu_exclusive,
    collect_candidate_checkpoints,
    safe_float,
    write_json,
)
from src.scripts.train_clean_aligned_online_finish import load_plain_state_dict
from src.scripts.train_frozen_clean_nav_reasoner_semidynamic import (
    preferred_bank_root,
    run_batch_gate,
    train_offline_reasoner,
)


DEFAULT_BASELINE_ROOT = (
    PROJECT_ROOT / "artifacts" / "reasoner_train_only_overfit" / "20260407_train500_frozen_nav_overfit_v1"
)
DEFAULT_ORACLE_ROOT = (
    PROJECT_ROOT / "artifacts" / "task_defined_oracle_sampler" / "20260407_train500_newpbest_h3"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact138_h3_v1"
)
DEFAULT_BRIDGE_PACKAGE = (
    PROJECT_ROOT / "artifacts" / "clean_navigator_v1" / "navigator_final_delivery_p_seed0_newdataset_currentrunner_20260406"
)
DEFAULT_INIT_CHECKPOINT = PROJECT_ROOT / "runs" / "clean_aligned_reasoner_mainline_f6549ff3d356" / "model_best.pt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
DEFAULT_BATCH_CANDIDATES = [256, 128, 64, 32]
RUNNER_VERSION = "same_case_stronger_source_overfit_v1"
PANEL_VERSION = "exact_same_case_train_only_h3_trainset_v1"


@dataclass
class ActionStep:
    case_id: str
    scenario_id: int
    part_id: int
    round_index: int
    global_ids: List[int]
    label: str


@dataclass
class CaseRecord:
    case_id: str
    scenario_id: int
    part_id: int
    dataset_index: int
    data: Any


class TempGraph:
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device):
        self.edge_index = edge_index.to(device=device, dtype=torch.long)
        self.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)


def move_payload(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_payload(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_payload(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_payload(v, device) for v in value)
    if hasattr(value, "__dataclass_fields__"):
        return type(value)(**{k: move_payload(getattr(value, k), device) for k in value.__dataclass_fields__.keys()})
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact same-case stronger-source train-only overfit verdict runner.")
    parser.add_argument("--baseline-root", type=str, default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--oracle-root", type=str, default=str(DEFAULT_ORACLE_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bridge-package-dir", type=str, default=str(DEFAULT_BRIDGE_PACKAGE))
    parser.add_argument("--init-checkpoint", type=str, default=str(DEFAULT_INIT_CHECKPOINT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--periodic-checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=DEFAULT_BATCH_CANDIDATES)
    parser.add_argument("--offline-workers", type=int, default=8)
    parser.add_argument("--offline-prefetch-factor", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_combo_field(value: str) -> List[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    return [int(v) for v in parsed]


def colon_case_id_from_data(data: Any, split_name: str, dataset_idx: int) -> tuple[str, int, int]:
    scenario_raw = getattr(data, "scenario_id", None)
    part_raw = getattr(data, "part_id", None)
    if isinstance(scenario_raw, torch.Tensor):
        scenario_val = int(scenario_raw.view(-1)[0].item())
    elif isinstance(scenario_raw, (list, tuple)):
        scenario_val = int(scenario_raw[0])
    elif scenario_raw is None:
        scenario_val = int(dataset_idx)
    else:
        scenario_val = int(scenario_raw)
    if isinstance(part_raw, torch.Tensor):
        part_val = int(part_raw.view(-1)[0].item())
    elif isinstance(part_raw, (list, tuple)):
        part_val = int(part_raw[0])
    elif part_raw is None:
        part_val = 0
    else:
        part_val = int(part_raw)
    return f"{split_name}:scenario{scenario_val}:part{part_val}", int(scenario_val), int(part_val)


def collect_dataset_assets(dataset: Any) -> Dict[str, Any]:
    asset_source_chain = []
    asset_source = dataset
    while asset_source is not None:
        asset_source_chain.append(asset_source)
        next_source = getattr(asset_source, "dataset", None)
        if next_source is asset_source:
            break
        asset_source = next_source

    def first_non_none_attr(name: str):
        for source in asset_source_chain:
            value = getattr(source, name, None)
            if value is not None:
                return value
        return None

    topology = first_non_none_attr("topology")
    return {
        "topology": topology,
        "global_edge_index": first_non_none_attr("global_edge_index"),
        "stt_dynamic_series": first_non_none_attr("stt_dynamic_series"),
        "num_global_nodes": first_non_none_attr("num_nodes"),
    }


def build_same_case_manifest(baseline_root: Path, oracle_root: Path, output_dir: Path) -> Dict[str, Any]:
    baseline_case_df = pd.read_csv(baseline_root / "train_eval" / "init" / "standardized_case_metrics.csv")
    oracle_case_df = pd.read_csv(oracle_root / "raw" / "oracle_case_rows.csv")
    baseline_ids = sorted(str(v) for v in baseline_case_df["case_id"].dropna().astype(str).tolist())
    oracle_ids = sorted(str(v) for v in oracle_case_df["case_id"].dropna().astype(str).tolist())
    exact_ids = sorted(set(baseline_ids) & set(oracle_ids))
    rows = []
    for case_id in exact_ids:
        match = baseline_case_df[baseline_case_df["case_id"] == case_id].iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "scenario_id": int(match["scenario_id"]),
                "part_id": int(match["part_id"]),
            }
        )
    manifest = {
        "runner_version": RUNNER_VERSION,
        "baseline_root": str(baseline_root),
        "oracle_root": str(oracle_root),
        "case_id_alignment_contract": "colon form `train:scenario{scenario_id}:part{part_id}` from current authoritative artifacts",
        "baseline_subset_size": int(len(baseline_ids)),
        "oracle_covered_size": int(len(oracle_ids)),
        "exact_intersection_size": int(len(exact_ids)),
        "baseline_part_counts": baseline_case_df["part_id"].value_counts().sort_index().to_dict(),
        "oracle_part_counts": oracle_case_df["part_id"].astype(int).value_counts().sort_index().to_dict(),
        "exact_case_ids_path": str(output_dir / "same_case_manifest.csv"),
    }
    same_case_df = pd.DataFrame(rows).sort_values(["scenario_id", "part_id"]).reset_index(drop=True)
    same_case_df.to_csv(output_dir / "same_case_manifest.csv", index=False)
    write_json(output_dir / "same_case_manifest.json", manifest)
    return manifest


def filter_replayable_cases(
    *,
    cases: Sequence[CaseRecord],
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
) -> tuple[List[CaseRecord], List[str]]:
    env = CleanTwoChannelEvidenceEnv()
    topology = dataset_assets["topology"]
    kept = []
    dropped = []
    for case in cases:
        rollout = PracticalRollout(
            event_data=deepcopy(case.data),
            global_edge_index=dataset_assets["global_edge_index"],
            stt_dynamic_series=dataset_assets["stt_dynamic_series"],
            num_global_nodes=int(dataset_assets["num_global_nodes"]),
            num_episodes=int(num_episodes),
            samples_per_episode=int(action_budget),
            episode_duration_min=float(episode_duration_min),
        )
        history = ObservationWitnessHistory()
        _ = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=num_episodes,
            action_budget=action_budget,
            frontier_role_mode=frontier_role_mode,
        )
        if resolve_source_local_idx(rollout) is None:
            dropped.append(case.case_id)
        else:
            kept.append(case)
    return kept, dropped


def load_action_plans(oracle_root: Path, same_case_ids: Sequence[str]) -> Dict[str, Dict[str, List[ActionStep]]]:
    step_df = pd.read_csv(oracle_root / "raw" / "oracle_step_rows.csv")
    step_df = step_df[step_df["case_id"].isin(list(same_case_ids))].copy()
    plans: Dict[str, Dict[str, List[ActionStep]]] = {"frozen": {}, "oracle": {}}
    for arm_name, combo_col, label_value in [
        ("frozen", "frozen_combo_global", "frozen_pbest"),
        ("oracle", "oracle_combo_global", "oracle_combo_label"),
    ]:
        arm_rows: Dict[str, List[ActionStep]] = {}
        for case_id, group in step_df.groupby("case_id"):
            ordered = group.sort_values("round_index")
            case_steps = []
            for row in ordered.itertuples(index=False):
                case_steps.append(
                    ActionStep(
                        case_id=str(row.case_id),
                        scenario_id=int(row.scenario_id),
                        part_id=int(row.part_id),
                        round_index=int(row.round_index),
                        global_ids=parse_combo_field(getattr(row, combo_col)),
                        label=str(getattr(row, label_value) if label_value == "oracle_combo_label" else label_value),
                    )
                )
            arm_rows[str(case_id)] = case_steps
        plans[arm_name] = arm_rows
    return plans


def write_action_plan_artifacts(output_dir: Path, plans: Dict[str, Dict[str, List[ActionStep]]]) -> None:
    for arm_name, case_map in plans.items():
        rows = []
        for case_id, steps in sorted(case_map.items()):
            for step in steps:
                rows.append(
                    {
                        "case_id": case_id,
                        "scenario_id": step.scenario_id,
                        "part_id": step.part_id,
                        "round_index": step.round_index,
                        "global_ids": json.dumps(step.global_ids),
                        "label": step.label,
                    }
                )
        pd.DataFrame(rows).to_csv(output_dir / f"{arm_name}_action_plan.csv", index=False)


def load_same_cases(
    *,
    cfg_path: Path,
    cache_dir: Path,
    target_case_ids: Sequence[str],
) -> tuple[List[CaseRecord], Dict[str, Any]]:
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    life_support = payload.get("life_support")
    if isinstance(life_support, dict) and str(life_support.get("profile")) == "custom_direct_edit":
        payload["life_support"] = {k: v for k, v in life_support.items() if k != "profile"}
    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(payload)
    cfg.training.enable_eval = False
    cfg.training.train_only = True
    cfg.training.enable_wandb = False
    cfg.data.skip_lmdb = False
    cfg.data.max_samples = None
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.paths.cache_dir = str(cache_dir)
    train_loader, _, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        eval_batch_size=1,
        skip_lmdb=False,
        train_only=True,
    )
    dataset = train_loader.dataset
    assets = collect_dataset_assets(dataset)
    if assets.get("topology") is None:
        assets["topology"] = HydraulicTopology(cfg.paths.foundation_path)
    found: Dict[str, CaseRecord] = {}
    target_set = set(str(v) for v in target_case_ids)
    for dataset_idx in range(len(dataset)):
        data = dataset[dataset_idx]
        case_id, scenario_id, part_id = colon_case_id_from_data(data, "train", dataset_idx)
        if case_id not in target_set:
            continue
        found[case_id] = CaseRecord(
            case_id=case_id,
            scenario_id=scenario_id,
            part_id=part_id,
            dataset_index=int(dataset_idx),
            data=deepcopy(data),
        )
        if len(found) == len(target_set):
            break
    missing = sorted(target_set - set(found.keys()))
    if missing:
        raise RuntimeError(f"Could not locate all exact same-case IDs in current train split. Missing sample: {missing[:10]}")
    ordered = [found[str(case_id)] for case_id in sorted(target_set)]
    return ordered, assets


def build_reasoner_cfg(
    *,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_dir: Path,
    epochs: int,
    periodic_every: int,
    batch_size: int,
) -> Any:
    overrides = build_overfit_overrides(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_version="same_case_stronger_source_cfg",
        cache_dir=cache_dir,
        batch_size=batch_size,
        epochs=epochs,
        periodic_checkpoint_every=periodic_every,
        num_workers=0,
        prefetch_factor=2,
        max_samples=1,
        run_name="same_case_stronger_source_cfg",
    )
    return prepare_cfg(overrides, run_name="same_case_stronger_source_cfg", max_epochs=int(epochs), seed=45)


def build_state_input(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "valid_mask": state["valid_mask"],
        "evidence_state": state["evidence_state"],
        "observation_state": state["observation_state"],
        "constraint_state": state["constraint_state"],
        "nav_state_summary": state["graph_features"].view(1, -1),
    }


def score_reasoner_state(reasoner_module: torch.nn.Module, state: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    source_local_idx = resolve_source_local_idx(state["rollout"])
    valid_mask = state["valid_mask"].view(-1).bool()
    if source_local_idx is None or not bool(valid_mask[int(source_local_idx)].item()):
        return {
            "valid_case": False,
            "true_source_rank": None,
            "mrr": 0.0,
            "top1_hit": False,
            "top3_hit": False,
            "top5_hit": False,
            "softmax_candidate_count": int(valid_mask.sum().item()),
        }
    graph = TempGraph(state["edge_index"], int(valid_mask.numel()), device)
    state_input = move_payload(build_state_input(state), device)
    physics_ctx = move_payload(state["phys_ctx"].__dict__, device)
    with torch.no_grad():
        out = reasoner_module(state_input, graph, physics_ctx=physics_ctx)
    logits = out["logits"].detach().float().view(-1).cpu()
    safe_logits = logits.clone()
    safe_logits[~valid_mask.cpu()] = -float("inf")
    order = torch.argsort(safe_logits, descending=True)
    valid_order = order[torch.isfinite(safe_logits[order])]
    positions = (valid_order == int(source_local_idx)).nonzero(as_tuple=True)[0]
    if positions.numel() <= 0:
        return {
            "valid_case": False,
            "true_source_rank": None,
            "mrr": 0.0,
            "top1_hit": False,
            "top3_hit": False,
            "top5_hit": False,
            "softmax_candidate_count": int(valid_mask.sum().item()),
        }
    rank = int(positions.min().item()) + 1
    probs = torch.softmax(safe_logits[valid_order], dim=0) if valid_order.numel() > 0 else torch.empty(0)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-9))).sum().item()) if probs.numel() > 0 else 0.0
    return {
        "valid_case": True,
        "true_source_rank": int(rank),
        "mrr": 1.0 / float(rank),
        "top1_hit": bool(rank <= 1),
        "top3_hit": bool(rank <= 3),
        "top5_hit": bool(rank <= 5),
        "softmax_candidate_count": int(valid_order.numel()),
        "entropy": entropy,
        "max_prob": float(probs.max().item()) if probs.numel() > 0 else float("nan"),
    }


def translate_global_ids(rollout: PracticalRollout, global_ids: Sequence[int]) -> List[int]:
    global_to_local = {int(v): int(i) for i, v in enumerate(rollout.g_ids.tolist())}
    selected = []
    for gid in global_ids:
        if int(gid) not in global_to_local:
            raise RuntimeError(f"Planned global id {gid} is missing from current rollout candidate set.")
        selected.append(int(global_to_local[int(gid)]))
    deduped = []
    seen = set()
    for idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(int(idx))
    return deduped


def make_rollout_state(
    *,
    case: CaseRecord,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    num_episodes: int,
    action_budget: int,
    frontier_role_mode: str,
) -> Dict[str, Any]:
    obs_partial, _, _, _ = rollout.observe_current_state()
    state = build_state_bundle(
        rollout=rollout,
        history=history,
        env=env,
        topology=topology,
        num_episodes=int(num_episodes),
        action_budget=int(action_budget),
        frontier_role_mode=str(frontier_role_mode),
    )
    state["case_id"] = case.case_id
    state["scenario_id"] = case.scenario_id
    state["part_id"] = case.part_id
    state["rollout"] = rollout
    state["observation_state"] = obs_partial
    state["evidence_state"] = state["pack"]["evidence_state_mini"]
    return state


def write_bank_from_plan(
    *,
    arm_name: str,
    output_dir: Path,
    bank_root: Path,
    cases: Sequence[CaseRecord],
    action_plan: Dict[str, List[ActionStep]],
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
) -> Dict[str, Any]:
    env = CleanTwoChannelEvidenceEnv()
    topology = dataset_assets["topology"]
    bank_path = bank_root / f"{output_dir.name}_{arm_name}.lmdb"
    writer = SemiDynamicTrajectoryBankWriter(bank_path)
    bank_rows = []
    bank_start = time.perf_counter()
    sample_count = 0
    for case_idx, case in enumerate(cases):
        rollout = PracticalRollout(
            event_data=deepcopy(case.data),
            global_edge_index=dataset_assets["global_edge_index"],
            stt_dynamic_series=dataset_assets["stt_dynamic_series"],
            num_global_nodes=int(dataset_assets["num_global_nodes"]),
            num_episodes=int(num_episodes),
            samples_per_episode=int(action_budget),
            episode_duration_min=float(episode_duration_min),
        )
        history = ObservationWitnessHistory()
        steps = action_plan.get(case.case_id, [])
        for step in steps:
            state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=num_episodes,
                action_budget=action_budget,
                frontier_role_mode=frontier_role_mode,
            )
            source_local_idx = resolve_source_local_idx(rollout)
            if source_local_idx is None:
                raise RuntimeError(f"{arm_name} bank cannot resolve source local idx for {case.case_id} round {step.round_index}")
            state_input = build_state_input(state)
            payload = build_clean_aligned_feature_payload(
                state_input,
                batch_index=torch.zeros(int(state["valid_mask"].numel()), dtype=torch.long),
                edge_index=state["edge_index"].view(2, -1).long(),
                physics_ctx=state["phys_ctx"].__dict__,
                frontier_mode="unresolved_without_pair",
            )
            source_mask = torch.zeros_like(state["valid_mask"].view(-1).float())
            source_mask[int(source_local_idx)] = 1.0
            sample = build_bank_sample(
                node_features=payload["node_features"],
                edge_index=state["edge_index"].view(2, -1).long(),
                graph_features=payload["graph_features_by_graph"].view(1, -1)[0],
                valid_mask=payload["valid_mask"],
                source_mask=source_mask,
                case_id=int(case_idx),
                trajectory_id=0,
                step_id=int(step.round_index),
                budget_used=float(rollout.revealed_mask.sum().item()),
                t_sim_minutes=float(rollout.current_time_min),
            )
            writer.add(sample)
            sample_count += 1
            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"{arm_name}_slot_{idx}" for idx in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            bank_rows.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "part_id": case.part_id,
                    "round_index": int(step.round_index),
                    "selected_global_ids": json.dumps(step.global_ids),
                    "selected_local_ids": json.dumps(local_ids),
                }
            )
    writer.close()
    manifest = {
        "arm_name": arm_name,
        "lmdb_path": str(bank_path),
        "lmdb_size_bytes": int(path_size_bytes(bank_path)),
        "sample_count": int(sample_count),
        "case_count": int(len(cases)),
        "generation_wall_seconds": float(time.perf_counter() - bank_start),
        "stats": SemiDynamicBankStats(
            case_count=int(len(cases)),
            trajectory_count=int(len(cases)),
            sample_count=int(sample_count),
            total_steps=int(sample_count),
            total_selected=int(sum(len(json.loads(row["selected_local_ids"])) for row in bank_rows)),
            unique_signature_count=int(len(cases)),
        ).to_dict(),
        "bank_rows_csv": str(output_dir / arm_name / "trajectory_bank" / "bank_rows.csv"),
    }
    bank_dir = output_dir / arm_name / "trajectory_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bank_rows).to_csv(bank_dir / "bank_rows.csv", index=False)
    write_json(bank_dir / "trajectory_bank_manifest.json", manifest)
    return manifest


def build_arm_cfg(
    *,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_dir: Path,
    epochs: int,
    periodic_every: int,
    batch_size: int,
) -> Any:
    cfg = build_reasoner_cfg(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=epochs,
        periodic_every=periodic_every,
        batch_size=batch_size,
    )
    cfg.training.enable_eval = False
    cfg.training.train_only = True
    cfg.training.seed = 45
    return cfg


def evaluate_arm_checkpoint(
    *,
    checkpoint_path: Path,
    cfg: Any,
    cases: Sequence[CaseRecord],
    action_plan: Dict[str, List[ActionStep]],
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    env = CleanTwoChannelEvidenceEnv()
    topology = dataset_assets["topology"]
    model = ModelBuilder.build_model(cfg).to(device)
    model.load_state_dict(load_plain_state_dict(checkpoint_path), strict=True)
    model.eval()
    reasoner_module = getattr(model, "reasoner_module", model)
    raw_case_rows = []
    raw_step_rows = []
    for case in cases:
        rollout = PracticalRollout(
            event_data=deepcopy(case.data),
            global_edge_index=dataset_assets["global_edge_index"],
            stt_dynamic_series=dataset_assets["stt_dynamic_series"],
            num_global_nodes=int(dataset_assets["num_global_nodes"]),
            num_episodes=int(num_episodes),
            samples_per_episode=int(action_budget),
            episode_duration_min=float(episode_duration_min),
        )
        history = ObservationWitnessHistory()
        plan_steps = action_plan.get(case.case_id, [])
        first_success_episode = None
        for step in plan_steps:
            pre_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=num_episodes,
                action_budget=action_budget,
                frontier_role_mode=frontier_role_mode,
            )
            pre_metrics = score_reasoner_state(reasoner_module, pre_state, device)
            local_ids = translate_global_ids(rollout, step.global_ids)
            rollout.step_with_actions(local_ids, sample_types=[f"eval_slot_{idx}" for idx in range(len(local_ids))])
            if rollout.history_steps:
                history.append_from_history_step(rollout.history_steps[-1])
            post_state = make_rollout_state(
                case=case,
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=num_episodes,
                action_budget=action_budget,
                frontier_role_mode=frontier_role_mode,
            )
            post_metrics = score_reasoner_state(reasoner_module, post_state, device)
            if first_success_episode is None and bool(post_metrics.get("top1_hit")):
                first_success_episode = int(step.round_index)
            observed_count = int(rollout.revealed_mask.sum().item())
            candidate_total = int(pre_state["valid_mask"].numel())
            raw_step_rows.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id,
                    "part_id": case.part_id,
                    "episode_index": int(step.round_index) + 1,
                    "true_source_rank": post_metrics.get("true_source_rank"),
                    "top1_hit": post_metrics.get("top1_hit"),
                    "top3_hit": post_metrics.get("top3_hit"),
                    "top5_hit": post_metrics.get("top5_hit"),
                    "mrr": post_metrics.get("mrr"),
                    "logits_candidate_size": post_metrics.get("softmax_candidate_count"),
                    "pre_action_valid_size": int(pre_state["valid_mask"].sum().item()),
                    "post_action_valid_size": int(post_state["valid_mask"].sum().item()),
                    "unrevealed_candidate_ratio": float(post_metrics.get("softmax_candidate_count") or 0) / max(int(post_state["valid_mask"].sum().item()), 1),
                    "total_nodes": candidate_total,
                    "revealed_ratio": float(observed_count) / max(candidate_total, 1),
                    "observed_count": observed_count,
                    "fallback_triggered": False,
                }
            )
        final_state = make_rollout_state(
            case=case,
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=num_episodes,
            action_budget=action_budget,
            frontier_role_mode=frontier_role_mode,
        )
        final_metrics = score_reasoner_state(reasoner_module, final_state, device)
        final_observed = int(rollout.revealed_mask.sum().item())
        total_nodes = int(final_state["valid_mask"].numel())
        final_candidate_count = int(final_metrics.get("softmax_candidate_count") or 0)
        raw_case_rows.append(
            {
                "case_id": case.case_id,
                "scenario_id": case.scenario_id,
                "part_id": case.part_id,
                "success": bool(final_metrics.get("top1_hit")),
                "budget_used": float(final_observed),
                "episodes_completed": float(len(plan_steps)),
                "physical_time_mins": float(rollout.current_time_min),
                "first_success_episode": first_success_episode,
                "step_count_observed": int(len(plan_steps)),
                "final_top1_hit": final_metrics.get("top1_hit"),
                "final_top3_hit": final_metrics.get("top3_hit"),
                "final_top5_hit": final_metrics.get("top5_hit"),
                "final_mrr": final_metrics.get("mrr"),
                "final_true_source_rank": final_metrics.get("true_source_rank"),
                "final_entropy": final_metrics.get("entropy"),
                "final_max_prob": final_metrics.get("max_prob"),
                "final_logits_candidate_size": final_candidate_count,
                "final_pre_action_valid_size": int(final_state["valid_mask"].sum().item()),
                "final_post_action_valid_size": int(final_state["valid_mask"].sum().item()),
                "final_revealed_ratio": float(final_observed) / max(total_nodes, 1),
                "final_revealed_candidate_count": max(total_nodes - final_candidate_count, 0),
                "final_unrevealed_candidate_count": final_candidate_count,
                "final_unrevealed_candidate_ratio": float(final_candidate_count) / max(total_nodes, 1),
                "total_nodes": total_nodes,
                "final_confirmed_source_count": 0,
                "final_confirmed_non_source_count": final_observed,
                "final_no_resample_count": final_observed,
                "valid_case": bool(final_metrics.get("valid_case")),
            }
        )
    raw_case_df = pd.DataFrame(raw_case_rows).sort_values("case_id").reset_index(drop=True)
    raw_step_df = pd.DataFrame(raw_step_rows).sort_values(["case_id", "episode_index"]).reset_index(drop=True)
    std_step_df = standardize_step_df(raw_step_df, split="train")
    std_case_df = standardize_case_df(raw_case_df, std_step_df, split="train")
    valid = std_case_df[std_case_df["valid_case"] == True].copy()
    summary = {
        "case_count": int(len(std_case_df)),
        "valid_final_ranking_case_count": int(len(valid)),
        "top1_hit": float(valid["final_top1_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "top3_hit": float(valid["final_top3_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "top5_hit": float(valid["final_top5_hit"].astype(float).mean()) if len(valid) else float("nan"),
        "mrr_valid": float(valid["final_mrr"].mean()) if len(valid) else float("nan"),
        "true_source_rank_mean": float(valid["final_true_source_rank"].mean()) if len(valid) else float("nan"),
        "median_true_source_rank": float(valid["final_true_source_rank"].median()) if len(valid) else float("nan"),
    }
    return {
        "raw_case_df": raw_case_df,
        "raw_step_df": raw_step_df,
        "std_case_df": std_case_df,
        "std_step_df": std_step_df,
        "summary": summary,
    }


def summarize_train_metrics(metric_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(metric_rows)
    ranked = df.sort_values(
        ["mrr_valid", "top1_hit", "top3_hit", "top5_hit", "true_source_rank_mean"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0].to_dict() if len(ranked) else {}
    return {"best_by_train_mrr": best}


def run_arm(
    *,
    arm_name: str,
    output_dir: Path,
    bridge_package_dir: Path,
    init_checkpoint: Path,
    cache_dir: Path,
    epochs: int,
    periodic_every: int,
    batch_candidates: Sequence[int],
    offline_workers: int,
    offline_prefetch_factor: int,
    cases: Sequence[CaseRecord],
    action_plan: Dict[str, List[ActionStep]],
    dataset_assets: Dict[str, Any],
    num_episodes: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    arm_dir = output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    bank_root = preferred_bank_root(arm_dir)
    bank_manifest = write_bank_from_plan(
        arm_name=arm_name,
        output_dir=output_dir,
        bank_root=bank_root,
        cases=cases,
        action_plan=action_plan,
        dataset_assets=dataset_assets,
        num_episodes=num_episodes,
        action_budget=action_budget,
        episode_duration_min=episode_duration_min,
        frontier_role_mode=frontier_role_mode,
    )
    sample_count = int(bank_manifest["sample_count"])
    eligible_candidates = [int(v) for v in batch_candidates if int(v) * 2 <= max(sample_count, 1)]
    if not eligible_candidates:
        eligible_candidates = [max(1, min(sample_count, int(batch_candidates[-1])))]

    cfg = build_arm_cfg(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=epochs,
        periodic_every=periodic_every,
        batch_size=min(eligible_candidates),
    )
    batch_gate = run_batch_gate(
        bank_lmdb_path=Path(bank_manifest["lmdb_path"]),
        cfg=cfg,
        init_checkpoint=init_checkpoint,
        output_dir=arm_dir,
        candidate_batch_sizes=eligible_candidates,
        num_workers=int(offline_workers),
        prefetch_factor=int(offline_prefetch_factor),
        device=device,
    )
    chosen_batch_size = int(batch_gate["chosen_batch_size"])
    cfg = build_arm_cfg(
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=epochs,
        periodic_every=periodic_every,
        batch_size=chosen_batch_size,
    )
    train_result = train_offline_reasoner(
        bank_lmdb_path=Path(bank_manifest["lmdb_path"]),
        cfg=cfg,
        init_checkpoint=init_checkpoint,
        output_dir=arm_dir,
        epochs=int(epochs),
        batch_size=chosen_batch_size,
        num_workers=int(offline_workers),
        prefetch_factor=int(offline_prefetch_factor),
        use_amp=True,
        periodic_every=int(periodic_every),
        device=device,
    )
    run_dir = Path(train_result["run_dir"])
    history_rows = read_json(arm_dir / "train" / "train_loss_curve.json")
    train_loss_by_epoch = {int(row["epoch"]): safe_float(row.get("train_loss")) for row in history_rows if row.get("epoch") is not None}
    checkpoints = collect_candidate_checkpoints(run_dir, int(epochs), int(periodic_every))
    checkpoints.append({"epoch": int(epochs), "path": Path(train_result["best_model_state_path"]), "label": "best_train_loss"})
    seen_labels = set()
    metric_rows = []
    for checkpoint in checkpoints:
        if checkpoint["label"] in seen_labels:
            continue
        seen_labels.add(checkpoint["label"])
        checkpoint_path = init_checkpoint if checkpoint["path"] is None else Path(checkpoint["path"])
        evaluated = evaluate_arm_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            cases=cases,
            action_plan=action_plan,
            dataset_assets=dataset_assets,
            num_episodes=num_episodes,
            action_budget=action_budget,
            episode_duration_min=episode_duration_min,
            frontier_role_mode=frontier_role_mode,
            device=device,
        )
        epoch_dir = arm_dir / "train_eval" / checkpoint["label"]
        epoch_dir.mkdir(parents=True, exist_ok=True)
        evaluated["raw_case_df"].to_csv(epoch_dir / "raw_case_metrics.csv", index=False)
        evaluated["raw_step_df"].to_csv(epoch_dir / "raw_step_metrics.csv", index=False)
        evaluated["std_case_df"].to_csv(epoch_dir / "standardized_case_metrics.csv", index=False)
        evaluated["std_step_df"].to_csv(epoch_dir / "standardized_step_metrics.csv", index=False)
        row = {
            "epoch": int(checkpoint["epoch"]),
            "label": checkpoint["label"],
            "checkpoint_path": str(checkpoint_path),
            "train_loss": None if checkpoint["label"] == "init" else train_loss_by_epoch.get(int(checkpoint["epoch"])),
            **evaluated["summary"],
        }
        metric_rows.append(row)
        write_json(epoch_dir / "summary.json", row)
    metric_df = pd.DataFrame(metric_rows).sort_values(["epoch", "label"]).reset_index(drop=True)
    metric_df.to_csv(arm_dir / "train_eval" / "train_checkpoint_metrics.csv", index=False)
    write_json(arm_dir / "train_eval" / "train_checkpoint_metrics.json", metric_rows)
    best_summary = summarize_train_metrics(metric_rows)
    write_json(arm_dir / "train_eval" / "best_checkpoint_summary.json", best_summary)
    run_manifest = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "arm_name": arm_name,
        "init_checkpoint": str(init_checkpoint),
        "bridge_package_dir": str(bridge_package_dir),
        "trajectory_bank_manifest_path": str(arm_dir / "trajectory_bank" / "trajectory_bank_manifest.json"),
        "batch_gate_summary_path": str(arm_dir / "batch_gate" / "batch_gate_summary.json"),
        "throughput_summary_path": str(arm_dir / "throughput_summary.json"),
        "train_eval_metrics_path": str(arm_dir / "train_eval" / "train_checkpoint_metrics.csv"),
        "run_dir": str(run_dir),
        "epochs": int(epochs),
        "periodic_checkpoint_every": int(periodic_every),
        "chosen_batch_size": int(chosen_batch_size),
        "offline_workers": int(offline_workers),
        "offline_prefetch_factor": int(offline_prefetch_factor),
        "seed": 45,
        "train_only": True,
        "val_used": False,
        "test_used": False,
        "case_count": int(len(cases)),
        "sample_count": int(sample_count),
        "action_source": "oracle_step_rows.csv exact same-case fixed action plan",
    }
    write_json(arm_dir / "run_manifest.json", run_manifest)
    return {"run_manifest": run_manifest, "metric_rows": metric_rows}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gpu_snapshot = check_gpu_exclusive()
    write_json(output_dir / "gpu_snapshot_before.json", gpu_snapshot)
    if gpu_snapshot.get("available") and gpu_snapshot.get("exclusive_ok") is False:
        raise RuntimeError(f"GPU is already occupied: {gpu_snapshot['processes']}")

    baseline_root = Path(args.baseline_root)
    oracle_root = Path(args.oracle_root)
    bridge_package_dir = Path(args.bridge_package_dir)
    init_checkpoint = Path(args.init_checkpoint)
    cache_dir = Path(args.cache_dir)

    same_case_manifest = build_same_case_manifest(baseline_root, oracle_root, output_dir)
    same_case_df = pd.read_csv(output_dir / "same_case_manifest.csv")
    same_case_ids = same_case_df["case_id"].astype(str).tolist()
    plans = load_action_plans(oracle_root, same_case_ids)
    write_action_plan_artifacts(output_dir, plans)

    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    cases, dataset_assets = load_same_cases(
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        target_case_ids=same_case_ids,
    )
    seed_meta = read_json(bridge_package_dir / "seed_metadata.json")
    control_bundle = load_control_bundle(Path(seed_meta["control_dir"]))
    nav_args = build_namespace_from_control_args(control_bundle["args"], "cpu")
    num_episodes = int(getattr(nav_args, "num_episodes"))
    action_budget = int(getattr(nav_args, "action_budget"))
    episode_duration_min = float(getattr(nav_args, "episode_duration_min"))
    frontier_role_mode = str(getattr(nav_args, "frontier_role_mode"))
    replayable_cases, dropped_case_ids = filter_replayable_cases(
        cases=cases,
        dataset_assets=dataset_assets,
        num_episodes=num_episodes,
        action_budget=action_budget,
        episode_duration_min=episode_duration_min,
        frontier_role_mode=frontier_role_mode,
    )
    if not replayable_cases:
        raise RuntimeError("No replayable exact same-case cases remained after source-resolution filtering.")
    replayable_ids = {case.case_id for case in replayable_cases}
    write_json(
        output_dir / "same_case_replayable_manifest.json",
        {
            "runner_version": RUNNER_VERSION,
            "exact_intersection_size": int(len(cases)),
            "replayable_exact_size": int(len(replayable_cases)),
            "dropped_nonreplayable_case_count": int(len(dropped_case_ids)),
            "dropped_nonreplayable_case_ids": dropped_case_ids,
            "replayable_manifest_csv": str(output_dir / "same_case_replayable_manifest.csv"),
        },
    )
    pd.DataFrame(
        [
            {"case_id": case.case_id, "scenario_id": case.scenario_id, "part_id": case.part_id, "dataset_index": case.dataset_index}
            for case in replayable_cases
        ]
    ).to_csv(output_dir / "same_case_replayable_manifest.csv", index=False)
    plans = {
        arm_name: {case_id: steps for case_id, steps in arm_map.items() if case_id in replayable_ids}
        for arm_name, arm_map in plans.items()
    }

    frozen_result = run_arm(
        arm_name="arm_a_frozen_pbest",
        output_dir=output_dir,
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=int(args.epochs),
        periodic_every=int(args.periodic_checkpoint_every),
        batch_candidates=[int(v) for v in args.batch_candidates],
        offline_workers=int(args.offline_workers),
        offline_prefetch_factor=int(args.offline_prefetch_factor),
        cases=replayable_cases,
        action_plan=plans["frozen"],
        dataset_assets=dataset_assets,
        num_episodes=num_episodes,
        action_budget=action_budget,
        episode_duration_min=episode_duration_min,
        frontier_role_mode=frontier_role_mode,
        device=device,
    )
    oracle_result = run_arm(
        arm_name="arm_b_task_defined_oracle",
        output_dir=output_dir,
        bridge_package_dir=bridge_package_dir,
        init_checkpoint=init_checkpoint,
        cache_dir=cache_dir,
        epochs=int(args.epochs),
        periodic_every=int(args.periodic_checkpoint_every),
        batch_candidates=[int(v) for v in args.batch_candidates],
        offline_workers=int(args.offline_workers),
        offline_prefetch_factor=int(args.offline_prefetch_factor),
        cases=replayable_cases,
        action_plan=plans["oracle"],
        dataset_assets=dataset_assets,
        num_episodes=num_episodes,
        action_budget=action_budget,
        episode_duration_min=episode_duration_min,
        frontier_role_mode=frontier_role_mode,
        device=device,
    )
    write_json(
        output_dir / "summary.json",
        {
            "runner_version": RUNNER_VERSION,
            "panel_version": PANEL_VERSION,
            "same_case_manifest_path": str(output_dir / "same_case_manifest.json"),
            "same_case_replayable_manifest_path": str(output_dir / "same_case_replayable_manifest.json"),
            "frozen_arm_manifest_path": str(output_dir / "arm_a_frozen_pbest" / "run_manifest.json"),
            "oracle_arm_manifest_path": str(output_dir / "arm_b_task_defined_oracle" / "run_manifest.json"),
            "same_case_manifest": same_case_manifest,
            "replayable_exact_case_count": int(len(replayable_cases)),
            "dropped_nonreplayable_case_ids": dropped_case_ids,
            "train_only": True,
            "val_used": False,
            "test_used": False,
            "seed": 45,
            "frozen_train_eval_rows": int(len(frozen_result["metric_rows"])),
            "oracle_train_eval_rows": int(len(oracle_result["metric_rows"])),
        },
    )


if __name__ == "__main__":
    main()
