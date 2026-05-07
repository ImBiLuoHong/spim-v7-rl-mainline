from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.two_channel_clean import (
    CleanTwoChannelEvidenceEnv,
    ObservationWitnessHistory,
    SupportScoreContext,
)
from src.modeling.navigators.clean_v1 import (
    CleanNavigatorV1,
    compute_mean_pairwise_jaccard_overlap,
    compute_mean_pairwise_overlap,
    compute_clean_transition_metrics,
    derive_two_channel_features,
    bound_nonnegative_score,
    masked_mean,
    pick_topk_valid,
    random_valid_pick,
)
from src.modeling.state.schema import ConstraintState
from src.scripts.audit.utils_practical_rollout import PracticalRollout


DEFAULT_CONFIG = "configs/evidence_v1/formal_campaign/official_clean_ref.yaml"
NODE_FEATURE_NAMES = [
    "support_score",
    "contradiction_score",
    "support_bounded",
    "contradiction_bounded",
    "live_plausibility",
    "conflict_mass",
    "ignorance_mass",
    "positive_anchor_potential",
    "safe_pair_potential",
    "positive_reachability",
    "safe_reachability",
    "positive_distance_summary",
    "safe_distance_summary",
    "pair_available",
    "eligible_safe_witness_count_bounded",
    "top_pair_margin_bounded",
    "feasible_mask",
    "sampled_mask",
    "no_resample_mask",
    "degree_norm",
    "valid_mask",
]
GRAPH_FEATURE_NAMES = [
    "episode_index_norm",
    "remaining_episodes_norm",
    "positive_witness_count_norm",
    "safe_witness_count_norm",
    "candidate_fraction",
    "current_time_norm",
]
REDUNDANCY_FEATURE_NAMES = [
    "positive_anchor_potential",
    "safe_pair_potential",
    "positive_reachability",
    "safe_reachability",
    "positive_distance_summary",
    "safe_distance_summary",
    "pair_available",
    "eligible_safe_witness_count_bounded",
    "top_pair_margin_bounded",
]
ROLE_POTENTIAL_NAMES = [
    "anchor_role_potential",
    "frontier_role_potential",
    "pair_role_potential",
]
GOVERNING_RELATIVE_FILES = [
    "src/scripts/diagnostics/run_clean_navigator_v1.py",
    "src/modeling/navigators/clean_v1.py",
    "src/data/v6/loader.py",
    "src/modeling/evidence/two_channel_clean.py",
    "src/scripts/audit/utils_practical_rollout.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Clean Navigator v1 on the clean mini evidence environment.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-train-cases", type=int, default=32)
    parser.add_argument("--max-val-cases", type=int, default=8)
    parser.add_argument("--max-test-cases", type=int, default=8)
    parser.add_argument("--train-pool-limit", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--action-budget", type=int, default=3)
    parser.add_argument("--episode-duration-min", type=float, default=45.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--optimizer-step-cases", type=int, default=1)
    parser.add_argument("--train-conflict-bonus-weight", type=float, default=0.0)
    parser.add_argument("--train-masked-negative-conflict-bonus-weight", type=float, default=0.0)
    parser.add_argument("--credit-mode", type=str, default="state_value", choices=["state_value", "action_set_q"])
    parser.add_argument(
        "--advantage-consumer-mode",
        type=str,
        default="canonical",
        choices=["canonical", "regime_masked_canonical_state_value", "regime_masked_alt_action_conditioned_ridge"],
    )
    parser.add_argument("--advantage-critic-path", type=str, default="")
    parser.add_argument("--advantage-mask-pair-active-threshold", type=float, default=0.05)
    parser.add_argument("--advantage-clip-min", type=float, default=None)
    parser.add_argument("--advantage-clip-max", type=float, default=None)
    parser.add_argument("--train-cases-per-epoch", type=int, default=0)
    parser.add_argument("--sampler-mode", type=str, default="uniform", choices=["uniform", "availability"])
    parser.add_argument("--availability-horizon", type=int, default=0)
    parser.add_argument("--availability-score-power", type=float, default=1.0)
    parser.add_argument("--availability-min-weight", type=float, default=0.25)
    parser.add_argument("--availability-focus-start", type=float, default=0.85)
    parser.add_argument("--availability-focus-end", type=float, default=0.35)
    parser.add_argument(
        "--frontier-role-mode",
        type=str,
        default="witness_pair_breadth",
        choices=["witness_pair_breadth", "legacy_distance_unresolved"],
    )
    parser.add_argument("--role-mode", type=str, default="none", choices=["none", "slot_bias"])
    parser.add_argument("--role-bias-weight", type=float, default=0.0)
    parser.add_argument(
        "--diversity-mode",
        type=str,
        default="none",
        choices=["none", "history_relation_overlap", "witness_pair_jaccard"],
    )
    parser.add_argument("--diversity-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--complementarity-mode",
        type=str,
        default="none",
        choices=["none", "role_aware_witness_pair_jaccard", "slot1_frontier_anchor_witness_pair_jaccard"],
    )
    parser.add_argument("--complementarity-penalty-weight", type=float, default=0.0)
    parser.add_argument("--skip-lmdb", action="store_true", default=True)
    parser.add_argument("--use-lmdb", action="store_true")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument(
        "--training-sampling-mode",
        type=str,
        default="seeded_case_generator",
        choices=["seeded_case_generator", "global_torch", "global_seeded_cpu"],
    )
    parser.add_argument("--trace-train-actions", action="store_true")
    parser.add_argument("--trace-max-train-cases", type=int, default=0)
    parser.add_argument("--trace-rng-states", action="store_true")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--init-checkpoint-strict", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/clean_navigator_v1",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    seed_everything(int(args.seed))
    settings: Dict[str, Any] = {
        "seed": int(args.seed),
        "device": str(device),
        "strict_determinism": bool(args.strict_determinism),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if bool(args.strict_determinism):
        if device.type == "cuda" and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            raise RuntimeError(
                "Strict determinism on CUDA requires CUBLAS_WORKSPACE_CONFIG to be set before process start."
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except RuntimeError:
            pass
    settings.update(
        {
            "torch_deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_tf32_allowed": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", True)),
            "cudnn_tf32_allowed": bool(getattr(torch.backends.cudnn, "allow_tf32", True)),
        }
    )
    return settings


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_tensor_state(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def global_rng_state_hashes(device: torch.device) -> Dict[str, Any]:
    hashes: Dict[str, Any] = {"cpu": hash_tensor_state(torch.get_rng_state())}
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        hashes["cuda_per_device"] = [hash_tensor_state(state) for state in cuda_states]
        hashes["cuda_combined"] = sha256_bytes(
            b"".join(state.detach().cpu().contiguous().numpy().tobytes() for state in cuda_states)
        )
    return hashes


def generator_state_hash(generator: Optional[torch.Generator]) -> Optional[str]:
    if generator is None:
        return None
    return hash_tensor_state(generator.get_state())


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cfg(config_path: str, skip_lmdb: bool) -> Config:
    cfg = Config(root_dir=str(PROJECT_ROOT))
    with open(PROJECT_ROOT / config_path, "r", encoding="utf-8") as handle:
        cfg.apply_overrides(yaml.safe_load(handle) or {})
    cfg.data.use_dataloader_v6 = True
    cfg.data.filter_no_source = True
    cfg.model.architecture = "phase4_5"
    cfg.data.skip_lmdb = bool(skip_lmdb)
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.paths.cache_dir = str(PROJECT_ROOT / "data" / "cache_lmdb")
    cfg.efficiency.batch_size = 1
    return cfg


def create_case_splits(
    cfg: Config,
    seed: int,
    limits: Dict[str, int],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Any, Dict[str, Any]]:
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        eval_batch_size=1,
        skip_lmdb=bool(cfg.data.skip_lmdb),
    )
    topology = None
    for loader in (train_loader, val_loader, test_loader):
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "topology") and dataset.topology is not None:
            topology = dataset.topology
            break
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "topology") and dataset.dataset.topology is not None:
            topology = dataset.dataset.topology
            break

    split_map = {"train": train_loader.dataset, "val": val_loader.dataset, "test": test_loader.dataset}
    asset_source_chain = []
    asset_source = train_loader.dataset
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

    global_edge_index = first_non_none_attr("global_edge_index")
    stt_dynamic_series = first_non_none_attr("stt_dynamic_series")
    num_global_nodes = first_non_none_attr("num_nodes")
    if num_global_nodes is None and topology is not None:
        num_global_nodes = getattr(topology, "num_nodes", None)
    if num_global_nodes is None and global_edge_index is not None:
        try:
            num_global_nodes = int(global_edge_index.max().item()) + 1
        except Exception:
            num_global_nodes = None
    dataset_assets = {
        "global_edge_index": global_edge_index,
        "stt_dynamic_series": stt_dynamic_series,
        "num_global_nodes": num_global_nodes,
    }
    rng = random.Random(seed)
    cases: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, dataset in split_map.items():
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        limit = min(int(limits[split_name]), len(indices))
        split_cases = []
        for dataset_idx in indices[:limit]:
            data = dataset[dataset_idx]
            split_cases.append(
                {
                    "split": split_name,
                    "dataset_index": int(dataset_idx),
                    "case_id": case_id_from_data(data, split_name, dataset_idx),
                    "data": deepcopy(data),
                }
            )
        cases[split_name] = split_cases
    if topology is None:
        topology = HydraulicTopology(cfg.paths.foundation_path)
    return cases, topology, dataset_assets


def case_id_from_data(data: Any, split_name: str, dataset_idx: int) -> str:
    scenario_id = getattr(data, "scenario_id", None)
    part_id = getattr(data, "part_id", None)
    scenario_val = int(scenario_id[0].item()) if isinstance(scenario_id, torch.Tensor) and scenario_id.numel() > 0 else scenario_id
    part_val = int(part_id[0].item()) if isinstance(part_id, torch.Tensor) and part_id.numel() > 0 else part_id
    if scenario_val is None:
        scenario_val = dataset_idx
    if part_val is None:
        return f"{split_name}_scenario{scenario_val}"
    return f"{split_name}_scenario{scenario_val}_part{part_val}"


def build_constraint_state(rollout: PracticalRollout) -> ConstraintState:
    sampled = rollout.revealed_mask.float()
    zeros = torch.zeros_like(sampled)
    return ConstraintState(
        confirmed_non_source_mask=zeros.clone(),
        confirmed_source_mask=zeros.clone(),
        sampled_mask=sampled.clone(),
        no_resample_mask=sampled.clone(),
    )


def compute_degree_norm(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    src, dst = edge_index
    degree = torch.bincount(src, minlength=num_nodes).float() + torch.bincount(dst, minlength=num_nodes).float()
    max_degree = degree.max().clamp_min(1.0)
    return (degree / max_degree).to(device=device, dtype=torch.float32)


def compute_history_relation_features(
    env: CleanTwoChannelEvidenceEnv,
    history: ObservationWitnessHistory,
    phys_ctx,
    num_nodes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    zeros = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    empty_signature = torch.zeros((num_nodes, 0), device=device, dtype=torch.float32)
    positives = history.positive_records()
    safes = history.safe_records()
    if not positives:
        pos_anchor = zeros.clone()
        pos_reach = zeros.clone()
        pos_dist = zeros.clone()
        pos_signature = empty_signature
        pos_arrivals = None
        pos_finite = None
        pos_conf = None
    else:
        pos_arrivals = env.physics_gate.stack_arrival_times(
            positives,
            num_nodes=num_nodes,
            device=device,
            phys_ctx_mode="current_time_physctx",
            current_phys_ctx=phys_ctx,
        )
        inf_thresh = float(env.physics_gate.reachability.infinity / 2)
        pos_finite = pos_arrivals < inf_thresh
        t_pos = torch.tensor([float(row.absolute_time_min) for row in positives], device=device, dtype=torch.float32)
        pos_conf = torch.tensor([float(row.confidence) for row in positives], device=device, dtype=torch.float32)
        anchor_raw = (t_pos.unsqueeze(0) - pos_arrivals) / env.tau_anchor_min
        anchor_strength = torch.sigmoid(anchor_raw) * pos_conf.unsqueeze(0)
        pos_signature = torch.where(pos_finite, anchor_strength, torch.zeros_like(anchor_strength))
        pos_anchor = torch.where(pos_finite, anchor_strength, torch.zeros_like(anchor_strength)).amax(dim=1)
        pos_reach = pos_finite.any(dim=1).float()
        pos_min_arrival = torch.where(pos_finite, pos_arrivals, torch.full_like(pos_arrivals, float("inf"))).amin(dim=1)
        pos_dist = torch.where(
            pos_reach > 0.5,
            torch.sigmoid((env.tau_anchor_min - pos_min_arrival) / env.tau_anchor_min),
            zeros.clone(),
        )

    if not safes:
        safe_pair = zeros.clone()
        safe_reach = zeros.clone()
        safe_dist = zeros.clone()
        safe_signature = empty_signature
    else:
        safe_arrivals = env.physics_gate.stack_arrival_times(
            safes,
            num_nodes=num_nodes,
            device=device,
            phys_ctx_mode="current_time_physctx",
            current_phys_ctx=phys_ctx,
        )
        inf_thresh = float(env.physics_gate.reachability.infinity / 2)
        safe_finite = safe_arrivals < inf_thresh
        t_safe = torch.tensor([float(row.absolute_time_min) for row in safes], device=device, dtype=torch.float32)
        safe_gate_raw = (t_safe.unsqueeze(0) - safe_arrivals + env.safe_gate_slack_min) / env.tau_safe_gate_min
        safe_gate = torch.sigmoid(safe_gate_raw)
        safe_signature = torch.where(safe_finite, safe_gate, torch.zeros_like(safe_gate))
        safe_reach = safe_finite.any(dim=1).float()
        safe_dist = torch.where(
            safe_reach > 0.5,
            torch.where(safe_finite, safe_gate, torch.zeros_like(safe_gate)).amax(dim=1),
            zeros.clone(),
        )
        if pos_arrivals is None or pos_finite is None or pos_conf is None:
            safe_pair = zeros.clone()
        else:
            pair_valid = safe_finite.unsqueeze(2) & pos_finite.unsqueeze(1)
            obs_margin = t_safe.view(1, -1, 1) - torch.tensor(
                [float(row.absolute_time_min) for row in positives],
                device=device,
                dtype=torch.float32,
            ).view(1, 1, -1)
            arrival_margin = safe_arrivals.unsqueeze(2) - pos_arrivals.unsqueeze(1)
            delta = obs_margin - arrival_margin
            anchor_raw = (
                torch.tensor([float(row.absolute_time_min) for row in positives], device=device, dtype=torch.float32).unsqueeze(0)
                - pos_arrivals
            ) / env.tau_anchor_min
            anchor_strength = torch.sigmoid(anchor_raw) * pos_conf.unsqueeze(0)
            margin_term = F.softplus(delta / env.tau_margin_min)
            pair_score = anchor_strength.unsqueeze(1) * safe_gate.unsqueeze(2) * margin_term
            pair_score = torch.where(pair_valid, pair_score, torch.zeros_like(pair_score))
            safe_pair = bound_nonnegative_score(pair_score.amax(dim=(1, 2)))
    return {
        "positive_anchor_potential": pos_anchor,
        "safe_pair_potential": safe_pair,
        "positive_reachability": pos_reach,
        "safe_reachability": safe_reach,
        "positive_distance_summary": pos_dist,
        "safe_distance_summary": safe_dist,
        "positive_witness_signature": pos_signature,
        "safe_witness_signature": safe_signature,
    }


def compute_pairwise_action_distance(
    env: CleanTwoChannelEvidenceEnv,
    phys_ctx,
    selected_indices: Sequence[int],
    num_nodes: int,
    device: torch.device,
) -> float:
    if len(selected_indices) < 2:
        return 0.0
    weights = env.physics_gate._resolve_distance_weights(phys_ctx).to(device)
    dists: List[float] = []
    for left, right in combinations(selected_indices, 2):
        seed = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        seed[int(left)] = 1.0
        dist = env.physics_gate.reachability.compute_distance(seed, phys_ctx, weights, num_nodes)
        value = dist[int(right)]
        if torch.isfinite(value):
            dists.append(float(value.item()))
    if not dists:
        return 0.0
    return float(sum(dists) / len(dists))


def compute_slot_role_potentials(
    relation: Dict[str, torch.Tensor],
    witness_pair_signature: torch.Tensor,
    derived: Optional[Dict[str, torch.Tensor]] = None,
    pair_available: Optional[torch.Tensor] = None,
    frontier_role_mode: str = "witness_pair_breadth",
) -> Dict[str, torch.Tensor]:
    anchor_role_potential = relation["positive_anchor_potential"].view(-1).float()
    pair_role_potential = relation["safe_pair_potential"].view(-1).float()
    if str(frontier_role_mode) == "legacy_distance_unresolved":
        if derived is None or pair_available is None:
            raise ValueError("legacy_distance_unresolved frontier role requires derived and pair_available inputs")
        unresolved_mass = derived["unresolved_mass"].view(-1).float()
        positive_distance = relation["positive_distance_summary"].view(-1).float()
        safe_distance = relation["safe_distance_summary"].view(-1).float()
        pair_available = pair_available.view(-1).float().clamp(0.0, 1.0)
        frontier_role_potential = (
            torch.sqrt((positive_distance * safe_distance).clamp_min(0.0))
            * unresolved_mass
            * (1.0 - pair_available)
        )
    elif witness_pair_signature.numel() == 0 or witness_pair_signature.size(1) == 0:
        frontier_role_potential = torch.zeros_like(anchor_role_potential)
    else:
        # Frontier/disambiguation tracks breadth over admissible safe-positive witness pairs,
        # distinct from the strongest single-pair score used by `safe_pair_potential`.
        frontier_role_potential = bound_nonnegative_score(
            (witness_pair_signature > 1e-6).float().sum(dim=1)
        )
    role_potentials = torch.stack(
        [
            anchor_role_potential,
            frontier_role_potential,
            pair_role_potential,
        ],
        dim=1,
    )
    return {
        "anchor_role_potential": anchor_role_potential,
        "frontier_role_potential": frontier_role_potential,
        "pair_role_potential": pair_role_potential,
        "role_potentials": role_potentials,
    }


def compute_witness_pair_stats(
    selected_indices: Sequence[int],
    witness_pair_signature: torch.Tensor,
) -> Dict[str, float]:
    if len(selected_indices) < 2 or witness_pair_signature.numel() == 0 or witness_pair_signature.size(1) == 0:
        return {
            "selected_witness_pair_overlap": 0.0,
            "selected_witness_pair_coverage_fraction": 0.0,
            "selected_witness_pair_union_count": 0.0,
            "selected_witness_pair_active_node_fraction": 0.0,
            "witness_pair_slot_collapse": 0.0,
        }
    selected_binary = (witness_pair_signature[selected_indices].float() > 1e-6).float()
    union_count = float((selected_binary.sum(dim=0) > 0.0).float().sum().item())
    total_dims = max(int(selected_binary.size(1)), 1)
    overlap = compute_mean_pairwise_jaccard_overlap(selected_indices, witness_pair_signature)
    active_node_fraction = float((selected_binary.sum(dim=1) > 0.0).float().mean().item())
    return {
        "selected_witness_pair_overlap": float(overlap),
        "selected_witness_pair_coverage_fraction": float(union_count / float(total_dims)),
        "selected_witness_pair_union_count": union_count,
        "selected_witness_pair_active_node_fraction": active_node_fraction,
        "witness_pair_slot_collapse": float(overlap >= 0.9),
    }


def build_state_bundle(
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    env: CleanTwoChannelEvidenceEnv,
    topology,
    num_episodes: int,
    action_budget: int,
    frontier_role_mode: str,
) -> Dict[str, Any]:
    obs_partial, _, phys_ctx, info = rollout.observe_current_state()
    constraint_state = build_constraint_state(rollout)
    support_context = SupportScoreContext(
        global_node_ids=rollout.g_ids,
        absolute_snapshot_idx=int(info["absolute_snapshot_idx"]),
        topology=topology,
    )
    pack = env.build_evidence_state_mini(
        observation_state=obs_partial,
        physics_context=phys_ctx,
        history=history,
        current_time_min=float(info["time_min"]),
        support_context=support_context,
        constraint_state=constraint_state,
        phys_ctx_mode="current_time_physctx",
    )
    support_score = pack["support_score"].view(-1).float()
    contradiction_score = pack["contradiction_score"].view(-1).float()
    derived = derive_two_channel_features(support_score, contradiction_score)
    contra_meta = pack["contradiction_meta"]
    valid_mask = pack["runtime_constraints"]["valid_mask"].view(-1).bool()
    feasible_mask = (
        phys_ctx.feasible_mask.view(-1).float()
        if phys_ctx.feasible_mask is not None
        else torch.ones_like(support_score)
    )
    sampled_mask = constraint_state.sampled_mask.view(-1).float()
    no_resample_mask = constraint_state.no_resample_mask.view(-1).float()
    degree_norm = compute_degree_norm(phys_ctx.edge_index, rollout.num_nodes, support_score.device)
    relation = compute_history_relation_features(
        env=env,
        history=history,
        phys_ctx=phys_ctx,
        num_nodes=rollout.num_nodes,
        device=support_score.device,
    )
    top_pair_margin_bounded = torch.sigmoid(contra_meta["top_pair_margin"].view(-1).float() / env.tau_margin_min)
    eligible_safe_bounded = bound_nonnegative_score(contra_meta["eligible_safe_witness_count"].view(-1).float())
    redundancy_signature = torch.cat(
        [
            relation["positive_witness_signature"],
            relation["safe_witness_signature"],
        ],
        dim=1,
    )
    pair_valid_tensor = contra_meta.get("pair_valid_tensor")
    if isinstance(pair_valid_tensor, torch.Tensor) and pair_valid_tensor.dim() == 3:
        witness_pair_signature = pair_valid_tensor.reshape(rollout.num_nodes, -1).float()
    else:
        witness_pair_signature = torch.zeros((rollout.num_nodes, 0), device=support_score.device, dtype=torch.float32)
    role_terms = compute_slot_role_potentials(
        relation=relation,
        witness_pair_signature=witness_pair_signature,
        derived=derived,
        pair_available=contra_meta["pair_available"].view(-1).float(),
        frontier_role_mode=frontier_role_mode,
    )
    node_features = torch.stack(
        [
            support_score,
            contradiction_score,
            derived["support_bounded"],
            derived["contradiction_bounded"],
            derived["live_plausibility"],
            derived["conflict_mass"],
            derived["ignorance_mass"],
            relation["positive_anchor_potential"],
            relation["safe_pair_potential"],
            relation["positive_reachability"],
            relation["safe_reachability"],
            relation["positive_distance_summary"],
            relation["safe_distance_summary"],
            contra_meta["pair_available"].view(-1).float(),
            eligible_safe_bounded,
            top_pair_margin_bounded,
            feasible_mask,
            sampled_mask,
            no_resample_mask,
            degree_norm,
            valid_mask.float(),
        ],
        dim=1,
    )
    positive_count = len(history.positive_records())
    safe_count = len(history.safe_records())
    graph_features = torch.tensor(
        [
            float(rollout.current_episode) / max(int(num_episodes), 1),
            float(max(int(num_episodes) - int(rollout.current_episode), 0)) / max(int(num_episodes), 1),
            float(positive_count) / max(float(rollout.num_nodes), 1.0),
            float(safe_count) / max(float(rollout.num_nodes), 1.0),
            float(valid_mask.float().mean().item()),
            float(info["time_min"]) / max(float(num_episodes) * float(rollout.episode_duration_min), 1.0),
        ],
        device=support_score.device,
        dtype=torch.float32,
    )
    return {
        "node_features": node_features,
        "graph_features": graph_features,
        "edge_index": phys_ctx.edge_index,
        "valid_mask": valid_mask,
        "support_score": support_score,
        "contradiction_score": contradiction_score,
        "pair_available": contra_meta["pair_available"].view(-1).float(),
        "eligible_safe_witness_count": contra_meta["eligible_safe_witness_count"].view(-1).float(),
        "top_pair_margin": contra_meta["top_pair_margin"].view(-1).float(),
        "redundancy_signature": redundancy_signature,
        "witness_pair_signature": witness_pair_signature,
        **role_terms,
        "constraint_state": constraint_state,
        "phys_ctx": phys_ctx,
        "info": info,
        "positive_count": positive_count,
        "safe_count": safe_count,
        "pair_budget": int(action_budget),
        "derived": derived,
        "pack": pack,
    }


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / max(len(values), 1))


def resolve_availability_horizon(args: argparse.Namespace) -> int:
    if int(args.availability_horizon) > 0:
        return max(1, min(int(args.num_episodes), int(args.availability_horizon)))
    return max(1, min(int(args.num_episodes), 6))


def select_probe_actions(
    valid_mask: torch.Tensor,
    concentration: torch.Tensor,
    action_budget: int,
) -> List[int]:
    valid_mask = valid_mask.view(-1).bool()
    concentration = concentration.view(-1).float()
    positive_idx = torch.nonzero(valid_mask & (concentration > 0.1), as_tuple=True)[0]
    safe_idx = torch.nonzero(valid_mask & (concentration <= 0.1), as_tuple=True)[0]

    selected: List[int] = []
    if positive_idx.numel() > 0:
        pos_scores = concentration[positive_idx]
        top_pos = positive_idx[torch.argsort(pos_scores, descending=True)]
        selected.append(int(top_pos[0].item()))

    if safe_idx.numel() > 0:
        safe_scores = concentration[safe_idx]
        ordered_safe = safe_idx[torch.argsort(safe_scores, descending=False)]
        for idx in ordered_safe.tolist():
            idx_int = int(idx)
            if idx_int in selected:
                continue
            selected.append(idx_int)
            if len(selected) >= int(action_budget):
                break

    if len(selected) < int(action_budget):
        fallback_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
        for idx in fallback_idx.tolist():
            idx_int = int(idx)
            if idx_int in selected:
                continue
            selected.append(idx_int)
            if len(selected) >= int(action_budget):
                break
    return selected[: int(action_budget)]


def probe_case_availability(
    case: Dict[str, Any],
    env: CleanTwoChannelEvidenceEnv,
    topology,
    dataset_assets: Dict[str, Any],
    availability_horizon: int,
    action_budget: int,
    episode_duration_min: float,
    frontier_role_mode: str,
) -> Dict[str, Any]:
    event_data = deepcopy(case["data"])
    rollout = PracticalRollout(
        event_data=event_data,
        global_edge_index=dataset_assets["global_edge_index"],
        stt_dynamic_series=dataset_assets["stt_dynamic_series"],
        num_global_nodes=int(dataset_assets["num_global_nodes"]),
        num_episodes=int(availability_horizon),
        samples_per_episode=int(action_budget),
        episode_duration_min=float(episode_duration_min),
    )
    history = ObservationWitnessHistory()
    total_reward = 0.0
    total_unresolved_delta = 0.0
    positive_episode_count = 0
    max_positive_nodes_visible = 0
    probe_positive_count = 0
    probe_pair_available_episode_count = 0
    probe_max_pair_available_mean = 0.0
    probe_max_pair_available_any = 0.0
    insufficient_valid_episodes = 0

    for _episode_idx in range(int(availability_horizon)):
        pre_state = build_state_bundle(
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=int(availability_horizon),
            action_budget=int(action_budget),
            frontier_role_mode=str(frontier_role_mode),
        )
        valid_mask = pre_state["valid_mask"].view(-1).bool()
        if int(valid_mask.sum().item()) <= 0:
            insufficient_valid_episodes += 1
            break

        t_snapshot_idx = int(pre_state["info"]["t_snapshot_idx"])
        concentration = event_data.x_raw[:, t_snapshot_idx, 1].view(-1).float()
        positive_visible = int(((concentration > 0.1) & valid_mask).sum().item())
        if positive_visible > 0:
            positive_episode_count += 1
        max_positive_nodes_visible = max(max_positive_nodes_visible, positive_visible)

        selected_indices = select_probe_actions(
            valid_mask=valid_mask,
            concentration=concentration,
            action_budget=int(action_budget),
        )
        if not selected_indices:
            insufficient_valid_episodes += 1
            break
        rollout.step_with_actions(selected_indices, sample_types=[f"probe_slot_{idx}" for idx in range(len(selected_indices))])
        if rollout.history_steps:
            prior_records = len(history.records)
            history.append_from_history_step(rollout.history_steps[-1])
            probe_positive_count += sum(1 for row in history.records[prior_records:] if row.label == "positive")

        post_state = build_state_bundle(
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=int(availability_horizon),
            action_budget=int(action_budget),
            frontier_role_mode=str(frontier_role_mode),
        )
        transition = compute_clean_transition_metrics(pre_state, post_state)
        total_reward += float(transition["reward"])
        total_unresolved_delta += float(transition["unresolved_delta"])

        pair_available = post_state["pair_available"][post_state["valid_mask"]]
        if pair_available.numel() > 0:
            pair_mean = float(pair_available.mean().item())
            pair_any = float(pair_available.max().item())
            if bool((pair_available > 0).any().item()):
                probe_pair_available_episode_count += 1
            probe_max_pair_available_mean = max(probe_max_pair_available_mean, pair_mean)
            probe_max_pair_available_any = max(probe_max_pair_available_any, pair_any)

    probe_positive_hit = float(probe_positive_count > 0)
    probe_any_pair_available = float(probe_pair_available_episode_count > 0)
    positive_episode_fraction = float(positive_episode_count / max(int(availability_horizon), 1))
    probe_pair_available_fraction = float(probe_pair_available_episode_count / max(int(availability_horizon), 1))
    probe_transition_hit = float((abs(total_reward) > 1e-8) or (abs(total_unresolved_delta) > 1e-8))
    reward_signal = min(max(float(total_reward), 0.0), 1.0)
    unresolved_signal = min(max(float(total_unresolved_delta), 0.0), 1.0)
    informative_score_raw = (
        probe_positive_hit
        + positive_episode_fraction
        + probe_any_pair_available
        + probe_pair_available_fraction
        + 0.5 * probe_transition_hit
        + reward_signal
        + unresolved_signal
        + min(probe_max_pair_available_mean, 1.0)
    )
    informative_case = float(
        probe_positive_hit > 0.0
        or probe_any_pair_available > 0.0
        or probe_transition_hit > 0.0
    )
    zero_signal_case = float(informative_case <= 0.0)
    return {
        "split": case["split"],
        "case_id": case["case_id"],
        "dataset_index": int(case["dataset_index"]),
        "availability_horizon": int(availability_horizon),
        "schedule_only_oracle_probe": 1.0,
        "positive_episode_count": int(positive_episode_count),
        "positive_episode_fraction": float(positive_episode_fraction),
        "max_positive_nodes_visible": int(max_positive_nodes_visible),
        "probe_positive_hit": float(probe_positive_hit),
        "probe_positive_count": int(probe_positive_count),
        "probe_any_pair_available": float(probe_any_pair_available),
        "probe_pair_available_episode_count": int(probe_pair_available_episode_count),
        "probe_pair_available_fraction": float(probe_pair_available_fraction),
        "probe_max_pair_available_mean": float(probe_max_pair_available_mean),
        "probe_max_pair_available_any": float(probe_max_pair_available_any),
        "probe_reward_total": float(total_reward),
        "probe_unresolved_delta_total": float(total_unresolved_delta),
        "probe_transition_hit": float(probe_transition_hit),
        "informative_case": float(informative_case),
        "zero_signal_case": float(zero_signal_case),
        "informative_score_raw": float(informative_score_raw),
        "insufficient_valid_episodes": int(insufficient_valid_episodes),
    }


def summarise_availability_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"case_count": 0}
    return {
        "case_count": int(len(rows)),
        "informative_case_fraction": safe_mean([row["informative_case"] for row in rows]),
        "zero_signal_case_fraction": safe_mean([row["zero_signal_case"] for row in rows]),
        "probe_positive_hit_fraction": safe_mean([row["probe_positive_hit"] for row in rows]),
        "probe_pair_available_fraction": safe_mean([row["probe_any_pair_available"] for row in rows]),
        "probe_transition_hit_fraction": safe_mean([row["probe_transition_hit"] for row in rows]),
        "positive_episode_fraction_mean": safe_mean([row["positive_episode_fraction"] for row in rows]),
        "probe_pair_available_episode_fraction_mean": safe_mean([row["probe_pair_available_fraction"] for row in rows]),
        "probe_max_pair_available_mean": safe_mean([row["probe_max_pair_available_mean"] for row in rows]),
        "probe_reward_total_mean": safe_mean([row["probe_reward_total"] for row in rows]),
        "probe_unresolved_delta_total_mean": safe_mean([row["probe_unresolved_delta_total"] for row in rows]),
        "informative_score_raw_mean": safe_mean([row["informative_score_raw"] for row in rows]),
    }


def build_availability_rows(
    cases: Dict[str, List[Dict[str, Any]]],
    env: CleanTwoChannelEvidenceEnv,
    topology,
    dataset_assets: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    availability_horizon = resolve_availability_horizon(args)
    rows: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        for case in cases.get(split_name, []):
            rows.append(
                probe_case_availability(
                    case=case,
                    env=env,
                    topology=topology,
                    dataset_assets=dataset_assets,
                    availability_horizon=int(availability_horizon),
                    action_budget=int(args.action_budget),
                    episode_duration_min=float(args.episode_duration_min),
                    frontier_role_mode=str(args.frontier_role_mode),
                )
            )

    train_rows = [row for row in rows if row["split"] == "train"]
    max_raw_score = max((float(row["informative_score_raw"]) for row in train_rows), default=1.0)
    max_raw_score = max(max_raw_score, 1e-6)
    row_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        score_norm = float(row["informative_score_raw"]) / max_raw_score
        row["informative_score_norm"] = float(score_norm)
        row["sampling_weight"] = float(
            max(float(args.availability_min_weight), 0.0)
            + max(score_norm, 0.0) ** max(float(args.availability_score_power), 1e-6)
        )
        row_map[row["case_id"]] = row

    summary = {
        "mode": "offline_schedule_only_oracle_probe",
        "availability_horizon": int(availability_horizon),
        "action_budget": int(args.action_budget),
        "episode_duration_min": float(args.episode_duration_min),
        "train_score_max": float(max_raw_score),
        "splits": {
            split_name: summarise_availability_rows([row for row in rows if row["split"] == split_name])
            for split_name in ("train", "val", "test")
        },
    }
    return rows, row_map, summary


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_jsonable(row))


def clamp_unit_interval(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def sample_train_cases(
    train_cases: Sequence[Dict[str, Any]],
    availability_by_case: Optional[Dict[str, Dict[str, Any]]],
    args: argparse.Namespace,
    epoch_index: int,
    epoch_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not train_cases:
        return [], {"sampler_mode": str(args.sampler_mode), "train_pool_case_count": 0, "sampled_case_count": 0}

    cases_per_epoch = len(train_cases)
    if int(args.train_cases_per_epoch) > 0:
        cases_per_epoch = max(1, int(args.train_cases_per_epoch))

    rng = np.random.default_rng(int(epoch_seed))
    if str(args.sampler_mode) == "availability" and availability_by_case:
        uniform_weights = np.ones(len(train_cases), dtype=np.float64)
        avail_weights = np.asarray(
            [max(float(availability_by_case[case["case_id"]]["sampling_weight"]), 1e-6) for case in train_cases],
            dtype=np.float64,
        )
        if len(train_cases) <= 1 or int(args.epochs) <= 1:
            focus = float(args.availability_focus_start)
        else:
            progress = float(max(int(epoch_index) - 1, 0)) / float(max(int(args.epochs) - 1, 1))
            focus = float(args.availability_focus_start) + (
                float(args.availability_focus_end) - float(args.availability_focus_start)
            ) * progress
        focus = clamp_unit_interval(focus)
        effective_weights = focus * avail_weights + (1.0 - focus) * uniform_weights
        probs = effective_weights / effective_weights.sum()
        selected_indices = rng.choice(
            len(train_cases),
            size=max(cases_per_epoch, 1),
            replace=True,
            p=probs,
        ).tolist()
        selected_cases = [train_cases[int(idx)] for idx in selected_indices]
        unique_case_fraction = float(len(set(int(idx) for idx in selected_indices)) / max(len(selected_indices), 1))
        sampler_info = {
            "sampler_mode": "availability",
            "availability_focus": float(focus),
            "train_pool_case_count": int(len(train_cases)),
            "sampled_case_count": int(len(selected_cases)),
            "replacement_enabled": 1.0,
            "sampled_unique_case_fraction": float(unique_case_fraction),
            "sampled_repeat_case_fraction": float(1.0 - unique_case_fraction),
        }
        return selected_cases, sampler_info

    selected_indices = rng.permutation(len(train_cases))[:cases_per_epoch].tolist()
    selected_cases = [train_cases[int(idx)] for idx in selected_indices]
    sampler_info = {
        "sampler_mode": "uniform",
        "availability_focus": 0.0,
        "train_pool_case_count": int(len(train_cases)),
        "sampled_case_count": int(len(selected_cases)),
        "replacement_enabled": 0.0,
        "sampled_unique_case_fraction": float(len(set(int(idx) for idx in selected_indices)) / max(len(selected_indices), 1)),
        "sampled_repeat_case_fraction": 0.0,
    }
    return selected_cases, sampler_info


def summarise_sampler_exposure(
    selected_cases: Sequence[Dict[str, Any]],
    availability_by_case: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    if not selected_cases:
        return {
            "sampled_case_count": 0,
            "informative_case_fraction": 0.0,
            "zero_signal_case_fraction": 0.0,
            "probe_positive_hit_fraction": 0.0,
            "probe_pair_available_fraction": 0.0,
            "probe_transition_hit_fraction": 0.0,
            "informative_score_norm_mean": 0.0,
            "sampling_weight_mean": 0.0,
        }
    if not availability_by_case:
        return {"sampled_case_count": int(len(selected_cases))}
    rows = [availability_by_case[case["case_id"]] for case in selected_cases]
    return {
        "sampled_case_count": int(len(selected_cases)),
        "informative_case_fraction": safe_mean([row["informative_case"] for row in rows]),
        "zero_signal_case_fraction": safe_mean([row["zero_signal_case"] for row in rows]),
        "probe_positive_hit_fraction": safe_mean([row["probe_positive_hit"] for row in rows]),
        "probe_pair_available_fraction": safe_mean([row["probe_any_pair_available"] for row in rows]),
        "probe_transition_hit_fraction": safe_mean([row["probe_transition_hit"] for row in rows]),
        "informative_score_norm_mean": safe_mean([row["informative_score_norm"] for row in rows]),
        "sampling_weight_mean": safe_mean([row["sampling_weight"] for row in rows]),
        "unique_case_fraction": float(len({case["case_id"] for case in selected_cases}) / max(len(selected_cases), 1)),
    }


def build_sampler_preview(
    train_cases: Sequence[Dict[str, Any]],
    availability_by_case: Optional[Dict[str, Dict[str, Any]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    uniform_args = deepcopy(args)
    uniform_args.sampler_mode = "uniform"
    availability_args = deepcopy(args)
    availability_args.sampler_mode = "availability"

    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        epoch_seed = int(args.seed) + int(epoch)
        uniform_cases, uniform_info = sample_train_cases(
            train_cases=train_cases,
            availability_by_case=availability_by_case,
            args=uniform_args,
            epoch_index=int(epoch),
            epoch_seed=int(epoch_seed),
        )
        availability_cases, availability_info = sample_train_cases(
            train_cases=train_cases,
            availability_by_case=availability_by_case,
            args=availability_args,
            epoch_index=int(epoch),
            epoch_seed=int(epoch_seed),
        )
        uniform_exposure = summarise_sampler_exposure(uniform_cases, availability_by_case)
        availability_exposure = summarise_sampler_exposure(availability_cases, availability_by_case)
        epoch_rows.append(
            {
                "epoch": int(epoch),
                "uniform_sampler": {**uniform_info, **uniform_exposure},
                "availability_sampler": {**availability_info, **availability_exposure},
            }
        )

    def aggregate(prefix: str) -> Dict[str, float]:
        if not epoch_rows:
            return {}
        keys = [
            "informative_case_fraction",
            "zero_signal_case_fraction",
            "probe_positive_hit_fraction",
            "probe_pair_available_fraction",
            "probe_transition_hit_fraction",
            "informative_score_norm_mean",
            "sampling_weight_mean",
            "unique_case_fraction",
        ]
        out: Dict[str, float] = {}
        for key in keys:
            out[key] = safe_mean([float(row[prefix].get(key, 0.0)) for row in epoch_rows])
        return out

    return {
        "train_pool_case_count": int(len(train_cases)),
        "epochs": int(args.epochs),
        "uniform_sampler_mean": aggregate("uniform_sampler"),
        "availability_sampler_mean": aggregate("availability_sampler"),
        "epoch_rows": epoch_rows,
    }


def build_slot_sampling_trace(
    *,
    pre_state: Dict[str, Any],
    model_out: Optional[Dict[str, torch.Tensor]],
    selected_indices: Sequence[int],
    rollout: PracticalRollout,
) -> List[Dict[str, Any]]:
    if model_out is None:
        return []
    slot_logits = model_out.get("slot_logits") or []
    available = pre_state["valid_mask"].view(-1).bool().detach().cpu().clone()
    slot_rows: List[Dict[str, Any]] = []
    for slot_idx, chosen_idx in enumerate(selected_indices):
        if slot_idx >= len(slot_logits):
            break
        candidate_idx = torch.nonzero(available, as_tuple=True)[0]
        logits = slot_logits[slot_idx].detach().cpu().view(-1).float()
        candidate_logits = logits[candidate_idx]
        candidate_probs = torch.softmax(candidate_logits, dim=0)
        candidate_idx_list = [int(idx) for idx in candidate_idx.tolist()]
        chosen_local_idx = candidate_idx_list.index(int(chosen_idx))
        slot_rows.append(
            {
                "slot_idx": int(slot_idx),
                "candidate_count": int(len(candidate_idx_list)),
                "candidate_local_indices": candidate_idx_list,
                "candidate_global_ids": [int(rollout.g_ids[int(idx)].item()) for idx in candidate_idx_list],
                "candidate_logits": [float(val) for val in candidate_logits.tolist()],
                "candidate_probs": [float(val) for val in candidate_probs.tolist()],
                "chosen_local_idx": int(chosen_local_idx),
                "chosen_index": int(chosen_idx),
                "chosen_global_id": int(rollout.g_ids[int(chosen_idx)].item()),
                "chosen_prob": float(candidate_probs[int(chosen_local_idx)].item()),
            }
        )
        available[int(chosen_idx)] = False
    return slot_rows


def select_action(
    policy_name: str,
    state: Dict[str, Any],
    model: Optional[CleanNavigatorV1],
    generator: torch.Generator,
    cpu_generator: Optional[torch.Generator],
    training_sampling_mode: str,
    deterministic: bool,
) -> Tuple[List[int], Optional[Dict[str, torch.Tensor]]]:
    valid_mask = state["valid_mask"]
    budget = int(state["pair_budget"])
    if policy_name == "random_valid":
        return random_valid_pick(valid_mask, budget, cpu_generator if cpu_generator is not None else generator), None
    if policy_name == "top_support":
        return pick_topk_valid(state["support_score"], valid_mask, budget), None
    if policy_name == "top_safe_pair_potential":
        scores = state["node_features"][:, NODE_FEATURE_NAMES.index("safe_pair_potential")]
        return pick_topk_valid(scores, valid_mask, budget), None
    if model is None:
        raise ValueError(f"Model is required for policy_name={policy_name}")
    model_device = next(model.parameters()).device
    redundancy_signature = state["redundancy_signature"]
    if getattr(model, "diversity_mode", "none") == "witness_pair_jaccard":
        redundancy_signature = state["witness_pair_signature"]
    complementarity_signature = None
    if getattr(model, "complementarity_mode", "none") in {
        "role_aware_witness_pair_jaccard",
        "slot1_frontier_anchor_witness_pair_jaccard",
    }:
        complementarity_signature = state["witness_pair_signature"]
    model_out = model.act(
        node_features=state["node_features"].to(model_device),
        edge_index=state["edge_index"].to(model_device),
        valid_mask=state["valid_mask"].to(model_device),
        graph_features=state["graph_features"].to(model_device),
        deterministic=deterministic,
        redundancy_features=redundancy_signature.to(model_device),
        complementarity_features=(
            complementarity_signature.to(model_device) if complementarity_signature is not None else None
        ),
        role_potentials=state["role_potentials"].to(model_device),
        generator=(
            None
            if str(training_sampling_mode) == "global_torch"
            else (cpu_generator if cpu_generator is not None else generator)
        ),
        training_sampling_mode=(
            "legacy_categorical"
            if str(training_sampling_mode) == "global_torch"
            else "explicit_cpu_multinomial"
        ),
    )
    return model_out["selected_indices"].detach().cpu().tolist(), model_out


def resolve_source_local_idx(rollout: PracticalRollout) -> Optional[int]:
    src_global = getattr(rollout.event_data, "global_injection_node", None)
    if src_global is None:
        return None
    if isinstance(src_global, torch.Tensor):
        src_global = int(src_global.view(-1)[0].item())
    matches = (rollout.g_ids == int(src_global)).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


def run_case_rollout(
    case: Dict[str, Any],
    policy_name: str,
    env: CleanTwoChannelEvidenceEnv,
    topology,
    dataset_assets: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    model: Optional[CleanNavigatorV1] = None,
    generator: Optional[torch.Generator] = None,
    cpu_generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    trace_rows: Optional[List[Dict[str, Any]]] = None,
    trace_context: Optional[Dict[str, Any]] = None,
    buffer_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    event_data = deepcopy(case["data"])
    rollout = PracticalRollout(
        event_data=event_data,
        global_edge_index=dataset_assets["global_edge_index"],
        stt_dynamic_series=dataset_assets["stt_dynamic_series"],
        num_global_nodes=int(dataset_assets["num_global_nodes"]),
        num_episodes=int(args.num_episodes),
        samples_per_episode=int(args.action_budget),
        episode_duration_min=float(args.episode_duration_min),
    )
    history = ObservationWitnessHistory()
    src_local = resolve_source_local_idx(rollout)
    gen = generator if generator is not None else torch.Generator(device="cpu")
    step_rows: List[Dict[str, Any]] = []
    train_tensors: List[Dict[str, torch.Tensor]] = []
    totals = {
        "reward_total": 0.0,
        "ignorance_delta_total": 0.0,
        "conflict_delta_total": 0.0,
        "pair_delta_total": 0.0,
        "live_delta_total": 0.0,
        "support_delta_total": 0.0,
        "unresolved_delta_total": 0.0,
        "action_valid_count": 0.0,
        "exact_k_count": 0.0,
        "unique_count": 0.0,
        "pairwise_distance_total": 0.0,
        "evidence_overlap_total": 0.0,
        "witness_pair_overlap_total": 0.0,
        "witness_pair_coverage_total": 0.0,
        "witness_pair_union_total": 0.0,
        "witness_pair_active_node_total": 0.0,
        "anchor_slot_alignment_total": 0.0,
        "frontier_slot_alignment_total": 0.0,
        "pair_slot_alignment_total": 0.0,
        "role_alignment_total": 0.0,
        "designated_anchor_total": 0.0,
        "designated_frontier_total": 0.0,
        "designated_pair_total": 0.0,
        "step_count": 0,
        "budget_used": 0.0,
        "source_sampled_count": 0.0,
        "insufficient_valid_episodes": 0.0,
        "unique_global_count": 0.0,
        "slot_collapse_count": 0.0,
        "witness_pair_slot_collapse_count": 0.0,
    }
    initial_state = build_state_bundle(
        rollout=rollout,
        history=history,
        env=env,
        topology=topology,
        num_episodes=int(args.num_episodes),
        action_budget=int(args.action_budget),
        frontier_role_mode=str(args.frontier_role_mode),
    )
    initial_unresolved = float(initial_state["derived"]["unresolved_mass"][initial_state["valid_mask"]].mean().item()) if bool(initial_state["valid_mask"].any()) else 0.0
    for episode_idx in range(int(args.num_episodes)):
        pre_state = build_state_bundle(
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=int(args.num_episodes),
            action_budget=int(args.action_budget),
            frontier_role_mode=str(args.frontier_role_mode),
        )
        if int(pre_state["valid_mask"].sum().item()) < int(args.action_budget):
            totals["insufficient_valid_episodes"] += 1.0
            break
        rng_before = (
            {
                "global": global_rng_state_hashes(device),
                "generator": generator_state_hash(gen),
                "cpu_generator": generator_state_hash(cpu_generator),
            }
            if bool(args.trace_rng_states)
            else None
        )
        selected_indices, model_out = select_action(
            policy_name=policy_name,
            state=pre_state,
            model=model,
            generator=gen,
            cpu_generator=cpu_generator,
            training_sampling_mode=str(args.training_sampling_mode),
            deterministic=deterministic,
        )
        rng_after = (
            {
                "global": global_rng_state_hashes(device),
                "generator": generator_state_hash(gen),
                "cpu_generator": generator_state_hash(cpu_generator),
            }
            if bool(args.trace_rng_states)
            else None
        )
        selected_indices = [int(idx) for idx in selected_indices]
        exact_k = float(len(selected_indices) == int(args.action_budget))
        unique_sel = float(len(set(selected_indices)) == len(selected_indices))
        valid_sel = float(all(bool(pre_state["valid_mask"][idx].item()) for idx in selected_indices))
        source_sampled = float(src_local is not None and any(int(idx) == int(src_local) for idx in selected_indices))
        selected_global_ids = [int(rollout.g_ids[idx].item()) for idx in selected_indices]
        if trace_rows is not None:
            trace_rows.append(
                {
                    **(trace_context or {}),
                    "policy_name": str(policy_name),
                    "training_sampling_mode": str(args.training_sampling_mode),
                    "deterministic": bool(deterministic),
                    "episode_idx": int(episode_idx),
                    "selected_indices": [int(idx) for idx in selected_indices],
                    "selected_global_ids": [int(idx) for idx in selected_global_ids],
                    "source_local_idx": None if src_local is None else int(src_local),
                    "source_sampled": float(source_sampled),
                    "rng_before_select": rng_before,
                    "rng_after_select": rng_after,
                    "slot_sampling": build_slot_sampling_trace(
                        pre_state=pre_state,
                        model_out=model_out,
                        selected_indices=selected_indices,
                        rollout=rollout,
                    ),
                }
            )
        unique_global_sel = float(len(set(selected_global_ids)) == len(selected_global_ids))
        pairwise_distance = compute_pairwise_action_distance(
            env=env,
            phys_ctx=pre_state["phys_ctx"],
            selected_indices=selected_indices,
            num_nodes=rollout.num_nodes,
            device=device,
        )
        evidence_overlap = compute_mean_pairwise_overlap(
            selected_indices=selected_indices,
            overlap_features=pre_state["redundancy_signature"],
        )
        witness_pair_stats = compute_witness_pair_stats(
            selected_indices=selected_indices,
            witness_pair_signature=pre_state["witness_pair_signature"],
        )
        role_potentials = pre_state["role_potentials"]
        anchor_selected = [float(role_potentials[int(idx), 0].item()) for idx in selected_indices]
        frontier_selected = [float(role_potentials[int(idx), 1].item()) for idx in selected_indices]
        pair_selected = [float(role_potentials[int(idx), 2].item()) for idx in selected_indices]
        role_alignment_flags = [
            float(len(anchor_selected) > 0 and anchor_selected[0] >= max(anchor_selected) - 1e-8),
            float(len(frontier_selected) > 1 and frontier_selected[1] >= max(frontier_selected) - 1e-8),
            float(len(pair_selected) > 2 and pair_selected[2] >= max(pair_selected) - 1e-8),
        ]
        role_alignment_mean = float(sum(role_alignment_flags) / max(len(role_alignment_flags), 1))
        slot_collapse = float(evidence_overlap >= 0.9)
        rollout.step_with_actions(selected_indices, sample_types=[f"slot_{slot}" for slot in range(len(selected_indices))])
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])
        post_state = build_state_bundle(
            rollout=rollout,
            history=history,
            env=env,
            topology=topology,
            num_episodes=int(args.num_episodes),
            action_budget=int(args.action_budget),
            frontier_role_mode=str(args.frontier_role_mode),
        )
        metrics = compute_clean_transition_metrics(pre_state, post_state)
        metrics["selected_pairwise_distance"] = float(pairwise_distance)
        metrics["selected_evidence_overlap"] = float(evidence_overlap)
        metrics.update(witness_pair_stats)
        metrics["selected_anchor_role_potentials"] = anchor_selected
        metrics["selected_frontier_role_potentials"] = frontier_selected
        metrics["selected_pair_role_potentials"] = pair_selected
        metrics["selected_designated_role_potentials"] = [
            anchor_selected[0] if anchor_selected else 0.0,
            frontier_selected[1] if len(frontier_selected) > 1 else 0.0,
            pair_selected[2] if len(pair_selected) > 2 else 0.0,
        ]
        metrics["anchor_slot_alignment"] = role_alignment_flags[0]
        metrics["frontier_slot_alignment"] = role_alignment_flags[1]
        metrics["pair_slot_alignment"] = role_alignment_flags[2]
        metrics["role_alignment_mean"] = role_alignment_mean
        metrics["slot_collapse"] = float(slot_collapse)
        metrics["exact_k"] = exact_k
        metrics["unique_selection"] = unique_sel
        metrics["unique_global_selection"] = unique_global_sel
        metrics["valid_selection"] = valid_sel
        metrics["budget_used"] = float(len(selected_indices))
        metrics["source_sampled"] = source_sampled
        metrics["episode"] = int(episode_idx + 1)
        metrics["selected_indices"] = list(selected_indices)
        metrics["selected_global_ids"] = list(selected_global_ids)
        metrics["positive_witness_count_after"] = int(post_state["positive_count"])
        metrics["safe_witness_count_after"] = int(post_state["safe_count"])
        step_rows.append(metrics)
        totals["reward_total"] += metrics["reward"]
        totals["ignorance_delta_total"] += metrics["ignorance_delta"]
        totals["conflict_delta_total"] += metrics["conflict_delta"]
        totals["pair_delta_total"] += metrics["pair_available_delta"]
        totals["live_delta_total"] += metrics["live_delta"]
        totals["support_delta_total"] += metrics["support_delta"]
        totals["unresolved_delta_total"] += metrics["unresolved_delta"]
        totals["action_valid_count"] += valid_sel
        totals["exact_k_count"] += exact_k
        totals["unique_count"] += unique_sel
        totals["unique_global_count"] += unique_global_sel
        totals["pairwise_distance_total"] += pairwise_distance
        totals["evidence_overlap_total"] += evidence_overlap
        totals["witness_pair_overlap_total"] += float(witness_pair_stats["selected_witness_pair_overlap"])
        totals["witness_pair_coverage_total"] += float(witness_pair_stats["selected_witness_pair_coverage_fraction"])
        totals["witness_pair_union_total"] += float(witness_pair_stats["selected_witness_pair_union_count"])
        totals["witness_pair_active_node_total"] += float(witness_pair_stats["selected_witness_pair_active_node_fraction"])
        totals["anchor_slot_alignment_total"] += role_alignment_flags[0]
        totals["frontier_slot_alignment_total"] += role_alignment_flags[1]
        totals["pair_slot_alignment_total"] += role_alignment_flags[2]
        totals["role_alignment_total"] += role_alignment_mean
        totals["designated_anchor_total"] += anchor_selected[0] if anchor_selected else 0.0
        totals["designated_frontier_total"] += frontier_selected[1] if len(frontier_selected) > 1 else 0.0
        totals["designated_pair_total"] += pair_selected[2] if len(pair_selected) > 2 else 0.0
        totals["budget_used"] += float(len(selected_indices))
        totals["source_sampled_count"] += source_sampled
        totals["step_count"] += 1
        totals["slot_collapse_count"] += slot_collapse
        totals["witness_pair_slot_collapse_count"] += float(witness_pair_stats["witness_pair_slot_collapse"])
        if model_out is not None:
            metrics["state_value"] = float(model_out["value"].detach().cpu().item())
            metrics["set_value"] = float(model_out["set_value"].detach().cpu().item())
            old_slot_log_probs = extract_slot_log_probs(
                model_out.get("slot_logits", []),
                selected_indices,
            )
            state_action_feature_vector = None
            if str(getattr(args, "advantage_consumer_mode", "canonical")) == "regime_masked_alt_action_conditioned_ridge":
                state_action_feature_vector = model_out["state_action_feature_vector"].detach().cpu().to(torch.float64)
            if buffer_rows is not None:
                redundancy_signature = pre_state["redundancy_signature"]
                if getattr(model, "diversity_mode", "none") == "witness_pair_jaccard":
                    redundancy_signature = pre_state["witness_pair_signature"]
                complementarity_signature = None
                if getattr(model, "complementarity_mode", "none") in {
                    "role_aware_witness_pair_jaccard",
                    "slot1_frontier_anchor_witness_pair_jaccard",
                }:
                    complementarity_signature = pre_state["witness_pair_signature"]
                buffer_rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "episode": int(metrics["episode"]),
                        "reward": float(metrics["reward"]),
                        "conflict_delta": float(metrics["conflict_delta"]),
                        "pair_available_delta": float(metrics["pair_available_delta"]),
                        "selected_indices": list(selected_indices),
                        "old_log_prob": float(model_out["log_prob"].detach().cpu().item()),
                        "old_slot_log_probs": [float(value) for value in old_slot_log_probs],
                        "old_value": float(model_out["value"].detach().cpu().item()),
                        "old_set_value": float(model_out["set_value"].detach().cpu().item()),
                        "state_action_feature_vector": model_out["state_action_feature_vector"].detach().cpu().to(torch.float32),
                        "node_features": pre_state["node_features"].detach().cpu().clone().to(torch.float32),
                        "edge_index": pre_state["edge_index"].detach().cpu().clone().to(torch.long),
                        "valid_mask": pre_state["valid_mask"].detach().cpu().clone().to(torch.bool),
                        "graph_features": pre_state["graph_features"].detach().cpu().clone().to(torch.float32),
                        "redundancy_features": redundancy_signature.detach().cpu().clone().to(torch.float32),
                        "complementarity_features": None
                        if complementarity_signature is None
                        else complementarity_signature.detach().cpu().clone().to(torch.float32),
                        "role_potentials": pre_state["role_potentials"].detach().cpu().clone().to(torch.float32),
                    }
                )
            train_tensors.append(
                {
                    "log_prob": model_out["log_prob"],
                    "entropy": model_out["entropy"],
                    "value": model_out["value"],
                    "set_value": model_out["set_value"],
                    "reward": torch.tensor(metrics["reward"], device=device, dtype=torch.float32),
                    "conflict_delta": torch.tensor(metrics["conflict_delta"], device=device, dtype=torch.float32),
                    "pair_available_delta": torch.tensor(metrics["pair_available_delta"], device=device, dtype=torch.float32),
                    "state_action_feature_vector": state_action_feature_vector,
                }
            )

    final_state = build_state_bundle(
        rollout=rollout,
        history=history,
        env=env,
        topology=topology,
        num_episodes=int(args.num_episodes),
        action_budget=int(args.action_budget),
        frontier_role_mode=str(args.frontier_role_mode),
    )
    final_unresolved = float(final_state["derived"]["unresolved_mass"][final_state["valid_mask"]].mean().item()) if bool(final_state["valid_mask"].any()) else 0.0
    step_denom = max(int(totals["step_count"]), 1)
    case_summary = {
        "case_id": case["case_id"],
        "split": case["split"],
        "dataset_index": int(case["dataset_index"]),
        "policy_name": policy_name,
        "step_count": int(totals["step_count"]),
        "reward_total": float(totals["reward_total"]),
        "reward_mean": float(totals["reward_total"] / step_denom),
        "ignorance_delta_total": float(totals["ignorance_delta_total"]),
        "conflict_delta_total": float(totals["conflict_delta_total"]),
        "pair_delta_total": float(totals["pair_delta_total"]),
        "live_delta_total": float(totals["live_delta_total"]),
        "support_delta_total": float(totals["support_delta_total"]),
        "unresolved_delta_total": float(totals["unresolved_delta_total"]),
        "action_validity_rate": float(totals["action_valid_count"] / step_denom),
        "exact_k_rate": float(totals["exact_k_count"] / step_denom),
        "unique_action_rate": float(totals["unique_count"] / step_denom),
        "unique_global_action_rate": float(totals["unique_global_count"] / step_denom),
        "selected_pairwise_distance_mean": float(totals["pairwise_distance_total"] / step_denom),
        "selected_evidence_overlap_mean": float(totals["evidence_overlap_total"] / step_denom),
        "selected_witness_pair_overlap_mean": float(totals["witness_pair_overlap_total"] / step_denom),
        "selected_witness_pair_coverage_fraction_mean": float(totals["witness_pair_coverage_total"] / step_denom),
        "selected_witness_pair_union_count_mean": float(totals["witness_pair_union_total"] / step_denom),
        "selected_witness_pair_active_node_fraction_mean": float(totals["witness_pair_active_node_total"] / step_denom),
        "anchor_slot_alignment_rate": float(totals["anchor_slot_alignment_total"] / step_denom),
        "frontier_slot_alignment_rate": float(totals["frontier_slot_alignment_total"] / step_denom),
        "pair_slot_alignment_rate": float(totals["pair_slot_alignment_total"] / step_denom),
        "role_alignment_mean": float(totals["role_alignment_total"] / step_denom),
        "designated_anchor_role_mean": float(totals["designated_anchor_total"] / step_denom),
        "designated_frontier_role_mean": float(totals["designated_frontier_total"] / step_denom),
        "designated_pair_role_mean": float(totals["designated_pair_total"] / step_denom),
        "budget_used": float(totals["budget_used"]),
        "budget_used_mean": float(totals["budget_used"] / step_denom),
        "source_sampled_rate_debug": float(totals["source_sampled_count"] / step_denom),
        "insufficient_valid_episodes": float(totals["insufficient_valid_episodes"]),
        "slot_collapse_step_fraction": float(totals["slot_collapse_count"] / step_denom),
        "witness_pair_slot_collapse_step_fraction": float(totals["witness_pair_slot_collapse_count"] / step_denom),
        "positive_witness_count_final": int(final_state["positive_count"]),
        "safe_witness_count_final": int(final_state["safe_count"]),
        "candidate_fraction_final": float(final_state["valid_mask"].float().mean().item()),
        "initial_unresolved_mean": float(initial_unresolved),
        "final_unresolved_mean": float(final_unresolved),
        "initial_live_mean": float(masked_mean(initial_state["derived"]["live_plausibility"], initial_state["valid_mask"]).item()),
        "final_live_mean": float(masked_mean(final_state["derived"]["live_plausibility"], final_state["valid_mask"]).item()),
        "final_pair_available_mean": float(masked_mean(final_state["pair_available"], final_state["valid_mask"]).item()),
        "debug_true_source_in_subgraph": float(src_local is not None),
    }
    return {
        "summary": case_summary,
        "steps": step_rows,
        "train_tensors": train_tensors,
    }


def compute_returns(
    rewards: Sequence[torch.Tensor],
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    running = torch.tensor(0.0, device=device)
    returns: List[torch.Tensor] = []
    for reward in reversed(rewards):
        running = reward + float(gamma) * running
        returns.append(running)
    returns.reverse()
    return torch.stack(returns) if returns else torch.zeros(0, device=device)


def load_advantage_critic(path: str) -> Optional[Dict[str, Any]]:
    critic_path = str(path).strip()
    if not critic_path:
        return None
    payload = json.loads(Path(critic_path).read_text(encoding="utf-8"))
    critic_type = str(payload.get("type", ""))
    if critic_type != "action_conditioned_ridge":
        raise ValueError(f"Unsupported advantage critic type: {critic_type}")
    return {
        "type": critic_type,
        "target": str(payload.get("target", "")),
        "alpha": float(payload.get("alpha", 0.0)),
        "mean": torch.tensor(payload["mean"], dtype=torch.float64),
        "std": torch.tensor(payload["std"], dtype=torch.float64),
        "weights": torch.tensor(payload["weights"], dtype=torch.float64),
        "rescaling": payload.get("rescaling"),
        "advantage_clip": payload.get("advantage_clip"),
        "path": critic_path,
    }


def build_state_action_feature_vector(
    model: CleanNavigatorV1,
    state: Dict[str, Any],
    selected_indices: Sequence[int],
) -> torch.Tensor:
    model_device = next(model.parameters()).device
    with torch.no_grad():
        node_features = state["node_features"].to(model_device)
        edge_index = state["edge_index"].to(model_device)
        graph_features = state["graph_features"].to(model_device).view(-1).float()
        node_embeddings = model.encode(node_features, edge_index)
        graph_context = node_embeddings.mean(dim=0)
        selected_tensor = torch.tensor(list(selected_indices), dtype=torch.long, device=model_device)
        selected_slot_features = model.build_selected_slot_features(node_embeddings, selected_tensor)
        return torch.cat([graph_context, graph_features, selected_slot_features], dim=0).detach().cpu().to(torch.float64)


def extract_slot_log_probs(
    slot_logits: Sequence[torch.Tensor],
    selected_indices: Sequence[int],
) -> List[float]:
    slot_log_probs: List[float] = []
    for logits, chosen_idx in zip(slot_logits, selected_indices):
        log_probs = torch.log_softmax(logits.detach().view(-1).float(), dim=0)
        slot_log_probs.append(float(log_probs[int(chosen_idx)].item()))
    return slot_log_probs


def predict_ridge_values(
    critic: Dict[str, Any],
    feature_matrix: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_eval = feature_matrix.detach().cpu().to(torch.float64)
    x_scaled = (x_eval - critic["mean"]) / critic["std"]
    ones = torch.ones((x_scaled.size(0), 1), dtype=torch.float64)
    design = torch.cat([ones, x_scaled], dim=1)
    pred = design @ critic["weights"]
    return pred.to(device=device, dtype=dtype)


def apply_advantage_critic_rescaling(
    raw_q_values: torch.Tensor,
    *,
    critic: Dict[str, Any],
    state_values: Optional[torch.Tensor],
) -> torch.Tensor:
    rescaling = critic.get("rescaling")
    if not isinstance(rescaling, dict):
        return raw_q_values
    mode = str(rescaling.get("name", rescaling.get("variant", "raw_q_alt"))).strip() or "raw_q_alt"
    if mode in {"raw_q_alt"}:
        return raw_q_values
    if mode in {"q_alt_minus_train_v_mean", "minus_train_state_value_mean"}:
        return raw_q_values - float(rescaling.get("train_state_value_mean", 0.0))
    if mode in {"q_alt_div_train_v_std_plus_mean", "divide_by_train_state_value_std_plus_mean"}:
        train_v_std = max(float(rescaling.get("train_state_value_std", 0.0)), 1e-6)
        train_v_mean = float(rescaling.get("train_state_value_mean", 0.0))
        return raw_q_values / train_v_std + train_v_mean
    if mode in {"q_alt_clip_unit", "clip_signed_unit_interval"}:
        return raw_q_values.clamp(min=-1.0, max=1.0)
    if mode in {"q_alt_plus_half_v", "optimism_shift_half_state_value"}:
        if state_values is None:
            raise ValueError("q_alt_plus_half_v rescaling requires the masked state_values tensor")
        return raw_q_values + 0.5 * state_values
    raise ValueError(f"Unsupported advantage critic rescaling mode: {mode}")


def predict_advantage_critic_values(
    critic: Dict[str, Any],
    feature_matrix: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    state_values: Optional[torch.Tensor],
) -> torch.Tensor:
    critic_type = str(critic.get("type", ""))
    if critic_type == "action_conditioned_ridge":
        raw_q_values = predict_ridge_values(
            critic,
            feature_matrix,
            device=device,
            dtype=dtype,
        )
    elif critic_type == "action_conditioned_mlp":
        module = critic.get("module")
        if module is None:
            raise ValueError("action_conditioned_mlp critic requires a loaded module")
        module_device = next(module.parameters()).device
        x_eval = feature_matrix.detach().to(device=module_device, dtype=torch.float32)
        mean = critic["mean"].to(device=module_device, dtype=torch.float32)
        std = critic["std"].to(device=module_device, dtype=torch.float32)
        x_scaled = (x_eval - mean) / std
        with torch.no_grad():
            raw_q_values = module(x_scaled).view(-1).to(device=device, dtype=dtype)
    elif critic_type == "action_conditioned_prefix_mlp":
        module = critic.get("module")
        if module is None:
            raise ValueError("action_conditioned_prefix_mlp critic requires a loaded module")
        module_device = next(module.parameters()).device
        x_eval = feature_matrix.detach().to(device=module_device, dtype=torch.float32)
        mean = critic["mean"].to(device=module_device, dtype=torch.float32)
        std = critic["std"].to(device=module_device, dtype=torch.float32)
        x_scaled = (x_eval - mean) / std
        with torch.no_grad():
            module_out = module(x_scaled)
            if isinstance(module_out, dict):
                raw_q_values = module_out["values"][-1].view(-1).to(device=device, dtype=dtype)
            else:
                raw_q_values = module_out.view(-1).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported advantage critic type: {critic_type}")
    return apply_advantage_critic_rescaling(
        raw_q_values,
        critic=critic,
        state_values=state_values,
    )


def apply_advantage_critic_clipping(
    advantages: torch.Tensor,
    *,
    critic: Dict[str, Any],
) -> torch.Tensor:
    clip_cfg = critic.get("advantage_clip")
    if not isinstance(clip_cfg, dict):
        return advantages
    clip_min = clip_cfg.get("min")
    clip_max = clip_cfg.get("max")
    clip_min_value = None if clip_min is None else float(clip_min)
    clip_max_value = None if clip_max is None else float(clip_max)
    if clip_min_value is None and clip_max_value is None:
        return advantages
    if clip_min_value is None:
        clip_min_value = float("-inf")
    if clip_max_value is None:
        clip_max_value = float("inf")
    return advantages.clamp(min=clip_min_value, max=clip_max_value)


def std_from_sums(total: float, total_sq: float, count: int) -> float:
    if count <= 0:
        return 0.0
    mean = float(total / count)
    variance = max(float(total_sq / count) - mean * mean, 0.0)
    return float(variance ** 0.5)


def summarise_case_rows(case_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not case_rows:
        return {}
    keys = [
        "reward_total",
        "reward_mean",
        "ignorance_delta_total",
        "conflict_delta_total",
        "pair_delta_total",
        "live_delta_total",
        "support_delta_total",
        "unresolved_delta_total",
        "action_validity_rate",
        "exact_k_rate",
        "unique_action_rate",
        "unique_global_action_rate",
        "selected_pairwise_distance_mean",
        "selected_evidence_overlap_mean",
        "selected_witness_pair_overlap_mean",
        "selected_witness_pair_coverage_fraction_mean",
        "selected_witness_pair_union_count_mean",
        "selected_witness_pair_active_node_fraction_mean",
        "anchor_slot_alignment_rate",
        "frontier_slot_alignment_rate",
        "pair_slot_alignment_rate",
        "role_alignment_mean",
        "designated_anchor_role_mean",
        "designated_frontier_role_mean",
        "designated_pair_role_mean",
        "budget_used",
        "budget_used_mean",
        "positive_witness_count_final",
        "safe_witness_count_final",
        "candidate_fraction_final",
        "initial_unresolved_mean",
        "final_unresolved_mean",
        "initial_live_mean",
        "final_live_mean",
        "final_pair_available_mean",
        "source_sampled_rate_debug",
        "slot_collapse_step_fraction",
        "witness_pair_slot_collapse_step_fraction",
    ]
    summary = {"case_count": float(len(case_rows))}
    for key in keys:
        summary[key] = float(sum(float(row[key]) for row in case_rows) / max(len(case_rows), 1))
    summary["zero_reward_case_fraction"] = float(
        sum(float(abs(float(row["reward_total"])) <= 1e-12) for row in case_rows) / max(len(case_rows), 1)
    )
    summary["nonpositive_reward_case_fraction"] = float(
        sum(float(float(row["reward_total"]) <= 1e-12) for row in case_rows) / max(len(case_rows), 1)
    )
    summary["zero_pair_delta_case_fraction"] = float(
        sum(float(abs(float(row["pair_delta_total"])) <= 1e-12) for row in case_rows) / max(len(case_rows), 1)
    )
    summary["zero_positive_final_case_fraction"] = float(
        sum(float(int(row["positive_witness_count_final"]) == 0) for row in case_rows) / max(len(case_rows), 1)
    )
    return summary


def summarise_step_rows(step_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not step_rows:
        return {}
    denom = max(len(step_rows), 1)
    return {
        "zero_reward_step_fraction": float(
            sum(float(abs(float(row["reward"])) <= 1e-12) for row in step_rows) / denom
        ),
        "zero_pair_delta_step_fraction": float(
            sum(float(abs(float(row["pair_available_delta"])) <= 1e-12) for row in step_rows) / denom
        ),
        "selected_evidence_overlap_step_mean": float(
            sum(float(row["selected_evidence_overlap"]) for row in step_rows) / denom
        ),
        "selected_witness_pair_overlap_step_mean": float(
            sum(float(row.get("selected_witness_pair_overlap", 0.0)) for row in step_rows) / denom
        ),
        "selected_witness_pair_coverage_fraction_step_mean": float(
            sum(float(row.get("selected_witness_pair_coverage_fraction", 0.0)) for row in step_rows) / denom
        ),
        "role_alignment_step_mean": float(
            sum(float(row.get("role_alignment_mean", 0.0)) for row in step_rows) / denom
        ),
        "slot_collapse_step_fraction": float(
            sum(float(row["slot_collapse"]) for row in step_rows) / denom
        ),
        "witness_pair_slot_collapse_step_fraction": float(
            sum(float(row.get("witness_pair_slot_collapse", 0.0)) for row in step_rows) / denom
        ),
        "duplicate_global_step_fraction": float(
            sum(float(float(row["unique_global_selection"]) < 1.0) for row in step_rows) / denom
        ),
        "zero_positive_after_step_fraction": float(
            sum(float(int(row["positive_witness_count_after"]) == 0) for row in step_rows) / denom
        ),
    }


def train_epoch(
    model: CleanNavigatorV1,
    optimizer: torch.optim.Optimizer,
    train_cases: Sequence[Dict[str, Any]],
    availability_by_case: Optional[Dict[str, Dict[str, Any]]],
    env: CleanTwoChannelEvidenceEnv,
    topology,
    dataset_assets: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    epoch_seed: int,
    epoch_number: int,
    global_training_generator: Optional[torch.Generator] = None,
    trace_rows: Optional[List[Dict[str, Any]]] = None,
    advantage_critic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model.train()
    order, sampler_info = sample_train_cases(
        train_cases=train_cases,
        availability_by_case=availability_by_case,
        args=args,
        epoch_index=int(epoch_number),
        epoch_seed=int(epoch_seed),
    )
    exposure_summary = summarise_sampler_exposure(order, availability_by_case)
    case_rows: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_state_value_loss = 0.0
    total_set_value_loss = 0.0
    total_entropy = 0.0
    advantage_step_count = 0
    advantage_consumed_step_count = 0
    advantage_total = 0.0
    advantage_total_sq = 0.0
    advantage_preclip_total = 0.0
    advantage_preclip_total_sq = 0.0
    canonical_advantage_total = 0.0
    canonical_advantage_total_sq = 0.0
    advantage_abs_delta_total = 0.0
    advantage_sign_flip_count = 0
    advantage_clip_count = 0
    masked_reward_bonus_total = 0.0
    masked_reward_bonus_step_count = 0
    clip_min = getattr(args, "advantage_clip_min", None)
    clip_max = getattr(args, "advantage_clip_max", None)
    grad_norms: List[float] = []
    optimizer_step_cases = max(1, int(getattr(args, "optimizer_step_cases", 1)))
    accumulation_counter = 0
    optimizer.zero_grad(set_to_none=True)

    for case_idx, case in enumerate(order):
        generator: Optional[torch.Generator]
        if str(args.training_sampling_mode) == "global_torch":
            generator = None
        elif str(args.training_sampling_mode) == "global_seeded_cpu":
            generator = global_training_generator
            if generator is None:
                raise ValueError("global_seeded_cpu requires a global_training_generator")
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(epoch_seed * 1000 + case_idx)
        trace_this_case = (
            trace_rows
            if bool(args.trace_train_actions)
            and (int(args.trace_max_train_cases) <= 0 or case_idx < int(args.trace_max_train_cases))
            else None
        )
        rollout_out = run_case_rollout(
            case=case,
            policy_name="clean_navigator_v1",
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
            model=model,
            generator=generator,
            deterministic=False,
            trace_rows=trace_this_case,
            trace_context={
                "epoch": int(epoch_number),
                "case_idx": int(case_idx),
                "case_id": str(case["case_id"]),
                "before_first_weight_update": bool(case_idx == 0),
            },
        )
        case_rows.append(rollout_out["summary"])
        train_tensors = rollout_out["train_tensors"]
        if not train_tensors:
            continue
        rewards = [
            row["reward"] + float(args.train_conflict_bonus_weight) * row["conflict_delta"]
            for row in train_tensors
        ]
        values = torch.stack([row["value"] for row in train_tensors])
        set_values = torch.stack([row["set_value"] for row in train_tensors])
        log_probs = torch.stack([row["log_prob"] for row in train_tensors])
        entropies = torch.stack([row["entropy"] for row in train_tensors])
        masked_negative_conflict_bonus_weight = float(
            getattr(args, "train_masked_negative_conflict_bonus_weight", 0.0)
        )
        if str(args.credit_mode) == "action_set_q":
            returns = compute_returns(rewards, gamma=float(args.gamma), device=device)
            state_value_loss = F.mse_loss(values, returns)
            set_value_loss = F.mse_loss(set_values, returns)
            canonical_advantages = returns - values
            advantages = set_values - values
            value_loss = 0.5 * (state_value_loss + set_value_loss)
        else:
            consumer_mode = str(getattr(args, "advantage_consumer_mode", "canonical"))
            if consumer_mode in {"regime_masked_canonical_state_value", "regime_masked_alt_action_conditioned_ridge"}:
                pair_active_threshold = float(getattr(args, "advantage_mask_pair_active_threshold", 0.05))
                regime_mask = torch.tensor(
                    [
                        float(row["pair_available_delta"].item()) >= pair_active_threshold
                        and float(row["conflict_delta"].item()) < 0.0
                        for row in train_tensors
                    ],
                    device=device,
                    dtype=torch.bool,
                )
                if not bool(regime_mask.any()):
                    continue
                if masked_negative_conflict_bonus_weight > 0.0:
                    updated_rewards = []
                    for reward, row, use_row in zip(rewards, train_tensors, regime_mask.detach().cpu().tolist()):
                        reward_bonus = (
                            masked_negative_conflict_bonus_weight * max(-float(row["conflict_delta"].item()), 0.0)
                            if bool(use_row)
                            else 0.0
                        )
                        updated_rewards.append(reward + reward_bonus)
                        if reward_bonus > 0.0:
                            masked_reward_bonus_total += float(reward_bonus)
                            masked_reward_bonus_step_count += 1
                    rewards = updated_rewards
                returns = compute_returns(rewards, gamma=float(args.gamma), device=device)
                canonical_advantages = returns[regime_mask] - values[regime_mask]
                state_value_loss = F.mse_loss(values[regime_mask], returns[regime_mask])
                set_value_loss = F.mse_loss(set_values[regime_mask], returns[regime_mask])
                if consumer_mode == "regime_masked_alt_action_conditioned_ridge":
                    if advantage_critic is None:
                        raise ValueError("advantage_consumer_mode requires a loaded advantage critic")
                    regime_mask_list = regime_mask.detach().cpu().tolist()
                    feature_matrix = torch.stack(
                        [
                            row["state_action_feature_vector"]
                            for row, use_row in zip(train_tensors, regime_mask_list)
                            if bool(use_row) and row["state_action_feature_vector"] is not None
                        ],
                        dim=0,
                    )
                    alt_q_values = predict_advantage_critic_values(
                        advantage_critic,
                        feature_matrix,
                        device=device,
                        dtype=values.dtype,
                        state_values=values[regime_mask],
                    )
                    advantages = alt_q_values - values[regime_mask]
                    advantages = apply_advantage_critic_clipping(
                        advantages,
                        critic=advantage_critic,
                    )
                    advantage_sign_flip_count += int(((canonical_advantages * advantages) < 0.0).sum().item())
                else:
                    advantages = canonical_advantages
                log_probs = log_probs[regime_mask]
                entropies = entropies[regime_mask]
                advantage_consumed_step_count += int(regime_mask.sum().item())
            else:
                returns = compute_returns(rewards, gamma=float(args.gamma), device=device)
                state_value_loss = F.mse_loss(values, returns)
                set_value_loss = F.mse_loss(set_values, returns)
                canonical_advantages = returns - values
                advantages = canonical_advantages
            value_loss = state_value_loss
        preclip_advantages = advantages
        if clip_min is not None or clip_max is not None:
            clipped_advantages = preclip_advantages.clamp(
                min=float("-inf") if clip_min is None else float(clip_min),
                max=float("inf") if clip_max is None else float(clip_max),
            )
            advantage_clip_count += int((clipped_advantages != preclip_advantages).sum().item())
            advantages = clipped_advantages
        advantage_preclip_total += float(preclip_advantages.detach().sum().item())
        advantage_preclip_total_sq += float((preclip_advantages.detach() ** 2).sum().item())
        advantage_step_count += int(advantages.numel())
        advantage_total += float(advantages.detach().sum().item())
        advantage_total_sq += float((advantages.detach() ** 2).sum().item())
        canonical_advantage_total += float(canonical_advantages.detach().sum().item())
        canonical_advantage_total_sq += float((canonical_advantages.detach() ** 2).sum().item())
        advantage_abs_delta_total += float((advantages.detach() - canonical_advantages.detach()).abs().sum().item())
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_bonus = entropies.mean()
        loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy_bonus
        scaled_loss = loss / float(optimizer_step_cases)
        scaled_loss.backward()
        accumulation_counter += 1
        grad_norm = None
        should_step = accumulation_counter >= optimizer_step_cases or case_idx == (len(order) - 1)
        if should_step:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

        total_loss += float(loss.item())
        total_policy_loss += float(policy_loss.item())
        total_value_loss += float(value_loss.item())
        total_state_value_loss += float(state_value_loss.item())
        total_set_value_loss += float(set_value_loss.item())
        total_entropy += float(entropy_bonus.item())
        if grad_norm is not None:
            grad_norms.append(grad_norm)

    epoch_summary = summarise_case_rows(case_rows)
    case_count = max(int(epoch_summary.get("case_count", 0.0)), 1)
    epoch_summary.update(
        {
            "train_loss_mean": float(total_loss / case_count),
            "policy_loss_mean": float(total_policy_loss / case_count),
            "value_loss_mean": float(total_value_loss / case_count),
            "state_value_loss_mean": float(total_state_value_loss / case_count),
            "set_value_loss_mean": float(total_set_value_loss / case_count),
            "entropy_mean": float(total_entropy / case_count),
            "grad_norm_mean": float(sum(grad_norms) / max(len(grad_norms), 1)),
            "optimizer_step_cases": int(optimizer_step_cases),
            "optimizer_step_count": int(len(grad_norms)),
            "advantage_step_count": int(advantage_step_count),
            "advantage_consumed_step_count": int(advantage_consumed_step_count),
            "advantage_consumed_step_fraction": float(advantage_consumed_step_count / max(advantage_step_count, 1)),
            "advantage_mean": float(advantage_total / max(advantage_step_count, 1)),
            "advantage_std": float(std_from_sums(advantage_total, advantage_total_sq, advantage_step_count)),
            "advantage_preclip_mean": float(advantage_preclip_total / max(advantage_step_count, 1)),
            "advantage_preclip_std": float(
                std_from_sums(advantage_preclip_total, advantage_preclip_total_sq, advantage_step_count)
            ),
            "advantage_clip_count": int(advantage_clip_count),
            "advantage_clip_fraction": float(advantage_clip_count / max(advantage_step_count, 1)),
            "advantage_clip_min": None if clip_min is None else float(clip_min),
            "advantage_clip_max": None if clip_max is None else float(clip_max),
            "canonical_advantage_mean": float(canonical_advantage_total / max(advantage_step_count, 1)),
            "canonical_advantage_std": float(std_from_sums(canonical_advantage_total, canonical_advantage_total_sq, advantage_step_count)),
            "mean_abs_advantage_delta_vs_canonical": float(advantage_abs_delta_total / max(advantage_step_count, 1)),
            "advantage_sign_flip_count": int(advantage_sign_flip_count),
            "advantage_sign_flip_fraction_consumed": float(advantage_sign_flip_count / max(advantage_consumed_step_count, 1)),
            "masked_reward_bonus_total": float(masked_reward_bonus_total),
            "masked_reward_bonus_step_count": int(masked_reward_bonus_step_count),
            **sampler_info,
            **{f"sampler_{key}": value for key, value in exposure_summary.items()},
        }
    )
    return {
        "summary": epoch_summary,
        "case_rows": case_rows,
        "selected_case_ids": [case["case_id"] for case in order],
    }


def evaluate_policy(
    policy_name: str,
    cases: Sequence[Dict[str, Any]],
    env: CleanTwoChannelEvidenceEnv,
    topology,
    dataset_assets: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    model: Optional[CleanNavigatorV1] = None,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.seed) * 1000 + case_idx)
        rollout_out = run_case_rollout(
            case=case,
            policy_name=policy_name,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
            model=model,
            generator=generator,
            deterministic=True,
        )
        case_rows.append(rollout_out["summary"])
        for row in rollout_out["steps"]:
            step_rows.append({"case_id": case["case_id"], "policy_name": policy_name, **row})
    return {
        "summary": {
            **summarise_case_rows(case_rows),
            **summarise_step_rows(step_rows),
        },
        "case_rows": case_rows,
        "step_rows": step_rows,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row)) + "\n")


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_governing_file_hashes(config_path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for rel_path in [config_path, *GOVERNING_RELATIVE_FILES]:
        path = PROJECT_ROOT / rel_path
        entries.append(
            {
                "path": rel_path,
                "exists": bool(path.exists()),
                "sha256": compute_sha256(path) if path.exists() else None,
                "size_bytes": int(path.stat().st_size) if path.exists() else None,
            }
        )
    return entries


def collect_git_status(paths: Sequence[str]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": [line for line in proc.stdout.splitlines() if line.strip()],
            "stderr": [line for line in proc.stderr.splitlines() if line.strip()],
        }
    except Exception as exc:
        return {
            "returncode": None,
            "stdout": [],
            "stderr": [f"{type(exc).__name__}: {exc}"],
        }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    runtime_settings = configure_runtime(args, device)
    advantage_critic = load_advantage_critic(str(args.advantage_critic_path))
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_lmdb = False if bool(getattr(args, "use_lmdb", False)) else bool(args.skip_lmdb)
    cfg = build_cfg(args.config, skip_lmdb=skip_lmdb)
    train_pool_limit = int(args.train_pool_limit) if int(args.train_pool_limit) > 0 else int(args.max_train_cases)
    case_limits = {
        "train": int(train_pool_limit),
        "val": int(args.max_val_cases),
        "test": int(args.max_test_cases),
    }
    cases, topology, dataset_assets = create_case_splits(cfg, seed=int(args.seed), limits=case_limits)
    if topology is None:
        raise ValueError("Unable to resolve topology engine from datasets")
    if dataset_assets["global_edge_index"] is None or dataset_assets["stt_dynamic_series"] is None:
        raise ValueError("Dataset-global hydraulic assets are required for the clean rollout lane")

    env = CleanTwoChannelEvidenceEnv()
    availability_rows, availability_by_case, availability_summary = build_availability_rows(
        cases=cases,
        env=env,
        topology=topology,
        dataset_assets=dataset_assets,
        args=args,
    )
    sampler_preview = build_sampler_preview(
        train_cases=cases["train"],
        availability_by_case=availability_by_case,
        args=args,
    )
    model = CleanNavigatorV1(
        node_feature_dim=len(NODE_FEATURE_NAMES),
        graph_feature_dim=len(GRAPH_FEATURE_NAMES),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_slots=int(args.action_budget),
        greedy_eval=True,
        role_mode=str(args.role_mode),
        role_bias_weight=float(args.role_bias_weight),
        diversity_mode=str(args.diversity_mode),
        diversity_penalty_weight=float(args.diversity_penalty_weight),
        complementarity_mode=str(args.complementarity_mode),
        complementarity_penalty_weight=float(args.complementarity_penalty_weight),
        credit_mode=str(args.credit_mode),
    ).to(device)
    if str(args.init_checkpoint).strip():
        init_checkpoint_path = Path(str(args.init_checkpoint)).expanduser()
        if not init_checkpoint_path.is_absolute():
            init_checkpoint_path = PROJECT_ROOT / init_checkpoint_path
        checkpoint_state = torch.load(init_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_state, strict=bool(args.init_checkpoint_strict))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    history_rows: List[Dict[str, Any]] = []
    train_case_rows_all: List[Dict[str, Any]] = []
    best_val_score = -float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    training_trace_rows: List[Dict[str, Any]] = []
    global_training_generator: Optional[torch.Generator] = None
    if str(args.training_sampling_mode) == "global_seeded_cpu":
        global_training_generator = torch.Generator(device="cpu")
        global_training_generator.manual_seed(int(args.seed))

    for epoch in range(1, int(args.epochs) + 1):
        train_out = train_epoch(
            model=model,
            optimizer=optimizer,
            train_cases=cases["train"],
            availability_by_case=availability_by_case,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
            epoch_seed=int(args.seed) + epoch,
            epoch_number=epoch,
            global_training_generator=global_training_generator,
            trace_rows=training_trace_rows,
            advantage_critic=advantage_critic,
        )
        model.eval()
        with torch.no_grad():
            val_out = evaluate_policy(
                policy_name="clean_navigator_v1",
                cases=cases["val"],
                env=env,
                topology=topology,
                dataset_assets=dataset_assets,
                args=args,
                device=device,
                model=model,
            )
        val_score = float(val_out["summary"].get("reward_total", 0.0))
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        epoch_row = {
            "epoch": int(epoch),
            "train_summary": train_out["summary"],
            "val_summary": val_out["summary"],
            "selected_case_ids": train_out["selected_case_ids"],
        }
        history_rows.append(epoch_row)
        for row in train_out["case_rows"]:
            train_case_rows_all.append({"epoch": int(epoch), **row})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_reward_total": train_out["summary"].get("reward_total", 0.0),
                    "train_sampler_informative_fraction": train_out["summary"].get("sampler_informative_case_fraction", 0.0),
                    "train_sampler_zero_signal_fraction": train_out["summary"].get("sampler_zero_signal_case_fraction", 0.0),
                    "val_reward_total": val_out["summary"].get("reward_total", 0.0),
                    "val_unresolved_delta_total": val_out["summary"].get("unresolved_delta_total", 0.0),
                    "val_action_validity_rate": val_out["summary"].get("action_validity_rate", 0.0),
                }
            )
        )

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "clean_navigator_v1_best.pt")

    model.eval()
    with torch.no_grad():
        model_val = evaluate_policy(
            policy_name="clean_navigator_v1",
            cases=cases["val"],
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
            model=model,
        )
        model_test = evaluate_policy(
            policy_name="clean_navigator_v1",
            cases=cases["test"],
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
            model=model,
        )
        informative_test_cases = [
            case
            for case in cases["test"]
            if float(availability_by_case.get(case["case_id"], {}).get("informative_case", 0.0)) > 0.5
        ]
        informative_model_test = evaluate_policy(
            policy_name="clean_navigator_v1",
            cases=informative_test_cases,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
            model=model,
        )
        random_test = evaluate_policy(
            policy_name="random_valid",
            cases=cases["test"],
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
        )
        informative_random_test = evaluate_policy(
            policy_name="random_valid",
            cases=informative_test_cases,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
        )
        top_support_test = evaluate_policy(
            policy_name="top_support",
            cases=cases["test"],
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
        )
        informative_top_support_test = evaluate_policy(
            policy_name="top_support",
            cases=informative_test_cases,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
        )
        top_pair_test = evaluate_policy(
            policy_name="top_safe_pair_potential",
            cases=cases["test"],
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
        )
        informative_top_pair_test = evaluate_policy(
            policy_name="top_safe_pair_potential",
            cases=informative_test_cases,
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args=args,
            device=device,
        )

    write_jsonl(output_dir / "availability_case_summary.jsonl", availability_rows)
    write_csv_rows(output_dir / "availability_case_summary.csv", availability_rows)
    with open(output_dir / "availability_summary.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(availability_summary), handle, indent=2)
    with open(output_dir / "sampler_preview.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(sampler_preview), handle, indent=2)
    write_jsonl(output_dir / "train_history.jsonl", history_rows)
    write_jsonl(output_dir / "train_case_rows.jsonl", train_case_rows_all)
    if training_trace_rows:
        write_jsonl(output_dir / "train_action_trace.jsonl", training_trace_rows)
    write_jsonl(output_dir / "model_test_case_rows.jsonl", model_test["case_rows"])
    write_jsonl(output_dir / "model_test_step_rows.jsonl", model_test["step_rows"])
    write_jsonl(output_dir / "baseline_random_test_case_rows.jsonl", random_test["case_rows"])
    write_jsonl(output_dir / "baseline_top_support_test_case_rows.jsonl", top_support_test["case_rows"])
    write_jsonl(output_dir / "baseline_top_pair_test_case_rows.jsonl", top_pair_test["case_rows"])

    comparison_rows = [
        {"policy_name": "clean_navigator_v1", **model_test["summary"]},
        {"policy_name": "random_valid", **random_test["summary"]},
        {"policy_name": "top_support", **top_support_test["summary"]},
        {"policy_name": "top_safe_pair_potential", **top_pair_test["summary"]},
    ]
    with open(output_dir / "official_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(comparison_rows), handle, indent=2)
    informative_comparison_rows = [
        {"policy_name": "clean_navigator_v1", **informative_model_test["summary"]},
        {"policy_name": "random_valid", **informative_random_test["summary"]},
        {"policy_name": "top_support", **informative_top_support_test["summary"]},
        {"policy_name": "top_safe_pair_potential", **informative_top_pair_test["summary"]},
    ]
    with open(output_dir / "official_comparison_informative.json", "w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(
                {
                    "informative_case_count": int(len(informative_test_cases)),
                    "rows": informative_comparison_rows,
                }
            ),
            handle,
            indent=2,
        )

    governing_file_hashes = build_governing_file_hashes(str(args.config))
    with open(output_dir / "governing_file_hashes.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(governing_file_hashes), handle, indent=2)
    reproducibility_manifest = {
        "command": ["python", *sys.argv],
        "args": vars(args),
        "runtime_settings": runtime_settings,
        "init_checkpoint": str(args.init_checkpoint) if str(args.init_checkpoint).strip() else None,
        "init_checkpoint_strict": bool(args.init_checkpoint_strict),
        "checkpoint_selection_rule": {
            "metric": "val_summary.reward_total",
            "comparison": "strict_greater_than",
            "tie_break": "earliest_epoch_wins",
        },
        "split_selection_rule": {
            "function": "create_case_splits",
            "seed": int(args.seed),
            "limits": case_limits,
        },
        "sampler_selection_rule": {
            "function": "sample_train_cases",
            "epoch_seed_rule": "seed + epoch",
            "train_case_generator_rule": "epoch_seed * 1000 + case_idx",
            "eval_case_generator_rule": "seed * 1000 + case_idx",
            "global_training_generator_rule": (
                "seeded_once_with_run_seed"
                if str(args.training_sampling_mode) == "global_seeded_cpu"
                else None
            ),
        },
        "governing_files": governing_file_hashes,
        "git_status_for_governing_files": collect_git_status(
            [str(args.config), *GOVERNING_RELATIVE_FILES]
        ),
        "trust_boundary": {
            "tracked_source_required": False,
            "warning": "Governance relies on emitted file hashes because governing clean Navigator files may be untracked in this workspace.",
        },
    }
    with open(output_dir / "reproducibility_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(reproducibility_manifest), handle, indent=2)
    with open(output_dir / "deterministic_settings_audit.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(runtime_settings), handle, indent=2)

    summary = {
        "args": vars(args),
        "runtime_settings": runtime_settings,
        "checkpoint_selection_rule": "best validation reward_total, strict greater-than only, earliest epoch wins ties",
        "device": str(device),
        "node_feature_names": NODE_FEATURE_NAMES,
        "graph_feature_names": GRAPH_FEATURE_NAMES,
        "redundancy_feature_names": REDUNDANCY_FEATURE_NAMES,
        "role_potential_names": ROLE_POTENTIAL_NAMES,
        "best_epoch": int(best_epoch),
        "best_val_reward_total": float(best_val_score),
        "train_case_count": len(cases["train"]),
        "epoch_train_case_count": int(args.train_cases_per_epoch) if int(args.train_cases_per_epoch) > 0 else len(cases["train"]),
        "val_case_count": len(cases["val"]),
        "test_case_count": len(cases["test"]),
        "availability_summary": availability_summary,
        "sampler_preview": sampler_preview,
        "model_val_summary": model_val["summary"],
        "model_test_summary": model_test["summary"],
        "informative_subset_test_summary": informative_model_test["summary"],
        "baseline_test_summaries": {
            "random_valid": random_test["summary"],
            "top_support": top_support_test["summary"],
            "top_safe_pair_potential": top_pair_test["summary"],
        },
        "baseline_informative_subset_test_summaries": {
            "random_valid": informative_random_test["summary"],
            "top_support": informative_top_support_test["summary"],
            "top_safe_pair_potential": informative_top_pair_test["summary"],
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)


if __name__ == "__main__":
    main()
