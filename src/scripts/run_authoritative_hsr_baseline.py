from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

LEGACY_HSR_ROOT = PROJECT_ROOT / "tools" / "legacy" / "src_baselines_archive"
if str(LEGACY_HSR_ROOT) not in sys.path:
    sys.path.append(str(LEGACY_HSR_ROOT))

from hsr_agent import HSRAgent

from src.config.core import Config
from src.modeling.evidence.two_channel_clean import ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_audit import load_runtime_context


DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT / "artifacts" / "reasoner_same_case_stronger_source_overfit" / "20260407_exact136_h3_formal_v1"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache_lmdb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "authoritative_hsr_baseline" / "20260409_exact136_hsr_authoritative_v1"
RUNNER_VERSION = "authoritative_hsr_baseline_v1"
PANEL_VERSION = "exact_same_case_train_only_h3_trainset_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Authoritative HSR greedy baseline under current exact136 runtime.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--progress-every-cases", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=0)
    parser.add_argument("--actions-per-round", type=int, default=0)
    parser.add_argument("--episode-duration-min", type=float, default=0.0)
    parser.add_argument("--case-subset-csv", type=str, default="")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_foundation_graph_path(source_root: Path) -> Path:
    summary = read_json(source_root / "summary.json")
    oracle_root = Path(summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    life_support = payload.get("life_support")
    if isinstance(life_support, dict) and str(life_support.get("profile")) == "custom_direct_edit":
        payload["life_support"] = {k: v for k, v in life_support.items() if k != "profile"}
    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(payload)
    return Path(cfg.paths.foundation_path) / "graph.npz"


def load_foundation_graph(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        edge_index = f["edge_index"]
        if "edge_attr_summary" in f:
            summary = f["edge_attr_summary"]
            p_forward = summary[:, 0]
            median_stt = summary[:, 1]
            min_stt = summary[:, 3]
        else:
            edge_count = edge_index.shape[1]
            p_forward = np.ones(edge_count, dtype=np.float32)
            median_stt = np.ones(edge_count, dtype=np.float32) * 0.02
            min_stt = np.ones(edge_count, dtype=np.float32) * 0.01
    return {
        "edge_index": edge_index,
        "p_forward": p_forward,
        "median_stt": median_stt,
        "min_stt": min_stt,
    }


def build_hsr_agent(*, graph_data: Dict[str, np.ndarray], runtime: Dict[str, Any]) -> HSRAgent:
    agent = HSRAgent(
        graph_data=graph_data,
        time_step_hours=float(runtime["episode_duration_min"]) / 60.0,
        tolerance_hours=24.0,
    )
    agent.min_stts = graph_data["min_stt"]
    agent._build_reverse_graph()
    return agent


def _extract_trigger_global(case_data: Any) -> Optional[int]:
    trigger = getattr(case_data, "global_trigger_node", None)
    if trigger is None:
        return None
    if isinstance(trigger, torch.Tensor):
        return int(trigger.view(-1)[0].item())
    return int(trigger)


def _extract_source_global(rollout: PracticalRollout) -> Optional[int]:
    source_local = resolve_source_local_idx(rollout)
    if source_local is None:
        return None
    return int(rollout.g_ids[int(source_local)].item())


def _seed_case(seed: int, case_id: str) -> None:
    digest = hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()
    stable_case_seed = int(digest[:8], 16)
    local_seed = (int(seed) ^ stable_case_seed) & 0xFFFFFFFF
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(int(local_seed % (2**31)))


def run_case_hsr(
    *,
    case: Any,
    runtime: Dict[str, Any],
    agent: HSRAgent,
    seed: int,
) -> Dict[str, Any]:
    _seed_case(seed, case.case_id)
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

    source_global = _extract_source_global(rollout)
    trigger_global = _extract_trigger_global(case.data)
    initial_candidates = [int(v) for v in rollout.g_ids.detach().cpu().tolist()]
    agent.reset(candidates=initial_candidates, trigger_node=trigger_global, t_start=0)
    if trigger_global is not None:
        # Align the HSR clock with the paper semantics: the first detecting sensor is already
        # known at tau*=0 before the first adaptive sample is selected.
        agent.current_time_step = -1
        agent.step({int(trigger_global): (1.0, 1)})

    step_rows: List[Dict[str, Any]] = []
    hit_round: Optional[int] = None
    hit_sample_index: Optional[int] = None
    termination_reason = "budget_exhausted"
    total_samples = 0

    for round_index in range(1, int(runtime["num_episodes"]) + 1):
        remaining_candidates = list(agent.candidate_set - agent.sampled_nodes)
        if not remaining_candidates:
            termination_reason = "no_remaining_candidates"
            break

        candidate_count_pre = int(len(agent.candidate_set))
        actions_global = [int(v) for v in agent.get_action_hsr_scalable(k=int(runtime["action_budget"]))]
        global_to_local = {int(gid): int(idx) for idx, gid in enumerate(rollout.g_ids.detach().cpu().tolist())}
        selected_global: List[int] = []
        selected_local: List[int] = []
        seen_global = set()
        for gid in actions_global:
            if gid in seen_global:
                continue
            seen_global.add(int(gid))
            local_idx = global_to_local.get(int(gid))
            if local_idx is None:
                continue
            if bool(rollout.revealed_mask[int(local_idx)].item()):
                continue
            selected_global.append(int(gid))
            selected_local.append(int(local_idx))
            if len(selected_local) >= int(runtime["action_budget"]):
                break
        if not selected_local:
            termination_reason = "no_valid_local_actions"
            break

        round_hit = source_global is not None and int(source_global) in set(selected_global)
        if round_hit and hit_round is None:
            hit_round = int(round_index)
            source_slot = selected_global.index(int(source_global)) + 1
            hit_sample_index = int((round_index - 1) * int(runtime["action_budget"]) + source_slot)

        rollout.step_with_actions(selected_local, sample_types=[f"hsr_slot_{i}" for i in range(len(selected_local))])
        if rollout.history_steps:
            history.append_from_history_step(rollout.history_steps[-1])

        total_samples += int(len(selected_local))
        latest_step = rollout.history_steps[-1] if rollout.history_steps else None
        observations: Dict[int, tuple[float, int]] = {}
        if latest_step is not None:
            for sample in latest_step.samples:
                observations[int(sample.global_idx)] = (
                    float(sample.concentration),
                    int(1 if bool(sample.is_positive) else 0),
                )
        if not round_hit and observations:
            agent.step(observations)

        final_candidate_count = int(len(agent.candidate_set))
        source_in_candidates = (
            float(source_global is not None and int(source_global) in agent.candidate_set)
            if source_global is not None
            else None
        )
        step_rows.append(
            {
                "case_id": case.case_id,
                "round_index": int(round_index),
                "selected_global_ids": json.dumps(selected_global),
                "selected_local_ids": json.dumps(selected_local),
                "selected_count": int(len(selected_local)),
                "source_hit_in_round": float(bool(round_hit)),
                "hit_sample_index": hit_sample_index if round_hit else None,
                "candidate_count_pre": candidate_count_pre,
                "candidate_count_post": final_candidate_count,
                "candidate_fraction_post": float(final_candidate_count / max(len(initial_candidates), 1)),
                "source_in_candidate_post": source_in_candidates,
                "positive_observation_count": int(sum(int(label > 0) for _, label in observations.values())),
                "safe_observation_count": int(sum(int(label <= 0) for _, label in observations.values())),
                "budget_used_cumulative": int(total_samples),
            }
        )

        if round_hit:
            termination_reason = "source_hit"
            break
        if final_candidate_count <= 0:
            termination_reason = "empty_candidate_set"
            break

    final_candidate_count = int(len(agent.candidate_set))
    final_source_in_candidates = (
        float(source_global is not None and int(source_global) in agent.candidate_set)
        if source_global is not None
        else None
    )
    final_actions_global = [int(v) for v in agent.get_action_hsr_scalable(k=int(runtime["action_budget"]))] if final_candidate_count > 0 else []
    final_topk_contains_source = (
        float(source_global is not None and int(source_global) in set(final_actions_global))
        if source_global is not None and final_actions_global
        else 0.0
    )
    return {
        "case_row": {
            "case_id": case.case_id,
            "scenario_id": int(case.scenario_id),
            "part_id": int(case.part_id),
            "success_rate": float(hit_round is not None),
            "hit_round": hit_round,
            "hit_sample_index": hit_sample_index,
            "budget_used": int(total_samples),
            "final_candidate_count": final_candidate_count,
            "final_candidate_fraction": float(final_candidate_count / max(len(initial_candidates), 1)),
            "final_source_in_candidate_set": final_source_in_candidates,
            "final_topk_contains_source": float(final_topk_contains_source),
            "termination_reason": termination_reason,
            "initial_candidate_count": int(len(initial_candidates)),
            "trigger_global_id": trigger_global,
            "source_global_id": source_global,
        },
        "step_rows": step_rows,
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime_context(source_root, cache_dir)
    if int(args.num_rounds) > 0:
        runtime["num_episodes"] = int(args.num_rounds)
    if int(args.actions_per_round) > 0:
        runtime["action_budget"] = int(args.actions_per_round)
    if float(args.episode_duration_min) > 0:
        runtime["episode_duration_min"] = float(args.episode_duration_min)
    if str(args.case_subset_csv).strip():
        subset_df = pd.read_csv(Path(args.case_subset_csv))
        keep = set(subset_df["case_id"].astype(str).tolist())
        runtime["cases"] = [case for case in runtime["cases"] if str(case.case_id) in keep]
    foundation_graph_path = resolve_foundation_graph_path(source_root)
    graph_data = load_foundation_graph(foundation_graph_path)
    agent = build_hsr_agent(graph_data=graph_data, runtime=runtime)

    protocol_audit = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "authoritative_hsr_implementation": {
            "selected": str(LEGACY_HSR_ROOT / "hsr_agent.py"),
            "why": [
                "contains the original HSR candidate-set pruning and get_action_hsr_scalable Monte Carlo voting semantics",
                "closest surviving implementation to the original HSR baseline intent",
            ],
            "rejected_alternatives": [
                {
                    "path": "src/scripts/experiments/run_hsr_baseline.py",
                    "reason": "broken today because it imports deprecated src.baselines.hsr_agent",
                },
                {
                    "path": "src/scripts/diagnostics/run_clean_navigator_v1.py::top_support",
                    "reason": "current executable baseline but not original HSR; it is support-score greedy under the clean navigator contract",
                },
                {
                    "path": "tools/legacy/src_baselines_archive/eval_hsr.py",
                    "reason": "old executable contract uses older dataset scan and B=60-style lane; not aligned with current exact136 same-case runtime",
                },
            ],
        },
        "authoritative_protocol": {
            "split": "train-only exact136 replayable same-case panel",
            "case_count": int(len(runtime["cases"])),
            "source": str(source_root / "same_case_replayable_manifest.csv"),
            "num_rounds": int(runtime["num_episodes"]),
            "actions_per_round": int(runtime["action_budget"]),
            "episode_duration_min": float(runtime["episode_duration_min"]),
            "total_budget": int(runtime["num_episodes"]) * int(runtime["action_budget"]),
            "success_definition": "SR = budgeted direct source hit, i.e. true source local/global node is explicitly sampled within the current round budget",
            "stop_rule": "terminate on source hit, empty candidate set, no valid local actions, no remaining candidates, or budget exhaustion",
            "alignment_note": "this matches the current exact136 same-case runtime contract used by recent planner/value audits rather than the older B=60 legacy HSR lane",
        },
        "minimal_fix": {
            "what": [
                "resolved legacy HSRAgent from tools/legacy/src_baselines_archive instead of deprecated src.baselines import path",
                "bound graph asset to current foundation graph from resolved config",
                "wrapped HSRAgent in current PracticalRollout same-case runtime and global/local id translation",
            ],
            "why_still_original_hsr": "the action policy remains HSRAgent.get_action_hsr_scalable(k=3); glue only restores executability under the current runtime contract",
        },
    }
    write_json(output_dir / "protocol_audit.json", protocol_audit)

    case_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for case_idx, case in enumerate(runtime["cases"], start=1):
        out = run_case_hsr(case=case, runtime=runtime, agent=agent, seed=int(args.seed))
        case_rows.append(out["case_row"])
        step_rows.extend(out["step_rows"])
        if case_idx % int(args.progress_every_cases) == 0 or case_idx == len(runtime["cases"]):
            pd.DataFrame(case_rows).to_csv(output_dir / "hsr_case_rows.partial.csv", index=False)
            pd.DataFrame(step_rows).to_csv(output_dir / "hsr_step_rows.partial.csv", index=False)
            print(
                f"[progress] case={case_idx}/{len(runtime['cases'])} "
                f"partial_sr={pd.DataFrame(case_rows)['success_rate'].mean():.4f}",
                flush=True,
            )

    case_df = pd.DataFrame(case_rows)
    step_df = pd.DataFrame(step_rows)
    case_df.to_csv(output_dir / "hsr_case_rows.csv", index=False)
    step_df.to_csv(output_dir / "hsr_step_rows.csv", index=False)

    round_curve_rows: List[Dict[str, Any]] = []
    for round_index in range(1, int(runtime["num_episodes"]) + 1):
        hit_mask = case_df["hit_round"].fillna(10**9) <= int(round_index)
        round_curve_rows.append(
            {
                "round_index": int(round_index),
                "cumulative_success_rate": float(hit_mask.mean()),
                "new_success_rate": float((case_df["hit_round"] == int(round_index)).mean()),
            }
        )
    round_curve_df = pd.DataFrame(round_curve_rows)
    round_curve_df.to_csv(output_dir / "hsr_roundwise_success_curve.csv", index=False)

    budget_curve_rows: List[Dict[str, Any]] = []
    total_budget = int(runtime["num_episodes"]) * int(runtime["action_budget"])
    for sample_budget in range(1, total_budget + 1):
        hit_mask = case_df["hit_sample_index"].fillna(10**9) <= int(sample_budget)
        budget_curve_rows.append(
            {
                "sample_budget": int(sample_budget),
                "cumulative_success_rate": float(hit_mask.mean()),
            }
        )
    budget_curve_df = pd.DataFrame(budget_curve_rows)
    budget_curve_df.to_csv(output_dir / "hsr_budget_success_curve.csv", index=False)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "cache_version": cache_dir.name,
        "foundation_graph_path": str(foundation_graph_path),
        "evaluated_case_count": int(len(case_df)),
        "sr": float(case_df["success_rate"].mean()),
        "avg_hit_round_conditional": (
            float(case_df.loc[case_df["success_rate"] > 0.5, "hit_round"].mean())
            if (case_df["success_rate"] > 0.5).any()
            else None
        ),
        "avg_hit_sample_conditional": (
            float(case_df.loc[case_df["success_rate"] > 0.5, "hit_sample_index"].mean())
            if (case_df["success_rate"] > 0.5).any()
            else None
        ),
        "final_source_in_candidate_set_rate": (
            float(case_df["final_source_in_candidate_set"].dropna().mean())
            if case_df["final_source_in_candidate_set"].notna().any()
            else None
        ),
        "final_candidate_fraction_mean": float(case_df["final_candidate_fraction"].mean()),
        "final_topk_contains_source_rate": float(case_df["final_topk_contains_source"].mean()),
        "budget_used_mean": float(case_df["budget_used"].mean()),
        "termination_reason_counts": case_df["termination_reason"].value_counts().to_dict(),
        "authoritative_protocol": protocol_audit["authoritative_protocol"],
        "minimal_aux_metrics": [
            "avg_hit_round_conditional",
            "hsr_roundwise_success_curve",
            "final_source_in_candidate_set_rate",
            "final_candidate_fraction_mean",
        ],
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
