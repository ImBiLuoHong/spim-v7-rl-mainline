from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.navigators.clean_v1 import CleanNavigatorV1
from src.scripts.diagnostics.run_clean_navigator_v1 import (
    NODE_FEATURE_NAMES,
    build_cfg,
    build_governing_file_hashes,
    build_state_bundle,
    compute_clean_transition_metrics,
    compute_pairwise_action_distance,
    compute_witness_pair_stats,
    configure_runtime,
    create_case_splits,
    get_device,
    resolve_source_local_idx,
    select_action,
)
from src.modeling.navigators.clean_v1 import compute_mean_pairwise_jaccard_overlap, compute_mean_pairwise_overlap
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout


DEFAULT_CONTROL_DIR = "artifacts/clean_navigator_v1/stage_slot1only_globaltorch_control_current_20260401"
DEFAULT_OUTPUT_DIR = "artifacts/clean_navigator_v1/stage_slot1_counterfactual_audit_20260401"
GOVERNING_PATHS = [
    "configs/evidence_v1/formal_campaign/official_clean_ref.yaml",
    "src/scripts/diagnostics/run_clean_navigator_v1.py",
    "src/modeling/navigators/clean_v1.py",
    "src/data/v6/loader.py",
    "src/modeling/evidence/two_channel_clean.py",
    "src/scripts/audit/utils_practical_rollout.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded slot-1 frontier counterfactual leverage audit.")
    parser.add_argument("--control-dir", type=str, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--failure-case-count", type=int, default=6)
    parser.add_argument("--success-case-count", type=int, default=2)
    parser.add_argument("--low-yield-case-count", type=int, default=2)
    parser.add_argument("--feature-topk", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--material-reward-gain", type=float, default=0.02)
    parser.add_argument("--material-conflict-gain", type=float, default=0.01)
    parser.add_argument("--material-unresolved-gain", type=float, default=0.02)
    parser.add_argument("--fidelity-atol", type=float, default=1e-6)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_governing_hashes() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for rel_path in GOVERNING_PATHS:
        path = PROJECT_ROOT / rel_path
        rows.append({"path": rel_path, "sha256": sha256_file(path)})
    return rows


def verify_control_matches_current_source(control_dir: Path) -> Dict[str, Any]:
    artifact_hashes = load_json(control_dir / "governing_file_hashes.json")
    current_hashes = current_governing_hashes()
    mismatches = []
    for current in current_hashes:
        matching = next((row for row in artifact_hashes if row["path"] == current["path"]), None)
        if matching is None or matching.get("sha256") != current["sha256"]:
            mismatches.append(
                {
                    "path": current["path"],
                    "artifact_sha256": None if matching is None else matching.get("sha256"),
                    "current_sha256": current["sha256"],
                }
            )
    return {
        "control_dir": str(control_dir),
        "matched": len(mismatches) == 0,
        "mismatches": mismatches,
        "current_hashes": current_hashes,
        "artifact_hashes": artifact_hashes,
    }


def load_control_bundle(control_dir: Path) -> Dict[str, Any]:
    summary = load_json(control_dir / "summary.json")
    return {
        "control_dir": str(control_dir),
        "summary": summary,
        "args": summary["args"],
        "runtime_settings": summary.get("runtime_settings", {}),
        "test_case_rows": load_jsonl(control_dir / "model_test_case_rows.jsonl"),
        "test_step_rows": load_jsonl(control_dir / "model_test_step_rows.jsonl"),
        "reproducibility_manifest": load_json(control_dir / "reproducibility_manifest.json"),
        "checkpoint_path": str(control_dir / "clean_navigator_v1_best.pt"),
    }


def load_failure_case_counts() -> Counter[str]:
    counter: Counter[str] = Counter()
    for rel_path in [
        "artifacts/clean_navigator_v1/stage_roleaware_globaltorch_audit_20260401/selected_failure_cases.json",
        "artifacts/clean_navigator_v1/stage_slot1only_globaltorch_audit_20260401/selected_failure_cases.json",
    ]:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        payload = load_json(path)
        for candidate_cases in payload.values():
            for row in candidate_cases:
                counter[str(row["case_id"])] += 1
    return counter


def choose_audit_cases(control_case_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    case_map = {str(row["case_id"]): row for row in control_case_rows}
    failure_counts = load_failure_case_counts()
    selected: List[Dict[str, Any]] = []
    used: set[str] = set()

    recurrent_failures = sorted(
        [case_id for case_id in failure_counts if case_id in case_map],
        key=lambda case_id: (-failure_counts[case_id], float(case_map[case_id]["conflict_delta_total"]), -float(case_map[case_id]["reward_total"]), case_id),
    )
    for case_id in recurrent_failures[: int(args.failure_case_count)]:
        selected.append(
            {
                "case_id": case_id,
                "category": "recurrent_failure",
                "selection_reason": f"appeared {failure_counts[case_id]} times in prior complementarity failure packs",
            }
        )
        used.add(case_id)

    success_candidates = [
        row
        for row in control_case_rows
        if str(row["case_id"]) not in used
        and float(row["reward_total"]) > 0.0
        and float(row["conflict_delta_total"]) >= -0.01
    ]
    success_candidates = sorted(
        success_candidates,
        key=lambda row: (-float(row["reward_total"]), -float(row["unresolved_delta_total"]), str(row["case_id"])),
    )
    for row in success_candidates[: int(args.success_case_count)]:
        selected.append(
            {
                "case_id": str(row["case_id"]),
                "category": "success_anchor",
                "selection_reason": "strong canonical case with near-nonnegative conflict handling",
            }
        )
        used.add(str(row["case_id"]))

    low_yield_candidates = [
        row
        for row in control_case_rows
        if str(row["case_id"]) not in used
        and float(row["reward_total"]) > 0.0
    ]
    low_yield_candidates = sorted(
        low_yield_candidates,
        key=lambda row: (float(row["reward_total"]), float(row["conflict_delta_total"]), str(row["case_id"])),
    )
    for row in low_yield_candidates[: int(args.low_yield_case_count)]:
        selected.append(
            {
                "case_id": str(row["case_id"]),
                "category": "low_yield",
                "selection_reason": "low-yield canonical case with nonzero signal",
            }
        )
        used.add(str(row["case_id"]))
    return selected


def choose_audit_steps(
    audit_cases: Sequence[Dict[str, Any]],
    control_step_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    step_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in control_step_rows:
        step_map[str(row["case_id"])].append(row)

    selected_steps: List[Dict[str, Any]] = []
    for case_row in audit_cases:
        case_id = str(case_row["case_id"])
        rows = step_map.get(case_id, [])
        if not rows:
            continue
        if case_row["category"] == "success_anchor":
            chosen = max(
                rows,
                key=lambda row: (
                    float(row["reward"]),
                    float(row["unresolved_delta"]),
                    float(row["conflict_delta"]),
                    -int(row["episode"]),
                ),
            )
            step_reason = "highest one-step canonical reward in a success-anchor case"
        else:
            chosen = min(
                rows,
                key=lambda row: (
                    float(row["conflict_delta"]),
                    float(row["unresolved_delta"]),
                    -float(row["pair_available_delta"]),
                    int(row["episode"]),
                ),
            )
            step_reason = "worst one-step canonical conflict/disambiguation step in this case"
        selected_steps.append({**case_row, "episode": int(chosen["episode"]), "step_selection_reason": step_reason, "artifact_step_row": chosen})
    return selected_steps


def build_namespace_from_control_args(control_args: Dict[str, Any], device_arg: str) -> SimpleNamespace:
    payload = dict(control_args)
    payload["device"] = device_arg if device_arg != "auto" else str(control_args.get("device", "cuda"))
    return SimpleNamespace(**payload)


def load_model_from_control(control_args: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> CleanNavigatorV1:
    model = CleanNavigatorV1(
        node_feature_dim=len(NODE_FEATURE_NAMES),
        graph_feature_dim=6,
        hidden_dim=int(control_args["hidden_dim"]),
        num_layers=int(control_args["num_layers"]),
        num_slots=int(control_args["action_budget"]),
        greedy_eval=True,
        role_mode=str(control_args["role_mode"]),
        role_bias_weight=float(control_args["role_bias_weight"]),
        diversity_mode=str(control_args.get("diversity_mode", "none")),
        diversity_penalty_weight=float(control_args.get("diversity_penalty_weight", 0.0)),
        complementarity_mode=str(control_args.get("complementarity_mode", "none")),
        complementarity_penalty_weight=float(control_args.get("complementarity_penalty_weight", 0.0)),
        credit_mode=str(control_args.get("credit_mode", "state_value")),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def sort_valid_topk(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    exclude: Iterable[int],
    topk: int,
) -> List[int]:
    excluded = {int(idx) for idx in exclude}
    candidates = torch.nonzero(valid_mask.view(-1).bool(), as_tuple=True)[0].tolist()
    ordered = sorted(
        [int(idx) for idx in candidates if int(idx) not in excluded],
        key=lambda idx: (-float(scores[int(idx)].item()), int(idx)),
    )
    return ordered[: max(int(topk), 0)]


def compute_anchor_pair_jaccard(
    witness_pair_signature: torch.Tensor,
    left_idx: int,
    right_idx: int,
) -> float:
    if witness_pair_signature.numel() == 0 or witness_pair_signature.size(1) == 0:
        return 0.0
    left = (witness_pair_signature[int(left_idx)].float() > 1e-6).float()
    right = (witness_pair_signature[int(right_idx)].float() > 1e-6).float()
    intersection = float((left * right).sum().item())
    union = float(((left + right) > 0.0).float().sum().item())
    return intersection / union if union > 0.0 else 0.0


def generate_slot1_candidates(
    pre_state: Dict[str, Any],
    canonical_selected: Sequence[int],
    *,
    feature_topk: int,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    slot0 = int(canonical_selected[0])
    slot1 = int(canonical_selected[1])
    slot2 = int(canonical_selected[2])
    valid_mask = pre_state["valid_mask"].view(-1).bool()
    excluded = {slot0, slot2}
    feature_lists = {
        "canonical_slot1": [slot1],
        "frontier_role_potential": sort_valid_topk(pre_state["role_potentials"][:, 1], valid_mask, exclude=excluded, topk=feature_topk),
        "conflict_mass": sort_valid_topk(pre_state["derived"]["conflict_mass"], valid_mask, exclude=excluded, topk=feature_topk),
        "unresolved_mass": sort_valid_topk(pre_state["derived"]["unresolved_mass"], valid_mask, exclude=excluded, topk=feature_topk),
        "pair_available": sort_valid_topk(pre_state["pair_available"], valid_mask, exclude=excluded, topk=feature_topk),
    }
    reasons: Dict[int, set[str]] = defaultdict(set)
    for reason, idxs in feature_lists.items():
        for idx in idxs:
            reasons[int(idx)].add(reason)
    ordered = sorted(
        reasons.keys(),
        key=lambda idx: (
            -len(reasons[idx]),
            -float(pre_state["role_potentials"][int(idx), 1].item()),
            -float(pre_state["derived"]["conflict_mass"][int(idx)].item()),
            -float(pre_state["derived"]["unresolved_mass"][int(idx)].item()),
            -float(pre_state["pair_available"][int(idx)].item()),
            int(idx),
        ),
    )
    ordered = ordered[: max(int(max_candidates), 1)]
    if slot1 not in ordered:
        ordered = [slot1] + [idx for idx in ordered if idx != slot1]
        ordered = ordered[: max(int(max_candidates), 1)]
    rows: List[Dict[str, Any]] = []
    for idx in ordered:
        rows.append({"slot1_index": int(idx), "reasons": sorted(reasons[int(idx)])})
    return rows


def candidate_feature_row(
    *,
    pre_state: Dict[str, Any],
    canonical_selected: Sequence[int],
    slot1_idx: int,
) -> Dict[str, float]:
    slot0 = int(canonical_selected[0])
    slot2 = int(canonical_selected[2])
    return {
        "slot1_support_score": float(pre_state["support_score"][int(slot1_idx)].item()),
        "slot1_contradiction_score": float(pre_state["contradiction_score"][int(slot1_idx)].item()),
        "slot1_conflict_mass": float(pre_state["derived"]["conflict_mass"][int(slot1_idx)].item()),
        "slot1_unresolved_mass": float(pre_state["derived"]["unresolved_mass"][int(slot1_idx)].item()),
        "slot1_pair_available": float(pre_state["pair_available"][int(slot1_idx)].item()),
        "slot1_top_pair_margin": float(pre_state["top_pair_margin"][int(slot1_idx)].item()),
        "slot1_eligible_safe_witness_count": float(pre_state["eligible_safe_witness_count"][int(slot1_idx)].item()),
        "slot1_anchor_role_potential": float(pre_state["role_potentials"][int(slot1_idx), 0].item()),
        "slot1_frontier_role_potential": float(pre_state["role_potentials"][int(slot1_idx), 1].item()),
        "slot1_pair_role_potential": float(pre_state["role_potentials"][int(slot1_idx), 2].item()),
        "slot1_anchor_witness_pair_jaccard": float(compute_anchor_pair_jaccard(pre_state["witness_pair_signature"], slot0, int(slot1_idx))),
        "slot1_pair_witness_pair_jaccard": float(compute_anchor_pair_jaccard(pre_state["witness_pair_signature"], slot2, int(slot1_idx))),
    }


def evaluate_slot1_counterfactual(
    *,
    rollout: PracticalRollout,
    history: ObservationWitnessHistory,
    pre_state: Dict[str, Any],
    selected_indices: Sequence[int],
    slot1_idx: int,
    env: CleanTwoChannelEvidenceEnv,
    topology: Any,
    args_ns: SimpleNamespace,
    device: torch.device,
) -> Dict[str, Any]:
    cf_rollout = deepcopy(rollout)
    cf_history = deepcopy(history)
    action_set = [int(selected_indices[0]), int(slot1_idx), int(selected_indices[2])]
    cf_rollout.step_with_actions(action_set, sample_types=["slot_0", "slot_1", "slot_2"])
    if cf_rollout.history_steps:
        cf_history.append_from_history_step(cf_rollout.history_steps[-1])
    post_state = build_state_bundle(
        rollout=cf_rollout,
        history=cf_history,
        env=env,
        topology=topology,
        num_episodes=int(args_ns.num_episodes),
        action_budget=int(args_ns.action_budget),
        frontier_role_mode=str(args_ns.frontier_role_mode),
    )
    metrics = compute_clean_transition_metrics(pre_state, post_state)
    witness_pair_stats = compute_witness_pair_stats(action_set, pre_state["witness_pair_signature"])
    evidence_overlap = compute_mean_pairwise_overlap(action_set, pre_state["redundancy_signature"])
    witness_pair_overlap = compute_mean_pairwise_jaccard_overlap(action_set, pre_state["witness_pair_signature"])
    pairwise_distance = compute_pairwise_action_distance(
        env=env,
        phys_ctx=pre_state["phys_ctx"],
        selected_indices=action_set,
        num_nodes=cf_rollout.num_nodes,
        device=device,
    )
    metrics.update(
        {
            "selected_pairwise_distance": float(pairwise_distance),
            "selected_evidence_overlap": float(evidence_overlap),
            "selected_witness_pair_overlap": float(witness_pair_overlap),
            **witness_pair_stats,
            "positive_witness_count_after": int(post_state["positive_count"]),
            "safe_witness_count_after": int(post_state["safe_count"]),
            "final_pair_available_mean": float(post_state["pair_available"][post_state["valid_mask"]].float().mean().item()) if bool(post_state["valid_mask"].any()) else 0.0,
            "final_unresolved_mean": float(post_state["derived"]["unresolved_mass"][post_state["valid_mask"]].float().mean().item()) if bool(post_state["valid_mask"].any()) else 0.0,
        }
    )
    return metrics


def rank_of(values: Sequence[float], index: int) -> int:
    ordered = sorted(range(len(values)), key=lambda idx: (-float(values[idx]), idx))
    return int(ordered.index(int(index)) + 1)


def build_definition_payload(
    *,
    control_bundle: Dict[str, Any],
    control_match: Dict[str, Any],
    audit_steps: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "audit_name": "slot1_frontier_counterfactual_leverage_audit",
        "question_targeted": "whether canonical slot-1 frontier choices are leaving one-step conflict/disambiguation value on the table under the current strict global_torch control",
        "control_reference": {
            "control_dir": control_bundle["control_dir"],
            "checkpoint_path": control_bundle["checkpoint_path"],
            "training_sampling_mode": control_bundle["args"]["training_sampling_mode"],
            "strict_determinism": control_bundle["args"]["strict_determinism"],
            "source_hash_match": bool(control_match["matched"]),
        },
        "held_fixed": [
            "current-source strict canonical checkpoint and args",
            "exact pre-episode rollout state before the audited action set",
            "slot 0 canonical choice",
            "slot 2 canonical choice",
            "environment transition path via PracticalRollout.step_with_actions",
            "post-step scoring via compute_clean_transition_metrics",
        ],
        "recomputed": [
            "slot 1 action only",
            "post-step evidence state",
            "next-step conflict_delta / unresolved_delta / pair_delta / reward",
            "set-level overlap and witness-pair coverage diagnostics for the replacement action set",
        ],
        "candidate_generation": {
            "feature_topk": int(args.feature_topk),
            "max_candidates": int(args.max_candidates),
            "sources": [
                "actual canonical slot 1",
                "top frontier_role_potential candidates",
                "top conflict_mass candidates",
                "top unresolved_mass candidates",
                "top pair_available candidates",
            ],
            "exclusions": [
                "canonical slot 0",
                "canonical slot 2",
                "invalid nodes under the frozen pre-state valid_mask",
            ],
        },
        "audit_subset": [
            {
                "case_id": str(row["case_id"]),
                "category": str(row["category"]),
                "selection_reason": str(row["selection_reason"]),
                "episode": int(row["episode"]),
                "step_selection_reason": str(row["step_selection_reason"]),
            }
            for row in audit_steps
        ],
        "material_thresholds": {
            "reward_gain": float(args.material_reward_gain),
            "conflict_delta_gain": float(args.material_conflict_gain),
            "unresolved_delta_gain": float(args.material_unresolved_gain),
        },
    }


def definition_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Slot-1 Counterfactual Audit Definition",
        "",
        "## Question",
        f"- {payload['question_targeted']}",
        "",
        "## Control reference",
        f"- Control dir: `{payload['control_reference']['control_dir']}`",
        f"- Checkpoint path: `{payload['control_reference']['checkpoint_path']}`",
        f"- training_sampling_mode: `{payload['control_reference']['training_sampling_mode']}`",
        f"- strict_determinism: `{payload['control_reference']['strict_determinism']}`",
        f"- current-source hash match: `{payload['control_reference']['source_hash_match']}`",
        "",
        "## Held fixed",
    ]
    for item in payload["held_fixed"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Recomputed"])
    for item in payload["recomputed"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Candidate generation"])
    cg = payload["candidate_generation"]
    lines.append(f"- feature_topk: `{cg['feature_topk']}`")
    lines.append(f"- max_candidates: `{cg['max_candidates']}`")
    for item in cg["sources"]:
        lines.append(f"- source: {item}")
    for item in cg["exclusions"]:
        lines.append(f"- exclusion: {item}")
    lines.extend(["", "## Audit subset"])
    for row in payload["audit_subset"]:
        lines.append(
            f"- {row['case_id']} ({row['category']}), episode `{row['episode']}`: {row['selection_reason']}; {row['step_selection_reason']}"
        )
    lines.extend(["", "## Material thresholds"])
    for key, value in payload["material_thresholds"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def aggregate_choice_summary(
    audited_steps: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not audited_steps:
        return {"audited_step_count": 0}
    reward_ranks = [int(step["canonical_reward_rank"]) for step in audited_steps]
    conflict_ranks = [int(step["canonical_conflict_rank"]) for step in audited_steps]
    unresolved_ranks = [int(step["canonical_unresolved_rank"]) for step in audited_steps]
    candidate_counts = [int(step["candidate_count"]) for step in audited_steps]
    best_reward_gains = [float(step["best_reward_gain"]) for step in audited_steps]
    best_conflict_gains = [float(step["best_conflict_gain"]) for step in audited_steps]
    best_unresolved_gains = [float(step["best_unresolved_gain"]) for step in audited_steps]
    best_reward_conflict_gains = [float(step["best_reward_conflict_gain"]) for step in audited_steps]
    material_reward_steps = [
        step
        for step in audited_steps
        if float(step["best_reward_gain"]) >= float(args.material_reward_gain)
    ]
    return {
        "audited_step_count": int(len(audited_steps)),
        "fidelity_pass_count": int(sum(int(bool(step["fidelity_pass"])) for step in audited_steps)),
        "candidate_count_mean": float(mean(candidate_counts)),
        "canonical_reward_rank_mean": float(mean(reward_ranks)),
        "canonical_reward_rank_median": float(median(reward_ranks)),
        "canonical_reward_top1_fraction": float(sum(int(rank == 1) for rank in reward_ranks) / len(reward_ranks)),
        "canonical_reward_top3_fraction": float(sum(int(rank <= 3) for rank in reward_ranks) / len(reward_ranks)),
        "canonical_conflict_top1_fraction": float(sum(int(rank == 1) for rank in conflict_ranks) / len(conflict_ranks)),
        "canonical_unresolved_top1_fraction": float(sum(int(rank == 1) for rank in unresolved_ranks) / len(unresolved_ranks)),
        "steps_with_better_reward_alternative_fraction": float(sum(int(gain > 1e-12) for gain in best_reward_gains) / len(best_reward_gains)),
        "steps_with_material_reward_alternative_fraction": float(sum(int(gain >= float(args.material_reward_gain)) for gain in best_reward_gains) / len(best_reward_gains)),
        "steps_with_material_conflict_alternative_fraction": float(sum(int(gain >= float(args.material_conflict_gain)) for gain in best_conflict_gains) / len(best_conflict_gains)),
        "steps_with_material_unresolved_alternative_fraction": float(sum(int(gain >= float(args.material_unresolved_gain)) for gain in best_unresolved_gains) / len(best_unresolved_gains)),
        "best_reward_gain_mean": float(mean(best_reward_gains)),
        "best_conflict_gain_mean": float(mean(best_conflict_gains)),
        "best_unresolved_gain_mean": float(mean(best_unresolved_gains)),
        "best_reward_conflict_gain_mean": float(mean(best_reward_conflict_gains)),
        "material_reward_winner_nonnegative_conflict_fraction": (
            float(
                sum(
                    int(float(step["best_reward_conflict_gain"]) >= -1e-12)
                    for step in material_reward_steps
                )
                / len(material_reward_steps)
            )
            if material_reward_steps
            else 0.0
        ),
        "material_reward_winner_negative_conflict_fraction": (
            float(
                sum(
                    int(float(step["best_reward_conflict_gain"]) < -1e-12)
                    for step in material_reward_steps
                )
                / len(material_reward_steps)
            )
            if material_reward_steps
            else 0.0
        ),
    }


def infer_feature_pattern(result_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[(str(row["case_id"]), int(row["episode"]))].append(row)

    feature_names = [
        "slot1_support_score",
        "slot1_contradiction_score",
        "slot1_conflict_mass",
        "slot1_unresolved_mass",
        "slot1_pair_available",
        "slot1_top_pair_margin",
        "slot1_eligible_safe_witness_count",
        "slot1_anchor_role_potential",
        "slot1_frontier_role_potential",
        "slot1_pair_role_potential",
        "slot1_anchor_witness_pair_jaccard",
        "slot1_pair_witness_pair_jaccard",
    ]
    deltas: Dict[str, List[float]] = {name: [] for name in feature_names}
    example_rows: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        canonical = next((row for row in rows if bool(row["is_canonical_slot1"])), None)
        if canonical is None:
            continue
        better = max(rows, key=lambda row: (float(row["reward"]), float(row["conflict_delta"]), float(row["unresolved_delta"]), -int(row["slot1_index"])))
        if bool(better["is_canonical_slot1"]):
            continue
        reward_gain = float(better["reward"]) - float(canonical["reward"])
        if reward_gain < float(args.material_reward_gain):
            continue
        example_rows.append(
            {
                "case_id": key[0],
                "episode": key[1],
                "canonical_slot1_index": int(canonical["slot1_index"]),
                "better_slot1_index": int(better["slot1_index"]),
                "reward_gain": reward_gain,
                "conflict_gain": float(better["conflict_delta"]) - float(canonical["conflict_delta"]),
                "unresolved_gain": float(better["unresolved_delta"]) - float(canonical["unresolved_delta"]),
            }
        )
        for feature_name in feature_names:
            deltas[feature_name].append(float(better[feature_name]) - float(canonical[feature_name]))
    summaries = []
    for feature_name, values in deltas.items():
        if not values:
            continue
        summaries.append(
            {
                "feature": feature_name,
                "example_count": int(len(values)),
                "mean_delta": float(mean(values)),
                "median_delta": float(median(values)),
                "positive_fraction": float(sum(int(value > 1e-12) for value in values) / len(values)),
                "negative_fraction": float(sum(int(value < -1e-12) for value in values) / len(values)),
            }
        )
    summaries = sorted(summaries, key=lambda row: (-abs(float(row["mean_delta"])), row["feature"]))
    return {
        "material_reward_pattern_example_count": int(len(example_rows)),
        "feature_deltas": summaries,
        "examples": example_rows[:8],
    }


def classify_leverage(choice_summary: Dict[str, Any]) -> str:
    if not choice_summary or int(choice_summary.get("audited_step_count", 0)) <= 0:
        return "not_proven"
    if (
        float(choice_summary.get("steps_with_material_reward_alternative_fraction", 0.0)) >= 0.6
        and float(choice_summary.get("canonical_reward_top1_fraction", 1.0)) <= 0.4
        and (
            float(choice_summary.get("steps_with_material_conflict_alternative_fraction", 0.0)) >= 0.4
            or float(choice_summary.get("material_reward_winner_nonnegative_conflict_fraction", 0.0)) >= 0.5
        )
    ):
        return "A"
    if (
        float(choice_summary.get("canonical_reward_top3_fraction", 0.0)) >= 0.7
        and float(choice_summary.get("steps_with_material_reward_alternative_fraction", 1.0)) <= 0.2
        and float(choice_summary.get("steps_with_material_conflict_alternative_fraction", 1.0)) <= 0.3
    ):
        return "B"
    return "C"


def build_final_judgment(
    *,
    control_bundle: Dict[str, Any],
    control_match: Dict[str, Any],
    choice_summary: Dict[str, Any],
    feature_pattern: Dict[str, Any],
) -> Dict[str, Any]:
    classification = classify_leverage(choice_summary)
    if classification == "A":
        hard = "Slot-1 frontier choice remains a live leverage point."
        proven = "partially_proven"
        next_step = "Design one bounded frontier-ranking intervention that targets the winning counterfactual feature pattern and re-test against the same canonical control."
    elif classification == "B":
        hard = "Slot-1 frontier choice is probably not the main remaining bottleneck."
        proven = "partially_proven" if bool(control_match["matched"]) else "not_proven"
        next_step = "Keep the canonical control frozen and deprioritize frontier-choice shaping; diagnose another bottleneck class instead of extending this line."
    else:
        hard = "The slot-1 leverage answer is mixed, without enough evidence to keep blaming frontier choice alone."
        proven = "partially_proven" if bool(control_match["matched"]) else "not_proven"
        next_step = "If continuing on slot 1 at all, target the specific feature pattern from the winning counterfactuals rather than adding another broad shaping mechanism."
    return {
        "control_dir": control_bundle["control_dir"],
        "current_source_hash_match": bool(control_match["matched"]),
        "classification": classification,
        "hard_judgment": hard,
        "proven_status": proven,
        "choice_summary": choice_summary,
        "feature_pattern_example_count": int(feature_pattern.get("material_reward_pattern_example_count", 0)),
        "single_best_next_step": next_step,
    }


def judgment_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Final Judgment",
        "",
        f"- Hard judgment: {payload['hard_judgment']}",
        f"- Proven status: {payload['proven_status']}",
        f"- Classification: {payload['classification']}",
        f"- Current-source hash match: {payload['current_source_hash_match']}",
        f"- Single best next step: {payload['single_best_next_step']}",
    ]
    return "\n".join(lines) + "\n"


def nearly_equal(left: float, right: float, atol: float) -> bool:
    return abs(float(left) - float(right)) <= float(atol)


def main() -> None:
    args = parse_args()
    control_dir = PROJECT_ROOT / args.control_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    control_bundle = load_control_bundle(control_dir)
    control_match = verify_control_matches_current_source(control_dir)
    if not bool(control_match["matched"]):
        raise RuntimeError(f"Control artifact does not match current source: {control_match['mismatches']}")

    control_args = dict(control_bundle["args"])
    args_ns = build_namespace_from_control_args(control_args, args.device)
    device = get_device(str(args_ns.device))
    configure_runtime(SimpleNamespace(seed=int(args_ns.seed), strict_determinism=bool(args_ns.strict_determinism), device=str(device)), device)

    cfg = build_cfg(str(args_ns.config), skip_lmdb=bool(args_ns.skip_lmdb))
    case_limits = {
        "train": int(args_ns.train_pool_limit) if int(args_ns.train_pool_limit) > 0 else int(args_ns.max_train_cases),
        "val": int(args_ns.max_val_cases),
        "test": int(args_ns.max_test_cases),
    }
    cases, topology, dataset_assets = create_case_splits(cfg, seed=int(args_ns.seed), limits=case_limits)
    env = CleanTwoChannelEvidenceEnv()
    model = load_model_from_control(control_args, checkpoint_path=control_dir / "clean_navigator_v1_best.pt", device=device)

    control_case_rows = control_bundle["test_case_rows"]
    control_step_rows = control_bundle["test_step_rows"]
    audit_cases = choose_audit_cases(control_case_rows, args)
    audit_steps = choose_audit_steps(audit_cases, control_step_rows)

    definition = build_definition_payload(
        control_bundle=control_bundle,
        control_match=control_match,
        audit_steps=audit_steps,
        args=args,
    )
    write_json(output_dir / "slot1_counterfactual_audit_definition.json", definition)
    (output_dir / "slot1_counterfactual_audit_definition.md").write_text(definition_markdown(definition), encoding="utf-8")

    case_lookup = {str(case["case_id"]): (idx, case) for idx, case in enumerate(cases["test"])}
    result_rows: List[Dict[str, Any]] = []
    step_summaries: List[Dict[str, Any]] = []
    selected_case_diagnostics: List[Dict[str, Any]] = []

    for target in audit_steps:
        case_id = str(target["case_id"])
        if case_id not in case_lookup:
            raise RuntimeError(f"Unable to find case_id={case_id} in recreated canonical test split")
        case_idx, case = case_lookup[case_id]
        target_episode = int(target["episode"])
        artifact_step = dict(target["artifact_step_row"])
        event_data = deepcopy(case["data"])
        rollout = PracticalRollout(
            event_data=event_data,
            global_edge_index=dataset_assets["global_edge_index"],
            stt_dynamic_series=dataset_assets["stt_dynamic_series"],
            num_global_nodes=int(dataset_assets["num_global_nodes"]),
            num_episodes=int(args_ns.num_episodes),
            samples_per_episode=int(args_ns.action_budget),
            episode_duration_min=float(args_ns.episode_duration_min),
        )
        history = ObservationWitnessHistory()
        src_local = resolve_source_local_idx(rollout)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args_ns.seed) * 1000 + int(case_idx))

        audited = False
        for episode_idx in range(int(args_ns.num_episodes)):
            pre_state = build_state_bundle(
                rollout=rollout,
                history=history,
                env=env,
                topology=topology,
                num_episodes=int(args_ns.num_episodes),
                action_budget=int(args_ns.action_budget),
                frontier_role_mode=str(args_ns.frontier_role_mode),
            )
            if int(pre_state["valid_mask"].sum().item()) < int(args_ns.action_budget):
                break
            selected_indices, _model_out = select_action(
                policy_name="clean_navigator_v1",
                state=pre_state,
                model=model,
                generator=generator,
                cpu_generator=None,
                training_sampling_mode=str(args_ns.training_sampling_mode),
                deterministic=True,
            )
            selected_indices = [int(idx) for idx in selected_indices]
            if int(episode_idx + 1) != target_episode:
                rollout.step_with_actions(selected_indices, sample_types=[f"slot_{slot}" for slot in range(len(selected_indices))])
                if rollout.history_steps:
                    history.append_from_history_step(rollout.history_steps[-1])
                continue

            fidelity = {
                "selected_indices_match": list(selected_indices) == [int(idx) for idx in artifact_step["selected_indices"]],
                "reward_match": nearly_equal(artifact_step["reward"], artifact_step["reward"], args.fidelity_atol),
            }

            canonical_metrics = evaluate_slot1_counterfactual(
                rollout=rollout,
                history=history,
                pre_state=pre_state,
                selected_indices=selected_indices,
                slot1_idx=int(selected_indices[1]),
                env=env,
                topology=topology,
                args_ns=args_ns,
                device=device,
            )
            fidelity["reward_match"] = nearly_equal(canonical_metrics["reward"], artifact_step["reward"], args.fidelity_atol)
            fidelity["conflict_match"] = nearly_equal(canonical_metrics["conflict_delta"], artifact_step["conflict_delta"], args.fidelity_atol)
            fidelity["unresolved_match"] = nearly_equal(canonical_metrics["unresolved_delta"], artifact_step["unresolved_delta"], args.fidelity_atol)
            fidelity["pair_match"] = nearly_equal(canonical_metrics["pair_available_delta"], artifact_step["pair_available_delta"], args.fidelity_atol)
            fidelity_pass = all(bool(v) for v in fidelity.values())

            candidate_rows = generate_slot1_candidates(
                pre_state=pre_state,
                canonical_selected=selected_indices,
                feature_topk=int(args.feature_topk),
                max_candidates=int(args.max_candidates),
            )
            evaluated_rows: List[Dict[str, Any]] = []
            for candidate in candidate_rows:
                slot1_idx = int(candidate["slot1_index"])
                metrics = canonical_metrics if slot1_idx == int(selected_indices[1]) else evaluate_slot1_counterfactual(
                    rollout=rollout,
                    history=history,
                    pre_state=pre_state,
                    selected_indices=selected_indices,
                    slot1_idx=slot1_idx,
                    env=env,
                    topology=topology,
                    args_ns=args_ns,
                    device=device,
                )
                feature_row = candidate_feature_row(pre_state=pre_state, canonical_selected=selected_indices, slot1_idx=slot1_idx)
                evaluated_rows.append(
                    {
                        "case_id": case_id,
                        "category": str(target["category"]),
                        "episode": int(target_episode),
                        "step_selection_reason": str(target["step_selection_reason"]),
                        "selection_reason": str(target["selection_reason"]),
                        "canonical_slot0_index": int(selected_indices[0]),
                        "canonical_slot1_index": int(selected_indices[1]),
                        "canonical_slot2_index": int(selected_indices[2]),
                        "canonical_slot0_global_id": int(rollout.g_ids[int(selected_indices[0])].item()),
                        "canonical_slot1_global_id": int(rollout.g_ids[int(selected_indices[1])].item()),
                        "canonical_slot2_global_id": int(rollout.g_ids[int(selected_indices[2])].item()),
                        "slot1_index": int(slot1_idx),
                        "slot1_global_id": int(rollout.g_ids[int(slot1_idx)].item()),
                        "is_canonical_slot1": bool(slot1_idx == int(selected_indices[1])),
                        "candidate_reasons": "|".join(candidate["reasons"]),
                        "source_local_idx": None if src_local is None else int(src_local),
                        "reward": float(metrics["reward"]),
                        "conflict_delta": float(metrics["conflict_delta"]),
                        "unresolved_delta": float(metrics["unresolved_delta"]),
                        "pair_delta": float(metrics["pair_available_delta"]),
                        "support_delta": float(metrics["support_delta"]),
                        "live_delta": float(metrics["live_delta"]),
                        "selected_pairwise_distance": float(metrics["selected_pairwise_distance"]),
                        "selected_evidence_overlap": float(metrics["selected_evidence_overlap"]),
                        "selected_witness_pair_overlap": float(metrics["selected_witness_pair_overlap"]),
                        "selected_witness_pair_coverage_fraction": float(metrics["selected_witness_pair_coverage_fraction"]),
                        "selected_witness_pair_union_count": float(metrics["selected_witness_pair_union_count"]),
                        "selected_witness_pair_active_node_fraction": float(metrics["selected_witness_pair_active_node_fraction"]),
                        "final_pair_available_mean": float(metrics["final_pair_available_mean"]),
                        "final_unresolved_mean": float(metrics["final_unresolved_mean"]),
                        "fidelity_pass": bool(fidelity_pass),
                        **feature_row,
                    }
                )

            canonical_row = next(row for row in evaluated_rows if bool(row["is_canonical_slot1"]))
            reward_values = [float(row["reward"]) for row in evaluated_rows]
            conflict_values = [float(row["conflict_delta"]) for row in evaluated_rows]
            unresolved_values = [float(row["unresolved_delta"]) for row in evaluated_rows]
            canonical_idx = evaluated_rows.index(canonical_row)
            canonical_reward_rank = rank_of(reward_values, canonical_idx)
            canonical_conflict_rank = rank_of(conflict_values, canonical_idx)
            canonical_unresolved_rank = rank_of(unresolved_values, canonical_idx)
            best_reward_row = max(evaluated_rows, key=lambda row: (float(row["reward"]), float(row["conflict_delta"]), float(row["unresolved_delta"]), -int(row["slot1_index"])))
            best_conflict_row = max(evaluated_rows, key=lambda row: (float(row["conflict_delta"]), float(row["reward"]), float(row["unresolved_delta"]), -int(row["slot1_index"])))
            best_unresolved_row = max(evaluated_rows, key=lambda row: (float(row["unresolved_delta"]), float(row["reward"]), float(row["conflict_delta"]), -int(row["slot1_index"])))
            step_summary = {
                "case_id": case_id,
                "category": str(target["category"]),
                "episode": int(target_episode),
                "candidate_count": int(len(evaluated_rows)),
                "canonical_slot1_index": int(canonical_row["slot1_index"]),
                "canonical_slot1_global_id": int(canonical_row["slot1_global_id"]),
                "canonical_reward_rank": int(canonical_reward_rank),
                "canonical_conflict_rank": int(canonical_conflict_rank),
                "canonical_unresolved_rank": int(canonical_unresolved_rank),
                "best_reward_gain": float(best_reward_row["reward"] - canonical_row["reward"]),
                "best_reward_conflict_gain": float(best_reward_row["conflict_delta"] - canonical_row["conflict_delta"]),
                "best_reward_unresolved_gain": float(best_reward_row["unresolved_delta"] - canonical_row["unresolved_delta"]),
                "best_conflict_gain": float(best_conflict_row["conflict_delta"] - canonical_row["conflict_delta"]),
                "best_unresolved_gain": float(best_unresolved_row["unresolved_delta"] - canonical_row["unresolved_delta"]),
                "fidelity_pass": bool(fidelity_pass),
            }
            step_summaries.append(step_summary)
            selected_case_diagnostics.append(
                {
                    "case_id": case_id,
                    "category": str(target["category"]),
                    "selection_reason": str(target["selection_reason"]),
                    "episode": int(target_episode),
                    "step_selection_reason": str(target["step_selection_reason"]),
                    "artifact_step_row": artifact_step,
                    "fidelity": fidelity,
                    "canonical": canonical_row,
                    "best_by_reward": best_reward_row,
                    "best_by_conflict_delta": best_conflict_row,
                    "best_by_unresolved_delta": best_unresolved_row,
                    "top3_by_reward": sorted(
                        evaluated_rows,
                        key=lambda row: (float(row["reward"]), float(row["conflict_delta"]), float(row["unresolved_delta"]), -int(row["slot1_index"])),
                        reverse=True,
                    )[:3],
                }
            )
            result_rows.extend(evaluated_rows)
            audited = True
            break
        if not audited:
            raise RuntimeError(f"Failed to audit selected case step: {case_id} episode {target_episode}")

    choice_summary = aggregate_choice_summary(step_summaries, args)
    feature_pattern = infer_feature_pattern(result_rows, args)
    final_judgment = build_final_judgment(
        control_bundle=control_bundle,
        control_match=control_match,
        choice_summary=choice_summary,
        feature_pattern=feature_pattern,
    )

    write_csv(output_dir / "slot1_counterfactual_results.csv", result_rows)
    write_json(output_dir / "slot1_counterfactual_results.json", result_rows)
    write_json(output_dir / "slot1_choice_rank_summary.json", {**choice_summary, "step_summaries": step_summaries, "feature_pattern": feature_pattern})
    write_json(output_dir / "selected_case_diagnostics.json", selected_case_diagnostics)
    write_json(output_dir / "final_judgment.json", final_judgment)
    (output_dir / "final_judgment.md").write_text(judgment_markdown(final_judgment), encoding="utf-8")


if __name__ == "__main__":
    main()
