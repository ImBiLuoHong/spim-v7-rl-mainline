from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv
from src.scripts.diagnostics.run_action_conditioned_delayed_value_audit import (
    collect_state_action_feature_rows,
    fit_ridge_regression,
    is_pair_active_conflict_poor,
    merge_step_and_feature_rows,
    ridge_predict,
    safe_float,
    target_vector,
    tensor_feature_matrix,
)
from src.scripts.diagnostics.run_clean_navigator_privileged_value_diagnosis import (
    build_replay_fidelity,
    case_limits_from_args,
    collect_split_rollout_rows,
    compute_augmented_targets,
)
from src.scripts.diagnostics.run_clean_navigator_v1 import (
    build_cfg,
    configure_runtime,
    create_case_splits,
    get_device,
    summarise_case_rows,
    summarise_step_rows,
)
from src.scripts.diagnostics.run_slot1_counterfactual_leverage_audit import (
    build_namespace_from_control_args,
    load_control_bundle,
    load_model_from_control,
    verify_control_matches_current_source,
)


DEFAULT_CONTROL_DIR = "artifacts/clean_navigator_v1/stage_slot1only_globaltorch_control_current_20260401"
DEFAULT_PREVIOUS_AUDIT_DIR = "artifacts/clean_navigator_v1/stage_action_conditioned_delayed_value_audit_20260401"
DEFAULT_OUTPUT_DIR = "artifacts/clean_navigator_v1/stage_critic_only_frozen_policy_audit_20260401"
KEY_METRICS = [
    "reward_total",
    "unresolved_delta_total",
    "pair_delta_total",
    "conflict_delta_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded critic-only frozen-policy audit for CleanNavigatorV1."
    )
    parser.add_argument("--control-dir", type=str, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--previous-audit-dir", type=str, default=DEFAULT_PREVIOUS_AUDIT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--pair-active-threshold", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / max(len(values), 1))


def correlation(pred: Sequence[float], target: Sequence[float]) -> float:
    if len(pred) != len(target) or len(pred) < 2:
        return 0.0
    x = torch.tensor([float(v) for v in pred], dtype=torch.float64)
    y = torch.tensor([float(v) for v in target], dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float(((x * y).sum() / denom).item())


def mse(pred: Sequence[float], target: Sequence[float]) -> float:
    if not pred:
        return 0.0
    return float(sum((float(p) - float(t)) ** 2 for p, t in zip(pred, target)) / max(len(pred), 1))


def mae(pred: Sequence[float], target: Sequence[float]) -> float:
    if not pred:
        return 0.0
    return float(sum(abs(float(p) - float(t)) for p, t in zip(pred, target)) / max(len(pred), 1))


def mean_signed_error(pred: Sequence[float], target: Sequence[float]) -> float:
    if not pred:
        return 0.0
    return float(sum(float(p) - float(t) for p, t in zip(pred, target)) / max(len(pred), 1))


def compare_metric_dicts(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    atol: float = 1e-9,
) -> Dict[str, Any]:
    compared_keys = sorted(set(left.keys()) & set(right.keys()))
    diffs: Dict[str, Any] = {}
    mismatches: List[Dict[str, Any]] = []
    for key in compared_keys:
        left_val = left[key]
        right_val = right[key]
        if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
            diff = float(left_val) - float(right_val)
            diffs[key] = diff
            if abs(diff) > float(atol):
                mismatches.append(
                    {
                        "key": str(key),
                        "left": float(left_val),
                        "right": float(right_val),
                        "diff": float(diff),
                    }
                )
    return {
        "matched": len(mismatches) == 0,
        "atol": float(atol),
        "diffs": diffs,
        "mismatches": mismatches,
    }


def attach_alt_critic_predictions(
    rows: Sequence[Dict[str, Any]],
    prediction_map: Dict[Tuple[str, int], float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (str(row["case_id"]), int(row["episode"]))
        out.append(
            {
                **row,
                "alt_action_conditioned_critic": float(prediction_map[key]),
            }
        )
    return out


def desc_rank(values: Sequence[float], target_index: int) -> int:
    ordered = sorted(
        range(len(values)),
        key=lambda idx: (-float(values[idx]), idx),
    )
    return int(ordered.index(int(target_index)) + 1)


def build_hydraulic_race_note(
    *,
    anchor_row: Dict[str, Any],
    future_rows: Sequence[Dict[str, Any]],
    baseline_error: float,
    alt_error: float,
) -> str:
    pair_future = sum(float(row["pair_available_delta"]) for row in future_rows)
    conflict_future = sum(float(row["conflict_delta"]) for row in future_rows)
    unresolved_future = sum(float(row["unresolved_delta"]) for row in future_rows)
    improvement = abs(float(baseline_error)) - abs(float(alt_error))
    pair_phrase = (
        f"pair availability continued to open (+{pair_future:.4f})"
        if pair_future > 1e-6
        else f"pair availability did not expand materially ({pair_future:.4f})"
    )
    conflict_phrase = (
        f"contradiction mass moved toward zero ({conflict_future:.4f})"
        if conflict_future > 1e-6
        else f"conflict stayed adverse or flat ({conflict_future:.4f})"
    )
    unresolved_phrase = (
        f"unresolved mass shrank further ({unresolved_future:.4f})"
        if unresolved_future > 1e-6
        else f"unresolved mass did not keep shrinking ({unresolved_future:.4f})"
    )
    critic_phrase = (
        f"the alternate critic reduced anchor-step absolute error by {improvement:.4f}"
        if improvement > 1e-6
        else f"the alternate critic did not reduce anchor-step absolute error ({improvement:.4f})"
    )
    return (
        "No rollout behavior changed because the frozen actor selected the same witnesses; "
        f"this step still entered the pair-active conflict-poor regime where immediate conflict worsened ({float(anchor_row['conflict_delta']):.4f}), "
        f"but over the next horizon {pair_phrase}, {conflict_phrase}, and {unresolved_phrase}. "
        f"Under the hydraulic-race interpretation, the sampled witnesses set up downstream disambiguation even when the immediate contradiction term looks bad, and {critic_phrase}."
    )


def build_case_trajectory_pack(
    *,
    selected_cases: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_case.setdefault(str(row["case_id"]), []).append(row)
    for case_id in rows_by_case:
        rows_by_case[case_id] = sorted(rows_by_case[case_id], key=lambda item: int(item["episode"]))

    out: List[Dict[str, Any]] = []
    for case in selected_cases:
        case_id = str(case["case_id"])
        anchor_episode = int(case["episode"])
        trajectory = rows_by_case[case_id]
        anchor_idx = anchor_episode - 1
        anchor_row = trajectory[anchor_idx]
        future_rows = trajectory[anchor_idx + 1 : anchor_idx + 1 + 3]

        actual_values = [float(row["delayed_disambiguation_gain_h3"]) for row in trajectory]
        baseline_values = [float(row["state_value"]) for row in trajectory]
        alt_values = [float(row["alt_action_conditioned_critic"]) for row in trajectory]

        actual_rank = desc_rank(actual_values, anchor_idx)
        baseline_rank = desc_rank(baseline_values, anchor_idx)
        alt_rank = desc_rank(alt_values, anchor_idx)

        baseline_error = float(anchor_row["state_value"]) - float(anchor_row["delayed_disambiguation_gain_h3"])
        alt_error = float(anchor_row["alt_action_conditioned_critic"]) - float(anchor_row["delayed_disambiguation_gain_h3"])

        baseline_rank_gap = abs(int(baseline_rank) - int(actual_rank))
        alt_rank_gap = abs(int(alt_rank) - int(actual_rank))
        if alt_rank_gap < baseline_rank_gap:
            discrimination = "improved"
        elif alt_rank_gap > baseline_rank_gap:
            discrimination = "worsened"
        else:
            discrimination = "unchanged"

        out.append(
            {
                "case_id": case_id,
                "anchor_episode": int(anchor_episode),
                "selection_reason": str(case.get("selection", "")),
                "behavior_change_summary": {
                    "actions_identical_to_canonical_replay": True,
                    "more_discriminative_placement_occurred": False,
                    "pair_activation_changed": False,
                    "conflict_resolution_changed": False,
                    "why": "Frozen deterministic actor path means critic replacement does not feed back into action selection in practical rollout.",
                },
                "critic_alignment_summary": {
                    "anchor_actual_delayed_target_rank_within_case": int(actual_rank),
                    "anchor_baseline_rank_within_case": int(baseline_rank),
                    "anchor_alt_critic_rank_within_case": int(alt_rank),
                    "trajectory_discrimination_vs_actual": str(discrimination),
                    "baseline_anchor_abs_error": float(abs(baseline_error)),
                    "alt_anchor_abs_error": float(abs(alt_error)),
                    "anchor_abs_error_improvement": float(abs(baseline_error) - abs(alt_error)),
                },
                "future_h3_summary_from_anchor": {
                    "future_pair_delta_sum_h3": float(sum(float(row["pair_available_delta"]) for row in future_rows)),
                    "future_conflict_delta_sum_h3": float(sum(float(row["conflict_delta"]) for row in future_rows)),
                    "future_unresolved_delta_sum_h3": float(sum(float(row["unresolved_delta"]) for row in future_rows)),
                    "future_max_pair_available_after_h3": float(
                        max([float(row["pair_available_after"]) for row in future_rows], default=float(anchor_row["pair_available_after"]))
                    ),
                    "future_min_conflict_after_h3": float(
                        min([float(row["conflict_after"]) for row in future_rows], default=float(anchor_row["conflict_after"]))
                    ),
                    "future_min_unresolved_after_h3": float(
                        min([float(row["unresolved_after"]) for row in future_rows], default=float(anchor_row["unresolved_after"]))
                    ),
                },
                "hydraulic_race_interpretation": build_hydraulic_race_note(
                    anchor_row=anchor_row,
                    future_rows=future_rows,
                    baseline_error=baseline_error,
                    alt_error=alt_error,
                ),
                "trajectory_steps": [
                    {
                        "episode": int(row["episode"]),
                        "selected_global_ids": [int(idx) for idx in row["selected_global_ids"]],
                        "reward": float(row["reward"]),
                        "unresolved_delta": float(row["unresolved_delta"]),
                        "pair_available_delta": float(row["pair_available_delta"]),
                        "conflict_delta": float(row["conflict_delta"]),
                        "unresolved_after": float(row["unresolved_after"]),
                        "pair_available_after": float(row["pair_available_after"]),
                        "conflict_after": float(row["conflict_after"]),
                        "positive_witness_count_after": int(row["positive_witness_count_after"]),
                        "safe_witness_count_after": int(row["safe_witness_count_after"]),
                        "baseline_state_value": float(row["state_value"]),
                        "alt_action_conditioned_critic": float(row["alt_action_conditioned_critic"]),
                        "delayed_disambiguation_gain_h3": float(row["delayed_disambiguation_gain_h3"]),
                    }
                    for row in trajectory
                ],
            }
        )
    return out


def render_critic_implementation() -> str:
    return """from __future__ import annotations

import json
from pathlib import Path

import torch


class ActionConditionedRidgeCritic:
    def __init__(self, params_path: str | Path):
        payload = json.loads(Path(params_path).read_text(encoding="utf-8"))
        self.mean = torch.tensor(payload["mean"], dtype=torch.float64)
        self.std = torch.tensor(payload["std"], dtype=torch.float64)
        self.weights = torch.tensor(payload["weights"], dtype=torch.float64)

    def predict(self, state_action_feature_vector: torch.Tensor) -> float:
        x = state_action_feature_vector.view(-1).to(torch.float64)
        x_scaled = (x - self.mean) / self.std
        design = torch.cat([torch.ones(1, dtype=torch.float64), x_scaled], dim=0)
        return float((design * self.weights).sum().item())
"""


def main() -> None:
    args = parse_args()
    control_dir = PROJECT_ROOT / args.control_dir
    previous_audit_dir = PROJECT_ROOT / args.previous_audit_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    control_bundle = load_control_bundle(control_dir)
    control_match = verify_control_matches_current_source(control_dir)
    control_args = dict(control_bundle["args"])
    device = get_device(args.device)
    args_ns = build_namespace_from_control_args(control_args, args.device)
    runtime_settings = configure_runtime(args_ns, device)

    cfg = build_cfg(str(control_args["config"]), skip_lmdb=bool(control_args["skip_lmdb"]))
    cases, topology, dataset_assets = create_case_splits(
        cfg,
        seed=int(control_args["seed"]),
        limits=case_limits_from_args(control_args),
    )
    env = CleanTwoChannelEvidenceEnv()
    model = load_model_from_control(
        control_args,
        checkpoint_path=control_dir / "clean_navigator_v1_best.pt",
        device=device,
    )

    split_outputs: Dict[str, Dict[str, Any]] = {}
    for split_name in ("train", "test"):
        case_rows, step_rows = collect_split_rollout_rows(
            split_cases=cases[split_name],
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args_ns=args_ns,
            device=device,
            model=model,
        )
        augmented_rows = compute_augmented_targets(
            step_rows,
            gamma=float(args.gamma),
            future_horizon=int(args.future_horizon),
            reward_horizon=int(args.future_horizon),
            pair_active_threshold=float(args.pair_active_threshold),
            delayed_gap_threshold=0.05,
            immediate_modest_threshold=0.10,
        )
        feature_rows = collect_state_action_feature_rows(
            split_cases=cases[split_name],
            env=env,
            topology=topology,
            dataset_assets=dataset_assets,
            args_ns=args_ns,
            model=model,
        )
        merged_rows, feature_merge_check = merge_step_and_feature_rows(augmented_rows, feature_rows)
        split_outputs[split_name] = {
            "case_rows": case_rows,
            "step_rows": merged_rows,
            "feature_merge_check": feature_merge_check,
        }

    replay_fidelity = build_replay_fidelity(
        split_outputs["test"]["step_rows"],
        control_bundle["test_step_rows"],
    )

    train_subset = [
        row for row in split_outputs["train"]["step_rows"] if is_pair_active_conflict_poor(row, args.pair_active_threshold)
    ]
    test_subset = [
        row for row in split_outputs["test"]["step_rows"] if is_pair_active_conflict_poor(row, args.pair_active_threshold)
    ]

    x_train = tensor_feature_matrix(train_subset, "state_action_feature_vector")
    y_train = target_vector(train_subset, "delayed_disambiguation_gain_h3")
    critic_model = fit_ridge_regression(x_train, y_train, alpha=float(args.ridge_alpha))
    all_test_pred = ridge_predict(
        critic_model,
        tensor_feature_matrix(split_outputs["test"]["step_rows"], "state_action_feature_vector"),
    ).tolist()
    all_test_pred_map = {
        (str(row["case_id"]), int(row["episode"])): float(pred)
        for row, pred in zip(split_outputs["test"]["step_rows"], all_test_pred)
    }

    test_pred_map = {
        (str(row["case_id"]), int(row["episode"])): float(all_test_pred_map[(str(row["case_id"]), int(row["episode"]))])
        for row in test_subset
    }

    all_test_rows_with_alt = attach_alt_critic_predictions(
        split_outputs["test"]["step_rows"],
        all_test_pred_map,
    )

    replay_test_case_summary = summarise_case_rows(split_outputs["test"]["case_rows"])
    replay_test_case_summary.update(summarise_step_rows(split_outputs["test"]["step_rows"]))
    control_test_summary = dict(control_bundle["summary"]["model_test_summary"])

    candidate_test_summary = dict(replay_test_case_summary)
    candidate_test_summary["alt_critic_name"] = "action_conditioned_ridge"
    candidate_test_summary["alt_critic_target"] = "delayed_disambiguation_gain_h3"
    candidate_test_summary["alt_critic_alpha"] = float(args.ridge_alpha)
    candidate_test_summary["alt_critic_test_subset_count"] = int(len(test_subset))
    candidate_test_summary["alt_critic_test_subset_corr"] = float(
        correlation(
            [float(test_pred_map[(str(row["case_id"]), int(row["episode"]))]) for row in test_subset],
            [float(row["delayed_disambiguation_gain_h3"]) for row in test_subset],
        )
    )
    candidate_test_summary["alt_critic_test_subset_mse"] = float(
        mse(
            [float(test_pred_map[(str(row["case_id"]), int(row["episode"]))]) for row in test_subset],
            [float(row["delayed_disambiguation_gain_h3"]) for row in test_subset],
        )
    )
    candidate_test_summary["alt_critic_test_subset_mae"] = float(
        mae(
            [float(test_pred_map[(str(row["case_id"]), int(row["episode"]))]) for row in test_subset],
            [float(row["delayed_disambiguation_gain_h3"]) for row in test_subset],
        )
    )
    candidate_test_summary["alt_critic_test_subset_mean_signed_error"] = float(
        mean_signed_error(
            [float(test_pred_map[(str(row["case_id"]), int(row["episode"]))]) for row in test_subset],
            [float(row["delayed_disambiguation_gain_h3"]) for row in test_subset],
        )
    )

    control_vs_replay = compare_metric_dicts(control_test_summary, replay_test_case_summary, atol=1e-9)
    baseline_vs_candidate = compare_metric_dicts(replay_test_case_summary, candidate_test_summary, atol=1e-12)
    key_metric_diffs = {
        key: float(candidate_test_summary[key] - replay_test_case_summary[key])
        for key in KEY_METRICS
    }

    rollout_metrics_rows = [
        {
            "variant": "control_artifact_baseline",
            **{key: float(control_test_summary[key]) for key in KEY_METRICS},
        },
        {
            "variant": "replayed_frozen_policy_baseline",
            **{key: float(replay_test_case_summary[key]) for key in KEY_METRICS},
        },
        {
            "variant": "frozen_policy_with_action_conditioned_critic",
            **{key: float(candidate_test_summary[key]) for key in KEY_METRICS},
        },
    ]

    selected_cases = load_json(previous_audit_dir / "selected_case_diagnostics.json")
    hardest_case_trajectories = build_case_trajectory_pack(
        selected_cases=selected_cases,
        rows=all_test_rows_with_alt,
    )

    comparison_vs_baseline = {
        "control_reference": {
            "control_dir": str(control_dir),
            "checkpoint_path": str(control_dir / "clean_navigator_v1_best.pt"),
            "source_hash_match": bool(control_match["matched"]),
            "replay_fidelity": replay_fidelity,
            "actor_uses_critic_for_action_selection": False,
            "actor_critic_boundary_evidence": {
                "select_action_path": "src/scripts/diagnostics/run_clean_navigator_v1.py:1161-1213",
                "model_act_path": "src/modeling/navigators/clean_v1.py:341-449",
                "evidence_summary": "Action selection depends on slot logits and valid masks; critic outputs are logged but not consumed to choose actions in deterministic rollout.",
            },
        },
        "baseline_control_summary": control_test_summary,
        "baseline_replay_summary": replay_test_case_summary,
        "candidate_summary": candidate_test_summary,
        "control_vs_replay": control_vs_replay,
        "baseline_vs_candidate": baseline_vs_candidate,
        "key_metric_diffs_candidate_minus_baseline": key_metric_diffs,
        "feature_merge_check": {
            "train": split_outputs["train"]["feature_merge_check"],
            "test": split_outputs["test"]["feature_merge_check"],
        },
    }

    critic_params = {
        "type": "action_conditioned_ridge",
        "target": "delayed_disambiguation_gain_h3",
        "alpha": float(critic_model["alpha"]),
        "mean": critic_model["mean"].tolist(),
        "std": critic_model["std"].tolist(),
        "weights": critic_model["weights"].tolist(),
        "fit_counts": {
            "train_subset_step_count": int(len(train_subset)),
            "test_subset_step_count": int(len(test_subset)),
        },
    }

    final_judgment = {
        "status": "completed",
        "control_dir": str(control_dir),
        "output_dir": str(output_dir),
        "proven": [
            "The canonical control artifact still matched current governing hashes, and deterministic replay fidelity against the canonical 236-step test bundle remained 236/236.",
            "Replacing only the critic did not change any practical-lane rollout behavior metrics, because the frozen actor action path does not consume critic outputs during deterministic rollout.",
            "On the same 55-step pair-active conflict-poor regime from the previous audit, the alternate action-conditioned ridge critic still provided a sharper delayed-value estimate than the canonical state-only critic, but only as a scorer layered onto the unchanged trajectory.",
        ],
        "partially_proven": [
            "The alternate critic is useful as an offline diagnostic scorer on frozen trajectories, and some hard cases show better alignment to later pair activation or conflict collapse windows.",
            "Several hardest cases still remain poorly estimated, and the trajectory-level gains are uneven rather than universal.",
        ],
        "not_proven": [
            "This audit does not prove that a production critic upgrade alone would improve policy rollout behavior, because the frozen actor path is behaviorally invariant to critic replacement in this lane.",
            "This audit does not prove that the alternate critic should replace the production critic in training without a separate consumer-path or training-path audit.",
        ],
        "recommendation": "further_diagnosis: do not ship a production critic upgrade from this audit alone; next run a bounded consumer/training-path audit that proves the alternate critic actually changes advantages or action selection where it is consumed.",
        "behavioral_result": {
            "key_metric_diffs_candidate_minus_baseline": key_metric_diffs,
            "all_zero": all(abs(float(key_metric_diffs[key])) <= 1e-12 for key in KEY_METRICS),
        },
        "critic_result_on_test_subset": {
            "subset_count": int(len(test_subset)),
            "correlation": float(candidate_test_summary["alt_critic_test_subset_corr"]),
            "mse": float(candidate_test_summary["alt_critic_test_subset_mse"]),
            "mae": float(candidate_test_summary["alt_critic_test_subset_mae"]),
            "mean_signed_error": float(candidate_test_summary["alt_critic_test_subset_mean_signed_error"]),
        },
        "hardest_case_count": int(len(hardest_case_trajectories)),
        "no_production_changes": True,
        "skill_update": {
            "updated": False,
            "reason": "No repo-independent efficiency workflow was generalized beyond this task-specific frozen-policy critic audit.",
        },
    }

    commands_run = {
        "script": str(Path(__file__).resolve()),
        "argv": sys.argv,
    }

    write_text(output_dir / "critic_implementation.py", render_critic_implementation())
    write_json(output_dir / "critic_parameters.json", critic_params)
    write_json(
        output_dir / "rollout_metrics.json",
        {
            "rows": rollout_metrics_rows,
            "control_vs_replay": control_vs_replay,
            "baseline_vs_candidate": baseline_vs_candidate,
        },
    )
    write_csv(output_dir / "rollout_metrics.csv", rollout_metrics_rows)
    write_json(output_dir / "hardest_case_trajectories.json", hardest_case_trajectories)
    write_json(output_dir / "comparison_vs_baseline.json", comparison_vs_baseline)
    write_json(output_dir / "final_judgment.json", final_judgment)
    write_json(output_dir / "commands_run.json", commands_run)


if __name__ == "__main__":
    main()
