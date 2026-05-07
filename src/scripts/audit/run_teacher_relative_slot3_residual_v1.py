from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn

from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv
from src.scripts.audit.run_spim_regret_upper_bound_gateability_audit import (
    _compute_gate_table,
    _load_panel_teacher_audit,
    _panel_rounds_from_name,
    _run_policy_rollout,
    _summarize_panel,
)
from src.scripts.run_posterior_like_belief_audit import write_json
from src.scripts.run_spim_policy_eval_strict import build_runtime_strict
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SOURCE_ROOT,
    get_device,
    seed_everything,
)


RUNNER_VERSION = "teacher_relative_slot3_residual_v1"
STATE_FEATURES = [
    "remaining_budget",
    "candidate_count",
    "posterior_entropy",
    "mass_cover_0p7",
    "top1_mass",
    "top3_mass",
    "top1_top2_margin",
]
CANDIDATE_FEATURES = [
    "is_teacher_slot3",
    "proxy_action_mass",
    "proxy_action_gap_to_top1",
    "proxy_rank_percentile",
    "proxy_expected_positive",
    "proxy_disagreement",
    "proxy_cover_shrink",
    "proxy_entropy_drop",
]


@dataclass
class StatePack:
    state_key: str
    case_id: str
    teacher_slot3: int
    candidate_slot3: List[int]
    x: torch.Tensor
    y: int
    y_delta: torch.Tensor
    best_delta_return: float
    best_delta_success: float


class ResidualScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal teacher-relative slot3 residual learner.")
    parser.add_argument("--audit-root", type=str, default="")
    parser.add_argument("--train-audit-root", type=str, default="")
    parser.add_argument("--eval-audit-root", type=str, default="")
    parser.add_argument("--train-panel-b30", type=str, default="train_B30")
    parser.add_argument("--eval-panel-b30", type=str, default="val_B30")
    parser.add_argument("--eval-panel-b60", type=str, default="val_B60")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--teacher-family", type=str, default="hsr_soft_scenario_posterior_v3")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--state-batch-size", type=int, default=128)
    parser.add_argument("--train-case-fraction", type=float, default=0.8)
    parser.add_argument(
        "--gated-main-family",
        type=str,
        default="entropy_high",
        choices=["entropy_high", "margin_low_and_entropy_high"],
    )
    parser.add_argument(
        "--baseline-summary-root",
        type=str,
        default="artifacts/conservative_corrective_rl_v1/20260412_main_b30_v1",
    )
    parser.add_argument("--eval-runtime-split", type=str, default="auto", choices=["auto", "exact136", "train", "val", "test"])
    parser.add_argument("--eval-case-limit", type=int, default=0)
    parser.add_argument("--eval-train-max-cases", type=int, default=0)
    parser.add_argument("--eval-train-cache-version", type=str, default="")
    return parser.parse_args()


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _build_state_packs(slot3_candidate_df: pd.DataFrame) -> List[StatePack]:
    packs: List[StatePack] = []
    for state_key, g in slot3_candidate_df.groupby("state_key", sort=False):
        g = g.copy().reset_index(drop=True)
        teacher_rows = g[g["is_teacher_slot3"].astype(float) > 0.5]
        if len(teacher_rows) <= 0:
            continue
        teacher_slot = int(teacher_rows.iloc[0]["candidate_slot3_local"])
        feats: List[List[float]] = []
        candidate_slot3: List[int] = []
        deltas = g["delta_return_vs_teacher_slot3"].astype(float).tolist()
        teacher_idx = int(teacher_rows.index[0])
        best_idx = int(max(range(len(deltas)), key=lambda i: deltas[i]))
        if float(deltas[best_idx]) <= 1e-12:
            best_idx = int(teacher_idx)
        for _, r in g.iterrows():
            fv = [_safe_float(r[c]) for c in STATE_FEATURES] + [_safe_float(r[c]) for c in CANDIDATE_FEATURES]
            feats.append(fv)
            candidate_slot3.append(int(r["candidate_slot3_local"]))
        packs.append(
            StatePack(
                state_key=str(state_key),
                case_id=str(g.iloc[0]["case_id"]),
                teacher_slot3=int(teacher_slot),
                candidate_slot3=candidate_slot3,
                x=torch.tensor(feats, dtype=torch.float32),
                y=int(best_idx),
                y_delta=torch.tensor(deltas, dtype=torch.float32),
                best_delta_return=float(g["delta_return_vs_teacher_slot3"].astype(float).max()),
                best_delta_success=float(g["delta_success_vs_teacher_slot3"].astype(float).max()),
            )
        )
    return packs


def _split_cases(packs: Sequence[StatePack], seed: int, train_fraction: float) -> Tuple[List[str], List[str]]:
    case_ids = sorted({p.case_id for p in packs})
    rnd = random.Random(int(seed))
    rnd.shuffle(case_ids)
    n_train = int(round(float(train_fraction) * len(case_ids)))
    n_train = max(1, min(n_train, len(case_ids) - 1))
    train_case_ids = sorted(case_ids[:n_train])
    val_case_ids = sorted(case_ids[n_train:])
    return train_case_ids, val_case_ids


def _partition_packs(packs: Sequence[StatePack], case_ids: Sequence[str]) -> List[StatePack]:
    cset = set(case_ids)
    return [p for p in packs if p.case_id in cset]


def _state_regression_loss(model: ResidualScorer, batch: Sequence[StatePack], device: torch.device) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for p in batch:
        x = p.x.to(device)
        pred = model(x)
        target = p.y_delta.to(device)
        losses.append(nn.functional.mse_loss(pred, target))
    return torch.stack(losses).mean()


def _evaluate_loss_and_acc(model: ResidualScorer, packs: Sequence[StatePack], device: torch.device) -> Dict[str, float]:
    if len(packs) <= 0:
        return {"loss": 0.0, "acc": 0.0}
    model.eval()
    losses: List[float] = []
    correct = 0
    with torch.no_grad():
        for p in packs:
            logits = model(p.x.to(device))
            target = torch.tensor([int(p.y)], dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(logits.view(1, -1), target)
            losses.append(float(loss.item()))
            pred = int(torch.argmax(logits).item())
            correct += int(pred == int(p.y))
    return {"loss": float(sum(losses) / max(len(losses), 1)), "acc": float(correct / max(len(packs), 1))}


def _train_model(
    *,
    train_packs: Sequence[StatePack],
    val_packs: Sequence[StatePack],
    in_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[ResidualScorer, List[Dict[str, float]], Dict[str, float]]:
    model = ResidualScorer(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best = {"epoch": 0, "val_loss": float("inf"), "val_acc": 0.0}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []

    for ep in range(1, int(epochs) + 1):
        model.train()
        idx = list(range(len(train_packs)))
        random.shuffle(idx)
        train_losses: List[float] = []
        for i in range(0, len(idx), int(batch_size)):
            batch = [train_packs[j] for j in idx[i : i + int(batch_size)]]
            if not batch:
                continue
            loss = _state_regression_loss(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(float(loss.item()))

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_metrics = _evaluate_loss_and_acc(model, val_packs, device)
        row = {
            "epoch": float(ep),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
        }
        history.append(row)
        if float(val_metrics["loss"]) < float(best["val_loss"]):
            best = {"epoch": float(ep), "val_loss": float(val_metrics["loss"]), "val_acc": float(val_metrics["acc"])}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, history, best


def _predict_map_and_stats(
    *,
    model: ResidualScorer,
    slot3_candidate_df: pd.DataFrame,
    gate_map: Optional[Dict[str, bool]],
    device: torch.device,
) -> Tuple[Dict[str, int], pd.DataFrame, Dict[str, float]]:
    correction_map: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for state_key, g in slot3_candidate_df.groupby("state_key", sort=False):
            g = g.copy().reset_index(drop=True)
            teacher_rows = g[g["is_teacher_slot3"].astype(float) > 0.5]
            if len(teacher_rows) <= 0:
                continue
            teacher_slot = int(teacher_rows.iloc[0]["candidate_slot3_local"])
            feats = []
            for _, r in g.iterrows():
                feats.append([_safe_float(r[c]) for c in STATE_FEATURES] + [_safe_float(r[c]) for c in CANDIDATE_FEATURES])
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            logits = model(x).view(-1)
            pred_idx = int(torch.argmax(logits).item())
            pred_slot = int(g.iloc[pred_idx]["candidate_slot3_local"])
            gate_triggered = True if gate_map is None else bool(gate_map.get(str(state_key), False))
            applied = bool(gate_triggered and (pred_slot != teacher_slot))
            if applied:
                correction_map[str(state_key)] = int(pred_slot)
            chosen_row = g.iloc[pred_idx]
            rows.append(
                {
                    "state_key": str(state_key),
                    "case_id": str(g.iloc[0]["case_id"]),
                    "teacher_slot3_local": int(teacher_slot),
                    "pred_slot3_local": int(pred_slot),
                    "replace_predicted": float(pred_slot != teacher_slot),
                    "gate_triggered": float(gate_triggered),
                    "replace_applied": float(applied),
                    "pred_delta_return_vs_teacher_slot3": float(chosen_row["delta_return_vs_teacher_slot3"]),
                    "pred_delta_success_vs_teacher_slot3": float(chosen_row["delta_success_vs_teacher_slot3"]),
                }
            )
    decision_df = pd.DataFrame(rows)
    stats = {
        "state_count": int(len(decision_df)),
        "replace_rate_predicted": float(decision_df["replace_predicted"].mean()) if len(decision_df) else 0.0,
        "gate_trigger_rate": float(decision_df["gate_triggered"].mean()) if len(decision_df) else 0.0,
        "replace_rate_applied": float(decision_df["replace_applied"].mean()) if len(decision_df) else 0.0,
    }
    return correction_map, decision_df, stats


def _load_baseline_summary_row(summary_csv: Path, policy_name: str) -> Dict[str, float]:
    df = pd.read_csv(summary_csv)
    row = df[df["policy_name"].astype(str) == str(policy_name)]
    if len(row) <= 0:
        return {}
    r = row.iloc[0]
    budget_mean = float(r["budget_used_mean"])
    return {
        "success_rate": float(r["success_rate"]),
        "avg_hit_round_conditional": float(r["avg_hit_round_conditional"]),
        "avg_return_r0": float(float(r["success_rate"]) - budget_mean / 30.0),
        "avg_budget_used": float(budget_mean),
    }


def _panel_eval_with_map(
    *,
    source_root: Path,
    cache_dir: Path,
    teacher_family: str,
    panel_name: str,
    correction_map: Dict[str, int],
    runtime_split: str,
    eval_case_limit: int,
    eval_train_max_cases: int,
    eval_train_cache_version: str,
) -> Dict[str, Any]:
    rounds = _panel_rounds_from_name(panel_name)
    runtime, _ = build_runtime_strict(
        source_root=source_root,
        cache_dir=cache_dir,
        split=str(runtime_split),
        num_rounds=int(rounds),
        actions_per_round=3,
        train_max_cases=int(eval_train_max_cases),
        train_cache_version=str(eval_train_cache_version),
        case_limit=int(eval_case_limit),
    )
    df = _run_policy_rollout(
        runtime=runtime,
        family=str(teacher_family),
        env=CleanTwoChannelEvidenceEnv(),
        action_budget=3,
        top_source_k=8,
        include_surrogate_features=False,
        correction_slot3_map=correction_map,
    )
    return _summarize_panel(df, num_rounds=rounds, action_budget=3)


def _infer_runtime_split(panel_name: str, split_override: str) -> str:
    override = str(split_override).strip()
    if override and override != "auto":
        return override
    name = str(panel_name)
    if name.startswith("train_"):
        return "train"
    if name.startswith("val_"):
        return "val"
    if name.startswith("test_"):
        return "test"
    return "exact136"


def _load_slot3_candidate_rows(root: Path, panel_name: str) -> pd.DataFrame:
    path = root / f"{panel_name}_slot3_candidate_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing slot3 candidate rows: {path}")
    df = pd.read_csv(path)
    return df[df["policy_source"].astype(str) == "teacher"].copy()


def _resolve_roots(args: argparse.Namespace) -> Tuple[Path, Path]:
    audit_root = Path(str(args.audit_root).strip()) if str(args.audit_root).strip() else None
    train_root = Path(str(args.train_audit_root).strip()) if str(args.train_audit_root).strip() else audit_root
    eval_root = Path(str(args.eval_audit_root).strip()) if str(args.eval_audit_root).strip() else audit_root
    if train_root is None or eval_root is None:
        raise ValueError("Provide either --audit-root (single-root mode) or both --train-audit-root and --eval-audit-root.")
    return train_root, eval_root


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    random.seed(int(args.seed))
    device = get_device(str(args.device))

    train_audit_root, eval_audit_root = _resolve_roots(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_panel_b30 = str(args.train_panel_b30)
    eval_panel_b30 = str(args.eval_panel_b30)
    eval_panel_b60 = str(args.eval_panel_b60)

    train_candidates = _load_slot3_candidate_rows(train_audit_root, train_panel_b30)
    b30_candidates = _load_slot3_candidate_rows(eval_audit_root, eval_panel_b30)
    b60_candidates = None
    b60_path = eval_audit_root / f"{eval_panel_b60}_slot3_candidate_rows.csv"
    if b60_path.exists():
        b60_candidates = _load_slot3_candidate_rows(eval_audit_root, eval_panel_b60)

    packs = _build_state_packs(train_candidates)
    if len(packs) <= 0:
        raise RuntimeError("No trainable teacher slot3 states found in train panel candidate rows.")

    train_case_ids = sorted({str(v) for v in train_candidates["case_id"].astype(str).tolist()})
    eval_case_ids_b30 = sorted({str(v) for v in b30_candidates["case_id"].astype(str).tolist()})
    overlap_b30 = sorted(set(train_case_ids).intersection(set(eval_case_ids_b30)))
    overlap_b60: List[str] = []
    eval_case_ids_b60: List[str] = []
    if b60_candidates is not None:
        eval_case_ids_b60 = sorted({str(v) for v in b60_candidates["case_id"].astype(str).tolist()})
        overlap_b60 = sorted(set(train_case_ids).intersection(set(eval_case_ids_b60)))
    if overlap_b30 or overlap_b60:
        raise RuntimeError(
            f"Data leakage detected: train/eval case overlap found. overlap_b30={len(overlap_b30)}, overlap_b60={len(overlap_b60)}"
        )
    split_integrity = {
        "train_audit_root": str(train_audit_root),
        "eval_audit_root": str(eval_audit_root),
        "train_panel_b30": str(train_panel_b30),
        "eval_panel_b30": str(eval_panel_b30),
        "eval_panel_b60": str(eval_panel_b60) if b60_candidates is not None else None,
        "train_case_count": int(len(train_case_ids)),
        "eval_b30_case_count": int(len(eval_case_ids_b30)),
        "eval_b60_case_count": int(len(eval_case_ids_b60)) if b60_candidates is not None else 0,
        "train_eval_overlap_b30_count": int(len(overlap_b30)),
        "train_eval_overlap_b60_count": int(len(overlap_b60)),
        "train_eval_overlap_b30_cases": overlap_b30,
        "train_eval_overlap_b60_cases": overlap_b60,
    }
    write_json(output_dir / "split_integrity_audit.json", split_integrity)

    train_cases, val_cases = _split_cases(packs, seed=int(args.seed), train_fraction=float(args.train_case_fraction))
    train_packs = _partition_packs(packs, train_cases)
    val_packs = _partition_packs(packs, val_cases)
    in_dim = int(train_packs[0].x.size(1))

    model, train_history, best = _train_model(
        train_packs=train_packs,
        val_packs=val_packs,
        in_dim=in_dim,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.state_batch_size),
        device=device,
    )
    torch.save(model.state_dict(), output_dir / "teacher_relative_slot3_residual_v1.pt")
    pd.DataFrame(train_history).to_csv(output_dir / "train_history.csv", index=False)

    # Ungated predictions on B30/B60 panels.
    ungated_map_b30, ungated_decision_b30, ungated_stats_b30 = _predict_map_and_stats(
        model=model, slot3_candidate_df=b30_candidates, gate_map=None, device=device
    )
    b30_ungated_decisions_path = output_dir / f"{eval_panel_b30}_ungated_decisions.csv"
    b30_gated_decisions_path = output_dir / f"{eval_panel_b30}_gated_main_decisions.csv"
    b30_gate_metrics_path = output_dir / f"{eval_panel_b30}_gate_family_metrics.csv"
    ungated_decision_b30.to_csv(b30_ungated_decisions_path, index=False)

    teacher_b30_df = _load_panel_teacher_audit(eval_audit_root, eval_panel_b30)
    gate_table_b30, gate_map_full_b30, gate_thresholds_b30 = _compute_gate_table(teacher_b30_df)
    gate_name = str(args.gated_main_family)
    gated_map_b30, gated_decision_b30, gated_stats_b30 = _predict_map_and_stats(
        model=model, slot3_candidate_df=b30_candidates, gate_map=gate_map_full_b30[gate_name], device=device
    )
    gated_decision_b30.to_csv(b30_gated_decisions_path, index=False)
    gate_table_b30.to_csv(b30_gate_metrics_path, index=False)

    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    runtime_split_b30 = _infer_runtime_split(eval_panel_b30, str(args.eval_runtime_split))
    runtime_split_b60 = _infer_runtime_split(eval_panel_b60, str(args.eval_runtime_split))
    teacher_metrics_b30 = _panel_eval_with_map(
        source_root=source_root,
        cache_dir=cache_dir,
        teacher_family=str(args.teacher_family),
        panel_name=eval_panel_b30,
        correction_map={},
        runtime_split=runtime_split_b30,
        eval_case_limit=int(args.eval_case_limit),
        eval_train_max_cases=int(args.eval_train_max_cases),
        eval_train_cache_version=str(args.eval_train_cache_version),
    )
    ungated_metrics_b30 = _panel_eval_with_map(
        source_root=source_root,
        cache_dir=cache_dir,
        teacher_family=str(args.teacher_family),
        panel_name=eval_panel_b30,
        correction_map=ungated_map_b30,
        runtime_split=runtime_split_b30,
        eval_case_limit=int(args.eval_case_limit),
        eval_train_max_cases=int(args.eval_train_max_cases),
        eval_train_cache_version=str(args.eval_train_cache_version),
    )
    gated_metrics_b30 = _panel_eval_with_map(
        source_root=source_root,
        cache_dir=cache_dir,
        teacher_family=str(args.teacher_family),
        panel_name=eval_panel_b30,
        correction_map=gated_map_b30,
        runtime_split=runtime_split_b30,
        eval_case_limit=int(args.eval_case_limit),
        eval_train_max_cases=int(args.eval_train_max_cases),
        eval_train_cache_version=str(args.eval_train_cache_version),
    )
    oracle_map_b30 = {
        str(r["state_key"]): int(r["best_slot3_local"])
        for _, r in teacher_b30_df.iterrows()
        if float(r["best_delta_return_vs_teacher_slot3"]) > 1e-12
    }
    oracle_metrics_b30 = _panel_eval_with_map(
        source_root=source_root,
        cache_dir=cache_dir,
        teacher_family=str(args.teacher_family),
        panel_name=eval_panel_b30,
        correction_map=oracle_map_b30,
        runtime_split=runtime_split_b30,
        eval_case_limit=int(args.eval_case_limit),
        eval_train_max_cases=int(args.eval_train_max_cases),
        eval_train_cache_version=str(args.eval_train_cache_version),
    )
    oracle_gated_map_b30 = {
        str(r["state_key"]): int(r["best_slot3_local"])
        for _, r in teacher_b30_df.iterrows()
        if float(r["best_delta_return_vs_teacher_slot3"]) > 1e-12 and bool(gate_map_full_b30[gate_name].get(str(r["state_key"]), False))
    }
    oracle_gated_metrics_b30 = _panel_eval_with_map(
        source_root=source_root,
        cache_dir=cache_dir,
        teacher_family=str(args.teacher_family),
        panel_name=eval_panel_b30,
        correction_map=oracle_gated_map_b30,
        runtime_split=runtime_split_b30,
        eval_case_limit=int(args.eval_case_limit),
        eval_train_max_cases=int(args.eval_train_max_cases),
        eval_train_cache_version=str(args.eval_train_cache_version),
    )

    panel_outputs: Dict[str, Any] = {
        str(eval_panel_b30): {
            "teacher": teacher_metrics_b30,
            "oracle_slot3_upper_bound": {
                **oracle_metrics_b30,
                "delta_success_vs_teacher": float(oracle_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]),
                "delta_return_r0_vs_teacher": float(oracle_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]),
            },
            "ungated": {
                **ungated_metrics_b30,
                "delta_success_vs_teacher": float(ungated_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]),
                "delta_return_r0_vs_teacher": float(ungated_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]),
                "replace_rate": float(ungated_stats_b30["replace_rate_applied"]),
                "recover_fraction_success": (
                    0.0
                    if abs(oracle_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]) <= 1e-12
                    else float((ungated_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]) / (oracle_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]))
                ),
                "recover_fraction_return_r0": (
                    0.0
                    if abs(oracle_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]) <= 1e-12
                    else float((ungated_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]) / (oracle_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]))
                ),
            },
            "gated_main": {
                **gated_metrics_b30,
                "delta_success_vs_teacher": float(gated_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]),
                "delta_return_r0_vs_teacher": float(gated_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]),
                "gate_family": gate_name,
                "gate_trigger_rate": float(gated_stats_b30["gate_trigger_rate"]),
                "replace_rate": float(gated_stats_b30["replace_rate_applied"]),
                "recover_fraction_success_vs_gated_oracle": (
                    0.0
                    if abs(oracle_gated_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]) <= 1e-12
                    else float((gated_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]) / (oracle_gated_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]))
                ),
                "recover_fraction_return_r0_vs_gated_oracle": (
                    0.0
                    if abs(oracle_gated_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]) <= 1e-12
                    else float((gated_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]) / (oracle_gated_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]))
                ),
            },
            "gated_main_oracle_upper_bound": {
                **oracle_gated_metrics_b30,
                "delta_success_vs_teacher": float(oracle_gated_metrics_b30["success_rate"] - teacher_metrics_b30["success_rate"]),
                "delta_return_r0_vs_teacher": float(oracle_gated_metrics_b30["avg_return_r0"] - teacher_metrics_b30["avg_return_r0"]),
                "num_corrected_states": int(len(oracle_gated_map_b30)),
            },
            "gate_thresholds": gate_thresholds_b30,
        }
    }

    if b60_candidates is not None:
        ungated_map_b60, ungated_decision_b60, ungated_stats_b60 = _predict_map_and_stats(
            model=model, slot3_candidate_df=b60_candidates, gate_map=None, device=device
        )
        b60_ungated_decisions_path = output_dir / f"{eval_panel_b60}_ungated_decisions.csv"
        b60_gated_decisions_path = output_dir / f"{eval_panel_b60}_gated_main_decisions.csv"
        b60_gate_metrics_path = output_dir / f"{eval_panel_b60}_gate_family_metrics.csv"
        ungated_decision_b60.to_csv(b60_ungated_decisions_path, index=False)

        teacher_b60_df = _load_panel_teacher_audit(eval_audit_root, eval_panel_b60)
        gate_table_b60, gate_map_full_b60, gate_thresholds_b60 = _compute_gate_table(teacher_b60_df)
        gated_map_b60, gated_decision_b60, gated_stats_b60 = _predict_map_and_stats(
            model=model, slot3_candidate_df=b60_candidates, gate_map=gate_map_full_b60[gate_name], device=device
        )
        gated_decision_b60.to_csv(b60_gated_decisions_path, index=False)
        gate_table_b60.to_csv(b60_gate_metrics_path, index=False)

        teacher_metrics_b60 = _panel_eval_with_map(
            source_root=source_root,
            cache_dir=cache_dir,
            teacher_family=str(args.teacher_family),
            panel_name=eval_panel_b60,
            correction_map={},
            runtime_split=runtime_split_b60,
            eval_case_limit=int(args.eval_case_limit),
            eval_train_max_cases=int(args.eval_train_max_cases),
            eval_train_cache_version=str(args.eval_train_cache_version),
        )
        ungated_metrics_b60 = _panel_eval_with_map(
            source_root=source_root,
            cache_dir=cache_dir,
            teacher_family=str(args.teacher_family),
            panel_name=eval_panel_b60,
            correction_map=ungated_map_b60,
            runtime_split=runtime_split_b60,
            eval_case_limit=int(args.eval_case_limit),
            eval_train_max_cases=int(args.eval_train_max_cases),
            eval_train_cache_version=str(args.eval_train_cache_version),
        )
        gated_metrics_b60 = _panel_eval_with_map(
            source_root=source_root,
            cache_dir=cache_dir,
            teacher_family=str(args.teacher_family),
            panel_name=eval_panel_b60,
            correction_map=gated_map_b60,
            runtime_split=runtime_split_b60,
            eval_case_limit=int(args.eval_case_limit),
            eval_train_max_cases=int(args.eval_train_max_cases),
            eval_train_cache_version=str(args.eval_train_cache_version),
        )
        oracle_map_b60 = {
            str(r["state_key"]): int(r["best_slot3_local"])
            for _, r in teacher_b60_df.iterrows()
            if float(r["best_delta_return_vs_teacher_slot3"]) > 1e-12
        }
        oracle_metrics_b60 = _panel_eval_with_map(
            source_root=source_root,
            cache_dir=cache_dir,
            teacher_family=str(args.teacher_family),
            panel_name=eval_panel_b60,
            correction_map=oracle_map_b60,
            runtime_split=runtime_split_b60,
            eval_case_limit=int(args.eval_case_limit),
            eval_train_max_cases=int(args.eval_train_max_cases),
            eval_train_cache_version=str(args.eval_train_cache_version),
        )
        oracle_gated_map_b60 = {
            str(r["state_key"]): int(r["best_slot3_local"])
            for _, r in teacher_b60_df.iterrows()
            if float(r["best_delta_return_vs_teacher_slot3"]) > 1e-12 and bool(gate_map_full_b60[gate_name].get(str(r["state_key"]), False))
        }
        oracle_gated_metrics_b60 = _panel_eval_with_map(
            source_root=source_root,
            cache_dir=cache_dir,
            teacher_family=str(args.teacher_family),
            panel_name=eval_panel_b60,
            correction_map=oracle_gated_map_b60,
            runtime_split=runtime_split_b60,
            eval_case_limit=int(args.eval_case_limit),
            eval_train_max_cases=int(args.eval_train_max_cases),
            eval_train_cache_version=str(args.eval_train_cache_version),
        )

        panel_outputs[str(eval_panel_b60)] = {
            "teacher": teacher_metrics_b60,
            "oracle_slot3_upper_bound": {
                **oracle_metrics_b60,
                "delta_success_vs_teacher": float(oracle_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]),
                "delta_return_r0_vs_teacher": float(oracle_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]),
            },
            "ungated": {
                **ungated_metrics_b60,
                "delta_success_vs_teacher": float(ungated_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]),
                "delta_return_r0_vs_teacher": float(ungated_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]),
                "replace_rate": float(ungated_stats_b60["replace_rate_applied"]),
                "recover_fraction_success": (
                    0.0
                    if abs(oracle_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]) <= 1e-12
                    else float((ungated_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]) / (oracle_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]))
                ),
                "recover_fraction_return_r0": (
                    0.0
                    if abs(oracle_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]) <= 1e-12
                    else float((ungated_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]) / (oracle_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]))
                ),
            },
            "gated_main": {
                **gated_metrics_b60,
                "delta_success_vs_teacher": float(gated_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]),
                "delta_return_r0_vs_teacher": float(gated_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]),
                "gate_family": gate_name,
                "gate_trigger_rate": float(gated_stats_b60["gate_trigger_rate"]),
                "replace_rate": float(gated_stats_b60["replace_rate_applied"]),
                "recover_fraction_success_vs_gated_oracle": (
                    0.0
                    if abs(oracle_gated_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]) <= 1e-12
                    else float((gated_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]) / (oracle_gated_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]))
                ),
                "recover_fraction_return_r0_vs_gated_oracle": (
                    0.0
                    if abs(oracle_gated_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]) <= 1e-12
                    else float((gated_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]) / (oracle_gated_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]))
                ),
            },
            "gated_main_oracle_upper_bound": {
                **oracle_gated_metrics_b60,
                "delta_success_vs_teacher": float(oracle_gated_metrics_b60["success_rate"] - teacher_metrics_b60["success_rate"]),
                "delta_return_r0_vs_teacher": float(oracle_gated_metrics_b60["avg_return_r0"] - teacher_metrics_b60["avg_return_r0"]),
                "num_corrected_states": int(len(oracle_gated_map_b60)),
            },
            "gate_thresholds": gate_thresholds_b60,
        }

    baseline_root = Path(args.baseline_summary_root)
    baselines = {
        str(eval_panel_b30): {
            "teacher_greedy": _load_baseline_summary_row(baseline_root / "policy_summary.csv", "teacher"),
            "trusted_bc": _load_baseline_summary_row(baseline_root / "policy_summary.csv", "bc_student"),
            "old_rl_representative": _load_baseline_summary_row(baseline_root / "policy_summary.csv", "rl_student"),
        },
        str(eval_panel_b60): {
            "teacher_greedy": _load_baseline_summary_row(baseline_root / "policy_summary_b60.csv", "teacher"),
            "trusted_bc": _load_baseline_summary_row(baseline_root / "policy_summary_b60.csv", "bc_student"),
            "old_rl_representative": _load_baseline_summary_row(baseline_root / "policy_summary_b60.csv", "rl_student"),
        },
    }

    summary = {
        "runner_version": RUNNER_VERSION,
        "seed": int(args.seed),
        "device": str(device),
        "fixed_contract": {
            "posterior_family": "hsr_soft_scenario_posterior_v3",
            "teacher_policy": "posterior_greedy",
            "trusted_bc_checkpoint": "artifacts/spim_mainline_lock_confirmation/20260411_trainfull4823_val_b30b60_v1/prep_bc/bc_teacher_warm_start.pt",
            "candidate_pool": "posterior_topk_legal_unsampled_slot3_top6",
            "gated_main_family": gate_name,
            "actions_per_round": 3,
            "eval_runtime_split_b30": runtime_split_b30,
            "eval_runtime_split_b60": runtime_split_b60 if b60_candidates is not None else None,
            "eval_case_limit": int(args.eval_case_limit),
        },
        "train": {
            "total_states": int(len(packs)),
            "train_states": int(len(train_packs)),
            "val_states": int(len(val_packs)),
            "train_case_count": int(len(train_cases)),
            "val_case_count": int(len(val_cases)),
            "state_feature_names": STATE_FEATURES,
            "candidate_feature_names": CANDIDATE_FEATURES,
            "best_checkpoint": best,
            "history_csv": str(output_dir / "train_history.csv"),
            "checkpoint": str(output_dir / "teacher_relative_slot3_residual_v1.pt"),
        },
        "split_integrity": split_integrity,
        "panel_results": panel_outputs,
        "baseline_reference": baselines,
        "artifacts": {
            "b30_ungated_decisions": str(b30_ungated_decisions_path),
            "b30_gated_main_decisions": str(b30_gated_decisions_path),
            "b30_gate_family_metrics": str(b30_gate_metrics_path),
            "b60_ungated_decisions": str(b60_ungated_decisions_path) if b60_candidates is not None else None,
            "b60_gated_main_decisions": str(b60_gated_decisions_path) if b60_candidates is not None else None,
            "b60_gate_family_metrics": str(b60_gate_metrics_path) if b60_candidates is not None else None,
            "split_integrity_audit": str(output_dir / "split_integrity_audit.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
