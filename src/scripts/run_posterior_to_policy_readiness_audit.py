from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.modeling.belief_updaters.evidence_posterior_like import _masked_zscore
from src.scripts.run_posterior_like_belief_acceptability_audit import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONTRAST_ROOT,
    DEFAULT_SOURCE_ROOT,
    _normalized_entropy,
    _safe_softmax,
    collect_step_payloads,
)
from src.scripts.run_posterior_like_belief_audit import load_frozen_reasoner, load_runtime_context, write_json


DEFAULT_ACCEPTABILITY_ROOT = PROJECT_ROOT / "artifacts" / "posterior_like_belief_acceptability_audit" / "20260407_exact136_belief_acceptability_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "posterior_to_policy_readiness_audit" / "20260407_exact136_policy_readiness_v1"
RUNNER_VERSION = "posterior_to_policy_readiness_audit_v1"
PANEL_VERSION = "exact136_train_only_posterior_to_policy_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posterior-to-policy readiness audit on fixed calibrated fused posterior.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--contrast-root", type=str, default=str(DEFAULT_CONTRAST_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--acceptability-root", type=str, default=str(DEFAULT_ACCEPTABILITY_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=45)
    return parser.parse_args()


def _spearman(df: pd.DataFrame, x: str, y: str) -> float | None:
    sub = df[[x, y]].dropna()
    if len(sub) < 5:
        return None
    val = sub[x].corr(sub[y], method="spearman")
    return None if pd.isna(val) else float(val)


def _get_calibrated_params(acceptability_root: Path) -> Dict[str, float]:
    summary = json.loads((acceptability_root / "summary.json").read_text())
    return dict(summary["head_definitions"]["calibrated_fused_posterior"])


def _compute_probs(row: Dict[str, Any], params: Dict[str, float]) -> torch.Tensor:
    mask = row["candidate_mask"]
    q_z = _masked_zscore(row["q_score"], mask)
    l_z = _masked_zscore(row["reasoner_logits"], mask)
    c_z = _masked_zscore(row["contrast_signal"], mask)
    d_z = _masked_zscore(row["contradiction_score"], mask)
    energy = (
        float(params["lambda_reasoner"]) * l_z
        + float(params["lambda_q"]) * q_z
        + float(params["lambda_contrast"]) * c_z
        - float(params["lambda_contradiction"]) * d_z
    )
    return _safe_softmax(energy, mask, float(params["temperature"]))


def _sorted_valid_indices(probs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_idx = mask.nonzero(as_tuple=True)[0]
    vals = probs[valid_idx]
    order = torch.argsort(vals, descending=True)
    return valid_idx[order], vals[order]


def _select_mass_cover_threshold(head_df: pd.DataFrame) -> float:
    candidates = [0.5, 0.6, 0.7]
    best_mass = candidates[0]
    best_score = -1e9
    for mass in candidates:
        col = f"mass_cover_{int(mass*100)}_ratio"
        src = f"mass_cover_{int(mass*100)}_source_cover"
        hard = f"mass_cover_{int(mass*100)}_hard_cover"
        score = float(head_df[src].mean()) + 0.5 * float(head_df[hard].mean()) - float(head_df[col].mean())
        if score > best_score:
            best_score = score
            best_mass = mass
    return float(best_mass)


def _summary_from_probs(
    probs: torch.Tensor,
    mask: torch.Tensor,
    source_local: int | None,
    hard_local: int | None,
    *,
    mass_cover_threshold: float,
    margin_band: float,
    eff_support_scale: float,
) -> Dict[str, Dict[str, float]]:
    sorted_idx, sorted_vals = _sorted_valid_indices(probs, mask)
    candidate_count = int(mask.float().sum().item())
    top1_prob = float(sorted_vals[0].item()) if len(sorted_vals) else 0.0
    log_vals = torch.log(sorted_vals.clamp_min(1e-12))
    cumsum = torch.cumsum(sorted_vals, dim=0)

    mass_idx = int((cumsum >= mass_cover_threshold).nonzero(as_tuple=True)[0][0].item()) + 1 if len(sorted_vals) else 0
    mass_set = set(sorted_idx[:mass_idx].tolist())
    src_in_mass = float(source_local is not None and int(source_local) in mass_set)
    hard_in_mass = float(hard_local is not None and int(hard_local) in mass_set)

    band_mask = (log_vals >= float(log_vals[0].item()) - margin_band) if len(log_vals) else torch.zeros(0, dtype=torch.bool)
    band_idx = sorted_idx[band_mask]
    band_set = set(band_idx.tolist())
    src_in_band = float(source_local is not None and int(source_local) in band_set)
    hard_in_band = float(hard_local is not None and int(hard_local) in band_set)

    entropy, _ = _normalized_entropy(probs, mask)
    eff_k = max(1, min(candidate_count, int(math.ceil(math.exp(entropy) * eff_support_scale))))
    eff_set = set(sorted_idx[:eff_k].tolist())
    src_in_eff = float(source_local is not None and int(source_local) in eff_set)
    hard_in_eff = float(hard_local is not None and int(hard_local) in eff_set)

    def pack(summary_set: set[int], src_cover: float, hard_cover: float, name: str) -> Dict[str, float]:
        size = float(len(summary_set))
        summary_mass = float(probs[list(summary_set)].sum().item()) if summary_set else 0.0
        confuser_mass = float(summary_mass - top1_prob)
        return {
            f"{name}_size": size,
            f"{name}_size_ratio": float(size / candidate_count) if candidate_count > 0 else 0.0,
            f"{name}_mass": summary_mass,
            f"{name}_confuser_mass": confuser_mass,
            f"{name}_source_cover": src_cover,
            f"{name}_hard_cover": hard_cover,
            f"{name}_both_cover": float(src_cover > 0.5 and hard_cover > 0.5),
        }

    return {
        "mass_cover": pack(mass_set, src_in_mass, hard_in_mass, "mass_cover"),
        "margin_band": pack(band_set, src_in_band, hard_in_band, "margin_band"),
        "effective_support": pack(eff_set, src_in_eff, hard_in_eff, "effective_support"),
    }


def _delta_summary(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    for key in ["size", "size_ratio", "mass", "confuser_mass"]:
        col = f"{prefix}_{key}"
        out[f"delta_{prefix}_{key}"] = out.groupby("case_id")[col].shift(1) - out[col]
    return out


def _summarise_summary(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    out = {
        "size_mean": float(df[f"{prefix}_size"].mean()),
        "size_median": float(df[f"{prefix}_size"].median()),
        "size_ratio_mean": float(df[f"{prefix}_size_ratio"].mean()),
        "source_cover_rate": float(df[f"{prefix}_source_cover"].mean()),
        "hard_cover_rate": float(df[f"{prefix}_hard_cover"].mean()),
        "both_cover_rate": float(df[f"{prefix}_both_cover"].mean()),
        "mass_mean": float(df[f"{prefix}_mass"].mean()),
        "confuser_mass_mean": float(df[f"{prefix}_confuser_mass"].mean()),
        "proxy_corr_size_to_next_delta_margin": _spearman(df, f"{prefix}_size_ratio", "next_delta_margin"),
        "proxy_corr_confuser_mass_to_next_delta_margin": _spearman(df, f"{prefix}_confuser_mass", "next_delta_margin"),
        "proxy_corr_size_to_next_delta_true_mass": _spearman(df, f"{prefix}_size_ratio", "next_delta_true_mass"),
    }
    delta_df = df[df[f"delta_{prefix}_size"].notna()].copy()
    for key in ["size", "mass", "confuser_mass"]:
        col = f"delta_{prefix}_{key}"
        out[f"{col}_mean"] = float(delta_df[col].mean()) if len(delta_df) else None
        out[f"{col}_std"] = float(delta_df[col].std()) if len(delta_df) else None
        out[f"{col}_q25"] = float(delta_df[col].quantile(0.25)) if len(delta_df) else None
        out[f"{col}_q50"] = float(delta_df[col].quantile(0.50)) if len(delta_df) else None
        out[f"{col}_q75"] = float(delta_df[col].quantile(0.75)) if len(delta_df) else None
        out[f"{col}_positive_rate"] = float((delta_df[col] > 0).mean()) if len(delta_df) else None
        std = float(delta_df[col].std()) if len(delta_df) else float("nan")
        out[f"{col}_snr"] = float(delta_df[col].mean() / std) if len(delta_df) and std > 1e-9 else None
    return out


def _summarise_reward_candidates(df: pd.DataFrame, best_prefix: str) -> pd.DataFrame:
    reward_cols = {
        "delta_true_mass": "next_delta_true_mass",
        "delta_entropy": "next_delta_entropy",
        "delta_margin": "next_delta_margin",
        f"delta_{best_prefix}_size": "next_delta_margin",
        f"delta_{best_prefix}_confuser_mass": "next_delta_margin",
        f"delta_{best_prefix}_mass": "next_delta_true_mass",
    }
    rows = []
    for col, target in reward_cols.items():
        sub = df[[col, target]].dropna().copy()
        if len(sub) == 0:
            continue
        mean = float(sub[col].mean())
        std = float(sub[col].std())
        rows.append(
            {
                "delta_name": col,
                "mean": mean,
                "std": std,
                "q25": float(sub[col].quantile(0.25)),
                "q50": float(sub[col].quantile(0.50)),
                "q75": float(sub[col].quantile(0.75)),
                "snr": float(mean / std) if std > 1e-9 else None,
                "positive_rate": float((sub[col] > 0).mean()),
                "corr_to_target": _spearman(sub, col, target),
                "target_name": target,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    source_root = Path(args.source_root)
    contrast_root = Path(args.contrast_root)
    cache_dir = Path(args.cache_dir)
    acceptability_root = Path(args.acceptability_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = load_runtime_context(source_root, cache_dir)
    _, frozen_checkpoint, reasoner_module = load_frozen_reasoner(runtime, cache_dir, device)
    step_payloads = collect_step_payloads(runtime, reasoner_module, argparse.Namespace(
        support_plausible_delta=0.25,
        not_ruled_out_threshold=0.5,
    ), device)
    calibrated = _get_calibrated_params(acceptability_root)

    rows: List[Dict[str, Any]] = []
    for row in step_payloads:
        probs = _compute_probs(row, calibrated)
        mask = row["candidate_mask"]
        valid_sorted, _ = _sorted_valid_indices(probs, mask)
        hard_local = None
        for idx in valid_sorted.tolist():
            if row["source_local"] is None or int(idx) != int(row["source_local"]):
                hard_local = int(idx)
                break
        entropy, norm_entropy = _normalized_entropy(probs, mask)
        rank = None
        if row["source_local"] is not None and bool(mask[int(row["source_local"])].item()):
            source_pos = (valid_sorted == int(row["source_local"])).nonzero(as_tuple=True)[0]
            if source_pos.numel():
                rank = int(source_pos[0].item()) + 1
        rows.append(
            {
                "case_id": row["case_id"],
                "scenario_id": row["scenario_id"],
                "part_id": row["part_id"],
                "episode_index": row["episode_index"],
                "candidate_count": int(mask.float().sum().item()),
                "rank": rank,
                "top1_hit": float(rank == 1) if rank is not None else 0.0,
                "mrr": float(1.0 / rank) if rank is not None else 0.0,
                "true_mass": float(probs[int(row["source_local"])].item()) if row["source_local"] is not None and bool(mask[int(row["source_local"])].item()) else None,
                "entropy": entropy,
                "normalized_entropy": norm_entropy,
                "effective_support": float(math.exp(entropy)),
                "effective_support_ratio": float(math.exp(entropy) / max(int(mask.float().sum().item()), 1)),
                "top1_mass": float(probs[valid_sorted[0]].item()) if len(valid_sorted) else 0.0,
                "top3_mass": float(probs[valid_sorted[: min(3, len(valid_sorted))]].sum().item()) if len(valid_sorted) else 0.0,
                "margin_true_vs_hard": (
                    float(probs[int(row["source_local"])].item() - probs[int(hard_local)].item())
                    if row["source_local"] is not None and hard_local is not None and bool(mask[int(row["source_local"])].item())
                    else None
                ),
                "hard_local": hard_local,
            }
        )
    df = pd.DataFrame(rows)

    # data-driven threshold selection ingredients
    temp = []
    for idx, row in enumerate(step_payloads):
        probs = _compute_probs(row, calibrated)
        mask = row["candidate_mask"]
        valid_sorted, sorted_vals = _sorted_valid_indices(probs, mask)
        if len(sorted_vals) >= 2:
            temp.append(float((torch.log(sorted_vals[0].clamp_min(1e-12)) - torch.log(sorted_vals[1].clamp_min(1e-12))).item()))
    margin_band = float(pd.Series(temp).median()) if temp else 0.5
    eff_scale = 0.5

    # tiny data-driven selection among majority-mass covers
    mc_probe_rows = []
    for mc in [0.5, 0.6, 0.7]:
        vals = []
        for row, base_row in zip(step_payloads, rows):
            probs = _compute_probs(row, calibrated)
            mask = row["candidate_mask"]
            valid_sorted, _ = _sorted_valid_indices(probs, mask)
            hard_local = base_row["hard_local"]
            summaries = _summary_from_probs(
                probs,
                mask,
                row["source_local"],
                hard_local,
                mass_cover_threshold=mc,
                margin_band=margin_band,
                eff_support_scale=eff_scale,
            )
            vals.append(
                {
                    f"mass_cover_{int(mc*100)}_ratio": summaries["mass_cover"]["mass_cover_size_ratio"],
                    f"mass_cover_{int(mc*100)}_source_cover": summaries["mass_cover"]["mass_cover_source_cover"],
                    f"mass_cover_{int(mc*100)}_hard_cover": summaries["mass_cover"]["mass_cover_hard_cover"],
                }
            )
        probe_df = pd.DataFrame(vals)
        mc_probe_rows.append(probe_df)
    mc_probe_df = pd.concat(mc_probe_rows, axis=1)
    mass_cover_threshold = _select_mass_cover_threshold(mc_probe_df)

    final_rows = []
    for row, base_row in zip(step_payloads, rows):
        probs = _compute_probs(row, calibrated)
        mask = row["candidate_mask"]
        summaries = _summary_from_probs(
            probs,
            mask,
            row["source_local"],
            base_row["hard_local"],
            mass_cover_threshold=mass_cover_threshold,
            margin_band=margin_band,
            eff_support_scale=eff_scale,
        )
        merged = dict(base_row)
        for family in summaries.values():
            merged.update(family)
        final_rows.append(merged)
    summary_df = pd.DataFrame(final_rows)

    summary_df["delta_true_mass"] = summary_df.groupby("case_id")["true_mass"].shift(1) - summary_df["true_mass"]
    summary_df["delta_true_mass"] = -summary_df["delta_true_mass"]
    summary_df["delta_entropy"] = summary_df.groupby("case_id")["entropy"].shift(1) - summary_df["entropy"]
    summary_df["delta_margin"] = summary_df["margin_true_vs_hard"] - summary_df.groupby("case_id")["margin_true_vs_hard"].shift(1)

    for prefix in ["mass_cover", "margin_band", "effective_support"]:
        summary_df = _delta_summary(summary_df, prefix)

    summary_df["next_delta_true_mass"] = summary_df.groupby("case_id")["true_mass"].shift(-1) - summary_df["true_mass"]
    summary_df["next_delta_entropy"] = summary_df["entropy"] - summary_df.groupby("case_id")["entropy"].shift(-1)
    summary_df["next_delta_margin"] = summary_df.groupby("case_id")["margin_true_vs_hard"].shift(-1) - summary_df["margin_true_vs_hard"]

    summary_df.to_csv(output_dir / "policy_readiness_step_rows.csv", index=False)

    family_summary = {
        "mass_cover": {
            "definition": f"smallest top set whose posterior mass >= {mass_cover_threshold:.2f}",
            "threshold": float(mass_cover_threshold),
            **_summarise_summary(summary_df, "mass_cover"),
        },
        "margin_band": {
            "definition": "candidates with log p(top1) - log p(u) <= empirical median top1-top2 log-gap",
            "threshold": float(margin_band),
            **_summarise_summary(summary_df, "margin_band"),
        },
        "effective_support": {
            "definition": f"top ceil(exp(H) * {eff_scale:.2f}) posterior candidates",
            "threshold": float(eff_scale),
            **_summarise_summary(summary_df, "effective_support"),
        },
    }
    family_compare = pd.DataFrame([{ "summary_name": k, **v } for k, v in family_summary.items()])
    family_compare.to_csv(output_dir / "confusion_summary_compare.csv", index=False)

    # choose best summary by balanced policy-readiness score
    def score(row: pd.Series) -> float:
        return (
            1.5 * float(row["both_cover_rate"])
            + 0.5 * float(row["source_cover_rate"])
            - 1.0 * float(row["size_ratio_mean"])
            + 0.5 * abs(float(row["proxy_corr_confuser_mass_to_next_delta_margin"]) if pd.notna(row["proxy_corr_confuser_mass_to_next_delta_margin"]) else 0.0)
            + 0.25 * abs(float(row["proxy_corr_size_to_next_delta_true_mass"]) if pd.notna(row["proxy_corr_size_to_next_delta_true_mass"]) else 0.0)
        )
    family_compare["policy_readiness_score"] = family_compare.apply(score, axis=1)
    family_compare.to_csv(output_dir / "confusion_summary_compare.csv", index=False)
    best_summary_name = str(family_compare.sort_values("policy_readiness_score", ascending=False).iloc[0]["summary_name"])

    reward_df = _summarise_reward_candidates(summary_df, best_summary_name)
    reward_df.to_csv(output_dir / "reward_readiness_compare.csv", index=False)

    summary = {
        "runner_version": RUNNER_VERSION,
        "panel_version": PANEL_VERSION,
        "seed": int(args.seed),
        "reasoner_asset": str(frozen_checkpoint),
        "fixed_posterior": {
            "type": "lightly_calibrated_fused_posterior",
            **calibrated,
        },
        "summary_families": family_summary,
        "recommended_summary": best_summary_name,
        "reward_ready_candidates_ranked": reward_df.sort_values(["snr", "corr_to_target"], ascending=[False, False]).to_dict(orient="records"),
    }
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
