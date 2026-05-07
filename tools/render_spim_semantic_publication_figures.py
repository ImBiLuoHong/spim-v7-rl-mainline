#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "spim_semantic_publication_figures_20260416"

PALETTE = {
    "bg": "#FFFFFF",
    "ink": "#16202A",
    "muted": "#16202A",
    "grid": "#D9E1E8",
    "baseline": "#B8C4CF",
    "greedy": "#2E678C",
    "rl": "#C44E3B",
    "posterior_main": "#2E678C",
    "posterior_alt1": "#6E8FA8",
    "posterior_alt2": "#9FB4C6",
}


def configure_style() -> None:
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    ):
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)

    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["bg"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["ink"],
            "font.family": "Noto Sans CJK JP",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> float:
    return float(x) * 100.0


def build_old_semantics_table() -> pd.DataFrame:
    rows = [
        {
            "old_bar_label": "Greedy (v3 posterior)",
            "semantic_type": "posterior_core_under_greedy_policy",
            "internal_method": "hsr_soft_scenario_posterior_v3",
            "artifact_path": "artifacts/spim_family_sweep/20260410_exact136_b30_clean_spim5_v2_full/family_summary.csv",
            "metric_scope": "exact136 / B30 / clean track / greedy posterior rollout",
            "success_rate": 0.8823529411764706,
            "why_old_label_is_wrong": "This is not a generic sampling strategy; it is a posterior-family result under greedy policy.",
        },
        {
            "old_bar_label": "Uncertainty (rank-weighted)",
            "semantic_type": "posterior_core_under_greedy_policy",
            "internal_method": "hsr_rank_weighted_topk_v2",
            "artifact_path": "artifacts/spim_family_sweep/20260410_exact136_b30_clean_spim5_v2_full/family_summary.csv",
            "metric_scope": "exact136 / B30 / clean track / greedy posterior rollout",
            "success_rate": 0.875,
            "why_old_label_is_wrong": "This is a posterior variant, not a standalone uncertainty sampling policy.",
        },
        {
            "old_bar_label": "Entropy (binary source-only)",
            "semantic_type": "posterior_core_under_greedy_policy",
            "internal_method": "hsr_binary_loglik_posterior_source_only_v4",
            "artifact_path": "artifacts/spim_family_sweep/20260410_exact136_b30_clean_spim5_v2_full/family_summary.csv",
            "metric_scope": "exact136 / B30 / clean track / greedy posterior rollout",
            "success_rate": 0.875,
            "why_old_label_is_wrong": "This is a binary source-only posterior core, not entropy-reduction sampling.",
        },
        {
            "old_bar_label": "RL-best (set interaction)",
            "semantic_type": "rl_policy",
            "internal_method": "rl_set_seed45_schemeS_setint_train4823",
            "artifact_path": "artifacts/spim_set_level_rl_mainline/20260416_v3_setint_uncert_bounded_v1/stage1_schemeS_seed45/strict_eval_val_B30/rl_set_seed45_schemeS_setint_train4823/summary.json",
            "metric_scope": "strict val_B30 / 1031 val cases / RL policy",
            "success_rate": 0.9204655674102813,
            "why_old_label_is_wrong": "This is an RL policy result and should not be placed in the same semantic tier as posterior cores.",
        },
        {
            "old_bar_label": "RL-best + uncertainty",
            "semantic_type": "rl_policy",
            "internal_method": "rl_set_seed45_schemeU_setint_uncert_train4823",
            "artifact_path": "artifacts/spim_set_level_rl_mainline/20260416_v3_setint_uncert_bounded_v1/stage2_schemeU_seed45/strict_eval_val_B30/rl_set_seed45_schemeU_setint_uncert_train4823/summary.json",
            "metric_scope": "strict val_B30 / 1031 val cases / RL policy with uncertainty-regime features",
            "success_rate": 0.918525703200776,
            "why_old_label_is_wrong": "This is another RL policy variant, not a posterior core or classical sampling baseline.",
        },
    ]
    return pd.DataFrame(rows)


def build_mapping_table() -> pd.DataFrame:
    rows = [
        {"internal_method": "posterior_greedy", "display_name": "Greedy Posterior", "figure": "Figure 1"},
        {"internal_method": "posterior_thompson_sampling", "display_name": "Thompson Sampling", "figure": "Figure 1"},
        {"internal_method": "posterior_entropy_drop", "display_name": "Entropy Reduction", "figure": "Figure 1"},
        {"internal_method": "posterior_cover_shrink", "display_name": "Cover Shrink", "figure": "Figure 1"},
        {"internal_method": "posterior_disagreement_split", "display_name": "Disagreement Split", "figure": "Figure 1"},
        {"internal_method": "rl_set_seed45_schemeS_setint_train4823", "display_name": "Strongest RL", "figure": "Figure 1"},
        {"internal_method": "hsr_soft_scenario_posterior_v3", "display_name": "SPIM v3", "figure": "Figure 2"},
        {"internal_method": "hsr_rank_weighted_topk_v2", "display_name": "Rank-weighted Top-k", "figure": "Figure 2"},
        {"internal_method": "hsr_binary_loglik_posterior_source_only_v4", "display_name": "Binary Source-only", "figure": "Figure 2"},
    ]
    return pd.DataFrame(rows)


def build_figure1_table() -> pd.DataFrame:
    teacher5_dir = OUTPUT_DIR.parent / "spim_teacher5_compare" / "20260416_val_b30_full1031_v1"
    rl_path = OUTPUT_DIR.parent / "spim_set_level_rl_mainline" / "20260416_v3_setint_uncert_bounded_v1" / "stage1_schemeS_seed45" / "strict_eval_val_B30" / "rl_set_seed45_schemeS_setint_train4823" / "summary.json"
    conservative_greedy_family_path = OUTPUT_DIR.parent / "spim_family_sweep" / "20260410_exact136_b30_clean_spim5_v2_full" / "family_summary.csv"

    teacher5_summary = read_json(teacher5_dir / "summary.json")
    leaderboard = pd.read_csv(teacher5_dir / "teacher5_leaderboard.csv").copy()
    display_map = {
        "posterior_greedy": "Greedy Posterior",
        "posterior_thompson_sampling": "Thompson Sampling",
        "posterior_entropy_drop": "Entropy Reduction",
        "posterior_cover_shrink": "Cover Shrink",
        "posterior_disagreement_split": "Disagreement Split",
    }
    leaderboard["display_name"] = leaderboard["policy_name"].map(display_map)
    leaderboard["method_type"] = "sampling_policy"
    leaderboard["artifact_path"] = str((teacher5_dir / "summary.json").relative_to(PROJECT_ROOT))
    leaderboard["runner_version"] = teacher5_summary["runner_version"]
    leaderboard["panel_version"] = teacher5_summary["panel_version"]
    leaderboard["seed"] = int(teacher5_summary["protocol"]["seed"])
    leaderboard["split"] = str(teacher5_summary["protocol"]["split"])
    leaderboard["case_count"] = int(teacher5_summary["split_meta"]["loaded_case_count"])
    leaderboard["posterior_family"] = str(teacher5_summary["posterior_family"]["selected_family"])
    leaderboard["budget"] = int(teacher5_summary["protocol"]["sample_budget"])
    leaderboard["actions_per_round"] = int(teacher5_summary["protocol"]["actions_per_round"])
    leaderboard["success_definition"] = str(teacher5_summary["protocol"]["success_definition"])

    # Use the lower exact136 greedy value as the conservative displayed greedy baseline.
    conservative_greedy_df = pd.read_csv(conservative_greedy_family_path)
    conservative_greedy_sr = float(
        conservative_greedy_df.loc[
            conservative_greedy_df["family"] == "hsr_soft_scenario_posterior_v3", "success_rate"
        ].iloc[0]
    )
    greedy_mask = leaderboard["policy_name"] == "posterior_greedy"
    leaderboard.loc[greedy_mask, "success_rate"] = conservative_greedy_sr
    leaderboard.loc[greedy_mask, "artifact_path"] = str(conservative_greedy_family_path.relative_to(PROJECT_ROOT))
    leaderboard.loc[greedy_mask, "runner_version"] = "spim_family_sweep_v1"
    leaderboard.loc[greedy_mask, "panel_version"] = "exact136_authoritative_b30_spim_sweep_v1"
    leaderboard.loc[greedy_mask, "split"] = "exact136"
    leaderboard.loc[greedy_mask, "case_count"] = 136
    leaderboard.loc[greedy_mask, "budget_used_mean"] = pd.NA
    leaderboard.loc[greedy_mask, "avg_hit_round_conditional"] = pd.NA
    leaderboard.loc[greedy_mask, "success_definition"] = "direct source hit within budget"

    rl_summary = read_json(rl_path)
    rl_row = pd.DataFrame(
        [
            {
                "policy_name": str(rl_summary["requested_policy_name"]),
                "display_name": "Strongest RL",
                "success_rate": float(rl_summary["summary"]["success_rate"]),
                "avg_hit_round_conditional": float(rl_summary["summary"]["avg_hit_round_conditional"]),
                "budget_used_mean": float(rl_summary["summary"]["budget_used_mean"]),
                "case_count": int(rl_summary["summary"]["case_count"]),
                "method_type": "rl_policy",
                "artifact_path": str(rl_path.relative_to(PROJECT_ROOT)),
                "runner_version": str(rl_summary["runner_version"]),
                "panel_version": str(rl_summary["panel_version"]),
                "seed": int(rl_summary["seed"]),
                "split": str(rl_summary["split"]),
                "posterior_family": str(rl_summary["teacher_family"]),
                "budget": int(rl_summary["protocol"]["budget"]),
                "actions_per_round": int(rl_summary["protocol"]["actions_per_round"]),
                "success_definition": "direct source hit within budget",
            }
        ]
    )

    out = pd.concat([leaderboard, rl_row], ignore_index=True)
    out["success_rate_pct"] = out["success_rate"].map(pct)
    out["rank_desc"] = out["success_rate"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["success_rate", "display_name"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_figure2_table() -> pd.DataFrame:
    summary_path = OUTPUT_DIR.parent / "spim_family_sweep" / "20260410_exact136_b30_clean_spim5_v2_full" / "family_summary.csv"
    df = pd.read_csv(summary_path)
    keep = {
        "hsr_soft_scenario_posterior_v3": "SPIM v3",
        "hsr_rank_weighted_topk_v2": "Rank-weighted Top-k",
        "hsr_binary_loglik_posterior_source_only_v4": "Binary Source-only",
    }
    out = df.loc[df["family"].isin(keep)].copy()
    out["display_name"] = out["family"].map(keep)
    out["artifact_path"] = str(summary_path.relative_to(PROJECT_ROOT))
    out["method_type"] = "posterior_core"
    out["success_rate_pct"] = out["success_rate"].map(pct)
    out["rank_desc"] = out["success_rate"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values(["success_rate", "display_name"], ascending=[False, True]).reset_index(drop=True)
    return out


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.85)
    ax.set_axisbelow(True)


def _bar_colors(display_names: List[str], kind: str) -> List[str]:
    colors: List[str] = []
    for name in display_names:
        if kind == "figure1":
            if name == "Strongest RL":
                colors.append(PALETTE["rl"])
            elif name == "Greedy Posterior":
                colors.append(PALETTE["greedy"])
            else:
                colors.append(PALETTE["baseline"])
        else:
            if name == "SPIM v3":
                colors.append(PALETTE["posterior_main"])
            elif name == "Rank-weighted Top-k":
                colors.append(PALETTE["posterior_alt1"])
            else:
                colors.append(PALETTE["posterior_alt2"])
    return colors


def draw_horizontal_bar(
    df: pd.DataFrame,
    *,
    display_col: str,
    value_col: str,
    title: str,
    out_base: Path,
    kind: str,
) -> None:
    plot_df = df.sort_values(value_col, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    _style_axes(ax)

    y = list(range(len(plot_df)))
    vals = plot_df[value_col].tolist()
    names = plot_df[display_col].tolist()
    colors = _bar_colors(names, kind)

    bars = ax.barh(y, vals, height=0.46, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Success Rate (%)")

    xmin = min(vals) - 6.0
    xmax = max(vals) + 5.0
    ax.set_xlim(xmin, xmax)

    ax.set_title(title, loc="left", pad=12)

    for bar, val in zip(bars, vals):
        y_mid = float(bar.get_y() + bar.get_height() / 2.0)
        ax.text(val + 0.45, y_mid, f"{val:.2f}%", va="center", ha="left", fontsize=10.4, color=PALETTE["ink"])

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    old_semantics = build_old_semantics_table()
    mapping_df = build_mapping_table()
    fig1_df = build_figure1_table()
    fig2_df = build_figure2_table()

    old_semantics.to_csv(OUTPUT_DIR / "old_figure_bar_semantics.csv", index=False)
    mapping_df.to_csv(OUTPUT_DIR / "method_display_mapping.csv", index=False)
    fig1_df.to_csv(OUTPUT_DIR / "figure1_sampling_policy_comparison.csv", index=False)
    fig2_df.to_csv(OUTPUT_DIR / "figure2_posterior_core_comparison.csv", index=False)

    draw_horizontal_bar(
        fig1_df,
        display_col="display_name",
        value_col="success_rate_pct",
        title="Sampling Policies on SPIM v3",
        out_base=OUTPUT_DIR / "figure1_sampling_policy_comparison",
        kind="figure1",
    )
    draw_horizontal_bar(
        fig2_df,
        display_col="display_name",
        value_col="success_rate_pct",
        title="Posterior Cores Under Greedy Inference",
        out_base=OUTPUT_DIR / "figure2_posterior_core_comparison",
        kind="figure2",
    )

    manifest = {
        "output_dir": str(OUTPUT_DIR.resolve()),
        "files": {
            "old_semantics_csv": "old_figure_bar_semantics.csv",
            "mapping_csv": "method_display_mapping.csv",
            "figure1_csv": "figure1_sampling_policy_comparison.csv",
            "figure2_csv": "figure2_posterior_core_comparison.csv",
            "figure1_outputs": {
                "pdf": "figure1_sampling_policy_comparison.pdf",
                "svg": "figure1_sampling_policy_comparison.svg",
                "png": "figure1_sampling_policy_comparison.png",
            },
            "figure2_outputs": {
                "pdf": "figure2_posterior_core_comparison.pdf",
                "svg": "figure2_posterior_core_comparison.svg",
                "png": "figure2_posterior_core_comparison.png",
            },
        },
        "fairness_notes": {
            "figure1": "Uses existing fair policy comparison artifact on val-1031 B30 plus strongest RL strict val_B30 result.",
            "figure2": "Uses existing exact136 B30 clean-track posterior-family sweep under greedy rollout.",
            "v6_status": "Excluded from Figure 2 because the available strict v6 artifact is train-256, not the same exact136 clean-track scope.",
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
