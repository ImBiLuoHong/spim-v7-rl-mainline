from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
PASS1_ROOT = ARTIFACTS_ROOT / "paper_analysis" / "20260420_080826"
PASS2_ROOT = ARTIFACTS_ROOT / "paper_analysis_pass2" / "20260420_084219"
PASS3_ROOT = ARTIFACTS_ROOT / "paper_freeze_pass3" / "20260420_101724"

EXPORT_DPI = 400
TITLE_Y = 0.985
SUBTITLE_Y = 0.94
ROUND_FAIL_SENTINEL = 11.0

MAIN_FIGURE_ORDER = [
    "main_comparison_bar",
    "success_vs_budget",
    "delta_success_vs_budget",
    "hard_state_heatmap",
    "paired_case_taxonomy",
    "policy_overlap_summary",
]

APPENDIX_FIGURE_ORDER = [
    "hit_round_ecdf",
    "hit_round_scatter",
    "hybrid_replay_ladder",
    "slot_tendency",
    "relaxed_radius_curve",
    "frontier_boundary_check",
    "representative_case_panels",
]

METHOD_ORDER = [
    "strongest_rl",
    "posterior_greedy",
    "posterior_thompson_sampling",
    "posterior_entropy_drop",
    "posterior_cover_shrink",
    "posterior_disagreement_split",
]

CASE_GROUP_ORDER = [
    "rl_unique_win",
    "both_hit_rl_earlier",
    "both_hit_same_or_near",
    "both_hit_greedy_earlier",
    "greedy_unique_win",
    "both_fail",
]

CASE_GROUP_LABELS = {
    "rl_unique_win": "RL unique win",
    "both_hit_rl_earlier": "Both hit, RL earlier",
    "both_hit_same_or_near": "Both hit, same/near",
    "both_hit_greedy_earlier": "Greedy earlier",
    "greedy_unique_win": "Greedy unique win",
    "both_fail": "Both fail",
}

HYBRID_ORDER = [
    "posterior_greedy",
    "rl1_g23",
    "g1_rl23",
    "rl12_g3",
    "g12_rl3",
    "strongest_rl",
]

HYBRID_LABELS = {
    "posterior_greedy": "Greedy",
    "rl1_g23": "RL1 + G23",
    "g1_rl23": "G1 + RL23",
    "rl12_g3": "RL12 + G3",
    "g12_rl3": "G12 + RL3",
    "strongest_rl": "RL",
}

PALETTE = {
    "strongest_rl": "#9E4254",
    "posterior_greedy": "#D8877B",
    "posterior_thompson_sampling": "#C9A07A",
    "posterior_entropy_drop": "#5E7D97",
    "posterior_cover_shrink": "#7F98AB",
    "posterior_disagreement_split": "#A6B9C7",
    "axis": "#312B2A",
    "muted_text": "#6D6461",
    "grid": "#E1DBD8",
    "panel_bg": "#FBF9F8",
    "neutral": "#948B88",
    "neutral_light": "#C9C1BE",
    "both_fail": "#BEB6B2",
    "greedy_adv": "#7A93A9",
    "same_or_near": "#8C857F",
    "boundary_dark": "#6A6664",
    "boundary_light": "#C9C6C4",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_delta",
    ["#6E8CA7", "#F7F3F1", "#A04658"],
)

FIGURE_SPECS = {
    "main_comparison_bar": {
        "intended_use": "main",
        "short_claim_supported": "Under the aligned held-out exact-hit B30 contract, Strongest RL leads Greedy Posterior and the heuristic baselines.",
        "caution_level": "proven",
    },
    "success_vs_budget": {
        "intended_use": "main",
        "short_claim_supported": "The RL advantage is visible early in the budget curve rather than only at budget exhaustion.",
        "caution_level": "proven",
    },
    "delta_success_vs_budget": {
        "intended_use": "main",
        "short_claim_supported": "The RL-minus-Greedy exact-hit gap opens before the final budget regime.",
        "caution_level": "proven",
    },
    "hard_state_heatmap": {
        "intended_use": "main",
        "short_claim_supported": "RL gains are larger in harder initial states, shown here as a descriptive entropy-by-support map.",
        "caution_level": "partially proven",
    },
    "paired_case_taxonomy": {
        "intended_use": "main",
        "short_claim_supported": "The net gain comes from RL-only wins plus a large both-hit-but-RL-earlier bucket, not only from aggregate averaging.",
        "caution_level": "proven",
    },
    "policy_overlap_summary": {
        "intended_use": "main",
        "short_claim_supported": "RL is not simple imitation of Greedy at the 3-set level.",
        "caution_level": "proven",
    },
    "hit_round_ecdf": {
        "intended_use": "appendix",
        "short_claim_supported": "RL solves a larger share of cases by earlier rounds under the same B30 contract.",
        "caution_level": "proven",
    },
    "hit_round_scatter": {
        "intended_use": "appendix",
        "short_claim_supported": "The paired round-by-round comparison shows where RL is earlier, where Greedy is earlier, and where either policy misses B30.",
        "caution_level": "proven",
    },
    "hybrid_replay_ladder": {
        "intended_use": "appendix",
        "short_claim_supported": "On the bounded 24-case subset, later-pick replacement helps more than replacing only the first pick.",
        "caution_level": "partially proven",
    },
    "slot_tendency": {
        "intended_use": "appendix",
        "short_claim_supported": "Slot statistics are consistent with an exploit-first and later-complement tendency on the bounded subset.",
        "caution_level": "partially proven",
    },
    "relaxed_radius_curve": {
        "intended_use": "appendix",
        "short_claim_supported": "Relaxed-radius success remains secondary context; exact hit stays primary.",
        "caution_level": "proven",
    },
    "frontier_boundary_check": {
        "intended_use": "appendix",
        "short_claim_supported": "The targeted replay subset remains posterior-dominated, which does not support a strong novelty-frontier mechanism claim.",
        "caution_level": "boundary-only",
    },
    "representative_case_panels": {
        "intended_use": "appendix",
        "short_claim_supported": "Frozen representative cases illustrate the four main paired-outcome categories without replacing the aggregate evidence.",
        "caution_level": "partially proven",
    },
}


@dataclass
class BundleRefs:
    pass1_root: Path
    pass2_root: Path
    pass3_root: Path
    output_dir: Path
    pass1_manifest: Dict[str, Any]
    pass2_manifest: Dict[str, Any]
    pass3_manifest: Dict[str, Any]
    rl_case_rows_path: Path
    rl_step_rows_path: Path
    teacher5_case_rows_path: Path
    teacher5_step_rows_path: Path
    teacher_full_case_rows_path: Path
    teacher_full_step_rows_path: Path


@dataclass
class FigureRecord:
    figure_name: str
    source_artifact_files: List[str]
    intended_use: str
    short_claim_supported: str
    caution_level: str
    png_path: str
    vector_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a publication-style SPIM-v3 + RL paper figure package from frozen artifacts.")
    parser.add_argument("--pass1-root", type=str, default=str(PASS1_ROOT))
    parser.add_argument("--pass2-root", type=str, default=str(PASS2_ROOT))
    parser.add_argument("--pass3-root", type=str, default=str(PASS3_ROOT))
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir(raw: str) -> Path:
    if raw:
        output_dir = Path(raw)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = ARTIFACTS_ROOT / "paper_figures" / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_bundle_refs(args: argparse.Namespace) -> BundleRefs:
    pass1_root = Path(args.pass1_root)
    pass2_root = Path(args.pass2_root)
    pass3_root = Path(args.pass3_root)
    output_dir = ensure_output_dir(args.output_dir)
    pass1_manifest = load_json(pass1_root / "run_manifest.json")
    pass2_manifest = load_json(pass2_root / "run_manifest.json")
    pass3_manifest = load_json(pass3_root / "run_manifest.json")
    rl_case_rows_path = Path(pass1_manifest["selected_rl_artifact"]["case_rows_path"])
    rl_step_rows_path = Path(pass1_manifest["selected_rl_artifact"]["step_rows_path"])
    teacher5_case_rows_path = Path(pass1_manifest["selected_teacher5_artifact"]["case_rows_path"])
    teacher5_step_rows_path = Path(pass1_manifest["selected_teacher5_artifact"]["step_rows_path"])
    teacher_full_case_rows_path = rl_case_rows_path.parents[1] / "teacher_full" / "case_rows.csv"
    teacher_full_step_rows_path = rl_case_rows_path.parents[1] / "teacher_full" / "step_rows.csv"
    return BundleRefs(
        pass1_root=pass1_root,
        pass2_root=pass2_root,
        pass3_root=pass3_root,
        output_dir=output_dir,
        pass1_manifest=pass1_manifest,
        pass2_manifest=pass2_manifest,
        pass3_manifest=pass3_manifest,
        rl_case_rows_path=rl_case_rows_path,
        rl_step_rows_path=rl_step_rows_path,
        teacher5_case_rows_path=teacher5_case_rows_path,
        teacher5_step_rows_path=teacher5_step_rows_path,
        teacher_full_case_rows_path=teacher_full_case_rows_path,
        teacher_full_step_rows_path=teacher_full_step_rows_path,
    )


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "axes.edgecolor": PALETTE["axis"],
            "axes.facecolor": PALETTE["panel_bg"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "grid.linestyle": "-",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": PALETTE["axis"],
            "ytick.color": PALETTE["axis"],
            "text.color": PALETTE["axis"],
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.facecolor": "white",
            "legend.edgecolor": "#D5CFCC",
            "legend.framealpha": 0.95,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": EXPORT_DPI,
            "figure.dpi": 120,
        }
    )


def figure_heading(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.suptitle(title, x=0.02, y=TITLE_Y, ha="left", va="top", fontweight="bold", fontsize=11.5)
    if subtitle:
        fig.text(0.02, SUBTITLE_Y, subtitle, ha="left", va="top", fontsize=8.4, color=PALETTE["muted_text"])


def finish_axes(ax: plt.Axes, xgrid: bool = True, ygrid: bool = False) -> None:
    ax.grid(False)
    if xgrid:
        ax.grid(axis="x", zorder=0)
    if ygrid:
        ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, figure_name: str) -> tuple[str, str]:
    png_path = output_dir / f"{figure_name}.png"
    pdf_path = output_dir / f"{figure_name}.pdf"
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(png_path), str(pdf_path)


def method_color(method_id: str) -> str:
    return PALETTE.get(method_id, PALETTE["neutral"])


def case_group_color(group: str) -> str:
    if group in {"rl_unique_win", "both_hit_rl_earlier"}:
        return PALETTE["strongest_rl"]
    if group in {"greedy_unique_win", "both_hit_greedy_earlier"}:
        return PALETTE["greedy_adv"]
    if group == "both_hit_same_or_near":
        return PALETTE["same_or_near"]
    return PALETTE["both_fail"]


def classify_case(row: pd.Series) -> str:
    rl_success = bool(row["rl_success"])
    greedy_success = bool(row["greedy_success"])
    if rl_success and not greedy_success:
        return "rl_unique_win"
    if greedy_success and not rl_success:
        return "greedy_unique_win"
    if not rl_success and not greedy_success:
        return "both_fail"
    diff = float(row["greedy_hit_round"]) - float(row["rl_hit_round"])
    if diff > 1:
        return "both_hit_rl_earlier"
    if diff < -1:
        return "both_hit_greedy_earlier"
    return "both_hit_same_or_near"


def safe_quantile_buckets(series: pd.Series, labels: Sequence[str]) -> pd.Series:
    values = series.astype(float)
    quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, len(labels) + 1)))
    if len(quantiles) - 1 == len(labels):
        return pd.cut(values, bins=quantiles, labels=labels, include_lowest=True, duplicates="drop").astype(str)
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels).astype(str)


def bootstrap_mean_ci(values: Sequence[float], seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1:
        point = float(arr[0])
        return point, point, point
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    mean = float(arr.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi)


def scenario_short(case_id: str) -> str:
    parts = case_id.split(":")
    if len(parts) >= 2:
        return parts[1]
    return case_id


def representative_case_label(index: int) -> str:
    return f"Case{chr(ord('A') + index)}"


def format_hit(value: float) -> str:
    if pd.isna(value):
        return "miss"
    return str(int(value))


def load_tables(refs: BundleRefs) -> Dict[str, Any]:
    comparison = pd.read_csv(refs.pass1_root / "comparison_table.csv")
    success_budget = pd.read_csv(refs.pass1_root / "success_vs_budget.csv")
    case_taxonomy = pd.read_csv(refs.pass1_root / "case_taxonomy.csv")
    difficulty_bucket = pd.read_csv(refs.pass1_root / "difficulty_bucket_tables.csv")
    policy_behavior = pd.read_csv(refs.pass1_root / "policy_behavior_tables.csv")
    relaxed_metric = pd.read_csv(refs.pass1_root / "relaxed_metric_tables.csv")
    representative_cases = json.loads((refs.pass1_root / "representative_cases.json").read_text(encoding="utf-8"))

    complementarity = pd.read_csv(refs.pass2_root / "complementarity_tables.csv")
    hard_state = pd.read_csv(refs.pass2_root / "hard_state_mechanism_tables.csv")
    slot_tendency = pd.read_csv(refs.pass2_root / "slot_role_tendency_tables.csv")
    frontier = pd.read_csv(refs.pass2_root / "frontier_composition_tables.csv")
    targeted_slot_rows = pd.read_csv(refs.pass2_root / "targeted_hybrid_replay_slot_rows.csv")
    representative_manifest = pd.read_csv(refs.pass2_root / "representative_case_figures" / "figure_manifest.csv")
    claim_ledger = pd.read_csv(refs.pass3_root / "claim_evidence_ledger.csv")

    rl_case = pd.read_csv(refs.rl_case_rows_path)
    rl_step = pd.read_csv(refs.rl_step_rows_path)
    teacher5_case = pd.read_csv(refs.teacher5_case_rows_path)
    teacher5_step = pd.read_csv(refs.teacher5_step_rows_path)
    teacher_full_case = pd.read_csv(refs.teacher_full_case_rows_path)
    teacher_full_step = pd.read_csv(refs.teacher_full_step_rows_path)

    greedy_case = teacher5_case[teacher5_case["policy_name"] == "posterior_greedy"].copy()
    greedy_step = teacher5_step[teacher5_step["policy_name"] == "posterior_greedy"].copy()

    initial_round = teacher_full_step[teacher_full_step["round_index"] == 1].copy()
    initial_features = initial_round[
        [
            "case_id",
            "candidate_count",
            "posterior_entropy",
            "top1_top2_margin",
        ]
    ].rename(
        columns={
            "candidate_count": "initial_candidate_count",
            "posterior_entropy": "initial_posterior_entropy",
            "top1_top2_margin": "initial_top1_top2_margin",
        }
    )

    paired = teacher_full_case[["case_id", "source_global_id", "trigger_global_id"]].merge(initial_features, on="case_id", how="left")
    paired = paired.merge(
        rl_case[["case_id", "success_rate", "hit_round", "hit_sample_index", "budget_used"]].rename(
            columns={
                "success_rate": "rl_success_rate",
                "hit_round": "rl_hit_round",
                "hit_sample_index": "rl_hit_sample_index",
                "budget_used": "rl_budget_used",
            }
        ),
        on="case_id",
        how="left",
    )
    paired = paired.merge(
        greedy_case[["case_id", "success_rate", "hit_round", "hit_sample_index", "budget_used"]].rename(
            columns={
                "success_rate": "greedy_success_rate",
                "hit_round": "greedy_hit_round",
                "hit_sample_index": "greedy_hit_sample_index",
                "budget_used": "greedy_budget_used",
            }
        ),
        on="case_id",
        how="left",
    )
    paired["rl_success"] = paired["rl_success_rate"] > 0.5
    paired["greedy_success"] = paired["greedy_success_rate"] > 0.5
    paired["taxonomy_group"] = paired.apply(classify_case, axis=1)
    return {
        "comparison": comparison,
        "success_budget": success_budget,
        "case_taxonomy": case_taxonomy,
        "difficulty_bucket": difficulty_bucket,
        "policy_behavior": policy_behavior,
        "relaxed_metric": relaxed_metric,
        "representative_cases": representative_cases,
        "complementarity": complementarity,
        "hard_state": hard_state,
        "slot_tendency": slot_tendency,
        "frontier": frontier,
        "targeted_slot_rows": targeted_slot_rows,
        "representative_manifest": representative_manifest,
        "claim_ledger": claim_ledger,
        "rl_case": rl_case,
        "rl_step": rl_step,
        "greedy_case": greedy_case,
        "greedy_step": greedy_step,
        "paired": paired,
    }


def plot_main_comparison_bar(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    comparison = tables["comparison"].set_index("method_id").loc[METHOD_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.15))
    figure_heading(
        fig,
        "Held-out exact Success@B30",
        "Aligned val / B30 / SPIM-v3 contract; Strongest RL highlighted without truncating the 0-1 scale.",
    )
    y = np.arange(len(comparison))
    bars = ax.barh(
        y,
        comparison["success_at_B30"],
        color=[method_color(method) for method in comparison["method_id"]],
        edgecolor="white",
        linewidth=0.9,
        height=0.7,
        zorder=3,
    )
    for bar, method_id, value in zip(bars, comparison["method_id"], comparison["success_at_B30"]):
        ax.text(
            min(float(value) + 0.012, 0.985),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=8.3,
            fontweight="bold" if method_id == "strongest_rl" else "normal",
            color=PALETTE["axis"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(comparison["display_name"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Exact Success@B30")
    finish_axes(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(top=0.83, left=0.29, right=0.97, bottom=0.12)
    png_path, vector_path = save_figure(fig, refs.output_dir, "main_comparison_bar")
    sources = [str(refs.pass1_root / "comparison_table.csv")]
    spec = FIGURE_SPECS["main_comparison_bar"]
    return FigureRecord("main_comparison_bar", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_success_vs_budget(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    success_budget = tables["success_budget"]
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    figure_heading(
        fig,
        "Success vs budget",
        "Strongest RL, Greedy Posterior, and Thompson are emphasized; weaker heuristics stay visible but muted.",
    )
    for method_id in METHOD_ORDER:
        sub = success_budget[success_budget["method_id"] == method_id].sort_values("sample_budget")
        color = method_color(method_id)
        if method_id == "strongest_rl":
            ax.plot(
                sub["sample_budget"],
                sub["cumulative_success_rate"],
                color=color,
                linewidth=2.6,
                marker="o",
                markersize=4.0,
                markevery=[0, 3, 9, 19, 29],
                label="Strongest RL",
                zorder=5,
            )
        elif method_id == "posterior_greedy":
            ax.plot(
                sub["sample_budget"],
                sub["cumulative_success_rate"],
                color=color,
                linewidth=2.2,
                marker="s",
                markersize=3.8,
                markevery=[0, 3, 9, 19, 29],
                label="Greedy Posterior",
                zorder=4,
            )
        elif method_id == "posterior_thompson_sampling":
            ax.plot(
                sub["sample_budget"],
                sub["cumulative_success_rate"],
                color=color,
                linewidth=1.9,
                linestyle="-.",
                label="Thompson Sampling",
                zorder=3,
            )
        else:
            ax.plot(
                sub["sample_budget"],
                sub["cumulative_success_rate"],
                color=color,
                linewidth=1.3,
                linestyle="--",
                alpha=0.75,
                label=sub["display_name"].iloc[0],
                zorder=2,
            )
    ax.set_xlabel("Sample budget")
    ax.set_ylabel("Cumulative exact success")
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    finish_axes(ax, xgrid=False, ygrid=True)
    ax.legend(ncol=2, loc="lower right")
    fig.subplots_adjust(top=0.83, left=0.11, right=0.98, bottom=0.13)
    png_path, vector_path = save_figure(fig, refs.output_dir, "success_vs_budget")
    sources = [str(refs.pass1_root / "success_vs_budget.csv")]
    spec = FIGURE_SPECS["success_vs_budget"]
    return FigureRecord("success_vs_budget", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_delta_success_vs_budget(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    success_budget = tables["success_budget"]
    rl = success_budget[success_budget["method_id"] == "strongest_rl"][["sample_budget", "cumulative_success_rate"]].rename(
        columns={"cumulative_success_rate": "rl_success"}
    )
    greedy = success_budget[success_budget["method_id"] == "posterior_greedy"][["sample_budget", "cumulative_success_rate"]].rename(
        columns={"cumulative_success_rate": "greedy_success"}
    )
    merged = rl.merge(greedy, on="sample_budget", how="inner")
    merged["delta_pp"] = (merged["rl_success"] - merged["greedy_success"]) * 100.0
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    figure_heading(
        fig,
        "RL advantage over Greedy by budget",
        "Positive values indicate exact-hit gain in percentage points relative to Greedy Posterior.",
    )
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=1.0, linestyle="--", zorder=1)
    ax.plot(
        merged["sample_budget"],
        merged["delta_pp"],
        color=PALETTE["strongest_rl"],
        linewidth=2.4,
        marker="o",
        markersize=3.6,
        markevery=[0, 3, 9, 19, 29],
        zorder=4,
    )
    ax.fill_between(
        merged["sample_budget"],
        0.0,
        merged["delta_pp"],
        where=merged["delta_pp"] >= 0.0,
        color=PALETTE["strongest_rl"],
        alpha=0.16,
        zorder=2,
    )
    ax.fill_between(
        merged["sample_budget"],
        0.0,
        merged["delta_pp"],
        where=merged["delta_pp"] < 0.0,
        color=PALETTE["greedy_adv"],
        alpha=0.18,
        zorder=2,
    )
    ax.set_xlabel("Sample budget")
    ax.set_ylabel("Delta exact success (pp)")
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    finish_axes(ax, xgrid=False, ygrid=True)
    fig.subplots_adjust(top=0.83, left=0.13, right=0.98, bottom=0.13)
    png_path, vector_path = save_figure(fig, refs.output_dir, "delta_success_vs_budget")
    sources = [str(refs.pass1_root / "success_vs_budget.csv")]
    spec = FIGURE_SPECS["delta_success_vs_budget"]
    return FigureRecord("delta_success_vs_budget", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_hard_state_heatmap(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    paired = tables["paired"].copy()
    paired["entropy_bucket"] = safe_quantile_buckets(
        paired["initial_posterior_entropy"],
        ["Q1", "Q2", "Q3", "Q4"],
    )
    paired["support_bucket"] = safe_quantile_buckets(
        paired["initial_candidate_count"],
        ["Small", "Medium", "Large"],
    )
    grouped = (
        paired.groupby(["support_bucket", "entropy_bucket"], dropna=False)
        .agg(
            delta=("rl_success", lambda s: float(s.mean()) - float(paired.loc[s.index, "greedy_success"].mean())),
            case_count=("case_id", "count"),
        )
        .reset_index()
    )
    support_order = ["Large", "Medium", "Small"]
    entropy_order = ["Q1", "Q2", "Q3", "Q4"]
    pivot = (
        grouped.assign(delta_pp=lambda df: df["delta"] * 100.0)
        .pivot(index="support_bucket", columns="entropy_bucket", values="delta_pp")
        .reindex(index=support_order, columns=entropy_order)
    )
    count_pivot = grouped.pivot(index="support_bucket", columns="entropy_bucket", values="case_count").reindex(
        index=support_order,
        columns=entropy_order,
    )
    value_max = float(np.nanmax(np.abs(pivot.values)))
    norm = TwoSlopeNorm(vmin=-value_max, vcenter=0.0, vmax=value_max)
    fig, ax = plt.subplots(figsize=(5.95, 4.65))
    figure_heading(
        fig,
        "Exact-hit delta across hard-state buckets",
        "Descriptive entropy-by-support heatmap; support size is used because the frozen margin bucket collapsed and was not cleanly 2D-plottable.",
    )
    im = ax.imshow(pivot.values, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(entropy_order)))
    ax.set_yticks(np.arange(len(support_order)))
    ax.set_xticklabels(entropy_order)
    ax.set_yticklabels([f"{label} support" for label in support_order])
    ax.set_xlabel("Initial entropy bucket")
    ax.set_ylabel("Initial support-size bucket")
    for row_idx, support_bucket in enumerate(support_order):
        for col_idx, entropy_bucket in enumerate(entropy_order):
            val = pivot.loc[support_bucket, entropy_bucket]
            count_value = count_pivot.loc[support_bucket, entropy_bucket]
            count = 0 if pd.isna(count_value) else int(count_value)
            if pd.isna(val):
                label = "NA"
                text_color = PALETTE["axis"]
            else:
                label = f"{val:+.1f} pp\nn={count}"
                text_color = "white" if abs(float(val)) >= max(3.0, value_max * 0.55) else PALETTE["axis"]
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8.0, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("RL minus Greedy (pp)")
    ax.grid(False)
    fig.subplots_adjust(top=0.83, left=0.18, right=0.92, bottom=0.14)
    png_path, vector_path = save_figure(fig, refs.output_dir, "hard_state_heatmap")
    sources = [
        str(refs.pass1_root / "difficulty_bucket_tables.csv"),
        str(refs.pass2_root / "hard_state_mechanism_tables.csv"),
        str(refs.pass1_root / "run_manifest.json"),
        str(refs.rl_case_rows_path),
        str(refs.teacher5_case_rows_path),
        str(refs.teacher_full_step_rows_path),
    ]
    spec = FIGURE_SPECS["hard_state_heatmap"]
    return FigureRecord("hard_state_heatmap", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_paired_case_taxonomy(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    case_taxonomy = tables["case_taxonomy"].set_index("taxonomy_group").loc[CASE_GROUP_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(6.55, 4.25))
    figure_heading(
        fig,
        "Paired case taxonomy",
        "Counts and shares across the six paired RL-versus-Greedy outcome categories under the same held-out B30 contract.",
    )
    y = np.arange(len(case_taxonomy))
    ax.hlines(
        y,
        0.0,
        case_taxonomy["percentage"],
        color=[case_group_color(group) for group in case_taxonomy["taxonomy_group"]],
        linewidth=2.2,
        zorder=2,
    )
    ax.scatter(
        case_taxonomy["percentage"],
        y,
        s=58,
        color=[case_group_color(group) for group in case_taxonomy["taxonomy_group"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for idx, row in case_taxonomy.iterrows():
        ax.text(
            min(float(row["percentage"]) + 0.02, 0.71),
            idx,
            f"{int(row['case_count'])} | {row['percentage'] * 100.0:.1f}%",
            ha="left",
            va="center",
            fontsize=8.2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([CASE_GROUP_LABELS[group] for group in case_taxonomy["taxonomy_group"]])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.70)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of held-out cases")
    finish_axes(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(top=0.83, left=0.31, right=0.97, bottom=0.14)
    png_path, vector_path = save_figure(fig, refs.output_dir, "paired_case_taxonomy")
    sources = [str(refs.pass1_root / "case_taxonomy.csv")]
    spec = FIGURE_SPECS["paired_case_taxonomy"]
    return FigureRecord("paired_case_taxonomy", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_policy_overlap_summary(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    row = tables["policy_behavior"][tables["policy_behavior"]["slice"] == "overall"].iloc[0]
    metrics = pd.DataFrame(
        {
            "metric": ["Exact 3-set match", "Mean Jaccard overlap", "Mean slot overlap"],
            "value": [
                float(row["exact_action_set_match_rate"]),
                float(row["mean_jaccard_overlap"]),
                float(row["mean_slot_overlap_fraction"]),
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.75))
    figure_heading(
        fig,
        "Set-level overlap with Greedy",
        "Lower values indicate less imitation; all three summaries stay far from full overlap.",
    )
    y = np.arange(len(metrics))
    ax.hlines(y, 0.0, metrics["value"], color=PALETTE["neutral_light"], linewidth=2.0, zorder=2)
    ax.scatter(metrics["value"], y, s=62, color=PALETTE["strongest_rl"], edgecolor="white", linewidth=0.8, zorder=3)
    for idx, value in enumerate(metrics["value"]):
        ax.text(min(float(value) + 0.03, 0.97), idx, f"{value * 100.0:.1f}%", ha="left", va="center", fontsize=8.3)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics["metric"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Overlap with Greedy Posterior")
    finish_axes(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(top=0.82, left=0.30, right=0.97, bottom=0.16)
    png_path, vector_path = save_figure(fig, refs.output_dir, "policy_overlap_summary")
    sources = [str(refs.pass1_root / "policy_behavior_tables.csv")]
    spec = FIGURE_SPECS["policy_overlap_summary"]
    return FigureRecord("policy_overlap_summary", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_hit_round_ecdf(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    rl_case = tables["rl_case"]
    greedy_case = tables["greedy_case"]
    rounds = np.arange(1, 11)
    rl_curve = [float(((rl_case["success_rate"] > 0.5) & (rl_case["hit_round"] <= round_idx)).mean()) for round_idx in rounds]
    greedy_curve = [float(((greedy_case["success_rate"] > 0.5) & (greedy_case["hit_round"] <= round_idx)).mean()) for round_idx in rounds]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    figure_heading(
        fig,
        "Solved by round",
        "Curves are computed over all held-out cases, so they end below 100% when B30 misses remain unresolved.",
    )
    ax.step(rounds, rl_curve, where="post", color=PALETTE["strongest_rl"], linewidth=2.4, label="Strongest RL")
    ax.step(rounds, greedy_curve, where="post", color=PALETTE["posterior_greedy"], linewidth=2.2, label="Greedy Posterior")
    ax.scatter(rounds, rl_curve, color=PALETTE["strongest_rl"], s=18, zorder=3)
    ax.scatter(rounds, greedy_curve, color=PALETTE["posterior_greedy"], marker="s", s=18, zorder=3)
    ax.set_xlim(1, 10)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Hit round")
    ax.set_ylabel("Share solved by round")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    finish_axes(ax, xgrid=False, ygrid=True)
    ax.legend(loc="lower right")
    fig.subplots_adjust(top=0.83, left=0.12, right=0.98, bottom=0.13)
    png_path, vector_path = save_figure(fig, refs.output_dir, "hit_round_ecdf")
    sources = [str(refs.pass1_root / "run_manifest.json"), str(refs.rl_case_rows_path), str(refs.teacher5_case_rows_path)]
    spec = FIGURE_SPECS["hit_round_ecdf"]
    return FigureRecord("hit_round_ecdf", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_hit_round_scatter(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    paired = tables["paired"].copy()
    rng = np.random.default_rng(20260421)
    paired["greedy_plot_round"] = paired["greedy_hit_round"].fillna(ROUND_FAIL_SENTINEL)
    paired["rl_plot_round"] = paired["rl_hit_round"].fillna(ROUND_FAIL_SENTINEL)
    paired["x_jitter"] = rng.uniform(-0.12, 0.12, size=len(paired))
    paired["y_jitter"] = rng.uniform(-0.12, 0.12, size=len(paired))
    fig, ax = plt.subplots(figsize=(6.1, 5.0))
    figure_heading(
        fig,
        "Paired hit-round comparison",
        "Round 11 denotes a B30 miss; points below the diagonal favor RL, while points above favor Greedy.",
    )
    for group in CASE_GROUP_ORDER:
        sub = paired[paired["taxonomy_group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["greedy_plot_round"] + sub["x_jitter"],
            sub["rl_plot_round"] + sub["y_jitter"],
            s=22,
            alpha=0.58,
            color=case_group_color(group),
            edgecolor="white",
            linewidth=0.25,
            label=CASE_GROUP_LABELS[group],
            zorder=3,
        )
    ax.plot([1, ROUND_FAIL_SENTINEL], [1, ROUND_FAIL_SENTINEL], color=PALETTE["neutral"], linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(10.5, color=PALETTE["neutral_light"], linewidth=0.9, linestyle=":")
    ax.axhline(10.5, color=PALETTE["neutral_light"], linewidth=0.9, linestyle=":")
    ax.set_xlim(0.7, 11.35)
    ax.set_ylim(0.7, 11.35)
    ticks = list(range(1, 11)) + [11]
    labels = [str(x) for x in range(1, 11)] + ["miss"]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Greedy hit round")
    ax.set_ylabel("RL hit round")
    finish_axes(ax, xgrid=False, ygrid=False)
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.96)
    fig.subplots_adjust(top=0.83, left=0.14, right=0.98, bottom=0.13)
    png_path, vector_path = save_figure(fig, refs.output_dir, "hit_round_scatter")
    sources = [str(refs.pass1_root / "run_manifest.json"), str(refs.rl_case_rows_path), str(refs.teacher5_case_rows_path)]
    spec = FIGURE_SPECS["hit_round_scatter"]
    return FigureRecord("hit_round_scatter", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_hybrid_replay_ladder(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    complementarity = tables["complementarity"]
    subset = complementarity[
        (complementarity["slice"] == "overall_subset") & (complementarity["analysis_type"] == "targeted_hybrid_replay")
    ].set_index("strategy").loc[HYBRID_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    figure_heading(
        fig,
        "Hybrid replay ladder",
        "Bounded 24-case hard subset only; synthetic prefix/suffix replay is mechanism evidence, not a new full-panel benchmark.",
    )
    y = np.arange(len(subset))
    x = subset["success_rate"] * 100.0
    ax.plot(x, y, color=PALETTE["neutral_light"], linewidth=1.5, zorder=1)
    for idx, row in subset.iterrows():
        strategy = row["strategy"]
        color = method_color(strategy) if strategy in {"strongest_rl", "posterior_greedy"} else PALETTE["boundary_dark"]
        ax.scatter(x.iloc[idx], y[idx], s=62, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(min(float(x.iloc[idx]) + 1.8, 98.5), y[idx], f"{x.iloc[idx]:.1f}%", ha="left", va="center", fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels([HYBRID_LABELS[strategy] for strategy in subset["strategy"]])
    ax.invert_yaxis()
    ax.set_xlim(50.0, 100.0)
    ax.set_xlabel("Subset exact success")
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.0f}%")
    finish_axes(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(top=0.83, left=0.27, right=0.97, bottom=0.15)
    png_path, vector_path = save_figure(fig, refs.output_dir, "hybrid_replay_ladder")
    sources = [str(refs.pass2_root / "complementarity_tables.csv"), str(refs.pass2_root / "run_manifest.json")]
    spec = FIGURE_SPECS["hybrid_replay_ladder"]
    return FigureRecord("hybrid_replay_ladder", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_slot_tendency(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    slot_rows = tables["targeted_slot_rows"]
    subset = slot_rows[slot_rows["strategy"].isin(["strongest_rl", "posterior_greedy"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.95))
    figure_heading(
        fig,
        "Slot-level tendency on the bounded subset",
        "Point ranges show bootstrap mean 95% CIs; slot 1 has no previous-pick distance by definition.",
    )
    posterior_ax, hop_ax = axes
    slot_positions = np.array([1, 2, 3], dtype=float)
    offsets = {"strongest_rl": -0.12, "posterior_greedy": 0.12}
    markers = {"strongest_rl": "o", "posterior_greedy": "s"}
    labels = {"strongest_rl": "Strongest RL", "posterior_greedy": "Greedy Posterior"}
    seed_base = 700
    for idx, strategy in enumerate(["strongest_rl", "posterior_greedy"]):
        sub = subset[subset["strategy"] == strategy]
        xs: List[float] = []
        means: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        for slot in [1, 2, 3]:
            values = sub[sub["slot_index"] == slot]["posterior_rank"].tolist()
            mean, lo, hi = bootstrap_mean_ci(values, seed=seed_base + idx * 10 + slot)
            xs.append(slot + offsets[strategy])
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        posterior_ax.errorbar(
            xs,
            means,
            yerr=[lower, upper],
            color=method_color(strategy),
            marker=markers[strategy],
            linestyle="-",
            linewidth=1.8,
            markersize=5.0,
            capsize=3.0,
            label=labels[strategy],
            zorder=3,
        )
        xs = []
        means = []
        lower = []
        upper = []
        for slot in [2, 3]:
            values = sub[sub["slot_index"] == slot]["prev_selected_hop_mean"].tolist()
            mean, lo, hi = bootstrap_mean_ci(values, seed=seed_base + 100 + idx * 10 + slot)
            xs.append(slot + offsets[strategy])
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        hop_ax.errorbar(
            xs,
            means,
            yerr=[lower, upper],
            color=method_color(strategy),
            marker=markers[strategy],
            linestyle="-",
            linewidth=1.8,
            markersize=5.0,
            capsize=3.0,
            label=labels[strategy],
            zorder=3,
        )
    posterior_ax.set_xticks(slot_positions)
    posterior_ax.set_xlabel("Slot index")
    posterior_ax.set_ylabel("Posterior rank")
    posterior_ax.set_title("Posterior rank", loc="left", pad=6)
    finish_axes(posterior_ax, xgrid=False, ygrid=True)
    hop_ax.set_xticks([2, 3])
    hop_ax.set_xlabel("Slot index")
    hop_ax.set_ylabel("Hop to previous pick")
    hop_ax.set_title("Mean hop to previous pick", loc="left", pad=6)
    finish_axes(hop_ax, xgrid=False, ygrid=True)
    handles, legend_labels = posterior_ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(top=0.81, left=0.09, right=0.99, bottom=0.22, wspace=0.28)
    png_path, vector_path = save_figure(fig, refs.output_dir, "slot_tendency")
    sources = [
        str(refs.pass2_root / "slot_role_tendency_tables.csv"),
        str(refs.pass2_root / "targeted_hybrid_replay_slot_rows.csv"),
        str(refs.pass2_root / "run_manifest.json"),
    ]
    spec = FIGURE_SPECS["slot_tendency"]
    return FigureRecord("slot_tendency", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_relaxed_radius_curve(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    relaxed_metric = tables["relaxed_metric"]
    subset = relaxed_metric[
        (relaxed_metric["sample_budget"] == 30)
        & (relaxed_metric["method_id"].isin(["strongest_rl", "posterior_greedy"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(5.75, 3.8))
    figure_heading(
        fig,
        "Relaxed-radius success at B30",
        "Secondary metric only; exact node-level hit remains the primary endpoint in the paper package.",
    )
    for method_id, marker in [("strongest_rl", "o"), ("posterior_greedy", "s")]:
        sub = subset[subset["method_id"] == method_id].sort_values("radius_hops")
        ax.plot(
            sub["radius_hops"],
            sub["success_rate"],
            color=method_color(method_id),
            linewidth=2.0,
            marker=marker,
            markersize=5.0,
            label=sub["display_name"].iloc[0],
        )
    ax.set_xlim(-0.05, 2.05)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Radius (hops)")
    ax.set_ylabel("Relaxed success")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    finish_axes(ax, xgrid=False, ygrid=True)
    ax.legend(loc="lower right")
    fig.subplots_adjust(top=0.83, left=0.12, right=0.98, bottom=0.16)
    png_path, vector_path = save_figure(fig, refs.output_dir, "relaxed_radius_curve")
    sources = [str(refs.pass1_root / "relaxed_metric_tables.csv")]
    spec = FIGURE_SPECS["relaxed_radius_curve"]
    return FigureRecord("relaxed_radius_curve", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def plot_frontier_boundary_check(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    frontier = tables["frontier"]
    subset = frontier[
        (frontier["table_scope"] == "targeted_slot_frontier_source") & (frontier["slice"] == "overall_subset")
    ].copy()
    order = ["posterior", "disagreement", "novelty", "fill", "outside_slate"]
    subset = subset.set_index("frontier_source").reindex(order).reset_index()
    color_map = {
        "posterior": PALETTE["boundary_dark"],
        "disagreement": "#9DA7B0",
        "novelty": "#D5D2D0",
        "fill": "#E1DEDC",
        "outside_slate": "#ECE9E8",
    }
    fig, ax = plt.subplots(figsize=(6.05, 3.85))
    figure_heading(
        fig,
        "Frontier boundary check",
        "Boundary-only appendix panel: the targeted subset remains posterior-dominated, so a strong mixed-frontier mechanism claim is not supported.",
    )
    y = np.arange(len(subset))
    ax.barh(
        y,
        subset["fraction"],
        color=[color_map[source] for source in subset["frontier_source"]],
        edgecolor="white",
        linewidth=0.8,
        height=0.7,
        zorder=3,
    )
    for idx, value in enumerate(subset["fraction"]):
        ax.text(min(float(value) + 0.02, 0.99), idx, f"{value * 100.0:.1f}%", ha="left", va="center", fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels([source.replace("_", " ") for source in subset["frontier_source"]])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of selected nodes")
    finish_axes(ax, xgrid=True, ygrid=False)
    fig.subplots_adjust(top=0.82, left=0.28, right=0.98, bottom=0.16)
    png_path, vector_path = save_figure(fig, refs.output_dir, "frontier_boundary_check")
    sources = [str(refs.pass2_root / "frontier_composition_tables.csv")]
    spec = FIGURE_SPECS["frontier_boundary_check"]
    return FigureRecord("frontier_boundary_check", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def select_representative_case_ids(representative_manifest: pd.DataFrame) -> pd.DataFrame:
    required_groups = ["rl_unique_win", "both_hit_rl_earlier", "greedy_unique_win", "both_fail"]
    rows = []
    for group in required_groups:
        sub = representative_manifest[representative_manifest["taxonomy_group"] == group]
        if sub.empty:
            raise RuntimeError(f"Missing representative case for group {group}")
        rows.append(sub.iloc[0])
    return pd.DataFrame(rows)


def build_representative_case_key(representative_manifest: pd.DataFrame) -> List[Dict[str, Any]]:
    selected = select_representative_case_ids(representative_manifest)
    key_rows: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(selected.iterrows()):
        key_rows.append(
            {
                "case_label": representative_case_label(idx),
                "taxonomy_group": str(row["taxonomy_group"]),
                "taxonomy_label": CASE_GROUP_LABELS[str(row["taxonomy_group"])],
                "case_id": str(row["case_id"]),
                "scenario_id": scenario_short(str(row["case_id"])),
                "initial_entropy": float(row["initial_entropy"]),
                "rl_hit_round": format_hit(row["rl_hit_round"]),
                "greedy_hit_round": format_hit(row["greedy_hit_round"]),
            }
        )
    return key_rows


def trajectory_for_case(step_df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    case_df = step_df[step_df["case_id"] == case_id].copy()
    order_col = "round_index" if "round_index" in case_df.columns else "episode_index"
    return case_df.sort_values(order_col).copy()


def plot_representative_case_panels(tables: Mapping[str, Any], refs: BundleRefs) -> tuple[FigureRecord, List[str]]:
    selected = select_representative_case_ids(tables["representative_manifest"])
    case_labels = {str(row["case_id"]): representative_case_label(idx) for idx, (_, row) in enumerate(selected.iterrows())}
    rl_step = tables["rl_step"]
    greedy_step = tables["greedy_step"]
    fig = plt.figure(figsize=(10.4, 7.1))
    figure_heading(
        fig,
        "Representative paired cases",
        "One frozen representative case per required category; panels illustrate the paired taxonomy but do not replace the aggregate evidence.",
    )
    outer = fig.add_gridspec(2, 2, left=0.05, right=0.99, top=0.82, bottom=0.06, hspace=0.30, wspace=0.18)
    legend_handles = [
        Line2D([0], [0], color=PALETTE["strongest_rl"], linewidth=2.1, marker="o", markersize=4.5, label="Strongest RL"),
        Line2D([0], [0], color=PALETTE["posterior_greedy"], linewidth=2.0, marker="s", markersize=4.5, label="Greedy Posterior"),
    ]
    for idx, (_, row) in enumerate(selected.iterrows()):
        case_id = row["case_id"]
        case_label = case_labels[str(case_id)]
        inner = outer[idx // 2, idx % 2].subgridspec(2, 2, height_ratios=[0.34, 1.0], hspace=0.10, wspace=0.20)
        header_ax = fig.add_subplot(inner[0, :])
        entropy_ax = fig.add_subplot(inner[1, 0])
        mass_ax = fig.add_subplot(inner[1, 1])
        header_ax.axis("off")
        group = str(row["taxonomy_group"])
        header_ax.text(
            0.0,
            0.92,
            CASE_GROUP_LABELS[group],
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color=case_group_color(group),
            transform=header_ax.transAxes,
        )
        header_ax.text(
            0.0,
            0.35,
            f"{case_label} | H0={float(row['initial_entropy']):.2f} | RL hit={format_hit(row['rl_hit_round'])} | G hit={format_hit(row['greedy_hit_round'])}",
            ha="left",
            va="top",
            fontsize=8.0,
            color=PALETTE["axis"],
            transform=header_ax.transAxes,
        )
        rl_case = trajectory_for_case(rl_step, case_id)
        greedy_case = trajectory_for_case(greedy_step, case_id)
        entropy_ax.plot(
            rl_case["round_index"],
            rl_case["posterior_entropy"],
            color=PALETTE["strongest_rl"],
            linewidth=1.9,
            marker="o",
            markersize=3.4,
        )
        entropy_ax.plot(
            greedy_case["episode_index"],
            greedy_case["posterior_entropy"],
            color=PALETTE["posterior_greedy"],
            linewidth=1.8,
            marker="s",
            markersize=3.2,
        )
        entropy_ax.set_title("Posterior entropy", loc="left", pad=4, fontsize=8.8)
        entropy_ax.set_xlim(1, 10)
        entropy_ax.set_xticks([1, 5, 10])
        entropy_ax.set_xlabel("Round")
        entropy_ax.set_ylabel("Entropy")
        finish_axes(entropy_ax, xgrid=False, ygrid=True)
        mass_ax.plot(
            rl_case["round_index"],
            rl_case["top3_mass"],
            color=PALETTE["strongest_rl"],
            linewidth=1.9,
            marker="o",
            markersize=3.4,
        )
        mass_ax.plot(
            greedy_case["episode_index"],
            greedy_case["top3_mass"],
            color=PALETTE["posterior_greedy"],
            linewidth=1.8,
            marker="s",
            markersize=3.2,
        )
        mass_ax.set_title("Top-3 mass", loc="left", pad=4, fontsize=8.8)
        mass_ax.set_xlim(1, 10)
        mass_ax.set_xticks([1, 5, 10])
        mass_ax.set_xlabel("Round")
        mass_ax.set_ylabel("Top-3 mass")
        finish_axes(mass_ax, xgrid=False, ygrid=True)
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 0.93), ncol=2)
    png_path, vector_path = save_figure(fig, refs.output_dir, "representative_case_panels")
    sources = [
        str(refs.pass2_root / "representative_case_figures" / "figure_manifest.csv"),
        str(refs.pass1_root / "representative_cases.json"),
        str(refs.pass1_root / "run_manifest.json"),
        str(refs.rl_step_rows_path),
        str(refs.teacher5_step_rows_path),
    ]
    spec = FIGURE_SPECS["representative_case_panels"]
    return FigureRecord("representative_case_panels", sources, spec["intended_use"], spec["short_claim_supported"], spec["caution_level"], png_path, vector_path), sources


def build_style_guide() -> str:
    return """# Style Guide

## Shared visual system
- Font family: `DejaVu Sans` for all figure text.
- Title style: left-aligned bold figure heading plus one muted subtitle line.
- Axis style: open-top/open-right axes, 0.9 pt axis lines, light horizontal or vertical grid only where it aids comparison.
- Line widths: main lines `2.2-2.6 pt`, secondary lines `1.3-2.0 pt`.
- Marker sizes: `3.2-5.0 pt`, with consistent circle/square usage for RL vs Greedy where paired comparison matters.
- Background: off-white panel background (`#FBF9F8`) with restrained gray grid (`#E1DBD8`).
- Export: PNG at `400 dpi` plus vector PDF for every figure.

## Palette
- Strongest RL: `#9E4254`
- Greedy Posterior: `#D8877B`
- Thompson Sampling: `#C9A07A`
- Weaker heuristics: `#5E7D97`, `#7F98AB`, `#A6B9C7`
- Neutral / boundary panels: `#6A6664`, `#948B88`, `#C9C1BE`
- Hard-state delta heatmap: muted blue to off-white to muted rose diverging scale

## Canvas family
- Most single-panel charts: about `6.2-6.5 in` wide and `3.8-4.3 in` tall.
- Heatmap / scatter panels: about `6.0-6.1 in` wide and `4.6-5.0 in` tall.
- Representative case montage: `10.4 x 7.1 in`.

## Interpretation guardrails encoded in style
- Headline figures use the strongest contrast only for `Strongest RL`; comparison baselines remain fully readable.
- Appendix-only boundary checks use grays and muted tones rather than celebratory colors.
- Secondary metrics and bounded-subset mechanism panels are explicitly labeled in subtitles.
- No truncated success-rate bars, no rainbow palette, and no decorative effects were used.
"""


def build_figure_notes() -> str:
    return """# Figure Notes

## main_comparison_bar
- What it shows: held-out exact `Success@B30` for Strongest RL, Greedy Posterior, Thompson Sampling, and the weaker heuristic baselines under the aligned val / B30 / SPIM-v3 contract.
- Safe interpretation: this is the headline empirical comparison under the locked exact-hit contract.
- What it does NOT prove: it does not justify claims about other datasets, other budgets, or other evaluation contracts.

## success_vs_budget
- What it shows: cumulative exact success as sample budget increases, with RL, Greedy Posterior, and Thompson emphasized.
- Safe interpretation: RL's advantage appears before budget exhaustion and is not purely a final-budget artifact.
- What it does NOT prove: it does not by itself identify the causal mechanism of the gain.

## delta_success_vs_budget
- What it shows: `RL success - Greedy success` in percentage points across budget.
- Safe interpretation: use it as a direct view of where the RL gain opens and how large it becomes.
- What it does NOT prove: it does not prove why the gap opens.

## hard_state_heatmap
- What it shows: a descriptive 2D map of exact-hit delta across initial entropy quartiles and support-size tertiles.
- Safe interpretation: larger positive cells are consistent with RL helping more in harder initial states.
- What it does NOT prove: it is not a causal mechanism plot, and it should not be read as proof of early ambiguity reduction.
- Boundary note: support-size buckets were used instead of margin buckets because the frozen margin evidence collapsed and did not support a clean 2D matrix.

## paired_case_taxonomy
- What it shows: the six paired RL-versus-Greedy case categories.
- Safe interpretation: the net gain combines RL-only wins with a large both-hit-but-RL-earlier bucket.
- What it does NOT prove: it does not imply RL wins on every type of case.

## policy_overlap_summary
- What it shows: exact 3-set match rate, mean Jaccard overlap, and mean slot overlap between RL and Greedy.
- Safe interpretation: RL is materially different from Greedy at the selected-set level.
- What it does NOT prove: it does not identify the alternative mechanism beyond “not simple imitation”.

## hit_round_ecdf
- What it shows: the share of all held-out cases solved by each round.
- Safe interpretation: the curve ends below 100% because B30 misses remain unresolved.
- What it does NOT prove: it does not replace the sample-budget curves, which remain the primary dynamic panel.

## hit_round_scatter
- What it shows: paired RL and Greedy hit rounds, with `miss` encoded explicitly.
- Safe interpretation: points below the diagonal favor RL and points above the diagonal favor Greedy.
- What it does NOT prove: it does not by itself establish a mechanism.

## hybrid_replay_ladder
- What it shows: bounded-subset synthetic hybrid replay success across Greedy, mixed prefixes/suffixes, and RL.
- Safe interpretation: this is bounded mechanism evidence only, consistent with later-pick completion being more important than a pure first-pick explanation.
- What it does NOT prove: it does not prove global suboptimality of the original RL checkpoint and it is not a new headline benchmark.

## slot_tendency
- What it shows: bounded-subset slot statistics for posterior rank and hop-to-previous.
- Safe interpretation: the evidence supports a slot-level tendency only.
- What it does NOT prove: it does not prove three fixed semantic heads or stable semantic roles.

## relaxed_radius_curve
- What it shows: relaxed-radius success at B30 for RL and Greedy.
- Safe interpretation: this is secondary context only; exact hit remains primary.
- What it does NOT prove: it does not redefine the task objective.

## frontier_boundary_check
- What it shows: selected-node frontier-source fractions on the bounded targeted replay subset.
- Safe interpretation: the policy remains strongly posterior-dominated there, so a strong novelty-frontier or mixed-frontier mechanism claim is not supported.
- What it does NOT prove: it should not be used as positive evidence for a frontier-mixing story.

## representative_case_panels
- What it shows: one frozen representative case per required paired-outcome category, re-rendered with the same style family as the rest of the package.
- Safe interpretation: these are illustrative examples anchored to already-frozen representative selections.
- What it does NOT prove: representative examples do not replace the aggregate comparisons or statistical case taxonomy.
"""


def figure_manifest_dataframe(records: Sequence[FigureRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "figure_name": record.figure_name,
                "source_artifact_files": json.dumps(record.source_artifact_files),
                "intended_use": record.intended_use,
                "short_claim_supported": record.short_claim_supported,
                "caution_level": record.caution_level,
                "png_path": record.png_path,
                "vector_path": record.vector_path,
            }
        )
    return pd.DataFrame(rows)


def write_supporting_files(
    refs: BundleRefs,
    records: Sequence[FigureRecord],
    extra_sources: Iterable[str],
    representative_case_key: Sequence[Mapping[str, Any]],
) -> None:
    manifest_df = figure_manifest_dataframe(records)
    manifest_df.to_csv(refs.output_dir / "figure_manifest.csv", index=False)
    (refs.output_dir / "style_guide.md").write_text(build_style_guide(), encoding="utf-8")
    (refs.output_dir / "figure_notes.md").write_text(build_figure_notes(), encoding="utf-8")
    (refs.output_dir / "representative_case_key.json").write_text(
        json.dumps(list(representative_case_key), indent=2),
        encoding="utf-8",
    )
    source_files = sorted({*extra_sources, *(src for record in records for src in record.source_artifact_files)})
    run_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(refs.output_dir),
        "pass1_root": str(refs.pass1_root),
        "pass2_root": str(refs.pass2_root),
        "pass3_root": str(refs.pass3_root),
        "execution_constraints": {
            "new_training_run": False,
            "new_broad_evaluation": False,
            "result_values_changed": False,
            "frozen_artifact_only": True,
        },
        "shared_style": {
            "font_family": "DejaVu Sans",
            "export_dpi_png": EXPORT_DPI,
            "vector_format": "pdf",
            "palette": {
                "strongest_rl": PALETTE["strongest_rl"],
                "posterior_greedy": PALETTE["posterior_greedy"],
                "posterior_thompson_sampling": PALETTE["posterior_thompson_sampling"],
                "weaker_heuristics": [
                    PALETTE["posterior_entropy_drop"],
                    PALETTE["posterior_cover_shrink"],
                    PALETTE["posterior_disagreement_split"],
                ],
            },
        },
        "generated_figures": [record.figure_name for record in records],
        "supporting_outputs": [
            "figure_manifest.csv",
            "style_guide.md",
            "figure_notes.md",
            "representative_case_key.json",
            "run_manifest.json",
        ],
        "source_artifact_files": source_files,
        "representative_case_key_path": str(refs.output_dir / "representative_case_key.json"),
        "main_recommendation_order": MAIN_FIGURE_ORDER,
        "appendix_recommendation_order": APPENDIX_FIGURE_ORDER,
        "unsupported_requested_figures": [],
    }
    (refs.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    refs = build_bundle_refs(args)
    apply_publication_style()
    tables = load_tables(refs)
    figure_builders = [
        plot_main_comparison_bar,
        plot_success_vs_budget,
        plot_delta_success_vs_budget,
        plot_hard_state_heatmap,
        plot_paired_case_taxonomy,
        plot_policy_overlap_summary,
        plot_hit_round_ecdf,
        plot_hit_round_scatter,
        plot_hybrid_replay_ladder,
        plot_slot_tendency,
        plot_relaxed_radius_curve,
        plot_frontier_boundary_check,
        plot_representative_case_panels,
    ]
    records: List[FigureRecord] = []
    all_sources: List[str] = []
    for builder in figure_builders:
        record, sources = builder(tables, refs)
        records.append(record)
        all_sources.extend(sources)
    representative_case_key = build_representative_case_key(tables["representative_manifest"])
    write_supporting_files(refs, records, all_sources, representative_case_key)


if __name__ == "__main__":
    main()
