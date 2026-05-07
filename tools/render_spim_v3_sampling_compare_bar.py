#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "spim_sampling_compare_20260416"


PALETTE = {
    "bg": "#F5F1EA",
    "panel": "#FFFDFC",
    "grid": "#D7CDC0",
    "ink": "#1E2A32",
    "muted": "#5C6B73",
    "greedy": "#5B7C99",
    "uncertainty": "#7FAE8A",
    "entropy": "#D5A35B",
    "rl": "#B5523B",
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
            "axes.facecolor": PALETTE["panel"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "axes.titlecolor": PALETTE["ink"],
            "font.family": "Noto Sans CJK JP",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.45,
            "mathtext.fontset": "stix",
        }
    )


def read_success_rate(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["summary"]["success_rate"])


def read_family_success_rate(path: Path, family: str) -> float:
    df = pd.read_csv(path)
    row = df.loc[df["family"] == family]
    if row.empty:
        raise KeyError(f"family not found: {family} in {path}")
    return float(row.iloc[0]["success_rate"])


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_summary_path = (
        PROJECT_ROOT / "artifacts/spim_family_sweep/20260410_exact136_b30_clean_spim5_v2_full/family_summary.csv"
    )

    # All values are loaded from repository artifacts instead of handwritten literals.
    methods = [
        {
            "label": "Greedy\n(v3 posterior)",
            "value": read_family_success_rate(family_summary_path, "hsr_soft_scenario_posterior_v3"),
            "source_family": "hsr_soft_scenario_posterior_v3",
            "color": PALETTE["greedy"],
        },
        {
            "label": "Uncertainty\n(rank-weighted)",
            "value": read_family_success_rate(family_summary_path, "hsr_rank_weighted_topk_v2"),
            "source_family": "hsr_rank_weighted_topk_v2",
            "color": PALETTE["uncertainty"],
        },
        {
            "label": "Entropy\n(binary source-only)",
            "value": read_family_success_rate(family_summary_path, "hsr_binary_loglik_posterior_source_only_v4"),
            "source_family": "hsr_binary_loglik_posterior_source_only_v4",
            "color": PALETTE["entropy"],
        },
        {
            "label": "RL-best\n(set interaction)",
            "value": read_success_rate(
                PROJECT_ROOT
                / "artifacts/spim_set_level_rl_mainline/20260416_v3_setint_uncert_bounded_v1/stage1_schemeS_seed45/strict_eval_val_B30/rl_set_seed45_schemeS_setint_train4823/summary.json"
            ),
            "source_family": "schemeS_set_interaction",
            "color": PALETTE["rl"],
        },
        {
            "label": "RL-best +\nuncertainty",
            "value": read_success_rate(
                PROJECT_ROOT
                / "artifacts/spim_set_level_rl_mainline/20260416_v3_setint_uncert_bounded_v1/stage2_schemeU_seed45/strict_eval_val_B30/rl_set_seed45_schemeU_setint_uncert_train4823/summary.json"
            ),
            "source_family": "schemeU_set_interaction_uncert",
            "color": "#D97A4A",
        },
    ]

    values = np.array([m["value"] * 100.0 for m in methods], dtype=np.float64)
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=(0, (4, 4)), linewidth=0.9)
    ax.xaxis.grid(False)

    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=[m["color"] for m in methods],
        edgecolor="#F7F3EE",
        linewidth=1.2,
        zorder=3,
    )

    ax.axhline(90.0, color="#8A4B37", linestyle=(0, (3, 3)), linewidth=1.0, alpha=0.9, zorder=2)
    ax.text(4.46, 90.7, "90%", ha="right", va="bottom", fontsize=9.5, color="#8A4B37")

    for idx, (bar, item) in enumerate(zip(bars, methods)):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.9,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=PALETTE["ink"],
            fontweight="semibold" if "RL-best" in item["label"] else "normal",
        )
        ax.text(
            idx,
            71.6,
            item["source_family"],
            ha="center",
            va="bottom",
            fontsize=7.9,
            color=PALETTE["muted"],
            rotation=90,
        )

    ax.set_ylim(70, 96)
    ax.set_yticks(np.arange(70, 97, 5))
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(x, [m["label"] for m in methods])
    ax.set_title("SPIM v3 Sampling Strategy Comparison Under the Strongest RL Regime", pad=12)
    ax.text(
        0.0,
        1.01,
        "Exact136/B30 benchmark; repository artifact values; RL bars use strict val_B30 evaluation",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=PALETTE["muted"],
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])

    fig.tight_layout()
    png_path = OUTPUT_DIR / "spim_v3_sampling_compare_bar.png"
    pdf_path = OUTPUT_DIR / "spim_v3_sampling_compare_bar.pdf"
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
