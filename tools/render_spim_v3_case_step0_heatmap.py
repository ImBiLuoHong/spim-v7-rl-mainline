from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_audit import load_runtime_context
from src.scripts.run_reasoner_same_case_stronger_source_overfit import CaseRecord, make_rollout_state, read_json
from src.scripts.run_spim_family_sweep import _extract_trigger_global, _pick_topk_unsampled, PaperLikeHSRState
from src.scripts.run_spim_teacher_imitation_rl_pilot import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SOURCE_ROOT,
    compute_teacher_belief,
)
from src.scripts.audit.utils_practical_rollout import PracticalRollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SPIM v3 greedy step0 heatmap for one event case.")
    parser.add_argument(
        "--case-tag",
        type=str,
        default="event_20260115_222959_7013_mode_HIGH",
        help="Event tag prefix without _viewA suffix.",
    )
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "spim_case_heatmap" / "20260415_step0_v3_greedy"),
    )
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--actions-per-round", type=int, default=3)
    parser.add_argument("--paper-like-alpha", type=float, default=0.55)
    parser.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    parser.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    parser.add_argument("--soft-scenario-beta", type=float, default=2.0)
    return parser.parse_args()


def _load_cfg_from_source(source_root: Path) -> dict[str, Any]:
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if isinstance(payload.get("life_support"), dict) and str(payload["life_support"].get("profile")) == "custom_direct_edit":
        payload = dict(payload)
        payload["life_support"] = {k: v for k, v in payload["life_support"].items() if k != "profile"}
    return payload


def _build_train_dataset(cfg_payload: dict[str, Any], cache_dir: Path):
    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(cfg_payload)
    cfg.training.enable_eval = False
    cfg.training.train_only = True
    cfg.training.enable_wandb = False
    cfg.data.skip_lmdb = True
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.max_samples = None
    cfg.data.rebuild_cache = False
    cfg.paths.cache_dir = str(cache_dir)

    train_loader, _, _, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1,
        eval_batch_size=1,
        skip_lmdb=True,
        train_only=True,
    )
    return train_loader.dataset, cfg


def _find_case_index(dataset, case_tag: str) -> int:
    needle = f"{case_tag}_viewA.npz"
    for idx, group in enumerate(dataset.groups):
        for p in group:
            if needle in str(p):
                return int(idx)
    raise ValueError(f"case not found in train groups: {needle}")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_payload = _load_cfg_from_source(source_root)
    dataset, cfg = _build_train_dataset(cfg_payload, cache_dir)
    case_idx = _find_case_index(dataset, args.case_tag)
    data = dataset[case_idx]

    topology = HydraulicTopology(cfg.paths.foundation_path)
    runtime_base = load_runtime_context(source_root, cache_dir)
    episode_duration_min = float(runtime_base["episode_duration_min"])
    frontier_role_mode = str(runtime_base["frontier_role_mode"])

    case = CaseRecord(
        case_id=f"train:scenario{case_idx}:part0",
        scenario_id=int(case_idx),
        part_id=0,
        dataset_index=int(case_idx),
        data=deepcopy(data),
    )
    rollout = PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=dataset.global_edge_index,
        stt_dynamic_series=dataset.stt_dynamic_series,
        num_global_nodes=int(dataset.global_pos.shape[0]),
        num_episodes=int(args.num_rounds),
        samples_per_episode=int(args.actions_per_round),
        episode_duration_min=float(episode_duration_min),
    )
    history = ObservationWitnessHistory()
    env = CleanTwoChannelEvidenceEnv()

    state = make_rollout_state(
        case=case,
        rollout=rollout,
        history=history,
        env=env,
        topology=topology,
        num_episodes=int(args.num_rounds),
        action_budget=int(args.actions_per_round),
        frontier_role_mode=str(frontier_role_mode),
    )
    trigger_global = _extract_trigger_global(case.data)
    paper_state = PaperLikeHSRState(source_prior=None)
    onset_grid = [-float(episode_duration_min), 0.0, float(episode_duration_min)]
    belief_ctx = compute_teacher_belief(
        family="hsr_soft_scenario_posterior_v3",
        rollout=rollout,
        state=state,
        history=history,
        trigger_global=trigger_global,
        paper_state=paper_state,
        onset_offsets_min=onset_grid,
        paper_like_alpha=float(args.paper_like_alpha),
        paper_like_topk_fraction=float(args.paper_like_topk_fraction),
        paper_like_time_tol_min=float(args.paper_like_time_tol_min),
        soft_scenario_beta=float(args.soft_scenario_beta),
    )

    belief = belief_ctx["belief"].view(-1).float().cpu()
    candidate_mask = belief_ctx["candidate_mask"].view(-1).bool().cpu()
    probs = torch.zeros_like(belief)
    if bool(candidate_mask.any()):
        probs[candidate_mask] = belief[candidate_mask] / belief[candidate_mask].sum().clamp_min(1e-12)
    selected_locals = _pick_topk_unsampled(probs, candidate_mask, rollout, int(args.actions_per_round))
    selected_globals = [int(rollout.g_ids[int(i)].item()) for i in selected_locals]

    source_local = resolve_source_local_idx(rollout)
    source_global = None if source_local is None else int(rollout.g_ids[int(source_local)].item())
    gids = rollout.g_ids.detach().cpu().numpy().astype(np.int64)
    coords = topology.node_coords[gids]
    values = probs.detach().cpu().numpy()

    edge_index = rollout.sub_edge_index.detach().cpu().numpy()
    seg = np.stack([coords[edge_index[0]], coords[edge_index[1]]], axis=1)

    fig, ax = plt.subplots(figsize=(9.8, 8.2))
    ax.set_title("SPIM v3 Greedy Step0 Posterior Heatmap", loc="left", fontsize=14, pad=8)
    ax.add_collection(LineCollection(seg, colors="#BFC8D1", linewidths=0.45, alpha=0.35, zorder=1))

    vmin = float(values[candidate_mask.numpy()].min()) if bool(candidate_mask.any()) else 0.0
    vmax = float(values[candidate_mask.numpy()].max()) if bool(candidate_mask.any()) else 1.0
    if abs(vmax - vmin) < 1e-12:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="magma", s=26, alpha=0.95, edgecolors="none", vmin=0.0, vmax=max(vmax * 1.2, 1e-3), zorder=2)
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="magma", s=26, alpha=0.95, edgecolors="none", zorder=2)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("posterior mass (step0)")
    if trigger_global is not None:
        txy = topology.node_coords[int(trigger_global)]
        ax.scatter([txy[0]], [txy[1]], s=210, marker="D", color="#F4A62A", edgecolors="white", linewidths=0.9, zorder=5, label="trigger")
    if source_global is not None:
        sxy = topology.node_coords[int(source_global)]
        ax.scatter([sxy[0]], [sxy[1]], s=250, marker="*", color="#D7263D", edgecolors="white", linewidths=0.9, zorder=6, label="true source")
    if selected_globals:
        sel_xy = topology.node_coords[np.array(selected_globals, dtype=np.int64)]
        ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=96, marker="o", facecolors="none", edgecolors="#1F9D8A", linewidths=1.5, zorder=7, label="greedy top-k")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.015,
        0.985,
        (
            f"case={args.case_tag} | scenario={case_idx} | step=0 | candidate={int(candidate_mask.sum().item())}\n"
            f"mass range on candidate: [{vmin:.6f}, {vmax:.6f}]"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#263238",
        bbox={"facecolor": "white", "edgecolor": "#D0D7DE", "boxstyle": "round,pad=0.24", "alpha": 0.92},
    )
    fig.tight_layout()

    png_path = output_dir / f"{args.case_tag}_spim_v3_greedy_step0_heatmap_v2.png"
    pdf_path = output_dir / f"{args.case_tag}_spim_v3_greedy_step0_heatmap_v2.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case_tag": args.case_tag,
        "scenario_index": int(case_idx),
        "case_id": case.case_id,
        "family": "hsr_soft_scenario_posterior_v3",
        "policy": "posterior_greedy",
        "step": 0,
        "trigger_global": None if trigger_global is None else int(trigger_global),
        "source_global": source_global,
        "candidate_count": int(candidate_mask.sum().item()),
        "topk_local": [int(v) for v in selected_locals],
        "topk_global": [int(v) for v in selected_globals],
        "topk_probs": [float(probs[int(v)].item()) for v in selected_locals],
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }
    (output_dir / f"{args.case_tag}_spim_v3_greedy_step0_summary_v2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
