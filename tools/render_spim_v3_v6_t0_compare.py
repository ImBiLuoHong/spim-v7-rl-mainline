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
from matplotlib.colors import LinearSegmentedColormap, Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.data.v6.topology import HydraulicTopology
from src.modeling.evidence.two_channel_clean import CleanTwoChannelEvidenceEnv, ObservationWitnessHistory
from src.scripts.audit.utils_practical_rollout import PracticalRollout
from src.scripts.diagnostics.run_clean_navigator_v1 import resolve_source_local_idx
from src.scripts.run_posterior_like_belief_audit import load_runtime_context
from src.scripts.run_reasoner_same_case_stronger_source_overfit import CaseRecord, make_rollout_state, read_json
from src.scripts.run_spim_family_sweep import _extract_trigger_global, _pick_topk_unsampled, PaperLikeHSRState
from src.scripts.run_spim_teacher_imitation_rl_pilot import DEFAULT_CACHE_DIR, DEFAULT_SOURCE_ROOT, compute_teacher_belief


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SPIM v3 vs v6 posterior distribution at t0.")
    p.add_argument("--case-tag", type=str, default="event_20260115_224139_1865_mode_LOW")
    p.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output" / "thesis_figures_v11"))
    p.add_argument("--case-label", type=str, default="案例A", help="Display/output case name, e.g., 案例A/案例B.")
    p.add_argument("--actions-per-round", type=int, default=3)
    p.add_argument("--paper-like-alpha", type=float, default=0.55)
    p.add_argument("--paper-like-topk-fraction", type=float, default=0.12)
    p.add_argument("--paper-like-time-tol-min", type=float, default=30.0)
    p.add_argument("--soft-scenario-beta", type=float, default=2.0)
    return p.parse_args()


def _safe_label(label: str) -> str:
    buf = []
    for ch in str(label):
        if ch.isalnum() or ch in ("-", "_"):
            buf.append(ch)
    return "".join(buf) or "Case"


def _load_cfg_payload(source_root: Path) -> dict[str, Any]:
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
        if any(needle in str(p) for p in group):
            return int(idx)
    raise ValueError(f"case not found in train groups: {needle}")


def _entropy(prob: torch.Tensor, mask: torch.Tensor) -> float:
    vals = prob[mask].clamp_min(1e-12)
    if vals.numel() == 0:
        return 0.0
    return float((-(vals * torch.log(vals))).sum().item())


def _run_family_step0(
    family: str,
    case: CaseRecord,
    rollout: PracticalRollout,
    topology: HydraulicTopology,
    history: ObservationWitnessHistory,
    env: CleanTwoChannelEvidenceEnv,
    trigger_global: int | None,
    episode_duration_min: float,
    frontier_role_mode: str,
    actions_per_round: int,
    paper_like_alpha: float,
    paper_like_topk_fraction: float,
    paper_like_time_tol_min: float,
    soft_scenario_beta: float,
):
    paper_state = PaperLikeHSRState(source_prior=None)
    onset_grid = [-float(episode_duration_min), 0.0, float(episode_duration_min)]
    state = make_rollout_state(
        case=case,
        rollout=rollout,
        history=history,
        env=env,
        topology=topology,
        num_episodes=10,
        action_budget=int(actions_per_round),
        frontier_role_mode=str(frontier_role_mode),
    )
    belief_ctx = compute_teacher_belief(
        family=family,
        rollout=rollout,
        state=state,
        history=history,
        trigger_global=trigger_global,
        paper_state=paper_state,
        onset_offsets_min=onset_grid,
        paper_like_alpha=float(paper_like_alpha),
        paper_like_topk_fraction=float(paper_like_topk_fraction),
        paper_like_time_tol_min=float(paper_like_time_tol_min),
        soft_scenario_beta=float(soft_scenario_beta),
    )
    belief = belief_ctx["belief"].view(-1).float().cpu()
    candidate_mask = belief_ctx["candidate_mask"].view(-1).bool().cpu()
    probs = torch.zeros_like(belief)
    if bool(candidate_mask.any()):
        probs[candidate_mask] = belief[candidate_mask] / belief[candidate_mask].sum().clamp_min(1e-12)
    topk_local = _pick_topk_unsampled(probs, candidate_mask, rollout, int(actions_per_round))
    topk_global = [int(rollout.g_ids[int(i)].item()) for i in topk_local]
    return {
        "probs": probs,
        "mask": candidate_mask,
        "entropy": _entropy(probs, candidate_mask),
        "candidate_count": int(candidate_mask.sum().item()),
        "topk_local": [int(v) for v in topk_local],
        "topk_global": [int(v) for v in topk_global],
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_payload = _load_cfg_payload(source_root)
    dataset, cfg = _build_train_dataset(cfg_payload, cache_dir)
    case_idx = _find_case_index(dataset, args.case_tag)
    data = dataset[case_idx]

    runtime_base = load_runtime_context(source_root, cache_dir)
    topology = HydraulicTopology(cfg.paths.foundation_path)
    case = CaseRecord(
        case_id=f"train:scenario{case_idx}:part0",
        scenario_id=int(case_idx),
        part_id=0,
        dataset_index=int(case_idx),
        data=deepcopy(data),
    )
    trigger_global = _extract_trigger_global(case.data)
    episode_duration_min = float(runtime_base["episode_duration_min"])
    frontier_role_mode = str(runtime_base["frontier_role_mode"])

    rollout_v3 = PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=dataset.global_edge_index,
        stt_dynamic_series=dataset.stt_dynamic_series,
        num_global_nodes=int(dataset.global_pos.shape[0]),
        num_episodes=10,
        samples_per_episode=int(args.actions_per_round),
        episode_duration_min=episode_duration_min,
    )
    rollout_v6 = PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=dataset.global_edge_index,
        stt_dynamic_series=dataset.stt_dynamic_series,
        num_global_nodes=int(dataset.global_pos.shape[0]),
        num_episodes=10,
        samples_per_episode=int(args.actions_per_round),
        episode_duration_min=episode_duration_min,
    )
    env = CleanTwoChannelEvidenceEnv()
    hist_v3 = ObservationWitnessHistory()
    hist_v6 = ObservationWitnessHistory()

    v3 = _run_family_step0(
        "hsr_soft_scenario_posterior_v3",
        case,
        rollout_v3,
        topology,
        hist_v3,
        env,
        trigger_global,
        episode_duration_min,
        frontier_role_mode,
        args.actions_per_round,
        args.paper_like_alpha,
        args.paper_like_topk_fraction,
        args.paper_like_time_tol_min,
        args.soft_scenario_beta,
    )
    v6 = _run_family_step0(
        "hsr_soft_scenario_posterior_v6",
        case,
        rollout_v6,
        topology,
        hist_v6,
        env,
        trigger_global,
        episode_duration_min,
        frontier_role_mode,
        args.actions_per_round,
        args.paper_like_alpha,
        args.paper_like_topk_fraction,
        args.paper_like_time_tol_min,
        args.soft_scenario_beta,
    )

    gids = rollout_v3.g_ids.detach().cpu().numpy().astype(np.int64)
    coords = topology.node_coords[gids]
    edge_index = rollout_v3.sub_edge_index.detach().cpu().numpy()
    seg = np.stack([coords[edge_index[0]], coords[edge_index[1]]], axis=1)

    source_local = resolve_source_local_idx(rollout_v3)
    source_global = None if source_local is None else int(rollout_v3.g_ids[int(source_local)].item())

    vals_v3 = v3["probs"].numpy()
    vals_v6 = v6["probs"].numpy()
    mask_v3 = v3["mask"].numpy().astype(bool)
    mask_v6 = v6["mask"].numpy().astype(bool)
    union_mask = mask_v3 | mask_v6
    all_cands = np.concatenate([vals_v3[v3["mask"].numpy()], vals_v6[v6["mask"].numpy()]])
    if all_cands.size:
        # Stretch contrast: compress low tail and saturate top tail to pure red.
        vmin = float(np.percentile(all_cands, 12))
        vmax = float(np.percentile(all_cands, 99.2))
        if vmax <= vmin:
            vmin = float(np.min(all_cands))
            vmax = float(np.max(all_cands))
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    hot_red = LinearSegmentedColormap.from_list(
        "posterior_hot_red",
        ["#F3F5F9", "#FFD9A8", "#FF6B4A", "#FF0000"],
    )

    # Zoom to the candidate area to make topology legible.
    zoom_pts = coords[union_mask] if bool(union_mask.any()) else coords
    xmin, ymin = np.min(zoom_pts, axis=0)
    xmax, ymax = np.max(zoom_pts, axis=0)
    dx = max(float(xmax - xmin), 1e-6)
    dy = max(float(ymax - ymin), 1e-6)
    mx = dx * 0.20
    my = dy * 0.20
    xlim = (float(xmin - mx), float(xmax + mx))
    ylim = (float(ymin - my), float(ymax + my))

    # Family-specific candidate edges for stronger topology readability.
    cand_edge_v3 = mask_v3[edge_index[0]] & mask_v3[edge_index[1]]
    cand_edge_v6 = mask_v6[edge_index[0]] & mask_v6[edge_index[1]]
    seg_v3 = seg[cand_edge_v3]
    seg_v6 = seg[cand_edge_v6]

    fig, axes = plt.subplots(1, 2, figsize=(16.8, 7.3))
    for ax, title, vals, stat, color, c_mask, c_seg in [
        (axes[0], "SPIM v3 @ t0", vals_v3, v3, "#1F9D8A", mask_v3, seg_v3),
        (axes[1], "SPIM v6 @ t0", vals_v6, v6, "#7C3AED", mask_v6, seg_v6),
    ]:
        ax.set_facecolor("#F8F7F3")
        # Layer 1: full local topology base.
        ax.add_collection(LineCollection(seg, colors="#D0D6DC", linewidths=0.85, alpha=0.62, zorder=1))
        # Layer 2: candidate-induced topology for this family.
        if len(c_seg) > 0:
            ax.add_collection(LineCollection(c_seg, colors=color, linewidths=1.5, alpha=0.70, zorder=2))

        # Background nodes keep topology context but do not compete with candidates.
        bg_mask = ~c_mask
        if bool(bg_mask.any()):
            ax.scatter(
                coords[bg_mask, 0],
                coords[bg_mask, 1],
                s=16,
                color="#C3CBD3",
                alpha=0.55,
                edgecolors="none",
                zorder=2,
            )

        # Candidate nodes: color + size double-encoding.
        cand_vals = vals[c_mask]
        if cand_vals.size > 0:
            cmin = float(np.min(cand_vals))
            cmax = float(np.max(cand_vals))
            if abs(cmax - cmin) < 1e-12:
                cand_size = np.full(cand_vals.shape[0], 54.0, dtype=np.float32)
            else:
                cand_size = 42.0 + 110.0 * ((cand_vals - cmin) / (cmax - cmin))
            sc = ax.scatter(
                coords[c_mask, 0],
                coords[c_mask, 1],
                c=cand_vals,
                cmap=hot_red,
                norm=norm,
                s=cand_size,
                alpha=0.96,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        else:
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=hot_red, norm=norm, s=26, alpha=0.93, edgecolors="none", zorder=3)

        if trigger_global is not None:
            txy = topology.node_coords[int(trigger_global)]
            ax.scatter([txy[0]], [txy[1]], s=190, marker="D", color="#F4A62A", edgecolors="white", linewidths=0.9, zorder=6)
        if source_global is not None:
            sxy = topology.node_coords[int(source_global)]
            ax.scatter([sxy[0]], [sxy[1]], s=240, marker="*", color="#A62E2E", edgecolors="white", linewidths=0.9, zorder=7)
        if stat["topk_global"]:
            sel_xy = topology.node_coords[np.array(stat["topk_global"], dtype=np.int64)]
            ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=140, marker="o", facecolors="#F8D36B", edgecolors=color, linewidths=1.8, zorder=8)
            for i, (x, y) in enumerate(sel_xy):
                ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=8, color="#3D2A00", zorder=9)
        ax.set_title(title, loc="left", fontsize=13.2)
        ax.text(
            0.01,
            0.985,
            (
                f"candidate={stat['candidate_count']}  H={stat['entropy']:.3f}\n"
                f"topk={stat['topk_global'][:3]}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.1,
            color="#263238",
            bbox={"facecolor": "white", "edgecolor": "#D0D7DE", "boxstyle": "round,pad=0.20", "alpha": 0.92},
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    # Keep colorbar at far-right to avoid splitting the two panels visually.
    cax = fig.add_axes([0.925, 0.16, 0.016, 0.70])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("posterior mass (t0) | contrast-stretched")
    fig.suptitle(f"SPIM V3 vs V6 at t0 | {args.case_label}", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])

    label_slug = _safe_label(args.case_label)
    png = output_dir / f"{label_slug}_spim_v3_v6_t0_compare.png"
    pdf = output_dir / f"{label_slug}_spim_v3_v6_t0_compare.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case_tag": args.case_tag,
        "scenario_index": int(case_idx),
        "v3": {
            "candidate_count": int(v3["candidate_count"]),
            "entropy": float(v3["entropy"]),
            "topk_global": [int(x) for x in v3["topk_global"]],
        },
        "v6": {
            "candidate_count": int(v6["candidate_count"]),
            "entropy": float(v6["entropy"]),
            "topk_global": [int(x) for x in v6["topk_global"]],
        },
        "output_png": str(png),
        "output_pdf": str(pdf),
    }
    summary_path = output_dir / f"{label_slug}_spim_v3_v6_t0_compare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
