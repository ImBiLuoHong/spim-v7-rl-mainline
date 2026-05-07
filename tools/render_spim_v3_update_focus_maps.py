from __future__ import annotations

import argparse
import importlib.util
import math
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
    p = argparse.ArgumentParser(description="Render SPIM full-trace storyboard (space/posterior/decision).")
    p.add_argument("--case-tag", type=str, default="event_20260115_222959_7013_mode_HIGH")
    p.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output" / "thesis_figures_v11"))
    p.add_argument("--topk-bars", type=int, default=8)
    p.add_argument(
        "--family",
        type=str,
        default="hsr_soft_scenario_posterior_v6",
        choices=["hsr_soft_scenario_posterior_v3", "hsr_soft_scenario_posterior_v6"],
    )
    p.add_argument("--output-stem", type=str, default="spim_v6_full_trace_maps")
    p.add_argument("--case-label", type=str, default="案例A", help="Display/output case name, e.g., 案例A/案例B.")
    return p.parse_args()


def _safe_label(label: str) -> str:
    buf = []
    for ch in str(label):
        if ch.isalnum() or ch in ("-", "_"):
            buf.append(ch)
    return "".join(buf) or "Case"


def _panel4_base(case_tag: str):
    mod_path = PROJECT_ROOT / "tools" / "generate_thesis_figures_v11.py"
    spec = importlib.util.spec_from_file_location("v11fig", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v11fig"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    graph = mod.load_graph_bundle()
    case_views = mod.load_case_views(f"{case_tag}.npz")
    target = case_views["A"]
    stages = mod.reconstruct_candidates(graph, target.group_id, target)
    union_nodes = np.unique(np.concatenate([np.fromiter(s, dtype=np.int32) for s in stages.values() if s]))
    local_bbox = mod.padded_bbox(graph.coords[union_nodes], margin_ratio=0.14)
    local_edge_index = mod.filter_edges_in_bbox(graph.coords, graph.edge_index, local_bbox)
    local_segments = mod.edge_segments(graph.coords, local_edge_index)
    final_nodes = np.array(sorted(stages["final"]), dtype=np.int32)
    final_set = set(stages["final"])
    final_lookup = {n: i for i, n in enumerate(final_nodes)}
    final_coords = graph.coords[final_nodes]
    final_local_edges = mod.induced_edge_index(local_edge_index, final_set, final_lookup)
    final_segments = mod.edge_segments(final_coords, final_local_edges)
    return graph, target, local_bbox, local_edge_index, local_segments, final_coords, final_segments


def _build_dataset(source_root: Path, cache_dir: Path):
    source_summary = read_json(source_root / "summary.json")
    oracle_root = Path(source_summary["same_case_manifest"]["oracle_root"])
    oracle_manifest = read_json(oracle_root / "manifest.json")
    cfg_path = Path(oracle_manifest["config_path"])
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if isinstance(payload.get("life_support"), dict) and str(payload["life_support"].get("profile")) == "custom_direct_edit":
        payload = dict(payload)
        payload["life_support"] = {k: v for k, v in payload["life_support"].items() if k != "profile"}
    cfg = Config(root_dir=str(PROJECT_ROOT))
    cfg.apply_overrides(payload)
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
    for i, g in enumerate(dataset.groups):
        if any(needle in str(p) for p in g):
            return int(i)
    raise ValueError(f"case not found: {needle}")


def _entropy(prob: torch.Tensor, mask: torch.Tensor) -> float:
    vals = prob[mask].clamp_min(1e-12)
    if vals.numel() == 0:
        return 0.0
    return float((-(vals * torch.log(vals))).sum().item())


def _rank_and_mass(prob: torch.Tensor, mask: torch.Tensor, gids: np.ndarray, source_global: int | None) -> tuple[int | None, float]:
    if source_global is None:
        return None, 0.0
    if not bool(mask.any()):
        return None, 0.0
    source_idx = np.where(gids == int(source_global))[0]
    if source_idx.size == 0:
        return None, 0.0
    sidx = int(source_idx[0])
    sval = float(prob[sidx].item())
    cand_vals = prob[mask].numpy()
    rank = 1 + int(np.sum(cand_vals > (sval + 1e-12)))
    return rank, sval


def _fmt_nodes(xs: list[int], limit: int = 3) -> str:
    if not xs:
        return "{}"
    head = ", ".join([str(int(v)) for v in xs[:limit]])
    return "{" + head + (" ..." if len(xs) > limit else "") + "}"


SEGMENT_CMAP = LinearSegmentedColormap.from_list(
    "posterior_segments",
    [
        "#fff1cc",
        "#ffbf69",
        "#ff7f50",
        "#ff0000",
        "#6f0000",
    ],
)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_graph, panel_target, bbox, local_edge_index, local_segments, final_coords, final_segments = _panel4_base(args.case_tag)
    dataset, cfg = _build_dataset(source_root, cache_dir)
    case_idx = _find_case_index(dataset, args.case_tag)
    rt = load_runtime_context(source_root, cache_dir)
    case = CaseRecord(
        case_id=f"train:scenario{case_idx}:part0",
        scenario_id=int(case_idx),
        part_id=0,
        dataset_index=int(case_idx),
        data=deepcopy(dataset[case_idx]),
    )
    rollout = PracticalRollout(
        event_data=deepcopy(case.data),
        global_edge_index=dataset.global_edge_index,
        stt_dynamic_series=dataset.stt_dynamic_series,
        num_global_nodes=int(dataset.global_pos.shape[0]),
        num_episodes=10,
        samples_per_episode=3,
        episode_duration_min=float(rt["episode_duration_min"]),
    )
    topo = HydraulicTopology(cfg.paths.foundation_path)
    env = CleanTwoChannelEvidenceEnv()
    hist = ObservationWitnessHistory()
    paper = PaperLikeHSRState(source_prior=None)
    trigger_global = _extract_trigger_global(case.data)
    onset = [-float(rt["episode_duration_min"]), 0.0, float(rt["episode_duration_min"])]

    snapshots = []
    observations_by_round: list[list[dict[str, Any]]] = []
    sampled_so_far: list[int] = []
    for ridx in range(3):
        st = make_rollout_state(
            case=case,
            rollout=rollout,
            history=hist,
            env=env,
            topology=topo,
            num_episodes=10,
            action_budget=3,
            frontier_role_mode=str(rt["frontier_role_mode"]),
        )
        b = compute_teacher_belief(
            family=str(args.family),
            rollout=rollout,
            state=st,
            history=hist,
            trigger_global=trigger_global,
            paper_state=paper,
            onset_offsets_min=onset,
            paper_like_alpha=0.55,
            paper_like_topk_fraction=0.12,
            paper_like_time_tol_min=30.0,
            soft_scenario_beta=2.0,
        )
        p = b["belief"].view(-1).float().cpu()
        m = b["candidate_mask"].view(-1).bool().cpu()
        q = torch.zeros_like(p)
        if bool(m.any()):
            q[m] = p[m] / p[m].sum().clamp_min(1e-12)

        a = _pick_topk_unsampled(q, m, rollout, 3)
        ag = [int(rollout.g_ids[int(i)].item()) for i in a]
        snapshots.append(
            {
                "round_idx": ridx,
                "prob": q.clone(),
                "mask": m.clone(),
                "selected_global": list(ag),
                "sampled_before": list(sampled_so_far),
                "candidate_count": int(m.sum().item()),
                "entropy": _entropy(q, m),
            }
        )
        if ridx < 2 and a:
            rollout.step_with_actions(a, sample_types=["slot0", "slot1", "slot2"][: len(a)])
            this_obs = []
            if rollout.history_steps:
                hist.append_from_history_step(rollout.history_steps[-1])
                for smp in rollout.history_steps[-1].samples:
                    this_obs.append(
                        {
                            "global_idx": int(smp.global_idx),
                            "is_positive": bool(smp.is_positive),
                            "is_safe": bool(smp.is_safe),
                        }
                    )
            observations_by_round.append(this_obs)
            sampled_so_far.extend(ag)

    gids = rollout.g_ids.detach().cpu().numpy().astype(np.int64)
    g2idx = {int(g): i for i, g in enumerate(gids)}

    source_local = resolve_source_local_idx(rollout)
    source_global = None if source_local is None else int(rollout.g_ids[int(source_local)].item())
    for snap in snapshots:
        rank, mass = _rank_and_mass(snap["prob"], snap["mask"], gids, source_global)
        snap["source_rank"] = rank
        snap["source_mass"] = mass

    # Edge masses are derived from node posterior (max endpoint mass) and scaled like demo_single_probe_visualization.
    local_u = local_edge_index[0].astype(np.int64)
    local_v = local_edge_index[1].astype(np.int64)

    def edge_mass_from_prob(prob_t: torch.Tensor, mask_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        q = prob_t.numpy()
        p_u = np.array([q[g2idx[int(u)]] if int(u) in g2idx else 0.0 for u in local_u], dtype=np.float32)
        p_v = np.array([q[g2idx[int(v)]] if int(v) in g2idx else 0.0 for v in local_v], dtype=np.float32)
        cand = mask_t.numpy().astype(bool)
        c_u = np.array([cand[g2idx[int(u)]] if int(u) in g2idx else False for u in local_u], dtype=bool)
        c_v = np.array([cand[g2idx[int(v)]] if int(v) in g2idx else False for v in local_v], dtype=bool)
        return np.maximum(p_u, p_v), np.logical_and(c_u, c_v)

    edge_masses = []
    edge_candidate_masks = []
    for snap in snapshots:
        em, cm = edge_mass_from_prob(snap["prob"], snap["mask"])
        edge_masses.append(em)
        edge_candidate_masks.append(cm)

    positive_sets = []
    global_max = 0.0
    for em, cm in zip(edge_masses, edge_candidate_masks):
        pos = em[np.logical_and(em > 0, cm)]
        if pos.size:
            positive_sets.append(pos)
            global_max = max(global_max, float(np.max(pos)))
    if positive_sets and global_max > 0:
        all_pos = np.concatenate(positive_sets)
        scale_floor = float(np.percentile(all_pos, 12))
        scale_floor = max(scale_floor, global_max * 1e-4, 1e-12)
        scale_ceiling = float(np.percentile(all_pos, 99.7))
        scale_ceiling = max(scale_ceiling, global_max * 0.18, scale_floor * 1.05)
        scale_ceiling = min(scale_ceiling, global_max)
        scale_ceiling = max(scale_ceiling, scale_floor * 1.05)
    else:
        scale_floor, scale_ceiling = 1e-9, 1.0

    # Edge rendering is normalized to [0, 1] after log scaling.
    pnorm = Normalize(vmin=0.0, vmax=1.0)
    local_node_mask = (
        (panel_graph.coords[:, 0] >= bbox[0])
        & (panel_graph.coords[:, 0] <= bbox[1])
        & (panel_graph.coords[:, 1] >= bbox[2])
        & (panel_graph.coords[:, 1] <= bbox[3])
    )
    local_bg_coords = panel_graph.coords[local_node_mask]

    states = [
        {
            "key": "t0",
            "title": "A  t0 Initial Posterior",
            "snap": 0,
            "show_selected": [],
            "show_prev": [],
            "show_next": snapshots[0]["selected_global"],
            "show_obs_rounds": 0,
            "show_source": False,
            "caption": "Initial posterior at trace start",
        },
        {
            "key": "t1",
            "title": "B  t0 Action Set #1",
            "snap": 0,
            "show_selected": snapshots[0]["selected_global"],
            "show_prev": [],
            "show_next": [],
            "show_obs_rounds": 0,
            "show_source": False,
            "caption": "Execute first sampling points",
        },
        {
            "key": "t2",
            "title": "C  t1 Posterior Update",
            "snap": 1,
            "show_selected": [],
            "show_prev": snapshots[1]["sampled_before"],
            "show_next": snapshots[1]["selected_global"],
            "show_obs_rounds": 1,
            "show_source": False,
            "caption": "Update with round-1 observations + suggest next points",
        },
        {
            "key": "t3",
            "title": "D  t1 Action Set #2",
            "snap": 1,
            "show_selected": snapshots[1]["selected_global"],
            "show_prev": snapshots[1]["sampled_before"],
            "show_next": [],
            "show_obs_rounds": 1,
            "show_source": False,
            "caption": "Execute second sampling points + keep history",
        },
        {
            "key": "t4",
            "title": "E  t2 Posterior Update",
            "snap": 2,
            "show_selected": [],
            "show_prev": snapshots[2]["sampled_before"],
            "show_next": snapshots[2]["selected_global"],
            "show_obs_rounds": 2,
            "show_source": True,
            "caption": "Posterior update after round-2 + next suggestions",
        },
    ]

    def draw_base(ax, show_source: bool):
        ax.set_facecolor("#F8F7F3")
        ax.add_collection(LineCollection(local_segments, colors="#BCC5CE", linewidths=0.62, alpha=0.58, zorder=1))
        ax.scatter(local_bg_coords[:, 0], local_bg_coords[:, 1], s=4.0, color="#B4BCC5", edgecolors="none", alpha=0.18, zorder=1)
        if len(final_segments) > 0:
            ax.add_collection(LineCollection(final_segments, colors="#9FA8B2", linewidths=0.95, alpha=0.40, zorder=2))
        if trigger_global is not None:
            txy = panel_graph.coords[int(trigger_global)]
            ax.scatter([txy[0]], [txy[1]], s=130, marker="D", color="#F4A62A", edgecolors="white", linewidths=0.8, zorder=7)
        if source_global is not None:
            sxy = panel_graph.coords[int(source_global)]
            ax.scatter(
                [sxy[0]],
                [sxy[1]],
                s=210,
                marker="*",
                color="#A62E2E",
                edgecolors="white",
                linewidths=0.9,
                alpha=0.98 if show_source else 0.82,
                zorder=8,
            )
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        ax.set_xticks([])
        ax.set_yticks([])

    def add_callout(ax, xy: np.ndarray, text: str, xytext: tuple[float, float], edge: str, fc: str = "white") -> None:
        ax.annotate(
            text,
            xy=(float(xy[0]), float(xy[1])),
            xycoords="data",
            xytext=xytext,
            textcoords="axes fraction",
            ha="left" if xytext[0] < 0.8 else "right",
            va="center",
            fontsize=7.4,
            color="#24313B",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": fc, "edgecolor": edge, "alpha": 0.96},
            arrowprops={
                "arrowstyle": "-|>",
                "color": edge,
                "lw": 1.0,
                "shrinkA": 4,
                "shrinkB": 4,
                "connectionstyle": "arc3,rad=0.12",
            },
            annotation_clip=False,
            zorder=20,
        )

    fig = plt.figure(figsize=(27.5, 7.6), constrained_layout=True)
    outer = fig.add_gridspec(1, 1)
    top = outer[0].subgridspec(1, 5, wspace=0.08)
    top_axes = [fig.add_subplot(top[0, i]) for i in range(5)]

    last_sc = None
    for ci, state in enumerate(states):
        snap = snapshots[state["snap"]]
        e_mass = edge_masses[state["snap"]]
        e_mask = edge_candidate_masks[state["snap"]]
        e_norm = np.zeros_like(e_mass, dtype=np.float32)
        pos = e_mass > 0
        if np.any(pos):
            if scale_ceiling <= scale_floor:
                e_norm[pos] = np.clip(e_mass[pos] / max(scale_ceiling, 1e-12), 0.0, 1.0)
            else:
                e_norm[pos] = (
                    np.log1p(e_mass[pos] / scale_floor)
                    / max(np.log1p(scale_ceiling / scale_floor), 1e-12)
                )
            e_norm = np.clip(e_norm, 0.0, 1.0)
        e_plot = np.where(e_mask, e_norm, 0.0)

        if e_mass.size > 0:
            e_width = 0.45 + 4.10 * e_plot
        else:
            e_width = np.zeros((0,), dtype=np.float32)

        ax = top_axes[ci]
        draw_base(ax, bool(state["show_source"]))
        # Primary posterior rendering is on pipe segments (edges), not nodes.
        lc = LineCollection(local_segments, cmap=SEGMENT_CMAP, norm=pnorm, linewidths=e_width, alpha=0.98, zorder=4)
        lc.set_array(e_plot)
        ax.add_collection(lc)
        last_sc = lc
        cax = ax.inset_axes([1.01, 0.10, 0.028, 0.76])
        cbar = fig.colorbar(lc, cax=cax)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=6.6, length=2)

        if state["show_prev"]:
            prev_xy = panel_graph.coords[np.array(state["show_prev"], dtype=np.int64)]
            ax.scatter(
                prev_xy[:, 0],
                prev_xy[:, 1],
                s=84,
                marker="o",
                facecolors="none",
                edgecolors="#E8C977",
                linewidths=1.2,
                zorder=9,
            )
        if state["show_selected"]:
            cur_xy = panel_graph.coords[np.array(state["show_selected"], dtype=np.int64)]
            ax.scatter(
                cur_xy[:, 0],
                cur_xy[:, 1],
                s=125,
                marker="D",
                facecolors="#D8A33D",
                edgecolors="white",
                linewidths=0.8,
                zorder=10,
            )
            for si, xy in enumerate(cur_xy):
                add_callout(ax, xy, f"Probe {si + 1}", (0.94, 0.86 - 0.10 * si), "#B7791F", fc="#FFF7E0")
        if state["show_next"]:
            nxt_xy = panel_graph.coords[np.array(state["show_next"], dtype=np.int64)]
            ax.scatter(
                nxt_xy[:, 0],
                nxt_xy[:, 1],
                s=122,
                marker="^",
                facecolors="#A8E6CF",
                edgecolors="#2D6A4F",
                linewidths=1.1,
                zorder=10,
            )
            for ni, xy in enumerate(nxt_xy):
                add_callout(ax, xy, f"Next {ni + 1}", (0.92, 0.84 - 0.10 * ni), "#2D6A4F", fc="#EAF7F0")

        pos_nodes = []
        neg_nodes = []
        for ridx in range(state["show_obs_rounds"]):
            if ridx >= len(observations_by_round):
                continue
            for obs in observations_by_round[ridx]:
                gid = int(obs["global_idx"])
                if bool(obs["is_positive"]):
                    pos_nodes.append(gid)
                elif bool(obs["is_safe"]):
                    neg_nodes.append(gid)
        if pos_nodes:
            pxy = panel_graph.coords[np.array(sorted(set(pos_nodes)), dtype=np.int64)]
            ax.scatter(
                pxy[:, 0],
                pxy[:, 1],
                s=98,
                marker="o",
                facecolors="none",
                edgecolors="#D95F5F",
                linewidths=1.4,
                zorder=11,
            )
        if neg_nodes:
            nxy = panel_graph.coords[np.array(sorted(set(neg_nodes)), dtype=np.int64)]
            ax.scatter(
                nxy[:, 0],
                nxy[:, 1],
                s=98,
                marker="o",
                facecolors="none",
                edgecolors="#2B6CB0",
                linewidths=1.4,
                zorder=11,
            )
        if state["show_obs_rounds"] > 0:
            obs_now = observations_by_round[state["show_obs_rounds"] - 1] if state["show_obs_rounds"] - 1 < len(observations_by_round) else []
            cur_pos = [int(o["global_idx"]) for o in obs_now if bool(o["is_positive"])]
            cur_neg = [int(o["global_idx"]) for o in obs_now if bool(o["is_safe"])]
            if cur_pos:
                cxy = panel_graph.coords[np.array(cur_pos, dtype=np.int64)]
                ax.scatter(cxy[:, 0], cxy[:, 1], s=118, marker="o", facecolors="none", edgecolors="#D95F5F", linewidths=2.2, zorder=12)
            if cur_neg:
                cxy = panel_graph.coords[np.array(cur_neg, dtype=np.int64)]
                ax.scatter(cxy[:, 0], cxy[:, 1], s=118, marker="o", facecolors="none", edgecolors="#2B6CB0", linewidths=2.2, zorder=12)

        if trigger_global is not None:
            add_callout(ax, panel_graph.coords[int(trigger_global)], "Trigger", (0.08, 0.14), "#C97A13", fc="#FFF4DD")
        if source_global is not None:
            add_callout(ax, panel_graph.coords[int(source_global)], "Source", (0.08, 0.88), "#9B2C2C", fc="#FDECEC")
        if state["show_prev"]:
            prev_arr = np.array(state["show_prev"], dtype=np.int64)
            prev_centroid = panel_graph.coords[prev_arr].mean(axis=0)
            add_callout(ax, prev_centroid, f"History ({len(prev_arr)})", (0.90, 0.16), "#9A7B12", fc="#FCF8E3")

        rank_str = "-" if snap["source_rank"] is None else str(int(snap["source_rank"]))
        metric_line = (
            f"|C|={snap['candidate_count']}   "
            f"H={snap['entropy']:.3f}   "
            f"rank*={rank_str}   "
            f"mass*={snap['source_mass']:.4f}"
        )
        ax.text(
            0.01,
            0.985,
            metric_line,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            color="#263238",
        )
        ax.set_title(state["title"], loc="left", fontsize=12.2)
        ax.text(0.01, -0.08, state["caption"], transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color="#34495E")

    fig.suptitle(
        f"Case Study: {args.case_label} | Full Trace ({args.family})",
        x=0.01,
        ha="left",
        fontsize=15.0,
    )
    label_slug = _safe_label(args.case_label)
    out_png = out_dir / f"{label_slug}_{args.output_stem}.png"
    out_pdf = out_dir / f"{label_slug}_{args.output_stem}.pdf"
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
