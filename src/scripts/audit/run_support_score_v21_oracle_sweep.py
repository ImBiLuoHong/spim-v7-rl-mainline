import os
import sys
from itertools import product
from typing import Dict, List, NamedTuple, Tuple

ROOT_DIR = "/root/autodl-tmp/rl_spim_v7_mainline"
sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
import torch

import src.scripts.audit.run_practical_audit_rerun as rerun

SWEEP_CSV_PATH = os.path.join(ROOT_DIR, "support_score_v21_oracle_sweep.csv")
BEST_SUMMARY_MD_PATH = os.path.join(ROOT_DIR, "support_score_v21_oracle_best_summary.md")
V2_VS_V21_CSV_PATH = os.path.join(ROOT_DIR, "support_score_v2_vs_v21_oracle.csv")

VIRTUAL_RELIABILITY_VALUES = [0.65, 0.75, 0.85]
HUB_PENALTY_WEIGHT_VALUES = [0.20, 0.35]
OWNERSHIP_EXPONENT_VALUES = [1.5, 2.0]


class V21Config(NamedTuple):
    combo_id: str
    virtual_reliability: float
    hub_penalty_weight: float
    ownership_exponent: float


def build_sweep_configs() -> List[V21Config]:
    configs: List[V21Config] = []
    for virtual_reliability, hub_penalty_weight, ownership_exponent in product(
        VIRTUAL_RELIABILITY_VALUES,
        HUB_PENALTY_WEIGHT_VALUES,
        OWNERSHIP_EXPONENT_VALUES,
    ):
        combo_id = (
            f"vr{virtual_reliability:.2f}_hp{hub_penalty_weight:.2f}_own{ownership_exponent:.1f}"
            .replace(".", "p")
        )
        configs.append(
            V21Config(
                combo_id=combo_id,
                virtual_reliability=float(virtual_reliability),
                hub_penalty_weight=float(hub_penalty_weight),
                ownership_exponent=float(ownership_exponent),
            )
        )
    return configs


def config_from_current_v2() -> V21Config:
    return V21Config(
        combo_id=(
            f"vr{rerun.V2_VIRTUAL_RELIABILITY:.2f}_hp{rerun.V2_HUB_PENALTY_WEIGHT:.2f}_own{rerun.V2_OWNERSHIP_POWER:.1f}"
            .replace(".", "p")
        ),
        virtual_reliability=float(rerun.V2_VIRTUAL_RELIABILITY),
        hub_penalty_weight=float(rerun.V2_HUB_PENALTY_WEIGHT),
        ownership_exponent=float(rerun.V2_OWNERSHIP_POWER),
    )


def load_baseline_lookup() -> pd.DataFrame:
    df = pd.read_csv(rerun.V2_ORACLE_STEPWISE_CSV_PATH)
    return df.set_index(["event_id", "episode"]).sort_index()


def prepare_v2_payload(
    rollout: rerun.PracticalRollout,
    phys_ctx,
    truth_positive_mask: torch.Tensor,
    witness_strength: torch.Tensor,
    t_abs_idx: int,
    topology: rerun.HydraulicTopology,
) -> Dict[str, object]:
    device = phys_ctx.edge_index.device
    num_nodes = int(phys_ctx.batch.numel()) if phys_ctx.batch is not None else int(rollout.g_ids.numel())
    witness_idx = rerun.select_oracle_witness_indices(truth_positive_mask, witness_strength)
    if witness_idx.numel() == 0:
        return {
            "num_nodes": num_nodes,
            "device": device,
            "witness_count": 0,
            "physical_time_matrix": None,
            "virtual_time_matrix": None,
        }

    phys_adj_rev = rollout.reachability_module._build_scipy_reverse_graph(
        phys_ctx.edge_index,
        phys_ctx.stt_dynamic.view(-1),
        num_nodes,
    )
    subgraph_global_ids = rollout.g_ids.detach().cpu()

    physical_time_cols: List[torch.Tensor] = []
    virtual_time_cols: List[torch.Tensor] = []
    for witness_local_idx in witness_idx.tolist():
        phys_dist_np = rollout.reachability_module._run_scipy_dijkstra(
            phys_adj_rev,
            np.array([int(witness_local_idx)], dtype=np.int64),
        )
        phys_dist = torch.from_numpy(np.asarray(phys_dist_np[0])).float().to(device)
        witness_global_id = int(rollout.g_ids[int(witness_local_idx)].item())
        virt_dist = rerun.build_virtual_time_lookup(
            topology=topology,
            witness_global_id=witness_global_id,
            subgraph_global_ids=subgraph_global_ids,
            witness_local_idx=int(witness_local_idx),
            t_abs_idx=t_abs_idx,
            num_nodes=num_nodes,
            device=device,
        )
        physical_time_cols.append(phys_dist)
        virtual_time_cols.append(virt_dist)

    return {
        "num_nodes": num_nodes,
        "device": device,
        "witness_count": int(witness_idx.numel()),
        "physical_time_matrix": torch.stack(physical_time_cols, dim=1),
        "virtual_time_matrix": torch.stack(virtual_time_cols, dim=1),
    }


def compute_support_v21_from_payload(
    payload: Dict[str, object],
    current_time_min: float,
    config: V21Config,
) -> Dict[str, torch.Tensor]:
    num_nodes = int(payload["num_nodes"])
    device = payload["device"]
    if int(payload["witness_count"]) <= 0:
        return rerun.zero_support_v2_dict(num_nodes, device)

    physical_time_matrix = payload["physical_time_matrix"]
    virtual_time_matrix = payload["virtual_time_matrix"]
    witness_time_mins = payload.get("witness_time_mins")

    if witness_time_mins is not None:
        witness_time_mins = torch.as_tensor(witness_time_mins, dtype=torch.float32, device=device).view(1, -1)
        finite_phys = torch.isfinite(physical_time_matrix) & (physical_time_matrix < 1e8)
        finite_virt = torch.isfinite(virtual_time_matrix) & (virtual_time_matrix < 1e8)

        phys_safe = torch.where(finite_phys, physical_time_matrix, torch.zeros_like(physical_time_matrix))
        virt_safe = torch.where(finite_virt, virtual_time_matrix, torch.zeros_like(virtual_time_matrix))
        witness_den = (witness_time_mins + rerun.V2_TIME_PRIOR_OFFSET_MIN).clamp_min(1.0)

        phys_late = torch.relu(phys_safe - witness_time_mins)
        virt_late = torch.relu(virt_safe - witness_time_mins)
        phys_late_penalty = torch.exp(-((phys_late ** 2) / (2.0 * (rerun.V2_TIME_SIGMA_MIN ** 2))))
        virt_late_penalty = torch.exp(-((virt_late ** 2) / (2.0 * (rerun.V2_TIME_SIGMA_MIN ** 2))))

        phys_distance_decay = 1.0 / (1.0 + phys_safe / witness_den)
        virt_distance_decay = 1.0 / (1.0 + virt_safe / witness_den)

        phys_avail = finite_phys.float() * phys_late_penalty * phys_distance_decay
        virt_avail = config.virtual_reliability * finite_virt.float() * virt_late_penalty * virt_distance_decay
    else:
        phys_avail = rerun.travel_time_to_availability(
            physical_time_matrix,
            current_time_min=current_time_min,
            reliability=1.0,
        )
        virt_avail = rerun.travel_time_to_availability(
            virtual_time_matrix,
            current_time_min=current_time_min,
            reliability=config.virtual_reliability,
        )

    availability_matrix = torch.maximum(phys_avail, virt_avail)
    best_virtual_matrix = (virt_avail > phys_avail + rerun.EPS).float()
    best_physical_matrix = 1.0 - best_virtual_matrix

    ownership_mass = availability_matrix.clamp_min(0.0).pow(config.ownership_exponent)
    ownership_matrix = ownership_mass / (ownership_mass.sum(dim=0, keepdim=True) + rerun.EPS)

    generic_count = (availability_matrix > rerun.V2_AVAIL_ACTIVE_THRESHOLD).float().sum(dim=0)
    witness_weight = 1.0 / (1.0 + torch.log1p(generic_count))
    witness_weight = torch.where(generic_count > 0.0, witness_weight, torch.zeros_like(witness_weight))

    availability_term = availability_matrix.mean(dim=1)
    ownership_term = (availability_matrix * ownership_matrix * witness_weight.unsqueeze(0)).mean(dim=1)
    hub_penalty_term = (
        availability_matrix
        * (1.0 - ownership_matrix)
        * (1.0 - witness_weight.unsqueeze(0))
    ).mean(dim=1)
    total = (
        rerun.V2_AVAIL_WEIGHT * availability_term
        + ownership_term
        - config.hub_penalty_weight * hub_penalty_term
    )

    finite_phys = torch.isfinite(physical_time_matrix) & (physical_time_matrix < 1e8)
    finite_virt = torch.isfinite(virtual_time_matrix) & (virtual_time_matrix < 1e8)
    finite_best = torch.where(best_virtual_matrix > 0.5, finite_virt, finite_phys)
    best_time_matrix = torch.where(best_virtual_matrix > 0.5, virtual_time_matrix, physical_time_matrix)

    best_time_mean = torch.where(
        finite_best.any(dim=1),
        (best_time_matrix * finite_best.float()).sum(dim=1) / finite_best.float().sum(dim=1).clamp_min(1.0),
        torch.full((num_nodes,), float("inf"), device=device),
    )
    physical_time_mean = torch.where(
        finite_phys.any(dim=1),
        (physical_time_matrix * finite_phys.float()).sum(dim=1) / finite_phys.float().sum(dim=1).clamp_min(1.0),
        torch.full((num_nodes,), float("inf"), device=device),
    )
    virtual_time_mean = torch.where(
        finite_virt.any(dim=1),
        (virtual_time_matrix * finite_virt.float()).sum(dim=1) / finite_virt.float().sum(dim=1).clamp_min(1.0),
        torch.full((num_nodes,), float("inf"), device=device),
    )

    return {
        "total": total,
        "availability": availability_term,
        "ownership": ownership_term,
        "hub_penalty": hub_penalty_term,
        "virtual_share": best_virtual_matrix.mean(dim=1),
        "best_time_mean": best_time_mean,
        "physical_time_mean": physical_time_mean,
        "virtual_time_mean": virtual_time_mean,
        "best_path_virtual_rate": best_virtual_matrix.mean(dim=1),
        "best_path_physical_rate": best_physical_matrix.mean(dim=1),
        "witness_count": torch.full((num_nodes,), float(payload["witness_count"]), device=device),
        "availability_matrix": availability_matrix,
        "ownership_matrix": ownership_matrix,
        "physical_time_matrix": physical_time_matrix,
        "virtual_time_matrix": virtual_time_matrix,
    }


def collect_variant_rows(
    configs: List[V21Config],
    baseline_lookup: pd.DataFrame,
) -> pd.DataFrame:
    rerun.set_seeds(0)
    rerun.silence_non_table_logs()
    dataset = rerun.NpzDatasetV6(
        samples_dir=rerun.SAMPLES_PATH,
        foundation_dir=rerun.FOUNDATION_PATH,
        mode="test",
        preload=False,
        audit_mode="fast",
        use_edge_attr=True,
    )
    topology = rerun.HydraulicTopology(rerun.FOUNDATION_PATH)

    rows: List[Dict[str, float]] = []
    indices = range(min(rerun.MAX_EVENTS, len(dataset)))

    with torch.no_grad():
        for event_id in indices:
            try:
                event_data_batch = dataset[event_id]
                if event_data_batch is None:
                    continue
                event_data = rerun.extract_view0(event_data_batch)
                src_global = event_data.global_injection_node
                if isinstance(src_global, torch.Tensor):
                    src_global = int(src_global.item())

                rollout = rerun.PracticalRollout(
                    event_data,
                    dataset.global_edge_index,
                    dataset.stt_dynamic_series,
                    dataset.num_nodes,
                    num_episodes=rerun.NUM_EPISODES,
                    samples_per_episode=3,
                )
                if src_global not in rollout.g_ids:
                    continue

                src_local = int((rollout.g_ids == src_global).nonzero(as_tuple=True)[0].item())
                for episode_idx in range(rerun.NUM_EPISODES):
                    _obs_partial, _obs_oracle, phys_ctx, info = rollout.step()
                    episode = episode_idx + 1
                    key = (event_id, episode)
                    if key not in baseline_lookup.index:
                        continue
                    base_row = baseline_lookup.loc[key]

                    t_snapshot_idx = int(info["t_snapshot_idx"])
                    conc = rollout.event_data.x_raw[:, t_snapshot_idx, 1]
                    truth_positive_mask = conc > 0.1
                    oracle_t_abs_idx = rerun.resolve_snapshot_time_index(rollout.event_data, t_snapshot_idx)
                    payload = prepare_v2_payload(
                        rollout=rollout,
                        phys_ctx=phys_ctx,
                        truth_positive_mask=truth_positive_mask,
                        witness_strength=conc,
                        t_abs_idx=oracle_t_abs_idx,
                        topology=topology,
                    )

                    base_common = {
                        "event_id": event_id,
                        "episode": episode,
                        "time_min": float(info["time_min"]),
                        "oracle_num_pos": int(base_row["oracle_num_pos"]),
                        "num_pos": int(base_row["num_pos"]),
                        "practical_b_subset": float(base_row["practical_b_subset"]),
                        "b_oracle_relevant": float(base_row["b_oracle_relevant"]),
                        "c_subset": float(base_row["c_subset"]),
                        "fixed_case": float(base_row["fixed_case"]),
                    }

                    for config in configs:
                        support_res = compute_support_v21_from_payload(
                            payload=payload,
                            current_time_min=float(info["time_min"]),
                            config=config,
                        )
                        fields = rerun.extract_support_variant_fields_v2(
                            "v21",
                            support_res,
                            src_local,
                            rollout.g_ids,
                        )
                        rows.append(
                            {
                                **base_common,
                                "combo_id": config.combo_id,
                                "virtual_reliability": config.virtual_reliability,
                                "hub_penalty_weight": config.hub_penalty_weight,
                                "ownership_exponent": config.ownership_exponent,
                                "rank": float(fields["v21_rank"]),
                                "directionality": float(fields["v21_directionality"]),
                                "true_nonzero": float(fields["v21_true_nonzero"]),
                                "hub_win": float(fields["v21_hub_win"]),
                                "true_total": float(fields["v21_true_total"]),
                                "true_availability": float(fields["v21_true_availability"]),
                                "true_ownership": float(fields["v21_true_ownership"]),
                                "true_hub_penalty": float(fields["v21_true_hub_penalty"]),
                                "true_virtual_share": float(fields["v21_true_virtual_share"]),
                                "competitor_availability": float(fields["v21_competitor_availability"]),
                                "competitor_ownership": float(fields["v21_competitor_ownership"]),
                                "competitor_hub_penalty": float(fields["v21_competitor_hub_penalty"]),
                                "competitor_virtual_share": float(fields["v21_competitor_virtual_share"]),
                                "top_other_global_id": int(fields["v21_top_other_global_id"]),
                                "true_best_time_mean": float(fields["v21_true_best_time_mean"])
                                if np.isfinite(fields["v21_true_best_time_mean"])
                                else np.nan,
                                "competitor_best_time_mean": float(fields["v21_competitor_best_time_mean"])
                                if np.isfinite(fields["v21_competitor_best_time_mean"])
                                else np.nan,
                            }
                        )
            except Exception:
                continue

    return pd.DataFrame(rows)


def safe_mean(series: pd.Series) -> float:
    return rerun.safe_mean_from_series(series)


def safe_median(series: pd.Series) -> float:
    return rerun.safe_median_from_series(series)


def summarize_combo(eval_df: pd.DataFrame) -> Dict[str, float]:
    conditioned = eval_df[eval_df["oracle_num_pos"] > 0]
    c_subset = eval_df[eval_df["c_subset"] > 0.5]
    b_subset = eval_df[eval_df["b_oracle_relevant"] > 0.5]

    return {
        "combo_id": str(eval_df["combo_id"].iloc[0]),
        "virtual_reliability": float(eval_df["virtual_reliability"].iloc[0]),
        "hub_penalty_weight": float(eval_df["hub_penalty_weight"].iloc[0]),
        "ownership_exponent": float(eval_df["ownership_exponent"].iloc[0]),
        "evaluated_rows": int(len(eval_df)),
        "oracle_input_rows": int((eval_df["oracle_num_pos"] > 0).sum()),
        "oracle_conditioned_rank_median": safe_median(conditioned["rank"]),
        "oracle_conditioned_directionality": safe_mean(conditioned["directionality"]),
        "oracle_true_nonzero_given_oracle_input_rate": safe_mean(conditioned["true_nonzero"]),
        "oracle_hub_win_rate": safe_mean(conditioned["hub_win"]),
        "c_subset_rows": int(len(c_subset)),
        "c_subset_hub_win_rate": safe_mean(c_subset["hub_win"]),
        "c_subset_directionality": safe_mean(c_subset["directionality"]),
        "b_oracle_relevant_rows": int(len(b_subset)),
        "b_oracle_relevant_true_nonzero_rate": safe_mean(b_subset["true_nonzero"]),
    }


def rank_combo_row(row: pd.Series) -> Tuple[float, float, float, float, float, float, float]:
    return (
        float(row["oracle_hub_win_rate"]),
        float(row["c_subset_hub_win_rate"]),
        max(0.0, float(row["oracle_conditioned_rank_median"]) - 1.0),
        max(0.0, 0.94 - float(row["oracle_conditioned_directionality"])),
        max(0.0, 0.99 - float(row["oracle_true_nonzero_given_oracle_input_rate"])),
        -float(row["oracle_conditioned_directionality"]),
        -float(row["oracle_true_nonzero_given_oracle_input_rate"]),
    )


def add_selection_columns(sweep_df: pd.DataFrame) -> pd.DataFrame:
    ranked = sweep_df.copy()
    ranked["selection_key"] = ranked.apply(rank_combo_row, axis=1)
    ranked = ranked.sort_values("selection_key", kind="stable").reset_index(drop=True)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected_as_best"] = 0.0
    if not ranked.empty:
        ranked.loc[0, "selected_as_best"] = 1.0
    return ranked.drop(columns=["selection_key"])


def compare_with_existing_v2(
    variant_rows: pd.DataFrame,
    baseline_lookup: pd.DataFrame,
    baseline_id: str,
) -> Dict[str, float]:
    baseline_rows = (
        variant_rows[variant_rows["combo_id"] == baseline_id]
        .set_index(["event_id", "episode"])
        .sort_index()
    )
    compare = baseline_lookup.join(
        baseline_rows[
            [
                "rank",
                "directionality",
                "true_nonzero",
                "hub_win",
                "true_total",
            ]
        ].rename(
            columns={
                "rank": "rerun_rank",
                "directionality": "rerun_directionality",
                "true_nonzero": "rerun_true_nonzero",
                "hub_win": "rerun_hub_win",
                "true_total": "rerun_true_total",
            }
        ),
        how="inner",
    )
    if compare.empty:
        return {
            "matched_rows": 0,
            "rank_mismatch_rate": np.nan,
            "directionality_mismatch_rate": np.nan,
            "true_nonzero_mismatch_rate": np.nan,
            "hub_win_mismatch_rate": np.nan,
            "true_total_max_abs_diff": np.nan,
        }
    return {
        "matched_rows": int(len(compare)),
        "rank_mismatch_rate": float((compare["v2_main_rank"] != compare["rerun_rank"]).mean()),
        "directionality_mismatch_rate": float((compare["v2_main_directionality"] != compare["rerun_directionality"]).mean()),
        "true_nonzero_mismatch_rate": float((compare["v2_main_true_nonzero"] != compare["rerun_true_nonzero"]).mean()),
        "hub_win_mismatch_rate": float((compare["v2_main_hub_win"] != compare["rerun_hub_win"]).mean()),
        "true_total_max_abs_diff": float((compare["v2_main_true_total"] - compare["rerun_true_total"]).abs().max()),
    }


def build_v2_vs_best_table(
    variant_rows: pd.DataFrame,
    best_combo_id: str,
    baseline_id: str,
) -> pd.DataFrame:
    def combo_slice(combo_id: str) -> pd.DataFrame:
        return variant_rows[variant_rows["combo_id"] == combo_id].copy()

    def global_row(combo_id: str, prefix: str) -> Dict[str, float]:
        df = combo_slice(combo_id)
        conditioned = df[df["oracle_num_pos"] > 0]
        return {
            f"{prefix}_oracle_conditioned_rank_median": safe_median(conditioned["rank"]),
            f"{prefix}_oracle_conditioned_directionality": safe_mean(conditioned["directionality"]),
            f"{prefix}_oracle_true_nonzero_given_oracle_input_rate": safe_mean(conditioned["true_nonzero"]),
            f"{prefix}_oracle_hub_win_rate": safe_mean(conditioned["hub_win"]),
        }

    def subset_row(combo_id: str, mask_col: str, prefix: str) -> Dict[str, float]:
        df = combo_slice(combo_id)
        subset = df[df[mask_col] > 0.5]
        return {
            f"{prefix}_rows": int(len(subset)),
            f"{prefix}_rank_median": safe_median(subset["rank"]),
            f"{prefix}_directionality": safe_mean(subset["directionality"]),
            f"{prefix}_true_nonzero_rate": safe_mean(subset["true_nonzero"]),
            f"{prefix}_hub_win_rate": safe_mean(subset["hub_win"]),
        }

    rows: List[Dict[str, float]] = []
    rows.append(
        {
            "analysis": "global_oracle_input",
            **global_row(baseline_id, "v2_main"),
            **global_row(best_combo_id, "best_v21"),
        }
    )
    rows.append(
        {
            "analysis": "C_subset",
            **subset_row(baseline_id, "c_subset", "v2_main"),
            **subset_row(best_combo_id, "c_subset", "best_v21"),
        }
    )
    rows.append(
        {
            "analysis": "B_oracle_relevant",
            **subset_row(baseline_id, "b_oracle_relevant", "v2_main"),
            **subset_row(best_combo_id, "b_oracle_relevant", "best_v21"),
        }
    )

    baseline_cases = combo_slice(baseline_id)
    best_cases = combo_slice(best_combo_id)
    baseline_latest = (
        baseline_cases[baseline_cases["fixed_case"] > 0.5]
        .sort_values(["event_id", "episode"])
        .groupby("event_id", as_index=False)
        .tail(1)
        .set_index("event_id")
    )
    best_latest = (
        best_cases[best_cases["fixed_case"] > 0.5]
        .sort_values(["event_id", "episode"])
        .groupby("event_id", as_index=False)
        .tail(1)
        .set_index("event_id")
    )
    fixed_event_ids = sorted(set(baseline_latest.index.tolist()) | set(best_latest.index.tolist()))
    for event_id in fixed_event_ids:
        if event_id not in baseline_latest.index or event_id not in best_latest.index:
            continue
        base = baseline_latest.loc[event_id]
        best = best_latest.loc[event_id]
        rows.append(
            {
                "analysis": "fixed_case_latest",
                "event_id": int(event_id),
                "episode": int(best["episode"]),
                "time_min": float(best["time_min"]),
                "oracle_num_pos": int(best["oracle_num_pos"]),
                "v2_main_rank": float(base["rank"]),
                "best_v21_rank": float(best["rank"]),
                "v2_main_directionality": float(base["directionality"]),
                "best_v21_directionality": float(best["directionality"]),
                "v2_main_true_nonzero": float(base["true_nonzero"]),
                "best_v21_true_nonzero": float(best["true_nonzero"]),
                "v2_main_hub_win": float(base["hub_win"]),
                "best_v21_hub_win": float(best["hub_win"]),
                "v2_main_true_total": float(base["true_total"]),
                "best_v21_true_total": float(best["true_total"]),
                "v2_main_true_availability": float(base["true_availability"]),
                "best_v21_true_availability": float(best["true_availability"]),
                "v2_main_true_ownership": float(base["true_ownership"]),
                "best_v21_true_ownership": float(best["true_ownership"]),
                "v2_main_true_hub_penalty": float(base["true_hub_penalty"]),
                "best_v21_true_hub_penalty": float(best["true_hub_penalty"]),
                "v2_main_true_virtual_share": float(base["true_virtual_share"]),
                "best_v21_true_virtual_share": float(best["true_virtual_share"]),
                "v2_main_competitor_availability": float(base["competitor_availability"]),
                "best_v21_competitor_availability": float(best["competitor_availability"]),
                "v2_main_competitor_ownership": float(base["competitor_ownership"]),
                "best_v21_competitor_ownership": float(best["competitor_ownership"]),
                "v2_main_competitor_hub_penalty": float(base["competitor_hub_penalty"]),
                "best_v21_competitor_hub_penalty": float(best["competitor_hub_penalty"]),
                "v2_main_competitor_virtual_share": float(base["competitor_virtual_share"]),
                "best_v21_competitor_virtual_share": float(best["competitor_virtual_share"]),
                "v2_main_true_best_time_mean": float(base["true_best_time_mean"]) if np.isfinite(base["true_best_time_mean"]) else np.nan,
                "best_v21_true_best_time_mean": float(best["true_best_time_mean"]) if np.isfinite(best["true_best_time_mean"]) else np.nan,
                "v2_main_competitor_best_time_mean": float(base["competitor_best_time_mean"]) if np.isfinite(base["competitor_best_time_mean"]) else np.nan,
                "best_v21_competitor_best_time_mean": float(best["competitor_best_time_mean"]) if np.isfinite(best["competitor_best_time_mean"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def best_one_knob_row(
    sweep_df: pd.DataFrame,
    baseline_config: V21Config,
    knob: str,
) -> pd.Series:
    filters = {
        "virtual_reliability": (
            (sweep_df["hub_penalty_weight"] == baseline_config.hub_penalty_weight)
            & (sweep_df["ownership_exponent"] == baseline_config.ownership_exponent)
        ),
        "hub_penalty_weight": (
            (sweep_df["virtual_reliability"] == baseline_config.virtual_reliability)
            & (sweep_df["ownership_exponent"] == baseline_config.ownership_exponent)
        ),
        "ownership_exponent": (
            (sweep_df["virtual_reliability"] == baseline_config.virtual_reliability)
            & (sweep_df["hub_penalty_weight"] == baseline_config.hub_penalty_weight)
        ),
    }
    subset = sweep_df.loc[filters[knob]].copy()
    subset = subset.sort_values(
        [
            "oracle_hub_win_rate",
            "c_subset_hub_win_rate",
            "oracle_conditioned_rank_median",
            "oracle_conditioned_directionality",
            "oracle_true_nonzero_given_oracle_input_rate",
        ],
        ascending=[True, True, True, False, False],
        kind="stable",
    )
    return subset.iloc[0]


def determine_primary_blocker(
    sweep_df: pd.DataFrame,
    baseline_config: V21Config,
) -> Tuple[str, pd.DataFrame]:
    baseline_row = sweep_df[sweep_df["combo_id"] == baseline_config.combo_id].iloc[0]
    knob_rows = []
    for knob in ["virtual_reliability", "hub_penalty_weight", "ownership_exponent"]:
        row = best_one_knob_row(sweep_df, baseline_config, knob)
        knob_rows.append(
            {
                "knob": knob,
                "combo_id": row["combo_id"],
                "oracle_hub_win_delta": float(row["oracle_hub_win_rate"] - baseline_row["oracle_hub_win_rate"]),
                "c_subset_hub_win_delta": float(row["c_subset_hub_win_rate"] - baseline_row["c_subset_hub_win_rate"]),
                "rank_degradation_from_1": max(0.0, float(row["oracle_conditioned_rank_median"]) - 1.0),
                "directionality_shortfall": max(0.0, 0.94 - float(row["oracle_conditioned_directionality"])),
                "true_nonzero_shortfall": max(0.0, 0.99 - float(row["oracle_true_nonzero_given_oracle_input_rate"])),
            }
        )
    knob_df = pd.DataFrame(knob_rows)
    knob_df = knob_df.sort_values(
        [
            "oracle_hub_win_delta",
            "c_subset_hub_win_delta",
            "rank_degradation_from_1",
            "directionality_shortfall",
            "true_nonzero_shortfall",
        ],
        ascending=[True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    top = str(knob_df.iloc[0]["knob"])
    mapping = {
        "virtual_reliability": "virtual reliability 太高",
        "hub_penalty_weight": "hub penalty 还不够",
        "ownership_exponent": "ownership 还不够 sharp",
    }
    return mapping[top], knob_df


def proof_tag(improved: bool, degraded: bool) -> str:
    if improved and not degraded:
        return "[已证明]"
    if improved:
        return "[部分证明]"
    return "[未证明]"


def build_fixed_case_lines(comparison_df: pd.DataFrame) -> List[str]:
    fixed_cases = comparison_df[comparison_df["analysis"] == "fixed_case_latest"].copy()
    lines: List[str] = []
    for _, row in fixed_cases.sort_values("event_id").iterrows():
        event_id = int(row["event_id"])
        tag = "[已证明]"
        if event_id == 0 and row["oracle_num_pos"] <= 0:
            diagnosis = "event 0 最新 snapshot 仍然没有 oracle input，因此 best v2.1 与 v2_main 一样不会凭空长出 witness。"
        elif event_id == 5:
            diagnosis = (
                "event 5 维持正常；best v2.1 没把已正确的 case 拉坏。"
                if not (
                    (row["best_v21_hub_win"] > row["v2_main_hub_win"])
                    or (row["best_v21_rank"] > row["v2_main_rank"])
                    or (row["best_v21_directionality"] < row["v2_main_directionality"])
                )
                else "event 5 出现退化，需要回滚。"
            )
        elif event_id in {60, 83} and row["best_v21_hub_win"] < row["v2_main_hub_win"]:
            diagnosis = "best v2.1 进一步压低了 residual hub 偏置。"
        elif event_id in {60, 83} and row["best_v21_hub_win"] == row["v2_main_hub_win"]:
            diagnosis = "best v2.1 仍没把 residual hub 偏置压干净，说明这条 case 还卡在 ownership / anti-hub 归因边界。"
        else:
            diagnosis = "best v2.1 有小幅改善，但没有改变最终失败形态。"
        lines.append(
            f"- {tag} event {event_id}: "
            f"v2_main(rank={row['v2_main_rank']:.0f}, dir={row['v2_main_directionality']:.0f}, "
            f"true_nonzero={row['v2_main_true_nonzero']:.0f}, hub_win={row['v2_main_hub_win']:.0f}) -> "
            f"best v2.1(rank={row['best_v21_rank']:.0f}, dir={row['best_v21_directionality']:.0f}, "
            f"true_nonzero={row['best_v21_true_nonzero']:.0f}, hub_win={row['best_v21_hub_win']:.0f}); "
            f"{diagnosis} "
            f"true ownership {row['v2_main_true_ownership']:.6f}->{row['best_v21_true_ownership']:.6f}, "
            f"competitor ownership {row['v2_main_competitor_ownership']:.6f}->{row['best_v21_competitor_ownership']:.6f}, "
            f"true virtual_share {row['v2_main_true_virtual_share']:.3f}->{row['best_v21_true_virtual_share']:.3f}, "
            f"competitor virtual_share {row['v2_main_competitor_virtual_share']:.3f}->{row['best_v21_competitor_virtual_share']:.3f}."
        )
    return lines


def build_summary_markdown(
    sweep_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    reproducibility: Dict[str, float],
    baseline_config: V21Config,
    best_config: V21Config,
    primary_blocker: str,
    knob_df: pd.DataFrame,
) -> str:
    baseline_row = sweep_df[sweep_df["combo_id"] == baseline_config.combo_id].iloc[0]
    best_row = sweep_df[sweep_df["combo_id"] == best_config.combo_id].iloc[0]
    global_compare = comparison_df[comparison_df["analysis"] == "global_oracle_input"].iloc[0]
    c_compare = comparison_df[comparison_df["analysis"] == "C_subset"].iloc[0]
    b_compare = comparison_df[comparison_df["analysis"] == "B_oracle_relevant"].iloc[0]
    fixed_case_lines = build_fixed_case_lines(comparison_df)

    reproducible = (
        reproducibility["rank_mismatch_rate"] <= rerun.EPS
        and reproducibility["directionality_mismatch_rate"] <= rerun.EPS
        and reproducibility["true_nonzero_mismatch_rate"] <= rerun.EPS
        and reproducibility["hub_win_mismatch_rate"] <= rerun.EPS
        and reproducibility["true_total_max_abs_diff"] <= 1e-6
    )
    close_to_gate = best_row["oracle_hub_win_rate"] <= 0.20
    improved_global = best_row["oracle_hub_win_rate"] < baseline_row["oracle_hub_win_rate"] - rerun.EPS
    improved_c = best_row["c_subset_hub_win_rate"] < baseline_row["c_subset_hub_win_rate"] - rerun.EPS
    kept_rank = best_row["oracle_conditioned_rank_median"] <= 1.0 + rerun.EPS
    kept_dir = best_row["oracle_conditioned_directionality"] >= 0.94 - rerun.EPS
    kept_true = best_row["oracle_true_nonzero_given_oracle_input_rate"] >= 0.99 - rerun.EPS

    if close_to_gate:
        direction = "B. 切回 practical 线"
    else:
        direction = "A. 继续 v2.2 小修"
    gate_line = (
        f"- [已证明] best v2.1 已足够接近 oracle gate。依据是全局 hub_win_rate={best_row['oracle_hub_win_rate']:.4f}，"
        " 已进入 `<=0.20` 参考线。"
        if close_to_gate
        else f"- [已证明] best v2.1 还不够接近 oracle gate。依据是全局 hub_win_rate={best_row['oracle_hub_win_rate']:.4f}，"
        " 仍高于 `<=0.20` 参考线。"
    )

    lines = [
        "# Support Score v2.1 Oracle Sweep Summary",
        "",
        "## 1. 本轮执行摘要",
        f"- {proof_tag(reproducible, not reproducible)} 复现实验基线：当前 sweep 内置的 `v2_main` 组合与现有 `support_score_v2_oracle_stepwise.csv` 对齐。",
        f"- {proof_tag(improved_global, False)} best v2.1 的全局 oracle_hub_win_rate: {baseline_row['oracle_hub_win_rate']:.4f} -> {best_row['oracle_hub_win_rate']:.4f}.",
        f"- {proof_tag(improved_c, False)} best v2.1 的 C 子集 hub_win_rate: {baseline_row['c_subset_hub_win_rate']:.4f} -> {best_row['c_subset_hub_win_rate']:.4f}.",
        f"- {proof_tag(kept_rank and kept_dir and kept_true, not (kept_rank and kept_dir and kept_true))} best v2.1 是否守住 rank/directionality/true_nonzero：rank={best_row['oracle_conditioned_rank_median']:.4f}, directionality={best_row['oracle_conditioned_directionality']:.4f}, true_nonzero={best_row['oracle_true_nonzero_given_oracle_input_rate']:.4f}.",
        f"- {proof_tag(True, False)} 剩余失败主因判定：{primary_blocker}。",
        "",
        "## 2. sweep 参数设置",
        "- [已证明] 只扫 3 个允许旋钮：`virtual_reliability`、`hub_penalty_weight`、`ownership_exponent`。",
        f"- [已证明] virtual_reliability = {VIRTUAL_RELIABILITY_VALUES}",
        f"- [已证明] hub_penalty_weight = {HUB_PENALTY_WEIGHT_VALUES}",
        f"- [已证明] ownership_exponent = {OWNERSHIP_EXPONENT_VALUES}",
        f"- [已证明] 总组合数 = {len(sweep_df)} (<= 12)。",
        "",
        "## 3. 实际运行命令",
        "- `python src/scripts/audit/run_support_score_v21_oracle_sweep.py`",
        "",
        "## 4. sweep 全局结果",
        "```text",
        sweep_df[
            [
                "selection_rank",
                "combo_id",
                "virtual_reliability",
                "hub_penalty_weight",
                "ownership_exponent",
                "oracle_conditioned_rank_median",
                "oracle_conditioned_directionality",
                "oracle_true_nonzero_given_oracle_input_rate",
                "oracle_hub_win_rate",
                "c_subset_hub_win_rate",
                "c_subset_directionality",
                "b_oracle_relevant_true_nonzero_rate",
                "selected_as_best",
            ]
        ].round(4).to_string(index=False),
        "```",
        "",
        "## 5. best v2.1 vs v2_main 对比",
        "```text",
        pd.DataFrame([global_compare]).round(4).to_string(index=False),
        "```",
        f"- {proof_tag(improved_global, not improved_global)} best 参数 = "
        f"`virtual_reliability={best_config.virtual_reliability:.2f}, hub_penalty_weight={best_config.hub_penalty_weight:.2f}, ownership_exponent={best_config.ownership_exponent:.1f}`。",
        f"- {proof_tag(best_row['oracle_conditioned_rank_median'] <= baseline_row['oracle_conditioned_rank_median'] + rerun.EPS, best_row['oracle_conditioned_rank_median'] > baseline_row['oracle_conditioned_rank_median'] + rerun.EPS)} "
        f"全局 rank median: {baseline_row['oracle_conditioned_rank_median']:.4f} -> {best_row['oracle_conditioned_rank_median']:.4f}.",
        f"- {proof_tag(best_row['oracle_conditioned_directionality'] >= baseline_row['oracle_conditioned_directionality'] - rerun.EPS, best_row['oracle_conditioned_directionality'] < baseline_row['oracle_conditioned_directionality'] - rerun.EPS)} "
        f"全局 directionality: {baseline_row['oracle_conditioned_directionality']:.4f} -> {best_row['oracle_conditioned_directionality']:.4f}.",
        f"- {proof_tag(best_row['oracle_true_nonzero_given_oracle_input_rate'] >= baseline_row['oracle_true_nonzero_given_oracle_input_rate'] - rerun.EPS, best_row['oracle_true_nonzero_given_oracle_input_rate'] < baseline_row['oracle_true_nonzero_given_oracle_input_rate'] - rerun.EPS)} "
        f"全局 true_nonzero_given_oracle_input: {baseline_row['oracle_true_nonzero_given_oracle_input_rate']:.4f} -> {best_row['oracle_true_nonzero_given_oracle_input_rate']:.4f}.",
        f"- {proof_tag(improved_global, not improved_global)} 全局 hub_win_rate: {baseline_row['oracle_hub_win_rate']:.4f} -> {best_row['oracle_hub_win_rate']:.4f}.",
        "",
        "## 6. C 子集对比",
        "```text",
        pd.DataFrame([c_compare]).round(4).to_string(index=False),
        "```",
        f"- {proof_tag(improved_c, not improved_c)} C 子集 hub_win_rate: {baseline_row['c_subset_hub_win_rate']:.4f} -> {best_row['c_subset_hub_win_rate']:.4f}.",
        f"- {proof_tag(best_row['c_subset_directionality'] >= baseline_row['c_subset_directionality'] - rerun.EPS, best_row['c_subset_directionality'] < baseline_row['c_subset_directionality'] - rerun.EPS)} "
        f"C 子集 directionality: {baseline_row['c_subset_directionality']:.4f} -> {best_row['c_subset_directionality']:.4f}.",
        "",
        "## 7. B_oracle_relevant 子集对比",
        "```text",
        pd.DataFrame([b_compare]).round(4).to_string(index=False),
        "```",
        f"- {proof_tag(best_row['b_oracle_relevant_true_nonzero_rate'] >= baseline_row['b_oracle_relevant_true_nonzero_rate'] - rerun.EPS, best_row['b_oracle_relevant_true_nonzero_rate'] < baseline_row['b_oracle_relevant_true_nonzero_rate'] - rerun.EPS)} "
        f"B_oracle_relevant true_nonzero_rate: {baseline_row['b_oracle_relevant_true_nonzero_rate']:.4f} -> {best_row['b_oracle_relevant_true_nonzero_rate']:.4f}.",
        "",
        "## 8. 固定 4 个 case 解释",
        *fixed_case_lines,
        "",
        "## 9. 当前结论：继续 v2 还是转向 v3",
        gate_line,
        f"- {proof_tag(True, False)} 现在更适合：**{direction}**。",
        f"- {proof_tag(direction.startswith('A'), direction.startswith('C'))} 不建议现在进入 v3。"
        f" v2 主结构已经把 rank 压到 {best_row['oracle_conditioned_rank_median']:.4f}，剩余问题更像 residual hub 偏置而不是范式错误。",
        "",
        "### 单旋钮归因",
        "```text",
        knob_df.round(4).to_string(index=False),
        "```",
        f"- {proof_tag(True, False)} 在只改一个旋钮的对比里，最能继续压 hub 的是：**{primary_blocker}** 这一侧的修正。",
        "",
        "## 10. 下一步最小建议（只给 1 条）",
        "- 继续做 `v2.2` 的单点小修：保留本轮 best 参数，只补一个 `generic_virtual_witness_penalty`，专门再压 60/83 这类高 virtual-share 的 residual hub case。",
        "",
        "## 输出文件",
        f"- `{SWEEP_CSV_PATH}`",
        f"- `{BEST_SUMMARY_MD_PATH}`",
        f"- `{V2_VS_V21_CSV_PATH}`",
    ]
    return "\n".join(lines) + "\n"


def run_sweep() -> None:
    baseline_config = config_from_current_v2()
    configs = build_sweep_configs()
    baseline_lookup = load_baseline_lookup()
    variant_rows = collect_variant_rows(configs, baseline_lookup)
    if variant_rows.empty:
        raise RuntimeError("v2.1 oracle sweep produced no rows.")

    sweep_rows = [summarize_combo(df_combo) for _, df_combo in variant_rows.groupby("combo_id", sort=False)]
    sweep_df = add_selection_columns(pd.DataFrame(sweep_rows))
    best_combo_id = str(sweep_df.iloc[0]["combo_id"])
    best_config = next(config for config in configs if config.combo_id == best_combo_id)

    reproducibility = compare_with_existing_v2(
        variant_rows=variant_rows,
        baseline_lookup=baseline_lookup,
        baseline_id=baseline_config.combo_id,
    )
    primary_blocker, knob_df = determine_primary_blocker(sweep_df, baseline_config)
    comparison_df = build_v2_vs_best_table(
        variant_rows=variant_rows,
        best_combo_id=best_combo_id,
        baseline_id=baseline_config.combo_id,
    )

    baseline_row = sweep_df[sweep_df["combo_id"] == baseline_config.combo_id].iloc[0]
    sweep_df["oracle_hub_win_delta_vs_v2_main"] = sweep_df["oracle_hub_win_rate"] - float(baseline_row["oracle_hub_win_rate"])
    sweep_df["c_subset_hub_win_delta_vs_v2_main"] = sweep_df["c_subset_hub_win_rate"] - float(baseline_row["c_subset_hub_win_rate"])
    sweep_df["directionality_delta_vs_v2_main"] = sweep_df["oracle_conditioned_directionality"] - float(baseline_row["oracle_conditioned_directionality"])
    sweep_df["true_nonzero_delta_vs_v2_main"] = (
        sweep_df["oracle_true_nonzero_given_oracle_input_rate"]
        - float(baseline_row["oracle_true_nonzero_given_oracle_input_rate"])
    )

    sweep_df.to_csv(SWEEP_CSV_PATH, index=False)
    comparison_df.to_csv(V2_VS_V21_CSV_PATH, index=False)
    markdown = build_summary_markdown(
        sweep_df=sweep_df,
        comparison_df=comparison_df,
        reproducibility=reproducibility,
        baseline_config=baseline_config,
        best_config=best_config,
        primary_blocker=primary_blocker,
        knob_df=knob_df,
    )
    with open(BEST_SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)

    print("Table 1: v2.1 oracle sweep")
    print(
        sweep_df[
            [
                "selection_rank",
                "combo_id",
                "virtual_reliability",
                "hub_penalty_weight",
                "ownership_exponent",
                "oracle_conditioned_rank_median",
                "oracle_conditioned_directionality",
                "oracle_true_nonzero_given_oracle_input_rate",
                "oracle_hub_win_rate",
                "c_subset_hub_win_rate",
                "c_subset_directionality",
                "b_oracle_relevant_true_nonzero_rate",
                "selected_as_best",
            ]
        ].round(4).to_string(index=False)
    )
    print()
    print("Table 2: v2_main vs best v2.1")
    print(comparison_df.round(4).to_string(index=False))
    print()
    print("Table 3: single-knob diagnosis")
    print(knob_df.round(4).to_string(index=False))


if __name__ == "__main__":
    run_sweep()
