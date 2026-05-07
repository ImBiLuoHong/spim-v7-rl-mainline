from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CANDIDATE_TRAIN = Path(
    "/root/autodl-tmp/rl_spim_v7_mainline/datanew/production_data/foundation_20260114_164946_86d5023e/"
    "subgraph_v11_prod/event_20260115_001755_2874_mode_LOW_viewA.npz"
)
DEFAULT_CANDIDATE_VAL = Path(
    "/root/autodl-tmp/rl_spim_v7_mainline/datanew/production_data/foundation_20260114_164946_86d5023e/"
    "subgraph_v11_prod/event_20260116_004044_8180_mode_LOW_viewA.npz"
)


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group_view_files(paths: Sequence[str]) -> List[List[str]]:
    grouped: Dict[str, List[str]] = {}
    for path in paths:
        name = Path(path).name
        if "_view" in name:
            key = name.rsplit("_view", 1)[0]
        elif "_aug" in name:
            key = name.rsplit("_aug", 1)[0]
        else:
            key = name.rsplit(".npz", 1)[0]
        grouped.setdefault(key, []).append(path)
    return [sorted(grouped[k]) for k in sorted(grouped.keys())]


def _group_id_from_path(path: str) -> str:
    name = Path(path).name
    if "_view" in name:
        return name.rsplit("_view", 1)[0]
    if "_aug" in name:
        return name.rsplit("_aug", 1)[0]
    return name.rsplit(".npz", 1)[0]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        return {k: handle[k] for k in handle.files}


def _scalar(v: Any) -> Any:
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _jsonable(v.item())
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_hash(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _array_summary(arr: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _array_hash(arr),
    }
    if arr.size == 0:
        out["size"] = 0
        return out
    if arr.dtype.kind in "fiu":
        out.update(
            {
                "size": int(arr.size),
                "nonzero": int(np.count_nonzero(arr)),
                "sum": float(arr.sum()),
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    else:
        out["size"] = int(arr.size)
        out["preview"] = [str(x) for x in arr.reshape(-1)[: min(8, arr.size)].tolist()]
    return out


def _timeline_fingerprint(data: np.ndarray) -> Dict[str, Any]:
    if data.ndim != 3:
        return {"shape": list(data.shape), "note": "not_3d"}
    time_len, chan_len = data.shape[0], data.shape[1]
    channel_summaries: List[Dict[str, Any]] = []
    for ch in range(chan_len):
        series = data[:, ch, ...]
        if series.ndim == 2:
            time_active = np.any(series != 0, axis=1)
            active_vals = series[time_active]
        else:
            time_active = series != 0
            active_vals = series[time_active]
        active_steps = np.where(time_active)[0]
        channel_summaries.append(
            {
                "channel": ch,
                "active_steps": int(active_steps.size),
                "first_active_step": int(active_steps[0]) if active_steps.size else None,
                "last_active_step": int(active_steps[-1]) if active_steps.size else None,
                "active_value_sha256": _sha256_bytes(np.ascontiguousarray(active_vals).tobytes()) if active_vals.size else None,
            }
        )
    return {
        "shape": list(data.shape),
        "time_len": int(time_len),
        "channels": channel_summaries,
    }


def _timeline_shift_analysis(left: np.ndarray, right: np.ndarray, max_shift: int = 16) -> Dict[str, Any]:
    if left.ndim < 1 or right.ndim < 1 or left.shape != right.shape:
        return {"available": False, "reason": "shape_mismatch"}
    left_series = left.reshape(left.shape[0], -1).sum(axis=1).astype(np.float64)
    right_series = right.reshape(right.shape[0], -1).sum(axis=1).astype(np.float64)
    best_shift = 0
    best_mae = float("inf")
    for shift in range(-int(max_shift), int(max_shift) + 1):
        if shift >= 0:
            l = left_series[shift:]
            r = right_series[: len(l)]
        else:
            l = left_series[:shift]
            r = right_series[-shift:]
        if l.size <= 0:
            continue
        mae = float(np.mean(np.abs(l - r)))
        if mae < best_mae:
            best_mae = mae
            best_shift = int(shift)
    return {
        "available": True,
        "left_nonzero_steps": [int(v) for v in np.where(np.abs(left_series) > 1e-9)[0].tolist()],
        "right_nonzero_steps": [int(v) for v in np.where(np.abs(right_series) > 1e-9)[0].tolist()],
        "best_shift_steps_left_to_right": int(best_shift),
        "best_shift_mae": float(best_mae),
    }


def _format_value(v: Any, max_len: int = 220) -> str:
    js = json.dumps(_jsonable(v), ensure_ascii=True, sort_keys=True)
    if len(js) <= max_len:
        return js
    return js[: max_len - 3] + "..."


def _load_sample_from_subgraph_path(subgraph_path: Path) -> Path:
    return Path(str(subgraph_path).replace("/subgraph_v11_prod/", "/samples/")).with_name(
        _group_id_from_path(subgraph_path.name) + ".npz"
    )


def _view_filenames(group_paths: Sequence[str]) -> List[str]:
    return sorted(Path(p).name for p in group_paths)


def _select_groups(split_file: Path, max_groups: int) -> List[List[str]]:
    raw_paths = _read_lines(split_file)
    groups = _group_view_files(raw_paths)
    if max_groups > 0:
        groups = groups[:max_groups]
    return groups


def _strict_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    sample = record["sample"]
    subgraph = record["subgraph"]
    return (
        sample["meta"][0],
        int(sample["label"][0]),
        int(sample["label"][1]),
        float(sample["meta"][1]),
        float(sample["meta"][2]),
        sample["meta"][3],
        sample["data_sha256"],
        sample["node_indices_sha256"],
        int(subgraph["global_injection_node"]),
        int(subgraph["global_trigger_node"]),
        int(subgraph["global_start_step"]),
        int(subgraph["trigger_time_step"]),
        subgraph["global_node_indices_sha256"],
        subgraph["global_edge_indices_sha256"],
        subgraph["x_sha256"],
        subgraph["edge_attr_sha256"],
    )


def _legacy_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    subgraph = record["subgraph"]
    return (
        int(subgraph["global_injection_node"]),
        int(subgraph["global_trigger_node"]),
        int(subgraph["global_start_step"]),
        int(subgraph["trigger_time_step"]),
        tuple(int(x) for x in subgraph["global_node_indices"]),
        tuple(int(x) for x in subgraph["global_edge_indices"]),
    )


def _risk_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    subgraph = record["subgraph"]
    sample = record["sample"]
    return (
        int(subgraph["global_injection_node"]),
        int(subgraph["global_trigger_node"]),
        int(subgraph["global_start_step"]),
        int(subgraph["trigger_time_step"]),
        int(subgraph["trigger_node_local"]),
        int(subgraph["injection_node_local"]),
        tuple(int(x) for x in subgraph["global_node_indices"]),
        tuple(int(x) for x in subgraph["global_edge_indices"]),
        sample["meta"][0],
    )


def _build_record(group_paths: Sequence[str]) -> Dict[str, Any]:
    subgraph_path = Path(group_paths[0])
    sample_path = _load_sample_from_subgraph_path(subgraph_path)
    sample = _load_npz(sample_path)
    subgraph = _load_npz(subgraph_path)
    sample_data = sample["data"]
    subgraph_x = subgraph["x"]
    return {
        "group_id": _group_id_from_path(group_paths[0]),
        "group_files": list(group_paths),
        "view_files": _view_filenames(group_paths),
        "sample_path": str(sample_path),
        "subgraph_path": str(subgraph_path),
        "sample": {
            "data": sample_data,
            "data_sha256": _array_hash(sample_data),
            "data_summary": _array_summary(sample_data),
            "timeline_fingerprint": _timeline_fingerprint(sample_data),
            "node_indices": sample["node_indices"],
            "node_indices_sha256": _array_hash(sample["node_indices"]),
            "label": sample["label"],
            "meta": sample["meta"],
        },
        "subgraph": {
            "x": subgraph_x,
            "x_sha256": _array_hash(subgraph_x),
            "x_summary": _array_summary(subgraph_x),
            "edge_attr": subgraph["edge_attr"],
            "edge_attr_sha256": _array_hash(subgraph["edge_attr"]),
            "edge_attr_summary": _array_summary(subgraph["edge_attr"]),
            "edge_index": subgraph["edge_index"],
            "edge_index_sha256": _array_hash(subgraph["edge_index"]),
            "edge_index_summary": _array_summary(subgraph["edge_index"]),
            "trigger_time_step": _scalar(subgraph["trigger_time_step"]),
            "trigger_node_local": _scalar(subgraph["trigger_node_local"]),
            "injection_node_local": _scalar(subgraph["injection_node_local"]),
            "global_node_indices": subgraph["global_node_indices"],
            "global_node_indices_sha256": _array_hash(subgraph["global_node_indices"]),
            "global_edge_indices": subgraph["global_edge_indices"],
            "global_edge_indices_sha256": _array_hash(subgraph["global_edge_indices"]),
            "global_trigger_node": _scalar(subgraph["global_trigger_node"]),
            "global_injection_node": _scalar(subgraph["global_injection_node"]),
            "anchor_type": _scalar(subgraph["anchor_type"]),
            "view_type": _scalar(subgraph["view_type"]),
            "group_id": _scalar(subgraph["group_id"]),
            "true_spread_size": _scalar(subgraph["true_spread_size"]),
            "global_start_step": _scalar(subgraph["global_start_step"]),
        },
    }


def _compare_value(left: Any, right: Any) -> Tuple[bool, str]:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False, "type_mismatch"
        same = np.array_equal(left, right)
        if same:
            return True, "exact_array_equal"
        if left.shape == right.shape and left.dtype.kind in "fiu" and right.dtype.kind in "fiu":
            diff = left.astype(np.float64) - right.astype(np.float64)
            return False, f"array_diff_nonzero={int(np.count_nonzero(diff))};max_abs={float(np.max(np.abs(diff)))}"
        return False, "array_mismatch"
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            same = abs(float(left) - float(right)) <= 1e-9
        except Exception:
            return False, "numeric_parse_error"
        return same, "exact_numeric_equal" if same else f"numeric_delta={float(right) - float(left)}"
    same = left == right
    return bool(same), "exact_equal" if same else "value_mismatch"


def _comparison_rows(train_record: Dict[str, Any], val_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(section: str, field: str, left: Any, right: Any) -> None:
        same, note = _compare_value(left, right)
        rows.append(
            {
                "section": section,
                "field": field,
                "train_value": _format_value(left),
                "val_value": _format_value(right),
                "equal": bool(same),
                "note": note,
            }
        )

    train_sample = train_record["sample"]
    val_sample = val_record["sample"]
    train_sub = train_record["subgraph"]
    val_sub = val_record["subgraph"]

    add("sample", "sample_path", train_record["sample_path"], val_record["sample_path"])
    add("sample", "data_sha256", train_sample["data_sha256"], val_sample["data_sha256"])
    add("sample", "data_summary", train_sample["data_summary"], val_sample["data_summary"])
    add("sample", "timeline_fingerprint", train_sample["timeline_fingerprint"], val_sample["timeline_fingerprint"])
    add("sample", "node_indices", train_sample["node_indices"], val_sample["node_indices"])
    add("sample", "node_indices_sha256", train_sample["node_indices_sha256"], val_sample["node_indices_sha256"])
    add("sample", "label", train_sample["label"], val_sample["label"])
    add("sample", "meta", train_sample["meta"], val_sample["meta"])

    add("subgraph", "subgraph_path", train_record["subgraph_path"], val_record["subgraph_path"])
    add("subgraph", "group_id", train_sub["group_id"], val_sub["group_id"])
    add("subgraph", "view_type", train_sub["view_type"], val_sub["view_type"])
    add("subgraph", "global_trigger_node", train_sub["global_trigger_node"], val_sub["global_trigger_node"])
    add("subgraph", "global_injection_node", train_sub["global_injection_node"], val_sub["global_injection_node"])
    add("subgraph", "trigger_node_local", train_sub["trigger_node_local"], val_sub["trigger_node_local"])
    add("subgraph", "injection_node_local", train_sub["injection_node_local"], val_sub["injection_node_local"])
    add("subgraph", "trigger_time_step", train_sub["trigger_time_step"], val_sub["trigger_time_step"])
    add("subgraph", "global_start_step", train_sub["global_start_step"], val_sub["global_start_step"])
    add("subgraph", "anchor_type", train_sub["anchor_type"], val_sub["anchor_type"])
    add("subgraph", "true_spread_size", train_sub["true_spread_size"], val_sub["true_spread_size"])
    add("subgraph", "global_node_indices", train_sub["global_node_indices"], val_sub["global_node_indices"])
    add("subgraph", "global_node_indices_sha256", train_sub["global_node_indices_sha256"], val_sub["global_node_indices_sha256"])
    add("subgraph", "global_edge_indices", train_sub["global_edge_indices"], val_sub["global_edge_indices"])
    add("subgraph", "global_edge_indices_sha256", train_sub["global_edge_indices_sha256"], val_sub["global_edge_indices_sha256"])
    add("subgraph", "edge_index", train_sub["edge_index"], val_sub["edge_index"])
    add("subgraph", "edge_index_sha256", train_sub["edge_index_sha256"], val_sub["edge_index_sha256"])
    add("subgraph", "edge_attr", train_sub["edge_attr"], val_sub["edge_attr"])
    add("subgraph", "edge_attr_sha256", train_sub["edge_attr_sha256"], val_sub["edge_attr_sha256"])
    add("subgraph", "x", train_sub["x"], val_sub["x"])
    add("subgraph", "x_sha256", train_sub["x_sha256"], val_sub["x_sha256"])
    add("subgraph", "x_summary", train_sub["x_summary"], val_sub["x_summary"])

    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict overlap audit on train subset 500 vs val 1031.")
    parser.add_argument("--train-split", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--val-split", type=Path, default=Path("data/val.txt"))
    parser.add_argument("--train-max-groups", type=int, default=500)
    parser.add_argument("--val-max-groups", type=int, default=1031)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-train-path", type=Path, default=DEFAULT_CANDIDATE_TRAIN)
    parser.add_argument("--candidate-val-path", type=Path, default=DEFAULT_CANDIDATE_VAL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_groups = _select_groups(args.train_split, args.train_max_groups)
    val_groups = _select_groups(args.val_split, args.val_max_groups)

    train_records = [_build_record(group) for group in train_groups]
    val_records = [_build_record(group) for group in val_groups]

    train_legacy = {_legacy_identity(record): record for record in train_records}
    val_legacy = {_legacy_identity(record): record for record in val_records}
    train_strict = {_strict_identity(record): record for record in train_records}
    val_strict = {_strict_identity(record): record for record in val_records}
    train_risk = {_risk_identity(record): record for record in train_records}
    val_risk = {_risk_identity(record): record for record in val_records}

    legacy_overlap_keys = sorted(set(train_legacy) & set(val_legacy), key=lambda x: json.dumps(_jsonable(x), ensure_ascii=True, sort_keys=True))
    strict_overlap_keys = sorted(set(train_strict) & set(val_strict), key=lambda x: json.dumps(_jsonable(x), ensure_ascii=True, sort_keys=True))
    alias_risk_keys = sorted(set(train_risk) & set(val_risk), key=lambda x: json.dumps(_jsonable(x), ensure_ascii=True, sort_keys=True))

    train_candidate = _build_record([str(args.candidate_train_path)])
    val_candidate = _build_record([str(args.candidate_val_path)])
    candidate_rows = _comparison_rows(train_candidate, val_candidate)
    _write_csv(args.output_dir / "suspicious_pair_raw_comparison.csv", candidate_rows)

    candidate_strict_equal = _strict_identity(train_candidate) == _strict_identity(val_candidate)
    candidate_legacy_equal = _legacy_identity(train_candidate) == _legacy_identity(val_candidate)
    candidate_timeline_shift = _timeline_shift_analysis(
        np.asarray(train_candidate["subgraph"]["x"]),
        np.asarray(val_candidate["subgraph"]["x"]),
    )
    if candidate_strict_equal:
        alias_conclusion = "ALIAS_CONFIRMED"
    elif candidate_legacy_equal:
        alias_conclusion = "ALIAS_NOT_CONFIRMED"
    else:
        alias_conclusion = "ALIAS_NOT_CONFIRMED"

    suspicious_groups: List[Dict[str, Any]] = []
    for key in strict_overlap_keys:
        train_record = train_strict[key]
        val_record = val_strict[key]
        suspicious_groups.append(
            {
                "strict_identity": _jsonable(key),
                "train_group_id": train_record["group_id"],
                "val_group_id": val_record["group_id"],
                "train_sample_path": train_record["sample_path"],
                "val_sample_path": val_record["sample_path"],
            }
        )
    alias_risk_groups: List[Dict[str, Any]] = []
    for key in alias_risk_keys:
        train_record = train_risk[key]
        val_record = val_risk[key]
        # Risk exists when coarse identity collides but strict identity does not.
        if _strict_identity(train_record) == _strict_identity(val_record):
            continue
        alias_risk_groups.append(
            {
                "risk_identity": _jsonable(key),
                "train_group_id": train_record["group_id"],
                "val_group_id": val_record["group_id"],
                "train_subgraph_path": train_record["subgraph_path"],
                "val_subgraph_path": val_record["subgraph_path"],
                "train_data_sha256": train_record["sample"]["data_sha256"],
                "val_data_sha256": val_record["sample"]["data_sha256"],
                "train_x_sha256": train_record["subgraph"]["x_sha256"],
                "val_x_sha256": val_record["subgraph"]["x_sha256"],
            }
        )

    payload = {
        "audit_version": "strict_overlap_audit_v1",
        "split_scope": {
            "train_split": str(args.train_split.resolve()),
            "val_split": str(args.val_split.resolve()),
            "train_subset_group_count": len(train_records),
            "val_subset_group_count": len(val_records),
            "train_subset_case_count_lmdb": len(train_records),
            "val_subset_case_count_lmdb": len(val_records),
        },
        "candidate_pair": {
            "train_path": str(args.candidate_train_path),
            "val_path": str(args.candidate_val_path),
            "alias_conclusion": alias_conclusion,
            "legacy_identity_equal": bool(candidate_legacy_equal),
            "strict_identity_equal": bool(candidate_strict_equal),
            "timeline_shift_analysis": candidate_timeline_shift,
        },
        "legacy_overlap_audit": {
            "overlap_count": len(legacy_overlap_keys),
            "sample": [_jsonable(k) for k in legacy_overlap_keys[:5]],
        },
        "strict_overlap_audit": {
            "overlap_count": len(strict_overlap_keys),
            "sample": [_jsonable(k) for k in strict_overlap_keys[:5]],
            "suspicious_groups": suspicious_groups,
        },
        "alias_risk_audit": {
            "risk_count": len(alias_risk_groups),
            "risk_groups": alias_risk_groups,
        },
        "candidate_raw_comparison": {
            "row_count": len(candidate_rows),
            "sections": sorted(set(row["section"] for row in candidate_rows)),
            "summary": {
                "exact_equal_fields": [row["field"] for row in candidate_rows if row["equal"]],
                "mismatched_fields": [row["field"] for row in candidate_rows if not row["equal"]],
            },
        },
    }

    (args.output_dir / "strict_overlap_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
