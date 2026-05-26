from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


VIEWS = ("_viewA.npz", "_viewB.npz", "_viewC.npz")


def _scalar_str(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _derive_group_id(path: Path, record) -> str:
    if "group_id" in record:
        return _scalar_str(record["group_id"])

    name = path.name
    for suffix in VIEWS:
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".npz"
    return name


def _infer_num_nodes_full(foundation_dir: Path) -> int:
    graph_path = foundation_dir / "graph.npz"
    with np.load(graph_path, allow_pickle=True) as graph:
        if "num_nodes" in graph:
            return int(graph["num_nodes"])
        if "coordinates" in graph:
            return int(graph["coordinates"].shape[0])
        if "node_coordinates" in graph:
            return int(graph["node_coordinates"].shape[0])
        if "node_coords" in graph:
            return int(graph["node_coords"].shape[0])
        if "node_ids" in graph:
            return int(graph["node_ids"].shape[0])
    raise KeyError(f"Cannot infer full node count from {graph_path}")


def convert_one(
    v11_path: Path,
    *,
    raw_samples_dir: Path,
    foundation_dir: Path,
    output_dir: Path,
    num_nodes_full: int,
) -> tuple[bool, str]:
    try:
        with np.load(v11_path, allow_pickle=True) as record:
            group_id = _derive_group_id(v11_path, record)
            sample_path = raw_samples_dir / group_id
            if not sample_path.exists():
                return False, f"missing raw sample: {sample_path}"

            fields = {
                "sample_path": np.str_(str(sample_path)),
                "foundation_dir": np.str_(str(foundation_dir)),
                "edge_index": record["edge_index"].astype(np.int32),
                "y": record["y"],
                "node_rpe_hops": record["node_rpe_hops"].astype(np.int16),
                "node_rpe_stt": record["node_rpe_stt"].astype(np.float32),
                "node_rpe_euclidean": record["node_rpe_euclidean"].astype(np.float32),
                "global_node_indices": record["global_node_indices"].astype(np.int32),
                "global_edge_indices": record["global_edge_indices"].astype(np.int32),
                "trigger_time_step": record["trigger_time_step"],
                "trigger_node_local": record["trigger_node_local"],
                "injection_node_local": record["injection_node_local"],
                "global_trigger_node": record["global_trigger_node"],
                "global_injection_node": record["global_injection_node"],
                "anchor_type": record["anchor_type"],
                "view_type": record["view_type"],
                "group_id": np.str_(group_id),
                "true_spread_size": record["true_spread_size"],
                "global_start_step": record["global_start_step"],
                "num_channels": np.array(int(record["x"].shape[-1]) if "x" in record else 2, dtype=np.int64),
                "num_nodes_full": np.array(int(num_nodes_full), dtype=np.int64),
            }

        np.savez_compressed(output_dir / v11_path.name, **fields)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert V11 dense subgraph NPZ files to V12 indexer NPZ files.")
    parser.add_argument("--foundation-dir", required=True)
    parser.add_argument("--v11-dir", required=True)
    parser.add_argument("--raw-samples-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any V11 file cannot be converted.",
    )
    args = parser.parse_args()

    foundation_dir = Path(args.foundation_dir).resolve()
    v11_dir = Path(args.v11_dir).resolve()
    raw_samples_dir = Path(args.raw_samples_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes_full = _infer_num_nodes_full(foundation_dir)

    files = sorted(v11_dir.glob("*.npz"))
    converted = 0
    failures: list[tuple[str, str]] = []
    for path in tqdm(files, desc="v11->v12-indexer"):
        good, message = convert_one(
            path,
            raw_samples_dir=raw_samples_dir,
            foundation_dir=foundation_dir,
            output_dir=output_dir,
            num_nodes_full=num_nodes_full,
        )
        if good:
            converted += 1
        else:
            failures.append((path.name, message))

    print(f"converted={converted} total={len(files)} failed={len(failures)} output={output_dir}")
    for name, message in failures[:20]:
        print(f"FAIL {name}: {message}")

    if failures and bool(args.strict):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
