import os
import numpy as np


def _read_edges_csv_generic(path):
    pairs = []
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        pairs.append((int(parts[0]), int(parts[1])))
        except Exception:
            pass
    return pairs


def load_eval_edges(cfg_obj, run_dir=None):
    candidates = []
    root = ""
    try:
        root = str(getattr(getattr(cfg_obj, "paths", object()), "root_dir", os.getcwd()) or os.getcwd())
    except Exception:
        root = os.getcwd()

    candidates.extend([
        os.path.join(run_dir or "", "artifacts", "edge_index_main.csv"),
        os.path.join(root, "artifacts", "edge_index_main.csv"),
        os.path.join(root, "data_split", "pipe_map.csv"),
    ])
    for p in candidates:
        if p and os.path.exists(p):
            edges = _read_edges_csv_generic(p)
            if edges:
                arr = np.array(edges, dtype=np.int64).T
                if arr.ndim == 2 and arr.shape[0] == 2:
                    return arr
    return None


def load_adj(cfg, run_dir=None):
    edge_index = load_eval_edges(cfg, run_dir)
    adj = {}
    if edge_index is not None and edge_index.ndim == 2:
        for i in range(int(edge_index.shape[1])):
            u = int(edge_index[0, i])
            v = int(edge_index[1, i])
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    return adj


def connected_recall_at_k(scores_b, true_id, K, adj):
    return float("nan")
