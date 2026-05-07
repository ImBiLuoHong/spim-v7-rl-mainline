import numpy as np
import os
import argparse
from tqdm import tqdm

def precompute_edge_attrs(graph_path):
    print(f"Loading {graph_path}...")
    with np.load(graph_path) as data:
        # Load all existing keys to memory so we can save them back
        content = dict(data)
    
    if 'edge_attr_dynamic' not in content:
        print("Error: edge_attr_dynamic not found in graph.npz")
        return

    # Shape: (T, E, C) or (E, T, C)?
    # Script check said: (288, 68557, 3) -> (T, E, C)
    dynamic = content['edge_attr_dynamic']
    T, E, C = dynamic.shape
    print(f"Dynamic attributes shape: {dynamic.shape}")

    # Ch0: Flow, Ch1: STT
    flow = dynamic[..., 0] # (T, E)
    stt = dynamic[..., 1]  # (T, E)

    print("Computing p_forward...")
    # p_forward: Probability that flow is positive (u -> v)
    # We use > 1e-6 to avoid floating point noise around 0
    p_forward = (flow > 1e-6).mean(axis=0) # (E,)

    print("Computing STT stats...")
    # q50, q90, min
    q50_stt = np.quantile(stt, 0.5, axis=0) # (E,)
    q90_stt = np.quantile(stt, 0.9, axis=0) # (E,)
    min_stt = np.min(stt, axis=0)           # (E,)

    print("Computing flip_rate...")
    # Flip rate: frequency of sign changes
    # sign: -1, 0, 1
    signs = np.sign(flow)
    # We care about direction changes.
    # A simple way: count how many times sign(t) != sign(t-1)
    # But 0 is tricky.
    # Let's count "crossing zero".
    # (sign[t] * sign[t-1] < 0)
    
    # We'll use diff of signs.
    # If +1 -> -1: diff = -2. abs=2.
    # If +1 -> 0: diff = -1. abs=1.
    # If 0 -> -1: diff = -1. abs=1.
    # If we want strictly reversals, we check sign product < 0.
    
    # Vectorized check for sign product < 0 across time
    # signs[:-1] * signs[1:]
    sign_prod = signs[:-1] * signs[1:] # (T-1, E)
    flip_count = (sign_prod < 0).sum(axis=0) # (E,)
    flip_rate = flip_count / (T - 1)

    # Stack features
    # Order: [p_forward, q50_stt, q90_stt, min_stt, flip_rate]
    # Shape: (E, 5)
    new_attrs = np.stack([p_forward, q50_stt, q90_stt, min_stt, flip_rate], axis=1)
    print(f"Computed edge_attr_summary shape: {new_attrs.shape}")
    
    # Add to content
    content['edge_attr_summary'] = new_attrs.astype(np.float32)
    
    # Save back
    print("Saving graph.npz...")
    np.savez_compressed(graph_path, **content)
    print("Done.")

if __name__ == "__main__":
    path = '/root/autodl-tmp/rl_spim_v7_mainline/datanew/production_data/foundation_20260114_164946_86d5023e/graph.npz'
    precompute_edge_attrs(path)
