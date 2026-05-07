
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.modeling.architectures.phase4_5_model import Phase45Model

# --- CONTRACT CONSTANTS ---
BUDGET = 60
ROUNDS = 20
SAMPLES_PER_ROUND = 3

def calculate_random_hit_prob(M, B):
    if M <= 0: return 0.0
    if B >= M: return 1.0
    return B / M

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='runs/ceiling_decomp')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Config & Model
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, "resolved_config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = Config()
        if 'model' in cfg_dict:
            for k, v in cfg_dict['model'].items():
                if hasattr(cfg.model, k): setattr(cfg.model, k, v)
        if 'loss' in cfg_dict: cfg.loss = cfg_dict['loss']
    else:
        cfg = Config()
        
    device = torch.device(args.device)
    model = Phase45Model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load Data
    if args.data_root: cfg.paths.samples_path = args.data_root
    _, _, test_loader, _ = create_dataloaders(data_root=cfg.paths.samples_path, cfg=cfg, batch_size=1)
    
    results = []
    
    print(f"Starting Policy Decomposition (Limit={args.limit})")
    
    for i, batch in tqdm(enumerate(test_loader)):
        if args.limit and i >= args.limit: break
        
        batch = batch.to(device)
        M = batch.num_nodes
        y = batch.y
        if y.dim() > 1: y = y.squeeze()
        source_indices = (y > 0.5).nonzero(as_tuple=False).view(-1)
        if len(source_indices) == 0: continue
        
        # 1. Random Baseline
        p_random = calculate_random_hit_prob(M, BUDGET)
        
        # 2. APT (Nav-Guided) - The standard system
        with torch.no_grad():
            out_apt = model(batch, inference_mode=False, max_steps=ROUNDS, action_k=SAMPLES_PER_ROUND, action_policy='nav_guided')
        sr_apt = out_apt['step_metrics']['success']
        
        # 3. Oracle (Greedy Posterior) - The theoretical ceiling of current belief
        with torch.no_grad():
            out_oracle = model(batch, inference_mode=False, max_steps=ROUNDS, action_k=SAMPLES_PER_ROUND, action_policy='greedy')
        sr_oracle = out_oracle['step_metrics']['success']
        
        results.append({
            'episode_idx': i,
            'M': M,
            'Random': p_random,
            'APT_NavGuided': sr_apt,
            'Oracle_Greedy': sr_oracle
        })
        
    df = pd.DataFrame(results)
    
    print("\n=== Policy Decomposition (Strict B=60) ===")
    print(f"Episodes: {len(df)}")
    print(df[['Random', 'APT_NavGuided', 'Oracle_Greedy']].mean())
    
    df.to_csv(os.path.join(args.output_dir, "decomp_results.csv"), index=False)

if __name__ == "__main__":
    main()
