
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.special import comb
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.getcwd())

from src.config.core import Config
from src.data.v6.loader import create_dataloaders
from src.modeling.architectures.phase4_5_model import Phase45Model
from src.baselines.hsr_agent import HSRAgent
from src.baselines.zju_agent import ZJUAgent

# --- CONSTANTS ---
BUDGET = 60
ROUNDS = 20
SAMPLES_PER_ROUND = 3

def load_graph_data(data_root):
    # Try to find graph.npz
    # data_root is usually .../subgraph_v11_prod
    foundation_path = os.path.join(os.path.dirname(data_root), "graph.npz")
    if os.path.exists(foundation_path):
        return foundation_path
    return None

def init_baselines(data_root):
    graph_path = load_graph_data(data_root)
    if not graph_path:
        print("WARNING: graph.npz not found. Baselines will be disabled.")
        return None, None
        
    print(f"Initializing Baselines with graph: {graph_path}")
    
    # HSR
    try:
        hsr_agent = HSRAgent(graph_path=graph_path)
    except Exception as e:
        print(f"Failed to load HSRAgent: {e}")
        hsr_agent = None
        
    # MGSM (ZJU)
    try:
        # ZJUAgent needs graph_data object usually, but let's see constructor
        # It takes graph_data. We can pass HSRAgent's graph_data if compatible
        # or load it. HSRAgent loads it into self.edge_index etc.
        # But ZJUAgent expects a PyG-like object or similar.
        # Let's peek at ZJUAgent.__init__: (self, graph_data, ...)
        # It uses graph_data.edge_index, graph_data.edge_attr.
        # We can construct a simple object.
        class GraphData:
            def __init__(self, path):
                data = np.load(path)
                self.edge_index = torch.tensor(data['edge_index'])
                self.edge_attr = torch.tensor(data['edge_attr']) if 'edge_attr' in data else None
                self.num_nodes = data['x'].shape[0] if 'x' in data else self.edge_index.max().item() + 1
                
        gdata = GraphData(graph_path)
        mgsm_agent = ZJUAgent(graph_data=gdata)
    except Exception as e:
        print(f"Failed to load ZJUAgent: {e}")
        mgsm_agent = None
        
    return hsr_agent, mgsm_agent

def load_model_and_data(checkpoint_path, data_root=None, device='cpu'):
    # Reuse loading logic (simplified)
    run_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(run_dir, "resolved_config.yaml")
    
    if not os.path.exists(config_path):
        cfg = Config()
        cfg.model.architecture = 'phase4_5' 
    else:
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = Config()
        if 'model' in cfg_dict:
            for k, v in cfg_dict['model'].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
                elif isinstance(cfg.model, dict):
                    cfg.model[k] = v
        if 'loss' in cfg_dict:
            cfg.loss = cfg_dict['loss']

    device = torch.device(device)
    print(f"Loading model from {checkpoint_path}...")
    model = Phase45Model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("Loading Data...")
    if data_root:
        cfg.paths.samples_path = data_root
        
    _, _, test_loader, _ = create_dataloaders(
        data_root=cfg.paths.samples_path,
        cfg=cfg,
        batch_size=1
    )
    
    # Load Baselines
    hsr_agent, mgsm_agent = init_baselines(cfg.paths.samples_path)

    return model, test_loader, hsr_agent, mgsm_agent

def calculate_random_hit_prob(M, B):
    if M <= 0: return 0.0
    if B >= M: return 1.0
    # P(Hit) = 1 - C(M-1, B)/C(M, B) = 1 - (M-B)/M = B/M
    return B / M

def run_baseline_episode(agent, batch, agent_type='hsr', max_rounds=ROUNDS, k=SAMPLES_PER_ROUND):
    if agent is None: return False, 0
    
    # Extract Episode Info
    # global_indices: batch.n_id
    if not hasattr(batch, 'n_id'): return False, 0
    global_indices = batch.n_id.cpu().numpy()
    global_to_local = {gid: i for i, gid in enumerate(global_indices)}
    
    # Trigger Info
    # Check Channel 4 (Anchor/Trigger) for initial non-zero
    # batch.x shape [N, C] or [N, T, C]
    # Phase45Model usually works with static [N, C]. 
    # But if we want true time-series feedback, we assume batch.x might be static snapshot?
    # Wait, Phase45Model uses T_sim updates but doesn't seem to update X.
    # We will assume batch.x contains "Ground Truth" in Channel 0 (Signal).
    # And Trigger in Channel 4.
    
    x = batch.x
    if x.dim() == 2:
        # [N, C]
        signal = x[:, 0] # Signal
        trigger_mask = x[:, 4] > 0
    elif x.dim() == 3:
        # [N, T, C] -> Take first step? Or aggregation?
        # Assuming [N, C] for now as per Phase45Model
        signal = x[:, 0, 0] # ?
        return False, 0 # Not supporting temporal X yet
    else:
        return False, 0
        
    if trigger_mask.sum() == 0:
        # No trigger?
        return False, 0
    
    # Get first trigger node
    trigger_local = torch.where(trigger_mask)[0][0].item()
    trigger_global = int(global_indices[trigger_local])
    
    # True Source
    y = batch.y
    if y.dim() > 1: y = y.squeeze()
    source_mask = y > 0.5
    if source_mask.sum() == 0: return False, 0
    source_local = torch.where(source_mask)[0][0].item()
    source_global = int(global_indices[source_local])
    
    # Reset Agent
    if agent_type == 'hsr':
        agent.reset(candidates=global_indices, trigger_node=trigger_global)
    elif agent_type == 'mgsm':
        agent.reset(initial_trigger_node=trigger_global)
    
    success = False
    rounds_taken = 0
    samples_taken = 0
    
    for r in range(max_rounds):
        rounds_taken += 1
        
        # Get Action
        if agent_type == 'hsr':
            # HSR scalable returns list of global IDs
            actions = agent.get_action_hsr_scalable(k=k)
        elif agent_type == 'mgsm':
            # MGSM returns list of global IDs
            actions = agent.get_action(k=k)
            if isinstance(actions, int): actions = [actions]
            
        # Execute Actions
        for action_global in actions:
            samples_taken += 1
            
            # Check Hit
            if action_global == source_global:
                success = True
                break
            
            # Feedback
            # Need local index
            if action_global in global_to_local:
                local_idx = global_to_local[action_global]
                # Observation: Is contaminated?
                # Using Channel 0 (Signal) > 0.5 (or epsilon)
                is_contaminated = 1 if signal[local_idx] > 1e-6 else 0
                
                if agent_type == 'hsr':
                    # HSR step takes dict {node: (val, type)}?
                    # agent.step(obs)
                    # obs = {global_id: (val, type)}
                    # val is float. type 1=sensor?
                    agent.step({action_global: (float(signal[local_idx]), 1)})
                elif agent_type == 'mgsm':
                    # ZJUAgent step(action, result)
                    agent.step(action_global, is_contaminated)
            else:
                # Action outside subgraph?
                # Should not happen if candidates restricted to subgraph
                pass
                
        if success:
            break
            
    return success, rounds_taken

def run_audit(args):
    model, loader, hsr_agent, mgsm_agent = load_model_and_data(args.checkpoint, args.data_root, args.device)
    
    results = []
    
    for i, batch in tqdm(enumerate(loader)):
        if args.limit and i >= args.limit: break
        
        batch = batch.to(args.device)
        M = batch.num_nodes
        y = batch.y
        if y.dim() > 1: y = y.squeeze()
        source_indices = (y > 0.5).nonzero(as_tuple=False).view(-1)
        if len(source_indices) == 0:
            continue
        
        # B1: Random
        p_random = calculate_random_hit_prob(M, BUDGET)
        
        # B2: Greedy Posterior
        with torch.no_grad():
            out_greedy = model(batch, inference_mode=False, max_steps=ROUNDS, action_k=SAMPLES_PER_ROUND, action_policy='greedy')
        sr_greedy = out_greedy['step_metrics']['success']
        
        # B3: Nav Guided
        with torch.no_grad():
            out_nav = model(batch, inference_mode=False, max_steps=ROUNDS, action_k=SAMPLES_PER_ROUND, action_policy='nav_guided')
        sr_nav = out_nav['step_metrics']['success']
        
        # B4: HSR
        hsr_success, hsr_rounds = run_baseline_episode(hsr_agent, batch, 'hsr')
        
        # B5: MGSM
        mgsm_success, mgsm_rounds = run_baseline_episode(mgsm_agent, batch, 'mgsm')
        
        results.append({
            'episode_idx': i,
            'M': M,
            'random_p': p_random,
            'greedy_success': sr_greedy,
            'nav_success': sr_nav,
            'hsr_success': float(hsr_success),
            'mgsm_success': float(mgsm_success)
        })
        
    df = pd.DataFrame(results)
    
    # Summarize
    print("\n=== Policy Upper Bound Audit (B=60) ===")
    print(f"Episodes: {len(df)}")
    print(f"Random Baseline (Theoretical): {df['random_p'].mean():.2%}")
    print(f"Greedy Posterior (Top3):       {df['greedy_success'].mean():.2%}")
    print(f"Nav Guided (Top1+Nav2):        {df['nav_success'].mean():.2%}")
    print(f"HSR (B=60):                    {df['hsr_success'].mean():.2%}")
    print(f"MGSM (B=60):                   {df['mgsm_success'].mean():.2%}")
    
    # Save
    df.to_json(os.path.join(args.output_dir, "oracle_policy_results.jsonl"), orient='records', lines=True)
    
    with open("docs/CEILING_ORACLE_POLICY_v2.md", "w") as f:
        f.write("# Task B: Policy Upper Bound v2\n\n")
        f.write(f"Budget B={BUDGET} ({ROUNDS}x{SAMPLES_PER_ROUND})\n\n")
        f.write("| Policy | SR |\n|---|---|\n")
        f.write(f"| Random | {df['random_p'].mean():.2%} |\n")
        f.write(f"| Greedy Posterior | {df['greedy_success'].mean():.2%} |\n")
        f.write(f"| Nav Guided | {df['nav_success'].mean():.2%} |\n")
        f.write(f"| HSR | {df['hsr_success'].mean():.2%} |\n")
        f.write(f"| MGSM | {df['mgsm_success'].mean():.2%} |\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='runs/oracle_v2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    run_audit(args)
