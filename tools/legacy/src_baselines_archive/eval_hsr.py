
import os
import sys
import numpy as np
import glob
from tqdm import tqdm
import torch
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from baselines.hsr_agent import HSRAgent

def load_foundation_graph(path):
    print(f"Loading Foundation Graph from {path}...")
    with np.load(path, allow_pickle=True) as f:
        edge_index = f['edge_index']
        if 'edge_attr_summary' in f:
            summary = f['edge_attr_summary']
            p_forward = summary[:, 0]
            median_stt = summary[:, 1]
            min_stt = summary[:, 3] # Ch3 is min_stt
        else:
            E = edge_index.shape[1]
            p_forward = np.ones(E)
            median_stt = np.ones(E) * 0.02
            min_stt = np.ones(E) * 0.01
            
    return {
        'edge_index': edge_index,
        'p_forward': p_forward,
        'median_stt': median_stt,
        'min_stt': min_stt
    }

def evaluate_hsr(data_dir, foundation_path, budget=20, limit=5000):
    # 1. Init Agent
    graph_data = load_foundation_graph(foundation_path)
    # Pass min_stt to agent for relaxed pruning
    agent = HSRAgent(graph_data=graph_data, time_step_hours=0.5, tolerance_hours=24.0)
    agent.min_stts = graph_data['min_stt']
    agent._build_reverse_graph() # Rebuild with min_stt
    
    # 2. Scan Dataset
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if limit:
        files = files[:limit]
        
    print(f"Evaluating on {len(files)} samples...")
    
    metrics = {
        'success': 0,
        'exhaustion': 0,
        'samples_at_success': [],
        'rounds_at_success': [],
        'time_at_success_hours': [],
        'source_pruned_count': 0,
        'empty_candidate_count': 0,
    }
    
    for fpath in tqdm(files):
        try:
            with np.load(fpath, allow_pickle=True) as sample:
                # 3. Setup Episode
                # Nodes
                if 'global_node_indices' in sample:
                    global_indices = sample['global_node_indices']
                elif 'node_indices' in sample:
                    global_indices = sample['node_indices']
                else:
                    continue
                
                # True Source
                if 'y' in sample:
                    y = sample['y']
                    if y.ndim == 1 and len(y) == len(global_indices):
                         source_mask = y > 0.5
                         if source_mask.sum() == 0:
                             continue
                         local_source_idx = np.where(source_mask)[0][0]
                         true_source_global = global_indices[local_source_idx]
                    else:
                        continue
                else:
                    continue
                
                # Initial Trigger (Anchor)
                trigger_node = None
                if 'trigger_node_index' in sample:
                    trigger_node = int(sample['trigger_node_index'])
                elif 'global_trigger_node' in sample:
                    trigger_node = int(sample['global_trigger_node'])
                
                # Trigger info for simulation start time
                if 'trigger_time_step' in sample:
                    t_start = int(sample['trigger_time_step'])
                elif 'global_trigger_time' in sample:
                    t_start = int(sample['global_trigger_time'])
                else:
                    t_start = 0 

                agent.reset(candidates=global_indices, trigger_node=trigger_node, t_start=t_start)
                
                # Simulation Loop
                hit = False
                total_samples_used = 0
                rounds_count = 0
                
                # Pre-load data for simulation
                x_data = sample['x'] if 'x' in sample else sample['data']
                
                # Robust Shape Handling
                shape = x_data.shape
                if shape[0] == 128:
                    if shape[1] == 2: pass
                    elif shape[2] == 2: x_data = x_data.transpose(0, 2, 1)
                    else: continue
                elif shape[1] == 128:
                    if shape[0] == 2: x_data = x_data.transpose(1, 0, 2)
                    elif shape[2] == 2: x_data = x_data.transpose(1, 2, 0)
                    else: continue
                else:
                    if shape[2] == 128 and shape[1] == 2: x_data = x_data.transpose(2, 1, 0)
                    else: continue
                
                # Map global ID -> local index for data lookup
                global_to_local = {gid: i for i, gid in enumerate(global_indices)}
                
                # Each "step" in simulation uses up to 3 samplers (HSR-Scalable)
                max_rounds = (budget + 2) // 3
                for round_idx in range(max_rounds):
                    rounds_count += 1
                    
                    # 1. Select Nodes (Top-3)
                    actions_global = agent.get_action_hsr_scalable(k=3)
                    
                    observations = {}
                    
                    for action_global in actions_global:
                        total_samples_used += 1
                        if action_global == true_source_global:
                            hit = True
                            break
                        
                        t_curr_idx = min(t_start + round_idx, 127)
                        if action_global in global_to_local:
                            loc_idx = global_to_local[action_global]
                            val = x_data[t_curr_idx, 1, loc_idx]
                            label = 1 if val > 1e-6 else 0
                            observations[action_global] = (val, label)
                    
                    if hit: break
                        
                    if observations:
                        agent.step(observations)
                    
                    # Diagnostics: Is the true source still in candidate_set?
                    if true_source_global not in agent.candidate_set:
                        metrics['source_pruned_count'] += 1
                        break # Source is gone, will never find it
                    
                    if not agent.candidate_set:
                        metrics['empty_candidate_count'] += 1
                        break

                    if total_samples_used >= budget: break
                
                # Record
                if hit:
                    metrics['success'] += 1
                    metrics['samples_at_success'].append(total_samples_used)
                    metrics['rounds_at_success'].append(rounds_count)
                    metrics['time_at_success_hours'].append(round_idx * agent.time_step_hours)
                else:
                    metrics['exhaustion'] += 1
                    
        except Exception as e:
            continue

    # Summary
    n_processed = metrics['success'] + metrics['exhaustion']
    if n_processed == 0:
        print("No samples processed.")
        return

    success_rate = metrics['success'] / n_processed
    avg_samples = np.mean(metrics['samples_at_success']) if metrics['samples_at_success'] else 0
    avg_rounds = np.mean(metrics['rounds_at_success']) if metrics['rounds_at_success'] else 0
    avg_time_hours = np.mean(metrics['time_at_success_hours']) if metrics['time_at_success_hours'] else 0
    
    print("\n" + "="*40)
    print(f"HSR-Scalable Evaluation (Phase 4.5 Metrics)")
    print(f"Dataset: {os.path.basename(data_dir)}")
    print(f"Samples: {n_processed}")
    print("-" * 40)
    print(f"Success Rate (Hit@1): {success_rate:.4f}")
    print(f"Avg Samples Used:     {avg_samples:.2f}")
    print(f"Avg Rounds:           {avg_rounds:.2f}")
    print(f"Avg Time to Hit:      {avg_time_hours*60:.2f} min")
    print(f"Exhaustion Rate:      {1-success_rate:.4f}")
    print(f"Source Pruned Rate:   {metrics['source_pruned_count']/n_processed:.4f}")
    print(f"Empty Set Rate:       {metrics['empty_candidate_count']/n_processed:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Config
    DATA_DIR = '/root/autodl-tmp/rl_spim_v7_mainline/datanew/production_data/foundation_20260114_164946_86d5023e/subgraph_v11_prod'
    FOUNDATION_PATH = '/root/autodl-tmp/rl_spim_v7_mainline/datanew/production_data/foundation_20260114_164946_86d5023e/graph.npz'
    
    # Run with standard budget (20) and large sample size (5000)
    print("Running HSR-Scalable Evaluation (Budget=20, K=3, T=30min, N=5000)...")
    evaluate_hsr(DATA_DIR, FOUNDATION_PATH, budget=20, limit=5000)
