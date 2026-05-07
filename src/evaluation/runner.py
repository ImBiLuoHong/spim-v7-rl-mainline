import torch
import numpy as np
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_scipy_sparse_matrix
from src.evaluation.evaluator import Evaluator

def compute_geodesic_distances(batch):
    """
    Computes geodesic distance from anchor (Ch4) to target (y=1) for each graph in the batch.
    Returns a tensor of distances [B], with -1.0 for unreachable/invalid graphs.
    """
    batch_size = batch.num_graphs
    dists = torch.full((batch_size,), -1.0)
    
    try:
        # Ch4 is Anchor (1=Trigger, -1=Negative Trigger). Use ABS to capture both.
        start_mask = (torch.abs(batch.x[:, 4]) > 0.5)
        # [Runner DEBUG]
        # print(f"[Runner] Computing Dists: Start Mask Sum={start_mask.sum().item()}, Total Nodes={len(batch.x)}")
        start_indices = start_mask.nonzero().view(-1).cpu().numpy()
        
        # Targets (y=1)
        target_mask = (batch.y > 0.5)
        target_indices = target_mask.nonzero().view(-1).cpu().numpy()
        
        if len(start_indices) > 0 and len(target_indices) > 0:
            adj = to_scipy_sparse_matrix(batch.edge_index, num_nodes=batch.num_nodes)
            # Compute from all starts to all nodes (unweighted)
            dist_matrix = shortest_path(adj, directed=False, unweighted=True, indices=start_indices)
            # dist_matrix shape: [n_starts, num_nodes]
            
            batch_ids = batch.batch.cpu().numpy()
            
            # Map graph_idx -> start_row_idx
            graph_to_start_row = {}
            for i, s_idx in enumerate(start_indices):
                g_id = batch_ids[s_idx]
                if g_id not in graph_to_start_row:
                    graph_to_start_row[g_id] = i
                    
            # For each target, look up dist
            for t_idx in target_indices:
                g_id = batch_ids[t_idx]
                if g_id in graph_to_start_row:
                    row_idx = graph_to_start_row[g_id]
                    d = dist_matrix[row_idx, t_idx]
                    if not np.isinf(d):
                        dists[g_id] = float(d)
        
        # Identify Hard Samples (Geodesic Dist > 3) [Lowered from 6]
        # And ensure it's reachable (dist != -1)
        # We store dists in batch for analysis
    except Exception as e:
        # Fallback or keep -1.0
        pass
        
    return dists

def evaluate_mode(model, loader, mode='standard', device='cuda'):
    """
    Runs evaluation loop (Phase 4.5 End-to-End).
    Decoupled from training script.
    """
    model.eval()
    evaluator = Evaluator(model.cfg)
    
    # [SSOT] Load max_eval_episodes from config
    max_episodes = 10 # Default to 10 as per user requirement (max_episodes)
    if hasattr(model.cfg, 'training'):
        # If config has explicit max_eval_episodes, use it, otherwise default to 10
        max_episodes = getattr(model.cfg.training, 'max_eval_episodes', 10)
        if max_episodes <= 0: max_episodes = 10
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # 1. Run Inference (Closed-Loop)
            # Call run_scenario instead of run_episode
            out = model(batch, inference_mode=True, max_episodes=max_episodes, return_trajectory=True)
            
            metrics = out['step_metrics']
            trajectory = out.get('trajectory', [])
            
            # Move to CPU in bulk with safety defaults
            raw_success = metrics.get('raw_success', torch.zeros(batch.num_graphs)).cpu() 
            raw_predict_hit = metrics.get('raw_predict_hit', torch.zeros(batch.num_graphs)).cpu()
            raw_predict_hit_valid = metrics.get('raw_predict_hit_valid', torch.zeros(batch.num_graphs)).cpu()
            raw_steps = metrics.get('raw_steps', torch.zeros(batch.num_graphs)).cpu()
            raw_budget = metrics.get('raw_budget', torch.zeros(batch.num_graphs)).cpu()
            raw_rounds = metrics.get('raw_rounds', torch.zeros(batch.num_graphs)).cpu()
            raw_max_hit_prob = metrics.get('raw_max_hit_prob', torch.zeros(batch.num_graphs)).cpu()
            raw_core_mass_before = metrics.get('raw_core_mass_before', torch.zeros(batch.num_graphs)).cpu()
            raw_core_mass_after = metrics.get('raw_core_mass_after', torch.zeros(batch.num_graphs)).cpu()
            raw_core_mass_delta = metrics.get('raw_core_mass_delta', torch.zeros(batch.num_graphs)).cpu()
            raw_core_size_before = metrics.get('raw_core_size_before', torch.zeros(batch.num_graphs)).cpu()
            raw_core_size_after = metrics.get('raw_core_size_after', torch.zeros(batch.num_graphs)).cpu()
            raw_core_size_delta = metrics.get('raw_core_size_delta', torch.zeros(batch.num_graphs)).cpu()
            raw_uncertainty_before = metrics.get('raw_uncertainty_before', torch.zeros(batch.num_graphs)).cpu()
            raw_uncertainty_after = metrics.get('raw_uncertainty_after', torch.zeros(batch.num_graphs)).cpu()
            raw_uncertainty_collapse = metrics.get('raw_uncertainty_collapse', torch.zeros(batch.num_graphs)).cpu()
            raw_closure_success = metrics.get('raw_closure_success', torch.zeros(batch.num_graphs)).cpu()
            raw_decisive_closure = metrics.get('raw_decisive_closure', torch.zeros(batch.num_graphs)).cpu()
            raw_budget_to_closure = metrics.get('raw_budget_to_closure', raw_budget).cpu()
            raw_budget_efficiency = metrics.get('raw_budget_efficiency', torch.zeros(batch.num_graphs)).cpu()
            raw_evidence_gain = metrics.get('raw_evidence_gain_per_sample', torch.zeros(batch.num_graphs)).cpu()
            raw_harmful_drift = metrics.get('raw_harmful_drift', torch.zeros(batch.num_graphs)).cpu()
            raw_focus_core_delta = metrics.get('raw_focus_core_delta', torch.zeros(batch.num_graphs)).cpu()
            raw_wasted_budget_fraction = metrics.get('raw_wasted_budget_fraction', torch.zeros(batch.num_graphs)).cpu()
            raw_empty_selection_fraction = metrics.get('raw_empty_selection_fraction', torch.zeros(batch.num_graphs)).cpu()
            raw_terminal_budget_bonus = metrics.get('raw_terminal_budget_bonus', torch.zeros(batch.num_graphs)).cpu()
            
            # Extract Trajectory for Step-Level AUROC
            traj_probs = []
            traj_hits = []
            if trajectory:
                for t_step in trajectory:
                    traj_probs.append(t_step['hit_prob_surrogate'].cpu().numpy())
                    traj_hits.append(t_step['is_hit'].cpu().numpy())
                
                # Stack to [Steps, Batch]
                traj_probs = np.stack(traj_probs)
                traj_hits = np.stack(traj_hits)
            
            # 2. Extract Hardness (Geodesic Dist)
            dists = compute_geodesic_distances(batch)
            
            # 3. Populate Evaluator
            success_np = raw_success.numpy()
            predict_hit_np = raw_predict_hit.numpy() # Audit
            predict_hit_valid_np = raw_predict_hit_valid.numpy()
            steps_np = raw_steps.numpy()
            budget_np = raw_budget.numpy()
            rounds_np = raw_rounds.numpy() # Audit
            max_hit_prob_np = raw_max_hit_prob.numpy()
            dists_np = dists.numpy()
            
            batch_size = batch.num_graphs
            for i in range(batch_size):
                episode_data = {
                    'success': bool(success_np[i] > 0.5),
                    'predict_hit': bool(predict_hit_np[i] > 0.5), # Audit
                    'predict_hit_valid': bool(predict_hit_valid_np[i] > 0.5),
                    'physical_time_mins': float(steps_np[i]), # Was 'steps' -> Physical Time (mins)
                    'total_samples': float(budget_np[i]), # Was 'budget' -> Total Unique Samples
                    'episodes_completed': float(rounds_np[i]), # Was 'rounds' -> Episode Count
                    'max_hit_prob': float(max_hit_prob_np[i]),
                    'geodesic_dist': float(dists_np[i]),
                    'trajectory_probs': traj_probs[:, i].tolist() if len(traj_probs) > 0 else [],
                    'trajectory_hits': traj_hits[:, i].tolist() if len(traj_hits) > 0 else [],
                    'core_mass_before': float(raw_core_mass_before[i].item()),
                    'core_mass_after': float(raw_core_mass_after[i].item()),
                    'core_mass_delta': float(raw_core_mass_delta[i].item()),
                    'core_size_before': float(raw_core_size_before[i].item()),
                    'core_size_after': float(raw_core_size_after[i].item()),
                    'core_size_delta': float(raw_core_size_delta[i].item()),
                    'uncertainty_before': float(raw_uncertainty_before[i].item()),
                    'uncertainty_after': float(raw_uncertainty_after[i].item()),
                    'uncertainty_collapse': float(raw_uncertainty_collapse[i].item()),
                    'closure_success': float(raw_closure_success[i].item()),
                    'decisive_closure': float(raw_decisive_closure[i].item()),
                    'budget_to_closure': float(raw_budget_to_closure[i].item()),
                    'budget_efficiency': float(raw_budget_efficiency[i].item()),
                    'evidence_gain_per_sample': float(raw_evidence_gain[i].item()),
                    'harmful_drift': float(raw_harmful_drift[i].item()),
                    'focus_core_delta': float(raw_focus_core_delta[i].item()),
                    'wasted_budget_fraction': float(raw_wasted_budget_fraction[i].item()),
                    'empty_selection_fraction': float(raw_empty_selection_fraction[i].item()),
                    'terminal_budget_bonus': float(raw_terminal_budget_bonus[i].item()),
                }
                evaluator.update_episode(episode_data)
                
    return evaluator.summarize()
