
import torch
import torch.nn.functional as F
import numpy as np
from src.modeling.evidence.contradiction_oracle_v1 import OracleHistorySample, OracleHistoryStep
from src.modeling.evidence.dynamic_reachability import DynamicReachabilityRuleModule
from src.modeling.state.schema import ObservationState, PhysicsContext
from torch_geometric.utils import subgraph

class PracticalRollout:
    """
    Manages the practical rollout protocol for evidence auditing.
    """
    def __init__(self, event_data, global_edge_index, stt_dynamic_series, num_global_nodes, num_episodes=10, samples_per_episode=3, episode_duration_min=45):
        self.event_data = event_data
        self.global_edge_index = global_edge_index
        self.stt_dynamic_series = stt_dynamic_series
        self.num_global_nodes = num_global_nodes
        self.num_episodes = num_episodes
        self.samples_per_episode = samples_per_episode
        self.episode_duration_min = episode_duration_min
        
        self.num_nodes = event_data.num_nodes
        self.device = event_data.x.device
        
        # Initialize State
        self.revealed_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        self.current_episode = 0
        self.current_time_min = 0.0
        self.history_steps = []
        
        # Physics Engine
        self.reachability_module = DynamicReachabilityRuleModule()
        
        # Pre-compute Edge Mask for STT slicing
        self.g_ids = event_data.n_id
        # Use subgraph to get mask for physical edges
        # Note: dataset.edge_index might contain virtual edges at the end.
        # We assume the first N_phys edges correspond to physical edges in global graph.
        # To be safe, we re-run subgraph logic.
        self.sub_edge_index, _, self.edge_mask = subgraph(
            self.g_ids, 
            self.global_edge_index, 
            relabel_nodes=True, 
            num_nodes=self.num_global_nodes, # Fixed: Global num nodes
            return_edge_mask=True
        )
        # Note: subgraph relabels nodes to 0..N_sub-1.
        # The edge_mask corresponds to edges in global_edge_index that connect nodes in g_ids.
        
    def _resolve_snapshot_index(self):
        step_seconds = getattr(self.event_data, 'step_seconds', 900)
        step_min = step_seconds / 60.0
        trigger_time_step = getattr(self.event_data, 'trigger_time_step', 0)
        if isinstance(trigger_time_step, torch.Tensor):
            trigger_time_step = trigger_time_step.item()
        current_step_rel = int(self.current_time_min / step_min)
        t_snapshot_idx = trigger_time_step + current_step_rel
        x_raw = getattr(self.event_data, 'x_raw', None)
        if x_raw is not None:
            max_t = x_raw.shape[1] - 1
            if t_snapshot_idx > max_t:
                t_snapshot_idx = max_t
        return int(t_snapshot_idx)

    def _resolve_absolute_snapshot_index(self, t_snapshot_idx: int) -> int:
        x_raw = getattr(self.event_data, "x_raw", None)
        global_start = getattr(self.event_data, "global_start_step", 96)
        if isinstance(global_start, torch.Tensor):
            global_start = global_start.item()
        is_v11 = bool(x_raw is not None and x_raw.shape[1] > 200)
        scale_factor = 1 if is_v11 else 3
        t_abs_idx = int(global_start) + (int(t_snapshot_idx) // scale_factor)
        return max(0, min(int(self.stt_dynamic_series.shape[0]) - 1, int(t_abs_idx)))

    def _build_physics_context(self, t_snapshot_idx: int) -> PhysicsContext:
        t_abs_idx = self._resolve_absolute_snapshot_index(t_snapshot_idx)

        stt_slice_global = self.stt_dynamic_series[t_abs_idx]
        stt_slice_sub = stt_slice_global[self.edge_mask]

        num_physical = stt_slice_sub.shape[0]
        num_total = self.event_data.edge_index.shape[1]
        if num_total > num_physical:
            padding = torch.full((num_total - num_physical,), 0.1, device=self.device)
            stt_dynamic = torch.cat([stt_slice_sub, padding], dim=0).unsqueeze(1)
        else:
            stt_dynamic = stt_slice_sub.unsqueeze(1)

        return PhysicsContext(
            edge_index=self.event_data.edge_index,
            edge_attr=self.event_data.edge_attr,
            stt_dynamic=stt_dynamic,
            batch=torch.zeros(self.num_nodes, dtype=torch.long, device=self.device),
            stt=torch.zeros(self.event_data.edge_index.size(1), device=self.device),
            direction=torch.zeros(self.event_data.edge_index.size(1), device=self.device),
            feasible_mask=torch.ones(self.num_nodes, dtype=torch.bool, device=self.device),
        )

    def _build_history_samples(self, sampled_nodes, x_raw, t_snapshot_idx: int):
        if not sampled_nodes:
            return []
        conc = x_raw[:, t_snapshot_idx, 1]
        signal = x_raw[:, t_snapshot_idx, 0]
        history_samples = []
        for node_idx, sample_type in sampled_nodes:
            node_idx = int(node_idx)
            concentration = float(conc[node_idx].item())
            history_samples.append(
                OracleHistorySample(
                    local_idx=node_idx,
                    global_idx=int(self.g_ids[node_idx].item()),
                    time_min=float(self.current_time_min),
                    t_snapshot_idx=int(t_snapshot_idx),
                    is_positive=bool(concentration > 0.1),
                    is_safe=bool(concentration <= 0.1),
                    sample_type=str(sample_type),
                    concentration=concentration,
                    signal=float(signal[node_idx].item()),
                )
            )
        return history_samples

    def observe_current_state(self):
        """
        Return the current partial/oracle observation state and physics context
        without advancing time or mutating the rollout.
        """
        t_snapshot_idx = self._resolve_snapshot_index()
        x_raw = getattr(self.event_data, "x_raw", None)
        obs_state_partial = self._build_observation_state(self.revealed_mask, x_raw, t_snapshot_idx)
        oracle_mask = torch.ones_like(self.revealed_mask)
        obs_state_oracle = self._build_observation_state(oracle_mask, x_raw, t_snapshot_idx)
        physics_context = self._build_physics_context(t_snapshot_idx)
        info = {
            "episode": self.current_episode,
            "time_min": self.current_time_min,
            "revealed_count": int(self.revealed_mask.sum().item()),
            "t_snapshot_idx": int(t_snapshot_idx),
            "absolute_snapshot_idx": int(self._resolve_absolute_snapshot_index(t_snapshot_idx)),
        }
        return obs_state_partial, obs_state_oracle, physics_context, info

    def _apply_episode_samples(self, sampled_nodes, t_snapshot_idx: int):
        x_raw = getattr(self.event_data, "x_raw", None)
        for node_idx, _sample_type in sampled_nodes:
            self.revealed_mask[int(node_idx)] = True

        obs_state_partial = self._build_observation_state(self.revealed_mask, x_raw, t_snapshot_idx)
        oracle_mask = torch.ones_like(self.revealed_mask)
        obs_state_oracle = self._build_observation_state(oracle_mask, x_raw, t_snapshot_idx)
        physics_context = self._build_physics_context(t_snapshot_idx)
        absolute_snapshot_idx = self._resolve_absolute_snapshot_index(t_snapshot_idx)
        history_samples = self._build_history_samples(sampled_nodes, x_raw, t_snapshot_idx)
        if history_samples:
            self.history_steps.append(
                OracleHistoryStep(
                    episode=int(self.current_episode),
                    time_min=float(self.current_time_min),
                    t_snapshot_idx=int(t_snapshot_idx),
                    absolute_snapshot_idx=int(absolute_snapshot_idx),
                    phys_ctx=physics_context,
                    samples=history_samples,
                )
            )

        info = {
            "episode": self.current_episode,
            "time_min": self.current_time_min,
            "samples": [int(node_idx) for node_idx, _ in sampled_nodes],
            "types": [str(sample_type) for _, sample_type in sampled_nodes],
            "revealed_count": int(self.revealed_mask.sum().item()),
            "t_snapshot_idx": int(t_snapshot_idx),
            "absolute_snapshot_idx": int(absolute_snapshot_idx),
        }
        return obs_state_partial, obs_state_oracle, physics_context, info

    def step_with_actions(self, action_nodes, sample_types=None):
        """
        Advance one episode using an explicit size-k action set chosen from the
        frozen pre-episode state.
        """
        self.current_episode += 1
        self.current_time_min += self.episode_duration_min
        t_snapshot_idx = self._resolve_snapshot_index()

        sample_types = list(sample_types) if sample_types is not None else []
        sanitized = []
        seen = set()
        for idx, node in enumerate(action_nodes):
            node_idx = int(node)
            if node_idx < 0 or node_idx >= self.num_nodes:
                continue
            if node_idx in seen or bool(self.revealed_mask[node_idx].item()):
                continue
            seen.add(node_idx)
            sample_type = sample_types[idx] if idx < len(sample_types) else f"slot_{idx}"
            sanitized.append((node_idx, str(sample_type)))

        return self._apply_episode_samples(sanitized, t_snapshot_idx)

    def step(self):
        """
        Advance one episode:
        1. Update Time
        2. Sample Nodes (Toxic, Safe, Random)
        3. Reveal Observations
        4. Return Current Physics & Observation State
        """
        self.current_episode += 1
        self.current_time_min += self.episode_duration_min

        t_snapshot_idx = self._resolve_snapshot_index()
        x_raw = getattr(self.event_data, 'x_raw', None)
        conc = x_raw[:, t_snapshot_idx, 1]

        toxic_pool = (conc > 0.1).nonzero(as_tuple=True)[0]
        safe_pool = (conc <= 0.1).nonzero(as_tuple=True)[0]
        planning_mask = self.revealed_mask.clone()

        sampled_nodes = []

        cand_toxic = [n.item() for n in toxic_pool if not planning_mask[n]]
        if cand_toxic:
            s_toxic = np.random.choice(cand_toxic)
            sample_type_A = "toxic"
        else:
            # Fallback: Random Unobserved
            cand_random = (~self.revealed_mask).nonzero(as_tuple=True)[0].tolist()
            if cand_random:
                s_toxic = np.random.choice(cand_random)
                sample_type_A = "fallback_random"
            else:
                s_toxic = None
                sample_type_A = "none"
        if s_toxic is not None:
            sampled_nodes.append((int(s_toxic), sample_type_A))
            planning_mask[int(s_toxic)] = True

        cand_safe = [n.item() for n in safe_pool if not planning_mask[n]]
        if cand_safe:
            s_safe = np.random.choice(cand_safe)
            sample_type_B = "safe"
        else:
            # Fallback: Random Unobserved
            cand_random = (~self.revealed_mask).nonzero(as_tuple=True)[0].tolist()
            if cand_random:
                s_safe = np.random.choice(cand_random)
                sample_type_B = "fallback_random"
            else:
                s_safe = None
                sample_type_B = "none"
        if s_safe is not None:
            sampled_nodes.append((int(s_safe), sample_type_B))
            planning_mask[int(s_safe)] = True

        cand_random = (~planning_mask).nonzero(as_tuple=True)[0].tolist()
        if cand_random:
            s_random = np.random.choice(cand_random)
            sample_type_C = "random"
            sampled_nodes.append((int(s_random), sample_type_C))
        else:
            sample_type_C = "none"
        return self._apply_episode_samples(sampled_nodes, t_snapshot_idx)

    def _build_observation_state(self, mask, x_raw, t_idx):
        """
        Construct ObservationState from mask and raw data at t_idx.
        """
        # 1. Observed Flag
        observed_flag = mask.float()
        
        # 2. Toxic Positive (Observed & Conc > 0.1)
        conc = x_raw[:, t_idx, 1]
        is_toxic = (conc > 0.1).float()
        toxic_positive = observed_flag * is_toxic
        
        # 3. Toxic Negative (Observed & Conc <= 0.1)
        # BUG FIX: Remove legacy is_sensor dependency.
        # Any observed node that is NOT toxic is a negative evidence.
        is_safe = (conc <= 0.1).float()
        toxic_negative = observed_flag * is_safe

        
        # 4. Chlorine Deviation
        # Ch0 is signal deviation.
        signal = x_raw[:, t_idx, 0]
        chlorine_deviation = observed_flag * signal
        
        # 5. Freshness
        freshness = observed_flag.clone() # 1.0 for observed, 0.0 for unobserved
        
        return ObservationState(
            observed_flag=observed_flag,
            toxic_positive_flag=toxic_positive,
            toxic_negative_flag=toxic_negative,
            chlorine_deviation=chlorine_deviation,
            freshness=freshness
        )

def compute_audit_support(reach_module, physics_context, obs_state, t_sim=None):
    """
    Implements the Practical Evidence Audit Support Semantic.
    support(s) = mean( u_{s,i} ) over positive i
    u_{s,i} = r_s^alpha * h_s^beta
    """
    # 1. Get Strict Dynamic Reachability & Distances
    # We use compute_reachability to get the full reachability map
    # But we need Distances T(s, i) for Ranking.
    # reach_module.compute_reachability returns 'dist_soft_pos' which is min_i(T(s, i)).
    # But we need T(s, i) for EACH i to calculate Rank per i.
    # So we must iterate over positive observations.
    
    pos_indices = obs_state.toxic_positive_flag.nonzero().squeeze(1)
    num_nodes = obs_state.observed_flag.size(0)
    device = obs_state.observed_flag.device
    
    if pos_indices.numel() == 0:
        return torch.zeros(num_nodes, device=device), {}
        
    # Weights for distances
    stt_val = physics_context.stt_dynamic.view(-1).abs()
    
    # Pre-compute Reverse Graph for efficiency (SciPy)
    adj_rev = reach_module._build_scipy_reverse_graph(
        physics_context.edge_index, 
        physics_context.stt_dynamic.view(-1), 
        num_nodes
    )
    
    support_scores = torch.zeros(num_nodes, device=device)
    
    # Store sub-metrics for audit
    metric_r = torch.zeros(num_nodes, device=device)
    metric_h = torch.zeros(num_nodes, device=device)
    
    # Limit max positives to avoid timeout
    MAX_POS = 20
    if pos_indices.numel() > MAX_POS:
        pos_indices = pos_indices[:MAX_POS]
        
    for idx in pos_indices:
        # 1. Compute Distances T(s, i) for this specific i
        # Use SciPy Dijkstra from single source (i) in reverse graph
        idx_np = np.array([idx.item()])
        dist_matrix = reach_module._run_scipy_dijkstra(adj_rev, idx_np) # [1, N]
        dists = torch.from_numpy(dist_matrix[0]).float().to(device)
        
        # Mask strict reachable (Time < Infinity)
        # Assuming Strict Reachability means "Physically Connected" (d < Inf)
        # AND "Temporally Feasible" (d <= t_sim)?
        # The prompt says "strict dynamic reachability gate".
        # If t_sim is provided, we should gate by t_sim.
        # But T(s, i) is travel time. Feasible if T(s, i) <= current_time?
        # Or if "path exists" (Topo)?
        # Usually Support checks if "Could have arrived".
        # Let's use T(s, i) < Inf as the base Strict Reachability.
        # And if t_sim is used, maybe gate T(s, i) <= t_sim + Buffer?
        # Prompt: "strict reachability gate: if s對i strict dynamic unreachable -> 0"
        # Let's use Topo Reachable (d < Inf) for now, as t_sim gate is implicit in rank?
        # Actually, if T > t_sim, it's physically impossible.
        # So we should gate by T <= t_sim + 60 (Buffer).
        
        if t_sim is not None:
            # t_sim is tensor [1] usually
            t_val = t_sim.mean().item()
            is_reachable = (dists <= t_val + 60.0)
        else:
            is_reachable = (dists < 1e9)
            
        reachable_indices = is_reachable.nonzero().squeeze(1)
        
        if reachable_indices.numel() == 0:
            continue
            
        # 2. Relative Arrival Rank (r_s)
        # Sort candidates by T(s, i)
        dists_reachable = dists[reachable_indices]
        # sort
        sorted_dists, sorted_idx = torch.sort(dists_reachable)
        # Rank: 1-based.
        # ranks = 1 + arange
        # But we need rank for each s.
        # Map back to original indices.
        
        # Rank Calculation (Normalized)
        # r_s = 1 - (rank - 1) / (|U| - 1 + eps)
        U_size = reachable_indices.numel()
        ranks = torch.arange(1, U_size + 1, device=device).float()
        
        if U_size > 1:
            r_vals = 1.0 - (ranks - 1.0) / (U_size - 1.0 + 1e-6)
        else:
            r_vals = torch.ones(1, device=device)
            
        # Assign r_vals to nodes
        r_map = torch.zeros(num_nodes, device=device)
        node_indices = reachable_indices[sorted_idx] # sorted nodes
        r_map[node_indices] = r_vals
        
        # 3. Trunk/Hub Penalty (h_s)
        # a_s = Count of strict reachable candidates upstream of s
        # In Reverse Graph (Obs->Src), this is "Descendants of s".
        # We need to count descendants in the tree formed by Dijkstra?
        # Or in the DAG of all reachable paths?
        # Dijkstra gives a Shortest Path Tree (Predecessors).
        # Using the SP Tree is a good approximation and efficient.
        
        # Re-run Dijkstra with return_predecessors=True
        from scipy.sparse.csgraph import dijkstra
        _, preds = dijkstra(adj_rev, indices=idx_np, directed=True, return_predecessors=True)
        preds = preds[0] # [N]
        
        # Build Tree from preds
        # preds[v] = u means u -> v in Dijkstra tree (Obs -> Src)
        # Root is idx. preds[idx] = -9999
        
        # We want to count descendants for each node in the reachable set.
        # Convert preds to children list
        children = [[] for _ in range(num_nodes)]
        # Filter to reachable nodes only
        reachable_set = set(reachable_indices.cpu().numpy().tolist())
        
        # Build tree structure
        # Note: Scipy preds returns -9999 for unreachable or root
        preds_cpu = preds
        
        # Iterate over reachable nodes to build children map
        # Optimization: Only process reachable nodes
        sorted_nodes_cpu = node_indices.cpu().numpy()
        
        for u in sorted_nodes_cpu:
            if u == idx.item(): continue
            p = preds_cpu[u]
            if p != -9999 and p in reachable_set:
                children[p].append(u)
                
        # Count descendants (Memoized DFS)
        descendant_counts = {}
        
        def get_descendants(u):
            if u in descendant_counts: return descendant_counts[u]
            count = 0
            for v in children[u]:
                count += 1 + get_descendants(v)
            descendant_counts[u] = count
            return count
            
        # Compute for all reachable
        h_vals = torch.zeros(num_nodes, device=device)
        
        # We need a_s for s in reachable
        # a_s = get_descendants(s)
        # h_s = 1 / (1 + log(1 + a_s))
        
        # Calculate for all reachable nodes
        # We can iterate since len(reachable) is usually < 1000
        for s in sorted_nodes_cpu:
            a_s = get_descendants(s)
            h_s = 1.0 / (1.0 + np.log1p(a_s))
            h_vals[s] = h_s
            
        # 4. Local Support u_{s, i}
        # u = r^alpha * h^beta
        alpha = 1.0
        beta = 1.0
        
        u_vals = (r_map ** alpha) * (h_vals ** beta)
        
        # Accumulate
        support_scores += u_vals
        metric_r += r_map
        metric_h += h_vals
        
    # Mean over positives
    support_scores = support_scores / (pos_indices.numel() + 1e-6)
    metric_r = metric_r / (pos_indices.numel() + 1e-6)
    metric_h = metric_h / (pos_indices.numel() + 1e-6)
    
    return support_scores, {"r": metric_r, "h": metric_h}
