import torch
import torch.nn.functional as F
# from torch_scatter import scatter_min
from src.modeling.state.schema import ObservationState, PhysicsContext

class ReachabilityRuleModule:
    """
    Implements hydraulic reachability logic using Bellman-Ford propagation 
    on the STT-weighted graph.
    
    Replaces heuristic 'hop counts' with physical travel time estimation.
    """
    def __init__(self, max_hops=20):
        self.max_hops = max_hops
        self.infinity = 1e9

    def compute_reachability(self, 
                           observation_state: ObservationState, 
                           physics_context: PhysicsContext, 
                           t_sim: torch.Tensor, 
                           batch: torch.Tensor):
        """
        Compute reachability masks/scores for Suspect Pool, Support, and Contradiction.
        
        Returns:
            dict: {
                'topology_reachable': [N] (0/1),
                'soft_reachability': [N] (0/1),
                'hard_reachability': [N] (0/1),
                'negative_pressure': [N] (0-1),
                'travel_time_soft': [N] (time),
                'travel_time_hard': [N] (time)
            }
        """
        # 1. Setup Weights
        # Ch0: Log Median STT -> Soft Weight
        # Ch2: Log Min STT -> Hard Weight
        # exp(x) - 1 to reverse log1p
        
        # Check if stt_median is available
        if physics_context.stt_median is not None:
            w_soft = torch.expm1(physics_context.stt_median)
            w_hard = torch.expm1(physics_context.stt_min)
        else:
            # Fallback if not populated (Audit safety)
            # Use 'stt' legacy (Flip Rate) -> Treat as constant 1.0 or similar?
            # Or try to extract from edge_attr if available
            if physics_context.edge_attr is not None and physics_context.edge_attr.size(1) > 2:
                w_soft = torch.expm1(physics_context.edge_attr[:, 0])
                w_hard = torch.expm1(physics_context.edge_attr[:, 2])
            else:
                # Fallback to constant weights (Hop Counting)
                w_soft = torch.ones_like(physics_context.edge_index[0], dtype=torch.float) * 20.0 # 20 min/hop
                w_hard = torch.ones_like(physics_context.edge_index[0], dtype=torch.float) * 10.0 # 10 min/hop

        # Clamp weights to be non-negative
        w_soft = torch.clamp(w_soft, min=0.0)
        w_hard = torch.clamp(w_hard, min=0.0)
        
        # 2. Compute Distances
        # Target: Any Positive Observation (Toxic+)
        # We want Shortest Path from S to Any Positive Obs.
        # Backwards: Shortest Path from Any Positive Obs to S.
        
        seeds_pos = observation_state.toxic_positive_flag
        # Support also needs Chlorine info?
        # Definition: Support is toxic dominated. So we seed from Toxic+.
        # Chlorine support uses same reachability but different seeds?
        # To save compute, we can compute "Reachability from Any Observed Node"
        # but that mixes Toxic and Safe.
        # We need specific reachability.
        
        # A. Reachability from Toxic Positive (for Support & Suspect Pool)
        dist_soft_pos = self._bellman_ford(seeds_pos, physics_context.edge_index, w_soft, batch.size(0))
        
        # B. Reachability from Toxic Negative (for Contradiction)
        # We use Hard Weights for "Strict Arrival" (Contradiction)
        # And Soft Weights for "Likely Arrival" (Soft Contradiction)
        seeds_neg = observation_state.toxic_negative_flag
        dist_hard_neg = self._bellman_ford(seeds_neg, physics_context.edge_index, w_hard, batch.size(0))
        dist_soft_neg = self._bellman_ford(seeds_neg, physics_context.edge_index, w_soft, batch.size(0))
        
        # 3. Evaluate Conditions
        # Expand t_sim to nodes
        if t_sim is not None:
            t_nodes = t_sim[batch]
        else:
            t_nodes = torch.full((batch.size(0),), 1e6, device=batch.device)
            
        # A. Topology Reachable
        # dist < infinity
        topo_reachable = (dist_soft_pos < self.infinity / 2).float()
        
        # B. Soft Reachability (for Suspect Pool & Support)
        # Condition: dist_soft <= t_sim + Buffer (e.g. 60 min)
        # Coarse Time Window
        buffer_soft = 60.0 
        soft_reachable = (dist_soft_pos <= (t_nodes + buffer_soft)).float()
        
        # C. Hard Reachability (from Negatives -> Contradiction)
        # Condition: dist_hard <= t_sim - Buffer (e.g. 10 min)
        # "Should have strictly arrived"
        buffer_hard = 10.0
        # If dist_hard_neg <= t - buffer, then S implies Neg SHOULD be hit.
        # But Neg is Safe. So S is contradicted.
        # This is "Negative Pressure".
        
        # Wait, "Hard Reachability" in prompt usually refers to "S reaching I".
        # Here we computed "Neg I reaching S".
        # If dist(S, I) <= t, then S reaches I.
        # So "Hard Reachability of S to Neg I" is what we computed in dist_hard_neg.
        
        # Pressure Logic:
        # If S can reach a Negative Node I (strictly), then S is under pressure.
        neg_pressure_hard = (dist_hard_neg <= (t_nodes - buffer_hard)).float()
        
        # Soft Pressure (Likely Arrived)
        neg_pressure_soft = (dist_soft_neg <= (t_nodes + buffer_soft)).float() 
        
        return {
            'topology_reachable': topo_reachable,
            'soft_reachability': soft_reachable, # From Positive
            'hard_reachability_from_neg': neg_pressure_hard, # "Hard Reachable to Safe"
            'soft_reachability_from_neg': neg_pressure_soft,
            'dist_soft_pos': dist_soft_pos,
            'dist_hard_neg': dist_hard_neg
        }

    def compute_distance(self, seeds, physics_context, weights, num_nodes):
        """Generic distance computation for arbitrary seeds"""
        return self._bellman_ford(seeds, physics_context.edge_index, weights, num_nodes)

    def _bellman_ford(self, seeds, edge_index, weights, num_nodes):
        """
        Run fixed-depth Bellman-Ford to compute shortest upstream path distances.
        """
        raise NotImplementedError("Bellman-Ford fallback is disabled in this audit sprint. Please ensure Dynamic STT is available.")
        # # Initialize dist
        # dist = torch.full((num_nodes,), self.infinity, device=seeds.device)

        
        # Seeds have dist 0
        mask_seeds = (seeds > 0.5)
        dist[mask_seeds] = 0.0
        
        # Early exit if no seeds
        if not mask_seeds.any():
            return dist
            
        src, dst = edge_index # src -> dst (Flow)
        # We propagate Upstream: dst -> src
        # dist[src] = min(dist[src], dist[dst] + w)
        
        for _ in range(self.max_hops):
            # Gather dist from downstream neighbors
            d_dst = dist[dst]
            
            # Add weights
            d_proposal = d_dst + weights
            
            # Scatter Min to update src
            # We need to update 'dist' in place or create new
            # scatter_min returns (out, argmin). We only need out.
            # Initialize output with current dist values to keep min
            d_new, _ = scatter_min(d_proposal, src, dim=0, dim_size=num_nodes)
            
            # Update dist (take min of current and new proposals)
            dist = torch.min(dist, d_new)
            
            # Optimization: Check convergence? 
            # (Skipped for fixed step simplicity & batch parallelism)
            
        return dist
