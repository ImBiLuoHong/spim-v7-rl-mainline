import os
import time
import torch
import torch.nn.functional as F
import numpy as np
# from torch_scatter import scatter_min
from src.modeling.state.schema import ObservationState, PhysicsContext

class DynamicReachabilityRuleModule:
    """
    Implements hydraulic reachability logic using Dynamic STT (Spatio-Temporal-Transitional) weights.
    
    Key Features:
    1. Uses dynamic STT from PhysicsContext.stt_dynamic (sliced for current time).
    2. Respects flow direction:
       - STT > 0: Flow u -> v
       - STT < 0: Flow v -> u
       - STT ~ 0: Disconnected (Valve Closed / Dead Water)
    3. Computes Causal Distance: Dist(S, O) following flow.
       - Implemented as Reverse Propagation from O to S on Reversed Graph.
    """
    def __init__(self, max_hops=20):
        self.max_hops = max_hops
        self.infinity = 1e9
        self.stt_epsilon = 1e-6 # Threshold for closed edge
        self._profile_enabled = os.environ.get("EVIDENCE_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
        self._profile = {}

    def _profile_add(self, key: str, value: float):
        if not self._profile_enabled:
            return
        self._profile[key] = self._profile.get(key, 0.0) + float(value)

    def _profile_inc(self, key: str, value: float = 1.0):
        self._profile_add(key, value)

    def reset_profile(self):
        self._profile.clear()

    def get_profile(self):
        return dict(self._profile)

    def compute_reachability(self, 
                           observation_state: ObservationState, 
                           physics_context: PhysicsContext, 
                           t_sim: torch.Tensor, 
                           batch: torch.Tensor):
        """
        Compute reachability masks/scores for Suspect Pool, Support, and Contradiction.
        Using Dynamic STT and Strict SciPy Dijkstra (Baseline for Audit).
        
        Prioritizing correctness over performance for this sprint.
        """
        t_total = time.perf_counter()
        # 1. Prepare Dynamic Propagation Graph
        t_seg = time.perf_counter()
        stt_dyn = physics_context.stt_dynamic
        if stt_dyn is None:
            return self._static_fallback(observation_state, physics_context, t_sim, batch)
            
        stt_val = stt_dyn.view(-1)
        num_nodes = observation_state.observed_flag.size(0)
        self._profile_add("reachability/setup_s", time.perf_counter() - t_seg)
        
        # [Strictness Fix] Use SciPy Dijkstra Solver
        # Convert to CPU/Numpy for SciPy
        t_seg = time.perf_counter()
        seeds_pos = observation_state.toxic_positive_flag
        seeds_neg = observation_state.toxic_negative_flag
        
        # Build SciPy Graph
        adj_rev = self.get_scipy_reverse_graph(physics_context, num_nodes)
        self._profile_add("reachability/reverse_graph_lookup_s", time.perf_counter() - t_seg)
        
        # A. Reachability from Toxic Positive (for Support & Suspect Pool)
        # Dijkstra from ALL positive observations to ALL candidates
        # SciPy dijkstra with indices argument returns distances FROM those indices TO all others.
        # We want: Dist(Source, Obs).
        # In Reverse Graph: Dist(Obs, Source) = Dist(Source, Obs).
        # So we seed with Obs.
        
        t_seg = time.perf_counter()
        pos_indices = seeds_pos.nonzero().view(-1).cpu().numpy()
        neg_indices = seeds_neg.nonzero().view(-1).cpu().numpy()
        self._profile_add("reachability/seed_extract_s", time.perf_counter() - t_seg)
        
        # Compute Distances (CPU)
        # If no seeds, return infinity
        if len(pos_indices) > 0:
            # min_only=True logic: we want min distance from ANY positive observation.
            # SciPy `dijkstra(..., indices=pos_indices)` returns matrix [N_seeds, N_nodes].
            # We take min over dim 0.
            # But for large graph, [N_seeds, N_nodes] is big.
            # Optimization: If N_seeds is large, this is slow. But usually N_seeds is small (<50).
            self._profile_inc("reachability/pos_dijkstra_calls")
            t_seg = time.perf_counter()
            dist_soft_pos_np = self._run_scipy_dijkstra(adj_rev, pos_indices, min_only=True)
            self._profile_add("reachability/pos_dijkstra_s", time.perf_counter() - t_seg)
        else:
            dist_soft_pos_np = np.full(num_nodes, self.infinity)
            
        if len(neg_indices) > 0:
            self._profile_inc("reachability/neg_dijkstra_calls")
            t_seg = time.perf_counter()
            dist_hard_neg_np = self._run_scipy_dijkstra(adj_rev, neg_indices, min_only=True)
            self._profile_add("reachability/neg_dijkstra_s", time.perf_counter() - t_seg)
        else:
            dist_hard_neg_np = np.full(num_nodes, self.infinity)
            
        # Convert back to Tensor
        t_seg = time.perf_counter()
        device = seeds_pos.device
        dist_soft_pos = torch.from_numpy(dist_soft_pos_np).float().to(device)
        dist_hard_neg = torch.from_numpy(dist_hard_neg_np).float().to(device)
        self._profile_add("reachability/torch_convert_s", time.perf_counter() - t_seg)
        
        # 3. Evaluate Conditions
        t_seg = time.perf_counter()
        if t_sim is not None:
            t_nodes = t_sim[batch]
        else:
            t_nodes = torch.full((batch.size(0),), 1e6, device=device)
            
        # A. Topology Reachable
        topo_reachable = (dist_soft_pos < self.infinity / 2).float()
        
        # B. Soft Reachability
        # Condition: dist <= t_sim + Buffer
        # Use tighter buffer for dynamic? Or keep standard to allow for STT noise?
        # User prompt says: "Don't tune buffer". Keep 60.0.
        buffer_soft = 60.0 
        soft_reachable = (dist_soft_pos <= (t_nodes + buffer_soft)).float()
        
        # C. Hard Reachability (Negative Pressure)
        # Condition: dist <= t_sim - Buffer
        buffer_hard = 10.0
        neg_pressure_hard = (dist_hard_neg <= (t_nodes - buffer_hard)).float()
        
        # Soft Pressure (Likely Arrived)
        neg_pressure_soft = (dist_hard_neg <= (t_nodes + buffer_soft)).float()
        self._profile_add("reachability/gate_filter_s", time.perf_counter() - t_seg)
        self._profile_add("reachability/total_s", time.perf_counter() - t_total)
        
        return {
            'topology_reachable': topo_reachable,
            'soft_reachability': soft_reachable,
            'hard_reachability_from_neg': neg_pressure_hard,
            'soft_reachability_from_neg': neg_pressure_soft,
            'dist_soft_pos': dist_soft_pos,
            'dist_hard_neg': dist_hard_neg
        }

    def compute_reachability_bundle(
        self,
        observation_state: ObservationState,
        physics_context: PhysicsContext,
        t_sim: torch.Tensor,
        batch: torch.Tensor,
        max_pos_seeds: int | None = None,
    ):
        """
        Compute the usual reachability outputs plus the full positive-seed distance
        matrix so support scoring can reuse the same Dijkstra work instead of
        recomputing it.
        """
        t_total = time.perf_counter()
        stt_dyn = physics_context.stt_dynamic
        if stt_dyn is None:
            reach_res = self._static_fallback(observation_state, physics_context, t_sim, batch)
            num_nodes = observation_state.observed_flag.size(0)
            empty = torch.empty((num_nodes, 0), device=observation_state.observed_flag.device)
            return reach_res, empty

        num_nodes = observation_state.observed_flag.size(0)
        seeds_pos = observation_state.toxic_positive_flag
        seeds_neg = observation_state.toxic_negative_flag

        t_seg = time.perf_counter()
        adj_rev = self.get_scipy_reverse_graph(physics_context, num_nodes)
        self._profile_add("reachability_bundle/reverse_graph_lookup_s", time.perf_counter() - t_seg)

        t_seg = time.perf_counter()
        pos_indices = seeds_pos.nonzero().view(-1).cpu().numpy()
        neg_indices = seeds_neg.nonzero().view(-1).cpu().numpy()
        if max_pos_seeds is not None and len(pos_indices) > int(max_pos_seeds):
            pos_indices = pos_indices[: int(max_pos_seeds)]
        self._profile_add("reachability_bundle/seed_extract_s", time.perf_counter() - t_seg)

        if len(pos_indices) > 0:
            self._profile_inc("reachability_bundle/pos_dijkstra_calls")
            t_seg = time.perf_counter()
            pos_dist_matrix_np = self._run_scipy_dijkstra(adj_rev, pos_indices)
            self._profile_add("reachability_bundle/pos_dijkstra_s", time.perf_counter() - t_seg)
            pos_dist_matrix_np = np.asarray(pos_dist_matrix_np)
            if pos_dist_matrix_np.ndim == 1:
                pos_dist_matrix_np = pos_dist_matrix_np[None, :]
            t_seg = time.perf_counter()
            dist_soft_pos_np = pos_dist_matrix_np.min(axis=0)
            self._profile_add("reachability_bundle/pos_reduce_s", time.perf_counter() - t_seg)
        else:
            pos_dist_matrix_np = np.empty((0, num_nodes), dtype=np.float64)
            dist_soft_pos_np = np.full(num_nodes, self.infinity)

        if len(neg_indices) > 0:
            self._profile_inc("reachability_bundle/neg_dijkstra_calls")
            t_seg = time.perf_counter()
            dist_hard_neg_np = self._run_scipy_dijkstra(adj_rev, neg_indices, min_only=True)
            self._profile_add("reachability_bundle/neg_dijkstra_s", time.perf_counter() - t_seg)
        else:
            dist_hard_neg_np = np.full(num_nodes, self.infinity)

        t_seg = time.perf_counter()
        device = seeds_pos.device
        dist_soft_pos = torch.from_numpy(dist_soft_pos_np).float().to(device)
        dist_hard_neg = torch.from_numpy(dist_hard_neg_np).float().to(device)
        pos_distance_matrix = torch.from_numpy(pos_dist_matrix_np.T).float().to(device)
        self._profile_add("reachability_bundle/torch_convert_s", time.perf_counter() - t_seg)

        t_seg = time.perf_counter()
        if t_sim is not None:
            t_nodes = t_sim[batch]
        else:
            t_nodes = torch.full((batch.size(0),), 1e6, device=device)

        topo_reachable = (dist_soft_pos < self.infinity / 2).float()
        buffer_soft = 60.0
        soft_reachable = (dist_soft_pos <= (t_nodes + buffer_soft)).float()
        buffer_hard = 10.0
        neg_pressure_hard = (dist_hard_neg <= (t_nodes - buffer_hard)).float()
        neg_pressure_soft = (dist_hard_neg <= (t_nodes + buffer_soft)).float()
        self._profile_add("reachability_bundle/gate_filter_s", time.perf_counter() - t_seg)
        self._profile_add("reachability_bundle/total_s", time.perf_counter() - t_total)

        return {
            'topology_reachable': topo_reachable,
            'soft_reachability': soft_reachable,
            'hard_reachability_from_neg': neg_pressure_hard,
            'soft_reachability_from_neg': neg_pressure_soft,
            'dist_soft_pos': dist_soft_pos,
            'dist_hard_neg': dist_hard_neg,
        }, pos_distance_matrix

    def compute_distance(self, seeds, physics_context, weights, num_nodes):
        """
        Generic distance computation for arbitrary seeds.
        Used by EvidenceBuilder for Support Score (Source-wise).
        Using SciPy strict solver.
        """
        stt_dyn = physics_context.stt_dynamic
        if stt_dyn is None:
            from src.modeling.evidence.reachability import ReachabilityRuleModule
            return ReachabilityRuleModule()._bellman_ford(seeds, physics_context.edge_index, weights, num_nodes)
        adj_rev = self.get_scipy_reverse_graph(physics_context, num_nodes)
        
        seed_indices = seeds.nonzero().view(-1).cpu().numpy()
        
        if len(seed_indices) == 0:
            return torch.full((num_nodes,), self.infinity, device=seeds.device)
            
        dist_np = self._run_scipy_dijkstra(adj_rev, seed_indices, min_only=True)
        
        return torch.from_numpy(dist_np).float().to(seeds.device)

    def compute_distance_matrix(self, seed_indices: torch.Tensor, physics_context: PhysicsContext, num_nodes: int):
        t_total = time.perf_counter()
        stt_dyn = physics_context.stt_dynamic
        if stt_dyn is None:
            if physics_context.stt_median is not None:
                weights = torch.expm1(physics_context.stt_median)
            else:
                weights = torch.ones(physics_context.edge_index.size(1), device=seed_indices.device)
            rows = []
            for idx in seed_indices.view(-1):
                seeds = torch.zeros((num_nodes,), device=seed_indices.device)
                seeds[idx] = 1.0
                rows.append(self.compute_distance(seeds, physics_context, weights, num_nodes))
            if not rows:
                return torch.empty((num_nodes, 0), device=seed_indices.device)
            return torch.stack(rows, dim=1)

        t_seg = time.perf_counter()
        adj_rev = self.get_scipy_reverse_graph(physics_context, num_nodes)
        self._profile_add("distance_matrix/reverse_graph_lookup_s", time.perf_counter() - t_seg)
        t_seg = time.perf_counter()
        seed_idx_np = seed_indices.view(-1).cpu().numpy()
        self._profile_add("distance_matrix/seed_numpy_s", time.perf_counter() - t_seg)
        if len(seed_idx_np) == 0:
            return torch.empty((num_nodes, 0), device=seed_indices.device)

        self._profile_inc("distance_matrix/dijkstra_calls")
        t_seg = time.perf_counter()
        dist_matrix = self._run_scipy_dijkstra(adj_rev, seed_idx_np)
        self._profile_add("distance_matrix/dijkstra_s", time.perf_counter() - t_seg)
        t_seg = time.perf_counter()
        dist_np = np.asarray(dist_matrix)
        if dist_np.ndim == 1:
            dist_np = dist_np[None, :]
        dist_np = dist_np.T
        self._profile_add("distance_matrix/aggregate_s", time.perf_counter() - t_seg)
        t_seg = time.perf_counter()
        dist_tensor = torch.from_numpy(dist_np).float().to(seed_indices.device)
        self._profile_add("distance_matrix/torch_convert_s", time.perf_counter() - t_seg)
        self._profile_add("distance_matrix/total_s", time.perf_counter() - t_total)
        return dist_tensor

    def get_scipy_reverse_graph(self, physics_context: PhysicsContext, num_nodes: int):
        t_total = time.perf_counter()
        cached_adj = getattr(physics_context, "_cached_scipy_reverse_graph", None)
        cached_nodes = getattr(physics_context, "_cached_scipy_reverse_graph_nodes", None)
        if cached_adj is not None and cached_nodes == int(num_nodes):
            self._profile_inc("reverse_graph/cache_hit_count")
            self._profile_add("reverse_graph/lookup_total_s", time.perf_counter() - t_total)
            return cached_adj

        stt_dyn = physics_context.stt_dynamic
        if stt_dyn is None:
            return None
        adj_rev = self._build_scipy_reverse_graph(physics_context.edge_index, stt_dyn.view(-1), num_nodes)
        physics_context._cached_scipy_reverse_graph = adj_rev
        physics_context._cached_scipy_reverse_graph_nodes = int(num_nodes)
        self._profile_inc("reverse_graph/cache_miss_count")
        self._profile_add("reverse_graph/lookup_total_s", time.perf_counter() - t_total)
        return adj_rev

    def _build_scipy_reverse_graph(self, edge_index, stt_val, num_nodes):
        """
        Build Scipy CSR for Reverse Propagation.
        SSOT with GT Audit Script.
        """
        import scipy.sparse as sp
        import numpy as np
        
        t_total = time.perf_counter()
        t_seg = time.perf_counter()
        u = edge_index[0].cpu().numpy()
        v = edge_index[1].cpu().numpy()
        stt = stt_val.detach().cpu().numpy()
        self._profile_add("reverse_graph/numpy_export_s", time.perf_counter() - t_seg)
        
        epsilon = self.stt_epsilon
        
        # Forward Flow (STT > 0): u -> v.  Reverse Prop: v -> u.
        t_seg = time.perf_counter()
        mask_fwd = stt > epsilon
        src_fwd = v[mask_fwd]
        dst_fwd = u[mask_fwd]
        w_fwd = stt[mask_fwd]
        
        # Backward Flow (STT < 0): v -> u. Reverse Prop: u -> v.
        mask_bwd = stt < -epsilon
        src_bwd = u[mask_bwd]
        dst_bwd = v[mask_bwd]
        w_bwd = np.abs(stt[mask_bwd])
        
        src = np.concatenate([src_fwd, src_bwd])
        dst = np.concatenate([dst_fwd, dst_bwd])
        w = np.concatenate([w_fwd, w_bwd])
        self._profile_add("reverse_graph/partition_s", time.perf_counter() - t_seg)
        
        t_seg = time.perf_counter()
        if len(src) == 0:
            adj = sp.csr_matrix((num_nodes, num_nodes))
        else:
            adj = sp.csr_matrix((w, (src, dst)), shape=(num_nodes, num_nodes))
        self._profile_add("reverse_graph/csr_construct_s", time.perf_counter() - t_seg)
        self._profile_add("reverse_graph/build_total_s", time.perf_counter() - t_total)
            
        return adj

    def _run_scipy_dijkstra(self, adj, indices, min_only=False):
        from scipy.sparse.csgraph import dijkstra
        return dijkstra(adj, indices=indices, directed=True, return_predecessors=False, min_only=min_only)

    def _build_reverse_flow_graph(self, edge_index, stt_val):
        """Legacy Torch Graph Builder (Kept for reference or future GPU optimization)"""
        # ... (Existing implementation) ...
        u, v = edge_index[0], edge_index[1]
        
        # Mask for active edges
        is_active = (stt_val.abs() > self.stt_epsilon)
        
        # Forward Flow (STT > 0): u -> v
        # Reverse Prop: v -> u
        mask_fwd = (stt_val > self.stt_epsilon)
        src_fwd = v[mask_fwd]
        dst_fwd = u[mask_fwd]
        w_fwd = stt_val[mask_fwd]
        
        # Backward Flow (STT < 0): v -> u
        # Reverse Prop: u -> v
        mask_bwd = (stt_val < -self.stt_epsilon)
        src_bwd = u[mask_bwd]
        dst_bwd = v[mask_bwd]
        w_bwd = stt_val[mask_bwd].abs()
        
        # Combine
        prop_src = torch.cat([src_fwd, src_bwd])
        prop_dst = torch.cat([dst_fwd, dst_bwd])
        prop_w = torch.cat([w_fwd, w_bwd])
        
        prop_edge_index = torch.stack([prop_src, prop_dst], dim=0)
        
        return prop_edge_index, prop_w

    def _bellman_ford(self, seeds, edge_index, weights, num_nodes):
        """Legacy Torch Solver (Kept for reference)"""
        raise NotImplementedError("Bellman-Ford fallback is disabled in this audit sprint. Please ensure Dynamic STT is available.")
        # ... (Existing implementation) ...
        # dist = torch.full((num_nodes,), self.infinity, device=seeds.device)

        mask_seeds = (seeds > 0.5)
        dist[mask_seeds] = 0.0
        
        if not mask_seeds.any():
            return dist
            
        if edge_index.size(1) == 0:
            return dist
            
        src, dst = edge_index # src -> dst (Propagation Direction)
        
        # Propagate
        for _ in range(self.max_hops):
            d_src = dist[src]
            d_proposal = d_src + weights
            
            # Update dst with min(d_proposal)
            # Note: In standard BF, we iterate edges.
            # Here we scatter min to dst.
            
            d_new, _ = scatter_min(d_proposal, dst, dim=0, dim_size=num_nodes)
            
            # Update dist
            dist = torch.min(dist, d_new)
            
        return dist

    def _static_fallback(self, observation_state, physics_context, t_sim, batch):
        from src.modeling.evidence.reachability import ReachabilityRuleModule
        return ReachabilityRuleModule().compute_reachability(observation_state, physics_context, t_sim, batch)
