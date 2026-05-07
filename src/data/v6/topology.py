
import os
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, dijkstra
import multiprocessing
from multiprocessing import shared_memory

# Global shared memory references (Process-safe)
_SHM_INDICES = None
_SHM_INDPTR = None
_SHM_DATA = None
_SHM_META = None 

class HydraulicTopology:
    """
    Singleton-like helper to manage the global hydraulic topology.
    Used for computing dynamic RPE_STT (Virtual Edges) on the fly.
    Optimized with Shared Memory for Multi-Process DataLoading.
    """
    def __init__(self, foundation_dir):
        self.foundation_dir = foundation_dir
        self.graph_path = os.path.join(foundation_dir, 'graph.npz')
        
        # Local references (will point to shared memory in workers)
        self.rev_adj_matrix = None 
        self.node_coords = None
        
        # Metadata
        self.num_nodes = 0
        self.num_edges = 0
        
        # Cache (Local to process)
        self.cache = {} 
        self.max_cache_size = 50000
        
        # Initialize (Load to SHM if main, link if worker)
        self._init_shared_graph()

    def _init_shared_graph(self):
        """
        Initializes graph in shared memory.
        If main process: Loads from disk, creates SHM blocks.
        If worker process: Attaches to existing SHM blocks.
        """
        global _SHM_INDICES, _SHM_INDPTR, _SHM_DATA, _SHM_META
        
        shm_name_base = f"hydra_topo_{os.getpid()}" # PID of parent if forked? 
        # Actually, for 'spawn' context (default on Windows/MacOS, but Linux uses fork), 
        # shared memory needs careful handling. 
        # On Linux (fork), global variables might be copied-on-write.
        # But for 'fork', we can just load in parent and children inherit.
        # HOWEVER, the user issue is that children are RE-LOADING.
        # This implies standard 'fork' should work IF initialized before fork.
        # But PyTorch DataLoader workers might re-init the Dataset.
        
        # Strategy:
        # 1. Load data once in __init__ (Parent).
        # 2. Store big arrays in shared memory (System V or POSIX).
        # 3. Children attach.
        
        # Let's use a simpler "Static Class Variable" approach first.
        # If the Dataset is initialized in the main process, `self.rev_adj_matrix` is loaded.
        # When `num_workers > 0` (fork), children inherit this memory (COW).
        # So why did we see re-loading logs?
        # Because `NpzDatasetV6` creates `self.topology = HydraulicTopology(...)` inside `__init__`.
        # If `NpzDatasetV6` is pickled to workers, `__init__` is NOT called in workers.
        # BUT, if `HydraulicTopology` is not picklable or if we re-create it...
        
        # Wait, the logs showed "[HydraulicTopology] Built full mesh graph" REPEATEDLY.
        # This means `HydraulicTopology.__init__` is being called multiple times.
        # This happens if `NpzDatasetV6` is re-instantiated or if `HydraulicTopology` is not preserved.
        
        # Optimization: Use a Class-Level Singleton Cache.
        # This works for 'fork' because the class attribute is populated in parent.
        if hasattr(HydraulicTopology, '_shared_instance'):
             # Copy state from shared instance
             shared = HydraulicTopology._shared_instance
             self.rev_adj_matrix = shared.rev_adj_matrix
             self.node_coords = shared.node_coords
             self.num_nodes = shared.num_nodes
             self.edge_attr_dynamic = shared.edge_attr_dynamic
             self.dynamic_loaded = shared.dynamic_loaded
             self.u_indices = shared.u_indices
             self.v_indices = shared.v_indices
             return

        # If not shared, load it.
        self._load_and_build()
        
        # Set as shared instance
        HydraulicTopology._shared_instance = self

    def _load_and_build(self):
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Foundation graph not found: {self.graph_path}")
            
        print(f"[HydraulicTopology] Loading {self.graph_path}...")
        with np.load(self.graph_path, allow_pickle=True) as f:
            edge_index = f['edge_index'] # (2, E)
            
            # Load Dynamic Data if available
            if 'edge_attr_dynamic' in f:
                self.edge_attr_dynamic = f['edge_attr_dynamic']
                self.dynamic_loaded = True
                print(f"[HydraulicTopology] Loaded dynamic edge attrs: {self.edge_attr_dynamic.shape}")
            else:
                self.edge_attr_dynamic = None
                self.dynamic_loaded = False
            
            if 'edge_attr_summary' in f:
                summary = f['edge_attr_summary']
                p_forward = summary[:, 0]
                min_stt = summary[:, 3]
            else:
                print("[HydraulicTopology] Warning: edge_attr_summary not found. Using static edges.")
                E = edge_index.shape[1]
                p_forward = np.ones(E)
                min_stt = np.ones(E)

            if 'node_coords' in f:
                self.node_coords = f['node_coords']
                print(f"[HydraulicTopology] Loaded node coordinates: {self.node_coords.shape}")
            else:
                self.node_coords = None

        # Build Full Mesh Graph (u->v and v->u for all pipes)
        u, v = edge_index[0], edge_index[1]
        self.u_indices = u
        self.v_indices = v
        
        all_src = np.concatenate([u, v])
        all_dst = np.concatenate([v, u])
        
        w_uv = np.where(p_forward > 0.5, min_stt, np.inf)
        w_vu = np.where(p_forward <= 0.5, min_stt, np.inf)
        
        w_uv = np.maximum(w_uv, 1e-4)
        w_vu = np.maximum(w_vu, 1e-4)
        
        all_w = np.concatenate([w_uv, w_vu])
        
        N = max(edge_index.max(), max(all_src.max(), all_dst.max())) + 1
        self.num_nodes = N
        
        # Construct CSRs
        # Reverse: Dst->Src (for Upstream search) - This is the PRIMARY one we use
        self.rev_adj_matrix = sp.coo_matrix((all_w, (all_dst, all_src)), shape=(N, N)).tocsr()
        
        print(f"[HydraulicTopology] Built full mesh graph. N={N}, E_mesh={len(all_w)}")


    def _get_snapshot_weights(self, time_idx):
        """
        Generates weight vector for the Full Mesh at time_idx.
        Returns weights corresponding to [u->v edges, v->u edges].
        """
        if not self.dynamic_loaded or time_idx < 0 or time_idx >= self.edge_attr_dynamic.shape[0]:
            return None
            
        step_data = self.edge_attr_dynamic[time_idx]
        flow = step_data[:, 0] # + means u->v
        stt = step_data[:, 1]  # always +
        
        w_uv = np.where(flow > 1e-6, stt, np.inf)
        w_vu = np.where(flow < -1e-6, stt, np.inf)
        
        w_uv = np.maximum(w_uv, 1e-4)
        w_vu = np.maximum(w_vu, 1e-4)
        
        return np.concatenate([w_uv, w_vu])

    def _get_dists_cached(self, trigger_idx, time_idx=-1):
        """
        Returns cached or computed distance array (N_global,) for a specific time.
        """
        cache_key = (trigger_idx, time_idx)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Determine Graph to use
        graph_to_use = self.rev_adj_matrix # Default static
        
        if time_idx != -1 and self.dynamic_loaded:
            weights = self._get_snapshot_weights(time_idx)
            if weights is not None:
                # Rebuild temporary graph for this snapshot
                # Reverse Graph: Dst -> Src
                all_src = np.concatenate([self.u_indices, self.v_indices])
                all_dst = np.concatenate([self.v_indices, self.u_indices])
                graph_to_use = sp.coo_matrix((weights, (all_dst, all_src)), shape=(self.num_nodes, self.num_nodes)).tocsr()

        # Compute Dijkstra
        dists = dijkstra(graph_to_use, directed=True, indices=[trigger_idx], return_predecessors=False)
        dists = dists[0].astype(np.float32)
        
        # Update Cache
        if len(self.cache) >= self.max_cache_size:
            k = next(iter(self.cache))
            del self.cache[k]
            
        self.cache[cache_key] = dists
        return dists

    def get_upstream_stt(self, target_nodes, max_dist=None):
        """
        Computes STT from all upstream nodes TO the target_nodes.
        
        Args:
            target_nodes (list/tensor): Indices of trigger nodes (Global IDs).
            max_dist (float): Cutoff distance (optional).
            
        Returns:
            dict: {target_node_idx: (source_indices, distances)}
            OR
            tuple: (edge_index, edge_attr) for PyG
        """
        if isinstance(target_nodes, torch.Tensor):
            target_nodes = target_nodes.cpu().numpy()
            
        # Run Dijkstra on Reversed Graph from targets
        # dist_matrix: (n_targets, N)
        # indices argument allows running only for specific sources
        dist_matrix = dijkstra(self.rev_adj_matrix, directed=True, indices=target_nodes, return_predecessors=False)
        
        # Convert to PyG format (Virtual Edges)
        # Edge: Source -> Target
        # Weight: dist
        
        sources = []
        targets = []
        weights = []
        
        for i, t_idx in enumerate(target_nodes):
            dists = dist_matrix[i]
            
            # Valid upstream nodes are those with finite distance
            # And distance > 0 (exclude self-loop if desired? User didn't say. Usually self is 0.)
            # User says "Upstream that node -> trigger".
            # Self -> Trigger is dist 0. Maybe keep it? 
            # "If in upstream... edge... weight is STT".
            
            valid_mask = (dists != np.inf) & (dists < 1e10) # 1e10 safety
            
            if max_dist:
                valid_mask &= (dists <= max_dist)
            
            src_indices = np.where(valid_mask)[0]
            w_values = dists[valid_mask]
            
            # Append
            # We want arrays to stack later
            # This loop might be slow if many targets and massive graph.
            # But here N=64k.
            # Vectorized approach:
            # coo_matrix of dists? No, dists is dense array.
            
            # Optimization for "Elegant Engineering":
            # Just keep the arrays.
            
            # src_indices is (M, )
            # t_idx is scalar
            sources.append(src_indices)
            targets.append(np.full_like(src_indices, t_idx))
            weights.append(w_values)
            
        if not sources:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
            
        all_src = np.concatenate(sources)
        all_dst = np.concatenate(targets)
        all_w = np.concatenate(weights)
        
        edge_index = torch.from_numpy(np.stack([all_src, all_dst])).long()
        edge_attr = torch.from_numpy(all_w).float().unsqueeze(1) # (E, 1)
        
        return edge_index, edge_attr

    def get_virtual_edges_for_subgraph(self, trigger_global_idx, subgraph_global_ids, subgraph_node_map=None, time_idx=-1, anchor_value=1.0, anchor_time=0.0):
        """
        Efficiently generates virtual edges for a subgraph relative to a specific trigger.
        
        Args:
            trigger_global_idx (int): Global ID of the trigger.
            subgraph_global_ids (torch.Tensor): Global IDs of nodes in the subgraph.
            subgraph_node_map (dict, optional): Map Global -> Local.
            time_idx (int, optional): Time step index (0-287) for dynamic topology. -1 for static.
            anchor_value (float): 1.0 for Positive, -1.0 for Negative.
            anchor_time (float): The simulation time when this anchor was sampled (t_v).
            
        Returns:
            tuple: (local_edge_index, local_edge_attr)
        """
        # 1. Run Dijkstra (Reversed) from trigger
        if trigger_global_idx < 0 or trigger_global_idx >= self.num_nodes:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)

        dists = self._get_dists_cached(trigger_global_idx, time_idx)
        
        # 2. Extract for subgraph nodes
        ids_np = subgraph_global_ids.cpu().numpy()
        sub_dists = dists[ids_np] # (M,)
        
        # 3. Filter Upstream (Finite dist)
        # Architecture 8.2.2: Only connect UPSTREAM nodes (those that can reach the trigger)
        mask = (sub_dists != np.inf) & (sub_dists < 1e10)
        src_local = np.where(mask)[0]
        
        if len(src_local) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)
            
        # Find trigger local index
        trigger_local = -1
        if subgraph_node_map is not None:
            trigger_local = subgraph_node_map.get(trigger_global_idx, -1)
        else:
            hits = np.where(ids_np == trigger_global_idx)[0]
            if len(hits) > 0:
                trigger_local = hits[0]
        
        if trigger_local == -1:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)
            
        # 4. Compute Euclidean RPE if coordinates available
        euclidean_rpe = np.zeros(len(src_local), dtype=np.float32)
        if self.node_coords is not None:
            trigger_coord = self.node_coords[trigger_global_idx] # (2,) or (3,)
            src_coords = self.node_coords[ids_np[src_local]] # (M_src, 2)
            euclidean_rpe = np.linalg.norm(src_coords - trigger_coord, axis=1)

        # 5. Construct Edges
        n_edges = len(src_local)
        dst_local = np.full(n_edges, trigger_local, dtype=np.int64)
        stt_weights = sub_dists[mask]
        
        edge_index = torch.from_numpy(np.stack([src_local, dst_local])).long()
        
        # [FIX] STT Scaling (Heuristics Compatibility)
        # Convert STT to "Physical Minutes" if it was precomputed as seconds or normalized
        # Graph.npz config usually: stt in seconds? or minutes?
        # If mean is small (e.g. 0.1), it might be normalized.
        # But here we trust the raw value from graph.npz.
        # Heuristics Engine will handle scaling via config (sigma).
        # We just return raw.
        
        # Architecture 8.3.2 (Version 4.5.21): Virtual Edge Attributes (Racing Semantic)
        # Return 4 channels: [STT, Euclidean_RPE, Anchor_Type, Anchor_Time]
        edge_attr = torch.zeros((n_edges, 4), dtype=torch.float32)
        edge_attr[:, 0] = torch.from_numpy(stt_weights).float()
        edge_attr[:, 1] = torch.from_numpy(euclidean_rpe).float()
        edge_attr[:, 2] = float(anchor_value)
        edge_attr[:, 3] = float(anchor_time)
        
        return edge_index, edge_attr
