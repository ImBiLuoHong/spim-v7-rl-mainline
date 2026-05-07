
import torch
import torch.sparse
import numpy as np
import logging

class PhysicsPropagator:
    """
    GPU-Accelerated Physics Propagator (Hyperedge Logic)
    
    Transforms sparse "Point Samples" into dense "Field States" using hydraulic connectivity.
    Logic: State_{t+1} = State_t + Propagate(Observation, Reverse_Flow_Graph)
    """
    
    def __init__(self, edge_index, edge_attr_summary, num_nodes, device):
        """
        Args:
            edge_index: [2, E] tensor (Source -> Target flow)
            edge_attr_summary: [E, D] tensor (includes p_forward, STT, etc.)
            num_nodes: Total nodes in graph
            device: torch device
        """
        self.device = device
        self.num_nodes = num_nodes
        
        # 1. Build Reverse Adjacency Matrix (Target -> Source) for Upstream Propagation
        # If u -> v (Flow), then contamination at u implies v is upstream? 
        # Wait: Contamination flows u -> v.
        # If we detect at v, the source must be upstream (u).
        # So we need to propagate "Suspicion" from v back to u.
        # So we need Reverse Graph: v -> u.
        
        src, dst = edge_index
        # Reverse: dst -> src
        rev_src = dst
        rev_dst = src
        
        # 2. Compute Weights (Propagability)
        # We use p_forward to gate propagation. 
        # If p_forward(u->v) is high, then v is reliably downstream of u.
        # So if v is contaminated, u is reliably upstream.
        # We can also use STT to decay influence? Or just binary reachability for now.
        # Let's use p_forward as weight.
        
        # edge_attr_summary: [p_forward, q50, q90, min, flip]
        # Index 2 is p_forward (based on my previous check, wait check test script output)
        # Test script: Names = ['Length', 'Diam', 'p_forward', 'q50', 'q90', 'min', 'flip']
        # p_forward is index 2.
        
        if edge_attr_summary.shape[1] >= 3:
            p_fwd = edge_attr_summary[:, 2].to(device)
            # Clip to [0, 1] just in case
            p_fwd = torch.clamp(p_fwd, 0.0, 1.0)
        else:
            # Fallback
            p_fwd = torch.ones(edge_index.shape[1], device=device)
            
        # Create Sparse Matrix (N x N)
        # Row: v (Sensor), Col: u (Upstream Candidate)
        # Value: Probability/Strength
        
        # Coalesce to handle multi-edges
        indices = torch.stack([rev_src, rev_dst], dim=0).to(device)
        values = p_fwd
        
        self.adj_rev = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes)
        ).coalesce()
        
        # Normalize?
        # Maybe not. We want "Is Upstream?" boolean-like logic but differentiable.
        # If we propagate multiple steps, values might explode or vanish.
        # Let's keep it simple: 1-hop propagation for now, or pre-compute k-hop?
        # User said "GPU sparse matrix multiplication". This implies we can do Iterative Propagation.
        
    def propagate(self, sample_indices, sample_values, steps=3, decay=0.8):
        """
        Propagate observations upstream.
        
        Args:
            sample_indices: [B, K] Global indices of sampled nodes
            sample_values: [B, K] Observed values (e.g. Signal Strength)
            steps: Number of propagation steps (Depth)
            decay: Decay factor per step
            
        Returns:
            upstream_map: [B, N] Dense map of upstream influence
        """
        B, K = sample_indices.shape
        N = self.num_nodes
        
        # 1. Create Sparse Signal Matrix X [N, B] (Transposed for matmul)
        # We want Result [B, N] = Sample [B, N] @ Adj [N, N]
        # But sparse mm usually requires (Sparse @ Dense) or (Sparse @ Sparse).
        # PyTorch sparse mm: sparse @ dense -> dense.
        # So we use Adj_Rev [N, N] @ Signal_T [N, B] -> Upstream_T [N, B]
        
        # Flatten indices for sparse construction
        batch_idx = torch.arange(B, device=self.device).repeat_interleave(K)
        node_idx = sample_indices.flatten()
        vals = sample_values.flatten()
        
        # Filter zero values (no contamination -> no upstream suspicion?)
        # Or maybe we propagate "Safe" signal too? 
        # User prompt: "mark the entire upstream path as Suspicious if positive, or Cleared if negative"
        # Let's handle Positive (Suspicion) first.
        
        mask_pos = vals > 1e-6
        if not mask_pos.any():
            return torch.zeros((B, N), device=self.device)
            
        # Sparse Signal (N x B)
        # Row: Node, Col: Batch
        indices_pos = torch.stack([node_idx[mask_pos], batch_idx[mask_pos]], dim=0)
        values_pos = vals[mask_pos]
        
        signal_pos = torch.sparse_coo_tensor(
            indices_pos, values_pos, (N, B)
        ).to_dense() # (N, B) - Dense is okay if B is small (e.g. 128). 128*60k floats is ~30MB. Fine.
        
        # Propagation Loop
        # H_0 = Signal
        # H_{k+1} = A @ H_k * decay + H_0 (Restart/Residual)
        
        current_h = signal_pos
        accumulated = signal_pos.clone()
        
        for _ in range(steps):
            # A [N, N] @ H [N, B] -> [N, B]
            next_h = torch.sparse.mm(self.adj_rev, current_h) * decay
            accumulated += next_h
            current_h = next_h
            
        # Transpose back to [B, N]
        return accumulated.t()

    def get_stt_field(self, trigger_indices, max_stt=60.0):
        """
        Approximates STT field from triggers using propagation.
        Note: Precise STT requires Dijkstra. This is a diffusion proxy.
        High value = Close (Low STT).
        """
        # Similar to propagate but interpret values as "Proximity"
        pass
