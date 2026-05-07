import torch
import numpy as np
import networkx as nx
from collections import Counter
import scipy.stats

class AdvancedMetrics:
    """
    Implements advanced metrics for AutoSamplingGNN evaluation:
    1. Mean Hop Error (MHE)
    2. Mean Euclidean Error (MEE)
    3. Signal Capture Rate (SCR)
    4. Sampling Entropy
    
    Requires global graph topology for MHE.
    """
    
    def __init__(self, global_edge_index, global_pos):
        """
        Args:
            global_edge_index: [2, E] tensor, full graph topology
            global_pos: [N, 2] tensor, node coordinates
        """
        self.global_edge_index = global_edge_index.cpu().numpy()
        self.global_pos = global_pos.cpu().numpy()
        
        # Build NetworkX graph for shortest path
        print("[Metrics] Building Global Graph for MHE calculation...")
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.global_pos.shape[0]))
        edges = list(zip(self.global_edge_index[0], self.global_edge_index[1]))
        self.G.add_edges_from(edges)
        print(f"[Metrics] Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        
    def compute_mhe(self, pred_indices, true_indices):
        """
        Computes Mean Hop Error between predicted and true source nodes.
        Args:
            pred_indices: list or array of predicted global node IDs
            true_indices: list or array of true source global node IDs
        Returns:
            mean_hop_error (float)
        """
        total_hops = 0
        valid_count = 0
        
        for p, t in zip(pred_indices, true_indices):
            try:
                # Use bidirectional BFS for speed
                dist = nx.shortest_path_length(self.G, source=int(p), target=int(t))
                total_hops += dist
                valid_count += 1
            except nx.NetworkXNoPath:
                # Should not happen in connected water network, but safe to ignore or penalize
                total_hops += 50 # Penalty for disconnected (rare)
                valid_count += 1
            except Exception:
                pass
                
        return total_hops / max(1, valid_count)

    def compute_mee(self, pred_indices, true_indices):
        """
        Computes Mean Euclidean Error.
        """
        total_dist = 0.0
        valid_count = 0
        
        for p, t in zip(pred_indices, true_indices):
            p_coord = self.global_pos[int(p)]
            t_coord = self.global_pos[int(t)]
            dist = np.linalg.norm(p_coord - t_coord)
            total_dist += dist
            valid_count += 1
            
        return total_dist / max(1, valid_count)

    def compute_scr(self, sampled_masks, dynamic_signal, signal_threshold=0.01):
        """
        Computes Signal Capture Rate (SCR).
        Args:
            sampled_masks: [B, N] binary mask indicating sampled nodes
            dynamic_signal: [B, N] continuous signal values (Ch 0)
            signal_threshold: threshold to consider signal "valid"
        Returns:
            scr (float): Proportion of samples (graphs) where at least one sampled node has signal > threshold.
        """
        # Element-wise multiply
        captured_signal = sampled_masks * dynamic_signal
        
        # Max signal per graph
        max_captured = captured_signal.max(dim=1)[0] # [B]
        
        # Check threshold
        is_captured = (max_captured > signal_threshold).float()
        
        return is_captured.mean().item()

    def compute_entropy(self, all_sampled_indices, num_nodes):
        """
        Computes Sampling Entropy across the test set.
        Args:
            all_sampled_indices: List of global node IDs that were sampled.
            num_nodes: Total number of nodes in graph.
        Returns:
            entropy (float)
        """
        # Count frequency of each node being sampled
        counts = Counter(all_sampled_indices)
        
        # Convert to probability distribution
        probs = np.zeros(num_nodes)
        total_samples = len(all_sampled_indices)
        
        for node_idx, count in counts.items():
            if node_idx < num_nodes:
                probs[node_idx] = count / total_samples
            
        # Compute entropy
        # Add epsilon to avoid log(0)
        entropy = scipy.stats.entropy(probs + 1e-9)
        return entropy
