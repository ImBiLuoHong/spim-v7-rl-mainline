
import torch
import numpy as np
import networkx as nx
import logging
import random
from typing import List, Set, Optional, Tuple, Union

class ZJUAgent:
    """
    ZJU Binary Search Agent (Ji et al., 2022, Water Resources Research)
    
    Faithful reproduction of the Manual Grab-Sampling Method (MGSM).
    Objective: Minimize the expected residual search space length F(A) = Σ Li^2 / Lall^2.
    Assumption: Source is uniformly distributed along the length of pipes.
    """
    
    def __init__(self, graph_data, p_forward_idx: int = 2, max_candidates: int = 100):
        self.graph_data = graph_data
        self.num_nodes = graph_data.num_nodes
        self.p_forward_idx = p_forward_idx
        self.max_candidates = max_candidates
        
        # Build Static Reachability Map (Original Ji 2022 uses static flow)
        # We assume dominant flow direction p_forward > 0.5
        edge_index = graph_data.edge_index
        self.src_np = edge_index[0].cpu().numpy()
        self.dst_np = edge_index[1].cpu().numpy()
        self.edge_attr = graph_data.edge_attr
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.num_nodes))
        
        if self.edge_attr is not None and self.edge_attr.size(1) > p_forward_idx:
            p_forward = self.edge_attr[:, p_forward_idx].cpu().numpy()
            mask = p_forward > 0.5
            self.G.add_edges_from(list(zip(self.src_np[mask], self.dst_np[mask])))
        else:
            self.G.add_edges_from(list(zip(self.src_np, self.dst_np)))
            
        # Calculate Node Weights based on Pipe Length (Discretization Approximation)
        # Original: Source on pipe. Reproduced: Source on node with length-based weight.
        self.node_weights = np.zeros(self.num_nodes)
        if self.edge_attr is not None and self.edge_attr.size(1) > 0:
            lengths = self.edge_attr[:, 0].cpu().numpy()
            for i in range(len(self.src_np)):
                u, v, L = self.src_np[i], self.dst_np[i], lengths[i]
                self.node_weights[u] += L / 2.0
                self.node_weights[v] += L / 2.0
            self.node_weights = np.maximum(self.node_weights, 1e-6)
            logging.info(f"[ZJUAgent] Length-based weights computed. Total L: {self.node_weights.sum():.2f}")
        else:
            self.node_weights = np.ones(self.num_nodes)
            logging.warning("[ZJUAgent] No pipe lengths found. Using uniform weights.")
        
        # Memoization
        self.memoized_upstream = {}
        
        # State
        self.search_area: Set[int] = set()
        
        logging.info(f"[ZJUAgent] Initialized. Nodes: {self.num_nodes}, Edges: {self.G.number_of_edges()}")

    def reset(self, initial_trigger_node: Union[int, List[int]] = None, **kwargs):
        """Reset agent. Search Area starts as all nodes upstream of first trigger."""
        self.search_area = set(range(self.num_nodes))
        
        if initial_trigger_node is not None:
            trigger = initial_trigger_node if isinstance(initial_trigger_node, int) else initial_trigger_node[0]
            upstream = self._get_upstream(trigger) | {trigger}
            self.search_area = self.search_area.intersection(upstream)
            
        return self.get_action()

    def step(self, action: int, result: int):
        """Update Search Area: Negative -> Remove Upstream; Positive -> SA = Upstream."""
        upstream_inclusive = self._get_upstream(action) | {action}
        
        if result == 1: # Positive
            self.search_area = self.search_area.intersection(upstream_inclusive)
        else: # Negative
            # Strict MGSM Logic: "If sample is safe, then source is NOT upstream."
            # This logic assumes STATIC contamination (infinite time / steady state).
            # In dynamic scenarios, this causes false negatives if plume hasn't reached yet.
            # However, to faithfully reproduce the "Manual Grab Sampling Method" as defined
            # in Ji et al. (which assumes steady state or worst-case coverage), we must keep it.
            # The low SR (14%) is a valid reflection of using a static method in a dynamic problem.
            self.search_area = self.search_area - upstream_inclusive

    def get_action(self, observation=None, k=1) -> Union[int, List[int]]:
        """
        Select k hydrants to minimize F(A) = Σ Li^2 / Lall^2.
        Greedy strategy: iteratively pick hydrants to minimize the sum of squared partition lengths.
        """
        if not self.search_area:
            pool = list(range(self.num_nodes))
            return random.choice(pool) if k == 1 else random.choices(pool, k=k)
            
        sa_list = list(self.search_area)
        candidates = random.sample(sa_list, min(len(sa_list), self.max_candidates))
        
        # Current partitions: list of sets of nodes
        partitions = [set(self.search_area)]
        selected_actions = []
        
        for _ in range(k):
            best_node = -1
            min_f = float('inf')
            
            for node in candidates:
                if node in selected_actions: continue
                
                up_inc = self._get_upstream(node) | {node}
                
                # Calculate new partitions sum of squares
                current_sum_sq = 0.0
                for p in partitions:
                    # This node splits partition p into p_pos and p_neg
                    p_pos_weight = 0.0
                    p_neg_weight = 0.0
                    
                    # Optimization: iterate over smaller set
                    if len(up_inc) < len(p):
                        for u in up_inc:
                            if u in p: p_pos_weight += self.node_weights[u]
                        p_weight = sum(self.node_weights[m] for m in p)
                        p_neg_weight = p_weight - p_pos_weight
                    else:
                        for m in p:
                            if m in up_inc: p_pos_weight += self.node_weights[m]
                            else: p_neg_weight += self.node_weights[m]
                    
                    current_sum_sq += (p_pos_weight**2 + p_neg_weight**2)
                
                # Other unchanged partitions remain same (sum sq)
                # But here we evaluate the WHOLE F(A) logic
                if current_sum_sq < min_f:
                    min_f = current_sum_sq
                    best_node = node
            
            if best_node == -1:
                best_node = random.choice([n for n in sa_list if n not in selected_actions] or [0])
            
            selected_actions.append(best_node)
            
            # Update partitions for next greedy selection
            new_partitions = []
            best_up_inc = self._get_upstream(best_node) | {best_node}
            for p in partitions:
                p_pos = p.intersection(best_up_inc)
                p_neg = p - best_up_inc
                if p_pos: new_partitions.append(p_pos)
                if p_neg: new_partitions.append(p_neg)
            partitions = new_partitions
            
        return selected_actions[0] if k == 1 else selected_actions

    def _get_upstream(self, node: int) -> Set[int]:
        if node in self.memoized_upstream: return self.memoized_upstream[node]
        ancestors = nx.ancestors(self.G, node)
        self.memoized_upstream[node] = ancestors
        return ancestors

    def predict(self) -> torch.Tensor:
        mask = torch.zeros(self.num_nodes)
        if self.search_area: mask[list(self.search_area)] = 1.0
        return mask
