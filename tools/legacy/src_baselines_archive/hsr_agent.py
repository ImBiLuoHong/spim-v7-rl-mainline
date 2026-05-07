
import numpy as np
import networkx as nx
import os
import sys

class HSRAgent:
    """
    Heuristic Space Reduction (HSR) Agent
    
    A rule-based baseline for source localization in water distribution networks.
    Operates purely on Set Theory and Hydraulic Rules (Upstream, Time, Negative constraints).
    
    Reference: ACS EST Water 2025
    """
    
    def __init__(self, graph_path=None, graph_data=None, shared_rev_graph=None, time_step_hours=0.25, tolerance_hours=0.0):
        """
        Initialize the HSR Agent.
        
        Args:
            graph_path (str): Path to graph.npz. If None, tries to locate it automatically.
            graph_data (dict): Pre-loaded graph data dictionary/object to avoid reloading. 
                               Must contain 'edge_index', 'p_forward', 'median_stt'.
            shared_rev_graph (nx.DiGraph): Pre-built reverse graph to share across instances.
            time_step_hours (float): Duration of one time step in hours (default 0.25 = 15 min).
            tolerance_hours (float): Tolerance epsilon for Time Rule (H2).
        """
        self.time_step_hours = time_step_hours
        self.epsilon = tolerance_hours
        self.current_time_step = 0
        self.G_rev = shared_rev_graph
        
        # Load Graph
        if graph_data is not None:
            self._init_from_data(graph_data)
        else:
            if graph_path is None:
                # Default path based on exploration
                graph_path = '/root/autodl-tmp/rl_spim_v7_mainline/datanew/production_data/foundation_20260114_164946_86d5023e/graph.npz'
            self._load_graph(graph_path)
        
        # State
        self.candidate_set = set(range(self.num_nodes))
        self.sampled_nodes = set()
        self.safe_observations = {} # {node_idx: t_observed}
        self.trigger_sensor = None
        self.t_start = 0

    def _init_from_data(self, data):
        """Initialize graph structures from pre-loaded data."""
        self.edge_index = data['edge_index']
        self.p_forward = data['p_forward']
        self.median_stt = data['median_stt']
        if self.G_rev is None:
            self._build_reverse_graph()
        else:
            # Just set num_nodes from G_rev if available, or data
            self.num_nodes = self.G_rev.number_of_nodes()

    def _load_graph(self, path):
        print(f"[HSRAgent] Loading graph from {path}...")
        try:
            with np.load(path, allow_pickle=True) as f:
                self.edge_index = f['edge_index'] # (2, E)
                if 'edge_attr_summary' in f:
                    # Ch0: p_forward, Ch1: median_stt
                    summary = f['edge_attr_summary']
                    self.p_forward = summary[:, 0]
                    self.median_stt = summary[:, 1]
                else:
                    print("[HSRAgent] Warning: edge_attr_summary not found. Using defaults.")
                    E = self.edge_index.shape[1]
                    self.p_forward = np.ones(E)
                    self.median_stt = np.ones(E) * 0.02 # ~1 min default
        except Exception as e:
            print(f"[HSRAgent] Error loading graph: {e}")
            raise e
        
        if self.G_rev is None:
            self._build_reverse_graph()
        else:
             self.num_nodes = self.G_rev.number_of_nodes()

    def _build_reverse_graph(self):
        """
        Build a reverse graph (dst -> src) for upstream search.
        To handle hydraulic instability (backflow) and align with the 
        Scenario-based logic of the original HSR paper, we build a 
        'Relaxed' graph that includes all possible flow directions.
        """
        self.num_nodes = max(self.edge_index[0].max(), self.edge_index[1].max()) + 1
        
        u_orig, v_orig = self.edge_index[0], self.edge_index[1]
        
        # Original HSR logic often assumes a dominant direction (p_forward > 0.5).
        # However, for 60k+ nodes with backflow, this leads to massive mis-pruning.
        # To be mathematically equivalent to a Scenario Database (which contains 
        # all physical possibilities), we must include any edge direction that 
        # appears in our data.
        
        # p_forward > 0.001: u -> v has occurred
        mask_fwd = self.p_forward > 0.001
        # p_forward < 0.999: v -> u has occurred (backflow)
        mask_rev = self.p_forward < 0.999
        
        # Physical Edges
        phy_u = np.concatenate([u_orig[mask_fwd], v_orig[mask_rev]])
        phy_v = np.concatenate([v_orig[mask_fwd], u_orig[mask_rev]])
        
        # Use MIN_STT for pruning rules (H1, H2, H3) to be conservative.
        # This ensures we don't prune a source that could have reached the sensor 
        # faster than the median time.
        if hasattr(self, 'min_stts') and self.min_stts is not None:
            phy_w = np.concatenate([self.min_stts[mask_fwd], self.min_stts[mask_rev]])
        else:
            phy_w = np.concatenate([self.median_stt[mask_fwd], self.median_stt[mask_rev]])
        
        self.G_rev = nx.DiGraph()
        self.G_rev.add_nodes_from(range(self.num_nodes))
        
        # Reverse Graph: dst -> src with weight = STT
        for u, v, w in zip(phy_u, phy_v, phy_w):
            self.G_rev.add_edge(v, u, weight=max(w, 1e-6))
            
        print(f"[HSRAgent] Relaxed Reverse Graph built. Edges: {self.G_rev.number_of_edges()} (included backflow directions)")

    def reset(self, candidates=None, trigger_node=None, t_start=0):
        """
        Reset the agent state for a new episode.
        
        Args:
            candidates (list/array, optional): Initial set of candidate node IDs. 
                                               If None, uses all nodes.
            trigger_node (int, optional): The node that first detected the poison.
            t_start (int, optional): The time step when the poison was first detected.
        """
        if candidates is not None:
            self.candidate_set = set(candidates)
        else:
            self.candidate_set = set(range(self.num_nodes))
            
        self.sampled_nodes = set()
        self.safe_observations = {}
        self.current_time_step = 0
        self.trigger_sensor = trigger_node
        self.t_start = t_start
        
        if trigger_node is not None:
            self.sampled_nodes.add(trigger_node)
            
        return self.get_belief()

    def step(self, observation):
        """
        Update candidate set based on observation.
        observation: dict {node_idx: (signal, label)}
        """
        # Batch update: Time only moves ONCE per step call
        self.current_time_step += 1
        current_t_hours = self.current_time_step * self.time_step_hours
        
        # Batch Pruning logic
        for node_idx, (signal, label) in observation.items():
            node_idx = int(node_idx)
            self.sampled_nodes.add(node_idx)
            
            is_positive = signal > 1e-6 
            
            if is_positive:
                if self.trigger_sensor is None:
                    self.trigger_sensor = node_idx
                    self.t_start = self.current_time_step
                
                # H1 & H2: Positive => Must be upstream within time
                cutoff_dist = current_t_hours + self.epsilon
                valid_upstream = nx.single_source_dijkstra_path_length(
                    self.G_rev, node_idx, cutoff=cutoff_dist, weight='weight'
                )
                self.candidate_set.intersection_update(set(valid_upstream.keys()))
            else:
                # Negative => Safe Observation
                self.safe_observations[node_idx] = current_t_hours
                
                # H3: Negative Pruning (Conservative)
                # Only prune if it SHOULD have arrived by now (T_arrival < T_now)
                # We use Static graph for conservative H3 pruning as per paper
                if not hasattr(self, 'G_rev_static'):
                    self._build_static_reverse_graph()
                
                # "Nodes that DEFINITELY reach node_idx within current_t_hours"
                # If they did, they would have caused a Positive signal.
                # Since signal is Negative, they are not the source.
                cutoff_dist = current_t_hours
                impure_upstream = nx.single_source_dijkstra_path_length(
                    self.G_rev_static, node_idx, cutoff=cutoff_dist, weight='weight'
                )
                self.candidate_set.difference_update(set(impure_upstream.keys()))

    def _build_static_reverse_graph(self):
        """
        Build a static reverse graph based on dominant flow (p_forward > 0.5).
        Used for H3 pruning to avoid over-pruning in the relaxed graph.
        """
        u_orig, v_orig = self.edge_index[0], self.edge_index[1]
        mask_fwd = self.p_forward > 0.5
        mask_rev = self.p_forward <= 0.5
        
        phy_u = np.concatenate([u_orig[mask_fwd], v_orig[mask_rev]])
        phy_v = np.concatenate([v_orig[mask_fwd], u_orig[mask_rev]])
        phy_w = np.concatenate([self.median_stt[mask_fwd], self.median_stt[mask_rev]])
        
        self.G_rev_static = nx.DiGraph()
        self.G_rev_static.add_nodes_from(range(self.num_nodes))
        for u, v, w in zip(phy_u, phy_v, phy_w):
            self.G_rev_static.add_edge(v, u, weight=max(w, 1e-6))

    def get_stt(self, source_nodes, target_node):
        """
        Calculate Shortest Travel Time (STT) from multiple sources to one target.
        Uses G_rev to find distances from target to sources.
        """
        if target_node is None:
            return np.zeros(len(source_nodes))
            
        # Dijkstra from target_node in G_rev
        dists = nx.single_source_dijkstra_path_length(self.G_rev, target_node, weight='weight')
        
        stt_values = []
        for node in source_nodes:
            stt_values.append(dists.get(node, 1e9)) # 1e9 if unreachable
            
        return np.array(stt_values)

    def get_action_hsr_scalable(self, k=3):
        """
        HSR-Scalable Core Operator based on Monte Carlo Voting.
        """
        vote_counts, candidate_nodes = self._compute_hsr_vote_counts()
        if not candidate_nodes:
            return [self.get_action()]
        sorted_suspects = sorted(vote_counts.keys(), key=lambda n: vote_counts[n], reverse=True)
        return sorted_suspects[:k]

    def _compute_hsr_vote_counts(self):
        """
        Shared Monte Carlo suspect weighting used by the scalable baseline.
        Returns:
            vote_counts: {candidate_node: integer votes}
            candidate_nodes: remaining unsampled candidates
        """
        candidate_nodes = list(self.candidate_set - self.sampled_nodes)
        if not candidate_nodes:
            return {}, []
            
        if self.trigger_sensor is None:
            return {int(node): 0 for node in candidate_nodes}, candidate_nodes

        # 1. Initialize票箱
        vote_counts = {node: 0 for node in candidate_nodes}
        
        # 2. 蒙特卡洛循环 (N=50 次足以逼近分布)
        N_MC = 50
        t_elapsed = (self.current_time_step - self.t_start) * self.time_step_hours
        
        # 获取基础物理 STT (从所有 candidate 到 trigger sensor)
        base_stts = self.get_stt(candidate_nodes, self.trigger_sensor)
        
        # 为了 H3 (Sampler State Comparison) 反馈逻辑，预计算到所有 Safe 节点的 STT
        safe_nodes = list(self.safe_observations.keys())
        safe_base_stts = {} # {safe_node: stts_from_candidates}
        for snode in safe_nodes:
            safe_base_stts[snode] = self.get_stt(candidate_nodes, snode)

        for _ in range(N_MC):
            # --- A. 注入不确定性 (已根据实验结果恢复导师建议的 20% 噪声，这能提高鲁棒性) ---
            noise = np.random.normal(loc=1.0, scale=0.2, size=len(candidate_nodes))
            noisy_stts = base_stts * noise
            
            # --- B. 水力竞速校验 (H2 Equivalence) ---
            # 计算相对于 trigger sensor 的时间残差
            time_residuals = np.abs(noisy_stts - 0.0) # Relative to trigger detection at t=0
            # Note: t_elapsed is the time since trigger. 
            # The STT from Source to Trigger is what we are matching against the "ideal" 0 
            # if we consider the simulation starting at the moment of detection.
            # Wait, the teacher says: |(Tnow - Tstart) - STT(n, s)| <= epsilon
            # My t_elapsed is (Tnow - Tstart). So:
            time_residuals = np.abs(noisy_stts - t_elapsed)
            
            # --- C. 采样状态校验 (H3 Equivalence) ---
            # 如果某节点在当前噪声下会污染 Safe 节点，则在这一轮被排除
            valid_mask = np.ones(len(candidate_nodes), dtype=bool)
            for snode, t_safe in self.safe_observations.items():
                snode_noisy_stts = safe_base_stts[snode] * noise
                # 如果预测到达时间 < 观察到 Safe 的时间，说明不匹配
                # t_arrival_at_snode = T_start_at_source + STT(source, snode)
                # We don't know T_start_at_source exactly, but we know:
                # T_trigger = T_start_at_source + STT(source, trigger)
                # So T_start_at_source = T_trigger - STT(source, trigger)
                # T_arrival_at_snode = (T_trigger - STT(source, trigger)) + STT(source, snode)
                # Since we observe Snode at T_now (which is t_elapsed since T_trigger),
                # if T_arrival_at_snode < T_now, then Snode should be Positive.
                # If it's Safe, then this source is invalid.
                
                # In relative terms:
                # STT(source, snode) - STT(source, trigger) < t_elapsed
                # where t_elapsed = T_now - T_trigger
                
                arrival_diff = snode_noisy_stts - noisy_stts
                if (arrival_diff < t_elapsed).any():
                    # For nodes where arrival_diff < t_elapsed, they are invalid
                    valid_mask &= (arrival_diff >= t_elapsed)

            # --- D. 剧本胜出者 (H4 Ranking) ---
            if valid_mask.any():
                # 只在有效的候选者中选残差最小的
                masked_residuals = time_residuals.copy()
                masked_residuals[~valid_mask] = 1e10
                winner_idx = np.argmin(masked_residuals)
                winner_node = candidate_nodes[winner_idx]
                vote_counts[winner_node] += 1

        return vote_counts, candidate_nodes

    def get_action_hsr_hybrid_disagreement(self, k=3):
        """
        Hybrid Step-1 improvement:
        - keep the strongest current suspect as an exploitation slot
        - spend the remaining slots on candidate nodes whose expected binary state
          best splits the remaining plausible sources

        This stays HSR-style because it only uses:
        - the current candidate set
        - trigger-relative travel-time consistency
        - binary positive/negative observation semantics
        - an interpretable disagreement score over remaining source hypotheses
        """
        vote_counts, candidate_nodes = self._compute_hsr_vote_counts()
        if not candidate_nodes:
            return [self.get_action()]
        if self.trigger_sensor is None:
            return [self.get_action()]

        total_votes = float(sum(vote_counts.values()))
        if total_votes <= 0.0:
            sorted_suspects = sorted(candidate_nodes)
            return sorted_suspects[:k]

        sorted_by_vote = sorted(vote_counts.keys(), key=lambda n: vote_counts[n], reverse=True)
        weights = {int(node): float(vote_counts[int(node)]) / total_votes for node in sorted_by_vote}
        exploit_nodes = [int(sorted_by_vote[0])]
        if int(k) <= 1:
            return exploit_nodes[: int(k)]

        source_nodes = [int(v) for v in sorted_by_vote]
        trigger_stts = self.get_stt(source_nodes, self.trigger_sensor)
        t_elapsed = (self.current_time_step - self.t_start) * self.time_step_hours

        disagreement_rows = []
        for action_node in candidate_nodes:
            if int(action_node) in exploit_nodes:
                continue
            action_stts = self.get_stt(source_nodes, int(action_node))

            pos_mass = 0.0
            finite_mass = 0.0
            for idx, source_node in enumerate(source_nodes):
                w = float(weights[int(source_node)])
                trigger_stt = float(trigger_stts[idx])
                action_stt = float(action_stts[idx])
                if trigger_stt >= 1e8 or action_stt >= 1e8:
                    continue
                finite_mass += w
                arrival_diff = action_stt - trigger_stt
                if arrival_diff <= t_elapsed:
                    pos_mass += w

            if finite_mass <= 1e-12:
                disagreement = 0.0
                pos_ratio = 0.0
            else:
                pos_ratio = pos_mass / finite_mass
                disagreement = min(pos_ratio, 1.0 - pos_ratio)

            disagreement_rows.append(
                (
                    float(disagreement),
                    float(weights.get(int(action_node), 0.0)),
                    int(action_node),
                )
            )

        disagreement_rows.sort(reverse=True)
        chosen = list(exploit_nodes)
        for _disagreement, _self_weight, node in disagreement_rows:
            if int(node) not in chosen:
                chosen.append(int(node))
            if len(chosen) >= int(k):
                break

        if len(chosen) < int(k):
            for node in sorted_by_vote:
                if int(node) not in chosen:
                    chosen.append(int(node))
                if len(chosen) >= int(k):
                    break
        return chosen[: int(k)]

    def get_action(self, candidate_mask=None):
        """
        Select next node to sample using Centroid Greedy strategy.
        
        Args:
            candidate_mask: Optional mask (not used, we use internal candidate_set)
            
        Returns:
            int: Node index to sample
        """
        candidates = list(self.candidate_set - self.sampled_nodes)
        
        if not candidates:
            # If no candidates or all sampled, pick random unsampled
            unsampled = list(set(range(self.num_nodes)) - self.sampled_nodes)
            if unsampled:
                return np.random.choice(unsampled)
            else:
                return 0 # All sampled
        
        # Centroid Greedy
        # Subgraph induced by candidates
        # If set is too large, use Degree. If small, use Betweenness?
        # 60k nodes -> Degree is O(1) if precomputed, or O(N) on subgraph.
        # Betweenness is O(VE).
        
        # Let's use Degree Centrality on the subgraph.
        # Subgraph: Nodes = candidates. Edges = induced.
        # This might be slow to build explicitly with NetworkX for 60k nodes.
        # Strategy:
        # 1. Global Degree Centrality (fast, static) restricted to candidates?
        # 2. Or "Degree in Candidate Set": For each u in C, count neighbors v also in C.
        
        # "Degree Centrality on the subgraph induced by candidate_set"
        
        if len(candidates) > 1000:
            # Approximation: Use global degree or just random?
            # Global degree is better than random.
            # But "Degree in Subgraph" means "How many other CANDIDATES am I connected to?"
            # This is important.
            
            # Fast calculation of subgraph degree:
            # Iterate edges? No.
            # Use G_rev (or forward graph) adj.
            # Since we have edge_index, we can do this fast with numpy if we had adjacency matrix.
            # But we are in NetworkX/Python.
            
            # Let's try simply picking the node with highest Global Degree among candidates
            # This is a good proxy and very fast.
            # Building subgraph for >1000 nodes is okay, but Betweenness is slow.
            # Degree on subgraph is fast.
            
            subgraph = self.G_rev.subgraph(candidates) # View
            degrees = dict(subgraph.degree())
            # Pick max
            best_node = max(degrees, key=degrees.get)
            return best_node
            
        else:
            # For small sets, use Betweenness Centrality (approximating Centroid)
            # This aligns with "Centroid Greedy"
            subgraph = self.G_rev.subgraph(candidates)
            try:
                # k=None means all pairs. fast for < 500 nodes.
                centrality = nx.betweenness_centrality(subgraph, weight='weight') 
                best_node = max(centrality, key=centrality.get)
                return best_node
            except:
                # Fallback to degree
                degrees = dict(subgraph.degree())
                if not degrees: return candidates[0]
                best_node = max(degrees, key=degrees.get)
                return best_node

    def get_belief(self):
        """
        Return binary mask of suspects.
        """
        mask = np.zeros(self.num_nodes, dtype=np.int32)
        if self.candidate_set:
            mask[list(self.candidate_set)] = 1
        return mask
    
    def predict(self):
        """
        Alias for get_belief as per instructions.
        """
        return self.get_belief()

# Unit Test
if __name__ == "__main__":
    print("Testing HSRAgent...", flush=True)
    try:
        agent = HSRAgent()
        
        # Mock Graph Data reset
        agent.reset()
        print("Reset complete.", flush=True)
        
        # Mock Step
        # Assume Node 100 is trigger.
        # Observation: Node 100 is Positive.
        obs = {100: (1.0, 1)}
        print("Stepping with obs 1...", flush=True)
        agent.step(obs)
        
        belief = agent.get_belief()
        print(f"Candidates after positive: {belief.sum()}", flush=True)
        
        # Mock Step 2
        # Node 200 (upstream of 100) is Negative.
        obs = {200: (0.0, 0)}
        print("Stepping with obs 2...", flush=True)
        agent.step(obs)
        
        belief = agent.get_belief()
        print(f"Candidates after negative: {belief.sum()}", flush=True)
        
        actions = agent.get_action_hsr_scalable(k=3)
        print(f"Next Actions (HSR-Scalable): {actions}", flush=True)
        
        action = agent.get_action()
        print(f"Next Action (Centroid): {action}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
