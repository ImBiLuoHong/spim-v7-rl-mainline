import numpy as np
import networkx as nx

from hsr_agent import HSRAgent


class HSRSoftNegativeAgent(HSRAgent):
    """
    Step-1 heuristic variant:
    - keep positive evidence as hard candidate retention
    - turn negative evidence into a soft contradiction penalty instead of
      irreversible candidate deletion
    """

    def __init__(
        self,
        *args,
        safe_penalty_weight=1.0,
        mc_samples=50,
        noise_scale=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.safe_penalty_weight = float(safe_penalty_weight)
        self.mc_samples = int(mc_samples)
        self.noise_scale = float(noise_scale)

    def step(self, observation):
        """
        Update state from observations.

        Positive evidence still hard-prunes the candidate set.
        Negative evidence is recorded as a contradiction constraint and used
        later in action scoring instead of deleting candidates immediately.
        """
        self.current_time_step += 1
        current_t_hours = self.current_time_step * self.time_step_hours

        for node_idx, (signal, label) in observation.items():
            node_idx = int(node_idx)
            self.sampled_nodes.add(node_idx)

            is_positive = signal > 1e-6

            if is_positive:
                if self.trigger_sensor is None:
                    self.trigger_sensor = node_idx
                    self.t_start = self.current_time_step

                cutoff_dist = current_t_hours + self.epsilon
                valid_upstream = nx.single_source_dijkstra_path_length(
                    self.G_rev, node_idx, cutoff=cutoff_dist, weight="weight"
                )
                self.candidate_set.intersection_update(set(valid_upstream.keys()))
            else:
                self.safe_observations[node_idx] = current_t_hours

    def get_action_hsr_soft_negative(self, k=3):
        candidate_nodes = list(self.candidate_set - self.sampled_nodes)
        if not candidate_nodes:
            return [self.get_action()]

        if self.trigger_sensor is None:
            return [self.get_action()]

        vote_counts = {node: 0 for node in candidate_nodes}
        mean_costs = {node: 0.0 for node in candidate_nodes}
        t_elapsed = (self.current_time_step - self.t_start) * self.time_step_hours

        base_stts = self.get_stt(candidate_nodes, self.trigger_sensor)
        safe_nodes = list(self.safe_observations.keys())
        safe_base_stts = {}
        for snode in safe_nodes:
            safe_base_stts[snode] = self.get_stt(candidate_nodes, snode)

        for _ in range(self.mc_samples):
            noise = np.random.normal(loc=1.0, scale=self.noise_scale, size=len(candidate_nodes))
            noisy_stts = base_stts * noise
            time_residuals = np.abs(noisy_stts - t_elapsed)

            contradiction_fraction = np.zeros(len(candidate_nodes), dtype=np.float32)
            if safe_nodes:
                contradiction_count = np.zeros(len(candidate_nodes), dtype=np.float32)
                for snode in safe_nodes:
                    snode_noisy_stts = safe_base_stts[snode] * noise
                    arrival_diff = snode_noisy_stts - noisy_stts
                    contradiction_count += (arrival_diff < t_elapsed).astype(np.float32)
                contradiction_fraction = contradiction_count / float(len(safe_nodes))

            soft_cost = time_residuals + (self.safe_penalty_weight * contradiction_fraction)
            winner_idx = int(np.argmin(soft_cost))
            winner_node = candidate_nodes[winner_idx]
            vote_counts[winner_node] += 1

            for idx, node in enumerate(candidate_nodes):
                mean_costs[node] += float(soft_cost[idx])

        sorted_suspects = sorted(
            candidate_nodes,
            key=lambda node: (
                -vote_counts[node],
                mean_costs[node] / float(max(self.mc_samples, 1)),
                node,
            ),
        )
        return sorted_suspects[:k]
