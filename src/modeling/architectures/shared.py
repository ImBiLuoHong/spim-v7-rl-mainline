import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class TriggerInvariantFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, batch_idx, global_node_ids, scenario_ids, max_nodes_global=100000):
        node_scenario_ids = scenario_ids[batch_idx]
        unique_keys = node_scenario_ids * max_nodes_global + global_node_ids
        unique_sorted, inverse_indices = torch.unique(unique_keys, return_inverse=True, sorted=True)
        h_fused = scatter_mean(h, inverse_indices, dim=0)
        # Note: scenario_ids, global_ids, batch are discrete. Mean is dangerous if they are not uniform per fused node.
        # But in our problem, fused nodes are same physical node, so IDs should be identical.
        # However, floating point mean might introduce errors for large IDs.
        # Use scatter_max to be safe for IDs.
        from torch_scatter import scatter_max
        fused_scenario_ids = scatter_max(node_scenario_ids, inverse_indices, dim=0)[0]
        fused_global_ids = scatter_max(global_node_ids, inverse_indices, dim=0)[0]
        fused_batch = scatter_max(batch_idx, inverse_indices, dim=0)[0]
        return h_fused, inverse_indices, fused_scenario_ids, fused_global_ids, fused_batch
