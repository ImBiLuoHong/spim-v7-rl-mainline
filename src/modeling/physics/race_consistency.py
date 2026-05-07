import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from src.modeling.interfaces.base import PhysicsConsistencyBase
from src.modeling.registry import PHYSICS_REGISTRY
from src.modeling.components.physics import build_upstream_feasible_mask

@PHYSICS_REGISTRY.register("race_consistency_v4_5")
class RaceConsistencyModule(PhysicsConsistencyBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, physics_in: Dict[str, Any]) -> Dict[str, Any]:
        # Map physics_in back to the logic expected by legacy compute
        t_sim = physics_in['t_sim']
        anchor_type = physics_in['anchor_type']
        anchor_time = physics_in['anchor_time']
        edge_index = physics_in['edge_index']
        edge_stt = physics_in['edge_stt']
        fused_batch = physics_in['batch']
        
        physics_ctx = {}
        
        # [Diagnosis] Compute Feasible Mask (Physical Pruning)
        # We use Infinite Window (Reachability) for robustness in diagnostic mode
        if edge_index.numel() > 0:
            # anomaly_mask: [N] (Fused Nodes)
            anomaly_mask = (anchor_type > 0.5).float()
            
            # Check if any anomaly exists
            if anomaly_mask.sum() > 0:
                # We treat fused graph as a single large graph (is_batched=False)
                # edge_index is [2, E]
                # anomaly_mask is [N]. Reshape to [1, N] for function.
                
                # Note: build_upstream_feasible_mask expects edge_index [2, E] for single graph
                # and anomaly_mask [1, N]
                
                feasible = build_upstream_feasible_mask(
                    edge_index_flow=edge_index,
                    travel_time=edge_stt,
                    anomaly_mask=anomaly_mask.unsqueeze(0),
                    window_s=1e9 # Infinite window -> Reachability
                )
                physics_ctx['feasible_mask'] = feasible.squeeze(0) # [N]
            else:
                # No anomalies -> All nodes are feasible (or None?)
                # If no anomaly, we have no "Upstream".
                # But we might have "Downstream" constraints?
                # HeuristicSTT needs a Suspect Pool.
                # If no anomaly, Suspect Pool is All Physical Nodes?
                # Let's return Ones.
                physics_ctx['feasible_mask'] = torch.ones_like(anchor_type)
        else:
             physics_ctx['feasible_mask'] = torch.ones_like(anchor_type)

        if edge_index.numel() > 0:
            src, dst = edge_index
            t_curr = t_sim[fused_batch[src]].unsqueeze(-1)
            
            # anchor_type and anchor_time are already node-level in physics_in
            # But the legacy logic used edge-level from edge_attr.
            # Here we use node-level mapping.
            a_type = anchor_type[dst].unsqueeze(-1)
            a_time = anchor_time[dst].unsqueeze(-1)
            stt = edge_stt.unsqueeze(-1)
            
            # 1. Gating
            is_neg = (a_type < -0.5).float()
            gate = torch.sigmoid(((t_curr - a_time) - stt) / torch.abs(self.beta + 1e-6))
            physics_ctx['neg_gate'] = torch.where(is_neg > 0.5, gate, torch.ones_like(gate))
            
            # 2. Energy
            delta = (t_curr - a_time) - stt
            penalty_pos = F.relu(-delta)
            penalty_neg = F.relu(delta)
            e_edge = torch.where(a_type > 0.5, penalty_pos, penalty_neg)
            
            energy = torch.zeros((anchor_type.size(0), 1), device=anchor_type.device, dtype=anchor_type.dtype)
            energy.scatter_add_(0, src.unsqueeze(-1), e_edge)
            physics_ctx['race_energy'] = energy
            physics_ctx['logit_bias'] = -energy # Default bias
            
        return physics_ctx

    def capabilities(self) -> Dict[str, Any]:
        return {
            'provides_race_energy': True,
            'provides_bias': True,
            'provides_neg_gating': True
        }

    def compute(self, graph, t_sim):
        # Legacy support
        from src.modeling.physics.bricks import BaseBrick
        mock_brick = BaseBrick(self.cfg)
        physics_in = mock_brick._prepare_input(graph, t_sim)
        return self.forward(physics_in)
