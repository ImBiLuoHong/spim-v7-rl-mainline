import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from src.modeling.interfaces.belief_updater import BeliefUpdaterBase, BeliefStateBase, BeliefUpdaterCapabilities
from src.modeling.registry import BELIEF_UPDATER_REGISTRY

@dataclass
class NodeGRUState(BeliefStateBase):
    node_h: torch.Tensor  # [B, N, H] or [TotalNodes, H]
    step_count: int = 0

    def detach(self):
        return NodeGRUState(node_h=self.node_h.detach(), step_count=self.step_count)
    
    def to(self, device: torch.device):
        return NodeGRUState(node_h=self.node_h.to(device), step_count=self.step_count)

@BELIEF_UPDATER_REGISTRY.register("node_gru")
class BeliefNodeGRU(BeliefUpdaterBase):
    """
    Maintains node-level state H_t based on node-wise features.
    """
    def __init__(self, hidden_dim: int, input_dim: int, detach_every: int = 0, clamp_norm: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.detach_every = detach_every
        self.clamp_norm = clamp_norm
        
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, num_nodes: Optional[int] = None, device: Optional[torch.device] = None) -> NodeGRUState:
        if num_nodes is None:
            raise ValueError("NodeGRU requires num_nodes for initialization")
        device = device or torch.device('cpu')
        return NodeGRUState(
            node_h=torch.zeros(num_nodes, self.hidden_dim, device=device),
            step_count=0
        )

    def _step_impl(self, state: NodeGRUState, step_in: Dict[str, Any]) -> Tuple[NodeGRUState, Dict[str, Any]]:
        # 1. Construct node-wise input
        # Feature Firewall: node_embeddings must not contain raw ground truth
        x_t = step_in.get('node_embeddings')
        if x_t is None:
            # Fallback to zero if not provided
            x_t = torch.zeros(state.node_h.size(0), self.input_dim, device=state.node_h.device)
        
        if x_t.size(-1) != self.input_dim:
            # Projection if dims don't match (in a real system we'd use a dedicated layer)
            if not hasattr(self, 'proj'):
                self.proj = nn.Linear(x_t.size(-1), self.input_dim).to(x_t.device)
            x_t = self.proj(x_t)

        # 2. GRU Update
        new_node_h = self.gru_cell(x_t, state.node_h)
        
        # 3. Clamp norm if configured
        if self.clamp_norm > 0:
            new_node_h = torch.clamp(new_node_h, -self.clamp_norm, self.clamp_norm)
        
        # 4. Detach if configured
        detach_applied = False
        new_step_count = state.step_count + 1
        if self.detach_every > 0 and new_step_count % self.detach_every == 0:
            new_node_h = new_node_h.detach()
            detach_applied = True
            
        new_state = NodeGRUState(node_h=new_node_h, step_count=new_step_count)
        
        # 5. Build belief_ctx
        belief_ctx = {
            'node_h': new_node_h,
            'audit': {
                't_sim': step_in.get('t_sim'),
                'step_idx': step_in.get('step_idx'),
                'state_norm_mean': new_node_h.norm(dim=-1).mean().item(),
                'state_norm_max': new_node_h.norm(dim=-1).max().item(),
                'detach_applied': detach_applied
            }
        }
        return new_state, belief_ctx

    def capabilities(self) -> BeliefUpdaterCapabilities:
        return {
            'provides_node_belief': True,
            'provides_global_belief': False,
            'provides_memory_bank': False,
            'output_fields': ['node_h']
        }
