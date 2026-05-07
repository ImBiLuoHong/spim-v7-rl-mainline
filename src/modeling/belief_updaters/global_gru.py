import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from src.modeling.interfaces.belief_updater import BeliefUpdaterBase, BeliefStateBase, BeliefUpdaterCapabilities
from src.modeling.registry import BELIEF_UPDATER_REGISTRY

@dataclass
class GlobalGRUState(BeliefStateBase):
    h_t: torch.Tensor  # [B, H]
    step_count: int = 0

    def detach(self):
        return GlobalGRUState(h_t=self.h_t.detach(), step_count=self.step_count)
    
    def to(self, device: torch.device):
        return GlobalGRUState(h_t=self.h_t.to(device), step_count=self.step_count)

@BELIEF_UPDATER_REGISTRY.register("global_gru")
class BeliefGlobalGRU(BeliefUpdaterBase):
    """
    Maintains a global hidden state h_t based on statistical features.
    """
    def __init__(self, hidden_dim: int, input_keys: List[str], detach_every: int = 0, clamp_norm: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_keys = input_keys
        self.detach_every = detach_every
        self.clamp_norm = clamp_norm
        
        # Simple MLP to process input features into a fixed-size vector for GRU
        self.input_dim = len(input_keys)
        self.gru_cell = nn.GRUCell(self.input_dim, hidden_dim)

    def init_state(self, batch_size: int, num_nodes: Optional[int] = None, device: Optional[torch.device] = None) -> GlobalGRUState:
        device = device or torch.device('cpu')
        return GlobalGRUState(
            h_t=torch.zeros(batch_size, self.hidden_dim, device=device),
            step_count=0
        )

    def _step_impl(self, state: GlobalGRUState, step_in: Dict[str, Any]) -> Tuple[GlobalGRUState, Dict[str, Any]]:
        # 1. Extract and normalize input features
        input_features = []
        for key in self.input_keys:
            feat = self._extract_feature(key, step_in)
            input_features.append(feat)
        
        x_t = torch.stack(input_features, dim=-1) # [B, len(input_keys)]
        
        # 2. GRU Update
        new_h_t = self.gru_cell(x_t, state.h_t)
        
        # 3. Clamp norm if configured
        if self.clamp_norm > 0:
            new_h_t = torch.clamp(new_h_t, -self.clamp_norm, self.clamp_norm)
        
        # 4. Detach if configured
        detach_applied = False
        new_step_count = state.step_count + 1
        if self.detach_every > 0 and new_step_count % self.detach_every == 0:
            new_h_t = new_h_t.detach()
            detach_applied = True
            
        new_state = GlobalGRUState(h_t=new_h_t, step_count=new_step_count)
        
        # 5. Build belief_ctx
        belief_ctx = {
            'global_h': new_h_t,
            'summary_features': x_t,
            'audit': {
                't_sim': step_in.get('t_sim'),
                'step_idx': step_in.get('step_idx'),
                'state_norm_mean': new_h_t.norm(dim=-1).mean().item(),
                'state_norm_max': new_h_t.norm(dim=-1).max().item(),
                'detach_applied': detach_applied
            }
        }
        return new_state, belief_ctx

    def _extract_feature(self, key: str, step_in: Dict[str, Any]) -> torch.Tensor:
        """Extracts statistical features from step_in."""
        batch_size = step_in['t_sim'].size(0)
        device = self.gru_cell.weight_ih.device
        
        if key == 'entropy':
            val = step_in.get('fov_params', {}).get('entropy', 0.0)
            return torch.full((batch_size,), val, device=device)
        
        # Default fallback
        return torch.zeros(batch_size, device=device)

    def capabilities(self) -> BeliefUpdaterCapabilities:
        return {
            'provides_node_belief': False,
            'provides_global_belief': True,
            'provides_memory_bank': False,
            'output_fields': ['global_h']
        }
