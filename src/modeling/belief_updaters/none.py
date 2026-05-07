import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from src.modeling.interfaces.belief_updater import BeliefUpdaterBase, BeliefStateBase, BeliefUpdaterCapabilities
from src.modeling.registry import BELIEF_UPDATER_REGISTRY

@dataclass
class NoneBeliefState(BeliefStateBase):
    def detach(self):
        return self
    
    def to(self, device: torch.device):
        return self

@BELIEF_UPDATER_REGISTRY.register("none")
class BeliefNone(BeliefUpdaterBase):
    """
    No-op belief updater for ablation.
    """
    def init_state(self, batch_size: int, num_nodes: Optional[int] = None, device: Optional[torch.device] = None) -> NoneBeliefState:
        return NoneBeliefState()

    def _step_impl(self, state: NoneBeliefState, step_in: Dict[str, Any]) -> Tuple[NoneBeliefState, Dict[str, Any]]:
        belief_ctx = {
            'audit': {
                't_sim': step_in.get('t_sim'),
                'step_idx': step_in.get('step_idx'),
                'state_norm_mean': 0.0,
                'state_norm_max': 0.0,
                'detach_applied': False
            }
        }
        return state, belief_ctx

    def capabilities(self) -> BeliefUpdaterCapabilities:
        return {
            'provides_node_belief': False,
            'provides_global_belief': False,
            'provides_memory_bank': False,
            'output_fields': []
        }
