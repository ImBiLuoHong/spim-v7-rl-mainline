import torch
import torch.nn as nn
from src.modeling.interfaces.base import NavigatorBase
from src.modeling.registry import NAVIGATOR_REGISTRY
from src.modeling.navigators.samplers import GumbelTopKSTSampler
from src.modeling.navigators.backbones import SageBackbone

@NAVIGATOR_REGISTRY.register("random")
class RandomNavigator(NavigatorBase):
    """
    Random Navigator:
    Selects candidates uniformly at random from the valid set.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Dummy parameter to register as module
        self.dummy = nn.Parameter(torch.empty(0))
        self.sampler = GumbelTopKSTSampler(cfg)
        # [Fix] Provide backbone for Phase45Model feature fusion (Perception)
        self.backbone = SageBackbone(cfg)

    def forward(self, state, graph, physics_ctx=None, belief_ctx=None):
        valid_mask = state.get('valid_mask') # [N]
        if valid_mask is None:
            # Fallback if no mask provided (shouldn't happen in Stepper)
            N = graph.num_nodes if hasattr(graph, 'num_nodes') else state['h_fused'].size(0)
            device = state['h_fused'].device
        else:
            N = valid_mask.size(0)
            device = valid_mask.device
            
        # Generate Random Logits (Normal Distribution for robust tie-breaking)
        logits = torch.randn(N, device=device)
        
        # Apply Mask
        if valid_mask is not None:
             # valid_mask: 1=valid, 0=invalid
             # We set invalid to -inf
             logits = logits + (1.0 - valid_mask.float().view(-1)) * -1e9
             
        # Delegate to Sampler (handles K, batching, etc.)
        # Note: Sampler uses 'logits' to determine probabilities.
        # Random logits -> Random probabilities.
        sampler_out = self.sampler(logits, state)
        
        return {
            'logits': logits,
            'value': None,
            **sampler_out
        }

    def capabilities(self):
        return self.sampler.capabilities()
