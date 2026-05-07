import torch
import torch.nn as nn
from src.modeling.registry import REASONER_REGISTRY
from src.modeling.reasoners.bayesian import BayesianReasoner

@REASONER_REGISTRY.register("calibrated_v4_5")
class CalibratedReasoner(BayesianReasoner):
    """
    Slot 7 Variant: Calibrated Reasoner
    Extends BayesianReasoner with learnable temperature scaling for uncertainty calibration.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Learnable temperature (initialized to 1.0 or slightly higher for softening)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, state, graph, physics_ctx=None):
        # 1. Base Logic (Frozen Data Semantics)
        out = super().forward(state, graph, physics_ctx)
        
        # 2. Calibration Layer
        # Ensure temp is positive
        temp = torch.abs(self.temperature) + 1e-6
        
        if 'logits' in out:
            out['logits'] = out['logits'] / temp
            
        # Add metadata for audit
        out['temperature'] = temp
        
        return out
