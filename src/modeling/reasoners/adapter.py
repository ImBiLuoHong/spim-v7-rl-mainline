import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class ReasonerInputAdapter(nn.Module):
    """
    [DEPRECATED] Manual Feature Construction.
    This class bypasses ObservationState and ObservationEncoder.
    It is marked for removal in System v3.
    """
    def __init__(self, cfg):
        super().__init__()
        import warnings
        warnings.warn(
            "ReasonerInputAdapter is deprecated and bypasses ObservationState contract. "
            "Use ObservationEncoder instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.cfg = cfg
        self.firewall_enabled = not getattr(cfg.model, 'disable_firewall', False)

    def forward(self, state: Dict[str, Any], graph: Any) -> Dict[str, Any]:
        """
        Args:
            state: Contains 'h_fused', 'causal_anchors', 'accumulated_mask', etc.
            graph: PyG graph object.
        Returns:
            Dict containing 'x_reasoner' and 'firewall_report'.
        """
        # 1. Extract raw signals
        # Assuming causal_anchors and accumulated_mask are already in state
        # These are usually [N, 1]
        causal_anchors = state.get('causal_anchors')
        accumulated_mask = state.get('accumulated_mask')
        
        # Freshness or other time-related features could be here
        # For now, let's use a standard 3-channel input: [Anchor, Mask, Freshness]
        # We'll mock freshness as 1.0 for sampled nodes for now if not provided
        freshness = state.get('freshness', accumulated_mask.clone())

        # [Refactor] Firewall REMOVED.
        # We trust the upstream (EpisodeStepper) to provide correctly zeroed/masked inputs.
        
        # Construct input features directly
        # x_anchor: [N, 1]
        # x_mask: [N, 1]
        # x_fresh: [N, 1]
        
        x_anchor = causal_anchors
        x_mask = accumulated_mask
        x_fresh = freshness
        
        x_reasoner = torch.cat([x_anchor, x_mask, x_fresh], dim=-1)
        
        # Firewall Report (Dummy)
        report = {
            'sampled_count': accumulated_mask.sum().item(),
            'leakage_detected': False 
        }

        return {
            'x_reasoner': x_reasoner,
            'firewall_report': report
        }
