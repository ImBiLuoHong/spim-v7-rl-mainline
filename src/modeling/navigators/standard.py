import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from src.modeling.interfaces.base import NavigatorBase
from src.modeling.registry import NAVIGATOR_REGISTRY, NAV_BACKBONE_REGISTRY
from src.modeling.navigators.samplers import GumbelTopKSTSampler

from torch_scatter import scatter_max

@NAVIGATOR_REGISTRY.register("standard_v4_5")
class StandardNavigator(NavigatorBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nav_cfg = cfg.model.navigator
        hidden_dim = nav_cfg.get('hidden_dim', 64) if isinstance(nav_cfg, dict) else getattr(nav_cfg, 'hidden_dim', 64)
        
        # Backbone Selection (SSOT)
        backbone_type = nav_cfg.get('backbone_type', 'sage_backbone') if isinstance(nav_cfg, dict) else getattr(nav_cfg, 'backbone_type', 'sage_backbone')
        try:
            backbone_cls = NAV_BACKBONE_REGISTRY.get(backbone_type)
            self.backbone = backbone_cls(cfg)
        except KeyError:
            # Fallback for compatibility or if not registered
            from src.modeling.navigators.backbones import SageBackbone
            self.backbone = SageBackbone(cfg)
        
        nav_state_cfg = getattr(cfg.model, 'nav_state_summary', {})
        nav_state_dim = int(nav_state_cfg.get('dim', 0)) if nav_state_cfg.get('enabled', False) else 0
        proxy_ig_enabled = bool(getattr(cfg.model, 'proxy_ig', {}).get('enabled', False))
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + nav_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim + nav_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ) if proxy_ig_enabled else None
        
        # Add Sampler for Action Selection
        self.sampler = GumbelTopKSTSampler(cfg)
        
        # [System v2] Policy Interface Config
        # Fallback to dict get or attribute access
        if isinstance(nav_cfg, dict):
            self.use_evidence = nav_cfg.get('use_evidence', False)
            self.evidence_mode = nav_cfg.get('evidence_mode', 'bias')
            self.support_scale = nav_cfg.get('support_bias_scale', 1.0)
            self.suspect_scale = nav_cfg.get('suspect_bias_scale', 0.0)
            self.contradiction_scale = nav_cfg.get('contradiction_bias_scale', 0.0)
            self.uncertainty_scale = nav_cfg.get('uncertainty_bias_scale', 0.5)
        else:
            self.use_evidence = getattr(nav_cfg, 'use_evidence', False)
            self.evidence_mode = getattr(nav_cfg, 'evidence_mode', 'bias')
            self.support_scale = getattr(nav_cfg, 'support_bias_scale', 1.0)
            self.suspect_scale = getattr(nav_cfg, 'suspect_bias_scale', 0.0)
            self.contradiction_scale = getattr(nav_cfg, 'contradiction_bias_scale', 0.0)
            self.uncertainty_scale = getattr(nav_cfg, 'uncertainty_bias_scale', 0.5)

    def forward(self, state, graph, physics_ctx=None, belief_ctx=None):
        # [SSOT Fix] Pass edge_attr to backbone to enable physics-aware GNN
        h = self.backbone(state['x_nav'], graph.edge_index, edge_attr=getattr(graph, 'edge_attr', None))
        
        # Fusion logic (simplified for interface)
        # In a real scenario, this would be part of a larger state object
        h_fused = state.get('h_fused', h) 
        nav_state_summary = state.get('nav_state_summary')
        
        # [Cleanup Phase B] Removed Legacy Belief Injection
        # Use EvidenceState interface for policy modulation.
        
        x = h_fused
        if nav_state_summary is not None:
            # Handle potential None in nav_state_summary or dimension mismatch
            if isinstance(nav_state_summary, torch.Tensor):
                # If sizes match (e.g. node-level state), cat directly
                if nav_state_summary.size(0) == h_fused.size(0):
                    x = torch.cat([h_fused, nav_state_summary], dim=1)
                # If sizes mismatch (e.g. graph-level state [B, D] vs [N, D]), expand using batch index
                elif hasattr(graph, 'batch') and graph.batch is not None:
                    # Broadcast graph state to nodes
                    nav_state_summary_expanded = nav_state_summary[graph.batch]
                    x = torch.cat([h_fused, nav_state_summary_expanded], dim=1)
                else:
                    # Fallback or error if no batch info
                    # Try to broadcast if size(0) is 1? No, B is usually > 1.
                    pass
            
        logits = self.mlp(x)
        
        # Apply Logit Bias from physics_ctx if available
        if physics_ctx and 'logit_bias' in physics_ctx:
            logits = logits + physics_ctx['logit_bias']
            
        # [System v2] Policy Interface: Evidence Integration
        if self.use_evidence:
            ev_state = state.get('evidence_state')
            if ev_state is not None:
                logits = self._apply_evidence_bias(logits, ev_state)
            
        value = self.value_mlp(x) if self.value_mlp is not None else None
        
        # Perform Sampling
        sampler_out = self.sampler(logits, state)
        
        return {
            'logits': logits, 
            'value': value,
            **sampler_out
        }

    def _apply_evidence_bias(self, logits, ev_state):
        """
        Apply Soft Bias from EvidenceState to Navigator Logits.
        Contract: Evidence fields must be in Fused Space aligned with logits.
        """
        def get_ev(attr_name):
            if isinstance(ev_state, dict):
                return ev_state.get(attr_name, None)
            return getattr(ev_state, attr_name, None)

        def apply_bias(field_name, tensor, scale, sign=1.0):
            if tensor is None or abs(float(scale)) <= 1e-12:
                return logits
            # Strict Contract Check
            if tensor.size(0) != logits.size(0):
                raise ValueError(
                    f"Policy Interface Error: {field_name} dimension {tensor.shape} "
                    f"does not match Navigator logits {logits.shape}. "
                    "Evidence must be in Fused Space."
                )
            bias = tensor.view_as(logits) * float(scale) * float(sign)
            return logits + bias

        # 1. Support Score (Mainline Evidence)
        logits = apply_bias('support_score', get_ev('support_score'), self.support_scale, sign=1.0)

        # 2. Suspect Pool (Optional Soft Prior)
        logits = apply_bias('suspect_pool', get_ev('suspect_pool'), self.suspect_scale, sign=1.0)

        # 3. Contradiction Score (Auxiliary Compare Only)
        logits = apply_bias('contradiction_score', get_ev('contradiction_score'), self.contradiction_scale, sign=-1.0)

        # 4. Uncertainty Gap (Exploration Prior)
        logits = apply_bias('uncertainty_gap', get_ev('uncertainty_gap'), self.uncertainty_scale, sign=1.0)

        return logits
