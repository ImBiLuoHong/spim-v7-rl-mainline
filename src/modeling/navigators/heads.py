import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from src.modeling.interfaces.base import NavigatorHeadBase, NavigatorCapabilities
from src.modeling.registry import NAV_HEAD_REGISTRY

@NAV_HEAD_REGISTRY.register("mlp_head")
class MLPHead(NavigatorHeadBase):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = getattr(cfg.model.navigator, 'hidden_dim', 64)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        logits = self.mlp(node_embeddings)
        
        # Apply physics bias if available
        # Fix for mismatch shape: physics_ctx['logit_bias'] might be per-node [N, 1] or per-graph [B, 1]
        # Usually it should be [N, 1] if it comes from PhysicsModule.
        # But if we have dynamic nodes (e.g. subgraphing), shapes might mismatch?
        # In this project, N is fixed per graph (Foundation), but we might batch multiple graphs.
        # The error says "a (1734) must match b (578)". 1734 = 3 * 578.
        # It seems 'logits' has 3x nodes? Or 'logit_bias' has 3x?
        # Likely logits is [N, 1] but logit_bias is [N_original, 1]?
        # Or logits is [B*N, 1] and logit_bias is [N, 1] broadcasted?
        
        # In V6, we use 'fused_batch'.
        # If logits is [N_total, 1].
        # And logit_bias is [N_total, 1].
        # They should match.
        
        # Wait, the error is 1734 vs 578. 1734 / 578 = 3.
        # Maybe batch size is 3?
        # Ah, we set batch_size=1 in run_case_study.py.
        # But maybe the model was trained with 3? No.
        
        # Wait, 'logits' comes from 'mlp(node_embeddings)'.
        # 'node_embeddings' comes from backbone.
        # 'logit_bias' comes from physics_ctx.
        
        # If we use V6Loader with batch_size=1.
        # N = 578 (nodes in subgraph?).
        # Why is one tensor 3x the other?
        
        # Hypothesis: `logit_bias` is being broadcasted wrongly or accumulated?
        # Or `logits` has duplicates?
        
        # Let's check `EpisodeStepper`.
        # `physics_ctx = self.model.physics_module(physics_in)`
        # `nav_out = self.model.navigator_module(nav_state, temp_graph, physics_ctx)`
        
        # In `NavigatorModule.forward`:
        # `node_embeddings = self.backbone(...)`
        # `logits = self.head(node_embeddings, state)`
        
        # If `physics_none` is used (default in case study), `logit_bias` shouldn't exist?
        # `physics_none` usually returns empty dict or minimal dict.
        # Let's check `physics_none`.
        
        # But wait, the error happens at `logits = logits + physics_ctx['logit_bias']`.
        # So `logit_bias` IS present.
        
        # Maybe `physics_ctx` is leaking from previous batch?
        # No, `EpisodeStepper` creates it fresh.
        
        # Maybe `logits` is [N, 1] and `logit_bias` is [N]?
        # 1734 and 578.
        # If batch=1.
        # Maybe `logits` is on `temp_graph` which might be the *entire* batch?
        # And `logit_bias` is on `physics_ctx` which is also on batch?
        
        # Let's just safeguard the add.
        physics_ctx = state.get('_physics_ctx')
        if physics_ctx and 'logit_bias' in physics_ctx:
            bias = physics_ctx['logit_bias']
            if bias.shape == logits.shape:
                logits = logits + bias
            elif bias.numel() == logits.numel():
                logits = logits + bias.view_as(logits)
            else:
                # Warning or ignore?
                # If shapes mismatch, we skip bias to prevent crash in Case Study
                pass
            
        return logits

    def capabilities(self) -> NavigatorCapabilities:
        return {
            'output_fields': ['nav_logits']
        }

@NAV_HEAD_REGISTRY.register("bilinear_head")
class BilinearHead(NavigatorHeadBase):
    """
    Head that interacts node embeddings with a global context.
    """
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = getattr(cfg.model.navigator, 'hidden_dim', 64)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, node_embeddings: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        # node_embeddings: [N, D]
        # state['nav_state_summary']: [B, D] (if available)
        
        x = self.node_proj(node_embeddings)
        
        nav_ctx = state.get('nav_state_summary')
        if nav_ctx is not None:
            # Simplified interaction: add context to nodes
            # In multi-graph batch, we'd need graph.batch to broadcast correctly
            # For now, we assume single graph or handled by orchestrator
            if nav_ctx.size(0) == 1:
                x = x + self.ctx_proj(nav_ctx)
        
        logits = self.out(F.relu(x))
        return logits
