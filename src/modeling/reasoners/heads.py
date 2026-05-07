import torch
import torch.nn as nn
from typing import Optional
from src.modeling.interfaces.base import ReasonerHeadBase
from src.modeling.registry import REASONER_HEAD_REGISTRY

@REASONER_HEAD_REGISTRY.register("reasoner_linear_head")
class LinearHead(ReasonerHeadBase):
    """
    Simple Linear Head for Reasoner.
    """
    def __init__(self, cfg):
        super().__init__()
        reason_cfg = getattr(cfg.model, 'reasoner', None)
        self.hidden_dim = getattr(reason_cfg, 'hidden_dim', getattr(cfg.model, 'hidden_dim', 128))
        self.proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, node_embeddings: torch.Tensor, memory_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_embeddings: [N, D]
            memory_state: [Batch, D] (global memory)
        """
        logits = self.proj(node_embeddings)
        
        # If global memory exists, we could use it to bias logits, 
        # but for a 'linear_head' we keep it simple.
        return logits
