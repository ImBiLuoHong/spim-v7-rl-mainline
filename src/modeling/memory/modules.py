import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from src.modeling.interfaces.base import MemoryBase
from src.modeling.registry import MEMORY_REGISTRY

@MEMORY_REGISTRY.register("memory_none")
class NoMemory(MemoryBase):
    """
    Pure Markovian: No hidden state maintained.
    """
    def init_state(self, batch_size: int, device: torch.device) -> Any:
        return None

    def step_update(self, current_state: Any, step_ctx: Dict[str, Any]) -> Any:
        return None

@MEMORY_REGISTRY.register("memory_gru_global")
class GRUGlobalMemory(MemoryBase):
    """
    Maintains a global hidden state h_t for each graph in the batch.
    Updates h_t using a GRUCell based on pooled node embeddings.
    """
    def __init__(self, cfg):
        super().__init__()
        reason_cfg = getattr(cfg.model, 'reasoner', None)
        self.hidden_dim = getattr(reason_cfg, 'hidden_dim', getattr(cfg.model, 'hidden_dim', 128))
        # Assuming input to GRU is pooled node embeddings
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        
    def init_state(self, batch_size: int, device: torch.device) -> Any:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step_update(self, current_state: Any, step_ctx: Dict[str, Any]) -> Any:
        """
        step_ctx contains:
            'node_embeddings': [N, D]
            'batch': [N] (graph indices)
        """
        from torch_scatter import scatter_mean
        node_embeddings = step_ctx['node_embeddings']
        batch = step_ctx['batch']
        
        # Pool node embeddings to get graph-level summary
        graph_summary = scatter_mean(node_embeddings, batch, dim=0)
        
        # Update GRU state
        new_state = self.gru(graph_summary, current_state)
        return new_state

@MEMORY_REGISTRY.register("memory_belief_bank")
class BeliefBankMemory(MemoryBase):
    """
    (Placeholder) Minimal Belief Bank: EMA update over node embeddings.
    """
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = getattr(cfg.model, 'hidden_dim', 128)
        self.momentum = getattr(cfg.model, 'memory_momentum', 0.9)

    def init_state(self, batch_size: int, device: torch.device) -> Any:
        # We don't know N here, so we might need to initialize on the fly or 
        # return a zero tensor that gets updated.
        return None 

    def step_update(self, current_state: Any, step_ctx: Dict[str, Any]) -> Any:
        node_embeddings = step_ctx['node_embeddings']
        if current_state is None:
            return node_embeddings
        return self.momentum * current_state + (1 - self.momentum) * node_embeddings
