import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, APPNP, GATv2Conv
from typing import Optional, Dict, Any
from src.modeling.interfaces.base import ReasonerBackboneBase
from src.modeling.registry import REASONER_BACKBONE_REGISTRY

@REASONER_BACKBONE_REGISTRY.register("reasoner_gnn_mpnn")
class MPNNBackbone(ReasonerBackboneBase):
    """
    Standard Message Passing GNN (using GATv2 for robustness).
    """
    def __init__(self, cfg):
        super().__init__()
        # Use reasoner specific hidden_dim if available, else global
        reason_cfg = getattr(cfg.model, 'reasoner', None)
        self.hidden_dim = getattr(reason_cfg, 'hidden_dim', getattr(cfg.model, 'hidden_dim', 128))
        
        self.input_dim = 3 # [Anchor, Mask, Freshness] from adapter
        # Output dim will be exactly self.hidden_dim
        self.conv1 = GATv2Conv(self.input_dim, self.hidden_dim // 4, heads=4)
        self.conv2 = GATv2Conv(self.hidden_dim, self.hidden_dim // 4, heads=4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

@REASONER_BACKBONE_REGISTRY.register("reasoner_appnp_backbone")
class APPNPBackbone(ReasonerBackboneBase):
    """
    APPNP Backbone: Strong global propagation.
    """
    def __init__(self, cfg):
        super().__init__()
        reason_cfg = getattr(cfg.model, 'reasoner', None)
        self.hidden_dim = getattr(reason_cfg, 'hidden_dim', getattr(cfg.model, 'hidden_dim', 128))
        self.input_dim = 3
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.prop = APPNP(K=10, alpha=0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.prop(x, edge_index)
        return x

@REASONER_BACKBONE_REGISTRY.register("reasoner_gps_min")
class GPSMinBackbone(ReasonerBackboneBase):
    """
    Minimal GPS-like: Local GNN + Global Attention.
    """
    def __init__(self, cfg):
        super().__init__()
        reason_cfg = getattr(cfg.model, 'reasoner', None)
        self.hidden_dim = getattr(reason_cfg, 'hidden_dim', getattr(cfg.model, 'hidden_dim', 128))
        self.input_dim = 3
        self.local_gnn = GCNConv(self.input_dim, self.hidden_dim)
        self.global_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Local Message Passing
        h_local = F.relu(self.local_gnn(x, edge_index))
        
        # 2. Global Attention (Simplified: Treat all nodes in batch as a sequence)
        # In a real GPS, this should be per-graph. 
        # For 'gps_min' dry-run, we use a simple global pooling/broadcast or mock it.
        # Here we just return local for now but keep the structure.
        return h_local

@REASONER_BACKBONE_REGISTRY.register("reasoner_gnn_deep")
class DeepGNNBackbone(ReasonerBackboneBase):
    """
    Deep GNN Backbone with configurable layers.
    """
    def __init__(self, cfg):
        super().__init__()
        reason_cfg = getattr(cfg.model, 'reasoner', None)
        # Default to 128
        self.hidden_dim = 128
        if reason_cfg:
            self.hidden_dim = getattr(reason_cfg, 'hidden_dim', getattr(cfg.model, 'hidden_dim', 128))
        else:
            self.hidden_dim = getattr(cfg.model, 'hidden_dim', 128)
            
        # Layers: Default to 5
        self.layers = 5
        if reason_cfg:
            if isinstance(reason_cfg, dict):
                self.layers = reason_cfg.get('layers', 5)
            else:
                self.layers = getattr(reason_cfg, 'layers', 5)
        
        self.input_dim = 3 # [Anchor, Mask, Freshness]
        
        self.convs = nn.ModuleList()
        # GATv2Conv with 4 heads, so output per head = hidden_dim // 4
        heads = 4
        out_per_head = self.hidden_dim // heads
        
        # First layer
        self.convs.append(GATv2Conv(self.input_dim, out_per_head, heads=heads))
        
        # Hidden layers
        for _ in range(self.layers - 1):
            self.convs.append(GATv2Conv(self.hidden_dim, out_per_head, heads=heads))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply activation and dropout for all layers except potentially the last one?
            # Usually we apply activation after all layers in GNN backbones before passing to Head.
            x = F.elu(x) # ELU is common with GAT
            x = F.dropout(x, p=0.2, training=self.training)
        return x
