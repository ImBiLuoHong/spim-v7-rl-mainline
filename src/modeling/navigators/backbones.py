import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GPSConv
from src.modeling.interfaces.base import NavigatorBackboneBase
from src.modeling.registry import NAV_BACKBONE_REGISTRY

def get_input_dim(cfg):
    # SSOT V6: Read from config
    if hasattr(cfg.model, 'input_dim'):
        input_dim = cfg.model.input_dim
    else:
        # Logic extracted from standard.py
        disable_firewall = getattr(cfg.model, 'disable_firewall', False)
        # [Protocol Update] Firewall now allows 7 channels: 0-6
        input_dim = 10 if disable_firewall else 7
    
    hint_cfg = getattr(getattr(cfg, 'features', object()), 'evidence_hint', {})
    if isinstance(hint_cfg, dict) and hint_cfg.get('enabled', False):
        input_dim += int(hint_cfg.get('dim', 0))
        
    state_cfg = getattr(getattr(cfg, 'interaction', object()), 'state_transition', {})
    if isinstance(state_cfg, dict) and state_cfg.get('enabled', False):
        input_dim += int(state_cfg.get('h_dim', 0)) + 1
    return input_dim

@NAV_BACKBONE_REGISTRY.register("sage_backbone")
class SageBackbone(NavigatorBackboneBase):
    def __init__(self, cfg):
        super().__init__()
        in_dim = get_input_dim(cfg)
        nav_cfg = cfg.model.navigator
        hidden_dim = nav_cfg.get('hidden_dim', 64) if isinstance(nav_cfg, dict) else getattr(nav_cfg, 'hidden_dim', 64)
        layers = nav_cfg.get('layers', 2) if isinstance(nav_cfg, dict) else getattr(nav_cfg, 'layers', 2)
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        # [SSOT Fix] Edge Attribute Injection (Physics-Aware)
        self.edge_dim = getattr(cfg.model, 'edge_dim', 8) 
        self.edge_encoder = nn.Linear(self.edge_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # [SSOT Fix] Inject Edge Physics (STT, Diameter, Length, etc.)
        edge_agg = None
        if edge_attr is not None:
             # Ensure edge_attr dimension matches (or project)
             if edge_attr.size(-1) == self.edge_dim:
                 edge_emb = self.edge_encoder(edge_attr)
                 # Aggregate to target nodes
                 from torch_scatter import scatter_mean
                 edge_agg = scatter_mean(edge_emb, edge_index[1], dim=0, dim_size=x.size(0))
             else:
                 pass

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Inject AFTER first layer (when x is hidden_dim)
            if i == 0 and edge_agg is not None:
                x = x + edge_agg
                
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return self.bn(x)

@NAV_BACKBONE_REGISTRY.register("gru_backbone")
class GRUBackbone(NavigatorBackboneBase):
    """
    Backbone with Global GRU Memory (Exp E + F).
    Combines SAGE encoding with a Graph-Level GRU update.
    Accepts explicit context (Entropy, Action History) if provided.
    """
    def __init__(self, cfg):
        super().__init__()
        in_dim = get_input_dim(cfg)
        nav_cfg = cfg.model.navigator
        hidden_dim = nav_cfg.get('hidden_dim', 64) if isinstance(nav_cfg, dict) else getattr(nav_cfg, 'hidden_dim', 64)
        layers = nav_cfg.get('layers', 2) if isinstance(nav_cfg, dict) else getattr(nav_cfg, 'layers', 2)
        
        # 1. Base Encoder (Deep Capacity SAGE)
        self.encoder = nn.ModuleList()
        self.encoder.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
        for _ in range(layers - 1):
            self.encoder.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            
        # 2. Global Memory Core
        self.memory_dim = hidden_dim
        # Use scatter_mean manually, AdaptiveAvgPool1d doesn't work with batch vector properly in this context
        
        # [Exp F] Explicit Context Projector (Scalar -> Vector)
        # Context: Entropy (1) + MaxProb (1) + Budget (1) = 3 dims
        self.context_dim = 3 
        self.context_proj = nn.Sequential(
            nn.Linear(self.context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # GRU Input: GraphSummary (D) + ProjectedContext (16)
        self.gru_input_dim = hidden_dim + 16
        self.gru_cell = nn.GRUCell(self.gru_input_dim, hidden_dim) # Hidden: Memory
        
        # 3. Context Injector (Broadcast back to nodes)
        self.ctx_inject = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, memory_state=None, batch=None, explicit_context=None):
        # 1. Encode Nodes
        h = x
        for i, conv in enumerate(self.encoder):
            h = conv(h, edge_index)
            if i < len(self.encoder) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=0.2, training=self.training)
        
        # 2. Global Pooling (Readout)
        if batch is None:
            h_graph = h.mean(dim=0, keepdim=True)
        else:
            from torch_scatter import scatter_mean
            h_graph = scatter_mean(h, batch, dim=0) # [B, D]
            
        # 3. Process Explicit Context
        if explicit_context is None:
            # Default zero context if not provided
            batch_size = h_graph.size(0)
            explicit_context = torch.zeros(batch_size, self.context_dim, device=h.device)
            
        ctx_emb = self.context_proj(explicit_context) # [B, 16]
        
        # Concatenate Graph Summary + Context
        gru_input = torch.cat([h_graph, ctx_emb], dim=-1) # [B, D+16]
            
        # 4. GRU Update
        if memory_state is None:
            memory_state = torch.zeros_like(h_graph)
            
        h_memory_next = self.gru_cell(gru_input, memory_state)
        
        # 5. Inject Memory back to Nodes
        if batch is None:
            h_context = h_memory_next.expand(h.size(0), -1)
        else:
            h_context = h_memory_next[batch]
            
        # Fuse: Node + Context
        h_out = torch.cat([h, h_context], dim=-1)
        h_out = self.ctx_inject(h_out)
        h_out = F.elu(h_out)
        
        return self.bn(h_out), h_memory_next

@NAV_BACKBONE_REGISTRY.register("gps_backbone")
class GPSBackbone(NavigatorBackboneBase):
    """
    GraphGPS Backbone Placeholder.
    Uses a simple Message Passing + Global Attention as a baseline.
    """
    def __init__(self, cfg):
        super().__init__()
        in_dim = get_input_dim(cfg)
        nav_cfg = cfg.model.navigator
        hidden_dim = nav_cfg.get('hidden_dim', 64) if isinstance(nav_cfg, dict) else getattr(nav_cfg, 'hidden_dim', 64)
        
        # Minimal implementation using MPNN + Global Attention
        self.pre_linear = nn.Linear(in_dim, hidden_dim)
        self.mpnn = SAGEConv(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.pre_linear(x))
        # MPNN step
        x_local = self.mpnn(x, edge_index)
        # Global Attention step (simplified)
        # Note: In a real GPS, this would handle batches correctly.
        # Here we assume a single graph or use node-wise attention.
        x_global, _ = self.attn(x_local.unsqueeze(0), x_local.unsqueeze(0), x_local.unsqueeze(0))
        x = x_local + x_global.squeeze(0)
        return self.bn(x)
