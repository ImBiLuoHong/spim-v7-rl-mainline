import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class DynamicFeatureGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gate_cfg = getattr(cfg.model, 'dynamic_gate', {})
        self.mode = gate_cfg.get('mode', 'off')
        self.target_indices = gate_cfg.get('target_indices', [1, 2, 3])
        self.context_dim = gate_cfg.get('context_dim', 16)
        
        if self.mode == 'learnable':
            input_dim = len(self.target_indices)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.context_dim),
                nn.ReLU(),
                nn.Linear(self.context_dim, input_dim),
                nn.Sigmoid()
            )
            
            # Initialize bias to start open (g close to 1)
            # Sigmoid(2.0) ~= 0.88
            nn.init.constant_(self.mlp[-2].bias, 2.0) 

    def forward(self, x, batch_idx):
        """
        Args:
            x: [N, C] input features (x_nav)
            batch_idx: [N] batch indices
        Returns:
            x_gated: [N, C]
            gate_info: dict containing 'gate_values' (B, 3) or None
        """
        if self.mode == 'off':
            return x, {}
            
        if self.mode == 'fixed':
            mask = torch.ones_like(x)
            mask[:, self.target_indices] = 0.0
            return x * mask, {}
            
        if self.mode == 'learnable':
            # 1. Extract dynamic features
            x_dyn = x[:, self.target_indices] # [N, 3]
            
            # 2. Global Pooling (Context)
            # We use mean of absolute values to capture "magnitude/presence"
            x_dyn_abs = torch.abs(x_dyn)
            # Handle empty batch case if necessary, but batch_idx should be fine
            ctx = scatter_mean(x_dyn_abs, batch_idx, dim=0) # [B, 3]
            
            # 3. Predict Gate
            g = self.mlp(ctx) # [B, 3] in [0, 1]
            
            # 4. Apply Gate
            # Broadcast g back to N
            g_broadcast = g[batch_idx] # [N, 3]
            
            # [Fix] Out-of-place update to avoid in-place gradient errors
            # x_out = x.clone()
            # x_out[:, self.target_indices] = x_dyn * g_broadcast
            
            # Construct Safe Mask
            mask = torch.ones_like(x)
            mask[:, self.target_indices] = 0.0
            x_base = x * mask
            
            # Construct Update
            x_update = torch.zeros_like(x)
            x_update[:, self.target_indices] = x_dyn * g_broadcast
            
            x_out = x_base + x_update
            
            # 5. Return info for regularization
            return x_out, {'gate_values': g}
            
        return x, {}
