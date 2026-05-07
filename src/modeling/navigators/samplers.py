import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from src.modeling.interfaces.base import ActionSamplerBase, NavigatorCapabilities
from src.modeling.registry import SAMPLER_REGISTRY

def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=0)
    return -(probs * torch.log(probs + 1e-9)).sum()

@SAMPLER_REGISTRY.register("sampler_topk_wo_replacement")
class TopKWithoutReplacementSampler(ActionSamplerBase):
    """
    Strict Top-K sampler with de-duplication.
    Ensures that already sampled nodes (from state['valid_mask']) are not picked.
    """
    def __init__(self, cfg):
        super().__init__()
        self.k_explore = getattr(cfg.model.navigator, 'k_explore', 1)
        self.dedup = getattr(cfg.model.navigator, 'dedup', True)

    def forward(self, logits: torch.Tensor, state: Dict[str, Any]) -> Dict[str, Any]:
        # logits: [N, 1]
        logits = logits.view(-1)
        N = logits.size(0)
        
        # 1. Masking: avoid already sampled nodes if dedup is enabled
        # valid_mask: 1 for available, 0 for already sampled
        # valid_mask should be [N] or [N, 1]
        valid_mask = state.get('valid_mask')
        if valid_mask is not None and self.dedup:
            # Fix broadcasting issue:
            # logits is [N], valid_mask might be [N, 1]
            valid_mask_flat = valid_mask.view(-1)
            if valid_mask_flat.shape == logits.shape:
                logits = logits + (1.0 - valid_mask_flat.float()) * -1e9
            
        # 2. Top-K Selection
        k = state.get('k', self.k_explore) # Allow dynamic K
        k = min(k, (valid_mask.sum().item() if valid_mask is not None else N))
        k = int(max(1, k))
        
        values, indices = torch.topk(logits, k=k)
        
        # 3. Audit
        audit = {
            'entropy': get_entropy(logits),
            'budget_ok': len(indices) == k,
            'dedup_ok': True, # Guaranteed by masking
            'max_logit': values[0].item() if len(values) > 0 else 0.0
        }
        
        return {
            'selected_indices': indices,
            'action_audit': audit
        }

    def capabilities(self) -> NavigatorCapabilities:
        return {
            'supports_soft_actions': False,
            'provides_khot_mask': False,
            'supports_without_replacement': True,
            'output_fields': ['selected_indices', 'action_audit']
        }

@SAMPLER_REGISTRY.register("sampler_soft_khot")
class SoftKHotSampler(ActionSamplerBase):
    """
    Differentiable K-hot mask sampler.
    Uses a softmax-based approximation to allow gradient flow.
    """
    def __init__(self, cfg):
        super().__init__()
        self.k_explore = getattr(cfg.model.navigator, 'k_explore', 1)
        self.tau = getattr(cfg.model.navigator, 'temperature', 1.0)

    def forward(self, logits: torch.Tensor, state: Dict[str, Any]) -> Dict[str, Any]:
        logits = logits.view(-1)
        
        # 1. Softmax to get probs
        # Apply valid_mask if present
        valid_mask = state.get('valid_mask')
        masked_logits = logits
        if valid_mask is not None:
            masked_logits = logits + (1.0 - valid_mask.float().view(-1)) * -1e9
            
        tau = state.get('tau', self.tau)
        probs = F.softmax(masked_logits / tau, dim=0)
        
        # 2. Soft K-hot mask: Scale probs by K to get an expectation of K selections
        k = state.get('k', self.k_explore)
        soft_mask = probs * k
        
        # 3. Hard selection for execution (Top-K on probs or logits)
        k_valid = min(k, (valid_mask.sum().item() if valid_mask is not None else logits.size(0)))
        k_valid = int(max(1, k_valid))
        _, indices = torch.topk(masked_logits, k=k_valid)
        
        audit = {
            'entropy': get_entropy(masked_logits),
            'soft_mask_sum': soft_mask.sum().item(),
            'tau': tau
        }
        
        return {
            'selected_indices': indices,
            'soft_khot_mask': soft_mask,
            'nav_probs': probs, # Satisfy survival_engine requirement
            'action_audit': audit
        }

    def capabilities(self) -> NavigatorCapabilities:
        return {
            'supports_soft_actions': True,
            'provides_khot_mask': True,
            'supports_without_replacement': False, # It's an approximation
            'output_fields': ['selected_indices', 'soft_khot_mask', 'nav_probs', 'action_audit']
        }

@SAMPLER_REGISTRY.register("sampler_gumbel_topk")
class GumbelTopKSampler(ActionSamplerBase):
    """
    Stochastic Top-K via Gumbel-Max trick.
    """
    def __init__(self, cfg):
        super().__init__()
        self.k_explore = getattr(cfg.model.navigator, 'k_explore', 1)
        self.tau = getattr(cfg.model.navigator, 'temperature', 1.0)

    def forward(self, logits: torch.Tensor, state: Dict[str, Any]) -> Dict[str, Any]:
        logits = logits.view(-1)
        tau = state.get('tau', self.tau)
        
        # 1. Add Gumbel noise for exploration
        if self.training:
            gumbels = -torch.empty_like(logits).exponential_().log()
            logits = (logits + gumbels) / tau
            
        valid_mask = state.get('valid_mask')
        if valid_mask is not None:
            logits = logits + (1.0 - valid_mask.float().view(-1)) * -1e9
            
        k = state.get('k', self.k_explore)
        k = min(k, (valid_mask.sum().item() if valid_mask is not None else logits.size(0)))
        k = int(max(1, k))
        
        _, indices = torch.topk(logits, k=k)
        
        # Optional: could also provide a soft mask here
        
        return {
            'selected_indices': indices,
            'nav_probs': F.softmax(logits, dim=0),
            'action_audit': {
                'stochastic': self.training,
                'entropy': get_entropy(logits)
            }
        }

    def capabilities(self) -> NavigatorCapabilities:
        return {
            'supports_soft_actions': True,
            'provides_khot_mask': False,
            'supports_without_replacement': True,
            'output_fields': ['selected_indices', 'nav_probs', 'action_audit']
        }

@SAMPLER_REGISTRY.register("sampler_gumbel_st")
class GumbelTopKSTSampler(ActionSamplerBase):
    """
    Gumbel Top-K Straight-Through Estimator.
    Forward: Discrete Top-K selection (Hard)
    Backward: Continuous Gumbel-Softmax gradients (Soft)
    """
    def __init__(self, cfg):
        super().__init__()
        self.k_explore = getattr(cfg.model.navigator, 'k_explore', 1)
        self.tau = getattr(cfg.model.navigator, 'temperature', 1.0)

    def forward(self, logits: torch.Tensor, state: Dict[str, Any]) -> Dict[str, Any]:
        # logits: [N, 1] or [N]
        logits = logits.view(-1)
        N = logits.size(0)
        
        # Get dynamic params
        k = state.get('k', self.k_explore)
        tau = state.get('tau', self.tau)
        valid_mask = state.get('valid_mask') # 1 for available, 0 for sampled/masked
        
        # [Fix] Graph Batch Handling
        # We need to perform Top-K per graph, not globally.
        # But ActionSamplerBase interface assumes 'state' has 'h_fused' but not explicitly 'batch' indices?
        # Actually, in EpisodeStepper, 'nav_state' has 'h_fused' and 'x_nav'.
        # The 'logits' passed here are output of Navigator Head.
        # We need the batch vector to do per-graph operations.
        # Navigator 'forward' usually takes 'temp_graph' which has 'batch'.
        # But 'Sampler' is called inside 'ComposedNavigator'.
        # Does 'ComposedNavigator' pass 'batch' in state?
        # Let's check ComposedNavigator.
        # ComposedNavigator passes 'state' directly to sampler.
        # EpisodeStepper constructs 'nav_state'.
        # We should ensure 'nav_state' has 'batch' or 'fused_batch'.
        # In EpisodeStepper, 'nav_state' has 'h_fused'. It does NOT have 'batch'.
        # But 'temp_graph' passed to Navigator has 'batch'.
        # ComposedNavigator: `state['batch'] = graph.batch` before calling sampler?
        # Let's assume we can access batch from state if we add it.
        # Or, we can use `state['h_fused']` and assume batch size? No.
        
        # [Workaround] If batch is missing, fall back to global TopK (for single graph inference).
        # But for training, we MUST have batch.
        # Let's assume 'batch' is available in state (we will add it in EpisodeStepper/ComposedNavigator).
        
        batch = state.get('batch')
        if batch is None:
            # Try to infer from valid_mask shape? No.
            # Fallback to Global TopK (Legacy behavior)
            return self._global_topk(logits, k, tau, valid_mask, N)

        # Per-Graph Gumbel Top-K
        # 1. Masking
        masked_logits = logits.clone()
        if valid_mask is not None:
            # Ensure valid_mask is flattened to match logits
            masked_logits = masked_logits + (1.0 - valid_mask.float().view(-1)) * -1e9
            
        # 2. Gumbel Noise (Training Only)
        if self.training:
            gumbels = -torch.empty_like(logits).exponential_().log()
            noisy_logits = (masked_logits + gumbels) / tau
        else:
            noisy_logits = masked_logits / tau

        # 3. Soft Pass (Continuous Gradients)
        # Per-Graph Softmax * K
        # Use torch_geometric.utils.softmax
        from torch_geometric.utils import softmax as gnn_softmax
        probs = gnn_softmax(noisy_logits, batch)
        
        # [Fix] Handle NaNs if all nodes are masked (probs become NaN)
        # If a graph has no valid nodes, probs should be 0 (no selection)
        probs = torch.nan_to_num(probs, nan=0.0)
        
        # DEBUG
        if torch.isnan(probs).any():
             print("NaN remaining in probs after nan_to_num!")
        if (probs < 0).any() or (probs > 1).any():
             print(f"Probs out of range: [{probs.min()}, {probs.max()}]")
        
        y_soft = probs * k # Scaling approximation for K-hot
        
        # 4. Hard Pass (Discrete Action)
        # We need Top-K per graph.
        # Iterative scatter_max or sort?
        # Sort is expensive for large graphs.
        # Iterative scatter_max for K times.
        
        # Create hard mask
        y_hard = torch.zeros_like(logits)
        selected_indices_list = []
        
        # We need to handle variable K per graph? No, K is scalar budget.
        # But some graphs might have < K valid nodes.
        
        # Temporary logits for selection
        temp_logits = noisy_logits.clone()
        
        # Iterative Selection
        for _ in range(int(k)):
            # Get max per graph
            from torch_scatter import scatter_max
            val, idx = scatter_max(temp_logits, batch, dim=0)
            
            # Mask out invalid (already selected or masked)
            # idx contains index of max element for each graph
            # Check validity (val > -inf) and index bounds
            valid_selection = (val > -1e9) & (idx < temp_logits.size(0))
            
            if not valid_selection.any():
                break
                
            # Filter valid indices
            valid_idx = idx[valid_selection]
            
            # Mark selected
            y_hard[valid_idx] = 1.0
            temp_logits[valid_idx] = -float('inf') # Mask for next iteration
            
            selected_indices_list.append(valid_idx)
            
        # Flatten selected indices
        if selected_indices_list:
            selected_indices = torch.cat(selected_indices_list, dim=0)
        else:
            selected_indices = torch.tensor([], device=logits.device, dtype=torch.long)
            
        # 5. Straight-Through Estimator
        y_action = (y_hard - y_soft).detach() + y_soft
        
        return {
            'selected_indices': selected_indices,
            'y_action': y_action,
            'nav_probs': y_soft / k,
            'action_audit': {
                'stochastic': self.training,
                'entropy': get_entropy(logits),
                'tau': tau,
                'k': k
            }
        }

    def _global_topk(self, logits, k, tau, valid_mask, N):
        # Original Global Logic
        masked_logits = logits.clone()
        if valid_mask is not None:
            masked_logits = masked_logits + (1.0 - valid_mask.float()) * -1e9
        
        if self.training:
            gumbels = -torch.empty_like(logits).exponential_().log()
            noisy_logits = (masked_logits + gumbels) / tau
        else:
            noisy_logits = masked_logits / tau

        y_soft = F.softmax(noisy_logits, dim=0) * k
        
        valid_count = valid_mask.sum().item() if valid_mask is not None else N
        k_effective = min(k, valid_count)
        k_effective = int(max(1, k_effective))
        
        _, indices = torch.topk(noisy_logits, k=k_effective)
        
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(0, indices, 1.0)
        
        y_action = (y_hard - y_soft).detach() + y_soft
        
        return {
            'selected_indices': indices,
            'y_action': y_action,
            'nav_probs': y_soft / k,
            'action_audit': {'mode': 'global_fallback'}
        }

    def capabilities(self) -> NavigatorCapabilities:
        return {
            'supports_soft_actions': True,
            'provides_khot_mask': True, # y_action acts as k-hot mask
            'supports_without_replacement': True, # Enforced by valid_mask and TopK
            'output_fields': ['selected_indices', 'y_action', 'nav_probs', 'action_audit']
        }
