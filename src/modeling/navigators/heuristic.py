import torch
import torch.nn as nn
from src.modeling.interfaces.base import NavigatorBase
from src.modeling.registry import NAVIGATOR_REGISTRY
from src.modeling.navigators.samplers import GumbelTopKSTSampler
from src.modeling.navigators.backbones import SageBackbone
from torch_scatter import scatter_sum, scatter_max, scatter_mean

@NAVIGATOR_REGISTRY.register("heuristic_stt")
class HeuristicSTTNavigator(NavigatorBase):
    """
    Heuristic Navigator:
    Supports multiple fixed sampling strategies:
    - 'stt_var': Variance of STT from suspect pool (Score(a) = Var(STT(s->a)))
    - 'time_split': Information Gain on Time (Score(a) = |<T| * |>T|)
    - 'hybrid': 0.5 * stt_var + 0.5 * time_split
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sampler = GumbelTopKSTSampler(cfg)
        self.backbone = SageBackbone(cfg)
        
        # [Config] Load sub-mode
        self.heuristic_mode = getattr(cfg.model, 'heuristic_sub_mode', 'stt_var')
        self.hybrid_alpha = getattr(cfg.model, 'heuristic_hybrid_alpha', 0.5)

    def forward(self, state, graph, physics_ctx=None, belief_ctx=None):
        valid_mask = state.get('valid_mask')
        edge_index = state.get('fused_edge_index')
        edge_attr = state.get('fused_edge_attr')
        
        if valid_mask is not None:
            N = valid_mask.size(0)
            device = valid_mask.device
        else:
            N = state['h_fused'].size(0)
            device = state['h_fused'].device

        logits = torch.randn(N, device=device) * 0.01 
        
        # [Physical Suspect Pool]
        is_suspect = torch.zeros(N, dtype=torch.bool, device=device)
        feasible_mask = None
        if physics_ctx is not None:
            feasible_mask = physics_ctx.get('feasible_mask')
            
        if feasible_mask is not None and feasible_mask.view(-1).size(0) == N:
            is_suspect = (feasible_mask.view(-1) > 0.5)
        elif valid_mask is not None:
            is_suspect = (valid_mask > 0.5)
        else:
            is_suspect = torch.ones(N, dtype=torch.bool, device=device)

        if edge_index is not None and edge_attr is not None and is_suspect.any():
             src, dst = edge_index
             mask_edges = is_suspect[src]
             
             if mask_edges.any():
                 valid_src = src[mask_edges]
                 valid_dst = dst[mask_edges]
                 
                 # STT (Ch0: LogMed) - Need raw STT? 
                 # LogMed = log(1 + STT). We can use it directly as proxy.
                 if edge_attr.dim() > 1 and edge_attr.size(1) > 0:
                     stt_val = edge_attr[mask_edges, 0]
                 else:
                     stt_val = torch.zeros(mask_edges.sum().item(), device=device)
                 
                 # --- H1: STT Variance ---
                 score_var = torch.zeros(N, device=device)
                 if self.heuristic_mode in ['stt_var', 'hybrid']:
                     ones = torch.ones_like(stt_val)
                     count = scatter_sum(ones, valid_dst, dim=0, dim_size=N)
                     sum_stt = scatter_sum(stt_val, valid_dst, dim=0, dim_size=N)
                     sum_stt_sq = scatter_sum(stt_val**2, valid_dst, dim=0, dim_size=N)
                     
                     mask_valid_a = (count > 1)
                     if mask_valid_a.any():
                         mean = sum_stt[mask_valid_a] / count[mask_valid_a]
                         mean_sq = sum_stt_sq[mask_valid_a] / count[mask_valid_a]
                         score_var[mask_valid_a] = mean_sq - mean**2

                 # --- H2: Time Split ---
                 score_split = torch.zeros(N, device=device)
                 if self.heuristic_mode in ['time_split', 'hybrid']:
                     # Get current T (t_sim)
                     # physics_ctx['t_sim'] is [B]. We need node-level T.
                     t_sim_batch = physics_ctx.get('t_sim', torch.zeros(1, device=device))
                     if t_sim_batch.size(0) > 1:
                         # Use batch mapping
                         fused_batch = state.get('batch')
                         if fused_batch is not None:
                             # Map t_sim to edges?
                             # valid_dst is node index.
                             t_node = t_sim_batch[fused_batch[valid_dst]]
                             
                             # LogMed STT vs LogMed T?
                             # STT in edge_attr is Log(1+T).
                             # So we compare with log(1 + t_sim).
                             t_log = torch.log1p(t_node)
                             
                             is_arrived = (stt_val <= t_log).float()
                             is_not_arrived = (stt_val > t_log).float()
                             
                             n_arrived = scatter_sum(is_arrived, valid_dst, dim=0, dim_size=N)
                             n_not = scatter_sum(is_not_arrived, valid_dst, dim=0, dim_size=N)
                             
                             score_split = n_arrived * n_not
                     else:
                         # Fallback single T
                         t_val = t_sim_batch.mean()
                         t_log = torch.log1p(t_val)
                         is_arrived = (stt_val <= t_log).float()
                         n_arrived = scatter_sum(is_arrived, valid_dst, dim=0, dim_size=N)
                         n_not = scatter_sum((stt_val > t_log).float(), valid_dst, dim=0, dim_size=N)
                         score_split = n_arrived * n_not

                 # --- Combine ---
                 final_score = torch.zeros(N, device=device)
                 if self.heuristic_mode == 'stt_var':
                     final_score = score_var
                 elif self.heuristic_mode == 'time_split':
                     final_score = score_split
                 elif self.heuristic_mode == 'hybrid':
                     # Normalize to [0, 1] per graph? Or global?
                     # Global max is simpler.
                     def norm(x):
                         mx = x.max()
                         return x / (mx + 1e-6)
                     
                     s1 = norm(score_var)
                     s2 = norm(score_split)
                     final_score = self.hybrid_alpha * s1 + (1 - self.hybrid_alpha) * s2
                 
                 # Assign to logits
                 logits = logits + final_score

        if valid_mask is not None:
             logits = logits + (1.0 - valid_mask.float().view(-1)) * -1e9
             
        sampler_out = self.sampler(logits, state)
        
        return {
            'logits': logits,
            'value': None,
            **sampler_out
        }

    def capabilities(self):
        return self.sampler.capabilities()
