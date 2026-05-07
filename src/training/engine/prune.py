import os
import torch
from src.modeling.components.physics import build_upstream_feasible_mask, write_feasible_stats


def apply_upstream_prune(class_logits: torch.Tensor, batch, model, batch_idx: int, logs_dir: str) -> torch.Tensor:
    """Apply upstream feasibility pruning to classification logits if enabled.
    - Uses model.prune_upstream_enabled and model.prune_upstream_window_s.
    - Reads edge_index_flow, travel_time, and anomaly/subgraph targets from batch when available.
    - Writes feasibility stats on first batch.
    Returns masked class_logits (or original on disable/invalid input).
    """
    try:
        prune_enabled = bool(getattr(model, 'prune_upstream_enabled', False))
        if not prune_enabled:
            return class_logits
        if not isinstance(batch, dict):
            if batch_idx == 0:
                print("[CAUSAL][SKIP][PRUNE] 非字典 batch，无法访问因果键，跳过硬裁剪")
            return class_logits
        dev = class_logits.device
        ei_flow = batch.get('edge_index_flow')
        tt = batch.get('travel_time')
        anom = batch.get('anomaly_nodes', batch.get('subgraph_targets'))
        if isinstance(ei_flow, torch.Tensor):
            ei_flow = ei_flow.to(dev, non_blocking=True)
        if isinstance(tt, torch.Tensor):
            tt = tt.to(dev, non_blocking=True)
        if isinstance(anom, torch.Tensor):
            anom = anom.to(dev, non_blocking=True).to(torch.float32)
        window_s = float(getattr(model, 'prune_upstream_window_s', 0.0) or 0.0)
        if window_s <= 0 or (not isinstance(ei_flow, torch.Tensor)) or (not isinstance(anom, torch.Tensor)):
            if batch_idx == 0:
                print("[CAUSAL][SKIP][PRUNE] 缺少 edge_index_flow/anomaly_nodes 或 window_s<=0，跳过硬裁剪")
            return class_logits
        feasible = build_upstream_feasible_mask(ei_flow, tt, anom, window_s=window_s)  # [B,N]
        if batch_idx == 0:
            try:
                kept = int(feasible.sum().item())
                ratio = float(kept / max(1, feasible.numel()))
                has_tt = isinstance(tt, torch.Tensor)
                _ = write_feasible_stats(logs_dir or '', kept, ratio, window_s, has_tt)
                print(f"[PRUNE] window_s={window_s} kept={kept} ratio={ratio:.6f} has_tt={has_tt}")
            except Exception:
                pass
        mask = (feasible > 0.5)
        neg_inf = torch.finfo(class_logits.dtype).min if class_logits.dtype.is_floating_point else -1e9
        return class_logits.masked_fill(~mask, neg_inf)
    except Exception:
        return class_logits

