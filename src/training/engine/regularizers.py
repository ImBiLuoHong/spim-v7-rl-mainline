import torch
from src.shared.logging.core import log_tv_stats
from src.modeling.components.physics import write_causal_stats
from src.modeling.components.physics import compute_causal_loss_static


def apply_coverage_mix(loss, p: torch.Tensor, m: torch.Tensor, model, soft_targets: torch.Tensor):
    """Apply coverage mixing to the loss if enabled.
    Returns: (loss_after, cov_term_value: float, cov_base_value: float)
    Prints nothing; caller is responsible for epoch-level averaging/printing.
    """
    try:
        if m is None:
            return loss, 0.0, 0.0
        eps = 1e-12
        K_for_cov = max(1.0, float(int(getattr(model, 'selector_target_k', 0) or m.shape[1])))
        cov_base = ((m * p).sum(dim=1) / K_for_cov).mean()
        cov_term = -torch.log(cov_base + eps)
        coverage_enabled = bool(getattr(model, 'loss_coverage_enabled', False))
        coverage_alpha = float(getattr(model, 'loss_coverage_alpha', 0.0) or 0.0)
        coverage_weight = float(getattr(model, 'loss_coverage_weight', 1.0) or 1.0)
        if coverage_enabled and coverage_alpha > 0.0 and coverage_weight > 0.0:
            try:
                ce_only = float(getattr(getattr(loss, 'fn', object()), 'last_ce', 0.0) or 0.0)
            except Exception:
                try:
                    ce_only = float(getattr(getattr(soft_targets, 'criterion', object()), 'last_ce', 0.0) or 0.0)
                except Exception:
                    # fallback recompute CE
                    hard_targets = soft_targets.argmax(dim=1).long()
                    ce_only = torch.nn.functional.cross_entropy(p.log(), hard_targets.long()).item()
            ce_tensor = torch.tensor(ce_only, dtype=torch.float32, device=p.device)
            total_mix = (1.0 - coverage_alpha) * ce_tensor + coverage_alpha * (coverage_weight * cov_term)
            loss = total_mix
        return loss, float(cov_term.detach().item()), float(cov_base.detach().item())
    except Exception:
        return loss, 0.0, 0.0


def apply_size_regularization(loss, m: torch.Tensor, model):
    """Apply size quadratic penalty if enabled.
    Returns: (loss_after, size_term_value: float)
    """
    try:
        size_enabled = bool(getattr(model, 'loss_size_enabled', False))
        size_lambda = float(getattr(model, 'loss_size_weight', 0.0) or 0.0)
        if size_enabled and size_lambda > 0.0 and (m is not None):
            K = int(getattr(model, 'selector_target_k', 0) or m.shape[1])
            size_term = ((m.sum(dim=1) - float(K)) ** 2).mean()
            loss = loss + size_lambda * size_term
            return loss, float(size_term.detach().item())
    except Exception:
        pass
    return loss, 0.0


def apply_tv_regularization(loss, m: torch.Tensor, batch, class_logits: torch.Tensor, model, batch_idx: int, logs_dir: str):
    """Apply TV regularization from edge_index_tv/w_tv if enabled.
    Returns: (loss_after, tv_term_value: float)
    """
    try:
        tv_enabled = bool(getattr(model, 'loss_tv_enabled', False))
        tv_lambda = float(getattr(model, 'loss_tv_weight', 0.0) or 0.0)
        if not (tv_enabled and tv_lambda > 0.0):
            return loss, 0.0
        dev = class_logits.device
        edge_index_tv = None
        w_tv = None
        if isinstance(batch, dict):
            edge_index_tv = batch.get('edge_index_tv')
            w_tv = batch.get('w_tv')
        if isinstance(edge_index_tv, torch.Tensor):
            edge_index_tv = edge_index_tv.to(dev, non_blocking=True).to(torch.long)
        if isinstance(w_tv, torch.Tensor):
            w_tv = w_tv.to(dev, non_blocking=True).to(torch.float32)
        if (edge_index_tv is None) or (w_tv is None) or (m is None):
            return loss, 0.0
        B = int(m.shape[0])
        tv_acc = 0.0
        if batch_idx == 0:
            try:
                _ = log_tv_stats(logs_dir or '', edge_index_tv, w_tv)
            except Exception:
                pass
        if edge_index_tv.dim() == 3:
            for b in range(B):
                ei_b = edge_index_tv[b]
                i = ei_b[:, 0].long(); j = ei_b[:, 1].long()
                w_b = torch.clamp(w_tv[b].view(-1), 0.0, 1.0).to(m.dtype)
                diff = torch.abs(m[b, i] - m[b, j])
                tv_acc += float((diff * w_b).mean().item())
            tv_term = torch.tensor(tv_acc / max(1, B), device=dev)
        elif edge_index_tv.dim() == 2 and edge_index_tv.shape[0] == 2:
            i = edge_index_tv[0].long(); j = edge_index_tv[1].long()
            w_flat = torch.clamp(w_tv.view(-1), 0.0, 1.0).to(m.dtype)
            diff = torch.abs(m[:, i] - m[:, j])
            tv_term = torch.tensor(float((diff.mean(dim=1) * w_flat.mean()).mean().item()), device=dev)
        else:
            tv_term = None
        if isinstance(tv_term, torch.Tensor):
            loss = loss + tv_lambda * tv_term
            if batch_idx == 0:
                try:
                    tv_loss_val = float((tv_lambda * tv_term.detach()).item())
                    print(f"[TV] tv_loss={tv_loss_val:.6f}")
                except Exception:
                    pass
            return loss, float(tv_term.detach().item())
    except Exception:
        pass
    return loss, 0.0


def apply_conn_regularization(loss, m: torch.Tensor, batch, class_logits: torch.Tensor, model, batch_idx: int):
    """Apply lightweight connectivity regularization using flow edges and optional TT weights.
    Returns: (loss_after, conn_term_value: float)
    """
    try:
        conn_enabled = bool(getattr(model, 'loss_conn_enabled', False))
        conn_lambda = float(getattr(model, 'loss_conn_weight', 0.0) or 0.0)
        if not (conn_enabled and conn_lambda > 0.0) or (m is None) or (not isinstance(batch, dict)):
            return loss, 0.0
        dev = class_logits.device
        ei_flow = batch.get('edge_index_flow')
        tt = batch.get('travel_time')
        if isinstance(ei_flow, torch.Tensor):
            ei_flow = ei_flow.to(dev, non_blocking=True).to(torch.long)
        if isinstance(tt, torch.Tensor):
            tt = tt.to(dev, non_blocking=True).to(torch.float32)
        if not isinstance(ei_flow, torch.Tensor):
            if batch_idx == 0:
                print("[CONN][SKIP][LOSS] 缺少 edge_index_flow 或类型不匹配，跳过连通性正则")
            return loss, 0.0
        use_tt = bool(getattr(model, 'loss_conn_use_tt', False))
        tau_t = float(getattr(model, 'loss_conn_tau_time', 1200.0) or 1200.0)
        w_min = 0.0; w_max = 1.0; w_mean = 1.0
        if ei_flow.dim() == 3:
            B = int(ei_flow.shape[0])
            conn_acc = 0.0
            for b in range(B):
                ei_b = ei_flow[b]
                i = ei_b[:, 0].long(); j = ei_b[:, 1].long()
                if use_tt and isinstance(tt, torch.Tensor) and tt.dim() >= 2 and int(tt.shape[0]) >= (b+1):
                    w_b = torch.exp(-torch.clamp(tt[b].view(-1), min=0.0) / max(1e-6, tau_t))
                else:
                    w_b = torch.ones(i.shape[0], dtype=torch.float32, device=dev)
                w_b = torch.clamp(w_b, 0.0, 1.0).to(m.dtype)
                diff = m[b, i] - m[b, j]
                mode = str(getattr(model, 'loss_conn_mode', 'laplacian')).lower()
                term_b = (diff.pow(2) * w_b).mean() if mode == 'laplacian' else (diff.abs() * w_b).mean()
                conn_acc += float(term_b.item())
            conn_term = torch.tensor(conn_acc / max(1, B), device=dev)
            if batch_idx == 0:
                try:
                    wb0 = None
                    if use_tt and isinstance(tt, torch.Tensor) and tt.dim() >= 2:
                        wb0 = torch.exp(-torch.clamp(tt[0].view(-1), min=0.0) / max(1e-6, tau_t))
                    if wb0 is None:
                        wb0 = torch.ones(int(ei_flow.shape[1]), dtype=torch.float32, device=dev)
                    w_min = float(wb0.min().item()); w_max = float(wb0.max().item()); w_mean = float(wb0.mean().item())
                except Exception:
                    pass
        elif ei_flow.dim() == 2 and int(ei_flow.shape[0]) == 2:
            i = ei_flow[0].long(); j = ei_flow[1].long()
            E = int(ei_flow.shape[1])
            if use_tt and isinstance(tt, torch.Tensor):
                if tt.dim() == 1:
                    w_flat = torch.exp(-torch.clamp(tt.view(-1), min=0.0) / max(1e-6, tau_t))
                else:
                    w_flat = torch.exp(-torch.clamp(tt, min=0.0) / max(1e-6, tau_t)).mean(dim=0)
            else:
                w_flat = torch.ones(E, dtype=torch.float32, device=dev)
            w_flat = torch.clamp(w_flat, 0.0, 1.0).to(m.dtype)
            diff = m[:, i] - m[:, j]
            mode = str(getattr(model, 'loss_conn_mode', 'laplacian')).lower()
            conn_term = (diff.pow(2) * w_flat.view(1, -1)).mean() if mode == 'laplacian' else (diff.abs() * w_flat.view(1, -1)).mean()
            if batch_idx == 0:
                try:
                    w_min = float(w_flat.min().item()); w_max = float(w_flat.max().item()); w_mean = float(w_flat.mean().item())
                except Exception:
                    pass
        else:
            conn_term = None
        if isinstance(conn_term, torch.Tensor):
            loss = loss + conn_lambda * conn_term
            if batch_idx == 0:
                try:
                    edges_cnt = int(ei_flow.shape[1]) if ei_flow.dim() in (2,3) else 0
                    print(f"[CONN] L={float((conn_lambda * conn_term).detach().item()):.6f} edges={edges_cnt} w_min={w_min:.4f} w_max={w_max:.4f} w_mean={w_mean:.4f}")
                except Exception:
                    pass
            return loss, float(conn_term.detach().item())
    except Exception:
        pass
    return loss, 0.0


def apply_causal_regularization(loss, m: torch.Tensor, batch, class_logits: torch.Tensor, model, batch_idx: int, logs_dir: str):
    """Apply causal soft regularization (static) using flow edges.
    Returns: (loss_after, l_causal_value: float)
    """
    try:
        causal_enabled = bool(getattr(model, 'loss_causal_enabled', False))
        causal_w = float(getattr(model, 'loss_causal_weight', 0.0) or 0.0)
        if not (causal_enabled and causal_w > 0.0) or (m is None) or (not isinstance(batch, dict)):
            return loss, 0.0
        dev = class_logits.device
        ei_flow = batch.get('edge_index_flow')
        tt = batch.get('travel_time')
        if isinstance(ei_flow, torch.Tensor):
            ei_flow = ei_flow.to(dev, non_blocking=True)
        if isinstance(tt, torch.Tensor):
            tt = tt.to(dev, non_blocking=True)
        tau_c = float(getattr(model, 'loss_causal_tau', 1200.0) or 1200.0)
        l_causal, stats = compute_causal_loss_static(m, ei_flow, tt, tau=tau_c)
        loss = loss + causal_w * l_causal
        if batch_idx == 0:
            try:
                _ = write_causal_stats(
                    logs_dir or '',
                    edges=int(stats.get('edges', 0)),
                    viol_ratio=float(stats.get('viol_ratio', 0.0)),
                    w_min=float(stats.get('w_min', 0.0)),
                    w_max=float(stats.get('w_max', 1.0)),
                    w_mean=float(stats.get('w_mean', 1.0)),
                    l_causal=float((causal_w * l_causal).detach().item()),
                    mode=str(stats.get('mode', 'static')),
                )
                print(f"[CAUSAL] L={float((causal_w * l_causal).detach().item()):.6f} edges={int(stats.get('edges', 0))} viol={float(stats.get('viol_ratio', 0.0)):.4f}")
            except Exception:
                pass
        return loss, float(l_causal.detach().item())
    except Exception:
        pass
    return loss, 0.0
