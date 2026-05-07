import torch
from src.shared.logging.core import write_selector_mask_csv


def build_selector_mask(out: dict, batch, model, batch_idx: int, logs_dir: str):
    """Build selector mask m and report normalization used.
    Priority: selector_logits -> selector_prob -> selector_mask -> fallback to subgraph_targets in batch.
    Returns: (m: Tensor or None, norm_used: str, K_csv: int)
    Side effects: on first batch, writes selector_mask CSV and prints summary.
    """
    try:
        class_logits = out.get('classification')
        B = int(class_logits.shape[0]) if isinstance(class_logits, torch.Tensor) else 0
        N = int(class_logits.shape[1]) if isinstance(class_logits, torch.Tensor) else 0
        m = None
        norm_used = 'none'
        u = None
        if 'selector_logits' in out:
            u = out['selector_logits']
            if isinstance(u, torch.Tensor) and u.dim() == 2:
                pass
            elif isinstance(u, torch.Tensor) and u.dim() == 1:
                u = u.view(B, -1)
        else:
            pb = None
            if 'selector_prob' in out:
                pb = out['selector_prob']
            elif 'selector_mask' in out:
                pb = out['selector_mask']
            if isinstance(pb, torch.Tensor):
                pb = pb.view(B, -1)
                pb = torch.clamp(pb, min=1e-6, max=1.0 - 1e-6)
                u = torch.log(pb) - torch.log(1.0 - pb)
        if u is not None:
            tau = float(getattr(model, 'softmax_temperature', 0.9) or 0.9)
            K = int(getattr(model, 'selector_target_k', 0) or N)
            m = torch.softmax(u / max(1e-6, tau), dim=1) * float(K)
            norm_used = 'softmaxK'
        else:
            norm_used = 'fallback_subgraph_targets'
            if isinstance(batch, dict) and ('subgraph_targets' in batch):
                m2 = batch['subgraph_targets']
                if isinstance(m2, torch.Tensor):
                    if m2.dim() == 1:
                        m = m2.view(1, -1)
                    elif m2.dim() == 2:
                        m = m2
        # First-batch CSV and print
        if (m is not None) and (batch_idx == 0):
            try:
                K_cfg = int(getattr(model, 'selector_target_k', 0) or 0)
                K_csv = K_cfg if K_cfg > 0 else int(m.shape[1])
                _ = write_selector_mask_csv(logs_dir or '', m.detach().cpu(), K=K_csv, norm_type=norm_used)
                sm_avg = float(m.sum(dim=1).mean().item())
                K_dbg = K_csv
                ratio = (sm_avg / float(K_cfg)) if K_cfg > 0 else 0.0
                print(f"[SELECTOR] sum(m)={sm_avg:.4f} K_cfg={K_dbg} ratio={ratio:.4f} norm={norm_used}")
                return m, norm_used, K_csv
            except Exception:
                pass
        return m, norm_used, int(getattr(model, 'selector_target_k', 0) or (m.shape[1] if isinstance(m, torch.Tensor) else 0))
    except Exception:
        return None, 'none', 0

