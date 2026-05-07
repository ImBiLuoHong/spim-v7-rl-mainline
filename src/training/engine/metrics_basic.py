import torch
import sys


def hard_labels_from(targets: torch.Tensor) -> torch.Tensor:
    if targets is None:
        raise ValueError("targets is None")
    if targets.dtype in (torch.float16, torch.float32, torch.float64):
        if targets.dim() == 2:
            return targets.argmax(dim=1).long()
        elif targets.dim() == 1:
            return torch.argmax(targets).view(1).long()
    return targets.view(-1).long()


def compute_metrics(class_logits: torch.Tensor, targets: torch.Tensor):
    """Return (accuracy, precision, recall, f1) with lightweight defaults for P/R/F1."""
    # Binary Classification Case (V6 Architecture)
    if (class_logits.dim() == 2 and class_logits.shape[1] == 1) or (class_logits.dim() == 1):
        preds = (class_logits.view(-1) > 0).long()
        gts = targets.view(-1).long()
    else:
        # Multiclass Case
        preds = class_logits.argmax(dim=1)
        gts = hard_labels_from(targets)
    
    # DEBUG
    # sys.stderr.write(f"[METRICS] Preds: {preds.shape} {preds.float().mean():.4f} | GTs: {gts.shape} {gts.float().mean():.4f}\n")
    # sys.stderr.flush()
    
    correct = int((preds == gts).sum().item())
    total = int(preds.numel())
    acc = correct / max(1, total)
    return acc, 0.0, 0.0, 0.0


def compute_rank_metrics(class_logits: torch.Tensor, targets: torch.Tensor, ks: tuple = (1, 3, 5, 10), ndcg_k: int = 5):
    try:
        B, C = class_logits.shape
        ks = tuple(int(k) for k in ks)
        k_max = max(ks + (int(ndcg_k),))
        
        # Adjust k_max if C is small
        if C < k_max:
            k_max = C
            
        gts = hard_labels_from(targets)  # [B]
        topk_vals, topk_idx = torch.topk(class_logits, k=k_max, dim=1, largest=True, sorted=True)
        
        # DEBUG PRINT
        if torch.rand(1).item() < 0.00:
             print(f"\n[METRICS DEBUG]")
             print(f"  GTs (first 5): {gts[:5].tolist()}")
             print(f"  Preds (first 5): {topk_idx[:5, 0].tolist()}")
             print(f"  Logits Max (first 5): {class_logits[:5].max(dim=1).values.tolist()}")
             print(f"  Targets Max (first 5): {targets[:5].max(dim=1).values.tolist()}")
        
        hits = {}
        for k in ks:
            if k > C:
                hits[int(k)] = 1.0 # Trivial? Or 0.0? If k > C, we cover all? 
                # If we select all C, and source is in C (always true if C is all nodes), then hit is 1.0.
                # But topk_idx only has C columns.
                # Let's just use min(k, C)
                k_eff = min(k, C)
                mask = (topk_idx[:, :k_eff] == gts.view(-1, 1))
            else:
                mask = (topk_idx[:, :k] == gts.view(-1, 1))
            
            hit_k = (mask.any(dim=1).to(torch.float32)).mean().item()
            hits[int(k)] = float(hit_k)
            
        idx_sorted = torch.argsort(class_logits, dim=1, descending=True)
        pos_matrix = (idx_sorted == gts.view(-1, 1)).to(torch.float32)
        positions = pos_matrix * torch.arange(C, device=class_logits.device, dtype=torch.float32).view(1, C)
        ranks = positions.sum(dim=1) + 1.0
        mrr = (1.0 / ranks).mean().item()
        
        if ndcg_k > C: ndcg_k = C
        ndcg_mask = (ranks <= float(ndcg_k)).to(torch.float32)
        ndcg_vals = ndcg_mask * (1.0 / torch.log2(ranks + 1.0))
        ndcg = ndcg_vals.mean().item()
        return hits, float(mrr), float(ndcg)
    except Exception as e:
        # print(f"[METRICS] Error: {e}")
        return ({int(k): 0.0 for k in ks}, 0.0, 0.0)

