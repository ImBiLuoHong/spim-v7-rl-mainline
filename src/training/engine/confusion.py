import torch
import torch.nn.functional as F


def confusion_counts_from_logits_targets(class_logits: torch.Tensor, targets: torch.Tensor):
    """Return per-class TP/FP/FN and P counts (shape [C])."""
    try:
        B, C = class_logits.shape
        preds = class_logits.argmax(dim=1)
        if targets is None:
            gts = preds
        elif targets.dtype in (torch.float16, torch.float32, torch.float64) and targets.dim() == 2:
            gts = targets.argmax(dim=1)
        else:
            gts = targets.view(-1).long()
        one_hot_pred = F.one_hot(preds, num_classes=C).to(torch.float32)
        one_hot_gt = F.one_hot(gts, num_classes=C).to(torch.float32)
        tp = (one_hot_pred * one_hot_gt).sum(dim=0)
        fp = (one_hot_pred * (1.0 - one_hot_gt)).sum(dim=0)
        fn = ((1.0 - one_hot_pred) * one_hot_gt).sum(dim=0)
        p = one_hot_gt.sum(dim=0)
        return tp, fp, fn, p
    except Exception:
        return (
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )


def balanced_acc_and_macro_f1_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, p: torch.Tensor):
    """Compute Balanced-Acc and Macro-F1 from per-class counts."""
    try:
        eps = 1e-12
        tp = tp.to(torch.float32)
        fp = fp.to(torch.float32)
        fn = fn.to(torch.float32)
        p = p.to(torch.float32)
        valid = (p > 0.0)
        recall_c = torch.where(valid, tp / (p + eps), torch.zeros_like(p))
        balanced_acc = float(recall_c.mean().item()) if recall_c.numel() > 0 else 0.0
        precision_c = tp / (tp + fp + eps)
        f1_c = 2.0 * precision_c * recall_c / (precision_c + recall_c + eps)
        macro_f1 = float(f1_c.mean().item()) if f1_c.numel() > 0 else 0.0
    except Exception:
        balanced_acc = 0.0
        macro_f1 = 0.0
    return balanced_acc, macro_f1

