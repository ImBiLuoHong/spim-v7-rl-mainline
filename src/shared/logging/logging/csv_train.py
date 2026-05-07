import os
import json
from typing import Iterable, Dict


def ensure_train_log_csv_format(log_csv_path: str, ks: Iterable[int] = (1, 3, 5, 10), ndcg_k: int = 5) -> str:
    """Ensure the train_log.csv exists with a proper header (minimal, test-compatible).

    Columns (in order):
    epoch,train_loss,train_acc(Hit@1),train_node_acc,train_hit@k...,train_mrr,train_ndcg@{ndcg_k},val_loss,val_acc(Hit@1),val_node_acc,val_hit@k...,val_mrr,val_ndcg@{ndcg_k},lr
    """
    try:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        # Always overwrite header if file is empty or explicitly requested (not easily possible here without flag)
        # For now, we rely on the file being empty or non-existent for header creation.
        if not os.path.exists(log_csv_path) or os.path.getsize(log_csv_path) == 0:
            ks = list(ks or [])
            cols = ['epoch', 'train_loss', 'train_acc', 'train_node_acc', 'train_scr']
            for k in ks:
                cols.append(f'train_hit@{int(k)}')
            cols += ['train_mrr', f'train_ndcg@{int(ndcg_k)}', 'val_loss', 'val_acc', 'val_node_acc']
            for k in ks:
                cols.append(f'val_hit@{int(k)}')
            cols += ['val_mrr', f'val_ndcg@{int(ndcg_k)}', 'lr']
            with open(log_csv_path, 'w', encoding='utf-8') as f:
                f.write(','.join(cols) + '\n')
        return log_csv_path
    except Exception as e:
        try:
            print(f"[WARN] ensure_train_log_csv_format failed: {e}")
        except Exception:
            pass
        return log_csv_path


def append_log(
    log_csv_path: str,
    epoch: int,
    train_loss: float,
    train_acc: float,
    train_hits: Dict[int, float],
    train_mrr: float,
    train_ndcg: float,
    val_loss: float,
    val_acc: float,
    val_hits: Dict[int, float],
    val_mrr: float,
    val_ndcg: float,
    lr_now: float,
    train_node_acc: float = 0.0,
    val_node_acc: float = 0.0,
    train_scr: float = 0.0,
    ks: Iterable[int] = (1, 3, 5, 10),
    ndcg_k: int = 5,
    **kwargs,
) -> str:
    """Append one row to train_log.csv according to the minimal header.

    Extra keyword arguments are accepted and ignored to maintain compatibility.
    """
    try:
        ensure_train_log_csv_format(log_csv_path, ks=ks, ndcg_k=ndcg_k)
        ks_list = list(ks or [])
        vals = [int(epoch), float(train_loss), float(train_acc), float(train_node_acc), float(train_scr)]
        for k in ks_list:
            vals.append(float(train_hits.get(int(k), 0.0)))
        vals += [float(train_mrr), float(train_ndcg), float(val_loss), float(val_acc), float(val_node_acc)]
        for k in ks_list:
            vals.append(float(val_hits.get(int(k), 0.0)))
        vals += [float(val_mrr), float(val_ndcg), float(lr_now)]
        def _fmt_val(v):
            try:
                if isinstance(v, int):
                    return str(v)
                fv = float(v)
                return f"{fv:.4f}"
            except Exception:
                return str(v)
        with open(log_csv_path, 'a', encoding='utf-8') as f:
            f.write(','.join(_fmt_val(v) for v in vals) + '\n')
        return log_csv_path
    except Exception as e:
        try:
            print(f"[WARN] append_log failed: {e}")
        except Exception:
            pass
        return log_csv_path


def append_grad_log(csv_path: str, epoch_index: int, batch_in_epoch: int, step_index: int, total_grad_norm: float) -> str:
    """Append gradient norm entry to gradients.csv. Creates header if missing."""
    try:
        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = 'epoch,batch,step,total_grad_norm\n'
        need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', encoding='utf-8') as f:
            if need_header:
                f.write(header)
            f.write(f"{int(epoch_index)},{int(batch_in_epoch)},{int(step_index)},{float(total_grad_norm)}\n")
        return csv_path
    except Exception as e:
        try:
            print(f"[WARN] append_grad_log failed: {e}")
        except Exception:
            pass
        return csv_path


def append_ce_kl(
    csv_path: str,
    epoch_index: int,
    batch_in_epoch: int,
    step_index: int,
    ce: float,
    kl: float,
    alpha: float,
    T: float,
    *extra_args,
    **extra_kwargs,
) -> str:
    """Append a CE/KL record to cekl_csv_path. Creates header if missing."""
    try:
        def _as_float(x):
            try:
                import torch  # type: ignore
                if isinstance(x, torch.Tensor):
                    return float(x.detach().item())
            except Exception:
                pass
            try:
                import numpy as np  # type: ignore
                if isinstance(x, (np.floating, np.integer)):
                    return float(x)
            except Exception:
                pass
            try:
                return float(x)
            except Exception:
                return 0.0

        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = 'epoch,batch,step,ce,kl,alpha,T\n'
        need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

        ce_v = _as_float(extra_kwargs.get('ce', ce))
        kl_v = _as_float(extra_kwargs.get('kl', kl))
        alpha_v = _as_float(extra_kwargs.get('alpha', alpha))
        T_v = _as_float(extra_kwargs.get('T', T))

        with open(csv_path, 'a', encoding='utf-8') as f:
            if need_header:
                f.write(header)
            f.write(
                f"{int(epoch_index)},{int(batch_in_epoch)},{int(step_index)},{ce_v:.6f},{kl_v:.6f},{alpha_v:.6f},{T_v:.6f}\n"
            )
        return csv_path
    except Exception as e:
        try:
            print(f"[WARN] append_ce_kl failed: {e}")
        except Exception:
            pass
        return csv_path