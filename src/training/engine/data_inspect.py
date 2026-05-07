import torch
import math


def infer_num_features_from_batch(sample_batch) -> int:
    """Infer node feature channels from first batch."""
    try:
        fts = None
        if isinstance(sample_batch, (list, tuple)):
            if len(sample_batch) >= 1 and hasattr(sample_batch[0], 'shape'):
                fts = sample_batch[0]
        elif isinstance(sample_batch, dict):
            for k in ('features', 'x_seq', 'x', 'fts', 'inputs'):
                v = sample_batch.get(k)
                if v is not None and hasattr(v, 'shape'):
                    fts = v
                    break
        if fts is None or (not hasattr(fts, 'shape')):
            # V6 Static: x is [N, 9] -> dim=2
            # V2 Dynamic: x is [B, W, N, F] -> dim=4
            # If we fail to infer, try to look for PyG 'x'
            if isinstance(sample_batch, (list, tuple)) and len(sample_batch) > 0 and hasattr(sample_batch[0], 'x'):
                return int(sample_batch[0].x.shape[-1])
            # PyG Batch object
            if hasattr(sample_batch, 'x') and hasattr(sample_batch.x, 'shape'):
                return int(sample_batch.x.shape[-1])
            return 1
        if len(fts.shape) >= 1:
            return int(fts.shape[-1])
        return 1
    except Exception:
        return 1


def infer_edge_dims(train_loader) -> dict:
    """Infer main and virtual edge feature channel dims (V1/V2 compatible)."""
    try:
        ds = getattr(train_loader, 'dataset', None)
        tt_enabled = True
        flow_enabled = True
        if ds is not None:
            try:
                tt_enabled = bool(getattr(ds, 'travel_time_edge_enabled', True))
            except Exception:
                tt_enabled = True
            try:
                flow_enabled = bool(getattr(ds, 'flow_edge_enabled', True))
            except Exception:
                flow_enabled = True
        sample_batch = next(iter(train_loader))
        eattr = None
        eattr_v = None
        if isinstance(sample_batch, (list, tuple)):
            if len(sample_batch) >= 4 and hasattr(sample_batch[3], 'shape'):
                eattr = sample_batch[3]
            if len(sample_batch) >= 6 and hasattr(sample_batch[5], 'shape'):
                eattr_v = sample_batch[5]
        elif isinstance(sample_batch, dict):
            eattr = sample_batch.get('edge_attr')
            eattr_v = sample_batch.get('eattr_v')
        # Try to infer from PyG Batch
        if hasattr(sample_batch, 'edge_attr') and hasattr(sample_batch.edge_attr, 'shape'):
            eattr = sample_batch.edge_attr
            if eattr.dim() == 2:
                edge_dim_main = int(eattr.shape[-1])
                print(f"[DATA] edge_dim_main(inferred_pyg)={edge_dim_main} edge_attr.shape={tuple(eattr.shape)}")
                return {'edge_dim_main': edge_dim_main, 'edge_dim_virtual': 0}

        if (not tt_enabled) and (not flow_enabled):
            edge_dim_main = 0
            print(f"[DATA] 所有边特征均已关闭 -> edge_dim_main=0")
        else:
            if (eattr is not None) and hasattr(eattr, 'shape') and eattr.dim() >= 4:
                edge_dim_main = int(eattr.shape[-1])
                print(f"[DATA] edge_dim_main(inferred)={edge_dim_main} edge_attr.shape(sample)={tuple(eattr.shape)}")
            else:
                print("[DATA][WARN] 批次结构异常，无法推断 edge_dim_main（默认1）")
                edge_dim_main = 1
        if (eattr_v is not None) and hasattr(eattr_v, 'shape') and eattr_v.dim() >= 4:
            edge_dim_virtual = int(eattr_v.shape[-1])
            print(f"[DATA] edge_dim_virtual(inferred)={edge_dim_virtual} eattr_v.shape(sample)={tuple(eattr_v.shape)}")
        else:
            edge_dim_virtual = 0
        return {
            'edge_dim_main': int(edge_dim_main),
            'edge_dim_virtual': int(edge_dim_virtual),
        }
    except Exception as _e:
        print(f"[DATA][WARN] 推断边维度失败: {_e}")
        return {
            'edge_dim_main': 1,
            'edge_dim_virtual': 0,
        }


def print_travel_time_samples(train_loader, max_edges: int = 10) -> None:
    """Randomly sample and print several edges' geometry and travel_time stats for minimal sanity check."""
    try:
        sample_batch = next(iter(train_loader))
        eattr = None
        if isinstance(sample_batch, (list, tuple)):
            if len(sample_batch) >= 4:
                eattr = sample_batch[3]
        elif isinstance(sample_batch, dict):
            eattr = sample_batch.get('edge_attr')
        else:
            print("[DATA][WARN] 批次结构异常，无法打印 travel_time_norm 取样")
            return
        ds = train_loader.dataset
        if (not isinstance(eattr, torch.Tensor)) or (eattr.dim() < 4) or (eattr.shape[-1] == 0):
            print("[DATA] edge_attr channels=0（已全关边特征），跳过 travel_time 取样打印。")
            return
        B, T, E, D = eattr.shape
        if D < 2 or not bool(getattr(ds, 'travel_time_edge_enabled', True)):
            idxs = torch.randperm(E)[:min(max_edges, E)]
            q_mean = eattr[0, :, :, 0].mean(dim=0)
            lengths = getattr(ds, 'pipe_length_arr', None)
            diams = getattr(ds, 'pipe_diam_arr', None)
            eps = 1e-6
            if isinstance(lengths, torch.Tensor) and isinstance(diams, torch.Tensor) and lengths.numel() == E and diams.numel() == E:
                area = (math.pi * (diams / 2.0) ** 2).to(torch.float32)
                denom = torch.maximum(q_mean, torch.tensor(eps, dtype=torch.float32))
                tau = lengths * area / denom
            else:
                tau = torch.ones((E,), dtype=torch.float32)
            print("[DATA] sample edges (idx, L, d, |Q|, tau): [tt_norm disabled]")
            for i in idxs.tolist():
                L_i = float(lengths[i].item()) if isinstance(lengths, torch.Tensor) else float('nan')
                d_i = float(diams[i].item()) if isinstance(diams, torch.Tensor) else float('nan')
                Q_i = float(q_mean[i].item())
                tau_i = float(tau[i].item())
                print(f"  edge[{i}]: L={L_i:.4f} d={d_i:.4f} |Q|={Q_i:.6f} tau={tau_i:.6f}")
            return
        # travel_time_norm enabled and channels>=2: can be extended for richer prints
    except Exception as _e:
        print(f"[DATA][WARN] 打印 travel_time 取样失败: {_e}")

