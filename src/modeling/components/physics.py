import os
from typing import Optional, Tuple
import torch

__all__ = [
    'build_upstream_feasible_mask',
    'compute_causal_loss_static',
    'write_feasible_stats',
    'write_causal_stats',
]


def _normalize_edge_index_flow(edge_index_flow: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    规范化 edge_index_flow 到形状 [2, E]（u->v），返回 (edge_index_2xe, batched)
    支持输入：[B, E, 2] 或 [2, E]
    """
    if edge_index_flow is None:
        return None, False
    if isinstance(edge_index_flow, torch.Tensor):
        if edge_index_flow.dim() == 3 and edge_index_flow.shape[-1] == 2:
            # [B,E,2] -> 逐 batch 处理，外部循环
            return edge_index_flow, True
        if edge_index_flow.dim() == 2 and edge_index_flow.shape[0] == 2:
            return edge_index_flow, False
    return None, False


def build_upstream_feasible_mask(
    edge_index_flow: torch.Tensor,
    travel_time: Optional[torch.Tensor],
    anomaly_mask: Optional[torch.Tensor],
    window_s: float,
) -> torch.Tensor:
    """
    构建上游可行掩码 feasible[B,N]：对每个样本，沿反向边从 anomaly 节点出发，累积 travel_time<=window_s 的所有可达节点置 1。
    - edge_index_flow: [B,E,2] 或 [2,E]
    - travel_time: [B,E] 或 [E]（单位秒）；缺失时按权重=1近似
    - anomaly_mask: [B,N]（二值）；缺失则返回全1掩码
    - window_s: 时间窗阈值（秒）
    返回：feasible[B,N] ∈ {0,1}
    """
    ei, is_batched = _normalize_edge_index_flow(edge_index_flow)
    if ei is None or anomaly_mask is None:
        # 缺关键数据返回全1（不裁剪）
        if isinstance(anomaly_mask, torch.Tensor):
            B, N = anomaly_mask.shape
        else:
            # 尝试从边推测 N；若失败返回空张量
            if isinstance(ei, torch.Tensor):
                if is_batched:
                    B = ei.shape[0]
                    N = int(max(int(ei[..., 0].max().item()), int(ei[..., 1].max().item())) + 1)
                else:
                    B = 1
                    N = int(max(int(ei[0].max().item()), int(ei[1].max().item())) + 1)
            else:
                B, N = 1, 1
        return torch.ones((B, N), dtype=torch.float32, device=(anomaly_mask.device if isinstance(anomaly_mask, torch.Tensor) else None))

    device = anomaly_mask.device
    B, N = anomaly_mask.shape
    feasible = torch.zeros((B, N), dtype=torch.float32, device=device)

    # 处理 travel_time 尺度
    def _edge_tt_for_batch(tt, b, E):
        if tt is None:
            return torch.ones((E,), dtype=torch.float32, device=device)
        if tt.dim() == 1:
            return tt.to(device).view(-1).float()
        if tt.dim() == 2:
            return tt[b].to(device).view(-1).float()
        return torch.ones((E,), dtype=torch.float32, device=device)

    if is_batched:
        # [B,E,2]
        for b in range(B):
            ei_b = ei[b]  # [E,2]
            src = ei_b[:, 0].long()
            dst = ei_b[:, 1].long()
            E = ei_b.shape[0]
            tt_b = _edge_tt_for_batch(travel_time, b, E)

            # 建立逆向邻接：dst -> src
            rev_adj = [[] for _ in range(N)]
            for e in range(E):
                u = int(src[e].item()); v = int(dst[e].item())
                if 0 <= u < N and 0 <= v < N:
                    rev_adj[v].append((u, float(tt_b[e].item())))
            # 多源起点：anomaly 节点
            starts = (anomaly_mask[b] > 0.5).nonzero(as_tuple=False).view(-1).tolist()
            # 从起点逆向 BFS，累积 TT<=window_s
            import heapq
            dist = [float('inf')] * N
            hq = []
            for s in starts:
                dist[s] = 0.0
                heapq.heappush(hq, (0.0, int(s)))
            while hq:
                d, v = heapq.heappop(hq)
                if d > window_s:
                    continue
                feasible[b, v] = 1.0
                for (u, w) in rev_adj[v]:
                    nd = d + (float(w) if travel_time is not None else 1.0)
                    if nd < dist[u] and nd <= window_s + 1e-6:
                        dist[u] = nd
                        heapq.heappush(hq, (nd, u))
    else:
        # 单图 [2,E]
        src = ei[0].long(); dst = ei[1].long(); E = ei.shape[1]
        tt = _edge_tt_for_batch(travel_time, 0, E)
        rev_adj = [[] for _ in range(N)]
        for e in range(E):
            u = int(src[e].item()); v = int(dst[e].item())
            if 0 <= u < N and 0 <= v < N:
                rev_adj[v].append((u, float(tt[e].item())))
        import heapq
        for b in range(B):
            starts = (anomaly_mask[b] > 0.5).nonzero(as_tuple=False).view(-1).tolist()
            dist = [float('inf')] * N
            hq = []
            for s in starts:
                dist[s] = 0.0
                heapq.heappush(hq, (0.0, int(s)))
            while hq:
                d, v = heapq.heappop(hq)
                if d > window_s:
                    continue
                feasible[b, v] = 1.0
                for (u, w) in rev_adj[v]:
                    nd = d + (float(w) if travel_time is not None else 1.0)
                    if nd < dist[u] and nd <= window_s + 1e-6:
                        dist[u] = nd
                        heapq.heappush(hq, (nd, u))

    return feasible


def compute_causal_loss_static(
    m: torch.Tensor,  # [B,N]
    edge_index_flow: torch.Tensor,  # [B,E,2] or [2,E]
    travel_time: Optional[torch.Tensor],  # [B,E] or [E] or None
    tau: float = 1200.0,
) -> Tuple[torch.Tensor, dict]:
    """
    计算静态因果软正则：mean( w_uv * ReLU(m_v - m_u) )，w = 1 或 exp(-tt/tau)。
    返回：(L_causal, stats)
    """
    ei, is_batched = _normalize_edge_index_flow(edge_index_flow)
    if ei is None or m is None:
        return torch.tensor(0.0, device=(m.device if isinstance(m, torch.Tensor) else 'cpu')), {
            'edges': 0,
            'viol_ratio': 0.0,
            'w_min': 0.0,
            'w_max': 0.0,
            'w_mean': 0.0,
        }
    B, N = m.shape
    device = m.device

    def _edge_weight(tt):
        if tt is None:
            return None
        w = torch.exp(-tt.to(device).float() / max(1e-6, float(tau)))
        return torch.clamp(w, 0.0, 1.0)

    total = 0.0
    edges_total = 0
    viol_cnt = 0
    w_min = 1.0
    w_max = 0.0
    w_sum = 0.0

    if is_batched:
        for b in range(B):
            ei_b = ei[b]
            i = ei_b[:, 0].long(); j = ei_b[:, 1].long()
            E = ei_b.shape[0]
            tt_b = None
            if isinstance(travel_time, torch.Tensor):
                if travel_time.dim() == 2:
                    tt_b = travel_time[b].view(-1)
                elif travel_time.dim() == 1:
                    tt_b = travel_time.view(-1)
            w_b = _edge_weight(tt_b) if tt_b is not None else None
            diff = torch.relu(m[b, j] - m[b, i])  # [E]
            if w_b is not None:
                w_use = torch.clamp(w_b, 0.0, 1.0).to(diff.dtype)
            else:
                w_use = torch.ones_like(diff)
            total = total + float((diff * w_use).mean().item())
            edges_total += int(E)
            viol_cnt += int((diff > 0).float().mean().item() * E)
            w_min = min(w_min, float(w_use.min().item()))
            w_max = max(w_max, float(w_use.max().item()))
            w_sum += float(w_use.mean().item())
        lval = torch.tensor(total / max(1, B), device=device)
        w_mean = (w_sum / max(1, B)) if edges_total > 0 else 0.0
    else:
        i = ei[0].long(); j = ei[1].long(); E = ei.shape[1]
        tt_flat = travel_time if (isinstance(travel_time, torch.Tensor) and travel_time.dim() == 1) else None
        w = _edge_weight(tt_flat)
        diff = torch.relu(m[:, j] - m[:, i])  # [B,E]
        if w is not None:
            w_use = torch.clamp(w.view(1, -1), 0.0, 1.0).to(diff.dtype)
            val = (diff * w_use).mean()
            w_min = float(w.min().item()); w_max = float(w.max().item()); w_mean = float(w.mean().item())
        else:
            val = diff.mean()
            w_min = 1.0; w_max = 1.0; w_mean = 1.0
        lval = val
        edges_total = int(E)
        viol_cnt = int((diff > 0).float().mean().item() * E)

    stats = {
        'edges': int(edges_total),
        'viol_ratio': float(viol_cnt / max(1, edges_total)),
        'w_min': float(w_min),
        'w_max': float(w_max),
        'w_mean': float(w_mean),
        'mode': 'static',
    }
    return lval, stats


def write_feasible_stats(logs_dir: str, kept: int, ratio: float, window_s: float, has_tt: bool) -> Optional[str]:
    try:
        path = os.path.join(logs_dir, 'feasible_stats.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = (not os.path.exists(path))
        with open(path, 'a', encoding='utf-8') as f:
            if header:
                f.write('kept,ratio,window_s,has_tt\n')
            f.write(f"{int(kept)},{float(ratio):.6f},{float(window_s):.1f},{bool(has_tt)}\n")
        return path
    except Exception:
        return None


def write_causal_stats(logs_dir: str, edges: int, viol_ratio: float, w_min: float, w_max: float, w_mean: float, l_causal: float, mode: str = 'static') -> Optional[str]:
    try:
        path = os.path.join(logs_dir, 'causal_stats.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = (not os.path.exists(path))
        with open(path, 'a', encoding='utf-8') as f:
            if header:
                f.write('edges,viol_ratio,w_min,w_max,w_mean,L_causal,mode\n')
            f.write(f"{int(edges)},{float(viol_ratio):.6f},{float(w_min):.6f},{float(w_max):.6f},{float(w_mean):.6f},{float(l_causal):.8f},{mode}\n")
        return path
    except Exception:
        return None
import torch
from typing import Optional


def derive_prior_from_batch(batch, N: int) -> Optional[torch.Tensor]:
    """最小先验推导：从 batch 中读取 'prior' 或 'prior_prob'，形状对齐为 [B,N]。
    - 允许 [B,N], [N], 或 [B,N,1]；数值将截断到 [1e-9, 1.0].
    - 若不存在合适键，则返回 None。
    """
    try:
        if not isinstance(batch, dict):
            return None
        prior = None
        for key in ['prior', 'prior_prob', 'prior_mask']:
            if key in batch:
                prior = batch[key]
                break
        if prior is None or (not isinstance(prior, torch.Tensor)):
            return None
        # 标准化形状
        if prior.dim() == 1:
            prior = prior.view(1, -1)
        elif prior.dim() == 3 and int(prior.shape[2]) == 1:
            prior = prior.squeeze(-1)
        if prior.dim() != 2:
            return None
        # 对齐列数
        if int(prior.shape[1]) != int(N):
            return None
        prior = prior.to(torch.float32)
        prior = torch.clamp(prior, min=1e-9, max=1.0)
        return prior
    except Exception:
        return None


def fuse_logits_with_prior(
    logits: torch.Tensor,
    prior: torch.Tensor,
    mode: str = 'log_add',
    alpha: float = 0.5,
    beta: float = 0.0,
    eps: float = 1e-12,
    base_temperature: float = 1.0,
):
    """先验软融合（纯加法）：不改外部 forward 接口。
    - mode='log_add': logits += alpha * log(prior + eps)
    - mode='temp_scale': T_eff = base_T / (1 + beta * prior)
        在 softmax 使用：softmax(logits / T_eff)
        但为保持接口与下游一致，这里返回更新后的 logits 以及 T_eff 的均值（审计用）。
    返回：更新后的 logits，T_eff_mean
    """
    mode = str(mode or 'log_add').lower()
    B, N = int(logits.shape[0]), int(logits.shape[1])
    T_eff = torch.full((B, N), float(base_temperature), dtype=torch.float32, device=logits.device)
    if mode == 'log_add':
        add_term = alpha * torch.log(prior + eps)
        logits = logits + add_term.to(logits.dtype)
        return logits, float(T_eff.mean().item())
    elif mode == 'temp_scale':
        # 注意：不直接改变 softmax 的实现；将温度缩放等效为 logits 预缩放
        T_eff = torch.clamp(base_temperature / (1.0 + beta * prior), min=1e-6)
        logits = logits / T_eff.to(logits.dtype)
        return logits, float(T_eff.mean().item())
    else:
        # 未知模式：不做处理
        return logits, float(T_eff.mean().item())

