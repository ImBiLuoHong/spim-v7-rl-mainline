import torch
import torch.nn as nn
import torch.nn.functional as F

class SubgraphLoss(nn.Module):
    """
    子图损失组合：
    - 覆盖 BCE：将模型分类概率视为节点选择概率 p_i，与目标子图掩码 y_i ∈ {0,1} 做 BCE。
    - TV（Total Variation）边平滑：鼓励相邻节点选择概率一致，∑_{(i,j)∈E} |p_i - p_j|。
    - Size 正则：鼓励选择节点数量接近目标规模（若提供目标规模），或惩罚过大/过小。

    所有权重默认为 0（安全关闭），仅当 cfg.features.subgraph_* 显式开启时生效。
    """
    def __init__(self,
                 bce_weight: float = 0.0,
                 tv_weight: float = 0.0,
                 size_weight: float = 0.0,
                 target_size: int | None = None,
                 eps: float = 1e-6,
                 tv_mode: str = 'adj_binary'):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.tv_weight = float(tv_weight)
        self.size_weight = float(size_weight)
        self.target_size = int(target_size) if (target_size is not None) else None
        self.eps = float(eps)
        # tv_mode: 'adj_binary' 基于邻接的 |pi-pj| 平均；'travel_time' 使用 edge_attr 加权
        self.tv_mode = str(tv_mode).lower() if tv_mode else 'adj_binary'

    def forward(self,
                class_logits: torch.Tensor,
                subgraph_targets: torch.Tensor | None,
                edge_index: torch.Tensor | None,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """
        输入：
        - class_logits: [B, N]（来自模型分类分支）
        - subgraph_targets: [B, N] 二值掩码（可选；缺失则 BCE 项为0）
        - edge_index: [B, T, E, 2] 或 [B*W*E, 2]（支持简单解耦，TV项按首时间步近似）
        返回：标量损失（加权和）
        """
        total = torch.tensor(0.0, device=class_logits.device)
        # 概率，detach后参与TV/size，BCE使用概率前向
        probs = torch.softmax(class_logits, dim=1)  # [B,N]
        # BCE 覆盖
        if self.bce_weight > 0.0 and (subgraph_targets is not None):
            y = subgraph_targets.to(probs.dtype)
            bce = F.binary_cross_entropy(probs, y, reduction='mean')
            total = total + self.bce_weight * bce
        # TV 边平滑（近似：取首时间步 T=0 的边集）；tv_mode 控制权重
        if self.tv_weight > 0.0 and (edge_index is not None):
            try:
                if edge_index.dim() == 4:
                    # [B,T,E,2] -> 仅取 T=0
                    ei0 = edge_index[:, 0, :, :]  # [B,E,2]
                    tv_sum = 0.0
                    B = probs.shape[0]
                    for b in range(B):
                        ei_b = ei0[b]
                        i = ei_b[:, 0].long(); j = ei_b[:, 1].long()
                        diff_ij = torch.abs(probs[b, i] - probs[b, j])  # [E]
                        if self.tv_mode == 'travel_time' and (edge_attr is not None) and edge_attr.dim() >= 4:
                            # 选择 travel_time 权重通道：优先使用通道索引1；若不可用，退回最后一维
                            try:
                                eattr0 = edge_attr[b, 0]  # [E, D]
                                D = int(eattr0.shape[-1])
                                if D >= 2:
                                    w = eattr0[:, 1].to(diff_ij.dtype)  # [E]
                                else:
                                    w = eattr0[:, -1].to(diff_ij.dtype)
                                # 归一化权重以稳定尺度
                                w = w / (w.mean() + self.eps)
                                diff = (diff_ij * w).mean()
                            except Exception:
                                diff = diff_ij.mean()
                        else:
                            diff = diff_ij.mean()
                        tv_sum += float(diff.item())
                    tv = torch.tensor(tv_sum / max(1, probs.shape[0]), device=class_logits.device)
                elif edge_index.dim() == 2 and edge_index.shape[0] == 2:
                    # [2, E] 视为单图（无批次），近似平均
                    i = edge_index[0].long(); j = edge_index[1].long()
                    # 使用 batch 平均
                    diff_ij = torch.abs(probs[:, i] - probs[:, j])  # [B,E]
                    if self.tv_mode == 'travel_time' and (edge_attr is not None) and edge_attr.dim() == 2:
                        try:
                            # edge_attr: [E, D]，选用通道1或最后一维
                            D = int(edge_attr.shape[-1])
                            if D >= 2:
                                w = edge_attr[:, 1].view(1, -1).to(diff_ij.dtype)
                            else:
                                w = edge_attr[:, -1].view(1, -1).to(diff_ij.dtype)
                            w = w / (w.mean() + self.eps)
                            diff = (diff_ij * w).mean()
                        except Exception:
                            diff = diff_ij.mean()
                    else:
                        diff = diff_ij.mean()
                    tv = diff
                else:
                    tv = torch.tensor(0.0, device=class_logits.device)
            except Exception:
                tv = torch.tensor(0.0, device=class_logits.device)
            total = total + self.tv_weight * tv
        # Size 正则：与目标大小的 L2 偏差或稀疏惩罚
        if self.size_weight > 0.0:
            size = probs.sum(dim=1)  # [B]
            if self.target_size is not None and self.target_size > 0:
                target = torch.full_like(size, float(self.target_size))
                sz_loss = torch.mean((size - target) ** 2)
            else:
                # 稀疏惩罚：鼓励较小总和（近似L1）
                sz_loss = torch.mean(size)
            total = total + self.size_weight * sz_loss
        return total