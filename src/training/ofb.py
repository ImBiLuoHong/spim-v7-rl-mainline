import os
import torch
from typing import Tuple

from src.training.utils import _batch_to_device, _compute_metrics, hard_labels_from
from src.shared.logging.core import append_ce_kl
from src.modeling.losses import CombinedLoss


def train_epoch_ofb(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    criterion_ce_only,
    device: torch.device,
    fixed_batch_cpu: bool,
    overfit_steps: int,
    diag_print_grad: bool,
    diag_print_top1: bool,
    print_ce_kl: bool,
    soft_alpha: float,
    kl_t: float,
    clip_norm: float,
    class_w: float,
    label_smooth: float,
    cekl_csv_path: str,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    单批过拟合训练一个 epoch（诊断用）。

    返回：avg_loss, avg_acc, avg_ce, avg_kl, avg_class, avg_total, avg_alpha, avg_T
    """
    # 要求GPU以保证速度（如需CPU，请在外部禁用 OFB）
    if not torch.cuda.is_available():
        raise RuntimeError("[OFB] 默认要求使用GPU执行。请关闭 OVERFIT_ONE_BATCH 或在具备CUDA的环境下运行。")

    model.train()
    _dev = next(model.parameters()).device
    if _dev.type != 'cuda':
        model.to(torch.device('cuda'))
        _dev = next(model.parameters()).device

    # 本地禁用 AMP/GradScaler，确保纯记忆路径
    scaler = torch.amp.GradScaler(enabled=False)
    print("[CFG][OFB] 已强制禁用 AMP/GradScaler（enabled=False），确保OFB纯记忆路径。")

    # 取首个 batch
    data_iter = iter(train_loader)
    try:
        batch0 = next(data_iter)
    except StopIteration:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, soft_alpha, kl_t

    # 将首个批次映射到目标设备
    # 这里复用通用的批次搬运工具，显式传入当前设备
    features_cpu, soft_targets_cpu, edge_index_cpu, edge_attr_cpu, eidx_v_cpu, eattr_v_cpu = _batch_to_device(batch0, _dev)

    # 将 batch 固定到设备（或保持 CPU 后每步再传）
    eidx_v_cpu = None
    eattr_v_cpu = None
    if fixed_batch_cpu:
        # 已在CPU，后续每步再传设备；兼容4/6元素批次
        if isinstance(batch0, (list, tuple)) and len(batch0) == 6:
            features_cpu, soft_targets_cpu, edge_index_cpu, edge_attr_cpu, eidx_v_cpu, eattr_v_cpu = batch0
        else:
            features_cpu, soft_targets_cpu, edge_index_cpu, edge_attr_cpu = batch0
            eidx_v_cpu = None
            eattr_v_cpu = None
    else:
        # 直接传到设备；统一取6元组（后两项可为None）
        features_cpu, soft_targets_cpu, edge_index_cpu, edge_attr_cpu, eidx_v_cpu, eattr_v_cpu = _batch_to_device(batch0, _dev)

    total_loss = 0.0
    total_acc = 0.0
    ce_sum = kl_sum = class_sum = total_sum = 0.0
    alpha_sum = 0.0
    T_sum = 0.0

    # OFB 使用 CE-only 以实现快速记忆
    local_criterion = criterion_ce_only

    for step in range(max(1, overfit_steps)):
        if fixed_batch_cpu:
            # 每步将CPU批次传到设备，兼容4/6返回结构
            if eidx_v_cpu is not None and eattr_v_cpu is not None:
                features, soft_targets, edge_index, edge_attr, eidx_v, eattr_v = _batch_to_device(
                    (features_cpu, soft_targets_cpu, edge_index_cpu, edge_attr_cpu, eidx_v_cpu, eattr_v_cpu),
                    _dev,
                )
            else:
                features, soft_targets, edge_index, edge_attr, eidx_v, eattr_v = _batch_to_device(
                    (features_cpu, soft_targets_cpu, edge_index_cpu, edge_attr_cpu),
                    _dev,
                )
        else:
            # 已在设备上，直接复用；保持6元组一致性
            features, soft_targets, edge_index, edge_attr, eidx_v, eattr_v = (
                features_cpu,
                soft_targets_cpu,
                edge_index_cpu,
                edge_attr_cpu,
                eidx_v_cpu,
                eattr_v_cpu,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
            out = model(features, edge_index, edge_attr, eidx_v, eattr_v)
            class_logits = out['classification']
            # CE-only：以软目标分布或索引标签统一派生硬标签
            hard_targets = hard_labels_from(soft_targets)
            loss = local_criterion(class_logits, hard_targets)

        # 反向与优化（已禁用AMP）
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if clip_norm is not None and clip_norm > 0:
                scaler.unscale_(optimizer)
        else:
            loss.backward()

        if clip_norm is not None and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # 统计（准确率依旧以硬标签评估）
        total_loss += float(loss.item())
        hard_targets = hard_labels_from(soft_targets)
        acc, _, _, _ = _compute_metrics(class_logits.detach(), hard_targets)
        total_acc += acc

        # 诊断：逐步打印 Top-1 预测与 GT（便于定位停留在 0.75 的样本）
        # 可通过环境变量或配置项 DIAG_PRINT_TOP1 控制开关
        if diag_print_top1:
            try:
                preds = class_logits.detach().argmax(dim=1)
                correct = int((preds == hard_targets).sum().item())
                total = int(preds.numel())
                acc_step = correct / max(1, total)
                # 仅打印前若干步，避免刷屏（>50步时每10步打印一次）
                if step < 10 or (step % 10 == 0):
                    print(f"[OFB][epoch={epoch}][step={step+1}/{overfit_steps}] acc_step={acc_step:.4f} correct={correct}/{total}; preds={preds.tolist()}, gts={hard_targets.tolist()}")
            except Exception as _e:
                print(f"[OFB][diag] Top-1 打印失败: {_e}")

        # 诊断：分模块梯度范数打印（GAT / temporal / head）
        if diag_print_grad:
            try:
                grad_gat_main = 0.0
                grad_gat_virtual = 0.0
                grad_temporal = 0.0
                grad_head = 0.0
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    n = name.lower()
                    g = float(p.grad.data.norm(2).item())
                    # GAT 主分支（排除虚拟）
                    if ('gat' in n) and ('_v' not in n):
                        grad_gat_main += g
                        continue
                    # GAT 虚拟分支
                    if 'gat' in n and '_v' in n:
                        grad_gat_virtual += g
                        continue
                    # 时间骨干（TCN/Transformer）
                    if ('temporal' in n) or ('tcn' in n) or ('encoder' in n):
                        grad_temporal += g
                        continue
                    # 分类头
                    if 'class_node_fc' in n:
                        grad_head += g
                if step < 10 or (step % 10 == 0):
                    print(
                        f"[OFB][epoch={epoch}][step={step+1}] grad_norms: "
                        f"gat_main={grad_gat_main:.6f} gat_virtual={grad_gat_virtual:.6f} "
                        f"temporal={grad_temporal:.6f} head={grad_head:.6f}"
                    )
            except Exception as _e:
                print(f"[OFB][diag] Grad 打印失败: {_e}")

        # 诊断值累计（CE-only）
        try:
            ce = float(getattr(local_criterion, 'last_ce', 0.0) or 0.0)
            kl = float(getattr(local_criterion, 'last_kl', 0.0) or 0.0)
            clsmix = float(getattr(local_criterion, 'last_class', 0.0) or 0.0)
            alpha = float(getattr(local_criterion, 'last_alpha', 1.0))
            Tnow = float(getattr(local_criterion, 'last_T', kl_t))
        except Exception:
            ce = kl = clsmix = 0.0
            alpha = 1.0
            Tnow = kl_t

        ce_sum += ce
        kl_sum += kl
        class_sum += clsmix
        total_sum += float(loss.item())
        alpha_sum += alpha
        T_sum += Tnow

    # 汇总
    denom = max(1, overfit_steps)
    avg_loss = total_loss / denom
    avg_acc = total_acc / denom
    avg_ce = ce_sum / denom
    avg_kl = kl_sum / denom
    avg_class = class_sum / denom
    avg_total = total_sum / denom
    avg_alpha = alpha_sum / denom
    avg_T = T_sum / denom

    try:
        print(
            f"[OFB][epoch={epoch}] avg_loss={avg_loss:.6f}, avg_acc={avg_acc:.4f}; "
            f"ce={avg_ce:.6f}, kl={avg_kl:.6f}, class_mix={avg_class:.6f}, total={avg_total:.6f}, "
            f"alpha={avg_alpha:.4f}, T={avg_T:.4f}"
        )
        if print_ce_kl:
            append_ce_kl(cekl_csv_path, epoch, 'train_ofb', avg_ce, avg_kl, avg_class, avg_total, avg_alpha, avg_T)
    except Exception as _e:
        print(f"[OFB] 平均诊断打印失败: {_e}")

    return (
        avg_loss,
        avg_acc,
        avg_ce,
        avg_kl,
        avg_class,
        avg_total,
        avg_alpha,
        avg_T,
    )

# 统一使用 utils.train_utils.hard_labels_from，已移除本地实现