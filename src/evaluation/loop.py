import os
import json
import torch
from typing import Tuple
from datetime import datetime

from src.training.utils import _batch_to_device, _compute_metrics, _compute_rank_metrics, _confusion_counts_from_logits_targets, _balanced_acc_and_macro_f1_from_counts, hard_labels_from, compute_topk_label_mass_and_cov_tilde
from src.shared.logging.core import write_effective_config_snapshot, write_selector_mask_csv
from src.config.paths import derive_logs_dir
from src.evaluation.metrics.connectivity import load_adj, connected_recall_at_k, load_eval_edges
from src.modeling.components.physics import build_upstream_feasible_mask, compute_causal_loss_static
from src.modeling.components.physics import derive_prior_from_batch, fuse_logits_with_prior
from src.evaluation.metrics.oracle import coverage_CK, oracle_hit_at_k
from src.evaluation.metrics.domain import model_hit_at_k_sensor, cap_mask_ratio
from src.evaluation.metrics.correlation import spearman_batch_logits_and_labels, spearman_batch_probs_and_labels
from src.evaluation.metrics.connectivity_metrics import connectivity_rate, island_rate


def evaluate(
    loader,
    model: torch.nn.Module,
    criterion,
    device: torch.device,
    use_soft_labels: bool,
    soft_alpha_default: float,
    kl_t_default: float,
    max_eval_steps: int,
    rank_ks: tuple = (1, 3, 5, 10),
    ndcg_k: int = 5,
    subgraph_criterion=None,
) -> tuple:
    """标准评估循环：不进行反向传播与优化，返回基础8项 + 动态命中率@k + MRR + NDCG@k + Balanced-Acc + Macro-F1。"""
    model.eval()
    # AMP 自动侦测（评估期仅用于前向推理提速，不改变指标与候选范围）
    amp_enabled = False
    try:
        # 优先读取训练期记录的 AMP 状态；其次读取 cfg.training.use_amp
        last_amp = bool(getattr(model, 'last_amp_enabled', False))
    except Exception:
        last_amp = False
    try:
        cfg_obj = getattr(model, 'cfg', None)
        cfg_amp = bool(getattr(getattr(cfg_obj, 'training', object()), 'use_amp', False)) if cfg_obj is not None else False
    except Exception:
        cfg_amp = False
    try:
        amp_enabled = (device.type == 'cuda') and (last_amp or cfg_amp)
    except Exception:
        amp_enabled = False
    total_loss = 0.0
    total_acc = 0.0
    ce_sum = kl_sum = class_sum = total_sum = 0.0
    alpha_sum = 0.0
    T_sum = 0.0
    hit_sums = {int(k): 0.0 for k in rank_ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    # TopKLabelMass 与 CovK_tilde 累计（评估期按样本加权）
    topk_label_mass_sums = {int(k): 0.0 for k in rank_ks}
    cov_tilde_sums = {int(k): 0.0 for k in rank_ks}
    # 追加：Oracle/Domain/Correlation/Selection/Connectivity 健康指标累计（按样本加权）
    oracle_cov_sums = {int(k): 0.0 for k in rank_ks}
    oracle_hit_sums = {int(k): 0.0 for k in rank_ks}
    sensor_hit_sums = {int(k): 0.0 for k in rank_ks}
    cap_ratio_sum = 0.0
    spearman_s_sum = 0.0
    spearman_p_sum = 0.0
    conn_rate_sum = 0.0
    island_rate_sum = 0.0
    # 混淆计数累计（用于 Balanced-Acc 与 Macro-F1）
    tp_sum = fp_sum = fn_sum = p_sum = None
    samp_sum = 0
    count = 0
    # Connected recall@K（可选特性）累计
    conn_enabled = False
    conn_hit_sums = {int(k): 0.0 for k in rank_ks}
    adj = {}
    # 评估审计：scope 字段（候选池大小/是否应用裁剪/掩码来源/是否全图评估/样本数）
    metrics_full_graph_flag = True
    candidate_pool_size = None
    prune_applied_flag = False
    mask_source = 'none'
    # 推导日志目录（用于 eval 期的可行性/因果/选择器诊断 CSV），优先来自 cfg 派生 -> model.logs_dir，其次 RUN_DIR/logs
    # 纯追加：不移除原有 env 回退，只是提前设置 model.logs_dir 以满足 G3（不读 env 优先）
    try:
        cfg_obj = getattr(model, 'cfg', None)
        if cfg_obj is not None:
            derived_logs = derive_logs_dir(cfg_obj)
            try:
                setattr(model, 'logs_dir', derived_logs)
            except Exception:
                pass
            try:
                run_dir_from_cfg = getattr(getattr(cfg_obj, 'paths', None), 'run_dir', None)
                if run_dir_from_cfg:
                    setattr(model, 'run_dir', run_dir_from_cfg)
            except Exception:
                pass
    except Exception:
        pass
    try:
        logs_dir = str(getattr(model, 'logs_dir', '') or '')
    except Exception:
        logs_dir = ''
    if not logs_dir:
        try:
            run_dir = os.getenv('RUN_DIR', '')
            if run_dir:
                logs_dir = os.path.join(run_dir, 'logs')
        except Exception:
            logs_dir = ''
    # 在评估首段统一写 effective_config.json（纯追加）
    try:
        if isinstance(cfg_obj, object):
            # 构造一个最小 effective dict（训练/特征/损失摘要）
            eff = {
                'paths': {
                    'run_dir': getattr(getattr(cfg_obj, 'paths', None), 'run_dir', None),
                    'logs_dir': logs_dir,
                },
                'features': {
                    'selector_head_enabled': bool(getattr(model, 'selector_head_enabled', False)),
                    'selector_normalization': str(getattr(model, 'selector_normalization', 'sigmoid')),
                    'selector_target_k': int(getattr(model, 'selector_target_k', 0) or 0),
                    'softmax_temperature': float(getattr(model, 'softmax_temperature', 1.0) or 1.0),
                    'prune_upstream': {
                        'enabled': bool(getattr(model, 'prune_upstream_enabled', False)),
                        'window_s': float(getattr(model, 'prune_upstream_window_s', 0.0) or 0.0),
                    },
                    'connected_eval_enabled': bool(getattr(model, 'connected_eval_enabled', False)),
                },
                'loss': {
                    'causal': {
                        'enabled': bool(getattr(model, 'loss_causal_enabled', False)),
                        'weight': float(getattr(model, 'loss_causal_weight', 0.0) or 0.0),
                        'tau': float(getattr(model, 'loss_causal_tau', 0.0) or 0.0),
                        'mode': 'static',
                    },
                    'tv': {
                        'enabled': bool(getattr(model, 'loss_tv_enabled', False)),
                        'weight': float(getattr(model, 'loss_tv_weight', 0.0) or 0.0),
                        'tau': float(getattr(model, 'loss_tv_tau', 0.0) or 0.0),
                    },
                    'coverage': {
                        # 兼容两套命名：loss_coverage_*（train.py 注入）与历史 loss_cov_* / coverage_alpha
                        'enabled': bool(getattr(model, 'loss_coverage_enabled', getattr(model, 'loss_cov_enabled', False))),
                        'alpha': float(getattr(model, 'loss_coverage_alpha', getattr(model, 'coverage_alpha', 0.0)) or 0.0),
                        'weight': float(getattr(model, 'loss_coverage_weight', getattr(model, 'loss_cov_weight', 0.0)) or 0.0),
                    },
                    'size': {
                        'enabled': bool(getattr(model, 'loss_size_enabled', False)),
                        'weight': float(getattr(model, 'loss_size_weight', 0.0) or 0.0),
                    },
                },
                'training': {
                    'rank_ks': tuple(int(k) for k in rank_ks),
                },
                'model': {
                    'selector_head_enabled': bool(getattr(model, 'selector_head_enabled', False)),
                },
            }
            applied_overrides_list = []
            try:
                applied_overrides_list = list(getattr(model, 'applied_overrides', []) or [])
            except Exception:
                applied_overrides_list = []
            # ΔS4：确保 applied_overrides 显式包含 selector_head 两键（纯追加）
            try:
                if 'features.selector_head_enabled' not in applied_overrides_list:
                    applied_overrides_list.append('features.selector_head_enabled')
                if 'model.selector_head_enabled' not in applied_overrides_list:
                    applied_overrides_list.append('model.selector_head_enabled')
            except Exception:
                pass
            # run_dir 可能为空，传空以由函数内部派生
            try:
                run_dir_for_snapshot = getattr(getattr(cfg_obj, 'paths', None), 'run_dir', None) or ''
            except Exception:
                run_dir_for_snapshot = ''
            write_effective_config_snapshot(run_dir_for_snapshot, cfg_obj, eff, applied_overrides_list)
        # 确保 logs_dir 目录存在
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
        # 记录 metrics_full_graph 开关（用于审计证据）
        try:
            feat_ce = getattr(getattr(cfg_obj, 'features', object()), 'connected_eval', None)
            metrics_full_graph = True
            if feat_ce is not None and hasattr(feat_ce, 'metrics_full_graph'):
                metrics_full_graph = bool(getattr(feat_ce, 'metrics_full_graph'))
            else:
                # 若缺失键，默认 True 并记录一次
                metrics_full_graph = True
            # 硬性协议：评估端强制全图评估
            metrics_full_graph = True
            if logs_dir:
                os.makedirs(logs_dir, exist_ok=True)
                fb_path = os.path.join(logs_dir, 'eval_fallback.jsonl')
                rec = {
                    'event': 'config',
                    'reason': 'metrics_full_graph',
                    'metrics_full_graph': bool(metrics_full_graph),
                    'ts': datetime.now().isoformat(timespec='seconds'),
                }
                with open(fb_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # 同步到评估审计标志
                metrics_full_graph_flag = bool(metrics_full_graph)
        except Exception:
            pass
        # ΔS1：检测旧键 features.enable_selector_head 的存在并记录统一动作（仅审计，不中断）
        try:
            feat = getattr(cfg_obj, 'features', None)
            if feat is not None and hasattr(feat, 'enable_selector_head'):
                # 若已存在统一键，则不再记录重复键审计，避免噪声
                if not hasattr(feat, 'selector_head_enabled'):
                    extra = {
                        'detail': 'features.enable_selector_head present; unified to features.selector_head_enabled'
                    }
                    try:
                        path = os.path.join(logs_dir, 'eval_fallback.jsonl')
                        rec = {'event': 'config', 'reason': 'duplicate_selector_head_keys', 'ts': datetime.now().isoformat(timespec='seconds')}
                        rec.update(extra)
                        with open(path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
            mdl = getattr(cfg_obj, 'model', None)
            if mdl is not None and hasattr(mdl, 'enable_selector_head'):
                # 同理：若统一键已存在，则忽略重复键审计
                if not hasattr(mdl, 'selector_head_enabled'):
                    extra = {
                        'detail': 'model.enable_selector_head present; unified to model.selector_head_enabled'
                    }
                    try:
                        path = os.path.join(logs_dir, 'eval_fallback.jsonl')
                        rec = {'event': 'config', 'reason': 'duplicate_selector_head_keys', 'ts': datetime.now().isoformat(timespec='seconds')}
                        rec.update(extra)
                        with open(path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass

    # ΔS1：放宽形状检查助手（尽量适配为 [B,N]；失败返回 None）。
    def _ensure_selector_like(u: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """宽松模式：尝试将 u 适配为形状 [B,N]。
        支持以下常见情况：
        - [B,N]
        - [N] 且 B==1
        - [B,1,N] 或 [B,N,1]
        - [N,B]（将转置为 [B,N]）
        其余情况保持 None。
        """
        try:
            if not isinstance(u, torch.Tensor):
                return None
            # 直接匹配 [B,N]
            if (u.dim() == 2) and (int(u.shape[0]) == B) and (int(u.shape[1]) == N):
                return u
            # [N] 且 B==1
            if (u.dim() == 1) and (B == 1) and (int(u.shape[0]) == N):
                return u.view(1, N)
            # [B,1,N] 或 [B,N,1]
            if u.dim() == 3:
                if (int(u.shape[0]) == B) and (int(u.shape[1]) == 1) and (int(u.shape[2]) == N):
                    return u.view(B, N)
                if (int(u.shape[0]) == B) and (int(u.shape[1]) == N) and (int(u.shape[2]) == 1):
                    return u.view(B, N)
            # [N,B] -> [B,N]
            if (u.dim() == 2) and (int(u.shape[0]) == N) and (int(u.shape[1]) == B):
                return u.transpose(0, 1).contiguous()
            return None
        except Exception:
            return None

    # 小工具：写入 eval 回退 JSONL（缺键时仅审计，不中断）
    def _append_eval_fallback(event: str, reason: str, extra: dict = None):
        try:
            if not logs_dir:
                return
            os.makedirs(logs_dir, exist_ok=True)
            path = os.path.join(logs_dir, 'eval_fallback.jsonl')
            rec = {'event': str(event), 'reason': str(reason), 'ts': datetime.now().isoformat(timespec='seconds')}
            if isinstance(extra, dict):
                rec.update(extra)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # 读取连通性评估开关并惰性加载邻接表/边列表
    try:
        # 优先从模型属性读取
        conn_enabled = bool(getattr(model, 'connected_eval_enabled', False))
        # 硬性协议：评估端禁用 connected_eval
        conn_enabled = False
        if conn_enabled:
            # 推导 run_dir：优先 model.run_dir，否则 logs_dir 的父目录
            run_dir = ''
            try:
                run_dir = str(getattr(model, 'run_dir', '') or '')
            except Exception:
                run_dir = ''
            if not run_dir and logs_dir:
                run_dir = os.path.dirname(logs_dir)
            # cfg 可选注入（若模型持有 cfg）
            cfg_obj = getattr(model, 'cfg', None)
            # P11：改为从边文件读取并构建邻接；保持原键名与写入格式
            edge_index_eval = None
            try:
                edge_index_eval = load_eval_edges(cfg_obj, run_dir)  # [2,E], int64
            except Exception:
                edge_index_eval = None
            # 记录候选路径（用于越界审计）
            edge_source_path = ''
            try:
                root = str(getattr(getattr(cfg_obj, 'paths', object()), 'root_dir', os.getcwd()) or os.getcwd())
            except Exception:
                root = os.getcwd()
            candidates = [
                os.path.join(run_dir or '', 'artifacts', 'edge_index_main.csv'),
                os.path.join(root, 'artifacts', 'edge_index_main.csv'),
                os.path.join(root, 'data_split', 'pipe_map.csv'),
            ]
            for p in candidates:
                if p and os.path.exists(p):
                    edge_source_path = p
                    break
            # 将边转换成邻接；若为空则保留空邻接并依赖 load_eval_edges 的审计
            try:
                import numpy as _np
                if (edge_index_eval is not None) and isinstance(edge_index_eval, _np.ndarray) and (edge_index_eval.ndim == 2) and (edge_index_eval.shape[1] > 0):
                    # 构建无向邻接
                    adj = {}
                    E = int(edge_index_eval.shape[1])
                    for i in range(E):
                        u = int(edge_index_eval[0, i]); v = int(edge_index_eval[1, i])
                        if u not in adj:
                            adj[u] = set()
                        if v not in adj:
                            adj[v] = set()
                        adj[u].add(v); adj[v].add(u)
                    # 暂存最大索引用于越界检测（N 需等到 logits 可得）
                    try:
                        edge_max_idx = int(edge_index_eval.max())
                    except Exception:
                        edge_max_idx = -1
                    # 记录加载来源
                    try:
                        # 依据候选路径顺序，选择第一个真实包含边的路径
                        from src.evaluation.metrics.connectivity import _read_edges_csv_generic as _read_pairs
                        chosen = ''
                        for p in candidates:
                            if p and os.path.exists(p):
                                pairs = _read_pairs(p)
                                if pairs and len(pairs) == E:
                                    chosen = p
                                    break
                                # 若边数不匹配但非空，优先采用第一条非空路径
                                if pairs:
                                    chosen = p
                                    break
                        if not chosen:
                            chosen = edge_source_path or ''
                        _append_eval_fallback('conn_eval', 'edge_index_loaded', {'edge_source': chosen, 'E': int(E)})
                    except Exception:
                        pass
                else:
                    adj = {}
                    edge_max_idx = -1
                    # 明确记录读取失败（保持原语义）
                    _append_eval_fallback('conn_eval', 'adj_empty_or_missing', {'run_dir': run_dir})
            except Exception:
                adj = {}
                edge_max_idx = -1
    except Exception:
        conn_enabled = False
    # 记录：是否已经写入一条 edges>0 的因果统计（满足 G2）
    edges_positive_logged = False
    # ΔS2 强制产物存在性：若评估首批未写因果统计，则在循环后段进行兜底写入（首批执行一次）
    causal_csv_checked = False
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
                
            # Check for PyG Batch (V6 Architecture)
            # Use same robust check as training loop
            is_pyg_batch = hasattr(batch, 'x') and hasattr(batch, 'edge_index')
            
            if not is_pyg_batch:
                print(f"[EVAL][DEBUG] Batch is not PyG. Type: {type(batch)}")
                if hasattr(batch, 'keys'):
                     print(f"[EVAL][DEBUG] Batch keys: {batch.keys()}")
            
            if is_pyg_batch:
                features = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                edge_attr = batch.edge_attr.to(device)
                # Ensure batch index exists, otherwise create all zeros (single graph case)
                if hasattr(batch, 'batch') and batch.batch is not None:
                    batch_idx_tensor = batch.batch.to(device)
                else:
                    batch_idx_tensor = torch.zeros(features.size(0), dtype=torch.long, device=device)
                
                soft_targets = batch.y.to(device)
                
                needs_full_batch = getattr(model, '_needs_full_batch', False)
                is_autosampling = hasattr(model, 'set_gumbel_temperature')

                # Eval Mode: Revealed (Stage 2) or Sparse Sensor Network
                # Apply masking logic to simulate realistic sparse observations
                if hasattr(batch, 'x_raw_signal') and not is_autosampling:
                    raw_signal = batch.x_raw_signal.to(device)
                    # Channel 4: Trigger (Always visible)
                    is_trigger = (features[:, 4] == 1.0).float()
                    # Channel 5: Sensor (Fixed sensors)
                    is_sensor = (features[:, 5] == 1.0).float()
                    
                    # Determine Mask
                    # Priority 1: Use Fixed Sensors if available
                    if is_sensor.sum() > 0.5:
                        mask = torch.max(is_trigger, is_sensor)
                        # Set source flag for audit
                        mask_source = 'sensors'
                    else:
                        # Priority 2: Fallback to Stage 2 Random Sampling (Simulate sparse checks)
                        # Use deterministic seed based on batch_idx for reproducibility during Eval
                        g_cpu = torch.Generator()
                        g_cpu.manual_seed(batch_idx + 10000) # Fixed seed per batch
                        mask_noise = (torch.rand(is_trigger.shape, generator=g_cpu).to(device) < 0.05).float()
                        mask = torch.max(is_trigger, mask_noise)
                        # Set source flag for audit
                        mask_source = 'stage2_random'
                    
                    # Apply Mask to Input Features
                    # Channel 3: Obs_Valid_Mask
                    features[:, 3] = mask
                    # Channel 0: Raw Signal (Masked)
                    features[:, 0] = raw_signal * mask
                    
                    # Update batch.x for T2I-BDASNet
                    batch.x = features
                
                # Evaluation Phase uses AMP
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    if needs_full_batch:
                        out = model(batch)
                    else:
                        out = model(features, edge_index, edge_attr, batch=batch_idx_tensor)
                    
                    if isinstance(out, dict):
                        class_logits = out['classification']
                    else:
                        class_logits = out
                
                # Convert to dense for metrics
                try:
                    from torch_geometric.utils import to_dense_batch
                    # logits: [Total_N, 1] -> [B, Max_N, 1]
                    dense_logits, mask = to_dense_batch(class_logits, batch_idx_tensor)
                    dense_logits = dense_logits.squeeze(-1) # [B, Max_N]
                    # Pad invalid positions with -inf
                    dense_logits[~mask] = -float('inf')
                    class_logits_metrics = dense_logits
                    class_logits = dense_logits # Also update class_logits for subsequent logic
                    
                    # targets: [Total_N] -> [B, Max_N]
                    dense_targets, _ = to_dense_batch(soft_targets, batch_idx_tensor)
                    soft_targets = dense_targets
                except ImportError:
                    pass
            else:
                features, soft_targets, edge_index, edge_attr, eidx_v, eattr_v = _batch_to_device(batch, device)
                # 评估期使用 AMP 自动类型转换（仅在 CUDA 且训练期启用 AMP 时）
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    out = model(features, edge_index, edge_attr, eidx_v, eattr_v)
                    class_logits = out['classification']
                # 为指标评估保留未裁剪版本（全图评估）
                class_logits_metrics = class_logits
            # 记录候选池大小（节点总数）
            try:
                candidate_pool_size = int(class_logits.shape[1])
            except Exception:
                pass

            # === 选择器 m（评估仅用于统计/因果项；首批写 selector 诊断） ===
            m = None
            norm_used = ''
            try:
                B = int(class_logits.shape[0])
                N = int(class_logits.shape[1])
                u = None
                keys_tried = []
                u_raw = None
                if isinstance(out, dict):
                    # 1) 优先找 selector logits（容忍常见别名），并进行宽松形状适配
                    for key in ['selector_logits', 'logits_selector', 'head_selector_logits', 'selector_scores']:
                        if key in out:
                            keys_tried.append(key)
                            u_raw = out[key]
                            u = _ensure_selector_like(u_raw, B, N)
                            if u is not None:
                                break
                    # 2) 若 logits 不可用，尝试从 prob/mask 恢复为 logits（logit = log(p/(1-p)))
                    if u is None:
                        pb = None
                        for key in ['selector_prob', 'prob_selector', 'selector_mask', 'mask_selector', 'selector_mask_hard']:
                            if key in out:
                                keys_tried.append(key)
                                pb = out[key]
                                # 统一视作概率/得分，尽量适配形状为 [B,N]
                                pb2 = _ensure_selector_like(pb, B, N)
                                if isinstance(pb2, torch.Tensor):
                                    pb2 = torch.clamp(pb2, min=1e-6, max=1.0 - 1e-6)
                                    u = torch.log(pb2) - torch.log(1.0 - pb2)
                                    break
                    # 3) 记录形状信息（仅审计），不抛异常
                    if (u is None) and keys_tried:
                        shape_detail = None
                        try:
                            if isinstance(u_raw, torch.Tensor):
                                shape_detail = [int(x) for x in list(u_raw.shape)]
                        except Exception:
                            shape_detail = None
                        _append_eval_fallback('selector', 'bad_shape_selector_source', {
                            'B': B, 'N': N,
                            'got_shape': shape_detail,
                            'keys_tried': keys_tried,
                            'strict': False,
                        })
                # 4) 若仍无 u，允许使用 subgraph_targets 作为回退来源（仅用于审计与覆盖损失统计）
                if u is not None:
                    tau = float(getattr(model, 'softmax_temperature', 0.9) or 0.9)
                    # K：优先 cfg.features.selector_target_k；若未设置则回退为 min(30, N)
                    K_cfg = int(getattr(model, 'selector_target_k', 0) or 0)
                    K_eff = int(K_cfg if K_cfg > 0 else min(30, N))
                    m = torch.softmax(u / max(1e-6, tau), dim=1) * float(K_eff)
                    norm_used = 'softmaxK'
                else:
                    # 回退：使用 subgraph_targets 作为 m（形状对齐为 [B,N]）
                    norm_used = 'fallback_subgraph_targets'
                    if isinstance(batch, dict) and ('subgraph_targets' in batch):
                        m2 = batch['subgraph_targets']
                        if isinstance(m2, torch.Tensor):
                            if m2.dim() == 1 and B == 1:
                                m = m2.view(1, -1)
                            elif m2.dim() == 2:
                                m = m2
                    if m is None:
                        _append_eval_fallback('selector', 'missing_selector_all_sources', {'B': B, 'N': N})
                if (m is not None) and (batch_idx == 0):
                    try:
                        os.makedirs(logs_dir, exist_ok=True)
                        # 使用统一 CSV 写入工具，确保首行形如 norm=softmaxK,K=...,B=...,N=...,sum_m_avg=...
                        K_cfg = int(getattr(model, 'selector_target_k', 0) or 0)
                        if K_cfg <= 0:
                            K_cfg = int(min(30, int(m.shape[1])))
                        try:
                            write_selector_mask_csv(logs_dir, m, K=K_cfg, norm_type=norm_used)
                        except Exception:
                            pass
                        # 兼容保留原追加日志（不删除原逻辑）
                        try:
                            csv_path = os.path.join(logs_dir, 'selector_mask_eval.csv')
                            header_needed = (not os.path.exists(csv_path))
                            with open(csv_path, 'a', encoding='utf-8') as f:
                                if header_needed:
                                    f.write('timestamp,batch_size,num_nodes,mean_m,sum_m,K,norm_type,node_ids\n')
                                from datetime import datetime as _dt
                                ts = _dt.now().isoformat(timespec='seconds')
                                mean_m = float(m.mean().item())
                                sum_m = float(m.sum().item())
                                # 近似 Top-K 节点（仅首样本）：按 m 的首行取前 K 索引
                                try:
                                    import torch as _torch
                                    N0 = int(m.shape[1])
                                    K_eff = K_cfg if K_cfg > 0 else max(1, min(5, N0))
                                    topk = _torch.topk(m[0], k=min(K_eff, N0), dim=0, largest=True, sorted=False)
                                    nodes = '[' + ','.join(str(int(x)) for x in topk.indices.detach().cpu().view(-1).tolist()) + ']'
                                except Exception:
                                    nodes = ''
                                f.write(f"{ts},{int(m.shape[0])},{int(m.shape[1])},{mean_m:.6f},{sum_m:.6f},{K_cfg},{norm_used},{nodes}\n")
                            # 结束兼容追加日志的 try 块
                            
                        except Exception:
                            pass
                        # 控制台诊断
                        sm_avg = float(m.sum(dim=1).mean().item())
                        ratio = (sm_avg / float(K_cfg)) if K_cfg > 0 else 0.0
                        msg = f"[SELECTOR][eval] sum(m)={sm_avg:.4f} K_cfg={K_cfg} ratio={ratio:.4f} norm={norm_used}"
                        print(msg)
                        # 追加 gate 摘要到文件
                        try:
                            with open(os.path.join(logs_dir, 'selector_gate.txt'), 'w', encoding='utf-8') as gf:
                                gf.write(msg + "\n")
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                # 选择器诊断失败不影响评估
                pass
            # 若未能构造 m，仍在首批强制打印一行（norm 显示为 fallback_subgraph_targets）
            if (m is None) and (batch_idx == 0):
                try:
                    K_cfg = int(getattr(model, 'selector_target_k', 0) or 0)
                    if K_cfg <= 0:
                        K_cfg = int(min(30, int(class_logits.shape[1])))
                    print(f"[SELECTOR][eval] sum(m)=NA K_cfg={K_cfg} ratio=NA norm=fallback_subgraph_targets")
                except Exception:
                    pass

            # === 上游可达硬裁剪（评估阶段参与推断，首批写统计 CSV 到 *_eval.csv） ===
            try:
                prune_enabled = bool(getattr(model, 'prune_upstream_enabled', False))
                # 硬性协议：评估端禁用上游可达裁剪
                if prune_enabled:
                    _append_eval_fallback('prune', 'disabled_in_eval_protocol', {'was_enabled': True})
                prune_enabled = False
                if prune_enabled and isinstance(batch, dict):
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
                    if window_s > 0 and isinstance(ei_flow, torch.Tensor) and isinstance(anom, torch.Tensor):
                        feasible = build_upstream_feasible_mask(ei_flow, tt, anom, window_s=window_s)  # [B,N]
                        if batch_idx == 0:
                            try:
                                kept = int(feasible.sum().item())
                                ratio = float(kept / max(1, feasible.numel()))
                                has_tt = isinstance(tt, torch.Tensor)
                                os.makedirs(logs_dir, exist_ok=True)
                                path = os.path.join(logs_dir, 'feasible_stats_eval.csv')
                                header = (not os.path.exists(path))
                                with open(path, 'a', encoding='utf-8') as f:
                                    if header:
                                        f.write('kept,ratio,window_s,has_tt\n')
                                    f.write(f"{kept},{ratio:.6f},{float(window_s):.1f},{bool(has_tt)}\n")
                                print(f"[PRUNE][eval] kept={kept} ratio={ratio:.6f} window_s={window_s} has_tt={has_tt}")
                                # 若比例不在目标区间，给出 window_s 调整建议，写入文件并记录回退
                                try:
                                    suggest_msg = ''
                                    new_ws = None
                                    if ratio > 0.60:
                                        new_ws = max(1.0, float(window_s) * 0.75)  # 例如 2400->1800
                                        suggest_msg = f"ratio>0.60，建议下调 window_s 至 {new_ws:.1f}（如 {window_s:.1f}→{new_ws:.1f}）"
                                    elif ratio < 0.35:
                                        new_ws = float(window_s) * 1.25  # 例如 2400->3000
                                        suggest_msg = f"ratio<0.35，建议上调 window_s 至 {new_ws:.1f}（如 {window_s:.1f}→{new_ws:.1f}）"
                                    if suggest_msg:
                                        print(f"[PRUNE][suggest] {suggest_msg}")
                                        try:
                                            with open(os.path.join(logs_dir, 'prune_suggestion.txt'), 'a', encoding='utf-8') as sf:
                                                sf.write(suggest_msg + "\n")
                                        except Exception:
                                            pass
                                        _append_eval_fallback('prune', 'ratio_out_of_range', {'ratio': ratio, 'window_s': float(window_s), 'suggest_window_s': float(new_ws or window_s)})
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        # 评估阶段：若启用全图指标，避免硬裁剪影响 recall/hit@K；否则按旧逻辑掩码
                        mask = (feasible > 0.5)
                        neg_inf = torch.finfo(class_logits.dtype).min if class_logits.dtype.is_floating_point else -1e9
                        from_cfg = getattr(getattr(getattr(model, 'cfg', object()), 'features', object()), 'connected_eval', None)
                        try:
                            metrics_full_graph_flag = True
                            if from_cfg is not None and hasattr(from_cfg, 'metrics_full_graph'):
                                metrics_full_graph_flag = bool(getattr(from_cfg, 'metrics_full_graph'))
                        except Exception:
                            metrics_full_graph_flag = True
                        # 硬性协议：强制全图评估
                        metrics_full_graph_flag = True
                        if metrics_full_graph_flag:
                            # 对指标使用全图 logits（class_logits_metrics 不变）；若需要“推断掩码”，保持 class_logits 原样
                            class_logits_metrics = class_logits
                        else:
                            # 非全图评估时，沿用掩码（保持向后兼容）
                            class_logits = class_logits.masked_fill(~mask, neg_inf)
                            class_logits_metrics = class_logits
                            prune_applied_flag = True
                            mask_source = 'upstream_feasible'
                    else:
                        if batch_idx == 0:
                            print("[EVAL][SKIP][PRUNE reason=missing_keys_or_window<=0]")
                            _append_eval_fallback('prune', 'missing_keys_or_window<=0', {'has_edge_index_flow': isinstance(ei_flow, torch.Tensor), 'has_anomaly_mask': isinstance(anom, torch.Tensor), 'window_s': window_s})
                elif prune_enabled and batch_idx == 0:
                    print("[EVAL][SKIP][PRUNE reason=non_dict_batch]")
                    _append_eval_fallback('prune', 'non_dict_batch')
            except Exception:
                pass

            # === 先验软约束融合（不改 forward 接口；评估仅影响 logits，不得缩减候选集） ===
            try:
                cfg_prior = getattr(getattr(getattr(model, 'cfg', object()), 'features', object()), 'prior', None)
                prior_enabled = bool(getattr(cfg_prior, 'enabled', False)) if cfg_prior is not None else False
                if prior_enabled:
                    # 从 batch 推导先验（若可用）；否则跳过融合
                    B = int(class_logits.shape[0]); N = int(class_logits.shape[1])
                    prior_tensor = derive_prior_from_batch(batch, N)
                    if prior_tensor is not None:
                        mode = str(getattr(cfg_prior, 'mode', 'log_add') or 'log_add')
                        alpha = float(getattr(cfg_prior, 'alpha', 0.0) or 0.0)
                        beta = float(getattr(cfg_prior, 'beta', 0.0) or 0.0)
                        eps = float(getattr(cfg_prior, 'eps', 1e-12) or 1e-12)
                        base_T = float(getattr(model, 'softmax_temperature', 1.0) or 1.0)
                        class_logits, T_eff = fuse_logits_with_prior(class_logits, prior_tensor, mode=mode, alpha=alpha, beta=beta, eps=eps, base_temperature=base_T)
                        # 指标评估也使用融合后的 logits（不影响候选范围）
                        class_logits_metrics = class_logits
                        # 记录一次融合事件（首批）
                        if batch_idx == 0 and logs_dir:
                            _append_eval_fallback('prior', 'applied_prior_fusion', {'mode': mode, 'alpha': alpha, 'beta': beta, 'eps': eps, 'T_eff': float(T_eff)})
                else:
                    if batch_idx == 0 and logs_dir:
                        _append_eval_fallback('prior', 'disabled')
            except Exception:
                # 任何异常不影响评估主流程
                pass

            # === 因果软正则（评估仅统计，不入验证损失） ===
            try:
                causal_enabled = bool(getattr(model, 'loss_causal_enabled', False))
                if causal_enabled and isinstance(batch, dict):
                    dev = class_logits.device
                    ei_flow = batch.get('edge_index_flow')
                    tt = batch.get('travel_time')
                    has_flow = isinstance(ei_flow, torch.Tensor)
                    # travel_time 可选：compute_causal_loss_static 在 tt 缺失时会退化为均权重 1.0
                    has_tt = isinstance(tt, torch.Tensor)
                    if has_flow:
                        ei_flow = ei_flow.to(dev, non_blocking=True)
                    if has_tt:
                        tt = tt.to(dev, non_blocking=True)
                    tau_c = float(getattr(model, 'loss_causal_tau', 1200.0) or 1200.0)
                    # 注意：评估阶段 tt 缺失不再阻止统计（与 compute_causal_loss_static 保持一致）；仅要求 edge_index_flow 与 m。
                    if not (has_flow and (m is not None)):
                        # 缺键或 m 缺失：记录回退并打印静默行（edges=0）
                        _append_eval_fallback('causal', 'missing_keys', {'has_flow': bool(has_flow), 'has_tt': bool(has_tt), 'has_m': bool(m is not None)})
                        # 细化缺失原因，便于排查
                        try:
                            if not has_flow:
                                _append_eval_fallback('causal', 'missing_edge_index_flow')
                            if not has_tt:
                                _append_eval_fallback('causal', 'missing_travel_time')
                            if m is None:
                                _append_eval_fallback('causal', 'missing_selector_logits')
                        except Exception:
                            pass
                        if batch_idx == 0:
                            try:
                                print(f"[CAUSAL][eval] edges=0 viol=0.0000 L_causal=0.000000 tau={float(tau_c):.1f} mode=static")
                                # 写入 CSV 首行（edges=0），以便后续补充 edges>0 行
                                os.makedirs(logs_dir, exist_ok=True)
                                path = os.path.join(logs_dir, 'causal_stats_eval.csv')
                                header = (not os.path.exists(path))
                                with open(path, 'a', encoding='utf-8') as f:
                                    if header:
                                        f.write('edges,viol_ratio,w_min,w_max,w_mean,L_causal,mode,tau\n')
                                    f.write(f"0,0.000000,0.000000,1.000000,1.000000,0.00000000,static,{float(tau_c):.1f}\n")
                            except Exception:
                                pass
                    else:
                        l_causal, stats = compute_causal_loss_static(m, ei_flow, tt, tau=tau_c)
                        # 写入 CSV 与控制台打印：首批必打印；若后续首次出现 edges>0，也追加一行并打印
                        try:
                            os.makedirs(logs_dir, exist_ok=True)
                            path = os.path.join(logs_dir, 'causal_stats_eval.csv')
                            header = (not os.path.exists(path))
                            row = f"{int(stats.get('edges', 0))},{float(stats.get('viol_ratio', 0.0)):.6f},{float(stats.get('w_min', 0.0)):.6f},{float(stats.get('w_max', 1.0)):.6f},{float(stats.get('w_mean', 1.0)):.6f},{float(l_causal.detach().item()):.8f},static,{float(tau_c):.1f}\n"
                            write_row = False
                            if batch_idx == 0:
                                write_row = True
                            elif (not edges_positive_logged) and int(stats.get('edges', 0)) > 0:
                                write_row = True
                            if write_row:
                                with open(path, 'a', encoding='utf-8') as f:
                                    if header:
                                        f.write('edges,viol_ratio,w_min,w_max,w_mean,L_causal,mode,tau\n')
                                    f.write(row)
                                if int(stats.get('edges', 0)) > 0:
                                    edges_positive_logged = True
                                else:
                                    # ΔS3：当真实计算得到 edges==0 时，记录因果无边的审计事件
                                    _append_eval_fallback('causal', 'causal_no_edges', {'tau': float(tau_c)})
                            # 控制台打印
                            if (batch_idx == 0) or int(stats.get('edges', 0)) > 0:
                                print(f"[CAUSAL][eval] edges={int(stats.get('edges', 0))} viol={float(stats.get('viol_ratio', 0.0)):.4f} L_causal={float(l_causal.detach().item()):.6f} tau={float(tau_c):.1f} mode=static")
                        except Exception:
                            pass
                elif causal_enabled and (batch_idx == 0):
                    # 非 dict 批次：首批也生成一行静态因果统计，满足产物存在性要求
                    tau_c = float(getattr(model, 'loss_causal_tau', 1200.0) or 1200.0)
                    # 控制台打印（尽可能不影响后续文件写入）
                    try:
                        print(f"[CAUSAL][eval] edges=0 viol=0.0000 L_causal=0.000000 tau={float(tau_c):.1f} mode=static")
                    except Exception:
                        pass
                    # 写入占位 CSV 行
                    try:
                        os.makedirs(logs_dir, exist_ok=True)
                        path = os.path.join(logs_dir, 'causal_stats_eval.csv')
                        header = (not os.path.exists(path))
                        with open(path, 'a', encoding='utf-8') as f:
                            if header:
                                f.write('edges,viol_ratio,w_min,w_max,w_mean,L_causal,mode,tau\n')
                            f.write(f"0,0.000000,0.000000,1.000000,1.000000,0.00000000,static,{float(tau_c):.1f}\n")
                    except Exception:
                        pass
            except Exception:
                pass

            # 首批记录 metrics_scope，用于后续写 outputs/val/metrics_scope.json
            try:
                if batch_idx == 0:
                    try:
                        N_total = int(class_logits.shape[1])
                    except Exception:
                        N_total = int(candidate_pool_size or 0)
                    N_eval_used = int(class_logits_metrics.shape[1]) if isinstance(class_logits_metrics, torch.Tensor) else int(N_total)
                    scope = {
                        'N_total': int(N_total),
                        'N_eval_used': int(N_eval_used),
                        'candidate_pool_size': int(candidate_pool_size or N_total or 0),
                        'metrics_full_graph': True,
                        'prune_applied': bool(prune_applied_flag),
                        'mask_source': str(mask_source or 'eval_full_graph'),
                    }
                    setattr(model, 'metrics_scope_last', scope)
            except Exception:
                pass

            # ΔS2 兜底：如果首批尚未生成 causal_stats_eval.csv，则写入一行静态占位，确保文件存在
            try:
                if (batch_idx == 0) and (not causal_csv_checked):
                    os.makedirs(logs_dir, exist_ok=True)
                    path = os.path.join(logs_dir, 'causal_stats_eval.csv')
                    need_header = (not os.path.exists(path))
                    if need_header:
                        tau_c = float(getattr(model, 'loss_causal_tau', 1200.0) or 1200.0)
                        with open(path, 'a', encoding='utf-8') as f:
                            f.write('edges,viol_ratio,w_min,w_max,w_mean,L_causal,mode,tau\n')
                            f.write(f"0,0.000000,0.000000,1.000000,1.000000,0.00000000,static,{float(tau_c):.1f}\n")
                        # 控制台打印（仅一次）
                        try:
                            print(f"[CAUSAL][eval] edges=0 viol=0.0000 L_causal=0.000000 tau={float(tau_c):.1f} mode=static")
                        except Exception:
                            pass
                    causal_csv_checked = True
            except Exception:
                pass

            if use_soft_labels:
                loss = criterion(class_logits, soft_targets)
            else:
                hard_targets = hard_labels_from(soft_targets)
                loss = criterion(class_logits, hard_targets)
            # 可选：评估子图损失（仅统计，不影响流程）；传入 edge_attr
            if (subgraph_criterion is not None) and isinstance(batch, dict) and ('subgraph_targets' in batch):
                try:
                    subg_loss = subgraph_criterion(
                        class_logits,
                        batch['subgraph_targets'].to(class_logits.device),
                        edge_index,
                        edge_attr,
                    )
                    loss = loss + subg_loss
                except Exception:
                    pass
            total_loss += float(loss.item())
            acc, _, _, _ = _compute_metrics(class_logits_metrics.detach(), soft_targets)
            total_acc += acc
            try:
                ce = float(getattr(criterion, 'last_ce', 0.0) or 0.0)
                kl = float(getattr(criterion, 'last_kl', 0.0) or 0.0)
                clsmix = float(getattr(criterion, 'last_class', 0.0) or 0.0)
                alpha = float(getattr(criterion, 'last_alpha', soft_alpha_default))
                Tnow = float(getattr(criterion, 'last_T', kl_t_default))
            except Exception:
                ce = kl = clsmix = 0.0; alpha = soft_alpha_default; Tnow = kl_t_default
            ce_sum += ce; kl_sum += kl; class_sum += clsmix; total_sum += float(loss.item()); alpha_sum += alpha; T_sum += Tnow
            hits, mrr_b, ndcg_b = _compute_rank_metrics(class_logits_metrics.detach(), soft_targets, ks=rank_ks, ndcg_k=int(ndcg_k))
            bsz = int(class_logits_metrics.shape[0])
            for k in hit_sums.keys():
                hit_sums[k] += hits.get(k, 0.0) * bsz
            mrr_sum += mrr_b * bsz
            ndcg_sum += ndcg_b * bsz
            # === 评估期 TopKLabelMass@k 与 CovK_tilde@k 累计（按样本加权） ===
            try:
                # DISABLED for Performance: Heavy metrics causing eval slowdown
                pass
                # p = torch.softmax(class_logits_metrics, dim=1)
                # for k_dyn in rank_ks:
                #     lm_avg_b, cov_avg_b = compute_topk_label_mass_and_cov_tilde(p, int(k_dyn))
                #     topk_label_mass_sums[int(k_dyn)] += float(lm_avg_b) * float(bsz)
                #     cov_tilde_sums[int(k_dyn)] += float(cov_avg_b) * float(bsz)
                # # === Oracle 覆盖与命中（基于软标签 y） ===
                # try:
                #     # anomaly mask（可选）：优先 anomaly_nodes 其次 subgraph_targets；缺失时退化逻辑由 oracle_hit_at_k 处理
                #     anomaly_mask = None
                #     if isinstance(batch, dict):
                #         am = batch.get('anomaly_nodes', batch.get('subgraph_targets'))
                #         if isinstance(am, torch.Tensor):
                #             anomaly_mask = am.to(soft_targets.device)
                #     for k_dyn in rank_ks:
                #         cov_b = coverage_CK(soft_targets, int(k_dyn))
                #         hit_b = oracle_hit_at_k(soft_targets, int(k_dyn), anomaly_mask)
                #         oracle_cov_sums[int(k_dyn)] += float(cov_b) * float(bsz)
                #         oracle_hit_sums[int(k_dyn)] += float(hit_b) * float(bsz)
                # except Exception:
                #     pass
                # # === 领域指标：传感器域命中率@K（基于模型概率） ===
                # try:
                #     sensor_mask = None
                #     if isinstance(batch, dict) and ('sensor_mask' in batch):
                #         sm = batch.get('sensor_mask')
                #         if isinstance(sm, torch.Tensor):
                #             sensor_mask = sm.to(p.device)
                #     for k_dyn in rank_ks:
                #         s_hit_b = model_hit_at_k_sensor(p, int(k_dyn), sensor_mask)
                #         sensor_hit_sums[int(k_dyn)] += float(s_hit_b) * float(bsz)
                # except Exception:
                #     pass
                # # === 选择器掩码占比（若 m 可用） ===
                # try:
                #     if m is not None:
                #         cap_ratio_b = cap_mask_ratio(m)
                #         cap_ratio_sum += float(cap_ratio_b) * float(bsz)
                # except Exception:
                #     pass
                # # === Spearman 相关（logits/probs vs labels） ===
                # try:
                #     s_logits = spearman_batch_logits_and_labels(class_logits_metrics.detach(), soft_targets)
                #     s_probs = spearman_batch_probs_and_labels(p, soft_targets)
                #     spearman_s_sum += float(s_logits) * float(bsz)
                #     spearman_p_sum += float(s_probs) * float(bsz)
                # except Exception:
                #     pass
                # === 连通性健康：选中子图的 LCC 覆盖率与孤岛率（基于 edge_index 与 m 的 TopK） ===
                # DISABLED for Performance: O(B*E) complexity in Python causes huge slowdown per batch.
                # try:
                #     if (m is not None) and isinstance(edge_index, torch.Tensor) and (edge_index.dim() == 2) and (int(edge_index.shape[0]) == 2) and (int(edge_index.shape[1]) > 0):
                #         k_eff = int(getattr(model, 'selector_target_k', 0) or int(m.shape[1]))
                #         k_eff = max(1, min(k_eff, int(m.shape[1])))
                #         # 预取边列表到CPU以减少重复 .detach()
                #         ei0 = edge_index[0].detach().cpu().view(-1).tolist()
                #         ei1 = edge_index[1].detach().cpu().view(-1).tolist()
                #         Bm = int(m.shape[0])
                #         for bb in range(Bm):
                #             try:
                #                 topk_idx = torch.topk(m[bb], k=k_eff, dim=0, largest=True, sorted=False).indices.detach().cpu().view(-1).tolist()
                #             except Exception:
                #                 topk_idx = []
                #             sel = set(int(i) for i in topk_idx)
                #             adj_b = {}
                #             if sel:
                #                 E2 = len(ei0)
                #                 for i_e in range(E2):
                #                     u = int(ei0[i_e]); v = int(ei1[i_e])
                #                     if (u in sel) and (v in sel):
                #                         if u not in adj_b:
                #                             adj_b[u] = set()
                #                         if v not in adj_b:
                #                             adj_b[v] = set()
                #                         adj_b[u].add(v); adj_b[v].add(u)
                #                 # 确保孤立点出现于邻接中
                #                 for u in sel:
                #                     if u not in adj_b:
                #                         adj_b[u] = set()
                #             # 若 sel 为空，adj_b 保持空字典，指标将返回 0.0
                #             conn_rate_sum += float(connectivity_rate(adj_b))
                #             island_rate_sum += float(island_rate(adj_b))
                # except Exception:
                #     pass
            except Exception:
                pass
            # === 节点越界审计（仅一次，发现越界则当轮降级为空邻接） ===
            try:
                if conn_enabled and (batch_idx == 0):
                    N = int(class_logits.shape[1])
                    if 'edge_max_idx' in locals():
                        max_idx = int(edge_max_idx)
                    else:
                        max_idx = -1
                    if (max_idx >= 0) and (max_idx >= N):
                        _append_eval_fallback('conn_eval', 'edge_index_out_of_range', {'N': int(N), 'max_idx': int(max_idx), 'path': edge_source_path})
                        # 降级为空邻接，避免异常中断
                        adj = {}
            except Exception:
                pass

            # === Connected recall@K（可选） ===
            try:
                if conn_enabled and adj and isinstance(class_logits_metrics, torch.Tensor) and isinstance(soft_targets, torch.Tensor):
                    # 逐样本
                    for b in range(int(class_logits_metrics.shape[0])):
                        # true_id 从 soft_targets argmax 提取
                        true_id = int(torch.argmax(soft_targets[b]).detach().item())
                        scores_b = class_logits_metrics[b].detach()
                        for k in rank_ks:
                            val = connected_recall_at_k(scores_b, true_id=true_id, K=int(k), adj=adj)
                            conn_hit_sums[int(k)] += float(val)
                elif conn_enabled and (not adj) and batch_idx == 0:
                    # 尝试从 dataloader 的边作为回退来源，避免出现 adj_empty_or_missing
                    try:
                        if isinstance(edge_index, torch.Tensor) and (edge_index.dim() == 2) and (int(edge_index.shape[0]) == 2) and (int(edge_index.shape[1]) > 0):
                            E2 = int(edge_index.shape[1])
                            adj2 = {}
                            for i in range(E2):
                                u = int(edge_index[0, i].detach().item()); v = int(edge_index[1, i].detach().item())
                                if u < 0 or v < 0:
                                    continue
                                if u not in adj2:
                                    adj2[u] = set()
                                if v not in adj2:
                                    adj2[v] = set()
                                adj2[u].add(v); adj2[v].add(u)
                            if adj2:
                                adj = adj2
                                _append_eval_fallback('conn_eval', 'edge_index_loaded', {'edge_source': 'dataloader.edge_index', 'E': int(E2)})
                    except Exception:
                        pass
            except Exception:
                # 连通性评估失败不阻断整体评估
                pass
            # 混淆计数累计
            try:
                tp, fp, fn, p = _confusion_counts_from_logits_targets(class_logits_metrics.detach(), soft_targets)
                if tp_sum is None:
                    tp_sum, fp_sum, fn_sum, p_sum = tp.clone(), fp.clone(), fn.clone(), p.clone()
                else:
                    tp_sum += tp; fp_sum += fp; fn_sum += fn; p_sum += p
            except Exception:
                pass
            samp_sum += bsz
            count += 1
            if max_eval_steps > 0 and count >= max_eval_steps:
                break
    denom = max(1, count)
    samp_denom = max(1, samp_sum)
    base = (
        total_loss / denom,
        total_acc / denom,
        ce_sum / denom,
        kl_sum / denom,
        class_sum / denom,
        total_sum / denom,
        alpha_sum / denom,
        T_sum / denom,
    )
    hit_vals = tuple((hit_sums[k] / samp_denom) for k in rank_ks)
    tail = (
        mrr_sum / samp_denom,
        ndcg_sum / samp_denom,
    )
    # 若启用连通性评估，写入到模型属性以便上层合并到 JSON
    try:
        if conn_enabled:
            conn_out = {f'conn_recall@{int(k)}': float(conn_hit_sums[int(k)] / samp_denom) for k in rank_ks}
            setattr(model, 'conn_recall_last', conn_out)
    except Exception:
        pass
    # 计算 Balanced-Acc 与 Macro-F1
    try:
        if tp_sum is not None and fp_sum is not None and fn_sum is not None and p_sum is not None:
            bal_acc, macro_f1 = _balanced_acc_and_macro_f1_from_counts(tp_sum, fp_sum, fn_sum, p_sum)
        else:
            bal_acc, macro_f1 = 0.0, 0.0
    except Exception:
        bal_acc, macro_f1 = 0.0, 0.0
    # 写入评估 scope 到模型属性，以便上层写 metrics_scope.json
    try:
        scope = {
            'N_total': int(samp_sum),
            'N_eval_used': int(samp_sum),
            'prune_applied': bool(prune_applied_flag),
            'candidate_pool_size': (int(candidate_pool_size) if candidate_pool_size is not None else None),
            'mask_source': str(mask_source),
            'metrics_full_graph': bool(metrics_full_graph_flag),
        }
        setattr(model, 'metrics_scope_last', scope)
    except Exception:
        pass
    # 记录评估样本数到模型属性，用于吞吐统计
    try:
        setattr(model, 'last_eval_samp_sum', int(samp_sum))
    except Exception:
        pass
    # 将 TopKLabelMass 与 CovK_tilde 的 epoch 均值写入模型属性（供日志/JSON使用）
    try:
        last_label_mass_eval = {int(k): float(topk_label_mass_sums[int(k)] / samp_denom) for k in rank_ks}
        last_cov_tilde_eval = {int(k): float(cov_tilde_sums[int(k)] / samp_denom) for k in rank_ks}
        setattr(model, 'last_label_mass_eval', last_label_mass_eval)
        setattr(model, 'last_cov_tilde_eval', last_cov_tilde_eval)
    except Exception:
        pass
    # 写入新增评估指标到模型属性（供日志/JSON）
    try:
        last_oracle_cov_eval = {int(k): float(oracle_cov_sums[int(k)] / samp_denom) for k in rank_ks}
        last_oracle_hit_eval = {int(k): float(oracle_hit_sums[int(k)] / samp_denom) for k in rank_ks}
        last_sensor_hit_eval = {int(k): float(sensor_hit_sums[int(k)] / samp_denom) for k in rank_ks}
        setattr(model, 'last_oracle_cov_eval', last_oracle_cov_eval)
        setattr(model, 'last_oracle_hit_eval', last_oracle_hit_eval)
        setattr(model, 'last_sensor_hit_eval', last_sensor_hit_eval)
    except Exception:
        pass
    try:
        setattr(model, 'last_cap_ratio_eval', float(cap_ratio_sum / samp_denom))
    except Exception:
        pass
    try:
        setattr(model, 'last_spearman_logits_eval', float(spearman_s_sum / samp_denom))
        setattr(model, 'last_spearman_probs_eval', float(spearman_p_sum / samp_denom))
    except Exception:
        pass
    try:
        setattr(model, 'last_conn_rate_eval', float(conn_rate_sum / max(1, int(getattr(model, 'last_eval_samp_sum', samp_sum)))))
        setattr(model, 'last_island_rate_eval', float(island_rate_sum / max(1, int(getattr(model, 'last_eval_samp_sum', samp_sum)))))
    except Exception:
        pass
    return base + hit_vals + tail + (bal_acc, macro_f1)
