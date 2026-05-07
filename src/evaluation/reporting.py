import os
import json
from typing import Dict, Optional, Tuple
from datetime import datetime


def ensure_output_dirs(run_dir: str) -> Dict[str, str]:
    """确保评估报告的目录结构存在，返回各 split 的目录路径。

    结构：
      RUN_DIR/
        outputs/
          val/
          test/
    """
    outputs_dir = os.path.join(run_dir, 'outputs')
    val_dir = os.path.join(outputs_dir, 'val')
    test_dir = os.path.join(outputs_dir, 'test')
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return {'outputs': outputs_dir, 'val': val_dir, 'test': test_dir}


def metrics_tuple_to_dict(metrics: Tuple[float, ...], ks: Tuple[int, ...] = (1, 3, 5, 10), ndcg_k: int = 5) -> Dict[str, float]:
    """将评估/训练循环返回的指标转换为结构化字典。
    格式：基础8项 + 命中率@k(按ks顺序) + MRR + NDCG@k [+ 可选 balanced_acc + macro_f1]
    """
    base_count = 8
    dyn_count = len(ks) + 2  # hits@k + mrr + ndcg
    if len(metrics) < base_count + dyn_count:
        raise ValueError("metrics 元组长度不足以解析动态 ks 指标")
    (
        loss,
        acc,
        ce,
        kl,
        class_mix,
        total,
        alpha,
        T,
    ) = metrics[:base_count]
    hits_vec = metrics[base_count:base_count + len(ks)]
    mrr = metrics[base_count + len(ks)]
    ndcg = metrics[base_count + len(ks) + 1]
    out = {
        'loss': float(loss),
        'acc': float(acc),
        'ce': float(ce),
        'kl': float(kl),
        'class_mix': float(class_mix),
        'total': float(total),
        'alpha': float(alpha),
        'T': float(T),
        'mrr': float(mrr),
        f'ndcg@{int(ndcg_k)}': float(ndcg),
    }
    for i, k in enumerate(ks):
        val = float(hits_vec[i])
        out[f'hit@{int(k)}'] = val
        # 对齐 recall@k（此任务中与命中率一致）
        out[f'recall@{int(k)}'] = val
    # 可选：解析 balanced_acc 与 macro_f1（若存在）
    try:
        if len(metrics) >= base_count + dyn_count + 2:
            balanced_acc = float(metrics[base_count + dyn_count])
            macro_f1 = float(metrics[base_count + dyn_count + 1])
            out['balanced_acc'] = balanced_acc
            out['macro_f1'] = macro_f1
    except Exception:
        pass
    return out


def write_eval_json(run_dir: str, split: str, metrics: Dict[str, float], epoch: Optional[int] = None) -> str:
    """写入评估报告 JSON 文件，返回写入路径。
    
    统一采用扁平命名：
    - val_epoch_X.json
    - train_epoch_X.json
    - test_final.json
    """
    dirs = ensure_output_dirs(run_dir)
    out_dir = dirs.get(split, dirs['outputs'])  # 使用各 split 子目录，回退到 outputs
    
    if split == 'val':
        fname = f"val_epoch_{int(epoch)}.json" if epoch is not None else "val_epoch_last.json"
    elif split == 'train':
        fname = f"train_epoch_{int(epoch)}.json" if epoch is not None else "train_epoch_last.json"
    elif split == 'test':
        fname = "test_final.json"
    else:
        fname = f"{split}_summary.json"

    out_path = os.path.join(out_dir, fname)

    # 写入
    payload = {
        'split': split,
        'epoch': int(epoch) if epoch is not None else None,
        'metrics': metrics,
        'ts': datetime.now().isoformat(timespec='seconds')
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def write_metrics_scope_json(run_dir: str, scope: Dict[str, object]) -> str:
    """写入 outputs/metrics_scope.json，记录全图评估审计字段并计算合规标志。
    """
    try:
        dirs = ensure_output_dirs(run_dir)
        out_dir = dirs['outputs']
        path = os.path.join(out_dir, 'metrics_scope.json')
        payload = dict(scope or {})
        # 默认值与纠偏
        N_total = int(payload.get('N_total') or 0)
        N_eval_used = int(payload.get('N_eval_used') or 0)
        candidate_pool_size = payload.get('candidate_pool_size')
        try:
            candidate_pool_size = int(candidate_pool_size) if candidate_pool_size is not None else None
        except Exception:
            candidate_pool_size = None
        metrics_full_graph = bool(payload.get('metrics_full_graph', True))
        prune_applied = bool(payload.get('prune_applied', False))
        mask_source = str(payload.get('mask_source', 'none') or 'none')
        # 若候选池大小缺失，则回填为 N_total
        if candidate_pool_size in (None, 0) and N_total > 0:
            candidate_pool_size = N_total
        payload['N_total'] = N_total
        payload['N_eval_used'] = N_eval_used
        payload['candidate_pool_size'] = int(candidate_pool_size or 0)
        payload['metrics_full_graph'] = bool(metrics_full_graph)
        payload['prune_applied'] = bool(prune_applied)
        payload['mask_source'] = mask_source
        # 合规判定
        scope_ok = (
            bool(metrics_full_graph) is True and
            bool(prune_applied) is False and
            int(N_total) > 0 and
            int(N_eval_used) == int(N_total) and
            int(candidate_pool_size or 0) == int(N_total) and
            mask_source in ("none", "eval_full_graph")
        )
        payload['scope_compliant'] = bool(scope_ok)
        payload['ts'] = datetime.now().isoformat(timespec='seconds')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return ''
