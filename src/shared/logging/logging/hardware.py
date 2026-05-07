import os
import json
from datetime import datetime
from typing import Optional, Dict

from src.shared.logging.logging.common import _to_serializable


def write_hardware_stats(
    logs_dir: str,
    phase: str,
    model: Optional[object] = None,
    epoch_time_s: Optional[float] = None,
    grad_accum_steps: Optional[int] = None,
    extra: Optional[Dict[str, object]] = None,
) -> str:
    """写入/追加硬件统计到 {logs_dir}/hardware_stats.json。

    字段包含：
    - ts: 时间戳
    - phase: 'train' 或 'eval'
    - gpu_name, cuda_available, cuda_device_count
    - amp_enabled
    - batch_size_reported
    - grad_accum_steps
    - samp_sum
    - epoch_time_s, tps
    - gpu_peak_mem_mb, gpu_mem_alloc_mb, gpu_mem_reserved_mb
    - notes (extra)
    """
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass
    out_path = os.path.join(logs_dir, 'hardware_stats.json')

    record: Dict[str, object] = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'phase': str(phase or 'train'),
    }
    try:
        import torch as _torch
        cuda_avail = _torch.cuda.is_available()
        record['cuda_available'] = bool(cuda_avail)
        record['cuda_device_count'] = int(_torch.cuda.device_count()) if cuda_avail else 0
        if cuda_avail:
            try:
                record['gpu_name'] = _torch.cuda.get_device_name(0)
            except Exception:
                record['gpu_name'] = None
            try:
                mem_alloc = _torch.cuda.memory_allocated(0)
                mem_reserved = _torch.cuda.memory_reserved(0)
                record['gpu_mem_alloc_mb'] = float(mem_alloc / (1024 * 1024))
                record['gpu_mem_reserved_mb'] = float(mem_reserved / (1024 * 1024))
            except Exception:
                pass
            try:
                peak = getattr(model, 'gpu_peak_mem_mb', None)
                if peak is None:
                    peak_bytes = getattr(_torch.cuda, 'max_memory_allocated', lambda device=None: 0)(0)
                    if peak_bytes:
                        peak = float(peak_bytes / (1024 * 1024))
                record['gpu_peak_mem_mb'] = (float(peak) if peak is not None else None)
            except Exception:
                record['gpu_peak_mem_mb'] = None
    except Exception:
        record['cuda_available'] = False
        record['cuda_device_count'] = 0

    try:
        amp_enabled = bool(getattr(model, 'last_amp_enabled', False))
        record['amp_enabled'] = amp_enabled
    except Exception:
        record['amp_enabled'] = None

    bs_reported = None
    try:
        cfg = getattr(model, 'cfg', None)
        if cfg is not None and hasattr(cfg, 'training'):
            bs_reported = getattr(cfg.training, 'batch_size', None)
    except Exception:
        bs_reported = None
    record['batch_size_reported'] = (int(bs_reported) if isinstance(bs_reported, (int, float)) else None)

    try:
        if grad_accum_steps is None:
            grad_accum_steps = getattr(getattr(model, 'cfg', object()), 'training', object()).__dict__.get('grad_accum_steps', None)
    except Exception:
        pass
    record['grad_accum_steps'] = (int(grad_accum_steps) if isinstance(grad_accum_steps, (int, float)) else None)

    samp_sum = None
    try:
        if str(phase) == 'train':
            samp_sum = getattr(model, 'last_train_samp_sum', None)
        else:
            samp_sum = getattr(model, 'last_eval_samp_sum', None)
    except Exception:
        samp_sum = None
    record['samp_sum'] = (int(samp_sum) if isinstance(samp_sum, (int, float)) else None)

    try:
        if (record['samp_sum'] is not None) and (epoch_time_s is not None) and (float(epoch_time_s) > 0):
            record['epoch_time_s'] = float(epoch_time_s)
            record['tps'] = float(record['samp_sum']) / float(epoch_time_s)
        else:
            if epoch_time_s is not None:
                record['epoch_time_s'] = float(epoch_time_s)
            record['tps'] = None
    except Exception:
        record['tps'] = None

    if isinstance(extra, dict):
        try:
            record.update({str(k): _to_serializable(v) for k, v in extra.items()})
        except Exception:
            pass

    try:
        prev = []
        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
                if not isinstance(prev, list):
                    prev = []
        prev.append(record)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(prev, f, ensure_ascii=False, indent=2)
    except Exception:
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump([record], f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return out_path