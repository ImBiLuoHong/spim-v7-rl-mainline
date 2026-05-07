import os
import json
from datetime import datetime
from typing import Optional, Dict, List

from src.shared.logging.logging.common import _get, _to_serializable


def write_model_architecture_doc(
    logs_dir: str,
    cfg,
    model,
    fts,
    eidx,
    eattr,
    eidx_v=None,
    eattr_v=None,
    out: Optional[Dict] = None,
) -> Optional[str]:
    """Write a lightweight model architecture/documentation JSON to logs/model_architecture.json."""
    try:
        import torch
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'model_architecture.json')

        def _shape(x):
            try:
                if isinstance(x, torch.Tensor):
                    return list(x.shape)
                if isinstance(x, (list, tuple)):
                    return [len(x)]
                return None
            except Exception:
                return None

        def _get_any(_cfg, keys: List[str], default=None):
            for k in keys:
                try:
                    v = _get(_cfg, k, None)
                except Exception:
                    v = None
                if v is not None:
                    return v
            return default

        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'model_class': type(model).__name__,
            'inputs': {
                'fts': _shape(fts),
                'edge_index': _shape(eidx),
                'edge_attr': _shape(eattr),
                'edge_index_v': _shape(eidx_v),
                'edge_attr_v': _shape(eattr_v),
            },
            'cfg': {
                'selector_head_enabled': _get_any(cfg, [
                    'features.selector_head_enabled',
                    'features.enable_selector_head',
                    'model.selector_head_enabled'
                ], None),
                'selector_normalization': _get(cfg, 'features.selector_normalization', None),
                'selector_target_k': _get(cfg, 'features.selector_target_k', None),
                'softmax_temperature': _get_any(cfg, [
                    'features.softmax_temperature',
                    'training.softmax_temperature'
                ], None),
                'prune_upstream.enabled': _get(cfg, 'features.prune_upstream.enabled', None),
                'prune_upstream.window_s': _get(cfg, 'features.prune_upstream.window_s', None),
                'connected_eval_enabled': _get_any(cfg, [
                    'features.connected_eval.enabled',
                    'features.connected_eval_enabled'
                ], None),
                'loss.causal.enabled': _get(cfg, 'loss.causal.enabled', None),
                'loss.causal.weight': _get(cfg, 'loss.causal.weight', None),
                'loss.causal.tau': _get(cfg, 'loss.causal.tau', None),
                'loss.causal.mode': _get(cfg, 'loss.causal.mode', None),
            },
        }

        try:
            if isinstance(out, dict):
                keep_keys = ['selector_mask', 'logits', 'attn_weights', 'selector_topk_idx']
                payload['outputs'] = {k: _shape(out.get(k)) for k in keep_keys if k in out}
        except Exception:
            pass

        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_model_architecture_doc failed: {e}")
        except Exception:
            pass
        return None