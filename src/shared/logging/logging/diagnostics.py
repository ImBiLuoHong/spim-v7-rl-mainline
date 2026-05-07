import os
import json
from datetime import datetime
from typing import Optional, Dict, List

from src.shared.logging.logging.common import _to_serializable


def write_selector_mask_csv(logs_dir: str, mask_tensor, K: Optional[int] = None, norm_type: str = 'softmaxK', nodes: Optional[List[int]] = None) -> Optional[str]:
    try:
        import torch
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'selector_mask_eval.csv')
        if isinstance(mask_tensor, torch.Tensor):
            m = mask_tensor.detach().cpu()
            if m.dim() == 1:
                m = m.view(1, -1)
        else:
            try:
                import numpy as np
                m = torch.tensor(np.array(mask_tensor), dtype=torch.float32)
                if m.dim() == 1:
                    m = m.view(1, -1)
            except Exception:
                return None
        B = int(m.shape[0]); N = int(m.shape[1])
        sum_m_avg = float(m.sum(dim=1).mean().item())
        with open(path, 'w', encoding='utf-8') as f:
            K_val = (int(K) if isinstance(K, int) and K > 0 else '')
            header = f"norm_type={norm_type},K={K_val},B={B},N={N},sum_m_avg={sum_m_avg:.6f}"
            if isinstance(nodes, (list, tuple)) and len(nodes) > 0:
                try:
                    nodes_str = ','.join(str(int(x)) for x in nodes)
                except Exception:
                    nodes_str = ''
                header = header + f";nodes=[{nodes_str}]"
            f.write(header + '\n')
            for b in range(B):
                vals = ','.join(f"{float(x):.6f}" for x in m[b].view(-1).tolist())
                if b == 0:
                    f.write(f"norm_type={norm_type},b{b},{vals}\n")
                else:
                    f.write(f"b{b},{vals}\n")
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_selector_mask_csv failed: {e}")
        except Exception:
            pass
        return None


def log_tv_stats(logs_dir: str, edge_index_tv, w_tv) -> Optional[str]:
    try:
        import torch
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'tv_stats_eval.csv')
        E_total = 0
        w_min = 0.0; w_max = 0.0; w_mean = 0.0
        if isinstance(edge_index_tv, torch.Tensor) and isinstance(w_tv, torch.Tensor):
            if edge_index_tv.dim() == 3:
                B = int(edge_index_tv.shape[0])
                E = int(edge_index_tv.shape[1])
                E_total = B * E
                wt = w_tv.view(-1).to(torch.float32)
                w_min = float(wt.min().item())
                w_max = float(wt.max().item())
                w_mean = float(wt.mean().item())
            elif edge_index_tv.dim() == 2:
                E_total = int(edge_index_tv.shape[1])
                wt = w_tv.view(-1).to(torch.float32)
                w_min = float(wt.min().item())
                w_max = float(wt.max().item())
                w_mean = float(wt.mean().item())
        with open(path, 'w', encoding='utf-8') as f:
            f.write('edges,w_min,w_max,w_mean\n')
            f.write(f"{int(E_total)},{w_min:.6f},{w_max:.6f},{w_mean:.6f}\n")
        return path
    except Exception as e:
        try:
            print(f"[WARN] log_tv_stats failed: {e}")
        except Exception:
            pass
        return None


def write_coverage_curve_csv(logs_dir: str, label_mass: Dict[int, float]) -> Optional[str]:
    try:
        os.makedirs(logs_dir, exist_ok=True)
        diag_dir = os.path.join(logs_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, 'coverage_curve.csv')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('k,label_mass\n')
            for k in sorted(int(x) for x in (label_mass or {}).keys()):
                try:
                    v = float(label_mass.get(int(k), 0.0))
                except Exception:
                    v = 0.0
                f.write(f"{int(k)},{v:.6f}\n")
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_coverage_curve_csv failed: {e}")
        except Exception:
            pass
        return None


def write_krho_stats_csv(logs_dir: str, spearman_logits: float = None, spearman_probs: float = None) -> Optional[str]:
    try:
        os.makedirs(logs_dir, exist_ok=True)
        diag_dir = os.path.join(logs_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, 'krho_stats.csv')
        s_l = 0.0 if spearman_logits is None else float(spearman_logits)
        s_p = 0.0 if spearman_probs is None else float(spearman_probs)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('spearman_logits,spearman_probs\n')
            f.write(f"{s_l:.6f},{s_p:.6f}\n")
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_krho_stats_csv failed: {e}")
        except Exception:
            pass
        return None


def write_eval_caps_csv(logs_dir: str, cap_mask_ratio: float = None) -> Optional[str]:
    try:
        os.makedirs(logs_dir, exist_ok=True)
        diag_dir = os.path.join(logs_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, 'eval_caps.csv')
        v = 0.0 if cap_mask_ratio is None else float(cap_mask_ratio)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('cap_mask_ratio\n')
            f.write(f"{v:.6f}\n")
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_eval_caps_csv failed: {e}")
        except Exception:
            pass
        return None


def write_tau_hist_csv(logs_dir: str, tv_tau: float = None, conn_tau_time: float = None) -> Optional[str]:
    try:
        os.makedirs(logs_dir, exist_ok=True)
        diag_dir = os.path.join(logs_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, 'tau_hist.csv')
        t_tv = '' if tv_tau is None else float(tv_tau)
        t_conn = '' if conn_tau_time is None else float(conn_tau_time)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('tv_tau,conn_tau_time\n')
            f.write(f"{t_tv},{t_conn}\n")
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_tau_hist_csv failed: {e}")
        except Exception:
            pass
        return None


def write_diag_report_md(logs_dir: str, summary: Dict[str, object]) -> Optional[str]:
    try:
        os.makedirs(logs_dir, exist_ok=True)
        diag_dir = os.path.join(logs_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, 'report.md')
        lines = ["# Diagnostic Report", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}"]
        def _line(k, alias=None):
            v = summary.get(k)
            if v is None:
                return None
            name = alias or k
            try:
                vv = float(v)
                return f"- {name}: {vv:.6f}"
            except Exception:
                return f"- {name}: {v}"
        for k, alias in [
            ('loss', 'Val loss'), ('acc', 'Val acc'),
            ('Spearman_s_y', 'Spearman(logits, y)'), ('Spearman_p_y', 'Spearman(probs, y)'),
            ('cap_mask_ratio', 'Capability mask ratio'),
            ('connectivity_rate', 'Connectivity rate'), ('island_rate', 'Island rate'),
        ]:
            ln = _line(k, alias)
            if ln:
                lines.append(ln)
        ks = []
        for k in list(summary.keys()):
            if isinstance(k, str) and k.startswith('TopKLabelMass@'):
                try:
                    ks.append(int(k.split('@')[1]))
                except Exception:
                    pass
        ks = sorted(set(ks))
        if ks:
            lines.append('')
            lines.append('## TopK Label Mass')
            for k in ks:
                v = summary.get(f'TopKLabelMass@{k}', 0.0)
                lines.append(f"- @ {k}: {float(v):.6f}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_diag_report_md failed: {e}")
        except Exception:
            pass
        return None


def write_diag_gates_json(logs_dir: str, summary: Dict[str, object], rank_k: int = 30) -> Optional[str]:
    try:
        os.makedirs(logs_dir, exist_ok=True)
        diag_dir = os.path.join(logs_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, 'gates.json')

        def _find_k(sd: Dict[str, object], prefer_k: int) -> int:
            try:
                ks = []
                for k in sd.keys():
                    if isinstance(k, str) and '@' in k:
                        try:
                            ks.append(int(k.split('@')[-1]))
                        except Exception:
                            pass
                if prefer_k in ks:
                    return prefer_k
                if ks:
                    return max(ks)
                return prefer_k
            except Exception:
                return prefer_k

        k_eff = _find_k(summary or {}, int(rank_k))

        def _get_float(sd: Dict[str, object], key: str, default: float = None):
            try:
                v = sd.get(key, default)
                return None if v is None else float(v)
            except Exception:
                return default

        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'rank_k': int(k_eff),
            'metrics': {
                f'OracleCov@{k_eff}': _get_float(summary, f'OracleCov@{k_eff}', None),
                f'OracleHit@{k_eff}': _get_float(summary, f'OracleHit@{k_eff}', None),
                f'ModelHitSensor@{k_eff}': _get_float(summary, f'ModelHitSensor@{k_eff}', None),
                'connectivity_rate': _get_float(summary, 'connectivity_rate', None),
                'island_rate': _get_float(summary, 'island_rate', None),
                'spearman_logits': _get_float(summary, 'Spearman_s_y', None),
                'spearman_probs': _get_float(summary, 'Spearman_p_y', None),
                'cap_mask_ratio': _get_float(summary, 'cap_mask_ratio', None),
            },
        }

        gates = {
            'oracle': {
                'cov_measured': payload['metrics'][f'OracleCov@{k_eff}'] is not None,
                'hit_measured': payload['metrics'][f'OracleHit@{k_eff}'] is not None,
            },
            'hit': {
                'sensor_measured': payload['metrics'][f'ModelHitSensor@{k_eff}'] is not None,
            },
            'connectivity': {
                'rate_measured': payload['metrics']['connectivity_rate'] is not None,
                'island_measured': payload['metrics']['island_rate'] is not None,
            },
            'correlation': {
                'spearman_logits_measured': payload['metrics']['spearman_logits'] is not None,
                'spearman_probs_measured': payload['metrics']['spearman_probs'] is not None,
            },
            'stability': {
                'cap_mask_measured': payload['metrics']['cap_mask_ratio'] is not None,
            },
            'integrity': {
                'has_eval_metrics': isinstance(summary, dict) and len(summary) > 0,
            },
        }
        payload['gates'] = gates

        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_diag_gates_json failed: {e}")
        except Exception:
            pass
        return None


def write_grads_csv(logs_dir: str, model) -> Optional[str]:
    try:
        import math
        import torch
        os.makedirs(logs_dir, exist_ok=True)
        diag_dir = os.path.join(logs_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(diag_dir, 'grads.csv')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('name,grad_norm,grad_mean,grad_std,nonzero_ratio\n')
            try:
                for name, p in getattr(model, 'named_parameters', lambda: [])():
                    g = getattr(p, 'grad', None)
                    if g is None or g.data is None:
                        f.write(f"{name},0.0,0.0,0.0,0.0\n")
                        continue
                    t = g.detach().float().view(-1)
                    if t.numel() == 0:
                        f.write(f"{name},0.0,0.0,0.0,0.0\n")
                        continue
                    gn = float(torch.linalg.norm(t).item())
                    gm = float(t.mean().item())
                    gs = float(t.std(unbiased=False).item())
                    nnz = float((t != 0).sum().item())
                    nzr = 0.0
                    try:
                        nzr = float(nnz / float(t.numel())) if t.numel() > 0 else 0.0
                    except Exception:
                        nzr = 0.0
                    f.write(f"{name},{gn:.6f},{gm:.6f},{gs:.6f},{nzr:.6f}\n")
            except Exception:
                pass
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_grads_csv failed: {e}")
        except Exception:
            pass
        return None