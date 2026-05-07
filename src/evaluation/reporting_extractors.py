#!/usr/bin/env python3
"""
报告抽取与回填工具（纯函数库，SSOT/DI）

职责：
- 仅提供读取/推断/格式化的纯函数，不持有全局状态；
- 统一健壮读取与字段提取，避免各脚本重复实现（DRY）。

约束：
- 不读取环境变量；所有路径由调用方传入（DI）。
- 不写入文件（除非明确由调用方要求）；本模块仅抽取。
"""
import os
import json
from statistics import mean
from typing import List, Tuple, Any, Optional


def read_json(path: str) -> Tuple[dict, Optional[str]]:
    """健壮读取 JSON（支持 UTF-8 与 UTF-8-SIG）。
    返回 (dict, err)，当读取失败时返回 ({}, "err:...")。
    """
    if not os.path.exists(path):
        return {}, f"err:missing:{path}"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                return json.load(f), None
        except Exception as e:
            return {}, f"err:decode:{e}"
    except Exception as e:
        return {}, f"err:read:{e}"


def read_text(path: str) -> str:
    """读取文本，strip 行尾；不存在返回空串。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ""


def find_best_epoch_json(run_dir: str) -> Tuple[str, List[str]]:
    """定位最佳 epoch 的验证 JSON 文件。
    - 优先使用 logs/best_epoch.txt 指示的 outputs/val/epoch_{best}.json
    - 其次回退到 outputs/val/epoch_best.json
    - 若二者都不存在，则在 outputs/val 目录中按时间/编号降序选择首个 epoch_*.json
    返回 (path, notes)。
    """
    notes: List[str] = []
    val_dir = os.path.join(run_dir, 'outputs', 'val')
    be_txt = os.path.join(run_dir, 'logs', 'best_epoch.txt')
    best = read_text(be_txt)
    if best.isdigit():
        cand = os.path.join(val_dir, f'epoch_{best}.json')
        if os.path.exists(cand):
            return cand, notes
        else:
            notes.append('best_epoch_json_missing')
    # 回退 epoch_best.json
    fallback = os.path.join(val_dir, 'epoch_best.json')
    if os.path.exists(fallback):
        return fallback, notes
    # 在 val_dir 中查找 epoch_*.json
    if os.path.isdir(val_dir):
        files = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.startswith('epoch_') and x.endswith('.json')]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if files:
            notes.append('metrics_fallback:any_epoch')
            return files[0], notes
    notes.append('metrics_missing')
    return '', notes


def extract_from_cfg(cfg_dict: dict, candidates: List[str], default: Any = "", notes: List[str] = None):
    """按候选键路径（点分割）逐一尝试；命中即返回值；全部失败记录 notes。"""
    if notes is None:
        notes = []
    for key in candidates:
        cur = cfg_dict
        ok = True
        for k in key.split('.'):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    notes.append('missing_cfg:' + '|'.join(candidates))
    return default


def extract_numeric(value: Any, fmt: Optional[str] = None, default: str = "") -> str:
    """容错数值格式化。"""
    try:
        x = float(value)
        if fmt:
            return (fmt % x) if '%' in fmt else f"{x:{fmt}}"
        return str(x)
    except Exception:
        return default


def _load_effective_cfg(run_dir: str) -> dict:
    eff_path = os.path.join(run_dir, 'logs', 'effective_config.json')
    j, _err = read_json(eff_path)
    root = j.get('effective') if isinstance(j.get('effective'), dict) else j
    return root or {}


def _parse_config_summary(run_dir: str) -> dict:
    """粗粒度解析 logs/config_summary.txt，用于兜底提取少量键值。"""
    fp = os.path.join(run_dir, 'logs', 'config_summary.txt')
    if not os.path.exists(fp):
        return {}
    text = read_text(fp)
    info = {}
    import re
    # 例：rank_ks=[1,5,30]; select_best=recall@30; model=APPNP
    m = re.search(r"rank_ks\s*=\s*\[([^\]]+)\]", text)
    if m:
        info['training.rank_ks'] = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
    m = re.search(r"select_best\s*=\s*([\w@]+)", text)
    if m:
        info['training.select_best'] = m.group(1)
    m = re.search(r"model\s*=\s*([A-Za-z0-9_\-]+)", text)
    if m:
        info['model.name'] = m.group(1)
    # K/alpha/temp 简易键值（若存在）
    m = re.search(r"selector_target_k\s*=\s*(\d+)", text)
    if m:
        info['features.selector_target_k'] = int(m.group(1))
    m = re.search(r"coverage_alpha\s*=\s*([0-9.]+)", text)
    if m:
        info['features.coverage_alpha'] = float(m.group(1))
    m = re.search(r"softmax_temperature\s*=\s*([0-9.]+)", text)
    if m:
        info['features.softmax_temperature'] = float(m.group(1))
    return info


def derive_backbone(cfg: dict, run_dir: str, notes: List[str]) -> str:
    val = cfg.get('model', {}).get('spatial_backbone') or cfg.get('model', {}).get('name') or cfg.get('model', {}).get('backbone')
    if not val:
        # 尝试从 config_summary 兜底
        info = _parse_config_summary(run_dir)
        val = info.get('model.name', '')
    if not val:
        # 进一步兜底：从 RUN 目录名解析（如 TOP5_gatv2_disabled_K30_A0.3_T0.9_*）
        name = os.path.basename(run_dir).lower()
        known = ['appnp', 'gatv2', 'gcn', 'sage', 'graphsage', 'gin', 'mlp']
        for k in known:
            if k in name:
                val = k.upper() if k != 'mlp' else 'MLP'
                break
    if not val:
        notes.append('unknown_backbone')
    return str(val or '')


def derive_tv_fields(cfg: dict, notes: List[str]) -> Tuple[str, str, str]:
    tv_enabled = bool(cfg.get('loss', {}).get('tv', {}).get('enabled', False))
    mode = cfg.get('loss', {}).get('tv', {}).get('mode')
    weight = cfg.get('loss', {}).get('tv', {}).get('weight')
    tau = cfg.get('loss', {}).get('tv', {}).get('tau')
    if not tv_enabled:
        return 'disabled', '0', '0'
    # 若启用但缺字段，记 notes 并回空串
    if mode is None:
        notes.append('missing_cfg:loss.tv.mode')
    if weight is None:
        notes.append('missing_cfg:loss.tv.weight')
    if tau is None:
        notes.append('missing_cfg:loss.tv.tau')
    return str(mode or ''), str(weight or ''), str(tau or '')


def _read_latest_selector_row(run_dir: str) -> str:
    """读取 selector_mask.csv 的最后一行；若不存在则回退读取 selector_mask_eval.csv。"""
    fp = os.path.join(run_dir, 'logs', 'selector_mask.csv')
    try:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                lines = f.read().strip().splitlines()
            if lines:
                return lines[-1]
    except Exception:
        pass
    # 回退到评估期快照
    fp2 = os.path.join(run_dir, 'logs', 'selector_mask_eval.csv')
    try:
        if os.path.exists(fp2):
            with open(fp2, 'r', encoding='utf-8') as f:
                lines = f.read().strip().splitlines()
            if lines:
                return lines[-1]
    except Exception:
        pass
    return ''


def derive_selector_fields(cfg: dict, run_dir: str, notes: List[str]) -> Tuple[str, str, str]:
    # 直接从 cfg 读取
    K = cfg.get('features', {}).get('selector_target_k')
    alpha = cfg.get('features', {}).get('coverage_alpha')
    temp = cfg.get('features', {}).get('softmax_temperature')
    # 兜底使用 config_summary
    info = _parse_config_summary(run_dir)
    K = K if K is not None else info.get('features.selector_target_k')
    alpha = alpha if alpha is not None else info.get('features.coverage_alpha')
    temp = temp if temp is not None else info.get('features.softmax_temperature')
    # 若 K 仍为空，尝试从 selector_mask.csv 最新一行推断
    if K is None:
        last = _read_latest_selector_row(run_dir)
        import re
        m = re.search(r"K\s*=\s*(\d+)", last)
        if m:
            K = int(m.group(1))
        else:
            # 回退：统计 node_ids 列近似长度
            m2 = re.search(r"\[([^\]]+)\]", last)
            if m2:
                arr = [t.strip() for t in m2.group(1).split(',') if t.strip()]
                if arr:
                    K = round(len(arr))
    if K is None:
        # 继续兜底：从 RUN 目录名解析 KXX
        base = os.path.basename(run_dir)
        import re
        mK = re.search(r"K(\d+)", base)
        if mK:
            try:
                K = int(mK.group(1))
            except Exception:
                pass
    if K is None:
        notes.append('missing_cfg:features.selector_target_k')
    if alpha is None:
        # 进一步兜底：loss.coverage.alpha 或 model.coverage_alpha
        alpha = cfg.get('loss', {}).get('coverage', {}).get('alpha') or cfg.get('model', {}).get('coverage_alpha')
    if alpha is None:
        # 从目录名解析 A<float>
        base = os.path.basename(run_dir)
        import re
        mA = re.search(r"A([0-9]+(?:\.[0-9]+)?)", base)
        if mA:
            try:
                alpha = float(mA.group(1))
            except Exception:
                pass
    if alpha is None:
        notes.append('missing_cfg:features.coverage_alpha')
    if temp is None:
        temp = cfg.get('model', {}).get('softmax_temperature')
    if temp is None:
        # 从目录名解析 T<float>
        base = os.path.basename(run_dir)
        import re
        mT = re.search(r"T([0-9]+(?:\.[0-9]+)?)", base)
        if mT:
            try:
                temp = float(mT.group(1))
            except Exception:
                pass
    if temp is None:
        notes.append('missing_cfg:features.softmax_temperature')
    return str(K or ''), str(alpha or ''), str(temp or '')


def extract_metrics(run_dir: str, notes: List[str]) -> Tuple[str, str, str, str]:
    fp, n2 = find_best_epoch_json(run_dir)
    notes.extend(n2)
    r30 = h30 = n30 = ''
    best_epoch = ''
    if fp:
        j, err = read_json(fp)
        if err:
            notes.append(err)
        if isinstance(j, dict):
            def fmt4(x):
                try:
                    return f"{float(x):.4f}"
                except Exception:
                    return ''
            has_r = 'recall@30' in j
            has_h = 'hit@30' in j
            has_n = 'ndcg@30' in j
            r30 = fmt4(j.get('recall@30'))
            h30 = fmt4(j.get('hit@30'))
            n30 = fmt4(j.get('ndcg@30'))
            if not (has_r and has_h and has_n):
                notes.append('metrics_fallback:missing_keys')
            # 从文件名中提取 best_epoch
            import re
            m = re.search(r"epoch_(\d+)\.json$", fp)
            if m:
                best_epoch = m.group(1)
            else:
                # 兼容 epoch_best.json，额外读取 logs/best_epoch.txt
                be = read_text(os.path.join(run_dir, 'logs', 'best_epoch.txt'))
                if be.isdigit():
                    best_epoch = be
    else:
        notes.append('metrics_missing')
    return r30, h30, n30, best_epoch


def extract_epoch_time(run_dir: str, notes: List[str]) -> str:
    fp = os.path.join(run_dir, 'logs', 'epoch_times.txt')
    if not os.path.exists(fp):
        notes.append('no_epoch_times')
        return ''
    try:
        vals = []
        with open(fp, 'r', encoding='utf-8') as f:
            for ln in f:
                try:
                    vals.append(float(ln.strip()))
                except Exception:
                    pass
        if not vals:
            notes.append('no_epoch_times')
            return ''
        return f"{mean(vals):.4f}"
    except Exception:
        notes.append('no_epoch_times')
        return ''


def extract_params_M(run_dir: str, notes: List[str]) -> str:
    fp = os.path.join(run_dir, 'logs', 'params.txt')
    if not os.path.exists(fp):
        notes.append('no_params')
        return ''
    try:
        line = read_text(fp)
        val = float(line)
        return f"{val:.1f}"
    except Exception:
        notes.append('no_params')
        return ''


def extract_components(run_dir: str, notes: List[str]) -> Tuple[str, bool]:
    fp = os.path.join(run_dir, 'logs', 'components_K_cc.txt')
    if os.path.exists(fp):
        val = read_text(fp)
        fb = os.path.join(run_dir, 'logs', 'cc_fallback.txt')
        try:
            float(val)  # 验证是数值
            return val, False
        except Exception:
            # 非数值，如 'NA'，若存在 fallback 文件，追加备注
            if os.path.exists(fb):
                notes.append('components_K_cc_fallback: see cc_fallback.txt')
            if val:
                return val, False
            return '', False
    fb = os.path.join(run_dir, 'logs', 'cc_fallback.txt')
    if os.path.exists(fb):
        notes.append('components_K_cc_fallback: see cc_fallback.txt')
        return '', False
    # 不存在，提示需要计算
    notes.append('components_K_cc_missing')
    return '', True


def load_cfg_for_run(run_dir: str) -> dict:
    """加载有效配置，按顺序：effective_config.json → config_snapshot.json → config_summary.txt。"""
    eff = _load_effective_cfg(run_dir)
    if eff:
        return eff
    snap_path = os.path.join(run_dir, 'logs', 'config_snapshot.json')
    j, _err = read_json(snap_path)
    if j:
        return j
    # 解析 summary
    return _parse_config_summary(run_dir)