"""
AuditingProbe: 统计分布报告生成器（纯加法模块）
================================================

用途：
- 针对 ComplaintAlertStrategy 的行为进行在线审计，生成一份结构化且人类可读的“统计分布报告”。
- 报告包含三部分：
  1) 投诉延迟分布（Complaint Latency Distribution）
  2) 候选集大小分布（Candidate Set Size Distribution）
  3) 源-诉跳数分布（Source-Complaint Hop Distance Distribution）

输入：
- 由外部采集的诊断事件（log_event 调用），每条事件至少包含：
  details = {
    'sample_id': int,
    't_ground_truth_sec': int,      # 真实污染注入时间（秒）
    't_complain_sec': int,          # 模型生成的投诉发生时间（秒），不含采样延迟
    'candidate_set_size': int,      # 在 t_complain 时刻，浓度 > 阈值 的候选节点数量
    'chosen_complaint_node': Any,   # 最终被选为投诉点的节点ID（字符串或整数）
    'ground_truth_source': Any      # 真实源点的节点ID（通常为整数）
  }

产出：
- {run_dir}/logs/audit_probe_report.json      结构化 JSON 报告（满足验收标准）
- {run_dir}/logs/audit_probe_report.md        人类可读的概览（可选）

依赖：
- utils.connectivity.load_adj 用于构建无向邻接字典，计算源-诉跳距的最短路径（BFS）。
- data/node_mapping.json（若存在）用于在“原始节点ID”和“索引”之间进行映射，从而适配不同的边表示。

设计原则：
- 纯加法：不修改既有逻辑；仅新增模块与产物文件。
- SSOT/DI：只从 cfg 读取需要的参数；所有路径来源于 cfg.paths 与 run_dir。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import math
import statistics

from src.evaluation.metrics.connectivity import load_adj


class AuditingProbe:
    """在线审计探针：收集诊断事件并生成分布报告。"""

    def __init__(self, cfg: Any, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.log_buffer: List[Dict[str, Any]] = []
        # 尝试构建邻接以供 hop 计算
        try:
            self._adj = load_adj(cfg, run_dir)
        except Exception:
            self._adj = {}
        # 读取节点映射（若存在）
        self._node_map: Dict[str, Dict[str, int]] = {}
        try:
            root = getattr(getattr(cfg, 'paths', object()), 'root_dir', os.getcwd()) or os.getcwd()
            p = os.path.join(str(root), 'data', 'node_mapping.json')
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                self._node_map = j.get('nodes', {}) or {}
        except Exception:
            self._node_map = {}

    # ---------------------- 对外接口 ----------------------

    def log_event(self, details: Dict[str, Any]) -> None:
        """收集一条诊断事件。

        期望字段：见模块顶部说明。缺失字段将用 None 填充并在报告中跳过相关计算。
        """
        # 最小字段校正
        rec = {
            'sample_id': details.get('sample_id'),
            't_ground_truth_sec': details.get('t_ground_truth_sec'),
            't_complain_sec': details.get('t_complain_sec'),
            'candidate_set_size': details.get('candidate_set_size'),
            'chosen_complaint_node': details.get('chosen_complaint_node'),
            'ground_truth_source': details.get('ground_truth_source'),
        }
        self.log_buffer.append(rec)

    def dump_report(self, output_dir: Optional[str] = None, write_markdown: bool = True) -> str:
        """根据 log_buffer 生成统计分布报告文件。

        返回：生成的 JSON 报告路径。
        """
        out_dir = output_dir or os.path.join(self.run_dir, 'logs')
        os.makedirs(out_dir, exist_ok=True)

        step_seconds = int(self._get_any(self.cfg, 'data.time.step_seconds', 900))
        delay_cfg = self._get_any(self.cfg, 'data.alert_manager.strategies.complaint_alert.delay_hours', {'min': 1, 'max': 3})
        delay_min_h = float(self._get_any(delay_cfg, 'min', 1.0))
        delay_max_h = float(self._get_any(delay_cfg, 'max', 3.0))

        # Part 1: 延迟分布
        delay_steps_list: List[int] = []
        delay_hours_list: List[float] = []

        for rec in self.log_buffer:
            t_gt = rec.get('t_ground_truth_sec')
            t_cmp = rec.get('t_complain_sec')
            if (t_gt is None) or (t_cmp is None):
                continue
            try:
                dsec = int(t_cmp) - int(t_gt)
            except Exception:
                continue
            if dsec < 0:
                # 保护性：负延迟无意义，跳过
                continue
            dsteps = int(round(dsec / max(1, step_seconds)))
            delay_steps_list.append(dsteps)
            delay_hours_list.append(dsec / 3600.0)

        # Part 2: 候选集大小分布
        cand_sizes: List[int] = []
        for rec in self.log_buffer:
            s = rec.get('candidate_set_size')
            if s is None:
                continue
            try:
                cand_sizes.append(int(s))
            except Exception:
                continue

        # Part 3: 源-诉跳距分布
        hop_list: List[int] = []
        hop_missing: int = 0
        for rec in self.log_buffer:
            src_raw = rec.get('ground_truth_source')
            dst_raw = rec.get('chosen_complaint_node')
            if (src_raw is None) or (dst_raw is None):
                hop_missing += 1
                continue
            src_id = self._normalize_node_id(src_raw)
            dst_id = self._normalize_node_id(dst_raw)
            if (src_id is None) or (dst_id is None):
                hop_missing += 1
                continue
            h = self._bfs_shortest_path(self._adj, src_id, dst_id)
            if h is None:
                hop_missing += 1
            else:
                hop_list.append(int(h))

        # 汇总统计
        delay_summary = self._summary(delay_hours_list)
        cand_summary = self._summary(cand_sizes)
        hop_summary = self._summary(hop_list)

        # 直方图
        delay_hist_steps = self._histogram_full_range(delay_steps_list)
        cand_hist = self._histogram(cand_sizes)
        hop_hist = self._histogram(hop_list)

        # 验收标准检查
        latency_range_ok = None
        if delay_summary['count'] > 0:
            # 注意步长量化误差，允许 ±1 个时间步的容差
            eps_h = (step_seconds / 3600.0)
            min_ok = (delay_summary['min'] is not None) and ((delay_summary['min'] + eps_h) >= delay_min_h)
            max_ok = (delay_summary['max'] is not None) and ((delay_summary['max'] - eps_h) <= delay_max_h)
            latency_range_ok = bool(min_ok and max_ok)

        small_cand_count = sum(1 for x in cand_sizes if x < 5)
        small_cand_frac = (small_cand_count / len(cand_sizes)) if cand_sizes else 0.0
        candidate_small_risk = (cand_summary['min'] == 1) or (small_cand_frac > 0.5)

        hop_near_count = sum(1 for x in hop_list if x <= 1)
        hop_near_frac = (hop_near_count / len(hop_list)) if hop_list else 0.0
        hop_risk = (hop_summary['min'] == 0) or (hop_near_frac > 0.2)

        report = {
            'time': {
                'step_seconds': step_seconds,
                'config_delay_hours_range': {'min': delay_min_h, 'max': delay_max_h}
            },
            'counts': {
                'total_events': len(self.log_buffer),
                'latency_samples': delay_summary['count'],
                'candidate_size_samples': cand_summary['count'],
                'hop_samples': hop_summary['count'],
                'hop_missing': hop_missing
            },
            'complaint_latency_distribution': {
                'summary_hours': { 'min': delay_summary['min'], 'max': delay_summary['max'], 'mean': delay_summary['mean'], 'median': delay_summary['median'] },
                'histogram_steps': delay_hist_steps,
                'acceptance': {
                    'range_ok': latency_range_ok
                }
            },
            'candidate_set_size_distribution': {
                'summary': { 'min': cand_summary['min'], 'max': cand_summary['max'], 'mean': cand_summary['mean'], 'median': cand_summary['median'] },
                'histogram': cand_hist,
                'acceptance': {
                    'min_size_is_one': cand_summary['min'] == 1,
                    'small_sizes_fraction_gt_0_5': small_cand_frac > 0.5,
                    'fraction_lt_5': small_cand_frac
                }
            },
            'source_complaint_hop_distribution': {
                'summary': { 'min': hop_summary['min'], 'max': hop_summary['max'], 'mean': hop_summary['mean'], 'median': hop_summary['median'] },
                'histogram': hop_hist,
                'acceptance': {
                    'min_hop_is_zero': hop_summary['min'] == 0,
                    'fraction_hop_le_1_gt_0_2': hop_near_frac > 0.2,
                    'fraction_hop_le_1': hop_near_frac
                }
            }
        }

        out_json = os.path.join(out_dir, 'audit_probe_report.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        if write_markdown:
            out_md = os.path.join(out_dir, 'audit_probe_report.md')
            self._write_md(out_md, report)

        return out_json

    # ---------------------- 内部工具 ----------------------

    @staticmethod
    def _get_any(obj: Any, dotted: str, default=None):
        try:
            parts = dotted.split('.')
            cur = obj
            for p in parts:
                if cur is None:
                    return default
                if isinstance(cur, dict):
                    cur = cur.get(p, None)
                else:
                    cur = getattr(cur, p, None)
            return cur if (cur is not None) else default
        except Exception:
            return default

    @staticmethod
    def _summary(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {'min': None, 'max': None, 'mean': None, 'median': None, 'count': 0}
        try:
            return {
                'min': float(min(values)),
                'max': float(max(values)),
                'mean': float(statistics.mean(values)),
                'median': float(statistics.median(values)),
                'count': len(values)
            }
        except Exception:
            return {'min': None, 'max': None, 'mean': None, 'median': None, 'count': len(values)}

    @staticmethod
    def _histogram(values: List[int]) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for x in values:
            try:
                k = str(int(x))
                hist[k] = hist.get(k, 0) + 1
            except Exception:
                continue
        return hist

    @staticmethod
    def _histogram_full_range(values: List[int]) -> Dict[str, int]:
        if not values:
            return {}
        try:
            mn = int(min(values)); mx = int(max(values))
        except Exception:
            return {}
        hist: Dict[str, int] = {}
        for k in range(mn, mx + 1):
            hist[str(k)] = 0
        for x in values:
            k = str(int(x))
            hist[k] = hist.get(k, 0) + 1
        return hist

    def _normalize_node_id(self, node: Any) -> Optional[int]:
        """尝试将不同形式的节点ID转换为邻接字典中的键（int）。

        规则：
        - 若 node 为 int 且存在于 adj 键集合，直接返回。
        - 若 node 为字符串，尝试提取末尾数字作为候选；若该数字在 adj 键集合或能映射到 node_index 且存在于 adj，则返回对应 int。
        - 若存在 data/node_mapping.json，则优先使用原始ID→node_index 的映射以适配使用索引的邻接。
        """
        try:
            adj_keys = set(int(k) for k in self._adj.keys())
        except Exception:
            adj_keys = set()

        # 1) 直接整数
        try:
            if isinstance(node, int):
                return int(node) if (int(node) in adj_keys or not adj_keys) else int(node)
        except Exception:
            pass

        # 2) 字符串形式，提取数字后缀
        s = str(node)
        # 提取末尾连续数字
        digits = ''
        for ch in reversed(s):
            if ch.isdigit():
                digits = ch + digits
            else:
                if digits:
                    break
        # 候选：原始ID数字
        cand_raw: Optional[int] = None
        if digits:
            try:
                cand_raw = int(digits)
            except Exception:
                cand_raw = None
        # 3) 映射到 node_index（若存在）
        cand_idx: Optional[int] = None
        if cand_raw is not None:
            nd = self._node_map.get(str(cand_raw))
            if isinstance(nd, dict) and ('node_index' in nd):
                try:
                    cand_idx = int(nd['node_index'])
                except Exception:
                    cand_idx = None
        # 4) 选择与邻接匹配的键
        if cand_raw is not None and (cand_raw in adj_keys):
            return cand_raw
        if cand_idx is not None and (cand_idx in adj_keys):
            return cand_idx
        # Fallback：若邻接为空（未能加载），直接返回原始数值或 None
        if cand_raw is not None:
            return cand_raw
        return None

    @staticmethod
    def _bfs_shortest_path(adj: Dict[int, set], src: int, dst: int) -> Optional[int]:
        try:
            if src == dst:
                return 0
            if (src not in adj) or (dst not in adj):
                return None
            from collections import deque
            q = deque([src])
            dist = {src: 0}
            while q:
                u = q.popleft()
                du = dist[u]
                for v in adj.get(u, set()):
                    if v not in dist:
                        dist[v] = du + 1
                        if v == dst:
                            return dist[v]
                        q.append(v)
            return None
        except Exception:
            return None

    @staticmethod
    def _write_md(path: str, report: Dict[str, Any]) -> None:
        """生成简要 Markdown 概览。"""
        try:
            lines: List[str] = []
            lines.append('# AuditingProbe 统计分布报告')
            lines.append('')
            lines.append(f"- step_seconds: {report.get('time', {}).get('step_seconds')}  配置延迟范围(小时): {report.get('time', {}).get('config_delay_hours_range')}")
            lines.append(f"- 事件计数: total={report.get('counts', {}).get('total_events')} latency={report.get('counts', {}).get('latency_samples')} candidates={report.get('counts', {}).get('candidate_size_samples')} hops={report.get('counts', {}).get('hop_samples')} missing_hops={report.get('counts', {}).get('hop_missing')}")
            lines.append('')
            # 延迟摘要
            lat = report.get('complaint_latency_distribution', {})
            lines.append('## Part 1: 投诉延迟分布')
            lines.append(f"summary_hours: {lat.get('summary_hours')}")
            lines.append(f"range_ok: {lat.get('acceptance', {}).get('range_ok')}")
            lines.append('')
            # 候选集摘要
            cs = report.get('candidate_set_size_distribution', {})
            lines.append('## Part 2: 候选集大小分布')
            lines.append(f"summary: {cs.get('summary')}")
            lines.append(f"risk_flags: min_size_is_one={cs.get('acceptance', {}).get('min_size_is_one')} fraction_lt_5={cs.get('acceptance', {}).get('fraction_lt_5')} frac>0.5?={cs.get('acceptance', {}).get('small_sizes_fraction_gt_0_5')}")
            lines.append('')
            # 跳距摘要
            hp = report.get('source_complaint_hop_distribution', {})
            lines.append('## Part 3: 源-诉跳数分布')
            lines.append(f"summary: {hp.get('summary')}")
            lines.append(f"risk_flags: min_hop_is_zero={hp.get('acceptance', {}).get('min_hop_is_zero')} fraction_hop_le_1={hp.get('acceptance', {}).get('fraction_hop_le_1')} frac>0.2?={hp.get('acceptance', {}).get('fraction_hop_le_1_gt_0_2')}")
            lines.append('')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        except Exception:
            # 忽略 MD 生成错误
            pass