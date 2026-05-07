import os
import json
from datetime import datetime
from typing import Optional, Dict, Any


class Probe:
    """轻量级诊断探针：在不影响训练/评估逻辑的前提下，记录事件到 logs 目录。

    - 默认写入 JSONL（每行一个事件对象），文件名由 cfg.diagnostics.events_filename 控制
    - 可选同时写入 CSV（基础列：ts,event；其余键尽量展开为扁平字段）
    - 任何写入失败都不会抛出异常（best-effort）
    """

    def __init__(
        self,
        enabled: bool,
        logs_dir: str,
        log_jsonl: bool = True,
        log_csv: bool = False,
        events_filename: str = 'diagnostics_events.jsonl',
    ):
        self.enabled = bool(enabled)
        self.logs_dir = str(logs_dir or '')
        self.log_jsonl = bool(log_jsonl)
        self.log_csv = bool(log_csv)
        self.events_filename = str(events_filename or 'diagnostics_events.jsonl')
        self._started = False
        self._csv_path = None
        self._jsonl_path = None
        try:
            if self.enabled and self.logs_dir:
                os.makedirs(self.logs_dir, exist_ok=True)
                # 解析 JSONL/CSV 路径
                self._jsonl_path = os.path.join(self.logs_dir, self.events_filename)
                self._csv_path = os.path.join(self.logs_dir, 'diagnostics_events.csv')
        except Exception:
            # 保持静默
            pass

    def _now(self) -> str:
        try:
            return datetime.now().isoformat(timespec='seconds')
        except Exception:
            return datetime.now().isoformat()

    def _safe_write_jsonl(self, rec: Dict[str, Any]):
        if not (self.enabled and self.log_jsonl and self._jsonl_path):
            return
        try:
            with open(self._jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def _safe_write_csv(self, rec: Dict[str, Any]):
        if not (self.enabled and self.log_csv and self._csv_path):
            return
        try:
            # 基础列：ts,event；其余键尽量扁平写入 key=value
            ts = rec.get('ts')
            ev = rec.get('event')
            extras = []
            for k, v in rec.items():
                if k in ('ts', 'event'):
                    continue
                try:
                    extras.append(f"{k}={v}")
                except Exception:
                    pass
            line = f"{ts},{ev},{' '.join(extras)}".rstrip()
            # 写入文件；若文件不存在则写表头
            need_header = not os.path.exists(self._csv_path)
            with open(self._csv_path, 'a', encoding='utf-8') as f:
                if need_header:
                    f.write("ts,event,extras\n")
                f.write(line + "\n")
        except Exception:
            pass

    def start(self, event: str = 'diagnostics.start', meta: Optional[Dict[str, Any]] = None):
        """标记诊断开始。"""
        if not self.enabled:
            return
        self._started = True
        rec = {'ts': self._now(), 'event': str(event or 'diagnostics.start')}
        if isinstance(meta, dict):
            try:
                for k, v in meta.items():
                    rec[str(k)] = v
            except Exception:
                pass
        self._safe_write_jsonl(rec)
        self._safe_write_csv(rec)

    def log(self, event: str, **kwargs):
        """记录任意事件。"""
        if not self.enabled:
            return
        rec = {'ts': self._now(), 'event': str(event or 'event')}
        try:
            for k, v in (kwargs or {}).items():
                rec[str(k)] = v
        except Exception:
            pass
        self._safe_write_jsonl(rec)
        self._safe_write_csv(rec)

    def finalize(self, info: Optional[Dict[str, Any]] = None):
        """标记诊断结束。"""
        if not self.enabled:
            return
        rec = {'ts': self._now(), 'event': 'diagnostics.finalize'}
        if isinstance(info, dict):
            try:
                for k, v in info.items():
                    rec[str(k)] = v
            except Exception:
                pass
        self._safe_write_jsonl(rec)
        self._safe_write_csv(rec)
        self._started = False