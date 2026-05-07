"""
System monitoring utilities: CPU/GPU real-time monitoring for training runs.

- Provides a background thread that periodically samples system metrics
- Prints a short status line to console (optional)
- Writes structured logs (JSONL) and optional CSV for post-run analysis

Design principles:
- Single responsibility: this module only handles system monitoring
- Resilient: gracefully degrades when NVML or GPUs are unavailable
- Non-blocking: runs in a dedicated daemon thread, minimal overhead
"""
from __future__ import annotations
import os
import sys
import time
import json
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# 新增：子进程调用用于 nvidia-smi 回退
import subprocess

# psutil for CPU and memory
try:
    import psutil  # type: ignore
except Exception as _e:
    psutil = None  # type: ignore

# pynvml for GPU metrics
try:
    import pynvml  # type: ignore
    _NVML_OK = True
except Exception:
    pynvml = None  # type: ignore
    _NVML_OK = False


def _now_ts() -> float:
    return time.time()


def _collect_cpu(proc: Optional[psutil.Process]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'percent': None,
        'system_mem_used_percent': None,
        'proc_mem_mb': None,
    }
    if psutil is None:
        return out
    try:
        out['percent'] = float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    try:
        vm = psutil.virtual_memory()
        out['system_mem_used_percent'] = float(vm.percent)
    except Exception:
        pass
    try:
        if proc is None:
            proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        out['proc_mem_mb'] = round(rss / (1024 * 1024), 2)
    except Exception:
        pass
    return out


def _init_nvml_once() -> bool:
    global _NVML_OK
    if not _NVML_OK or pynvml is None:
        return False
    try:
        # Init only once
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            # Already initialized or failure
            if 'Already Initialized' in str(e):
                pass
            else:
                _NVML_OK = False
                return False
        return True
    except Exception:
        _NVML_OK = False
        return False


# 新增：当 NVML 不可用时，通过 nvidia-smi 采集 GPU 指标
def _collect_gpus_smi() -> List[Dict[str, Any]]:
    """Collect GPU metrics using nvidia-smi as a resilient fallback.

    Returns a list of dicts with keys: index, name, util, mem_used_mb, mem_total_mb, temp.
    """
    gpus: List[Dict[str, Any]] = []
    try:
        fields = ['index', 'name', 'utilization.gpu', 'memory.used', 'memory.total', 'temperature.gpu']
        cmd = ['nvidia-smi', f"--query-gpu={','.join(fields)}", '--format=csv,noheader,nounits']
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            # 尝试使用 Windows 常见绝对路径
            exe2 = os.path.join(os.getenv('ProgramFiles', 'C:/Program Files'), 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe')
            cmd[0] = exe2
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            return gpus
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < len(fields):
                continue
            idx_str, name, util_str, mem_used_str, mem_total_str, temp_str = parts[:6]
            # 解析整数与浮点
            def _to_int(s: str) -> Optional[int]:
                try:
                    return int(s)
                except Exception:
                    try:
                        return int(s.split()[0])
                    except Exception:
                        return None
            def _to_float(s: str) -> Optional[float]:
                try:
                    return float(s)
                except Exception:
                    try:
                        return float(s.split()[0])
                    except Exception:
                        return None
            util = _to_int(util_str)
            mem_used = _to_float(mem_used_str)
            mem_total = _to_float(mem_total_str)
            temp = _to_int(temp_str)
            gpus.append({
                'index': int(idx_str) if idx_str.isdigit() else i,
                'name': name,
                'util': util if util is not None else 0,
                'mem_used_mb': round(mem_used, 2) if mem_used is not None else None,
                'mem_total_mb': round(mem_total, 2) if mem_total is not None else None,
                'temp': temp,
            })
    except Exception:
        # 保持回退的健壮性，不抛出异常
        return []
    return gpus


def _collect_gpus() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    if _NVML_OK and pynvml is not None:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8') if hasattr(pynvml.nvmlDeviceGetName(handle), 'decode') else pynvml.nvmlDeviceGetName(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = None
                    try:
                        temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        temp = None
                    gpus.append({
                        'index': i,
                        'name': name,
                        'util': int(getattr(util, 'gpu', 0)),
                        'mem_used_mb': round(int(getattr(mem, 'used', 0)) / (1024 * 1024), 2),
                        'mem_total_mb': round(int(getattr(mem, 'total', 0)) / (1024 * 1024), 2),
                        'temp': temp,
                    })
                except Exception:
                    # Continue collecting other devices even if one fails
                    continue
        except Exception:
            # If NVML fails at runtime, proceed to fallback
            pass
    # 回退：若 NVML 不可用或采集结果为空，尝试使用 nvidia-smi
    if not gpus:
        smi_gpus = _collect_gpus_smi()
        if smi_gpus:
            return smi_gpus
    return gpus


@dataclass
class MonitorOptions:
    interval: float = 1.0
    to_console: bool = True
    to_log_jsonl: bool = True
    to_log_csv: bool = False


class SystemMonitor:
    """Real-time system monitor for CPU/GPU usage.

    This component starts a background thread that periodically:
    - Samples CPU, process memory, and GPU metrics
    - Prints a short status line to console for real-time awareness
    - Writes a JSONL record (and optional CSV) to logs_dir for post-run analysis

    Args:
        logs_dir: Directory to write monitoring logs. Files:
            - logs_dir/sysmon.jsonl
            - logs_dir/sysmon.csv (optional)
        interval: Sampling interval in seconds
        to_console: Whether to print status lines to console
        to_log_jsonl: Whether to write JSONL structured log
        to_log_csv: Whether to write CSV log
    """
    def __init__(self, logs_dir: str, interval: float = 1.0, to_console: bool = True,
                 to_log_jsonl: bool = True, to_log_csv: bool = False):
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self.options = MonitorOptions(interval=interval, to_console=to_console,
                                      to_log_jsonl=to_log_jsonl, to_log_csv=to_log_csv)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[Any] = None
        self._jsonl_fp = None
        self._csv_fp = None
        # Attempt NVML init once
        _init_nvml_once()
        # Lazy psutil process
        if psutil is not None:
            try:
                self._proc = psutil.Process(os.getpid())
            except Exception:
                self._proc = None

    def start(self) -> None:
        if self._thread is not None:
            return
        # Open files lazily with UTF-8
        if self.options.to_log_jsonl:
            try:
                self._jsonl_fp = open(os.path.join(self.logs_dir, 'sysmon.jsonl'), 'a', encoding='utf-8')
            except Exception:
                self._jsonl_fp = None
        if self.options.to_log_csv:
            try:
                path = os.path.join(self.logs_dir, 'sysmon.csv')
                new_file = not os.path.exists(path)
                self._csv_fp = open(path, 'a', encoding='utf-8')
                if new_file:
                    self._csv_fp.write('ts,cpu_percent,proc_mem_mb,system_mem_used_percent,gpu_count,gpu0_util,gpu0_mem_used_mb,gpu0_mem_total_mb,gpu0_temp\n')
                    self._csv_fp.flush()
            except Exception:
                self._csv_fp = None
        # Start thread
        self._thread = threading.Thread(target=self._run, name='SystemMonitor', daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=timeout)
        finally:
            # Close files
            try:
                if self._jsonl_fp:
                    self._jsonl_fp.close()
            except Exception:
                pass
            try:
                if self._csv_fp:
                    self._csv_fp.close()
            except Exception:
                pass
            self._thread = None

    def _run(self) -> None:
        interval = float(max(0.1, self.options.interval))
        while not self._stop.is_set():
            ts = _now_ts()
            cpu = _collect_cpu(self._proc)
            gpus = _collect_gpus()
            rec: Dict[str, Any] = {
                'ts': ts,
                'cpu': cpu,
                'gpu': gpus,
            }
            # Write JSONL
            if self._jsonl_fp is not None:
                try:
                    self._jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    self._jsonl_fp.flush()
                except Exception:
                    pass
            # Write CSV (first GPU only, for quick glance)
            if self._csv_fp is not None:
                try:
                    gpu0 = gpus[0] if gpus else {}
                    line = (
                        f"{ts}," 
                        f"{cpu.get('percent') if cpu.get('percent') is not None else ''},"
                        f"{cpu.get('proc_mem_mb') if cpu.get('proc_mem_mb') is not None else ''},"
                        f"{cpu.get('system_mem_used_percent') if cpu.get('system_mem_used_percent') is not None else ''},"
                        f"{len(gpus)},"
                        f"{gpu0.get('util','')},"
                        f"{gpu0.get('mem_used_mb','')},"
                        f"{gpu0.get('mem_total_mb','')},"
                        f"{gpu0.get('temp','')}\n"
                    )
                    self._csv_fp.write(line)
                    self._csv_fp.flush()
                except Exception:
                    pass
            # Console output
            if self.options.to_console:
                try:
                    cpu_p = cpu.get('percent')
                    pmem = cpu.get('proc_mem_mb')
                    smem = cpu.get('system_mem_used_percent')
                    if gpus:
                        g0 = gpus[0]
                        msg = (
                            f"[SYS] CPU {cpu_p}% | Proc {pmem}MB | RAM {smem}% | "
                            f"GPU0 {g0.get('util','?')}% mem {g0.get('mem_used_mb','?')}/{g0.get('mem_total_mb','?')}MB"
                        )
                    else:
                        msg = f"[SYS] CPU {cpu_p}% | Proc {pmem}MB | RAM {smem}% | GPU N/A"
                    print(msg, flush=True)
                except Exception:
                    pass
            # Sleep
            try:
                time.sleep(interval)
            except Exception:
                break


__all__ = ['SystemMonitor', 'MonitorOptions']