import os
import json
from datetime import datetime
from typing import Optional, Dict


def write_heartbeat(logs_dir: str, epoch_index: int) -> Optional[str]:
    """Write/update a heartbeat file at logs/heartbeat.txt with last active epoch and timestamp."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'heartbeat.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'epoch': int(epoch_index),
                'status': 'running'
            }, ensure_ascii=False))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_heartbeat failed: {e}")
        except Exception:
            pass
        return None


def write_finished_ok(logs_dir: str) -> Optional[str]:
    """Mark training finished successfully by writing logs/finished.ok."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'finished.ok')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'timestamp': datetime.now().isoformat(timespec='seconds'), 'status': 'finished'}, ensure_ascii=False))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_finished_ok failed: {e}")
        except Exception:
            pass
        return None


def append_interruption_event(logs_dir: str, event: Dict) -> Optional[str]:
    """Append an interruption or notable event to logs/events.jsonl (JSON Lines)."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'events.jsonl')
        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'event': event or {},
        }
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        return path
    except Exception as e:
        try:
            print(f"[WARN] append_interruption_event failed: {e}")
        except Exception:
            pass
        return None


def write_paused_ok(logs_dir: str) -> Optional[str]:
    """Write logs/paused.ok to indicate a controlled pause (e.g., KeyboardInterrupt)."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'paused.ok')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'timestamp': datetime.now().isoformat(timespec='seconds'), 'status': 'paused'}, ensure_ascii=False))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_paused_ok failed: {e}")
        except Exception:
            pass
        return None


def should_stop(logs_dir: str) -> bool:
    """Return True if a stop request file exists under logs_dir (stop.req or paused.req)."""
    try:
        stop_paths = [
            os.path.join(logs_dir, 'stop.req'),
            os.path.join(logs_dir, 'paused.req'),
        ]
        for p in stop_paths:
            try:
                if os.path.exists(p) and os.path.getsize(p) >= 0:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False