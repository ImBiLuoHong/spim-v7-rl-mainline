import os
import json
from datetime import datetime
from typing import Optional, Dict

from src.shared.logging.logging.common import _to_serializable


def write_env_snapshot(logs_dir: str, runtime: Dict, env_vars: Dict) -> Optional[str]:
    """Write runtime and selected environment variables to logs/env_snapshot.json."""
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'env_snapshot.json')
        data = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'runtime': _to_serializable(runtime or {}),
            'env': _to_serializable(env_vars or {}),
        }
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_env_snapshot failed: {e}")
        except Exception:
            pass
        return None