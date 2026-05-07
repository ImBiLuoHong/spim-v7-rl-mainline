import os
import json
import hashlib
from datetime import datetime
from typing import Optional

from src.shared.logging.logging.common import _get, _to_serializable


def write_config_snapshot_with_hash(logs_dir: str, cfg) -> Optional[str]:
    """Write a config snapshot with a stable hash to logs/config_snapshot_with_hash.json.

    The output JSON contains keys: 'hash' and 'snapshot'.
    'hash' is the SHA256 of json.dumps(snapshot, sort_keys=True).
    """
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, 'config_snapshot_with_hash.json')
        cfg_dict = {}
        try:
            cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else {}
        except Exception:
            cfg_dict = {}

        snapshot = {
            'cfg': _to_serializable(cfg_dict),
            'seed': _get(cfg, 'seed', None),
        }
        try:
            snapshot_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
            h = hashlib.sha256(snapshot_json.encode('utf-8')).hexdigest()
        except Exception:
            h = hashlib.sha256(json.dumps({}, ensure_ascii=False).encode('utf-8')).hexdigest()
        payload = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'hash': h,
            'snapshot': snapshot,
        }
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        return path
    except Exception as e:
        try:
            print(f"[WARN] write_config_snapshot_with_hash failed: {e}")
        except Exception:
            pass
        return None