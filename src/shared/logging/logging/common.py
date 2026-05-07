import json


def _get(cfg, dotted: str, default=None):
    """Safely fetch nested attributes from cfg using dotted path (attr or key).
    Falls back to dict-style if available.
    """
    try:
        parts = dotted.split('.')
        cur = cfg
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


def _to_serializable(val):
    """Convert common objects to JSON serializable forms without altering semantics."""
    try:
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        if isinstance(val, (list, tuple)):
            return [_to_serializable(v) for v in val]
        if isinstance(val, dict):
            return {str(k): _to_serializable(v) for k, v in val.items()}
        # Generic objects: try to fetch __dict__ or str
        d = getattr(val, '__dict__', None)
        if isinstance(d, dict):
            return {str(k): _to_serializable(v) for k, v in d.items()}
        return str(val)
    except Exception:
        try:
            return str(val)
        except Exception:
            return None