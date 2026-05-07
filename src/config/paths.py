import os
from datetime import datetime
from typing import Optional

def ensure_unique_run_name(base_name: str) -> str:
    """Helper to generate unique run name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"

def derive_logs_dir(cfg: object) -> str:
    """
    Derive logs directory from config object.
    Expects cfg.paths.run_dir to be set.
    """
    try:
        run_dir = getattr(getattr(cfg, 'paths', None), 'run_dir', None)
        if run_dir:
            return os.path.join(run_dir, 'logs')
    except Exception:
        pass
    
    # Fallback to env or default
    run_dir = os.getenv('RUN_DIR', '')
    if run_dir:
        return os.path.join(run_dir, 'logs')
        
    return 'logs'
