
import os
import json
import hashlib
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("AuditCache")

class AuditCache:
    """
    Manages audit cache to avoid redundant heavy audits.
    """
    
    CACHE_DIR = ".cache/audit_pass"
    PLATFORM_VERSION = "1.0.0" # Should match run_audit.py
    EVAL_CONTRACT_VERSION = "v1" # Frozen
    DATA_CONTRACT_VERSION = "v1" # Frozen
    
    @staticmethod
    def _get_git_hash() -> str:
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "no-git"

    @staticmethod
    def _get_config_hash(cfg: Any) -> str:
        """
        Generate a hash for configuration relevant to audit.
        We only care about data paths, model architecture, and feature flags.
        Training hyperparameters (lr, epochs) do not affect data/model contract.
        """
        relevant_cfg = {
            'paths': {
                'foundation_path': getattr(cfg.paths, 'foundation_path', ''),
                'samples_path': getattr(cfg.paths, 'samples_path', ''),
            },
            'model': {
                'architecture': getattr(cfg.model, 'architecture', ''),
                'input_dim': getattr(cfg.model, 'input_dim', 0),
                'edge_dim': getattr(cfg.model, 'edge_dim', 0),
                'navigator': getattr(cfg.model, 'navigator', {}),
                'reasoner': getattr(cfg.model, 'reasoner', {}), # Check composition
                'physics': getattr(cfg.model, 'physics', {}),
            },
            'data': {
                'task_mode': getattr(cfg.data, 'task_mode', ''),
                'use_dataloader_v6': getattr(cfg.data, 'use_dataloader_v6', True),
                # Feature flags
                'use_chlorine_val': getattr(cfg.data, 'use_chlorine_val', True),
                'use_poison_bin': getattr(cfg.data, 'use_poison_bin', True),
                'use_is_revealed': getattr(cfg.data, 'use_is_revealed', True),
            },
            'loss': getattr(cfg, 'loss', {}), # Loss type affects model contract
        }
        
        # Helper to serialize
        def default(o):
            if hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)
            
        cfg_str = json.dumps(relevant_cfg, sort_keys=True, default=default)
        return hashlib.md5(cfg_str.encode()).hexdigest()[:12]

    @classmethod
    def generate_key(cls, cfg: Any) -> str:
        """
        Generates a unique cache key based on:
        - Platform Version
        - Git Commit
        - Data/Eval Contract Version
        - Config Hash (Data/Model specific)
        """
        git_hash = cls._get_git_hash()
        cfg_hash = cls._get_config_hash(cfg)
        
        components = [
            f"plat-{cls.PLATFORM_VERSION}",
            f"git-{git_hash}",
            f"eval-{cls.EVAL_CONTRACT_VERSION}",
            f"data-{cls.DATA_CONTRACT_VERSION}",
            f"cfg-{cfg_hash}"
        ]
        
        raw_key = "_".join(components)
        final_key = hashlib.md5(raw_key.encode()).hexdigest()
        
        # logger.debug(f"Audit Cache Key Components: {components} -> {final_key}")
        return final_key

    @classmethod
    def check(cls, key: str) -> bool:
        """
        Checks if the audit key exists in cache.
        """
        cache_path = os.path.join(cls.CACHE_DIR, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                # Touch the file to update mtime (LRU-like)
                os.utime(cache_path, None)
                return True
            except:
                pass
        return False

    @classmethod
    def write(cls, key: str, metadata: Dict[str, Any] = None):
        """
        Writes the audit key to cache.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(cls.CACHE_DIR, f"{key}.json")
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "metadata": metadata or {}
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            # logger.info(f"Audit cache written: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write audit cache: {e}")

