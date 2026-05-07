import os
import torch
# from src.modeling.architectures.main import SpatioTemporalModel
# from src.modeling.architectures.autosampling import AutoSamplingGNN
# from src.modeling.architectures.t2i_bdas import T2IBDASNet
from src.modeling.architectures.phase4_5_model import Phase45Model

def build_model(cfg, NUM_NODES: int, NUM_FEATURES: int, WINDOW_SIZE: int, device: torch.device):
    """Construct Model from cfg and inject runtime attributes.
    Responsibilities (SRP): model construction + attribute injection for losses/selector/prune/causal/conn.
    """

    architecture = getattr(cfg.model, 'architecture', 'spatiotemporal')
    
    if architecture == 'phase4_5' or architecture == 'Phase45':
        model = Phase45Model(cfg).to(device)
    else:
        # Fallback or Error
        print(f"[WARN] Architecture '{architecture}' is not supported or archived. Defaulting to Phase45Model.")
        model = Phase45Model(cfg).to(device)

    # Inject cfg and logging dirs
    try:
        setattr(model, 'cfg', cfg)
        setattr(model, 'logs_dir', getattr(cfg.paths, 'logs_dir', os.path.join(cfg.paths.run_dir, 'logs')))
        setattr(model, 'run_dir', getattr(cfg.paths, 'run_dir', os.path.dirname(getattr(cfg.paths, 'logs_dir', cfg.paths.run_dir))))
        _conn_enabled = False
        if hasattr(cfg.features, 'connected_eval') and getattr(cfg.features, 'connected_eval') is not None:
            _conn_enabled = bool(getattr(getattr(cfg.features, 'connected_eval'), 'enabled', False))
        if not _conn_enabled:
            _conn_enabled = bool(getattr(cfg.features, 'connected_eval_enabled', False))
        setattr(model, 'connected_eval_enabled', _conn_enabled)
        setattr(model, 'selector_head_enabled', bool(getattr(cfg.features, 'enable_selector_head', False)))
    except Exception:
        pass

    # Selector normalization and temperature
    try:
        setattr(model, 'selector_normalization', str(getattr(cfg.features, 'selector_normalization', 'softmaxK') or 'softmaxK'))
        setattr(model, 'softmax_temperature', float(getattr(cfg.features, 'softmax_temperature', 1.0) or 1.0))
    except Exception:
        pass

    try:
        # Helper to safely get dict or object attributes
        def safe_get(obj, key, default):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        loss_cfg = getattr(cfg, 'loss', None)
        cov_dict = safe_get(loss_cfg, 'coverage', {})
        tv_dict = safe_get(loss_cfg, 'tv', {})
        size_dict = safe_get(loss_cfg, 'size', {})
        causal_cfg = safe_get(loss_cfg, 'causal', {})
        conn_cfg = safe_get(loss_cfg, 'conn', {})
        
        # Features config
        feat_cfg = getattr(cfg, 'features', None)

        setattr(model, 'loss_coverage_enabled', bool(safe_get(cov_dict, 'enabled', False)))
        setattr(model, 'loss_coverage_alpha', float(safe_get(cov_dict, 'alpha', float(safe_get(feat_cfg, 'coverage_alpha', 0.0) or 0.0))))
        setattr(model, 'loss_coverage_weight', float(safe_get(cov_dict, 'weight', 1.0)))
        
        setattr(model, 'loss_tv_enabled', bool(safe_get(tv_dict, 'enabled', False)))
        setattr(model, 'loss_tv_weight', float(safe_get(tv_dict, 'weight', float(safe_get(feat_cfg, 'subgraph_lambda_tv', 0.0) or 0.0))))
        setattr(model, 'loss_tv_mode', str(safe_get(tv_dict, 'mode', 'adj_binary') or 'adj_binary'))
        setattr(model, 'loss_tv_tau', float(safe_get(tv_dict, 'tau', 1200.0)))
        
        setattr(model, 'loss_size_enabled', bool(safe_get(size_dict, 'enabled', False)))
        setattr(model, 'loss_size_weight', float(safe_get(size_dict, 'weight', float(safe_get(feat_cfg, 'subgraph_lambda_size', 0.0) or 0.0))))
        
        # Prune upstream
        prune_cfg = safe_get(feat_cfg, 'prune_upstream', {})
        # Safety check: prune_cfg might be None if feat_cfg is None
        if prune_cfg is None:
            prune_cfg = {}
        # safe_get expects an object with get() or getattr. prune_cfg is dict.
        _p_enabled = bool(prune_cfg.get('enabled', False)) if isinstance(prune_cfg, dict) else bool(getattr(prune_cfg, 'enabled', False))
        _p_window = float(prune_cfg.get('window_s', 0.0)) if isinstance(prune_cfg, dict) else float(getattr(prune_cfg, 'window_s', 0.0))
        
        setattr(model, 'prune_upstream_enabled', _p_enabled)
        setattr(model, 'prune_upstream_window_s', _p_window)
        
        # Causal
        setattr(model, 'loss_causal_enabled', bool(safe_get(causal_cfg, 'enabled', False)))
        setattr(model, 'loss_causal_weight', float(safe_get(causal_cfg, 'weight', 0.0)))
        setattr(model, 'loss_causal_tau', float(safe_get(causal_cfg, 'tau', 1200.0)))
        
        # Connectivity
        setattr(model, 'loss_conn_enabled', bool(safe_get(conn_cfg, 'enabled', False)))
        setattr(model, 'loss_conn_weight', float(safe_get(conn_cfg, 'weight', 0.0)))
        setattr(model, 'loss_conn_mode', str(safe_get(conn_cfg, 'mode', 'endpoints') or 'endpoints'))
        setattr(model, 'loss_conn_tau_time', int(safe_get(conn_cfg, 'tau_time', 600)))
        setattr(model, 'loss_conn_use_tt', bool(safe_get(conn_cfg, 'use_tt', True)))

        # Physics Propagator Flag (for loop.py)
        pp_cfg = getattr(cfg.model, 'physics_propagator', {})
        print(f"DEBUG: pp_cfg type: {type(pp_cfg)}, content: {pp_cfg}")
        # Handle both dict and OmegaConf object
        if isinstance(pp_cfg, dict):
            pp_enabled = bool(pp_cfg.get('enabled', False))
        else:
            pp_enabled = bool(getattr(pp_cfg, 'enabled', False))
            
        if pp_enabled:
            model.propagator = True # Flag to enable loop.py logic

    except Exception as e:
        print(f"[WARN] Model injection failed: {e}")
        import traceback
        traceback.print_exc()
        pass
    return model
