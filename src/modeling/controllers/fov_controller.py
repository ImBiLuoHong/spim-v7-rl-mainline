import torch
import torch.nn as nn
import numpy as np
from src.modeling.interfaces.base import FoVControllerBase
from src.modeling.registry import FOV_REGISTRY

@FOV_REGISTRY.register("entropy_driven")
class EntropyDrivenFoVController(FoVControllerBase):
    def __init__(self, cfg):
        self.cfg = cfg
        params = getattr(cfg.fov_controller, 'params', {})
        self.M_min = params.get('M_min', 10)
        self.M_max = params.get('M_max', 100)
        self.H_hi = params.get('entropy_hi', 2.0)
        self.H_lo = params.get('entropy_lo', 0.5)
        self.L_min = params.get('L_min', 1)
        self.L_max = params.get('L_max', 3)
        self.strength = getattr(cfg.fov_controller, 'strength', 1.0)

    def step(self, stats_dict):
        entropy = stats_dict.get('entropy', self.H_hi)
        
        # Normalize entropy to [0, 1]
        alpha = (entropy - self.H_lo) / (self.H_hi - self.H_lo + 1e-6)
        alpha = max(0.0, min(1.0, float(alpha)))
        
        # Scale by strength
        alpha = alpha * self.strength
        
        candidate_topM = int(self.M_min + alpha * (self.M_max - self.M_min))
        gnn_layers = int(self.L_min + alpha * (self.L_max - self.L_min))
        
        return {
            'candidate_topM': candidate_topM,
            'gnn_layers': gnn_layers,
            'alpha': alpha
        }

@FOV_REGISTRY.register("conflict_driven")
class ConflictDrivenFoVController(FoVControllerBase):
    def __init__(self, cfg):
        self.cfg = cfg
        params = getattr(cfg.fov_controller, 'params', {})
        self.M_min = params.get('M_min', 10)
        self.M_max = params.get('M_max', 100)
        self.E_hi = params.get('energy_hi', 10.0)
        self.E_lo = params.get('energy_lo', 1.0)
        self.strength = getattr(cfg.fov_controller, 'strength', 1.0)

    def step(self, stats_dict):
        energy = stats_dict.get('race_conflict_mean', self.E_hi)
        
        alpha = (energy - self.E_lo) / (self.E_hi - self.E_lo + 1e-6)
        alpha = max(0.0, min(1.0, float(alpha)))
        alpha = alpha * self.strength
        
        candidate_topM = int(self.M_min + alpha * (self.M_max - self.M_min))
        # Safely get gnn_layers from model config
        model_cfg = getattr(self.cfg, 'model', {})
        gnn_layers = model_cfg.get('gnn_layers', 2) if isinstance(model_cfg, dict) else getattr(model_cfg, 'gnn_layers', 2)
        
        return {
            'candidate_topM': candidate_topM,
            'gnn_layers': gnn_layers,
            'alpha': alpha
        }

@FOV_REGISTRY.register("none")
class NoneFoVController(FoVControllerBase):
    def __init__(self, cfg):
        self.cfg = cfg
        # [SSOT Fix] Do not set defaults that override model config
        pass

    def step(self, stats_dict):
        # Return empty dict so EpisodeStepper uses its own defaults (sample_budget)
        return {}

def build_fov_controller(cfg):
    # Check if fov_controller exists in cfg
    if not hasattr(cfg, 'fov_controller'):
        return NoneFoVController(cfg)
        
    fov_type = getattr(cfg.fov_controller, 'type', 'none')
    try:
        cls = FOV_REGISTRY.get(fov_type)
        return cls(cfg)
    except KeyError:
        return NoneFoVController(cfg)
