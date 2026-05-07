from typing import Dict, Any, Callable, Type

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str = None):
        def decorator(cls_or_func: Type):
            reg_name = name if name is not None else cls_or_func.__name__
            if reg_name in self._registry:
                raise ValueError(f"Name '{reg_name}' already registered in {self.name}")
            self._registry[reg_name] = cls_or_func
            return cls_or_func
        return decorator

    def get(self, name: str) -> Type:
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

# Global Registries
NAV_BACKBONE_REGISTRY = Registry("NAV_BACKBONE")
NAV_HEAD_REGISTRY = Registry("NAV_HEAD")
SAMPLER_REGISTRY = Registry("SAMPLER")
NAVIGATOR_REGISTRY = Registry("NAVIGATOR")

REASONER_BACKBONE_REGISTRY = Registry("REASONER_BACKBONE")
REASONER_HEAD_REGISTRY = Registry("REASONER_HEAD")
MEMORY_REGISTRY = Registry("MEMORY")
REASONER_REGISTRY = Registry("REASONER")
PHYSICS_REGISTRY = Registry("PHYSICS")
RACE_ENERGY_REGISTRY = Registry("RACE_ENERGY")
NEG_GATING_REGISTRY = Registry("NEG_GATING")
FOV_REGISTRY = Registry("FOV")
LOSS_REGISTRY = Registry("LOSS")
CURRICULUM_REGISTRY = Registry("CURRICULUM")
METRICS_REGISTRY = Registry("METRICS")
BELIEF_UPDATER_REGISTRY = Registry("BELIEF_UPDATER")
