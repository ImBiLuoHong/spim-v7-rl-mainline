from typing import Any, Dict
from src.modeling.registry import BELIEF_UPDATER_REGISTRY
from src.modeling.interfaces.belief_updater import BeliefUpdaterBase
import src.modeling.belief_updaters # Ensure registration

class BeliefUpdaterBuilder:
    @staticmethod
    def build(cfg: Any) -> BeliefUpdaterBase:
        """
        Builds a BeliefUpdater from configuration.
        Expected YAML structure:
        model:
          belief_updater:
            type: "none"
            params:
              hidden_dim: 128
              ...
        """
        belief_cfg = getattr(cfg.model, 'belief_updater', None)
        if belief_cfg is None:
            # Default to 'none' if not specified
            return BELIEF_UPDATER_REGISTRY.get("none")()
        
        updater_type = belief_cfg.type
        params = getattr(belief_cfg, 'params', {})
        
        # Convert DotDict or OmegaConf to regular dict if needed
        if hasattr(params, 'to_dict'):
            params = params.to_dict()
        elif not isinstance(params, dict):
            # Handle other config objects
            params = {k: getattr(params, k) for k in dir(params) if not k.startswith('_')}

        updater_cls = BELIEF_UPDATER_REGISTRY.get(updater_type)
        return updater_cls(**params)
