from src.modeling.registry import (
    NAVIGATOR_REGISTRY, REASONER_REGISTRY, PHYSICS_REGISTRY, 
    FOV_REGISTRY, LOSS_REGISTRY, NAV_BACKBONE_REGISTRY, NAV_HEAD_REGISTRY,
    SAMPLER_REGISTRY, REASONER_BACKBONE_REGISTRY, REASONER_HEAD_REGISTRY, MEMORY_REGISTRY
)
from src.modeling.controllers.fov_controller import build_fov_controller
from src.modeling.losses import build_primary_criterion
import logging

# Ensure all modules are registered
import src.modeling.navigators.standard
import src.modeling.navigators.composed
import src.modeling.navigators.backbones
import src.modeling.navigators.heads
import src.modeling.navigators.samplers
import src.modeling.navigators.vnext
import src.modeling.navigators.random
import src.modeling.navigators.heuristic
import src.modeling.navigators.frozen_clean_bridge
import src.modeling.reasoners.bayesian
import src.modeling.reasoners.clean_aligned
import src.modeling.reasoners.backbones
import src.modeling.reasoners.heads
import src.modeling.reasoners.composed
import src.modeling.memory.modules
import src.modeling.physics.race_consistency
import src.modeling.physics.bricks

logger = logging.getLogger(__name__)

class ModelBuilder:
    @staticmethod
    def build_model(cfg):
        from src.modeling.architectures.phase4_5_model import Phase45Model
        from src.modeling.loop.episode_stepper import EpisodeStepper
        
        navigator = ModelBuilder.build_navigator(cfg)
        reasoner = ModelBuilder.build_reasoner(cfg)
        physics = ModelBuilder.build_physics(cfg)
        fov = ModelBuilder.build_fov(cfg)
        
        model = Phase45Model(cfg, navigator, reasoner, physics, fov)
        
        return model

    @staticmethod
    def build_navigator(cfg):
        nav_mode = getattr(cfg.model, 'navigator_mode', 'learned')
        
        if nav_mode == 'random':
            logger.info(f"Building Navigator: Random")
            return NAVIGATOR_REGISTRY.get("random")(cfg)
        elif nav_mode == 'heuristic_stt':
            logger.info(f"Building Navigator: Heuristic STT")
            return NAVIGATOR_REGISTRY.get("heuristic_stt")(cfg)

        nav_cfg = getattr(cfg.model, 'navigator', None)
        
        # New 3D Composition Logic: Backbone x Head x Sampler
        if nav_cfg and hasattr(nav_cfg, 'backbone_type') and hasattr(nav_cfg, 'head_type'):
            backbone_type = nav_cfg.backbone_type
            head_type = nav_cfg.head_type
            sampler_type = getattr(nav_cfg, 'sampler_type', 'sampler_topk_wo_replacement')
            
            logger.info(f"Building composed navigator: {backbone_type} x {head_type} x {sampler_type}")
            
            backbone_cls = NAV_BACKBONE_REGISTRY.get(backbone_type)
            head_cls = NAV_HEAD_REGISTRY.get(head_type)
            sampler_cls = SAMPLER_REGISTRY.get(sampler_type)
            
            backbone = backbone_cls(cfg)
            head = head_cls(cfg)
            sampler = sampler_cls(cfg)
            
            # Use the ComposedNavigator wrapper
            from src.modeling.navigators.composed import ComposedNavigator
            return ComposedNavigator(backbone, head, sampler, cfg)
            
        # Backward Compatibility
        nav_type = getattr(cfg.model, 'navigator_type', 'standard_v4_5')
        if nav_type != 'standard_v4_5':
            logger.warning(f"Using legacy navigator_type: {nav_type}. Recommended: navigator.backbone_type & head_type")
            
        cls = NAVIGATOR_REGISTRY.get(nav_type)
        return cls(cfg)

    @staticmethod
    def build_reasoner(cfg):
        reason_cfg = getattr(cfg.model, 'reasoner', None)
        
        # New 3D Composition Logic: Backbone x Head x Memory
        if reason_cfg and hasattr(reason_cfg, 'backbone_type') and hasattr(reason_cfg, 'head_type'):
            backbone_type = reason_cfg.backbone_type
            head_type = reason_cfg.head_type
            memory_type = getattr(reason_cfg, 'memory_type', 'memory_none')
            
            logger.info(f"Building composed reasoner: {backbone_type} x {head_type} x {memory_type}")
            
            backbone_cls = REASONER_BACKBONE_REGISTRY.get(backbone_type)
            head_cls = REASONER_HEAD_REGISTRY.get(head_type)
            memory_cls = MEMORY_REGISTRY.get(memory_type)
            
            backbone = backbone_cls(cfg)
            head = head_cls(cfg)
            memory = memory_cls(cfg) if memory_type != 'memory_none' else None
            
            from src.modeling.reasoners.composed import ComposedReasoner
            return ComposedReasoner(backbone, head, memory, cfg)

        # Backward Compatibility
        reasoner_type = getattr(cfg.model, 'reasoner_type', 'bayesian_v4_5')
        if reasoner_type != 'bayesian_v4_5':
            logger.warning(f"Using legacy reasoner_type: {reasoner_type}. Recommended: reasoner.backbone_type & head_type")
            
        cls = REASONER_REGISTRY.get(reasoner_type)
        return cls(cfg)

    @staticmethod
    def build_physics(cfg):
        # [SSOT Refactor] Support new cfg.physics.rea_rules structure
        # Priority: cfg.physics.rea_rules.energy_model > cfg.model.physics_type (Legacy)
        
        phys_type = None
        
        # New SSOT Path
        if hasattr(cfg, 'physics') and hasattr(cfg.physics, 'rea_rules'):
            phys_type = getattr(cfg.physics.rea_rules, 'energy_model', None)
            
        # Fallback to Legacy
        if phys_type is None:
            phys_type = getattr(cfg.model, 'physics_type', None)
            phys_cfg = getattr(cfg.model, 'physics', {})
            if isinstance(phys_cfg, dict):
                phys_type = phys_cfg.get('type', phys_type)
            else:
                phys_type = getattr(phys_cfg, 'type', phys_type)
            
        if phys_type is None:
            phys_type = 'physics_none'
            
        logger.info(f"Building physics consistency module: {phys_type}")
        cls = PHYSICS_REGISTRY.get(phys_type)
        physics_module = cls(cfg)
        
        # Contract Check: If config requires bias injection but module doesn't provide it
        # [SSOT Refactor] Check cfg.physics.rea_rules.use_bias
        use_bias_injection = False
        if hasattr(cfg, 'physics') and hasattr(cfg.physics, 'rea_rules'):
             use_bias_injection = getattr(cfg.physics.rea_rules, 'use_bias', False)
        else:
             use_bias_injection = getattr(cfg.model, 'use_physics_bias', False)

        if hasattr(physics_module, 'capabilities'):
            caps = physics_module.capabilities()
            if use_bias_injection and not caps.get('provides_bias', False):
                raise ValueError(
                    f"YAML Configuration Error: 'use_physics_bias: true' requires a physics module that provides bias, "
                    f"but '{phys_type}' does not. Change physics type or set 'use_bias: false'."
                )
            
        return physics_module

    @staticmethod
    def build_fov(cfg):
        return build_fov_controller(cfg)

    @staticmethod
    def build_belief_updater(cfg):
        from src.modeling.builders.belief_updater_builder import BeliefUpdaterBuilder
        return BeliefUpdaterBuilder.build(cfg)

    @staticmethod
    def build_loss(cfg):
        criterion = build_primary_criterion(cfg)
        return criterion

    @staticmethod
    def validate_capabilities(navigator, loss_engine):
        """
        Capability Contract Validation: Ensure Navigator can provide what LossEngine requires.
        """
        nav_caps = navigator.capabilities()
        loss_reqs = loss_engine.requirements()
        
        errors = []
        if loss_reqs.get('requires_soft_actions') and not nav_caps.get('supports_soft_actions'):
            errors.append(f"LossEngine requires soft actions, but Sampler '{type(navigator.sampler).__name__}' does not support them.")
            
        if loss_reqs.get('requires_active_mask') and 'active_mask' not in nav_caps.get('output_fields', []) and 'active_mask' not in ['active_mask']:
            # active_mask is usually provided by environment/orchestrator
            pass
            
        required_fields = loss_reqs.get('required_fields', [])
        output_fields = nav_caps.get('output_fields', [])
        # Common fields provided by Orchestrator are excluded from this check
        orchestrator_fields = ['reasoner_logits', 'active_mask', 'fused_source_label', 'is_hit', 'logits']
        
        for field in required_fields:
            if field not in output_fields and field not in orchestrator_fields:
                errors.append(f"LossEngine requires field '{field}', but Navigator components {output_fields} do not provide it.")
                
        return errors
