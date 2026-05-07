class HorizonScheduler:
    """
    Slot 12: HorizonScheduler (Standardized Entry Point)
    
    Responsibilities:
    1. Unify management of max_train_steps and max_eval_steps.
    2. Provide interface for Curriculum Learning (dynamic horizon).
    """
    def __init__(self, cfg):
        self.cfg = cfg
        # SSOT: Defaults from Config
        # [Fix] Use correct config path: cfg.training.max_train_episodes
        self.training_max_steps = getattr(cfg.training, 'max_train_episodes', 3)
        self.eval_max_steps = getattr(cfg.training, 'max_eval_episodes', 10)
        
        # Safety Defaults
        if not self.training_max_steps: self.training_max_steps = 3
        if not self.eval_max_steps: self.eval_max_steps = 10

        # Curriculum Config
        self.curriculum = getattr(cfg.training, 'curriculum', {})
        self.enabled = self.curriculum.get('enabled', False) if isinstance(self.curriculum, dict) else getattr(self.curriculum, 'enabled', False)
        
        # Handle dict or object access for schedule
        if isinstance(self.curriculum, dict):
            self.horizon_schedule = self.curriculum.get('horizon_schedule', [])
            self.strength_schedule = self.curriculum.get('strength_schedule', [])
        else:
            self.horizon_schedule = getattr(self.curriculum, 'horizon_schedule', [])
            self.strength_schedule = getattr(self.curriculum, 'strength_schedule', [])

    def get_train_horizon(self, epoch=0):
        """
        Returns max_steps for training loop.
        Args:
            epoch: Current epoch (for curriculum)
        """
        # Default fallback
        current_steps = self.training_max_steps

        if not self.enabled:
            return current_steps
        
        # Dynamic Schedule: Find the latest applicable epoch threshold
        # horizon_schedule: list of [start_epoch, steps], e.g. [[0, 3], [10, 8]]
        
        # Sort descending by epoch to find the active stage
        if self.horizon_schedule:
            for start_epoch, horizon in sorted(self.horizon_schedule, key=lambda x: x[0], reverse=True):
                if epoch >= start_epoch:
                    current_steps = horizon
                    break
        
        return current_steps

    def get_eval_horizon(self):
        """
        Returns max_steps for evaluation loop.
        """
        return self.eval_max_steps
        
    def get_fov_strength(self, epoch=0):
        """
        Returns FoV strength for current epoch.
        """
        current_strength = 1.0
        if hasattr(self.cfg, 'fov_controller') and isinstance(self.cfg.fov_controller, dict):
             current_strength = self.cfg.fov_controller.get('strength', 1.0)
             
        if not self.enabled:
            return current_strength
            
        for start_epoch, strength in sorted(self.strength_schedule, key=lambda x: x[0], reverse=True):
            if epoch >= start_epoch:
                current_strength = strength
                break
        return current_strength
