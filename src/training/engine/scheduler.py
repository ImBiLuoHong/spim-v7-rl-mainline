import torch
import torch.optim.lr_scheduler as _ls


class WarmupThenCosine:
    """Warmup + Cosine scheduler composition.
    Called once per epoch via .step().
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, t_max: int, eta_min: float, start_factor: float = 0.1):
        self.optimizer = optimizer
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.start_factor = float(max(0.0, min(1.0, start_factor)))
        self._epoch = 0
        self.base_lrs = [float(pg.get('lr', 1e-3)) for pg in optimizer.param_groups]
        self.inner = _ls.CosineAnnealingLR(optimizer, T_max=int(max(1, t_max)), eta_min=float(eta_min))

    def step(self):
        if self._epoch < self.warmup_epochs:
            self._epoch += 1
            progress = float(self._epoch) / float(max(1, self.warmup_epochs))
            scale = self.start_factor + progress * (1.0 - self.start_factor)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = float(base_lr) * float(scale)
        else:
            self._epoch += 1
            self.inner.step()

    def state_dict(self):
        return {
            'epoch': self._epoch,
            'warmup_epochs': self.warmup_epochs,
            'start_factor': self.start_factor,
            'base_lrs': list(self.base_lrs),
            'inner': getattr(self.inner, 'state_dict', lambda: {})(),
        }

    def load_state_dict(self, state: dict):
        try:
            self._epoch = int(state.get('epoch', 0))
            self.warmup_epochs = int(state.get('warmup_epochs', self.warmup_epochs))
            self.start_factor = float(state.get('start_factor', self.start_factor))
            bl = state.get('base_lrs', None)
            if isinstance(bl, (list, tuple)) and len(bl) == len(self.base_lrs):
                self.base_lrs = [float(x) for x in bl]
            inner_state = state.get('inner', None)
            if inner_state is not None:
                self.inner.load_state_dict(inner_state)
        except Exception:
            pass


def build_scheduler(optimizer: torch.optim.Optimizer, training_cfg) -> object:
    """Build LR scheduler from cfg.training.
    Supports: plateau, cosine (+ optional warmup), none.
    """
    try:
        name = str(getattr(training_cfg, 'lr_scheduler', 'plateau') or 'plateau').lower()
    except Exception:
        name = 'plateau'

    if name in ('none', 'off', 'disabled'):
        return None

    if name in ('plateau', 'reduce_on_plateau', 'reduce_lr_on_plateau'):
        try:
            factor = float(getattr(training_cfg, 'lr_factor', 0.5))
        except Exception:
            factor = 0.5
        try:
            patience = int(getattr(training_cfg, 'lr_patience', 3))
        except Exception:
            patience = 3
        try:
            threshold = float(getattr(training_cfg, 'lr_threshold', 1e-4))
        except Exception:
            threshold = 1e-4
        try:
            cooldown = int(getattr(training_cfg, 'lr_cooldown', 0))
        except Exception:
            cooldown = 0
        try:
            min_lr = float(getattr(training_cfg, 'lr_min', 0.0))
        except Exception:
            min_lr = 0.0
        return _ls.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
        )

    if name in ('cosine', 'cosine_anneal', 'cosineannealing'):
        try:
            tmax = int(getattr(training_cfg, 'cosine_tmax_epochs', getattr(training_cfg, 'num_epochs', 20)))
        except Exception:
            tmax = 20
        try:
            eta_min = float(getattr(training_cfg, 'cosine_eta_min', 0.0))
        except Exception:
            eta_min = 0.0
        try:
            warmup = int(getattr(training_cfg, 'lr_warmup_epochs', 0))
        except Exception:
            warmup = 0
        if warmup and warmup > 0:
            return WarmupThenCosine(optimizer, warmup_epochs=warmup, t_max=tmax, eta_min=eta_min, start_factor=0.1)
        return _ls.CosineAnnealingLR(optimizer, T_max=int(max(1, tmax)), eta_min=float(eta_min))

    return None

