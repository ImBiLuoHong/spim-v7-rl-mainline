from dataclasses import dataclass
import torch

@dataclass
class EfficiencyConfig:
    """计算效率配置类 - 管理硬件优化、并行计算与资源调度相关的参数 (SSOT V6.1)"""
    
    def __init__(self):
        # Hardware Optimization Profile (SSOT)
        # Profiles: STANDARD, RTX_5090_EXTREME
        self.hardware_profile = 'RTX_5090_EXTREME'
        
        self.performance = {
            'use_torch_compile': False, # Disabled due to torch_scatter JIT issues
            'enable_nan_guard': True,   # Prevent explosion in physical layers
            'max_batch_size': 1024,
            'cuda_streams': True,       # Parallel data transfer
            'tf32': True               # [SSOT Change] Enable TensorFloat-32 for RTX 5090
        }
        
        if self.hardware_profile == 'RTX_5090_EXTREME':
            # === [PRODUCTION OPTIMIZED] RTX 5090 Saturation Profile (90GB RAM) ===
            self.batch_size = 256          # [OPTIMIZED] Push to Limit for TF32
            self.num_workers = 16          # [OPTIMIZED] Increase workers for high throughput
            self.prefetch_factor = 4       # [TUNED] Aggressive prefetch
            self.pin_memory = True         # [OPTIMIZED]
            self.preload = False           # [USER REQUEST] Disable RAM Preloading (Start Immediately)
            self.persistent_workers = True # [FIX] Add back missing persistent_workers
            self.non_blocking_transfer = True
            self.cudnn_benchmark = True
            self.use_amp = True            # Enable AMP by default for 5090
            
            # TF32 Control (Moved to train script setup, but default is True here)
            # We set the flag here, application logic is in script setup or here if import side effects allowed.
            # But adhering to SSOT, we just define the config here.
            pass
        else:
            self.batch_size = 128
            self.num_workers = 4
            self.pin_memory = True
            self.prefetch_factor = 2
            self.persistent_workers = False
            self.use_amp = False
            self.preload = False
            self.non_blocking_transfer = False
            self.cudnn_benchmark = False
