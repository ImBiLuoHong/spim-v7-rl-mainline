import torch
import logging

logger = logging.getLogger(__name__)

class DevicePrefetcher:
    """
    Asynchronously prefetches data to GPU to overlap data transfer with computation.
    Works with PyTorch Geometric Batch objects and standard tensors.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.loader_iter = None
        self.next_input = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input = next(self.loader_iter)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            if hasattr(self.next_input, 'to'):
                # PyG Batch or Tensor
                self.next_input = self.next_input.to(self.device, non_blocking=True)
            elif isinstance(self.next_input, (list, tuple)):
                self.next_input = [t.to(self.device, non_blocking=True) if hasattr(t, 'to') else t for t in self.next_input]
            elif isinstance(self.next_input, dict):
                self.next_input = {k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in self.next_input.items()}
            
    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        # Trigger next load while current batch is being processed
        self.preload()
        return input

def apply_hardware_optimizations(model, cfg):
    """
    Applies a suite of hardware optimizations for high-end GPUs like RTX 5090.
    Based on Phase 4.5 Performance Benchmarking.
    """
    profile = getattr(cfg.efficiency, 'hardware_profile', 'STANDARD')
    perf_cfg = getattr(cfg.efficiency, 'performance', {})
    
    logger.info(f"Applying Hardware Optimization Profile: {profile}")
    
    # 1. TensorFloat-32 (TF32) for Ampere/Ada/Blackwell
    if perf_cfg.get('tf32', True):
        torch.set_float32_matmul_precision('high')
        logger.info("  -> Enabled TF32 (High Precision Matmul)")
        
    # 2. cuDNN Benchmark
    if getattr(cfg.efficiency, 'cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        logger.info("  -> Enabled cuDNN Benchmark")
        
    # 3. Torch Compile (Optional, use with caution for GNNs)
    if perf_cfg.get('use_torch_compile', False):
        try:
            logger.info("  -> Attempting torch.compile()...")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"  -> torch.compile failed: {e}. Falling back.")
            
    return model

def get_device_and_scaler(cfg):
    """
    Standardizes device selection and AMP scaler initialization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = None
    if getattr(cfg.efficiency, 'use_amp', False) and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        logger.info("  -> Initialized AMP GradScaler")
    return device, scaler
