import torch


def select_device(force_gpu: bool) -> torch.device:
    """Select training device.
    - Prefer CUDA when available.
    - If force_gpu is True and CUDA is available, return CUDA; otherwise fall back to CPU.
    This function has no side effects and does not read environment variables.
    """
    try:
        if torch.cuda.is_available():
            return torch.device('cuda')
    except Exception:
        pass
    return torch.device('cpu')

