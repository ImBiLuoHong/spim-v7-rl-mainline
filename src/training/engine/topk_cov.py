import torch


def compute_topk_label_mass_and_cov_tilde(p: torch.Tensor, K: int) -> tuple:
    """Compute TopKLabelMass@K and CovK_tilde@K from probability matrix p [B,N]."""
    try:
        assert p.dim() == 2, "p must be [B,N]"
        B, N = int(p.shape[0]), int(p.shape[1])
        K_eff = max(1, min(int(K or N), N))
        vals, _ = torch.topk(p, k=K_eff, dim=1, largest=True, sorted=False)
        mass_per_b = vals.sum(dim=1)
        cov_tilde_per_b = mass_per_b / float(K_eff)
        topk_mass_avg = float(mass_per_b.mean().item()) if B > 0 else 0.0
        cov_tilde_avg = float(cov_tilde_per_b.mean().item()) if B > 0 else 0.0
        return topk_mass_avg, cov_tilde_avg
    except Exception:
        return 0.0, 0.0

