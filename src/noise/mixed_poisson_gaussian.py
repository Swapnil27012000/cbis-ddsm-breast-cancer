"""Mixed Poisson-Gaussian synthetic noise model.

IMPORTANT RESEARCH DISTINCTION:
--------------------------------
This is an artificially generated synthetic noise model developed solely for an
experimental image denoising benchmark study. It does NOT represent naturally occurring
physical noise in the CBIS-DDSM mammography dataset.
"""
from typing import Optional, Union
import numpy as np
import torch

def add_mixed_poisson_gaussian_noise(
    img: Union[np.ndarray, torch.Tensor],
    poisson_scale: float = 255.0,
    gaussian_var: float = 0.005,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Simulate mixed Poisson-Gaussian detector noise on GPU when available.

    Models quantum photon shot noise combined with additive electronic sensor noise:
        1. Quantum stage: shot_noisy = Poisson(Image * scale) / scale
        2. Electronic stage: mixed = shot_noisy + Normal(0, sqrt(var))

    Args:
        img: Input image in [0, 1] as float32 NumPy array or PyTorch Tensor.
        poisson_scale: Scaling factor for photon counting (default: 255.0).
        gaussian_var: Variance of additive electronic Gaussian noise (default: 0.005).
        seed: Random seed for deterministic reproducibility.
        device: PyTorch device ('cuda:0' or 'cpu').

    Returns:
        np.ndarray: Noisy float32 image with values clipped strictly to [0.0, 1.0].
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    target_dev = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img.astype(np.float32)).to(target_dev)
    else:
        t = img.to(device=target_dev, dtype=torch.float32)

    # 1. Poisson shot noise
    scaled = torch.clamp(t * float(poisson_scale), min=0.0)
    shot_noisy = torch.poisson(scaled) / float(poisson_scale)

    # 2. Additive electronic Gaussian noise
    sigma = float(np.sqrt(max(gaussian_var, 0.0)))
    gauss = torch.randn_like(shot_noisy) * sigma

    mixed = shot_noisy + gauss
    return torch.clamp(mixed, 0.0, 1.0).detach().cpu().numpy()
