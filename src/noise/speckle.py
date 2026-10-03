"""Multiplicative speckle noise simulation.

IMPORTANT RESEARCH DISTINCTION:
--------------------------------
This is an artificially generated synthetic noise model developed solely for an
experimental image denoising benchmark study. It does NOT represent naturally occurring
physical noise in the CBIS-DDSM mammography dataset.
"""
from typing import Optional, Union
import numpy as np
import torch

def add_speckle_noise(
    img: Union[np.ndarray, torch.Tensor],
    var: float = 0.04,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Add multiplicative speckle noise using PyTorch GPU acceleration when available.

    Formula: Noisy = Clip(Image + Image * Normal(0, sqrt(var)), 0.0, 1.0)

    Args:
        img: Input image in [0, 1] as float32 NumPy array or PyTorch Tensor.
        var: Variance of multiplicative Gaussian noise (default: 0.04).
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

    sigma = float(np.sqrt(max(var, 0.0)))
    noise = torch.randn_like(t) * sigma
    noisy = t + t * noise

    return torch.clamp(noisy, 0.0, 1.0).detach().cpu().numpy()
