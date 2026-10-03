"""Poisson (quantum shot) noise simulation.

IMPORTANT RESEARCH DISTINCTION:
--------------------------------
This is an artificially generated synthetic noise model developed solely for an
experimental image denoising benchmark study. It does NOT represent naturally occurring
physical noise in the CBIS-DDSM mammography dataset.
"""
from typing import Optional, Union
import numpy as np
import torch

def add_poisson_noise(
    img: Union[np.ndarray, torch.Tensor],
    scale: float = 255.0,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Simulate photon counting Poisson noise using PyTorch GPU acceleration when available.

    Formula: Noisy = Clip(Poisson(Image * scale) / scale, 0.0, 1.0)

    Args:
        img: Input image in [0, 1] as float32 NumPy array or PyTorch Tensor.
        scale: Intensity scaling factor simulating photon counts (default: 255.0).
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

    scaled = torch.clamp(t * float(scale), min=0.0)
    noisy = torch.poisson(scaled) / float(scale)

    return torch.clamp(noisy, 0.0, 1.0).detach().cpu().numpy()
