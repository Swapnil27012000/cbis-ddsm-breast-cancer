"""Impulse Salt & Pepper noise simulation.

IMPORTANT RESEARCH DISTINCTION:
--------------------------------
This is an artificially generated synthetic noise model developed solely for an
experimental image denoising benchmark study. It does NOT represent naturally occurring
physical noise in the CBIS-DDSM mammography dataset.
"""
from typing import Optional, Union
import numpy as np
import torch

def add_salt_pepper_noise(
    img: Union[np.ndarray, torch.Tensor],
    amount: float = 0.04,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Add salt-and-pepper impulse noise using PyTorch GPU acceleration when available.

    Args:
        img: Input image in [0, 1] as float32 NumPy array or PyTorch Tensor.
        amount: Overall noise density ratio in [0, 1] (default: 0.04).
        salt_vs_pepper: Proportion of salt (white pixels, 1.0) vs pepper (black pixels, 0.0).
        seed: Random seed for deterministic reproducibility.
        device: PyTorch device ('cuda:0' or 'cpu').

    Returns:
        np.ndarray: Noisy float32 image with impulse values clipped to [0.0, 1.0].
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

    noisy = t.clone()
    rand_matrix = torch.rand_like(t)

    # Thresholds for salt and pepper
    salt_thresh = float(amount * salt_vs_pepper)
    pepper_thresh = 1.0 - float(amount * (1.0 - salt_vs_pepper))

    # Apply impulse spikes in parallel
    noisy[rand_matrix < salt_thresh] = 1.0
    noisy[rand_matrix > pepper_thresh] = 0.0

    return torch.clamp(noisy, 0.0, 1.0).detach().cpu().numpy()
