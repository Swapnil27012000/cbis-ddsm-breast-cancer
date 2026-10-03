"""Kuan Adaptive Filter for Multiplicative Speckle Reduction in Medical Images.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: Hybrid (GPU accelerated on CUDA via PyTorch, CPU fallback via SciPy uniform_filter).
Implementation:
  - GPU: PyTorch 2D local moving average pooling (F.avg_pool2d) for local mean and variance.
  - CPU: SciPy scipy.ndimage.uniform_filter.
"""
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter


def denoise_kuan(
    img: np.ndarray,
    window_size: int = 7,
    noise_var: float = 0.04,
    damping: float = 1.0,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Apply Kuan adaptive filter for multiplicative speckle attenuation in mammographic tissue.

    Mathematical formulation (Kuan et al.):
        Local mean: y_bar = E[y]
        Local variance: sigma_y^2 = E[y^2] - (E[y])^2
        Adaptive weighting factor:
            W = max(0, (1 - (noise_var * y_bar^2) / max(sigma_y^2, eps)) / (1 + noise_var))
            Filtered = y_bar + (W * damping) * (y - y_bar)

    Computation Device: Hybrid (GPU accelerated when CUDA is active, CPU fallback via SciPy).

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        window_size: Size of square moving neighborhood (must be odd integer >= 3, default: 7).
        noise_var: Theoretical variance of the multiplicative speckle noise (default: 0.04).
        damping: Smoothing factor modifying weighting factor responsiveness (default: 1.0).
        device: Optional PyTorch device for computation ('cuda:0' or 'cpu').

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got: {window_size}")

    # Sanitize invalid values and create copy
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    target_dev = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    if target_dev.type == "cuda":
        # GPU execution using PyTorch tensor operations
        t = torch.from_numpy(work).to(device=target_dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        pad = window_size // 2

        # Compute local mean and local squared mean using reflective padding + avg_pool2d
        t_pad = F.pad(t, (pad, pad, pad, pad), mode="reflect")
        t_sq_pad = F.pad(t ** 2, (pad, pad, pad, pad), mode="reflect")

        local_mean = F.avg_pool2d(t_pad, kernel_size=window_size, stride=1)
        local_sq_mean = F.avg_pool2d(t_sq_pad, kernel_size=window_size, stride=1)
        local_var = torch.clamp(local_sq_mean - local_mean ** 2, min=1e-8)

        # Kuan weight calculation
        sigma_u2 = float(max(noise_var, 1e-8))
        ratio = (sigma_u2 * (local_mean ** 2)) / local_var
        w = torch.clamp((1.0 - ratio) / (1.0 + sigma_u2), 0.0, 1.0)

        # Reconstructed pixel intensity
        filtered = local_mean + (w * float(damping)) * (t - local_mean)
        out = torch.clamp(filtered.squeeze(0).squeeze(0), 0.0, 1.0).detach().cpu().numpy()
        return out

    # CPU fallback via SciPy
    local_mean_cpu = uniform_filter(work, size=int(window_size), mode="reflect")
    local_sq_mean_cpu = uniform_filter(work ** 2, size=int(window_size), mode="reflect")
    local_var_cpu = np.maximum(local_sq_mean_cpu - local_mean_cpu ** 2, 1e-8)

    sigma_u2 = float(max(noise_var, 1e-8))
    ratio = (sigma_u2 * (local_mean_cpu ** 2)) / local_var_cpu
    w = np.clip((1.0 - ratio) / (1.0 + sigma_u2), 0.0, 1.0)

    filtered_cpu = local_mean_cpu + (w * float(damping)) * (work - local_mean_cpu)
    clean_out = np.nan_to_num(filtered_cpu, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(clean_out, 0.0, 1.0).astype(np.float32)
