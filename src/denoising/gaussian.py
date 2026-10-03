"""Gaussian Spatial Smoothing Filter for Mammography Denoising.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: Hybrid (GPU accelerated when CUDA is active, CPU fallback via OpenCV).
Implementation:
  - GPU: PyTorch 2D tensor convolution with separable Gaussian kernel on CUDA.
  - CPU: OpenCV cv2.GaussianBlur (vectorized multi-threaded CPU execution).
"""
from typing import Tuple, Union, Optional
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


def denoise_gaussian(
    img: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]] = (5, 5),
    sigma: float = 1.0,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Apply 2D Gaussian linear spatial filtering for high-frequency noise attenuation.

    Computation Device: Hybrid (GPU accelerated when CUDA is available, CPU fallback).

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        kernel_size: Tuple (kx, ky) or int specifying Gaussian kernel dimensions.
        sigma: Standard deviation of Gaussian blur distribution.
        device: Optional PyTorch device for computation ('cuda:0' or 'cpu').

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].
    """
    # Sanitize invalid values and create copy
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    # Format kernel dimensions
    if isinstance(kernel_size, int):
        ksize = [kernel_size, kernel_size]
    else:
        ksize = [int(kernel_size[0]), int(kernel_size[1])]

    # Kernel sizes must be odd integers
    if ksize[0] % 2 == 0:
        ksize[0] += 1
    if ksize[1] % 2 == 0:
        ksize[1] += 1

    target_dev = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    if target_dev.type == "cuda":
        # GPU execution via PyTorch tensor operations
        t = torch.from_numpy(work).to(device=target_dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Apply Gaussian blur on GPU
        filtered_t = TF.gaussian_blur(t, kernel_size=ksize, sigma=[float(sigma), float(sigma)])
        out = torch.clamp(filtered_t.squeeze(0).squeeze(0), 0.0, 1.0).detach().cpu().numpy()
        return out

    # CPU fallback via OpenCV
    filtered_cpu = cv2.GaussianBlur(work, tuple(ksize), float(sigma))
    return np.clip(filtered_cpu, 0.0, 1.0).astype(np.float32)
