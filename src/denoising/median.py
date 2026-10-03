"""Standard Median Filtering for Medical Mammograms.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: CPU based
Implementation: OpenCV cv2.medianBlur (SIMD-accelerated multi-threaded CPU execution).
GPU Acceleration: Not natively available in OpenCV without custom CUDA kernels.
"""
from typing import Union
import cv2
import numpy as np


def denoise_median(
    img: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply standard 2D median filtering to suppress impulse and speckle noise.

    Computation Device: CPU based (OpenCV SIMD).

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        kernel_size: Size of the median square neighborhood (must be odd integer >= 3).

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].

    Raises:
        ValueError: If kernel_size is not an odd positive integer >= 3.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"Median kernel_size must be an odd integer >= 3, got: {kernel_size}")

    # Ensure clean float32 copy and sanitize invalid values
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    # OpenCV cv2.medianBlur operates efficiently on 8-bit unsigned integers
    img_8u = np.clip(work * 255.0, 0.0, 255.0).round().astype(np.uint8)
    denoised_8u = cv2.medianBlur(img_8u, int(kernel_size))

    # Convert back to normalized float32 [0.0, 1.0]
    out = denoised_8u.astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)
