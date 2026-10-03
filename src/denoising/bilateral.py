"""Edge-Preserving Bilateral Filter for Medical Mammograms.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: CPU based
Implementation: OpenCV cv2.bilateralFilter (multi-threaded SIMD CPU execution).
GPU Acceleration: Not natively available in OpenCV standard build without CUDA module.
"""
import cv2
import numpy as np


def denoise_bilateral(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral filtering to smooth tissue parenchyma while keeping lesion borders sharp.

    Combines spatial closeness (sigma_space) with radiometric intensity difference (sigma_color).

    Computation Device: CPU based (OpenCV).

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        d: Diameter of each pixel neighborhood (e.g. 5, 7, 9).
        sigma_color: Filter sigma in the color/intensity space.
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].
    """
    # Sanitize invalid values and create copy
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    # OpenCV bilateral filter on 32-bit float expects float32 in [0, 1] or uint8 in [0, 255]
    filtered = cv2.bilateralFilter(
        work,
        d=int(d),
        sigmaColor=float(sigma_color) / 255.0 if sigma_color > 1.0 else float(sigma_color),
        sigmaSpace=float(sigma_space),
    )

    clean_out = np.nan_to_num(filtered, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(clean_out, 0.0, 1.0).astype(np.float32)
