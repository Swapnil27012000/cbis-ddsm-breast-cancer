"""Peak Signal-to-Noise Ratio (PSNR) Image Quality Metric.

MATHEMATICAL DEFINITION:
-----------------------
PSNR represents the ratio between the maximum possible power of a signal and the power
of corrupting noise affecting the fidelity of its representation:

    PSNR = 10 * log10(MAX_I^2 / MSE) = 20 * log10(MAX_I / sqrt(MSE))

Units: Decibels (dB).
Range: [0.0, +inf), where higher values indicate closer fidelity to reference.
"""
from typing import Optional
import numpy as np
from .mse import compute_mse


def compute_psnr(
    original: np.ndarray,
    degraded: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR in dB) between reference and evaluated images.

    Input Assumptions:
        - original and degraded must have identical dimensions.
        - data_range is the peak dynamic range of the image. If None, it is automatically
          inferred:
            - If array dtype is uint8: data_range = 255.0
            - If array dtype is floating and max(original) <= 1.0: data_range = 1.0
            - Otherwise: max(original) - min(original)

    Edge Cases & Robustness:
        - When original and degraded are identical (MSE = 0), returns float('inf').
        - Avoids division-by-zero or negative log inputs using numerical epsilon guards.
        - Robust against pre-existing NaNs or Infs.

    Args:
        original: Reference clean image as NumPy array.
        degraded: Processed/noisy/denoised image as NumPy array.
        data_range: Maximum possible pixel dynamic range (e.g., 1.0 for float32 in [0, 1]).

    Returns:
        float: PSNR in decibels (dB), or float('inf') for identical images.
    """
    if data_range is None:
        if np.issubdtype(original.dtype, np.integer):
            dr = 255.0
        elif float(np.nanmax(original)) <= 1.05 and float(np.nanmin(original)) >= -0.05:
            dr = 1.0
        else:
            dr = float(np.nanmax(original) - np.nanmin(original))
            dr = max(dr, 1.0)
    else:
        dr = float(data_range)

    if dr <= 0.0:
        raise ValueError(f"data_range must be strictly positive, got: {dr}")

    mse = compute_mse(original, degraded)
    if mse == 0.0:
        return float("inf")

    psnr_val = float(10.0 * np.log10((dr ** 2) / mse))
    return psnr_val
