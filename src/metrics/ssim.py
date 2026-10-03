"""Structural Similarity Index Measure (SSIM) Image Quality Metric.

MATHEMATICAL DEFINITION:
-----------------------
SSIM evaluates perceptual image degradation across three structural components:
luminance, contrast, and structure:

    SSIM(x, y) = [l(x, y)]^alpha * [c(x, y)]^beta * [s(x, y)]^gamma

In standard form (Wang et al., 2004 with alpha=beta=gamma=1):

    SSIM(x, y) = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) /
                 ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

where C1 = (K1 * L)^2, C2 = (K2 * L)^2, with K1=0.01, K2=0.03, and L = dynamic range.
Range: [-1.0, 1.0], where 1.0 indicates identical structural fidelity.
"""
from typing import Optional
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def compute_ssim(
    original: np.ndarray,
    degraded: np.ndarray,
    data_range: Optional[float] = None,
    win_size: Optional[int] = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) between reference and evaluated images.

    Input Assumptions:
        - original and degraded must have identical dimensions (2D grayscale).
        - Dynamic range L is automatically determined if not provided.

    Edge Cases & Robustness:
        - Small images: if min(H, W) < 7, win_size is dynamically adjusted to min(H, W) (odd).
        - Identical arrays return exactly 1.0.
        - Sanitizes pre-existing NaNs or Infs.

    Args:
        original: Reference clean image as 2D NumPy array.
        degraded: Processed/noisy/denoised image as 2D NumPy array.
        data_range: Peak dynamic range (default: 1.0 for float32 in [0, 1]).
        win_size: Gaussian sliding window size (must be odd, default: 7).

    Returns:
        float: SSIM score bounded in [-1.0, 1.0].
    """
    if original.shape != degraded.shape:
        raise ValueError(
            f"Shape mismatch: original shape {original.shape} does not match degraded shape {degraded.shape}"
        )

    orig_f = np.nan_to_num(original.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    deg_f = np.nan_to_num(degraded.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)

    # Identical array shortcut
    if np.array_equal(orig_f, deg_f):
        return 1.0

    if data_range is None:
        if np.issubdtype(original.dtype, np.integer):
            dr = 255.0
        elif float(np.max(orig_f)) <= 1.05 and float(np.min(orig_f)) >= -0.05:
            dr = 1.0
        else:
            dr = float(np.max(orig_f) - np.min(orig_f))
            dr = max(dr, 1.0)
    else:
        dr = float(data_range)

    # Determine adaptive window size for small patches
    min_dim = min(orig_f.shape[0], orig_f.shape[1])
    if win_size is None:
        w_size = 7 if min_dim >= 7 else (min_dim if min_dim % 2 == 1 else min_dim - 1)
    else:
        w_size = int(win_size)
        if w_size > min_dim:
            w_size = min_dim if min_dim % 2 == 1 else min_dim - 1

    w_size = max(3, w_size)

    val = float(ssim_fn(orig_f, deg_f, data_range=dr, win_size=w_size))
    return float(np.clip(val, -1.0, 1.0))
