"""Mean Squared Error (MSE) Image Quality Metric.

MATHEMATICAL DEFINITION:
-----------------------
MSE measures the average squared difference between the reference (clean) image
and the processed (degraded/denoised) image:

    MSE = (1 / (M * N)) * sum_{i=1}^M sum_{j=1}^N (I_ref(i, j) - I_deg(i, j))^2

Range: [0.0, +inf), where 0.0 indicates perfect identity.
"""
from typing import Union
import numpy as np


def compute_mse(
    original: np.ndarray,
    degraded: np.ndarray,
) -> float:
    """Compute Mean Squared Error (MSE) between reference and evaluated images.

    Input Assumptions:
        - original and degraded must have identical dimensions (H, W) or (H, W, C).
        - Images should have comparable dynamic ranges (typically float32 in [0.0, 1.0]
          or uint8 in [0, 255]).

    Edge Cases & Robustness:
        - Mismatched shapes raise ValueError.
        - Identical images return exactly 0.0.
        - Pre-existing NaN or Inf values are sanitized to 0.0 to prevent silent propagation.

    Args:
        original: Reference (clean or ground-truth) image as NumPy array.
        degraded: Processed, noisy, or denoised image as NumPy array.

    Returns:
        float: Non-negative Mean Squared Error value.

    Raises:
        ValueError: If array dimensions do not match.
    """
    if original.shape != degraded.shape:
        raise ValueError(
            f"Shape mismatch: original shape {original.shape} does not match degraded shape {degraded.shape}"
        )

    # Cast safely to float64 to prevent numerical overflow during squaring
    orig_f = np.nan_to_num(original.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    deg_f = np.nan_to_num(degraded.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)

    diff = orig_f - deg_f
    mse_val = float(np.mean(diff ** 2))
    return max(0.0, mse_val)
