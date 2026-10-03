"""Two-Level Adaptive Median Filter for High-Density Impulse Noise.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: CPU based
Implementation: Vectorized multi-scale rank and neighborhood filtering via SciPy / OpenCV.
GPU Acceleration: Not currently GPU accelerated. Runs on CPU.
"""
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, median_filter


def denoise_adaptive_median(
    img: np.ndarray,
    max_window_size: int = 7,
) -> np.ndarray:
    """Apply two-level Adaptive Median Filter (AMF) with dynamic window expansion.

    Specifically removes high-density salt-and-pepper impulse noise while preserving
    fine geometric details and avoiding the blurring artifacts of fixed-window median filters.

    Algorithm (Gonzalez & Woods):
        Level A:
            A1 = z_med - z_min
            A2 = z_med - z_max
            If A1 > 0 and A2 < 0 -> Go to Level B
            Else -> Increase window size s += 2; if s <= max_window_size repeat Level A, else output z_med
        Level B:
            B1 = z_xy - z_min
            B2 = z_xy - z_max
            If B1 > 0 and B2 < 0 -> Output z_xy (clean pixel)
            Else -> Output z_med (impulse noise replaced by median)

    Computation Device: CPU based (Vectorized SciPy multi-window rank filters).

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        max_window_size: Maximum square filter window size (must be odd integer >= 3, default: 7).

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].
    """
    if max_window_size < 3 or max_window_size % 2 == 0:
        raise ValueError(f"max_window_size must be an odd integer >= 3, got: {max_window_size}")

    # Sanitize invalid values and create copy
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    # Initialize output array and unvisited pixel mask
    output = work.copy()
    unprocessed = np.ones(work.shape, dtype=bool)

    # Multi-scale window progression: 3, 5, ..., max_window_size
    s = 3
    while s <= max_window_size and np.any(unprocessed):
        # Compute local minimum, maximum, and median across window s
        z_min = minimum_filter(work, size=s, mode="reflect")
        z_max = maximum_filter(work, size=s, mode="reflect")
        z_med = median_filter(work, size=s, mode="reflect")

        # Level A condition: is median non-impulse?
        level_a = (z_med > z_min) & (z_med < z_max) & unprocessed

        # Level B condition on pixels where Level A passed
        level_b_clean = (work > z_min) & (work < z_max) & level_a

        # 1. Non-impulse pixels: retain original z_xy
        output[level_b_clean] = work[level_b_clean]
        unprocessed[level_b_clean] = False

        # 2. Impulse pixels with valid median: replace with z_med
        level_b_noise = level_a & (~level_b_clean)
        output[level_b_noise] = z_med[level_b_noise]
        unprocessed[level_b_noise] = False

        # Advance window size for pixels where Level A failed
        s += 2

    # For any remaining pixels that reached max_window_size without clean median, assign max-window median
    if np.any(unprocessed):
        z_med_max = median_filter(work, size=max_window_size, mode="reflect")
        output[unprocessed] = z_med_max[unprocessed]

    return np.clip(output, 0.0, 1.0).astype(np.float32)
