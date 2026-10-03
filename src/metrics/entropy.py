"""Shannon Information Entropy Image Quality Metric.

MATHEMATICAL DEFINITION:
-----------------------
Shannon entropy measures the information density, uncertainty, and complexity
of gray-level intensity distributions in an image:

    H = - sum_{k=1}^K p(r_k) * log2(p(r_k))

where:
    p(r_k) = n_k / N is the empirical probability of occurrence of intensity level r_k.
    K = number of discrete gray-level quantization bins (typically 256).

Units: bits per pixel.
Range: [0.0, log2(K)] (e.g., [0.0, 8.0] bits for 256 bins).

Interpretation in Mammography Denoising:
    - High entropy in corrupted images reflects unpredictable random noise fluctuations.
    - Excessively low entropy reflects loss of fine anatomical trabeculae and over-smoothing.
    - An optimal denoising method preserves structural tissue entropy while suppressing noise entropy.
"""
from typing import Optional
import numpy as np


def compute_entropy(
    img: np.ndarray,
    num_bins: int = 256,
) -> float:
    """Calculate Shannon information entropy (in bits) of a grayscale image.

    Input Assumptions:
        - img is a 2D or 3D NumPy array of real-valued pixel intensities.
        - Dynamic range can be float in [0.0, 1.0] or integers in [0, 255].

    Edge Cases & Robustness:
        - Constant/uniform image: returns exactly 0.0 bits.
        - Empty array: returns 0.0 bits.
        - Terms with p(r_k) == 0 are strictly excluded (0 * log2(0) -> 0).
        - Pre-existing NaNs or Infs are cleansed before histogram computation.

    Args:
        img: Input image as NumPy array.
        num_bins: Number of histogram bins for probability distribution estimation (default: 256).

    Returns:
        float: Non-negative Shannon entropy in bits per pixel.
    """
    work = np.nan_to_num(img.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    if work.size == 0:
        return 0.0

    min_val, max_val = float(np.min(work)), float(np.max(work))
    if min_val == max_val:
        # Constant image contains zero uncertainty/information
        return 0.0

    # Compute empirical histogram probability distribution
    counts, _ = np.histogram(work, bins=num_bins, range=(min_val, max_val))
    total_pixels = float(np.sum(counts))

    if total_pixels == 0.0:
        return 0.0

    probs = counts / total_pixels
    # Filter strictly positive probabilities (lim_{p->0} p * log2(p) = 0)
    non_zero_probs = probs[probs > 0.0]

    entropy_val = float(-np.sum(non_zero_probs * np.log2(non_zero_probs)))
    return max(0.0, entropy_val)
