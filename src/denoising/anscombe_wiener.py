"""Generalized Anscombe Variance-Stabilizing Transform + Wiener Filter.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: CPU based
Implementation: NumPy variance stabilization combined with SciPy adaptive Wiener filter.
GPU Acceleration: Not currently GPU accelerated. Runs on CPU.
"""
from typing import Tuple, Union
import numpy as np
from .wiener import denoise_wiener


def anscombe_transform(x: np.ndarray) -> np.ndarray:
    """Forward Anscombe variance-stabilizing transform: 2 * sqrt(max(x, 0) + 3/8).

    Transforms Poisson-distributed count noise into approximately Gaussian noise with unit variance.
    """
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + (3.0 / 8.0))


def inverse_anscombe_transform(y: np.ndarray) -> np.ndarray:
    """Asymptotically unbiased inverse Anscombe transform: (y / 2)^2 - 3/8."""
    return np.maximum((y / 2.0) ** 2 - (3.0 / 8.0), 0.0)


def denoise_anscombe_wiener(
    img: np.ndarray,
    scale: float = 255.0,
    mysize: Union[int, Tuple[int, int]] = (5, 5),
    sigma: float = 1.0,
) -> np.ndarray:
    """Denoise Poisson and mixed quantum noise via Anscombe variance stabilization and adaptive Wiener filtering.

    Computation Device: CPU based.

    Pipeline:
        1. Scale image to photon count space: x = img * scale
        2. Apply forward Anscombe transform: y = 2 * sqrt(x + 3/8)
        3. Apply adaptive Wiener filter with noise variance = 1.0
        4. Invert Anscombe transform and normalize back to [0.0, 1.0]

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        scale: Scaling factor mapping [0, 1] intensities to photon count domain (default: 255.0).
        mysize: Wiener filter local neighborhood window size.
        sigma: Noise variance in stabilized space (default: 1.0 for standard Anscombe).

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].
    """
    # Sanitize invalid values and create copy
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    # 1. Scale to photon count domain
    count_domain = work * float(scale)

    # 2. Forward Anscombe stabilization
    stabilized = anscombe_transform(count_domain)

    # 3. Wiener filtering (noise variance in stabilized domain is ~1.0)
    stabilized_denoised = denoise_wiener(stabilized, mysize=mysize, noise=float(sigma))

    # 4. Inverse Anscombe transform
    restored = inverse_anscombe_transform(stabilized_denoised) / float(scale)

    clean_out = np.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(clean_out, 0.0, 1.0).astype(np.float32)
