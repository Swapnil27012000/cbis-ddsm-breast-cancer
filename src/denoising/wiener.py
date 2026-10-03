"""Adaptive 2D Wiener Filter for Additive Noise Reduction.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: CPU based
Implementation: SciPy scipy.signal.wiener (adaptive minimum mean square error filtering).
GPU Acceleration: Not natively provided in SciPy; runs on CPU.
"""
from typing import Tuple, Optional, Union
import numpy as np
from scipy.signal import wiener


def denoise_wiener(
    img: np.ndarray,
    mysize: Union[int, Tuple[int, int]] = (5, 5),
    noise: Optional[float] = None,
) -> np.ndarray:
    """Apply adaptive 2D Wiener filter for optimal linear estimation in stationary additive noise.

    Formula:
        Output = LocalMean + (LocalVar - NoiseVar) / LocalVar * (Input - LocalMean)

    Computation Device: CPU based (SciPy).

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        mysize: Size of local neighborhood window as int or tuple (wx, wy).
        noise: Noise power/variance. If None, automatically estimated by SciPy.

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].
    """
    # Sanitize invalid values and create copy
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    if isinstance(mysize, int):
        window = (mysize, mysize)
    else:
        window = (int(mysize[0]), int(mysize[1]))

    # Execute adaptive Wiener filter on CPU
    noise_param = float(noise) if noise is not None else None
    filtered = wiener(work, mysize=window, noise=noise_param)

    # Sanitize and clip output
    clean_out = np.nan_to_num(filtered, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(clean_out, 0.0, 1.0).astype(np.float32)
