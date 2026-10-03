"""Non-Local Means (NLM) Patch-Based Filter for Mammography.

COMPUTATION DEVICE CLASSIFICATION:
---------------------------------
Architecture: CPU based
Implementation: OpenCV cv2.fastNlMeansDenoising (multi-threaded CPU execution).
GPU Acceleration: Not natively available in OpenCV standard build without CUDA module.
"""
import cv2
import numpy as np


def denoise_nlm(
    img: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """Apply Non-Local Means (NLM) patch-based filtering for rich texture and micro-calcification retention.

    Computes weighted average of non-local neighborhood patches based on Gaussian patch similarity.

    Computation Device: CPU based (OpenCV fastNlMeansDenoising).

    Args:
        img: Input image as float32 NumPy array with values in [0.0, 1.0].
        h: Filter strength parameter controlling luminance smoothing. Higher h = stronger blur.
        template_window_size: Size of template patch in pixels (must be odd, e.g. 7).
        search_window_size: Size of area where patch searches occur (must be odd, e.g. 21).

    Returns:
        np.ndarray: Denoised float32 image with values strictly clipped to [0.0, 1.0].
    """
    # Sanitize invalid values and create copy
    work = np.nan_to_num(img.copy().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    work = np.clip(work, 0.0, 1.0)

    # OpenCV fastNlMeans operates efficiently on 8-bit unsigned images
    img_8u = np.clip(work * 255.0, 0.0, 255.0).round().astype(np.uint8)

    # In config, h might be given as 0.1 (for normalized float) or 10.0 (for uint8). Convert safely:
    h_param = float(h) * 100.0 if float(h) <= 1.0 else float(h)

    denoised_8u = cv2.fastNlMeansDenoising(
        img_8u,
        None,
        h=h_param,
        templateWindowSize=int(template_window_size),
        searchWindowSize=int(search_window_size),
    )

    out = denoised_8u.astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)
