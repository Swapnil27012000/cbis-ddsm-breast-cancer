"""Contrast-to-Noise Ratio (CNR) Metric for Medical Mammography.

MATHEMATICAL DEFINITION:
-----------------------
Contrast-to-Noise Ratio (CNR) evaluates the perceptibility and detectability of a lesion
(mass or microcalcification) relative to the surrounding healthy fibroglandular parenchyma:

    CNR = |mu_ROI - mu_BG| / sqrt(sigma_ROI^2 + sigma_BG^2)

Alternative Rose-criterion formulation:
    CNR_rose = |mu_ROI - mu_BG| / sigma_BG

where:
    mu_ROI, sigma_ROI: Mean and standard deviation of pixel intensities within the lesion ROI.
    mu_BG,  sigma_BG : Mean and standard deviation of pixel intensities within the reference background.

REGION SELECTION STRATEGY:
-------------------------
To ensure rigorous clinical validity, regions are NEVER arbitrarily selected:
1. Foreground ROI:
   - Defined strictly by the radiologist-annotated CBIS-DDSM ground-truth lesion mask (roi_mask > 0).
2. Background Region:
   - If explicit background_mask is provided, those pixels are used directly.
   - If background_mask is None:
     A deterministic peri-tumoral annular zone (dilation ring around the lesion) is extracted:
         Background = (Dilate(ROI_mask, radius) - ROI_mask) INTERSECT Breast_Parenchyma
     Crucially, non-breast empty air (intensity < air_threshold) is strictly excluded so that
     CNR reflects true tissue contrast rather than detector background margin.
3. Vector Direct Mode:
   - Alternatively, 1D NumPy arrays of pre-extracted ROI and background pixels can be passed directly.
"""
from typing import Optional, Union, Tuple
import cv2
import numpy as np


def extract_roi_and_background_pixels(
    image: np.ndarray,
    roi_mask: np.ndarray,
    background_mask: Optional[np.ndarray] = None,
    dilation_radius: int = 15,
    air_threshold: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract foreground lesion pixels and local surrounding parenchymal background pixels.

    Args:
        image: 2D mammogram array (typically normalized float32 in [0, 1]).
        roi_mask: 2D binary mask of the abnormality (same shape as image).
        background_mask: Optional explicit 2D binary background mask.
        dilation_radius: Radius in pixels for morphological dilation ring if background_mask is None.
        air_threshold: Minimum pixel intensity to be considered breast parenchyma (excludes exterior air).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (roi_pixels_1d, background_pixels_1d).
    """
    if image.shape != roi_mask.shape:
        raise ValueError(f"Image shape {image.shape} does not match ROI mask shape {roi_mask.shape}")

    roi_bool = roi_mask > 0
    roi_pixels = image[roi_bool]

    if background_mask is not None:
        bg_bool = (background_mask > 0) & (~roi_bool)
        bg_pixels = image[bg_bool]
    else:
        # Construct deterministic peri-tumoral annular ring
        kernel_size = 2 * int(dilation_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(roi_mask.astype(np.uint8), kernel) > 0

        # Exclude ROI itself and exclude scanner air background
        parenchyma = image > float(air_threshold)
        bg_bool = dilated & (~roi_bool) & parenchyma
        bg_pixels = image[bg_bool]

        # Fallback if annular zone is completely empty
        if bg_pixels.size == 0:
            fallback_bg = (~roi_bool) & parenchyma
            bg_pixels = image[fallback_bg] if np.any(fallback_bg) else image[~roi_bool]

    return roi_pixels, bg_pixels


def compute_cnr(
    roi: np.ndarray,
    background: np.ndarray,
    definition: str = "standard",
) -> float:
    """Compute Contrast-to-Noise Ratio (CNR) between lesion ROI and background parenchyma.

    Input Assumptions:
        - roi and background are 1D arrays of pixel intensities or 2D image arrays.
        - Pixel values should be real-valued float or int.

    Edge Cases & Robustness:
        - If roi or background is empty, returns 0.0.
        - If total variance is zero (constant flat regions):
          returns 0.0 (if means are equal) or handles gracefully without division by zero.
        - Sanitizes pre-existing NaNs or Infs.

    Args:
        roi: 1D array of ROI pixel intensities, or pre-extracted ROI patch.
        background: 1D array of background pixel intensities.
        definition: 'standard' (ICRU/AAPM: sqrt(sigma_roi^2 + sigma_bg^2)) or 'rose' (sigma_bg).

    Returns:
        float: Non-negative Contrast-to-Noise Ratio value.
    """
    roi_f = np.nan_to_num(roi.astype(np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0)
    bg_f = np.nan_to_num(background.astype(np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0)

    if roi_f.size == 0 or bg_f.size == 0:
        return 0.0

    mu_roi = float(np.mean(roi_f))
    mu_bg = float(np.mean(bg_f))
    var_roi = float(np.var(roi_f))
    var_bg = float(np.var(bg_f))

    if definition.lower() == "rose":
        denom = np.sqrt(var_bg)
    else:
        denom = np.sqrt(var_roi + var_bg)

    if denom <= 1e-12:
        # Constant images: if means differ, infinite contrast/zero noise, but clinically meaningless -> 0.0
        return 0.0

    cnr_val = float(np.abs(mu_roi - mu_bg) / denom)
    return max(0.0, cnr_val)
