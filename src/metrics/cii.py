"""Contrast Improvement Index (CII) Metric for Medical Mammography.

MATHEMATICAL DEFINITION:
-----------------------
Contrast Improvement Index (CII) quantifies the enhancement factor of lesion-to-background
contrast achieved by an image processing (denoising, contrast-stretching) operation:

    CII = C_processed / C_original

where C is the Michelson contrast (or Weber contrast):

    C = |mu_ROI - mu_BG| / (mu_ROI + mu_BG + eps)

Interpretation:
    CII > 1.0 : Effective contrast enhancement of the abnormality.
    CII = 1.0 : Contrast preserved unchanged.
    CII < 1.0 : Loss of contrast between lesion and surrounding parenchyma.

REGION SELECTION STRATEGY:
-------------------------
In strict adherence to clinical standards, regions are never arbitrarily selected:
- Foreground ROI: Extracted from verified CBIS-DDSM lesion ground-truth annotations (roi_mask > 0).
- Background: Extracted from the adjacent peri-tumoral fibroglandular parenchyma (excluding air),
  ensuring both C_original and C_processed evaluate identical spatial tissue regions.
"""
from typing import Optional
import numpy as np


def compute_contrast(
    roi: np.ndarray,
    background: np.ndarray,
    method: str = "michelson",
    eps: float = 1e-7,
) -> float:
    """Compute radiometric contrast between ROI and background pixels.

    Args:
        roi: 1D array of ROI pixel intensities.
        background: 1D array of background pixel intensities.
        method: 'michelson' (|mu_roi - mu_bg| / (mu_roi + mu_bg)) or 'weber' (|mu_roi - mu_bg| / mu_bg).
        eps: Numerical guard to prevent division by zero.

    Returns:
        float: Non-negative contrast value.
    """
    roi_f = np.nan_to_num(roi.astype(np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0)
    bg_f = np.nan_to_num(background.astype(np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0)

    if roi_f.size == 0 or bg_f.size == 0:
        return 0.0

    mu_roi = float(np.mean(roi_f))
    mu_bg = float(np.mean(bg_f))

    diff = abs(mu_roi - mu_bg)

    if method.lower() == "weber":
        denom = mu_bg + eps
    else:  # Michelson
        denom = mu_roi + mu_bg + eps

    return float(diff / denom)


def compute_cii(
    enhanced_roi: np.ndarray,
    enhanced_bg: np.ndarray,
    orig_roi: np.ndarray,
    orig_bg: np.ndarray,
    method: str = "michelson",
    eps: float = 1e-7,
) -> float:
    """Compute Contrast Improvement Index (CII) between original and processed mammograms.

    Input Assumptions:
        - enhanced_roi and orig_roi sample the identical physical lesion area.
        - enhanced_bg and orig_bg sample the identical physical parenchymal background.
        - Pixel values are real-valued float or int.

    Edge Cases & Robustness:
        - If original contrast is zero: returns 1.0 if processed is also zero; else float ratio.
        - Division by zero guarded by numerical epsilon.
        - Returns a non-negative float.

    Args:
        enhanced_roi: Pixel intensities of ROI in the processed/denoised image.
        enhanced_bg: Pixel intensities of background in the processed/denoised image.
        orig_roi: Pixel intensities of ROI in the original/reference image.
        orig_bg: Pixel intensities of background in the original/reference image.
        method: Contrast formula ('michelson' or 'weber').
        eps: Small constant avoiding division by zero.

    Returns:
        float: Ratio C_enhanced / C_original.
    """
    c_orig = compute_contrast(orig_roi, orig_bg, method=method, eps=eps)
    c_enh = compute_contrast(enhanced_roi, enhanced_bg, method=method, eps=eps)

    if c_orig <= eps:
        if c_enh <= eps:
            return 1.0
        return float(c_enh / eps)

    cii_val = float(c_enh / c_orig)
    return max(0.0, cii_val)
