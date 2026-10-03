"""Signal-to-Noise Ratio (SNR) Image Quality Metric.

MATHEMATICAL DEFINITION:
-----------------------
SNR evaluates the ratio of the total reference signal power to corrupting noise power:

    SNR = 10 * log10( sum(I_ref^2) / sum((I_ref - I_deg)^2) )

Units: Decibels (dB).
Range: (-inf, +inf), where higher values indicate lower noise corruption.
"""
import numpy as np


def compute_snr(
    original: np.ndarray,
    degraded: np.ndarray,
) -> float:
    """Compute Signal-to-Noise Ratio (SNR in dB) between reference and evaluated images.

    Input Assumptions:
        - original and degraded must have identical dimensions.
        - Arrays are converted to float64 to ensure numerical precision.

    Edge Cases & Robustness:
        - Identical images (zero noise power): returns float('inf').
        - All-zero reference image (zero signal power): returns 0.0 dB.
        - Pre-existing NaNs or Infs are sanitized to 0.0.

    Args:
        original: Reference clean image as NumPy array.
        degraded: Processed/noisy/denoised image as NumPy array.

    Returns:
        float: SNR in decibels (dB), or float('inf') for identical images.
    """
    if original.shape != degraded.shape:
        raise ValueError(
            f"Shape mismatch: original shape {original.shape} does not match degraded shape {degraded.shape}"
        )

    orig_f = np.nan_to_num(original.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    deg_f = np.nan_to_num(degraded.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)

    signal_power = float(np.sum(orig_f ** 2))
    noise_power = float(np.sum((orig_f - deg_f) ** 2))

    if noise_power == 0.0:
        return float("inf")

    if signal_power == 0.0:
        return 0.0

    snr_val = float(10.0 * np.log10(signal_power / noise_power))
    return snr_val
