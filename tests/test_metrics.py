"""Comprehensive unit test suite for Step 10 image-quality metrics."""
import numpy as np
import pytest

from src.metrics.mse import compute_mse
from src.metrics.psnr import compute_psnr
from src.metrics.ssim import compute_ssim
from src.metrics.snr import compute_snr
from src.metrics.cnr import compute_cnr, extract_roi_and_background_pixels
from src.metrics.cii import compute_cii, compute_contrast
from src.metrics.entropy import compute_entropy


@pytest.fixture
def synthetic_pair():
    """Create a pair of 64x64 synthetic float32 images."""
    np.random.seed(42)
    clean = np.linspace(0.1, 0.9, 64 * 64, dtype=np.float32).reshape((64, 64))
    noise = np.random.normal(0, 0.05, (64, 64)).astype(np.float32)
    degraded = np.clip(clean + noise, 0.0, 1.0)
    return clean, degraded


def test_mse_metric(synthetic_pair):
    clean, degraded = synthetic_pair
    # Identical
    assert compute_mse(clean, clean) == 0.0
    # Degraded
    mse_val = compute_mse(clean, degraded)
    assert mse_val > 0.0
    # Shape mismatch error
    with pytest.raises(ValueError):
        compute_mse(clean, clean[:32, :32])
    # NaN robustness
    corrupt = clean.copy()
    corrupt[0, 0] = np.nan
    assert not np.isnan(compute_mse(clean, corrupt))


def test_psnr_metric(synthetic_pair):
    clean, degraded = synthetic_pair
    # Identical images have infinite PSNR
    assert compute_psnr(clean, clean) == float("inf")
    # Degraded image has positive finite PSNR
    psnr_val = compute_psnr(clean, degraded, data_range=1.0)
    assert 15.0 < psnr_val < 50.0
    # Inferred data_range
    assert np.isclose(psnr_val, compute_psnr(clean, degraded, data_range=None))


def test_ssim_metric(synthetic_pair):
    clean, degraded = synthetic_pair
    # Identical images
    assert np.isclose(compute_ssim(clean, clean), 1.0)
    # Degraded image has SSIM in [0, 1)
    ssim_val = compute_ssim(clean, degraded)
    assert 0.0 < ssim_val < 1.0
    # Small patch dynamic window test (e.g. 5x5)
    small_clean = clean[:5, :5]
    small_deg = degraded[:5, :5]
    assert compute_ssim(small_clean, small_clean) == 1.0


def test_snr_metric(synthetic_pair):
    clean, degraded = synthetic_pair
    # Identical images
    assert compute_snr(clean, clean) == float("inf")
    # Degraded image SNR
    snr_val = compute_snr(clean, degraded)
    assert snr_val > 0.0
    # All zero image
    zeros = np.zeros((64, 64), dtype=np.float32)
    assert compute_snr(zeros, degraded) == 0.0


def test_cnr_metric():
    # Direct pixel vectors
    roi_bright = np.array([0.8, 0.82, 0.79, 0.81], dtype=np.float32)
    bg_dark = np.array([0.2, 0.21, 0.19, 0.22], dtype=np.float32)

    cnr_val = compute_cnr(roi_bright, bg_dark)
    assert cnr_val > 0.0

    # Rose criterion definition
    cnr_rose = compute_cnr(roi_bright, bg_dark, definition="rose")
    assert cnr_rose > 0.0

    # Zero contrast / identical regions
    assert compute_cnr(roi_bright, roi_bright) == 0.0

    # Flat constant regions with zero noise
    flat_roi = np.ones(5, dtype=np.float32) * 0.8
    flat_bg = np.ones(5, dtype=np.float32) * 0.2
    assert compute_cnr(flat_roi, flat_bg) == 0.0


def test_cnr_roi_mask_extraction():
    # 64x64 synthetic mammogram with a lesion and surrounding parenchyma
    image = np.ones((64, 64), dtype=np.float32) * 0.3  # Background parenchyma
    image[0:10, :] = 0.0  # Exterior black scanner air (< 0.02)

    # 10x10 lesion in center
    roi_mask = np.zeros((64, 64), dtype=np.uint8)
    roi_mask[25:35, 25:35] = 1
    image[25:35, 25:35] = 0.8  # Dense mass

    roi_pixels, bg_pixels = extract_roi_and_background_pixels(
        image, roi_mask, dilation_radius=5, air_threshold=0.02
    )

    assert len(roi_pixels) == 100
    assert len(bg_pixels) > 0
    # Confirm exterior air was NOT included in background
    assert np.all(bg_pixels >= 0.02)
    assert np.isclose(np.mean(roi_pixels), 0.8)
    assert np.isclose(np.mean(bg_pixels), 0.3)


def test_cii_metric():
    # Case 1: Contrast improved (CII > 1.0)
    orig_roi = np.array([0.5], dtype=np.float32)
    orig_bg = np.array([0.4], dtype=np.float32)
    enh_roi = np.array([0.7], dtype=np.float32)
    enh_bg = np.array([0.3], dtype=np.float32)

    cii_val = compute_cii(enh_roi, enh_bg, orig_roi, orig_bg)
    assert cii_val > 1.0

    # Case 2: Contrast unchanged (CII == 1.0)
    assert np.isclose(compute_cii(orig_roi, orig_bg, orig_roi, orig_bg), 1.0)

    # Case 3: Contrast degraded (CII < 1.0)
    cii_degraded = compute_cii(orig_roi, orig_bg, enh_roi, enh_bg)
    assert cii_degraded < 1.0

    # Weber method
    cii_weber = compute_cii(enh_roi, enh_bg, orig_roi, orig_bg, method="weber")
    assert cii_weber > 1.0


def test_entropy_metric():
    # Constant image has 0 entropy
    constant_img = np.ones((50, 50), dtype=np.float32) * 0.5
    assert compute_entropy(constant_img) == 0.0

    # Perfectly balanced binary image has 1.0 bit entropy
    binary_img = np.zeros((100, 100), dtype=np.float32)
    binary_img[:50, :] = 1.0
    entropy_binary = compute_entropy(binary_img, num_bins=2)
    assert np.isclose(entropy_binary, 1.0, atol=1e-3)

    # Random noise has high entropy
    np.random.seed(42)
    rand_img = np.random.uniform(0.0, 1.0, (100, 100)).astype(np.float32)
    rand_entropy = compute_entropy(rand_img)
    assert rand_entropy > 6.0
