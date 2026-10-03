"""Comprehensive unit and integration tests for Step 9 denoising filters."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2
import torch

from src.denoising.median import denoise_median
from src.denoising.gaussian import denoise_gaussian
from src.denoising.wiener import denoise_wiener
from src.denoising.bilateral import denoise_bilateral
from src.denoising.non_local_means import denoise_nlm
from src.denoising.anscombe_wiener import denoise_anscombe_wiener
from src.denoising.adaptive_median import denoise_adaptive_median
from src.denoising.kuan import denoise_kuan
from src.denoising.run_denoising import run_denoising_pipeline


@pytest.fixture
def sample_image():
    """Create a 48x48 synthetic float32 image with known features and bounds."""
    np.random.seed(42)
    base = np.linspace(0.1, 0.9, 48 * 48, dtype=np.float32).reshape((48, 48))
    # Add small amount of noise
    noise = np.random.normal(0, 0.05, (48, 48)).astype(np.float32)
    return np.clip(base + noise, 0.0, 1.0)


def test_denoise_median(sample_image):
    out = denoise_median(sample_image, kernel_size=5)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_gaussian(sample_image):
    out = denoise_gaussian(sample_image, kernel_size=(5, 5), sigma=1.0)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_wiener(sample_image):
    out = denoise_wiener(sample_image, mysize=(5, 5))
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_bilateral(sample_image):
    out = denoise_bilateral(sample_image, d=5, sigma_color=50.0, sigma_space=50.0)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_nlm(sample_image):
    out = denoise_nlm(sample_image, h=0.1, template_window_size=5, search_window_size=11)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_anscombe_wiener(sample_image):
    out = denoise_anscombe_wiener(sample_image, scale=255.0, mysize=(5, 5))
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_adaptive_median(sample_image):
    out = denoise_adaptive_median(sample_image, max_window_size=7)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_denoise_kuan(sample_image):
    out = denoise_kuan(sample_image, window_size=5, noise_var=0.04)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_nan_and_immutability(sample_image):
    corrupted = sample_image.copy()
    corrupted[5, 5] = np.nan
    corrupted[10, 10] = np.inf
    backup = corrupted.copy()

    # Test all 8 filters handle NaN/Inf without raising and preserve input array
    filters = [
        lambda img: denoise_median(img, 3),
        lambda img: denoise_gaussian(img, 3, 1.0),
        lambda img: denoise_wiener(img, (3, 3)),
        lambda img: denoise_bilateral(img, 3, 25.0, 25.0),
        lambda img: denoise_nlm(img, 0.1, 3, 7),
        lambda img: denoise_anscombe_wiener(img, 255.0, (3, 3)),
        lambda img: denoise_adaptive_median(img, 5),
        lambda img: denoise_kuan(img, 5, 0.04),
    ]

    for fn in filters:
        res = fn(corrupted)
        assert not np.isnan(res).any(), "Result contains NaN"
        assert not np.isinf(res).any(), "Result contains Inf"
        assert res.min() >= 0.0 and res.max() <= 1.0
        # Verify original was not mutated
        assert np.array_equal(np.isnan(corrupted), np.isnan(backup))


def test_run_denoising_pipeline(tmp_path):
    # Setup mock noisy image
    noisy_dir = tmp_path / "data" / "noisy" / "gaussian"
    noisy_dir.mkdir(parents=True)
    noisy_img_path = noisy_dir / "sample_img.png"
    sample_uint8 = np.linspace(30, 220, 32 * 32, dtype=np.uint8).reshape((32, 32))
    cv2.imwrite(str(noisy_img_path), sample_uint8)

    # Setup mock noisy CSV
    noisy_meta_path = tmp_path / "mock_noisy_metadata.csv"
    mock_df = pd.DataFrame([{
        "noise_type": "gaussian",
        "noise_parameters": '{"mean": 0.0, "var": 0.01}',
        "source_image": "/path/to/clean.png",
        "output_image": str(noisy_img_path),
        "patient_id": "P_00044",
        "pathology": "BENIGN",
        "label": 0,
        "is_synthetic": True,
    }])
    mock_df.to_csv(str(noisy_meta_path), index=False)

    out_denoised_dir = tmp_path / "denoised"
    out_meta_path = tmp_path / "denoised_image_metadata.csv"

    res_df = run_denoising_pipeline(
        input_noisy_csv=str(noisy_meta_path),
        output_base_dir=str(out_denoised_dir),
        output_metadata_csv=str(out_meta_path),
        noise_type="gaussian",
        max_images=1,
        use_gpu=False,
    )

    # 1 image x 8 methods = 8 records
    assert len(res_df) == 8
    assert os.path.exists(str(out_meta_path))

    expected_cols = [
        "denoising_method",
        "hardware_architecture",
        "noise_type",
        "source_noisy_image",
        "output_denoised_image",
        "patient_id",
        "pathology",
        "label",
        "status",
    ]
    for c in expected_cols:
        assert c in res_df.columns

    # Verify that all 8 method subfolders contain the output image
    for m in [
        "median", "gaussian", "wiener", "bilateral",
        "non_local_means", "anscombe_wiener", "adaptive_median", "kuan"
    ]:
        m_folder = out_denoised_dir / m
        assert m_folder.exists()
        assert len(list(m_folder.glob("*.png"))) == 1
