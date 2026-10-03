"""Unit tests for Step 7 artificial noise simulation models."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2
import torch

from src.noise.gaussian import add_gaussian_noise
from src.noise.salt_pepper import add_salt_pepper_noise
from src.noise.speckle import add_speckle_noise
from src.noise.poisson import add_poisson_noise
from src.noise.mixed_poisson_gaussian import add_mixed_poisson_gaussian_noise
from src.noise.generate_noise import run_noise_generation

@pytest.fixture
def clean_image():
    """Create a 32x32 clean float32 test mammogram patch."""
    return np.linspace(0.1, 0.8, 32 * 32, dtype=np.float32).reshape((32, 32))

def test_gaussian_noise(clean_image):
    noisy = add_gaussian_noise(clean_image, mean=0.0, var=0.01, seed=42)
    assert noisy.shape == clean_image.shape
    assert noisy.dtype == np.float32
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert not np.array_equal(clean_image, noisy)

def test_salt_pepper_noise(clean_image):
    noisy = add_salt_pepper_noise(clean_image, amount=0.1, salt_vs_pepper=0.5, seed=42)
    assert noisy.shape == clean_image.shape
    assert (noisy == 1.0).any()
    assert (noisy == 0.0).any()
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0

def test_speckle_noise(clean_image):
    noisy = add_speckle_noise(clean_image, var=0.04, seed=42)
    assert noisy.shape == clean_image.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0

def test_poisson_noise(clean_image):
    noisy = add_poisson_noise(clean_image, scale=255.0, seed=42)
    assert noisy.shape == clean_image.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0

def test_mixed_poisson_gaussian_noise(clean_image):
    noisy = add_mixed_poisson_gaussian_noise(clean_image, poisson_scale=255.0, gaussian_var=0.005, seed=42)
    assert noisy.shape == clean_image.shape
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0

def test_run_noise_generation_pipeline(tmp_path):
    # Setup clean sharpened image
    sharp_dir = tmp_path / "sharpened"
    sharp_dir.mkdir()
    sharp_path = sharp_dir / "sample_sharp.png"
    sample_arr = np.linspace(20, 230, 32 * 32, dtype=np.uint8).reshape((32, 32))
    cv2.imwrite(str(sharp_path), sample_arr)

    # Setup mock sharpened metadata CSV
    meta_path = tmp_path / "mock_sharp_meta.csv"
    mock_df = pd.DataFrame([{
        "patient_id": "P_00044",
        "abnormality_category": "mass",
        "breast_side": "RIGHT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "label": 0,
        "dataset_split": "train",
        "original_image_path": "/path/to/raw.jpg",
        "normalized_image_path": "/path/to/norm.png",
        "contrast_image_path": "/path/to/contrast.png",
        "sharpened_image_path": str(sharp_path),
    }])
    mock_df.to_csv(str(meta_path), index=False)

    out_base = tmp_path / "noisy"
    out_meta = tmp_path / "noisy_image_metadata.csv"

    res_df = run_noise_generation(
        input_metadata_csv=str(meta_path),
        output_base_dir=str(out_base),
        output_metadata_csv=str(out_meta),
        override_test_mode=True,
        override_test_count=1,
        use_gpu=False # CPU for unit test runner
    )

    # 1 image * 5 noise models = 5 rows
    assert len(res_df) == 5
    assert os.path.exists(str(out_meta))

    expected_cols = [
        "noise_type",
        "noise_parameters",
        "source_image",
        "output_image",
        "patient_id",
        "pathology",
        "label",
    ]
    for c in expected_cols:
        assert c in res_df.columns

    # Verify subdirectories
    for n_type in ["gaussian", "salt_pepper", "speckle", "poisson", "mixed_poisson_gaussian"]:
        n_folder = out_base / n_type
        assert n_folder.exists()
        assert len(list(n_folder.glob("*.png"))) == 1

    # Verify original clean file was not modified
    reloaded_clean = cv2.imread(str(sharp_path), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(sample_arr, reloaded_clean)
