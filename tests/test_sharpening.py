"""Unit tests for Step 6 controlled image sharpening."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.preprocessing.sharpening import unsharp_mask, run_image_sharpening

def test_unsharp_mask_basic():
    img = np.linspace(0.1, 0.9, 100, dtype=np.float32).reshape((10, 10))
    sharpened = unsharp_mask(img, amount=1.0, sigma=1.0, kernel_size=(5, 5))
    assert sharpened.dtype == np.float32
    assert sharpened.min() >= 0.0
    assert sharpened.max() <= 1.0
    assert not np.isnan(sharpened).any()
    assert not np.isinf(sharpened).any()

def test_unsharp_mask_clips_extreme_values():
    img = np.ones((20, 20), dtype=np.float32) * 0.9
    img[10, 10] = 1.0
    sharpened = unsharp_mask(img, amount=2.0, sigma=1.0, kernel_size=(5, 5))
    assert sharpened.max() <= 1.0
    assert sharpened.min() >= 0.0

def test_run_image_sharpening_pipeline(tmp_path):
    # Setup mock contrast image
    contrast_dir = tmp_path / "contrast"
    contrast_dir.mkdir()
    contrast_path = contrast_dir / "sample_contrast.png"
    sample_arr = np.linspace(10, 240, 64 * 64, dtype=np.uint8).reshape((64, 64))
    cv2.imwrite(str(contrast_path), sample_arr)

    # Setup mock contrast metadata CSV
    meta_path = tmp_path / "mock_contrast_meta.csv"
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
        "contrast_image_path": str(contrast_path),
    }])
    mock_df.to_csv(str(meta_path), index=False)

    out_dir = tmp_path / "sharpened"
    out_meta = tmp_path / "sharpened_image_metadata.csv"

    res_df = run_image_sharpening(
        input_metadata_csv=str(meta_path),
        output_dir=str(out_dir),
        output_metadata_csv=str(out_meta),
        override_test_mode=True,
        override_test_count=1,
        override_amount=1.0,
        override_sigma=1.0,
    )

    assert len(res_df) == 1
    assert os.path.exists(str(out_meta))

    required_cols = [
        "patient_id",
        "abnormality_category",
        "breast_side",
        "image_view",
        "pathology",
        "label",
        "dataset_split",
        "original_image_path",
        "normalized_image_path",
        "contrast_image_path",
        "sharpened_image_path",
    ]
    for col in required_cols:
        assert col in res_df.columns

    # Verify input contrast file was not modified
    reloaded_contrast = cv2.imread(str(contrast_path), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(sample_arr, reloaded_contrast)

    # Verify saved sharpened file exists and has correct dimensions
    saved_sharp_path = res_df.iloc[0]["sharpened_image_path"]
    assert os.path.exists(saved_sharp_path)
    loaded_sharp = cv2.imread(saved_sharp_path, cv2.IMREAD_GRAYSCALE)
    assert loaded_sharp.shape == (64, 64)
