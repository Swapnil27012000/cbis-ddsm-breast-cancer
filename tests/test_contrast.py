"""Unit tests for controlled contrast stretching."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.preprocessing.contrast import stretch_contrast, run_contrast_stretching

def test_stretch_contrast_basic():
    img = np.linspace(0.1, 0.9, 100, dtype=np.float32).reshape((10, 10))
    stretched = stretch_contrast(img, lower_percentile=5.0, upper_percentile=95.0)
    assert stretched.dtype == np.float32
    assert stretched.min() >= 0.0
    assert stretched.max() <= 1.0
    assert not np.isnan(stretched).any()
    assert not np.isinf(stretched).any()

def test_stretch_contrast_constant_image():
    const_img = np.full((10, 10), 0.5, dtype=np.float32)
    stretched = stretch_contrast(const_img, lower_percentile=2.0, upper_percentile=98.0)
    assert stretched.shape == const_img.shape
    assert not np.isnan(stretched).any()
    assert not np.isinf(stretched).any()

def test_stretch_contrast_handles_nan_and_inf():
    dirty_img = np.array([[0.2, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    stretched = stretch_contrast(dirty_img)
    assert not np.isnan(stretched).any()
    assert not np.isinf(stretched).any()
    assert stretched.dtype == np.float32

def test_run_contrast_stretching_pipeline(tmp_path):
    # Setup mock normalized image
    norm_dir = tmp_path / "normalized"
    norm_dir.mkdir()
    norm_path = norm_dir / "sample_norm.png"
    sample_arr = np.linspace(20, 200, 64 * 64, dtype=np.uint8).reshape((64, 64))
    cv2.imwrite(str(norm_path), sample_arr)

    # Setup mock normalized metadata CSV
    meta_path = tmp_path / "mock_norm_meta.csv"
    mock_df = pd.DataFrame([{
        "patient_id": "P_00044",
        "abnormality_category": "mass",
        "breast_side": "RIGHT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "label": 0,
        "dataset_split": "train",
        "original_image_path": "/path/to/raw.jpg",
        "processed_image_path": str(norm_path),
    }])
    mock_df.to_csv(str(meta_path), index=False)

    out_dir = tmp_path / "contrast"
    out_meta = tmp_path / "contrast_image_metadata.csv"

    res_df = run_contrast_stretching(
        input_metadata_csv=str(meta_path),
        output_dir=str(out_dir),
        output_metadata_csv=str(out_meta),
        override_test_mode=True,
        override_test_count=1,
        override_lower_p=2.0,
        override_upper_p=98.0,
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
    ]
    for col in required_cols:
        assert col in res_df.columns

    # Verify original normalized file was untouched
    reloaded_norm = cv2.imread(str(norm_path), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(sample_arr, reloaded_norm)

    # Verify saved contrast image exists and is valid
    saved_contrast_path = res_df.iloc[0]["contrast_image_path"]
    assert os.path.exists(saved_contrast_path)
    loaded_contrast = cv2.imread(saved_contrast_path, cv2.IMREAD_GRAYSCALE)
    assert loaded_contrast.shape == (64, 64)
