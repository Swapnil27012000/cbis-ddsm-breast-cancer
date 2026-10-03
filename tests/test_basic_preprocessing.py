"""Unit tests for basic CBIS-DDSM preprocessing pipeline."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.preprocessing.basic_preprocessing import run_basic_preprocessing

def test_basic_preprocessing_pipeline(tmp_path):
    # Create sample image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    sample_img_path = img_dir / "sample_mammogram.png"
    cv2.imwrite(str(sample_img_path), np.random.randint(20, 220, (120, 100), dtype=np.uint8))

    # Create mock master metadata CSV
    meta_path = tmp_path / "mock_master_metadata.csv"
    mock_df = pd.DataFrame([{
        "patient_id": "P_00001",
        "abnormality_category": "mass",
        "left_or_right_breast": "LEFT",
        "image_view": "CC",
        "pathology": "MALIGNANT",
        "label": 1,
        "dataset_split": "train",
        "resolved_full_mammogram_path": str(sample_img_path),
        "image_path": str(sample_img_path),
        "file_exists": True
    }])
    mock_df.to_csv(str(meta_path), index=False)

    out_dir = tmp_path / "normalized"
    out_meta = tmp_path / "normalized_image_metadata.csv"

    res_df = run_basic_preprocessing(
        metadata_csv=str(meta_path),
        output_dir=str(out_dir),
        output_metadata_csv=str(out_meta),
        override_test_mode=True,
        override_image_count=5,
        override_image_size=128
    )

    assert len(res_df) == 1
    assert os.path.exists(str(out_meta))
    
    # Verify required columns
    required_cols = [
        "patient_id",
        "abnormality_category",
        "breast_side",
        "image_view",
        "pathology",
        "label",
        "dataset_split",
        "original_image_path",
        "processed_image_path",
    ]
    for col in required_cols:
        assert col in res_df.columns

    # Verify saved image properties
    saved_path = res_df.iloc[0]["processed_image_path"]
    assert os.path.exists(saved_path)
    loaded = cv2.imread(saved_path, cv2.IMREAD_GRAYSCALE)
    assert loaded.shape == (128, 128)
