"""Unit tests for preprocessing validation and visualization."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.preprocessing.visualize import validate_and_visualize_samples

def test_validate_and_visualize_samples(tmp_path):
    # Create mock original and processed images
    img_orig = np.random.randint(10, 240, (150, 100), dtype=np.uint8)
    img_proc = np.random.randint(10, 240, (64, 64), dtype=np.uint8)

    orig_path = tmp_path / "orig.png"
    proc_path = tmp_path / "proc.png"
    cv2.imwrite(str(orig_path), img_orig)
    cv2.imwrite(str(proc_path), img_proc)

    # Create mock metadata CSV
    meta_path = tmp_path / "normalized_meta.csv"
    mock_df = pd.DataFrame([{
        "patient_id": "P_00001",
        "abnormality_category": "mass",
        "breast_side": "LEFT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "label": 0,
        "dataset_split": "train",
        "original_image_path": str(orig_path),
        "processed_image_path": str(proc_path),
    }])
    mock_df.to_csv(str(meta_path), index=False)

    plot_path = tmp_path / "validation_plot.png"
    per_sample_dir = tmp_path / "per_sample"

    results = validate_and_visualize_samples(
        metadata_csv=str(meta_path),
        num_samples=1,
        save_path=str(plot_path),
        per_sample_dir=str(per_sample_dir)
    )

    assert len(results) == 1
    sample = results[0]
    assert sample["patient_id"] == "P_00001"
    assert sample["status"] == "PASSED"
    assert sample["orig_shape"] == (150, 100)
    assert sample["proc_shape"] == (64, 64)
    assert os.path.exists(str(plot_path))
    assert os.path.getsize(str(plot_path)) > 0
