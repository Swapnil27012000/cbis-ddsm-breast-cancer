"""Unit and integration tests for full-dataset denoising experiment runner."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.experiments.denoising_experiment import (
    run_denoising_experiment,
    extract_mammogram_regions,
    estimate_resources,
)


def test_extract_mammogram_regions():
    # 48x48 test image with parenchyma and simulated dense lesion
    img = np.zeros((48, 48), dtype=np.float32)
    img[10:40, 10:40] = 0.2  # Breast parenchyma
    img[20:30, 20:30] = 0.8  # Dense lesion

    roi_px, bg_px = extract_mammogram_regions(img, air_threshold=0.02)
    assert len(roi_px) > 0
    assert len(bg_px) > 0
    assert np.mean(roi_px) > np.mean(bg_px)


def test_estimate_resources():
    # 10 images x 5 noise models x 8 filters
    est = estimate_resources(num_images=10, num_noise_types=5, num_denoising_methods=8)
    assert est["num_images"] == 10
    assert est["num_noise_types"] == 5
    assert est["num_denoising_methods"] == 8
    assert est["total_experiments"] == 400
    assert est["noise_operations"] == 50
    assert est["denoising_operations"] == 400
    assert est["metric_evaluations"] == 2800
    assert est["total_operations"] == 3250
    assert est["total_saved_images"] == 450
    assert est["estimated_disk_mb"] > 0
    assert est["estimated_disk_gb"] > 0


def test_confirmation_guard(tmp_path):
    # Setup mock clean sharpened image
    sharp_dir = tmp_path / "data" / "sharpened"
    sharp_dir.mkdir(parents=True)
    sharp_img_path = sharp_dir / "case_001.png"

    clean_img = np.linspace(30, 220, 48 * 48, dtype=np.uint8).reshape((48, 48))
    cv2.imwrite(str(sharp_img_path), clean_img)

    sharp_csv = tmp_path / "mock_sharpened_meta.csv"
    pd.DataFrame([{
        "patient_id": "P_00044",
        "abnormality_category": "mass",
        "breast_side": "RIGHT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "label": 0,
        "dataset_split": "train",
        "original_image_path": "/path/to/orig.jpg",
        "normalized_image_path": "/path/to/norm.png",
        "contrast_image_path": "/path/to/contrast.png",
        "sharpened_image_path": str(sharp_img_path),
    }]).to_csv(str(sharp_csv), index=False)

    out_metrics_csv = tmp_path / "results" / "denoising_metrics.csv"
    out_denoised_dir = tmp_path / "data" / "denoised"
    out_noisy_dir = tmp_path / "data" / "noisy"

    # Full mode without confirm must return None and not write outputs
    res = run_denoising_experiment(
        input_metadata_csv=str(sharp_csv),
        output_metrics_csv=str(out_metrics_csv),
        denoised_base_dir=str(out_denoised_dir),
        noisy_base_dir=str(out_noisy_dir),
        test_mode=False,
        confirm=False,
        use_gpu=False,
    )
    assert res is None
    assert not os.path.exists(str(out_metrics_csv))


def test_run_denoising_experiment_pipeline_with_resume(tmp_path):
    # Setup mock clean sharpened image
    sharp_dir = tmp_path / "data" / "sharpened"
    sharp_dir.mkdir(parents=True)
    sharp_img_path = sharp_dir / "case_001.png"

    clean_img = np.linspace(30, 220, 48 * 48, dtype=np.uint8).reshape((48, 48))
    cv2.imwrite(str(sharp_img_path), clean_img)

    sharp_csv = tmp_path / "mock_sharpened_meta.csv"
    pd.DataFrame([{
        "patient_id": "P_00044",
        "abnormality_category": "mass",
        "breast_side": "RIGHT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "label": 0,
        "dataset_split": "train",
        "original_image_path": "/path/to/orig.jpg",
        "normalized_image_path": "/path/to/norm.png",
        "contrast_image_path": "/path/to/contrast.png",
        "sharpened_image_path": str(sharp_img_path),
    }]).to_csv(str(sharp_csv), index=False)

    out_metrics_csv = tmp_path / "results" / "denoising_metrics.csv"
    out_denoised_dir = tmp_path / "data" / "denoised"
    out_noisy_dir = tmp_path / "data" / "noisy"

    # Run pass 1: 1 image x 5 noise models x 8 denoising methods = 40 rows
    df_pass1 = run_denoising_experiment(
        input_metadata_csv=str(sharp_csv),
        output_metrics_csv=str(out_metrics_csv),
        denoised_base_dir=str(out_denoised_dir),
        noisy_base_dir=str(out_noisy_dir),
        test_mode=True,
        max_images=1,
        resume=True,
        use_gpu=False,
    )
    assert len(df_pass1) == 40
    assert os.path.exists(str(out_metrics_csv))

    # Run pass 2 (Resume): should detect existing 40 records and complete instantly
    df_pass2 = run_denoising_experiment(
        input_metadata_csv=str(sharp_csv),
        output_metrics_csv=str(out_metrics_csv),
        denoised_base_dir=str(out_denoised_dir),
        noisy_base_dir=str(out_noisy_dir),
        test_mode=True,
        max_images=1,
        resume=True,
        use_gpu=False,
    )
    assert len(df_pass2) == 40
