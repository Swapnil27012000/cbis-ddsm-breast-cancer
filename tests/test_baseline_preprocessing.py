"""Unit and integration tests for Stage 5: Primary Baseline Image Preprocessing."""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.preprocessing.baseline_preprocessing import (
    load_baseline_config,
    select_pilot_samples,
    load_full_dataset_samples,
    detect_breast_region_conservative,
    normalize_intensity_robust,
    process_single_full_mammogram,
    run_baseline_preprocessing,
    run_baseline_preprocessing_pilot,
    STATUS_SUCCESS,
    STATUS_REVIEW_REQUIRED,
    STATUS_FAILED,
)
from src.preprocessing.baseline_visualization import (
    select_visualization_samples,
    generate_baseline_comparison_plot,
)


@pytest.fixture
def mock_stage5_env(tmp_path):
    """Create a mock environment with Stage 2, 3, 4 metadata and raw full mammograms."""
    s2_dir = tmp_path / "metadata" / "stage2"
    s3_dir = tmp_path / "metadata" / "stage3"
    s4_dir = tmp_path / "metadata" / "stage4"
    s5_dir = tmp_path / "metadata" / "stage5"
    raw_dir = tmp_path / "raw" / "CBIS_DDSM" / "jpeg"
    base_out = tmp_path / "processed" / "baseline" / "full_mammogram"
    res_dir = tmp_path / "results" / "baseline"

    for d in (s2_dir, s3_dir, s4_dir, s5_dir, raw_dir, base_out, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy raw full mammograms:
    # 1. Normal breast mammogram: 1200x1500 with tissue on right side
    img1 = np.zeros((1500, 1200), dtype=np.uint8)
    img1[200:1300, 400:1150] = 130
    p1_rel = "case1/full.jpg"
    (raw_dir / "case1").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_dir / p1_rel), img1)

    # 2. Mammogram on left side
    img2 = np.zeros((1400, 1100), dtype=np.uint8)
    img2[150:1250, 50:800] = 110
    p2_rel = "case2/full.jpg"
    (raw_dir / "case2").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_dir / p2_rel), img2)

    # Stage 2 reference mapping CSV
    rows = [
        {
            "source_csv": "mass_case_description_train_set.csv",
            "row_number": 0,
            "patient_id": "P_00001",
            "abnormality_id": 1,
            "abnormality_category": "mass",
            "breast_side": "LEFT",
            "image_view": "CC",
            "pathology": "BENIGN",
            "label": 0,
            "assessment": 4,
            "subtlety": 4,
            "dataset_split": "train",
            "original_csv_reference": "ref/p1.dcm",
            "original_resolved_path": p1_rel,
            "original_mapping_status": "RESOLVED",
            "cropped_csv_reference": "ref/crop1.dcm",
            "cropped_resolved_path": "crop1.jpg",
            "cropped_mapping_status": "RESOLVED",
            "roi_mask_csv_reference": "ref/mask1.dcm",
            "roi_mask_resolved_path": "mask1.jpg",
            "roi_mask_mapping_status": "RESOLVED",
        },
        {
            "source_csv": "calc_case_description_train_set.csv",
            "row_number": 1,
            "patient_id": "P_00002",
            "abnormality_id": 1,
            "abnormality_category": "calcification",
            "breast_side": "RIGHT",
            "image_view": "MLO",
            "pathology": "MALIGNANT",
            "label": 1,
            "assessment": 5,
            "subtlety": 5,
            "dataset_split": "train",
            "original_csv_reference": "ref/p2.dcm",
            "original_resolved_path": p2_rel,
            "original_mapping_status": "RESOLVED",
            "cropped_csv_reference": "ref/crop2.dcm",
            "cropped_resolved_path": "crop2.jpg",
            "cropped_mapping_status": "RESOLVED",
            "roi_mask_csv_reference": "ref/mask2.dcm",
            "roi_mask_resolved_path": "mask2.jpg",
            "roi_mask_mapping_status": "RESOLVED",
        },
    ]
    pd.DataFrame(rows).to_csv(s2_dir / "CBIS_DDSM_reference_mapping.csv", index=False)

    return {
        "s2_dir": s2_dir,
        "s3_dir": s3_dir,
        "s4_dir": s4_dir,
        "s5_dir": s5_dir,
        "raw_dir": raw_dir,
        "base_out": base_out,
        "res_dir": res_dir,
        "p1_rel": p1_rel,
        "p2_rel": p2_rel,
        "tmp_path": tmp_path,
    }


def test_detect_breast_region_conservative():
    """Test conservative breast detection bounding box and safety margin calculation."""
    config = {
        "breast_region": {
            "crop_margin_ratio": 0.05,
            "min_foreground_area_ratio": 0.05,
            "max_foreground_area_ratio": 0.98,
        }
    }
    # Create 1000x800 image with centered 400x500 tissue
    img = np.zeros((1000, 800), dtype=np.uint8)
    img[200:700, 200:600] = 150

    bbox, area_ratio, crop_applied, status, notes = detect_breast_region_conservative(img, config)
    cx, cy, cw, ch = bbox

    assert crop_applied is True
    assert status == STATUS_SUCCESS
    assert area_ratio > 0.0
    # Cropped bounding box must enclose the active region
    assert cx <= 200
    assert cy <= 200
    assert cx + cw >= 600
    assert cy + ch >= 700


def test_detect_breast_region_flat_image():
    """Test safe fallback on completely flat or empty image."""
    config = {
        "breast_region": {
            "crop_margin_ratio": 0.05,
            "min_foreground_area_ratio": 0.05,
            "max_foreground_area_ratio": 0.98,
        }
    }
    img = np.zeros((500, 400), dtype=np.uint8)
    bbox, area_ratio, crop_applied, status, notes = detect_breast_region_conservative(img, config)

    assert crop_applied is False
    assert status == STATUS_REVIEW_REQUIRED
    assert bbox == (0, 0, 400, 500)


def test_normalize_intensity_robust():
    """Test robust percentile intensity normalization to [0.0, 1.0]."""
    img = np.array([[10, 50], [100, 200]], dtype=np.uint8)
    norm, p_low, p_high = normalize_intensity_robust(img, lower_percentile=0.0, upper_percentile=100.0)

    assert norm.shape == img.shape
    assert norm.dtype == np.float32
    assert float(np.min(norm)) == 0.0
    assert float(np.max(norm)) == 1.0
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()


def test_process_single_full_mammogram(mock_stage5_env):
    """Test processing single full mammogram and creating baseline lossless PNG."""
    config = load_baseline_config()
    sample_info = {
        "patient_id": "P_00001",
        "abnormality_id": 1,
        "abnormality_category": "mass",
        "breast_side": "LEFT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "label": 0,
        "dataset_split": "train",
        "original_image_path": mock_stage5_env["p1_rel"],
    }
    raw_dir = str(mock_stage5_env["raw_dir"])
    out_dir = str(mock_stage5_env["base_out"])

    meta_rec, val_rec = process_single_full_mammogram(
        sample_info=sample_info,
        jpeg_dir=raw_dir,
        output_dir=out_dir,
        config=config,
        stage3_reviews={},
        stage4_quality_map={},
    )

    assert meta_rec["processing_status"] in (STATUS_SUCCESS, STATUS_REVIEW_REQUIRED)
    assert meta_rec["baseline_width"] > 0
    assert meta_rec["baseline_height"] > 0
    assert os.path.exists(meta_rec["baseline_image_path"])

    # Validation record asserts
    assert val_rec["output_exists"] is True
    assert val_rec["can_reopen"] is True
    assert val_rec["is_grayscale"] is True
    assert val_rec["channels"] == 1
    assert val_rec["dtype"] == "uint8"
    assert val_rec["raw_source_unmodified"] is True
    assert val_rec["validation_status"] == "PASS"


def test_run_baseline_preprocessing_pilot_end_to_end(mock_stage5_env):
    """Test full end-to-end pilot run and metadata artifact generation."""
    s2 = str(mock_stage5_env["s2_dir"])
    s3 = str(mock_stage5_env["s3_dir"])
    s4 = str(mock_stage5_env["s4_dir"])
    raw_base = str(mock_stage5_env["tmp_path"] / "raw" / "CBIS_DDSM")
    s5_out = str(mock_stage5_env["s5_dir"])
    base_out = str(mock_stage5_env["base_out"])

    res = run_baseline_preprocessing_pilot(
        stage2_metadata_dir=s2,
        stage3_metadata_dir=s3,
        stage4_metadata_dir=s4,
        raw_data_dir=raw_base,
        stage5_metadata_dir=s5_out,
        baseline_output_dir=base_out,
    )

    assert res["pilot_count"] == 2
    assert res["success_count"] + res["review_count"] == 2
    assert res["fail_count"] == 0

    assert os.path.exists(os.path.join(s5_out, "baseline_preprocessing_metadata.csv"))
    assert os.path.exists(os.path.join(s5_out, "baseline_preprocessing_validation.csv"))

    # Test visualization generation
    df_meta = pd.read_csv(os.path.join(s5_out, "baseline_preprocessing_metadata.csv"))
    viz_samples = select_visualization_samples(df_meta, n_samples=2)
    assert len(viz_samples) == 2

    out_plot = str(mock_stage5_env["res_dir"] / "original_vs_baseline.png")
    generate_baseline_comparison_plot(
        samples=viz_samples,
        jpeg_dir=str(mock_stage5_env["raw_dir"]),
        output_path=out_plot,
    )
    assert os.path.exists(out_plot)


def test_optimization_equivalence(mock_stage5_env):
    """Test that original unoptimized algorithm and optimized pipeline yield bit-exact pixels."""
    from src.preprocessing.verify_optimization_equivalence import (
        run_equivalence_verification,
    )
    s2 = str(mock_stage5_env["s2_dir"])
    s3 = str(mock_stage5_env["s3_dir"])
    s4 = str(mock_stage5_env["s4_dir"])
    raw_base = str(mock_stage5_env["tmp_path"] / "raw" / "CBIS_DDSM")
    s5_out = str(mock_stage5_env["s5_dir"])
    base_out = str(mock_stage5_env["base_out"])

    run_baseline_preprocessing_pilot(
        stage2_metadata_dir=s2,
        stage3_metadata_dir=s3,
        stage4_metadata_dir=s4,
        raw_data_dir=raw_base,
        stage5_metadata_dir=s5_out,
        baseline_output_dir=base_out,
    )

    res = run_equivalence_verification(
        stage5_metadata_dir=s5_out,
        baseline_output_dir=base_out,
        raw_data_dir=raw_base,
    )

    assert res["total_images"] == 2
    assert res["exact_count"] == 2
    assert res["all_exact_equal"] is True


def test_load_full_dataset_samples(mock_stage5_env):
    """Test loading and unique deterministic filename assignment for full dataset."""
    s2 = str(mock_stage5_env["s2_dir"])
    df_ref = pd.read_csv(os.path.join(s2, "CBIS_DDSM_reference_mapping.csv"))
    samples = load_full_dataset_samples(df_ref)

    assert len(samples) == 2
    assert samples[0]["patient_id"] == "P_00001"
    assert samples[0]["breast_side"] == "LEFT"
    assert samples[0]["image_view"] == "CC"
    assert samples[0]["output_filename"] == "P_00001_LEFT_CC_1_baseline.png"
    assert samples[1]["patient_id"] == "P_00002"
    assert samples[1]["output_filename"] == "P_00002_RIGHT_MLO_1_baseline.png"


def test_run_baseline_preprocessing_full_dataset_mode(mock_stage5_env):
    """Test full dataset execution mode and comprehensive validation report generation."""
    s2 = str(mock_stage5_env["s2_dir"])
    s3 = str(mock_stage5_env["s3_dir"])
    s4 = str(mock_stage5_env["s4_dir"])
    raw_base = str(mock_stage5_env["tmp_path"] / "raw" / "CBIS_DDSM")
    s5_out = str(mock_stage5_env["s5_dir"] / "full_run")
    base_out = str(mock_stage5_env["base_out"] / "full_run")

    res = run_baseline_preprocessing(
        stage2_metadata_dir=s2,
        stage3_metadata_dir=s3,
        stage4_metadata_dir=s4,
        raw_data_dir=raw_base,
        stage5_metadata_dir=s5_out,
        baseline_output_dir=base_out,
        force_pilot=False,
    )

    assert res["total_eligible"] == 2
    assert res["success_count"] + res["review_count"] == 2
    assert res["fail_count"] == 0

    meta_p = os.path.join(s5_out, "baseline_preprocessing_metadata.csv")
    val_p = os.path.join(s5_out, "baseline_preprocessing_validation.csv")
    rep_p = os.path.join(s5_out, "full_dataset_baseline_report.txt")

    assert os.path.exists(meta_p)
    assert os.path.exists(val_p)
    assert os.path.exists(rep_p)

    with open(rep_p, "r", encoding="utf-8") as f:
        rep_text = f.read()

    assert "CBIS-DDSM FULL DATASET BASELINE PREPROCESSING REPORT" in rep_text
    assert "Total eligible FULL/ORIGINAL records: 2" in rep_text
    assert "Raw data modified:\nNO" in rep_text

