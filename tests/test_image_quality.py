"""Unit and integration tests for Stage 4: Image Quality Control and Validation."""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.image_quality import (
    load_quality_config,
    compute_exact_hash,
    compute_perceptual_hash,
    hamming_distance,
    audit_single_image,
    find_near_duplicates_in_buckets,
    render_flagged_contact_sheet,
    load_stage3_review_records,
    run_stage4_image_quality_control,
    STATUS_PASS,
    STATUS_REVIEW,
    STATUS_FAIL,
    DECODE_VALID,
    DECODE_MISSING,
    MASK_BINARY_LIKE,
    MASK_NON_BINARY,
    MASK_EMPTY,
    ROLE_FULL_ORIGINAL,
    ROLE_CROPPED_ABNORMALITY,
    ROLE_ROI_MASK,
)


@pytest.fixture
def mock_stage4_env(tmp_path):
    """Create self-contained mock Stage-2, Stage-3 metadata and raw images."""
    s2_dir = tmp_path / "metadata" / "stage2"
    s3_dir = tmp_path / "metadata" / "stage3"
    raw_dir = tmp_path / "raw" / "CBIS_DDSM" / "jpeg"
    s2_dir.mkdir(parents=True)
    s3_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    # 1. Full Mammogram: 1200x1500 with breast tissue
    full_img = np.zeros((1500, 1200), dtype=np.uint8)
    full_img[200:1300, 200:1000] = 120
    full_rel = "case1/full.jpg"
    (raw_dir / "case1").mkdir(parents=True)
    cv2.imwrite(str(raw_dir / full_rel), full_img)

    # 2. Cropped lesion patch: 120x120
    crop_img = np.ones((120, 120), dtype=np.uint8) * 160
    crop_rel = "case1/crop.jpg"
    cv2.imwrite(str(raw_dir / crop_rel), crop_img)

    # 3. Binary ROI Mask: 1500x1200
    mask_img = np.zeros((1500, 1200), dtype=np.uint8)
    mask_img[400:500, 400:500] = 255
    mask_rel = "case1/mask.jpg"
    cv2.imwrite(str(raw_dir / mask_rel), mask_img)

    # 4. Reference mapping CSV
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
            "original_csv_reference": "ref/full.dcm",
            "original_resolved_path": full_rel,
            "original_mapping_status": "RESOLVED",
            "cropped_csv_reference": "ref/crop.dcm",
            "cropped_resolved_path": crop_rel,
            "cropped_mapping_status": "RESOLVED",
            "roi_mask_csv_reference": "ref/mask.dcm",
            "roi_mask_resolved_path": mask_rel,
            "roi_mask_mapping_status": "RESOLVED",
        },
        {
            "source_csv": "calc_case_description_train_set.csv",
            "row_number": 1216,
            "patient_id": "P_01563",
            "abnormality_id": 2,
            "abnormality_category": "calcification",
            "breast_side": "RIGHT",
            "image_view": "MLO",
            "pathology": "MALIGNANT",
            "label": 1,
            "assessment": 5,
            "subtlety": 5,
            "dataset_split": "train",
            "original_csv_reference": "ref/p1563_full.dcm",
            "original_resolved_path": full_rel,
            "original_mapping_status": "RESOLVED",
            "cropped_csv_reference": "ref/missing.dcm",
            "cropped_resolved_path": "",
            "cropped_mapping_status": "UNRESOLVED",
            "roi_mask_csv_reference": "ref/missing.dcm",
            "roi_mask_resolved_path": "",
            "roi_mask_mapping_status": "UNRESOLVED",
        },
    ]
    df_ref = pd.DataFrame(rows)
    df_ref.to_csv(s2_dir / "CBIS_DDSM_reference_mapping.csv", index=False)

    # 5. Stage 3 review CSV
    st3_rows = [
        {
            "patient_id": "P_00004",
            "abnormality_id": 1,
            "image_role": "ROI_MASK",
            "review_resolution": "REVIEW_CONFIRMED",
            "final_status": "REVIEW",
            "notes": "Non-binary tissue patch",
        }
    ]
    pd.DataFrame(st3_rows).to_csv(s3_dir / "stage3_review_analysis.csv", index=False)

    return {
        "s2_dir": s2_dir,
        "s3_dir": s3_dir,
        "raw_dir": raw_dir,
        "full_rel": full_rel,
        "crop_rel": crop_rel,
        "mask_rel": mask_rel,
        "tmp_path": tmp_path,
    }


def test_audit_single_image_full(mock_stage4_env):
    """Test auditing a full mammogram for technical dimensions and foreground statistics."""
    config = load_quality_config()
    p = str(mock_stage4_env["raw_dir"] / mock_stage4_env["full_rel"])

    res = audit_single_image(p, ROLE_FULL_ORIGINAL, config)

    assert res["file_exists"] is True
    assert res["decode_status"] == DECODE_VALID
    assert res["width"] == 1200
    assert res["height"] == 1500
    assert res["aspect_ratio"] == 0.8
    assert res["channels"] == 1
    assert res["min_intensity"] == 0
    assert res["max_intensity"] == 120
    assert res["foreground_area_ratio"] > 0.0
    assert res["bounding_box_width"] > 0
    assert res["nan_count"] == 0
    assert res["inf_count"] == 0
    assert len(res["exact_hash"]) == 64
    assert len(res["perceptual_hash"]) > 0


def test_audit_single_image_mask(mock_stage4_env):
    """Test auditing an ROI mask for binary classification and nonzero coverage."""
    config = load_quality_config()
    p = str(mock_stage4_env["raw_dir"] / mock_stage4_env["mask_rel"])

    res = audit_single_image(p, ROLE_ROI_MASK, config)

    assert res["file_exists"] is True
    assert res["decode_status"] == DECODE_VALID
    assert res["unique_pixel_count"] == 2
    assert res["mask_classification"] == MASK_BINARY_LIKE
    assert res["binary_like_ratio"] == 1.0


def test_audit_missing_file():
    """Test auditing a non-existent file."""
    config = load_quality_config()
    res = audit_single_image("non_existent_file.jpg", ROLE_FULL_ORIGINAL, config)
    assert res["file_exists"] is False
    assert res["decode_status"] == DECODE_MISSING


def test_hash_functions(tmp_path):
    """Test SHA-256 and dHash perceptual hash calculation and Hamming distance."""
    dummy_p = str(tmp_path / "test.jpg")
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img1[20:80, 20:80] = 200
    cv2.imwrite(dummy_p, img1)

    h_exact = compute_exact_hash(dummy_p)
    assert len(h_exact) == 64

    phash1 = compute_perceptual_hash(img1)
    phash2 = compute_perceptual_hash(img1)
    assert phash1 == phash2
    assert hamming_distance(phash1, phash2) == 0


def test_render_contact_sheet(tmp_path):
    """Test rendering flagged contact sheets."""
    out_png = str(tmp_path / "flagged.png")
    records = [
        {
            "patient_id": "P_00001",
            "abnormality_id": 1,
            "image_role": "FULL_ORIGINAL",
            "image_path": "case1/full.jpg",
            "width": 1200,
            "height": 1500,
            "quality_status": "REVIEW",
            "quality_flags": "UNUSUAL_DIMENSIONS",
        }
    ]
    render_flagged_contact_sheet(records, str(tmp_path), out_png, "Test Sheet", max_images=4)
    assert os.path.exists(out_png)

    # Empty list test
    empty_png = str(tmp_path / "empty.png")
    render_flagged_contact_sheet([], str(tmp_path), empty_png, "Empty Sheet", max_images=4)
    assert os.path.exists(empty_png)


def test_run_stage4_quality_control_end_to_end(mock_stage4_env, tmp_path):
    """Integration test of run_stage4_image_quality_control pipeline."""
    s2 = str(mock_stage4_env["s2_dir"])
    s3 = str(mock_stage4_env["s3_dir"])
    raw_base = str(mock_stage4_env["tmp_path"] / "raw" / "CBIS_DDSM")
    s4_out = str(tmp_path / "metadata" / "stage4")
    res_out = str(tmp_path / "results" / "quality_control")

    summary = run_stage4_image_quality_control(
        stage2_metadata_dir=s2,
        stage3_metadata_dir=s3,
        raw_data_dir=raw_base,
        stage4_output_dir=s4_out,
        results_dir=res_out,
    )

    assert summary["total_audited"] == 6  # 2 rows x 3 references = 6
    assert summary["pass_count"] >= 1
    assert os.path.exists(os.path.join(s4_out, "image_quality_report.csv"))
    assert os.path.exists(os.path.join(s4_out, "image_quality_summary.csv"))
    assert os.path.exists(os.path.join(s4_out, "image_quality_flags.csv"))
    assert os.path.exists(os.path.join(s4_out, "image_quality_report.txt"))
    assert os.path.exists(os.path.join(res_out, "flagged_full_mammograms.png"))
    assert os.path.exists(os.path.join(res_out, "flagged_cropped_images.png"))
    assert os.path.exists(os.path.join(res_out, "flagged_roi_masks.png"))
