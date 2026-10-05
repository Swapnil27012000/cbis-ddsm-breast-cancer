"""Unit and integration tests for Stage 3: Image-Role Verification and Visual Inspection."""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.image_role_verification import (
    load_and_scale_image,
    select_stratified_samples,
    select_paired_triplets,
    select_overlay_cases,
    verify_image_record,
    plot_image_grid,
    plot_paired_triplets,
    plot_roi_overlays,
    generate_stage3_text_report,
    run_stage3_verification,
    STATUS_PASS,
    STATUS_REVIEW,
    STATUS_FAIL,
    ROLE_FULL_ORIGINAL,
    ROLE_CROPPED_ABNORMALITY,
    ROLE_ROI_MASK,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED_STATUS,
)


@pytest.fixture
def mock_dataset_env(tmp_path):
    """Create a self-contained mock Stage-2 metadata directory and raw image files."""
    meta_dir = tmp_path / "metadata" / "stage2"
    raw_dir = tmp_path / "raw" / "jpeg"
    meta_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    # Create dummy image files
    # 1. Full mammogram: large dimensions, grayscale
    full_img = np.ones((800, 600), dtype=np.uint8) * 128
    full_path_rel = "case1/full.jpg"
    (raw_dir / "case1").mkdir(parents=True)
    cv2.imwrite(str(raw_dir / full_path_rel), full_img)

    # 2. Cropped abnormality
    crop_img = np.ones((150, 150), dtype=np.uint8) * 180
    crop_path_rel = "case1/crop.jpg"
    cv2.imwrite(str(raw_dir / crop_path_rel), crop_img)

    # 3. Binary ROI mask
    mask_img = np.zeros((800, 600), dtype=np.uint8)
    mask_img[200:300, 200:300] = 255
    mask_path_rel = "case1/mask.jpg"
    cv2.imwrite(str(raw_dir / mask_path_rel), mask_img)

    # 4. Empty mask
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    empty_mask_rel = "case1/empty_mask.jpg"
    cv2.imwrite(str(raw_dir / empty_mask_rel), empty_mask)

    # 5. Non-binary mask (multiple grayscale values)
    multi_mask = np.zeros((100, 100), dtype=np.uint8)
    multi_mask[10:30, 10:30] = 100
    multi_mask[30:50, 30:50] = 200
    multi_mask_rel = "case1/multi_mask.jpg"
    cv2.imwrite(str(raw_dir / multi_mask_rel), multi_mask)

    # Create mock reference mapping CSV
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
            "original_resolved_path": full_path_rel,
            "original_mapping_status": STATUS_RESOLVED,
            "cropped_csv_reference": "ref/crop.dcm",
            "cropped_resolved_path": crop_path_rel,
            "cropped_mapping_status": STATUS_RESOLVED,
            "roi_mask_csv_reference": "ref/mask.dcm",
            "roi_mask_resolved_path": mask_path_rel,
            "roi_mask_mapping_status": STATUS_RESOLVED,
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
            "original_resolved_path": full_path_rel,
            "original_mapping_status": STATUS_RESOLVED,
            "cropped_csv_reference": "ref/missing.dcm",
            "cropped_resolved_path": "",
            "cropped_mapping_status": STATUS_UNRESOLVED_STATUS,
            "roi_mask_csv_reference": "ref/missing.dcm",
            "roi_mask_resolved_path": "",
            "roi_mask_mapping_status": STATUS_UNRESOLVED_STATUS,
        },
    ]
    df_ref = pd.DataFrame(rows)
    ref_csv_path = meta_dir / "CBIS_DDSM_reference_mapping.csv"
    df_ref.to_csv(ref_csv_path, index=False)

    return {
        "tmp_path": tmp_path,
        "meta_dir": meta_dir,
        "raw_dir": raw_dir,
        "ref_csv_path": ref_csv_path,
        "full_path_rel": full_path_rel,
        "crop_path_rel": crop_path_rel,
        "mask_path_rel": mask_path_rel,
        "empty_mask_rel": empty_mask_rel,
        "multi_mask_rel": multi_mask_rel,
    }


def test_load_and_scale_image(mock_dataset_env):
    """Test reading native image dimensions and stats without disk mutation."""
    raw_dir = mock_dataset_env["raw_dir"]
    full_path = os.path.join(raw_dir, mock_dataset_env["full_path_rel"])

    disp_img, stats = load_and_scale_image(full_path, max_dim=400)

    assert stats["readable"] is True
    assert stats["width"] == 600
    assert stats["height"] == 800
    assert stats["aspect_ratio"] == 0.75
    assert stats["channels"] == 1
    assert stats["min_val"] == 128
    assert stats["max_val"] == 128
    # Display downscaled copy conforms to max_dim
    assert max(disp_img.shape) == 400


def test_mask_metrics_and_classification(mock_dataset_env):
    """Test ROI mask geometric calculations and BINARY_LIKE, EMPTY, NON_BINARY classifications."""
    raw_dir = mock_dataset_env["raw_dir"]

    # 1. Binary-like mask
    mask_p = os.path.join(raw_dir, mock_dataset_env["mask_path_rel"])
    _, stats_binary = load_and_scale_image(mask_p)
    status, notes = verify_image_record(
        {"image_role": ROLE_ROI_MASK, "patient_id": "P_00001", "abnormality_id": 1, "category": "mass", "side": "LEFT", "view": "CC", "pathology": "BENIGN", "split": "train", "image_path": "case1/mask.jpg"},
        stats_binary,
    )
    assert status == STATUS_PASS
    assert "BINARY_LIKE" in notes
    assert stats_binary["nonzero_pixels"] == 10000

    # 2. Empty mask
    empty_p = os.path.join(raw_dir, mock_dataset_env["empty_mask_rel"])
    _, stats_empty = load_and_scale_image(empty_p)
    status_empty, notes_empty = verify_image_record(
        {"image_role": ROLE_ROI_MASK, "patient_id": "P_00001", "abnormality_id": 1, "category": "mass", "side": "LEFT", "view": "CC", "pathology": "BENIGN", "split": "train", "image_path": "case1/empty_mask.jpg"},
        stats_empty,
    )
    assert status_empty == STATUS_REVIEW
    assert "EMPTY" in notes_empty

    # 3. Non-binary mask
    multi_p = os.path.join(raw_dir, mock_dataset_env["multi_mask_rel"])
    _, stats_multi = load_and_scale_image(multi_p)
    status_multi, notes_multi = verify_image_record(
        {"image_role": ROLE_ROI_MASK, "patient_id": "P_00001", "abnormality_id": 1, "category": "mass", "side": "LEFT", "view": "CC", "pathology": "BENIGN", "split": "train", "image_path": "case1/multi_mask.jpg"},
        stats_multi,
    )
    assert status_multi == STATUS_REVIEW
    assert "NON_BINARY" in notes_multi


def test_select_stratified_samples(mock_dataset_env):
    """Test deterministic sample selection across strata."""
    df_ref = pd.read_csv(mock_dataset_env["ref_csv_path"])

    samples = select_stratified_samples(df_ref, ROLE_FULL_ORIGINAL, n_samples=2)
    assert len(samples) > 0
    assert samples[0]["image_role"] == ROLE_FULL_ORIGINAL
    assert samples[0]["patient_id"] in ("P_00001", "P_01563")


def test_select_paired_triplets(mock_dataset_env):
    """Test paired triplet selection matching exact patient_id and abnormality_id."""
    df_ref = pd.read_csv(mock_dataset_env["ref_csv_path"])

    triplets = select_paired_triplets(df_ref, n_samples=2)
    # Only P_00001 has all 3 resolved
    assert len(triplets) == 1
    t = triplets[0]
    assert t["patient_id"] == "P_00001"
    assert t["abnormality_id"] == 1
    assert t["original_path"] == mock_dataset_env["full_path_rel"]
    assert t["cropped_path"] == mock_dataset_env["crop_path_rel"]
    assert t["roi_mask_path"] == mock_dataset_env["mask_path_rel"]


def test_plot_visualizations(mock_dataset_env, tmp_path):
    """Test contact sheet, triplet, and overlay generation."""
    raw_dir = str(mock_dataset_env["raw_dir"])
    df_ref = pd.read_csv(mock_dataset_env["ref_csv_path"])

    full_samples = select_stratified_samples(df_ref, ROLE_FULL_ORIGINAL, n_samples=1)
    triplets = select_paired_triplets(df_ref, n_samples=1)
    overlays = select_overlay_cases(df_ref, raw_dir, n_samples=1)

    out_grid = str(tmp_path / "full.png")
    out_triplets = str(tmp_path / "triplets.png")
    out_overlays = str(tmp_path / "overlays.png")

    # 1. Grid
    records = plot_image_grid(full_samples, raw_dir, out_grid, "Test Full Mammograms")
    assert os.path.exists(out_grid)
    assert len(records) == 1
    assert records[0]["verification_status"] == STATUS_PASS

    # 2. Triplets
    plot_paired_triplets(triplets, raw_dir, out_triplets)
    assert os.path.exists(out_triplets)

    # 3. Overlays
    plot_roi_overlays(overlays, raw_dir, out_overlays)
    assert os.path.exists(out_overlays)


def test_generate_stage3_text_report(tmp_path):
    """Test text report generation matches exact Section 18 structure."""
    out_txt = str(tmp_path / "report.txt")
    verified_records = [
        {
            "image_role": ROLE_FULL_ORIGINAL,
            "verification_status": STATUS_PASS,
            "patient_id": "P_00001",
            "abnormality_id": 1,
            "image_path": "case1/full.jpg",
            "verification_notes": "Valid image",
        }
    ]
    p01563_status = {
        ROLE_FULL_ORIGINAL: "RESOLVED",
        ROLE_CROPPED_ABNORMALITY: "UNRESOLVED",
        ROLE_ROI_MASK: "UNRESOLVED",
    }

    report = generate_stage3_text_report(
        verified_records=verified_records,
        n_triplets_available=1,
        n_triplets_visualized=1,
        n_overlays_available=1,
        n_overlays_visualized=1,
        p01563_status=p01563_status,
        output_path=out_txt,
    )

    assert "CBIS-DDSM IMAGE ROLE VERIFICATION" in report
    assert "FULL_ORIGINAL SAMPLES" in report
    assert "CROPPED_ABNORMALITY SAMPLES" in report
    assert "ROI_MASK SAMPLES" in report
    assert "ORIGINAL + CROPPED + ROI PAIRS" in report
    assert "ROI OVERLAYS" in report
    assert "ISSUES REQUIRING REVIEW" in report
    assert "P_01563" in report
    assert "UNRESOLVED" in report


def test_run_stage3_verification_end_to_end(mock_dataset_env, tmp_path):
    """Integration test of run_stage3_verification pipeline."""
    meta_dir = str(mock_dataset_env["meta_dir"])
    raw_dir = str(mock_dataset_env["tmp_path"] / "raw")
    stage3_out = str(tmp_path / "metadata" / "stage3")
    results_out = str(tmp_path / "results" / "role_verification")

    res = run_stage3_verification(
        stage2_metadata_dir=meta_dir,
        raw_data_dir=raw_dir,
        stage3_output_dir=stage3_out,
        results_dir=results_out,
    )

    assert res["pass_count"] >= 1
    assert os.path.exists(os.path.join(stage3_out, "image_role_verification_report.csv"))
    assert os.path.exists(os.path.join(stage3_out, "image_role_sample_selection.csv"))
    assert os.path.exists(os.path.join(stage3_out, "image_role_verification_report.txt"))
    assert os.path.exists(os.path.join(results_out, "full_mammograms.png"))
    assert os.path.exists(os.path.join(results_out, "cropped_abnormalities.png"))
    assert os.path.exists(os.path.join(results_out, "roi_masks.png"))
    assert os.path.exists(os.path.join(results_out, "original_cropped_roi_pairs.png"))
    assert os.path.exists(os.path.join(results_out, "roi_overlays.png"))
