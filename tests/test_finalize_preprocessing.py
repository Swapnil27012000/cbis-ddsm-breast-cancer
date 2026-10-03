"""Unit and integration tests for final preprocessing dataset generation stage."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.preprocessing.finalize_preprocessing import (
    resolve_path,
    build_final_preprocessing_metadata,
    run_integrity_checks,
    generate_final_processing_report,
    run_final_preprocessing_stage,
)

EXPECTED_COLUMNS = [
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
    "noise_type",
    "denoising_method",
    "noisy_image_path",
    "denoised_image_path",
]


def test_build_final_preprocessing_metadata_structure():
    """Verify that build_final_preprocessing_metadata produces exact expected columns and records."""
    df = build_final_preprocessing_metadata(
        clean_metadata_csv="data/metadata/sharpened_image_metadata.csv",
        experiment_metrics_csv="results/preprocessing/denoising_metrics.csv",
        include_experiments=True
    )

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 410

    # 10 clean baseline records
    clean_subset = df[df["noise_type"].isna()]
    assert len(clean_subset) == 10
    assert clean_subset["denoising_method"].isna().all()
    assert clean_subset["noisy_image_path"].isna().all()
    assert clean_subset["denoised_image_path"].isna().all()

    # 400 experimental records
    exp_subset = df[df["noise_type"].notna()]
    assert len(exp_subset) == 400
    assert exp_subset["denoising_method"].notna().all()
    assert exp_subset["noisy_image_path"].notna().all()
    assert exp_subset["denoised_image_path"].notna().all()


def test_integrity_checks_on_generated_dataset():
    """Verify that all 7 integrity checks pass on the produced dataset."""
    metadata_path = resolve_path("data/metadata/final_preprocessing_metadata.csv")
    assert metadata_path and os.path.exists(metadata_path)

    df = pd.read_csv(metadata_path)
    integrity_results = run_integrity_checks(df)

    assert integrity_results["overall_passed"] is True, f"Integrity checks failed: {integrity_results['errors']}"

    checks = integrity_results["checks"]
    assert checks["every_referenced_file_exists"]["passed"] is True
    assert checks["every_referenced_file_exists"]["missing_count"] == 0

    assert checks["no_raw_files_modified"]["passed"] is True

    assert checks["no_train_test_patient_leakage"]["passed"] is True
    assert len(checks["no_train_test_patient_leakage"]["train_test_overlap"]) == 0

    assert checks["pathology_is_not_missing"]["passed"] is True
    assert checks["pathology_is_not_missing"]["missing_count"] == 0

    assert checks["label_matches_pathology"]["passed"] is True
    assert checks["label_matches_pathology"]["mismatches_count"] == 0

    assert checks["image_dimensions_are_valid"]["passed"] is True
    assert checks["image_dimensions_are_valid"]["dimension_errors_count"] == 0

    assert checks["no_duplicate_output_paths"]["passed"] is True
    assert checks["no_duplicate_output_paths"]["duplicate_count"] == 0


def test_integrity_check_detects_patient_leakage():
    """Verify that patient overlap between train and test splits is caught by integrity check."""
    metadata_path = resolve_path("data/metadata/final_preprocessing_metadata.csv")
    df = pd.read_csv(metadata_path).copy()

    # Artificially introduce patient leakage
    test_rows = df.tail(10).copy()
    test_rows["dataset_split"] = "test"
    # Keep the same patient_id from train in test
    leaked_df = pd.concat([df, test_rows], ignore_index=True)

    results = run_integrity_checks(leaked_df)
    assert results["checks"]["no_train_test_patient_leakage"]["passed"] is False
    assert results["overall_passed"] is False


def test_integrity_check_detects_pathology_missing():
    """Verify that missing/empty pathology classifications are caught."""
    metadata_path = resolve_path("data/metadata/final_preprocessing_metadata.csv")
    df = pd.read_csv(metadata_path).copy()

    # Inject missing pathology
    df.loc[0, "pathology"] = np.nan

    results = run_integrity_checks(df)
    assert results["checks"]["pathology_is_not_missing"]["passed"] is False
    assert results["overall_passed"] is False


def test_integrity_check_detects_label_discordance():
    """Verify that discordant binary labels trigger an integrity failure."""
    metadata_path = resolve_path("data/metadata/final_preprocessing_metadata.csv")
    df = pd.read_csv(metadata_path).copy()

    # Corrupt benign label to 1
    benign_idx = df[df["pathology"] == "BENIGN"].index[0]
    df.loc[benign_idx, "label"] = 1

    results = run_integrity_checks(df)
    assert results["checks"]["label_matches_pathology"]["passed"] is False
    assert results["overall_passed"] is False


def test_integrity_check_detects_duplicate_output_paths():
    """Verify that colliding output paths trigger an integrity failure."""
    metadata_path = resolve_path("data/metadata/final_preprocessing_metadata.csv")
    df = pd.read_csv(metadata_path).copy()

    # Duplicate first row's output path onto second row
    df.loc[1, "sharpened_image_path"] = df.loc[0, "sharpened_image_path"]

    results = run_integrity_checks(df)
    assert results["checks"]["no_duplicate_output_paths"]["passed"] is False
    assert results["overall_passed"] is False


def test_final_processing_report_contents():
    """Verify that final_processing_report.txt contains all required executive reporting sections."""
    report_path = resolve_path("results/preprocessing/final_processing_report.txt")
    assert report_path and os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Required sections
    assert "total records" in content.lower()
    assert "successful images" in content.lower()
    assert "failed images" in content.lower()
    assert "missing output" in content.lower()
    assert "processing time" in content.lower()
    assert "gpu" in content.lower() or "cuda" in content.lower()
    assert "noise experiment statistics" in content.lower()
    assert "denoising benchmark experiment statistics" in content.lower()

    # Check that all 7 checks are documented in report
    assert "Check 1: Every Referenced File Exists on Disk" in content
    assert "Check 2: No Raw Files Were Modified or Overwritten" in content
    assert "Check 3: No Train/Test Patient Leakage" in content
    assert "Check 4: Pathology is Not Missing" in content
    assert "Check 5: Label Matches Pathology" in content
    assert "Check 6: Image Dimensions Are Valid" in content
    assert "Check 7: No Duplicate Output Paths" in content
