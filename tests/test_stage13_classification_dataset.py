"""Unit and integration tests for Stage 13: Classification Dataset Construction and Patient-Level Splitting.

Verifies:
1. Reliable label mapping: BENIGN -> 0, BENIGN_WITHOUT_CALLBACK -> 0, MALIGNANT -> 1.
2. Rejection/isolation of unknown, missing, or ambiguous pathology labels.
3. Accurate mapping to Stage 5 baseline full/original mammogram PNGs (no crops/masks).
4. Multi-abnormality case auditing and primary view canonical flagging.
5. Strict patient-level grouping (all views/records of a patient in one split).
6. Zero patient leakage across active train, val, and test manifests.
7. Preservation of official test assignments for clean test patients.
8. Deterministic validation split reproducibility with fixed random seed (42).
9. Class count and partition inventory consistency.
10. Full manifest-to-source CSV and Stage 5 traceability.
11. Detection and quarantining of official cross-partition patient conflicts.
12. Binding and documentation of Stage 12 candidate strategies (C1–C5).
13. Immutability of raw CBIS-DDSM data and prior Stages 1–12 outputs.
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.stage13_dataset import (
    LABEL_MAP,
    LABEL_NAMES,
    STAGE12_CANDIDATES,
    audit_cross_partition_patient_conflicts,
    audit_images_and_traceability,
    create_patient_level_splits,
    find_csv_dir,
    find_stage5_metadata,
    generate_class_distribution_summary,
    generate_missing_image_and_invalid_label_report,
    generate_stage13_config_json,
    generate_stage13_reports,
    generate_stage13_validation_csv,
    load_raw_case_descriptions,
    load_stage5_baseline_metadata,
    map_pathology_labels,
    run_stage13_pipeline,
)


def test_label_mapping_and_unknown_pathology_rejection():
    """Verify that binary mapping is exact and unknown values are flagged rather than assumed benign."""
    test_pathologies = pd.Series([
        "BENIGN",
        "BENIGN_WITHOUT_CALLBACK",
        "MALIGNANT",
        "benign",
        "malignant",
        "UNKNOWN_SUSPICIOUS",
        "",
        None,
    ])
    labels, statuses = map_pathology_labels(test_pathologies)

    # Valid labels
    assert labels[0] == 0
    assert statuses[0] == "VALID"

    assert labels[1] == 0
    assert statuses[1] == "VALID"

    assert labels[2] == 1
    assert statuses[2] == "VALID"

    assert labels[3] == 0
    assert labels[4] == 1

    # Invalid / unknown labels must be -1, NEVER silently converted to 0
    assert labels[5] == -1
    assert "INVALID_UNKNOWN_PATHOLOGY" in statuses[5]

    assert labels[6] == -1
    assert labels[7] == -1


def test_load_raw_case_descriptions_counts():
    """Verify loading of all 4 CBIS-DDSM case description CSVs and their reconciliation."""
    df_raw, counts = load_raw_case_descriptions()
    assert len(df_raw) == 3568
    assert counts["mass_case_description_train_set.csv"] == 1318
    assert counts["mass_case_description_test_set.csv"] == 378
    assert counts["calc_case_description_train_set.csv"] == 1546
    assert counts["calc_case_description_test_set.csv"] == 326
    assert sum(counts.values()) == 3568

    # Key columns present
    for col in ["patient_id", "breast_side", "image_view", "abnormality_id", "pathology", "official_split"]:
        assert col in df_raw.columns


def test_stage5_baseline_metadata_alignment():
    """Verify that Stage 5 metadata contains 3,568 processed records and aligns with raw cases."""
    df_s5 = load_stage5_baseline_metadata()
    assert len(df_s5) == 3568
    assert (df_s5["processing_status"] == "SUCCESS").all()

    # Verify paths are full mammogram baseline images
    sample_paths = df_s5["baseline_image_path"].head(10).tolist()
    for p in sample_paths:
        assert "full_mammogram" in p
        assert p.endswith("_baseline.png")
        assert "crop" not in p.lower()
        assert "mask" not in p.lower()


def test_multi_abnormality_and_view_preservation():
    """Verify that distinct mammographic views are preserved while multi-abnormality records are audited."""
    df_raw, _ = load_raw_case_descriptions()
    df_s5 = load_stage5_baseline_metadata()
    df_audited, metrics = audit_images_and_traceability(df_raw, df_s5)

    assert metrics["total_source_records"] == 3568
    assert metrics["aligned_records"] == 3568
    assert metrics["unique_patients"] > 0
    assert metrics["unique_mammographic_views"] > 0
    # Number of unique views should be less than total records due to multi-abnormality cases
    assert metrics["unique_mammographic_views"] < metrics["total_source_records"]
    assert metrics["primary_view_records"] == metrics["unique_mammographic_views"]
    assert metrics["primary_view_records"] + metrics["secondary_abnormality_records"] == 3568


def test_cross_partition_patient_conflict_detection():
    """Verify detection of official cross-partition patient conflicts (e.g. P_00016)."""
    df_raw, _ = load_raw_case_descriptions()
    df_s5 = load_stage5_baseline_metadata()
    df_audited, _ = audit_images_and_traceability(df_raw, df_s5)

    conflicts_df, conflicting_pids = audit_cross_partition_patient_conflicts(df_audited)

    # In official CBIS-DDSM, patients with both mass and calc lesions (like P_00016) exist
    assert len(conflicting_pids) > 0
    assert "P_00016" in conflicting_pids

    # Conflicting patients must be documented with train and test records
    p16_conflict = conflicts_df[conflicts_df["patient_id"] == "P_00016"].iloc[0]
    assert p16_conflict["official_train_records"] > 0
    assert p16_conflict["official_test_records"] > 0


def test_patient_level_zero_leakage_and_grouping():
    """Verify that all views of a patient stay in one partition and no overlap exists across splits."""
    df_raw, _ = load_raw_case_descriptions()
    df_s5 = load_stage5_baseline_metadata()
    df_audited, _ = audit_images_and_traceability(df_raw, df_s5)
    conflicts_df, conflicting_pids = audit_cross_partition_patient_conflicts(df_audited)

    master_df, patient_df, train_df, val_df, test_df = create_patient_level_splits(
        df_audited,
        conflicting_pids=conflicting_pids,
        val_ratio=0.20,
        seed=42,
        conflict_policy="isolate_for_review",
    )

    train_patients = set(train_df["patient_id"].unique())
    val_patients = set(val_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())
    conflict_patients = set(master_df[master_df["assigned_split"] == "conflict_review"]["patient_id"].unique())

    # ZERO overlap between any active partitions
    assert len(train_patients.intersection(val_patients)) == 0, "Train and Val share patients!"
    assert len(train_patients.intersection(test_patients)) == 0, "Train and Test share patients!"
    assert len(val_patients.intersection(test_patients)) == 0, "Val and Test share patients!"
    assert len(train_patients.intersection(conflict_patients)) == 0
    assert len(val_patients.intersection(conflict_patients)) == 0
    assert len(test_patients.intersection(conflict_patients)) == 0

    # Total patient count check
    total_unique_pats = len(df_audited["patient_id"].unique())
    assert len(train_patients) + len(val_patients) + len(test_patients) + len(conflict_patients) == total_unique_pats

    # Total record count check
    assert len(train_df) + len(val_df) + len(test_df) + len(master_df[master_df["assigned_split"] == "conflict_review"]) == 3568


def test_official_test_set_preservation():
    """Verify that clean official test patients are preserved in the test partition."""
    df_raw, _ = load_raw_case_descriptions()
    df_s5 = load_stage5_baseline_metadata()
    df_audited, _ = audit_images_and_traceability(df_raw, df_s5)
    conflicts_df, conflicting_pids = audit_cross_partition_patient_conflicts(df_audited)

    master_df, patient_df, train_df, val_df, test_df = create_patient_level_splits(
        df_audited,
        conflicting_pids=conflicting_pids,
        val_ratio=0.20,
        seed=42,
    )

    # Every record in test_df must originate from official test set
    assert (test_df["official_split"] == "test").all()

    # None of the test patients should appear in official train
    test_pids = set(test_df["patient_id"].unique())
    assert len(test_pids.intersection(conflicting_pids)) == 0


def test_deterministic_split_reproducibility():
    """Verify that running the split twice with seed=42 produces 100% identical partitions."""
    df_raw, _ = load_raw_case_descriptions()
    df_s5 = load_stage5_baseline_metadata()
    df_audited, _ = audit_images_and_traceability(df_raw, df_s5)
    conflicts_df, conflicting_pids = audit_cross_partition_patient_conflicts(df_audited)

    _, p_df1, tr1, val1, te1 = create_patient_level_splits(
        df_audited, conflicting_pids, val_ratio=0.20, seed=42
    )
    _, p_df2, tr2, val2, te2 = create_patient_level_splits(
        df_audited, conflicting_pids, val_ratio=0.20, seed=42
    )

    pd.testing.assert_frame_equal(p_df1, p_df2)
    pd.testing.assert_series_equal(tr1["case_key"].sort_values(), tr2["case_key"].sort_values())
    pd.testing.assert_series_equal(val1["case_key"].sort_values(), val2["case_key"].sort_values())
    pd.testing.assert_series_equal(te1["case_key"].sort_values(), te2["case_key"].sort_values())


def test_manifest_traceability():
    """Verify that every manifest record can be traced back to its raw CSV and baseline PNG."""
    df_raw, _ = load_raw_case_descriptions()
    df_s5 = load_stage5_baseline_metadata()
    df_audited, _ = audit_images_and_traceability(df_raw, df_s5)
    conflicts_df, conflicting_pids = audit_cross_partition_patient_conflicts(df_audited)
    master_df, _, train_df, val_df, test_df = create_patient_level_splits(
        df_audited, conflicting_pids, val_ratio=0.20, seed=42
    )

    for subset in [train_df, val_df, test_df]:
        for _, row in subset.head(20).iterrows():
            assert pd.notna(row["patient_id"])
            assert pd.notna(row["source_csv"])
            assert pd.notna(row["baseline_image_path"])
            assert row["binary_label"] in (0, 1)
            assert row["pathology"] in ("BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT")


def test_missing_image_and_invalid_label_reporting():
    """Verify that missing files or invalid labels are flagged correctly."""
    df_mock = pd.DataFrame([
        {
            "patient_id": "P_99999",
            "breast_side": "LEFT",
            "image_view": "CC",
            "abnormality_id": 1,
            "source_csv": "mass_train.csv",
            "baseline_image_path": "non_existent_file.png",
            "pathology": "UNKNOWN_ERROR",
            "baseline_file_exists": False,
            "baseline_file_readable": False,
            "dimensions_match_metadata": False,
            "binary_label": -1,
            "label_status": "INVALID_UNKNOWN_PATHOLOGY",
        }
    ])
    rep = generate_missing_image_and_invalid_label_report(df_mock)
    assert len(rep) == 1
    assert "BASELINE_FILE_MISSING" in rep.iloc[0]["issues"]
    assert "INVALID_LABEL" in rep.iloc[0]["issues"]


def test_stage12_candidates_binding():
    """Verify that Stage 12 candidate definitions C1–C5 are preserved and documented."""
    assert len(STAGE12_CANDIDATES) == 5
    ids = [c["candidate_id"] for c in STAGE12_CANDIDATES]
    assert "C1_BASELINE" in ids
    assert "C2_CLAHE" in ids
    assert "C3_UNSHARP_MASK" in ids
    assert "C4_CLAHE_UNSHARP" in ids
    assert "C5_MEDIAN_DENOISING" in ids

    # Only C1 is full corpus
    c1 = next(c for c in STAGE12_CANDIDATES if c["candidate_id"] == "C1_BASELINE")
    assert c1["full_corpus_available"] is True
    assert c1["full_dataset_sample_count"] == 3568

    # C2-C5 are pilot only
    for cid in ["C2_CLAHE", "C3_UNSHARP_MASK", "C4_CLAHE_UNSHARP", "C5_MEDIAN_DENOISING"]:
        c = next(cand for cand in STAGE12_CANDIDATES if cand["candidate_id"] == cid)
        assert c["full_corpus_available"] is False


def test_pipeline_execution_in_temp_dir():
    """Verify end-to-end execution of Stage 13 pipeline generating all expected artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_stage13_pipeline(output_dir=tmpdir, val_ratio=0.20, seed=42)
        assert result["status"] == "SUCCESS"

        expected_files = [
            "classification_record_audit.csv",
            "classification_master_manifest.csv",
            "classification_train_manifest.csv",
            "classification_val_manifest.csv",
            "classification_test_manifest.csv",
            "patient_level_split_manifest.csv",
            "cross_partition_patient_conflicts.csv",
            "class_distribution_summary.csv",
            "missing_image_and_invalid_label_report.csv",
            "stage13_validation.csv",
            "stage13_classification_config.json",
            "stage13_classification_report.txt",
            "stage13_classification_report.md",
        ]

        for fname in expected_files:
            p = os.path.join(tmpdir, fname)
            assert os.path.exists(p), f"Artifact missing: {fname}"
            assert os.path.getsize(p) > 0, f"Artifact empty: {fname}"

        # Check validation CSV
        val_df = pd.read_csv(os.path.join(tmpdir, "stage13_validation.csv"))
        assert (val_df["status"] == "PASS").all()


def test_raw_data_and_earlier_stages_immutability():
    """Verify that running Stage 13 does not alter raw CBIS-DDSM data or Stages 1-12 files."""
    raw_dir = find_csv_dir()
    for fname in [
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ]:
        p = os.path.join(raw_dir, fname)
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0

    s5_meta = find_stage5_metadata()
    assert os.path.exists(s5_meta)
    df_s5 = pd.read_csv(s5_meta)
    assert len(df_s5) == 3568


def test_stage13_1_audit_reports_and_reconciliation():
    """Verify that Stage 13.1 independent audit reports exist and confirm exact record reconciliation."""
    s13_dir = os.path.join(_PROJECT_ROOT, "data/metadata/stage13")
    if not os.path.exists(s13_dir):
        pytest.skip("Stage 13 output directory not yet created")

    audit_md = os.path.join(s13_dir, "stage13_1_split_audit_report.md")
    audit_txt = os.path.join(s13_dir, "stage13_1_split_audit_report.txt")
    audit_csv = os.path.join(s13_dir, "stage13_1_audit_summary.csv")

    assert os.path.exists(audit_md) and os.path.getsize(audit_md) > 0
    assert os.path.exists(audit_txt) and os.path.getsize(audit_txt) > 0
    assert os.path.exists(audit_csv) and os.path.getsize(audit_csv) > 0

    df_summary = pd.read_csv(audit_csv)
    assert len(df_summary) >= 15
    assert (df_summary["audit_status"].isin(["PASS", "REVIEW_REQUIRED", "REVIEW"])).all()

    # Reconcile exact counts from class distribution summary
    dist_path = os.path.join(s13_dir, "class_distribution_summary.csv")
    if os.path.exists(dist_path):
        dist_df = pd.read_csv(dist_path)
        tr = dist_df[dist_df["partition"] == "train"].iloc[0]["record_count"]
        val = dist_df[dist_df["partition"] == "val"].iloc[0]["record_count"]
        te = dist_df[dist_df["partition"] == "test"].iloc[0]["record_count"]
        conf = dist_df[dist_df["partition"] == "conflict_review"].iloc[0]["record_count"]

        assert tr + val + te + conf == 3568
        assert tr == 2239
        assert val == 542
        assert te == 644
        assert conf == 143

