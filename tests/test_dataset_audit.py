"""Unit tests for Stage 1 dataset structure audit module."""
import os
import pytest
from src.data.dataset_audit import (
    audit_jpeg_directory,
    generate_audit_csv,
    generate_audit_text_report,
    generate_unresolved_references_csv,
    generate_reference_summary_csv,
    generate_unresolved_diagnostics_txt,
)


def test_audit_jpeg_directory_mock(tmp_path):
    jpeg_dir = tmp_path / "jpeg"
    jpeg_dir.mkdir()

    # Case folder 1: 1 image
    folder1 = jpeg_dir / "series1"
    folder1.mkdir()
    (folder1 / "1-1.jpg").write_bytes(b"dummy")

    # Case folder 2: 2 images (crop + mask)
    folder2 = jpeg_dir / "series2"
    folder2.mkdir()
    (folder2 / "1-1.jpg").write_bytes(b"dummy")
    (folder2 / "1-2.jpg").write_bytes(b"dummy")

    # Case folder 3: empty
    folder3 = jpeg_dir / "series3"
    folder3.mkdir()

    res = audit_jpeg_directory(str(jpeg_dir), sample_size=10)
    assert res["total_dirs"] == 4  # root + 3 subfolders
    assert res["total_files"] == 3
    assert res["folder_buckets"]["1_image"] == 1
    assert res["folder_buckets"]["2_images"] == 1
    assert res["folder_buckets"]["0_images"] == 2  # root + series3
    assert res["duplicate_filenames_count"] >= 1  # 1-1.jpg in both series1 and series2


def test_generate_audit_reports(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()

    dir_audit = {
        "jpeg_dir": str(tmp_path),
        "total_dirs": 5,
        "total_files": 10,
        "folder_buckets": {"0_images": 1, "1_image": 2, "2_images": 2, "3_images": 0, "4+_images": 0},
        "min_images_per_folder": 1,
        "max_images_per_folder": 2,
        "avg_images_per_folder": 1.5,
        "median_images_per_folder": 1.5,
        "depth_stats": {"min_depth": 0, "max_depth": 2, "avg_depth": 1.2},
        "duplicate_filenames_count": 2,
        "duplicate_paths_count": 0,
        "sample_dimensions": [],
        "extension_counts": {".jpg": 10},
    }
    linkage_audit = {
        "total_csv_records": 10,
        "records_with_resolved_image": 10,
        "records_fully_resolved": 10,
        "total_image_references": 20,
        "total_resolved_references": 20,
        "total_unresolved_references": 0,
        "reference_stats": {
            "full_mammogram": {"total": 10, "resolved": 10, "unresolved": 0},
            "cropped_image": {"total": 5, "resolved": 5, "unresolved": 0},
            "roi_mask": {"total": 5, "resolved": 5, "unresolved": 0},
        },
        "counts_by_status": {
            "exact": 0,
            "normalized": 20,
            "unique_filename": 0,
            "ambiguous": 0,
            "unresolved": 0,
        },
    }

    csv_path = str(stage1_dir / "dataset_structure_audit.csv")
    txt_path = str(stage1_dir / "dataset_structure_audit_report.txt")

    df = generate_audit_csv(dir_audit, linkage_audit, output_path=csv_path)
    assert os.path.exists(csv_path)
    assert len(df) > 10

    report = generate_audit_text_report(dir_audit, linkage_audit, output_path=txt_path)
    assert os.path.exists(txt_path)
    assert "CBIS-DDSM DATASET STRUCTURE & RESOLUTION AUDIT REPORT" in report

    # Test unresolved references and diagnostics generators
    unresolved_records = [{
        "source_csv": "calc_case_description_train_set.csv",
        "row_number": 1216,
        "patient_id": "P_01563",
        "abnormality_id": "2",
        "image_view": "RIGHT_MLO",
        "image_file_path": "path/mammo.dcm",
        "cropped_image_file_path": "path/crop.dcm",
        "roi_mask_file_path": "path/mask.dcm",
        "reference_type": "CROPPED_IMAGE",
        "referenced_path": "path/crop.dcm",
        "normalized_reference": "path/crop.dcm",
        "attempted_path": "/app/data/crop.dcm",
        "failure_reason": "Series absent",
        "reference_status": "UNRESOLVED",
        "reference_usable": False,
        "alternative_representation_available": True,
        "alternative_representation_path": "path/mammo.jpg",
    }]
    summary_by_csv = [{
        "source_csv": "test.csv",
        "reference_type": "full_mammogram",
        "total_references": 10,
        "resolved": 10,
        "unresolved": 0,
        "ambiguous": 0,
        "resolution_rate": "100.0%",
    }]

    unres_csv = str(stage1_dir / "dataset_unresolved_references.csv")
    unres_diag = str(stage1_dir / "dataset_unresolved_diagnostics.txt")
    ref_summary = str(stage1_dir / "dataset_reference_summary.csv")

    generate_unresolved_references_csv(unresolved_records, output_path=unres_csv)
    assert os.path.exists(unres_csv)

    generate_unresolved_diagnostics_txt(unresolved_records, output_path=unres_diag)
    assert os.path.exists(unres_diag)

    generate_reference_summary_csv(summary_by_csv, output_path=ref_summary)
    assert os.path.exists(ref_summary)
