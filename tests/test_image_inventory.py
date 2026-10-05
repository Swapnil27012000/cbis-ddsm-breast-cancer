"""Unit tests for Stage 2: CBIS-DDSM Image Inventory and Reference Mapping."""

import os
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.data.image_inventory import (
    find_jpeg_dir,
    inspect_image_file,
    scan_physical_inventory,
    resolve_case_csv_records,
    cross_reference_inventory,
    build_shared_references_data,
    build_patient_summary,
    build_image_role_summary,
    build_dimension_statistics,
    run_stage2_mapping,
    ROLE_FULL_ORIGINAL,
    ROLE_CROPPED_ABNORMALITY,
    ROLE_ROI_MASK,
    ROLE_UNKNOWN,
    STATUS_RESOLVED,
    STATUS_UNMAPPED,
    STATUS_UNRESOLVED_STATUS,
)
from src.data.path_resolver import PathResolver


@pytest.fixture
def mock_stage2_env(tmp_path):
    """Create isolated mock directory tree with raw images, CSVs, and output folder."""
    raw_dir = tmp_path / "raw"
    jpeg_dir = raw_dir / "jpeg"
    csv_dir = raw_dir / "csv"
    output_dir = tmp_path / "metadata" / "stage2"

    jpeg_dir.mkdir(parents=True)
    csv_dir.mkdir(parents=True)

    # 1. Create Patient 1 Full Mammogram Folder (1 image)
    p1_full_folder = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.111111111111"
    p1_full_folder.mkdir()
    p1_full_img = p1_full_folder / "1-137.jpg"
    img1 = Image.fromarray(np.full((100, 80), 128, dtype=np.uint8))
    img1.save(str(p1_full_img))

    # 2. Create Patient 1 Crop + Mask Pair Folder (2 images)
    p1_crop_folder = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.222222222222"
    p1_crop_folder.mkdir()
    p1_mask_img = p1_crop_folder / "1-1.jpg"
    p1_crop_img = p1_crop_folder / "1-2.jpg"
    img_mask = Image.fromarray(np.full((30, 20), 255, dtype=np.uint8))
    img_crop = Image.fromarray(np.full((30, 20), 100, dtype=np.uint8))
    img_mask.save(str(p1_mask_img))
    img_crop.save(str(p1_crop_img))

    # 3. Create Unmapped Image (exists on disk, not in CSV)
    unmapped_folder = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.999999999999"
    unmapped_folder.mkdir()
    unmapped_img = unmapped_folder / "unmapped.jpg"
    img_unmapped = Image.fromarray(np.full((50, 50), 200, dtype=np.uint8))
    img_unmapped.save(str(unmapped_img))

    # 4. Create Case CSVs
    # Case 1: Normal mass case referencing p1_full and p1_crop/mask
    # Case 2: Second abnormality for P_00001 referencing SAME full mammo (shared reference)
    mass_train_df = pd.DataFrame([
        {
            "patient_id": "P_00001",
            "breast_density": 3,
            "left or right breast": "LEFT",
            "image view": "CC",
            "abnormality id": 1,
            "abnormality type": "mass",
            "mass shape": "ROUND",
            "mass margins": "CIRCUMSCRIBED",
            "assessment": 3,
            "pathology": "BENIGN",
            "subtlety": 4,
            "image file path": "1.3.6.1.4.1.9590.100.1.2.111111111111/1-137.jpg",
            "cropped image file path": "1.3.6.1.4.1.9590.100.1.2.222222222222/000000.dcm",
            "ROI mask file path": "1.3.6.1.4.1.9590.100.1.2.222222222222/000001.dcm",
        },
        {
            "patient_id": "P_00001",
            "breast_density": 3,
            "left or right breast": "LEFT",
            "image view": "CC",
            "abnormality id": 2,
            "abnormality type": "mass",
            "mass shape": "OVAL",
            "mass margins": "CIRCUMSCRIBED",
            "assessment": 3,
            "pathology": "BENIGN",
            "subtlety": 4,
            "image file path": "1.3.6.1.4.1.9590.100.1.2.111111111111/1-137.jpg",
            "cropped image file path": "1.3.6.1.4.1.9590.100.1.2.222222222222/000000.dcm",
            "ROI mask file path": "1.3.6.1.4.1.9590.100.1.2.222222222222/000001.dcm",
        },
    ])
    mass_train_df.to_csv(csv_dir / "mass_case_description_train_set.csv", index=False)

    # Empty test set
    empty_mass_test = pd.DataFrame(columns=mass_train_df.columns)
    empty_mass_test.to_csv(csv_dir / "mass_case_description_test_set.csv", index=False)

    # Calc train set: includes P_01563 known exception
    calc_train_df = pd.DataFrame([
        {
            "patient_id": "P_01563",
            "breast density": 2,
            "left or right breast": "RIGHT",
            "image view": "MLO",
            "abnormality id": 2,
            "abnormality type": "calcification",
            "calc type": "PLEOMORPHIC",
            "calc distribution": "SEGMENTAL",
            "assessment": 4,
            "pathology": "MALIGNANT",
            "subtlety": 3,
            "image file path": "1.3.6.1.4.1.9590.100.1.2.111111111111/1-137.jpg",
            "cropped image file path": "Calc-Training_P_01563_RIGHT_MLO_2/1.3.6.1.4.1.9590.100.1.2.348822970413183698610798947061334416506/000001.dcm",
            "ROI mask file path": "Calc-Training_P_01563_RIGHT_MLO_2/1.3.6.1.4.1.9590.100.1.2.348822970413183698610798947061334416506/000000.dcm",
        }
    ])
    calc_train_df.to_csv(csv_dir / "calc_case_description_train_set.csv", index=False)

    # Empty calc test set
    empty_calc_test = pd.DataFrame(columns=calc_train_df.columns)
    empty_calc_test.to_csv(csv_dir / "calc_case_description_test_set.csv", index=False)

    # Minimal dicom_info.csv and meta.csv
    pd.DataFrame({"image_path": [], "SeriesInstanceUID": [], "SOPInstanceUID": []}).to_csv(
        csv_dir / "dicom_info.csv", index=False
    )
    pd.DataFrame({"Series UID": []}).to_csv(csv_dir / "meta.csv", index=False)

    return {
        "raw_dir": str(raw_dir),
        "jpeg_dir": str(jpeg_dir),
        "csv_dir": str(csv_dir),
        "output_dir": str(output_dir),
    }


def test_physical_image_scanning(mock_stage2_env):
    """Test recursive physical JPEG scanning and metadata extraction."""
    jpeg_dir = mock_stage2_env["jpeg_dir"]
    inventory = scan_physical_inventory(jpeg_dir)

    # 4 images created in mock (p1_full, p1_mask, p1_crop, unmapped)
    assert len(inventory) == 4
    for idx, item in enumerate(inventory, start=1):
        assert item["image_id"] == f"IMG_{idx:06d}"
        assert item["decodable"] is True
        assert item["readability_status"] == "READABLE"
        assert item["width"] > 0
        assert item["height"] > 0
        assert item["file_size_bytes"] > 0
        assert "/" in item["relative_image_path"] or "\\" not in item["relative_image_path"]


def test_reference_mapping_and_p01563_exception(mock_stage2_env):
    """Test resolution of CSV references including P_01563 exception."""
    raw_dir = mock_stage2_env["raw_dir"]
    csv_dir = mock_stage2_env["csv_dir"]
    resolver = PathResolver(raw_dir)

    mapping_rows, stats = resolve_case_csv_records(csv_dir, resolver)

    # 3 total case rows (2 in mass_train, 1 in calc_train)
    assert stats["total_case_records"] == 3
    assert stats["total_references"] == 9  # 3 * 3

    # P_01563 verification
    p_status = stats["p01563_status"]
    assert p_status[ROLE_FULL_ORIGINAL] == STATUS_RESOLVED
    assert p_status[ROLE_CROPPED_ABNORMALITY] == STATUS_UNRESOLVED_STATUS
    assert p_status[ROLE_ROI_MASK] == STATUS_UNRESOLVED_STATUS


def test_shared_and_unmapped_images(mock_stage2_env):
    """Test identifying shared images and unmapped physical files."""
    raw_dir = mock_stage2_env["raw_dir"]
    jpeg_dir = mock_stage2_env["jpeg_dir"]
    csv_dir = mock_stage2_env["csv_dir"]

    resolver = PathResolver(raw_dir)
    inventory = scan_physical_inventory(jpeg_dir)
    mapping_rows, _ = resolve_case_csv_records(csv_dir, resolver)
    enriched_inv, path_to_refs, unmapped = cross_reference_inventory(inventory, mapping_rows)

    # Unmapped image detection (1 unmapped image)
    assert len(unmapped) == 1
    assert unmapped[0]["mapping_status"] == STATUS_UNMAPPED
    assert "unmapped.jpg" in unmapped[0]["filename"]

    # Shared image detection (p1_full referenced by Case 1, Case 2, and P_01563 = 3 times)
    shared = build_shared_references_data(path_to_refs)
    assert len(shared) >= 1
    top_shared = [s for s in shared if "1-137.jpg" in s["resolved_image_path"]][0]
    assert top_shared["reference_count"] == 3
    assert top_shared["patient_count"] == 2  # P_00001 and P_01563


def test_complete_stage2_workflow_artifacts(mock_stage2_env):
    """Test full execution of run_stage2_mapping and verify all 9 artifacts exist."""
    raw_dir = mock_stage2_env["raw_dir"]
    csv_dir = mock_stage2_env["csv_dir"]
    output_dir = mock_stage2_env["output_dir"]

    results = run_stage2_mapping(
        raw_data_dir=raw_dir,
        raw_csv_dir=csv_dir,
        output_dir=output_dir,
    )

    required_files = [
        "CBIS_DDSM_image_inventory.csv",
        "CBIS_DDSM_reference_mapping.csv",
        "image_role_summary.csv",
        "image_mapping_report.txt",
        "shared_image_references.csv",
        "unmapped_jpeg_files.csv",
        "patient_image_mapping_summary.csv",
        "image_readability_report.csv",
        "image_dimension_statistics.csv",
        "stage1_stage2_mapping_comparison.csv",
        "stage1_stage2_mapping_comparison_report.txt",
    ]

    for rf in required_files:
        p = os.path.join(output_dir, rf)
        assert os.path.exists(p), f"Artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Artifact empty: {p}"

    # Verify column structure of inventory CSV
    df_inv = pd.read_csv(os.path.join(output_dir, "CBIS_DDSM_image_inventory.csv"))
    assert "image_id" in df_inv.columns
    assert "relative_image_path" in df_inv.columns
    assert "reference_count" in df_inv.columns
    assert "mapping_status" in df_inv.columns

    # Verify column structure of reference mapping CSV
    df_ref = pd.read_csv(os.path.join(output_dir, "CBIS_DDSM_reference_mapping.csv"))
    assert "original_resolved_path" in df_ref.columns
    assert "original_mapping_status" in df_ref.columns
    assert "cropped_resolved_path" in df_ref.columns
    assert "cropped_mapping_status" in df_ref.columns
    assert "roi_mask_resolved_path" in df_ref.columns
    assert "roi_mask_mapping_status" in df_ref.columns

    # Verify comparison CSV
    df_comp = pd.read_csv(os.path.join(output_dir, "stage1_stage2_mapping_comparison.csv"))
    assert "stage1_status" in df_comp.columns
    assert "stage2_status" in df_comp.columns
    assert "status_match" in df_comp.columns
    assert "path_match" in df_comp.columns
    assert df_comp["status_match"].all()
