"""Unit and integration tests for Stage 6: Contrast Enhancement Experiment."""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.contrast_enhancement import (
    load_contrast_config,
    load_stage5_metadata,
    select_contrast_pilot_samples,
    compute_image_statistics,
    apply_histogram_equalization,
    apply_clahe,
    process_case_contrast_methods,
    generate_contrast_summary_report,
    run_contrast_enhancement,
    METHOD_BASELINE,
    METHOD_HIST_EQ,
    METHOD_CLAHE,
    STATUS_SUCCESS,
)
from src.preprocessing.contrast_visualization import (
    group_metadata_by_case,
    generate_original_vs_methods_plot,
    generate_contrast_histogram_comparison,
    generate_difference_maps_plot,
    run_contrast_visualization,
)


@pytest.fixture
def mock_stage6_env(tmp_path):
    """Create a mock Stage 6 environment with Stage 5 baseline images and metadata."""
    s5_dir = tmp_path / "metadata" / "stage5"
    base_out = tmp_path / "processed" / "baseline" / "full_mammogram"
    contrast_out = tmp_path / "processed" / "contrast"
    meta_out = tmp_path / "metadata"
    res_dir = tmp_path / "results" / "contrast"

    for d in (s5_dir, base_out, contrast_out, meta_out, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create 2 synthetic Stage-5 baseline PNGs (uint8 [0, 255])
    # 1. Normal breast simulation (1200x1000)
    img1 = np.zeros((1200, 1000), dtype=np.uint8)
    img1[200:1000, 200:800] = 120
    img1[400:700, 400:600] = 200  # denser glandular tissue
    p1 = base_out / "P_00004_LEFT_CC_1_baseline.png"
    cv2.imwrite(str(p1), img1)

    # 2. Right MLO simulation (1100x900)
    img2 = np.zeros((1100, 900), dtype=np.uint8)
    img2[150:950, 100:700] = 100
    img2[300:600, 250:550] = 180
    p2 = base_out / "P_00009_RIGHT_MLO_1_baseline.png"
    cv2.imwrite(str(p2), img2)

    # Create mock Stage-5 metadata
    meta_rows = [
        {
            "patient_id": "P_00004",
            "abnormality_id": 1,
            "abnormality_category": "mass",
            "breast_side": "LEFT",
            "image_view": "CC",
            "pathology": "BENIGN",
            "label": 0,
            "dataset_split": "train",
            "original_image_path": "raw/p4.jpg",
            "baseline_image_path": str(p1),
            "original_width": 1000,
            "original_height": 1200,
            "baseline_width": 1000,
            "baseline_height": 1200,
            "processing_status": "SUCCESS",
        },
        {
            "patient_id": "P_00009",
            "abnormality_id": 1,
            "abnormality_category": "mass",
            "breast_side": "RIGHT",
            "image_view": "MLO",
            "pathology": "MALIGNANT",
            "label": 1,
            "dataset_split": "train",
            "original_image_path": "raw/p9.jpg",
            "baseline_image_path": str(p2),
            "original_width": 900,
            "original_height": 1100,
            "baseline_width": 900,
            "baseline_height": 1100,
            "processing_status": "SUCCESS",
        },
    ]
    df_s5 = pd.DataFrame(meta_rows)
    df_s5.to_csv(s5_dir / "baseline_preprocessing_metadata.csv", index=False)

    val_rows = [
        {"patient_id": "P_00004", "abnormality_id": 1, "validation_status": "PASS"},
        {"patient_id": "P_00009", "abnormality_id": 1, "validation_status": "PASS"},
    ]
    pd.DataFrame(val_rows).to_csv(s5_dir / "baseline_preprocessing_validation.csv", index=False)

    return {
        "s5_dir": str(s5_dir),
        "base_out": str(base_out),
        "contrast_out": str(contrast_out),
        "meta_out": str(meta_out),
        "res_dir": str(res_dir),
        "p1": str(p1),
        "p2": str(p2),
    }


def test_load_contrast_config():
    """Test configuration loading and default parameter structure."""
    cfg = load_contrast_config()
    assert cfg["enabled"] is True
    assert "methods" in cfg
    assert METHOD_BASELINE in cfg["methods"]
    assert METHOD_HIST_EQ in cfg["methods"]
    assert METHOD_CLAHE in cfg["methods"]
    assert cfg["clahe"]["clip_limit"] == 2.0
    assert cfg["clahe"]["tile_grid_size"]["width"] == 8
    assert cfg["clahe"]["tile_grid_size"]["height"] == 8


def test_apply_histogram_equalization():
    """Test global histogram equalization preserves dimensions and increases dynamic range."""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[50:150, 50:250] = 80
    eq = apply_histogram_equalization(img)

    assert eq.shape == (200, 300)
    assert eq.dtype == np.uint8
    assert eq.max() > img.max()


def test_apply_clahe():
    """Test CLAHE enhancement preserves dimensions and honors parameters."""
    img = np.zeros((400, 300), dtype=np.uint8)
    img[50:350, 50:250] = 100
    img[100:200, 100:200] = 150

    cl = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    assert cl.shape == (400, 300)
    assert cl.dtype == np.uint8
    assert cl.max() >= img.max()


def test_compute_image_statistics():
    """Test calculation of comprehensive descriptive intensity and quality statistics."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 128
    stats = compute_image_statistics(img)

    assert stats["min"] == 0.0
    assert stats["max"] == 128.0
    assert 0.0 < stats["mean"] < 128.0
    assert stats["std"] > 0.0
    assert "entropy" in stats
    assert stats["entropy"] > 0.0
    assert "breast_region_mean" in stats
    assert stats["breast_region_mean"] == 128.0
    assert "p01" in stats
    assert "p99" in stats


def test_process_case_contrast_methods_end_to_end(mock_stage6_env):
    """Test end-to-end processing of a case across Baseline, HistEq, and CLAHE."""
    cfg = load_contrast_config()
    sample = {
        "patient_id": "P_00004",
        "abnormality_id": 1,
        "abnormality_category": "mass",
        "breast_side": "LEFT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "label": 0,
        "dataset_split": "train",
        "baseline_image_path": mock_stage6_env["p1"],
    }
    meta_recs, val_recs = process_case_contrast_methods(
        sample_info=sample,
        contrast_output_root=mock_stage6_env["contrast_out"],
        config=cfg,
    )

    assert len(meta_recs) == 3
    assert len(val_recs) == 3

    methods_present = [r["method"] for r in meta_recs]
    assert METHOD_BASELINE in methods_present
    assert METHOD_HIST_EQ in methods_present
    assert METHOD_CLAHE in methods_present

    for vr in val_recs:
        assert vr["status"] == "PASS"
        assert vr["shape_match"] is True
        assert vr["output_exists"] is True


def test_run_contrast_enhancement_pilot(mock_stage6_env):
    """Test running complete contrast enhancement pipeline in pilot mode."""
    res = run_contrast_enhancement(
        stage5_metadata_dir=mock_stage6_env["s5_dir"],
        contrast_output_root=mock_stage6_env["contrast_out"],
        metadata_output_dir=mock_stage6_env["meta_out"],
        force_pilot=True,
    )

    assert res["num_input_cases"] == 2
    assert res["total_records"] == 6
    assert res["success_count"] == 6
    assert res["fail_count"] == 0

    assert os.path.exists(res["metadata_csv"])
    assert os.path.exists(res["validation_csv"])
    assert os.path.exists(res["report_txt"])


def test_contrast_visualizations(mock_stage6_env):
    """Test generating contact sheets, histogram distributions, and difference maps."""
    # Run enhancement to generate outputs
    run_contrast_enhancement(
        stage5_metadata_dir=mock_stage6_env["s5_dir"],
        contrast_output_root=mock_stage6_env["contrast_out"],
        metadata_output_dir=mock_stage6_env["meta_out"],
        force_pilot=True,
    )

    meta_csv = os.path.join(mock_stage6_env["meta_out"], "contrast_preprocessing_metadata.csv")
    viz_res = run_contrast_visualization(
        metadata_path=meta_csv,
        output_dir=mock_stage6_env["res_dir"],
    )

    assert os.path.exists(viz_res["original_vs_methods"])
    assert os.path.exists(viz_res["original_vs_methods_highres"])
    assert os.path.exists(viz_res["histogram_comparison"])
    assert os.path.exists(viz_res["difference_maps"])
