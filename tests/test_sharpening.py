"""Unit and integration tests for Stage 7: Sharpening Experiment."""

import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sharpening import (
    apply_unsharp_mask,
    unsharp_mask,
    apply_laplacian_sharpening,
    compute_image_statistics,
    compute_edge_statistics,
    compute_difference_metrics,
    load_sharpening_config,
    load_stage6_inputs,
    run_image_sharpening_stage7,
    generate_sharpening_summary_report,
    CONTRAST_BASELINE,
    CONTRAST_HIST_EQ,
    CONTRAST_CLAHE,
    SHARPENING_NONE,
    SHARPENING_UNSHARP,
    STATUS_SUCCESS,
    STATUS_PASS,
)
from src.preprocessing.sharpening_visualization import (
    group_sharpening_metadata,
    generate_sharpening_comparison_plot,
    generate_sharpening_difference_maps_plot,
    generate_edge_comparison_plot,
    run_sharpening_visualization,
)


def test_apply_unsharp_mask_basic():
    """Verify unsharp mask preserves float32 range [0, 1] without NaNs or Infs."""
    img = np.linspace(0.1, 0.9, 100, dtype=np.float32).reshape((10, 10))
    sharpened = apply_unsharp_mask(img, sigma=1.0, amount=1.0, threshold=0.0)
    assert sharpened.dtype == np.float32
    assert sharpened.min() >= 0.0
    assert sharpened.max() <= 1.0
    assert not np.isnan(sharpened).any()
    assert not np.isinf(sharpened).any()
    assert sharpened.shape == img.shape


def test_apply_unsharp_mask_clipping():
    """Verify extreme sharpening amount never overflows or underflows [0, 1]."""
    img = np.zeros((30, 30), dtype=np.float32)
    img[10:20, 10:20] = 1.0
    sharpened = apply_unsharp_mask(img, sigma=1.0, amount=5.0, threshold=0.0)
    assert sharpened.min() >= 0.0
    assert sharpened.max() <= 1.0


def test_apply_unsharp_mask_threshold_behavior():
    """Verify threshold > 0 suppresses sharpening on low-detail regions."""
    # Flat image with subtle variation and one strong step edge
    img = np.full((40, 40), 0.5, dtype=np.float32)
    img[10:15, 10:15] = 0.51  # subtle detail (diff = 0.01)
    img[25:35, 25:35] = 0.90  # strong detail (diff = 0.40)

    # With threshold = 0.05, subtle region should NOT be sharpened
    sharp_thresh = apply_unsharp_mask(img, sigma=1.0, amount=1.0, threshold=0.05)
    sharp_zero = apply_unsharp_mask(img, sigma=1.0, amount=1.0, threshold=0.0)

    # Subtle area should remain identical to original when threshold > subtle difference
    assert abs(float(sharp_thresh[12, 12]) - float(img[12, 12])) < 1e-6
    # But with threshold = 0, subtle area was modified
    assert abs(float(sharp_zero[12, 12]) - float(img[12, 12])) > 1e-5
    # Strong edge area is sharpened in both cases (at step boundary [25, 28])
    assert abs(float(sharp_thresh[25, 28]) - float(img[25, 28])) > 1e-3
    assert abs(float(sharp_zero[25, 28]) - float(img[25, 28])) > 1e-3


def test_apply_laplacian_sharpening():
    """Verify legacy Laplacian edge enhancement behaves predictably."""
    img = np.full((20, 20), 0.5, dtype=np.float32)
    out = apply_laplacian_sharpening(img, strength=0.5)
    assert out.shape == img.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_compute_image_statistics():
    """Verify image quality and saturation metric calculations."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[:20, :] = 0      # 20% near zero (0 <= 2.55)
    arr[20:80, :] = 128  # 60% mid-range
    arr[80:, :] = 255    # 20% near one (255 >= 252.45)

    stats = compute_image_statistics(arr, near_zero_thresh=0.01, near_one_thresh=0.99)
    assert stats["min"] == 0.0
    assert stats["max"] == 255.0
    assert abs(stats["near_zero_percentage"] - 20.0) < 1e-2
    assert abs(stats["near_one_percentage"] - 20.0) < 1e-2
    assert stats["entropy"] > 0.0
    assert stats["median"] == 128.0


def test_compute_edge_statistics():
    """Verify Sobel gradient magnitude and strong-edge percentage calculations."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[:, 32:] = 200  # Step edge down the center
    stats, mag = compute_edge_statistics(arr, sobel_ksize=3, strong_edge_threshold=25.5)

    assert stats["mean_gradient_magnitude"] > 0.0
    assert stats["std_gradient_magnitude"] > 0.0
    assert stats["strong_edge_percentage"] > 0.0
    assert mag.shape == (64, 64)


def test_compute_difference_metrics():
    """Verify difference metrics between sharpened and control."""
    ctrl = np.full((50, 50), 100, dtype=np.uint8)
    sharp = np.full((50, 50), 110, dtype=np.uint8)
    diff = compute_difference_metrics(sharp, ctrl)

    assert abs(diff["mean_absolute_difference"] - 10.0) < 1e-3
    assert abs(diff["max_absolute_difference"] - 10.0) < 1e-3
    assert abs(diff["std_absolute_difference"] - 0.0) < 1e-3


def test_load_sharpening_config():
    """Verify configuration loading with default values."""
    cfg = load_sharpening_config()
    assert cfg["enabled"] is True
    assert cfg["unsharp_mask"]["sigma"] == 1.0
    assert cfg["unsharp_mask"]["amount"] == 1.0
    assert cfg["unsharp_mask"]["threshold"] == 0.0
    assert SHARPENING_UNSHARP in cfg["methods"]
    assert SHARPENING_NONE in cfg["methods"]


@pytest.fixture
def mock_stage7_env(tmp_path):
    """Create a fully contained mock Stage-6 environment for testing Stage 7."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    proc_base = tmp_path / "processed" / "baseline" / "full_mammogram"
    proc_hist = tmp_path / "processed" / "contrast" / "histogram_equalization"
    proc_clahe = tmp_path / "processed" / "contrast" / "clahe"
    sharp_out = tmp_path / "processed" / "sharpening"

    for d in (proc_base, proc_hist, proc_clahe, sharp_out):
        d.mkdir(parents=True, exist_ok=True)

    # 2 mock cases: P_00004 (LEFT CC) and P_00009 (RIGHT MLO)
    cases_def = [
        ("P_00004", 1, "mass", "LEFT", "CC", "BENIGN", 0, "train", (600, 400)),
        ("P_00009", 1, "mass", "RIGHT", "MLO", "MALIGNANT", 1, "train", (500, 350)),
    ]

    meta_rows = []
    val_rows = []

    for pid, abn, cat, side, view, path, lbl, split, (h, w) in cases_def:
        # Generate non-flat synthetic images with gradient, edges, multiple intensity levels, and local detail
        y_coords = np.linspace(30, 200, h, dtype=np.float32)[:, None]
        x_coords = np.linspace(30, 200, w, dtype=np.float32)[None, :]
        img_b = ((x_coords + y_coords) / 2.0).astype(np.uint8)

        # Clear edge and structural region
        img_b[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = (
            img_b[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] // 2
        ) + 70
        # High contrast internal mass
        img_b[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)] = 195
        # Local fine detail
        img_b[int(h*0.45):int(h*0.55), int(w*0.45):int(w*0.55)] = 220
        pb = proc_base / f"{pid}_{side}_{view}_{abn}_baseline.png"
        cv2.imwrite(str(pb), img_b)

        img_h = cv2.equalizeHist(img_b)
        ph = proc_hist / f"{pid}_{side}_{view}_{abn}_histeq.png"
        cv2.imwrite(str(ph), img_h)

        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_c = clahe_obj.apply(img_b)
        pc = proc_clahe / f"{pid}_{side}_{view}_{abn}_clahe.png"
        cv2.imwrite(str(pc), img_c)

        methods_info = [
            (CONTRAST_BASELINE, str(pb)),
            (CONTRAST_HIST_EQ, str(ph)),
            (CONTRAST_CLAHE, str(pc)),
        ]

        for m_name, m_path in methods_info:
            meta_rows.append({
                "patient_id": pid,
                "abnormality_id": abn,
                "category": cat,
                "side": side,
                "view": view,
                "pathology": path,
                "label": lbl,
                "split": split,
                "method": m_name,
                "output_image_path": m_path,
                "original_height": h,
                "original_width": w,
                "output_height": h,
                "output_width": w,
                "status": STATUS_SUCCESS,
                "notes": f"Stage 6 {m_name}",
            })
            val_rows.append({
                "patient_id": pid,
                "abnormality_id": abn,
                "method": m_name,
                "input_exists": True,
                "output_exists": True,
                "shape_match": True,
                "status": STATUS_PASS,
                "notes": "Verified PASS",
            })

    df_m = pd.DataFrame(meta_rows)
    df_v = pd.DataFrame(val_rows)

    m_csv = meta_dir / "contrast_preprocessing_metadata.csv"
    v_csv = meta_dir / "contrast_preprocessing_validation.csv"
    df_m.to_csv(m_csv, index=False)
    df_v.to_csv(v_csv, index=False)

    return {
        "meta_csv": str(m_csv),
        "val_csv": str(v_csv),
        "sharp_out": str(sharp_out),
        "meta_dir": str(meta_dir),
    }


def test_stage7_end_to_end_mock_pipeline(mock_stage7_env, tmp_path):
    """Verify complete 3x2 factorial execution across mock cases."""
    m_csv = mock_stage7_env["meta_csv"]
    v_csv = mock_stage7_env["val_csv"]
    out_dir = mock_stage7_env["sharp_out"]

    out_meta_csv = str(tmp_path / "metadata" / "sharpening_preprocessing_metadata.csv")
    out_val_csv = str(tmp_path / "metadata" / "sharpening_preprocessing_validation.csv")
    out_rep_txt = str(tmp_path / "metadata" / "sharpening_preprocessing_report.txt")

    df_meta, df_val = run_image_sharpening_stage7(
        stage6_meta_path=m_csv,
        stage6_val_path=v_csv,
        output_base_dir=out_dir,
        metadata_output_csv=out_meta_csv,
        validation_output_csv=out_val_csv,
        report_output_txt=out_rep_txt,
    )

    # 2 cases x 3 contrast conditions x 2 sharpening conditions = 12 records
    assert len(df_meta) == 12
    assert len(df_val) == 12

    # Check that both files exist
    assert os.path.exists(out_meta_csv)
    assert os.path.exists(out_val_csv)
    assert os.path.exists(out_rep_txt)

    # Verify all records passed validation
    assert (df_val["status"] == STATUS_PASS).all()
    assert (df_meta["status"] == STATUS_SUCCESS).all()

    # Verify required columns exist
    expected_meta_cols = [
        "patient_id", "abnormality_id", "category", "side", "view", "pathology",
        "label", "split", "contrast_method", "sharpening_method",
        "input_image_path", "output_image_path", "original_height", "original_width",
        "output_height", "output_width", "sigma", "amount", "threshold",
        "input_min", "input_max", "input_mean", "input_std", "input_median",
        "input_p01", "input_p99", "output_min", "output_max", "output_mean",
        "output_std", "output_median", "output_p01", "output_p99", "entropy",
        "near_zero_percentage", "near_one_percentage", "mean_gradient_magnitude",
        "std_gradient_magnitude", "strong_edge_percentage", "mean_absolute_difference",
        "max_absolute_difference", "std_absolute_difference", "status", "notes",
    ]
    for col in expected_meta_cols:
        assert col in df_meta.columns

    # Verify control records referenced existing Stage-6 paths without duplicate files
    controls = df_meta[df_meta["sharpening_method"] == SHARPENING_NONE]
    assert len(controls) == 6
    for _, r in controls.iterrows():
        assert r["input_image_path"] == r["output_image_path"]
        assert r["mean_absolute_difference"] == 0.0

    # Verify sharpened outputs were created in expected subdirectories
    sharpened = df_meta[df_meta["sharpening_method"] == SHARPENING_UNSHARP]
    assert len(sharpened) == 6
    for _, r in sharpened.iterrows():
        out_p = r["output_image_path"]
        assert os.path.exists(out_p)
        assert "unsharp" in out_p
        # Image dimensions match input
        im = cv2.imread(out_p, cv2.IMREAD_GRAYSCALE)
        assert im.shape == (r["original_height"], r["original_width"])

    # Verify that unsharp output contains non-zero pixel differences overall
    assert (sharpened["mean_absolute_difference"] > 0.0).any()
    assert (sharpened["max_absolute_difference"] > 0.0).any()


def test_stage7_visualization_generation(mock_stage7_env, tmp_path):
    """Verify visualization generation creates all expected PNG figures."""
    m_csv = mock_stage7_env["meta_csv"]
    v_csv = mock_stage7_env["val_csv"]
    out_dir = mock_stage7_env["sharp_out"]

    out_meta_csv = str(tmp_path / "metadata" / "sharpening_preprocessing_metadata.csv")
    out_val_csv = str(tmp_path / "metadata" / "sharpening_preprocessing_validation.csv")
    out_rep_txt = str(tmp_path / "metadata" / "sharpening_preprocessing_report.txt")

    df_meta, _ = run_image_sharpening_stage7(
        stage6_meta_path=m_csv,
        stage6_val_path=v_csv,
        output_base_dir=out_dir,
        metadata_output_csv=out_meta_csv,
        validation_output_csv=out_val_csv,
        report_output_txt=out_rep_txt,
    )

    cases = group_sharpening_metadata(df_meta)
    assert len(cases) == 2

    viz_dir = str(tmp_path / "results" / "sharpening")

    p_comp_std, p_comp_hr = generate_sharpening_comparison_plot(cases, viz_dir, CONTRAST_CLAHE)
    assert os.path.exists(p_comp_std)
    assert os.path.exists(p_comp_hr)

    p_diff = generate_sharpening_difference_maps_plot(cases, viz_dir)
    assert os.path.exists(p_diff)

    p_edge = generate_edge_comparison_plot(cases, viz_dir, CONTRAST_CLAHE)
    assert os.path.exists(p_edge)
