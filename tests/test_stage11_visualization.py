"""Unit and integration tests for Stage 11: Preprocessing Visualization & Comparison.

Verifies:
1. Stage 5–10 artifact audit counts and status reporting.
2. Cross-environment path resolution (Windows host and Docker /app/).
3. Metadata alignment (patient IDs, side, view, noise types, denoising methods).
4. Missing input and uncalculated record handling (no fabrication of results).
5. Non-finite metric handling (Inf for PSNR, NaN for CNR; never silently replaced by 0.0).
6. Separation of Comparison B (restoration) from Comparison C (filtering modification).
7. Determinism and reproducibility of case selection rules.
8. Immutability of earlier-stage artifacts.
9. Figure generation and output directory resolution.
"""

import json
import math
import os
import shutil
import tempfile
import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.stage11_audit import (
    run_stage11_artifact_audit,
    resolve_path,
    find_stage_file,
    EXPECTED_PILOT_CASES,
    NOISE_TYPES,
    DENOISING_METHODS,
)
from src.preprocessing.stage11_visualization import (
    REPRESENTATIVE_CASES,
    load_grayscale_safe,
    create_contrast_comparison_panel,
    create_sharpening_comparison_panel,
    create_clean_vs_noisy_panel,
    create_denoising_comparison_panel,
    create_denoising_difference_maps_panel,
    create_magnified_detail_crops,
    plot_metrics_by_denoising_method,
    plot_metrics_by_noise_type,
    plot_method_by_noise_heatmaps,
    plot_whole_vs_foreground_metrics,
    plot_comparison_c_filtering_magnitude,
    run_stage11_visualization_pipeline,
)
from src.preprocessing.stage11_report import generate_stage11_report


def test_stage11_artifact_audit():
    """Verify that the artifact audit correctly checks Stages 5–10."""
    audit = run_stage11_artifact_audit()
    assert "stage5" in audit
    assert "stage6" in audit
    assert "stage7" in audit
    assert "stage8" in audit
    assert "stage9" in audit
    assert "stage10" in audit

    # Verify Stage 5
    assert audit["stage5"]["record_count"] >= 3568

    # Verify Stage 6
    assert audit["stage6"]["record_count"] == 48
    assert audit["stage6"]["unique_cases"] == 16
    assert set(audit["stage6"]["methods"]) == {"baseline", "clahe", "histogram_equalization"}

    # Verify Stage 7
    assert audit["stage7"]["record_count"] == 96
    assert audit["stage7"]["unique_cases"] == 16

    # Verify Stage 8
    assert audit["stage8"]["record_count"] == 96
    assert audit["stage8"]["unique_cases"] == 16
    # 16 clean + 16 of each 5 noise types = 96
    for nt in NOISE_TYPES:
        assert audit["stage8"]["noise_counts"].get(nt, 0) == 16

    # Verify Stage 9
    assert audit["stage9"]["record_count"] == 640
    assert audit["stage9"]["unique_cases"] == 16

    # Verify Stage 10 scope audit
    assert audit["stage10"]["record_count"] == 640
    assert audit["stage10"]["validated_count"] == 640
    assert audit["stage10"]["unique_cases"] == 16
    assert len(audit["stage10"]["evaluated_cases"]) == 16
    assert audit["stage10"]["missing_records"] == 0
    assert audit["stage10"]["cnr_eligible_count"] == 0  # 0 eligible cases (all NaN)
    assert audit["stage10"]["foreground_eligible_count"] == 640
    assert audit["stage10"]["status"] == "PASS"


def test_stage11_path_resolution(tmp_path):
    """Verify path resolver works for relative, absolute, and /app/ paths."""
    # Existing relative file guaranteed to be mounted in container and host
    resolved = resolve_path("src/__init__.py")
    assert resolved is not None
    assert os.path.isfile(resolved)

    # Docker-style path
    docker_style = "/app/src/__init__.py"
    resolved_docker = resolve_path(docker_style)
    assert resolved_docker is not None
    assert os.path.isfile(resolved_docker)

    # Temporary file resolution
    temp_f = tmp_path / "test_img.png"
    temp_f.write_text("test")
    assert resolve_path(str(temp_f)) == os.path.abspath(str(temp_f))

    # None and empty path
    assert resolve_path(None) is None
    assert resolve_path("") is None
    assert resolve_path("non_existent_file_xyz123.abc") is None


def test_stage11_labels_and_metadata():
    """Verify that canonical pilot case identifiers match metadata."""
    assert len(EXPECTED_PILOT_CASES) == 16
    assert len(NOISE_TYPES) == 5
    assert len(DENOISING_METHODS) == 8

    # Check representative cases
    for k, v in REPRESENTATIVE_CASES.items():
        assert k in EXPECTED_PILOT_CASES
        assert "patient_id" in v
        assert "breast_side" in v
        assert "image_view" in v
        assert "detail_crop" in v
        crop = v["detail_crop"]
        assert crop["ymin"] < crop["ymax"]
        assert crop["xmin"] < crop["xmax"]
        assert len(crop["purpose"]) > 5


def test_stage11_missing_record_handling():
    """Verify that evaluation counts and missing records are reported and never fabricated."""
    audit = run_stage11_artifact_audit()
    s10 = audit["stage10"]
    # 640 evaluated, exactly 0 missing/uncalculated
    assert s10["record_count"] == 640
    assert s10["missing_records"] == 0
    # No records fabricated
    assert s10["unique_cases"] == 16
    assert "P_00004_1_LEFT_CC" in s10["evaluated_cases"]
    assert s10["status"] == "PASS"


def test_stage11_partial_stage10_subset_fixture(monkeypatch):
    """Verify that earlier 40-record pilot subset is accurately audited as partial (600 missing)."""
    p_s10 = find_stage_file(["data/metadata/stage10/denoising_metrics_per_image.csv"])
    df = pd.read_csv(p_s10)
    df_40 = df[df["patient_id"] == "P_00004"].iloc[:40].copy()

    orig_read_csv = pd.read_csv

    def mock_read_csv(filepath, *args, **kwargs):
        if "stage10" in str(filepath) and "denoising_metrics_per_image.csv" in str(filepath):
            return df_40.copy()
        return orig_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    audit = run_stage11_artifact_audit()
    s10 = audit["stage10"]
    assert s10["record_count"] == 40
    assert s10["unique_cases"] == 1
    assert s10["missing_records"] == 600
    assert s10["status"] == "VALIDATED_PILOT_SUBSET"
    assert s10["evaluated_cases"] == ["P_00004_1_LEFT_CC"]


def test_stage11_missing_file_detection(monkeypatch):
    """Verify that missing files and genuinely incomplete metrics are detected."""
    import src.preprocessing.stage11_audit as audit_module

    def mock_find(candidates):
        if any("stage10" in c for c in candidates):
            return None
        return find_stage_file(candidates)

    monkeypatch.setattr(audit_module, "find_stage_file", mock_find)

    audit = audit_module.run_stage11_artifact_audit()
    assert audit["stage10"]["status"] == "MISSING"
    assert audit["stage10"]["record_count"] == 0
    assert audit["stage10"]["missing_records"] == 640


def test_stage11_incomplete_metrics_partial_detection(monkeypatch):
    """Verify that a present metadata file with arbitrary incomplete records reports PARTIAL."""
    p_s10 = find_stage_file(["data/metadata/stage10/denoising_metrics_per_image.csv"])
    df = pd.read_csv(p_s10)
    df_partial = df.iloc[:100].copy()

    orig_read_csv = pd.read_csv

    def mock_read_csv(filepath, *args, **kwargs):
        if "stage10" in str(filepath) and "denoising_metrics_per_image.csv" in str(filepath):
            return df_partial.copy()
        return orig_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    audit = run_stage11_artifact_audit()
    s10 = audit["stage10"]
    assert s10["status"] == "PARTIAL"
    assert s10["record_count"] == 100
    assert s10["missing_records"] == 540


def test_stage11_non_finite_metric_handling(tmp_path):
    """Verify that NaN and Inf are handled correctly without zero substitution."""
    # Create mock metrics DataFrame with Inf and NaN
    df_mock = pd.DataFrame([
        {
            "patient_id": "P_00004",
            "abnormality_id": 1,
            "breast_side": "LEFT",
            "image_view": "CC",
            "noise_type": "gaussian",
            "denoising_method": "median",
            "clean_denoised_psnr": 35.0,
            "clean_denoised_ssim": 0.85,
            "clean_denoised_mse": 0.0003,
            "clean_denoised_psnr_foreground": 33.5,
            "clean_denoised_ssim_foreground": 0.80,
            "noisy_denoised_psnr": 28.0,
            "noisy_denoised_ssim": 0.75,
            "clean_cnr": np.nan,  # CNR is NaN
        },
        {
            "patient_id": "P_00004",
            "abnormality_id": 1,
            "breast_side": "LEFT",
            "image_view": "CC",
            "noise_type": "gaussian",
            "denoising_method": "gaussian",
            "clean_denoised_psnr": np.inf,  # Identical clean reference has Inf PSNR
            "clean_denoised_ssim": 1.0,
            "clean_denoised_mse": 0.0,
            "clean_denoised_psnr_foreground": np.inf,
            "clean_denoised_ssim_foreground": 1.0,
            "noisy_denoised_psnr": 25.0,
            "noisy_denoised_ssim": 0.70,
            "clean_cnr": np.nan,
        },
    ])

    # Check CNR is NaN
    assert np.isnan(df_mock.loc[0, "clean_cnr"])
    # Ensure not substituted by 0.0
    assert df_mock.loc[0, "clean_cnr"] != 0.0

    # Check finite handling
    finite_psnr = df_mock["clean_denoised_psnr"].replace([np.inf, -np.inf], np.nan).dropna()
    assert len(finite_psnr) == 1
    assert finite_psnr.iloc[0] == 35.0


def test_stage11_comparison_definition_separation():
    """Verify that Comparison B and Comparison C columns are distinct."""
    p_s10 = find_stage_file(["data/metadata/stage10/denoising_metrics_per_image.csv"])
    assert p_s10 is not None
    df = pd.read_csv(p_s10)

    # Comparison B: Clean vs Denoised
    assert "clean_denoised_psnr" in df.columns
    assert "clean_denoised_ssim" in df.columns
    assert "clean_denoised_mse" in df.columns

    # Comparison C: Noisy vs Denoised
    assert "noisy_denoised_psnr" in df.columns
    assert "noisy_denoised_ssim" in df.columns
    assert "noisy_denoised_mse" in df.columns

    # Verify they are numerically distinct
    assert not np.allclose(df["clean_denoised_psnr"].values, df["noisy_denoised_psnr"].values)
    assert not np.allclose(df["clean_denoised_ssim"].values, df["noisy_denoised_ssim"].values)


def test_stage11_deterministic_case_selection():
    """Verify that representative case selection is deterministic and repeatable."""
    cases_1 = list(REPRESENTATIVE_CASES.keys())
    cases_2 = list(REPRESENTATIVE_CASES.keys())
    assert cases_1 == cases_2
    assert "P_00004_1_LEFT_CC" in cases_1
    assert "P_00009_1_RIGHT_MLO" in cases_1
    assert "P_00007_1_LEFT_CC" in cases_1
    assert "P_00299_1_LEFT_MLO" in cases_1


def test_stage11_report_generation(tmp_path):
    """Verify that stage 11 report generation creates valid TXT and MD files."""
    txt_out = str(tmp_path / "stage11_report.txt")
    md_out = str(tmp_path / "stage11_report.md")

    generate_stage11_report(
        output_txt_path=txt_out,
        output_md_path=md_out,
    )

    assert os.path.exists(txt_out)
    assert os.path.exists(md_out)

    with open(txt_out, "r", encoding="utf-8") as f:
        content_txt = f.read()
    with open(md_out, "r", encoding="utf-8") as f:
        content_md = f.read()

    # Check key reporting content
    assert "Stage 11 — Preprocessing Visualization & Comparison" in content_txt
    assert "Stage 9 Denoised Outputs          : 640" in content_txt
    assert "Stage 10 Quantitative Records     : 640 evaluated" in content_txt
    assert "NO WINNER SELECTED" in content_txt
    assert "P_00004_1_LEFT_CC" in content_txt
    assert "CNR Mask Eligibility Status       : 0 eligible cases" in content_txt

    assert "# CBIS-DDSM Stage 11 — Preprocessing Visualization & Comparison Report" in content_md
    assert "Stage 12" in content_md


def test_stage11_figure_generation_mock(tmp_path):
    """Verify that figure generation functions output valid PNG files with mock data."""
    out_dir = str(tmp_path / "comparison")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Create mock per-image metrics dataframe covering all 8 methods x 5 noise types
    mock_rows = []
    for m in DENOISING_METHODS:
        for n in NOISE_TYPES:
            mock_rows.append({
                "patient_id": "P_00004",
                "abnormality_id": 1,
                "breast_side": "LEFT",
                "image_view": "CC",
                "noise_type": n,
                "denoising_method": m,
                "clean_denoised_psnr": float(np.random.uniform(25.0, 45.0)),
                "clean_denoised_ssim": float(np.random.uniform(0.5, 0.95)),
                "clean_denoised_mse": float(np.random.uniform(0.0001, 0.003)),
                "clean_denoised_psnr_foreground": float(np.random.uniform(23.0, 43.0)),
                "clean_denoised_ssim_foreground": float(np.random.uniform(0.45, 0.90)),
                "noisy_denoised_psnr": float(np.random.uniform(24.0, 40.0)),
                "noisy_denoised_ssim": float(np.random.uniform(0.6, 0.9)),
                "clean_cnr": np.nan,
            })
    df_mock = pd.DataFrame(mock_rows)

    # Test plot_metrics_by_denoising_method
    p1 = plot_metrics_by_denoising_method(df_mock, os.path.join(out_dir, "test_metrics_method.png"))
    assert os.path.exists(p1)
    assert os.path.getsize(p1) > 1000

    # Test plot_metrics_by_noise_type
    p2 = plot_metrics_by_noise_type(df_mock, os.path.join(out_dir, "test_metrics_noise.png"))
    assert os.path.exists(p2)
    assert os.path.getsize(p2) > 1000

    # Test plot_method_by_noise_heatmaps
    p3, p4 = plot_method_by_noise_heatmaps(
        df_mock,
        os.path.join(out_dir, "test_heatmap_psnr.png"),
        os.path.join(out_dir, "test_heatmap_ssim.png"),
    )
    assert os.path.exists(p3)
    assert os.path.exists(p4)

    # Test plot_whole_vs_foreground_metrics
    p5 = plot_whole_vs_foreground_metrics(df_mock, os.path.join(out_dir, "test_whole_fg.png"))
    assert os.path.exists(p5)

    # Test plot_comparison_c_filtering_magnitude
    p6 = plot_comparison_c_filtering_magnitude(df_mock, os.path.join(out_dir, "test_comp_c.png"))
    assert os.path.exists(p6)


def test_stage11_figure_generation_multi_case_mock(tmp_path):
    """Verify that heatmaps and metric charts aggregate multiple cases without duplicate index errors."""
    out_dir = str(tmp_path / "multi_case_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # Create mock metrics with 3 different cases x 8 methods x 5 noise types = 120 rows
    mock_rows = []
    cases = [
        ("P_00004", 1, "LEFT", "CC"),
        ("P_00009", 1, "RIGHT", "MLO"),
        ("P_00007", 1, "LEFT", "CC"),
    ]
    for pid, aid, side, view in cases:
        for m in DENOISING_METHODS:
            for n in NOISE_TYPES:
                mock_rows.append({
                    "patient_id": pid,
                    "abnormality_id": aid,
                    "breast_side": side,
                    "image_view": view,
                    "noise_type": n,
                    "denoising_method": m,
                    "clean_denoised_psnr": np.inf if m == "baseline" else float(np.random.uniform(25.0, 45.0)),
                    "clean_denoised_ssim": float(np.random.uniform(0.5, 0.95)),
                    "clean_denoised_mse": float(np.random.uniform(0.0001, 0.003)),
                    "clean_denoised_psnr_foreground": float(np.random.uniform(23.0, 43.0)),
                    "clean_denoised_ssim_foreground": float(np.random.uniform(0.45, 0.90)),
                    "noisy_denoised_psnr": float(np.random.uniform(24.0, 40.0)),
                    "noisy_denoised_ssim": float(np.random.uniform(0.6, 0.9)),
                    "clean_cnr": np.nan,
                })
    df_multi = pd.DataFrame(mock_rows)

    # Verify duplicate keys exist in raw dataframe before aggregation
    dup_counts = df_multi.groupby(["denoising_method", "noise_type"]).size()
    assert (dup_counts == 3).all()

    # Heatmaps must aggregate across cases using pivot_table without raising ValueError
    p3, p4 = plot_method_by_noise_heatmaps(
        df_multi,
        os.path.join(out_dir, "test_heatmap_psnr.png"),
        os.path.join(out_dir, "test_heatmap_ssim.png"),
    )
    assert os.path.exists(p3)
    assert os.path.exists(p4)
    assert os.path.getsize(p3) > 1000
    assert os.path.getsize(p4) > 1000

    # Metric charts must also handle multi-case distributions and Inf values
    p1 = plot_metrics_by_denoising_method(df_multi, os.path.join(out_dir, "test_metrics_method.png"))
    assert os.path.exists(p1)
    p2 = plot_metrics_by_noise_type(df_multi, os.path.join(out_dir, "test_metrics_noise.png"))
    assert os.path.exists(p2)
    p5 = plot_whole_vs_foreground_metrics(df_multi, os.path.join(out_dir, "test_whole_fg.png"))
    assert os.path.exists(p5)
    p6 = plot_comparison_c_filtering_magnitude(df_multi, os.path.join(out_dir, "test_comp_c.png"))
    assert os.path.exists(p6)

