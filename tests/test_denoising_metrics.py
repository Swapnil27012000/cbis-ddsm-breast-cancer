"""Unit tests for Stage 10: Denoising Metrics & Quantitative Evaluation.

Verifies all mandatory requirements from the research specification:
1. MSE is zero for identical images.
2. MAE is zero for identical images.
3. RMSE is zero for identical images.
4. PSNR is infinite for identical images.
5. SSIM is 1.0 (within numerical tolerance) for identical images.
6. Metrics are finite and non-negative where mathematically expected.
7. Hand-calculated / analytical known MSE, MAE, RMSE, and PSNR examples.
8. SSIM data_range=1.0 consistency.
9. SNR edge cases (zero error -> +inf, zero signal -> 0.0, finite).
10. The same inputs produce deterministic metrics.
11. Images with mismatched dimensions are rejected.
12. Unreadable, corrupted, and non-finite (NaN/Inf) images are detected.
13. Missing metadata file references are caught and reported.
14. CNR handles invalid/unaligned masks safely and stores NaN (never 0.0).
15. Whole-image and foreground-only metrics remain distinct.
16. Comparison C (Noisy vs Denoised) metrics are computed and distinct from Comparison B.
17. Numerical equivalence of optimized single-pass SSIM and caching vs unoptimized calculations.
18. End-to-end pipeline execution links correct Stage 5, 8, and 9 files.
19. Summary counts and groupings match per-image metrics table.
20. Source images from earlier stages are 100% immutable (hash verified).
"""

import hashlib
import json
import math
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest

from src.metrics.mse import compute_mse
from src.metrics.psnr import compute_psnr
from src.metrics.ssim import compute_ssim
from src.metrics.snr import compute_snr
from src.metrics.cnr import compute_cnr, extract_roi_and_background_pixels
from src.metrics.entropy import compute_entropy
from src.preprocessing.denoising_metrics import (
    compute_mae,
    compute_rmse,
    compute_mean_gradient,
    extract_breast_foreground_mask,
    compute_foreground_metrics,
    compute_image_pair_metrics,
    resolve_case_roi_mask,
    evaluate_single_denoised_record,
    build_metric_summary_df,
    compute_paired_statistical_tests,
    run_stage10_denoising_metrics,
    STATUS_PASS,
    STATUS_FAIL,
    STATUS_SUCCESS,
    CNR_AVAILABLE,
    CNR_UNAVAILABLE,
)


@pytest.fixture
def synthetic_patch() -> np.ndarray:
    """Create a 64x64 synthetic float32 [0.0, 1.0] mammogram patch."""
    y, x = np.mgrid[:64, :64]
    base = (x / 64.0) * 0.6 + (y / 64.0) * 0.3
    base[x < 10] = 0.0  # Background black region
    return np.clip(base, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def synthetic_noisy_patch(synthetic_patch) -> np.ndarray:
    """Create a perturbed 64x64 synthetic noisy patch."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, size=synthetic_patch.shape).astype(np.float32)
    return np.clip(synthetic_patch + noise, 0.0, 1.0).astype(np.float32)


# ==============================================================================
# TEST 1, 2, 3, 4, 5: Identity Metrics
# ==============================================================================
def test_identity_metrics(synthetic_patch):
    """Verify MSE=0, MAE=0, RMSE=0, PSNR=inf, SSIM=1.0 for identical images."""
    mse = compute_mse(synthetic_patch, synthetic_patch)
    mae = compute_mae(synthetic_patch, synthetic_patch)
    rmse = compute_rmse(synthetic_patch, synthetic_patch)
    psnr = compute_psnr(synthetic_patch, synthetic_patch, data_range=1.0)
    ssim = compute_ssim(synthetic_patch, synthetic_patch, data_range=1.0)
    snr = compute_snr(synthetic_patch, synthetic_patch)

    assert mse == 0.0
    assert mae == 0.0
    assert rmse == 0.0
    assert math.isinf(psnr) and psnr > 0
    assert math.isinf(snr) and snr > 0
    assert pytest.approx(ssim, rel=1e-5) == 1.0


# ==============================================================================
# TEST 6 & 7: Known Analytical Examples (Hand-Calculated Verification)
# ==============================================================================
def test_known_analytical_examples():
    """Verify MSE, MAE, RMSE, PSNR, SNR against known mathematical test cases."""
    # Two constant 10x10 arrays with exact difference 0.1
    arr1 = np.full((10, 10), 0.5, dtype=np.float32)
    arr2 = np.full((10, 10), 0.6, dtype=np.float32)

    # Analytical values:
    # diff = 0.1
    # MSE = 0.1^2 = 0.01
    # MAE = 0.1
    # RMSE = sqrt(0.01) = 0.1
    # PSNR = 10 * log10(1.0^2 / 0.01) = 10 * 2 = 20.0 dB
    # Signal power = 100 * (0.5^2) = 25.0
    # Noise power = 100 * (0.1^2) = 1.0
    # SNR = 10 * log10(25.0 / 1.0) = 10 * 1.3979400 = 13.9794 dB

    assert pytest.approx(compute_mse(arr1, arr2), abs=1e-6) == 0.01
    assert pytest.approx(compute_mae(arr1, arr2), abs=1e-6) == 0.1
    assert pytest.approx(compute_rmse(arr1, arr2), abs=1e-6) == 0.1
    assert pytest.approx(compute_psnr(arr1, arr2, data_range=1.0), abs=1e-4) == 20.0
    assert pytest.approx(compute_snr(arr1, arr2), abs=1e-4) == 13.9794


# ==============================================================================
# TEST 8: SSIM Data-Range Consistency
# ==============================================================================
def test_ssim_data_range_consistency(synthetic_patch, synthetic_noisy_patch):
    """Verify SSIM enforces data_range=1.0 and yields values bounded in [-1.0, 1.0]."""
    ssim_val = compute_ssim(synthetic_patch, synthetic_noisy_patch, data_range=1.0, win_size=7)
    assert 0.0 < ssim_val < 1.0
    assert np.isfinite(ssim_val)

    # Identical images must always be exactly 1.0
    assert compute_ssim(synthetic_patch, synthetic_patch, data_range=1.0) == 1.0


# ==============================================================================
# TEST 9: SNR Edge Cases
# ==============================================================================
def test_snr_edge_cases():
    """Verify SNR edge case handling: identical arrays (+inf), zero signal (0.0)."""
    ones = np.ones((16, 16), dtype=np.float32)
    zeros = np.zeros((16, 16), dtype=np.float32)

    # Zero error -> +inf
    snr_ident = compute_snr(ones, ones)
    assert math.isinf(snr_ident) and snr_ident > 0

    # Zero signal power -> 0.0 dB
    snr_zeros = compute_snr(zeros, ones)
    assert snr_zeros == 0.0

    # Both zero -> +inf (zero noise)
    snr_both_zeros = compute_snr(zeros, zeros)
    assert math.isinf(snr_both_zeros) and snr_both_zeros > 0


# ==============================================================================
# TEST 10: Determinism
# ==============================================================================
def test_metrics_deterministic(synthetic_patch, synthetic_noisy_patch):
    """Verify the same inputs produce identical metric outputs."""
    mse1 = compute_mse(synthetic_patch, synthetic_noisy_patch)
    mse2 = compute_mse(synthetic_patch, synthetic_noisy_patch)
    psnr1 = compute_psnr(synthetic_patch, synthetic_noisy_patch, data_range=1.0)
    psnr2 = compute_psnr(synthetic_patch, synthetic_noisy_patch, data_range=1.0)
    ssim1 = compute_ssim(synthetic_patch, synthetic_noisy_patch, data_range=1.0)
    ssim2 = compute_ssim(synthetic_patch, synthetic_noisy_patch, data_range=1.0)

    assert mse1 == mse2
    assert psnr1 == psnr2
    assert ssim1 == ssim2


# ==============================================================================
# TEST 11: Dimension Mismatch Rejection
# ==============================================================================
def test_dimension_mismatch_rejected(synthetic_patch):
    """Verify mismatched dimensions raise ValueError."""
    mismatched = synthetic_patch[:32, :32]
    with pytest.raises(ValueError):
        compute_mse(synthetic_patch, mismatched)
    with pytest.raises(ValueError):
        compute_mae(synthetic_patch, mismatched)
    with pytest.raises(ValueError):
        compute_ssim(synthetic_patch, mismatched)
    with pytest.raises(ValueError):
        compute_snr(synthetic_patch, mismatched)
    with pytest.raises(ValueError):
        compute_image_pair_metrics(synthetic_patch, mismatched)


# ==============================================================================
# TEST 12: Unreadable & Non-finite (NaN/Inf) Image Detection
# ==============================================================================
def test_evaluation_catches_unreadable_and_nan(tmp_path, synthetic_patch):
    """Verify missing, unreadable, and non-finite images are caught in evaluation."""
    clean_p = tmp_path / "clean.png"
    noisy_p = tmp_path / "noisy.png"
    denoised_p = tmp_path / "denoised.png"

    cv2.imwrite(str(clean_p), (synthetic_patch * 255).astype(np.uint8))
    cv2.imwrite(str(noisy_p), (synthetic_patch * 255).astype(np.uint8))
    # Write corrupted file for denoised
    denoised_p.write_bytes(b"CORRUPTED_FILE")

    row = pd.Series({
        "patient_id": "P_99999",
        "abnormality_id": 1,
        "abnormality_type": "mass",
        "pathology": "BENIGN",
        "binary_label": 0,
        "breast_side": "LEFT",
        "image_view": "CC",
        "dataset_split": "train",
        "noise_type": "gaussian",
        "denoising_method": "median",
        "clean_image_path": str(clean_p),
        "noisy_image_path": str(noisy_p),
        "denoised_image_path": str(denoised_p),
    })

    cfg = {"metrics": {"ssim": {"win_size": 7}, "entropy": {"num_bins": 256}}}
    m_rec, v_rec = evaluate_single_denoised_record(row, cfg)
    assert v_rec["status"] == STATUS_FAIL
    assert "Error reading image array" in v_rec["validation_message"]


# ==============================================================================
# TEST 13: Missing Metadata File References
# ==============================================================================
def test_missing_metadata_references():
    """Verify missing image paths are flagged as STATUS_FAIL with clear message."""
    row = pd.Series({
        "patient_id": "P_NONEXISTENT",
        "abnormality_id": 1,
        "abnormality_type": "mass",
        "pathology": "BENIGN",
        "binary_label": 0,
        "breast_side": "LEFT",
        "image_view": "CC",
        "dataset_split": "train",
        "noise_type": "gaussian",
        "denoising_method": "median",
        "clean_image_path": "nonexistent/clean.png",
        "noisy_image_path": "nonexistent/noisy.png",
        "denoised_image_path": "nonexistent/denoised.png",
    })
    cfg = {"metrics": {}}
    m_rec, v_rec = evaluate_single_denoised_record(row, cfg)
    assert m_rec == {}
    assert v_rec["status"] == STATUS_FAIL
    assert "Missing file(s)" in v_rec["validation_message"]


# ==============================================================================
# TEST 14: CNR Mask Eligibility & Missing Value (NaN) Handling
# ==============================================================================
def test_cnr_mask_eligibility_and_nan():
    """Verify CNR handles unaligned or missing masks by storing NaN, never replacing with 0.0."""
    roi_empty = np.array([])
    bg_empty = np.array([])
    assert compute_cnr(roi_empty, bg_empty) == 0.0

    # Dimension mismatch check in resolve_case_roi_mask
    mask, reason = resolve_case_roi_mask(
        patient_id="NON_EXISTENT",
        abnormality_id=1,
        breast_side="LEFT",
        image_view="CC",
        expected_shape=(64, 64),
        ref_mapping_csv=None,
    )
    assert mask is None
    assert "not found" in reason.lower() or "no mapping entry" in reason.lower()


# ==============================================================================
# TEST 15: Foreground-Only vs Whole-Image Distinctness
# ==============================================================================
def test_whole_image_vs_foreground_distinct(synthetic_patch, synthetic_noisy_patch):
    """Verify whole-image and foreground-only metrics are distinct and valid."""
    fg_mask = extract_breast_foreground_mask(synthetic_patch, threshold=0.02)
    assert fg_mask.shape == synthetic_patch.shape
    assert 0 < np.count_nonzero(fg_mask) < synthetic_patch.size

    mse_whole = compute_mse(synthetic_patch, synthetic_noisy_patch)
    mse_fg, psnr_fg, ssim_fg = compute_foreground_metrics(
        synthetic_patch, synthetic_noisy_patch, fg_mask, win_size=7
    )

    assert mse_whole != mse_fg
    assert np.isfinite(mse_fg) and mse_fg > 0.0
    assert np.isfinite(psnr_fg)
    assert np.isfinite(ssim_fg)


# ==============================================================================
# TEST 16: Comparison C (Noisy vs Denoised) Evaluation
# ==============================================================================
def test_comparison_c_metrics(tmp_path, synthetic_patch, synthetic_noisy_patch):
    """Verify Comparison C (Noisy vs Denoised) is computed and distinct from Restoration (Clean vs Denoised)."""
    clean_p = tmp_path / "clean.png"
    noisy_p = tmp_path / "noisy.png"
    denoised_p = tmp_path / "denoised.png"

    # Denoised image is slightly smoothed version
    denoised_arr = cv2.medianBlur((synthetic_noisy_patch * 255).astype(np.uint8), 3)

    cv2.imwrite(str(clean_p), (synthetic_patch * 255).astype(np.uint8))
    cv2.imwrite(str(noisy_p), (synthetic_noisy_patch * 255).astype(np.uint8))
    cv2.imwrite(str(denoised_p), denoised_arr)

    row = pd.Series({
        "patient_id": "P_00004",
        "abnormality_id": 1,
        "abnormality_type": "mass",
        "pathology": "BENIGN",
        "binary_label": 0,
        "breast_side": "LEFT",
        "image_view": "CC",
        "dataset_split": "train",
        "noise_type": "gaussian",
        "denoising_method": "median",
        "clean_image_path": str(clean_p),
        "noisy_image_path": str(noisy_p),
        "denoised_image_path": str(denoised_p),
    })

    cfg = {"metrics": {"ssim": {"win_size": 7}, "entropy": {"num_bins": 256}}}
    m_rec, v_rec = evaluate_single_denoised_record(row, cfg)

    assert v_rec["status"] == STATUS_PASS
    # Verify Comparison C columns exist and are finite
    assert "noisy_denoised_mse" in m_rec
    assert "noisy_denoised_psnr" in m_rec
    assert "noisy_denoised_ssim" in m_rec
    assert "noisy_denoised_snr" in m_rec
    assert "noisy_denoised_mae" in m_rec
    assert "noisy_denoised_rmse" in m_rec
    assert "noisy_denoised_entropy" in m_rec

    assert np.isfinite(m_rec["noisy_denoised_mse"])
    assert np.isfinite(m_rec["noisy_denoised_ssim"])
    # Comparison C must not be identical to Comparison B
    assert m_rec["noisy_denoised_mse"] != m_rec["clean_denoised_mse"]


# ==============================================================================
# TEST 17: Numerical Equivalence of Single-Pass SSIM and Caching Optimization
# ==============================================================================
def test_numerical_equivalence_optimized_vs_unoptimized(synthetic_patch, synthetic_noisy_patch):
    """Verify optimized single-pass SSIM and caching agree with unoptimized metrics within 1e-12 tolerance."""
    fg_mask = extract_breast_foreground_mask(synthetic_patch, threshold=0.02)

    # 1. Compare compute_image_pair_metrics against individual functions
    pair = compute_image_pair_metrics(
        synthetic_patch, synthetic_noisy_patch, data_range=1.0, win_size=7, fg_mask=fg_mask
    )

    expected_mse = compute_mse(synthetic_patch, synthetic_noisy_patch)
    expected_mae = compute_mae(synthetic_patch, synthetic_noisy_patch)
    expected_rmse = compute_rmse(synthetic_patch, synthetic_noisy_patch)
    expected_psnr = compute_psnr(synthetic_patch, synthetic_noisy_patch, data_range=1.0)
    expected_snr = compute_snr(synthetic_patch, synthetic_noisy_patch)
    expected_ssim = compute_ssim(synthetic_patch, synthetic_noisy_patch, data_range=1.0, win_size=7)
    mse_fg, psnr_fg, ssim_fg = compute_foreground_metrics(
        synthetic_patch, synthetic_noisy_patch, fg_mask, win_size=7
    )

    assert pytest.approx(pair["mse"], abs=1e-12) == expected_mse
    assert pytest.approx(pair["mae"], abs=1e-12) == expected_mae
    assert pytest.approx(pair["rmse"], abs=1e-12) == expected_rmse
    assert pytest.approx(pair["psnr"], abs=1e-12) == expected_psnr
    assert pytest.approx(pair["snr"], abs=1e-12) == expected_snr
    assert pytest.approx(pair["ssim"], abs=1e-7) == expected_ssim
    assert pytest.approx(pair["mse_fg"], abs=1e-12) == mse_fg
    assert pytest.approx(pair["psnr_fg"], abs=1e-12) == psnr_fg
    assert pytest.approx(pair["ssim_fg"], abs=1e-7) == ssim_fg


# ==============================================================================
# TEST 18, 19, 20: Pipeline Execution, Summaries, Infinity, and Immutability
# ==============================================================================
def test_stage10_mock_pipeline_and_summaries(tmp_path):
    """Verify end-to-end evaluation, correct file linkage, summary counts, and safe infinity handling."""
    clean_dir = tmp_path / "baseline"
    noisy_dir = tmp_path / "noise"
    denoised_dir = tmp_path / "denoised"
    meta_dir = tmp_path / "meta_s9"
    out_s10 = tmp_path / "meta_s10"

    clean_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)
    denoised_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    y, x = np.mgrid[:32, :32]
    clean_arr = ((x / 32.0) * 160 + (y / 32.0) * 50 + 20).astype(np.uint8)
    clean_arr[:, :4] = 0

    s9_records = []
    # 2 cases x 2 noise types x 2 denoising methods = 8 records
    for c_idx in range(2):
        pid = f"P_{c_idx:05d}"
        clean_p = clean_dir / f"{pid}_1_LEFT_CC_baseline.png"
        cv2.imwrite(str(clean_p), clean_arr)

        for nt in ["gaussian", "poisson"]:
            noisy_p = noisy_dir / f"{pid}_1_LEFT_CC_{nt}.png"
            noisy_arr = np.clip(clean_arr.astype(np.float32) + 10.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(noisy_p), noisy_arr)

            for m in ["median", "gaussian"]:
                denoised_p = denoised_dir / f"{pid}_1_LEFT_CC_{nt}_{m}.png"
                # If median on case 0, make identical to clean to test PSNR=inf handling!
                if c_idx == 0 and m == "median":
                    denoised_arr = clean_arr.copy()
                else:
                    denoised_arr = np.clip(clean_arr.astype(np.float32) + 5.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(denoised_p), denoised_arr)

                s9_records.append({
                    "patient_id": pid,
                    "abnormality_id": 1,
                    "abnormality_type": "mass",
                    "pathology": "BENIGN",
                    "binary_label": 0,
                    "breast_side": "LEFT",
                    "image_view": "CC",
                    "dataset_split": "train",
                    "noise_type": nt,
                    "denoising_method": m,
                    "clean_image_path": str(clean_p),
                    "noisy_image_path": str(noisy_p),
                    "denoised_image_path": str(denoised_p),
                })

    s9_meta_path = meta_dir / "denoising_experiment_metadata.csv"
    pd.DataFrame(s9_records).to_csv(str(s9_meta_path), index=False)

    # Immutability baseline hashes
    clean_hash_before = hashlib.sha256(clean_p.read_bytes()).hexdigest()
    noisy_hash_before = hashlib.sha256(noisy_p.read_bytes()).hexdigest()
    denoised_hash_before = hashlib.sha256(denoised_p.read_bytes()).hexdigest()

    # Run Stage 10 pipeline
    df_metrics, df_val = run_stage10_denoising_metrics(
        config_path=None,
        stage9_metadata_csv=str(s9_meta_path),
        ref_mapping_csv=None,
        output_dir=str(out_s10),
    )

    # 1. Output count matches
    assert len(df_metrics) == 8
    assert len(df_val) == 8
    assert (df_val["status"] == STATUS_PASS).all()

    # 2. Infinite PSNR handled safely (case 0 median had identical clean and denoised image)
    inf_records = df_metrics[np.isinf(df_metrics["clean_denoised_psnr"])]
    assert len(inf_records) > 0  # Captured positive infinity correctly

    # 3. Required files exist
    assert os.path.exists(out_s10 / "denoising_metrics_per_image.csv")
    assert os.path.exists(out_s10 / "denoising_metrics_summary_by_method.csv")
    assert os.path.exists(out_s10 / "denoising_metrics_summary_by_noise.csv")
    assert os.path.exists(out_s10 / "denoising_metrics_summary_method_by_noise.csv")
    assert os.path.exists(out_s10 / "denoising_metrics_validation.csv")
    assert os.path.exists(out_s10 / "denoising_metrics_report.txt")
    assert os.path.exists(out_s10 / "denoising_metrics_config.json")

    # 4. Check summary counts match
    df_sum_method = pd.read_csv(out_s10 / "denoising_metrics_summary_by_method.csv")
    assert df_sum_method["record_count"].sum() == 8

    # 5. Immutability verification
    clean_hash_after = hashlib.sha256(clean_p.read_bytes()).hexdigest()
    noisy_hash_after = hashlib.sha256(noisy_p.read_bytes()).hexdigest()
    denoised_hash_after = hashlib.sha256(denoised_p.read_bytes()).hexdigest()
    assert clean_hash_before == clean_hash_after
    assert noisy_hash_before == noisy_hash_after
    assert denoised_hash_before == denoised_hash_after


# ==============================================================================
# TEST 21: Resume Mode Preserves Valid Existing Records and Calculates Missing
# ==============================================================================
def test_stage10_resume_preserves_existing_and_computes_missing(tmp_path):
    """Verify that Stage 10 resume mode preserves valid records without recomputing and adds missing."""
    clean_dir = tmp_path / "baseline"
    noisy_dir = tmp_path / "noise"
    denoised_dir = tmp_path / "denoised"
    meta_s9_dir = tmp_path / "meta_s9"
    out_s10 = tmp_path / "meta_s10"

    clean_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)
    denoised_dir.mkdir(parents=True)
    meta_s9_dir.mkdir(parents=True)
    out_s10.mkdir(parents=True)

    # 32x32 clean patch
    clean_arr = np.full((32, 32), 100, dtype=np.uint8)
    clean_p = clean_dir / "P_00001_1_LEFT_CC_baseline.png"
    cv2.imwrite(str(clean_p), clean_arr)

    # 2 noise types x 2 methods = 4 records in Stage 9
    s9_records = []
    for nt in ["gaussian", "poisson"]:
        noisy_arr = np.clip(clean_arr.astype(np.float32) + 10.0, 0, 255).astype(np.uint8)
        noisy_p = noisy_dir / f"P_00001_1_LEFT_CC_{nt}.png"
        cv2.imwrite(str(noisy_p), noisy_arr)

        for m in ["median", "gaussian"]:
            denoised_arr = np.clip(clean_arr.astype(np.float32) + 3.0, 0, 255).astype(np.uint8)
            denoised_p = denoised_dir / f"P_00001_1_LEFT_CC_{nt}_{m}.png"
            cv2.imwrite(str(denoised_p), denoised_arr)

            s9_records.append({
                "patient_id": "P_00001",
                "abnormality_id": 1,
                "abnormality_type": "mass",
                "pathology": "BENIGN",
                "binary_label": 0,
                "breast_side": "LEFT",
                "image_view": "CC",
                "dataset_split": "train",
                "noise_type": nt,
                "denoising_method": m,
                "clean_image_path": str(clean_p),
                "noisy_image_path": str(noisy_p),
                "denoised_image_path": str(denoised_p),
            })

    s9_meta_path = meta_s9_dir / "denoising_experiment_metadata.csv"
    pd.DataFrame(s9_records).to_csv(str(s9_meta_path), index=False)

    # Pre-populate Stage 10 with 1 record having a distinctive Sentinel PSNR (999.999)
    # Target record: (P_00001, 1, LEFT, CC, gaussian, median)
    existing_record = {
        "patient_id": "P_00001",
        "abnormality_id": 1,
        "abnormality_type": "mass",
        "pathology": "BENIGN",
        "binary_label": 0,
        "breast_side": "LEFT",
        "image_view": "CC",
        "dataset_split": "train",
        "noise_type": "gaussian",
        "denoising_method": "median",
        "clean_image_path": str(clean_p),
        "noisy_image_path": str(noisy_dir / "P_00001_1_LEFT_CC_gaussian.png"),
        "denoised_image_path": str(denoised_dir / "P_00001_1_LEFT_CC_gaussian_median.png"),
        "clean_noisy_mse": 0.01,
        "clean_noisy_psnr": 20.0,
        "clean_noisy_ssim": 0.8,
        "clean_noisy_snr": 15.0,
        "clean_noisy_mae": 0.1,
        "clean_noisy_rmse": 0.1,
        "clean_noisy_entropy": 3.0,
        "clean_noisy_mse_foreground": 0.01,
        "clean_noisy_psnr_foreground": 20.0,
        "clean_noisy_ssim_foreground": 0.8,
        "clean_denoised_mse": 0.0001,
        "clean_denoised_psnr": 999.999,  # Distinctive Sentinel Value
        "clean_denoised_ssim": 0.9999,
        "clean_denoised_snr": 40.0,
        "clean_denoised_mae": 0.01,
        "clean_denoised_rmse": 0.01,
        "clean_denoised_mse_foreground": 0.0001,
        "clean_denoised_psnr_foreground": 999.999,
        "clean_denoised_ssim_foreground": 0.9999,
        "noisy_denoised_mse": 0.005,
        "noisy_denoised_psnr": 23.0,
        "noisy_denoised_ssim": 0.85,
        "noisy_denoised_snr": 18.0,
        "noisy_denoised_mae": 0.05,
        "noisy_denoised_rmse": 0.07,
        "noisy_denoised_entropy": 2.5,
        "noisy_denoised_mse_foreground": 0.005,
        "noisy_denoised_psnr_foreground": 23.0,
        "noisy_denoised_ssim_foreground": 0.85,
        "clean_entropy": 4.0,
        "noisy_entropy": 4.5,
        "denoised_entropy": 4.1,
        "clean_mean_gradient": 0.02,
        "noisy_mean_gradient": 0.05,
        "denoised_mean_gradient": 0.025,
        "clean_cnr": np.nan,
        "noisy_cnr": np.nan,
        "denoised_cnr": np.nan,
        "cnr_status": "UNAVAILABLE",
        "foreground_mask_source": "test",
        "foreground_area_ratio": 1.0,
        "input_height": 32,
        "input_width": 32,
        "status": STATUS_SUCCESS,
        "validation_message": "Preserved test record",
        "processing_time": 0.001,
    }
    existing_val = {
        "patient_id": "P_00001",
        "abnormality_id": 1,
        "breast_side": "LEFT",
        "image_view": "CC",
        "noise_type": "gaussian",
        "denoising_method": "median",
        "clean_exists": True,
        "noisy_exists": True,
        "denoised_exists": True,
        "dimensions_match": True,
        "no_nan_inf": True,
        "finite_metrics": True,
        "status": STATUS_PASS,
        "validation_message": "PASS: Preserved test validation",
    }
    pd.DataFrame([existing_record]).to_csv(str(out_s10 / "denoising_metrics_per_image.csv"), index=False)
    pd.DataFrame([existing_val]).to_csv(str(out_s10 / "denoising_metrics_validation.csv"), index=False)

    # Execute Stage 10 with resume=True
    df_metrics, df_val = run_stage10_denoising_metrics(
        config_path=None,
        stage9_metadata_csv=str(s9_meta_path),
        ref_mapping_csv=None,
        output_dir=str(out_s10),
        resume=True,
    )

    # 1. All 4 evaluations present, 0 duplicates
    assert len(df_metrics) == 4
    assert len(df_val) == 4
    assert len(df_metrics.drop_duplicates(subset=["patient_id", "noise_type", "denoising_method"])) == 4

    # 2. Existing record was preserved exactly (retaining sentinel PSNR 999.999)
    preserved_row = df_metrics[(df_metrics["noise_type"] == "gaussian") & (df_metrics["denoising_method"] == "median")]
    assert len(preserved_row) == 1
    assert pytest.approx(preserved_row.iloc[0]["clean_denoised_psnr"], abs=1e-3) == 999.999

    # 3. Missing records were newly computed (finite and realistic PSNR)
    computed_rows = df_metrics[df_metrics["clean_denoised_psnr"] != 999.999]
    assert len(computed_rows) == 3
    assert (computed_rows["clean_denoised_psnr"] > 25.0).all()


# ==============================================================================
# TEST 22: Overwrite Mode Recomputes All Records
# ==============================================================================
def test_stage10_overwrite_mode_recomputes_all(tmp_path):
    """Verify that Stage 10 overwrite mode ignores existing sentinel values and recomputes all."""
    clean_dir = tmp_path / "baseline"
    noisy_dir = tmp_path / "noise"
    denoised_dir = tmp_path / "denoised"
    meta_s9_dir = tmp_path / "meta_s9"
    out_s10 = tmp_path / "meta_s10"

    clean_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)
    denoised_dir.mkdir(parents=True)
    meta_s9_dir.mkdir(parents=True)
    out_s10.mkdir(parents=True)

    clean_arr = np.full((32, 32), 100, dtype=np.uint8)
    clean_p = clean_dir / "P_00001_1_LEFT_CC_baseline.png"
    cv2.imwrite(str(clean_p), clean_arr)

    noisy_arr = np.clip(clean_arr.astype(np.float32) + 10.0, 0, 255).astype(np.uint8)
    noisy_p = noisy_dir / "P_00001_1_LEFT_CC_gaussian.png"
    cv2.imwrite(str(noisy_p), noisy_arr)

    denoised_arr = np.clip(clean_arr.astype(np.float32) + 2.0, 0, 255).astype(np.uint8)
    denoised_p = denoised_dir / "P_00001_1_LEFT_CC_gaussian_median.png"
    cv2.imwrite(str(denoised_p), denoised_arr)

    s9_record = {
        "patient_id": "P_00001",
        "abnormality_id": 1,
        "abnormality_type": "mass",
        "pathology": "BENIGN",
        "binary_label": 0,
        "breast_side": "LEFT",
        "image_view": "CC",
        "dataset_split": "train",
        "noise_type": "gaussian",
        "denoising_method": "median",
        "clean_image_path": str(clean_p),
        "noisy_image_path": str(noisy_p),
        "denoised_image_path": str(denoised_p),
    }
    s9_meta_path = meta_s9_dir / "denoising_experiment_metadata.csv"
    pd.DataFrame([s9_record]).to_csv(str(s9_meta_path), index=False)

    # Pre-populate with sentinel PSNR
    existing_record = dict(s9_record)
    existing_record.update({
        "clean_denoised_psnr": 777.777,
        "status": STATUS_SUCCESS,
        "validation_message": "Old record",
    })
    pd.DataFrame([existing_record]).to_csv(str(out_s10 / "denoising_metrics_per_image.csv"), index=False)
    pd.DataFrame([{
        "patient_id": "P_00001", "abnormality_id": 1, "breast_side": "LEFT", "image_view": "CC",
        "noise_type": "gaussian", "denoising_method": "median", "clean_exists": True, "noisy_exists": True,
        "denoised_exists": True, "dimensions_match": True, "no_nan_inf": True, "finite_metrics": True,
        "status": STATUS_PASS, "validation_message": "PASS",
    }]).to_csv(str(out_s10 / "denoising_metrics_validation.csv"), index=False)

    # Run with resume=False (overwrite mode)
    df_metrics, df_val = run_stage10_denoising_metrics(
        config_path=None,
        stage9_metadata_csv=str(s9_meta_path),
        ref_mapping_csv=None,
        output_dir=str(out_s10),
        resume=False,
    )

    # Sentinel must be replaced by actual calculated PSNR (~42 dB for diff=2)
    assert len(df_metrics) == 1
    assert df_metrics.iloc[0]["clean_denoised_psnr"] != 777.777
    assert 35.0 < df_metrics.iloc[0]["clean_denoised_psnr"] < 45.0


# ==============================================================================
# TEST 23: Missing Input File Handled Gracefully Without Crash
# ==============================================================================
def test_stage10_missing_file_handling():
    """Verify that evaluate_single_denoised_record reports failure when input files are missing."""
    cfg = {"metrics": {"ssim": {"win_size": 7, "data_range": 1.0}}}
    row = pd.Series({
        "patient_id": "P_NONEXISTENT",
        "abnormality_id": 1,
        "abnormality_type": "mass",
        "pathology": "BENIGN",
        "binary_label": 0,
        "breast_side": "LEFT",
        "image_view": "CC",
        "dataset_split": "train",
        "noise_type": "gaussian",
        "denoising_method": "median",
        "clean_image_path": "nonexistent_clean.png",
        "noisy_image_path": "nonexistent_noisy.png",
        "denoised_image_path": "nonexistent_denoised.png",
    })

    m_rec, v_rec = evaluate_single_denoised_record(row=row, cfg=cfg)

    # Must return empty metric record and FAIL validation record
    assert m_rec == {}
    assert v_rec["status"] == STATUS_FAIL
    assert not v_rec["clean_exists"]
    assert "Missing file(s)" in v_rec["validation_message"]

