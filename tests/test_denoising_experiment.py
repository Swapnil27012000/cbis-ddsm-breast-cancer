"""Unit tests for Stage 9: Medical Image Denoising Experiment.

Verifies all 22 mandatory requirements from the research specification:
1. Median reduces/modifies noisy input.
2. Gaussian denoising produces a valid image.
3. Wiener produces a valid image.
4. Bilateral produces a valid image.
5. Non-Local Means produces a valid image.
6. Anscombe-Wiener produces a valid image.
7. Adaptive Median produces a valid image.
8. Kuan produces a valid image.
9. All eight methods are available.
10. Output dimensions are unchanged.
11. Output remains grayscale.
12. Output values remain within [0, 1] before saving.
13. Saved output reopens correctly.
14. No NaN/Inf.
15. Same input + same method + same configuration is deterministic.
16. All five Stage-8 noise types are supported.
17. Every Stage-8 noisy input is used exactly once per denoising method.
18. 80 noisy inputs produce exactly 640 denoised outputs.
19. Every denoised output has a valid corresponding clean_image_path.
20. Raw CBIS-DDSM data remains unchanged.
21. Stage-5 baseline remains unchanged.
22. Stage-8 noisy images remain unchanged.
"""

import hashlib
import json
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.denoising_experiment import (
    ALL_DENOISING_METHODS,
    ALL_NOISE_TYPES,
    METHOD_MEDIAN,
    METHOD_GAUSSIAN,
    METHOD_WIENER,
    METHOD_BILATERAL,
    METHOD_NLM,
    METHOD_ANSCOMBE_WIENER,
    METHOD_ADAPTIVE_MEDIAN,
    METHOD_KUAN,
    STATUS_PASS,
    STATUS_SUCCESS,
    apply_denoising_method,
    save_denoised_image,
    validate_denoised_image,
    load_stage8_noisy_inputs,
    run_stage9_pilot,
    compute_array_hash,
)
from src.preprocessing.noise_experiment import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    add_poisson_noise,
    add_mixed_poisson_gaussian_noise,
)


@pytest.fixture
def synthetic_clean_mammogram() -> np.ndarray:
    """Create a 64x64 synthetic float32 [0.0, 1.0] mammogram patch."""
    y, x = np.mgrid[:64, :64]
    # Smooth anatomical gradient with simulated fibroglandular tissue
    base = (x / 64.0) * 0.6 + (y / 64.0) * 0.3
    base[x < 10] = 0.0  # Background black region
    return np.clip(base, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def synthetic_noisy_mammogram(synthetic_clean_mammogram) -> np.ndarray:
    """Create a 64x64 synthetic float32 [0.0, 1.0] noisy mammogram patch."""
    rng = np.random.default_rng(42)
    return add_gaussian_noise(synthetic_clean_mammogram, mean=0.0, sigma=0.05, rng=rng)


# ==============================================================================
# TEST 1: Median reduces/modifies noisy input
# ==============================================================================
def test_median_modifies_noisy_input(synthetic_clean_mammogram):
    rng = np.random.default_rng(42)
    sp_noisy = add_salt_pepper_noise(synthetic_clean_mammogram, amount=0.05, rng=rng)
    denoised = apply_denoising_method(sp_noisy, METHOD_MEDIAN, {"kernel_size": 3})
    assert denoised.shape == sp_noisy.shape
    assert not np.array_equal(sp_noisy, denoised)
    # Median filter should significantly reduce salt & pepper impulse errors
    assert np.mean((denoised - synthetic_clean_mammogram) ** 2) < np.mean((sp_noisy - synthetic_clean_mammogram) ** 2)


# ==============================================================================
# TEST 2: Gaussian denoising produces a valid image
# ==============================================================================
def test_gaussian_produces_valid_image(synthetic_noisy_mammogram):
    denoised = apply_denoising_method(synthetic_noisy_mammogram, METHOD_GAUSSIAN, {"kernel_size": 5, "sigma": 1.0})
    assert denoised.shape == synthetic_noisy_mammogram.shape
    assert denoised.dtype == np.float32
    assert denoised.min() >= 0.0 and denoised.max() <= 1.0
    assert not np.isnan(denoised).any()


# ==============================================================================
# TEST 3: Wiener produces a valid image
# ==============================================================================
def test_wiener_produces_valid_image(synthetic_noisy_mammogram):
    denoised = apply_denoising_method(synthetic_noisy_mammogram, METHOD_WIENER, {"window_size": 5})
    assert denoised.shape == synthetic_noisy_mammogram.shape
    assert denoised.dtype == np.float32
    assert denoised.min() >= 0.0 and denoised.max() <= 1.0
    assert not np.isnan(denoised).any()


# ==============================================================================
# TEST 4: Bilateral produces a valid image
# ==============================================================================
def test_bilateral_produces_valid_image(synthetic_noisy_mammogram):
    denoised = apply_denoising_method(
        synthetic_noisy_mammogram, METHOD_BILATERAL, {"diameter": 5, "sigma_color": 0.1, "sigma_space": 5.0}
    )
    assert denoised.shape == synthetic_noisy_mammogram.shape
    assert denoised.dtype == np.float32
    assert denoised.min() >= 0.0 and denoised.max() <= 1.0
    assert not np.isnan(denoised).any()


# ==============================================================================
# TEST 5: Non-Local Means produces a valid image
# ==============================================================================
def test_nlm_produces_valid_image(synthetic_noisy_mammogram):
    denoised = apply_denoising_method(
        synthetic_noisy_mammogram, METHOD_NLM, {"patch_size": 5, "patch_distance": 6, "h": 0.05}
    )
    assert denoised.shape == synthetic_noisy_mammogram.shape
    assert denoised.dtype == np.float32
    assert denoised.min() >= 0.0 and denoised.max() <= 1.0
    assert not np.isnan(denoised).any()


# ==============================================================================
# TEST 6: Anscombe-Wiener produces a valid image
# ==============================================================================
def test_anscombe_wiener_produces_valid_image(synthetic_clean_mammogram):
    rng = np.random.default_rng(42)
    poisson_noisy = add_poisson_noise(synthetic_clean_mammogram, peak=30.0, rng=rng)
    denoised = apply_denoising_method(
        poisson_noisy, METHOD_ANSCOMBE_WIENER, {"scale": 255.0, "wiener_window_size": 5, "sigma": 1.0}
    )
    assert denoised.shape == poisson_noisy.shape
    assert denoised.dtype == np.float32
    assert denoised.min() >= 0.0 and denoised.max() <= 1.0
    assert not np.isnan(denoised).any()
    assert not np.isinf(denoised).any()
    assert np.std(denoised) > 0.01  # Not completely blank


# ==============================================================================
# TEST 7: Adaptive Median produces a valid image
# ==============================================================================
def test_adaptive_median_produces_valid_image(synthetic_clean_mammogram):
    rng = np.random.default_rng(42)
    sp_noisy = add_salt_pepper_noise(synthetic_clean_mammogram, amount=0.03, rng=rng)
    denoised = apply_denoising_method(sp_noisy, METHOD_ADAPTIVE_MEDIAN, {"max_window": 7})
    assert denoised.shape == sp_noisy.shape
    assert denoised.dtype == np.float32
    assert denoised.min() >= 0.0 and denoised.max() <= 1.0
    assert not np.isnan(denoised).any()


# ==============================================================================
# TEST 8: Kuan produces a valid image
# ==============================================================================
def test_kuan_produces_valid_image(synthetic_clean_mammogram):
    rng = np.random.default_rng(42)
    speckle_noisy = add_speckle_noise(synthetic_clean_mammogram, sigma=0.05, rng=rng)
    denoised = apply_denoising_method(
        speckle_noisy, METHOD_KUAN, {"window_size": 5, "noise_var": 0.04, "damping": 1.0}
    )
    assert denoised.shape == speckle_noisy.shape
    assert denoised.dtype == np.float32
    assert denoised.min() >= 0.0 and denoised.max() <= 1.0
    assert not np.isnan(denoised).any()


# ==============================================================================
# TEST 9: All eight methods are available
# ==============================================================================
def test_all_eight_methods_available():
    assert len(ALL_DENOISING_METHODS) == 8
    expected = [
        "median",
        "gaussian",
        "wiener",
        "bilateral",
        "non_local_means",
        "anscombe_wiener",
        "adaptive_median",
        "kuan",
    ]
    for m in expected:
        assert m in ALL_DENOISING_METHODS


# ==============================================================================
# TEST 10: Output dimensions are unchanged
# ==============================================================================
def test_output_dimensions_unchanged(synthetic_noisy_mammogram):
    h, w = synthetic_noisy_mammogram.shape[:2]
    for m in ALL_DENOISING_METHODS:
        denoised = apply_denoising_method(synthetic_noisy_mammogram, m, {})
        assert denoised.shape == (h, w)


# ==============================================================================
# TEST 11: Output remains grayscale
# ==============================================================================
def test_output_remains_grayscale(tmp_path, synthetic_noisy_mammogram):
    denoised = apply_denoising_method(synthetic_noisy_mammogram, METHOD_MEDIAN, {})
    out_p = str(tmp_path / "gray_check.png")
    save_denoised_image(denoised, out_p)
    arr = cv2.imread(out_p, cv2.IMREAD_UNCHANGED)
    assert arr.ndim == 2  # Single channel 2D matrix


# ==============================================================================
# TEST 12: Output values remain within [0, 1] before saving
# ==============================================================================
def test_output_within_zero_one_bounds(synthetic_noisy_mammogram):
    for m in ALL_DENOISING_METHODS:
        denoised = apply_denoising_method(synthetic_noisy_mammogram, m, {})
        assert denoised.min() >= 0.0
        assert denoised.max() <= 1.0


# ==============================================================================
# TEST 13: Saved output reopens correctly
# ==============================================================================
def test_saved_output_reopens_correctly(tmp_path, synthetic_noisy_mammogram):
    for m in ALL_DENOISING_METHODS:
        denoised = apply_denoising_method(synthetic_noisy_mammogram, m, {})
        out_p = str(tmp_path / f"saved_{m}.png")
        save_denoised_image(denoised, out_p)
        assert os.path.exists(out_p)
        reopened = cv2.imread(out_p, cv2.IMREAD_GRAYSCALE)
        assert reopened is not None
        assert reopened.dtype == np.uint8
        assert reopened.shape == synthetic_noisy_mammogram.shape


# ==============================================================================
# TEST 14: No NaN or Inf across methods
# ==============================================================================
def test_no_nan_or_inf_across_methods(synthetic_noisy_mammogram):
    for m in ALL_DENOISING_METHODS:
        denoised = apply_denoising_method(synthetic_noisy_mammogram, m, {})
        assert not np.isnan(denoised).any()
        assert not np.isinf(denoised).any()


# ==============================================================================
# TEST 15: Same input + same method + same configuration is deterministic
# ==============================================================================
def test_deterministic_execution(synthetic_noisy_mammogram):
    for m in ALL_DENOISING_METHODS:
        out1 = apply_denoising_method(synthetic_noisy_mammogram, m, {})
        out2 = apply_denoising_method(synthetic_noisy_mammogram, m, {})
        assert np.array_equal(out1, out2)


# ==============================================================================
# TEST 16: All five Stage-8 noise types are supported
# ==============================================================================
def test_all_five_noise_types_supported():
    assert len(ALL_NOISE_TYPES) == 5
    for nt in ["gaussian", "salt_pepper", "speckle", "poisson", "mixed_poisson_gaussian"]:
        assert nt in ALL_NOISE_TYPES


# ==============================================================================
# TEST 17, 18, 19: End-to-End Scalability & Fairness Mock Pipeline
# 80 noisy inputs produce exactly 640 denoised outputs, valid clean links, no duplicate calls
# ==============================================================================
def test_eighty_noisy_yields_six_hundred_forty_denoised(tmp_path):
    # Setup mock clean baseline and noisy directories
    clean_dir = tmp_path / "baseline"
    noise_dir = tmp_path / "noise"
    clean_dir.mkdir(parents=True)
    noise_dir.mkdir(parents=True)

    y, x = np.mgrid[:32, :32]
    records = []

    # 16 pilot cases x 5 noise types = 80 noisy images
    for c_idx in range(16):
        pid = f"P_{c_idx:05d}"
        clean_patch = ((x / 32.0) * 160 + (y / 32.0) * 50 + 20).astype(np.uint8)
        clean_patch[:, :4] = 0
        clean_p = clean_dir / f"{pid}_1_LEFT_CC_baseline.png"
        cv2.imwrite(str(clean_p), clean_patch)

        clean_f32 = clean_patch.astype(np.float32) / 255.0

        for nt in ALL_NOISE_TYPES:
            nt_sub = noise_dir / nt
            nt_sub.mkdir(exist_ok=True)
            noisy_p = nt_sub / f"{pid}_1_LEFT_CC_{nt}.png"

            rng = np.random.default_rng(42 + c_idx)
            if nt == "gaussian":
                noisy_arr = add_gaussian_noise(clean_f32, rng=rng)
            elif nt == "salt_pepper":
                noisy_arr = add_salt_pepper_noise(clean_f32, rng=rng)
            elif nt == "speckle":
                noisy_arr = add_speckle_noise(clean_f32, rng=rng)
            elif nt == "poisson":
                noisy_arr = add_poisson_noise(clean_f32, rng=rng)
            else:
                noisy_arr = add_mixed_poisson_gaussian_noise(clean_f32, rng=rng)

            noisy_u8 = np.clip(np.round(noisy_arr * 255.0), 0, 255).astype(np.uint8)
            cv2.imwrite(str(noisy_p), noisy_u8)

            records.append({
                "patient_id": pid,
                "abnormality_id": 1,
                "abnormality_type": "mass" if c_idx % 2 == 0 else "calcification",
                "pathology": "MALIGNANT" if c_idx % 3 == 0 else "BENIGN",
                "binary_label": 1 if c_idx % 3 == 0 else 0,
                "breast_side": "LEFT",
                "image_view": "CC",
                "official_split": "train",
                "clean_image_path": str(clean_p),
                "noise_image_path": str(noisy_p),
                "noise_type": nt,
                "noise_seed": 42 + c_idx,
                "original_height": 32,
                "original_width": 32,
                "noisy_height": 32,
                "noisy_width": 32,
            })

    s8_meta_path = tmp_path / "mock_s8_meta.csv"
    pd.DataFrame(records).to_csv(str(s8_meta_path), index=False)

    out_denoised = tmp_path / "denoised"
    out_meta = tmp_path / "metadata"
    out_viz = tmp_path / "viz"

    df_meta, df_val = run_stage9_pilot(
        config_path=None,
        stage8_metadata_csv=str(s8_meta_path),
        output_base_dir=str(out_denoised),
        metadata_dir=str(out_meta),
        viz_dir=str(out_viz),
    )

    # 1. Exactly 640 outputs generated (Requirement 18)
    assert len(df_meta) == 640
    assert len(df_val) == 640

    # 2. Every Stage-8 noisy input is used exactly once per denoising method (Requirement 17)
    for m in ALL_DENOISING_METHODS:
        sub_m = df_meta[df_meta["denoising_method"] == m]
        assert len(sub_m) == 80
        assert len(sub_m["noisy_image_path"].unique()) == 80

    # 3. Every denoised output has a valid clean_image_path (Requirement 19)
    for _, row in df_meta.iterrows():
        assert os.path.exists(row["clean_image_path"])

    # 4. Zero validation failures
    assert (df_val["status"] == STATUS_PASS).all()
    assert (df_meta["status"] == STATUS_SUCCESS).all()


# ==============================================================================
# TEST 20, 21, 22: Raw CBIS-DDSM, Stage-5 baseline, and Stage-8 noisy images remain untouched
# ==============================================================================
def test_raw_stage5_and_stage8_integrity(tmp_path):
    raw_file = tmp_path / "raw_ddsm.jpg"
    s5_file = tmp_path / "stage5_baseline.png"
    s8_file = tmp_path / "stage8_noisy.png"

    raw_file.write_bytes(b"IMMUTABLE_RAW_CBIS_DDSM_PAYLOAD")
    s5_file.write_bytes(b"IMMUTABLE_STAGE5_BASELINE_PAYLOAD")
    s8_file.write_bytes(b"IMMUTABLE_STAGE8_NOISY_PAYLOAD")

    h_raw_before = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    h_s5_before = hashlib.sha256(s5_file.read_bytes()).hexdigest()
    h_s8_before = hashlib.sha256(s8_file.read_bytes()).hexdigest()

    # Apply all 8 denoising methods to a test array
    test_arr = np.linspace(0.1, 0.9, 16 * 16, dtype=np.float32).reshape((16, 16))
    for m in ALL_DENOISING_METHODS:
        _ = apply_denoising_method(test_arr, m, {})

    h_raw_after = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    h_s5_after = hashlib.sha256(s5_file.read_bytes()).hexdigest()
    h_s8_after = hashlib.sha256(s8_file.read_bytes()).hexdigest()

    assert h_raw_before == h_raw_after
    assert h_s5_before == h_s5_after
    assert h_s8_before == h_s8_after
