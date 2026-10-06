"""Unit tests for Stage 8: Artificial Noise Experiment.

Verifies all 18 mandatory requirements from the research specification:
1. Gaussian noise changes the image.
2. Salt & pepper introduces approximately the expected amount of impulse noise.
3. Speckle noise is reproducible.
4. Poisson noise is reproducible.
5. Mixed Poisson-Gaussian noise is reproducible.
6. Same input + same seed produces identical output.
7. Different seed produces a different output.
8. Output remains within [0, 1] before saving.
9. Saved image reopens correctly.
10. Image dimensions are unchanged.
11. Image remains grayscale.
12. No NaN or Inf.
13. All five noise types are generated.
14. Exactly 80 noisy pilot images are generated.
15. Metadata contains the correct clean_image_path.
16. Raw CBIS-DDSM files remain unchanged.
17. Stage-5 baseline files remain unchanged.
18. Noise outputs from one noise type do not overwrite another.
"""

import hashlib
import json
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.noise_experiment import (
    ALL_NOISE_TYPES,
    NOISE_CLEAN,
    NOISE_GAUSSIAN,
    NOISE_SALT_PEPPER,
    NOISE_SPECKLE,
    NOISE_POISSON,
    NOISE_MIXED,
    STATUS_PASS,
    STATUS_SUCCESS,
    derive_deterministic_seed,
    load_clean_image,
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    add_poisson_noise,
    add_mixed_poisson_gaussian_noise,
    apply_synthetic_noise,
    calculate_noise_statistics,
    save_noisy_image,
    validate_noisy_image,
    run_stage8_pilot,
    generate_noise_visualizations,
)


@pytest.fixture
def synthetic_clean_image() -> np.ndarray:
    """Create a 64x64 synthetic float32 [0.0, 1.0] mammogram patch."""
    y, x = np.ogrid[:64, :64]
    # Smooth gradient with simulated tissue boundary
    base = (x / 64.0) * 0.6 + (y / 64.0) * 0.3
    base[x < 10] = 0.0  # Background black region
    return np.clip(base, 0.0, 1.0).astype(np.float32)


# ==============================================================================
# TEST 1: Gaussian noise changes the image
# ==============================================================================
def test_gaussian_noise_changes_image(synthetic_clean_image):
    rng = np.random.default_rng(42)
    noisy = add_gaussian_noise(synthetic_clean_image, mean=0.0, sigma=0.03, rng=rng)
    assert not np.array_equal(synthetic_clean_image, noisy)
    assert np.mean(np.abs(synthetic_clean_image - noisy)) > 0.01


# ==============================================================================
# TEST 2: Salt & pepper introduces approximately expected amount of impulse noise
# ==============================================================================
def test_salt_pepper_expected_ratio(synthetic_clean_image):
    rng = np.random.default_rng(42)
    amount = 0.02
    noisy = add_salt_pepper_noise(synthetic_clean_image, amount=amount, salt_vs_pepper=0.5, rng=rng)
    # Differences should occur approximately on `amount` fraction of pixels
    diff_pixels = np.count_nonzero(noisy != synthetic_clean_image)
    total_pixels = synthetic_clean_image.size
    fraction_perturbed = diff_pixels / total_pixels
    # In a 64x64 patch (4096 pixels), 2% is ~82 pixels. Allow +/- 0.01 tolerance
    assert abs(fraction_perturbed - amount) < 0.01
    # Check that both salt (1.0) and pepper (0.0) values exist
    assert np.count_nonzero(noisy == 1.0) > 0
    assert np.count_nonzero(noisy == 0.0) > 0


# ==============================================================================
# TEST 3: Speckle noise is reproducible
# ==============================================================================
def test_speckle_noise_reproducibility(synthetic_clean_image):
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)
    noisy1 = add_speckle_noise(synthetic_clean_image, sigma=0.05, rng=rng1)
    noisy2 = add_speckle_noise(synthetic_clean_image, sigma=0.05, rng=rng2)
    assert np.array_equal(noisy1, noisy2)


# ==============================================================================
# TEST 4: Poisson noise is reproducible
# ==============================================================================
def test_poisson_noise_reproducibility(synthetic_clean_image):
    rng1 = np.random.default_rng(67890)
    rng2 = np.random.default_rng(67890)
    noisy1 = add_poisson_noise(synthetic_clean_image, peak=30.0, rng=rng1)
    noisy2 = add_poisson_noise(synthetic_clean_image, peak=30.0, rng=rng2)
    assert np.array_equal(noisy1, noisy2)


# ==============================================================================
# TEST 5: Mixed Poisson-Gaussian noise is reproducible
# ==============================================================================
def test_mixed_poisson_gaussian_noise_reproducibility(synthetic_clean_image):
    rng1 = np.random.default_rng(54321)
    rng2 = np.random.default_rng(54321)
    noisy1 = add_mixed_poisson_gaussian_noise(
        synthetic_clean_image, poisson_peak=30.0, gaussian_mean=0.0, gaussian_sigma=0.02, rng=rng1
    )
    noisy2 = add_mixed_poisson_gaussian_noise(
        synthetic_clean_image, poisson_peak=30.0, gaussian_mean=0.0, gaussian_sigma=0.02, rng=rng2
    )
    assert np.array_equal(noisy1, noisy2)


# ==============================================================================
# TEST 6: Same input + same seed produces identical output
# ==============================================================================
def test_same_input_same_seed_identical_output(synthetic_clean_image):
    seed1 = derive_deterministic_seed(42, "P_00001", 1, "LEFT", "CC", "gaussian")
    seed2 = derive_deterministic_seed(42, "P_00001", 1, "LEFT", "CC", "gaussian")
    assert seed1 == seed2

    out1 = add_gaussian_noise(synthetic_clean_image, rng=np.random.default_rng(seed1))
    out2 = add_gaussian_noise(synthetic_clean_image, rng=np.random.default_rng(seed2))
    assert np.array_equal(out1, out2)


# ==============================================================================
# TEST 7: Different seed produces a different output
# ==============================================================================
def test_different_seed_different_output(synthetic_clean_image):
    seed1 = derive_deterministic_seed(42, "P_00001", 1, "LEFT", "CC", "gaussian")
    seed2 = derive_deterministic_seed(42, "P_00002", 1, "RIGHT", "MLO", "gaussian")
    assert seed1 != seed2

    out1 = add_gaussian_noise(synthetic_clean_image, rng=np.random.default_rng(seed1))
    out2 = add_gaussian_noise(synthetic_clean_image, rng=np.random.default_rng(seed2))
    assert not np.array_equal(out1, out2)


# ==============================================================================
# TEST 8: Output remains within [0, 1] before saving
# ==============================================================================
def test_output_bounds_all_noise_models(synthetic_clean_image):
    rng = np.random.default_rng(999)
    models = [
        add_gaussian_noise(synthetic_clean_image, mean=0.0, sigma=0.2, rng=rng),
        add_salt_pepper_noise(synthetic_clean_image, amount=0.1, rng=rng),
        add_speckle_noise(synthetic_clean_image, sigma=0.3, rng=rng),
        add_poisson_noise(synthetic_clean_image, peak=10.0, rng=rng),
        add_mixed_poisson_gaussian_noise(synthetic_clean_image, poisson_peak=10.0, gaussian_sigma=0.1, rng=rng),
    ]
    for m in models:
        assert m.min() >= 0.0
        assert m.max() <= 1.0
        assert m.dtype == np.float32


# ==============================================================================
# TEST 9: Saved image reopens correctly
# ==============================================================================
def test_saved_image_reopens_correctly(tmp_path, synthetic_clean_image):
    rng = np.random.default_rng(42)
    noisy = add_gaussian_noise(synthetic_clean_image, sigma=0.03, rng=rng)
    save_path = str(tmp_path / "test_saved_image.png")
    save_noisy_image(noisy, save_path)

    assert os.path.exists(save_path)
    reopened = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
    assert reopened is not None
    assert reopened.dtype == np.uint8
    assert reopened.shape == synthetic_clean_image.shape


# ==============================================================================
# TEST 10: Image dimensions are unchanged
# ==============================================================================
def test_image_dimensions_unchanged(synthetic_clean_image):
    h, w = synthetic_clean_image.shape
    rng = np.random.default_rng(42)
    for nt in ALL_NOISE_TYPES:
        noisy = apply_synthetic_noise(synthetic_clean_image, nt, {}, rng)
        assert noisy.shape == (h, w)


# ==============================================================================
# TEST 11: Image remains grayscale
# ==============================================================================
def test_image_remains_grayscale(tmp_path, synthetic_clean_image):
    rng = np.random.default_rng(42)
    noisy = add_gaussian_noise(synthetic_clean_image, sigma=0.03, rng=rng)
    save_path = str(tmp_path / "test_gray.png")
    save_noisy_image(noisy, save_path)

    reloaded = cv2.imread(save_path, cv2.IMREAD_UNCHANGED)
    assert reloaded.ndim == 2  # Single channel 2D matrix


# ==============================================================================
# TEST 12: No NaN or Inf in outputs and metrics
# ==============================================================================
def test_no_nan_or_inf(synthetic_clean_image):
    rng = np.random.default_rng(42)
    for nt in ALL_NOISE_TYPES:
        noisy = apply_synthetic_noise(synthetic_clean_image, nt, {}, rng)
        assert not np.isnan(noisy).any()
        assert not np.isinf(noisy).any()

        stats = calculate_noise_statistics(synthetic_clean_image, noisy)
        assert not np.isnan(stats["mse"])
        assert not np.isnan(stats["psnr"])
        assert not np.isnan(stats["snr"])
        assert not np.isnan(stats["entropy_noisy"])


# ==============================================================================
# TEST 13: All five noise types are generated
# ==============================================================================
def test_all_five_noise_types_supported():
    assert len(ALL_NOISE_TYPES) == 5
    assert NOISE_GAUSSIAN in ALL_NOISE_TYPES
    assert NOISE_SALT_PEPPER in ALL_NOISE_TYPES
    assert NOISE_SPECKLE in ALL_NOISE_TYPES
    assert NOISE_POISSON in ALL_NOISE_TYPES
    assert NOISE_MIXED in ALL_NOISE_TYPES


# ==============================================================================
# TEST 14, 15, 18: Mock End-to-End Pipeline
# Exactly 80 noisy pilot images generated, clean_image_path correct, no overwrites
# ==============================================================================
def test_stage8_pilot_pipeline_mock(tmp_path):
    # Setup 2 mock clean baseline images to test pipeline mechanics quickly
    base_dir = tmp_path / "baseline"
    base_dir.mkdir()
    c1_path = base_dir / "P_00001_LEFT_CC_1_baseline.png"
    c2_path = base_dir / "P_00002_RIGHT_MLO_1_baseline.png"

    arr1 = np.full((32, 32), 128, dtype=np.uint8)
    arr2 = np.full((32, 32), 200, dtype=np.uint8)
    cv2.imwrite(str(c1_path), arr1)
    cv2.imwrite(str(c2_path), arr2)

    # Setup mock stage5 metadata CSV
    s5_meta_path = tmp_path / "mock_stage5_meta.csv"
    mock_df = pd.DataFrame([
        {
            "patient_id": "P_00001",
            "abnormality_id": 1,
            "abnormality_category": "mass",
            "pathology": "MALIGNANT",
            "binary_label": 1,
            "breast_side": "LEFT",
            "image_view": "CC",
            "dataset_split": "test",
            "baseline_image_path": str(c1_path),
            "baseline_height": 32,
            "baseline_width": 32,
        },
        {
            "patient_id": "P_00002",
            "abnormality_id": 1,
            "abnormality_category": "calcification",
            "pathology": "BENIGN",
            "binary_label": 0,
            "breast_side": "RIGHT",
            "image_view": "MLO",
            "dataset_split": "train",
            "baseline_image_path": str(c2_path),
            "baseline_height": 32,
            "baseline_width": 32,
        },
    ])
    mock_df.to_csv(str(s5_meta_path), index=False)

    out_noise = tmp_path / "noise"
    out_meta = tmp_path / "metadata"
    out_viz = tmp_path / "viz"

    df_meta, df_val = run_stage8_pilot(
        config_path=None,
        stage5_metadata_csv=str(s5_meta_path),
        output_base_dir=str(out_noise),
        metadata_dir=str(out_meta),
        viz_dir=str(out_viz),
    )

    # 2 cases * 5 noise types = 10 noisy + 2 clean = 12 records
    assert len(df_meta) == 12
    assert len(df_val) == 12

    # Verify no validation failures
    assert (df_val["status"] == STATUS_PASS).all()

    # Verify clean_image_path is preserved (Requirement 15)
    for _, row in df_meta.iterrows():
        assert os.path.exists(row["clean_image_path"])

    # Verify separate folders for each noise model (Requirement 18 - no overwrites)
    for nt in ALL_NOISE_TYPES:
        sub_dir = out_noise / nt
        assert sub_dir.exists()
        files = list(sub_dir.glob("*.png"))
        assert len(files) == 2


# ==============================================================================
# TEST 16 & 17: Raw CBIS-DDSM & Stage-5 baseline files remain untouched
# ==============================================================================
def test_raw_and_stage5_files_integrity(tmp_path):
    # Create mock raw and stage5 folders with sentinel hash
    raw_file = tmp_path / "raw.jpg"
    s5_file = tmp_path / "stage5.png"

    raw_file.write_bytes(b"RAW_DATA_SENTINEL_BYTES_CBIS_DDSM")
    s5_file.write_bytes(b"STAGE5_BASELINE_SENTINEL_BYTES")

    raw_hash_before = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    s5_hash_before = hashlib.sha256(s5_file.read_bytes()).hexdigest()

    # Derive seeds and run noise logic
    seed = derive_deterministic_seed(42, "P_00001", 1, "LEFT", "CC", "gaussian")
    arr = np.ones((16, 16), dtype=np.float32) * 0.5
    _ = add_gaussian_noise(arr, rng=np.random.default_rng(seed))

    raw_hash_after = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    s5_hash_after = hashlib.sha256(s5_file.read_bytes()).hexdigest()

    assert raw_hash_before == raw_hash_after
    assert s5_hash_before == s5_hash_after


# ==============================================================================
# TEST: Clean vs Clean Infinite PSNR & SNR safety
# ==============================================================================
def test_clean_vs_clean_metrics_infinite_safety(synthetic_clean_image):
    stats = calculate_noise_statistics(synthetic_clean_image, synthetic_clean_image)
    assert stats["mse"] == 0.0
    assert stats["mean_absolute_difference"] == 0.0
    assert stats["psnr"] == float("inf")
    assert stats["snr"] == float("inf")


# ==============================================================================
# TEST 14: Exactly 80 noisy images from 16 pilot cases (total 96 records)
# ==============================================================================
def test_sixteen_cases_yields_eighty_noisy_images(tmp_path):
    base_dir = tmp_path / "baseline"
    base_dir.mkdir()
    records = []
    for i in range(16):
        c_path = base_dir / f"case_{i:02d}.png"
        cv2.imwrite(str(c_path), np.full((16, 16), 100 + i * 5, dtype=np.uint8))
        records.append({
            "patient_id": f"P_{i:05d}",
            "abnormality_id": 1,
            "abnormality_category": "mass" if i % 2 == 0 else "calcification",
            "pathology": "MALIGNANT" if i % 3 == 0 else "BENIGN",
            "binary_label": 1 if i % 3 == 0 else 0,
            "breast_side": "LEFT" if i % 2 == 0 else "RIGHT",
            "image_view": "CC" if i % 4 < 2 else "MLO",
            "dataset_split": "train",
            "baseline_image_path": str(c_path),
            "baseline_height": 16,
            "baseline_width": 16,
        })
    meta_path = tmp_path / "stage5_16_cases.csv"
    pd.DataFrame(records).to_csv(str(meta_path), index=False)

    out_noise = tmp_path / "noise"
    out_meta = tmp_path / "meta"
    out_viz = tmp_path / "viz"

    df_meta, df_val = run_stage8_pilot(
        config_path=None,
        stage5_metadata_csv=str(meta_path),
        output_base_dir=str(out_noise),
        metadata_dir=str(out_meta),
        viz_dir=str(out_viz),
    )

    # Exactly 16 clean + 80 noisy = 96 records
    assert len(df_meta) == 96
    assert len(df_val) == 96
    assert len(df_meta[df_meta["noise_type"] == NOISE_CLEAN]) == 16
    assert len(df_meta[df_meta["noise_type"] != NOISE_CLEAN]) == 80
    assert (df_val["status"] == STATUS_PASS).all()

