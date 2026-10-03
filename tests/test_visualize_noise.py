"""Unit tests for Step 8 noise visualization and realism validation."""
import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.noise.visualize_noise import (
    validate_noise_realism,
    plot_noise_comparison,
    run_noise_visualizations,
)


@pytest.fixture
def sample_clean_and_noisy():
    """Create a mock clean mammogram patch and 5 realistic noisy variations."""
    np.random.seed(42)
    clean = np.linspace(30, 220, 64 * 64, dtype=np.uint8).reshape((64, 64))

    # 1. Gaussian
    gauss = np.clip(clean.astype(np.float32) + np.random.normal(0, 10, clean.shape), 0, 255).astype(np.uint8)

    # 2. Salt & Pepper
    sp = clean.copy()
    sp[0, 0] = 255
    sp[0, 1] = 0

    # 3. Speckle
    speckle = np.clip(clean.astype(np.float32) * (1 + np.random.normal(0, 0.1, clean.shape)), 0, 255).astype(np.uint8)

    # 4. Poisson
    poisson = np.clip(np.random.poisson(clean.astype(np.float32)), 0, 255).astype(np.uint8)

    # 5. Mixed
    mixed = np.clip(poisson.astype(np.float32) + np.random.normal(0, 5, clean.shape), 0, 255).astype(np.uint8)

    noisy_dict = {
        "gaussian": gauss,
        "salt_pepper": sp,
        "speckle": speckle,
        "poisson": poisson,
        "mixed_poisson_gaussian": mixed,
    }
    return clean, noisy_dict


def test_validate_noise_realism_pass(sample_clean_and_noisy):
    clean, noisy_dict = sample_clean_and_noisy
    report = validate_noise_realism(clean, noisy_dict, "P_TEST")
    assert report["overall_status"] == "PASSED"
    for ntype in ["gaussian", "salt_pepper", "speckle", "poisson", "mixed_poisson_gaussian"]:
        assert report[ntype]["status"] == "PASSED"
        assert report[ntype]["metrics"]["mse"] > 0


def test_validate_noise_realism_fail_impulse(sample_clean_and_noisy):
    clean, noisy_dict = sample_clean_and_noisy
    # Corrupt salt & pepper so neither 0 nor 255 exists
    bad_sp = np.clip(clean.copy(), 10, 240)
    noisy_dict["salt_pepper"] = bad_sp
    report = validate_noise_realism(clean, noisy_dict, "P_FAIL")
    assert report["salt_pepper"]["status"] == "FAILED"
    assert report["overall_status"] == "FAILED"


def test_plot_noise_comparison(sample_clean_and_noisy, tmp_path):
    clean, noisy_dict = sample_clean_and_noisy
    save_fig = tmp_path / "test_comparison.png"

    plot_noise_comparison(
        clean_img=clean,
        noisy_dict=noisy_dict,
        patient_id="P_00044",
        pathology="BENIGN",
        label=0,
        save_path=str(save_fig),
        params_dict={"gaussian": {"mean": 0.0, "var": 0.01}},
    )

    assert save_fig.exists()
    assert save_fig.stat().st_size > 0


def test_run_noise_visualizations_pipeline(sample_clean_and_noisy, tmp_path):
    clean, noisy_dict = sample_clean_and_noisy

    # Save mock files
    data_dir = tmp_path / "data"
    clean_dir = data_dir / "sharpened"
    clean_dir.mkdir(parents=True)
    clean_path = clean_dir / "case_001.png"
    cv2.imwrite(str(clean_path), clean)

    noisy_dir = data_dir / "noisy"
    meta_rows = []
    for ntype, img in noisy_dict.items():
        type_dir = noisy_dir / ntype
        type_dir.mkdir(parents=True)
        img_path = type_dir / "case_001.png"
        cv2.imwrite(str(img_path), img)

        meta_rows.append({
            "noise_type": ntype,
            "noise_parameters": '{"dummy": 1}',
            "source_image": str(clean_path),
            "output_image": str(img_path),
            "patient_id": "P_00044",
            "pathology": "BENIGN",
            "label": 0,
            "is_synthetic": True,
        })

    meta_csv = tmp_path / "noisy_metadata.csv"
    pd.DataFrame(meta_rows).to_csv(str(meta_csv), index=False)

    out_res_dir = tmp_path / "results" / "noise_comparisons"

    summary = run_noise_visualizations(
        noisy_metadata_csv=str(meta_csv),
        output_dir=str(out_res_dir),
        max_images=1,
    )

    assert len(summary) == 1
    assert summary[0]["patient_id"] == "P_00044"
    assert summary[0]["status"] == "PASSED"
    assert os.path.exists(summary[0]["figure_path"])
