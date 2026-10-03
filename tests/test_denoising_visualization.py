"""Unit tests for Step 12 denoising visualization and comparative summary."""
import os
import pytest
import pandas as pd
import numpy as np

from src.evaluation.denoising_visualization import run_denoising_visualization


def test_run_denoising_visualization_pipeline(tmp_path):
    # Setup mock metrics CSV
    mock_metrics_data = []
    noise_models = ["gaussian", "salt_pepper"]
    filters = ["median", "gaussian"]

    for n_type in noise_models:
        for f_name in filters:
            for img_idx in range(3):
                mock_metrics_data.append({
                    "patient_id": f"P_000{img_idx}",
                    "abnormality_category": "mass",
                    "breast_side": "LEFT",
                    "image_view": "CC",
                    "pathology": "BENIGN",
                    "label": 0,
                    "noise_type": n_type,
                    "denoising_method": f_name,
                    "MSE": float(np.random.uniform(0.001, 0.01)),
                    "PSNR": float(np.random.uniform(25.0, 35.0)),
                    "SSIM": float(np.random.uniform(0.7, 0.95)),
                    "SNR": float(np.random.uniform(18.0, 28.0)),
                    "CNR": float(np.random.uniform(2.0, 5.0)),
                    "CII": float(np.random.uniform(0.9, 1.4)),
                    "Entropy": float(np.random.uniform(5.0, 7.0)),
                    "source_image": f"/app/data/clean_{img_idx}.png",
                    "noisy_image": f"/app/data/{n_type}_{img_idx}.png",
                    "denoised_image": f"/app/data/{f_name}_{img_idx}.png",
                })

    metrics_csv = tmp_path / "mock_metrics.csv"
    pd.DataFrame(mock_metrics_data).to_csv(str(metrics_csv), index=False)

    out_dir = tmp_path / "comparison_plots"
    summary_csv = tmp_path / "denoising_summary.csv"

    summary_df, saved_plots = run_denoising_visualization(
        metrics_csv=str(metrics_csv),
        output_dir=str(out_dir),
        summary_csv=str(summary_csv),
    )

    # 2 noise types x 2 filters = 4 aggregated rows
    assert len(summary_df) == 4
    assert os.path.exists(str(summary_csv))

    expected_cols = [
        "noise_type",
        "denoising_method",
        "MSE_mean",
        "PSNR_mean",
        "SSIM_mean",
        "SNR_mean",
        "CNR_mean",
        "CII_mean",
        "Entropy_mean",
    ]
    for col in expected_cols:
        assert col in summary_df.columns

    # Verify that all 7 required metric plots exist
    required_plots = [
        "psnr_comparison.png",
        "ssim_comparison.png",
        "mse_comparison.png",
        "snr_comparison.png",
        "cnr_comparison.png",
        "cii_comparison.png",
        "entropy_comparison.png",
        "psnr_heatmap.png",
        "ssim_heatmap.png",
        "denoising_benchmark_dashboard.png",
    ]
    for p_name in required_plots:
        p_path = out_dir / p_name
        assert p_path.exists(), f"Missing plot: {p_name}"
        assert p_path.stat().st_size > 0, f"Empty plot: {p_name}"
