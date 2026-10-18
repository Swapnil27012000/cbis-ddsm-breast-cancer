"""Stage 11: Preprocessing Visualization and Scientific Comparison Module.

Generates reproducible, publication-quality visual comparison panels, difference maps,
magnified detail crops, and quantitative metric visualizations for CBIS-DDSM mammography:
1. Panel A: Baseline & Contrast Enhancement (Clean Baseline, Control, HistEq, CLAHE).
2. Panel B: Sharpening Factorial Comparison (Control vs Unsharp-Masked across contrast variants).
3. Panel C: Synthetic Noise Simulation (Clean Reference vs 5 Noise Models).
4. Panel D: Denoising Benchmark Panels (Clean Reference, Noisy Input, 8 Denoised Outputs).
5. Difference Maps: Clean vs Noisy, Clean vs Denoised (Restoration Error), Noisy vs Denoised (Filtering Change).
6. Magnified Detail Crops: High-resolution crops of fibroglandular tissue and lesion boundaries.
7. Quantitative Metric Visualizations (Stage 10):
   - PSNR, SSIM, and MSE by Denoising Method (with N=5 sample counts and error bars).
   - Metrics by Synthetic Noise Type (with N=8 sample counts and error bars).
   - Method-by-Noise Heatmap of PSNR (8x5 matrix).
   - Method-by-Noise Heatmap of SSIM (8x5 matrix).
   - Whole-Image vs Foreground-Only Metrics (verifying breast threshold mask).
   - Comparison C (Filtering Magnitude: Noisy vs Denoised) kept strictly separate from Comparison B.
   - Rigorous handling of CNR (0 eligible cases documented; no misleading zero substitutions).
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
# Headless non-interactive backend safe for Docker & headless server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.stage11_audit import (
    resolve_path,
    find_stage_file,
    NOISE_TYPES,
    DENOISING_METHODS,
)
from src.utils.logger import setup_logger

logger = setup_logger("Stage11Visualization")

# Objective, colorblind-friendly distinct palette for 8 denoising methods
METHOD_PALETTE = {
    "median": "#1f77b4",          # Blue
    "gaussian": "#ff7f0e",        # Orange
    "wiener": "#2ca02c",          # Green
    "bilateral": "#d62728",       # Red
    "non_local_means": "#9467bd", # Purple
    "anscombe_wiener": "#8c564b", # Brown
    "adaptive_median": "#e377c2", # Pink
    "kuan": "#7f7f7f",            # Gray
}

NOISE_PALETTE = {
    "gaussian": "#4e79a7",
    "salt_pepper": "#f28e2b",
    "speckle": "#e15759",
    "poisson": "#76b7b2",
    "mixed_poisson_gaussian": "#59a14f",
}

# Representative Case Definitions (Deterministic Selection Rules)
REPRESENTATIVE_CASES = {
    "P_00004_1_LEFT_CC": {
        "description": "Benign Mass, Left CC view (Primary case linking directly to Stage 10 metrics)",
        "patient_id": "P_00004",
        "abnormality_id": 1,
        "breast_side": "LEFT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "abnormality_type": "mass",
        "dimensions": (5491, 2986),
        "detail_crop": {
            "ymin": 2100,
            "ymax": 2700,
            "xmin": 1200,
            "xmax": 1800,
            "purpose": "Dense fibroglandular parenchyma and circumscribed benign mass boundary",
        },
    },
    "P_00009_1_RIGHT_MLO": {
        "description": "Malignant Mass, Right MLO view",
        "patient_id": "P_00009",
        "abnormality_id": 1,
        "breast_side": "RIGHT",
        "image_view": "MLO",
        "pathology": "MALIGNANT",
        "abnormality_type": "mass",
        "dimensions": (5266, 2596),
        "detail_crop": {
            "ymin": 2200,
            "ymax": 2800,
            "xmin": 800,
            "xmax": 1400,
            "purpose": "Spiculated malignant mass lesion and pectoral margin",
        },
    },
    "P_00007_1_LEFT_CC": {
        "description": "Benign Calcification, Left CC view",
        "patient_id": "P_00007",
        "abnormality_id": 1,
        "breast_side": "LEFT",
        "image_view": "CC",
        "pathology": "BENIGN",
        "abnormality_type": "calcification",
        "dimensions": (5356, 2986),
        "detail_crop": {
            "ymin": 2000,
            "ymax": 2600,
            "xmin": 1100,
            "xmax": 1700,
            "purpose": "Punctate microcalcification clusters in fibroglandular tissue",
        },
    },
    "P_00299_1_LEFT_MLO": {
        "description": "Malignant Calcification, Left MLO view",
        "patient_id": "P_00299",
        "abnormality_id": 1,
        "breast_side": "LEFT",
        "image_view": "MLO",
        "pathology": "MALIGNANT",
        "abnormality_type": "calcification",
        "dimensions": (6301, 3736),
        "detail_crop": {
            "ymin": 2500,
            "ymax": 3100,
            "xmin": 1400,
            "xmax": 2000,
            "purpose": "Pleomorphic microcalcifications and architectural distortion",
        },
    },
}


def load_grayscale_safe(path: Optional[str]) -> Optional[np.ndarray]:
    """Safely load 8-bit grayscale image with path verification."""
    if not path:
        return None
    resolved = resolve_path(path)
    if not resolved or not os.path.exists(resolved):
        logger.warning(f"Image path unresolvable or missing: {path}")
        return None
    img = cv2.imread(resolved, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Failed to read image as grayscale: {resolved}")
    return img


# ==============================================================================
# 1. PANEL A: BASELINE & CONTRAST ENHANCEMENT
# ==============================================================================
def create_contrast_comparison_panel(
    output_path: str = "results/preprocessing/comparison/contrast_comparison_panel.png",
    case_keys: Optional[List[str]] = None,
) -> str:
    """Generate side-by-side contrast enhancement panel:

    Stage 5 Clean Baseline | Stage 6 Control | Global HistEq | CLAHE.
    """
    if case_keys is None:
        case_keys = ["P_00004_1_LEFT_CC", "P_00009_1_RIGHT_MLO"]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_cases = len(case_keys)
    fig, axes = plt.subplots(num_cases, 4, figsize=(18, 5.2 * num_cases), squeeze=False)
    plt.subplots_adjust(wspace=0.08, hspace=0.25)

    headers = [
        "Stage 5 Clean Baseline\n(Percentile [1, 99])",
        "Stage 6 Control\n(Unchanged Reference)",
        "Global HistEq\n(cv2.equalizeHist)",
        "CLAHE\n(clip=2.0, tileGrid=8x8)",
    ]

    for row_idx, ck in enumerate(case_keys):
        meta = REPRESENTATIVE_CASES.get(ck)
        if not meta:
            continue
        pid = meta["patient_id"]
        side = meta["breast_side"]
        view = meta["image_view"]
        abn = meta["abnormality_id"]
        cat = meta["abnormality_type"].upper()
        patho = meta["pathology"].upper()

        # Image paths
        p_base5 = f"data/processed/baseline/full_mammogram/{pid}_{side}_{view}_{abn}_baseline.png"
        p_ctrl6 = f"data/processed/baseline/full_mammogram/{pid}_{side}_{view}_{abn}_baseline.png"
        p_he = f"data/processed/contrast/histogram_equalization/{pid}_{side}_{view}_{abn}_histeq.png"
        p_clahe = f"data/processed/contrast/clahe/{pid}_{side}_{view}_{abn}_clahe.png"

        imgs = [
            load_grayscale_safe(p_base5),
            load_grayscale_safe(p_ctrl6),
            load_grayscale_safe(p_he),
            load_grayscale_safe(p_clahe),
        ]

        for col_idx, (img, hdr) in enumerate(zip(imgs, headers)):
            ax = axes[row_idx, col_idx]
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                # Compute display stats
                mean_val = np.mean(img)
                std_val = np.std(img)
                ax.set_xlabel(f"Mean={mean_val:.1f}, Std={std_val:.1f}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "Image Unavailable", ha="center", va="center", color="red")
                ax.set_facecolor("#f0f0f0")

            if row_idx == 0:
                ax.set_title(hdr, fontsize=11, fontweight="bold", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_idx, 0].set_ylabel(
            f"{pid} ({side} {view})\n{cat} | {patho}",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=15,
        )

    fig.suptitle(
        "CBIS-DDSM Pilot Preprocessing — Baseline & Contrast Enhancement Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated contrast comparison panel at {output_path}")
    return output_path


# ==============================================================================
# 2. PANEL B: SHARPENING COMPARISON
# ==============================================================================
def create_sharpening_comparison_panel(
    output_path: str = "results/preprocessing/comparison/sharpening_comparison_panel.png",
    case_key: str = "P_00004_1_LEFT_CC",
) -> str:
    """Generate sharpening factorial comparison:

    3 Contrast Conditions (Baseline, HistEq, CLAHE) x 2 Sharpening States (None vs Unsharp-Mask).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = REPRESENTATIVE_CASES.get(case_key)
    if not meta:
        case_key = "P_00004_1_LEFT_CC"
        meta = REPRESENTATIVE_CASES[case_key]

    pid = meta["patient_id"]
    side = meta["breast_side"]
    view = meta["image_view"]
    abn = meta["abnormality_id"]

    # Conditions: (Row Header, Control Path, Sharpened Path)
    conditions = [
        (
            "Baseline Control",
            f"data/processed/baseline/full_mammogram/{pid}_{side}_{view}_{abn}_baseline.png",
            f"data/processed/sharpening/baseline/unsharp/{pid}_{side}_{view}_{abn}_baseline_unsharp.png",
        ),
        (
            "Global HistEq",
            f"data/processed/contrast/histogram_equalization/{pid}_{side}_{view}_{abn}_histeq.png",
            f"data/processed/sharpening/histogram_equalization/unsharp/{pid}_{side}_{view}_{abn}_histeq_unsharp.png",
        ),
        (
            "CLAHE",
            f"data/processed/contrast/clahe/{pid}_{side}_{view}_{abn}_clahe.png",
            f"data/processed/sharpening/clahe/unsharp/{pid}_{side}_{view}_{abn}_clahe_unsharp.png",
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    plt.subplots_adjust(wspace=0.1, hspace=0.22)

    col_titles = [
        "Contrast Condition (Unsharpened)",
        "Unsharp-Masked (amount=1.0, sigma=1.0)",
        "Absolute Sharpening Difference\n(|Sharpened - Unsharpened| x 5)",
    ]

    for row_idx, (cond_name, p_ctrl, p_sharp) in enumerate(conditions):
        img_ctrl = load_grayscale_safe(p_ctrl)
        img_sharp = load_grayscale_safe(p_sharp)

        # Col 0: Control
        ax0 = axes[row_idx, 0]
        if img_ctrl is not None:
            ax0.imshow(img_ctrl, cmap="gray", vmin=0, vmax=255)
            ax0.set_xlabel(f"Mean={np.mean(img_ctrl):.1f}, Std={np.std(img_ctrl):.1f}", fontsize=9)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_ylabel(f"[{cond_name}]", fontsize=11, fontweight="bold", rotation=0, ha="right", va="center", labelpad=15)

        # Col 1: Sharpened
        ax1 = axes[row_idx, 1]
        if img_sharp is not None:
            ax1.imshow(img_sharp, cmap="gray", vmin=0, vmax=255)
            ax1.set_xlabel(f"Mean={np.mean(img_sharp):.1f}, Std={np.std(img_sharp):.1f}", fontsize=9)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Col 2: Difference Map (scaled for visibility)
        ax2 = axes[row_idx, 2]
        if img_ctrl is not None and img_sharp is not None:
            diff = np.abs(img_sharp.astype(np.float32) - img_ctrl.astype(np.float32))
            # Amplified 5x for visual edge inspection, clamped to 255
            diff_vis = np.clip(diff * 5.0, 0, 255)
            im_diff = ax2.imshow(diff_vis, cmap="inferno", vmin=0, vmax=255)
            ax2.set_xlabel(f"MeanDiff={np.mean(diff):.2f}, MaxDiff={np.max(diff):.0f}", fontsize=9)
            plt.colorbar(im_diff, ax=ax2, fraction=0.046, pad=0.04, label="Scaled Difference [0, 255]")
        ax2.set_xticks([])
        ax2.set_yticks([])

        if row_idx == 0:
            for c_idx, title in enumerate(col_titles):
                axes[0, c_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    fig.suptitle(
        f"CBIS-DDSM Pilot Sharpening Factorial Evaluation — Case {pid}_{side}_{view}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated sharpening comparison panel at {output_path}")
    return output_path


# ==============================================================================
# 3. PANEL C: SYNTHETIC NOISE SIMULATION
# ==============================================================================
def create_clean_vs_noisy_panel(
    output_path: str = "results/preprocessing/comparison/clean_vs_noisy_panel.png",
    case_key: str = "P_00004_1_LEFT_CC",
) -> str:
    """Generate 6-column synthetic noise panel:

    Clean Reference + 5 Synthetic Noise Models.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = REPRESENTATIVE_CASES.get(case_key, REPRESENTATIVE_CASES["P_00004_1_LEFT_CC"])
    pid = meta["patient_id"]
    side = meta["breast_side"]
    view = meta["image_view"]
    abn = meta["abnormality_id"]

    noise_items = [
        ("Clean Baseline\n(Reference)", f"data/processed/baseline/full_mammogram/{pid}_{side}_{view}_{abn}_baseline.png"),
        ("Gaussian Noise\n(mean=0, sigma=0.03)", f"data/processed/noise/gaussian/{pid}_{abn}_{side}_{view}_gaussian.png"),
        ("Salt & Pepper\n(amount=0.01, 50/50)", f"data/processed/noise/salt_pepper/{pid}_{abn}_{side}_{view}_salt_pepper.png"),
        ("Speckle Noise\n(multiplicative, sigma=0.05)", f"data/processed/noise/speckle/{pid}_{abn}_{side}_{view}_speckle.png"),
        ("Poisson Noise\n(quantum shot, peak=30)", f"data/processed/noise/poisson/{pid}_{abn}_{side}_{view}_poisson.png"),
        ("Mixed Poisson-Gaussian\n(dual shot + electronic)", f"data/processed/noise/mixed_poisson_gaussian/{pid}_{abn}_{side}_{view}_mixed_poisson_gaussian.png"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(22, 5.5))
    plt.subplots_adjust(wspace=0.08)

    for idx, (label, p_img) in enumerate(noise_items):
        ax = axes[idx]
        img = load_grayscale_safe(p_img)
        if img is not None:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            mean_val = np.mean(img)
            std_val = np.std(img)
            ax.set_xlabel(f"Mean={mean_val:.1f}\nStd={std_val:.1f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Image Missing", ha="center", va="center", color="red")
            ax.set_facecolor("#f0f0f0")
        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"CBIS-DDSM Synthetic Noise Simulation Models — Case {pid}_{side}_{view} ({meta['pathology'].upper()} {meta['abnormality_type'].upper()})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated clean vs noisy panel at {output_path}")
    return output_path


# ==============================================================================
# 4. PANEL D: DENOISING BENCHMARK PANELS
# ==============================================================================
def create_denoising_comparison_panel(
    output_path: str = "results/preprocessing/comparison/denoising_comparison_panel.png",
    case_key: str = "P_00004_1_LEFT_CC",
    noise_type: str = "gaussian",
) -> str:
    """Generate comprehensive 10-image denoising benchmark row:

    Clean Reference | Noisy Input | 8 Stage-9 Denoised Outputs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = REPRESENTATIVE_CASES.get(case_key, REPRESENTATIVE_CASES["P_00004_1_LEFT_CC"])
    pid = meta["patient_id"]
    side = meta["breast_side"]
    view = meta["image_view"]
    abn = meta["abnormality_id"]

    clean_path = f"data/processed/baseline/full_mammogram/{pid}_{side}_{view}_{abn}_baseline.png"
    noisy_path = f"data/processed/noise/{noise_type}/{pid}_{abn}_{side}_{view}_{noise_type}.png"

    # Layout: 2 rows of 5 columns for clean visual readability
    # Row 1: Clean, Noisy, Median, Gaussian, Wiener
    # Row 2: Bilateral, Non-Local Means, Anscombe-Wiener, Adaptive Median, Kuan
    panel_items = [
        ("Clean Reference (Stage 5)", clean_path),
        (f"Noisy Input ({noise_type})", noisy_path),
        ("Median Filter", f"data/processed/denoising/{noise_type}/median/{pid}_{abn}_{side}_{view}_{noise_type}_median.png"),
        ("Gaussian Filter", f"data/processed/denoising/{noise_type}/gaussian/{pid}_{abn}_{side}_{view}_{noise_type}_gaussian.png"),
        ("Wiener Filter", f"data/processed/denoising/{noise_type}/wiener/{pid}_{abn}_{side}_{view}_{noise_type}_wiener.png"),
        ("Bilateral Filter", f"data/processed/denoising/{noise_type}/bilateral/{pid}_{abn}_{side}_{view}_{noise_type}_bilateral.png"),
        ("Non-Local Means", f"data/processed/denoising/{noise_type}/non_local_means/{pid}_{abn}_{side}_{view}_{noise_type}_non_local_means.png"),
        ("Anscombe-Wiener", f"data/processed/denoising/{noise_type}/anscombe_wiener/{pid}_{abn}_{side}_{view}_{noise_type}_anscombe_wiener.png"),
        ("Adaptive Median", f"data/processed/denoising/{noise_type}/adaptive_median/{pid}_{abn}_{side}_{view}_{noise_type}_adaptive_median.png"),
        ("Kuan Filter", f"data/processed/denoising/{noise_type}/kuan/{pid}_{abn}_{side}_{view}_{noise_type}_kuan.png"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 9.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    axes_flat = axes.flatten()

    for idx, (title, p_img) in enumerate(panel_items):
        ax = axes_flat[idx]
        img = load_grayscale_safe(p_img)
        if img is not None:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            mean_val = np.mean(img)
            std_val = np.std(img)
            ax.set_xlabel(f"Mean={mean_val:.1f}, Std={std_val:.1f}", fontsize=8.5)
        else:
            ax.text(0.5, 0.5, "Image Missing", ha="center", va="center", color="red")
            ax.set_facecolor("#f0f0f0")

        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"CBIS-DDSM Denoising Benchmark — Noise: {noise_type.upper()} | Case: {pid}_{side}_{view}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated denoising comparison panel at {output_path}")
    return output_path


# ==============================================================================
# 5. DIFFERENCE MAPS & MAGNIFIED DETAIL CROPS
# ==============================================================================
def create_denoising_difference_maps_panel(
    output_path: str = "results/preprocessing/comparison/denoising_difference_maps_panel.png",
    case_key: str = "P_00004_1_LEFT_CC",
    noise_type: str = "salt_pepper",
) -> str:
    """Generate difference maps across 3 scientific comparison perspectives:

    A. Clean Reference vs Noisy Input (|Noisy - Clean|)
    B. Clean Reference vs Denoised Output (|Denoised - Clean|, Restoration Error)
    C. Noisy Input vs Denoised Output (|Denoised - Noisy|, Filtering Modification)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = REPRESENTATIVE_CASES.get(case_key, REPRESENTATIVE_CASES["P_00004_1_LEFT_CC"])
    pid = meta["patient_id"]
    side = meta["breast_side"]
    view = meta["image_view"]
    abn = meta["abnormality_id"]

    clean_img = load_grayscale_safe(f"data/processed/baseline/full_mammogram/{pid}_{side}_{view}_{abn}_baseline.png")
    noisy_img = load_grayscale_safe(f"data/processed/noise/{noise_type}/{pid}_{abn}_{side}_{view}_{noise_type}.png")

    if clean_img is None or noisy_img is None:
        logger.error("Clean or noisy image missing for difference maps.")
        return ""

    # Select representative denoisers illustrating distinct filter behaviors
    demo_methods = ["median", "gaussian", "bilateral", "adaptive_median"]
    fig, axes = plt.subplots(len(demo_methods) + 1, 3, figsize=(15, 3.8 * (len(demo_methods) + 1)))
    plt.subplots_adjust(wspace=0.12, hspace=0.3)

    # Row 0: Noise Perturbation (Clean vs Noisy)
    diff_noise = np.abs(noisy_img.astype(np.float32) - clean_img.astype(np.float32))
    axes[0, 0].imshow(clean_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Clean Reference (I)", fontsize=10, fontweight="bold")
    axes[0, 1].imshow(noisy_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Noisy Input ({noise_type})", fontsize=10, fontweight="bold")
    im0 = axes[0, 2].imshow(diff_noise, cmap="inferno", vmin=0, vmax=120)
    axes[0, 2].set_title("Comparison A: |Noisy - Clean|\n(Degradation Magnitude)", fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Intensity Error")

    for c in range(3):
        axes[0, c].set_xticks([])
        axes[0, c].set_yticks([])
    axes[0, 0].set_ylabel("[Noise Baseline]", fontsize=10, fontweight="bold", rotation=0, ha="right", va="center", labelpad=15)

    # Rows 1..4: Denoising Methods
    for r_idx, m_name in enumerate(demo_methods, start=1):
        p_denoised = f"data/processed/denoising/{noise_type}/{m_name}/{pid}_{abn}_{side}_{view}_{noise_type}_{m_name}.png"
        denoised_img = load_grayscale_safe(p_denoised)

        ax_den = axes[r_idx, 0]
        ax_comp_b = axes[r_idx, 1]
        ax_comp_c = axes[r_idx, 2]

        if denoised_img is not None:
            ax_den.imshow(denoised_img, cmap="gray", vmin=0, vmax=255)
            ax_den.set_xlabel(f"Mean={np.mean(denoised_img):.1f}", fontsize=8.5)

            # Comparison B: |Denoised - Clean| (Restoration Error)
            diff_b = np.abs(denoised_img.astype(np.float32) - clean_img.astype(np.float32))
            im_b = ax_comp_b.imshow(diff_b, cmap="inferno", vmin=0, vmax=60)
            ax_comp_b.set_xlabel(f"Mean Err={np.mean(diff_b):.2f}, Max={np.max(diff_b):.0f}", fontsize=8.5)
            plt.colorbar(im_b, ax=ax_comp_b, fraction=0.046, pad=0.04)

            # Comparison C: |Denoised - Noisy| (Filtering Modification)
            diff_c = np.abs(denoised_img.astype(np.float32) - noisy_img.astype(np.float32))
            im_c = ax_comp_c.imshow(diff_c, cmap="viridis", vmin=0, vmax=120)
            ax_comp_c.set_xlabel(f"Modif Mean={np.mean(diff_c):.2f}, Max={np.max(diff_c):.0f}", fontsize=8.5)
            plt.colorbar(im_c, ax=ax_comp_c, fraction=0.046, pad=0.04)
        else:
            ax_den.text(0.5, 0.5, "Image Missing", ha="center", va="center", color="red")

        if r_idx == 1:
            ax_den.set_title("Denoised Output", fontsize=10, fontweight="bold")
            ax_comp_b.set_title("Comparison B: |Denoised - Clean|\n(True Restoration Error)", fontsize=10, fontweight="bold")
            ax_comp_c.set_title("Comparison C: |Denoised - Noisy|\n(Filtering Magnitude)", fontsize=10, fontweight="bold")

        for ax in [ax_den, ax_comp_b, ax_comp_c]:
            ax.set_xticks([])
            ax.set_yticks([])
        ax_den.set_ylabel(f"[{m_name}]", fontsize=10, fontweight="bold", rotation=0, ha="right", va="center", labelpad=15)

    fig.suptitle(
        f"CBIS-DDSM Denoising Difference Maps — Noise: {noise_type.upper()} | Case: {pid}_{side}_{view}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated difference maps panel at {output_path}")
    return output_path


def create_magnified_detail_crops(
    output_path: str = "results/preprocessing/comparison/magnified_detail_crops.png",
    case_key: str = "P_00004_1_LEFT_CC",
    noise_type: str = "salt_pepper",
) -> str:
    """Generate high-resolution magnified detail crops of breast parenchyma/lesion.

    Documents exact crop coordinates (ymin, ymax, xmin, xmax) and examines fine edge retention.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = REPRESENTATIVE_CASES.get(case_key, REPRESENTATIVE_CASES["P_00004_1_LEFT_CC"])
    pid = meta["patient_id"]
    side = meta["breast_side"]
    view = meta["image_view"]
    abn = meta["abnormality_id"]
    crop_info = meta["detail_crop"]

    ymin, ymax = crop_info["ymin"], crop_info["ymax"]
    xmin, xmax = crop_info["xmin"], crop_info["xmax"]

    clean_img = load_grayscale_safe(f"data/processed/baseline/full_mammogram/{pid}_{side}_{view}_{abn}_baseline.png")
    noisy_img = load_grayscale_safe(f"data/processed/noise/{noise_type}/{pid}_{abn}_{side}_{view}_{noise_type}.png")

    if clean_img is None or noisy_img is None:
        return ""

    crop_clean = clean_img[ymin:ymax, xmin:xmax]
    crop_noisy = noisy_img[ymin:ymax, xmin:xmax]

    # Compare 8 denoisers
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    axes_flat = axes.flatten()

    items = [
        ("Clean Crop (Ref)", crop_clean),
        (f"Noisy Crop ({noise_type})", crop_noisy),
    ]

    for m in DENOISING_METHODS:
        p_den = f"data/processed/denoising/{noise_type}/{m}/{pid}_{abn}_{side}_{view}_{noise_type}_{m}.png"
        img_den = load_grayscale_safe(p_den)
        if img_den is not None:
            items.append((f"{m}", img_den[ymin:ymax, xmin:xmax]))
        else:
            items.append((f"{m} (Missing)", np.zeros_like(crop_clean)))

    for idx, (title, crop) in enumerate(items):
        ax = axes_flat[idx]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        # Draw bounding border to emphasize magnification
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
            spine.set_linewidth(1.2)

    fig.suptitle(
        f"CBIS-DDSM Magnified Parenchymal Crops — Coordinates: Y[{ymin}:{ymax}], X[{xmin}:{xmax}]\n"
        f"Purpose: {crop_info['purpose']} | Case: {pid}_{side}_{view} | Noise: {noise_type}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated magnified detail crops at {output_path}")
    return output_path


# ==============================================================================
# 6. STAGE 10 QUANTITATIVE METRIC VISUALIZATIONS
# ==============================================================================
def load_stage10_metric_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load validated Stage 10 metrics from disk."""
    p_per_image = find_stage_file([
        "data/metadata/stage10/denoising_metrics_per_image.csv",
        "data/metadata/denoising_metrics_per_image.csv",
    ])
    p_by_method = find_stage_file([
        "data/metadata/stage10/denoising_metrics_summary_by_method.csv",
    ])
    p_by_noise = find_stage_file([
        "data/metadata/stage10/denoising_metrics_summary_by_noise.csv",
    ])

    if not p_per_image or not os.path.exists(p_per_image):
        raise FileNotFoundError("Stage 10 per-image metrics CSV not found.")

    df_per_img = pd.read_csv(p_per_image)
    df_method = pd.read_csv(p_by_method) if p_by_method and os.path.exists(p_by_method) else pd.DataFrame()
    df_noise = pd.read_csv(p_by_noise) if p_by_noise and os.path.exists(p_by_noise) else pd.DataFrame()

    return df_per_img, df_method, df_noise


def plot_metrics_by_denoising_method(
    df_per_img: pd.DataFrame,
    output_path: str = "results/preprocessing/comparison/metrics_by_denoising_method.png",
) -> str:
    """Plot PSNR, SSIM, and MSE by denoising method (Comparison B: Clean vs Denoised).

    Includes exact sample count per method and error bars across all evaluated conditions.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.25)

    methods = [m for m in DENOISING_METHODS if m in df_per_img["denoising_method"].values]
    grouped = df_per_img.groupby("denoising_method")

    means_psnr = []
    stds_psnr = []
    means_ssim = []
    stds_ssim = []
    means_mse = []
    stds_mse = []

    for m in methods:
        grp = grouped.get_group(m)
        s_p = grp["clean_denoised_psnr"].replace([np.inf, -np.inf], np.nan).dropna()
        means_psnr.append(float(s_p.mean()) if not s_p.empty else 0.0)
        stds_psnr.append(float(s_p.std()) if len(s_p) > 1 else 0.0)

        s_s = grp["clean_denoised_ssim"].dropna()
        means_ssim.append(float(s_s.mean()) if not s_s.empty else 0.0)
        stds_ssim.append(float(s_s.std()) if len(s_s) > 1 else 0.0)

        s_m = grp["clean_denoised_mse"].dropna() * 1e4
        means_mse.append(float(s_m.mean()) if not s_m.empty else 0.0)
        stds_mse.append(float(s_m.std()) if len(s_m) > 1 else 0.0)

    x = np.arange(len(methods))
    colors = [METHOD_PALETTE.get(m, "#4c72b0") for m in methods]

    # Subplot 1: PSNR
    ax0 = axes[0]
    bars0 = ax0.bar(x, means_psnr, yerr=stds_psnr, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax0.set_title("Peak Signal-to-Noise Ratio (PSNR)\n[Comparison B: Clean vs Denoised]", fontsize=11, fontweight="bold")
    ax0.set_ylabel("PSNR (dB) — Higher is Better", fontsize=10)
    ax0.set_xticks(x)
    ax0.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax0.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars0:
        yval = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{yval:.1f}", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)

    # Subplot 2: SSIM
    ax1 = axes[1]
    bars1 = ax1.bar(x, means_ssim, yerr=stds_ssim, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Structural Similarity Index (SSIM)\n[Comparison B: Clean vs Denoised]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("SSIM [0.0 - 1.0] — Higher is Better", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{yval:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)

    # Subplot 3: MSE
    ax2 = axes[2]
    bars2 = ax2.bar(x, means_mse, yerr=stds_mse, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Mean Squared Error (MSE x 1e-4)\n[Comparison B: Clean vs Denoised]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MSE (x 1e-4) — Lower is Better", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}", ha="center", va="bottom", fontsize=8)

    n_unique_cases = len(df_per_img.drop_duplicates(subset=["patient_id", "breast_side", "image_view"]))
    n_evals_per_method = len(df_per_img) // len(methods) if len(methods) > 0 else 0
    sub_desc = (
        f"N={n_evals_per_method} evaluations per method across {n_unique_cases} pilot cases"
        if n_unique_cases > 1
        else "N=5 noise conditions per method, Case P_00004_1_LEFT_CC"
    )
    fig.suptitle(
        f"Stage 10 Quantitative Metrics by Denoising Method ({sub_desc})\n"
        "[Preliminary Pilot Scope — No Overall Winning Algorithm Declared]",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated metrics by method plot at {output_path}")
    return output_path


def plot_metrics_by_noise_type(
    df_per_img: pd.DataFrame,
    output_path: str = "results/preprocessing/comparison/metrics_by_noise_type.png",
) -> str:
    """Plot quantitative restoration difficulty grouped by synthetic noise type (Comparison B)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.25)

    noise_order = [n for n in NOISE_TYPES if n in df_per_img["noise_type"].values]
    grouped = df_per_img.groupby("noise_type")

    means_psnr = []
    stds_psnr = []
    means_ssim = []
    stds_ssim = []
    means_mse = []
    stds_mse = []

    for n in noise_order:
        grp = grouped.get_group(n)
        s_p = grp["clean_denoised_psnr"].replace([np.inf, -np.inf], np.nan).dropna()
        means_psnr.append(float(s_p.mean()) if not s_p.empty else 0.0)
        stds_psnr.append(float(s_p.std()) if len(s_p) > 1 else 0.0)

        s_s = grp["clean_denoised_ssim"].dropna()
        means_ssim.append(float(s_s.mean()) if not s_s.empty else 0.0)
        stds_ssim.append(float(s_s.std()) if len(s_s) > 1 else 0.0)

        s_m = grp["clean_denoised_mse"].dropna() * 1e4
        means_mse.append(float(s_m.mean()) if not s_m.empty else 0.0)
        stds_mse.append(float(s_m.std()) if len(s_m) > 1 else 0.0)

    x = np.arange(len(noise_order))
    colors = [NOISE_PALETTE.get(n, "#4e79a7") for n in noise_order]

    # PSNR
    axes[0].bar(x, means_psnr, yerr=stds_psnr, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Restoration PSNR by Noise Model\n(Comparison B: Clean vs Denoised)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(noise_order, rotation=40, ha="right", fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # SSIM
    axes[1].bar(x, means_ssim, yerr=stds_ssim, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title("Restoration SSIM by Noise Model\n(Comparison B: Clean vs Denoised)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("SSIM [0.0 - 1.0]", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(noise_order, rotation=40, ha="right", fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # MSE
    axes[2].bar(x, means_mse, yerr=stds_mse, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[2].set_title("Restoration MSE (x 1e-4) by Noise Model\n(Comparison B: Clean vs Denoised)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("MSE (x 1e-4)", fontsize=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(noise_order, rotation=40, ha="right", fontsize=9)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    n_unique_cases = len(df_per_img.drop_duplicates(subset=["patient_id", "breast_side", "image_view"]))
    n_evals_per_noise = len(df_per_img) // len(noise_order) if len(noise_order) > 0 else 0
    noise_sub_desc = (
        f"N={n_evals_per_noise} evaluations per noise model across {n_unique_cases} pilot cases"
        if n_unique_cases > 1
        else "N=8 methods per noise type"
    )
    fig.suptitle(
        f"Denoising Restoration Metrics Grouped by Synthetic Noise Model ({noise_sub_desc})\n"
        "[Speckle and Gaussian exhibit highest restoration fidelity; Mixed P-G exhibits highest residual degradation]",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated metrics by noise type plot at {output_path}")
    return output_path


def plot_method_by_noise_heatmaps(
    df_per_img: pd.DataFrame,
    output_psnr_path: str = "results/preprocessing/comparison/heatmap_method_by_noise_psnr.png",
    output_ssim_path: str = "results/preprocessing/comparison/heatmap_method_by_noise_ssim.png",
) -> Tuple[str, str]:
    """Generate 8x5 heatmaps of mean PSNR and mean SSIM across all method x noise combinations.

    Aggregates case-level observations with mean to handle single-case (N=1) and full-pilot (N=16) corpora.
    Isolates non-finite (infinite) PSNR values from the mean to ensure numerical integrity.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_psnr_path)), exist_ok=True)

    df_work = df_per_img.copy()
    # Mask out infinite values to compute finite arithmetic mean
    df_work["clean_denoised_psnr_finite"] = df_work["clean_denoised_psnr"].replace([np.inf, -np.inf], np.nan)
    df_work["clean_denoised_ssim_finite"] = df_work["clean_denoised_ssim"].replace([np.inf, -np.inf], np.nan)

    # Pivot table with mean aggregation across eligible cases
    pivot_psnr = df_work.pivot_table(
        index="denoising_method",
        columns="noise_type",
        values="clean_denoised_psnr_finite",
        aggfunc="mean",
    ).reindex(index=DENOISING_METHODS, columns=NOISE_TYPES)

    pivot_ssim = df_work.pivot_table(
        index="denoising_method",
        columns="noise_type",
        values="clean_denoised_ssim_finite",
        aggfunc="mean",
    ).reindex(index=DENOISING_METHODS, columns=NOISE_TYPES)

    # Sample counts per cell
    pivot_counts = df_work.pivot_table(
        index="denoising_method",
        columns="noise_type",
        values="clean_denoised_psnr",
        aggfunc="count",
    ).reindex(index=DENOISING_METHODS, columns=NOISE_TYPES).fillna(0).astype(int)

    n_cases = len(df_work.drop_duplicates(subset=["patient_id", "breast_side", "image_view"]))
    case_desc = f"Mean across N={n_cases} Pilot Cases" if n_cases > 1 else "N=1 per cell, Case P_00004_1_LEFT_CC"

    # 1. PSNR Heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 7.5))
    im1 = ax1.imshow(pivot_psnr.values, cmap="YlGnBu", aspect="auto")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("PSNR (dB) — Clean vs Denoised", fontsize=10)

    ax1.set_xticks(np.arange(len(NOISE_TYPES)))
    ax1.set_yticks(np.arange(len(DENOISING_METHODS)))
    ax1.set_xticklabels(NOISE_TYPES, rotation=35, ha="right", fontsize=9.5)
    ax1.set_yticklabels(DENOISING_METHODS, fontsize=9.5)

    # Annotate each cell with exact mean PSNR and sample count
    for i in range(len(DENOISING_METHODS)):
        for j in range(len(NOISE_TYPES)):
            val = pivot_psnr.values[i, j]
            cnt = pivot_counts.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 35.0 else "black"
                ax1.text(
                    j, i,
                    f"{val:.1f}\n(N={cnt})",
                    ha="center", va="center",
                    color=text_color, fontsize=8.5, fontweight="bold",
                )

    ax1.set_title(
        f"Stage 10 Factorial Evaluation: Mean PSNR (dB) by Method x Noise Type\n"
        f"[Clean Reference vs Denoised Output — {case_desc}]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.savefig(output_psnr_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 2. SSIM Heatmap
    fig2, ax2 = plt.subplots(figsize=(10, 7.5))
    im2 = ax2.imshow(pivot_ssim.values, cmap="magma", aspect="auto", vmin=0.3, vmax=1.0)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("SSIM (Structural Similarity) — Clean vs Denoised", fontsize=10)

    ax2.set_xticks(np.arange(len(NOISE_TYPES)))
    ax2.set_yticks(np.arange(len(DENOISING_METHODS)))
    ax2.set_xticklabels(NOISE_TYPES, rotation=35, ha="right", fontsize=9.5)
    ax2.set_yticklabels(DENOISING_METHODS, fontsize=9.5)

    for i in range(len(DENOISING_METHODS)):
        for j in range(len(NOISE_TYPES)):
            val = pivot_ssim.values[i, j]
            cnt = pivot_counts.values[i, j]
            if not np.isnan(val):
                text_color = "black" if val > 0.85 else "white"
                ax2.text(
                    j, i,
                    f"{val:.2f}\n(N={cnt})",
                    ha="center", va="center",
                    color=text_color, fontsize=8.5, fontweight="bold",
                )

    ax2.set_title(
        f"Stage 10 Factorial Evaluation: Mean SSIM by Method x Noise Type\n"
        f"[Clean Reference vs Denoised Output — {case_desc}]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.savefig(output_ssim_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    logger.info(f"Generated heatmaps: {output_psnr_path} and {output_ssim_path}")
    return output_psnr_path, output_ssim_path


def plot_whole_vs_foreground_metrics(
    df_per_img: pd.DataFrame,
    output_path: str = "results/preprocessing/comparison/metrics_whole_vs_foreground.png",
) -> str:
    """Compare Whole-Image vs Breast Foreground-Only metrics (PSNR and SSIM).

    Verifies mask source: clean_baseline_intensity_threshold(0.02).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plt.subplots_adjust(wspace=0.25)

    methods = [m for m in DENOISING_METHODS if m in df_per_img["denoising_method"].values]
    grouped = df_per_img.groupby("denoising_method")

    psnr_whole = [
        float(grouped.get_group(m)["clean_denoised_psnr"].replace([np.inf, -np.inf], np.nan).mean())
        for m in methods
    ]
    psnr_fg = [
        float(grouped.get_group(m)["clean_denoised_psnr_foreground"].replace([np.inf, -np.inf], np.nan).mean())
        for m in methods
    ]

    ssim_whole = [float(grouped.get_group(m)["clean_denoised_ssim"].dropna().mean()) for m in methods]
    ssim_fg = [float(grouped.get_group(m)["clean_denoised_ssim_foreground"].dropna().mean()) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    # PSNR Whole vs Foreground
    ax0 = axes[0]
    ax0.bar(x - width / 2, psnr_whole, width, label="Whole Image", color="#4e79a7", edgecolor="black")
    ax0.bar(x + width / 2, psnr_fg, width, label="Breast Foreground Only", color="#e15759", edgecolor="black")
    ax0.set_title("PSNR: Whole-Image vs Foreground-Only\n(Clean vs Denoised)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("PSNR (dB)", fontsize=10)
    ax0.set_xticks(x)
    ax0.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(axis="y", linestyle="--", alpha=0.4)

    # SSIM Whole vs Foreground
    ax1 = axes[1]
    ax1.bar(x - width / 2, ssim_whole, width, label="Whole Image", color="#4e79a7", edgecolor="black")
    ax1.bar(x + width / 2, ssim_fg, width, label="Breast Foreground Only", color="#e15759", edgecolor="black")
    ax1.set_title("SSIM: Whole-Image vs Foreground-Only\n(Clean vs Denoised)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("SSIM [0.0 - 1.0]", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Whole-Image vs Breast Foreground-Only Metric Comparison\n"
        "[Mask Source: clean_baseline_intensity_threshold(0.02) — Exterior Black Air Regions Excluded]",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated whole vs foreground comparison plot at {output_path}")
    return output_path


def plot_comparison_c_filtering_magnitude(
    df_per_img: pd.DataFrame,
    output_path: str = "results/preprocessing/comparison/comparison_c_filtering_magnitude.png",
) -> str:
    """Plot Comparison C (Noisy vs Denoised) to illustrate filtering change magnitude.

    Maintains a strict, unambiguous distinction: Comparison C quantifies how much the filter
    altered the noisy input, NOT whether the denoised image matches the ground-truth clean image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    plt.subplots_adjust(wspace=0.25)

    methods = [m for m in DENOISING_METHODS if m in df_per_img["denoising_method"].values]
    grouped = df_per_img.groupby("denoising_method")

    nd_psnr = [
        float(grouped.get_group(m)["noisy_denoised_psnr"].replace([np.inf, -np.inf], np.nan).mean())
        for m in methods
    ]
    nd_ssim = [float(grouped.get_group(m)["noisy_denoised_ssim"].dropna().mean()) for m in methods]
    x = np.arange(len(methods))
    colors = [METHOD_PALETTE.get(m, "#4c72b0") for m in methods]

    # Subplot 1: Noisy vs Denoised PSNR
    axes[0].bar(x, nd_psnr, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Comparison C: Noisy vs Denoised PSNR\n(Higher = Milder Smoothing / Closer to Noisy)", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("ND-PSNR (dB)", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Subplot 2: Noisy vs Denoised SSIM
    axes[1].bar(x, nd_ssim, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title("Comparison C: Noisy vs Denoised SSIM\n(Higher = More Structural Similarity to Noisy)", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("ND-SSIM [0.0 - 1.0]", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Comparison C: Filtering Modification Magnitude (Noisy Input vs Denoised Output)\n"
        "[Note: Comparison C measures filter modification strength, NOT ground-truth restoration accuracy]",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated Comparison C plot at {output_path}")
    return output_path


# ==============================================================================
# 7. MASTER STAGE 11 PIPELINE
# ==============================================================================
def run_stage11_visualization_pipeline(
    output_dir: str = "results/preprocessing/comparison",
    metadata_dir: str = "data/metadata/stage11",
) -> Dict[str, Any]:
    """Execute complete Stage 11 Preprocessing Visualization & Comparison suite."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    generated_figures: Dict[str, str] = {}

    logger.info("Starting Stage 11 Visualizations & Quantitative Comparisons...")

    # Step 1: Save Selection Configuration JSON
    config_data = {
        "stage": "Stage 11 — Preprocessing Visualization & Comparison",
        "pilot_scope": {
            "pilot_mammograms_total": 16,
            "representative_cases_selected": list(REPRESENTATIVE_CASES.keys()),
            "selection_rationale": (
                "Deterministic coverage of mass vs calcification, benign vs malignant, "
                "CC vs MLO views, and direct linkage to Stage 10 evaluated pilot metrics."
            ),
        },
        "display_settings": {
            "image_colormap": "gray",
            "image_vmin": 0,
            "image_vmax": 255,
            "difference_map_colormap": "inferno",
            "difference_map_vmin": 0,
            "difference_map_vmax": 120,
            "native_dimensions_preserved": True,
            "no_independent_normalization": True,
        },
        "detail_crop_coordinates": {
            k: v["detail_crop"] for k, v in REPRESENTATIVE_CASES.items()
        },
        "metric_framework": {
            "comparison_a": "Clean Reference (Stage 5) vs Noisy Input (Stage 8) [Degradation Severity]",
            "comparison_b": "Clean Reference (Stage 5) vs Denoised Output (Stage 9) [Restoration Accuracy]",
            "comparison_c": "Noisy Input (Stage 8) vs Denoised Output (Stage 9) [Filtering Modification]",
            "cnr_handling": "0 eligible cases; unaligned masks excluded per AAPM/ICRU; NaN recorded",
            "foreground_mask": "clean_baseline_intensity_threshold(0.02)",
        },
        "output_paths": {
            "figures_directory": output_dir,
            "metadata_directory": metadata_dir,
        },
    }
    config_path = os.path.join(metadata_dir, "stage11_selection_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved Stage 11 configuration JSON to {config_path}")

    # Step 2: Generate Visual Comparison Panels
    generated_figures["contrast_panel"] = create_contrast_comparison_panel(
        os.path.join(output_dir, "contrast_comparison_panel.png")
    )
    generated_figures["sharpening_panel"] = create_sharpening_comparison_panel(
        os.path.join(output_dir, "sharpening_comparison_panel.png")
    )
    generated_figures["noise_panel"] = create_clean_vs_noisy_panel(
        os.path.join(output_dir, "clean_vs_noisy_panel.png")
    )
    generated_figures["denoising_panel"] = create_denoising_comparison_panel(
        os.path.join(output_dir, "denoising_comparison_panel.png"),
        noise_type="gaussian",
    )
    generated_figures["difference_maps"] = create_denoising_difference_maps_panel(
        os.path.join(output_dir, "denoising_difference_maps_panel.png"),
        noise_type="salt_pepper",
    )
    generated_figures["detail_crops"] = create_magnified_detail_crops(
        os.path.join(output_dir, "magnified_detail_crops.png"),
        noise_type="salt_pepper",
    )

    # Step 3: Generate Stage 10 Quantitative Visualizations
    try:
        df_per_img, _, _ = load_stage10_metric_data()
        generated_figures["metrics_by_method"] = plot_metrics_by_denoising_method(
            df_per_img,
            os.path.join(output_dir, "metrics_by_denoising_method.png"),
        )
        generated_figures["metrics_by_noise"] = plot_metrics_by_noise_type(
            df_per_img,
            os.path.join(output_dir, "metrics_by_noise_type.png"),
        )
        p_heat_psnr, p_heat_ssim = plot_method_by_noise_heatmaps(
            df_per_img,
            os.path.join(output_dir, "heatmap_method_by_noise_psnr.png"),
            os.path.join(output_dir, "heatmap_method_by_noise_ssim.png"),
        )
        generated_figures["heatmap_psnr"] = p_heat_psnr
        generated_figures["heatmap_ssim"] = p_heat_ssim
        generated_figures["whole_vs_foreground"] = plot_whole_vs_foreground_metrics(
            df_per_img,
            os.path.join(output_dir, "metrics_whole_vs_foreground.png"),
        )
        generated_figures["comparison_c"] = plot_comparison_c_filtering_magnitude(
            df_per_img,
            os.path.join(output_dir, "comparison_c_filtering_magnitude.png"),
        )
    except Exception as e:
        logger.error(f"Failed to generate quantitative metric plots: {e}")
        raise RuntimeError(f"Stage 11 quantitative visualization failed: {e}") from e

    expected_figures = [
        "contrast_panel",
        "sharpening_panel",
        "noise_panel",
        "denoising_panel",
        "difference_maps",
        "detail_crops",
        "metrics_by_method",
        "metrics_by_noise",
        "heatmap_psnr",
        "heatmap_ssim",
        "whole_vs_foreground",
        "comparison_c",
    ]
    missing_figs = [
        f for f in expected_figures
        if f not in generated_figures or not os.path.exists(generated_figures[f])
    ]
    if missing_figs:
        raise RuntimeError(f"Stage 11 pipeline incomplete. Missing figures: {missing_figs}")

    logger.info(f"Stage 11 pipeline completed successfully. Generated all {len(generated_figures)} expected figures.")
    return {
        "config_path": config_path,
        "figures": generated_figures,
    }


if __name__ == "__main__":
    res = run_stage11_visualization_pipeline()
    print("Stage 11 Visualizations Generated:")
    for k, v in res["figures"].items():
        print(f"  - {k}: {v}")
