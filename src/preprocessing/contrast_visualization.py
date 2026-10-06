"""Stage 6: CBIS-DDSM Contrast Enhancement Visualization Module.

Generates scientific comparison and diagnostic figures for contrast-enhanced mammograms:
1. results/preprocessing/contrast/original_vs_methods.png
   Side-by-side contact sheet of all 16 representative pilot cases:
   Stage-5 Baseline vs Global Histogram Equalization vs CLAHE.
   Also saves high-resolution copy: original_vs_methods_highres.png.

2. results/preprocessing/contrast/contrast_histogram_comparison.png
   Side-by-side mammogram images and intensity histograms across Baseline, HistEq, and CLAHE.

3. results/preprocessing/contrast/difference_maps.png
   Diagnostic visualization showing absolute difference:
   abs(enhanced_image - baseline_image) using inferno colormap to analyze
   background noise amplification, saturation, and local tissue enhancement.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend safe for Docker & servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.contrast_enhancement import (
    METHOD_BASELINE,
    METHOD_HIST_EQ,
    METHOD_CLAHE,
    resolve_image_path,
    find_metadata_path,
)


def load_contrast_metadata(metadata_path: Optional[str] = None) -> pd.DataFrame:
    """Load contrast enhancement metadata CSV."""
    if metadata_path is None or not os.path.exists(metadata_path):
        metadata_path = find_metadata_path("contrast_preprocessing_metadata.csv", "stage6")
    if not os.path.exists(metadata_path):
        # Fallback to data/metadata/ directly
        alt = os.path.join(_PROJECT_ROOT, "data", "metadata", "contrast_preprocessing_metadata.csv")
        if os.path.exists(alt):
            metadata_path = alt
        else:
            raise FileNotFoundError(f"Contrast metadata not found at: {metadata_path}")

    return pd.read_csv(metadata_path)


def group_metadata_by_case(df_meta: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group rows by case (patient_id, abnormality_id, side, view)."""
    cases_dict = {}
    for _, r in df_meta.iterrows():
        pid = str(r.get("patient_id", "")).strip()
        abn = int(r.get("abnormality_id", 1))
        side = str(r.get("side", "")).strip().upper()
        view = str(r.get("view", "")).strip().upper()
        key = (pid, abn, side, view)
        if key not in cases_dict:
            cases_dict[key] = {
                "patient_id": pid,
                "abnormality_id": abn,
                "side": side,
                "view": view,
                "category": str(r.get("category", "")),
                "pathology": str(r.get("pathology", "")),
                "split": str(r.get("split", "")),
                "methods": {},
            }
        method = str(r.get("method", "")).strip()
        cases_dict[key]["methods"][method] = r.to_dict()

    return list(cases_dict.values())


def generate_original_vs_methods_plot(
    cases: List[Dict[str, Any]],
    output_dir: str = "results/preprocessing/contrast",
) -> Tuple[str, str]:
    """Generate side-by-side contact sheet of all 16 cases across Baseline, HistEq, and CLAHE."""
    os.makedirs(output_dir, exist_ok=True)
    num_cases = len(cases)
    if num_cases == 0:
        return "", ""

    # Figure height proportional to number of cases (e.g. 3.5 inches per case)
    fig, axes = plt.subplots(num_cases, 3, figsize=(15, 4.2 * num_cases))
    if num_cases == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, c in enumerate(cases):
        pid = c["patient_id"]
        side = c["side"]
        view = c["view"]
        cat = c["category"].upper()
        path = c["pathology"].upper()
        methods = c["methods"]

        base_rec = methods.get(METHOD_BASELINE, {})
        he_rec = methods.get(METHOD_HIST_EQ, {})
        clahe_rec = methods.get(METHOD_CLAHE, {})

        base_path = resolve_image_path(base_rec.get("output_image_path", ""))
        he_path = resolve_image_path(he_rec.get("output_image_path", ""))
        clahe_path = resolve_image_path(clahe_rec.get("output_image_path", ""))

        img_base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(base_path) else None
        img_he = cv2.imread(he_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(he_path) else None
        img_clahe = cv2.imread(clahe_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(clahe_path) else None

        # Column 1: Baseline
        ax0 = axes[i, 0]
        if img_base is not None:
            ax0.imshow(img_base, cmap="gray", vmin=0, vmax=255)
            h, w = img_base.shape[:2]
            ent = base_rec.get("entropy", 0.0)
            ax0.set_title(f"Baseline (Control)\n{pid} {side} {view} [{cat} | {path}]\nDim: {w}x{h} | Entropy: {ent:.2f}", fontsize=9)
        else:
            ax0.text(0.5, 0.5, "Image Missing", ha="center", va="center")
        ax0.axis("off")

        # Column 2: Histogram Equalization
        ax1 = axes[i, 1]
        if img_he is not None:
            ax1.imshow(img_he, cmap="gray", vmin=0, vmax=255)
            ent = he_rec.get("entropy", 0.0)
            ax1.set_title(f"Global Histogram Equalization\nMean: {he_rec.get('output_mean', 0.0):.1f} | Entropy: {ent:.2f}\n[Notice Background Amplification]", fontsize=9, color="#9c1010")
        else:
            ax1.text(0.5, 0.5, "Image Missing", ha="center", va="center")
        ax1.axis("off")

        # Column 3: CLAHE
        ax2 = axes[i, 2]
        if img_clahe is not None:
            ax2.imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
            ent = clahe_rec.get("entropy", 0.0)
            ax2.set_title(f"CLAHE (clip=2.0, grid=8x8)\nMean: {clahe_rec.get('output_mean', 0.0):.1f} | Entropy: {ent:.2f}\n[Adaptive Local Contrast]", fontsize=9, color="#10509c")
        else:
            ax2.text(0.5, 0.5, "Image Missing", ha="center", va="center")
        ax2.axis("off")

    plt.suptitle("Stage 6: Controlled Contrast Enhancement Evaluation (16 Pilot Cases)", fontsize=14, y=0.998, fontweight="bold")
    plt.tight_layout()

    out_std = os.path.join(output_dir, "original_vs_methods.png")
    out_highres = os.path.join(output_dir, "original_vs_methods_highres.png")
    plt.savefig(out_std, dpi=120, bbox_inches="tight")
    plt.savefig(out_highres, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"[Stage 6 Viz] Saved contact sheets to:\n  - {out_std}\n  - {out_highres}")
    return out_std, out_highres


def generate_contrast_histogram_comparison(
    cases: List[Dict[str, Any]],
    output_dir: str = "results/preprocessing/contrast",
    n_sample_cases: int = 4,
) -> str:
    """Generate comparative image and intensity histogram distributions across methods."""
    os.makedirs(output_dir, exist_ok=True)
    if not cases:
        return ""

    # Choose representative diverse cases (e.g. 4 cases)
    sample_cases = cases[:min(len(cases), n_sample_cases)]
    n_rows = len(sample_cases)

    # 4 rows x 6 columns: [Base Img, Base Hist, HistEq Img, HistEq Hist, CLAHE Img, CLAHE Hist]
    fig, axes = plt.subplots(n_rows, 6, figsize=(22, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, c in enumerate(sample_cases):
        pid = c["patient_id"]
        side = c["side"]
        view = c["view"]
        cat = c["category"].upper()
        path = c["pathology"].upper()
        methods = c["methods"]

        base_rec = methods.get(METHOD_BASELINE, {})
        he_rec = methods.get(METHOD_HIST_EQ, {})
        clahe_rec = methods.get(METHOD_CLAHE, {})

        base_path = resolve_image_path(base_rec.get("output_image_path", ""))
        he_path = resolve_image_path(he_rec.get("output_image_path", ""))
        clahe_path = resolve_image_path(clahe_rec.get("output_image_path", ""))

        img_base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(base_path) else None
        img_he = cv2.imread(he_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(he_path) else None
        img_clahe = cv2.imread(clahe_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(clahe_path) else None

        # 1. Baseline Image & Hist
        axes[i, 0].imshow(img_base, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"Baseline\n{pid} {side} {view}\n[{cat} | {path}]", fontsize=8)
        axes[i, 0].axis("off")

        if img_base is not None:
            axes[i, 1].hist(img_base.ravel(), bins=64, range=(0, 255), color="#333333", alpha=0.8, density=True)
            axes[i, 1].set_title(f"Baseline Histogram\nEntropy: {base_rec.get('entropy', 0.0):.2f}", fontsize=8)
            axes[i, 1].set_xlim(0, 255)
            axes[i, 1].tick_params(labelsize=7)

        # 2. HistEq Image & Hist
        axes[i, 2].imshow(img_he, cmap="gray", vmin=0, vmax=255)
        axes[i, 2].set_title("Global HistEq Image", fontsize=8, color="#9c1010")
        axes[i, 2].axis("off")

        if img_he is not None:
            axes[i, 3].hist(img_he.ravel(), bins=64, range=(0, 255), color="#b32400", alpha=0.8, density=True)
            axes[i, 3].set_title(f"HistEq Histogram (Flat)\nEntropy: {he_rec.get('entropy', 0.0):.2f}", fontsize=8, color="#9c1010")
            axes[i, 3].set_xlim(0, 255)
            axes[i, 3].tick_params(labelsize=7)

        # 3. CLAHE Image & Hist
        axes[i, 4].imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
        axes[i, 4].set_title("CLAHE Image", fontsize=8, color="#10509c")
        axes[i, 4].axis("off")

        if img_clahe is not None:
            axes[i, 5].hist(img_clahe.ravel(), bins=64, range=(0, 255), color="#1a53ff", alpha=0.8, density=True)
            axes[i, 5].set_title(f"CLAHE Histogram\nEntropy: {clahe_rec.get('entropy', 0.0):.2f}", fontsize=8, color="#10509c")
            axes[i, 5].set_xlim(0, 255)
            axes[i, 5].tick_params(labelsize=7)

    plt.suptitle("Stage 6: Mammogram Intensity Distribution and Histogram Analysis", fontsize=13, y=0.995, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "contrast_histogram_comparison.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[Stage 6 Viz] Saved histogram comparison to: {out_path}")
    return out_path


def generate_difference_maps_plot(
    cases: List[Dict[str, Any]],
    output_dir: str = "results/preprocessing/contrast",
    n_sample_cases: int = 4,
) -> str:
    """Generate diagnostic absolute difference maps showing how methods alter the baseline."""
    os.makedirs(output_dir, exist_ok=True)
    if not cases:
        return ""

    sample_cases = cases[:min(len(cases), n_sample_cases)]
    n_rows = len(sample_cases)

    # 4 rows x 3 columns: [Baseline, |HistEq - Baseline|, |CLAHE - Baseline|]
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, c in enumerate(sample_cases):
        pid = c["patient_id"]
        side = c["side"]
        view = c["view"]
        cat = c["category"].upper()
        path = c["pathology"].upper()
        methods = c["methods"]

        base_rec = methods.get(METHOD_BASELINE, {})
        he_rec = methods.get(METHOD_HIST_EQ, {})
        clahe_rec = methods.get(METHOD_CLAHE, {})

        base_path = resolve_image_path(base_rec.get("output_image_path", ""))
        he_path = resolve_image_path(he_rec.get("output_image_path", ""))
        clahe_path = resolve_image_path(clahe_rec.get("output_image_path", ""))

        img_base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(base_path) else None
        img_he = cv2.imread(he_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(he_path) else None
        img_clahe = cv2.imread(clahe_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(clahe_path) else None

        if img_base is None or img_he is None or img_clahe is None:
            continue

        # Column 0: Baseline
        axes[i, 0].imshow(img_base, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(f"Baseline\n{pid} {side} {view} [{cat} | {path}]", fontsize=9)
        axes[i, 0].axis("off")

        # Column 1: Difference |HistEq - Baseline|
        diff_he = np.abs(img_he.astype(np.float32) - img_base.astype(np.float32))
        im1 = axes[i, 1].imshow(diff_he, cmap="inferno", vmin=0, vmax=255)
        mean_d_he = float(np.mean(diff_he))
        axes[i, 1].set_title(f"|HistEq - Baseline|\nMean Abs Diff: {mean_d_he:.1f} DN\n(Notice Background Noise Amplification)", fontsize=9, color="#9c1010")
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Column 2: Difference |CLAHE - Baseline|
        diff_clahe = np.abs(img_clahe.astype(np.float32) - img_base.astype(np.float32))
        im2 = axes[i, 2].imshow(diff_clahe, cmap="inferno", vmin=0, vmax=255)
        mean_d_cl = float(np.mean(diff_clahe))
        axes[i, 2].set_title(f"|CLAHE - Baseline|\nMean Abs Diff: {mean_d_cl:.1f} DN\n(Selective Glandular Local Changes)", fontsize=9, color="#10509c")
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.suptitle("Stage 6: Diagnostic Absolute Difference Maps (|Enhanced - Baseline|)", fontsize=13, y=0.998, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "difference_maps.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[Stage 6 Viz] Saved difference maps to: {out_path}")
    return out_path


def run_contrast_visualization(
    metadata_path: Optional[str] = None,
    output_dir: str = "results/preprocessing/contrast",
) -> Dict[str, str]:
    """Execute complete Stage 6 Contrast Visualization suite."""
    print("============================================================")
    print("STAGE 6 — CONTRAST ENHANCEMENT VISUALIZATION")
    print("============================================================")
    os.makedirs(output_dir, exist_ok=True)

    df_meta = load_contrast_metadata(metadata_path)
    print(f"[Stage 6 Viz] Loaded {len(df_meta)} metadata records.")

    cases = group_metadata_by_case(df_meta)
    print(f"[Stage 6 Viz] Grouped {len(cases)} cases for comparative visualization.")

    p1, p1_hr = generate_original_vs_methods_plot(cases, output_dir=output_dir)
    p2 = generate_contrast_histogram_comparison(cases, output_dir=output_dir, n_sample_cases=4)
    p3 = generate_difference_maps_plot(cases, output_dir=output_dir, n_sample_cases=4)

    print("============================================================")
    print("STAGE 6 VISUALIZATION COMPLETE")
    print(f"Figures saved in: {output_dir}/")
    print("============================================================")

    return {
        "original_vs_methods": p1,
        "original_vs_methods_highres": p1_hr,
        "histogram_comparison": p2,
        "difference_maps": p3,
    }


if __name__ == "__main__":
    run_contrast_visualization()
