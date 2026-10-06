"""Stage 5: Baseline Preprocessing Visual Validation Module.

Generates side-by-side comparative contact sheets (Original Full Mammogram vs Baseline Processed)
for 12 representative pilot cases covering CC/MLO, LEFT/RIGHT, MASS/CALCIFICATION, and available
pathologies without altering research image files.
"""

from collections import Counter
import os
import sys
from typing import Any, Dict, List, Optional

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

from src.data.image_inventory import find_jpeg_dir


def find_metadata_dir(stage_subdir: str = "stage5") -> str:
    """Locate metadata directory across Docker and local workspaces."""
    candidates = [
        f"data/metadata/{stage_subdir}",
        os.path.join(os.getcwd(), "data", "metadata", stage_subdir),
        f"/app/data/metadata/{stage_subdir}",
        "data/metadata",
        os.path.join(os.getcwd(), "data", "metadata"),
        "/app/data/metadata",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(f"data/metadata/{stage_subdir}")


def resolve_image_path(image_path: str, jpeg_dir: str) -> str:
    """Resolve physical path on disk across environments."""
    if not image_path:
        return ""
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return os.path.normpath(image_path)
    cand1 = os.path.normpath(os.path.join(jpeg_dir, image_path))
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.normpath(image_path)
    if os.path.exists(cand2):
        return cand2
    return cand1


def select_visualization_samples(
    df_meta: pd.DataFrame,
    n_samples: int = 12,
) -> List[Dict[str, Any]]:
    """Deterministically select balanced pilot subset for visual inspection."""
    if df_meta.empty:
        return []

    # Sort deterministically
    sorted_df = df_meta.sort_values(
        by=[
            "abnormality_category",
            "image_view",
            "breast_side",
            "pathology",
            "patient_id",
            "abnormality_id",
        ]
    ).copy()

    # Stratified target strata (cat, view, side, pathology)
    target_strata = [
        ("mass", "CC", "LEFT", "BENIGN"),
        ("mass", "MLO", "RIGHT", "MALIGNANT"),
        ("mass", "CC", "RIGHT", "MALIGNANT"),
        ("mass", "MLO", "LEFT", "BENIGN"),
        ("calcification", "CC", "LEFT", "BENIGN"),
        ("calcification", "MLO", "RIGHT", "BENIGN_WITHOUT_CALLBACK"),
        ("calcification", "MLO", "LEFT", "MALIGNANT"),
        ("calcification", "CC", "RIGHT", "MALIGNANT"),
        ("mass", "CC", "LEFT", "BENIGN_WITHOUT_CALLBACK"),
        ("calcification", "CC", "RIGHT", "BENIGN"),
        ("mass", "MLO", "RIGHT", "BENIGN_WITHOUT_CALLBACK"),
        ("calcification", "MLO", "LEFT", "MALIGNANT"),
    ]

    selected = []
    used_indices = set()

    for cat, view, side, path_val in target_strata:
        if len(selected) >= n_samples:
            break
        subset = sorted_df[
            (sorted_df["abnormality_category"].str.lower() == cat.lower())
            & (sorted_df["image_view"].str.upper() == view.upper())
            & (sorted_df["breast_side"].str.upper() == side.upper())
            & (sorted_df["pathology"].str.upper() == path_val.upper())
            & (~sorted_df.index.isin(used_indices))
        ]
        if not subset.empty:
            chosen = subset.iloc[0]
            used_indices.add(chosen.name)
            selected.append(chosen)

    # Fallback to remaining
    if len(selected) < n_samples:
        for _, r in sorted_df.iterrows():
            if r.name not in used_indices:
                selected.append(r)
                used_indices.add(r.name)
                if len(selected) >= n_samples:
                    break

    return [s.to_dict() for s in selected]


def generate_baseline_comparison_plot(
    samples: List[Dict[str, Any]],
    jpeg_dir: str,
    output_path: str,
    max_dim: int = 800,
) -> None:
    """Generate side-by-side comparative contact sheet: Original vs Baseline Processed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_items = len(samples)
    if n_items == 0:
        print("[Baseline Visualization] No sample records provided for plotting.")
        return

    # 2 columns per case: (LEFT: Original, RIGHT: Baseline)
    # We display each case as a 2-column pair in a row
    n_rows = n_items
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, n_rows * 5.2), facecolor="#1a1a1a")
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, s in enumerate(samples):
        ax_orig = axes[row_idx, 0]
        ax_base = axes[row_idx, 1]

        # 1. Load and scale Original Image for display canvas
        orig_p = resolve_image_path(s["original_image_path"], jpeg_dir)
        orig_disp = None
        if os.path.exists(orig_p):
            try:
                raw_img = cv2.imread(orig_p, cv2.IMREAD_UNCHANGED)
                if raw_img is not None:
                    h, w = raw_img.shape[:2]
                    scale = min(1.0, max_dim / max(h, w))
                    orig_disp = cv2.resize(raw_img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

        # 2. Load and scale Baseline Processed Image for display canvas
        base_p = s.get("baseline_image_path", "")
        if not os.path.isabs(base_p):
            base_p = os.path.join(os.getcwd(), base_p)

        base_disp = None
        if os.path.exists(base_p):
            try:
                base_img = cv2.imread(base_p, cv2.IMREAD_UNCHANGED)
                if base_img is not None:
                    h, w = base_img.shape[:2]
                    scale = min(1.0, max_dim / max(h, w))
                    base_disp = cv2.resize(base_img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

        # Plot Left: Original
        if orig_disp is not None:
            ax_orig.imshow(orig_disp, cmap="gray")
        else:
            ax_orig.text(0.5, 0.5, "ORIGINAL UNAVAILABLE", color="#ff5252", ha="center", va="center", fontsize=10)
        ax_orig.set_facecolor("#121212")
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        orig_title = (
            f"ORIGINAL FULL MAMMOGRAM\n"
            f"{s['patient_id']} | {s['abnormality_category'].upper()} | {s['breast_side']}_{s['image_view']}\n"
            f"Pathology: {s['pathology']} | Dims: {s['original_width']}x{s['original_height']}"
        )
        ax_orig.set_title(orig_title, color="#ffffff", fontsize=8.5, pad=6)
        for spine in ax_orig.spines.values():
            spine.set_color("#757575")
            spine.set_linewidth(1.5)

        # Plot Right: Baseline
        if base_disp is not None:
            ax_base.imshow(base_disp, cmap="gray")
        else:
            ax_base.text(0.5, 0.5, "BASELINE UNAVAILABLE", color="#ff5252", ha="center", va="center", fontsize=10)
        ax_base.set_facecolor("#121212")
        ax_base.set_xticks([])
        ax_base.set_yticks([])

        crop_info = f"Cropped to {s['baseline_width']}x{s['baseline_height']}" if s.get("crop_applied") else "Full image uncropped"
        status_color = "#4CAF50" if s["processing_status"] == "SUCCESS" else "#FFC107"

        base_title = (
            f"STAGE 5 BASELINE PROCESSED\n"
            f"{s['patient_id']} | Status: {s['processing_status']}\n"
            f"{crop_info} | Norm: [0, 1] uint8\n"
            f"Active Area: {s.get('breast_region_area_ratio', 0.0)*100:.1f}%"
        )
        ax_base.set_title(base_title, color="#ffffff", fontsize=8.5, pad=6)
        for spine in ax_base.spines.values():
            spine.set_color(status_color)
            spine.set_linewidth(2.0)

    fig.suptitle(
        "CBIS-DDSM BASELINE PREPROCESSING VALIDATION: ORIGINAL VS BASELINE PROCESSED",
        color="#ffffff", fontsize=13, fontweight="bold", y=0.998,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[Baseline Visualization] Successfully generated comparison contact sheet at: {output_path}")


def run_baseline_visualization(
    stage5_metadata_dir: str = "data/metadata/stage5",
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    results_dir: str = "results/preprocessing/baseline",
    n_samples: int = 12,
) -> None:
    """Execute complete visual comparison workflow for Stage 5."""
    os.makedirs(results_dir, exist_ok=True)
    meta_path = os.path.join(stage5_metadata_dir, "baseline_preprocessing_metadata.csv")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(find_metadata_dir("stage5"), "baseline_preprocessing_metadata.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Stage 5 metadata file not found at {meta_path}. Run baseline_preprocessing first.")

    df_meta = pd.read_csv(meta_path)
    jpeg_dir = find_jpeg_dir(raw_data_dir)

    print("============================================================")
    print("STAGE 5 — BASELINE PREPROCESSING VISUAL VALIDATION")
    print("============================================================")
    print(f"[Stage 5 Viz] Loaded {len(df_meta)} processed metadata records.")
    print(f"[Stage 5 Viz] Selecting {n_samples} representative cases across strata...")

    samples = select_visualization_samples(df_meta, n_samples=n_samples)
    out_plot_path = os.path.join(results_dir, "original_vs_baseline.png")

    generate_baseline_comparison_plot(
        samples=samples,
        jpeg_dir=jpeg_dir,
        output_path=out_plot_path,
    )
    print("============================================================")


if __name__ == "__main__":
    s5_dir = "data/metadata/stage5"
    raw_dir = "data/raw/CBIS_DDSM"
    res_dir = "results/preprocessing/baseline"

    if len(sys.argv) > 1:
        s5_dir = sys.argv[1]
    if len(sys.argv) > 2:
        raw_dir = sys.argv[2]
    if len(sys.argv) > 3:
        res_dir = sys.argv[3]

    run_baseline_visualization(
        stage5_metadata_dir=s5_dir,
        raw_data_dir=raw_dir,
        results_dir=res_dir,
    )
