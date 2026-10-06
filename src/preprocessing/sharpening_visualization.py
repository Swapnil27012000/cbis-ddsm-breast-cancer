"""Stage 7: CBIS-DDSM Sharpening Visualization Module.

Generates scientific comparison and diagnostic figures for image sharpening:
1. results/preprocessing/sharpening/sharpening_comparison.png
   Side-by-side contact sheet of all 16 representative pilot cases:
   Col 1: Stage-6 control image
   Col 2: Stage-7 sharpened image
   Col 3: Absolute difference map (|sharpened - control|)

2. results/preprocessing/sharpening/sharpening_difference_maps.png
   Diagnostic visualization comparing sharpening difference maps across
   the three contrast conditions: Baseline, HistEq, and CLAHE.

3. results/preprocessing/sharpening/edge_comparison.png
   Diagnostic 4-column figure:
   1. CONTROL
   2. SHARPENED
   3. CONTROL EDGE MAP
   4. SHARPENED EDGE MAP
   Evaluates structural edge enhancement, halos, noise amplification, and overshoot.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend safe for Docker & headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.sharpening import (
    CONTRAST_BASELINE,
    CONTRAST_HIST_EQ,
    CONTRAST_CLAHE,
    SHARPENING_NONE,
    SHARPENING_UNSHARP,
    find_metadata_path,
    resolve_image_path,
    compute_edge_statistics,
)


def load_sharpening_metadata(metadata_path: Optional[str] = None) -> pd.DataFrame:
    """Load Stage-7 sharpening metadata CSV."""
    if metadata_path is None or not os.path.exists(metadata_path):
        metadata_path = find_metadata_path("sharpening_preprocessing_metadata.csv", "stage7")
    if not os.path.exists(metadata_path):
        alt = os.path.join(_PROJECT_ROOT, "data", "metadata", "sharpening_preprocessing_metadata.csv")
        if os.path.exists(alt):
            metadata_path = alt
        else:
            raise FileNotFoundError(f"Sharpening metadata not found at: {metadata_path}")

    return pd.read_csv(metadata_path)


def group_sharpening_metadata(df_meta: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group rows by case (patient_id, abnormality_id, side, view)."""
    cases_dict: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for _, r in df_meta.iterrows():
        pid = str(r["patient_id"]).strip()
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
                "category": str(r.get("category", "")).strip(),
                "pathology": str(r.get("pathology", "")).strip(),
                "label": int(r.get("label", 0)) if pd.notna(r.get("label")) else 0,
                "split": str(r.get("split", "train")).strip(),
                "conditions": {},
            }

        cm = str(r.get("contrast_method", "")).strip()
        sm = str(r.get("sharpening_method", "")).strip()
        cases_dict[key]["conditions"][(cm, sm)] = r.to_dict()

    return list(cases_dict.values())


def generate_sharpening_comparison_plot(
    cases: List[Dict[str, Any]],
    output_dir: str = "results/preprocessing/sharpening",
    contrast_condition: str = CONTRAST_CLAHE,
) -> Tuple[str, str]:
    """Generate 3-column comparison figure for all 16 pilot cases.

    Column 1: Stage-6 control image
    Column 2: Stage-7 sharpened image
    Column 3: Absolute difference map (|sharpened - control|)
    """
    os.makedirs(output_dir, exist_ok=True)
    num_cases = len(cases)
    if num_cases == 0:
        return "", ""

    fig, axes = plt.subplots(num_cases, 3, figsize=(16, 4.4 * num_cases))
    if num_cases == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, c in enumerate(cases):
        pid = c["patient_id"]
        side = c["side"]
        view = c["view"]
        abn = c["abnormality_id"]
        cat = c["category"].upper()
        path = c["pathology"].upper()

        ctrl_rec = c["conditions"].get((contrast_condition, SHARPENING_NONE), {})
        sharp_rec = c["conditions"].get((contrast_condition, SHARPENING_UNSHARP), {})

        # If contrast condition not found, fallback to baseline
        if not ctrl_rec or not sharp_rec:
            ctrl_rec = c["conditions"].get((CONTRAST_BASELINE, SHARPENING_NONE), {})
            sharp_rec = c["conditions"].get((CONTRAST_BASELINE, SHARPENING_UNSHARP), {})

        ctrl_path = resolve_image_path(ctrl_rec.get("output_image_path", ""))
        sharp_path = resolve_image_path(sharp_rec.get("output_image_path", ""))

        img_ctrl = cv2.imread(ctrl_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ctrl_path) else None
        img_sharp = cv2.imread(sharp_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(sharp_path) else None

        # Column 1: Control Image
        ax0 = axes[i, 0]
        if img_ctrl is not None:
            ax0.imshow(img_ctrl, cmap="gray", vmin=0, vmax=255)
            h, w = img_ctrl.shape[:2]
            ent = ctrl_rec.get("entropy", 0.0)
            ax0.set_title(
                f"CONTROL ({contrast_condition.upper()})\n{pid} {side} {view} #{abn} [{cat} | {path}]\nDim: {w}x{h} | Entropy: {ent:.2f}",
                fontsize=9,
            )
        else:
            ax0.text(0.5, 0.5, "Control Missing", ha="center", va="center")
        ax0.axis("off")

        # Column 2: Sharpened Image
        ax1 = axes[i, 1]
        if img_sharp is not None:
            ax1.imshow(img_sharp, cmap="gray", vmin=0, vmax=255)
            ent_s = sharp_rec.get("entropy", 0.0)
            strong_e = sharp_rec.get("strong_edge_percentage", 0.0)
            ax1.set_title(
                f"SHARPENED (Unsharp Masking)\nSigma: 1.0, Amount: 1.0, Thresh: 0\nEntropy: {ent_s:.2f} | Strong Edges: {strong_e:.1f}%",
                fontsize=9,
                color="#0b5394",
            )
        else:
            ax1.text(0.5, 0.5, "Sharpened Missing", ha="center", va="center")
        ax1.axis("off")

        # Column 3: Absolute Difference Map
        ax2 = axes[i, 2]
        if img_ctrl is not None and img_sharp is not None:
            diff = np.abs(img_sharp.astype(np.float32) - img_ctrl.astype(np.float32))
            mean_d = float(np.mean(diff))
            max_d = float(np.max(diff))
            im_diff = ax2.imshow(diff, cmap="inferno", vmin=0, vmax=35)
            ax2.set_title(
                f"ABSOLUTE DIFFERENCE MAP\nMean Diff: {mean_d:.2f} | Max Diff: {max_d:.1f}\n[Notice Boundary Transitions]",
                fontsize=9,
                color="#a61c1c",
            )
            plt.colorbar(im_diff, ax=ax2, fraction=0.035, pad=0.04)
        else:
            ax2.text(0.5, 0.5, "Difference N/A", ha="center", va="center")
        ax2.axis("off")

    plt.suptitle(
        f"Stage 7: Controlled Image Sharpening Comparison — {contrast_condition.upper()} Condition (16 Pilot Cases)",
        fontsize=14,
        y=0.998,
        fontweight="bold",
    )
    plt.tight_layout()

    out_std = os.path.join(output_dir, "sharpening_comparison.png")
    out_highres = os.path.join(output_dir, "sharpening_comparison_highres.png")
    plt.savefig(out_std, dpi=120, bbox_inches="tight")
    plt.savefig(out_highres, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"[Stage 7 Viz] Saved comparison contact sheet to:\n  - {out_std}\n  - {out_highres}")
    return out_std, out_highres


def generate_sharpening_difference_maps_plot(
    cases: List[Dict[str, Any]],
    output_dir: str = "results/preprocessing/sharpening",
    n_display_cases: int = 8,
) -> str:
    """Generate diagnostic difference maps across Baseline, HistEq, and CLAHE.

    Evaluates whether unsharp masking amplifies noise differently depending
    on the prior contrast enhancement method.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not cases:
        return ""

    sample_cases = cases[:min(len(cases), n_display_cases)]
    n_rows = len(sample_cases)

    # 3 Columns: Baseline Diff Map, HistEq Diff Map, CLAHE Diff Map
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    contrast_list = [
        (CONTRAST_BASELINE, "Baseline + Unsharp Diff", "#274e13"),
        (CONTRAST_HIST_EQ, "HistEq + Unsharp Diff", "#783f04"),
        (CONTRAST_CLAHE, "CLAHE + Unsharp Diff", "#073763"),
    ]

    for i, c in enumerate(sample_cases):
        pid = c["patient_id"]
        side = c["side"]
        view = c["view"]
        abn = c["abnormality_id"]
        cat = c["category"].upper()

        for j, (cm, label, color) in enumerate(contrast_list):
            ax = axes[i, j]
            ctrl_rec = c["conditions"].get((cm, SHARPENING_NONE), {})
            sharp_rec = c["conditions"].get((cm, SHARPENING_UNSHARP), {})

            ctrl_path = resolve_image_path(ctrl_rec.get("output_image_path", ""))
            sharp_path = resolve_image_path(sharp_rec.get("output_image_path", ""))

            img_ctrl = cv2.imread(ctrl_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ctrl_path) else None
            img_sharp = cv2.imread(sharp_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(sharp_path) else None

            if img_ctrl is not None and img_sharp is not None:
                diff = np.abs(img_sharp.astype(np.float32) - img_ctrl.astype(np.float32))
                mean_d = float(np.mean(diff))
                max_d = float(np.max(diff))
                im_d = ax.imshow(diff, cmap="inferno", vmin=0, vmax=35)
                ax.set_title(
                    f"{label}\n{pid} {side} {view} #{abn} [{cat}]\nMean Diff: {mean_d:.2f} | Max Diff: {max_d:.1f}",
                    fontsize=9,
                    color=color,
                )
                plt.colorbar(im_d, ax=ax, fraction=0.035, pad=0.04)
            else:
                ax.text(0.5, 0.5, "Image Missing", ha="center", va="center")
            ax.axis("off")

    plt.suptitle(
        "Stage 7: Sharpening Difference Maps Across Contrast Methods (|Sharpened - Control|)",
        fontsize=14,
        y=0.998,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "sharpening_difference_maps.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[Stage 7 Viz] Saved difference maps comparison to:\n  - {out_path}")
    return out_path


def generate_edge_comparison_plot(
    cases: List[Dict[str, Any]],
    output_dir: str = "results/preprocessing/sharpening",
    contrast_condition: str = CONTRAST_CLAHE,
    n_display_cases: int = 6,
) -> str:
    """Generate diagnostic edge figure containing 4 columns:

    1. CONTROL
    2. SHARPENED
    3. CONTROL EDGE MAP
    4. SHARPENED EDGE MAP

    Inspects useful structural edges vs halos, artificial boundaries, and noise amplification.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not cases:
        return ""

    sample_cases = cases[:min(len(cases), n_display_cases)]
    n_rows = len(sample_cases)

    # 4 Columns: Control, Sharpened, Control Edge Map, Sharpened Edge Map
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, c in enumerate(sample_cases):
        pid = c["patient_id"]
        side = c["side"]
        view = c["view"]
        abn = c["abnormality_id"]
        cat = c["category"].upper()
        path = c["pathology"].upper()

        ctrl_rec = c["conditions"].get((contrast_condition, SHARPENING_NONE), {})
        sharp_rec = c["conditions"].get((contrast_condition, SHARPENING_UNSHARP), {})

        if not ctrl_rec or not sharp_rec:
            ctrl_rec = c["conditions"].get((CONTRAST_BASELINE, SHARPENING_NONE), {})
            sharp_rec = c["conditions"].get((CONTRAST_BASELINE, SHARPENING_UNSHARP), {})

        ctrl_path = resolve_image_path(ctrl_rec.get("output_image_path", ""))
        sharp_path = resolve_image_path(sharp_rec.get("output_image_path", ""))

        img_ctrl = cv2.imread(ctrl_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ctrl_path) else None
        img_sharp = cv2.imread(sharp_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(sharp_path) else None

        # 1. CONTROL
        ax0 = axes[i, 0]
        if img_ctrl is not None:
            ax0.imshow(img_ctrl, cmap="gray", vmin=0, vmax=255)
            ax0.set_title(
                f"1. CONTROL ({contrast_condition.upper()})\n{pid} {side} {view} #{abn} [{cat} | {path}]\nMean: {ctrl_rec.get('output_mean', 0.0):.1f}",
                fontsize=9,
            )
        else:
            ax0.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax0.axis("off")

        # 2. SHARPENED
        ax1 = axes[i, 1]
        if img_sharp is not None:
            ax1.imshow(img_sharp, cmap="gray", vmin=0, vmax=255)
            ax1.set_title(
                f"2. SHARPENED (Unsharp Masking)\nSigma: 1.0, Amount: 1.0\nMean: {sharp_rec.get('output_mean', 0.0):.1f}",
                fontsize=9,
                color="#0b5394",
            )
        else:
            ax1.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax1.axis("off")

        # 3. CONTROL EDGE MAP
        ax2 = axes[i, 2]
        if img_ctrl is not None:
            stats_ctrl, mag_ctrl = compute_edge_statistics(img_ctrl)
            im2 = ax2.imshow(mag_ctrl, cmap="viridis", vmin=0, vmax=60)
            ax2.set_title(
                f"3. CONTROL EDGE MAP (Sobel)\nMean Grad: {stats_ctrl['mean_gradient_magnitude']:.1f}\nStrong Edges: {stats_ctrl['strong_edge_percentage']:.1f}%",
                fontsize=9,
            )
            plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
        else:
            ax2.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax2.axis("off")

        # 4. SHARPENED EDGE MAP
        ax3 = axes[i, 3]
        if img_sharp is not None:
            stats_sharp, mag_sharp = compute_edge_statistics(img_sharp)
            im3 = ax3.imshow(mag_sharp, cmap="viridis", vmin=0, vmax=60)
            ax3.set_title(
                f"4. SHARPENED EDGE MAP (Sobel)\nMean Grad: {stats_sharp['mean_gradient_magnitude']:.1f}\nStrong Edges: {stats_sharp['strong_edge_percentage']:.1f}%",
                fontsize=9,
                color="#a61c1c",
            )
            plt.colorbar(im3, ax=ax3, fraction=0.035, pad=0.04)
        else:
            ax3.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax3.axis("off")

    plt.suptitle(
        f"Stage 7: Edge Analysis & Halo Inspection — {contrast_condition.upper()} Condition",
        fontsize=14,
        y=0.998,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "edge_comparison.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[Stage 7 Viz] Saved edge comparison figure to:\n  - {out_path}")
    return out_path


def run_sharpening_visualization(
    metadata_path: Optional[str] = None,
    output_dir: str = "results/preprocessing/sharpening",
) -> Dict[str, str]:
    """Execute complete visualization suite for Stage 7 Sharpening Experiment."""
    os.makedirs(output_dir, exist_ok=True)
    df_meta = load_sharpening_metadata(metadata_path)
    cases = group_sharpening_metadata(df_meta)

    print(f"[Stage 7 Viz] Generating visualizations for {len(cases)} pilot cases...")

    p_comp_std, p_comp_hr = generate_sharpening_comparison_plot(cases, output_dir, CONTRAST_CLAHE)
    p_diff = generate_sharpening_difference_maps_plot(cases, output_dir)
    p_edge = generate_edge_comparison_plot(cases, output_dir, CONTRAST_CLAHE)

    print("\n" + "=" * 65)
    print("Stage 7 Sharpening Visualizations Generated Successfully")
    print("=" * 65)
    print(f"1. Comparison Contact Sheet : {p_comp_std}")
    print(f"2. Difference Maps          : {p_diff}")
    print(f"3. Edge Comparison Figure   : {p_edge}")
    print("=" * 65 + "\n")

    return {
        "sharpening_comparison": p_comp_std,
        "sharpening_comparison_highres": p_comp_hr,
        "sharpening_difference_maps": p_diff,
        "edge_comparison": p_edge,
    }


def main():
    """CLI execution entrypoint for Stage 7 sharpening visualizations."""
    run_sharpening_visualization()


if __name__ == "__main__":
    main()
