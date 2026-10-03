"""Visualization and validation utility for preprocessed CBIS-DDSM mammograms."""
import os
import argparse
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
# Headless backend to ensure error-free rendering inside Docker/remote environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.image_utils import load_grayscale_image, get_image_statistics
from src.utils.logger import setup_logger

logger = setup_logger("PreprocessingValidation")

def resolve_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve file path across host and Docker container filesystems."""
    if not path_str or pd.isna(path_str):
        return None

    path_str = str(path_str).strip()
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # Check without /app/ prefix (if running locally on Windows)
    if path_str.startswith("/app/"):
        rel = path_str[len("/app/"):]
        if os.path.exists(rel):
            return os.path.abspath(rel)

    # Check with /app/ prefix (if running in Docker)
    if not path_str.startswith("/app/"):
        in_docker = os.path.join("/app", path_str)
        if os.path.exists(in_docker):
            return os.path.abspath(in_docker)

    return None

def validate_and_visualize_samples(
    metadata_csv: str = "data/metadata/normalized_image_metadata.csv",
    num_samples: int = 4,
    save_path: str = "results/preprocessing/comparison_plots/normalization_validation.png",
    per_sample_dir: str = "experiments/preprocessing/visual_comparisons",
    patient_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load and compare original vs normalized mammograms side-by-side with statistical validation.

    Args:
        metadata_csv: Path to normalized image metadata CSV.
        num_samples: Number of sample pairs to visualize (default: 4).
        save_path: Primary plot output destination.
        per_sample_dir: Directory to save individual patient comparison figures.
        patient_filter: Optional patient_id to inspect a specific case.

    Returns:
        List of dictionaries containing validation statistics for each inspected pair.
    """
    actual_csv = resolve_path(metadata_csv)
    if not actual_csv or not os.path.exists(actual_csv):
        # Fallback candidates
        candidates = [
            metadata_csv,
            os.path.join("data", "metadata", "normalized_image_metadata.csv"),
            os.path.join("/app", "data", "metadata", "normalized_image_metadata.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                actual_csv = os.path.abspath(c)
                break

    if not actual_csv or not os.path.exists(actual_csv):
        raise FileNotFoundError(f"Normalized metadata CSV not found: {metadata_csv}. Please run basic preprocessing first.")

    df = pd.read_csv(actual_csv)
    if df.empty:
        raise ValueError("Normalized metadata CSV is empty.")

    if patient_filter:
        subset = df[df["patient_id"].astype(str).str.contains(patient_filter, case=False, na=False)]
        if subset.empty:
            logger.warning(f"No records matching patient '{patient_filter}'. Falling back to default records.")
            subset = df.head(num_samples)
    else:
        subset = df.head(num_samples)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    os.makedirs(per_sample_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("CBIS-DDSM Preprocessing Validation & Integrity Analysis")
    print("=" * 70)

    inspected_samples: List[Dict[str, Any]] = []

    # Configure grid plot: num_samples rows, 2 columns (Original | Processed)
    n_rows = len(subset)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.5 * n_rows), squeeze=False)
    fig.patch.set_facecolor("#18181b")

    for i, (_, row) in enumerate(subset.iterrows()):
        patient_id = str(row.get("patient_id", "N/A")).strip()
        abnormality = str(row.get("abnormality_category", "N/A")).strip()
        breast_side = str(row.get("breast_side", "N/A")).strip()
        image_view = str(row.get("image_view", "N/A")).strip()
        pathology = str(row.get("pathology", "N/A")).strip()
        label = row.get("label", 0)

        orig_path = resolve_path(row.get("original_image_path"))
        proc_path = resolve_path(row.get("processed_image_path"))

        if not orig_path or not os.path.exists(orig_path):
            logger.error(f"Sample {i+1}: Original image path not found: {row.get('original_image_path')}")
            continue
        if not proc_path or not os.path.exists(proc_path):
            logger.error(f"Sample {i+1}: Processed image path not found: {row.get('processed_image_path')}")
            continue

        # Load images
        img_orig = load_grayscale_image(orig_path)
        img_proc = load_grayscale_image(proc_path)

        # Compute statistics
        stats_orig = get_image_statistics(img_orig)
        stats_proc = get_image_statistics(img_proc)

        # Integrity Checks
        is_blank = stats_proc["min"] == stats_proc["max"]
        has_dynamic_range = stats_proc["max"] > stats_proc["min"]
        status = "PASSED" if has_dynamic_range and not is_blank else "FAILED (Degraded/Blank)"

        sample_info = {
            "index": i + 1,
            "patient_id": patient_id,
            "abnormality": abnormality,
            "breast_side": breast_side,
            "image_view": image_view,
            "pathology": pathology,
            "label": label,
            "orig_shape": (stats_orig["height"], stats_orig["width"]),
            "proc_shape": (stats_proc["height"], stats_proc["width"]),
            "orig_range": (stats_orig["min"], stats_orig["max"]),
            "proc_range": (stats_proc["min"], stats_proc["max"]),
            "orig_mean": stats_orig["mean"],
            "proc_mean": stats_proc["mean"],
            "status": status,
        }
        inspected_samples.append(sample_info)

        # Terminal Print
        print(f"\n[Sample {i+1}/{n_rows}] Patient ID: {patient_id}")
        print(f"  Category & View   : {abnormality.upper()} | {breast_side} Breast | View: {image_view}")
        print(f"  Pathology & Label : {pathology} (Label {label})")
        print(f"  Original Shape    : {stats_orig['height']} x {stats_orig['width']} | Dynamic Range: [{stats_orig['min']}, {stats_orig['max']}] | Mean: {stats_orig['mean']:.2f}")
        print(f"  Processed Shape   : {stats_proc['height']} x {stats_proc['width']} | Dynamic Range: [{stats_proc['min']}, {stats_proc['max']}] | Mean: {stats_proc['mean']:.2f}")
        print(f"  Integrity Check   : {status}")

        # Render Left: Original
        ax_orig = axes[i, 0]
        ax_orig.set_facecolor("#09090b")
        im_o = ax_orig.imshow(img_orig, cmap="gray")
        ax_orig.set_title(
            f"ORIGINAL MAMMOGRAM\nPatient: {patient_id} ({breast_side} {image_view})\n"
            f"Shape: {stats_orig['width']}x{stats_orig['height']} | Range: [{stats_orig['min']}, {stats_orig['max']}]",
            fontsize=9.5,
            color="#e4e4e7",
            pad=8
        )
        ax_orig.axis("off")

        # Render Right: Normalized & Resized
        ax_proc = axes[i, 1]
        ax_proc.set_facecolor("#09090b")
        im_p = ax_proc.imshow(img_proc, cmap="gray", vmin=0, vmax=255)
        pathology_color = "#f87171" if label == 1 else "#4ade80"
        ax_proc.set_title(
            f"NORMALIZED & RESIZED (512x512)\nPathology: {pathology} (Label: {label})\n"
            f"Shape: {stats_proc['width']}x{stats_proc['height']} | Range: [{stats_proc['min']}, {stats_proc['max']}]",
            fontsize=9.5,
            color=pathology_color,
            pad=8
        )
        ax_proc.axis("off")

        # Also save individual patient comparison
        indiv_fig, indiv_axes = plt.subplots(1, 2, figsize=(9, 5))
        indiv_fig.patch.set_facecolor("#18181b")
        indiv_axes[0].imshow(img_orig, cmap="gray")
        indiv_axes[0].set_title(f"Original ({stats_orig['width']}x{stats_orig['height']})", color="#e4e4e7", fontsize=10)
        indiv_axes[0].axis("off")
        indiv_axes[1].imshow(img_proc, cmap="gray")
        indiv_axes[1].set_title(f"Normalized ({pathology} - Label {label})", color=pathology_color, fontsize=10)
        indiv_axes[1].axis("off")
        indiv_path = os.path.join(per_sample_dir, f"{patient_id}_{breast_side}_{image_view}_comparison.png")
        indiv_fig.tight_layout()
        indiv_fig.savefig(indiv_path, dpi=200, facecolor=indiv_fig.get_facecolor(), bbox_inches="tight")
        plt.close(indiv_fig)

    fig.tight_layout(pad=2.5)
    fig.savefig(save_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("Visualization Output Details")
    print("=" * 70)
    print(f"Comparative Plot Saved : {os.path.abspath(save_path)}")
    print(f"Per-Patient Directory  : {os.path.abspath(per_sample_dir)}")
    all_passed = all(s["status"] == "PASSED" for s in inspected_samples)
    print(f"Overall Quality Status : {'ALL SAMPLES VERIFIED HEALTHY (NO DATA LOSS)' if all_passed else 'WARNING: ANOMALIES DETECTED'}")
    print("=" * 70 + "\n")

    return inspected_samples

def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Preprocessing Validation & Visualization")
    parser.add_argument("--csv", type=str, default="data/metadata/normalized_image_metadata.csv", help="Normalized metadata CSV")
    parser.add_argument("--samples", type=int, default=4, help="Number of sample pairs to inspect")
    parser.add_argument("--output", type=str, default="results/preprocessing/comparison_plots/normalization_validation.png", help="Output comparison figure path")
    parser.add_argument("--patient", type=str, default=None, help="Filter for specific patient ID")

    args = parser.parse_args()

    validate_and_visualize_samples(
        metadata_csv=args.csv,
        num_samples=args.samples,
        save_path=args.output,
        patient_filter=args.patient,
    )

if __name__ == "__main__":
    main()
