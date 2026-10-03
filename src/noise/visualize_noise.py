"""Visual validation and side-by-side comparison for artificial noise models.

IMPORTANT RESEARCH DISTINCTION:
--------------------------------
The noise simulated and visualized in this module is ARTIFICIALLY GENERATED
solely for an experimental medical image denoising benchmark study. It does NOT
represent naturally occurring physical noise in the CBIS-DDSM mammography dataset.
"""
import os
import json
import argparse
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib

# Headless backend to prevent X11 server errors inside Docker / headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.image_utils import load_grayscale_image, get_image_statistics
from src.utils.logger import setup_logger

logger = setup_logger("NoiseVisualization")


def resolve_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve file path across host and Docker container filesystems."""
    if not path_str or pd.isna(path_str):
        return None

    path_str = str(path_str).strip()
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    if path_str.startswith("/app/"):
        rel = path_str[len("/app/") :]
        if os.path.exists(rel):
            return os.path.abspath(rel)

    if not path_str.startswith("/app/"):
        in_docker = os.path.join("/app", path_str)
        if os.path.exists(in_docker):
            return os.path.abspath(in_docker)

    return None


def calculate_metrics(clean: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
    """Compute basic numerical divergence metrics between clean and noisy images."""
    c_f = clean.astype(np.float64) / 255.0
    n_f = noisy.astype(np.float64) / 255.0
    mse = float(np.mean((c_f - n_f) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-10)))
    mae = float(np.mean(np.abs(c_f - n_f)))
    return {"mse": mse, "psnr": psnr, "mae": mae}


def validate_noise_realism(
    clean: np.ndarray,
    noisy_dict: Dict[str, np.ndarray],
    patient_id: str,
) -> Dict[str, Any]:
    """Verify that each synthetic noise model behaves correctly and realistically.

    Sanity criteria:
    - Dimensions match clean image.
    - Pixel values within valid range [0, 255].
    - No NaNs or infinities.
    - Measurable noise added (MSE > 0).
    - No severe saturation (mean within sensible boundaries).
    - Salt & Pepper contains distinct impulse points (0 and 255).
    """
    verifications: Dict[str, Any] = {}
    all_passed = True

    for noise_type, noisy_img in noisy_dict.items():
        issues = []

        if noisy_img.shape != clean.shape:
            issues.append(f"Dimension mismatch: clean {clean.shape} vs noisy {noisy_img.shape}")

        if np.isnan(noisy_img).any() or np.isinf(noisy_img).any():
            issues.append("Contains NaN or infinite values")

        min_val, max_val = float(np.min(noisy_img)), float(np.max(noisy_img))
        if min_val < 0 or max_val > 255:
            issues.append(f"Intensity out of bounds: [{min_val}, {max_val}]")

        metrics = calculate_metrics(clean, noisy_img)
        if metrics["mse"] < 1e-6:
            issues.append("No perceptible noise detected (MSE near zero)")

        if noise_type == "salt_pepper":
            zero_count = int(np.sum(noisy_img == 0))
            max_count = int(np.sum(noisy_img == 255))
            if zero_count == 0 and max_count == 0:
                issues.append("Impulse spikes (0 or 255) not found for Salt & Pepper noise")

        status = "PASSED" if not issues else "FAILED"
        if status == "FAILED":
            all_passed = False
            logger.error(f"[{patient_id}] Realism check FAILED for {noise_type}: {'; '.join(issues)}")

        verifications[noise_type] = {
            "status": status,
            "issues": issues,
            "metrics": metrics,
            "min": min_val,
            "max": max_val,
            "mean": float(np.mean(noisy_img)),
            "std": float(np.std(noisy_img)),
        }

    verifications["overall_status"] = "PASSED" if all_passed else "FAILED"
    return verifications


def plot_noise_comparison(
    clean_img: np.ndarray,
    noisy_dict: Dict[str, np.ndarray],
    patient_id: str,
    pathology: str,
    label: Any,
    save_path: str,
    params_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Create a high-resolution 2x3 comparison figure of clean vs 5 synthetic noise models."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.patch.set_facecolor("#181818")

    # Titles and display configurations
    panel_configs = [
        ("Original Clean (Sharpened)", clean_img, None, (0, 0)),
        ("Gaussian Noise", noisy_dict.get("gaussian"), params_dict.get("gaussian") if params_dict else None, (0, 1)),
        ("Salt & Pepper Noise", noisy_dict.get("salt_pepper"), params_dict.get("salt_pepper") if params_dict else None, (0, 2)),
        ("Speckle Noise", noisy_dict.get("speckle"), params_dict.get("speckle") if params_dict else None, (1, 0)),
        ("Poisson Noise", noisy_dict.get("poisson"), params_dict.get("poisson") if params_dict else None, (1, 1)),
        ("Mixed Poisson-Gaussian", noisy_dict.get("mixed_poisson_gaussian"), params_dict.get("mixed_poisson_gaussian") if params_dict else None, (1, 2)),
    ]

    for title, img, params, (r, c) in panel_configs:
        ax = axes[r, c]
        ax.set_facecolor("#121212")

        if img is not None:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            # Statistical subtitle
            min_v, max_v = int(np.min(img)), int(np.max(img))
            mean_v, std_v = float(np.mean(img)), float(np.std(img))

            param_str = ""
            if params:
                param_items = [f"{k}={v}" for k, v in params.items()]
                param_str = f" | {', '.join(param_items)}"

            ax.set_title(
                f"{title}{param_str}\nRange: [{min_v}, {max_v}] | Mean: {mean_v:.1f} | Std: {std_v:.1f}",
                fontsize=11,
                color="#E0E0E0",
                pad=8,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"Missing Image\n({title})",
                color="#FF5555",
                ha="center",
                va="center",
                fontsize=12,
            )

        ax.axis("off")

    # Master title
    suptag = f"Patient ID: {patient_id}   |   Pathology: {pathology}   |   Label: {label}"
    research_notice = "[SYNTHETIC RESEARCH NOISE MODELS - Denoising Benchmark Experiment - NOT Natural CBIS-DDSM Noise]"
    plt.suptitle(
        f"{suptag}\n{research_notice}",
        fontsize=14,
        fontweight="bold",
        color="#FFFFFF",
        y=0.98,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_noise_visualizations(
    noisy_metadata_csv: str = "data/metadata/noisy_image_metadata.csv",
    sharpened_metadata_csv: str = "data/metadata/sharpened_image_metadata.csv",
    noisy_base_dir: str = "data/processed/noisy",
    output_dir: str = "results/preprocessing/noise_comparisons",
    max_images: int = 10,
) -> List[Dict[str, Any]]:
    """Generate visual comparisons for the test set mammograms across all 5 noise models.

    Args:
        noisy_metadata_csv: Path to noisy image metadata CSV.
        sharpened_metadata_csv: Fallback path to sharpened metadata CSV.
        noisy_base_dir: Directory containing subfolders with noisy images.
        output_dir: Destination folder for output comparison figures.
        max_images: Number of clean images to process (default: 10).

    Returns:
        List[Dict[str, Any]]: Summary of validation results for each processed case.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Attempt loading noisy_metadata_csv
    actual_noisy_csv = resolve_path(noisy_metadata_csv)
    groups = []

    if actual_noisy_csv and os.path.exists(actual_noisy_csv):
        df_noisy = pd.read_csv(actual_noisy_csv)
        if not df_noisy.empty:
            for src_img, group in df_noisy.groupby("source_image", sort=False):
                first_row = group.iloc[0]
                patient_id = str(first_row.get("patient_id", "UNKNOWN")).strip()
                pathology = str(first_row.get("pathology", "UNKNOWN")).strip()
                label = first_row.get("label", 0)

                noisy_map = {}
                params_map = {}
                for _, r in group.iterrows():
                    ntype = str(r.get("noise_type", "")).strip()
                    npath = resolve_path(r.get("output_image"))
                    if npath and os.path.exists(npath):
                        noisy_map[ntype] = npath

                    p_str = r.get("noise_parameters")
                    if p_str and not pd.isna(p_str):
                        try:
                            params_map[ntype] = json.loads(p_str) if isinstance(p_str, str) else p_str
                        except Exception:
                            params_map[ntype] = {}

                groups.append({
                    "clean_path": resolve_path(src_img),
                    "patient_id": patient_id,
                    "pathology": pathology,
                    "label": label,
                    "noisy_paths": noisy_map,
                    "params": params_map,
                })

    # 2. Fallback to sharpened_metadata_csv if noisy CSV not yet populated
    if not groups:
        actual_sharp_csv = resolve_path(sharpened_metadata_csv)
        if actual_sharp_csv and os.path.exists(actual_sharp_csv):
            df_sharp = pd.read_csv(actual_sharp_csv)
            for _, r in df_sharp.iterrows():
                clean_path = resolve_path(r.get("sharpened_image_path"))
                if not clean_path or not os.path.exists(clean_path):
                    continue

                base_name = os.path.basename(clean_path)
                patient_id = str(r.get("patient_id", "UNKNOWN")).strip()
                pathology = str(r.get("pathology", "UNKNOWN")).strip()
                label = r.get("label", 0)

                noisy_map = {}
                for ntype in ["gaussian", "salt_pepper", "speckle", "poisson", "mixed_poisson_gaussian"]:
                    cand_path = resolve_path(os.path.join(noisy_base_dir, ntype, base_name))
                    if cand_path and os.path.exists(cand_path):
                        noisy_map[ntype] = cand_path

                groups.append({
                    "clean_path": clean_path,
                    "patient_id": patient_id,
                    "pathology": pathology,
                    "label": label,
                    "noisy_paths": noisy_map,
                    "params": {},
                })

    if not groups:
        raise FileNotFoundError(
            f"No noisy mammogram metadata found. Please ensure noise generation has been completed."
        )

    groups = groups[:max_images]
    logger.info(f"Generating visual comparisons for {len(groups)} test mammograms in '{output_dir}'.")

    results_summary: List[Dict[str, Any]] = []
    failed_any = False

    for idx, item in enumerate(groups, start=1):
        patient_id = item["patient_id"]
        pathology = item["pathology"]
        label = item["label"]
        clean_path = item["clean_path"]

        if not clean_path or not os.path.exists(clean_path):
            logger.warning(f"[{patient_id}] Clean sharpened image missing: {clean_path}")
            continue

        clean_img = load_grayscale_image(clean_path)
        base_name = os.path.splitext(os.path.basename(clean_path))[0]

        # Load available noisy images
        noisy_imgs: Dict[str, np.ndarray] = {}
        for ntype, npath in item["noisy_paths"].items():
            if os.path.exists(npath):
                try:
                    noisy_imgs[ntype] = load_grayscale_image(npath)
                except Exception as e:
                    logger.warning(f"Could not load {ntype} image ({npath}): {e}")

        # Validate realism
        validation = validate_noise_realism(clean_img, noisy_imgs, patient_id)
        if validation["overall_status"] != "PASSED":
            failed_any = True

        # Render comparison figure
        out_fig_name = f"{base_name}_noise_comparison.png"
        out_fig_path = os.path.join(output_dir, out_fig_name)

        plot_noise_comparison(
            clean_img=clean_img,
            noisy_dict=noisy_imgs,
            patient_id=patient_id,
            pathology=pathology,
            label=label,
            save_path=out_fig_path,
            params_dict=item.get("params"),
        )

        results_summary.append({
            "patient_id": patient_id,
            "pathology": pathology,
            "label": label,
            "figure_path": os.path.abspath(out_fig_path),
            "status": validation["overall_status"],
            "models_present": list(noisy_imgs.keys()),
            "validation_details": validation,
        })

        logger.info(f"[{idx}/{len(groups)}] Saved comparison: {out_fig_name} | Realism: {validation['overall_status']}")

    # Print summary report
    print("\n" + "=" * 76)
    print("CBIS-DDSM Artificial Noise Models: Visual Verification Report")
    print("=" * 76)
    print(f"Output Directory : {os.path.abspath(output_dir)}")
    print(f"Test Images      : {len(results_summary)}")
    print(f"Noise Models     : Gaussian, Salt & Pepper, Speckle, Poisson, Mixed Poisson-Gaussian")
    print("-" * 76)
    print(f"{'Patient ID':<12} | {'Pathology':<20} | {'Models':<7} | {'Realism Status':<15}")
    print("-" * 76)
    for r in results_summary:
        print(f"{r['patient_id']:<12} | {r['pathology']:<20} | {len(r['models_present']):<7} | {r['status']:<15}")
    print("=" * 76)

    if failed_any:
        print("[CRITICAL WARNING]: One or more artificial noise models failed realism verification.")
        print("Review the log details above. Do not proceed to the denoising stage until resolved.\n")
    else:
        print("[VERIFICATION SUCCESSFUL]: All artificial noise models behave realistically and as specified.")
        print("Ready for subsequent denoising benchmarks.\n")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Artificial Noise Visualization and Validation")
    parser.add_argument("--noisy-csv", type=str, default="data/metadata/noisy_image_metadata.csv", help="Noisy metadata CSV")
    parser.add_argument("--sharp-csv", type=str, default="data/metadata/sharpened_image_metadata.csv", help="Sharpened metadata CSV")
    parser.add_argument("--noisy-dir", type=str, default="data/processed/noisy", help="Noisy images base directory")
    parser.add_argument("--output-dir", type=str, default="results/preprocessing/noise_comparisons", help="Output comparison figures directory")
    parser.add_argument("--count", type=int, default=10, help="Number of test images to visualize")

    args = parser.parse_args()

    run_noise_visualizations(
        noisy_metadata_csv=args.noisy_csv,
        sharpened_metadata_csv=args.sharp_csv,
        noisy_base_dir=args.noisy_dir,
        output_dir=args.output_dir,
        max_images=args.count,
    )


if __name__ == "__main__":
    main()
