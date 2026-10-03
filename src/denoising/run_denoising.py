"""Batch runner CLI for medical image denoising filters benchmark.

Executes 8 distinct spatial, frequency, and adaptive denoising algorithms
across artificially degraded mammography images for denoising benchmark evaluation.
"""
import os
import json
import argparse
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.utils.image_utils import load_grayscale_image, save_image
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.gpu.device import get_device, get_device_info

from src.denoising.median import denoise_median
from src.denoising.gaussian import denoise_gaussian
from src.denoising.wiener import denoise_wiener
from src.denoising.bilateral import denoise_bilateral
from src.denoising.non_local_means import denoise_nlm
from src.denoising.anscombe_wiener import denoise_anscombe_wiener
from src.denoising.adaptive_median import denoise_adaptive_median
from src.denoising.kuan import denoise_kuan

logger = setup_logger("DenoisingPipeline")

METHOD_DEVICE_MAP = {
    "median": "CPU based (OpenCV SIMD)",
    "gaussian": "GPU accelerated / Hybrid (PyTorch CUDA / OpenCV)",
    "wiener": "CPU based (SciPy)",
    "bilateral": "CPU based (OpenCV)",
    "non_local_means": "CPU based (OpenCV fastNlMeans)",
    "anscombe_wiener": "CPU based (Anscombe + SciPy Wiener)",
    "adaptive_median": "CPU based (Vectorized SciPy multi-window rank)",
    "kuan": "GPU accelerated / Hybrid (PyTorch CUDA / SciPy)",
}


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


def run_denoising_pipeline(
    input_noisy_csv: str = "data/metadata/noisy_image_metadata.csv",
    sharpened_csv: str = "data/metadata/sharpened_image_metadata.csv",
    output_base_dir: str = "data/processed/denoised",
    output_metadata_csv: str = "data/metadata/denoised_image_metadata.csv",
    config_path: str = "config/preprocessing_config.yaml",
    noise_type: str = "gaussian",
    max_images: int = 10,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """Execute 8 denoising filters across 10 noisy mammogram samples.

    Matrix: 1 noise type x 8 denoising methods x 10 images = 80 output images.

    Args:
        input_noisy_csv: Path to noisy image metadata CSV.
        sharpened_csv: Fallback path to clean sharpened image metadata.
        output_base_dir: Root directory for denoised images.
        output_metadata_csv: Destination CSV path for denoised metadata.
        config_path: Configuration YAML path.
        noise_type: Chosen synthetic noise model to test (default: 'gaussian').
        max_images: Number of images to process (default: 10).
        use_gpu: Whether to utilize PyTorch CUDA for GPU-supported methods.

    Returns:
        pd.DataFrame: Table describing all generated denoised images.
    """
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    denoise_cfg = cfg.get("denoising", {})

    # Compute Hardware
    device = get_device(0) if use_gpu else torch.device("cpu")
    dev_info = get_device_info(0)
    device_name = dev_info["gpu_name"] if device.type == "cuda" else "CPU"
    logger.info(f"Compute Hardware: {device} ({device_name}) | GPU Acceleration: {'ACTIVE' if device.type == 'cuda' else 'INACTIVE'}")

    # Load noisy images for the chosen noise type
    actual_noisy_csv = resolve_path(input_noisy_csv)
    test_records = []

    if actual_noisy_csv and os.path.exists(actual_noisy_csv):
        df_noisy = pd.read_csv(actual_noisy_csv)
        filtered = df_noisy[df_noisy["noise_type"] == noise_type].copy()
        if not filtered.empty:
            for _, r in filtered.head(max_images).iterrows():
                test_records.append({
                    "patient_id": str(r.get("patient_id", "UNKNOWN")).strip(),
                    "pathology": str(r.get("pathology", "UNKNOWN")).strip(),
                    "label": r.get("label", 0),
                    "noisy_image_path": resolve_path(r.get("output_image")),
                    "clean_image_path": resolve_path(r.get("source_image")),
                })

    # Fallback to direct directory lookup if metadata not yet resolved
    if not test_records:
        noise_dir = resolve_path(os.path.join("data/processed/noisy", noise_type))
        if noise_dir and os.path.exists(noise_dir):
            noisy_files = [f for f in os.listdir(noise_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            for f in sorted(noisy_files)[:max_images]:
                test_records.append({
                    "patient_id": f.split("_")[0] if "_" in f else "UNKNOWN",
                    "pathology": "UNKNOWN",
                    "label": 0,
                    "noisy_image_path": os.path.abspath(os.path.join(noise_dir, f)),
                    "clean_image_path": resolve_path(os.path.join("data/processed/sharpened", f)),
                })

    if not test_records:
        raise FileNotFoundError(
            f"No noisy images found for noise_type='{noise_type}'. Please run artificial noise generation first."
        )

    logger.info(f"Preparing to denoise {len(test_records)} images with noise_type='{noise_type}' across 8 methods.")

    # Prepare output directories
    methods = [
        "median",
        "gaussian",
        "wiener",
        "bilateral",
        "non_local_means",
        "anscombe_wiener",
        "adaptive_median",
        "kuan",
    ]
    method_dirs = {}
    for m in methods:
        mdir = os.path.join(output_base_dir, m)
        os.makedirs(mdir, exist_ok=True)
        method_dirs[m] = mdir

    # Extract method configurations
    cfg_median = denoise_cfg.get("median", {"kernel_size": 5})
    cfg_gaussian = denoise_cfg.get("gaussian", {"kernel_size": [5, 5], "sigma": 1.0})
    cfg_wiener = denoise_cfg.get("wiener", {"noise_variance": None})
    cfg_bilateral = denoise_cfg.get("bilateral", {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0})
    cfg_nlm = denoise_cfg.get("non_local_means", {"h": 0.1, "template_window_size": 7, "search_window_size": 21})
    cfg_anscombe = denoise_cfg.get("anscombe_wiener", {"sigma": 0.02, "scale": 255.0})
    cfg_adaptive = denoise_cfg.get("adaptive_median", {"max_window_size": 7})
    cfg_kuan = denoise_cfg.get("kuan", {"window_size": 7, "damping": 1.0, "noise_var": 0.04})

    denoised_records: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    pbar = tqdm(test_records, desc=f"Denoising ({noise_type})", unit="img")

    for item in pbar:
        noisy_path = item["noisy_image_path"]
        if not noisy_path or not os.path.exists(noisy_path):
            failure_count += 8
            logger.warning(f"Noisy file not found: {noisy_path}")
            continue

        base_name = os.path.basename(noisy_path)
        patient_id = item["patient_id"]
        pathology = item["pathology"]
        label = item["label"]
        clean_path = item["clean_image_path"]

        # Load 8-bit image and convert to normalized float32 [0.0, 1.0]
        raw_noisy = load_grayscale_image(noisy_path)
        img_f = raw_noisy.astype(np.float32) / 255.0 if raw_noisy.dtype != np.float32 else raw_noisy
        img_f = np.clip(img_f, 0.0, 1.0)

        # Define 8 denoising operations
        filter_suite = [
            ("median", cfg_median, lambda: denoise_median(
                img_f, kernel_size=int(cfg_median.get("kernel_size", 5))
            )),
            ("gaussian", cfg_gaussian, lambda: denoise_gaussian(
                img_f,
                kernel_size=cfg_gaussian.get("kernel_size", (5, 5)),
                sigma=float(cfg_gaussian.get("sigma", 1.0)),
                device=device,
            )),
            ("wiener", cfg_wiener, lambda: denoise_wiener(
                img_f,
                mysize=cfg_wiener.get("mysize", (5, 5)),
                noise=cfg_wiener.get("noise_variance"),
            )),
            ("bilateral", cfg_bilateral, lambda: denoise_bilateral(
                img_f,
                d=int(cfg_bilateral.get("d", 9)),
                sigma_color=float(cfg_bilateral.get("sigma_color", 75.0)),
                sigma_space=float(cfg_bilateral.get("sigma_space", 75.0)),
            )),
            ("non_local_means", cfg_nlm, lambda: denoise_nlm(
                img_f,
                h=float(cfg_nlm.get("h", 0.1)),
                template_window_size=int(cfg_nlm.get("template_window_size", 7)),
                search_window_size=int(cfg_nlm.get("search_window_size", 21)),
            )),
            ("anscombe_wiener", cfg_anscombe, lambda: denoise_anscombe_wiener(
                img_f,
                scale=float(cfg_anscombe.get("scale", 255.0)),
                sigma=float(cfg_anscombe.get("sigma", 1.0)),
            )),
            ("adaptive_median", cfg_adaptive, lambda: denoise_adaptive_median(
                img_f,
                max_window_size=int(cfg_adaptive.get("max_window_size", 7)),
            )),
            ("kuan", cfg_kuan, lambda: denoise_kuan(
                img_f,
                window_size=int(cfg_kuan.get("window_size", 7)),
                noise_var=float(cfg_kuan.get("noise_var", 0.04)),
                damping=float(cfg_kuan.get("damping", 1.0)),
                device=device,
            )),
        ]

        for method_name, params, denoise_fn in filter_suite:
            try:
                denoised_arr = denoise_fn()
                # Verify dimensions and bounds
                if denoised_arr.shape != img_f.shape:
                    raise ValueError(f"Shape changed from {img_f.shape} to {denoised_arr.shape}")

                out_path = os.path.join(method_dirs[method_name], base_name)
                saved_path = save_image(denoised_arr, out_path)

                denoised_records.append({
                    "denoising_method": method_name,
                    "hardware_architecture": METHOD_DEVICE_MAP.get(method_name, "CPU"),
                    "noise_type": noise_type,
                    "source_noisy_image": noisy_path,
                    "clean_reference_image": clean_path,
                    "output_denoised_image": saved_path,
                    "patient_id": patient_id,
                    "pathology": pathology,
                    "label": int(label) if not pd.isna(label) else 0,
                    "parameters": json.dumps(params),
                    "status": "SUCCESS",
                })
                success_count += 1
            except Exception as e:
                failure_count += 1
                logger.error(f"Error applying {method_name} on {base_name}: {e}")

    out_df = pd.DataFrame(denoised_records)
    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_csv)), exist_ok=True)
    out_df.to_csv(output_metadata_csv, index=False)

    # Print summary table
    print("\n" + "=" * 78)
    print("CBIS-DDSM Denoising Benchmark: 8 Filters Execution Summary")
    print("=" * 78)
    print(f"Target Noise Model   : {noise_type.upper()}")
    print(f"Compute Hardware     : {device_name} ({device})")
    print(f"Sampled Images       : {len(test_records)}")
    print(f"Filter Methods (8)   : {', '.join(methods)}")
    print(f"Total Denoised Saved : {success_count} / {len(test_records) * 8} (Failures: {failure_count})")
    print(f"Output Directory     : {os.path.abspath(output_base_dir)}")
    print(f"Metadata CSV         : {os.path.abspath(output_metadata_csv)}")
    print("-" * 78)
    print(f"{'Method Name':<20} | {'Hardware Architecture':<36} | {'Count':<6}")
    print("-" * 78)
    for m in methods:
        cnt = sum(1 for r in denoised_records if r["denoising_method"] == m)
        print(f"{m:<20} | {METHOD_DEVICE_MAP.get(m, 'CPU'):<36} | {cnt:<6}")
    print("=" * 78 + "\n")

    return out_df


def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Denoising Benchmark Runner")
    parser.add_argument("--noisy-csv", type=str, default="data/metadata/noisy_image_metadata.csv", help="Path to noisy metadata CSV")
    parser.add_argument("--sharp-csv", type=str, default="data/metadata/sharpened_image_metadata.csv", help="Path to clean sharpened CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed/denoised", help="Denoised images base directory")
    parser.add_argument("--output-csv", type=str, default="data/metadata/denoised_image_metadata.csv", help="Output metadata CSV")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.yaml", help="Configuration YAML path")
    parser.add_argument("--noise-type", type=str, default="gaussian", help="Noise model to denoise (default: gaussian)")
    parser.add_argument("--count", type=int, default=10, help="Number of images to process")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    args = parser.parse_args()

    run_denoising_pipeline(
        input_noisy_csv=args.noisy_csv,
        sharpened_csv=args.sharp_csv,
        output_base_dir=args.output_dir,
        output_metadata_csv=args.output_csv,
        config_path=args.config,
        noise_type=args.noise_type,
        max_images=args.count,
        use_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
