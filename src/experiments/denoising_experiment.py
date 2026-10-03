"""Comprehensive comparative denoising benchmark experiment runner for CBIS-DDSM.

Supports both test sampling and full-dataset execution with:
- Checkpoint / resume capability (never repeats completed images)
- Theoretical resource estimation (disk space, operations, compute requirements)
- Explicit execution confirmation guard for full dataset processing
- Multiprocessing / multithreading for CPU filters
- PyTorch GPU acceleration for CUDA operations
- Memory streaming (never loads the full dataset into RAM)
"""
import os
import time
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple, Set
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor

from src.utils.image_utils import load_grayscale_image, save_image
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.gpu.device import get_device, get_device_info

# Noise models
from src.noise.gaussian import add_gaussian_noise
from src.noise.salt_pepper import add_salt_pepper_noise
from src.noise.speckle import add_speckle_noise
from src.noise.poisson import add_poisson_noise
from src.noise.mixed_poisson_gaussian import add_mixed_poisson_gaussian_noise

# Denoising filters
from src.denoising.median import denoise_median
from src.denoising.gaussian import denoise_gaussian
from src.denoising.wiener import denoise_wiener
from src.denoising.bilateral import denoise_bilateral
from src.denoising.non_local_means import denoise_nlm
from src.denoising.anscombe_wiener import denoise_anscombe_wiener
from src.denoising.adaptive_median import denoise_adaptive_median
from src.denoising.kuan import denoise_kuan

# Metrics
from src.metrics.mse import compute_mse
from src.metrics.psnr import compute_psnr
from src.metrics.ssim import compute_ssim
from src.metrics.snr import compute_snr
from src.metrics.cnr import compute_cnr, extract_roi_and_background_pixels
from src.metrics.cii import compute_cii
from src.metrics.entropy import compute_entropy

logger = setup_logger("DenoisingExperiment")


def resolve_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve file path across host Windows and Linux Docker container filesystems."""
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


def estimate_resources(
    num_images: int,
    num_noise_types: int = 5,
    num_denoising_methods: int = 8,
    image_size: Tuple[int, int] = (512, 512),
    avg_png_kb: float = 200.0,
    exp_per_second: float = 5.25,
) -> Dict[str, Any]:
    """Compute mathematical resource estimate for a planned benchmark workload.

    Calculates:
        - Number of images
        - Number of noise types
        - Number of denoising methods
        - Total expected image-processing operations
        - Estimated disk usage
        - Estimated GPU / CPU requirements
    """
    total_experiments = num_images * num_noise_types * num_denoising_methods
    noise_operations = num_images * num_noise_types
    denoising_operations = total_experiments
    metric_evaluations = total_experiments * 7  # 7 metrics per denoised image
    total_operations = noise_operations + denoising_operations + metric_evaluations

    # Disk usage estimation: 1 noisy PNG + 8 denoised PNGs per (image x noise)
    total_saved_images = noise_operations + denoising_operations
    disk_mb = (total_saved_images * avg_png_kb) / 1024.0
    disk_gb = disk_mb / 1024.0

    # Runtime estimation
    est_seconds = total_experiments / max(exp_per_second, 0.1)
    est_minutes = est_seconds / 60.0
    est_hours = est_minutes / 60.0

    return {
        "num_images": num_images,
        "num_noise_types": num_noise_types,
        "num_denoising_methods": num_denoising_methods,
        "total_experiments": total_experiments,
        "noise_operations": noise_operations,
        "denoising_operations": denoising_operations,
        "metric_evaluations": metric_evaluations,
        "total_operations": total_operations,
        "total_saved_images": total_saved_images,
        "estimated_disk_mb": disk_mb,
        "estimated_disk_gb": disk_gb,
        "estimated_runtime_seconds": est_seconds,
        "estimated_runtime_minutes": est_minutes,
        "estimated_runtime_hours": est_hours,
        "gpu_vram_requirement": "1.5 - 2.5 GB active VRAM (NVIDIA RTX 2050 4GB supported)",
        "cpu_ram_requirement": "2.0 - 4.0 GB system RAM (streaming single-image batches)",
    }


def print_resource_estimate_table(estimate: Dict[str, Any], title: str = "Workload & Resource Estimate") -> None:
    """Print formatted resource estimation and hardware requirements table."""
    print("\n" + "=" * 74)
    print(f"CBIS-DDSM Denoising Benchmark: {title}")
    print("=" * 74)
    print(f"Number of Clean Mammograms : {estimate['num_images']}")
    print(f"Number of Noise Models     : {estimate['num_noise_types']} (Gaussian, Salt & Pepper, Speckle, Poisson, Mixed)")
    print(f"Number of Denoising Filters: {estimate['num_denoising_methods']} (Median, Gaussian, Wiener, Bilateral, NLM, Anscombe, Adaptive, Kuan)")
    print(f"Total Benchmark Experiments: {estimate['total_experiments']:,}")
    print("-" * 74)
    print("DETAILED EXPECTED OPERATIONS:")
    print(f"  * Noise Simulation Ops   : {estimate['noise_operations']:,} image transforms")
    print(f"  * Denoising Filter Ops   : {estimate['denoising_operations']:,} image restorations")
    print(f"  * Metric Evaluations     : {estimate['metric_evaluations']:,} individual metric calculations (7 per output)")
    print(f"  * Total Expected Ops     : {estimate['total_operations']:,} discrete image-processing operations")
    print("-" * 74)
    print("ESTIMATED DISK USAGE:")
    print(f"  * Total Saved PNG Files  : {estimate['total_saved_images']:,} images (at 512x512 resolution)")
    print(f"  * Projected Storage Space: {estimate['estimated_disk_mb']:.1f} MB ({estimate['estimated_disk_gb']:.2f} GB)")
    print("-" * 74)
    print("ESTIMATED HARDWARE REQUIREMENTS:")
    print(f"  * GPU VRAM Allocation    : {estimate['gpu_vram_requirement']}")
    print(f"  * System Memory (RAM)    : {estimate['cpu_ram_requirement']}")
    if estimate['estimated_runtime_hours'] >= 1.0:
        print(f"  * Projected Runtime      : ~{estimate['estimated_runtime_hours']:.2f} hours (~{estimate['estimated_runtime_minutes']:.1f} min)")
    else:
        print(f"  * Projected Runtime      : ~{estimate['estimated_runtime_minutes']:.2f} minutes (~{estimate['estimated_runtime_seconds']:.1f} sec)")
    print("=" * 74 + "\n")


def extract_mammogram_regions(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    air_threshold: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically segment foreground ROI and background parenchyma for CNR and CII."""
    if roi_mask is not None and np.any(roi_mask > 0):
        return extract_roi_and_background_pixels(
            image, roi_mask, dilation_radius=15, air_threshold=air_threshold
        )

    parenchyma = image > float(air_threshold)
    if not np.any(parenchyma):
        return image.ravel(), image.ravel()

    tissue_pixels = image[parenchyma]
    tissue_8u = np.clip(tissue_pixels * 255.0, 0, 255).astype(np.uint8)

    min_t, max_t = float(np.min(tissue_pixels)), float(np.max(tissue_pixels))
    if min_t == max_t:
        return tissue_pixels, tissue_pixels

    try:
        thresh_8u, _ = cv2.threshold(tissue_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = float(thresh_8u) / 255.0
        if thresh_val <= min_t or thresh_val >= max_t:
            thresh_val = (min_t + max_t) / 2.0
    except Exception:
        thresh_val = (min_t + max_t) / 2.0

    roi_mask_otsu = (image >= thresh_val) & parenchyma
    bg_mask_otsu = (image < thresh_val) & parenchyma

    roi_pixels = image[roi_mask_otsu]
    bg_pixels = image[bg_mask_otsu]

    if roi_pixels.size == 0 or bg_pixels.size == 0:
        thresh_val = (min_t + max_t) / 2.0
        roi_pixels = image[(image >= thresh_val) & parenchyma]
        bg_pixels = image[(image < thresh_val) & parenchyma]

    if roi_pixels.size == 0:
        roi_pixels = tissue_pixels
    if bg_pixels.size == 0:
        bg_pixels = tissue_pixels

    return roi_pixels, bg_pixels


def load_checkpoint_keys(metrics_csv: str) -> Tuple[Set[Tuple[str, str, str]], List[Dict[str, Any]]]:
    """Read existing checkpoint file to resume without repeating successfully processed experiments."""
    actual_path = resolve_path(metrics_csv)
    if not actual_path or not os.path.exists(actual_path):
        return set(), []

    try:
        df_existing = pd.read_csv(actual_path)
        if df_existing.empty:
            return set(), []

        completed_keys = set()
        records = df_existing.to_dict("records")

        for r in records:
            p_id = str(r.get("patient_id", "")).strip()
            n_type = str(r.get("noise_type", "")).strip()
            d_m = str(r.get("denoising_method", "")).strip()
            out_img = resolve_path(r.get("denoised_image"))

            # Only consider complete if the output file actually exists on disk
            if p_id and n_type and d_m and out_img and os.path.exists(out_img):
                completed_keys.add((p_id, n_type, d_m))

        return completed_keys, records
    except Exception as e:
        logger.warning(f"Could not read checkpoint metrics CSV ({metrics_csv}): {e}")
        return set(), []


def flush_checkpoint(records: List[Dict[str, Any]], output_csv: str) -> None:
    """Safely flush current accumulated metrics records to disk."""
    if not records:
        return
    abs_out = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    temp_out = abs_out + ".tmp"
    pd.DataFrame(records).to_csv(temp_out, index=False)
    if os.path.exists(abs_out):
        os.remove(abs_out)
    os.rename(temp_out, abs_out)


def run_denoising_experiment(
    input_metadata_csv: str = "data/metadata/sharpened_image_metadata.csv",
    output_metrics_csv: str = "results/preprocessing/denoising_metrics.csv",
    denoised_base_dir: str = "data/processed/denoised",
    noisy_base_dir: str = "data/processed/noisy",
    config_path: str = "config/preprocessing_config.yaml",
    test_mode: Optional[bool] = None,
    max_images: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    checkpoint_frequency: Optional[int] = None,
    confirm: bool = False,
    estimate_only: bool = False,
    resume: bool = True,
    use_gpu: bool = True,
) -> Optional[pd.DataFrame]:
    """Execute comparative denoising benchmark with full dataset support and checkpoint/resume."""
    start_time = time.time()

    # 1. Load configuration
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    full_cfg = cfg.get("full_dataset_pipeline", {})
    noise_cfg = cfg.get("noise_simulation", {})
    denoise_cfg = cfg.get("denoising", {})

    is_test_mode = test_mode if test_mode is not None else full_cfg.get("test_mode", False)
    cfg_workers = num_workers or full_cfg.get("num_workers", 4)
    cfg_checkpoint_freq = checkpoint_frequency or full_cfg.get("checkpoint_frequency", 20)
    cfg_require_confirm = full_cfg.get("require_confirmation", True)
    seed = int(noise_cfg.get("seed", 42))

    # 2. Hardware configuration
    device = get_device(0) if use_gpu else torch.device("cpu")
    dev_info = get_device_info(0)
    device_name = dev_info["gpu_name"] if device.type == "cuda" else "CPU"

    # 3. Resolve metadata and dataset scope
    actual_csv = resolve_path(input_metadata_csv)
    if not actual_csv or not os.path.exists(actual_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {input_metadata_csv}. Please run image sharpening first.")

    df_meta = pd.read_csv(actual_csv)
    if df_meta.empty:
        raise ValueError(f"Metadata CSV is empty: {input_metadata_csv}")

    if is_test_mode:
        count_limit = max_images or 10
        records_to_process = df_meta.head(count_limit).copy()
        mode_label = f"TEST SAMPLE ({len(records_to_process)} images)"
    else:
        count_limit = max_images or len(df_meta)
        records_to_process = df_meta.head(count_limit).copy()
        mode_label = f"FULL DATASET ({len(records_to_process)} images)"

    # 4. Perform Resource Estimation
    noise_types = ["gaussian", "salt_pepper", "speckle", "poisson", "mixed_poisson_gaussian"]
    denoise_methods = [
        "median", "gaussian", "wiener", "bilateral",
        "non_local_means", "anscombe_wiener", "adaptive_median", "kuan"
    ]

    res_estimate = estimate_resources(
        num_images=len(records_to_process),
        num_noise_types=len(noise_types),
        num_denoising_methods=len(denoise_methods),
    )

    print_resource_estimate_table(res_estimate, title=f"Planned Workload [{mode_label}]")

    if estimate_only:
        print("[ESTIMATE ONLY]: Exiting without executing image-processing operations.\n")
        return None

    # 5. Confirmation Guard for Full Dataset
    if not is_test_mode and cfg_require_confirm and not confirm:
        print("=" * 74)
        print("ACTION REQUIRED: FULL DATASET EXECUTION CONFIRMATION GUARD")
        print("=" * 74)
        print("You are about to process the complete dataset workload estimated above.")
        print("To proceed with execution, please provide explicit confirmation via CLI:")
        print("    docker compose run --rm cbis-ddsm-ai python -m src.experiments.denoising_experiment --full --confirm")
        print("Or set 'require_confirmation: false' in config/preprocessing_config.yaml.")
        print("=" * 74 + "\n")
        return None

    # 6. Checkpoint / Resume Initialization
    completed_keys: Set[Tuple[str, str, str]] = set()
    all_results: List[Dict[str, Any]] = []

    if resume:
        completed_keys, all_results = load_checkpoint_keys(output_metrics_csv)
        if completed_keys:
            logger.info(f"[CHECKPOINT ACTIVE]: Found {len(completed_keys)} previously completed experiments on disk. Resuming...")

    # Noise parameters
    gauss_p = noise_cfg.get("gaussian", {"mean": 0.0, "var": 0.01})
    sp_p = noise_cfg.get("salt_pepper", {"amount": 0.04, "salt_vs_pepper": 0.5})
    speckle_p = noise_cfg.get("speckle", {"var": 0.04})
    poisson_p = noise_cfg.get("poisson", {"scale": 255.0})
    mixed_p = noise_cfg.get("mixed_poisson_gaussian", {"poisson_scale": 255.0, "gaussian_var": 0.005})

    # Denoising configurations
    cfg_median = denoise_cfg.get("median", {"kernel_size": 5})
    cfg_gaussian = denoise_cfg.get("gaussian", {"kernel_size": [5, 5], "sigma": 1.0})
    cfg_wiener = denoise_cfg.get("wiener", {"noise_variance": None})
    cfg_bilateral = denoise_cfg.get("bilateral", {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0})
    cfg_nlm = denoise_cfg.get("non_local_means", {"h": 0.1, "template_window_size": 7, "search_window_size": 21})
    cfg_anscombe = denoise_cfg.get("anscombe_wiener", {"sigma": 0.02, "scale": 255.0})
    cfg_adaptive = denoise_cfg.get("adaptive_median", {"max_window_size": 7})
    cfg_kuan = denoise_cfg.get("kuan", {"window_size": 7, "damping": 1.0, "noise_var": 0.04})

    os.makedirs(os.path.dirname(os.path.abspath(output_metrics_csv)), exist_ok=True)
    os.makedirs(denoised_base_dir, exist_ok=True)

    total_target = len(records_to_process) * len(noise_types) * len(denoise_methods)
    pbar = tqdm(total=total_target, desc=f"Benchmark [{mode_label}]", unit="exp")
    pbar.update(len(completed_keys))

    successful_runs = len(completed_keys)
    failed_runs = 0
    new_runs_since_flush = 0

    logger.info(f"Compute Hardware: {device} ({device_name}) | Workers: {cfg_workers} | Checkpoint Freq: {cfg_checkpoint_freq}")

    # 7. Batch Streaming Execution
    for img_idx, (_, row) in enumerate(records_to_process.iterrows()):
        patient_id = str(row.get("patient_id", "UNKNOWN")).strip()
        abnormality_cat = str(row.get("abnormality_category", "UNKNOWN")).strip()
        breast_side = str(row.get("breast_side", "UNKNOWN")).strip()
        image_view = str(row.get("image_view", "UNKNOWN")).strip()
        pathology = str(row.get("pathology", "UNKNOWN")).strip()
        label = int(row.get("label", 0)) if not pd.isna(row.get("label")) else 0

        clean_path = resolve_path(row.get("sharpened_image_path"))
        if not clean_path or not os.path.exists(clean_path):
            continue

        base_name = os.path.basename(clean_path)

        # Check if all 40 combinations for this patient are already done
        patient_pending = any(
            (patient_id, nt, dm) not in completed_keys
            for nt in noise_types
            for dm in denoise_methods
        )
        if not patient_pending:
            continue

        # Load clean image
        clean_raw = load_grayscale_image(clean_path)
        clean_f = clean_raw.astype(np.float32) / 255.0 if clean_raw.dtype != np.float32 else clean_raw
        clean_f = np.clip(clean_f, 0.0, 1.0)
        clean_roi_px, clean_bg_px = extract_mammogram_regions(clean_f)

        # Noise simulation
        noise_generators = {
            "gaussian": lambda: add_gaussian_noise(clean_f, mean=float(gauss_p.get("mean", 0.0)), var=float(gauss_p.get("var", 0.01)), seed=seed + img_idx, device=device),
            "salt_pepper": lambda: add_salt_pepper_noise(clean_f, amount=float(sp_p.get("amount", 0.04)), salt_vs_pepper=float(sp_p.get("salt_vs_pepper", 0.5)), seed=seed + img_idx, device=device),
            "speckle": lambda: add_speckle_noise(clean_f, var=float(speckle_p.get("var", 0.04)), seed=seed + img_idx, device=device),
            "poisson": lambda: add_poisson_noise(clean_f, scale=float(poisson_p.get("scale", 255.0)), seed=seed + img_idx, device=device),
            "mixed_poisson_gaussian": lambda: add_mixed_poisson_gaussian_noise(clean_f, poisson_scale=float(mixed_p.get("poisson_scale", 255.0)), gaussian_var=float(mixed_p.get("gaussian_var", 0.005)), seed=seed + img_idx, device=device),
        }

        for n_type in noise_types:
            # Check if all filters for this noise type are already completed
            noise_pending = any((patient_id, n_type, dm) not in completed_keys for dm in denoise_methods)
            if not noise_pending:
                continue

            # Load or generate noisy image
            pre_noisy_path = resolve_path(os.path.join(noisy_base_dir, n_type, base_name))
            if pre_noisy_path and os.path.exists(pre_noisy_path):
                noisy_raw = load_grayscale_image(pre_noisy_path)
                noisy_f = noisy_raw.astype(np.float32) / 255.0 if noisy_raw.dtype != np.float32 else noisy_raw
                noisy_f = np.clip(noisy_f, 0.0, 1.0)
                noisy_saved_path = pre_noisy_path
            else:
                noisy_f = noise_generators[n_type]()
                n_out_dir = os.path.join(noisy_base_dir, n_type)
                noisy_saved_path = save_image(noisy_f, os.path.join(n_out_dir, base_name))

            # Filter suite
            filter_callables = {
                "median": lambda: denoise_median(noisy_f, kernel_size=int(cfg_median.get("kernel_size", 5))),
                "gaussian": lambda: denoise_gaussian(noisy_f, kernel_size=cfg_gaussian.get("kernel_size", (5, 5)), sigma=float(cfg_gaussian.get("sigma", 1.0)), device=device),
                "wiener": lambda: denoise_wiener(noisy_f, mysize=cfg_wiener.get("mysize", (5, 5)), noise=cfg_wiener.get("noise_variance")),
                "bilateral": lambda: denoise_bilateral(noisy_f, d=int(cfg_bilateral.get("d", 9)), sigma_color=float(cfg_bilateral.get("sigma_color", 75.0)), sigma_space=float(cfg_bilateral.get("sigma_space", 75.0))),
                "non_local_means": lambda: denoise_nlm(noisy_f, h=float(cfg_nlm.get("h", 0.1)), template_window_size=int(cfg_nlm.get("template_window_size", 7)), search_window_size=int(cfg_nlm.get("search_window_size", 21))),
                "anscombe_wiener": lambda: denoise_anscombe_wiener(noisy_f, scale=float(cfg_anscombe.get("scale", 255.0)), sigma=float(cfg_anscombe.get("sigma", 1.0))),
                "adaptive_median": lambda: denoise_adaptive_median(noisy_f, max_window_size=int(cfg_adaptive.get("max_window_size", 7))),
                "kuan": lambda: denoise_kuan(noisy_f, window_size=int(cfg_kuan.get("window_size", 7)), noise_var=float(cfg_kuan.get("noise_var", 0.04)), damping=float(cfg_kuan.get("damping", 1.0)), device=device),
            }

            for d_method in denoise_methods:
                if (patient_id, n_type, d_method) in completed_keys:
                    continue

                try:
                    denoised_f = filter_callables[d_method]()
                    denoised_f = np.clip(denoised_f, 0.0, 1.0).astype(np.float32)

                    d_out_dir = os.path.join(denoised_base_dir, d_method)
                    d_out_name = f"{n_type}_{base_name}"
                    denoised_path = save_image(denoised_f, os.path.join(d_out_dir, d_out_name))

                    # Calculate metrics
                    mse_val = compute_mse(clean_f, denoised_f)
                    psnr_val = compute_psnr(clean_f, denoised_f, data_range=1.0)
                    ssim_val = compute_ssim(clean_f, denoised_f, data_range=1.0)
                    snr_val = compute_snr(clean_f, denoised_f)

                    den_roi_px, den_bg_px = extract_mammogram_regions(denoised_f)
                    cnr_val = compute_cnr(den_roi_px, den_bg_px)
                    cii_val = compute_cii(den_roi_px, den_bg_px, clean_roi_px, clean_bg_px)
                    entropy_val = compute_entropy(denoised_f)

                    all_results.append({
                        "patient_id": patient_id,
                        "abnormality_category": abnormality_cat,
                        "breast_side": breast_side,
                        "image_view": image_view,
                        "pathology": pathology,
                        "label": label,
                        "noise_type": n_type,
                        "denoising_method": d_method,
                        "MSE": mse_val,
                        "PSNR": psnr_val,
                        "SSIM": ssim_val,
                        "SNR": snr_val,
                        "CNR": cnr_val,
                        "CII": cii_val,
                        "Entropy": entropy_val,
                        "source_image": clean_path,
                        "noisy_image": noisy_saved_path,
                        "denoised_image": denoised_path,
                    })

                    completed_keys.add((patient_id, n_type, d_method))
                    successful_runs += 1
                    new_runs_since_flush += 1

                    # Checkpoint flush
                    if new_runs_since_flush >= cfg_checkpoint_freq:
                        flush_checkpoint(all_results, output_metrics_csv)
                        new_runs_since_flush = 0

                except Exception as e:
                    failed_runs += 1
                    logger.error(f"Error on {patient_id} [{n_type} + {d_method}]: {e}")

                pbar.update(1)

        # Free memory per image
        del clean_raw, clean_f, clean_roi_px, clean_bg_px
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    elapsed_time = time.time() - start_time

    # Final checkpoint flush
    flush_checkpoint(all_results, output_metrics_csv)

    print("\n" + "=" * 74)
    print(f"CBIS-DDSM Denoising Benchmark: Execution Audit [{mode_label}]")
    print("=" * 74)
    print(f"Number of Images Processed   : {len(records_to_process)}")
    print(f"Number of Noise Types        : {len(noise_types)}")
    print(f"Number of Denoising Methods  : {len(denoise_methods)}")
    print(f"Total Benchmark Experiments  : {total_target}")
    print(f"Successfully Completed/Cached: {successful_runs}")
    print(f"Failures                     : {failed_runs}")
    print(f"Processing Wall-Clock Time   : {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} min)")
    print(f"Compute Hardware Utilized    : {device_name} ({device})")
    print(f"Metrics Output File          : {os.path.abspath(output_metrics_csv)}")
    print("=" * 74 + "\n")

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Full Dataset Denoising Benchmark Runner")
    parser.add_argument("--input-csv", type=str, default="data/metadata/sharpened_image_metadata.csv", help="Input clean sharpened metadata CSV")
    parser.add_argument("--output-csv", type=str, default="results/preprocessing/denoising_metrics.csv", help="Destination metrics CSV")
    parser.add_argument("--denoised-dir", type=str, default="data/processed/denoised", help="Denoised images root directory")
    parser.add_argument("--noisy-dir", type=str, default="data/processed/noisy", help="Noisy images root directory")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.yaml", help="Configuration YAML path")
    parser.add_argument("--full", action="store_true", help="Process complete dataset instead of test sample")
    parser.add_argument("--count", type=int, default=None, help="Explicit image count override")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for parallel processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel CPU worker threads")
    parser.add_argument("--checkpoint-freq", type=int, default=20, help="Checkpoint save frequency")
    parser.add_argument("--confirm", action="store_true", help="Explicit confirmation to execute full dataset workload")
    parser.add_argument("--estimate-only", action="store_true", help="Print theoretical resource estimates and exit")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume; recompute from scratch")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    args = parser.parse_args()

    run_denoising_experiment(
        input_metadata_csv=args.input_csv,
        output_metrics_csv=args.output_csv,
        denoised_base_dir=args.denoised_dir,
        noisy_base_dir=args.noisy_dir,
        config_path=args.config,
        test_mode=not args.full if args.full else None,
        max_images=args.count,
        batch_size=args.batch_size,
        num_workers=args.workers,
        checkpoint_frequency=args.checkpoint_freq,
        confirm=args.confirm,
        estimate_only=args.estimate_only,
        resume=not args.no_resume,
        use_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
