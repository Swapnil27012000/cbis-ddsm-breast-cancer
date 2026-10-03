"""Batch artificial noise generation runner for CBIS-DDSM denoising benchmark study.

IMPORTANT RESEARCH DISTINCTION:
--------------------------------
The noise simulated by this module is ARTIFICIALLY GENERATED for an experimental
medical image denoising research benchmark. It does NOT represent naturally occurring
or inherent physical noise in the CBIS-DDSM mammography dataset.
"""
import os
import json
import argparse
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from src.utils.image_utils import load_grayscale_image, save_image
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.gpu.device import get_device, get_device_info
from src.noise.gaussian import add_gaussian_noise
from src.noise.salt_pepper import add_salt_pepper_noise
from src.noise.speckle import add_speckle_noise
from src.noise.poisson import add_poisson_noise
from src.noise.mixed_poisson_gaussian import add_mixed_poisson_gaussian_noise

logger = setup_logger("NoiseSimulation")

def resolve_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve file path across host and Docker container filesystems."""
    if not path_str or pd.isna(path_str):
        return None

    path_str = str(path_str).strip()
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    if path_str.startswith("/app/"):
        rel = path_str[len("/app/"):]
        if os.path.exists(rel):
            return os.path.abspath(rel)

    if not path_str.startswith("/app/"):
        in_docker = os.path.join("/app", path_str)
        if os.path.exists(in_docker):
            return os.path.abspath(in_docker)

    return None

def run_noise_generation(
    input_metadata_csv: str = "data/metadata/sharpened_image_metadata.csv",
    output_base_dir: str = "data/processed/noisy",
    output_metadata_csv: str = "data/metadata/noisy_image_metadata.csv",
    config_path: str = "config/preprocessing_config.yaml",
    override_test_mode: Optional[bool] = None,
    override_test_count: Optional[int] = None,
    use_gpu: bool = True
) -> pd.DataFrame:
    """Generate 5 artificial noise models across sharpened CBIS-DDSM mammograms.

    Args:
        input_metadata_csv: Path to sharpened image metadata CSV.
        output_base_dir: Root directory for noisy images.
        output_metadata_csv: Output CSV documenting generated noise.
        config_path: Configuration YAML path.
        override_test_mode: Overrides config test mode if provided.
        override_test_count: Overrides test image count if provided.
        use_gpu: Whether to utilize PyTorch GPU acceleration if available.

    Returns:
        pd.DataFrame: Table describing generated noisy datasets.
    """
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    noise_cfg = cfg.get("noise_simulation", {})
    basic_cfg = cfg.get("basic_preprocessing", {})

    test_mode = override_test_mode if override_test_mode is not None else noise_cfg.get("test_mode", basic_cfg.get("test_mode", True))
    test_count = override_test_count if override_test_count is not None else noise_cfg.get("test_image_count", basic_cfg.get("test_image_count", 10))
    seed = int(noise_cfg.get("seed", 42))

    # Device configuration
    device = get_device(0) if use_gpu else torch.device("cpu")
    dev_info = get_device_info(0)
    device_name = dev_info["gpu_name"] if device.type == "cuda" else "CPU"
    logger.info(f"Compute Hardware: {device} ({device_name}) | GPU Acceleration: {'ACTIVE' if device.type == 'cuda' else 'INACTIVE'}")

    actual_csv = resolve_path(input_metadata_csv)
    if not actual_csv or not os.path.exists(actual_csv):
        raise FileNotFoundError(f"Sharpened metadata CSV not found: {input_metadata_csv}. Please run image sharpening first.")

    df = pd.read_csv(actual_csv)
    if df.empty:
        raise ValueError("Sharpened metadata CSV is empty.")

    # Target noise subdirectories
    subdirs = {
        "gaussian": os.path.join(output_base_dir, "gaussian"),
        "salt_pepper": os.path.join(output_base_dir, "salt_pepper"),
        "speckle": os.path.join(output_base_dir, "speckle"),
        "poisson": os.path.join(output_base_dir, "poisson"),
        "mixed_poisson_gaussian": os.path.join(output_base_dir, "mixed_poisson_gaussian"),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_csv)), exist_ok=True)

    records_to_process = df.copy()
    if test_mode:
        records_to_process = records_to_process.head(test_count)
        logger.info(f"[TEST MODE ACTIVE] Generating noise for {len(records_to_process)} clean mammograms across 5 models.")
    else:
        logger.info(f"[FULL MODE ACTIVE] Generating noise for {len(records_to_process)} clean mammograms across 5 models.")

    # Extract noise parameters from config
    gauss_params = noise_cfg.get("gaussian", {"mean": 0.0, "var": 0.01})
    sp_params = noise_cfg.get("salt_pepper", {"amount": 0.04, "salt_vs_pepper": 0.5})
    speckle_params = noise_cfg.get("speckle", {"var": 0.04})
    poisson_params = noise_cfg.get("poisson", {"scale": 255.0})
    mixed_params = noise_cfg.get("mixed_poisson_gaussian", {"poisson_scale": 255.0, "gaussian_var": 0.005})

    noisy_records: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    pbar = tqdm(records_to_process.iterrows(), total=len(records_to_process), desc="Simulating Noise", unit="img")

    for idx, row in pbar:
        patient_id = str(row.get("patient_id", "UNKNOWN")).strip()
        pathology = str(row.get("pathology", "UNKNOWN")).strip()
        label = row.get("label", 0)

        # Source sharpened image
        sharp_path = resolve_path(row.get("sharpened_image_path"))
        if not sharp_path or not os.path.exists(sharp_path):
            failure_count += 1
            logger.warning(f"Sharpened image not found for patient {patient_id}: {row.get('sharpened_image_path')}")
            continue

        base_name = os.path.basename(sharp_path)

        try:
            # 1. Load clean image (normalized [0, 1] float)
            clean_img = load_grayscale_image(sharp_path)
            clean_f = clean_img.astype(np.float32) / 255.0 if clean_img.dtype != np.float32 else clean_img

            # 2. Simulate 5 noise models with deterministic seeds
            generators = [
                ("gaussian", gauss_params, lambda: add_gaussian_noise(
                    clean_f, mean=float(gauss_params.get("mean", 0.0)), var=float(gauss_params.get("var", 0.01)), seed=seed + idx, device=device
                )),
                ("salt_pepper", sp_params, lambda: add_salt_pepper_noise(
                    clean_f, amount=float(sp_params.get("amount", 0.04)), salt_vs_pepper=float(sp_params.get("salt_vs_pepper", 0.5)), seed=seed + idx, device=device
                )),
                ("speckle", speckle_params, lambda: add_speckle_noise(
                    clean_f, var=float(speckle_params.get("var", 0.04)), seed=seed + idx, device=device
                )),
                ("poisson", poisson_params, lambda: add_poisson_noise(
                    clean_f, scale=float(poisson_params.get("scale", 255.0)), seed=seed + idx, device=device
                )),
                ("mixed_poisson_gaussian", mixed_params, lambda: add_mixed_poisson_gaussian_noise(
                    clean_f, poisson_scale=float(mixed_params.get("poisson_scale", 255.0)), gaussian_var=float(mixed_params.get("gaussian_var", 0.005)), seed=seed + idx, device=device
                )),
            ]

            for noise_type, params, gen_fn in generators:
                noisy_arr = gen_fn()
                out_path = os.path.join(subdirs[noise_type], base_name)
                saved_path = save_image(noisy_arr, out_path)

                noisy_records.append({
                    "noise_type": noise_type,
                    "noise_parameters": json.dumps(params),
                    "source_image": sharp_path,
                    "output_image": saved_path,
                    "patient_id": patient_id,
                    "pathology": pathology,
                    "label": int(label) if not pd.isna(label) else 0,
                    "is_synthetic": True,
                })

            success_count += 1

        except Exception as e:
            failure_count += 1
            logger.warning(f"Error simulating noise for {sharp_path}: {e}")

    out_df = pd.DataFrame(noisy_records)
    out_df.to_csv(output_metadata_csv, index=False)

    print("\n" + "=" * 68)
    print("CBIS-DDSM Artificial Noise Simulation Summary")
    print("=" * 68)
    print(f"Research Designation  : Synthetic Noise for Benchmark Denoising Study")
    print(f"Compute Hardware      : {device_name} ({device})")
    print(f"Clean Images Sampled  : {len(records_to_process)}")
    print(f"Noise Models per Image: 5 (Gaussian, Salt & Pepper, Speckle, Poisson, Mixed)")
    print(f"Total Noisy Images    : {len(noisy_records)} (Successfully saved: {success_count * 5})")
    print(f"Failures              : {failure_count}")
    print(f"Noisy Images Directory: {os.path.abspath(output_base_dir)}")
    print(f"Metadata CSV Generated: {os.path.abspath(output_metadata_csv)}")
    print("=" * 68 + "\n")

    return out_df

def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Artificial Noise Generation")
    parser.add_argument("--input-csv", type=str, default="data/metadata/sharpened_image_metadata.csv", help="Input sharpened metadata CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed/noisy", help="Base directory for noisy outputs")
    parser.add_argument("--output-csv", type=str, default="data/metadata/noisy_image_metadata.csv", help="Output metadata CSV")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.yaml", help="Configuration YAML path")
    parser.add_argument("--count", type=int, default=None, help="Override test image count")
    parser.add_argument("--all", action="store_true", help="Process all images instead of test sample")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    args = parser.parse_args()

    override_test = False if args.all else None
    run_noise_generation(
        input_metadata_csv=args.input_csv,
        output_base_dir=args.output_dir,
        output_metadata_csv=args.output_csv,
        config_path=args.config,
        override_test_mode=override_test,
        override_test_count=args.count,
        use_gpu=not args.cpu,
    )

if __name__ == "__main__":
    main()
