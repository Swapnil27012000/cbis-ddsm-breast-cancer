"""Contrast enhancement and controlled contrast stretching for mammograms."""
import os
import argparse
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.image_utils import load_grayscale_image, save_image, get_image_statistics
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("ContrastEnhancement")

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

def stretch_contrast(
    img: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0
) -> np.ndarray:
    """Apply controlled contrast stretching using robust percentile clipping.

    Operates in float32 internally with input range [0, 1] and output range [0, 1].
    Safely handles constant images, zero-variance inputs, NaNs, and Infs.

    Args:
        img: Input image array (normalized float [0, 1] or uint8 [0, 255]).
        lower_percentile: Lower percentile threshold (default: 2.0).
        upper_percentile: Upper percentile threshold (default: 98.0).

    Returns:
        np.ndarray: Contrast-stretched float32 image with values in [0.0, 1.0].
    """
    if img is None or img.size == 0:
        raise ValueError("Input image array is empty or None.")

    # Convert to float32
    if np.issubdtype(img.dtype, np.integer):
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)

    # Sanitize invalid values (NaN, Inf)
    img_clean = np.nan_to_num(img_f, nan=0.0, posinf=1.0, neginf=0.0)
    img_clean = np.clip(img_clean, 0.0, 1.0)

    # Validate percentile arguments
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError(f"Invalid percentiles: lower={lower_percentile}, upper={upper_percentile}. Must satisfy 0 <= lower < upper <= 100.")

    v_min = float(np.percentile(img_clean, lower_percentile))
    v_max = float(np.percentile(img_clean, upper_percentile))

    # Constant or near-constant image safeguard
    if (v_max - v_min) <= 1e-7:
        return img_clean

    clipped = np.clip(img_clean, v_min, v_max)
    stretched = (clipped - v_min) / (v_max - v_min)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)

def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    orig_dtype = img.dtype
    if orig_dtype != np.uint8:
        norm_8u = ((img - np.min(img)) / (np.ptp(img) + 1e-7) * 255).astype(np.uint8)
    else:
        norm_8u = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(norm_8u)

    if orig_dtype == np.float32:
        return enhanced.astype(np.float32) / 255.0
    return enhanced

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Apply standard global histogram equalization."""
    orig_dtype = img.dtype
    if orig_dtype != np.uint8:
        norm_8u = ((img - np.min(img)) / (np.ptp(img) + 1e-7) * 255).astype(np.uint8)
    else:
        norm_8u = img

    eq = cv2.equalizeHist(norm_8u)
    if orig_dtype == np.float32:
        return eq.astype(np.float32) / 255.0
    return eq

def run_contrast_stretching(
    input_metadata_csv: str = "data/metadata/normalized_image_metadata.csv",
    output_dir: str = "data/processed/contrast",
    output_metadata_csv: str = "data/metadata/contrast_image_metadata.csv",
    config_path: str = "config/preprocessing_config.yaml",
    override_test_mode: Optional[bool] = None,
    override_test_count: Optional[int] = None,
    override_lower_p: Optional[float] = None,
    override_upper_p: Optional[float] = None,
) -> pd.DataFrame:
    """Batch contrast stretching execution for normalized CBIS-DDSM mammograms.

    Reads data/metadata/normalized_image_metadata.csv, applies percentile-based contrast stretching,
    saves output images to data/processed/contrast/, and generates data/metadata/contrast_image_metadata.csv.

    Args:
        input_metadata_csv: Path to normalized image metadata.
        output_dir: Output directory for contrast stretched images.
        output_metadata_csv: Path to save contrast metadata CSV.
        config_path: Path to YAML configuration.
        override_test_mode: Overrides config test mode if provided.
        override_test_count: Overrides image count if provided.
        override_lower_p: Overrides lower percentile if provided.
        override_upper_p: Overrides upper percentile if provided.

    Returns:
        pd.DataFrame: Metadata for successfully processed images.
    """
    # 1. Load configuration
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    contrast_cfg = cfg.get("preprocessing", {}).get("contrast", {})
    basic_cfg = cfg.get("basic_preprocessing", {})

    test_mode = override_test_mode if override_test_mode is not None else contrast_cfg.get("test_mode", basic_cfg.get("test_mode", True))
    test_count = override_test_count if override_test_count is not None else contrast_cfg.get("test_image_count", basic_cfg.get("test_image_count", 10))
    lower_p = override_lower_p if override_lower_p is not None else float(contrast_cfg.get("lower_percentile", 2.0))
    upper_p = override_upper_p if override_upper_p is not None else float(contrast_cfg.get("upper_percentile", 98.0))

    actual_csv = resolve_path(input_metadata_csv)
    if not actual_csv or not os.path.exists(actual_csv):
        raise FileNotFoundError(f"Input metadata CSV not found: {input_metadata_csv}. Please run basic preprocessing first.")

    df = pd.read_csv(actual_csv)
    if df.empty:
        raise ValueError("Input metadata CSV is empty.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_csv)), exist_ok=True)

    records_to_process = df.copy()
    if test_mode:
        records_to_process = records_to_process.head(test_count)
        logger.info(f"[TEST MODE ACTIVE] Processing {len(records_to_process)} images with percentiles [{lower_p}%, {upper_p}%].")
    else:
        logger.info(f"[FULL MODE ACTIVE] Processing {len(records_to_process)} images with percentiles [{lower_p}%, {upper_p}%].")

    processed_records: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    pbar = tqdm(records_to_process.iterrows(), total=len(records_to_process), desc="Contrast Stretching", unit="img")

    for _, row in pbar:
        patient_id = str(row.get("patient_id", "UNKNOWN")).strip()
        abnormality_cat = str(row.get("abnormality_category", "unknown")).strip()
        breast_side = str(row.get("breast_side", "UNKNOWN")).strip()
        image_view = str(row.get("image_view", "UNKNOWN")).strip()
        pathology = str(row.get("pathology", "UNKNOWN")).strip()
        label = row.get("label", 0)
        split = str(row.get("dataset_split", "train")).strip()
        orig_img_path = str(row.get("original_image_path", "")).strip()

        # The normalized image is the input for contrast stretching
        norm_img_path = resolve_path(row.get("processed_image_path"))
        if not norm_img_path or not os.path.exists(norm_img_path):
            failure_count += 1
            logger.warning(f"Normalized image file not found for patient {patient_id}: {row.get('processed_image_path')}")
            continue

        base_name = os.path.basename(norm_img_path)
        out_filepath = os.path.join(output_dir, base_name)

        try:
            # 1. Load normalized image
            img_norm = load_grayscale_image(norm_img_path)

            # 2. Apply controlled contrast stretching [0, 1] float32
            img_stretched = stretch_contrast(
                img_norm,
                lower_percentile=lower_p,
                upper_percentile=upper_p
            )

            # 3. Save contrast-stretched image (never modifies normalized image)
            saved_path = save_image(img_stretched, out_filepath)

            # 4. Record output metadata
            processed_records.append({
                "patient_id": patient_id,
                "abnormality_category": abnormality_cat,
                "breast_side": breast_side,
                "image_view": image_view,
                "pathology": pathology,
                "label": int(label) if not pd.isna(label) else 0,
                "dataset_split": split,
                "original_image_path": orig_img_path,
                "normalized_image_path": norm_img_path,
                "contrast_image_path": saved_path,
            })
            success_count += 1

        except Exception as e:
            failure_count += 1
            logger.warning(f"Error processing image {norm_img_path} for patient {patient_id}: {e}")

    out_df = pd.DataFrame(processed_records)
    out_df.to_csv(output_metadata_csv, index=False)

    print("\n" + "=" * 65)
    print("CBIS-DDSM Contrast Stretching Execution Summary")
    print("=" * 65)
    print(f"Test Mode Active            : {test_mode}")
    print(f"Percentile Limits (Low/High): [{lower_p}%, {upper_p}%]")
    print(f"Images Attempted            : {len(records_to_process)}")
    print(f"Successfully Processed      : {success_count}")
    print(f"Failures                    : {failure_count}")
    print(f"Input Normalized Dir        : data/processed/normalized")
    print(f"Output Contrast Dir         : {os.path.abspath(output_dir)}")
    print(f"Metadata CSV Generated      : {os.path.abspath(output_metadata_csv)}")
    print("=" * 65 + "\n")

    return out_df

def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Controlled Contrast Stretching")
    parser.add_argument("--input-csv", type=str, default="data/metadata/normalized_image_metadata.csv", help="Normalized metadata CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed/contrast", help="Output contrast directory")
    parser.add_argument("--output-csv", type=str, default="data/metadata/contrast_image_metadata.csv", help="Output metadata CSV")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.yaml", help="Configuration YAML path")
    parser.add_argument("--lower-p", type=float, default=None, help="Override lower percentile (e.g. 2.0)")
    parser.add_argument("--upper-p", type=float, default=None, help="Override upper percentile (e.g. 98.0)")
    parser.add_argument("--count", type=int, default=None, help="Override number of images to process in test mode")
    parser.add_argument("--all", action="store_true", help="Process all images instead of test sample")

    args = parser.parse_args()

    override_test = False if args.all else None
    run_contrast_stretching(
        input_metadata_csv=args.input_csv,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_csv,
        config_path=args.config,
        override_test_mode=override_test,
        override_test_count=args.count,
        override_lower_p=args.lower_p,
        override_upper_p=args.upper_p,
    )

if __name__ == "__main__":
    main()
