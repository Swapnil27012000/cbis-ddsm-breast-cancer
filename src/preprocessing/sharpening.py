"""Controlled medical image sharpening using unsharp masking for mammograms."""
import os
import argparse
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.image_utils import load_grayscale_image, save_image
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("ImageSharpening")

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

def unsharp_mask(
    img: np.ndarray,
    amount: float = 1.0,
    sigma: float = 1.0,
    kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """Apply controlled unsharp masking to enhance fine anatomical boundaries and subtle mass margins.

    MEDICAL IMAGING & CLINICAL RATIONALE:
    --------------------------------------
    Unsharp masking sharpens an image by subtracting a blurred version of the image from
    the original to obtain a high-pass frequency component, then adding a scaled portion
    back to the original:
        High-Pass Detail = Original - GaussianBlur(Original)
        Sharpened = Original + amount * High-Pass Detail

    CRITICAL WARNING REGARDING AGGRESSIVE SHARPENING:
    -------------------------------------------------
    In digital mammography, excessive or aggressive sharpening (e.g., amount > 1.5 or
    an excessively small blur radius) can:
      1. Greatly amplify high-frequency background noise and quantum mottle.
      2. Produce ringing halos along high-contrast skin boundaries and fibroglandular margins.
      3. Create artificial pseudo-speckles that mimic microcalcification clusters, introducing
         false positives in downstream CAD/AI models.
    Therefore, a controlled, non-aggressive strength (amount in [0.5, 1.2], sigma around 1.0)
    is strictly maintained to preserve authentic tissue morphology without introducing artifacts.

    Args:
        img: Input image array (normalized float [0, 1] or uint8 [0, 255]).
        amount: Sharpening strength factor (default: 1.0; recommended: 0.5 to 1.2).
        sigma: Standard deviation for Gaussian blur kernel (default: 1.0).
        kernel_size: Gaussian blur kernel size as odd tuple (default: (5, 5)).

    Returns:
        np.ndarray: Sharpened float32 image with pixel intensities clipped strictly to [0.0, 1.0].
    """
    if img is None or img.size == 0:
        raise ValueError("Input image array is empty or None.")

    # Convert safely to float32
    if np.issubdtype(img.dtype, np.integer):
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)

    # Sanitize NaNs and Infs
    img_clean = np.nan_to_num(img_f, nan=0.0, posinf=1.0, neginf=0.0)
    img_clean = np.clip(img_clean, 0.0, 1.0)

    # Ensure odd kernel dimensions
    kw, kh = kernel_size
    kw = kw if kw % 2 != 0 else kw + 1
    kh = kh if kh % 2 != 0 else kh + 1
    safe_kernel = (kw, kh)

    # Compute low-pass blurred component
    blurred = cv2.GaussianBlur(img_clean, safe_kernel, sigmaX=sigma, sigmaY=sigma)

    # High-pass detail extraction
    detail = img_clean - blurred

    # Controlled addition with strictly clipped output [0.0, 1.0]
    sharpened = img_clean + float(amount) * detail
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)

# Backward-compatible alias
apply_unsharp_mask = unsharp_mask

def apply_laplacian_sharpening(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply conservative Laplacian kernel edge enhancement."""
    if np.issubdtype(img.dtype, np.integer):
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)

    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    edges = cv2.filter2D(img_f, -1, kernel)
    out = img_f + strength * edges
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def run_image_sharpening(
    input_metadata_csv: str = "data/metadata/contrast_image_metadata.csv",
    output_dir: str = "data/processed/sharpened",
    output_metadata_csv: str = "data/metadata/sharpened_image_metadata.csv",
    config_path: str = "config/preprocessing_config.yaml",
    override_test_mode: Optional[bool] = None,
    override_test_count: Optional[int] = None,
    override_amount: Optional[float] = None,
    override_sigma: Optional[float] = None,
) -> pd.DataFrame:
    """Batch controlled image sharpening on contrast-enhanced CBIS-DDSM mammograms.

    Reads data/metadata/contrast_image_metadata.csv, applies unsharp masking,
    saves output images to data/processed/sharpened/, and outputs
    data/metadata/sharpened_image_metadata.csv. Never modifies prior outputs.

    Args:
        input_metadata_csv: Path to contrast metadata CSV.
        output_dir: Output directory for sharpened images.
        output_metadata_csv: Path to save sharpened metadata CSV.
        config_path: Path to configuration YAML.
        override_test_mode: Overrides test mode if provided.
        override_test_count: Overrides image count if provided.
        override_amount: Overrides sharpening strength factor if provided.
        override_sigma: Overrides Gaussian blur sigma if provided.

    Returns:
        pd.DataFrame: Metadata for successfully sharpened images.
    """
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    sharp_cfg = cfg.get("preprocessing", {}).get("sharpening", {})
    basic_cfg = cfg.get("basic_preprocessing", {})

    test_mode = override_test_mode if override_test_mode is not None else sharp_cfg.get("test_mode", basic_cfg.get("test_mode", True))
    test_count = override_test_count if override_test_count is not None else sharp_cfg.get("test_image_count", basic_cfg.get("test_image_count", 10))
    amount = override_amount if override_amount is not None else float(sharp_cfg.get("amount", 1.0))
    sigma = override_sigma if override_sigma is not None else float(sharp_cfg.get("sigma", 1.0))
    raw_k = sharp_cfg.get("kernel_size", [5, 5])
    kernel_size = (int(raw_k[0]), int(raw_k[1])) if isinstance(raw_k, (list, tuple)) else (5, 5)

    actual_csv = resolve_path(input_metadata_csv)
    if not actual_csv or not os.path.exists(actual_csv):
        raise FileNotFoundError(f"Contrast metadata CSV not found: {input_metadata_csv}. Please run contrast stretching first.")

    df = pd.read_csv(actual_csv)
    if df.empty:
        raise ValueError("Contrast metadata CSV is empty.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_csv)), exist_ok=True)

    records_to_process = df.copy()
    if test_mode:
        records_to_process = records_to_process.head(test_count)
        logger.info(f"[TEST MODE ACTIVE] Sharpening initial {len(records_to_process)} images (amount={amount}, sigma={sigma}, kernel={kernel_size}).")
    else:
        logger.info(f"[FULL MODE ACTIVE] Sharpening all {len(records_to_process)} images (amount={amount}, sigma={sigma}, kernel={kernel_size}).")

    processed_records: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    pbar = tqdm(records_to_process.iterrows(), total=len(records_to_process), desc="Sharpening Images", unit="img")

    for _, row in pbar:
        patient_id = str(row.get("patient_id", "UNKNOWN")).strip()
        abnormality_cat = str(row.get("abnormality_category", "unknown")).strip()
        breast_side = str(row.get("breast_side", "UNKNOWN")).strip()
        image_view = str(row.get("image_view", "UNKNOWN")).strip()
        pathology = str(row.get("pathology", "UNKNOWN")).strip()
        label = row.get("label", 0)
        split = str(row.get("dataset_split", "train")).strip()
        orig_img_path = str(row.get("original_image_path", "")).strip()
        norm_img_path = str(row.get("normalized_image_path", "")).strip()

        # The contrast-enhanced image is the direct input for sharpening
        contrast_img_path = resolve_path(row.get("contrast_image_path"))
        if not contrast_img_path or not os.path.exists(contrast_img_path):
            failure_count += 1
            logger.warning(f"Contrast image file not found for patient {patient_id}: {row.get('contrast_image_path')}")
            continue

        base_name = os.path.basename(contrast_img_path)
        out_filepath = os.path.join(output_dir, base_name)

        try:
            # 1. Load contrast-stretched image
            img_contrast = load_grayscale_image(contrast_img_path)

            # 2. Apply controlled unsharp masking [0, 1] float32
            img_sharpened = unsharp_mask(
                img_contrast,
                amount=amount,
                sigma=sigma,
                kernel_size=kernel_size
            )

            # 3. Save sharpened image (never modifies prior outputs)
            saved_path = save_image(img_sharpened, out_filepath)

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
                "contrast_image_path": contrast_img_path,
                "sharpened_image_path": saved_path,
            })
            success_count += 1

        except Exception as e:
            failure_count += 1
            logger.warning(f"Error sharpening image {contrast_img_path} for patient {patient_id}: {e}")

    out_df = pd.DataFrame(processed_records)
    out_df.to_csv(output_metadata_csv, index=False)

    print("\n" + "=" * 65)
    print("CBIS-DDSM Image Sharpening Execution Summary")
    print("=" * 65)
    print(f"Test Mode Active        : {test_mode}")
    print(f"Sharpening Parameters   : Strength={amount}, Sigma={sigma}, Kernel={kernel_size}")
    print(f"Images Attempted        : {len(records_to_process)}")
    print(f"Successfully Processed  : {success_count}")
    print(f"Failures                : {failure_count}")
    print(f"Input Contrast Dir      : data/processed/contrast")
    print(f"Output Sharpened Dir    : {os.path.abspath(output_dir)}")
    print(f"Metadata CSV Generated  : {os.path.abspath(output_metadata_csv)}")
    print("=" * 65 + "\n")

    return out_df

def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Controlled Image Sharpening")
    parser.add_argument("--input-csv", type=str, default="data/metadata/contrast_image_metadata.csv", help="Contrast metadata CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed/sharpened", help="Output sharpened directory")
    parser.add_argument("--output-csv", type=str, default="data/metadata/sharpened_image_metadata.csv", help="Output metadata CSV")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.yaml", help="Configuration YAML path")
    parser.add_argument("--amount", type=float, default=None, help="Override sharpening strength factor (e.g. 1.0)")
    parser.add_argument("--sigma", type=float, default=None, help="Override blur sigma (e.g. 1.0)")
    parser.add_argument("--count", type=int, default=None, help="Override number of images to process in test mode")
    parser.add_argument("--all", action="store_true", help="Process all images instead of test sample")

    args = parser.parse_args()

    override_test = False if args.all else None
    run_image_sharpening(
        input_metadata_csv=args.input_csv,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_csv,
        config_path=args.config,
        override_test_mode=override_test,
        override_test_count=args.count,
        override_amount=args.amount,
        override_sigma=args.sigma,
    )

if __name__ == "__main__":
    main()
