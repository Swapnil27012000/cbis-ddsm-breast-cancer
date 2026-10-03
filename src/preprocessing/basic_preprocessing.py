"""Basic CBIS-DDSM mammography preprocessing pipeline: load -> normalize -> resize -> save."""
import os
import argparse
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from tqdm import tqdm

from src.utils.image_utils import (
    load_grayscale_image,
    normalize_image,
    resize_image,
    save_image,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("BasicPreprocessing")

def resolve_existing_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve image path across host Windows and Linux Docker container environments."""
    if not path_str or pd.isna(path_str):
        return None

    path_str = str(path_str).strip()
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # Check without /app/ prefix (if running on host)
    if path_str.startswith("/app/"):
        rel_app = path_str[len("/app/"):]
        if os.path.exists(rel_app):
            return os.path.abspath(rel_app)

    # Check candidates
    candidates = [
        path_str.replace("\\", "/"),
        os.path.join("data", "raw", "CBIS_DDSM", "jpeg", os.path.basename(path_str)),
        os.path.join("dataset", "jpeg", os.path.basename(path_str)),
        os.path.join("/app", "data", "raw", "CBIS_DDSM", "jpeg", os.path.basename(path_str)),
        os.path.join("/app", "dataset", "jpeg", os.path.basename(path_str)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    return None

def run_basic_preprocessing(
    metadata_csv: str = "data/metadata/CBIS_DDSM_master_metadata.csv",
    config_path: str = "config/preprocessing_config.yaml",
    output_dir: str = "data/processed/normalized",
    output_metadata_csv: str = "data/metadata/normalized_image_metadata.csv",
    override_test_mode: Optional[bool] = None,
    override_image_count: Optional[int] = None,
    override_image_size: Optional[int] = None,
) -> pd.DataFrame:
    """Execute Stage 1 preprocessing on CBIS-DDSM mammograms.

    Pipeline:
        CSV -> Validate full mammogram path -> Load grayscale -> Normalize [0,1] -> Resize -> Save

    Args:
        metadata_csv: Path to master metadata CSV.
        config_path: Path to preprocessing configuration YAML.
        output_dir: Directory to save normalized images.
        output_metadata_csv: Path to save output metadata CSV.
        override_test_mode: If set, overrides config test_mode.
        override_image_count: If set, overrides config test_image_count.
        override_image_size: If set, overrides config image_size.

    Returns:
        pd.DataFrame: Metadata for successfully processed images.
    """
    # 1. Load configuration
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    basic_cfg = cfg.get("basic_preprocessing", {})

    test_mode = override_test_mode if override_test_mode is not None else basic_cfg.get("test_mode", True)
    test_count = override_image_count if override_image_count is not None else basic_cfg.get("test_image_count", 10)
    
    # Image size resolution
    size_cfg = override_image_size if override_image_size is not None else basic_cfg.get("image_size", cfg.get("preprocessing", {}).get("image_size", 512))
    if isinstance(size_cfg, (list, tuple)):
        target_size = (int(size_cfg[0]), int(size_cfg[1]))
    else:
        target_size = (int(size_cfg), int(size_cfg))

    # Resolve master metadata path
    actual_meta = metadata_csv
    if not os.path.exists(actual_meta) and actual_meta.startswith("data/"):
        alt_meta = os.path.join("/app", actual_meta)
        if os.path.exists(alt_meta):
            actual_meta = alt_meta

    if not os.path.exists(actual_meta):
        raise FileNotFoundError(f"Master metadata CSV not found at: {metadata_csv}. Please run metadata builder first.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_metadata_csv)), exist_ok=True)

    # 2. Read metadata and filter full mammograms
    df = pd.read_csv(actual_meta)
    
    # Identify valid full mammogram path column
    records_to_process: List[Dict[str, Any]] = []
    seen_source_paths = set()

    for idx, row in df.iterrows():
        # Prefer resolved_full_mammogram_path, fallback to image_path
        raw_candidate = row.get("resolved_full_mammogram_path")
        if not raw_candidate or pd.isna(raw_candidate):
            raw_candidate = row.get("image_path")

        resolved_path = resolve_existing_path(raw_candidate)
        if resolved_path and resolved_path not in seen_source_paths:
            seen_source_paths.add(resolved_path)
            record = row.to_dict()
            record["_source_path"] = resolved_path
            records_to_process.append(record)

    total_available = len(records_to_process)
    logger.info(f"Master metadata loaded: {len(df)} total rows. Unique available images on disk: {total_available}.")

    if test_mode:
        records_to_process = records_to_process[:test_count]
        logger.info(f"[TEST MODE ACTIVE] Processing initial sample of {len(records_to_process)} images.")
    else:
        logger.info(f"[FULL MODE ACTIVE] Processing all {len(records_to_process)} available images.")

    # 3. Process images sequentially
    processed_records: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    pbar = tqdm(records_to_process, desc="Normalizing Images", unit="img")

    for item in pbar:
        source_path = item["_source_path"]
        patient_id = str(item.get("patient_id", "UNKNOWN")).strip()
        abnormality_cat = str(item.get("abnormality_category", "unknown")).strip()
        breast_side = str(item.get("left_or_right_breast", "UNKNOWN")).strip()
        image_view = str(item.get("image_view", "UNKNOWN")).strip()
        pathology = str(item.get("pathology", "UNKNOWN")).strip()
        label = item.get("label", 0)
        split = str(item.get("dataset_split", "train")).strip()

        # Build clean, deterministic filename: <patient_id>_<cat>_<side>_<view>_<basename>
        src_stem = os.path.splitext(os.path.basename(source_path))[0]
        out_filename = f"{patient_id}_{abnormality_cat}_{breast_side}_{image_view}_{src_stem}.png"
        out_filepath = os.path.join(output_dir, out_filename)

        try:
            # 1. Load grayscale
            img = load_grayscale_image(source_path)

            # 2. Normalize to [0, 1] float32
            norm_img = normalize_image(img, target_min=0.0, target_max=1.0)

            # 3. Resize to target dimensions
            resized_img = resize_image(norm_img, target_size=target_size)

            # 4. Save processed image
            saved_path = save_image(resized_img, out_filepath)

            # 5. Record metadata
            processed_records.append({
                "patient_id": patient_id,
                "abnormality_category": abnormality_cat,
                "breast_side": breast_side,
                "image_view": image_view,
                "pathology": pathology,
                "label": int(label) if not pd.isna(label) else 0,
                "dataset_split": split,
                "original_image_path": source_path,
                "processed_image_path": saved_path,
                "width": target_size[0],
                "height": target_size[1],
            })
            success_count += 1

        except Exception as e:
            failure_count += 1
            logger.warning(f"Failed processing image '{source_path}' for patient {patient_id}: {e}")

    # 4. Export normalized metadata CSV
    out_df = pd.DataFrame(processed_records)
    out_df.to_csv(output_metadata_csv, index=False)

    print("\n" + "=" * 60)
    print("CBIS-DDSM Basic Preprocessing Execution Summary")
    print("=" * 60)
    print(f"Test Mode Active        : {test_mode}")
    print(f"Target Resolution       : {target_size[0]} x {target_size[1]}")
    print(f"Images Attempted        : {len(records_to_process)}")
    print(f"Successfully Processed  : {success_count}")
    print(f"Failures                : {failure_count}")
    print(f"Processed Images Dir    : {os.path.abspath(output_dir)}")
    print(f"Metadata CSV Generated  : {os.path.abspath(output_metadata_csv)}")
    print("=" * 60 + "\n")

    return out_df

def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM Basic Mammography Preprocessing")
    parser.add_argument("--config", type=str, default="config/preprocessing_config.yaml", help="Path to config YAML")
    parser.add_argument("--metadata", type=str, default="data/metadata/CBIS_DDSM_master_metadata.csv", help="Master metadata CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed/normalized", help="Output directory for normalized images")
    parser.add_argument("--output-csv", type=str, default="data/metadata/normalized_image_metadata.csv", help="Output CSV path")
    parser.add_argument("--all", action="store_true", help="Process entire dataset instead of test sample")
    parser.add_argument("--count", type=int, default=None, help="Override number of images to process in test mode")
    parser.add_argument("--size", type=int, default=None, help="Override square target image dimension")

    args = parser.parse_args()

    override_test = False if args.all else None
    run_basic_preprocessing(
        metadata_csv=args.metadata,
        config_path=args.config,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_csv,
        override_test_mode=override_test,
        override_image_count=args.count,
        override_image_size=args.size,
    )

if __name__ == "__main__":
    main()
