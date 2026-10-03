"""Full mammography preprocessing pipeline runner."""
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

from .validation import validate_mammogram
from .normalization import robust_min_max_normalize
from .roi_processing import segment_breast, remove_pectoral_muscle, crop_to_roi
from .contrast import apply_clahe
from .sharpening import apply_unsharp_mask
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("PreprocessingPipeline")

def run_preprocessing_pipeline(
    metadata_csv: str = "data/metadata/CBIS_DDSM_master_metadata.csv",
    config_path: str = "config/preprocessing_config.yaml",
    processed_dir: str = "data/processed",
    max_samples: Optional[int] = None
):
    """Execute stages: validate -> normalize -> roi segment -> crop -> contrast -> sharpen -> final format."""
    cfg = load_config(config_path) if os.path.exists(config_path) else {}
    target_size = tuple(cfg.get("preprocessing", {}).get("image_size", [512, 512]))

    # Target directories
    dirs = {
        "validated": os.path.join(processed_dir, "validated"),
        "normalized": os.path.join(processed_dir, "normalized"),
        "roi": os.path.join(processed_dir, "roi"),
        "cropped": os.path.join(processed_dir, "cropped"),
        "contrast": os.path.join(processed_dir, "contrast"),
        "sharpened": os.path.join(processed_dir, "sharpened"),
        "final_benign": os.path.join(processed_dir, "final", "benign"),
        "final_malignant": os.path.join(processed_dir, "final", "malignant")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(metadata_csv):
        logger.error(f"Metadata file not found: {metadata_csv}. Run metadata builder first.")
        return

    df = pd.read_csv(metadata_csv)
    valid_records = df[df["file_exists"] == True].copy()
    if max_samples:
        valid_records = valid_records.head(max_samples)

    logger.info(f"Processing {len(valid_records)} mammograms through full clinical pipeline...")

    for idx, row in tqdm(valid_records.iterrows(), total=len(valid_records), desc="Preprocessing"):
        raw_path = row["image_path"]
        patient_id = row["patient_id"]
        pathology = row["standardized_pathology"]
        base_name = f"{patient_id}_{os.path.splitext(os.path.basename(raw_path))[0]}.png"

        img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 1. Validation
        is_valid, _ = validate_mammogram(img)
        if not is_valid:
            continue
        cv2.imwrite(os.path.join(dirs["validated"], base_name), img)

        # 2. Normalization [0, 1] float
        norm_img = robust_min_max_normalize(img, p_low=1.0, p_high=99.0, target_min=0.0, target_max=1.0)
        norm_8u = np.clip(norm_img * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dirs["normalized"], base_name), norm_8u)

        # 3. ROI & Pectoral Muscle Suppression
        mask = segment_breast(norm_8u)
        roi_img = norm_8u * mask
        roi_img = remove_pectoral_muscle(roi_img)
        cv2.imwrite(os.path.join(dirs["roi"], base_name), roi_img)

        # 4. Cropped to bounding box & resize
        cropped = crop_to_roi(roi_img, mask)
        cropped_resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dirs["cropped"], base_name), cropped_resized)

        # 5. Contrast Enhancement (CLAHE)
        contrast = apply_clahe(cropped_resized, clip_limit=2.0)
        cv2.imwrite(os.path.join(dirs["contrast"], base_name), contrast)

        # 6. Sharpening (Unsharp Mask)
        sharpened = apply_unsharp_mask(contrast, amount=1.0)
        cv2.imwrite(os.path.join(dirs["sharpened"], base_name), sharpened)

        # 7. Final dataset classification buckets
        if pathology == "BENIGN":
            cv2.imwrite(os.path.join(dirs["final_benign"], base_name), sharpened)
        elif pathology == "MALIGNANT":
            cv2.imwrite(os.path.join(dirs["final_malignant"], base_name), sharpened)

    logger.info("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
