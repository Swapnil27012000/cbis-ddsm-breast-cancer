"""Metadata builder to consolidate CBIS-DDSM descriptions into master metadata and reports."""
import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Dict
from .csv_loader import load_cbis_csvs
from .path_resolver import PathResolver
from .dataset_split import create_patient_split

def standardize_pathology(pathology: str) -> str:
    """Standardize pathology labels:
    BENIGN or BENIGN_WITHOUT_CALLBACK -> BENIGN
    MALIGNANT -> MALIGNANT
    """
    if not isinstance(pathology, str):
        return "UNKNOWN"
    norm = pathology.strip().upper()
    if "BENIGN" in norm:
        return "BENIGN"
    elif "MALIGNANT" in norm:
        return "MALIGNANT"
    return "UNKNOWN"

def build_master_metadata(
    raw_csv_dir: str = "data/raw/CBIS_DDSM/csv",
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    output_dir: str = "data/metadata"
) -> Dict[str, pd.DataFrame]:
    """Build unified master metadata dataframe, image validation report, statistics, and patient splits."""
    os.makedirs(output_dir, exist_ok=True)
    dfs = load_cbis_csvs(raw_csv_dir)
    resolver = PathResolver(raw_data_dir)

    dfs["mass_train"]["abnormality_category"] = "mass"
    dfs["mass_train"]["dataset_split"] = "train"
    dfs["mass_test"]["abnormality_category"] = "mass"
    dfs["mass_test"]["dataset_split"] = "test"
    dfs["calc_train"]["abnormality_category"] = "calcification"
    dfs["calc_train"]["dataset_split"] = "train"
    dfs["calc_test"]["abnormality_category"] = "calcification"
    dfs["calc_test"]["dataset_split"] = "test"

    combined = pd.concat(
        [dfs["mass_train"], dfs["mass_test"], dfs["calc_train"], dfs["calc_test"]],
        ignore_index=True
    )

    # Standardize column names
    combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]

    # Standardize labels
    combined["standardized_pathology"] = combined["pathology"].apply(standardize_pathology)
    # Map benign: 0, malignant: 1
    combined["label"] = combined["standardized_pathology"].map({"BENIGN": 0, "MALIGNANT": 1})

    # Resolve local disk paths for full mammogram and cropped images
    full_paths = []
    cropped_paths = []
    roi_paths = []

    print("[MetadataBuilder] Resolving local image paths...")
    for _, row in combined.iterrows():
        f_p = resolver.resolve_image_path(row.get("image_file_path", None))
        c_p = resolver.resolve_image_path(row.get("cropped_image_file_path", None))
        r_p = resolver.resolve_image_path(row.get("roi_mask_file_path", None))
        full_paths.append(f_p)
        cropped_paths.append(c_p)
        roi_paths.append(r_p)

    combined["resolved_full_mammogram_path"] = full_paths
    combined["resolved_cropped_path"] = cropped_paths
    combined["resolved_roi_mask_path"] = roi_paths

    # Primary image path column (cropped or full)
    combined["image_path"] = np.where(
        combined["resolved_full_mammogram_path"].notna(),
        combined["resolved_full_mammogram_path"],
        combined["resolved_cropped_path"]
    )
    combined["file_exists"] = combined["image_path"].notna()

    # 1. Master metadata
    master_path = os.path.join(output_dir, "CBIS_DDSM_master_metadata.csv")
    combined.to_csv(master_path, index=False)
    print(f"[MetadataBuilder] Saved master metadata to: {master_path}")

    # 2. Image validation report
    validation_records = []
    print("[MetadataBuilder] Validating resolved images...")
    for idx, row in combined[combined["file_exists"]].head(200).iterrows(): # sample or full
        img_p = row["image_path"]
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        is_valid = img is not None and img.size > 0
        validation_records.append({
            "patient_id": row["patient_id"],
            "image_path": img_p,
            "is_valid": is_valid,
            "height": img.shape[0] if is_valid else None,
            "width": img.shape[1] if is_valid else None,
            "min_intensity": int(np.min(img)) if is_valid else None,
            "max_intensity": int(np.max(img)) if is_valid else None,
            "pathology": row["standardized_pathology"]
        })
    val_df = pd.DataFrame(validation_records)
    val_path = os.path.join(output_dir, "image_validation_report.csv")
    val_df.to_csv(val_path, index=False)
    print(f"[MetadataBuilder] Saved validation report to: {val_path}")

    # 3. Dataset statistics report
    stats_data = [
        {"metric": "total_records", "value": len(combined)},
        {"metric": "available_on_disk", "value": int(combined["file_exists"].sum())},
        {"metric": "total_patients", "value": combined["patient_id"].nunique()},
        {"metric": "benign_count", "value": int((combined["standardized_pathology"] == "BENIGN").sum())},
        {"metric": "malignant_count", "value": int((combined["standardized_pathology"] == "MALIGNANT").sum())},
        {"metric": "mass_cases", "value": int((combined["abnormality_category"] == "mass").sum())},
        {"metric": "calcification_cases", "value": int((combined["abnormality_category"] == "calcification").sum())},
    ]
    stats_df = pd.DataFrame(stats_data)
    stats_path = os.path.join(output_dir, "dataset_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[MetadataBuilder] Saved dataset statistics to: {stats_path}")

    # 4. Patient split report
    valid_df = combined[combined["label"].notna()].copy()
    train_df, val_df_split, test_df = create_patient_split(
        valid_df,
        patient_col="patient_id",
        label_col="label",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    split_summary = pd.DataFrame([
        {"split": "train", "cases": len(train_df), "patients": train_df["patient_id"].nunique(), "malignant_pct": float(train_df["label"].mean() * 100)},
        {"split": "val", "cases": len(val_df_split), "patients": val_df_split["patient_id"].nunique(), "malignant_pct": float(val_df_split["label"].mean() * 100)},
        {"split": "test", "cases": len(test_df), "patients": test_df["patient_id"].nunique(), "malignant_pct": float(test_df["label"].mean() * 100)},
    ])
    split_path = os.path.join(output_dir, "patient_split_report.csv")
    split_summary.to_csv(split_path, index=False)
    print(f"[MetadataBuilder] Saved patient split report to: {split_path}")

    return {
        "master": combined,
        "validation": val_df,
        "statistics": stats_df,
        "split": split_summary
    }

if __name__ == "__main__":
    build_master_metadata()
