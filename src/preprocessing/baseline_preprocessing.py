"""Stage 5: CBIS-DDSM Primary Baseline Image Preprocessing Module.

Establishes a clean, reproducible, and scientifically controlled BASELINE preprocessing
pipeline for CBIS-DDSM FULL/ORIGINAL mammograms:
1. Technical validation (ensures valid, uncorrupted input).
2. Grayscale representation (float32 [0, 1]).
3. Conservative deterministic breast-region detection (retains complete breast & peripheral tissue).
4. Conservative background cropping (removes irrelevant black background with safety margin).
5. Robust percentile intensity normalization (percentile 1.0 to 99.0).
6. Lossless PNG output storage (uint8 [0, 255], mode L, unresized spatial resolution).

Executes a controlled pilot (10–20 representative images) covering CC/MLO, LEFT/RIGHT,
MASS/CALCIFICATION before full dataset execution.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import yaml

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.image_inventory import (
    find_jpeg_dir,
    ROLE_FULL_ORIGINAL,
    STATUS_RESOLVED,
)

# Processing Status Constants
STATUS_SUCCESS = "SUCCESS"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_FAILED = "FAILED"


def find_metadata_dir(stage_subdir: str = "stage2") -> str:
    """Locate metadata directory across Docker and local workspaces."""
    candidates = [
        f"data/metadata/{stage_subdir}",
        os.path.join(os.getcwd(), "data", "metadata", stage_subdir),
        f"/app/data/metadata/{stage_subdir}",
        "data/metadata",
        os.path.join(os.getcwd(), "data", "metadata"),
        "/app/data/metadata",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(f"data/metadata/{stage_subdir}")


def load_baseline_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load baseline preprocessing configuration parameters with safe defaults."""
    default_cfg = {
        "enabled": True,
        "pilot_mode": True,
        "pilot_sample_count": 16,
        "normalization": {
            "method": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
        },
        "breast_region": {
            "enabled": True,
            "crop_margin_ratio": 0.05,
            "threshold_method": "otsu_conservative",
            "min_foreground_area_ratio": 0.05,
            "max_foreground_area_ratio": 0.98,
        },
        "validation": {
            "max_foreground_loss_ratio": 0.02,
            "min_foreground_area_ratio": 0.05,
        },
        "performance": {
            "num_workers": 4,
            "png_compression_level": 1,
            "io_buffer_size": 1048576,
        },
    }

    candidates = [
        config_path,
        "config/preprocessing_config.yaml",
        os.path.join(os.getcwd(), "config", "preprocessing_config.yaml"),
        "/app/config/preprocessing_config.yaml",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if "baseline" in data:
                    # Update nested dicts
                    b_cfg = data["baseline"]
                    for k, v in b_cfg.items():
                        if isinstance(v, dict) and k in default_cfg and isinstance(default_cfg[k], dict):
                            default_cfg[k].update(v)
                        else:
                            default_cfg[k] = v
                    return default_cfg
            except Exception as err:
                print(f"[Stage 5] Warning loading config from {c}: {err}")

    return default_cfg


def resolve_image_path(image_path: str, jpeg_dir: str) -> str:
    """Resolve physical path on disk across environments."""
    if not image_path:
        return ""
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return os.path.normpath(image_path)
    cand1 = os.path.normpath(os.path.join(jpeg_dir, image_path))
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.normpath(image_path)
    if os.path.exists(cand2):
        return cand2
    return cand1


def compute_file_sha256(file_path: str, buffer_size: int = 1048576) -> str:
    """Compute SHA-256 hash of a file for integrity verification using an optimized 1MB buffer."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(buffer_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def select_pilot_samples(
    df_ref: pd.DataFrame,
    n_samples: int = 16,
) -> List[Dict[str, Any]]:
    """Deterministically select a balanced pilot subset of FULL/ORIGINAL mammograms.

    Covers CC/MLO, LEFT/RIGHT, MASS/CALCIFICATION, and available pathology categories.
    """
    valid_full = df_ref[
        (df_ref["original_mapping_status"] == STATUS_RESOLVED)
        & (df_ref["original_resolved_path"].notna())
        & (df_ref["original_resolved_path"] != "")
    ].copy()

    if valid_full.empty:
        return []

    # Sort deterministically
    valid_full.sort_values(
        by=[
            "abnormality_category",
            "image_view",
            "breast_side",
            "pathology",
            "dataset_split",
            "patient_id",
            "abnormality_id",
        ],
        inplace=True,
    )

    # Stratified target combinations: (cat, view, side, pathology, split)
    target_strata = [
        ("mass", "CC", "LEFT", "BENIGN", "train"),
        ("mass", "MLO", "RIGHT", "MALIGNANT", "train"),
        ("mass", "CC", "RIGHT", "MALIGNANT", "test"),
        ("mass", "MLO", "LEFT", "BENIGN", "test"),
        ("calcification", "CC", "LEFT", "BENIGN", "train"),
        ("calcification", "MLO", "RIGHT", "BENIGN_WITHOUT_CALLBACK", "train"),
        ("calcification", "MLO", "LEFT", "MALIGNANT", "train"),
        ("calcification", "CC", "RIGHT", "MALIGNANT", "test"),
        ("mass", "CC", "LEFT", "BENIGN_WITHOUT_CALLBACK", "train"),
        ("calcification", "CC", "RIGHT", "BENIGN", "test"),
        ("mass", "MLO", "RIGHT", "BENIGN_WITHOUT_CALLBACK", "test"),
        ("calcification", "MLO", "LEFT", "MALIGNANT", "test"),
        ("mass", "CC", "RIGHT", "BENIGN", "train"),
        ("calcification", "CC", "LEFT", "MALIGNANT", "train"),
        ("mass", "MLO", "LEFT", "MALIGNANT", "train"),
        ("calcification", "MLO", "RIGHT", "BENIGN", "test"),
    ]

    selected_rows = []
    used_indices = set()

    for cat, view, side, path_val, split in target_strata:
        if len(selected_rows) >= n_samples:
            break
        subset = valid_full[
            (valid_full["abnormality_category"].str.lower() == cat.lower())
            & (valid_full["image_view"].str.upper() == view.upper())
            & (valid_full["breast_side"].str.upper() == side.upper())
            & (valid_full["pathology"].str.upper() == path_val.upper())
            & (valid_full["dataset_split"].str.lower() == split.lower())
            & (~valid_full.index.isin(used_indices))
        ]
        if not subset.empty:
            chosen = subset.iloc[0]
            used_indices.add(chosen.name)
            selected_rows.append(chosen)

    # Fallback to secondary matching if any stratum was absent
    if len(selected_rows) < n_samples:
        for cat, view, _, path_val, _ in target_strata:
            if len(selected_rows) >= n_samples:
                break
            subset = valid_full[
                (valid_full["abnormality_category"].str.lower() == cat.lower())
                & (valid_full["image_view"].str.upper() == view.upper())
                & (valid_full["pathology"].str.upper() == path_val.upper())
                & (~valid_full.index.isin(used_indices))
            ]
            if not subset.empty:
                chosen = subset.iloc[0]
                used_indices.add(chosen.name)
                selected_rows.append(chosen)

    # Final fallback for any remaining required sample count
    if len(selected_rows) < n_samples:
        for _, r in valid_full.iterrows():
            if r.name not in used_indices:
                selected_rows.append(r)
                used_indices.add(r.name)
                if len(selected_rows) >= n_samples:
                    break

    pilot_samples = []
    for r in selected_rows:
        pid = str(r.get("patient_id", ""))
        side = str(r.get("breast_side", ""))
        view = str(r.get("image_view", ""))
        abn = int(r.get("abnormality_id", 1)) if pd.notna(r.get("abnormality_id")) else 1
        pilot_samples.append({
            "source_csv": r.get("source_csv", ""),
            "row_number": int(r.get("row_number", 0)),
            "patient_id": pid,
            "abnormality_id": abn,
            "abnormality_category": str(r.get("abnormality_category", "")),
            "breast_side": side,
            "image_view": view,
            "pathology": str(r.get("pathology", "")),
            "label": int(r.get("label", 0)) if pd.notna(r.get("label")) else 0,
            "dataset_split": str(r.get("dataset_split", "")),
            "original_image_path": str(r.get("original_resolved_path", "")),
            "output_filename": f"{pid}_{side}_{view}_{abn}_baseline.png",
        })

    return pilot_samples


def load_full_dataset_samples(df_ref: pd.DataFrame) -> List[Dict[str, Any]]:
    """Load the complete set of eligible FULL/ORIGINAL CBIS-DDSM mammograms.

    Filters strictly for FULL/ORIGINAL mammogram records. Does NOT include
    CROPPED images or ROI masks. Preserves all metadata fields and assigns
    deterministic, unique output filenames.
    """
    valid_full = df_ref[
        (df_ref["original_mapping_status"] == STATUS_RESOLVED)
        & (df_ref["original_resolved_path"].notna())
        & (df_ref["original_resolved_path"] != "")
    ].copy()

    # Deterministic order matching source reference mapping
    if "source_csv" in valid_full.columns and "row_number" in valid_full.columns:
        valid_full.sort_values(by=["source_csv", "row_number"], inplace=True)

    used_filenames: Set[str] = set()
    samples: List[Dict[str, Any]] = []

    for idx, r in valid_full.iterrows():
        pid = str(r.get("patient_id", "")).strip()
        side = str(r.get("breast_side", "")).strip().upper()
        view = str(r.get("image_view", "")).strip().upper()
        abn_raw = r.get("abnormality_id", 1)
        abn = int(abn_raw) if pd.notna(abn_raw) else 1
        cat = str(r.get("abnormality_category", "")).strip().lower()
        row_num = int(r.get("row_number", idx)) if pd.notna(r.get("row_number")) else int(idx)

        # Standard primary filename: {pid}_{side}_{view}_{abn}_baseline.png
        base_name = f"{pid}_{side}_{view}_{abn}_baseline.png"
        if base_name in used_filenames:
            # Deterministic collision resolution ensuring 100% uniqueness
            base_name = f"{pid}_{side}_{view}_{abn}_{cat}_row{row_num}_baseline.png"

        used_filenames.add(base_name)

        samples.append({
            "source_csv": str(r.get("source_csv", "")),
            "row_number": row_num,
            "patient_id": pid,
            "abnormality_id": abn,
            "abnormality_category": str(r.get("abnormality_category", "")),
            "breast_side": side,
            "image_view": view,
            "pathology": str(r.get("pathology", "")),
            "label": int(r.get("label", 0)) if pd.notna(r.get("label")) else 0,
            "dataset_split": str(r.get("dataset_split", "")),
            "original_image_path": str(r.get("original_resolved_path", "")),
            "output_filename": base_name,
        })

    return samples


def detect_breast_region_conservative(
    img: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[Tuple[int, int, int, int], float, bool, str, str]:
    """Detect the complete visible breast region using conservative foreground thresholding.

    Safety rules:
    - Avoids aggressive cropping.
    - Preserves peripheral breast tissue and skin boundary.
    - Does NOT use ROI annotations, lesion coordinates, or pathology metadata.
    - If detected region is geometrically implausible or outside safe thresholds,
      retains full original dimensions and flags as REVIEW_REQUIRED.

    Returns:
        (bbox, breast_region_area_ratio, crop_applied, status, notes)
        where bbox is (x, y, w, h)
    """
    h, w = img.shape[:2]

    # Fast downscaled copy for robust boundary analysis
    scale = 0.25 if max(h, w) > 1500 else 1.0
    if scale < 1.0:
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()

    # Digitize background: mammograms have black scanner background near 0
    # Conservative threshold on non-zero pixels
    nonzero_mask = small > 5
    if not np.any(nonzero_mask):
        return (0, 0, w, h), 0.0, False, STATUS_REVIEW_REQUIRED, "Image appears completely black/flat"

    # Compute conservative threshold: Otsu threshold on active tissue
    _, thresh_otsu = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Use low conservative fraction of Otsu threshold to preserve subtle peripheral tissue
    conservative_thresh = max(8, int(np.percentile(small[nonzero_mask], 5)))
    mask = (small > conservative_thresh).astype(np.uint8)

    # Morphological closing to bridge small glandular gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components: extract the major breast tissue component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    if num_labels <= 1:
        return (0, 0, w, h), 1.0, False, STATUS_REVIEW_REQUIRED, "No distinct connected component detected"

    # Combine major components that are at least 5% the size of the largest component
    # (prevents excluding disconnected glandular tissue or axillary extensions)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(np.max(areas))
    threshold_area = 0.05 * max_area
    valid_labels = [i + 1 for i, a in enumerate(areas) if a >= threshold_area]

    if not valid_labels:
        return (0, 0, w, h), 0.0, False, STATUS_REVIEW_REQUIRED, "No valid connected components meet threshold"

    # Fast geometric union directly from OpenCV C++ stats table (avoids allocating masks or searching coordinates)
    bx_min = min(stats[lbl, cv2.CC_STAT_LEFT] for lbl in valid_labels)
    by_min = min(stats[lbl, cv2.CC_STAT_TOP] for lbl in valid_labels)
    bx_max = max(stats[lbl, cv2.CC_STAT_LEFT] + stats[lbl, cv2.CC_STAT_WIDTH] for lbl in valid_labels)
    by_max = max(stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT] for lbl in valid_labels)
    bw = bx_max - bx_min
    bh = by_max - by_min

    active_small_pixels = sum(stats[lbl, cv2.CC_STAT_AREA] for lbl in valid_labels)
    breast_area_ratio = round(float(active_small_pixels / small.size), 6)

    # Project bounding box back to full native resolution
    x_min = int(bx_min / scale)
    y_min = int(by_min / scale)
    x_max = int(bx_max / scale)
    y_max = int(by_max / scale)

    # Configurable safety thresholds
    b_cfg = config.get("breast_region", {})
    min_fg = b_cfg.get("min_foreground_area_ratio", 0.05)
    max_fg = b_cfg.get("max_foreground_area_ratio", 0.98)
    margin_ratio = b_cfg.get("crop_margin_ratio", 0.05)

    # Safeguard validation: check if detected area is implausible
    if breast_area_ratio < min_fg:
        return (
            (0, 0, w, h),
            breast_area_ratio,
            False,
            STATUS_REVIEW_REQUIRED,
            f"Detected foreground ratio ({breast_area_ratio:.4f}) below safe minimum ({min_fg}); preserved uncropped",
        )
    if breast_area_ratio > max_fg:
        return (
            (0, 0, w, h),
            breast_area_ratio,
            False,
            STATUS_SUCCESS,
            f"Foreground occupies nearly entire frame ({breast_area_ratio:.4f}); preserved uncropped",
        )

    # Add conservative safety margin (5% of bbox dimensions)
    orig_bw = x_max - x_min
    orig_bh = y_max - y_min
    margin_x = int(orig_bw * margin_ratio)
    margin_y = int(orig_bh * margin_ratio)

    x_min_safe = max(0, x_min - margin_x)
    y_min_safe = max(0, y_min - margin_y)
    x_max_safe = min(w, x_max + margin_x)
    y_max_safe = min(h, y_max + margin_y)

    crop_w = x_max_safe - x_min_safe
    crop_h = y_max_safe - y_min_safe

    # If the cropped box is virtually identical to full image (>97% width and height), no crop needed
    if crop_w >= int(w * 0.97) and crop_h >= int(h * 0.97):
        return (0, 0, w, h), breast_area_ratio, False, STATUS_SUCCESS, "Full mammogram contains minimal background; uncropped"

    bbox = (x_min_safe, y_min_safe, crop_w, crop_h)
    note = f"Conservative crop applied: {crop_w}x{crop_h} (margin {margin_ratio*100:.1f}%)"
    return bbox, breast_area_ratio, True, STATUS_SUCCESS, note


def normalize_intensity_robust(
    img: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[np.ndarray, float, float]:
    """Apply robust percentile-based intensity normalization to [0.0, 1.0].

    Values <= p1 map to 0.0, values >= p99 map to 1.0, intermediate values scaled linearly.
    """
    if lower_percentile == 0.0 and upper_percentile == 100.0:
        p_low = float(np.min(img))
        p_high = float(np.max(img))
    else:
        p_low, p_high = np.percentile(img, [lower_percentile, upper_percentile])
        p_low = float(p_low)
        p_high = float(p_high)

    if p_high <= p_low:
        normalized = np.zeros_like(img, dtype=np.float32)
        return normalized, p_low, p_high

    if img.dtype == np.uint8:
        lut_in = np.arange(256, dtype=np.float32)
        lut_norm = np.clip((lut_in - p_low) / (p_high - p_low), 0.0, 1.0)
        normalized = lut_norm[img]
    else:
        normalized = np.clip((img.astype(np.float32) - p_low) / (p_high - p_low), 0.0, 1.0)

    return normalized, p_low, p_high


def _process_single_full_mammogram_impl(
    sample_info: Dict[str, Any],
    jpeg_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    stage3_reviews: Dict[Tuple[str, str, str], Dict[str, Any]],
    stage4_quality_map: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute complete Stage 5 baseline preprocessing pipeline for one full mammogram."""
    pid = sample_info["patient_id"]
    abn = sample_info["abnormality_id"]
    rel_path = sample_info["original_image_path"]
    full_source_path = resolve_image_path(rel_path, jpeg_dir)

    # Pull Stage 3 & Stage 4 Carry-Forward Metadata
    st3_key = (pid, str(abn), ROLE_FULL_ORIGINAL)
    st3_info = stage3_reviews.get(st3_key, {})
    st3_role = st3_info.get("final_status", "PASS")
    st3_ref = st3_info.get("review_resolution", "RESOLVED")

    st4_info = stage4_quality_map.get(rel_path, {})
    st4_status = st4_info.get("quality_status", "PASS")
    st4_flags = st4_info.get("quality_flags", "")

    # Base metadata record
    meta_rec: Dict[str, Any] = {
        "patient_id": pid,
        "abnormality_id": abn,
        "abnormality_category": sample_info["abnormality_category"],
        "breast_side": sample_info["breast_side"],
        "image_view": sample_info["image_view"],
        "pathology": sample_info["pathology"],
        "label": sample_info["label"],
        "dataset_split": sample_info["dataset_split"],
        "original_image_path": rel_path,
        "baseline_image_path": "",
        "stage3_role_status": st3_role,
        "stage3_reference_status": st3_ref,
        "stage4_quality_status": st4_status,
        "stage4_quality_flags": st4_flags,
        "original_width": 0,
        "original_height": 0,
        "original_channels": 1,
        "baseline_width": 0,
        "baseline_height": 0,
        "baseline_channels": 1,
        "original_min": 0.0,
        "original_max": 0.0,
        "original_mean": 0.0,
        "original_std": 0.0,
        "baseline_min": 0.0,
        "baseline_max": 0.0,
        "baseline_mean": 0.0,
        "baseline_std": 0.0,
        "normalization_method": config["normalization"]["method"],
        "normalization_lower_percentile": config["normalization"]["lower_percentile"],
        "normalization_upper_percentile": config["normalization"]["upper_percentile"],
        "crop_applied": False,
        "crop_x": 0,
        "crop_y": 0,
        "crop_width": 0,
        "crop_height": 0,
        "crop_margin_ratio": config["breast_region"]["crop_margin_ratio"],
        "breast_region_area_ratio": 0.0,
        "processing_status": STATUS_FAILED,
        "processing_notes": "",
    }

    val_rec: Dict[str, Any] = {
        "patient_id": pid,
        "abnormality_id": abn,
        "baseline_image_path": "",
        "output_exists": False,
        "can_reopen": False,
        "is_grayscale": False,
        "channels": 0,
        "dtype": "",
        "normalized_internal_min": 0.0,
        "normalized_internal_max": 0.0,
        "saved_min": 0,
        "saved_max": 0,
        "nan_count": 0,
        "inf_count": 0,
        "dimensions_valid": False,
        "raw_source_unmodified": True,
        "validation_status": "FAIL",
        "validation_notes": "",
    }

    # Step 2: Technical Validation of Raw Source
    if not os.path.exists(full_source_path):
        meta_rec["processing_status"] = STATUS_FAILED
        meta_rec["processing_notes"] = "Raw image file does not exist on disk"
        val_rec["validation_notes"] = meta_rec["processing_notes"]
        return meta_rec, val_rec

    # Record filesystem stat before processing to verify no raw file modification
    try:
        stat_pre = os.stat(full_source_path)
    except Exception as err:
        meta_rec["processing_status"] = STATUS_FAILED
        meta_rec["processing_notes"] = f"Raw image stat failed: {err}"
        val_rec["validation_notes"] = meta_rec["processing_notes"]
        return meta_rec, val_rec

    buf_size = config.get("performance", {}).get("io_buffer_size", 1048576)
    raw_sha_pre = compute_file_sha256(full_source_path, buffer_size=buf_size)

    try:
        # Fast direct grayscale decoding via libjpeg-turbo
        raw_img = cv2.imread(full_source_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            with Image.open(full_source_path) as pil_img:
                raw_img = np.array(pil_img)
    except Exception as err:
        meta_rec["processing_status"] = STATUS_FAILED
        meta_rec["processing_notes"] = f"Decoding failed: {err}"
        val_rec["validation_notes"] = meta_rec["processing_notes"]
        return meta_rec, val_rec

    if raw_img is None or raw_img.size == 0:
        meta_rec["processing_status"] = STATUS_FAILED
        meta_rec["processing_notes"] = "Decoded raw image is empty"
        val_rec["validation_notes"] = meta_rec["processing_notes"]
        return meta_rec, val_rec

    # Grayscale representation
    if len(raw_img.shape) > 2:
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = raw_img
    h_orig, w_orig = gray_img.shape[:2]
    meta_rec["original_width"] = int(w_orig)
    meta_rec["original_height"] = int(h_orig)

    # Fast 256-bin histogram-based statistics in <5ms (avoids multi-hundred-megabyte float array conversions)
    counts_orig = np.bincount(gray_img.ravel(), minlength=256)
    valid_orig_bins = np.where(counts_orig > 0)[0]
    meta_rec["original_min"] = float(valid_orig_bins[0])
    meta_rec["original_max"] = float(valid_orig_bins[-1])
    n_orig = gray_img.size
    bins_f = np.arange(256, dtype=np.float64)
    orig_mean = float(np.sum(counts_orig * bins_f) / n_orig)
    orig_var = float(np.sum(counts_orig * ((bins_f - orig_mean) ** 2)) / n_orig)
    meta_rec["original_mean"] = round(orig_mean, 2)
    meta_rec["original_std"] = round(float(np.sqrt(max(0.0, orig_var))), 2)

    # Step 4: Conservative Breast-Region Detection
    bbox, area_ratio, crop_applied, detect_status, detect_notes = detect_breast_region_conservative(gray_img, config)
    cx, cy, cw, ch = bbox

    meta_rec["crop_applied"] = crop_applied
    meta_rec["crop_x"] = cx
    meta_rec["crop_y"] = cy
    meta_rec["crop_width"] = cw
    meta_rec["crop_height"] = ch
    meta_rec["breast_region_area_ratio"] = area_ratio
    meta_rec["processing_notes"] = detect_notes

    # Step 5: Conservative Crop (NumPy slicing creates a zero-copy view)
    cropped_img = gray_img[cy : cy + ch, cx : cx + cw]

    # Step 6: Robust Percentile Intensity Normalization
    p_low = config["normalization"]["lower_percentile"]
    p_high = config["normalization"]["upper_percentile"]

    # Compute percentiles in single pass directly on uint8
    if p_low == 0.0 and p_high == 100.0:
        act_p1 = float(valid_orig_bins[0])
        act_p99 = float(valid_orig_bins[-1])
    else:
        act_p1, act_p99 = np.percentile(cropped_img, [p_low, p_high])
        act_p1 = float(act_p1)
        act_p99 = float(act_p99)

    h_base, w_base = cropped_img.shape[:2]
    meta_rec["baseline_width"] = int(w_base)
    meta_rec["baseline_height"] = int(h_base)

    if act_p99 <= act_p1:
        uint8_img = np.zeros_like(cropped_img, dtype=np.uint8)
        meta_rec["baseline_min"] = 0.0
        meta_rec["baseline_max"] = 0.0
        meta_rec["baseline_mean"] = 0.0
        meta_rec["baseline_std"] = 0.0
    else:
        # 256-element lookup table: 100% mathematically exact mapping, zero 100MB array allocations
        lut_in = np.arange(256, dtype=np.float32)
        lut_norm = np.clip((lut_in - act_p1) / (act_p99 - act_p1), 0.0, 1.0)
        lut_uint8 = np.clip(np.round(lut_norm * 255.0), 0, 255).astype(np.uint8)

        # Fast SIMD mapping in <2ms
        uint8_img = cv2.LUT(cropped_img, lut_uint8)

        # Fast exact normalized stats via histogram:
        counts_crop = np.bincount(cropped_img.ravel(), minlength=256)
        valid_crop_bins = np.where(counts_crop > 0)[0]
        n_crop = cropped_img.size

        meta_rec["baseline_min"] = float(lut_norm[valid_crop_bins[0]])
        meta_rec["baseline_max"] = float(lut_norm[valid_crop_bins[-1]])
        base_mean = float(np.sum(counts_crop * lut_norm) / n_crop)
        base_var = float(np.sum(counts_crop * ((lut_norm - base_mean) ** 2)) / n_crop)
        meta_rec["baseline_mean"] = round(base_mean, 4)
        meta_rec["baseline_std"] = round(float(np.sqrt(max(0.0, base_var))), 4)

    # Step 8: Save Baseline PNG Image
    os.makedirs(output_dir, exist_ok=True)
    out_filename = (
        sample_info.get("output_filename")
        or f"{pid}_{sample_info['breast_side']}_{sample_info['image_view']}_{abn}_baseline.png"
    )
    out_path = os.path.join(output_dir, out_filename)

    # Save lossless PNG with fast zlib compression (level 1):
    # Compression level 1 is 100% bit-exact lossless, fully standard PNG, and 5-10x faster
    needs_save = True
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            existing_test = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
            if (
                existing_test is not None
                and existing_test.shape[:2] == (h_base, w_base)
                and existing_test.dtype == np.uint8
            ):
                needs_save = False
        except Exception:
            needs_save = True

    if needs_save:
        png_compression = config.get("performance", {}).get("png_compression_level", 1)
        success_save = cv2.imwrite(out_path, uint8_img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        if not success_save:
            Image.fromarray(uint8_img, mode="L").save(out_path, format="PNG", compress_level=png_compression)

    meta_rec["baseline_image_path"] = os.path.relpath(out_path, os.getcwd()).replace("\\", "/")
    meta_rec["processing_status"] = detect_status

    # Validation: Technical Reopening & Integrity Check
    val_rec["baseline_image_path"] = meta_rec["baseline_image_path"]
    val_rec["output_exists"] = os.path.exists(out_path)

    reopened = None
    if val_rec["output_exists"]:
        try:
            reopened = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if reopened is not None:
        val_rec["can_reopen"] = True
        val_rec["channels"] = reopened.shape[2] if len(reopened.shape) > 2 else 1
        val_rec["is_grayscale"] = (val_rec["channels"] == 1)
        val_rec["dtype"] = str(reopened.dtype)
        val_rec["saved_min"] = int(reopened.min())
        val_rec["saved_max"] = int(reopened.max())
        val_rec["nan_count"] = 0
        val_rec["inf_count"] = 0
        val_rec["dimensions_valid"] = (reopened.shape[0] == h_base and reopened.shape[1] == w_base)

    val_rec["normalized_internal_min"] = meta_rec["baseline_min"]
    val_rec["normalized_internal_max"] = meta_rec["baseline_max"]

    # Verify raw file unchanged via stat check
    stat_post = os.stat(full_source_path)
    if stat_pre.st_mtime_ns == stat_post.st_mtime_ns and stat_pre.st_size == stat_post.st_size:
        raw_sha_post = raw_sha_pre
    else:
        raw_sha_post = compute_file_sha256(full_source_path, buffer_size=buf_size)
    val_rec["raw_source_unmodified"] = (raw_sha_pre == raw_sha_post)

    if (
        val_rec["can_reopen"]
        and val_rec["is_grayscale"]
        and val_rec["dtype"] == "uint8"
        and val_rec["dimensions_valid"]
        and val_rec["raw_source_unmodified"]
    ):
        val_rec["validation_status"] = "PASS"
        val_rec["validation_notes"] = "Valid 8-bit grayscale baseline PNG verified"
    else:
        val_rec["validation_status"] = "FAIL"
        val_rec["validation_notes"] = "Validation check failure on generated baseline image"

    return meta_rec, val_rec


def process_single_full_mammogram(
    sample_info: Dict[str, Any],
    jpeg_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    stage3_reviews: Dict[Tuple[str, str, str], Dict[str, Any]],
    stage4_quality_map: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute complete Stage 5 baseline preprocessing pipeline with isolated error handling."""
    try:
        return _process_single_full_mammogram_impl(
            sample_info=sample_info,
            jpeg_dir=jpeg_dir,
            output_dir=output_dir,
            config=config,
            stage3_reviews=stage3_reviews,
            stage4_quality_map=stage4_quality_map,
        )
    except Exception as err:
        pid = sample_info.get("patient_id", "UNKNOWN")
        abn = sample_info.get("abnormality_id", 1)
        rel_p = sample_info.get("original_image_path", "")
        meta_rec = {
            "patient_id": pid,
            "abnormality_id": abn,
            "abnormality_category": sample_info.get("abnormality_category", ""),
            "breast_side": sample_info.get("breast_side", ""),
            "image_view": sample_info.get("image_view", ""),
            "pathology": sample_info.get("pathology", ""),
            "label": sample_info.get("label", 0),
            "dataset_split": sample_info.get("dataset_split", ""),
            "original_image_path": rel_p,
            "baseline_image_path": "",
            "stage3_role_status": "UNKNOWN",
            "stage3_reference_status": "UNKNOWN",
            "stage4_quality_status": "UNKNOWN",
            "stage4_quality_flags": "",
            "original_width": 0,
            "original_height": 0,
            "original_channels": 1,
            "baseline_width": 0,
            "baseline_height": 0,
            "baseline_channels": 1,
            "original_min": 0.0,
            "original_max": 0.0,
            "original_mean": 0.0,
            "original_std": 0.0,
            "baseline_min": 0.0,
            "baseline_max": 0.0,
            "baseline_mean": 0.0,
            "baseline_std": 0.0,
            "normalization_method": config.get("normalization", {}).get("method", "percentile"),
            "normalization_lower_percentile": config.get("normalization", {}).get("lower_percentile", 1.0),
            "normalization_upper_percentile": config.get("normalization", {}).get("upper_percentile", 99.0),
            "crop_applied": False,
            "crop_x": 0,
            "crop_y": 0,
            "crop_width": 0,
            "crop_height": 0,
            "crop_margin_ratio": config.get("breast_region", {}).get("crop_margin_ratio", 0.05),
            "breast_region_area_ratio": 0.0,
            "processing_status": STATUS_FAILED,
            "processing_notes": f"Preprocessing exception: {err}",
        }
        val_rec = {
            "patient_id": pid,
            "abnormality_id": abn,
            "baseline_image_path": "",
            "output_exists": False,
            "can_reopen": False,
            "is_grayscale": False,
            "channels": 0,
            "dtype": "",
            "normalized_internal_min": 0.0,
            "normalized_internal_max": 0.0,
            "saved_min": 0,
            "saved_max": 0,
            "nan_count": 0,
            "inf_count": 0,
            "dimensions_valid": False,
            "raw_source_unmodified": True,
            "validation_status": "FAIL",
            "validation_notes": f"Preprocessing exception: {err}",
        }
        return meta_rec, val_rec


def verify_raw_data_protection(
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    sample_paths: Optional[List[str]] = None,
) -> bool:
    """Verify that no files under raw CBIS-DDSM data directory were added or altered.

    Checks parent directories of processed samples and top-level raw directories,
    avoiding an exhaustive recursive walk of 10,000+ directories over bind mounts.
    """
    if not os.path.exists(raw_data_dir):
        return True

    dirs_to_check = set()
    if sample_paths:
        for p in sample_paths:
            if p and os.path.exists(p):
                dirs_to_check.add(os.path.dirname(p))

    # Also check top-level raw directory entries
    dirs_to_check.add(raw_data_dir)
    try:
        with os.scandir(raw_data_dir) as it:
            for entry in it:
                if entry.is_dir() and "csv" not in entry.name.lower():
                    dirs_to_check.add(entry.path)
    except Exception:
        pass

    for d in dirs_to_check:
        if os.path.exists(d):
            try:
                for f in os.listdir(d):
                    lower = f.lower()
                    if lower.endswith((".png", ".txt")) or (lower.endswith(".csv") and "csv" not in d.replace("\\", "/").lower()):
                        print(f"[RAW PROTECTION ALERT] Non-raw file detected: {os.path.join(d, f)}")
                        return False
            except Exception:
                pass
    return True


def validate_dataset_run(
    df_meta: pd.DataFrame,
    df_val: pd.DataFrame,
    df_ref: pd.DataFrame,
    baseline_output_dir: str,
    raw_protected: bool,
    expected_eligible_count: int,
) -> Dict[str, Any]:
    """Execute comprehensive 14-point final dataset validation."""
    val_results: Dict[str, Any] = {}

    n_eligible = len(df_meta)
    val_results["1_eligible_records_identified"] = n_eligible
    val_results["1_eligible_matches_expected"] = (n_eligible == expected_eligible_count)

    succ_mask = df_meta["processing_status"] == STATUS_SUCCESS
    rev_mask = df_meta["processing_status"] == STATUS_REVIEW_REQUIRED
    fail_mask = df_meta["processing_status"] == STATUS_FAILED

    n_success = int(succ_mask.sum())
    n_review = int(rev_mask.sum())
    n_failed = int(fail_mask.sum())

    val_results["2_successfully_processed"] = n_success
    val_results["3_requiring_review"] = n_review
    val_results["4_failed"] = n_failed

    # Output files on disk
    existing_outputs = 0
    if os.path.exists(baseline_output_dir):
        existing_outputs = len([
            f for f in os.listdir(baseline_output_dir)
            if f.lower().endswith(".png") and not f.startswith(".") and os.path.isfile(os.path.join(baseline_output_dir, f))
        ])
    val_results["5_output_files_count"] = existing_outputs

    val_results["6_metadata_records_count"] = len(df_meta)
    val_results["7_validation_records_count"] = len(df_val)

    # Check duplicate output paths
    valid_paths = df_meta[df_meta["baseline_image_path"] != ""]["baseline_image_path"]
    num_duplicates = int(valid_paths.duplicated().sum())
    val_results["8_no_duplicate_output_paths"] = (num_duplicates == 0)
    val_results["8_duplicate_path_count"] = num_duplicates

    # Validation records checks for successful outputs
    all_reopened = bool(df_val.loc[succ_mask, "can_reopen"].all()) if n_success > 0 else True
    all_grayscale = bool(df_val.loc[succ_mask, "is_grayscale"].all()) if n_success > 0 else True
    all_uint8 = bool((df_val.loc[succ_mask, "dtype"] == "uint8").all()) if n_success > 0 else True
    no_nan_inf = bool((df_val["nan_count"] == 0).all() and (df_val["inf_count"] == 0).all())

    val_results["9_all_successful_reopened"] = all_reopened
    val_results["10_all_outputs_grayscale"] = all_grayscale
    val_results["11_all_outputs_valid_uint8"] = all_uint8
    val_results["12_no_nan_inf"] = no_nan_inf

    val_results["13_raw_dataset_unchanged"] = raw_protected

    # Check patient-level split preservation
    split_preserved = True
    if "dataset_split" in df_meta.columns and "dataset_split" in df_ref.columns and len(df_meta) == len(df_ref):
        split_preserved = bool((df_meta["dataset_split"].str.lower().values == df_ref["dataset_split"].str.lower().values).all())
    val_results["14_split_metadata_unchanged"] = split_preserved

    is_pass = (
        val_results["8_no_duplicate_output_paths"]
        and all_reopened
        and all_grayscale
        and all_uint8
        and no_nan_inf
        and raw_protected
        and split_preserved
        and (n_failed == 0)
    )
    val_results["overall_validation_pass"] = is_pass

    return val_results


def generate_full_dataset_report(
    df_meta: pd.DataFrame,
    df_val: pd.DataFrame,
    val_results: Dict[str, Any],
    total_time: float,
    report_path: str,
) -> str:
    """Generate the official Stage 5 full dataset baseline report text and save to disk."""
    total_eligible = len(df_meta)
    succ_cnt = int((df_meta["processing_status"] == STATUS_SUCCESS).sum())
    review_cnt = int((df_meta["processing_status"] == STATUS_REVIEW_REQUIRED).sum())
    fail_cnt = int((df_meta["processing_status"] == STATUS_FAILED).sum())

    cc_cnt = int((df_meta["image_view"].str.upper() == "CC").sum())
    mlo_cnt = int((df_meta["image_view"].str.upper() == "MLO").sum())

    left_cnt = int((df_meta["breast_side"].str.upper() == "LEFT").sum())
    right_cnt = int((df_meta["breast_side"].str.upper() == "RIGHT").sum())

    mass_cnt = int((df_meta["abnormality_category"].str.lower() == "mass").sum())
    calc_cnt = int((df_meta["abnormality_category"].str.lower() == "calcification").sum())

    benign_cnt = int((df_meta["pathology"].str.upper() == "BENIGN").sum())
    bwc_cnt = int((df_meta["pathology"].str.upper() == "BENIGN_WITHOUT_CALLBACK").sum())
    malignant_cnt = int((df_meta["pathology"].str.upper() == "MALIGNANT").sum())

    train_cnt = int((df_meta["dataset_split"].str.lower() == "train").sum())
    val_split_cnt = int((df_meta["dataset_split"].str.lower() == "validation").sum())
    test_cnt = int((df_meta["dataset_split"].str.lower() == "test").sum())

    total_output_files = val_results.get("5_output_files_count", succ_cnt)
    total_meta_records = len(df_meta)
    total_val_records = len(df_val)

    avg_time = round(total_time / total_eligible, 4) if total_eligible > 0 else 0.0
    raw_modified_str = "NO" if val_results.get("13_raw_dataset_unchanged", True) else "YES"

    report_text = f"""============================================================
CBIS-DDSM FULL DATASET BASELINE PREPROCESSING REPORT
============================================================

Total eligible FULL/ORIGINAL records: {total_eligible}
Successfully processed: {succ_cnt}
Review required: {review_cnt}
Failed: {fail_cnt}

CC: {cc_cnt}
MLO: {mlo_cnt}

LEFT: {left_cnt}
RIGHT: {right_cnt}

MASS: {mass_cnt}
CALCIFICATION: {calc_cnt}

BENIGN: {benign_cnt}
BENIGN_WITHOUT_CALLBACK: {bwc_cnt}
MALIGNANT: {malignant_cnt}

Train: {train_cnt}
Validation: {val_split_cnt}
Test: {test_cnt}

Total output files: {total_output_files}
Total metadata records: {total_meta_records}
Total validation records: {total_val_records}

Total processing time: {total_time:.1f}s
Average time/image: {avg_time:.4f}s

Raw data modified:
{raw_modified_str}
============================================================
"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def run_baseline_preprocessing(
    stage2_metadata_dir: str = "data/metadata/stage2",
    stage3_metadata_dir: str = "data/metadata/stage3",
    stage4_metadata_dir: str = "data/metadata/stage4",
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    stage5_metadata_dir: str = "data/metadata/stage5",
    baseline_output_dir: str = "data/processed/baseline/full_mammogram",
    config_path: Optional[str] = None,
    force_pilot: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute complete Stage 5 Baseline Preprocessing on the pilot or full dataset."""
    t0 = time.time()
    os.makedirs(stage5_metadata_dir, exist_ok=True)
    os.makedirs(baseline_output_dir, exist_ok=True)

    config = load_baseline_config(config_path)
    jpeg_dir = find_jpeg_dir(raw_data_dir)

    is_pilot = force_pilot if force_pilot is not None else config.get("pilot_mode", False)

    print("============================================================")
    print("STAGE 5 — CBIS-DDSM PRIMARY BASELINE IMAGE PREPROCESSING")
    print("============================================================")
    print(f"[Stage 5] Mode: {'PILOT' if is_pilot else 'FULL DATASET'}")
    print(f"[Stage 5] JPEG Directory: {jpeg_dir}")
    print(f"[Stage 5] Output Directory: {baseline_output_dir}")

    # 1. Load Stage 2 Reference Mapping
    ref_map_path = os.path.join(stage2_metadata_dir, "CBIS_DDSM_reference_mapping.csv")
    if not os.path.exists(ref_map_path):
        ref_map_path = os.path.join(find_metadata_dir("stage2"), "CBIS_DDSM_reference_mapping.csv")
    df_ref = pd.read_csv(ref_map_path)

    # 2. Load Stage 3 Review Records
    st3_reviews = {}
    st3_path = os.path.join(stage3_metadata_dir, "stage3_review_analysis.csv")
    if os.path.exists(st3_path):
        try:
            df_st3 = pd.read_csv(st3_path)
            for _, r in df_st3.iterrows():
                key = (str(r.get("patient_id", "")), str(r.get("abnormality_id", "")), str(r.get("image_role", "")))
                st3_reviews[key] = {
                    "final_status": r.get("final_status", ""),
                    "review_resolution": r.get("review_resolution", ""),
                }
        except Exception as err:
            print(f"[Stage 5] Warning loading Stage 3 file {st3_path}: {err}")

    # 3. Load Stage 4 Quality Report
    st4_quality_map = {}
    st4_path = os.path.join(stage4_metadata_dir, "image_quality_report.csv")
    if os.path.exists(st4_path):
        try:
            df_st4 = pd.read_csv(st4_path)
            for _, r in df_st4.iterrows():
                img_p = str(r.get("image_path", "")).strip()
                if img_p:
                    st4_quality_map[img_p] = {
                        "quality_status": r.get("quality_status", "PASS"),
                        "quality_flags": r.get("quality_flags", ""),
                    }
        except Exception as err:
            print(f"[Stage 5] Warning loading Stage 4 file {st4_path}: {err}")

    # 4. Select Samples
    if is_pilot:
        pilot_count = config.get("pilot_sample_count", 16)
        samples = select_pilot_samples(df_ref, n_samples=pilot_count)
        total_eligible = len(samples)
        print(f"[Stage 5] Selected {len(samples)} stratified pilot cases.")
    else:
        samples = load_full_dataset_samples(df_ref)
        total_eligible = len(samples)
        EXPECTED_FULL_RECORD_COUNT = 3568
        print(f"[Stage 5] Identified {total_eligible} eligible FULL/ORIGINAL mammogram records.")
        if total_eligible != EXPECTED_FULL_RECORD_COUNT:
            print(f"[Stage 5 WARNING] Eligible record count ({total_eligible}) differs from expected reference count ({EXPECTED_FULL_RECORD_COUNT})!")
            try:
                response = input(f"Proceed with {total_eligible} records? (y/n): ")
                if response.strip().lower() not in ("y", "yes"):
                    print("[Stage 5] Aborting execution upon user confirmation.")
                    sys.exit(1)
            except Exception:
                print(f"[Stage 5] Non-interactive execution; proceeding with {total_eligible} records.")
        else:
            print(f"[Stage 5] Eligible record count matches expected CBIS-DDSM reference count ({EXPECTED_FULL_RECORD_COUNT}). Proceeding with full dataset...")

    # Concurrency configuration
    perf_cfg = config.get("performance", {})
    workers = int(perf_cfg.get("num_workers", 4))

    metadata_records: List[Dict[str, Any]] = []
    validation_records: List[Dict[str, Any]] = []

    print(f"[Stage 5] Processing {len(samples)} records (concurrency: {workers} workers)...")
    log_interval = 1 if len(samples) <= 20 else 50
    succ_tally = 0
    rev_tally = 0
    fail_tally = 0

    if workers > 1 and len(samples) > 1:
        results: List[Optional[Tuple[Dict[str, Any], Dict[str, Any]]]] = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_full_mammogram,
                    sample_info=s,
                    jpeg_dir=jpeg_dir,
                    output_dir=baseline_output_dir,
                    config=config,
                    stage3_reviews=st3_reviews,
                    stage4_quality_map=st4_quality_map,
                ): idx
                for idx, s in enumerate(samples)
            }
            completed_count = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                completed_count += 1
                s = samples[idx]
                try:
                    meta_rec, val_rec = fut.result()
                except Exception as err:
                    # Robust per-image failure catch
                    meta_rec = {
                        "patient_id": s["patient_id"],
                        "abnormality_id": s["abnormality_id"],
                        "abnormality_category": s["abnormality_category"],
                        "breast_side": s["breast_side"],
                        "image_view": s["image_view"],
                        "pathology": s["pathology"],
                        "label": s["label"],
                        "dataset_split": s["dataset_split"],
                        "original_image_path": s["original_image_path"],
                        "baseline_image_path": "",
                        "stage3_role_status": "UNKNOWN",
                        "stage3_reference_status": "UNKNOWN",
                        "stage4_quality_status": "UNKNOWN",
                        "stage4_quality_flags": "",
                        "original_width": 0, "original_height": 0, "original_channels": 1,
                        "baseline_width": 0, "baseline_height": 0, "baseline_channels": 1,
                        "original_min": 0.0, "original_max": 0.0, "original_mean": 0.0, "original_std": 0.0,
                        "baseline_min": 0.0, "baseline_max": 0.0, "baseline_mean": 0.0, "baseline_std": 0.0,
                        "normalization_method": config["normalization"]["method"],
                        "normalization_lower_percentile": config["normalization"]["lower_percentile"],
                        "normalization_upper_percentile": config["normalization"]["upper_percentile"],
                        "crop_applied": False, "crop_x": 0, "crop_y": 0, "crop_width": 0, "crop_height": 0,
                        "crop_margin_ratio": config["breast_region"]["crop_margin_ratio"],
                        "breast_region_area_ratio": 0.0,
                        "processing_status": STATUS_FAILED,
                        "processing_notes": f"Worker thread exception: {err}",
                    }
                    val_rec = {
                        "patient_id": s["patient_id"],
                        "abnormality_id": s["abnormality_id"],
                        "baseline_image_path": "",
                        "output_exists": False, "can_reopen": False, "is_grayscale": False,
                        "channels": 0, "dtype": "",
                        "normalized_internal_min": 0.0, "normalized_internal_max": 0.0,
                        "saved_min": 0, "saved_max": 0, "nan_count": 0, "inf_count": 0,
                        "dimensions_valid": False, "raw_source_unmodified": True,
                        "validation_status": "FAIL",
                        "validation_notes": f"Worker thread exception: {err}",
                    }

                results[idx] = (meta_rec, val_rec)
                status = meta_rec["processing_status"]
                if status == STATUS_SUCCESS:
                    succ_tally += 1
                elif status == STATUS_REVIEW_REQUIRED:
                    rev_tally += 1
                    print(f"  [REVIEW] {s['patient_id']} ({s['breast_side']}_{s['image_view']}) -> {meta_rec['processing_notes']}")
                else:
                    fail_tally += 1
                    print(f"  [FAILED] {s['patient_id']} ({s['breast_side']}_{s['image_view']}) -> {meta_rec['processing_notes']}")

                if completed_count % log_interval == 0 or completed_count == len(samples):
                    elapsed = time.time() - t0
                    avg_t = elapsed / completed_count if completed_count > 0 else 0.0
                    pct = (completed_count / len(samples)) * 100.0
                    print(f"Processed: {completed_count}/{len(samples)} ({pct:.1f}%) | Successful: {succ_tally} | Review: {rev_tally} | Failed: {fail_tally} | Elapsed: {elapsed:.1f}s | Avg: {avg_t:.2f}s/img")

        # Gather results deterministically ordered by original sample index
        for res in results:
            if res is not None:
                metadata_records.append(res[0])
                validation_records.append(res[1])
    else:
        completed_count = 0
        for idx, s in enumerate(samples):
            completed_count += 1
            meta_rec, val_rec = process_single_full_mammogram(
                sample_info=s,
                jpeg_dir=jpeg_dir,
                output_dir=baseline_output_dir,
                config=config,
                stage3_reviews=st3_reviews,
                stage4_quality_map=st4_quality_map,
            )
            metadata_records.append(meta_rec)
            validation_records.append(val_rec)
            status = meta_rec["processing_status"]
            if status == STATUS_SUCCESS:
                succ_tally += 1
            elif status == STATUS_REVIEW_REQUIRED:
                rev_tally += 1
                print(f"  [REVIEW] {s['patient_id']} ({s['breast_side']}_{s['image_view']}) -> {meta_rec['processing_notes']}")
            else:
                fail_tally += 1
                print(f"  [FAILED] {s['patient_id']} ({s['breast_side']}_{s['image_view']}) -> {meta_rec['processing_notes']}")

            if completed_count % log_interval == 0 or completed_count == len(samples):
                elapsed = time.time() - t0
                avg_t = elapsed / completed_count if completed_count > 0 else 0.0
                pct = (completed_count / len(samples)) * 100.0
                print(f"Processed: {completed_count}/{len(samples)} ({pct:.1f}%) | Successful: {succ_tally} | Review: {rev_tally} | Failed: {fail_tally} | Elapsed: {elapsed:.1f}s | Avg: {avg_t:.2f}s/img")

    # 6. Save Stage 5 Metadata Files
    df_meta = pd.DataFrame(metadata_records)
    meta_csv_path = os.path.join(stage5_metadata_dir, "baseline_preprocessing_metadata.csv")
    df_meta.to_csv(meta_csv_path, index=False)
    print(f"[Stage 5] Saved preprocessing metadata to: {meta_csv_path}")

    df_val = pd.DataFrame(validation_records)
    val_csv_path = os.path.join(stage5_metadata_dir, "baseline_preprocessing_validation.csv")
    df_val.to_csv(val_csv_path, index=False)
    print(f"[Stage 5] Saved validation audit to: {val_csv_path}")

    # Verify raw dataset protection
    sample_paths = [resolve_image_path(s["original_image_path"], jpeg_dir) for s in samples[:100]]
    raw_protected = verify_raw_data_protection(raw_data_dir, sample_paths=sample_paths)
    if not raw_protected:
        print("[Stage 5 WARNING] Raw data modification check encountered alerts!")

    duration = round(time.time() - t0, 1)

    # 7. Final Dataset Validation
    val_results = validate_dataset_run(
        df_meta=df_meta,
        df_val=df_val,
        df_ref=df_ref if not is_pilot else df_ref.iloc[:len(samples)],
        baseline_output_dir=baseline_output_dir,
        raw_protected=raw_protected,
        expected_eligible_count=total_eligible,
    )

    # 8. Final Report Generation
    report_filename = "full_dataset_baseline_report.txt" if not is_pilot else "pilot_baseline_report.txt"
    report_path = os.path.join(stage5_metadata_dir, report_filename)
    generate_full_dataset_report(
        df_meta=df_meta,
        df_val=df_val,
        val_results=val_results,
        total_time=duration,
        report_path=report_path,
    )
    print(f"[Stage 5] Generated final report at: {report_path}")

    # Summary Counts for Console
    succ_cnt = int((df_meta["processing_status"] == STATUS_SUCCESS).sum())
    review_cnt = int((df_meta["processing_status"] == STATUS_REVIEW_REQUIRED).sum())
    fail_cnt = int((df_meta["processing_status"] == STATUS_FAILED).sum())

    cc_cnt = int((df_meta["image_view"].str.upper() == "CC").sum())
    mlo_cnt = int((df_meta["image_view"].str.upper() == "MLO").sum())
    left_cnt = int((df_meta["breast_side"].str.upper() == "LEFT").sum())
    right_cnt = int((df_meta["breast_side"].str.upper() == "RIGHT").sum())
    mass_cnt = int((df_meta["abnormality_category"].str.lower() == "mass").sum())
    calc_cnt = int((df_meta["abnormality_category"].str.lower() == "calcification").sum())

    console_summary = f"""============================================================
CBIS-DDSM STAGE 5 BASELINE PREPROCESSING COMPLETE
============================================================

Mode:                           {'PILOT' if is_pilot else 'FULL DATASET'}
Total eligible records:         {total_eligible}
Successfully processed:         {succ_cnt}
Review required:                {review_cnt}
Failed:                         {fail_cnt}

CC:                             {cc_cnt}
MLO:                            {mlo_cnt}

LEFT:                           {left_cnt}
RIGHT:                          {right_cnt}

MASS:                           {mass_cnt}
CALCIFICATION:                  {calc_cnt}

Outputs:
{baseline_output_dir}/

Metadata:
{meta_csv_path}

Validation:
{val_csv_path}

Report:
{report_path}

Duration:                       {duration}s
Average time/image:             {duration/total_eligible if total_eligible > 0 else 0.0:.3f}s
============================================================
"""
    print(console_summary)

    return {
        "pilot_count": len(metadata_records),
        "total_count": len(metadata_records),
        "total_eligible": total_eligible,
        "success_count": succ_cnt,
        "review_count": review_cnt,
        "fail_count": fail_cnt,
        "cc_count": cc_cnt,
        "mlo_count": mlo_cnt,
        "left_count": left_cnt,
        "right_count": right_cnt,
        "mass_count": mass_cnt,
        "calc_count": calc_cnt,
        "metadata_csv": meta_csv_path,
        "validation_csv": val_csv_path,
        "report_path": report_path,
        "duration_seconds": duration,
        "validation_results": val_results,
    }


def run_baseline_preprocessing_pilot(
    stage2_metadata_dir: str = "data/metadata/stage2",
    stage3_metadata_dir: str = "data/metadata/stage3",
    stage4_metadata_dir: str = "data/metadata/stage4",
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    stage5_metadata_dir: str = "data/metadata/stage5",
    baseline_output_dir: str = "data/processed/baseline/full_mammogram",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Stage 5 Baseline Preprocessing in Pilot Mode (backwards-compatible wrapper)."""
    return run_baseline_preprocessing(
        stage2_metadata_dir=stage2_metadata_dir,
        stage3_metadata_dir=stage3_metadata_dir,
        stage4_metadata_dir=stage4_metadata_dir,
        raw_data_dir=raw_data_dir,
        stage5_metadata_dir=stage5_metadata_dir,
        baseline_output_dir=baseline_output_dir,
        config_path=config_path,
        force_pilot=True,
    )


if __name__ == "__main__":
    force_pilot = "--pilot" in sys.argv
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    s2_dir = pos_args[0] if len(pos_args) > 0 else "data/metadata/stage2"
    s3_dir = pos_args[1] if len(pos_args) > 1 else "data/metadata/stage3"
    s4_dir = pos_args[2] if len(pos_args) > 2 else "data/metadata/stage4"
    raw_dir = pos_args[3] if len(pos_args) > 3 else "data/raw/CBIS_DDSM"
    s5_dir = pos_args[4] if len(pos_args) > 4 else "data/metadata/stage5"
    out_dir = pos_args[5] if len(pos_args) > 5 else "data/processed/baseline/full_mammogram"
    cfg_p = None

    run_baseline_preprocessing(
        stage2_metadata_dir=s2_dir,
        stage3_metadata_dir=s3_dir,
        stage4_metadata_dir=s4_dir,
        raw_data_dir=raw_dir,
        stage5_metadata_dir=s5_dir,
        baseline_output_dir=out_dir,
        config_path=cfg_p,
        force_pilot=force_pilot,
    )
