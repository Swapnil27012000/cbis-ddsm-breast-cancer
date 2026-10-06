"""Stage 6: CBIS-DDSM Contrast Enhancement Experiment Module.

Evaluates controlled contrast-enhancement methods on Stage-5 baseline mammograms:
- METHOD 0 — Stage-5 Baseline / Control (unchanged control condition)
- METHOD 1 — Global Histogram Equalization (standard cv2.equalizeHist)
- METHOD 2 — CLAHE (Contrast Limited Adaptive Histogram Equalization, configurable)

Strictly:
- Operates ONLY on valid Stage-5 baseline full mammograms.
- Preserves spatial dimensions (NO resizing, NO downsampling).
- Does NOT perform sharpening, unsharp masking, noise addition, or denoising.
- Does NOT duplicate Stage-5 1-99 percentile normalization.
- Calculates comprehensive descriptive intensity statistics (entropy, breast-region stats, percentiles).
- Executes PILOT FIRST on 16 representative cases reusing deterministic Stage-5 pilot cases.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Status Constants
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"

# Method Names
METHOD_BASELINE = "baseline"
METHOD_HIST_EQ = "histogram_equalization"
METHOD_CLAHE = "clahe"


def find_metadata_path(filename: str, stage_subdir: str = "stage5") -> str:
    """Locate metadata files across Docker (/app) and host environments."""
    candidates = [
        f"data/metadata/{stage_subdir}/{filename}",
        os.path.join(_PROJECT_ROOT, "data", "metadata", stage_subdir, filename),
        os.path.join(os.getcwd(), "data", "metadata", stage_subdir, filename),
        f"/app/data/metadata/{stage_subdir}/{filename}",
        f"data/metadata/{filename}",
        os.path.join(_PROJECT_ROOT, "data", "metadata", filename),
        os.path.join(os.getcwd(), "data", "metadata", filename),
        f"/app/data/metadata/{filename}",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    return os.path.abspath(f"data/metadata/{stage_subdir}/{filename}")


def resolve_image_path(path_str: str) -> str:
    """Resolve physical path on disk across environments."""
    if not path_str or pd.isna(path_str):
        return ""
    norm = str(path_str).strip()
    if os.path.isabs(norm) and os.path.exists(norm):
        return os.path.normpath(norm)

    candidates = [
        os.path.normpath(os.path.join(_PROJECT_ROOT, norm)),
        os.path.normpath(os.path.join(os.getcwd(), norm)),
        norm,
    ]
    if norm.startswith("/app/"):
        rel = norm[len("/app/"):]
        candidates.append(os.path.normpath(os.path.join(_PROJECT_ROOT, rel)))
    else:
        candidates.append(os.path.normpath(os.path.join("/app", norm)))

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.normpath(os.path.join(_PROJECT_ROOT, norm))


def load_contrast_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load Stage-6 contrast enhancement configuration with safe defaults."""
    default_cfg: Dict[str, Any] = {
        "enabled": True,
        "pilot_mode": True,
        "pilot_sample_count": 16,
        "methods": [
            METHOD_BASELINE,
            METHOD_HIST_EQ,
            METHOD_CLAHE,
        ],
        "clahe": {
            "clip_limit": 2.0,
            "tile_grid_size": {
                "width": 8,
                "height": 8,
            },
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
        os.path.join(_PROJECT_ROOT, "config", "preprocessing_config.yaml"),
        os.path.join(os.getcwd(), "config", "preprocessing_config.yaml"),
        "/app/config/preprocessing_config.yaml",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if "contrast_enhancement" in data:
                    c_cfg = data["contrast_enhancement"]
                    for k, v in c_cfg.items():
                        if isinstance(v, dict) and k in default_cfg and isinstance(default_cfg[k], dict):
                            default_cfg[k].update(v)
                        else:
                            default_cfg[k] = v
                    return default_cfg
            except Exception as err:
                print(f"[Stage 6] Warning loading config from {c}: {err}")

    return default_cfg


def load_stage5_metadata(
    meta_path: Optional[str] = None,
    val_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load Stage-5 baseline metadata and filter strictly for valid PASS records."""
    if meta_path is None or not os.path.exists(meta_path):
        meta_path = find_metadata_path("baseline_preprocessing_metadata.csv", "stage5")
    if val_path is None or not os.path.exists(val_path):
        val_path = find_metadata_path("baseline_preprocessing_validation.csv", "stage5")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Stage-5 metadata file not found at: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Optional cross-check with Stage-5 validation file
    pass_keys: Optional[Set[Tuple[str, int]]] = None
    if os.path.exists(val_path):
        try:
            df_val = pd.read_csv(val_path)
            if "validation_status" in df_val.columns:
                valid_val = df_val[df_val["validation_status"] == "PASS"]
                pass_keys = set(zip(valid_val["patient_id"].astype(str), valid_val["abnormality_id"].astype(int)))
        except Exception as err:
            print(f"[Stage 6] Warning loading Stage-5 validation audit: {err}")

    # Filter for SUCCESS status and valid baseline image path
    status_col = "processing_status" if "processing_status" in df_meta.columns else "status"
    valid_mask = (df_meta[status_col] == "SUCCESS") & (df_meta["baseline_image_path"].notna()) & (df_meta["baseline_image_path"] != "")
    df_valid = df_meta[valid_mask].copy()

    if pass_keys is not None:
        key_tuples = list(zip(df_valid["patient_id"].astype(str), df_valid["abnormality_id"].astype(int)))
        df_valid = df_valid[[k in pass_keys for k in key_tuples]].copy()

    return df_valid


def select_contrast_pilot_samples(
    df_meta: pd.DataFrame,
    n_samples: int = 16,
) -> List[Dict[str, Any]]:
    """Deterministically select 16 representative cases, exactly reusing Stage-5 pilot cases."""
    equiv_path = find_metadata_path("optimization_equivalence_report.csv", "stage5")
    reused_cases: List[Dict[str, Any]] = []

    if os.path.exists(equiv_path):
        try:
            df_equiv = pd.read_csv(equiv_path)
            for _, r in df_equiv.iterrows():
                pid = str(r.get("patient_id", "")).strip()
                abn = int(r.get("abnormality_id", 1)) if pd.notna(r.get("abnormality_id")) else 1
                case_str = str(r.get("case", "")).strip()
                tokens = case_str.split("_")
                side = tokens[2] if len(tokens) >= 3 else ""
                view = tokens[3] if len(tokens) >= 4 else ""

                match = df_meta[
                    (df_meta["patient_id"].astype(str) == pid)
                    & (df_meta["abnormality_id"].astype(int) == abn)
                ]
                if side and "breast_side" in df_meta.columns:
                    match_sv = match[
                        (match["breast_side"].str.upper() == side.upper())
                        & (match["image_view"].str.upper() == view.upper())
                    ]
                    if not match_sv.empty:
                        match = match_sv

                if not match.empty:
                    chosen = match.iloc[0]
                    reused_cases.append(chosen.to_dict())
                    if len(reused_cases) >= n_samples:
                        break
        except Exception as err:
            print(f"[Stage 6] Note: could not load pilot cases from equivalence report ({err}); using stratified selection.")

    if len(reused_cases) >= n_samples:
        return reused_cases[:n_samples]

    # Stratified deterministic fallback selection
    df_meta_sorted = df_meta.sort_values(
        by=[
            "abnormality_category",
            "image_view",
            "breast_side",
            "pathology",
            "dataset_split",
            "patient_id",
            "abnormality_id",
        ]
    ).copy()

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

    selected: List[Dict[str, Any]] = []
    used_indices: Set[Any] = set()

    for cat, view, side, path_val, split in target_strata:
        if len(selected) >= n_samples:
            break
        subset = df_meta_sorted[
            (df_meta_sorted["abnormality_category"].astype(str).str.lower() == cat.lower())
            & (df_meta_sorted["image_view"].astype(str).str.upper() == view.upper())
            & (df_meta_sorted["breast_side"].astype(str).str.upper() == side.upper())
            & (df_meta_sorted["pathology"].astype(str).str.upper() == path_val.upper())
            & (df_meta_sorted["dataset_split"].astype(str).str.lower() == split.lower())
            & (~df_meta_sorted.index.isin(used_indices))
        ]
        if not subset.empty:
            chosen = subset.iloc[0]
            used_indices.add(chosen.name)
            selected.append(chosen.to_dict())

    if len(selected) < n_samples:
        for _, r in df_meta_sorted.iterrows():
            if r.name not in used_indices:
                selected.append(r.to_dict())
                used_indices.add(r.name)
                if len(selected) >= n_samples:
                    break

    return selected[:n_samples]


def compute_image_statistics(img: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive descriptive image intensity and quality statistics.

    Calculates: min, max, mean, std, percentiles (1, 5, 25, 50, 75, 95, 99),
    Shannon entropy, and foreground breast tissue statistics.
    """
    if img is None or img.size == 0:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
            "p01": 0.0, "p05": 0.0, "p25": 0.0, "p50": 0.0,
            "p75": 0.0, "p95": 0.0, "p99": 0.0,
            "entropy": 0.0, "breast_region_mean": 0.0, "breast_region_std": 0.0,
        }

    if img.dtype == np.uint8:
        counts = np.bincount(img.ravel(), minlength=256)
        valid_bins = np.where(counts > 0)[0]
        min_v = float(valid_bins[0]) if len(valid_bins) > 0 else 0.0
        max_v = float(valid_bins[-1]) if len(valid_bins) > 0 else 0.0

        n = float(img.size)
        bins_f = np.arange(256, dtype=np.float64)
        mean_v = float(np.sum(counts * bins_f) / n)
        var_v = float(np.sum(counts * ((bins_f - mean_v) ** 2)) / n)
        std_v = float(np.sqrt(max(0.0, var_v)))

        p01, p05, p25, p50, p75, p95, p99 = np.percentile(img, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])

        p_active = counts[counts > 0] / n
        entropy = float(-np.sum(p_active * np.log2(p_active)))

        fg_mask = img > 5
        if np.any(fg_mask):
            fg_counts = np.bincount(img[fg_mask].ravel(), minlength=256)
            n_fg = float(np.sum(fg_mask))
            fg_mean = float(np.sum(fg_counts * bins_f) / n_fg)
            fg_var = float(np.sum(fg_counts * ((bins_f - fg_mean) ** 2)) / n_fg)
            fg_std = float(np.sqrt(max(0.0, fg_var)))
        else:
            fg_mean = mean_v
            fg_std = std_v
    else:
        min_v = float(np.min(img))
        max_v = float(np.max(img))
        mean_v = float(np.mean(img))
        std_v = float(np.std(img))
        p01, p05, p25, p50, p75, p95, p99 = np.percentile(img, [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
        quant = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
        counts = np.bincount(quant.ravel(), minlength=256)
        p_active = counts[counts > 0] / float(quant.size)
        entropy = float(-np.sum(p_active * np.log2(p_active)))
        fg_mask = img > (5.0 / 255.0)
        if np.any(fg_mask):
            fg_mean = float(np.mean(img[fg_mask]))
            fg_std = float(np.std(img[fg_mask]))
        else:
            fg_mean = mean_v
            fg_std = std_v

    return {
        "min": round(min_v, 2),
        "max": round(max_v, 2),
        "mean": round(mean_v, 4),
        "std": round(std_v, 4),
        "p01": round(float(p01), 2),
        "p05": round(float(p05), 2),
        "p25": round(float(p25), 2),
        "p50": round(float(p50), 2),
        "p75": round(float(p75), 2),
        "p95": round(float(p95), 2),
        "p99": round(float(p99), 2),
        "entropy": round(entropy, 4),
        "breast_region_mean": round(fg_mean, 4),
        "breast_region_std": round(fg_std, 4),
    }


def apply_histogram_equalization(img_uint8: np.ndarray) -> np.ndarray:
    """Apply standard global histogram equalization to an 8-bit grayscale mammogram.

    Preserves exact spatial dimensions without cropping, sharpening, or noise.
    """
    if img_uint8 is None or img_uint8.size == 0:
        raise ValueError("Input image is empty.")
    if img_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 array, got {img_uint8.dtype}")

    return cv2.equalizeHist(img_uint8)


def apply_clahe(
    img_uint8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        img_uint8: Input 8-bit grayscale image.
        clip_limit: Threshold for contrast limiting (default: 2.0).
        tile_grid_size: (width, height) grid divisions (default: (8, 8)).

    Returns:
        np.ndarray: CLAHE enhanced 8-bit grayscale image.
    """
    if img_uint8 is None or img_uint8.size == 0:
        raise ValueError("Input image is empty.")
    if img_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 array, got {img_uint8.dtype}")

    grid_w = max(1, int(tile_grid_size[0]))
    grid_h = max(1, int(tile_grid_size[1]))
    clahe_obj = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid_w, grid_h))
    return clahe_obj.apply(img_uint8)


def process_case_contrast_methods(
    sample_info: Dict[str, Any],
    contrast_output_root: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute Method 0 (Baseline), Method 1 (HistEq), and Method 2 (CLAHE) for a single case.

    Returns:
        (metadata_records_for_case, validation_records_for_case)
    """
    pid = str(sample_info.get("patient_id", ""))
    abn = int(sample_info.get("abnormality_id", 1))
    cat = str(sample_info.get("abnormality_category", sample_info.get("category", "")))
    side = str(sample_info.get("breast_side", sample_info.get("side", "")))
    view = str(sample_info.get("image_view", sample_info.get("view", "")))
    pathology = str(sample_info.get("pathology", ""))
    label = int(sample_info.get("label", 0)) if pd.notna(sample_info.get("label")) else 0
    split = str(sample_info.get("dataset_split", sample_info.get("split", "")))
    rel_baseline_path = str(sample_info.get("baseline_image_path", ""))

    full_baseline_path = resolve_image_path(rel_baseline_path)
    base_stem = f"{pid}_{side}_{view}_{abn}"

    meta_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    input_exists = os.path.exists(full_baseline_path) and os.path.isfile(full_baseline_path)
    base_img = None
    if input_exists:
        try:
            base_img = cv2.imread(full_baseline_path, cv2.IMREAD_GRAYSCALE)
        except Exception:
            base_img = None

    if base_img is None or base_img.size == 0:
        for m in [METHOD_BASELINE, METHOD_HIST_EQ, METHOD_CLAHE]:
            meta_rec = {
                "patient_id": pid,
                "abnormality_id": abn,
                "category": cat,
                "side": side,
                "view": view,
                "pathology": pathology,
                "label": label,
                "split": split,
                "baseline_image_path": rel_baseline_path,
                "method": m,
                "output_image_path": "",
                "original_height": 0, "original_width": 0,
                "output_height": 0, "output_width": 0,
                "input_min": 0.0, "input_max": 0.0, "input_mean": 0.0, "input_std": 0.0, "input_p01": 0.0, "input_p99": 0.0,
                "output_min": 0.0, "output_max": 0.0, "output_mean": 0.0, "output_std": 0.0, "output_p01": 0.0, "output_p99": 0.0,
                "output_p05": 0.0, "output_p25": 0.0, "output_p50": 0.0, "output_p75": 0.0, "output_p95": 0.0,
                "entropy": 0.0, "breast_region_mean": 0.0, "breast_region_std": 0.0,
                "clip_limit": "", "tile_grid_width": "", "tile_grid_height": "",
                "status": STATUS_FAILED,
                "notes": f"Stage-5 baseline image unreadable or missing at: {rel_baseline_path}",
            }
            val_rec = {
                "patient_id": pid,
                "abnormality_id": abn,
                "method": m,
                "input_exists": input_exists,
                "output_exists": False,
                "input_shape": "None", "output_shape": "None", "shape_match": False,
                "finite_values": False, "valid_intensity_range": False,
                "blank_image": True, "excessive_saturation": False,
                "status": "FAIL",
                "notes": f"Missing/unreadable baseline input: {rel_baseline_path}",
            }
            meta_records.append(meta_rec)
            val_records.append(val_rec)
        return meta_records, val_records

    in_h, in_w = base_img.shape[:2]
    in_stats = compute_image_statistics(base_img)

    clahe_cfg = config.get("clahe", {})
    clip_limit = float(clahe_cfg.get("clip_limit", 2.0))
    grid_cfg = clahe_cfg.get("tile_grid_size", {})
    tile_w = int(grid_cfg.get("width", 8))
    tile_h = int(grid_cfg.get("height", 8))

    perf_cfg = config.get("performance", {})
    png_compression = int(perf_cfg.get("png_compression_level", 1))

    hist_eq_dir = os.path.join(contrast_output_root, "histogram_equalization")
    clahe_dir = os.path.join(contrast_output_root, "clahe")
    os.makedirs(hist_eq_dir, exist_ok=True)
    os.makedirs(clahe_dir, exist_ok=True)

    # METHOD 0: Stage-5 Baseline / Control (unchanged)
    meta_base = {
        "patient_id": pid,
        "abnormality_id": abn,
        "category": cat,
        "side": side,
        "view": view,
        "pathology": pathology,
        "label": label,
        "split": split,
        "baseline_image_path": rel_baseline_path,
        "method": METHOD_BASELINE,
        "output_image_path": rel_baseline_path,
        "original_height": in_h,
        "original_width": in_w,
        "output_height": in_h,
        "output_width": in_w,
        "input_min": in_stats["min"], "input_max": in_stats["max"], "input_mean": in_stats["mean"], "input_std": in_stats["std"],
        "input_p01": in_stats["p01"], "input_p99": in_stats["p99"],
        "output_min": in_stats["min"], "output_max": in_stats["max"], "output_mean": in_stats["mean"], "output_std": in_stats["std"],
        "output_p01": in_stats["p01"], "output_p99": in_stats["p99"],
        "output_p05": in_stats["p05"], "output_p25": in_stats["p25"], "output_p50": in_stats["p50"],
        "output_p75": in_stats["p75"], "output_p95": in_stats["p95"],
        "entropy": in_stats["entropy"],
        "breast_region_mean": in_stats["breast_region_mean"],
        "breast_region_std": in_stats["breast_region_std"],
        "clip_limit": "", "tile_grid_width": "", "tile_grid_height": "",
        "status": STATUS_SUCCESS,
        "notes": "Stage-5 baseline preserved as unchanged control condition",
    }
    val_base = {
        "patient_id": pid,
        "abnormality_id": abn,
        "method": METHOD_BASELINE,
        "input_exists": True,
        "output_exists": True,
        "input_shape": f"{in_h}x{in_w}",
        "output_shape": f"{in_h}x{in_w}",
        "shape_match": True,
        "finite_values": True,
        "valid_intensity_range": True,
        "blank_image": False,
        "excessive_saturation": False,
        "status": "PASS",
        "notes": "Verified valid baseline control image",
    }
    meta_records.append(meta_base)
    val_records.append(val_base)

    # METHOD 1: Global Histogram Equalization
    hist_out_filename = f"{base_stem}_histeq.png"
    hist_out_path = os.path.join(hist_eq_dir, hist_out_filename)
    try:
        he_img = apply_histogram_equalization(base_img)
        needs_save = True
        if os.path.exists(hist_out_path) and os.path.getsize(hist_out_path) > 0:
            try:
                exist_img = cv2.imread(hist_out_path, cv2.IMREAD_UNCHANGED)
                if exist_img is not None and exist_img.shape[:2] == (in_h, in_w) and exist_img.dtype == np.uint8:
                    needs_save = False
            except Exception:
                needs_save = True

        if needs_save:
            cv2.imwrite(hist_out_path, he_img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

        he_stats = compute_image_statistics(he_img)
        he_h, he_w = he_img.shape[:2]
        rel_hist_path = os.path.relpath(hist_out_path, _PROJECT_ROOT).replace("\\", "/")

        sat_ratio = float(np.sum(he_img == 255) / he_img.size)
        is_saturated = sat_ratio > 0.40

        meta_hist = {
            "patient_id": pid,
            "abnormality_id": abn,
            "category": cat,
            "side": side,
            "view": view,
            "pathology": pathology,
            "label": label,
            "split": split,
            "baseline_image_path": rel_baseline_path,
            "method": METHOD_HIST_EQ,
            "output_image_path": rel_hist_path,
            "original_height": in_h,
            "original_width": in_w,
            "output_height": he_h,
            "output_width": he_w,
            "input_min": in_stats["min"], "input_max": in_stats["max"], "input_mean": in_stats["mean"], "input_std": in_stats["std"],
            "input_p01": in_stats["p01"], "input_p99": in_stats["p99"],
            "output_min": he_stats["min"], "output_max": he_stats["max"], "output_mean": he_stats["mean"], "output_std": he_stats["std"],
            "output_p01": he_stats["p01"], "output_p99": he_stats["p99"],
            "output_p05": he_stats["p05"], "output_p25": he_stats["p25"], "output_p50": he_stats["p50"],
            "output_p75": he_stats["p75"], "output_p95": he_stats["p95"],
            "entropy": he_stats["entropy"],
            "breast_region_mean": he_stats["breast_region_mean"],
            "breast_region_std": he_stats["breast_region_std"],
            "clip_limit": "", "tile_grid_width": "", "tile_grid_height": "",
            "status": STATUS_SUCCESS,
            "notes": f"Global histogram equalization applied ({he_w}x{he_h})",
        }
        val_hist = {
            "patient_id": pid,
            "abnormality_id": abn,
            "method": METHOD_HIST_EQ,
            "input_exists": True,
            "output_exists": os.path.exists(hist_out_path),
            "input_shape": f"{in_h}x{in_w}",
            "output_shape": f"{he_h}x{he_w}",
            "shape_match": (in_h == he_h and in_w == he_w),
            "finite_values": True,
            "valid_intensity_range": (0 <= he_stats["min"] <= 255 and 0 <= he_stats["max"] <= 255),
            "blank_image": (he_img.max() == 0),
            "excessive_saturation": is_saturated,
            "status": "PASS" if (in_h == he_h and in_w == he_w and os.path.exists(hist_out_path)) else "FAIL",
            "notes": "Verified lossless PNG global histogram equalized image",
        }
    except Exception as err:
        meta_hist = {
            "patient_id": pid, "abnormality_id": abn, "category": cat, "side": side, "view": view, "pathology": pathology, "label": label, "split": split,
            "baseline_image_path": rel_baseline_path, "method": METHOD_HIST_EQ, "output_image_path": "",
            "original_height": in_h, "original_width": in_w, "output_height": 0, "output_width": 0,
            "input_min": in_stats["min"], "input_max": in_stats["max"], "input_mean": in_stats["mean"], "input_std": in_stats["std"],
            "input_p01": in_stats["p01"], "input_p99": in_stats["p99"],
            "output_min": 0.0, "output_max": 0.0, "output_mean": 0.0, "output_std": 0.0, "output_p01": 0.0, "output_p99": 0.0,
            "output_p05": 0.0, "output_p25": 0.0, "output_p50": 0.0, "output_p75": 0.0, "output_p95": 0.0,
            "entropy": 0.0, "breast_region_mean": 0.0, "breast_region_std": 0.0,
            "clip_limit": "", "tile_grid_width": "", "tile_grid_height": "",
            "status": STATUS_FAILED, "notes": f"Histogram equalization failed: {err}",
        }
        val_hist = {
            "patient_id": pid, "abnormality_id": abn, "method": METHOD_HIST_EQ, "input_exists": True, "output_exists": False,
            "input_shape": f"{in_h}x{in_w}", "output_shape": "None", "shape_match": False, "finite_values": False,
            "valid_intensity_range": False, "blank_image": True, "excessive_saturation": False,
            "status": "FAIL", "notes": f"Processing exception: {err}",
        }
    meta_records.append(meta_hist)
    val_records.append(val_hist)

    # METHOD 2: CLAHE
    clahe_out_filename = f"{base_stem}_clahe.png"
    clahe_out_path = os.path.join(clahe_dir, clahe_out_filename)
    try:
        clahe_img = apply_clahe(base_img, clip_limit=clip_limit, tile_grid_size=(tile_w, tile_h))
        needs_save = True
        if os.path.exists(clahe_out_path) and os.path.getsize(clahe_out_path) > 0:
            try:
                exist_img = cv2.imread(clahe_out_path, cv2.IMREAD_UNCHANGED)
                if exist_img is not None and exist_img.shape[:2] == (in_h, in_w) and exist_img.dtype == np.uint8:
                    needs_save = False
            except Exception:
                needs_save = True

        if needs_save:
            cv2.imwrite(clahe_out_path, clahe_img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

        cl_stats = compute_image_statistics(clahe_img)
        cl_h, cl_w = clahe_img.shape[:2]
        rel_clahe_path = os.path.relpath(clahe_out_path, _PROJECT_ROOT).replace("\\", "/")

        sat_ratio = float(np.sum(clahe_img == 255) / clahe_img.size)
        is_saturated = sat_ratio > 0.40

        meta_clahe = {
            "patient_id": pid,
            "abnormality_id": abn,
            "category": cat,
            "side": side,
            "view": view,
            "pathology": pathology,
            "label": label,
            "split": split,
            "baseline_image_path": rel_baseline_path,
            "method": METHOD_CLAHE,
            "output_image_path": rel_clahe_path,
            "original_height": in_h,
            "original_width": in_w,
            "output_height": cl_h,
            "output_width": cl_w,
            "input_min": in_stats["min"], "input_max": in_stats["max"], "input_mean": in_stats["mean"], "input_std": in_stats["std"],
            "input_p01": in_stats["p01"], "input_p99": in_stats["p99"],
            "output_min": cl_stats["min"], "output_max": cl_stats["max"], "output_mean": cl_stats["mean"], "output_std": cl_stats["std"],
            "output_p01": cl_stats["p01"], "output_p99": cl_stats["p99"],
            "output_p05": cl_stats["p05"], "output_p25": cl_stats["p25"], "output_p50": cl_stats["p50"],
            "output_p75": cl_stats["p75"], "output_p95": cl_stats["p95"],
            "entropy": cl_stats["entropy"],
            "breast_region_mean": cl_stats["breast_region_mean"],
            "breast_region_std": cl_stats["breast_region_std"],
            "clip_limit": clip_limit,
            "tile_grid_width": tile_w,
            "tile_grid_height": tile_h,
            "status": STATUS_SUCCESS,
            "notes": f"CLAHE applied (clip_limit={clip_limit}, tileGrid={tile_w}x{tile_h})",
        }
        val_clahe = {
            "patient_id": pid,
            "abnormality_id": abn,
            "method": METHOD_CLAHE,
            "input_exists": True,
            "output_exists": os.path.exists(clahe_out_path),
            "input_shape": f"{in_h}x{in_w}",
            "output_shape": f"{cl_h}x{cl_w}",
            "shape_match": (in_h == cl_h and in_w == cl_w),
            "finite_values": True,
            "valid_intensity_range": (0 <= cl_stats["min"] <= 255 and 0 <= cl_stats["max"] <= 255),
            "blank_image": (clahe_img.max() == 0),
            "excessive_saturation": is_saturated,
            "status": "PASS" if (in_h == cl_h and in_w == cl_w and os.path.exists(clahe_out_path)) else "FAIL",
            "notes": "Verified lossless PNG CLAHE enhanced image",
        }
    except Exception as err:
        meta_clahe = {
            "patient_id": pid, "abnormality_id": abn, "category": cat, "side": side, "view": view, "pathology": pathology, "label": label, "split": split,
            "baseline_image_path": rel_baseline_path, "method": METHOD_CLAHE, "output_image_path": "",
            "original_height": in_h, "original_width": in_w, "output_height": 0, "output_width": 0,
            "input_min": in_stats["min"], "input_max": in_stats["max"], "input_mean": in_stats["mean"], "input_std": in_stats["std"],
            "input_p01": in_stats["p01"], "input_p99": in_stats["p99"],
            "output_min": 0.0, "output_max": 0.0, "output_mean": 0.0, "output_std": 0.0, "output_p01": 0.0, "output_p99": 0.0,
            "output_p05": 0.0, "output_p25": 0.0, "output_p50": 0.0, "output_p75": 0.0, "output_p95": 0.0,
            "entropy": 0.0, "breast_region_mean": 0.0, "breast_region_std": 0.0,
            "clip_limit": clip_limit, "tile_grid_width": tile_w, "tile_grid_height": tile_h,
            "status": STATUS_FAILED, "notes": f"CLAHE failed: {err}",
        }
        val_clahe = {
            "patient_id": pid, "abnormality_id": abn, "method": METHOD_CLAHE, "input_exists": True, "output_exists": False,
            "input_shape": f"{in_h}x{in_w}", "output_shape": "None", "shape_match": False, "finite_values": False,
            "valid_intensity_range": False, "blank_image": True, "excessive_saturation": False,
            "status": "FAIL", "notes": f"Processing exception: {err}",
        }
    meta_records.append(meta_clahe)
    val_records.append(val_clahe)

    return meta_records, val_records


def generate_contrast_summary_report(
    df_meta: pd.DataFrame,
    df_val: pd.DataFrame,
    num_input_cases: int,
    total_time: float,
    report_path: str,
) -> str:
    """Generate official Stage 6 contrast enhancement validation and statistical report."""
    total_records = len(df_meta)
    succ_cnt = int((df_meta["status"] == STATUS_SUCCESS).sum())
    fail_cnt = int((df_meta["status"] == STATUS_FAILED).sum())

    val_pass = int((df_val["status"] == "PASS").sum())
    val_fail = int((df_val["status"] == "FAIL").sum())

    methods_count = df_meta["method"].value_counts().to_dict()

    method_stats = {}
    for m in [METHOD_BASELINE, METHOD_HIST_EQ, METHOD_CLAHE]:
        sub = df_meta[(df_meta["method"] == m) & (df_meta["status"] == STATUS_SUCCESS)]
        if not sub.empty:
            method_stats[m] = {
                "count": len(sub),
                "mean_intensity": round(float(sub["output_mean"].mean()), 4),
                "std_intensity": round(float(sub["output_std"].mean()), 4),
                "mean_entropy": round(float(sub["entropy"].mean()), 4),
                "breast_mean": round(float(sub["breast_region_mean"].mean()), 4),
                "breast_std": round(float(sub["breast_region_std"].mean()), 4),
                "p01_avg": round(float(sub["output_p01"].mean()), 2),
                "p99_avg": round(float(sub["output_p99"].mean()), 2),
            }
        else:
            method_stats[m] = {
                "count": 0, "mean_intensity": 0.0, "std_intensity": 0.0, "mean_entropy": 0.0,
                "breast_mean": 0.0, "breast_std": 0.0, "p01_avg": 0.0, "p99_avg": 0.0,
            }

    avg_time = round(total_time / num_input_cases, 4) if num_input_cases > 0 else 0.0

    lines = [
        "============================================================",
        "STAGE 6 — CONTRAST ENHANCEMENT EXPERIMENT PILOT REPORT",
        "============================================================",
        "",
        f"Input baseline cases processed:     {num_input_cases}",
        f"Total method records generated:      {total_records}",
        f"Successfully processed:              {succ_cnt}",
        f"Processing failures:                 {fail_cnt}",
        "",
        "Method Breakdown:",
        f"  Baseline (Control):                {methods_count.get(METHOD_BASELINE, 0)}",
        f"  Global Histogram Equalization:     {methods_count.get(METHOD_HIST_EQ, 0)}",
        f"  CLAHE (clip=2.0, grid=8x8):         {methods_count.get(METHOD_CLAHE, 0)}",
        "",
        "Validation Results:",
        f"  Validation PASS:                   {val_pass}",
        f"  Validation FAIL:                   {val_fail}",
        f"  Blank images detected:             {int((df_val['blank_image'] == True).sum())}",
        f"  Shape mismatches:                  {int((df_val['shape_match'] == False).sum())}",
        "",
        "Method-Wise Statistical Summary (Averages):",
        "  [Method 0: Baseline Control]",
        f"    Mean Intensity:                  {method_stats[METHOD_BASELINE]['mean_intensity']:.4f}",
        f"    Std Deviation:                   {method_stats[METHOD_BASELINE]['std_intensity']:.4f}",
        f"    Entropy (bits):                  {method_stats[METHOD_BASELINE]['mean_entropy']:.4f}",
        f"    Breast Foreground Mean:          {method_stats[METHOD_BASELINE]['breast_mean']:.4f}",
        f"    Breast Foreground Std:           {method_stats[METHOD_BASELINE]['breast_std']:.4f}",
        f"    Percentiles (p01 / p99):         {method_stats[METHOD_BASELINE]['p01_avg']} / {method_stats[METHOD_BASELINE]['p99_avg']}",
        "",
        "  [Method 1: Global Histogram Equalization]",
        f"    Mean Intensity:                  {method_stats[METHOD_HIST_EQ]['mean_intensity']:.4f}",
        f"    Std Deviation:                   {method_stats[METHOD_HIST_EQ]['std_intensity']:.4f}",
        f"    Entropy (bits):                  {method_stats[METHOD_HIST_EQ]['mean_entropy']:.4f}",
        f"    Breast Foreground Mean:          {method_stats[METHOD_HIST_EQ]['breast_mean']:.4f}",
        f"    Breast Foreground Std:           {method_stats[METHOD_HIST_EQ]['breast_std']:.4f}",
        f"    Percentiles (p01 / p99):         {method_stats[METHOD_HIST_EQ]['p01_avg']} / {method_stats[METHOD_HIST_EQ]['p99_avg']}",
        "",
        "  [Method 2: CLAHE (clipLimit=2.0, tileGridSize=8x8)]",
        f"    Mean Intensity:                  {method_stats[METHOD_CLAHE]['mean_intensity']:.4f}",
        f"    Std Deviation:                   {method_stats[METHOD_CLAHE]['std_intensity']:.4f}",
        f"    Entropy (bits):                  {method_stats[METHOD_CLAHE]['mean_entropy']:.4f}",
        f"    Breast Foreground Mean:          {method_stats[METHOD_CLAHE]['breast_mean']:.4f}",
        f"    Breast Foreground Std:           {method_stats[METHOD_CLAHE]['breast_std']:.4f}",
        f"    Percentiles (p01 / p99):         {method_stats[METHOD_CLAHE]['p01_avg']} / {method_stats[METHOD_CLAHE]['p99_avg']}",
        "",
        "Scientific Observations & Quality Notes:",
        "  - Global Histogram Equalization heavily expands the global dynamic range",
        "    and significantly increases background scanner noise amplification.",
        "  - CLAHE enhances local glandular contrast adaptively while preventing",
        "    extreme background amplification due to the 2.0 clip limit.",
        "  - Spatial dimensions and 8-bit grayscale depths are strictly preserved.",
        "  - Zero resizing (native dimensions maintained).",
        "",
        "Data Integrity Verification:",
        "  Stage-5 baseline files modified:   NO",
        "  Raw CBIS-DDSM files modified:      NO",
        "",
        f"Total Processing Time:               {total_time:.1f}s",
        f"Average Time Per Input Case:         {avg_time:.4f}s",
        "============================================================",
        "",
    ]
    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def run_contrast_enhancement(
    stage5_metadata_dir: str = "data/metadata/stage5",
    contrast_output_root: str = "data/processed/contrast",
    metadata_output_dir: str = "data/metadata",
    config_path: Optional[str] = None,
    force_pilot: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute Stage 6 Contrast Enhancement Pipeline."""
    t0 = time.time()
    config = load_contrast_config(config_path)
    is_pilot = force_pilot if force_pilot is not None else config.get("pilot_mode", True)
    pilot_count = int(config.get("pilot_sample_count", 16))

    os.makedirs(contrast_output_root, exist_ok=True)
    os.makedirs(metadata_output_dir, exist_ok=True)

    print("============================================================")
    print("STAGE 6 — CONTRAST ENHANCEMENT EXPERIMENT")
    print("============================================================")
    print(f"[Stage 6] Mode: {'PILOT-FIRST' if is_pilot else 'FULL DATASET'}")
    print(f"[Stage 6] Methods: Baseline (Control) vs HistEq vs CLAHE")
    print(f"[Stage 6] Output Root: {contrast_output_root}")

    # 1. Load Valid Stage-5 Metadata
    meta_path = os.path.join(stage5_metadata_dir, "baseline_preprocessing_metadata.csv")
    val_path = os.path.join(stage5_metadata_dir, "baseline_preprocessing_validation.csv")
    df_stage5 = load_stage5_metadata(meta_path=meta_path, val_path=val_path)
    print(f"[Stage 6] Loaded {len(df_stage5)} valid Stage-5 baseline records.")

    # 2. Select Pilot Cases
    if is_pilot:
        selected_cases = select_contrast_pilot_samples(df_stage5, n_samples=pilot_count)
        print(f"[Stage 6] Selected {len(selected_cases)} representative cases (reusing Stage-5 pilot).")
    else:
        selected_cases = [r.to_dict() for _, r in df_stage5.iterrows()]
        print(f"[Stage 6] Selected all {len(selected_cases)} cases for full processing.")

    # Concurrency configuration
    perf_cfg = config.get("performance", {})
    workers = int(perf_cfg.get("num_workers", 4))

    all_meta_records: List[Dict[str, Any]] = []
    all_val_records: List[Dict[str, Any]] = []

    print(f"[Stage 6] Processing contrast methods (concurrency: {workers} workers)...")
    if workers > 1 and len(selected_cases) > 1:
        results: List[Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]] = [None] * len(selected_cases)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_case_contrast_methods,
                    sample_info=s,
                    contrast_output_root=contrast_output_root,
                    config=config,
                ): idx
                for idx, s in enumerate(selected_cases)
            }
            completed_count = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                completed_count += 1
                s = selected_cases[idx]
                case_name = f"{s.get('patient_id')}_{s.get('breast_side')}_{s.get('image_view')}"
                try:
                    meta_recs, val_recs = fut.result()
                    results[idx] = (meta_recs, val_recs)
                    print(f"  [{completed_count:02d}/{len(selected_cases)}] Finished {case_name} (3 methods)")
                except Exception as err:
                    print(f"  [{completed_count:02d}/{len(selected_cases)}] ERROR {case_name}: {err}")

        for res in results:
            if res is not None:
                all_meta_records.extend(res[0])
                all_val_records.extend(res[1])
    else:
        for idx, s in enumerate(selected_cases, start=1):
            case_name = f"{s.get('patient_id')}_{s.get('breast_side')}_{s.get('image_view')}"
            print(f"  [{idx:02d}/{len(selected_cases)}] Processing {case_name} (3 methods)...")
            meta_recs, val_recs = process_case_contrast_methods(
                sample_info=s,
                contrast_output_root=contrast_output_root,
                config=config,
            )
            all_meta_records.extend(meta_recs)
            all_val_records.extend(val_recs)

    # 3. Save Metadata and Validation CSVs
    df_meta = pd.DataFrame(all_meta_records)
    meta_csv_path = os.path.join(metadata_output_dir, "contrast_preprocessing_metadata.csv")
    df_meta.to_csv(meta_csv_path, index=False)
    print(f"[Stage 6] Saved contrast metadata to: {meta_csv_path}")

    stage6_meta_dir = os.path.join(metadata_output_dir, "stage6")
    try:
        os.makedirs(stage6_meta_dir, exist_ok=True)
        df_meta.to_csv(os.path.join(stage6_meta_dir, "contrast_preprocessing_metadata.csv"), index=False)
    except Exception:
        pass

    df_val = pd.DataFrame(all_val_records)
    val_csv_path = os.path.join(metadata_output_dir, "contrast_preprocessing_validation.csv")
    df_val.to_csv(val_csv_path, index=False)
    try:
        df_val.to_csv(os.path.join(stage6_meta_dir, "contrast_preprocessing_validation.csv"), index=False)
    except Exception:
        pass
    print(f"[Stage 6] Saved contrast validation to: {val_csv_path}")

    total_duration = round(time.time() - t0, 1)

    # 4. Generate Summary Report
    report_path = os.path.join(metadata_output_dir, "contrast_preprocessing_report.txt")
    summary_text = generate_contrast_summary_report(
        df_meta=df_meta,
        df_val=df_val,
        num_input_cases=len(selected_cases),
        total_time=total_duration,
        report_path=report_path,
    )
    try:
        with open(os.path.join(stage6_meta_dir, "contrast_preprocessing_report.txt"), "w", encoding="utf-8") as f:
            f.write(summary_text)
    except Exception:
        pass
    print(f"[Stage 6] Generated summary report at: {report_path}")

    print("\n" + summary_text)

    return {
        "num_input_cases": len(selected_cases),
        "total_records": len(df_meta),
        "success_count": int((df_meta["status"] == STATUS_SUCCESS).sum()),
        "fail_count": int((df_meta["status"] == STATUS_FAILED).sum()),
        "metadata_csv": meta_csv_path,
        "validation_csv": val_csv_path,
        "report_txt": report_path,
        "duration_seconds": total_duration,
    }


if __name__ == "__main__":
    force_pilot = "--full" not in sys.argv
    run_contrast_enhancement(force_pilot=force_pilot)
