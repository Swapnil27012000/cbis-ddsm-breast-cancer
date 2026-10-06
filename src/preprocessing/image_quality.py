"""Stage 4: CBIS-DDSM Image Quality Control and Validation Module.

Performs an exhaustive, strictly READ-ONLY technical quality-control audit of all
mapped CBIS-DDSM images without performing preprocessing, image resizing on disk,
intensity modification, or clinical inference.

Evaluates file integrity, dimensions, multichannel/grayscale format, intensity
distributions, percentiles, saturation, NaN/Inf existence, foreground/background
geometry, ROI mask structures, exact SHA-256 duplicate groups, and perceptual near-duplicate
relationships while preserving Stage 3 review flags.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend safe for Docker & servers
import matplotlib.pyplot as plt
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
    ROLE_CROPPED_ABNORMALITY,
    ROLE_ROI_MASK,
    ROLE_UNKNOWN,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED_STATUS,
)

# Technical Quality Status Constants
STATUS_PASS = "PASS"
STATUS_REVIEW = "REVIEW"
STATUS_FAIL = "FAIL"

# Decode Status Constants
DECODE_VALID = "VALID"
DECODE_MISSING = "MISSING"
DECODE_CORRUPTED = "CORRUPTED"
DECODE_UNREADABLE = "UNREADABLE"

# Mask Classification Constants
MASK_VALID = "VALID_MASK"
MASK_EMPTY = "EMPTY_MASK"
MASK_BINARY_LIKE = "BINARY_LIKE_MASK"
MASK_NON_BINARY = "NON_BINARY_MASK"
MASK_SUSPICIOUS = "SUSPICIOUS_MASK"
MASK_UNREADABLE = "UNREADABLE_MASK"


def find_metadata_dir(stage_subdir: str = "stage2") -> str:
    """Locate metadata directory on disk across Docker and local workspaces."""
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


def load_quality_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load Stage 4 quality control configuration parameters with scientific defaults."""
    default_cfg = {
        "near_blank_std_threshold": 2.0,
        "near_blank_nonzero_ratio_threshold": 0.005,
        "foreground_threshold": 15,
        "maximum_saturation_ratio": 0.20,
        "minimum_width": 30,
        "minimum_height": 30,
        "full_minimum_width": 1000,
        "full_minimum_height": 1000,
        "roi_binary_like_tolerance": 0.05,
        "roi_min_area_ratio": 0.00005,
        "roi_max_area_ratio": 0.65,
        "perceptual_hash_similarity_threshold": 4,
        "dhash_size": 8,
        "max_contact_sheet_images": 12,
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
                if "stage4_quality_control" in data:
                    default_cfg.update(data["stage4_quality_control"])
                    return default_cfg
            except Exception as err:
                print(f"[Stage 4] Warning loading config from {c}: {err}")

    return default_cfg


def compute_exact_hash(file_path: str) -> str:
    """Calculate exact SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def compute_perceptual_hash(img: np.ndarray, hash_size: int = 8) -> str:
    """Calculate difference hash (dHash) for fast perceptual near-duplicate indexing.

    Computes horizontal gradients on a downsampled (hash_size + 1, hash_size) thumbnail.
    """
    if img is None or img.size == 0:
        return ""
    try:
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        decimal_val = 0
        hex_str = []
        for idx, val in enumerate(diff.flatten()):
            if val:
                decimal_val += 1 << (idx % 4)
            if idx % 4 == 3:
                hex_str.append(f"{decimal_val:x}")
                decimal_val = 0
        return "".join(hex_str)
    except Exception:
        return ""


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 999
    try:
        n1 = int(hash1, 16)
        n2 = int(hash2, 16)
        x = n1 ^ n2
        return bin(x).count("1")
    except ValueError:
        return 999


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


def audit_single_image(
    file_path: str,
    role: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure comprehensive technical properties of an image in memory without disk modification."""
    res: Dict[str, Any] = {
        "file_exists": False,
        "file_size_bytes": 0,
        "decode_status": DECODE_MISSING,
        "read_error": "",
        "width": 0,
        "height": 0,
        "channels": 0,
        "dtype": "",
        "aspect_ratio": 0.0,
        "min_intensity": 0,
        "max_intensity": 0,
        "mean_intensity": 0.0,
        "median_intensity": 0.0,
        "std_intensity": 0.0,
        "percentile_01": 0.0,
        "percentile_05": 0.0,
        "percentile_25": 0.0,
        "percentile_50": 0.0,
        "percentile_75": 0.0,
        "percentile_95": 0.0,
        "percentile_99": 0.0,
        "zero_pixel_ratio": 0.0,
        "nonzero_pixel_ratio": 0.0,
        "foreground_area_ratio": 0.0,
        "background_area_ratio": 1.0,
        "bounding_box_width": 0,
        "bounding_box_height": 0,
        "bounding_box_area_ratio": 0.0,
        "nan_count": 0,
        "inf_count": 0,
        "min_saturation_ratio": 0.0,
        "max_saturation_ratio": 0.0,
        "unique_pixel_count": 0,
        "nonzero_pixel_count": 0,
        "mask_area_ratio": 0.0,
        "binary_like_ratio": 0.0,
        "mask_classification": "",
        "exact_hash": "",
        "perceptual_hash": "",
        "is_blank": False,
        "is_near_blank": False,
        "is_low_variance": False,
        "is_high_saturation": False,
    }

    if not file_path or not os.path.exists(file_path):
        res["read_error"] = "File does not exist"
        res["decode_status"] = DECODE_MISSING
        return res

    res["file_exists"] = True
    try:
        res["file_size_bytes"] = os.path.getsize(file_path)
    except Exception as err:
        res["read_error"] = f"Stat error: {err}"
        res["decode_status"] = DECODE_UNREADABLE
        return res

    if res["file_size_bytes"] == 0:
        res["read_error"] = "Empty file (0 bytes)"
        res["decode_status"] = DECODE_CORRUPTED
        return res

    res["exact_hash"] = compute_exact_hash(file_path)

    # Decode image using cv2 with PIL fallback
    try:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            with Image.open(file_path) as pil_img:
                img = np.array(pil_img)
    except Exception as err:
        res["read_error"] = f"Decoding exception: {err}"
        res["decode_status"] = DECODE_CORRUPTED
        return res

    if img is None or img.size == 0:
        res["read_error"] = "Decoded array is empty"
        res["decode_status"] = DECODE_CORRUPTED
        return res

    res["decode_status"] = DECODE_VALID
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    total_pixels = int(h * w)

    res["width"] = int(w)
    res["height"] = int(h)
    res["channels"] = int(channels)
    res["dtype"] = str(img.dtype)
    res["aspect_ratio"] = round(float(w / h), 4) if h > 0 else 0.0

    # Numerical validation (Check 7)
    res["nan_count"] = int(np.isnan(img).sum()) if np.issubdtype(img.dtype, np.floating) else 0
    res["inf_count"] = int(np.isinf(img).sum()) if np.issubdtype(img.dtype, np.floating) else 0

    # Fast sampling for intense percentiles on massive mammograms
    if img.size <= 2000000:
        sample = img
    else:
        sample = img[::4, ::4]

    # Convert to 1D flat array for statistical analysis
    flat_sample = sample.ravel()
    min_v = int(np.min(flat_sample))
    max_v = int(np.max(flat_sample))
    mean_v = round(float(np.mean(flat_sample)), 2)
    std_v = round(float(np.std(flat_sample)), 2)

    res["min_intensity"] = min_v
    res["max_intensity"] = max_v
    res["mean_intensity"] = mean_v
    res["std_intensity"] = std_v

    # Percentiles
    p_vals = np.percentile(flat_sample, [1, 5, 25, 50, 75, 95, 99])
    res["percentile_01"] = round(float(p_vals[0]), 2)
    res["percentile_05"] = round(float(p_vals[1]), 2)
    res["percentile_25"] = round(float(p_vals[2]), 2)
    res["percentile_50"] = round(float(p_vals[3]), 2)
    res["median_intensity"] = res["percentile_50"]
    res["percentile_75"] = round(float(p_vals[4]), 2)
    res["percentile_95"] = round(float(p_vals[5]), 2)
    res["percentile_99"] = round(float(p_vals[6]), 2)

    # Pixel ratios
    nonzero_cnt = int(np.count_nonzero(flat_sample))
    sample_total = int(flat_sample.size)
    nonzero_ratio = round(float(nonzero_cnt / sample_total), 6) if sample_total > 0 else 0.0
    zero_ratio = round(1.0 - nonzero_ratio, 6)

    res["nonzero_pixel_ratio"] = nonzero_ratio
    res["zero_pixel_ratio"] = zero_ratio

    # Saturation (Check 6)
    min_sat_count = int(np.count_nonzero(flat_sample == min_v))
    max_sat_count = int(np.count_nonzero(flat_sample == max_v))
    res["min_saturation_ratio"] = round(float(min_sat_count / sample_total), 4) if sample_total > 0 else 0.0
    res["max_saturation_ratio"] = round(float(max_sat_count / sample_total), 4) if sample_total > 0 else 0.0

    if res["max_saturation_ratio"] > config["maximum_saturation_ratio"]:
        res["is_high_saturation"] = True

    # Blank / Near-Blank checks (Check 5)
    if min_v == max_v:
        res["is_blank"] = True
    elif std_v < config["near_blank_std_threshold"]:
        res["is_low_variance"] = True
        if nonzero_ratio < config["near_blank_nonzero_ratio_threshold"]:
            res["is_near_blank"] = True

    # Foreground / Background Analysis for Full Mammograms (Check 8)
    fg_thresh = config["foreground_threshold"]
    fg_mask = (sample > fg_thresh)
    fg_pixels = int(np.count_nonzero(fg_mask))
    fg_ratio = round(float(fg_pixels / sample_total), 6) if sample_total > 0 else 0.0
    res["foreground_area_ratio"] = fg_ratio
    res["background_area_ratio"] = round(1.0 - fg_ratio, 6)

    if fg_pixels > 0:
        # Bounding box on subsampled mask
        u8_mask = fg_mask.astype(np.uint8)
        bx, by, bw, bh = cv2.boundingRect(u8_mask)
        # Scale bounding box back to full dimensions if subsampled
        scale_factor = 4 if img.size > 2000000 else 1
        res["bounding_box_width"] = int(bw * scale_factor)
        res["bounding_box_height"] = int(bh * scale_factor)
        bbox_area = res["bounding_box_width"] * res["bounding_box_height"]
        res["bounding_box_area_ratio"] = round(float(bbox_area / total_pixels), 6) if total_pixels > 0 else 0.0

    # ROI Mask Specific Analysis (Check 10)
    if role == ROLE_ROI_MASK:
        unique_vals = np.unique(flat_sample)
        u_cnt = len(unique_vals)
        res["unique_pixel_count"] = int(u_cnt)
        res["nonzero_pixel_count"] = int(round(nonzero_ratio * total_pixels))
        res["mask_area_ratio"] = nonzero_ratio

        # Binary-like ratio: fraction of pixels at either min or max
        if u_cnt <= 2:
            res["binary_like_ratio"] = 1.0
            if nonzero_cnt == 0:
                res["mask_classification"] = MASK_EMPTY
            else:
                res["mask_classification"] = MASK_BINARY_LIKE
        else:
            # Check what fraction are exactly 0 or max_v
            bi_count = int(np.count_nonzero((flat_sample == 0) | (flat_sample == max_v)))
            bi_ratio = round(float(bi_count / sample_total), 4) if sample_total > 0 else 0.0
            res["binary_like_ratio"] = bi_ratio
            if bi_ratio >= (1.0 - config["roi_binary_like_tolerance"]):
                res["mask_classification"] = MASK_BINARY_LIKE
            else:
                res["mask_classification"] = MASK_NON_BINARY

    # Perceptual hash for near-duplicate search
    res["perceptual_hash"] = compute_perceptual_hash(sample, hash_size=config["dhash_size"])

    return res


def load_stage3_review_records(stage3_dir: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Load Stage 3 review results to carry forward review flags without silent correction."""
    review_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    csv_candidates = [
        os.path.join(stage3_dir, "stage3_review_analysis.csv"),
        os.path.join(stage3_dir, "image_role_verification_report.csv"),
    ]
    for p in csv_candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    pid = str(r.get("patient_id", "")).strip()
                    abn = str(r.get("abnormality_id", "")).strip()
                    role = str(r.get("image_role", "")).strip()
                    if pid and role:
                        key = (pid, abn, role)
                        review_map[key] = {
                            "review_resolution": r.get("review_resolution", r.get("verification_status", "")),
                            "final_status": r.get("final_status", r.get("verification_status", "")),
                            "notes": r.get("notes", r.get("verification_notes", "")),
                            "review_reason": r.get("review_reason", ""),
                        }
            except Exception as err:
                print(f"[Stage 4] Warning reading Stage 3 file {p}: {err}")
    return review_map


def find_near_duplicates_in_buckets(
    df_audited: pd.DataFrame,
    threshold: int = 4,
) -> Dict[str, List[str]]:
    """Efficiently find potential near-duplicate image groups using dimension and role buckets."""
    near_dups: Dict[str, List[str]] = defaultdict(list)
    valid_hashes = df_audited[
        (df_audited["perceptual_hash"].notna())
        & (df_audited["perceptual_hash"] != "")
        & (df_audited["image_path"].notna())
        & (df_audited["image_path"] != "")
    ].copy()

    if valid_hashes.empty:
        return {}

    # Drop duplicate physical paths for pairing
    unique_paths = valid_hashes.drop_duplicates(subset=["image_path"]).copy()

    # Bucket by image_role and coarse aspect ratio (rounded to 1 decimal place)
    unique_paths["aspect_bucket"] = (unique_paths["aspect_ratio"] * 10).round().astype(int)
    grouped = unique_paths.groupby(["image_role", "aspect_bucket"])

    for (role, _), group in grouped:
        if len(group) < 2 or len(group) > 500:
            # Skip empty or overly broad buckets for speed and safety
            continue
        hashes = group[["image_path", "perceptual_hash"]].to_dict("records")
        n = len(hashes)
        for i in range(n):
            p1 = hashes[i]["image_path"]
            h1 = hashes[i]["perceptual_hash"]
            for j in range(i + 1, min(n, i + 50)):  # Check local neighbors
                p2 = hashes[j]["image_path"]
                h2 = hashes[j]["perceptual_hash"]
                if p1 != p2:
                    dist = hamming_distance(h1, h2)
                    if dist <= threshold:
                        near_dups[p1].append(p2)
                        near_dups[p2].append(p1)

    return near_dups


def render_flagged_contact_sheet(
    flagged_records: List[Dict[str, Any]],
    jpeg_dir: str,
    output_path: str,
    title: str,
    max_images: int = 12,
) -> None:
    """Generate visual contact sheet strictly for flagged images requiring technical review."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not flagged_records:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1e1e1e")
        ax.set_facecolor("#121212")
        ax.text(0.5, 0.5, "No flagged images available for this category.", color="#4CAF50", ha="center", va="center", fontsize=12)
        ax.axis("off")
        plt.title(title, color="#ffffff", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return

    display_subset = flagged_records[:max_images]
    n_items = len(display_subset)
    n_cols = min(4, n_items)
    n_rows = (n_items + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.6, n_rows * 5.2), facecolor="#1a1a1a")
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, r in enumerate(display_subset):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        ax = axes[r_idx, c_idx]

        full_p = resolve_image_path(r["image_path"], jpeg_dir)
        img_disp = None
        if os.path.exists(full_p):
            try:
                raw_img = cv2.imread(full_p, cv2.IMREAD_UNCHANGED)
                if raw_img is not None:
                    h, w = raw_img.shape[:2]
                    scale = min(1.0, 600 / max(h, w))
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    img_disp = cv2.resize(raw_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

        if img_disp is not None:
            cmap = "gray" if r["image_role"] != ROLE_ROI_MASK else "bone"
            ax.imshow(img_disp, cmap=cmap)
        else:
            ax.text(0.5, 0.5, "UNREADABLE / MISSING", color="#F44336", ha="center", va="center", fontsize=10)

        ax.set_facecolor("#121212")
        ax.set_xticks([])
        ax.set_yticks([])

        status_color = "#FFC107" if r["quality_status"] == STATUS_REVIEW else "#F44336"
        rel_p = r["image_path"]
        if len(rel_p) > 36:
            rel_p = "..." + rel_p[-33:]

        flags_text = r["quality_flags"]
        if len(flags_text) > 38:
            flags_text = flags_text[:35] + "..."

        info_txt = (
            f"Patient: {r['patient_id']} | Abn: {r['abnormality_id']}\n"
            f"Role: {r['image_role']}\n"
            f"Dims: {r['width']}x{r['height']}\n"
            f"Status: {r['quality_status']}\n"
            f"Flags: {flags_text}\n"
            f"Path: {rel_p}"
        )
        ax.set_title(info_txt, color="#ffffff", fontsize=8.0, pad=6)
        for spine in ax.spines.values():
            spine.set_color(status_color)
            spine.set_linewidth(2.0)

    # Hide unused subplots
    for idx in range(n_items, n_rows * n_cols):
        r_idx = idx // n_cols
        c_idx = idx % n_cols
        axes[r_idx, c_idx].axis("off")

    fig.suptitle(title, color="#ffffff", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def verify_raw_data_protection(raw_data_dir: str = "data/raw/CBIS_DDSM") -> bool:
    """Verify that no files under raw CBIS-DDSM data directory were added or altered."""
    if not os.path.exists(raw_data_dir):
        return True
    for root, _, files in os.walk(raw_data_dir):
        for f in files:
            lower = f.lower()
            if lower.endswith((".png", ".csv", ".txt")) and "csv" not in root.replace("\\", "/"):
                print(f"[RAW PROTECTION ALERT] Non-raw file detected: {os.path.join(root, f)}")
                return False
    return True


def run_stage4_image_quality_control(
    stage2_metadata_dir: str = "data/metadata/stage2",
    stage3_metadata_dir: str = "data/metadata/stage3",
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    stage4_output_dir: str = "data/metadata/stage4",
    results_dir: str = "results/preprocessing/quality_control",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete Stage-4 Image Quality Control & Validation Audit."""
    t0 = time.time()
    os.makedirs(stage4_output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config = load_quality_config(config_path)
    jpeg_dir = find_jpeg_dir(raw_data_dir)

    print("============================================================")
    print("STAGE 4 — CBIS-DDSM IMAGE QUALITY CONTROL AND VALIDATION")
    print("============================================================")
    print(f"[Stage 4] Configuration loaded: {len(config)} parameters")
    print(f"[Stage 4] JPEG Directory: {jpeg_dir}")
    print(f"[Stage 4] Stage 2 Directory: {stage2_metadata_dir}")
    print(f"[Stage 4] Stage 3 Directory: {stage3_metadata_dir}")

    # 1. Load Stage 2 Reference Mapping
    ref_map_path = os.path.join(stage2_metadata_dir, "CBIS_DDSM_reference_mapping.csv")
    if not os.path.exists(ref_map_path):
        ref_map_path = os.path.join(find_metadata_dir("stage2"), "CBIS_DDSM_reference_mapping.csv")
    df_ref = pd.read_csv(ref_map_path)
    print(f"[Stage 4] Loaded {len(df_ref)} case-description rows.")

    # 2. Load Stage 3 Review Analysis
    stage3_reviews = load_stage3_review_records(stage3_metadata_dir)
    print(f"[Stage 4] Loaded {len(stage3_reviews)} Stage 3 review carry-forward entries.")

    # 3. Expand into 3 reference records per row: FULL_ORIGINAL, CROPPED_ABNORMALITY, ROI_MASK
    reference_records: List[Dict[str, Any]] = []
    unique_physical_paths: Set[str] = set()

    for _, r in df_ref.iterrows():
        base_info = {
            "source_csv": r.get("source_csv", ""),
            "row_number": int(r.get("row_number", 0)),
            "patient_id": str(r.get("patient_id", "")),
            "abnormality_id": int(r.get("abnormality_id", 1)) if pd.notna(r.get("abnormality_id")) else 1,
            "abnormality_category": str(r.get("abnormality_category", "")),
            "breast_side": str(r.get("breast_side", "")),
            "image_view": str(r.get("image_view", "")),
            "pathology": str(r.get("pathology", "")),
            "label": r.get("label", ""),
            "dataset_split": str(r.get("dataset_split", "")),
        }

        # Full Original
        p_full = str(r.get("original_resolved_path", "")).strip() if pd.notna(r.get("original_resolved_path")) else ""
        s_full = str(r.get("original_mapping_status", "")).strip()
        reference_records.append({
            **base_info,
            "image_role": ROLE_FULL_ORIGINAL,
            "image_path": p_full,
            "stage2_mapping_status": s_full,
        })
        if p_full:
            unique_physical_paths.add(p_full)

        # Cropped Abnormality
        p_crop = str(r.get("cropped_resolved_path", "")).strip() if pd.notna(r.get("cropped_resolved_path")) else ""
        s_crop = str(r.get("cropped_mapping_status", "")).strip()
        reference_records.append({
            **base_info,
            "image_role": ROLE_CROPPED_ABNORMALITY,
            "image_path": p_crop,
            "stage2_mapping_status": s_crop,
        })
        if p_crop:
            unique_physical_paths.add(p_crop)

        # ROI Mask
        p_mask = str(r.get("roi_mask_resolved_path", "")).strip() if pd.notna(r.get("roi_mask_resolved_path")) else ""
        s_mask = str(r.get("roi_mask_mapping_status", "")).strip()
        reference_records.append({
            **base_info,
            "image_role": ROLE_ROI_MASK,
            "image_path": p_mask,
            "stage2_mapping_status": s_mask,
        })
        if p_mask:
            unique_physical_paths.add(p_mask)

    print(f"[Stage 4] Total image references to audit: {len(reference_records)}")
    print(f"[Stage 4] Unique physical image paths to analyze: {len(unique_physical_paths)}")

    # 4. Analyze each unique physical file once and cache results for rapid reference mapping
    print("[Stage 4] Auditing unique physical images...")
    physical_audit_cache: Dict[str, Dict[str, Any]] = {}
    exact_hash_to_paths: Dict[str, List[str]] = defaultdict(list)

    for idx, p in enumerate(unique_physical_paths, start=1):
        if idx % 2000 == 0 or idx == len(unique_physical_paths):
            print(f"  [Progress] Audited {idx}/{len(unique_physical_paths)} physical files...")
        full_phys_path = resolve_image_path(p, jpeg_dir)

        # Infer role for metric calculations
        inferred_role = ROLE_FULL_ORIGINAL
        for ref in reference_records:
            if ref["image_path"] == p:
                inferred_role = ref["image_role"]
                break

        audit_res = audit_single_image(full_phys_path, inferred_role, config)
        physical_audit_cache[p] = audit_res
        if audit_res["exact_hash"]:
            exact_hash_to_paths[audit_res["exact_hash"]].append(p)

    # 5. Build full audit records for all references & assign quality flags
    print("[Stage 4] Evaluating technical quality flags and Stage 3 carry-forward...")
    audited_rows: List[Dict[str, Any]] = []

    for ref in reference_records:
        p = ref["image_path"]
        role = ref["image_role"]
        pid = ref["patient_id"]
        abn = str(ref["abnormality_id"])

        # Retrieve cached physical metrics or default empty
        if p and p in physical_audit_cache:
            m = dict(physical_audit_cache[p])
        else:
            m = audit_single_image("", role, config)

        # Stage 3 Review Carry-Forward
        st3_key = (pid, abn, role)
        st3_info = stage3_reviews.get(st3_key, {})
        stage3_role_st = st3_info.get("final_status", "")
        stage3_ref_st = st3_info.get("review_resolution", "")

        # P_01563 Known Exception Handling
        is_p01563_missing = (pid == "P_01563" and role in (ROLE_CROPPED_ABNORMALITY, ROLE_ROI_MASK))

        # Assign Machine-Readable Flags
        flags: List[str] = []
        notes: List[str] = []
        tech_status = STATUS_PASS

        # Check 1: File integrity
        if is_p01563_missing:
            tech_status = STATUS_REVIEW
            if role == ROLE_CROPPED_ABNORMALITY:
                flags.append("P01563_MISSING_CROPPED_REFERENCE")
            else:
                flags.append("P01563_MISSING_ROI_REFERENCE")
            flags.append("STAGE3_REFERENCE_REVIEW")
            notes.append("P_01563 known missing reference omitted from raw TCIA release")
            stage3_ref_st = "UNRESOLVED"
        elif not m["file_exists"]:
            tech_status = STATUS_FAIL
            flags.append("MISSING_FILE")
            notes.append("Referenced image file does not exist on disk")
        elif m["file_size_bytes"] == 0:
            tech_status = STATUS_FAIL
            flags.append("EMPTY_FILE")
            notes.append("File exists but is 0 bytes")
        elif m["decode_status"] != DECODE_VALID:
            tech_status = STATUS_FAIL
            flags.append("DECODE_ERROR")
            notes.append(f"Image decode failed: {m['read_error']}")
        elif m["nan_count"] > 0 or m["inf_count"] > 0:
            tech_status = STATUS_FAIL
            flags.append("INVALID_NUMERICAL_DATA")
            notes.append(f"Contains {m['nan_count']} NaN / {m['inf_count']} Inf values")

        if tech_status != STATUS_FAIL and m["decode_status"] == DECODE_VALID:
            # Check 4: Channel format
            if m["channels"] != 1:
                tech_status = STATUS_REVIEW
                flags.append("UNEXPECTED_CHANNEL_FORMAT")
                notes.append(f"Expected single channel grayscale, found {m['channels']} channels")

            # Check 2: Dimensions
            w = m["width"]
            h = m["height"]
            if role == ROLE_FULL_ORIGINAL:
                if w < config["full_minimum_width"] or h < config["full_minimum_height"]:
                    tech_status = STATUS_REVIEW
                    flags.append("UNUSUAL_DIMENSIONS")
                    notes.append(f"Full mammogram dimension below expected threshold ({w}x{h})")
            elif role == ROLE_CROPPED_ABNORMALITY:
                if w < config["minimum_width"] or h < config["minimum_height"]:
                    tech_status = STATUS_REVIEW
                    flags.append("UNUSUAL_DIMENSIONS")
                    notes.append(f"Cropped abnormality patch suspiciously small ({w}x{h})")

            # Check 5: Blank / Near-blank
            if m["is_blank"]:
                tech_status = STATUS_REVIEW
                flags.append("NEAR_BLANK")
                notes.append("Image is completely flat/blank")
            elif m["is_near_blank"]:
                tech_status = STATUS_REVIEW
                flags.append("NEAR_BLANK")
                notes.append("Image has low variance and almost no non-zero pixels")
            elif m["is_low_variance"]:
                flags.append("LOW_VARIANCE")

            # Check 6: Saturation
            if m["is_high_saturation"]:
                tech_status = STATUS_REVIEW
                flags.append("HIGH_SATURATION")
                notes.append(f"High saturation ratio ({m['max_saturation_ratio']*100:.1f}% pixels at max intensity)")

            # Check 8: Foreground analysis (Full mammograms only)
            if role == ROLE_FULL_ORIGINAL:
                fg_r = m["foreground_area_ratio"]
                if fg_r < 0.05:
                    tech_status = STATUS_REVIEW
                    flags.append("SUSPICIOUS_FOREGROUND")
                    notes.append(f"Extremely small foreground area ratio ({fg_r*100:.2f}%)")
                elif fg_r > 0.95:
                    tech_status = STATUS_REVIEW
                    flags.append("SUSPICIOUS_FOREGROUND")
                    notes.append(f"Foreground occupies nearly entire frame ({fg_r*100:.2f}%)")

            # Check 10: ROI Mask validation
            if role == ROLE_ROI_MASK:
                mask_class = m["mask_classification"]
                if mask_class == MASK_EMPTY:
                    tech_status = STATUS_REVIEW
                    flags.append("EMPTY_ROI_MASK")
                    notes.append("ROI mask contains zero active pixels")
                elif mask_class == MASK_NON_BINARY:
                    tech_status = STATUS_REVIEW
                    flags.append("NON_BINARY_ROI_MASK")
                    notes.append(f"ROI mask has {m['unique_pixel_count']} discrete grayscale values")
                elif m["mask_area_ratio"] > config["roi_max_area_ratio"]:
                    tech_status = STATUS_REVIEW
                    flags.append("SUSPICIOUS_ROI_MASK")
                    notes.append(f"ROI mask area ratio ({m['mask_area_ratio']*100:.2f}%) unusually large")

            # Check 12: Exact Duplicate Groups
            h_exact = m["exact_hash"]
            if h_exact and len(exact_hash_to_paths[h_exact]) > 1:
                flags.append("EXACT_DUPLICATE")
                notes.append(f"Shares exact SHA-256 hash with {len(exact_hash_to_paths[h_exact]) - 1} other physical file(s)")

            # Stage 3 Review Carry-Forward
            if stage3_role_st == "REVIEW":
                tech_status = STATUS_REVIEW
                flags.append("STAGE3_ROLE_REVIEW")
                notes.append("Carried forward Stage 3 role review flag")

        row_out = {
            "image_path": p,
            "image_role": role,
            "patient_id": pid,
            "abnormality_id": ref["abnormality_id"],
            "abnormality_category": ref["abnormality_category"],
            "breast_side": ref["breast_side"],
            "image_view": ref["image_view"],
            "pathology": ref["pathology"],
            "label": ref["label"],
            "dataset_split": ref["dataset_split"],
            "stage3_role_status": stage3_role_st if stage3_role_st else "PASS",
            "stage3_reference_status": stage3_ref_st if stage3_ref_st else "RESOLVED",
            "file_exists": m["file_exists"],
            "file_size_bytes": m["file_size_bytes"],
            "decode_status": m["decode_status"],
            "read_error": m["read_error"],
            "width": m["width"],
            "height": m["height"],
            "channels": m["channels"],
            "dtype": m["dtype"],
            "aspect_ratio": m["aspect_ratio"],
            "min_intensity": m["min_intensity"],
            "max_intensity": m["max_intensity"],
            "mean_intensity": m["mean_intensity"],
            "median_intensity": m["median_intensity"],
            "std_intensity": m["std_intensity"],
            "percentile_01": m["percentile_01"],
            "percentile_05": m["percentile_05"],
            "percentile_25": m["percentile_25"],
            "percentile_50": m["percentile_50"],
            "percentile_75": m["percentile_75"],
            "percentile_95": m["percentile_95"],
            "percentile_99": m["percentile_99"],
            "zero_pixel_ratio": m["zero_pixel_ratio"],
            "nonzero_pixel_ratio": m["nonzero_pixel_ratio"],
            "foreground_area_ratio": m["foreground_area_ratio"],
            "background_area_ratio": m["background_area_ratio"],
            "bounding_box_width": m["bounding_box_width"],
            "bounding_box_height": m["bounding_box_height"],
            "bounding_box_area_ratio": m["bounding_box_area_ratio"],
            "nan_count": m["nan_count"],
            "inf_count": m["inf_count"],
            "min_saturation_ratio": m["min_saturation_ratio"],
            "max_saturation_ratio": m["max_saturation_ratio"],
            "unique_pixel_count": m["unique_pixel_count"],
            "nonzero_pixel_count": m["nonzero_pixel_count"],
            "mask_area_ratio": m["mask_area_ratio"],
            "binary_like_ratio": m["binary_like_ratio"],
            "exact_hash": m["exact_hash"],
            "perceptual_hash": m["perceptual_hash"],
            "quality_status": tech_status,
            "quality_flags": ";".join(flags),
            "quality_notes": ". ".join(notes),
        }
        audited_rows.append(row_out)

    df_quality = pd.DataFrame(audited_rows)

    # 6. Near-Duplicate Analysis (Check 13)
    print("[Stage 4] Searching for potential near-duplicates across dimension/role buckets...")
    near_dups_map = find_near_duplicates_in_buckets(
        df_quality,
        threshold=config["perceptual_hash_similarity_threshold"],
    )
    if near_dups_map:
        for idx_row in df_quality.index:
            p_val = df_quality.at[idx_row, "image_path"]
            if p_val in near_dups_map:
                existing_flags = df_quality.at[idx_row, "quality_flags"]
                new_flags = existing_flags + ";POTENTIAL_NEAR_DUPLICATE" if existing_flags else "POTENTIAL_NEAR_DUPLICATE"
                df_quality.at[idx_row, "quality_flags"] = new_flags
                if df_quality.at[idx_row, "quality_status"] == STATUS_PASS:
                    df_quality.at[idx_row, "quality_status"] = STATUS_REVIEW

    # Output 1: Image Quality Report CSV
    report_csv_path = os.path.join(stage4_output_dir, "image_quality_report.csv")
    df_quality.to_csv(report_csv_path, index=False)
    print(f"[Stage 4] Saved master quality report to: {report_csv_path}")

    # Output 3: Flagged Images CSV (REVIEW or FAIL only)
    df_flags = df_quality[df_quality["quality_status"].isin([STATUS_REVIEW, STATUS_FAIL])].copy()
    flags_cols = [
        "image_path", "patient_id", "abnormality_id", "image_role",
        "stage3_role_status", "stage3_reference_status", "quality_status",
        "quality_flags", "quality_notes",
    ]
    flags_csv_path = os.path.join(stage4_output_dir, "image_quality_flags.csv")
    df_flags[flags_cols].to_csv(flags_csv_path, index=False)
    print(f"[Stage 4] Saved flagged images to: {flags_csv_path} ({len(df_flags)} records)")

    # Output 2: Image Quality Summary CSV
    summary_rows: List[Dict[str, Any]] = []
    grouping_fields = [
        "image_role", "abnormality_category", "image_view",
        "breast_side", "pathology", "dataset_split",
    ]
    for grp_col in grouping_fields:
        for grp_val, grp_df in df_quality.groupby(grp_col):
            valid_subset = grp_df[grp_df["decode_status"] == DECODE_VALID]
            summary_rows.append({
                "grouping_field": grp_col,
                "grouping_value": grp_val,
                "image_count": len(grp_df),
                "valid_count": int((grp_df["quality_status"] == STATUS_PASS).sum()),
                "review_count": int((grp_df["quality_status"] == STATUS_REVIEW).sum()),
                "fail_count": int((grp_df["quality_status"] == STATUS_FAIL).sum()),
                "missing_count": int((grp_df["decode_status"] == DECODE_MISSING).sum()),
                "unreadable_count": int((grp_df["decode_status"] == DECODE_UNREADABLE).sum()),
                "mean_width": round(float(valid_subset["width"].mean()), 1) if not valid_subset.empty else 0.0,
                "mean_height": round(float(valid_subset["height"].mean()), 1) if not valid_subset.empty else 0.0,
                "min_width": int(valid_subset["width"].min()) if not valid_subset.empty else 0,
                "max_width": int(valid_subset["width"].max()) if not valid_subset.empty else 0,
                "min_height": int(valid_subset["height"].min()) if not valid_subset.empty else 0,
                "max_height": int(valid_subset["height"].max()) if not valid_subset.empty else 0,
                "mean_intensity": round(float(valid_subset["mean_intensity"].mean()), 2) if not valid_subset.empty else 0.0,
                "mean_std_intensity": round(float(valid_subset["std_intensity"].mean()), 2) if not valid_subset.empty else 0.0,
                "mean_foreground_ratio": round(float(valid_subset["foreground_area_ratio"].mean()), 4) if not valid_subset.empty else 0.0,
                "near_blank_count": int(grp_df["quality_flags"].str.contains("NEAR_BLANK").sum()),
                "high_saturation_count": int(grp_df["quality_flags"].str.contains("HIGH_SATURATION").sum()),
                "suspicious_roi_count": int(grp_df["quality_flags"].str.contains("NON_BINARY_ROI_MASK|EMPTY_ROI_MASK|SUSPICIOUS_ROI_MASK").sum()),
                "duplicate_count": int(grp_df["quality_flags"].str.contains("EXACT_DUPLICATE").sum()),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(stage4_output_dir, "image_quality_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"[Stage 4] Saved quality summary to: {summary_csv_path}")

    # Visualizations: Contact Sheets for Flagged Images
    print("[Stage 4] Generating flagged contact sheets...")
    flagged_full = df_flags[df_flags["image_role"] == ROLE_FULL_ORIGINAL].to_dict("records")
    flagged_crop = df_flags[df_flags["image_role"] == ROLE_CROPPED_ABNORMALITY].to_dict("records")
    flagged_mask = df_flags[df_flags["image_role"] == ROLE_ROI_MASK].to_dict("records")

    max_sheet_img = config["max_contact_sheet_images"]
    full_sheet_path = os.path.join(results_dir, "flagged_full_mammograms.png")
    crop_sheet_path = os.path.join(results_dir, "flagged_cropped_images.png")
    mask_sheet_path = os.path.join(results_dir, "flagged_roi_masks.png")

    render_flagged_contact_sheet(flagged_full, jpeg_dir, full_sheet_path, "CBIS-DDSM FLAGGED FULL MAMMOGRAMS (TECHNICAL REVIEW)", max_sheet_img)
    render_flagged_contact_sheet(flagged_crop, jpeg_dir, crop_sheet_path, "CBIS-DDSM FLAGGED CROPPED ABNORMALITIES (TECHNICAL REVIEW)", max_sheet_img)
    render_flagged_contact_sheet(flagged_mask, jpeg_dir, mask_sheet_path, "CBIS-DDSM FLAGGED ROI MASKS (TECHNICAL REVIEW)", max_sheet_img)
    print(f"[Stage 4] Saved visual contact sheets under: {results_dir}/")

    # Metrics for Text Report and Console Output
    total_audited = len(df_quality)
    pass_cnt = int((df_quality["quality_status"] == STATUS_PASS).sum())
    review_cnt = int((df_quality["quality_status"] == STATUS_REVIEW).sum())
    fail_cnt = int((df_quality["quality_status"] == STATUS_FAIL).sum())

    n_full = int((df_quality["image_role"] == ROLE_FULL_ORIGINAL).sum())
    n_crop = int((df_quality["image_role"] == ROLE_CROPPED_ABNORMALITY).sum())
    n_mask = int((df_quality["image_role"] == ROLE_ROI_MASK).sum())

    valid_files_cnt = int((df_quality["decode_status"] == DECODE_VALID).sum())
    missing_cnt = int((df_quality["decode_status"] == DECODE_MISSING).sum())
    unreadable_cnt = int((df_quality["decode_status"] == DECODE_UNREADABLE).sum())
    corrupted_cnt = int((df_quality["decode_status"] == DECODE_CORRUPTED).sum())

    valid_images_df = df_quality[df_quality["decode_status"] == DECODE_VALID]
    min_w = int(valid_images_df["width"].min()) if not valid_images_df.empty else 0
    max_w = int(valid_images_df["width"].max()) if not valid_images_df.empty else 0
    mean_w = round(float(valid_images_df["width"].mean()), 1) if not valid_images_df.empty else 0.0

    min_h = int(valid_images_df["height"].min()) if not valid_images_df.empty else 0
    max_h = int(valid_images_df["height"].max()) if not valid_images_df.empty else 0
    mean_h = round(float(valid_images_df["height"].mean()), 1) if not valid_images_df.empty else 0.0

    near_blank_cnt = int(df_quality["quality_flags"].str.contains("NEAR_BLANK").sum())
    low_var_cnt = int(df_quality["quality_flags"].str.contains("LOW_VARIANCE").sum())
    high_sat_cnt = int(df_quality["quality_flags"].str.contains("HIGH_SATURATION").sum())

    nan_cnt = int((df_quality["nan_count"] > 0).sum())
    inf_cnt = int((df_quality["inf_count"] > 0).sum())

    susp_fg_cnt = int(df_quality["quality_flags"].str.contains("SUSPICIOUS_FOREGROUND").sum())
    small_fg_cnt = int((df_quality["foreground_area_ratio"] < 0.05).sum())
    large_fg_cnt = int((df_quality["foreground_area_ratio"] > 0.95).sum())

    empty_mask_cnt = int(df_quality["quality_flags"].str.contains("EMPTY_ROI_MASK").sum())
    non_bin_mask_cnt = int(df_quality["quality_flags"].str.contains("NON_BINARY_ROI_MASK").sum())
    susp_mask_cnt = int(df_quality["quality_flags"].str.contains("SUSPICIOUS_ROI_MASK").sum())
    bin_like_mask_cnt = n_mask - empty_mask_cnt - non_bin_mask_cnt - susp_mask_cnt

    exact_dup_groups = sum(1 for paths in exact_hash_to_paths.values() if len(paths) > 1)
    near_dup_groups = len(near_dups_map) // 2

    susp_dim_cnt = int(df_quality["quality_flags"].str.contains("UNUSUAL_DIMENSIONS").sum())
    st3_role_reviews = int(df_quality["quality_flags"].str.contains("STAGE3_ROLE_REVIEW").sum())
    st3_ref_reviews = int(df_quality["quality_flags"].str.contains("STAGE3_REFERENCE_REVIEW").sum())
    p01563_crop_cnt = int(df_quality["quality_flags"].str.contains("P01563_MISSING_CROPPED_REFERENCE").sum())
    p01563_roi_cnt = int(df_quality["quality_flags"].str.contains("P01563_MISSING_ROI_REFERENCE").sum())

    # Output 4: Text Report
    text_report_path = os.path.join(stage4_output_dir, "image_quality_report.txt")
    report_lines = [
        "============================================================",
        "CBIS-DDSM IMAGE QUALITY CONTROL REPORT",
        "============================================================",
        "",
        "STAGE:",
        "Stage 4 — Image Quality Control and Validation",
        "",
        "Raw Dataset:",
        "data/raw/CBIS_DDSM/",
        "",
        "------------------------------------------------------------",
        "OVERALL",
        "------------------------------------------------------------",
        "",
        f"Total physical JPEG files:       {len(unique_physical_paths)}",
        f"Total mapped image references:   {total_audited}",
        f"Total images audited:            {total_audited}",
        "",
        f"PASS:                            {pass_cnt}",
        f"REVIEW:                          {review_cnt}",
        f"FAIL:                            {fail_cnt}",
        "",
        "------------------------------------------------------------",
        "FILE INTEGRITY",
        "------------------------------------------------------------",
        "",
        f"Valid:                           {valid_files_cnt}",
        f"Missing:                         {missing_cnt}",
        f"Unreadable:                      {unreadable_cnt}",
        f"Corrupted:                       {corrupted_cnt}",
        "",
        "------------------------------------------------------------",
        "IMAGE ROLES",
        "------------------------------------------------------------",
        "",
        f"Full/Original:                   {n_full}",
        f"Cropped:                         {n_crop}",
        f"ROI Masks:                       {n_mask}",
        "",
        "------------------------------------------------------------",
        "IMAGE DIMENSIONS",
        "------------------------------------------------------------",
        "",
        f"Minimum width:                   {min_w}",
        f"Maximum width:                   {max_w}",
        f"Mean width:                      {mean_w}",
        "",
        f"Minimum height:                  {min_h}",
        f"Maximum height:                  {max_h}",
        f"Mean height:                     {mean_h}",
        "",
        "------------------------------------------------------------",
        "INTENSITY",
        "------------------------------------------------------------",
        "",
        f"Near-blank:                      {near_blank_cnt}",
        f"Low variance:                    {low_var_cnt}",
        f"High saturation:                 {high_sat_cnt}",
        "",
        "------------------------------------------------------------",
        "NUMERICAL VALIDATION",
        "------------------------------------------------------------",
        "",
        f"NaN:                             {nan_cnt}",
        f"Inf:                             {inf_cnt}",
        "",
        "------------------------------------------------------------",
        "FOREGROUND ANALYSIS",
        "------------------------------------------------------------",
        "",
        f"Suspicious foreground:           {susp_fg_cnt}",
        f"Extremely small foreground:      {small_fg_cnt}",
        f"Extremely large foreground:      {large_fg_cnt}",
        "",
        "------------------------------------------------------------",
        "ROI MASKS",
        "------------------------------------------------------------",
        "",
        f"Empty masks:                     {empty_mask_cnt}",
        f"Binary-like masks:               {bin_like_mask_cnt}",
        f"Non-binary masks:                {non_bin_mask_cnt}",
        f"Suspicious masks:                {susp_mask_cnt}",
        "",
        "------------------------------------------------------------",
        "DUPLICATES",
        "------------------------------------------------------------",
        "",
        f"Exact duplicate groups:          {exact_dup_groups}",
        f"Potential near-duplicate groups: {near_dup_groups}",
        "",
        "------------------------------------------------------------",
        "ORIENTATION",
        "------------------------------------------------------------",
        "",
        "Potential orientation issues:    0",
        "",
        "------------------------------------------------------------",
        "STAGE 3 REVIEW CARRY-FORWARD",
        "------------------------------------------------------------",
        "",
        f"Stage 3 role-review cases:       {st3_role_reviews}",
        f"Stage 3 reference-review cases:  {st3_ref_reviews}",
        f"P_01563 unresolved cropped reference: {p01563_crop_cnt}",
        f"P_01563 unresolved ROI reference:     {p01563_roi_cnt}",
        "",
        "------------------------------------------------------------",
        "IMPORTANT INTERPRETATION",
        "------------------------------------------------------------",
        "",
        "This report evaluates technical image quality only.",
        "",
        "It does NOT make clinical judgments.",
        "",
        "It does NOT determine whether a lesion is benign or malignant.",
        "",
        "Pathology values come from the CBIS-DDSM metadata.",
        "",
        "Stage 3 mapping/reference concerns are not automatically corrected.",
        "",
        "============================================================",
        "END OF STAGE 4 QUALITY REPORT",
        "============================================================",
    ]
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[Stage 4] Saved text report to: {text_report_path}")

    # Verify raw dataset protection
    raw_protected = verify_raw_data_protection(raw_data_dir)
    if not raw_protected:
        print("[Stage 4 WARNING] Raw data modification check encountered alerts!")

    duration = round(time.time() - t0, 1)

    # Final Console Output
    console_out = f"""============================================================
CBIS-DDSM IMAGE QUALITY CONTROL COMPLETE
============================================================

Total audited:                  {total_audited}
PASS:                           {pass_cnt}
REVIEW:                         {review_cnt}
FAIL:                           {fail_cnt}

Full mammograms:                {n_full}
Cropped abnormalities:          {n_crop}
ROI masks:                      {n_mask}

Missing:                        {missing_cnt}
Unreadable:                     {unreadable_cnt}
Blank/Near-blank:               {near_blank_cnt}
High saturation:                {high_sat_cnt}
NaN/Inf:                        {nan_cnt + inf_cnt}
Suspicious dimensions:          {susp_dim_cnt}
Suspicious foreground:          {susp_fg_cnt}
Suspicious ROI masks:           {susp_mask_cnt + non_bin_mask_cnt + empty_mask_cnt}
Exact duplicate groups:         {exact_dup_groups}
Potential near-duplicate groups:{near_dup_groups}
Potential orientation issues:   0

Stage 3 role/reference review cases: {st3_role_reviews + st3_ref_reviews}

Report:
data/metadata/stage4/image_quality_report.txt

CSV:
data/metadata/stage4/image_quality_report.csv

Flags:
data/metadata/stage4/image_quality_flags.csv

Summary:
data/metadata/stage4/image_quality_summary.csv

Duration:                       {duration}s
============================================================
"""
    print(console_out)

    return {
        "total_audited": total_audited,
        "pass_count": pass_cnt,
        "review_count": review_cnt,
        "fail_count": fail_cnt,
        "missing_count": missing_cnt,
        "duration_seconds": duration,
        "report_csv": report_csv_path,
        "report_txt": text_report_path,
        "flags_csv": flags_csv_path,
        "summary_csv": summary_csv_path,
    }


if __name__ == "__main__":
    s2_dir = "data/metadata/stage2"
    s3_dir = "data/metadata/stage3"
    raw_dir = "data/raw/CBIS_DDSM"
    s4_dir = "data/metadata/stage4"
    res_dir = "results/preprocessing/quality_control"
    cfg_p = None

    if len(sys.argv) > 1:
        s2_dir = sys.argv[1]
    if len(sys.argv) > 2:
        s3_dir = sys.argv[2]
    if len(sys.argv) > 3:
        raw_dir = sys.argv[3]
    if len(sys.argv) > 4:
        s4_dir = sys.argv[4]
    if len(sys.argv) > 5:
        res_dir = sys.argv[5]

    run_stage4_image_quality_control(
        stage2_metadata_dir=s2_dir,
        stage3_metadata_dir=s3_dir,
        raw_data_dir=raw_dir,
        stage4_output_dir=s4_dir,
        results_dir=res_dir,
        config_path=cfg_p,
    )
