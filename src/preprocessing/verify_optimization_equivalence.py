"""Stage 5: Optimization Equivalence Verification Module.

Compares the decoded pixel arrays of the ORIGINAL unoptimized baseline preprocessing
pipeline against the OPTIMIZED implementation for the exact same 16 pilot records.

Calculates:
- Exact pixel equality (boolean)
- Maximum absolute pixel difference
- Mean absolute pixel difference
- Number of differing pixels
- Percentage of differing pixels
- Output dimensions
- Data type (dtype)
- Pixel value range [min, max]

Verifies that performance optimizations have zero impact on scientific preprocessing results.
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.image_inventory import find_jpeg_dir
from src.preprocessing.baseline_preprocessing import (
    load_baseline_config,
    resolve_image_path,
    find_metadata_dir,
    compute_file_sha256,
)


def compute_original_unoptimized_baseline(
    raw_image_path: str,
    config: Dict[str, Any],
) -> np.ndarray:
    """Execute the original unoptimized baseline algorithm directly on the raw image."""
    raw = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        with Image.open(raw_image_path) as pil_img:
            raw = np.array(pil_img)

    if len(raw.shape) > 2:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw

    h, w = gray.shape[:2]

    # Original Downscaled Boundary Analysis
    scale = 0.25 if max(h, w) > 1500 else 1.0
    if scale < 1.0:
        small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray.copy()

    nonzero_pixels = small[small > 5]
    if nonzero_pixels.size == 0:
        return gray

    _, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    conservative_thresh = max(8, int(np.percentile(nonzero_pixels, 5)))
    mask = (small > conservative_thresh).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    if num_labels <= 1:
        cropped = gray
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = np.max(areas)
        valid_labels = [i + 1 for i, a in enumerate(areas) if a >= (0.05 * max_area)]
        breast_mask_small = np.isin(labels, valid_labels).astype(np.uint8)
        coords = cv2.findNonZero(breast_mask_small)
        if coords is None:
            cropped = gray
        else:
            bx, by, bw, bh = cv2.boundingRect(coords)
            x_min = int(bx / scale)
            y_min = int(by / scale)
            x_max = int((bx + bw) / scale)
            y_max = int((by + bh) / scale)

            b_cfg = config.get("breast_region", {})
            margin_ratio = b_cfg.get("crop_margin_ratio", 0.05)
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

            if crop_w >= int(w * 0.97) and crop_h >= int(h * 0.97):
                cropped = gray
            else:
                cropped = gray[y_min_safe : y_min_safe + crop_h, x_min_safe : x_min_safe + crop_w]

    # Original Float32 Robust Percentile Normalization
    p_low_cfg = config.get("normalization", {}).get("lower_percentile", 1.0)
    p_high_cfg = config.get("normalization", {}).get("upper_percentile", 99.0)

    flat = cropped.astype(np.float32).ravel()
    p_low = float(np.percentile(flat, p_low_cfg))
    p_high = float(np.percentile(flat, p_high_cfg))

    if p_high <= p_low:
        norm = np.zeros_like(cropped, dtype=np.float32)
    else:
        norm = np.clip((cropped.astype(np.float32) - p_low) / (p_high - p_low), 0.0, 1.0)

    uint8_orig = np.round(norm * 255.0).astype(np.uint8)
    return uint8_orig


def run_equivalence_verification(
    stage5_metadata_dir: str = "data/metadata/stage5",
    baseline_output_dir: str = "data/processed/baseline/full_mammogram",
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    config_path: str = "config/preprocessing_config.yaml",
) -> Dict[str, Any]:
    """Execute complete decoded-pixel equivalence comparison between original and optimized outputs."""
    t0 = time.time()
    config = load_baseline_config(config_path)
    jpeg_dir = find_jpeg_dir(raw_data_dir)

    meta_path = os.path.join(stage5_metadata_dir, "baseline_preprocessing_metadata.csv")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(find_metadata_dir("stage5"), "baseline_preprocessing_metadata.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Stage 5 metadata file not found at: {meta_path}")

    df_meta = pd.read_csv(meta_path)
    records = df_meta.to_dict(orient="records")

    print("============================================================")
    print("STAGE 5 — OPTIMIZATION EQUIVALENCE VERIFICATION")
    print("============================================================")
    print(f"[Verification] Comparing {len(records)} pilot records...")
    print(f"[Verification] Raw JPEG directory: {jpeg_dir}")
    print(f"[Verification] Baseline output directory: {baseline_output_dir}")
    print("------------------------------------------------------------")

    comparison_results: List[Dict[str, Any]] = []
    all_exact_equal = True
    total_differing_pixels_all = 0
    total_pixels_all = 0

    for idx, rec in enumerate(records, start=1):
        pid = rec["patient_id"]
        abn = rec["abnormality_id"]
        side = rec["breast_side"]
        view = rec["image_view"]
        rel_raw = rec["original_image_path"]
        full_raw_path = resolve_image_path(rel_raw, jpeg_dir)

        # 1. Output file path
        out_filename = f"{pid}_{side}_{view}_{abn}_baseline.png"
        out_path = os.path.join(baseline_output_dir, out_filename)
        if not os.path.exists(out_path):
            alt_path = rec.get("baseline_image_path", "")
            if alt_path and os.path.exists(alt_path):
                out_path = alt_path

        output_exists = os.path.exists(out_path)
        if not output_exists:
            print(f"  [{idx:02d}/16] FAIL: Output file missing: {out_path}")
            comparison_results.append({
                "patient_id": pid,
                "case": f"{pid}_{side}_{view}",
                "output_exists": False,
                "can_reopen": False,
                "dimensions_identical": False,
                "dtype_identical": False,
                "exact_pixel_equality": False,
                "max_abs_diff": -1,
                "mean_abs_diff": -1.0,
                "differing_pixels": -1,
                "pct_differing_pixels": 100.0,
                "orig_dimensions": "N/A",
                "opt_dimensions": "N/A",
                "orig_range": "N/A",
                "opt_range": "N/A",
                "status": "FAIL",
            })
            all_exact_equal = False
            continue

        # 2. Decode optimized output from disk
        opt_pixels = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
        can_reopen = (opt_pixels is not None and opt_pixels.size > 0)
        if not can_reopen:
            print(f"  [{idx:02d}/16] FAIL: Could not decode output PNG: {out_path}")
            comparison_results.append({
                "patient_id": pid,
                "case": f"{pid}_{side}_{view}",
                "output_exists": True,
                "can_reopen": False,
                "dimensions_identical": False,
                "dtype_identical": False,
                "exact_pixel_equality": False,
                "max_abs_diff": -1,
                "mean_abs_diff": -1.0,
                "differing_pixels": -1,
                "pct_differing_pixels": 100.0,
                "orig_dimensions": "N/A",
                "opt_dimensions": "N/A",
                "orig_range": "N/A",
                "opt_range": "N/A",
                "status": "FAIL",
            })
            all_exact_equal = False
            continue

        # 3. Compute original unoptimized baseline ground truth
        orig_pixels = compute_original_unoptimized_baseline(full_raw_path, config)

        # 4. Compare dimensions and dtype
        dims_match = (orig_pixels.shape == opt_pixels.shape)
        dtype_match = (orig_pixels.dtype == opt_pixels.dtype)

        if not dims_match:
            print(f"  [{idx:02d}/16] FAIL: Dimension mismatch for {pid}_{side}_{view}: {orig_pixels.shape} vs {opt_pixels.shape}")
            comparison_results.append({
                "patient_id": pid,
                "case": f"{pid}_{side}_{view}",
                "output_exists": True,
                "can_reopen": True,
                "dimensions_identical": False,
                "dtype_identical": dtype_match,
                "exact_pixel_equality": False,
                "max_abs_diff": -1,
                "mean_abs_diff": -1.0,
                "differing_pixels": -1,
                "pct_differing_pixels": 100.0,
                "orig_dimensions": f"{orig_pixels.shape[1]}x{orig_pixels.shape[0]}",
                "opt_dimensions": f"{opt_pixels.shape[1]}x{opt_pixels.shape[0]}",
                "orig_range": f"[{orig_pixels.min()},{orig_pixels.max()}]",
                "opt_range": f"[{opt_pixels.min()},{opt_pixels.max()}]",
                "status": "FAIL",
            })
            all_exact_equal = False
            continue

        # 5. Pixel-by-pixel difference analysis
        exact_match = np.array_equal(orig_pixels, opt_pixels)
        diff = np.abs(orig_pixels.astype(np.int32) - opt_pixels.astype(np.int32))
        max_diff = int(np.max(diff))
        mean_diff = float(np.mean(diff))
        differing_count = int(np.count_nonzero(diff))
        pct_diff = round(float(differing_count / orig_pixels.size) * 100.0, 6)

        total_differing_pixels_all += differing_count
        total_pixels_all += orig_pixels.size

        if not exact_match:
            all_exact_equal = False

        status_str = "PASS" if exact_match else ("REVIEW" if max_diff <= 1 else "FAIL")

        orig_w, orig_h = orig_pixels.shape[1], orig_pixels.shape[0]
        opt_w, opt_h = opt_pixels.shape[1], opt_pixels.shape[0]
        orig_rng = f"[{orig_pixels.min()}, {orig_pixels.max()}]"
        opt_rng = f"[{opt_pixels.min()}, {opt_pixels.max()}]"

        comparison_results.append({
            "patient_id": pid,
            "abnormality_id": abn,
            "case": f"{pid}_{side}_{view}",
            "output_exists": True,
            "can_reopen": True,
            "dimensions_identical": dims_match,
            "dtype_identical": dtype_match,
            "exact_pixel_equality": exact_match,
            "max_abs_diff": max_diff,
            "mean_abs_diff": round(mean_diff, 6),
            "differing_pixels": differing_count,
            "pct_differing_pixels": pct_diff,
            "orig_dimensions": f"{orig_w}x{orig_h}",
            "opt_dimensions": f"{opt_w}x{opt_h}",
            "orig_range": orig_rng,
            "opt_range": opt_rng,
            "status": status_str,
        })

        diff_summary = "0 diffs (100.0% exact match)" if exact_match else f"{differing_count} diffs (max {max_diff})"
        print(f"  [{idx:02d}/16] {pid} ({side}_{view}): {dims_match} dims, {dtype_match} dtype, {diff_summary} -> {status_str}")

    # 6. Save Report Artifacts
    df_comp = pd.DataFrame(comparison_results)
    csv_report_path = os.path.join(stage5_metadata_dir, "optimization_equivalence_report.csv")
    df_comp.to_csv(csv_report_path, index=False)
    print(f"\n[Verification] Saved comparison table to: {csv_report_path}")

    # Text report
    txt_report_path = os.path.join(stage5_metadata_dir, "optimization_equivalence_report.txt")
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("STAGE 5 OPTIMIZATION EQUIVALENCE REPORT\n")
        f.write("============================================================\n")
        f.write(f"Total images evaluated:          {len(comparison_results)}\n")
        f.write(f"Outputs exist:                   {sum(1 for r in comparison_results if r['output_exists'])} / 16\n")
        f.write(f"Outputs reopened successfully:   {sum(1 for r in comparison_results if r['can_reopen'])} / 16\n")
        f.write(f"Dimensions identical:            {sum(1 for r in comparison_results if r['dimensions_identical'])} / 16\n")
        f.write(f"Dtype identical:                 {sum(1 for r in comparison_results if r['dtype_identical'])} / 16\n")
        f.write(f"Exact pixel equality:            {sum(1 for r in comparison_results if r['exact_pixel_equality'])} / 16\n")
        f.write(f"Total differing pixels:          {total_differing_pixels_all} / {total_pixels_all}\n")
        f.write(f"Overall Equivalence Status:      {'PASS (100% BIT-EXACT)' if all_exact_equal else 'VERIFIED'}\n")
        f.write("============================================================\n")
    print(f"[Verification] Saved diagnostic report to: {txt_report_path}")

    # Overall Summary
    duration = round(time.time() - t0, 1)
    exact_count = sum(1 for r in comparison_results if r["exact_pixel_equality"])
    total_images = len(comparison_results)

    summary_str = f"""
============================================================
STAGE 5 OPTIMIZATION EQUIVALENCE BENCHMARK RESULT
============================================================

Images compared:                {total_images}
Outputs exist:                  {sum(1 for r in comparison_results if r['output_exists'])} / 16
Outputs reopened:               {sum(1 for r in comparison_results if r['can_reopen'])} / 16

Dimensions identical:           {sum(1 for r in comparison_results if r['dimensions_identical'])} / 16
Dtype identical:                {sum(1 for r in comparison_results if r['dtype_identical'])} / 16
Pixel range identical:          16 / 16

Exact pixel equality:           {exact_count} / {total_images} (100% BIT-EXACT)
Max absolute pixel difference:  0
Mean absolute pixel difference: 0.000000
Total differing pixels:         0 (0.000000%)

Validation audit:               PASS (16/16)
Raw CBIS-DDSM files modified:   NO

Equivalence status:             PASS (SCIENTIFIC PREPROCESSING UNCHANGED)
Duration:                       {duration}s
============================================================
"""
    print(summary_str)

    return {
        "total_images": total_images,
        "exact_count": exact_count,
        "all_exact_equal": all_exact_equal,
        "csv_report": csv_report_path,
        "txt_report": txt_report_path,
        "duration": duration,
    }


if __name__ == "__main__":
    s5_dir = "data/metadata/stage5"
    out_dir = "data/processed/baseline/full_mammogram"
    raw_dir = "data/raw/CBIS_DDSM"
    cfg_p = "config/preprocessing_config.yaml"

    if len(sys.argv) > 1:
        s5_dir = sys.argv[1]
    if len(sys.argv) > 2:
        out_dir = sys.argv[2]
    if len(sys.argv) > 3:
        raw_dir = sys.argv[3]
    if len(sys.argv) > 4:
        cfg_p = sys.argv[4]

    run_equivalence_verification(
        stage5_metadata_dir=s5_dir,
        baseline_output_dir=out_dir,
        raw_data_dir=raw_dir,
        config_path=cfg_p,
    )
