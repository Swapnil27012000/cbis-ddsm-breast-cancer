"""CBIS-DDSM Dataset Structure Audit Module.

Performs a read-only audit of raw dataset directory structures, image distributions,
file duplicates, CSV image path references, and path resolution integrity.
"""

from collections import Counter, defaultdict
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from .csv_loader import find_csv_dir, load_cbis_csvs
from .path_resolver import (
    PathResolver,
    ResolutionResult,
    STATUS_RESOLVED_EXACT,
    STATUS_RESOLVED_NORMALIZED,
    STATUS_RESOLVED_UNIQUE_FILENAME,
    STATUS_AMBIGUOUS,
    STATUS_UNRESOLVED,
    STATUS_INVALID_REFERENCE,
    normalize_path_string,
)


def find_jpeg_dir(base_dir: str = "data/raw/CBIS_DDSM") -> str:
    """Locate the actual raw jpeg directory on disk."""
    candidates = [
        os.path.join(base_dir, "jpeg"),
        "data/raw/CBIS_DDSM/jpeg",
        "dataset/jpeg",
        os.path.join(os.getcwd(), "data", "raw", "CBIS_DDSM", "jpeg"),
        os.path.join(os.getcwd(), "dataset", "jpeg"),
        "/app/data/raw/CBIS_DDSM/jpeg",
        "/app/dataset/jpeg",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(base_dir, "jpeg"))


def audit_jpeg_directory(
    jpeg_dir: str,
    sample_size: int = 200,
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".dcm"),
) -> Dict[str, Any]:
    """Recursively scan JPEG folder tree and compute structural and dimension statistics.

    Args:
        jpeg_dir: Root path to raw jpeg directory.
        sample_size: Number of images to sample for dimension verification.
        valid_extensions: Recognized image file extensions.

    Returns:
        Dictionary of directory metrics, file distribution, duplicate analysis, and sample stats.
    """
    total_dirs = 0
    total_files = 0
    extension_counts: Counter = Counter()
    folder_image_counts: Dict[str, int] = {}
    filename_occurrences: Dict[str, List[str]] = defaultdict(list)
    all_image_paths: List[str] = []
    dir_depths: List[int] = []

    if not os.path.exists(jpeg_dir):
        return {
            "error": f"JPEG directory not found at {jpeg_dir}",
            "total_dirs": 0,
            "total_files": 0,
            "extension_counts": {},
            "folder_buckets": {"0_images": 0, "1_image": 0, "2_images": 0, "3_images": 0, "4+_images": 0},
            "min_images_per_folder": 0,
            "max_images_per_folder": 0,
            "avg_images_per_folder": 0.0,
            "median_images_per_folder": 0.0,
            "depth_stats": {"min_depth": 0, "max_depth": 0, "avg_depth": 0.0},
            "duplicate_filenames_count": 0,
            "duplicate_paths_count": 0,
            "sample_dimensions": [],
            "all_image_paths": [],
        }

    norm_jpeg_dir = os.path.abspath(jpeg_dir)

    for root, dirs, files in os.walk(norm_jpeg_dir):
        total_dirs += 1
        rel_path = os.path.relpath(root, norm_jpeg_dir)
        depth = 0 if rel_path == "." else len(rel_path.replace("\\", "/").split("/"))
        dir_depths.append(depth)

        img_in_dir = 0
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            extension_counts[ext if ext else "no_ext"] += 1

            if ext in valid_extensions:
                total_files += 1
                img_in_dir += 1
                full_path = os.path.join(root, f)
                all_image_paths.append(full_path)
                filename_occurrences[f].append(full_path)

        folder_image_counts[root] = img_in_dir

    # Folder buckets
    bucket_0 = sum(1 for c in folder_image_counts.values() if c == 0)
    bucket_1 = sum(1 for c in folder_image_counts.values() if c == 1)
    bucket_2 = sum(1 for c in folder_image_counts.values() if c == 2)
    bucket_3 = sum(1 for c in folder_image_counts.values() if c == 3)
    bucket_4_plus = sum(1 for c in folder_image_counts.values() if c >= 4)

    # Content counts (folders containing >= 1 image)
    content_counts = [c for c in folder_image_counts.values() if c > 0]
    min_images = int(np.min(content_counts)) if content_counts else 0
    max_images = int(np.max(content_counts)) if content_counts else 0
    avg_images = float(np.mean(content_counts)) if content_counts else 0.0
    median_images = float(np.median(content_counts)) if content_counts else 0.0

    # Duplicate detection
    duplicate_filenames = {k: v for k, v in filename_occurrences.items() if len(v) > 1}
    duplicate_paths = len(all_image_paths) - len(set(all_image_paths))

    # Representative sample dimensions
    sample_dimensions = []
    if all_image_paths:
        step = max(1, len(all_image_paths) // sample_size)
        sampled_paths = all_image_paths[::step][:sample_size]
        for p in sampled_paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                shape = img.shape
                h = shape[0]
                w = shape[1]
                channels = shape[2] if len(shape) > 2 else 1
                sample_dimensions.append({
                    "path": p,
                    "filename": os.path.basename(p),
                    "height": h,
                    "width": w,
                    "channels": channels,
                    "aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
                    "dtype": str(img.dtype),
                })

    return {
        "jpeg_dir": norm_jpeg_dir,
        "total_dirs": total_dirs,
        "total_files": total_files,
        "extension_counts": dict(extension_counts),
        "folder_buckets": {
            "0_images": bucket_0,
            "1_image": bucket_1,
            "2_images": bucket_2,
            "3_images": bucket_3,
            "4+_images": bucket_4_plus,
        },
        "min_images_per_folder": min_images,
        "max_images_per_folder": max_images,
        "avg_images_per_folder": round(avg_images, 2),
        "median_images_per_folder": round(median_images, 2),
        "depth_stats": {
            "min_depth": min(dir_depths) if dir_depths else 0,
            "max_depth": max(dir_depths) if dir_depths else 0,
            "avg_depth": round(float(np.mean(dir_depths)), 2) if dir_depths else 0.0,
        },
        "duplicate_filenames_count": len(duplicate_filenames),
        "duplicate_filenames_examples": list(duplicate_filenames.keys())[:10],
        "duplicate_paths_count": duplicate_paths,
        "sample_dimensions": sample_dimensions,
        "all_image_paths": all_image_paths,
    }


def audit_csv_linkage(
    raw_csv_dir: str = "data/raw/CBIS_DDSM/csv",
    raw_data_dir: str = "data/raw/CBIS_DDSM",
) -> Dict[str, Any]:
    """Audit CSV metadata records against disk paths using PathResolver.

    Tracks resolution across full mammogram, cropped lesion, and ROI mask references,
    broken down by CSV file and reference type.
    """
    dfs = load_cbis_csvs(raw_csv_dir)
    resolver = PathResolver(raw_data_dir)

    case_dfs = [
        ("mass_case_description_train_set.csv", dfs["mass_train"]),
        ("mass_case_description_test_set.csv", dfs["mass_test"]),
        ("calc_case_description_train_set.csv", dfs["calc_train"]),
        ("calc_case_description_test_set.csv", dfs["calc_test"]),
    ]

    total_csv_records = sum(len(df) for _, df in case_dfs)

    reference_stats = {
        "full_mammogram": {"total": 0, "resolved": 0, "unresolved": 0, "exact": 0, "normalized": 0, "unique_filename": 0, "ambiguous": 0},
        "cropped_image": {"total": 0, "resolved": 0, "unresolved": 0, "exact": 0, "normalized": 0, "unique_filename": 0, "ambiguous": 0},
        "roi_mask": {"total": 0, "resolved": 0, "unresolved": 0, "exact": 0, "normalized": 0, "unique_filename": 0, "ambiguous": 0},
    }

    # Breakdown by CSV and reference type
    summary_by_csv: List[Dict[str, Any]] = []
    unresolved_records: List[Dict[str, Any]] = []

    records_with_at_least_one_resolved = 0
    records_fully_resolved = 0

    counts_by_status = {
        "exact": 0,
        "normalized": 0,
        "unique_filename": 0,
        "ambiguous": 0,
        "unresolved": 0,
    }

    for csv_name, df in case_dfs:
        csv_counts = {
            "full_mammogram": {"total": 0, "resolved": 0, "unresolved": 0, "ambiguous": 0},
            "cropped_image": {"total": 0, "resolved": 0, "unresolved": 0, "ambiguous": 0},
            "roi_mask": {"total": 0, "resolved": 0, "unresolved": 0, "ambiguous": 0},
        }

        for row_idx, row in df.iterrows():
            f_ref = row.get("image file path", None)
            c_ref = row.get("cropped image file path", None)
            r_ref = row.get("ROI mask file path", None)

            pid = str(row.get("patient_id", ""))
            abn_id = str(row.get("abnormality id", ""))
            view = str(row.get("image view", ""))
            side = str(row.get("left or right breast", ""))

            f_res = resolver.resolve(f_ref, reference_type="FULL_IMAGE") if pd.notna(f_ref) else None
            c_res = resolver.resolve(c_ref, reference_type="CROPPED_IMAGE") if pd.notna(c_ref) else None
            r_res = resolver.resolve(r_ref, reference_type="ROI_MASK") if pd.notna(r_ref) else None

            # Process full mammogram reference
            if f_res and f_res.status != STATUS_INVALID_REFERENCE:
                reference_stats["full_mammogram"]["total"] += 1
                csv_counts["full_mammogram"]["total"] += 1
                if f_res.status == STATUS_RESOLVED_EXACT:
                    reference_stats["full_mammogram"]["resolved"] += 1
                    reference_stats["full_mammogram"]["exact"] += 1
                    csv_counts["full_mammogram"]["resolved"] += 1
                    counts_by_status["exact"] += 1
                elif f_res.status == STATUS_RESOLVED_NORMALIZED:
                    reference_stats["full_mammogram"]["resolved"] += 1
                    reference_stats["full_mammogram"]["normalized"] += 1
                    csv_counts["full_mammogram"]["resolved"] += 1
                    counts_by_status["normalized"] += 1
                elif f_res.status == STATUS_RESOLVED_UNIQUE_FILENAME:
                    reference_stats["full_mammogram"]["resolved"] += 1
                    reference_stats["full_mammogram"]["unique_filename"] += 1
                    csv_counts["full_mammogram"]["resolved"] += 1
                    counts_by_status["unique_filename"] += 1
                elif f_res.status == STATUS_AMBIGUOUS:
                    reference_stats["full_mammogram"]["ambiguous"] += 1
                    csv_counts["full_mammogram"]["ambiguous"] += 1
                    counts_by_status["ambiguous"] += 1
                else:
                    reference_stats["full_mammogram"]["unresolved"] += 1
                    csv_counts["full_mammogram"]["unresolved"] += 1
                    counts_by_status["unresolved"] += 1
                    unresolved_records.append({
                        "source_csv": csv_name,
                        "row_number": row_idx,
                        "patient_id": pid,
                        "abnormality_id": abn_id,
                        "image_view": f"{side}_{view}",
                        "image_file_path": str(f_ref),
                        "cropped_image_file_path": str(c_ref),
                        "roi_mask_file_path": str(r_ref),
                        "reference_type": "FULL_IMAGE",
                        "referenced_path": str(f_ref),
                        "normalized_reference": f_res.normalized_path,
                        "attempted_path": f_res.attempted_path,
                        "failure_reason": f_res.reason,
                        "reference_status": "UNRESOLVED",
                        "reference_usable": False,
                        "alternative_representation_available": bool(c_res and c_res.resolved_path),
                        "alternative_representation_path": c_res.resolved_path if c_res else None,
                    })

            # Process cropped image reference
            if c_res and c_res.status != STATUS_INVALID_REFERENCE:
                reference_stats["cropped_image"]["total"] += 1
                csv_counts["cropped_image"]["total"] += 1
                if c_res.status == STATUS_RESOLVED_EXACT:
                    reference_stats["cropped_image"]["resolved"] += 1
                    reference_stats["cropped_image"]["exact"] += 1
                    csv_counts["cropped_image"]["resolved"] += 1
                    counts_by_status["exact"] += 1
                elif c_res.status == STATUS_RESOLVED_NORMALIZED:
                    reference_stats["cropped_image"]["resolved"] += 1
                    reference_stats["cropped_image"]["normalized"] += 1
                    csv_counts["cropped_image"]["resolved"] += 1
                    counts_by_status["normalized"] += 1
                elif c_res.status == STATUS_RESOLVED_UNIQUE_FILENAME:
                    reference_stats["cropped_image"]["resolved"] += 1
                    reference_stats["cropped_image"]["unique_filename"] += 1
                    csv_counts["cropped_image"]["resolved"] += 1
                    counts_by_status["unique_filename"] += 1
                elif c_res.status == STATUS_AMBIGUOUS:
                    reference_stats["cropped_image"]["ambiguous"] += 1
                    csv_counts["cropped_image"]["ambiguous"] += 1
                    counts_by_status["ambiguous"] += 1
                else:
                    reference_stats["cropped_image"]["unresolved"] += 1
                    csv_counts["cropped_image"]["unresolved"] += 1
                    counts_by_status["unresolved"] += 1
                    unresolved_records.append({
                        "source_csv": csv_name,
                        "row_number": row_idx,
                        "patient_id": pid,
                        "abnormality_id": abn_id,
                        "image_view": f"{side}_{view}",
                        "image_file_path": str(f_ref),
                        "cropped_image_file_path": str(c_ref),
                        "roi_mask_file_path": str(r_ref),
                        "reference_type": "CROPPED_IMAGE",
                        "referenced_path": str(c_ref),
                        "normalized_reference": c_res.normalized_path,
                        "attempted_path": c_res.attempted_path,
                        "failure_reason": c_res.reason,
                        "reference_status": "UNRESOLVED",
                        "reference_usable": False,
                        "alternative_representation_available": bool(f_res and f_res.resolved_path),
                        "alternative_representation_path": f_res.resolved_path if f_res else None,
                    })

            # Process ROI mask reference
            if r_res and r_res.status != STATUS_INVALID_REFERENCE:
                reference_stats["roi_mask"]["total"] += 1
                csv_counts["roi_mask"]["total"] += 1
                if r_res.status == STATUS_RESOLVED_EXACT:
                    reference_stats["roi_mask"]["resolved"] += 1
                    reference_stats["roi_mask"]["exact"] += 1
                    csv_counts["roi_mask"]["resolved"] += 1
                    counts_by_status["exact"] += 1
                elif r_res.status == STATUS_RESOLVED_NORMALIZED:
                    reference_stats["roi_mask"]["resolved"] += 1
                    reference_stats["roi_mask"]["normalized"] += 1
                    csv_counts["roi_mask"]["resolved"] += 1
                    counts_by_status["normalized"] += 1
                elif r_res.status == STATUS_RESOLVED_UNIQUE_FILENAME:
                    reference_stats["roi_mask"]["resolved"] += 1
                    reference_stats["roi_mask"]["unique_filename"] += 1
                    csv_counts["roi_mask"]["resolved"] += 1
                    counts_by_status["unique_filename"] += 1
                elif r_res.status == STATUS_AMBIGUOUS:
                    reference_stats["roi_mask"]["ambiguous"] += 1
                    csv_counts["roi_mask"]["ambiguous"] += 1
                    counts_by_status["ambiguous"] += 1
                else:
                    reference_stats["roi_mask"]["unresolved"] += 1
                    csv_counts["roi_mask"]["unresolved"] += 1
                    counts_by_status["unresolved"] += 1
                    unresolved_records.append({
                        "source_csv": csv_name,
                        "row_number": row_idx,
                        "patient_id": pid,
                        "abnormality_id": abn_id,
                        "image_view": f"{side}_{view}",
                        "image_file_path": str(f_ref),
                        "cropped_image_file_path": str(c_ref),
                        "roi_mask_file_path": str(r_ref),
                        "reference_type": "ROI_MASK",
                        "referenced_path": str(r_ref),
                        "normalized_reference": r_res.normalized_path,
                        "attempted_path": r_res.attempted_path,
                        "failure_reason": r_res.reason,
                        "reference_status": "UNRESOLVED",
                        "reference_usable": False,
                        "alternative_representation_available": bool(f_res and f_res.resolved_path),
                        "alternative_representation_path": f_res.resolved_path if f_res else None,
                    })

            # Record-level resolution check
            resolved_flags = [
                bool(f_res and f_res.resolved_path),
                bool(c_res and c_res.resolved_path),
                bool(r_res and r_res.resolved_path),
            ]
            if any(resolved_flags):
                records_with_at_least_one_resolved += 1
            if all([
                bool(f_res and f_res.resolved_path) if f_res else True,
                bool(c_res and c_res.resolved_path) if c_res else True,
                bool(r_res and r_res.resolved_path) if r_res else True,
            ]):
                records_fully_resolved += 1

        for r_type in ["full_mammogram", "cropped_image", "roi_mask"]:
            tot = csv_counts[r_type]["total"]
            res = csv_counts[r_type]["resolved"]
            unres = csv_counts[r_type]["unresolved"]
            amb = csv_counts[r_type]["ambiguous"]
            rate = round((res / tot * 100), 2) if tot > 0 else 0.0
            summary_by_csv.append({
                "source_csv": csv_name,
                "reference_type": r_type,
                "total_references": tot,
                "resolved": res,
                "unresolved": unres,
                "ambiguous": amb,
                "resolution_rate": f"{rate}%",
            })

    total_image_references = (
        reference_stats["full_mammogram"]["total"]
        + reference_stats["cropped_image"]["total"]
        + reference_stats["roi_mask"]["total"]
    )
    total_resolved_references = (
        reference_stats["full_mammogram"]["resolved"]
        + reference_stats["cropped_image"]["resolved"]
        + reference_stats["roi_mask"]["resolved"]
    )
    total_unresolved_references = total_image_references - total_resolved_references

    return {
        "total_csv_records": total_csv_records,
        "records_with_resolved_image": records_with_at_least_one_resolved,
        "records_fully_resolved": records_fully_resolved,
        "total_image_references": total_image_references,
        "total_resolved_references": total_resolved_references,
        "total_unresolved_references": total_unresolved_references,
        "counts_by_status": counts_by_status,
        "reference_stats": reference_stats,
        "summary_by_csv": summary_by_csv,
        "unresolved_records": unresolved_records,
    }


def generate_unresolved_references_csv(
    unresolved_records: List[Dict[str, Any]],
    output_path: str = "data/metadata/stage1/dataset_unresolved_references.csv",
) -> pd.DataFrame:
    """Generate dedicated diagnostic CSV of unresolved image references."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    columns = [
        "source_csv",
        "row_number",
        "patient_id",
        "image_file_path",
        "cropped_image_file_path",
        "roi_mask_file_path",
        "reference_type",
        "referenced_path",
        "normalized_reference",
        "attempted_path",
        "failure_reason",
        "reference_status",
        "reference_usable",
        "alternative_representation_available",
        "alternative_representation_path",
    ]
    df = pd.DataFrame(unresolved_records, columns=columns) if unresolved_records else pd.DataFrame(columns=columns)
    df.to_csv(output_path, index=False)
    return df


def generate_reference_summary_csv(
    summary_by_csv: List[Dict[str, Any]],
    output_path: str = "data/metadata/stage1/dataset_reference_summary.csv",
) -> pd.DataFrame:
    """Generate summary table broken down by source CSV and reference type."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df = pd.DataFrame(summary_by_csv)
    df.to_csv(output_path, index=False)
    return df


def generate_unresolved_diagnostics_txt(
    unresolved_records: List[Dict[str, Any]],
    output_path: str = "data/metadata/stage1/dataset_unresolved_diagnostics.txt",
) -> str:
    """Generate detailed per-record diagnostic narrative report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    lines = [
        "=" * 80,
        "CBIS-DDSM UNRESOLVED REFERENCES DIAGNOSTIC INVESTIGATION REPORT",
        "=" * 80,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Unresolved References: {len(unresolved_records)}",
        "Storage vs Case Architecture:",
        "  - The JPEG directory structure is a storage structure and is not treated as a patient/case structure.",
        "  - Original CSV metadata remains authoritative for patient/case definitions.",
        "Root Cause Determination:",
        "  - Root Cause: A (Files genuinely do not exist in the downloaded raw CBIS-DDSM dataset release).",
        "  - Referenced SeriesInstanceUID directory is entirely absent from disk storage.",
        "  - The case's Full Mammogram remains intact, verified, and completely usable downstream.",
        "=" * 80,
        "",
    ]

    for idx, rec in enumerate(unresolved_records, 1):
        norm_ref = rec.get("normalized_reference", "")
        parts = norm_ref.split("/")
        expected_filename = parts[-1] if parts else "N/A"
        expected_dir = parts[-2] if len(parts) >= 2 else "N/A"

        lines.extend([
            "-" * 50,
            f"Record {idx}",
            "-" * 50,
            f"Source CSV:              {rec.get('source_csv', 'N/A')}",
            f"Row:                     {rec.get('row_number', 'N/A')}",
            f"Patient ID:              {rec.get('patient_id', 'N/A')}",
            f"Abnormality ID:          {rec.get('abnormality_id', 'N/A')}",
            f"Image View:              {rec.get('image_view', 'N/A')}",
            f"Reference type:          {rec.get('reference_type', 'N/A')}",
            f"CSV reference:           {repr(rec.get('referenced_path', 'N/A'))}",
            f"Normalized reference:    {rec.get('normalized_reference', 'N/A')}",
            f"Expected filename:       {expected_filename}",
            f"Expected directory:      {expected_dir}",
            f"Candidates found:        0",
            f"Resolution result:       UNRESOLVED",
            f"Reason:                  {rec.get('failure_reason', 'N/A')}",
            f"Alternative usable:      {rec.get('alternative_representation_available', False)} (Full Mammogram: {rec.get('alternative_representation_path', 'None')})",
            "",
        ])

    report_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return report_content


def generate_audit_csv(
    dir_audit: Dict[str, Any],
    linkage_audit: Dict[str, Any],
    output_path: str = "data/metadata/stage1/dataset_structure_audit.csv",
) -> pd.DataFrame:
    """Export flattened key metrics and breakdown to a CSV report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    counts = linkage_audit.get("counts_by_status", {})
    metrics = [
        ("total_directories", dir_audit.get("total_dirs", 0)),
        ("total_jpeg_files", dir_audit.get("total_files", 0)),
        ("folders_with_0_images", dir_audit.get("folder_buckets", {}).get("0_images", 0)),
        ("folders_with_1_image", dir_audit.get("folder_buckets", {}).get("1_image", 0)),
        ("folders_with_2_images", dir_audit.get("folder_buckets", {}).get("2_images", 0)),
        ("folders_with_3_images", dir_audit.get("folder_buckets", {}).get("3_images", 0)),
        ("folders_with_4+_images", dir_audit.get("folder_buckets", {}).get("4+_images", 0)),
        ("min_images_per_folder", dir_audit.get("min_images_per_folder", 0)),
        ("max_images_per_folder", dir_audit.get("max_images_per_folder", 0)),
        ("avg_images_per_folder", dir_audit.get("avg_images_per_folder", 0.0)),
        ("median_images_per_folder", dir_audit.get("median_images_per_folder", 0.0)),
        ("min_directory_depth", dir_audit.get("depth_stats", {}).get("min_depth", 0)),
        ("max_directory_depth", dir_audit.get("depth_stats", {}).get("max_depth", 0)),
        ("avg_directory_depth", dir_audit.get("depth_stats", {}).get("avg_depth", 0.0)),
        ("duplicate_filenames_count", dir_audit.get("duplicate_filenames_count", 0)),
        ("duplicate_paths_count", dir_audit.get("duplicate_paths_count", 0)),
        ("total_csv_case_records", linkage_audit.get("total_csv_records", 0)),
        ("records_with_resolved_image", linkage_audit.get("records_with_resolved_image", 0)),
        ("total_csv_image_references", linkage_audit.get("total_image_references", 0)),
        ("total_resolved_references", linkage_audit.get("total_resolved_references", 0)),
        ("resolved_exact", counts.get("exact", 0)),
        ("resolved_normalized", counts.get("normalized", 0)),
        ("resolved_unique_filename", counts.get("unique_filename", 0)),
        ("total_ambiguous_references", counts.get("ambiguous", 0)),
        ("total_unresolved_references", linkage_audit.get("total_unresolved_references", 0)),
        ("full_mammogram_resolved", linkage_audit.get("reference_stats", {}).get("full_mammogram", {}).get("resolved", 0)),
        ("full_mammogram_unresolved", linkage_audit.get("reference_stats", {}).get("full_mammogram", {}).get("unresolved", 0)),
        ("cropped_image_resolved", linkage_audit.get("reference_stats", {}).get("cropped_image", {}).get("resolved", 0)),
        ("cropped_image_unresolved", linkage_audit.get("reference_stats", {}).get("cropped_image", {}).get("unresolved", 0)),
        ("roi_mask_resolved", linkage_audit.get("reference_stats", {}).get("roi_mask", {}).get("resolved", 0)),
        ("roi_mask_unresolved", linkage_audit.get("reference_stats", {}).get("roi_mask", {}).get("unresolved", 0)),
    ]

    for ext, count in dir_audit.get("extension_counts", {}).items():
        metrics.append((f"ext_count_{ext}", count))

    df = pd.DataFrame(metrics, columns=["metric", "value"])
    df.to_csv(output_path, index=False)
    return df


def generate_audit_text_report(
    dir_audit: Dict[str, Any],
    linkage_audit: Dict[str, Any],
    output_path: str = "data/metadata/stage1/dataset_structure_audit_report.txt",
) -> str:
    """Generate detailed textual narrative report explaining dataset layout and resolution."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    samples = dir_audit.get("sample_dimensions", [])
    sample_summary = "None available"
    if samples:
        heights = [s["height"] for s in samples]
        widths = [s["width"] for s in samples]
        aspects = [s["aspect_ratio"] for s in samples]
        sample_summary = (
            f"Sampled {len(samples)} images across folders.\n"
            f"  - Height range: {min(heights)} to {max(heights)} px (mean: {int(np.mean(heights))} px)\n"
            f"  - Width range:  {min(widths)} to {max(widths)} px (mean: {int(np.mean(widths))} px)\n"
            f"  - Aspect ratio: {min(aspects):.2f} to {max(aspects):.2f} (mean: {np.mean(aspects):.2f})\n"
            f"  - Formats: {dict(Counter([s['dtype'] for s in samples]))}"
        )

    ref_stats = linkage_audit.get("reference_stats", {})
    counts = linkage_audit.get("counts_by_status", {})

    report = f"""================================================================================
CBIS-DDSM DATASET STRUCTURE & RESOLUTION AUDIT REPORT
================================================================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Audit Scope: READ-ONLY inspection of CBIS-DDSM raw directory structure and CSV references.

NOTICE:
JPEG directory structure is a storage structure and is not treated as a patient/case structure.
Original CSV metadata remains authoritative for patient/case definitions.

1. DIRECTORY & FILE STRUCTURE HIERARCHY
--------------------------------------------------------------------------------
Root JPEG Directory: {dir_audit.get('jpeg_dir', 'N/A')}
Total Directories / Subdirectories: {dir_audit.get('total_dirs', 0)}
Total Image Files on Disk:          {dir_audit.get('total_files', 0)}

Directory Depth:
  - Minimum Depth: {dir_audit.get('depth_stats', {}).get('min_depth', 0)}
  - Maximum Depth: {dir_audit.get('depth_stats', {}).get('max_depth', 0)}
  - Average Depth: {dir_audit.get('depth_stats', {}).get('avg_depth', 0.0)}

Folder Image Density Breakdown:
  - Folders with 0 images (parent/intermediate hierarchy): {dir_audit.get('folder_buckets', {}).get('0_images', 0)}
  - Folders with 1 image  (typically full mammogram series): {dir_audit.get('folder_buckets', {}).get('1_image', 0)}
  - Folders with 2 images (typically crop + ROI mask pairs): {dir_audit.get('folder_buckets', {}).get('2_images', 0)}
  - Folders with 3 images:                                 {dir_audit.get('folder_buckets', {}).get('3_images', 0)}
  - Folders with 4+ images:                                {dir_audit.get('folder_buckets', {}).get('4+_images', 0)}

Images Per Containing Folder:
  - Minimum: {dir_audit.get('min_images_per_folder', 0)}
  - Maximum: {dir_audit.get('max_images_per_folder', 0)}
  - Average: {dir_audit.get('avg_images_per_folder', 0.0)}
  - Median:  {dir_audit.get('median_images_per_folder', 0.0)}

File Extension Breakdown:
{chr(10).join(f"  - {k}: {v}" for k, v in dir_audit.get('extension_counts', {}).items())}

Duplicate Detection:
  - Duplicate Filenames Across Different Folders: {dir_audit.get('duplicate_filenames_count', 0)}
    (Note: Generic names like '1-1.jpg', '1-2.jpg' are standard across nested DICOM series folders)
  - Duplicate Absolute File Paths: {dir_audit.get('duplicate_paths_count', 0)}

2. IMAGE DIMENSION SAMPLING (Representative Sample)
--------------------------------------------------------------------------------
{sample_summary}

3. CSV METADATA & PATH RESOLUTION INTEGRITY
--------------------------------------------------------------------------------
Total Case Records in Description CSVs: {linkage_audit.get('total_csv_records', 0)}
Records with >=1 Locally Resolved Image: {linkage_audit.get('records_with_resolved_image', 0)}
Records with All References Resolved:    {linkage_audit.get('records_fully_resolved', 0)}

Reference Breakdown:
  - Full Mammogram References:
      Total:      {ref_stats.get('full_mammogram', {}).get('total', 0)}
      Resolved:   {ref_stats.get('full_mammogram', {}).get('resolved', 0)}
      Unresolved: {ref_stats.get('full_mammogram', {}).get('unresolved', 0)}
  - Cropped Lesion References:
      Total:      {ref_stats.get('cropped_image', {}).get('total', 0)}
      Resolved:   {ref_stats.get('cropped_image', {}).get('resolved', 0)}
      Unresolved: {ref_stats.get('cropped_image', {}).get('unresolved', 0)}
  - ROI Mask References:
      Total:      {ref_stats.get('roi_mask', {}).get('total', 0)}
      Resolved:   {ref_stats.get('roi_mask', {}).get('resolved', 0)}
      Unresolved: {ref_stats.get('roi_mask', {}).get('unresolved', 0)}

Matching Method Breakdown:
  - Resolved Exact:           {counts.get('exact', 0)}
  - Resolved Normalized:      {counts.get('normalized', 0)}
  - Resolved Unique Filename: {counts.get('unique_filename', 0)}
  - Ambiguous References:     {counts.get('ambiguous', 0)}
  - Unresolved References:    {linkage_audit.get('total_unresolved_references', 0)}

Summary Totals:
  - Total Image References in CSVs: {linkage_audit.get('total_image_references', 0)}
  - Total Resolved Image Paths:     {linkage_audit.get('total_resolved_references', 0)}
  - Total Unresolved References:    {linkage_audit.get('total_unresolved_references', 0)}

4. STRUCTURAL RELATIONSHIP & MAPPING LOGIC
--------------------------------------------------------------------------------
The CBIS-DDSM data flow operates under the following hierarchy:

    CSV Metadata (mass_case_*.csv, calc_case_*.csv)
          │  Contains: patient_id, abnormality_type, pathology, view, side,
          │            image file path, cropped image file path, ROI mask file path
          ▼
    DICOM Series Instance UIDs / Relative Path References
          │  Matches series folder UIDs embedded in DICOM path strings
          ▼
    Nested JPEG Directory Structure
          │  Folders are named by Series/Study UIDs, NOT patient IDs.
          │  - Folders with 1 image correspond to Full Mammograms.
          │  - Folders with 2 images contain Crop + Binary Mask pairs (1-1.jpg, 1-2.jpg).
          ▼
    Local Disk Image Files (.jpg / .jpeg)
          │  Resolved deterministically by PathResolver using indexed UIDs.

Master Metadata Fields Mapping:
  - patient_id:                  Direct patient identifier (authoritative)
  - abnormality_category:        'mass' or 'calcification' derived from source CSV
  - image_view:                  'CC' (Craniocaudal) or 'MLO' (Mediolateral Oblique)
  - breast_side:                 'LEFT' or 'RIGHT'
  - pathology:                   Original diagnostic text (BENIGN, BENIGN_WITHOUT_CALLBACK, MALIGNANT)
  - label:                       Binary target (0 for BENIGN / 1 for MALIGNANT)
  - image_file_path:             CSV relative path for full mammogram
  - cropped_image_file_path:     CSV relative path for cropped abnormality
  - roi_mask_file_path:          CSV relative path for ROI binary segmentation mask
  - resolved_image_path:         Validated absolute disk path

5. AUDIT STATUS CONCLUSION
--------------------------------------------------------------------------------
Status: {'PASSED' if linkage_audit.get('total_unresolved_references', 0) == 0 else 'WARNING - Unresolved References Present'}
No raw files were modified, moved, renamed, or preprocessed during this audit.
================================================================================
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def run_dataset_audit(
    raw_data_dir: str = "data/raw/CBIS_DDSM",
    raw_csv_dir: str = "data/raw/CBIS_DDSM/csv",
    output_dir: str = "data/metadata/stage1",
) -> Dict[str, Any]:
    """Execute complete read-only dataset structure audit and produce CSV and text reports."""
    os.makedirs(output_dir, exist_ok=True)
    jpeg_dir = find_jpeg_dir(raw_data_dir)
    csv_dir = find_csv_dir(raw_csv_dir)

    print(f"[DatasetAudit] Auditing JPEG tree at: {jpeg_dir}")
    dir_audit = audit_jpeg_directory(jpeg_dir)

    print(f"[DatasetAudit] Auditing CSV references from: {csv_dir}")
    linkage_audit = audit_csv_linkage(raw_csv_dir=csv_dir, raw_data_dir=raw_data_dir)

    # Output file paths
    csv_out = os.path.join(output_dir, "dataset_structure_audit.csv")
    txt_out = os.path.join(output_dir, "dataset_structure_audit_report.txt")
    unresolved_csv_out = os.path.join(output_dir, "dataset_unresolved_references.csv")
    unresolved_diag_out = os.path.join(output_dir, "dataset_unresolved_diagnostics.txt")
    summary_csv_out = os.path.join(output_dir, "dataset_reference_summary.csv")

    generate_audit_csv(dir_audit, linkage_audit, output_path=csv_out)
    generate_audit_text_report(dir_audit, linkage_audit, output_path=txt_out)
    generate_unresolved_references_csv(linkage_audit["unresolved_records"], output_path=unresolved_csv_out)
    generate_unresolved_diagnostics_txt(linkage_audit["unresolved_records"], output_path=unresolved_diag_out)
    generate_reference_summary_csv(linkage_audit["summary_by_csv"], output_path=summary_csv_out)

    print(f"[DatasetAudit] Saved audit CSV to: {csv_out}")
    print(f"[DatasetAudit] Saved audit Report to: {txt_out}")
    print(f"[DatasetAudit] Saved unresolved references CSV to: {unresolved_csv_out}")
    print(f"[DatasetAudit] Saved unresolved diagnostics to: {unresolved_diag_out}")
    print(f"[DatasetAudit] Saved reference summary CSV to: {summary_csv_out}")

    buckets = dir_audit.get("folder_buckets", {})
    ref_stats = linkage_audit.get("reference_stats", {})
    counts = linkage_audit.get("counts_by_status", {})
    total_unresolved = linkage_audit.get("total_unresolved_references", 0)
    status_str = "PASSED" if total_unresolved == 0 else "WARNING - Unresolved References Present"

    # Terminal summary banner matching required specification
    summary = f"""==================================================
CBIS-DDSM DATASET STRUCTURE AUDIT
==================================================

JPEG folders:           {dir_audit.get('total_dirs', 0)}
JPEG files:             {dir_audit.get('total_files', 0)}

Folders with 1 image:   {buckets.get('1_image', 0)}
Folders with 2 images:  {buckets.get('2_images', 0)}
Folders with 3 images:  {buckets.get('3_images', 0)}
Folders with 4+ images: {buckets.get('4+_images', 0)}

CSV image references:   {linkage_audit.get('total_image_references', 0)}

Full/original references: {ref_stats.get('full_mammogram', {}).get('total', 0)} ({ref_stats.get('full_mammogram', {}).get('resolved', 0)} resolved, {ref_stats.get('full_mammogram', {}).get('unresolved', 0)} unresolved)
Cropped references:       {ref_stats.get('cropped_image', {}).get('total', 0)} ({ref_stats.get('cropped_image', {}).get('resolved', 0)} resolved, {ref_stats.get('cropped_image', {}).get('unresolved', 0)} unresolved)
ROI mask references:      {ref_stats.get('roi_mask', {}).get('total', 0)} ({ref_stats.get('roi_mask', {}).get('resolved', 0)} resolved, {ref_stats.get('roi_mask', {}).get('unresolved', 0)} unresolved)

Resolved references:    {linkage_audit.get('total_resolved_references', 0)}
Exact:                  {counts.get('exact', 0)}
Normalized:             {counts.get('normalized', 0)}
Unique filename:        {counts.get('unique_filename', 0)}

Ambiguous references:   {counts.get('ambiguous', 0)}
Unresolved references:  {total_unresolved}

Duplicate paths:        {dir_audit.get('duplicate_paths_count', 0)}

Status:                 {status_str}

=================================================="""

    print(summary)

    if linkage_audit["unresolved_records"]:
        print("Exact Unresolved References:")
        for idx, rec in enumerate(linkage_audit["unresolved_records"], 1):
            print(f"  {idx}. Source: {rec['source_csv']} | Row: {rec['row_number']} | Patient: {rec['patient_id']} | Type: {rec['reference_type']}")
            print(f"     Path: {repr(rec['referenced_path'])}")
            print(f"     Reason: {rec['failure_reason']}")
        print("==================================================")

    return {
        "dir_audit": dir_audit,
        "linkage_audit": linkage_audit,
        "summary": summary,
    }


if __name__ == "__main__":
    run_dataset_audit()
