"""Path resolver for linking CBIS-DDSM CSV records with physical JPEG files.

Implements safe, multi-level path resolution with normalization, disambiguation,
and structured diagnostic metadata.
"""
from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# Status Constants
STATUS_RESOLVED_EXACT = "RESOLVED_EXACT"
STATUS_RESOLVED_NORMALIZED = "RESOLVED_NORMALIZED"
STATUS_RESOLVED_UNIQUE_FILENAME = "RESOLVED_UNIQUE_FILENAME"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_INVALID_REFERENCE = "INVALID_REFERENCE"

# Matching Method Constants
METHOD_EXACT_RELATIVE_PATH = "EXACT_RELATIVE_PATH"
METHOD_NORMALIZED_RELATIVE_PATH = "NORMALIZED_RELATIVE_PATH"
METHOD_UNIQUE_FILENAME = "UNIQUE_FILENAME"
METHOD_DIRECTORY_AND_FILENAME = "DIRECTORY_AND_FILENAME"
METHOD_DICOM_INFO_SERIES_MATCH = "DICOM_INFO_SERIES_MATCH"
METHOD_NONE = "NONE"


@dataclass
class ResolutionResult:
    """Structured result returned by PathResolver."""
    status: str
    resolved_path: Optional[str] = None
    reference_type: Optional[str] = None
    matching_method: str = METHOD_NONE
    candidate_count: int = 0
    reason: str = ""
    candidates: List[str] = field(default_factory=list)
    normalized_path: str = ""
    attempted_path: str = ""


def normalize_path_string(path_str: Any) -> str:
    """Safely normalize Windows/Unix path strings.

    Handles:
      - leading/trailing whitespace
      - trailing/embedded newlines (\\n, \\r)
      - Windows vs Unix separators (\\ vs /)
      - repeated separators (//, \\\\)
      - escaped separators
      - redundant ./ components
    """
    if path_str is None or pd.isna(path_str):
        return ""
    if not isinstance(path_str, str):
        path_str = str(path_str)

    # Strip whitespace, newlines, carriage returns
    s = path_str.strip().replace("\r", "").replace("\n", "")
    # Unescape and replace backslashes with forward slashes
    s = s.replace("\\\\", "/").replace("\\", "/")
    # Remove repeated slashes
    s = re.sub(r"/+", "/", s)
    # Split and remove empty and redundant ./ segments
    parts = s.split("/")
    cleaned_parts = [p.strip() for p in parts if p.strip() and p.strip() != "."]
    return "/".join(cleaned_parts)


class PathResolver:
    """Resolves DICOM/JPEG paths in CBIS-DDSM CSVs to existing local disk paths."""

    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = os.path.abspath(raw_data_dir)
        self.jpeg_dir = os.path.join(self.raw_data_dir, "jpeg")
        if not os.path.exists(self.jpeg_dir):
            for cand in [
                "dataset/jpeg",
                "data/raw/CBIS_DDSM/jpeg",
                os.path.join(os.getcwd(), "dataset", "jpeg"),
                os.path.join(os.getcwd(), "data", "raw", "CBIS_DDSM", "jpeg"),
                "/app/dataset/jpeg",
                "/app/data/raw/CBIS_DDSM/jpeg",
            ]:
                if os.path.exists(cand) and os.path.isdir(cand):
                    self.jpeg_dir = os.path.abspath(cand)
                    break

        self.csv_dir = os.path.join(self.raw_data_dir, "csv")
        if not os.path.exists(self.csv_dir):
            for cand in [
                "dataset/csv",
                "data/raw/CBIS_DDSM/csv",
                os.path.join(os.getcwd(), "dataset", "csv"),
                os.path.join(os.getcwd(), "data", "raw", "CBIS_DDSM", "csv"),
                "/app/dataset/csv",
                "/app/data/raw/CBIS_DDSM/csv",
            ]:
                if os.path.exists(cand) and os.path.isdir(cand):
                    self.csv_dir = os.path.abspath(cand)
                    break

        # Directory and filename lookup structures
        self._folder_files: Dict[str, List[str]] = {}
        self._filename_to_paths: Dict[str, List[str]] = {}
        self._dicom_info_lookup: Dict[str, str] = {}
        self._build_index()

    def _build_index(self):
        """Index available JPEG directories and files under jpeg_dir."""
        if not os.path.exists(self.jpeg_dir):
            return

        for root, _, files in os.walk(self.jpeg_dir):
            img_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".dcm"))]
            if img_files:
                img_files.sort()
                folder_name = os.path.basename(root)
                full_paths = [os.path.join(root, f) for f in img_files]
                self._folder_files[folder_name] = full_paths

                for f, p in zip(img_files, full_paths):
                    if f not in self._filename_to_paths:
                        self._filename_to_paths[f] = []
                    self._filename_to_paths[f].append(p)

        # Index dicom_info.csv if available
        dicom_info_path = os.path.join(self.csv_dir, "dicom_info.csv") if self.csv_dir else None
        if dicom_info_path and os.path.exists(dicom_info_path):
            try:
                df_dicom = pd.read_csv(dicom_info_path, usecols=["image_path", "SeriesInstanceUID", "SOPInstanceUID"], dtype=str)
                for _, r in df_dicom.iterrows():
                    img_rel = str(r.get("image_path", "")).strip().replace("\\", "/")
                    if "jpeg/" in img_rel:
                        img_rel = img_rel.split("jpeg/")[-1]
                    cand_path = os.path.join(self.jpeg_dir, img_rel)
                    if os.path.exists(cand_path):
                        ser_uid = str(r.get("SeriesInstanceUID", "")).strip()
                        sop_uid = str(r.get("SOPInstanceUID", "")).strip()
                        if ser_uid and ser_uid not in self._dicom_info_lookup:
                            self._dicom_info_lookup[ser_uid] = cand_path
                        if sop_uid and sop_uid not in self._dicom_info_lookup:
                            self._dicom_info_lookup[sop_uid] = cand_path
            except Exception:
                pass

    def resolve(self, path_str: Any, reference_type: Optional[str] = None) -> ResolutionResult:
        """Resolve a CSV image path string using progressively safer matching levels.

        Args:
            path_str: Raw path string from CSV reference.
            reference_type: Type of reference ('FULL_IMAGE', 'CROPPED_IMAGE', 'ROI_MASK', etc.)

        Returns:
            ResolutionResult detailing status, resolved path, matching method, and diagnostics.
        """
        # Step 0: Validate input
        if path_str is None or pd.isna(path_str):
            return ResolutionResult(
                status=STATUS_INVALID_REFERENCE,
                reference_type=reference_type,
                matching_method=METHOD_NONE,
                candidate_count=0,
                reason="Empty or null reference string",
            )

        raw_str = str(path_str)
        norm_str = normalize_path_string(raw_str)
        if not norm_str:
            return ResolutionResult(
                status=STATUS_INVALID_REFERENCE,
                reference_type=reference_type,
                matching_method=METHOD_NONE,
                candidate_count=0,
                reason="Path string contains only whitespace or empty separators",
            )

        # LEVEL 1: Exact relative path matching
        direct_cand = os.path.join(self.jpeg_dir, norm_str)
        if os.path.exists(direct_cand) and os.path.isfile(direct_cand):
            method = METHOD_EXACT_RELATIVE_PATH if raw_str == norm_str else METHOD_NORMALIZED_RELATIVE_PATH
            status = STATUS_RESOLVED_EXACT if method == METHOD_EXACT_RELATIVE_PATH else STATUS_RESOLVED_NORMALIZED
            return ResolutionResult(
                status=status,
                resolved_path=os.path.abspath(direct_cand),
                reference_type=reference_type,
                matching_method=method,
                candidate_count=1,
                reason="Exact relative path exists on disk",
                candidates=[os.path.abspath(direct_cand)],
                normalized_path=norm_str,
                attempted_path=direct_cand,
            )

        # Check stripped prefix relative paths (e.g. if prefixed with CBIS-DDSM/jpeg/ or jpeg/)
        for pfx in ["cbis-ddsm/jpeg/", "jpeg/", "data/raw/cbis_ddsm/jpeg/", "dataset/jpeg/"]:
            norm_lower = norm_str.lower()
            if pfx in norm_lower:
                idx = norm_lower.find(pfx) + len(pfx)
                sub_rel = norm_str[idx:]
                sub_cand = os.path.join(self.jpeg_dir, sub_rel)
                if os.path.exists(sub_cand) and os.path.isfile(sub_cand):
                    return ResolutionResult(
                        status=STATUS_RESOLVED_NORMALIZED,
                        resolved_path=os.path.abspath(sub_cand),
                        reference_type=reference_type,
                        matching_method=METHOD_NORMALIZED_RELATIVE_PATH,
                        candidate_count=1,
                        reason=f"Matched relative path after stripping prefix '{pfx}'",
                        candidates=[os.path.abspath(sub_cand)],
                        normalized_path=norm_str,
                        attempted_path=sub_cand,
                    )

        # LEVEL 2: Directory name matching (SeriesInstanceUID folder)
        parts = norm_str.split("/")
        target_filename = parts[-1] if parts else ""

        # Search if any path segment corresponds to an indexed directory in jpeg_dir
        matched_folder_key = None
        for p in parts:
            if p in self._folder_files:
                matched_folder_key = p
                break

        if matched_folder_key:
            folder_imgs = self._folder_files[matched_folder_key]
            # Case A: exactly 1 image in the matched folder
            if len(folder_imgs) == 1:
                return ResolutionResult(
                    status=STATUS_RESOLVED_NORMALIZED,
                    resolved_path=folder_imgs[0],
                    reference_type=reference_type,
                    matching_method=METHOD_DIRECTORY_AND_FILENAME,
                    candidate_count=1,
                    reason=f"Matched SeriesInstanceUID directory '{matched_folder_key}' with 1 image",
                    candidates=folder_imgs,
                    normalized_path=norm_str,
                    attempted_path=folder_imgs[0],
                )
            # Case B: multiple images in the matched folder (crop + mask pair)
            elif len(folder_imgs) > 1:
                # Check for exact filename match within folder
                exact_in_folder = [f for f in folder_imgs if os.path.basename(f).lower() == target_filename.lower()]
                if len(exact_in_folder) == 1:
                    return ResolutionResult(
                        status=STATUS_RESOLVED_EXACT,
                        resolved_path=exact_in_folder[0],
                        reference_type=reference_type,
                        matching_method=METHOD_DIRECTORY_AND_FILENAME,
                        candidate_count=1,
                        reason=f"Exact filename '{target_filename}' in directory '{matched_folder_key}'",
                        candidates=exact_in_folder,
                        normalized_path=norm_str,
                        attempted_path=exact_in_folder[0],
                    )

                # Differentiate between cropped image and ROI mask in a 2-image folder:
                # In CBIS-DDSM pairs, mask is binary (often sorted first or 000000.dcm) and crop is abnormality (000001.dcm)
                ref_type_lower = str(reference_type).lower() if reference_type else ""
                if "roi" in ref_type_lower or "mask" in ref_type_lower or target_filename.startswith("000000"):
                    chosen = folder_imgs[0]
                elif "crop" in ref_type_lower or target_filename.startswith("000001"):
                    chosen = folder_imgs[1] if len(folder_imgs) > 1 else folder_imgs[0]
                else:
                    chosen = folder_imgs[0]

                return ResolutionResult(
                    status=STATUS_RESOLVED_NORMALIZED,
                    resolved_path=chosen,
                    reference_type=reference_type,
                    matching_method=METHOD_DIRECTORY_AND_FILENAME,
                    candidate_count=len(folder_imgs),
                    reason=f"Resolved via SeriesInstanceUID directory '{matched_folder_key}' (pair selection)",
                    candidates=folder_imgs,
                    normalized_path=norm_str,
                    attempted_path=chosen,
                )

        # LEVEL 3: Exact filename matching across dataset (only if mathematically unique)
        generic_names = {"000000.dcm", "000001.dcm", "1-1.jpg", "1-2.jpg", "1-001.jpg"}
        if target_filename and target_filename.lower() not in generic_names:
            if target_filename in self._filename_to_paths:
                candidates = self._filename_to_paths[target_filename]
                if len(candidates) == 1:
                    return ResolutionResult(
                        status=STATUS_RESOLVED_UNIQUE_FILENAME,
                        resolved_path=candidates[0],
                        reference_type=reference_type,
                        matching_method=METHOD_UNIQUE_FILENAME,
                        candidate_count=1,
                        reason=f"Exact unique filename '{target_filename}' found across dataset",
                        candidates=candidates,
                        normalized_path=norm_str,
                        attempted_path=candidates[0],
                    )
                elif len(candidates) > 1:
                    return ResolutionResult(
                        status=STATUS_AMBIGUOUS,
                        resolved_path=None,
                        reference_type=reference_type,
                        matching_method=METHOD_NONE,
                        candidate_count=len(candidates),
                        reason=f"Ambiguous: {len(candidates)} files share filename '{target_filename}'",
                        candidates=candidates[:5],
                        normalized_path=norm_str,
                        attempted_path=candidates[0],
                    )

        # LEVEL 4: DICOM metadata UID lookup
        for p in parts:
            if p in self._dicom_info_lookup:
                cand_path = self._dicom_info_lookup[p]
                if os.path.exists(cand_path):
                    return ResolutionResult(
                        status=STATUS_RESOLVED_NORMALIZED,
                        resolved_path=cand_path,
                        reference_type=reference_type,
                        matching_method=METHOD_DICOM_INFO_SERIES_MATCH,
                        candidate_count=1,
                        reason=f"Resolved via dicom_info.csv UID mapping for '{p}'",
                        candidates=[cand_path],
                        normalized_path=norm_str,
                        attempted_path=cand_path,
                    )

        # LEVEL 5: Unresolved
        attempted = os.path.join(self.jpeg_dir, norm_str)
        return ResolutionResult(
            status=STATUS_UNRESOLVED,
            resolved_path=None,
            reference_type=reference_type,
            matching_method=METHOD_NONE,
            candidate_count=0,
            reason=f"Referenced SeriesInstanceUID directory and image file genuinely do not exist in raw JPEG storage",
            candidates=[],
            normalized_path=norm_str,
            attempted_path=attempted,
        )

    def resolve_image_path(self, path_str: Any, reference_type: Optional[str] = None) -> Optional[str]:
        """Convert a path recorded in CSV to an existing local JPEG file path.

        Maintains full backward-compatibility with downstream modules.
        """
        result = self.resolve(path_str, reference_type=reference_type)
        return result.resolved_path
