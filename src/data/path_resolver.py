"""Path resolver for linking CBIS-DDSM CSV records with physical JPEG files."""
import os
import glob
from typing import Optional, Dict
import pandas as pd

class PathResolver:
    """Resolves DICOM/JPEG paths in CBIS-DDSM CSVs to existing local disk paths."""

    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = os.path.abspath(raw_data_dir)
        self.jpeg_dir = os.path.join(self.raw_data_dir, "jpeg")
        if not os.path.exists(self.jpeg_dir):
            for cand in ["dataset/jpeg", "data/raw/CBIS_DDSM/jpeg", "/app/dataset/jpeg", "/app/data/raw/CBIS_DDSM/jpeg"]:
                if os.path.exists(cand):
                    self.jpeg_dir = os.path.abspath(cand)
                    break
        self._lookup: Dict[str, str] = {}
        self._build_quick_index()

    def _build_quick_index(self):
        """Index available JPEG directories under raw/CBIS_DDSM/jpeg."""
        if not os.path.exists(self.jpeg_dir):
            return

        for folder_name in os.listdir(self.jpeg_dir):
            folder_path = os.path.join(self.jpeg_dir, folder_name)
            if os.path.isdir(folder_path):
                # Map folder name (SeriesInstanceUID) to first contained image
                files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if files:
                    # Sort so standard 1-xxx.jpg is chosen deterministically
                    files.sort()
                    self._lookup[folder_name] = os.path.join(folder_path, files[0])

    def resolve_image_path(self, path_str: str) -> Optional[str]:
        """Convert a path recorded in the CSV to an existing local JPEG file path.

        Args:
            path_str: Path string from case description or dicom_info CSV.

        Returns:
            Resolved absolute path if file exists, else None.
        """
        if not path_str or pd.isna(path_str):
            return None

        clean_str = str(path_str).strip().replace("\\", "/")
        parts = clean_str.split("/")

        # Check if any segment matches a known folder in jpeg/
        for p in parts:
            if p in self._lookup:
                return self._lookup[p]

        # Check direct path in jpeg_dir
        candidate = os.path.join(self.jpeg_dir, clean_str)
        if os.path.exists(candidate):
            return candidate

        # Check candidate without prefix (e.g. if starts with CBIS-DDSM/jpeg/)
        if "jpeg/" in clean_str:
            rel = clean_str.split("jpeg/")[-1]
            cand = os.path.join(self.jpeg_dir, rel)
            if os.path.exists(cand):
                return cand

        return None
