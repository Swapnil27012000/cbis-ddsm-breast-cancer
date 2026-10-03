"""Image indexing module to map SOP Instance / Patient IDs to file locations."""
import os
from typing import Dict, List

def build_image_index(jpeg_root: str) -> Dict[str, str]:
    """Scan the JPEG directory and build a lookup table of folder names / files.

    Args:
        jpeg_root: Root path of JPEG images.

    Returns:
        Dict mapping directory/filename keys to absolute file paths.
    """
    index = {}
    if not os.path.exists(jpeg_root):
        return index

    for root, _, files in os.walk(jpeg_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, f)
                rel_dir = os.path.basename(root)
                index[f] = full_path
                index[rel_dir] = full_path

    return index
