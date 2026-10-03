"""Reusable image loading, normalization, resizing, and saving utilities for CBIS-DDSM."""
import os
from typing import Dict, Any, Tuple, Union, Optional
import cv2
import numpy as np

def load_grayscale_image(file_path: Union[str, os.PathLike]) -> np.ndarray:
    """Load an image file safely from disk in 8-bit grayscale format.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        np.ndarray: 2D NumPy array of dtype uint8 representing the grayscale image.

    Raises:
        FileNotFoundError: If the specified file does not exist on disk.
        ValueError: If the file exists but cannot be decoded by OpenCV.
    """
    path_str = os.fspath(file_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Image file not found: {path_str}")

    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to decode image file at: {path_str}. The image may be corrupted or in an unsupported format.")

    return img

# Alias for backward compatibility
load_image = load_grayscale_image

def normalize_image(
    img: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0
) -> np.ndarray:
    """Safely convert and normalize pixel intensities to [target_min, target_max].

    Handles constant or zero-variance images safely without generating NaN or Inf.

    Args:
        img: Input NumPy image array.
        target_min: Lower bound of normalized dynamic range (default: 0.0).
        target_max: Upper bound of normalized dynamic range (default: 1.0).

    Returns:
        np.ndarray: float32 NumPy array with values within [target_min, target_max].
    """
    if img is None or img.size == 0:
        raise ValueError("Input image array is empty or None.")

    img_float = img.astype(np.float32)
    # Remove any pre-existing NaN or Inf values
    img_float = np.nan_to_num(img_float, nan=0.0, posinf=target_max, neginf=target_min)

    min_val = float(np.min(img_float))
    max_val = float(np.max(img_float))
    val_range = max_val - min_val

    # Constant or uniform image safeguard
    if val_range <= 1e-7:
        return np.full_like(img_float, target_min, dtype=np.float32)

    normalized = (img_float - min_val) / val_range
    scaled = normalized * (target_max - target_min) + target_min
    return scaled.astype(np.float32)

def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: Optional[int] = None
) -> np.ndarray:
    """Resize an image to target dimensions (width, height) while preserving grayscale dimensionality.

    Uses cv2.INTER_AREA for decimation (downsampling) and cv2.INTER_LINEAR for zoom (upsampling)
    unless explicitly overridden.

    Args:
        img: Input NumPy array (H, W) or (H, W, 1).
        target_size: Tuple of (target_width, target_height).
        interpolation: Optional OpenCV interpolation flag. Defaults to INTER_AREA when shrinking,
                       and INTER_LINEAR when enlarging.

    Returns:
        np.ndarray: Resized 2D NumPy array.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image array is empty or None.")

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target dimensions: {target_size}. Width and height must be positive.")

    src_h, src_w = img.shape[:2]

    # Select optimal interpolation if not specified
    if interpolation is None:
        if target_w < src_w or target_h < src_h:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # Ensure grayscale images remain 2D
    if resized.ndim == 3 and resized.shape[2] == 1:
        resized = resized.squeeze(axis=2)

    return resized

def save_image(img: np.ndarray, file_path: Union[str, os.PathLike]) -> str:
    """Save an image to disk safely, creating parent directories and preventing raw data overwrites.

    Floating-point images in [0, 1] are converted to 8-bit uint8 [0, 255] automatically.

    Args:
        img: NumPy image array to save.
        file_path: Destination path on disk.

    Returns:
        str: Absolute path of the saved file.

    Raises:
        PermissionError: If destination path is located within data/raw/.
        ValueError: If image cannot be written to disk.
    """
    abs_path = os.path.abspath(os.fspath(file_path))
    normalized_path = abs_path.replace("\\", "/").lower()

    # Rule safeguard: NEVER overwrite or save into data/raw/
    if "/data/raw/" in normalized_path or normalized_path.endswith("/data/raw"):
        raise PermissionError(f"Security restriction: Writing or modifying files inside data/raw/ is strictly prohibited ({file_path}).")

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    # Format conversion
    if np.issubdtype(img.dtype, np.floating):
        # Clip to [0, 1] or appropriate range
        clipped = np.clip(img, 0.0, 1.0)
        out_img = (clipped * 255.0).round().astype(np.uint8)
    else:
        out_img = np.clip(img, 0, 255).astype(np.uint8)

    success = cv2.imwrite(abs_path, out_img)
    if not success:
        raise IOError(f"Failed to write image file to disk: {abs_path}")

    return abs_path

def get_image_statistics(img: np.ndarray) -> Dict[str, Any]:
    """Calculate and return fundamental descriptive statistics for an image.

    Args:
        img: Input NumPy image array.

    Returns:
        Dict[str, Any] containing:
            - width (int)
            - height (int)
            - channels (int)
            - dtype (str)
            - min (float)
            - max (float)
            - mean (float)
            - std (float)
    """
    if img is None or img.size == 0:
        raise ValueError("Input image array is empty or None.")

    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]

    return {
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "dtype": str(img.dtype),
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
    }

def print_image_inspection(image_path: str) -> None:
    """CLI utility to load an image, compute statistics, and print a formatted summary."""
    print("=" * 55)
    print(f"CBIS-DDSM Image Inspection: {os.path.basename(image_path)}")
    print("=" * 55)
    try:
        img = load_grayscale_image(image_path)
        stats = get_image_statistics(img)
        print(f"File Path    : {image_path}")
        print(f"Dimensions   : {stats['width']} x {stats['height']} (Width x Height)")
        print(f"Channels     : {stats['channels']}")
        print(f"Data Type    : {stats['dtype']}")
        print(f"Min Value    : {stats['min']}")
        print(f"Max Value    : {stats['max']}")
        print(f"Mean Value   : {stats['mean']:.2f}")
        print(f"Std Dev      : {stats['std']:.2f}")
    except Exception as e:
        print(f"Error inspecting image: {e}")
    print("=" * 55)

if __name__ == "__main__":
    import sys
    import pandas as pd

    # Default to first available image in master metadata if no path provided
    target_path = None
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        meta_candidates = [
            "data/metadata/CBIS_DDSM_master_metadata.csv",
            "/app/data/metadata/CBIS_DDSM_master_metadata.csv",
        ]
        for m in meta_candidates:
            if os.path.exists(m):
                df = pd.read_csv(m)
                valid = df[df.get("file_exists", False) == True]
                if not valid.empty:
                    target_path = valid.iloc[0]["image_path"]
                    break

    if target_path and os.path.exists(target_path):
        print_image_inspection(target_path)
    else:
        print(f"Target image path not found or specified ({target_path}). Pass an image path as argument.")
