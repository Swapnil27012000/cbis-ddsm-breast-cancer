"""Region of Interest (ROI) extraction and pectoral muscle suppression."""
import cv2
import numpy as np
from typing import Tuple

def segment_breast(img: np.ndarray) -> np.ndarray:
    """Segment breast tissue from background using Otsu thresholding & largest connected component."""
    if img.dtype != np.uint8:
        # Scale to 8-bit for morphological operations
        img_8u = ((img - np.min(img)) / (np.ptp(img) + 1e-7) * 255).astype(np.uint8)
    else:
        img_8u = img

    _, thresh = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return np.ones_like(img_8u)

    # Exclude background (label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_label).astype(np.uint8)
    return mask

def remove_pectoral_muscle(img: np.ndarray, roi_fraction: float = 0.4) -> np.ndarray:
    """Suppress high-intensity triangular pectoral muscle in upper MLO view corner."""
    out = img.copy()
    h, w = out.shape[:2]
    max_h = int(h * roi_fraction)
    max_w = int(w * roi_fraction)

    # Suppress top-left corner pectoral candidate
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[0, 0], [max_w, 0], [0, max_h]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    
    # Apply soft attenuation or zero-fill
    out[mask == 1] = 0
    return out

def crop_to_roi(img: np.ndarray, mask: np.ndarray, margin: int = 10) -> np.ndarray:
    """Crop image to bounding box of the active breast mask."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = img.shape[:2]
    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(w_img, x + w + margin)
    y_max = min(h_img, y + h + margin)
    return img[y_min:y_max, x_min:x_max]
