"""Intensity normalization algorithms for mammograms."""
import numpy as np

def min_max_normalize(
    img: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0
) -> np.ndarray:
    """Standard Min-Max normalization to [target_min, target_max]."""
    img_f = img.astype(np.float32)
    min_val, max_val = np.min(img_f), np.max(img_f)
    if max_val - min_val == 0:
        return np.zeros_like(img_f)
    norm = (img_f - min_val) / (max_val - min_val)
    return norm * (target_max - target_min) + target_min

def robust_min_max_normalize(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    target_min: float = 0.0,
    target_max: float = 1.0
) -> np.ndarray:
    """Robust percentile-based Min-Max normalization discarding extreme artifacts."""
    img_f = img.astype(np.float32)
    v_min = np.percentile(img_f, p_low)
    v_max = np.percentile(img_f, p_high)
    if v_max - v_min == 0:
        return np.zeros_like(img_f)
    clipped = np.clip(img_f, v_min, v_max)
    norm = (clipped - v_min) / (v_max - v_min)
    return norm * (target_max - target_min) + target_min

def z_score_normalize(img: np.ndarray) -> np.ndarray:
    """Z-score normalization (zero mean, unit variance)."""
    img_f = img.astype(np.float32)
    mean = np.mean(img_f)
    std = np.std(img_f)
    if std == 0:
        return np.zeros_like(img_f)
    return (img_f - mean) / std
