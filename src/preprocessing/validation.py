"""Mammogram integrity and quality validation."""
import numpy as np
from typing import Tuple, Dict

def validate_mammogram(img: np.ndarray) -> Tuple[bool, Dict[str, any]]:
    """Validate image dimensions, pixel values, and lack of NaN/Inf.

    Args:
        img: Input image as numpy array.

    Returns:
        (is_valid, report_dict)
    """
    report = {
        "shape": img.shape if img is not None else None,
        "dtype": str(img.dtype) if img is not None else None,
        "min": float(np.min(img)) if img is not None and img.size > 0 else None,
        "max": float(np.max(img)) if img is not None and img.size > 0 else None,
        "has_nan": bool(np.isnan(img).any()) if img is not None else True,
        "has_inf": bool(np.isinf(img).any()) if img is not None else True,
        "is_constant": bool(np.min(img) == np.max(img)) if img is not None and img.size > 0 else True,
    }

    is_valid = (
        img is not None
        and img.size > 0
        and len(img.shape) in (2, 3)
        and not report["has_nan"]
        and not report["has_inf"]
        and not report["is_constant"]
    )

    report["valid"] = is_valid
    return is_valid, report
