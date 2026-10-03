"""Unit tests for mammogram validation and normalization."""
import numpy as np
import pytest
from src.preprocessing.validation import validate_mammogram
from src.preprocessing.normalization import min_max_normalize, robust_min_max_normalize

def test_validate_mammogram():
    valid_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    is_valid, report = validate_mammogram(valid_img)
    assert is_valid is True
    assert report["has_nan"] is False

    invalid_img = np.full((100, 100), np.nan)
    is_valid_nan, report_nan = validate_mammogram(invalid_img)
    assert is_valid_nan is False
    assert report_nan["has_nan"] is True

def test_normalization():
    img = np.array([[10, 20], [30, 40]], dtype=np.float32)
    norm = min_max_normalize(img, 0.0, 1.0)
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)
