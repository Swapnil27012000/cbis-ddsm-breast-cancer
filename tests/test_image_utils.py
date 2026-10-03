"""Unit tests for reusable image-loading and manipulation utilities."""
import os
import pytest
import numpy as np
import cv2

from src.utils.image_utils import (
    load_grayscale_image,
    normalize_image,
    resize_image,
    save_image,
    get_image_statistics,
)

@pytest.fixture
def sample_image(tmp_path):
    """Fixture providing a temporary 100x150 grayscale test image."""
    arr = np.linspace(0, 255, 100 * 150, dtype=np.uint8).reshape((100, 150))
    file_path = tmp_path / "test_mammogram.png"
    cv2.imwrite(str(file_path), arr)
    return str(file_path), arr

def test_load_grayscale_image_valid(sample_image):
    file_path, original = sample_image
    loaded = load_grayscale_image(file_path)
    assert loaded.ndim == 2
    assert loaded.shape == original.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, original)

def test_load_grayscale_image_missing_file():
    with pytest.raises(FileNotFoundError):
        load_grayscale_image("non_existent_file_path.png")

def test_normalize_image_range(sample_image):
    _, original = sample_image
    norm = normalize_image(original)
    assert norm.dtype == np.float32
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()

def test_normalize_image_constant():
    constant_img = np.full((50, 50), 128, dtype=np.uint8)
    norm = normalize_image(constant_img, target_min=0.0, target_max=1.0)
    assert norm.shape == constant_img.shape
    assert np.all(norm == 0.0)
    assert not np.isnan(norm).any()

def test_resize_image(sample_image):
    _, original = sample_image
    # Target size: (width=64, height=48)
    resized = resize_image(original, target_size=(64, 48))
    assert resized.shape == (48, 64)
    assert resized.ndim == 2

def test_save_image_float_to_uint8(tmp_path):
    float_img = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    out_path = tmp_path / "subdir" / "saved_float.png"
    saved_path = save_image(float_img, str(out_path))
    assert os.path.exists(saved_path)
    # Reload and inspect
    reloaded = cv2.imread(saved_path, cv2.IMREAD_GRAYSCALE)
    assert reloaded.dtype == np.uint8
    assert reloaded[0, 0] == 0
    assert reloaded[1, 1] == 255

def test_save_image_blocks_raw_directory(tmp_path):
    img = np.zeros((10, 10), dtype=np.uint8)
    forbidden_path = os.path.join(tmp_path, "data", "raw", "overwritten.png")
    with pytest.raises(PermissionError):
        save_image(img, forbidden_path)

def test_get_image_statistics(sample_image):
    _, original = sample_image
    stats = get_image_statistics(original)
    assert stats["width"] == 150
    assert stats["height"] == 100
    assert stats["channels"] == 1
    assert stats["dtype"] == "uint8"
    assert stats["min"] == 0.0
    assert stats["max"] == 255.0
    assert isinstance(stats["mean"], float)
    assert isinstance(stats["std"], float)
