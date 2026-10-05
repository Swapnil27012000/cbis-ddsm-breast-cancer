"""Unit tests for CBIS-DDSM PathResolver.

Tests all 10 resolution requirements using temporary isolated test fixtures:
1. exact valid path
2. Windows separator
3. Linux separator
4. whitespace around path
5. unique filename
6. ambiguous filename
7. nonexistent file
8. invalid/empty reference
9. duplicate candidate filenames
10. normalized relative path
"""
import os
import pytest
from src.data.path_resolver import (
    PathResolver,
    normalize_path_string,
    STATUS_RESOLVED_EXACT,
    STATUS_RESOLVED_NORMALIZED,
    STATUS_RESOLVED_UNIQUE_FILENAME,
    STATUS_AMBIGUOUS,
    STATUS_UNRESOLVED,
    STATUS_INVALID_REFERENCE,
    METHOD_EXACT_RELATIVE_PATH,
    METHOD_NORMALIZED_RELATIVE_PATH,
    METHOD_UNIQUE_FILENAME,
    METHOD_DIRECTORY_AND_FILENAME,
    METHOD_NONE,
)


@pytest.fixture
def mock_dataset_env(tmp_path):
    """Create a structured temporary JPEG directory mock for testing."""
    jpeg_dir = tmp_path / "jpeg"
    jpeg_dir.mkdir()

    # Series 1: Single image (Full mammogram)
    series1 = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.111111111111"
    series1.mkdir()
    img1 = series1 / "full_mammo.jpg"
    img1.write_bytes(b"image_content_1")

    # Series 2: Crop + Mask pair
    series2 = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.222222222222"
    series2.mkdir()
    mask2 = series2 / "1-1.jpg"
    mask2.write_bytes(b"mask_content")
    crop2 = series2 / "1-2.jpg"
    crop2.write_bytes(b"crop_content")

    # Series 3: Unique named file
    series3 = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.333333333333"
    series3.mkdir()
    unique_img = series3 / "unique_anomaly_test.jpg"
    unique_img.write_bytes(b"unique_content")

    # Series 4: Subdirectory with ambiguous duplicate filename
    series4 = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.444444444444"
    series4.mkdir()
    ambig1 = series4 / "ambiguous_file.jpg"
    ambig1.write_bytes(b"ambig_1")

    series5 = jpeg_dir / "1.3.6.1.4.1.9590.100.1.2.555555555555"
    series5.mkdir()
    ambig2 = series5 / "ambiguous_file.jpg"
    ambig2.write_bytes(b"ambig_2")

    resolver = PathResolver(str(tmp_path))
    return {
        "root": tmp_path,
        "jpeg_dir": jpeg_dir,
        "resolver": resolver,
        "img1": img1,
        "unique_img": unique_img,
    }


def test_1_exact_valid_path(mock_dataset_env):
    """Test 1: Exact valid relative path on disk."""
    resolver = mock_dataset_env["resolver"]
    res = resolver.resolve("1.3.6.1.4.1.9590.100.1.2.111111111111/full_mammo.jpg")
    assert res.status in (STATUS_RESOLVED_EXACT, STATUS_RESOLVED_NORMALIZED)
    assert res.resolved_path == str(mock_dataset_env["img1"])
    assert res.candidate_count == 1


def test_2_windows_separator(mock_dataset_env):
    """Test 2: Windows backslash separators in reference path."""
    resolver = mock_dataset_env["resolver"]
    res = resolver.resolve("1.3.6.1.4.1.9590.100.1.2.111111111111\\full_mammo.jpg")
    assert res.status in (STATUS_RESOLVED_EXACT, STATUS_RESOLVED_NORMALIZED)
    assert res.resolved_path == str(mock_dataset_env["img1"])


def test_3_linux_separator(mock_dataset_env):
    """Test 3: Linux forward slash separators in reference path."""
    resolver = mock_dataset_env["resolver"]
    res = resolver.resolve("1.3.6.1.4.1.9590.100.1.2.111111111111/full_mammo.jpg")
    assert res.status in (STATUS_RESOLVED_EXACT, STATUS_RESOLVED_NORMALIZED)
    assert res.resolved_path == str(mock_dataset_env["img1"])


def test_4_whitespace_and_newlines(mock_dataset_env):
    """Test 4: Leading/trailing whitespace and embedded newlines (e.g. \\n in CBIS-DDSM CSV)."""
    resolver = mock_dataset_env["resolver"]
    res = resolver.resolve("  1.3.6.1.4.1.9590.100.1.2.111111111111/full_mammo.jpg\n  ")
    assert res.status == STATUS_RESOLVED_NORMALIZED
    assert res.resolved_path == str(mock_dataset_env["img1"])


def test_5_unique_filename(mock_dataset_env):
    """Test 5: Resolution via mathematically unique filename across the dataset."""
    resolver = mock_dataset_env["resolver"]
    res = resolver.resolve("unique_anomaly_test.jpg")
    assert res.status == STATUS_RESOLVED_UNIQUE_FILENAME
    assert res.resolved_path == str(mock_dataset_env["unique_img"])
    assert res.candidate_count == 1


def test_6_ambiguous_filename(mock_dataset_env):
    """Test 6: Ambiguous filename occurring in multiple directories must NOT silently match."""
    resolver = mock_dataset_env["resolver"]
    res = resolver.resolve("ambiguous_file.jpg")
    assert res.status == STATUS_AMBIGUOUS
    assert res.resolved_path is None
    assert res.candidate_count == 2


def test_7_nonexistent_file(mock_dataset_env):
    """Test 7: Entirely nonexistent path/file must return UNRESOLVED."""
    resolver = mock_dataset_env["resolver"]
    res = resolver.resolve("Calc-Training_P_99999_RIGHT_MLO/1.2.3.4.5/1.2.3.4.5.6/nonexistent.dcm")
    assert res.status == STATUS_UNRESOLVED
    assert res.resolved_path is None
    assert res.candidate_count == 0


def test_8_invalid_or_empty_reference(mock_dataset_env):
    """Test 8: Invalid, None, or empty references."""
    resolver = mock_dataset_env["resolver"]
    assert resolver.resolve("").status == STATUS_INVALID_REFERENCE
    assert resolver.resolve("   ").status == STATUS_INVALID_REFERENCE
    assert resolver.resolve(None).status == STATUS_INVALID_REFERENCE


def test_9_duplicate_candidate_filenames_in_same_series(mock_dataset_env):
    """Test 9: Differentiating crop vs ROI mask in a pair folder."""
    resolver = mock_dataset_env["resolver"]
    res_crop = resolver.resolve("1.3.6.1.4.1.9590.100.1.2.222222222222/000001.dcm", reference_type="CROPPED_IMAGE")
    res_mask = resolver.resolve("1.3.6.1.4.1.9590.100.1.2.222222222222/000000.dcm", reference_type="ROI_MASK")

    assert res_crop.status == STATUS_RESOLVED_NORMALIZED
    assert res_mask.status == STATUS_RESOLVED_NORMALIZED
    assert res_crop.resolved_path is not None
    assert res_mask.resolved_path is not None
    assert res_crop.resolved_path != res_mask.resolved_path


def test_10_normalized_relative_path_with_redundant_tokens(mock_dataset_env):
    """Test 10: Paths with redundant ./ and repeated slashes."""
    resolver = mock_dataset_env["resolver"]
    norm = normalize_path_string("./CBIS-DDSM//jpeg///1.3.6.1.4.1.9590.100.1.2.111111111111/./full_mammo.jpg\r\n")
    assert norm == "CBIS-DDSM/jpeg/1.3.6.1.4.1.9590.100.1.2.111111111111/full_mammo.jpg"

    res = resolver.resolve("./CBIS-DDSM//jpeg///1.3.6.1.4.1.9590.100.1.2.111111111111/./full_mammo.jpg\r\n")
    assert res.status == STATUS_RESOLVED_NORMALIZED
    assert res.resolved_path == str(mock_dataset_env["img1"])
