"""Unit tests for metadata builder and standardizer."""
import pytest
from src.data.metadata_builder import standardize_pathology

def test_standardize_pathology():
    assert standardize_pathology("BENIGN") == "BENIGN"
    assert standardize_pathology("BENIGN_WITHOUT_CALLBACK") == "BENIGN"
    assert standardize_pathology("MALIGNANT") == "MALIGNANT"
    assert standardize_pathology("UNKNOWN_LABEL") == "UNKNOWN"
