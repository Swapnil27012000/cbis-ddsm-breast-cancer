"""Unit tests for CSV loading and path resolving."""
import pytest
import os
import pandas as pd
from src.data.path_resolver import PathResolver

def test_path_resolver_creation():
    resolver = PathResolver(raw_data_dir="data/raw/CBIS_DDSM")
    assert resolver is not None
    assert os.path.isabs(resolver.raw_data_dir)
