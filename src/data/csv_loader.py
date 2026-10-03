"""CSV loader for raw CBIS-DDSM dataset files."""
import os
from typing import Dict
import pandas as pd

def find_csv_dir(target_dir: str) -> str:
    """Locate existing directory containing CBIS-DDSM CSV files."""
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "mass_case_description_train_set.csv")):
        return target_dir

    candidates = [
        target_dir,
        "dataset/csv",
        "data/raw/CBIS_DDSM/csv",
        os.path.join(os.getcwd(), "dataset", "csv"),
        os.path.join(os.getcwd(), "data", "raw", "CBIS_DDSM", "csv"),
        "/app/dataset/csv",
        "/app/data/raw/CBIS_DDSM/csv",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "mass_case_description_train_set.csv")):
            return c
    return target_dir

def load_cbis_csvs(raw_csv_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all standard CBIS-DDSM CSV files into a dictionary of DataFrames.

    Args:
        raw_csv_dir: Path to directory containing raw CSVs.

    Returns:
        Dictionary containing DataFrames for mass, calc, dicom_info, and meta.
    """
    actual_dir = find_csv_dir(raw_csv_dir)
    csv_files = {
        "mass_train": "mass_case_description_train_set.csv",
        "mass_test": "mass_case_description_test_set.csv",
        "calc_train": "calc_case_description_train_set.csv",
        "calc_test": "calc_case_description_test_set.csv",
        "dicom_info": "dicom_info.csv",
        "meta": "meta.csv",
    }
    
    dfs = {}
    for key, filename in csv_files.items():
        file_path = os.path.join(actual_dir, filename)
        if os.path.exists(file_path):
            dfs[key] = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Expected CSV file not found: {file_path}")
            
    return dfs
