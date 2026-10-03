"""Patient-level train/validation/test stratified splitting."""
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def create_patient_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset ensuring all images from each patient stay in one split.

    Prevents data leakage across train, val, and test subsets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Group by patient
    patient_df = df.groupby(patient_col)[label_col].agg(lambda x: x.mode()[0]).reset_index()

    patients_train, patients_temp = train_test_split(
        patient_df,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=patient_df[label_col]
    )

    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    patients_val, patients_test = train_test_split(
        patients_temp,
        test_size=rel_test_ratio,
        random_state=seed,
        stratify=patients_temp[label_col]
    )

    train_set = df[df[patient_col].isin(patients_train[patient_col])].copy()
    val_set = df[df[patient_col].isin(patients_val[patient_col])].copy()
    test_set = df[df[patient_col].isin(patients_test[patient_col])].copy()

    train_set["split"] = "train"
    val_set["split"] = "val"
    test_set["split"] = "test"

    return train_set, val_set, test_set
