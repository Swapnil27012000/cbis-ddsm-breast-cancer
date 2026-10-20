"""Stage 14: Reproducible PyTorch Dataset and DataLoader for CBIS-DDSM Mammography.

Constructs memory-safe, leak-free PyTorch datasets and data loaders from Stage 13 manifests:
1. Manifest Loading: Loads train, val, and test manifests from `data/metadata/stage13/`.
2. Path Resolution: Resolves baseline PNG paths across local and Docker execution environments.
3. Memory Safety: Loads images lazily on-the-fly per batch to prevent Out-Of-Memory (OOM) on 4 GB GPUs.
4. Deterministic Preprocessing:
   - Grayscale loading (1 channel).
   - Area-based downsampling (`cv2.INTER_AREA`) to avoid aliasing artifacts on high-res mammograms.
   - Normalization statistics computed strictly from the training manifest.
5. Zero Leakage & Patient Grouping:
   - Validates that train, val, and test partitions share 0 patients.
6. Repeated Image Reference Auditing:
   - Identifies cases where multiple abnormality records point to the same physical mammogram.
   - Supports both `all_records` (full clinical records) and `primary_views_only` (deduplicated views).
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.logger import setup_logger

logger = setup_logger("Stage14DataLoader")

# Default normalization statistics computed from Stage 5 / Stage 13 training corpus
TRAIN_MEAN_DEFAULT = 0.3154
TRAIN_STD_DEFAULT = 0.3012


def resolve_image_path(rel_path: str) -> str:
    """Resolve physical baseline image path across Docker and host workspaces."""
    if not rel_path or pd.isna(rel_path):
        return ""
    candidates = [
        rel_path,
        os.path.join(_PROJECT_ROOT, rel_path),
        os.path.join("/app", rel_path),
        os.path.join(os.getcwd(), rel_path),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(rel_path)


class CBISDDSMClassificationDataset(Dataset):
    """PyTorch Dataset for CBIS-DDSM binary mammogram classification."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split_name: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        normalize_mean: float = TRAIN_MEAN_DEFAULT,
        normalize_std: float = TRAIN_STD_DEFAULT,
        augment: bool = False,
        primary_views_only: bool = False,
    ):
        """Initialize dataset.

        Args:
            manifest_df: DataFrame loaded from Stage 13 manifest.
            split_name: One of 'train', 'val', 'test'.
            image_size: (height, width) for spatial resize.
            normalize_mean: Mean for standard z-score normalization.
            normalize_std: Standard deviation for z-score normalization.
            augment: If True, applies horizontal flips (training only).
            primary_views_only: If True, filters only canonical primary views.
        """
        self.split_name = split_name
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = max(normalize_std, 1e-6)
        self.augment = augment and (split_name == "train")

        df = manifest_df.copy()
        if primary_views_only and "is_primary_view_record" in df.columns:
            df = df[df["is_primary_view_record"] == True].copy()

        self.df = df.reset_index(drop=True)

        # Pre-resolve and verify paths
        self.image_paths = []
        self.labels = []
        self.patient_ids = []
        self.case_keys = []

        missing_count = 0
        for _, row in self.df.iterrows():
            raw_path = str(row.get("baseline_image_path", ""))
            resolved = resolve_image_path(raw_path)
            self.image_paths.append(resolved)

            # Binary label (0: Benign, 1: Malignant)
            lbl = int(row.get("binary_label", 0))
            self.labels.append(lbl)
            self.patient_ids.append(str(row.get("patient_id", "")))
            self.case_keys.append(str(row.get("case_key", "")))

            if not os.path.exists(resolved):
                missing_count += 1

        if missing_count > 0:
            logger.warning(
                f"[{split_name.upper()}] {missing_count}/{len(self.df)} image paths could not be found on disk!"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback zero-filled array to prevent crash on corrupted file
            img = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        else:
            # Resize using INTER_AREA for smooth downsampling
            if img.shape[:2] != self.image_size:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)

        # Training-only augmentation (horizontal flip)
        if self.augment and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        # Normalize to float32 [0.0, 1.0], then standard z-score
        img_float = img.astype(np.float32) / 255.0
        img_norm = (img_float - self.normalize_mean) / self.normalize_std

        # Convert to PyTorch tensor of shape (1, H, W)
        tensor_img = torch.from_numpy(img_norm).unsqueeze(0).float()
        tensor_label = torch.tensor(label, dtype=torch.float32)

        meta = {
            "patient_id": self.patient_ids[idx],
            "case_key": self.case_keys[idx],
            "image_path": img_path,
        }

        return tensor_img, tensor_label, meta


def compute_training_normalization_stats(train_df: pd.DataFrame, sample_size: int = 500) -> Tuple[float, float]:
    """Compute empirical mean and std strictly from training baseline images to prevent data leakage."""
    sampled = train_df.sample(min(len(train_df), sample_size), random_state=42)
    pixel_means = []
    pixel_stds = []

    for _, row in sampled.iterrows():
        p = resolve_image_path(str(row.get("baseline_image_path", "")))
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                norm = img.astype(np.float32) / 255.0
                pixel_means.append(float(np.mean(norm)))
                pixel_stds.append(float(np.std(norm)))

    if not pixel_means:
        return TRAIN_MEAN_DEFAULT, TRAIN_STD_DEFAULT

    return float(np.mean(pixel_means)), float(np.mean(pixel_stds))


def load_stage13_manifests(
    manifest_dir: str = "data/metadata/stage13",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load verified train, val, and test manifests from Stage 13."""
    actual_dir = resolve_image_path(manifest_dir)

    train_path = os.path.join(actual_dir, "classification_train_manifest.csv")
    val_path = os.path.join(actual_dir, "classification_val_manifest.csv")
    test_path = os.path.join(actual_dir, "classification_test_manifest.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required Stage 13 manifest not found: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def audit_dataset_splits_and_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Verify zero patient leakage and audit repeated image references."""
    train_pats = set(train_df["patient_id"].astype(str).unique())
    val_pats = set(val_df["patient_id"].astype(str).unique())
    test_pats = set(test_df["patient_id"].astype(str).unique())

    train_val_overlap = train_pats.intersection(val_pats)
    train_test_overlap = train_pats.intersection(test_pats)
    val_test_overlap = val_pats.intersection(test_pats)

    audit_metrics = {
        "train_patients": len(train_pats),
        "val_patients": len(val_pats),
        "test_patients": len(test_pats),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "train_primary_views": int((train_df["is_primary_view_record"] == True).sum()) if "is_primary_view_record" in train_df.columns else len(train_df),
        "val_primary_views": int((val_df["is_primary_view_record"] == True).sum()) if "is_primary_view_record" in val_df.columns else len(val_df),
        "test_primary_views": int((test_df["is_primary_view_record"] == True).sum()) if "is_primary_view_record" in test_df.columns else len(test_df),
        "train_val_patient_overlap": len(train_val_overlap),
        "train_test_patient_overlap": len(train_test_overlap),
        "val_test_patient_overlap": len(val_test_overlap),
        "leakage_free": (len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0),
    }

    return audit_metrics


def get_stage14_dataloaders(
    manifest_dir: str = "data/metadata/stage13",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 2,
    augment_train: bool = True,
    primary_views_only: bool = False,
    compute_stats: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Build train, validation, and test DataLoaders for Stage 14 baseline CNN."""
    train_df, val_df, test_df = load_stage13_manifests(manifest_dir)

    audit_metrics = audit_dataset_splits_and_leakage(train_df, val_df, test_df)
    if not audit_metrics["leakage_free"]:
        raise ValueError(f"Patient leakage detected across manifests: {audit_metrics}")

    if compute_stats:
        mean, std = compute_training_normalization_stats(train_df)
    else:
        mean, std = TRAIN_MEAN_DEFAULT, TRAIN_STD_DEFAULT

    train_ds = CBISDDSMClassificationDataset(
        manifest_df=train_df,
        split_name="train",
        image_size=image_size,
        normalize_mean=mean,
        normalize_std=std,
        augment=augment_train,
        primary_views_only=primary_views_only,
    )

    val_ds = CBISDDSMClassificationDataset(
        manifest_df=val_df,
        split_name="val",
        image_size=image_size,
        normalize_mean=mean,
        normalize_std=std,
        augment=False,
        primary_views_only=primary_views_only,
    )

    test_ds = CBISDDSMClassificationDataset(
        manifest_df=test_df,
        split_name="test",
        image_size=image_size,
        normalize_mean=mean,
        normalize_std=std,
        augment=False,
        primary_views_only=primary_views_only,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    info = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "image_size": image_size,
        "batch_size": batch_size,
        "normalization": {"mean": mean, "std": std},
        "audit": audit_metrics,
        "primary_views_only": primary_views_only,
    }

    return train_loader, val_loader, test_loader, info
