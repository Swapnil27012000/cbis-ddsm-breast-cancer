"""CBIS-DDSM PyTorch Dataset implementation."""
import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable

class CBISDDSMDataset(Dataset):
    """PyTorch Dataset loading mammograms and labels from master metadata."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_col: str = "image_file_path",
        label_col: str = "label",
        transform: Optional[Callable] = None,
        in_channels: int = 1
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        
        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback black canvas if corrupt
            img = np.zeros((512, 512), dtype=np.uint8)

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        if self.in_channels == 3 and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        label = int(row[self.label_col]) if self.label_col in row and not pd.isna(row[self.label_col]) else 0
        return img_tensor, torch.tensor(label, dtype=torch.long)
