"""Generic mammogram dataset loading from folder or image paths."""
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, List

class MammogramDataset(Dataset):
    """Generic dataset loading raw or preprocessed mammogram images."""

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [0] * len(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((512, 512), dtype=np.uint8)

        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        return tensor, torch.tensor(self.labels[idx], dtype=torch.long)
