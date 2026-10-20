"""Stage 14: Baseline Convolutional Neural Network for CBIS-DDSM Mammography.

A modest, scientifically controlled CNN architecture tailored for binary breast cancer
classification (Benign vs. Malignant) on full-resolution baseline mammograms downsampled
to a memory-efficient resolution (e.g., 256x256), designed to train comfortably within
a 4 GB VRAM GPU budget (e.g., NVIDIA GeForce RTX 2050).

Key Architectural Principles:
1. Four Sequential Convolutional Blocks:
   - Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
   - Channel progression: 1 -> 32 -> 64 -> 128 -> 256.
   - Kernel size: 3x3 with padding=1 to preserve spatial context before pooling.
2. Global Average Pooling (GAP):
   - Compresses 256 spatial feature maps into a 1D vector (256,), minimizing total parameters
     and preventing fully-connected overfitting.
3. Regularized Classification Head:
   - Linear(256, 128) -> BatchNorm1d -> ReLU -> Dropout(0.4) -> Linear(128, 1).
4. Numerically Stable Binary Logits Output:
   - Outputs a single raw logit per image for use with BCEWithLogitsLoss.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineBreastCancerCNN(nn.Module):
    """Modest Convolutional Neural Network baseline for binary breast cancer classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1: Input (B, in_channels, H, W) -> (B, 32, H/2, W/2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: (B, 128, H/8, W/8) -> (B, 256, H/16, W/16)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling: (B, 256, H/16, W/16) -> (B, 256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Raw logits tensor of shape (B, 1) or (B, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


def get_baseline_cnn(
    in_channels: int = 1,
    num_classes: int = 1,
    dropout_rate: float = 0.4,
) -> BaselineBreastCancerCNN:
    """Factory function for baseline CNN model."""
    return BaselineBreastCancerCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
