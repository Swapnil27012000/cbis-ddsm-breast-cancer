"""Stage 16: DenseNet-121 Transfer Learning Architecture for CBIS-DDSM Mammography.

Adapts Densely Connected Convolutional Networks (DenseNet-121) for binary breast cancer classification
(Benign = 0 vs. Malignant = 1) on single-channel (grayscale) mammograms.

Key Architectural & Methodological Specifications:
1. Pretrained Backbone:
   - Uses torchvision DenseNet-121 pretrained on ImageNet-1K (DenseNet121_Weights.IMAGENET1K_V1).
   - Features dense connectivity where each layer receives direct inputs from all preceding layers,
     promoting continuous feature reuse, compact parameter count (~7.0M), and strong gradient propagation.
2. Grayscale Adaptation Strategy (Channel Weight Averaging):
   - DenseNet-121 natively expects 3-channel RGB input.
   - Mammograms are natively single-channel (grayscale) 256x256 images.
   - We replace features.conv0 (7x7, stride 2, padding 3) with in_channels=1 and initialize its weights
     as the mean across the original 3 RGB channels:
       W_gray = W_rgb.mean(dim=1, keepdim=True)
   - Mathematical Rationale:
       Preserves the early edge and texture detector response of ImageNet pretrained filters without
       triplicating input images in VRAM (preventing 3x memory overhead).
3. Binary Classification Head:
   - Replaces the 1,000-class ImageNet classifier (in_features=1024) with:
       nn.Sequential(
           nn.Dropout(p=dropout_rate),
           nn.Linear(1024, num_classes)  # num_classes=1 for BCEWithLogitsLoss
       )
   - Initialized via Kaiming uniform with zero bias.
4. Resource Optimization for 4 GB VRAM (RTX 2050):
   - DenseNet-121 has ~7.0M total parameters (only 30% of ResNet-50's 23.5M parameters).
   - Operates with an estimated peak memory of ~1.8 GB VRAM under AMP and batch size 8.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class Stage16DenseNet121(nn.Module):
    """DenseNet-121 architecture adapted for single-channel binary mammogram classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        weights: Optional[Any] = "DEFAULT",
    ):
        """Initialize DenseNet-121 with single-channel adaptation and binary classification head.

        Args:
            in_channels: Number of input image channels (default 1 for grayscale mammography).
            num_classes: Number of target output logits (default 1 for binary BCEWithLogitsLoss).
            pretrained: Whether to initialize backbone with ImageNet pretrained weights.
            dropout_rate: Dropout probability in classification head (default 0.3).
            weights: Specific torchvision weights enum or 'DEFAULT'.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained_requested = pretrained
        self.dropout_rate = dropout_rate

        self.weight_metadata: Dict[str, Any] = {
            "architecture": "DenseNet-121",
            "pretrained_requested": pretrained,
            "pretrained_loaded": False,
            "weight_version": "None (Random Initialization)",
            "adaptation_strategy": "channel_weight_averaging",
            "in_channels": in_channels,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "load_error": None,
        }

        # 1. Instantiate base DenseNet-121
        weight_enum = None
        if pretrained:
            if weights == "DEFAULT" or weights is True:
                weight_enum = DenseNet121_Weights.DEFAULT
            elif isinstance(weights, DenseNet121_Weights):
                weight_enum = weights
            else:
                weight_enum = DenseNet121_Weights.IMAGENET1K_V1

        try:
            if weight_enum is not None:
                base_model = densenet121(weights=weight_enum)
                self.weight_metadata["pretrained_loaded"] = True
                self.weight_metadata["weight_version"] = str(weight_enum)
            else:
                base_model = densenet121(weights=None)
                self.weight_metadata["pretrained_loaded"] = False
                self.weight_metadata["weight_version"] = "None (Random Initialization)"
        except Exception as e:
            self.weight_metadata["load_error"] = str(e)
            raise RuntimeError(
                f"Failed to load requested pretrained DenseNet-121 weights ({weight_enum}): {e}. "
                "Per project specification, random initialization is not silently substituted."
            ) from e

        # 2. Adapt first convolution (features.conv0) if single-channel
        orig_conv = base_model.features.conv0
        if in_channels == 1:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            if self.weight_metadata["pretrained_loaded"] and orig_conv.weight is not None:
                with torch.no_grad():
                    # Channel weight averaging: (64, 3, 7, 7) -> (64, 1, 7, 7)
                    avg_weights = orig_conv.weight.data.mean(dim=1, keepdim=True)
                    new_conv.weight.data.copy_(avg_weights)
                    if orig_conv.bias is not None and new_conv.bias is not None:
                        new_conv.bias.data.copy_(orig_conv.bias.data)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0.0)
            base_model.features.conv0 = new_conv
        elif in_channels != 3:
            base_model.features.conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            nn.init.kaiming_normal_(base_model.features.conv0.weight, mode="fan_out", nonlinearity="relu")

        # 3. DenseNet feature extractor
        self.features = base_model.features

        # 4. Binary classification head
        in_features = base_model.classifier.in_features  # 1024 for DenseNet-121
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        nn.init.kaiming_uniform_(self.classifier[1].weight, a=1.0)
        if self.classifier[1].bias is not None:
            nn.init.constant_(self.classifier[1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DenseNet-121.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Logits tensor of shape (B, num_classes), typically (B, 1).
        """
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        classifier = sum(p.numel() for p in self.classifier.parameters())
        backbone = total - classifier
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "backbone_parameters": backbone,
            "classifier_parameters": classifier,
        }

    def get_weight_metadata(self) -> Dict[str, Any]:
        """Retrieve documented metadata about pretrained weights and adaptation."""
        meta = dict(self.weight_metadata)
        meta.update(self.count_parameters())
        return meta


def get_stage16_densenet121(
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    weights: Optional[Any] = "DEFAULT",
) -> Stage16DenseNet121:
    """Factory function for Stage 16 DenseNet-121 model."""
    return Stage16DenseNet121(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        weights=weights,
    )
