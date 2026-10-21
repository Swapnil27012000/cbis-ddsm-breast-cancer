"""Stage 15: ResNet-50 Transfer Learning Architecture for CBIS-DDSM Mammography.

Adapts the deep 50-layer Residual Network (ResNet-50) for binary breast cancer classification
(Benign = 0 vs. Malignant = 1) on single-channel (grayscale) mammograms.

Key Architectural & Methodological Specifications:
1. Pretrained Backbone:
   - Uses PyTorch torchvision ResNet-50 pretrained on ImageNet-1K (ResNet50_Weights.DEFAULT / IMAGENET1K_V2).
   - Accurately captures hierarchical multiscale visual representations (edges, textures, shapes).
   - Transparently documents weight version and whether weights were successfully loaded.
2. Grayscale Adaptation Strategy (Channel Weight Averaging):
   - Standard ImageNet ResNet expects 3 input channels (RGB).
   - Mammograms are natively single-channel grayscale (1 channel).
   - Rather than triplicating images in memory (3x VRAM overhead), we replace conv1 (7x7, stride 2, padding 3)
     with in_channels=1 and initialize its weights as the mean across the original 3 RGB channels:
       W_gray = W_rgb.mean(dim=1, keepdim=True)
   - Mathematical Rationale:
       Luminance in RGB images is approximated as (R + G + B) / 3. Setting W_gray = (W_R + W_G + W_B) / 3
       preserves the expected filter activation magnitude and spatial response of ImageNet early edge detectors
       when presented with identical intensity grayscale inputs, avoiding gradient shocks during initial epochs.
3. Binary Classification Head:
   - Replaces the 1,000-class ImageNet fully connected layer (in_features=2048) with:
       nn.Sequential(
           nn.Dropout(p=dropout_rate),
           nn.Linear(2048, num_classes)  # num_classes=1 for BCEWithLogitsLoss
       )
   - Initialized via Kaiming uniform with zero bias.
4. Resource Optimization for 4 GB VRAM (RTX 2050):
   - Global Average Pooling produces a compact 2048-dim feature vector.
   - Outputs a single unconstrained logit per image for numerically stable BCEWithLogitsLoss.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Stage15ResNet50(nn.Module):
    """ResNet-50 architecture adapted for single-channel binary mammogram classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        weights: Optional[Any] = "DEFAULT",
    ):
        """Initialize ResNet-50 with single-channel adaptation and binary classification head.

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
            "architecture": "ResNet-50",
            "pretrained_requested": pretrained,
            "pretrained_loaded": False,
            "weight_version": "None (Random Initialization)",
            "adaptation_strategy": "channel_weight_averaging",
            "in_channels": in_channels,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate,
            "load_error": None,
        }

        # 1. Instantiate base ResNet-50
        weight_enum = None
        if pretrained:
            if weights == "DEFAULT" or weights is True:
                weight_enum = ResNet50_Weights.DEFAULT
            elif isinstance(weights, ResNet50_Weights):
                weight_enum = weights
            else:
                weight_enum = ResNet50_Weights.DEFAULT

        try:
            if weight_enum is not None:
                base_model = resnet50(weights=weight_enum)
                self.weight_metadata["pretrained_loaded"] = True
                self.weight_metadata["weight_version"] = str(weight_enum)
            else:
                base_model = resnet50(weights=None)
                self.weight_metadata["pretrained_loaded"] = False
                self.weight_metadata["weight_version"] = "None (Random Initialization)"
        except Exception as e:
            # Document exact failure rather than silently substituting random weights without notice
            self.weight_metadata["load_error"] = str(e)
            raise RuntimeError(
                f"Failed to load requested pretrained ResNet-50 weights ({weight_enum}): {e}. "
                "Per Stage 15 specifications, random initialization is not silently substituted."
            ) from e

        # 2. Adapt first convolution (conv1) if single-channel
        orig_conv = base_model.conv1
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
            base_model.conv1 = new_conv
        elif in_channels != 3:
            # Arbitrary in_channels fallback
            base_model.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            nn.init.kaiming_normal_(base_model.conv1.weight, mode="fan_out", nonlinearity="relu")

        # 3. Backbone layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # 4. Classification Head
        in_features = base_model.fc.in_features  # 2048 for ResNet-50
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        # Initialize the fresh classification head
        nn.init.kaiming_uniform_(self.fc[1].weight, a=1.0)
        if self.fc[1].bias is not None:
            nn.init.constant_(self.fc[1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-50.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Logits tensor of shape (B, num_classes), typically (B, 1).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters for fine-tuning control."""
        for param in self.conv1.parameters():
            param.requires_grad = not freeze
        for param in self.bn1.parameters():
            param.requires_grad = not freeze
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = not freeze

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        classifier = sum(p.numel() for p in self.fc.parameters())
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


def get_stage15_resnet50(
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    weights: Optional[Any] = "DEFAULT",
) -> Stage15ResNet50:
    """Factory function for Stage 15 ResNet-50 model."""
    return Stage15ResNet50(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        weights=weights,
    )
