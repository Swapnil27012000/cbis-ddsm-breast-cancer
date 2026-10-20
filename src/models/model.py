"""Unified model builder / dispatcher."""
import torch.nn as nn
from .resnet import get_resnet
from .densenet import get_densenet
from .efficientnet import get_efficientnet
from .baseline_cnn import get_baseline_cnn

def build_model(
    arch: str = "resnet50",
    in_channels: int = 1,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """Build model based on architecture name."""
    arch_lower = arch.lower()
    if "baseline" in arch_lower:
        # Single output logit for binary classification
        out_dim = 1 if num_classes == 2 else num_classes
        return get_baseline_cnn(in_channels=in_channels, num_classes=out_dim, dropout_rate=dropout)
    elif "resnet" in arch_lower:
        return get_resnet(arch=arch_lower, in_channels=in_channels, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif "densenet" in arch_lower:
        return get_densenet(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif "efficientnet" in arch_lower:
        return get_efficientnet(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
