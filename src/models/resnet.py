"""ResNet architectures adapted for mammography."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

def get_resnet(
    arch: str = "resnet50",
    in_channels: int = 1,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """Build ResNet-18 or ResNet-50 with single-channel or 3-channel input."""
    if arch == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    else:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)

    # Modify initial conv if 1 channel
    if in_channels != 3:
        orig_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )
        if pretrained and orig_conv.weight is not None:
            # Average pretrained weights across input channels
            with torch.no_grad():
                model.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feat, num_classes)
    )
    return model
