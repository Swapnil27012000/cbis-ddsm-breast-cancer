"""DenseNet architectures adapted for mammography."""
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def get_densenet(
    in_channels: int = 1,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """Build DenseNet-121 with modified input channels and classification head."""
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    if in_channels != 3:
        orig_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )
        if pretrained and orig_conv.weight is not None:
            with torch.no_grad():
                model.features.conv0.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

    in_feat = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_feat, num_classes)
    )
    return model
