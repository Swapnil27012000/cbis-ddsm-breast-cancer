"""Unit tests for model instantiation and forward pass."""
import torch
import pytest
from src.models.model import build_model

def test_model_build_and_forward():
    # Single-channel 2-class classifier
    model = build_model("resnet18", in_channels=1, num_classes=2, pretrained=False)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 2)
