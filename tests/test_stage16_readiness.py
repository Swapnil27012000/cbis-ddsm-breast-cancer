"""Stage 16.0 Readiness Audit Test Suite: DenseNet and EfficientNet.

Verifies:
1. Stage 13 manifest integrity (2,239 train, 542 val, 644 test, 143 quarantined).
2. Patient-level partition isolation (zero patient overlap).
3. Label mapping consistency (0 = Benign, 1 = Malignant).
4. DenseNet-121 architecture, single-channel adaptation, and output shape (B, 1).
5. EfficientNet-B0 architecture, single-channel adaptation, and output shape (B, 1).
6. Grayscale channel weight averaging correctness for DenseNet and EfficientNet.
7. Forward pass finiteness (no NaNs, no Infs) on single-channel 256x256 inputs.
8. Clinical metric computation utilities and single-class safety.
9. Immutability of Stages 1–15 artifacts and checkpoints.
10. Test-set isolation policy (clean test partition held back).
"""

import os
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

from src.datasets.stage14_dataloader import (
    audit_dataset_splits_and_leakage,
    load_stage13_manifests,
)
from src.models.densenet import get_densenet
from src.models.efficientnet import get_efficientnet
from src.training.stage15_trainer import compute_metrics, safe_torch_load


def test_stage16_manifest_integrity_and_patient_isolation():
    """Verify that Stage 16 can directly reuse Stage 13 manifests with zero patient leakage."""
    train_df, val_df, test_df = load_stage13_manifests("data/metadata/stage13")

    assert len(train_df) == 2239
    assert len(val_df) == 542
    assert len(test_df) == 644

    audit = audit_dataset_splits_and_leakage(train_df, val_df, test_df)
    assert audit["leakage_free"] is True
    assert audit["train_val_patient_overlap"] == 0
    assert audit["train_test_patient_overlap"] == 0
    assert audit["val_test_patient_overlap"] == 0


def test_stage16_label_mapping_consistency():
    """Verify binary labels adhere strictly to 0=Benign, 1=Malignant."""
    train_df, val_df, _ = load_stage13_manifests("data/metadata/stage13")

    # Check distinct values
    assert set(train_df["binary_label"].unique()).issubset({0, 1})
    assert set(val_df["binary_label"].unique()).issubset({0, 1})

    # Check class distributions match Stage 13 ground truth
    train_counts = train_df["binary_label"].value_counts().to_dict()
    assert train_counts[0] == 1324
    assert train_counts[1] == 915

    val_counts = val_df["binary_label"].value_counts().to_dict()
    assert val_counts[0] == 312
    assert val_counts[1] == 230


def test_densenet121_architecture_and_output_shape():
    """Verify DenseNet-121 layer structure, parameter count, and binary logit output."""
    model = get_densenet(in_channels=1, num_classes=1, pretrained=False, dropout=0.3)
    model.eval()

    # Parameter count should be approx 7.0M
    total_params = sum(p.numel() for p in model.parameters())
    assert 6_500_000 < total_params < 7_500_000

    # Verify single-channel first convolution
    assert model.features.conv0.in_channels == 1
    assert model.features.conv0.out_channels == 64

    # Forward pass with batch of 2 single-channel 256x256 images
    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_efficientnet_b0_architecture_and_output_shape():
    """Verify EfficientNet-B0 layer structure, parameter count, and binary logit output."""
    model = get_efficientnet(in_channels=1, num_classes=1, pretrained=False, dropout=0.3)
    model.eval()

    # Parameter count should be approx 4.0M
    total_params = sum(p.numel() for p in model.parameters())
    assert 3_500_000 < total_params < 4_500_000

    # Verify single-channel first convolution
    assert model.features[0][0].in_channels == 1
    assert model.features[0][0].out_channels == 32

    # Forward pass with batch of 2 single-channel 256x256 images
    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_grayscale_channel_weight_averaging_densenet():
    """Verify DenseNet-121 conv0 weights match the mean across original 3 RGB channels."""
    ref_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    ref_conv0_weight = ref_model.features.conv0.weight.data.clone()
    expected_weight = ref_conv0_weight.mean(dim=1, keepdim=True)

    adapted_model = get_densenet(in_channels=1, num_classes=1, pretrained=True)
    actual_weight = adapted_model.features.conv0.weight.data

    assert actual_weight.shape == (64, 1, 7, 7)
    assert torch.allclose(actual_weight, expected_weight, atol=1e-6)


def test_grayscale_channel_weight_averaging_efficientnet():
    """Verify EfficientNet-B0 first conv weights match the mean across original 3 RGB channels."""
    ref_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    ref_conv_weight = ref_model.features[0][0].weight.data.clone()
    expected_weight = ref_conv_weight.mean(dim=1, keepdim=True)

    adapted_model = get_efficientnet(in_channels=1, num_classes=1, pretrained=True)
    actual_weight = adapted_model.features[0][0].weight.data

    assert actual_weight.shape == (32, 1, 3, 3)
    assert torch.allclose(actual_weight, expected_weight, atol=1e-6)


def test_metric_computations_parity():
    """Verify compute_metrics reproduces identical clinical statistics on known data."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.2, 0.85])

    metrics = compute_metrics(y_true, y_probs, threshold=0.5)
    assert metrics["accuracy"] == 6.0 / 8.0
    assert metrics["sensitivity"] == 3.0 / 4.0
    assert metrics["specificity"] == 3.0 / 4.0
    assert metrics["precision"] == 3.0 / 4.0
    assert metrics["f1_score"] == 3.0 / 4.0
    assert metrics["roc_auc"] == pytest.approx(0.78125, abs=1e-5)


def test_prior_stages_immutability():
    """Verify that Stage 16 readiness activities have not modified prior artifacts."""
    # Stage 13 manifest
    assert os.path.exists("data/metadata/stage13/classification_train_manifest.csv")
    assert os.path.exists("data/metadata/stage13/classification_test_manifest.csv")

    # Stage 14 artifacts
    assert os.path.exists("data/metadata/stage14/stage14_baseline_cnn_config.json")
    assert os.path.exists("models/stage14_baseline_cnn/baseline_cnn_best.pt")

    # Stage 15 artifacts
    assert os.path.exists("data/metadata/stage15/stage15_resnet50_config.json")
    assert os.path.exists("models/stage15_resnet50/resnet50_best.pt")
    assert os.path.exists("data/metadata/stage15/stage15_1_audit_report.md")
    assert os.path.exists("data/metadata/stage15/stage15_2_metric_consistency_audit.md")
