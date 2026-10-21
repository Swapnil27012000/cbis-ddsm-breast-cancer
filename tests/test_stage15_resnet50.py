"""Unit and integration tests for Stage 15: ResNet-50 Transfer Learning Pipeline.

Verifies:
1. ResNet-50 architecture and binary output shape (B, 1).
2. Pretrained ImageNet weight loading and transparent metadata documentation.
3. Grayscale conversion strategy (channel weight averaging).
4. Forward/backward pass with finite loss under BCEWithLogitsLoss.
5. Stage 13 manifest loading and zero patient leakage across partitions.
6. Checkpoint saving and reloading with weights_only=True safe settings.
7. Reproducible initialization with fixed seed.
8. CPU fallback execution and hardware agnosticism.
9. Immutability of Stages 1–14 artifacts and baseline CNN metrics.
10. Smoke-test mode isolation (clean test set is not evaluated).
11. Default test-set deferral policy (held back for fair final comparison).
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from src.datasets.stage14_dataloader import (
    CBISDDSMClassificationDataset,
    audit_dataset_splits_and_leakage,
    get_stage14_dataloaders,
    load_stage13_manifests,
)
from src.models.stage15_resnet50 import Stage15ResNet50, get_stage15_resnet50
from src.training.stage15_trainer import (
    compute_metrics,
    run_stage15_training,
    safe_torch_load,
    set_seed,
)


def test_resnet50_architecture_and_binary_output_shape():
    """Verify ResNet-50 layer structure, parameter count (~23.5M), and tensor shapes."""
    model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False)
    model.eval()

    params = model.count_parameters()
    # ResNet-50 should have approx 23.5 million parameters
    assert 23_000_000 < params["total_parameters"] < 25_000_000
    assert params["trainable_parameters"] == params["total_parameters"]
    assert params["classifier_parameters"] == 2049  # 2048 weights + 1 bias

    # Test forward pass with batch of 2 single-channel 256x256 images
    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_pretrained_weight_loading_metadata():
    """Verify documented pretrained ImageNet weight loading and metadata accuracy."""
    model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=True)
    meta = model.get_weight_metadata()

    assert meta["architecture"] == "ResNet-50"
    assert meta["pretrained_requested"] is True
    assert meta["pretrained_loaded"] is True
    assert "ResNet50_Weights" in meta["weight_version"]
    assert meta["adaptation_strategy"] == "channel_weight_averaging"
    assert meta["in_channels"] == 1
    assert meta["num_classes"] == 1
    assert meta["load_error"] is None


def test_grayscale_channel_weight_averaging_exactness():
    """Verify that conv1 weights are precisely the mean across original 3 RGB channels."""
    # Obtain original 3-channel weights directly from torchvision
    ref_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    ref_conv1_weight = ref_model.conv1.weight.data.clone()  # shape: (64, 3, 7, 7)
    expected_gray_weight = ref_conv1_weight.mean(dim=1, keepdim=True)  # shape: (64, 1, 7, 7)

    # Instantiate adapted Stage 15 model
    adapted_model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=True)
    actual_conv1_weight = adapted_model.conv1.weight.data  # shape: (64, 1, 7, 7)

    assert actual_conv1_weight.shape == (64, 1, 7, 7)
    assert torch.allclose(actual_conv1_weight, expected_gray_weight, atol=1e-6)


def test_forward_and_backward_pass_finite_loss():
    """Verify forward and backward gradients with BCEWithLogitsLoss."""
    model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False)
    model.train()

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.45]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dummy_input = torch.randn(2, 1, 256, 256)
    dummy_target = torch.tensor([[0.0], [1.0]])

    optimizer.zero_grad()
    logits = model(dummy_input)
    loss = criterion(logits, dummy_target)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert model.conv1.weight.grad is not None
    assert not torch.isnan(model.conv1.weight.grad).any()
    assert not torch.isinf(model.conv1.weight.grad).any()


def test_manifest_loading_and_patient_separation():
    """Verify loading of Stage 13 manifests and guarantee zero patient leakage."""
    train_df, val_df, test_df = load_stage13_manifests("data/metadata/stage13")

    assert len(train_df) == 2239
    assert len(val_df) == 542
    assert len(test_df) == 644

    audit = audit_dataset_splits_and_leakage(train_df, val_df, test_df)
    assert audit["leakage_free"] is True
    assert audit["train_val_patient_overlap"] == 0
    assert audit["train_test_patient_overlap"] == 0
    assert audit["val_test_patient_overlap"] == 0


def test_checkpoint_save_and_safe_load():
    """Verify that model state can be saved and accurately reloaded with safe_torch_load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_resnet50_ckpt.pt")
        model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False)
        initial_weights = model.conv1.weight.clone()

        torch.save({
            "model_state_dict": model.state_dict(),
            "val_loss": 0.38,
            "val_metrics": {"accuracy": 0.85},
        }, ckpt_path)
        assert os.path.exists(ckpt_path)

        new_model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False)
        # Alter weights
        with torch.no_grad():
            new_model.conv1.weight.fill_(0.0)
        assert not torch.equal(new_model.conv1.weight, initial_weights)

        # Reload using safe_torch_load
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")
        new_model.load_state_dict(ckpt["model_state_dict"])
        assert torch.equal(new_model.conv1.weight, initial_weights)
        assert ckpt["val_loss"] == 0.38


def test_reproducible_initialization():
    """Verify that setting the seed produces identical head initialization weights."""
    set_seed(42)
    m1 = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False)
    w1 = m1.fc[1].weight.clone()

    set_seed(42)
    m2 = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False)
    w2 = m2.fc[1].weight.clone()

    assert torch.equal(w1, w2)


def test_cpu_fallback_execution():
    """Verify that model executes seamlessly on CPU without requiring CUDA."""
    device = torch.device("cpu")
    model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False).to(device)
    model.eval()

    inp = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        out = model(inp)

    assert out.device.type == "cpu"
    assert out.shape == (1, 1)


def test_immutability_of_prior_stages():
    """Verify that Stage 15 does not modify raw CSVs or Stages 1-14 metadata/artifacts."""
    # Stage 13 manifests
    train_manifest = "data/metadata/stage13/classification_train_manifest.csv"
    assert os.path.exists(train_manifest)
    df_tr = pd.read_csv(train_manifest)
    assert len(df_tr) == 2239

    test_manifest = "data/metadata/stage13/classification_test_manifest.csv"
    assert os.path.exists(test_manifest)
    df_te = pd.read_csv(test_manifest)
    assert len(df_te) == 644

    # Stage 14 artifacts
    s14_config = "data/metadata/stage14/stage14_baseline_cnn_config.json"
    assert os.path.exists(s14_config)
    s14_test_metrics = "data/metadata/stage14/stage14_test_metrics.csv"
    assert os.path.exists(s14_test_metrics)
    df_s14_test = pd.read_csv(s14_test_metrics)
    assert pytest.approx(df_s14_test["accuracy"].iloc[0], abs=1e-4) == 0.59472


def test_smoke_test_mode_isolation():
    """Verify that smoke test mode strictly excludes clean test set evaluation."""
    summary = run_stage15_training(
        epochs=1,
        batch_size=8,
        accum_steps=1,
        smoke_test=True,
        pretrained=False,
        device_name="cpu",
    )
    assert summary["status"] == "SMOKE_TEST_SUCCESS"
    assert summary["clean_test_evaluated"] is False
    assert summary["checkpoint_reloaded"] is True
    assert "val_metrics" in summary


def test_test_set_deferral_policy():
    """Verify that test evaluation is held back by default in run_stage15_training."""
    # When evaluate_test is False, test metrics should record HELD_BACK_FOR_FINAL_COMPARISON
    from src.training.stage15_trainer import run_stage15_training
    # The default setting in function signature must be evaluate_test=False
    import inspect
    sig = inspect.signature(run_stage15_training)
    assert sig.parameters["evaluate_test"].default is False
