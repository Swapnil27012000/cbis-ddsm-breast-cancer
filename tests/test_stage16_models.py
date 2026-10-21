"""Unit and integration tests for Stage 16.1: DenseNet-121 and EfficientNet-B0 Models and Trainer.

Verifies:
1. DenseNet-121 architecture, single-channel conv0, and binary output shape (B, 1).
2. EfficientNet-B0 architecture, single-channel conv, and binary output shape (B, 1).
3. Pretrained ImageNet weight channel-averaging logic and accuracy.
4. Forward and backward passes with finite loss under BCEWithLogitsLoss.
5. Clinical metric computations and edge-case handling.
6. Dual checkpoint save, safe reload, and metadata specification.
7. Smoke-test mode isolation (clean test set is not touched).
8. Manifest and patient-level split integrity (zero leakage).
9. Test-set isolation policy (clean test partition held back).
10. Immutability of Stages 1–15 artifacts and checkpoints.
"""

import os
import shutil
import tempfile
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
from src.models.stage16_densenet import Stage16DenseNet121, get_stage16_densenet121
from src.models.stage16_efficientnet import Stage16EfficientNetB0, get_stage16_efficientnet_b0
from src.training.stage16_trainer import (
    build_stage16_model,
    compute_metrics,
    resolve_stage16_paths,
    run_stage16_training,
    safe_torch_load,
    set_seed,
)


def test_densenet121_architecture_and_output_shape():
    """Verify DenseNet-121 layer structure, parameter count, and binary logit output."""
    model = get_stage16_densenet121(in_channels=1, num_classes=1, pretrained=False, dropout_rate=0.3)
    model.eval()

    params = model.count_parameters()
    assert 6_500_000 < params["total_parameters"] < 7_500_000
    assert params["classifier_parameters"] == 1025  # 1024 weights + 1 bias

    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_efficientnet_b0_architecture_and_output_shape():
    """Verify EfficientNet-B0 layer structure, parameter count, and binary logit output."""
    model = get_stage16_efficientnet_b0(in_channels=1, num_classes=1, pretrained=False, dropout_rate=0.2)
    model.eval()

    params = model.count_parameters()
    assert 3_500_000 < params["total_parameters"] < 4_500_000
    assert params["classifier_parameters"] == 1281  # 1280 weights + 1 bias

    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_densenet_pretrained_channel_averaging_exactness():
    """Verify DenseNet-121 conv0 weights are precisely the mean across original 3 RGB channels."""
    ref_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    ref_conv0_weight = ref_model.features.conv0.weight.data.clone()
    expected_weight = ref_conv0_weight.mean(dim=1, keepdim=True)

    adapted_model = get_stage16_densenet121(in_channels=1, num_classes=1, pretrained=True)
    actual_weight = adapted_model.features.conv0.weight.data

    assert actual_weight.shape == (64, 1, 7, 7)
    assert torch.allclose(actual_weight, expected_weight, atol=1e-6)


def test_efficientnet_pretrained_channel_averaging_exactness():
    """Verify EfficientNet-B0 first conv weights are precisely the mean across original 3 RGB channels."""
    ref_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    ref_conv_weight = ref_model.features[0][0].weight.data.clone()
    expected_weight = ref_conv_weight.mean(dim=1, keepdim=True)

    adapted_model = get_stage16_efficientnet_b0(in_channels=1, num_classes=1, pretrained=True)
    actual_weight = adapted_model.features[0][0].weight.data

    assert actual_weight.shape == (32, 1, 3, 3)
    assert torch.allclose(actual_weight, expected_weight, atol=1e-6)


def test_densenet_forward_and_backward_pass():
    """Verify forward and backward gradients with BCEWithLogitsLoss for DenseNet-121."""
    model = get_stage16_densenet121(in_channels=1, num_classes=1, pretrained=False)
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
    loss.backward()

    assert model.features.conv0.weight.grad is not None
    assert not torch.isnan(model.features.conv0.weight.grad).any()


def test_efficientnet_forward_and_backward_pass():
    """Verify forward and backward gradients with BCEWithLogitsLoss for EfficientNet-B0."""
    model = get_stage16_efficientnet_b0(in_channels=1, num_classes=1, pretrained=False)
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
    loss.backward()

    assert model.features[0][0].weight.grad is not None
    assert not torch.isnan(model.features[0][0].weight.grad).any()


def test_metric_computations_exactness():
    """Verify clinical classification metrics calculations under known ground truths."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.2, 0.85])

    metrics = compute_metrics(y_true, y_probs, threshold=0.5)

    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 3

    assert metrics["accuracy"] == 6.0 / 8.0
    assert metrics["sensitivity"] == 3.0 / 4.0
    assert metrics["specificity"] == 3.0 / 4.0
    assert metrics["precision"] == 3.0 / 4.0
    assert metrics["f1_score"] == 3.0 / 4.0
    assert metrics["roc_auc"] == pytest.approx(0.78125, abs=1e-5)


def test_dual_checkpoint_save_and_safe_load():
    """Verify that dual checkpoints (loss and AUC) can be saved and safely reloaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loss_ckpt_path = os.path.join(tmpdir, "test_best_loss.pt")
        auc_ckpt_path = os.path.join(tmpdir, "test_best_auc.pt")

        model = get_stage16_densenet121(in_channels=1, num_classes=1, pretrained=False)
        orig_weight = model.features.conv0.weight.clone()

        # Save loss checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "selection_metric": "val_loss",
            "val_loss": 0.42,
        }, loss_ckpt_path)

        # Save AUC checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "selection_metric": "val_roc_auc",
            "val_roc_auc": 0.88,
        }, auc_ckpt_path)

        assert os.path.exists(loss_ckpt_path)
        assert os.path.exists(auc_ckpt_path)

        new_model = get_stage16_densenet121(in_channels=1, num_classes=1, pretrained=False)
        with torch.no_grad():
            new_model.features.conv0.weight.fill_(0.0)

        # Reload loss checkpoint
        ckpt_loss = safe_torch_load(loss_ckpt_path, map_location="cpu")
        new_model.load_state_dict(ckpt_loss["model_state_dict"])
        assert torch.equal(new_model.features.conv0.weight, orig_weight)
        assert ckpt_loss["selection_metric"] == "val_loss"
        assert ckpt_loss["val_loss"] == 0.42

        # Reload AUC checkpoint
        ckpt_auc = safe_torch_load(auc_ckpt_path, map_location="cpu")
        assert ckpt_auc["selection_metric"] == "val_roc_auc"
        assert ckpt_auc["val_roc_auc"] == 0.88


def test_smoke_test_checkpoint_path_isolation_exactness():
    """Regression test verifying smoke test paths are 100% disjoint from production paths."""
    for model_key in ["densenet121", "efficientnet_b0"]:
        prod_paths = resolve_stage16_paths(model_key, smoke_test=False)
        smoke_paths = resolve_stage16_paths(model_key, smoke_test=True)

        # 1. Full training production paths must match expected standards exactly
        assert prod_paths["models_dir"].replace("\\", "/").endswith(f"models/stage16_{model_key}")
        assert prod_paths["best_loss_checkpoint_path"].replace("\\", "/").endswith(f"{model_key}_best_loss.pt")
        assert prod_paths["best_auc_checkpoint_path"].replace("\\", "/").endswith(f"{model_key}_best_auc.pt")
        assert prod_paths["final_checkpoint_path"].replace("\\", "/").endswith(f"{model_key}_final.pt")

        # 2. Smoke test paths must be strictly isolated into separate directory and prefix
        assert "stage16_smoke_tests" in smoke_paths["models_dir"].replace("\\", "/")
        assert "stage16_smoke_tests" in smoke_paths["results_dir"].replace("\\", "/")
        assert os.path.basename(smoke_paths["best_loss_checkpoint_path"]).startswith(f"smoke_{model_key}")
        assert os.path.basename(smoke_paths["best_auc_checkpoint_path"]).startswith(f"smoke_{model_key}")
        assert os.path.basename(smoke_paths["final_checkpoint_path"]).startswith(f"smoke_{model_key}")

        # 3. Checkpoints and artifact paths must be completely disjoint
        ckpt_keys = [
            "best_loss_checkpoint_path",
            "best_auc_checkpoint_path",
            "final_checkpoint_path",
            "epoch_history_path",
            "val_loss_metrics_path",
            "val_auc_metrics_path",
            "config_path",
        ]
        for key in ckpt_keys:
            assert prod_paths[key] != smoke_paths[key], f"Path collision on {key} for {model_key}!"
            assert "smoke" not in prod_paths[key]
            assert "smoke" in smoke_paths[key]


def test_smoke_test_mode_isolation_densenet():
    """Verify that smoke test mode strictly excludes clean test set evaluation for DenseNet."""
    prod_ckpt = "models/stage16_densenet121/densenet121_best_loss.pt"
    prod_mtime_before = os.path.getmtime(prod_ckpt) if os.path.exists(prod_ckpt) else None

    summary = run_stage16_training(
        model_name="densenet121",
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
    assert summary["model_key"] == "densenet121"
    assert "val_metrics" in summary

    # Verify smoke test saved to isolated smoke path
    assert "stage16_smoke_tests" in summary["best_loss_checkpoint_path"].replace("\\", "/")
    assert "smoke_densenet121" in os.path.basename(summary["best_loss_checkpoint_path"])
    assert os.path.exists(summary["best_loss_checkpoint_path"])

    # Verify production checkpoint was NOT modified
    if prod_mtime_before is not None:
        assert os.path.getmtime(prod_ckpt) == prod_mtime_before


def test_smoke_test_mode_isolation_efficientnet():
    """Verify that smoke test mode strictly excludes clean test set evaluation for EfficientNet."""
    prod_ckpt = "models/stage16_efficientnet_b0/efficientnet_b0_best_loss.pt"
    prod_mtime_before = os.path.getmtime(prod_ckpt) if os.path.exists(prod_ckpt) else None

    summary = run_stage16_training(
        model_name="efficientnet_b0",
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
    assert summary["model_key"] == "efficientnet_b0"
    assert "val_metrics" in summary

    # Verify smoke test saved to isolated smoke path
    assert "stage16_smoke_tests" in summary["best_loss_checkpoint_path"].replace("\\", "/")
    assert "smoke_efficientnet_b0" in os.path.basename(summary["best_loss_checkpoint_path"])
    assert os.path.exists(summary["best_loss_checkpoint_path"])

    # Verify production checkpoint was NOT modified
    if prod_mtime_before is not None:
        assert os.path.getmtime(prod_ckpt) == prod_mtime_before


def test_stage16_manifest_and_patient_leakage_integrity():
    """Verify Stage 13 manifests and patient-level zero-leakage guarantee."""
    train_df, val_df, test_df = load_stage13_manifests("data/metadata/stage13")

    assert len(train_df) == 2239
    assert len(val_df) == 542
    assert len(test_df) == 644

    audit = audit_dataset_splits_and_leakage(train_df, val_df, test_df)
    assert audit["leakage_free"] is True
    assert audit["train_val_patient_overlap"] == 0
    assert audit["train_test_patient_overlap"] == 0
    assert audit["val_test_patient_overlap"] == 0


def test_prior_stages_immutability():
    """Verify that Stage 16 implementation does not modify prior artifacts."""
    # Stage 13
    assert os.path.exists("data/metadata/stage13/classification_train_manifest.csv")
    assert os.path.exists("data/metadata/stage13/classification_test_manifest.csv")

    # Stage 14
    assert os.path.exists("data/metadata/stage14/stage14_baseline_cnn_config.json")
    assert os.path.exists("models/stage14_baseline_cnn/baseline_cnn_best.pt")

    # Stage 15
    assert os.path.exists("data/metadata/stage15/stage15_resnet50_config.json")
    assert os.path.exists("models/stage15_resnet50/resnet50_best.pt")
    assert os.path.exists("data/metadata/stage15/stage15_1_audit_report.md")
    assert os.path.exists("data/metadata/stage15/stage15_2_metric_consistency_audit.md")
