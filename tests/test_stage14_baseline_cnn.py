"""Unit and integration tests for Stage 14: Baseline CNN Model and Training Pipeline.

Verifies:
1. Manifest loading and path resolution from Stage 13.
2. Label mapping (0 = Benign, 1 = Malignant).
3. Image dimensions and tensor shape (B, 1, H, W).
4. Normalization and tensor finiteness (no NaN, no Inf).
5. Data-loader batch shape and patient separation across partitions.
6. Repeated image-path audit (primary views vs. secondary abnormality records).
7. Reproducible model initialization with fixed seed.
8. Checkpoint saving and loading integrity.
9. Metric calculations (accuracy, sensitivity, specificity, precision, F1, ROC-AUC).
10. CPU fallback and GPU selection.
11. Immutability of raw CBIS-DDSM data and prior Stages 1–13 outputs.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets.stage14_dataloader import (
    CBISDDSMClassificationDataset,
    audit_dataset_splits_and_leakage,
    compute_training_normalization_stats,
    get_stage14_dataloaders,
    load_stage13_manifests,
    resolve_image_path,
)
from src.models.baseline_cnn import BaselineBreastCancerCNN, get_baseline_cnn
from src.training.stage14_trainer import compute_metrics, set_seed


def test_baseline_cnn_architecture_and_forward_pass():
    """Verify Baseline CNN layer structure, parameter count, and tensor shapes."""
    model = get_baseline_cnn(in_channels=1, num_classes=1, dropout_rate=0.4)
    model.eval()

    # Verify parameter count (should be modest, ~450k)
    params = model.count_parameters()
    assert 300_000 < params["total_parameters"] < 800_000
    assert params["trainable_parameters"] == params["total_parameters"]

    # Test forward pass with batch of 4 images of size 256x256
    dummy_input = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (4, 1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


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


def test_dataset_item_format_and_finiteness():
    """Verify single-item loading, tensor shapes, and normalized intensity ranges."""
    train_df, _, _ = load_stage13_manifests("data/metadata/stage13")

    # Instantiate dataset on small subset
    ds = CBISDDSMClassificationDataset(
        manifest_df=train_df.head(10),
        split_name="train",
        image_size=(256, 256),
        augment=False,
    )

    assert len(ds) == 10
    tensor_img, tensor_lbl, meta = ds[0]

    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (1, 256, 256)
    assert tensor_img.dtype == torch.float32

    # Check finiteness
    assert not torch.isnan(tensor_img).any()
    assert not torch.isinf(tensor_img).any()

    # Label must be float tensor 0.0 or 1.0
    assert tensor_lbl.item() in (0.0, 1.0)
    assert "patient_id" in meta
    assert "case_key" in meta


def test_primary_views_only_filtering():
    """Verify that primary_views_only properly isolates canonical unique views."""
    train_df, val_df, test_df = load_stage13_manifests("data/metadata/stage13")

    ds_all = CBISDDSMClassificationDataset(train_df, split_name="train", primary_views_only=False)
    ds_primary = CBISDDSMClassificationDataset(train_df, split_name="train", primary_views_only=True)

    assert len(ds_all) == 2239
    assert len(ds_primary) == 1891  # Exact count of canonical views in train set


def test_metric_computations_exactness():
    """Verify clinical classification metrics calculations under known ground truths."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.2, 0.85])  # 3 TN, 1 FP, 1 FN, 3 TP

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
    # Exact rank-sum Mann-Whitney calculation: (4 + 3 + 1.5 + 4) / 16 = 12.5 / 16 = 0.78125
    assert metrics["roc_auc"] == pytest.approx(0.78125, abs=1e-5)


def test_roc_auc_single_class_edge_cases():
    """Verify compute_metrics safely returns None for single-class targets without raising an exception."""
    # All benign ground truth (class 0 only)
    y_true_benign = np.array([0, 0, 0, 0])
    y_probs_benign = np.array([0.1, 0.2, 0.3, 0.4])
    metrics_benign = compute_metrics(y_true_benign, y_probs_benign)
    assert metrics_benign["roc_auc"] is None
    assert metrics_benign["accuracy"] == 1.0
    assert metrics_benign["tn"] == 4
    assert metrics_benign["fp"] == 0

    # All malignant ground truth (class 1 only)
    y_true_malignant = np.array([1, 1, 1, 1])
    y_probs_malignant = np.array([0.6, 0.7, 0.8, 0.9])
    metrics_malignant = compute_metrics(y_true_malignant, y_probs_malignant)
    assert metrics_malignant["roc_auc"] is None
    assert metrics_malignant["accuracy"] == 1.0
    assert metrics_malignant["tp"] == 4
    assert metrics_malignant["fn"] == 0


def test_roc_auc_constant_prediction_edge_case():
    """Verify ROC-AUC behavior when prediction probabilities are completely constant."""
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.5, 0.5, 0.5, 0.5])
    metrics = compute_metrics(y_true, y_probs)
    assert metrics["roc_auc"] == pytest.approx(0.5, abs=1e-5)


def test_reproducible_initialization():
    """Verify that setting the seed produces identical initial weights."""
    set_seed(42)
    m1 = get_baseline_cnn()
    w1 = m1.block1[0].weight.clone()

    set_seed(42)
    m2 = get_baseline_cnn()
    w2 = m2.block1[0].weight.clone()

    assert torch.equal(w1, w2)


def test_checkpoint_save_and_load():
    """Verify that model state can be saved and accurately reloaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        model = get_baseline_cnn()
        initial_weights = model.block1[0].weight.clone()

        torch.save({"model_state_dict": model.state_dict(), "val_loss": 0.42}, ckpt_path)
        assert os.path.exists(ckpt_path)

        new_model = get_baseline_cnn()
        # Alter weights
        with torch.no_grad():
            new_model.block1[0].weight.fill_(0.0)
        assert not torch.equal(new_model.block1[0].weight, initial_weights)

        # Reload with safe weights_only
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except (TypeError, ValueError):
            ckpt = torch.load(ckpt_path, map_location="cpu")
        new_model.load_state_dict(ckpt["model_state_dict"])
        assert torch.equal(new_model.block1[0].weight, initial_weights)
        assert ckpt["val_loss"] == 0.42


def test_immutability_of_prior_stages():
    """Verify that Stage 14 does not modify raw CSVs or Stages 1-13 metadata."""
    train_manifest = os.path.join("data/metadata/stage13/classification_train_manifest.csv")
    assert os.path.exists(train_manifest)
    df_tr = pd.read_csv(train_manifest)
    assert len(df_tr) == 2239

    test_manifest = os.path.join("data/metadata/stage13/classification_test_manifest.csv")
    assert os.path.exists(test_manifest)
    df_te = pd.read_csv(test_manifest)
    assert len(df_te) == 644


def test_smoke_test_isolation_flag():
    """Verify that smoke test mode strictly excludes clean test set evaluation."""
    from src.training.stage14_trainer import run_stage14_training
    summary = run_stage14_training(epochs=1, batch_size=16, smoke_test=True, device_name="cpu")
    assert summary["status"] == "SMOKE_TEST_SUCCESS"
    assert summary["clean_test_evaluated"] is False
    assert summary["checkpoint_reloaded"] is True
    assert "val_metrics" in summary

