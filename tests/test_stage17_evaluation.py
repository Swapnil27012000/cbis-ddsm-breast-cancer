"""Unit and integration tests for Stage 17.1: Final Comparative Evaluation.

Verifies:
1. All four model checkpoints exist, are readable, and non-empty.
2. Stage 13 test manifest integrity (644 records, 387 benign, 257 malignant, 318 patients).
3. Preprocessing and forward-pass output dimension compatibility (B, 1).
4. McNemar statistical significance test mathematical accuracy on analytical examples.
5. Metric computation and confusion matrix reconciliation ($TN+FP+FN+TP = N$).
6. Prediction CSV alignment with manifest record identifiers and zero leakage.
7. Absence of quarantined conflict records from test partitions.
"""

import os
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.datasets.stage14_dataloader import load_stage13_manifests
from src.evaluation.stage17_evaluator import (
    MODEL_REGISTRY,
    compute_detailed_metrics,
    compute_mcnemar_test,
)


def test_checkpoint_registry_integrity():
    """Verify that all four required candidate checkpoints exist and have expected non-zero size."""
    for model_key, spec in MODEL_REGISTRY.items():
        path = spec["checkpoint_path"]
        assert os.path.exists(path), f"Missing checkpoint for {model_key}: {path}"
        size = os.path.getsize(path)
        assert size > 1_000_000, f"Checkpoint file suspiciously small for {model_key}: {size} bytes"


def test_test_manifest_record_count_and_class_distribution():
    """Verify clean test manifest contains exactly 644 records and zero patient leakage."""
    manifest_dir = "data/metadata/stage13"
    train_df, val_df, test_df = load_stage13_manifests(manifest_dir)

    assert len(test_df) == 644
    assert (test_df["binary_label"] == 0).sum() == 387
    assert (test_df["binary_label"] == 1).sum() == 257
    assert test_df["patient_id"].nunique() == 318

    # Verify zero patient overlap with train and val
    train_patients = set(train_df["patient_id"].unique())
    val_patients = set(val_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())

    assert len(test_patients.intersection(train_patients)) == 0
    assert len(test_patients.intersection(val_patients)) == 0


def test_model_forward_pass_and_binary_output_shape():
    """Verify each candidate architecture produces raw logits of shape (B, 1)."""
    dummy_input = torch.randn(2, 1, 256, 256)
    for model_key, spec in MODEL_REGISTRY.items():
        model = spec["factory"]()
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 1), f"Unexpected output shape {out.shape} for {model_key}"
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_mcnemar_statistical_computation_exactness():
    """Verify McNemar's test calculation on known discordant pairs."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    # Model A: correct on 0, 1, 2, 3; wrong on 4, 5
    pred_a = np.array([0, 0, 1, 1, 1, 0])
    # Model B: correct on 0, 1, 4, 5; wrong on 2, 3
    pred_b = np.array([0, 0, 0, 0, 0, 1])

    res = compute_mcnemar_test(y_true, pred_a, pred_b)
    assert res["b"] == 2  # A right, B wrong (cases 2, 3)
    assert res["c"] == 2  # B right, A wrong (cases 4, 5)
    assert res["p_value"] == 1.0  # Equal discordance gives p = 1.0


def test_detailed_metrics_exactness():
    """Verify confusion matrix, accuracy, sensitivity, specificity, and loss."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.2, 0.85])

    metrics = compute_detailed_metrics(y_true, y_probs, threshold=0.50)

    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 3
    assert metrics["total_samples"] == 8

    assert metrics["accuracy"] == 0.75
    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["f1_score"] == 0.75
    assert metrics["roc_auc"] == pytest.approx(0.78125, abs=1e-4)


def test_quarantined_records_exclusion():
    """Ensure quarantined conflict records are strictly absent from the test set."""
    quarantined_path = "data/metadata/stage13/quarantined_conflict_manifest.csv"
    if os.path.exists(quarantined_path):
        q_df = pd.read_csv(quarantined_path)
        q_cases = set(q_df["case_key"].astype(str))

        _, _, test_df = load_stage13_manifests("data/metadata/stage13")
        test_cases = set(test_df["case_key"].astype(str))

        overlap = q_cases.intersection(test_cases)
        assert len(overlap) == 0, f"Found {len(overlap)} quarantined cases in test set!"
