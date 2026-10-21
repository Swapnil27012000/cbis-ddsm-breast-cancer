"""Stage 17.1: Final Comparative Evaluation Pipeline for CBIS-DDSM Mammography.

Executes a frozen, reproducible comparative evaluation of the four candidate models:
1. Baseline CNN (Stage 14 best-loss checkpoint, Epoch 7)
2. ResNet-50 (Stage 15 best-loss checkpoint, Epoch 4)
3. DenseNet-121 (Stage 16 best-loss checkpoint, Epoch 3)
4. EfficientNet-B0 (Stage 16 best-loss checkpoint, Epoch 2)

Evaluates on the exact 644-record untouched Stage 13 test manifest:
- Pre-fixed decision threshold: tau = 0.50
- Strict read-only access (no threshold tuning, no retraining, zero test leakage)
- Unweighted & weighted BCE loss (pos_weight = 1.446995)
- Standardized metrics: Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC-AUC
- Statistical significance tests: McNemar's test across all pairwise model combinations
- Granular per-record prediction manifests and subgroup analysis (mass vs. calcification)
"""

import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.datasets.stage14_dataloader import (
    CBISDDSMClassificationDataset,
    load_stage13_manifests,
    resolve_image_path,
)
from src.gpu.device import get_device, get_device_info
from src.gpu.gpu_utils import clean_gpu_memory
from src.models.baseline_cnn import get_baseline_cnn
from src.models.stage15_resnet50 import get_stage15_resnet50
from src.models.stage16_densenet import get_stage16_densenet121
from src.models.stage16_efficientnet import get_stage16_efficientnet_b0
from src.utils.logger import setup_logger

logger = setup_logger("Stage17Evaluator")

# Model configuration registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "baseline_cnn": {
        "display_name": "Baseline CNN",
        "stage": 14,
        "checkpoint_path": os.path.join(_PROJECT_ROOT, "models/stage14_baseline_cnn/baseline_cnn_best.pt"),
        "expected_epoch": 7,
        "factory": lambda: get_baseline_cnn(in_channels=1, num_classes=1, dropout_rate=0.4),
        "params": 421601,
        "prediction_filename": "baseline_cnn_predictions.csv",
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "stage": 15,
        "checkpoint_path": os.path.join(_PROJECT_ROOT, "models/stage15_resnet50/resnet50_best.pt"),
        "expected_epoch": 4,
        "factory": lambda: get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False, dropout_rate=0.3),
        "params": 23503809,
        "prediction_filename": "resnet50_predictions.csv",
    },
    "densenet121": {
        "display_name": "DenseNet-121",
        "stage": 16,
        "checkpoint_path": os.path.join(_PROJECT_ROOT, "models/stage16_densenet121/densenet121_best_loss.pt"),
        "expected_epoch": 3,
        "factory": lambda: get_stage16_densenet121(in_channels=1, num_classes=1, pretrained=False, dropout_rate=0.3),
        "params": 6948609,
        "prediction_filename": "densenet121_predictions.csv",
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "stage": 16,
        "checkpoint_path": os.path.join(_PROJECT_ROOT, "models/stage16_efficientnet_b0/efficientnet_b0_best_loss.pt"),
        "expected_epoch": 2,
        "factory": lambda: get_stage16_efficientnet_b0(in_channels=1, num_classes=1, pretrained=False, dropout_rate=0.2),
        "params": 4008253,
        "prediction_filename": "efficientnet_b0_predictions.csv",
    },
}

# Positive class weight derived strictly from training split: 1,324 benign vs 915 malignant
POS_WEIGHT_SCALAR = 1324.0 / 915.0  # 1.446995


def compute_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """Compute McNemar's test for paired classification discordance."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B incorrect; c: A incorrect, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_discordant = b + c

    if n_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b": b,
            "c": c,
            "significant_at_05": False,
            "method": "Exact (No discordant pairs)",
        }

    if n_discordant < 25:
        # Exact binomial test for small sample sizes
        res = binomtest(b, n_discordant, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        stat = float(b)
        method = "Exact Binomial"
    else:
        # Edwards continuity correction
        stat = float(((abs(b - c) - 1.0) ** 2) / n_discordant)
        p_val = float(chi2.sf(stat, df=1))
        method = "Chi-squared with continuity correction"

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "b": b,
        "c": c,
        "significant_at_05": p_val < 0.05,
        "method": method,
    }


def compute_detailed_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
    pos_weight: float = POS_WEIGHT_SCALAR,
) -> Dict[str, Any]:
    """Compute clinical metrics with exact confusion matrix reconciliation."""
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = int(len(y_true))

    # Standard clinical metrics
    acc = float(accuracy_score(y_true, y_pred))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else None
    except Exception:
        auc = None

    # BCE Loss: unweighted and weighted
    eps = 1e-12
    p_clipped = np.clip(y_probs, eps, 1.0 - eps)
    unweighted_losses = -(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    unweighted_bce = float(np.mean(unweighted_losses))

    weighted_losses = -(pos_weight * y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    weighted_bce = float(np.mean(weighted_losses))

    return {
        "total_samples": total,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "unweighted_bce_loss": round(unweighted_bce, 5),
        "weighted_bce_loss": round(weighted_bce, 5),
        "pos_weight_applied": round(pos_weight, 5),
        "decision_threshold": threshold,
    }


def evaluate_single_model(
    model_key: str,
    test_loader: DataLoader,
    manifest_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute inference for a single candidate model against the test dataset."""
    spec = MODEL_REGISTRY[model_key]
    disp_name = spec["display_name"]
    ckpt_path = spec["checkpoint_path"]

    logger.info(f"Evaluating {disp_name} from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")

    # Build fresh architecture
    model = spec["factory"]().to(device)
    model.eval()

    # Load state dict
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_meta = {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "val_roc_auc": ckpt.get("val_roc_auc"),
            "selection_metric": ckpt.get("selection_metric", "val_loss"),
        }
    else:
        model.load_state_dict(ckpt)
        ckpt_meta = {"epoch": None, "val_loss": None, "selection_metric": "unknown"}

    all_logits = []
    all_probs = []
    all_targets = []
    all_meta_case_keys = []
    all_meta_patient_ids = []
    all_meta_image_paths = []

    eval_start_time = time.time()
    with torch.no_grad():
        for images, targets, meta in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_logits.extend(logits.squeeze(1).cpu().numpy().tolist())
            all_probs.extend(probs.squeeze(1).cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

            all_meta_case_keys.extend(meta["case_key"])
            all_meta_patient_ids.extend(meta["patient_id"])
            all_meta_image_paths.extend(meta["image_path"])

    eval_duration = time.time() - eval_start_time

    y_true = np.array(all_targets, dtype=int)
    y_probs = np.array(all_probs, dtype=float)
    y_logits = np.array(all_logits, dtype=float)
    y_preds = (y_probs >= 0.50).astype(int)

    # Sanity check record count
    if len(y_true) != len(manifest_df):
        raise RuntimeError(
            f"Record count mismatch for {disp_name}: predicted {len(y_true)}, expected {len(manifest_df)}"
        )

    # Compute overall metrics
    metrics = compute_detailed_metrics(y_true, y_probs, threshold=0.50)
    metrics["model_key"] = model_key
    metrics["display_name"] = disp_name
    metrics["stage"] = spec["stage"]
    metrics["checkpoint_path"] = ckpt_path
    metrics["checkpoint_metadata"] = ckpt_meta
    metrics["parameter_count"] = spec["params"]
    metrics["evaluation_time_seconds"] = round(eval_duration, 2)

    # Build per-record prediction DataFrame aligned with test manifest
    pred_df = manifest_df.copy()
    pred_df["predicted_logit"] = y_logits
    pred_df["predicted_probability"] = y_probs
    pred_df["predicted_class"] = y_preds
    pred_df["decision_threshold"] = 0.50
    pred_df["model_key"] = model_key
    pred_df["model_name"] = disp_name
    pred_df["checkpoint_path"] = ckpt_path
    pred_df["prediction_correct"] = (pred_df["predicted_class"] == pred_df["binary_label"])

    return metrics, pred_df


def plot_comparative_roc_and_confusion(
    all_metrics: Dict[str, Dict[str, Any]],
    all_predictions: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Dict[str, str]:
    """Generate comparative ROC curve overlay and confusion matrix grid."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    colors = {
        "baseline_cnn": "#1f77b4",
        "resnet50": "#2ca02c",
        "densenet121": "#d62728",
        "efficientnet_b0": "#9467bd",
    }

    # 1. Overlay ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    for model_key, spec in MODEL_REGISTRY.items():
        if model_key not in all_predictions:
            continue
        df = all_predictions[model_key]
        y_true = df["binary_label"].values
        y_probs = df["predicted_probability"].values
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc_val = all_metrics[model_key]["roc_auc"] or 0.0

        label_str = f"{spec['display_name']} (AUC = {auc_val:.4f})"
        ax.plot(fpr, tpr, color=colors.get(model_key, "black"), lw=2, label=label_str)

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("CBIS-DDSM Stage 17 — Clean Test Set Comparative ROC Curves (N = 644)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(output_dir, "stage17_1_comparative_roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    plot_paths["comparative_roc_curves"] = roc_path

    # 2. Confusion Matrix Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=150)
    classes = ["Benign (0)", "Malignant (1)"]

    model_keys = list(MODEL_REGISTRY.keys())
    for idx, model_key in enumerate(model_keys):
        r, c = divmod(idx, 2)
        ax_sub = axes[r, c]
        if model_key not in all_metrics:
            ax_sub.axis("off")
            continue

        cm = np.array(all_metrics[model_key]["confusion_matrix"])
        disp_name = MODEL_REGISTRY[model_key]["display_name"]
        acc = all_metrics[model_key]["accuracy"]
        f1 = all_metrics[model_key]["f1_score"]

        im = ax_sub.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax_sub.set_title(f"{disp_name}\nAcc: {acc:.4f} | F1: {f1:.4f}", fontsize=11, fontweight="bold")
        ax_sub.set_xticks([0, 1])
        ax_sub.set_yticks([0, 1])
        ax_sub.set_xticklabels(classes, fontsize=9)
        ax_sub.set_yticklabels(classes, fontsize=9)
        ax_sub.set_xlabel("Predicted Diagnosis", fontsize=9)
        ax_sub.set_ylabel("True Diagnosis", fontsize=9)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_sub.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold",
                )

    plt.suptitle("Stage 17.1 Test Confusion Matrices (Threshold = 0.50)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "stage17_1_confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    plot_paths["confusion_matrices_grid"] = cm_path

    return plot_paths


def run_stage17_evaluation(
    device_name: str = "auto",
    batch_size: int = 16,
    num_workers: int = 2,
) -> Dict[str, Any]:
    """Execute Stage 17 final evaluation pipeline across all 4 models."""
    start_time = time.time()

    # 1. Device Setup
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    logger.info("=" * 80)
    logger.info("CBIS-DDSM STAGE 17.1: FINAL COMPARATIVE MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Target compute device: {device}")
    if device.type == "cuda":
        clean_gpu_memory(0)
        gpu_info = get_device_info(0)
        logger.info(f"GPU: {gpu_info.get('gpu_name')} | VRAM: {gpu_info.get('total_memory_gb', 0)} GB")

    # 2. Output Directories
    metadata_dir = os.path.join(_PROJECT_ROOT, "data/metadata/stage17")
    results_dir = os.path.join(_PROJECT_ROOT, "results/stage17")
    for d in [metadata_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # 3. Load Stage 13 Test Manifest
    manifest_dir = os.path.join(_PROJECT_ROOT, "data/metadata/stage13")
    _, _, test_df = load_stage13_manifests(manifest_dir)
    logger.info(f"Loaded Stage 13 clean test manifest: {len(test_df)} records across {test_df['patient_id'].nunique()} patients.")

    if len(test_df) != 644:
        raise ValueError(f"Expected 644 test records, found {len(test_df)}")

    # Build clean test DataLoader
    test_ds = CBISDDSMClassificationDataset(
        manifest_df=test_df,
        split_name="test",
        image_size=(256, 256),
        augment=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 4. Evaluate each model sequentially
    all_metrics: Dict[str, Dict[str, Any]] = {}
    all_predictions: Dict[str, pd.DataFrame] = {}

    for model_key, spec in MODEL_REGISTRY.items():
        logger.info("-" * 80)
        logger.info(f"Evaluating candidate: {spec['display_name']} (Stage {spec['stage']})")
        logger.info("-" * 80)

        metrics, pred_df = evaluate_single_model(
            model_key=model_key,
            test_loader=test_loader,
            manifest_df=test_df,
            device=device,
        )
        all_metrics[model_key] = metrics
        all_predictions[model_key] = pred_df

        # Save individual prediction CSV
        pred_out_path = os.path.join(results_dir, spec["prediction_filename"])
        pred_df.to_csv(pred_out_path, index=False)
        logger.info(f"Saved {spec['display_name']} predictions: {pred_out_path} ({len(pred_df)} rows)")
        logger.info(
            f"Results: Acc={metrics['accuracy']:.4f} | Sens={metrics['sensitivity']:.4f} | "
            f"Spec={metrics['specificity']:.4f} | Prec={metrics['precision']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | AUC={metrics['roc_auc'] or 0.0:.4f} | "
            f"BCE={metrics['unweighted_bce_loss']:.4f}"
        )

    # 5. Pairwise McNemar Statistical Tests
    mcnemar_results = []
    keys = list(MODEL_REGISTRY.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k_a, k_b = keys[i], keys[j]
            y_true = all_predictions[k_a]["binary_label"].values
            pred_a = all_predictions[k_a]["predicted_class"].values
            pred_b = all_predictions[k_b]["predicted_class"].values

            test_res = compute_mcnemar_test(y_true, pred_a, pred_b)
            record = {
                "model_a": MODEL_REGISTRY[k_a]["display_name"],
                "model_b": MODEL_REGISTRY[k_b]["display_name"],
                "statistic": test_res["statistic"],
                "p_value": test_res["p_value"],
                "significant_at_05": test_res["significant_at_05"],
                "discordant_a_correct_b_wrong": test_res["b"],
                "discordant_b_correct_a_wrong": test_res["c"],
                "test_method": test_res["method"],
            }
            mcnemar_results.append(record)

    mcnemar_df = pd.DataFrame(mcnemar_results)
    mcnemar_path = os.path.join(results_dir, "stage17_1_mcnemar_significance.csv")
    mcnemar_df.to_csv(mcnemar_path, index=False)

    # 6. Subgroup Analysis: Masses vs Calcifications
    subgroup_records = []
    for model_key in keys:
        df = all_predictions[model_key]
        for sub_type in ["mass", "calcification"]:
            sub_df = df[df["abnormality_type"].str.lower() == sub_type]
            if len(sub_df) > 0:
                y_sub = sub_df["binary_label"].values
                p_sub = sub_df["predicted_probability"].values
                sub_metrics = compute_detailed_metrics(y_sub, p_sub, threshold=0.50)
                sub_metrics["model"] = MODEL_REGISTRY[model_key]["display_name"]
                sub_metrics["abnormality_type"] = sub_type
                subgroup_records.append(sub_metrics)

    subgroup_df = pd.DataFrame(subgroup_records)
    subgroup_path = os.path.join(results_dir, "stage17_1_subgroup_metrics.csv")
    subgroup_df.to_csv(subgroup_path, index=False)

    # 7. Model Comparison Table
    comparison_records = []
    for k in keys:
        m = all_metrics[k]
        comparison_records.append({
            "model_key": k,
            "architecture": m["display_name"],
            "stage": m["stage"],
            "parameters": m["parameter_count"],
            "selected_epoch": m["checkpoint_metadata"]["epoch"],
            "test_samples": m["total_samples"],
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
            "unweighted_bce_loss": m["unweighted_bce_loss"],
            "weighted_bce_loss": m["weighted_bce_loss"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"],
            "eval_time_sec": m["evaluation_time_seconds"],
            "checkpoint_path": m["checkpoint_path"],
        })

    comp_df = pd.DataFrame(comparison_records)
    comp_path = os.path.join(results_dir, "stage17_1_model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)

    # Save Confusion Matrices JSON
    cm_dict = {
        k: {
            "display_name": MODEL_REGISTRY[k]["display_name"],
            "confusion_matrix": all_metrics[k]["confusion_matrix"],
            "tn": all_metrics[k]["tn"],
            "fp": all_metrics[k]["fp"],
            "fn": all_metrics[k]["fn"],
            "tp": all_metrics[k]["tp"],
        }
        for k in keys
    }
    cm_json_path = os.path.join(results_dir, "stage17_1_confusion_matrices.json")
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)

    # 8. Plots
    plot_paths = plot_comparative_roc_and_confusion(all_metrics, all_predictions, results_dir)

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"STAGE 17.1 FINAL EVALUATION COMPLETE IN {total_time:.2f} SECONDS")
    logger.info(f"Comparison Table: {comp_path}")
    logger.info("=" * 80)

    return {
        "status": "SUCCESS",
        "models_evaluated": list(keys),
        "total_test_records": len(test_df),
        "comparison_table_path": comp_path,
        "mcnemar_path": mcnemar_path,
        "subgroup_path": subgroup_path,
        "confusion_matrix_path": cm_json_path,
        "plots": plot_paths,
        "metrics": all_metrics,
        "total_time_seconds": round(total_time, 2),
    }


def main():
    """CLI execution entrypoint."""
    run_stage17_evaluation()


if __name__ == "__main__":
    main()
