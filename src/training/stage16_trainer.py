"""Stage 16: Unified Transfer Learning Pipeline for DenseNet-121 and EfficientNet-B0.

Provides modular transfer learning for CBIS-DDSM mammography classification using:
1. DenseNet-121 (Dense feature concatenation, ~7.0M parameters).
2. EfficientNet-B0 (Compound scaling with MBConv, ~4.0M parameters).

Features:
- Single-channel luminance adaptation via channel weight averaging.
- Positive class weighting (pos_weight = 1324/915 ≈ 1.447) under BCEWithLogitsLoss.
- Mixed precision training (AMP) with gradient accumulation for 4 GB VRAM (RTX 2050).
- Dual checkpointing:
    * <model>_best_loss.pt (selected via minimal validation BCE loss, calibrated probabilities).
    * <model>_best_auc.pt  (selected via maximal validation ROC-AUC, discriminative ranking).
- Explicit metadata export distinguishing best-loss metrics from best-AUC metrics.
- Clean test partition strictly held back unblinded by default (evaluate_test=False).
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.datasets.stage14_dataloader import (
    get_stage14_dataloaders,
    resolve_image_path,
)
from src.gpu.device import get_device, get_device_info
from src.gpu.gpu_utils import clean_gpu_memory, empty_cache, get_memory_stats
from src.models.stage16_densenet import Stage16DenseNet121, get_stage16_densenet121
from src.models.stage16_efficientnet import Stage16EfficientNetB0, get_stage16_efficientnet_b0
from src.utils.logger import setup_logger

logger = setup_logger("Stage16Trainer")


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_torch_load(path: str, map_location: Any = None) -> Any:
    """Load a PyTorch checkpoint safely using weights_only=True where supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, ValueError):
        return torch.load(path, map_location=map_location)


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive clinical binary classification metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Specificity
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = None
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            auc = None

    return {
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "roc_auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    accum_steps: int = 1,
) -> float:
    """Run one epoch of training with optional AMP and gradient accumulation."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    use_cuda = (device.type == "cuda")
    optimizer.zero_grad()

    for step_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        if scaler and use_cuda:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
                loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader and return loss, metrics, true targets, and predicted probabilities."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    use_cuda = (device.type == "cuda")

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        if use_cuda:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.extend(targets.squeeze(1).cpu().numpy().tolist())
        all_probs.extend(probs.squeeze(1).cpu().numpy().tolist())

    epoch_loss = running_loss / max(total_samples, 1)
    y_true = np.array(all_targets, dtype=int)
    y_probs = np.array(all_probs, dtype=float)

    metrics = compute_metrics(y_true, y_probs)
    return epoch_loss, metrics, y_true, y_probs


def plot_stage16_results(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    history: List[Dict[str, Any]],
    output_dir: str,
    model_name: str,
    prefix: str = "val",
    y_test_true: Optional[np.ndarray] = None,
    y_test_probs: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Generate and save confusion matrix, ROC curve, and training history curves."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    y_eval_true = y_test_true if y_test_true is not None else y_val_true
    y_eval_probs = y_test_probs if y_test_probs is not None else y_val_probs
    eval_title = "Test" if y_test_true is not None else "Validation"

    # 1. Confusion Matrix
    y_pred = (y_eval_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_eval_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Benign (0)", "Malignant (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Stage 16 {model_name} — {eval_title} Confusion Matrix",
        ylabel="True Diagnosis",
        xlabel="Predicted Diagnosis",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    plot_paths[f"{prefix}_confusion_matrix"] = cm_path

    # 2. ROC Curve
    if len(np.unique(y_eval_true)) > 1:
        fpr, tpr, _ = roc_curve(y_eval_true, y_eval_probs)
        auc_val = roc_auc_score(y_eval_true, y_eval_probs)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"{model_name} (AUC = {auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(f"Stage 16 {model_name} — {eval_title} ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        plot_paths[f"{prefix}_roc_curve"] = roc_path

    # 3. Training & Validation Curves
    if history:
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_f1s = [h["val_f1"] for h in history]
        val_accs = [h["val_accuracy"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

        # Loss curves
        ax1.plot(epochs, train_losses, "b-o", label="Train Loss", lw=1.8)
        ax1.plot(epochs, val_losses, "r-s", label="Val Loss", lw=1.8)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE Loss")
        ax1.set_title(f"{model_name} Loss History")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Metric curves
        ax2.plot(epochs, val_accs, "g-^", label="Val Accuracy", lw=1.8)
        ax2.plot(epochs, val_f1s, "m-d", label="Val F1-Score", lw=1.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_title(f"{model_name} Validation Metrics")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(curves_path)
        plt.close()
        plot_paths["training_curves"] = curves_path

def resolve_stage16_paths(
    model_name: str,
    smoke_test: bool = False,
    project_root: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve directory and file paths for Stage 16 artifacts.

    Guarantees strict separation between smoke-test mode and full-training
    artifacts to prevent accidental overwrites of production checkpoints.

    Args:
        model_name: Architecture identifier ('densenet121' or 'efficientnet_b0').
        smoke_test: Whether paths are for a smoke test run.
        project_root: Optional project root path (defaults to _PROJECT_ROOT).

    Returns:
        Dictionary mapping artifact keys to their absolute or resolved paths.
    """
    root = project_root if project_root is not None else _PROJECT_ROOT
    canonical_model = "densenet121" if "dense" in model_name.lower() else "efficientnet_b0"

    metadata_dir = os.path.join(root, "data/metadata/stage16")

    if smoke_test:
        models_dir = os.path.join(root, "models/stage16_smoke_tests", canonical_model)
        results_dir = os.path.join(root, "results/stage16_smoke_tests", canonical_model)
        prefix = f"smoke_{canonical_model}"
        summary_filename = f"stage16_{canonical_model}_smoke_test_summary.json"
        epoch_history_filename = f"stage16_{canonical_model}_smoke_epoch_history.csv"
        val_loss_filename = f"stage16_{canonical_model}_smoke_validation_metrics_best_loss.csv"
        val_auc_filename = f"stage16_{canonical_model}_smoke_validation_metrics_best_auc.csv"
        config_filename = f"stage16_{canonical_model}_smoke_config.json"
    else:
        models_dir = os.path.join(root, f"models/stage16_{canonical_model}")
        results_dir = os.path.join(root, f"results/stage16_{canonical_model}")
        prefix = canonical_model
        summary_filename = f"stage16_{canonical_model}_smoke_test_summary.json"
        epoch_history_filename = f"stage16_{canonical_model}_epoch_history.csv"
        val_loss_filename = f"stage16_{canonical_model}_validation_metrics_best_loss.csv"
        val_auc_filename = f"stage16_{canonical_model}_validation_metrics_best_auc.csv"
        config_filename = f"stage16_{canonical_model}_config.json"

    return {
        "canonical_model": canonical_model,
        "metadata_dir": metadata_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "best_loss_checkpoint_path": os.path.join(models_dir, f"{prefix}_best_loss.pt"),
        "best_auc_checkpoint_path": os.path.join(models_dir, f"{prefix}_best_auc.pt"),
        "final_checkpoint_path": os.path.join(models_dir, f"{prefix}_final.pt"),
        "smoke_summary_path": os.path.join(metadata_dir, summary_filename),
        "epoch_history_path": os.path.join(metadata_dir, epoch_history_filename),
        "val_loss_metrics_path": os.path.join(metadata_dir, val_loss_filename),
        "val_auc_metrics_path": os.path.join(metadata_dir, val_auc_filename),
        "test_metrics_path": os.path.join(metadata_dir, f"stage16_{canonical_model}_test_metrics.csv"),
        "config_path": os.path.join(metadata_dir, config_filename),
    }


def build_stage16_model(
    model_name: str,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
) -> Tuple[nn.Module, str]:
    """Factory helper to build requested Stage 16 architecture."""
    norm_name = model_name.lower().replace("-", "").replace("_", "")
    if "dense" in norm_name:
        model = get_stage16_densenet121(
            in_channels=1,
            num_classes=1,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
        arch_display = "DenseNet-121"
    elif "efficient" in norm_name:
        model = get_stage16_efficientnet_b0(
            in_channels=1,
            num_classes=1,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
        arch_display = "EfficientNet-B0"
    else:
        raise ValueError(f"Unsupported Stage 16 model '{model_name}'. Choose 'densenet121' or 'efficientnet_b0'.")

    return model, arch_display


def run_stage16_training(
    model_name: str = "densenet121",
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-4,
    accum_steps: int = 2,
    early_stopping_patience: int = 5,
    image_size: int = 256,
    num_workers: int = 2,
    seed: int = 42,
    smoke_test: bool = False,
    evaluate_test: bool = False,
    primary_views_only: bool = False,
    pretrained: bool = True,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Execute Stage 16 transfer learning with dual checkpointing and validation selection."""
    start_total_time = time.time()
    set_seed(seed)

    # 1. Resolve compute device and clean memory
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda":
        clean_gpu_memory(0)
        gpu_info = get_device_info(0)
    else:
        gpu_info = {"device": "cpu"}

    # Format model identifier
    canonical_model = "densenet121" if "dense" in model_name.lower() else "efficientnet_b0"

    logger.info("=" * 80)
    logger.info(f"STARTING STAGE 16 — {canonical_model.upper()} TRANSFER LEARNING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Target compute device: {device} | Smoke test mode: {smoke_test} | Evaluate test: {evaluate_test}")
    if device.type == "cuda":
        logger.info(f"GPU: {gpu_info.get('gpu_name')} | VRAM: {gpu_info.get('total_memory_gb', 0)} GB")

    # Dedicated Stage 16 directories & artifact path resolution
    paths = resolve_stage16_paths(canonical_model, smoke_test=smoke_test)
    metadata_dir = paths["metadata_dir"]
    models_dir = paths["models_dir"]
    results_dir = paths["results_dir"]

    for d in [metadata_dir, models_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Build DataLoaders from Stage 13 manifests
    logger.info("Loading Stage 13 classification manifests (2,239 train, 542 val, 644 clean test)...")
    train_loader, val_loader, test_loader, loader_info = get_stage14_dataloaders(
        manifest_dir="data/metadata/stage13",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        num_workers=num_workers,
        augment_train=True,
        primary_views_only=primary_views_only,
    )

    if smoke_test:
        logger.info("[SMOKE TEST] Truncating dataloaders: 32 train samples and 16 val samples...")
        logger.info("[SMOKE TEST] Clean test manifest is strictly excluded from smoke test to maintain isolation.")
        epochs = 1
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(min(32, len(train_loader.dataset)))),
            batch_size=min(16, batch_size), shuffle=False, num_workers=0
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(min(16, len(val_loader.dataset)))),
            batch_size=min(16, batch_size), shuffle=False, num_workers=0
        )
        test_loader = None

    # 3. Instantiate model
    model, arch_display = build_stage16_model(canonical_model, pretrained=pretrained, dropout_rate=0.3)
    model = model.to(device)

    weight_meta = model.get_weight_metadata()
    logger.info(f"Architecture: {arch_display} ({weight_meta['total_parameters']:,} parameters)")
    logger.info(f"Pretrained weights status: loaded={weight_meta['pretrained_loaded']}, version={weight_meta['weight_version']}")

    # 4. Loss function with positive class weighting
    # Training distribution: 1,324 benign (0) vs. 915 malignant (1)
    pos_weight = torch.tensor([1324.0 / 915.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 5. Dual Checkpoint Tracking
    best_val_loss = float("inf")
    best_val_loss_metrics = {}
    best_loss_epoch = 0

    best_val_auc = -1.0
    best_val_auc_metrics = {}
    best_auc_epoch = 0

    patience_counter = 0
    epoch_history = []

    best_loss_checkpoint_path = paths["best_loss_checkpoint_path"]
    best_auc_checkpoint_path = paths["best_auc_checkpoint_path"]
    final_checkpoint_path = paths["final_checkpoint_path"]

    logger.info(f"Commencing training: {epochs} max epochs, batch_size={batch_size}, accum_steps={accum_steps}, lr={learning_rate}...")
    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, accum_steps)
            val_loss, val_metrics, y_val_true, y_val_probs = evaluate_loader(model, val_loader, criterion, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if "out of memory" in str(oom_err).lower():
                clean_gpu_memory(0)
                mem = get_memory_stats(0)
                logger.error(
                    f"CUDA Out of Memory during epoch {epoch}! Allocated: {mem.get('allocated_mb')}MB. "
                    f"Please reduce --batch-size (current: {batch_size}) or increase --accum-steps."
                )
            raise oom_err

        scheduler.step(val_loss)
        ep_time = time.time() - ep_start

        current_auc = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else 0.0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_sensitivity": round(val_metrics["sensitivity"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_f1": round(val_metrics["f1_score"], 4),
            "val_roc_auc": round(current_auc, 4),
            "epoch_time_seconds": round(ep_time, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_history.append(record)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
            f"Val AUC: {current_auc:.4f}"
        )

        # 5a. Primary Checkpoint Selection: Validation Loss Minimization
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_metrics = val_metrics
            best_loss_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "selection_metric": "val_loss",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "weight_metadata": weight_meta,
                "config": {
                    "architecture": arch_display,
                    "model_key": canonical_model,
                    "seed": seed,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "accum_steps": accum_steps,
                    "lr": learning_rate,
                },
            }, best_loss_checkpoint_path)
            logger.info(f"--> [BEST LOSS] Saved new best-loss checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # 5b. Secondary Checkpoint Selection: Validation ROC-AUC Maximization
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            best_val_auc_metrics = val_metrics
            best_auc_epoch = epoch

            torch.save({
                "epoch": epoch,
                "selection_metric": "val_roc_auc",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_roc_auc": current_auc,
                "val_metrics": val_metrics,
                "weight_metadata": weight_meta,
                "config": {
                    "architecture": arch_display,
                    "model_key": canonical_model,
                    "seed": seed,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "accum_steps": accum_steps,
                    "lr": learning_rate,
                },
            }, best_auc_checkpoint_path)
            logger.info(f"--> [BEST AUC] Saved new best-AUC checkpoint at epoch {epoch} (Val AUC: {current_auc:.4f})")

        # Early stopping based on primary metric (val_loss)
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (No loss improvement for {patience_counter} epochs).")
            break

    total_training_time = time.time() - train_start_time

    # Save final model state
    torch.save(model.state_dict(), final_checkpoint_path)

    # 6. Smoke-Test Early Exit
    if smoke_test:
        logger.info("=" * 80)
        logger.info("[SMOKE TEST] VALIDATING CHECKPOINT SAVE AND LOAD INTEGRITY...")
        logger.info("=" * 80)
        test_reload_model, _ = build_stage16_model(canonical_model, pretrained=False, dropout_rate=0.3)
        test_reload_model = test_reload_model.to(device)

        loaded_ckpt = safe_torch_load(best_loss_checkpoint_path, map_location=device)
        test_reload_model.load_state_dict(loaded_ckpt["model_state_dict"])
        logger.info(f"[SMOKE TEST] Best-loss checkpoint reloaded successfully into fresh {arch_display} instance.")
        logger.info(
            f"[SMOKE TEST] Verified metrics from best epoch {best_loss_epoch}: "
            f"Val Loss={best_val_loss:.4f}, Val Acc={best_val_loss_metrics['accuracy']:.4f}"
        )
        logger.info("[SMOKE TEST] Clean test manifest was NOT evaluated to maintain strict data isolation.")
        logger.info("=" * 80)

        smoke_summary = {
            "status": "SMOKE_TEST_SUCCESS",
            "architecture": arch_display,
            "model_key": canonical_model,
            "pretrained_status": weight_meta["pretrained_loaded"],
            "weight_version": weight_meta["weight_version"],
            "epochs_run": 1,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "train_loss": round(train_loss, 5),
            "val_loss": round(best_val_loss, 5),
            "val_metrics": best_val_loss_metrics,
            "checkpoint_reloaded": True,
            "best_loss_checkpoint_path": best_loss_checkpoint_path,
            "best_auc_checkpoint_path": best_auc_checkpoint_path,
            "clean_test_evaluated": False,
            "total_time_seconds": round(time.time() - start_total_time, 2),
        }
        summary_path = paths["smoke_summary_path"]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(smoke_summary, f, indent=2)

        return smoke_summary

    # 7. Generate Validation Plots & Save Training History
    logger.info("Generating validation figures...")
    plot_paths = plot_stage16_results(
        y_val_true=y_val_true,
        y_val_probs=y_val_probs,
        history=epoch_history,
        output_dir=results_dir,
        model_name=arch_display,
        prefix="val",
    )

    history_df = pd.DataFrame(epoch_history)
    history_df.to_csv(paths["epoch_history_path"], index=False)

    # Export best-loss validation metrics
    val_loss_df = pd.DataFrame([best_val_loss_metrics])
    val_loss_df.to_csv(paths["val_loss_metrics_path"], index=False)

    # Export best-auc validation metrics separately to prevent conflation
    val_auc_df = pd.DataFrame([best_val_auc_metrics])
    val_auc_df.to_csv(paths["val_auc_metrics_path"], index=False)

    # 8. Clean Test Evaluation Handling (Held back by default)
    test_metrics = None
    test_loss = None
    total_test_time = 0.0

    if evaluate_test:
        logger.info("=" * 80)
        logger.info(f"EVALUATING BEST {arch_display.upper()} CHECKPOINT ON CLEAN TEST SET...")
        logger.info("=" * 80)
        best_ckpt = safe_torch_load(best_loss_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_start_time = time.time()
        test_loss, test_metrics, y_test_true, y_test_probs = evaluate_loader(model, test_loader, criterion, device)
        total_test_time = time.time() - test_start_time

        logger.info(f"Test Evaluation Complete in {total_test_time:.2f}s:")
        logger.info(f"Test Loss:        {test_loss:.4f}")
        logger.info(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
        logger.info(f"Test Specificity: {test_metrics['specificity']:.4f}")
        logger.info(f"Test Precision:   {test_metrics['precision']:.4f}")
        logger.info(f"Test F1-Score:    {test_metrics['f1_score']:.4f}")
        logger.info(f"Test ROC-AUC:     {test_metrics['roc_auc'] or 0.0:.4f}")

        test_plot_paths = plot_stage16_results(
            y_val_true=y_val_true,
            y_val_probs=y_val_probs,
            history=[],
            output_dir=results_dir,
            model_name=arch_display,
            prefix="test",
            y_test_true=y_test_true,
            y_test_probs=y_test_probs,
        )
        plot_paths.update(test_plot_paths)

        test_metrics_record = dict(test_metrics)
        test_metrics_record["test_loss"] = round(test_loss, 5)
        test_metrics_record["test_time_seconds"] = round(total_test_time, 2)
        test_metrics_record["total_test_samples"] = len(y_test_true)
        test_metrics_df = pd.DataFrame([test_metrics_record])
        test_metrics_df.to_csv(paths["test_metrics_path"], index=False)
    else:
        logger.info("=" * 80)
        logger.info(f"[STAGE 16] CLEAN TEST SET EVALUATION HELD BACK FOR {arch_display.upper()}.")
        logger.info("Preserving pristine test set for final comparative evaluation across models.")
        logger.info("=" * 80)
        test_status_df = pd.DataFrame([{
            "status": "HELD_BACK_FOR_FINAL_COMPARISON",
            "model": canonical_model,
            "rationale": "Deferred until all comparative models are frozen to protect against test exposure",
            "clean_test_manifest": "data/metadata/stage13/classification_test_manifest.csv",
            "total_test_samples": 644,
        }])
        test_status_df.to_csv(paths["test_metrics_path"], index=False)

    # 9. Save Configuration JSON
    stage16_config = {
        "stage": 16,
        "stage_title": f"{arch_display} Transfer Learning for CBIS-DDSM",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "device_info": gpu_info,
        },
        "hyperparameters": {
            "model_key": canonical_model,
            "architecture": arch_display,
            "epochs": epochs,
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "effective_batch_size": batch_size * accum_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "image_size": [image_size, image_size],
            "num_workers": num_workers,
            "seed": seed,
            "smoke_test": smoke_test,
            "evaluate_test": evaluate_test,
            "primary_views_only": primary_views_only,
        },
        "model_architecture": {
            "name": f"Stage16{arch_display.replace('-', '').replace(' ', '')}",
            "in_channels": 1,
            "num_classes": 1,
            "pretrained_status": weight_meta["pretrained_loaded"],
            "weight_version": weight_meta["weight_version"],
            "adaptation_strategy": "channel_weight_averaging",
            "parameters": {
                "total_parameters": weight_meta["total_parameters"],
                "trainable_parameters": weight_meta["trainable_parameters"],
                "backbone_parameters": weight_meta["backbone_parameters"],
                "classifier_parameters": weight_meta["classifier_parameters"],
            },
        },
        "loss_function": {
            "name": "BCEWithLogitsLoss",
            "pos_weight": float(pos_weight.item()),
            "rationale": "Compensates for 1.45:1 benign-to-malignant class ratio in training set (1,324 benign vs. 915 malignant)",
        },
        "training_summary": {
            "best_loss_epoch": best_loss_epoch,
            "best_val_loss": round(best_val_loss, 5),
            "best_val_loss_metrics": best_val_loss_metrics,
            "best_auc_epoch": best_auc_epoch,
            "best_val_auc": round(best_val_auc, 5),
            "best_val_auc_metrics": best_val_auc_metrics,
            "total_training_time_seconds": round(total_training_time, 2),
        },
        "checkpoints": {
            "best_loss_checkpoint": best_loss_checkpoint_path,
            "best_auc_checkpoint": best_auc_checkpoint_path,
            "final_checkpoint": final_checkpoint_path,
        },
        "test_results": {
            "evaluated": evaluate_test,
            "status": "COMPLETED" if evaluate_test else "HELD_BACK_FOR_FINAL_COMPARISON",
            "loss": round(test_loss, 5) if test_loss is not None else None,
            "metrics": test_metrics,
        },
        "comparative_references": {
            "stage14_baseline_cnn": {
                "architecture": "BaselineBreastCancerCNN (421,601 params)",
                "test_accuracy": 0.5947,
                "test_sensitivity": 0.7237,
                "test_specificity": 0.5090,
                "test_f1_score": 0.5877,
                "test_roc_auc": 0.6669,
            },
            "stage15_resnet50": {
                "architecture": "Stage15ResNet50 (23,503,809 params)",
                "val_loss": 0.60488,
                "val_accuracy": 0.7528,
                "val_sensitivity": 0.7261,
                "val_specificity": 0.7724,
                "val_f1_score": 0.7137,
                "val_roc_auc": 0.8326,
            },
        },
    }

    config_path = paths["config_path"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(stage16_config, f, indent=2)

    return {
        "status": "SUCCESS",
        "model": canonical_model,
        "best_loss_epoch": best_loss_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_metrics": best_val_loss_metrics,
        "best_auc_epoch": best_auc_epoch,
        "best_val_auc": best_val_auc,
        "best_val_auc_metrics": best_val_auc_metrics,
        "test_evaluated": evaluate_test,
        "test_metrics": test_metrics,
        "plot_paths": plot_paths,
        "total_time": round(time.time() - start_total_time, 2),
    }


def main():
    """CLI entrypoint for Stage 16 training pipeline."""
    parser = argparse.ArgumentParser(description="CBIS-DDSM Stage 16: DenseNet-121 and EfficientNet-B0 Transfer Learning")
    parser.add_argument("--model", type=str, default="densenet121", choices=["densenet121", "efficientnet_b0"],
                        help="Target architecture ('densenet121' or 'efficientnet_b0')")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU step (default: 8)")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution H and W (default: 256)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 1-epoch smoke test without evaluating test set")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate best checkpoint on clean test set")
    parser.add_argument("--primary-views-only", action="store_true", help="Filter to primary unique mammographic views")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ImageNet weights")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")

    args = parser.parse_args()

    run_stage16_training(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        accum_steps=args.accum_steps,
        early_stopping_patience=args.patience,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        smoke_test=args.smoke_test,
        evaluate_test=args.evaluate_test,
        primary_views_only=args.primary_views_only,
        pretrained=not args.no_pretrained,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
