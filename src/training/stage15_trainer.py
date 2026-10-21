"""Stage 15: ResNet-50 Transfer Learning Pipeline for CBIS-DDSM Mammography.

Implements transfer-learning classification using a pretrained ResNet-50 backbone,
single-channel luminance-preserving convolution adaptation, positive-class weighted BCE loss,
and strict patient-level split isolation.

Designed for resource-conscious execution within a 4 GB VRAM GPU budget (e.g., NVIDIA RTX 2050):
1. Conservative default batch size (8 or 16) with optional gradient accumulation.
2. Automatic Mixed Precision (AMP via torch.amp.autocast and GradScaler).
3. CUDA VRAM caching cleanup and graceful Out-Of-Memory diagnostic handling.
4. Safe CPU fallback.
5. Model selection strictly on validation data (val_loss).
6. Clean test evaluation deferred / held back by default to protect against repeated test exposure.
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
from src.models.stage15_resnet50 import Stage15ResNet50, get_stage15_resnet50
from src.utils.logger import setup_logger

logger = setup_logger("Stage15Trainer")


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


def plot_stage15_results(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    history: List[Dict[str, Any]],
    output_dir: str,
    prefix: str = "val",
    y_test_true: Optional[np.ndarray] = None,
    y_test_probs: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Generate and save confusion matrix, ROC curve, and training history curves."""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # 1. Validation (or Test) Confusion Matrix
    y_eval_true = y_test_true if y_test_true is not None else y_val_true
    y_eval_probs = y_test_probs if y_test_probs is not None else y_val_probs
    eval_title = "Test" if y_test_true is not None else "Validation"

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
        title=f"Stage 15 ResNet-50 — {eval_title} Confusion Matrix",
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
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ResNet-50 (AUC = {auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title(f"Stage 15 ResNet-50 — {eval_title} ROC Curve")
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
        ax1.set_title("ResNet-50 Loss History")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Metric curves
        ax2.plot(epochs, val_accs, "g-^", label="Val Accuracy", lw=1.8)
        ax2.plot(epochs, val_f1s, "m-d", label="Val F1-Score", lw=1.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_title("ResNet-50 Validation Metrics")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(curves_path)
        plt.close()
        plot_paths["training_curves"] = curves_path

    return plot_paths


def run_stage15_training(
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
    """Execute Stage 15 ResNet-50 transfer learning, validation checkpoint selection, and audit reporting."""
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

    logger.info("=" * 80)
    logger.info("STARTING STAGE 15 — RESNET-50 TRANSFER LEARNING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Target compute device: {device} | Smoke test mode: {smoke_test} | Evaluate test: {evaluate_test}")
    if device.type == "cuda":
        logger.info(f"GPU: {gpu_info.get('gpu_name')} | VRAM: {gpu_info.get('total_memory_gb', 0)} GB")

    # Set up dedicated Stage 15 directories
    metadata_dir = os.path.join(_PROJECT_ROOT, "data/metadata/stage15")
    models_dir = os.path.join(_PROJECT_ROOT, "models/stage15_resnet50")
    results_dir = os.path.join(_PROJECT_ROOT, "results/stage15_resnet50")

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

    # 3. Instantiate Stage 15 ResNet-50
    logger.info("Instantiating Stage15ResNet50 with single-channel luminance adaptation...")
    try:
        model = get_stage15_resnet50(
            in_channels=1,
            num_classes=1,
            pretrained=pretrained,
            dropout_rate=0.3,
        ).to(device)
    except Exception as e:
        logger.error(f"Failed to instantiate ResNet-50: {e}")
        raise

    weight_meta = model.get_weight_metadata()
    logger.info(f"Model parameters: {weight_meta['total_parameters']:,} total ({weight_meta['trainable_parameters']:,} trainable)")
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

    # 5. Training Loop with Early Stopping
    logger.info(f"Commencing training: {epochs} max epochs, batch_size={batch_size}, accum_steps={accum_steps}, lr={learning_rate}...")
    best_val_loss = float("inf")
    best_val_metrics = {}
    best_epoch = 0
    patience_counter = 0
    epoch_history = []

    best_checkpoint_path = os.path.join(models_dir, "resnet50_best.pt")
    final_checkpoint_path = os.path.join(models_dir, "resnet50_final.pt")

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
                    f"Please reduce --batch-size (current: {batch_size}) or use --accum-steps."
                )
            raise oom_err

        scheduler.step(val_loss)
        ep_time = time.time() - ep_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_sensitivity": round(val_metrics["sensitivity"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_f1": round(val_metrics["f1_score"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4) if val_metrics["roc_auc"] is not None else None,
            "epoch_time_seconds": round(ep_time, 2),
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_history.append(record)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc'] or 0.0:.4f}"
        )

        # Checkpoint selection strictly on validation data
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "weight_metadata": weight_meta,
                "config": {
                    "architecture": "ResNet-50",
                    "seed": seed,
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "accum_steps": accum_steps,
                    "lr": learning_rate,
                },
            }, best_checkpoint_path)
            logger.info(f"--> Saved new best checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {patience_counter} epochs).")
                break

    total_training_time = time.time() - train_start_time

    # Save final model state
    torch.save(model.state_dict(), final_checkpoint_path)

    # 6. Smoke-Test Early Exit
    if smoke_test:
        logger.info("=" * 80)
        logger.info("[SMOKE TEST] VALIDATING CHECKPOINT SAVE AND LOAD INTEGRITY...")
        logger.info("=" * 80)
        test_reload_model = get_stage15_resnet50(in_channels=1, num_classes=1, pretrained=False).to(device)
        loaded_ckpt = safe_torch_load(best_checkpoint_path, map_location=device)
        test_reload_model.load_state_dict(loaded_ckpt["model_state_dict"])
        logger.info("[SMOKE TEST] Checkpoint reloaded successfully into fresh ResNet-50 model instance.")
        logger.info(
            f"[SMOKE TEST] Verified metrics from best epoch {best_epoch}: "
            f"Val Loss={best_val_loss:.4f}, Val Acc={best_val_metrics['accuracy']:.4f}"
        )
        logger.info("[SMOKE TEST] Clean test manifest was NOT evaluated to maintain strict data isolation.")
        logger.info("=" * 80)

        smoke_summary = {
            "status": "SMOKE_TEST_SUCCESS",
            "architecture": "ResNet-50",
            "pretrained_status": weight_meta["pretrained_loaded"],
            "weight_version": weight_meta["weight_version"],
            "epochs_run": 1,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "train_loss": round(train_loss, 5),
            "val_loss": round(best_val_loss, 5),
            "val_metrics": best_val_metrics,
            "checkpoint_reloaded": True,
            "best_checkpoint_path": best_checkpoint_path,
            "clean_test_evaluated": False,
            "total_time_seconds": round(time.time() - start_total_time, 2),
        }
        with open(os.path.join(metadata_dir, "stage15_smoke_test_summary.json"), "w", encoding="utf-8") as f:
            json.dump(smoke_summary, f, indent=2)

        return smoke_summary

    # 7. Generate Validation Plots & Save Training History
    logger.info("Generating validation figures...")
    plot_paths = plot_stage15_results(
        y_val_true=y_val_true,
        y_val_probs=y_val_probs,
        history=epoch_history,
        output_dir=results_dir,
        prefix="val",
    )

    history_df = pd.DataFrame(epoch_history)
    history_df.to_csv(os.path.join(metadata_dir, "stage15_epoch_history.csv"), index=False)

    val_metrics_df = pd.DataFrame([best_val_metrics])
    val_metrics_df.to_csv(os.path.join(metadata_dir, "stage15_validation_metrics.csv"), index=False)

    # 8. Clean Test Evaluation Handling (Held back by default)
    test_metrics = None
    test_loss = None
    total_test_time = 0.0

    if evaluate_test:
        logger.info("=" * 80)
        logger.info("EVALUATING BEST RESNET-50 CHECKPOINT ON CLEAN TEST SET...")
        logger.info("=" * 80)
        best_ckpt = safe_torch_load(best_checkpoint_path, map_location=device)
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
        logger.info(f"Confusion Matrix: TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}")

        test_plot_paths = plot_stage15_results(
            y_val_true=y_val_true,
            y_val_probs=y_val_probs,
            history=[],
            output_dir=results_dir,
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
        test_metrics_df.to_csv(os.path.join(metadata_dir, "stage15_test_metrics.csv"), index=False)
    else:
        logger.info("=" * 80)
        logger.info("[STAGE 15] CLEAN TEST SET EVALUATION HELD BACK BY DEFAULT.")
        logger.info("Preserving pristine test set for final comparative evaluation across models.")
        logger.info("=" * 80)
        # Record explicit held-back status in test metrics CSV
        test_status_df = pd.DataFrame([{
            "status": "HELD_BACK_FOR_FINAL_COMPARISON",
            "rationale": "Deferred until all comparative models are frozen to protect against test exposure",
            "clean_test_manifest": "data/metadata/stage13/classification_test_manifest.csv",
            "total_test_samples": 644,
        }])
        test_status_df.to_csv(os.path.join(metadata_dir, "stage15_test_metrics.csv"), index=False)

    # 9. Save Configuration JSON
    stage15_config = {
        "stage": 15,
        "stage_title": "ResNet-50 Transfer Learning for CBIS-DDSM",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "device_info": gpu_info,
        },
        "hyperparameters": {
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
            "name": "Stage15ResNet50",
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
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 5),
            "best_val_metrics": best_val_metrics,
            "total_training_time_seconds": round(total_training_time, 2),
        },
        "test_results": {
            "evaluated": evaluate_test,
            "status": "COMPLETED" if evaluate_test else "HELD_BACK_FOR_FINAL_COMPARISON",
            "loss": round(test_loss, 5) if test_loss is not None else None,
            "metrics": test_metrics,
        },
        "stage14_baseline_reference": {
            "architecture": "BaselineBreastCancerCNN (421,601 params)",
            "test_accuracy": 0.5947,
            "test_sensitivity": 0.7237,
            "test_specificity": 0.5090,
            "test_f1_score": 0.5877,
            "test_roc_auc": 0.6669,
            "note": "Reference only. Do not claim ResNet-50 outperforms baseline until frozen comparison is executed.",
        },
    }

    with open(os.path.join(metadata_dir, "stage15_resnet50_config.json"), "w", encoding="utf-8") as f:
        json.dump(stage15_config, f, indent=2)

    with open(os.path.join(results_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stage15_config, f, indent=2)

    # 10. Generate Text and Markdown Reports
    txt_report = generate_stage15_text_report(stage15_config, best_val_metrics, test_metrics)
    md_report = generate_stage15_markdown_report(stage15_config, best_val_metrics, test_metrics)

    with open(os.path.join(metadata_dir, "stage15_resnet50_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt_report)

    with open(os.path.join(metadata_dir, "stage15_resnet50_report.md"), "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info("=" * 80)
    logger.info("STAGE 15 RESNET-50 PIPELINE COMPLETED SUCCESSFULLY.")
    logger.info(f"Artifacts saved under: {metadata_dir}")
    logger.info("=" * 80)

    return {
        "status": "SUCCESS",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_metrics": best_val_metrics,
        "test_evaluated": evaluate_test,
        "test_metrics": test_metrics,
        "plot_paths": plot_paths,
        "total_time": round(time.time() - start_total_time, 2),
    }


def generate_stage15_text_report(
    cfg: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]],
) -> str:
    """Produce human-readable plain text report for Stage 15."""
    lines = [
        "=" * 80,
        "CBIS-DDSM STAGE 15 — RESNET-50 TRANSFER LEARNING REPORT",
        "=" * 80,
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device:    {cfg['hardware']['device']}",
        f"Architecture: {cfg['model_architecture']['name']} ({cfg['model_architecture']['parameters']['total_parameters']:,} params)",
        f"Pretrained Weights: {cfg['model_architecture']['weight_version']} (Loaded: {cfg['model_architecture']['pretrained_status']})",
        f"Input Adaptation:  {cfg['model_architecture']['adaptation_strategy']}",
        "",
        "1. HARDWARE & RESOURCE MANAGEMENT",
        f"  - Device:             {cfg['hardware']['device']}",
        f"  - GPU Name:           {cfg['hardware']['device_info'].get('gpu_name', 'N/A')}",
        f"  - Batch Size:         {cfg['hyperparameters']['batch_size']} (Accum steps: {cfg['hyperparameters']['accum_steps']}, Effective batch: {cfg['hyperparameters']['effective_batch_size']})",
        f"  - Image Resolution:   {cfg['hyperparameters']['image_size'][0]}x{cfg['hyperparameters']['image_size'][1]}",
        f"  - Mixed Precision:    AMP (autocast + GradScaler)",
        "",
        "2. DATASET SPLIT ISOLATION",
        "  - Training Set:       2,239 records (Gradient updates only)",
        "  - Validation Set:     542 records (Model selection & early stopping)",
        "  - Clean Test Set:     644 records (Pristine held-back partition)",
        "  - Quarantined:        143 records (Excluded: cross-partition patient conflicts)",
        "  - Patient Leakage:    0.0% (Zero patient overlap verified in Stage 13.1)",
        "",
        "3. VALIDATION PERFORMANCE (MODEL SELECTION)",
        f"  - Best Epoch:         {cfg['training_summary']['best_epoch']}",
        f"  - Best Val Loss:      {cfg['training_summary']['best_val_loss']:.4f}",
        f"  - Val Accuracy:       {val_metrics.get('accuracy', 0.0):.4f}",
        f"  - Val Sensitivity:    {val_metrics.get('sensitivity', 0.0):.4f}",
        f"  - Val Specificity:    {val_metrics.get('specificity', 0.0):.4f}",
        f"  - Val Precision:      {val_metrics.get('precision', 0.0):.4f}",
        f"  - Val F1-Score:       {val_metrics.get('f1_score', 0.0):.4f}",
        f"  - Val ROC-AUC:        {val_metrics.get('roc_auc') or 0.0:.4f}",
        "",
        "4. CLEAN TEST EVALUATION STATUS",
    ]

    if test_metrics:
        lines.extend([
            f"  - Test Status:        EVALUATED",
            f"  - Test Loss:          {cfg['test_results']['loss']:.4f}",
            f"  - Test Accuracy:      {test_metrics['accuracy']:.4f}",
            f"  - Test Sensitivity:   {test_metrics['sensitivity']:.4f}",
            f"  - Test Specificity:   {test_metrics['specificity']:.4f}",
            f"  - Test Precision:     {test_metrics['precision']:.4f}",
            f"  - Test F1-Score:      {test_metrics['f1_score']:.4f}",
            f"  - Test ROC-AUC:       {test_metrics.get('roc_auc') or 0.0:.4f}",
            f"  - Confusion Matrix:   TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}",
        ])
    else:
        lines.extend([
            "  - Test Status:        HELD_BACK_FOR_FINAL_COMPARISON",
            "  - Rationale:          Test partition held pristine to avoid repeated testing before cross-model comparison.",
        ])

    lines.extend([
        "",
        "5. STAGE 14 BASELINE CNN COMPARISON REFERENCE",
        "  - Architecture:       BaselineBreastCancerCNN (421,601 parameters)",
        "  - Stage 14 Test Acc:  59.47%",
        "  - Stage 14 Test Sens: 72.37%",
        "  - Stage 14 Test Spec: 50.90%",
        "  - Stage 14 Test F1:   58.77%",
        "  - Stage 14 Test AUC:  0.6669",
        "  - Note: Do not claim ResNet-50 outperforms baseline until frozen comparison is executed.",
        "=" * 80,
    ])
    return "\n".join(lines)


def generate_stage15_markdown_report(
    cfg: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]],
) -> str:
    """Produce structured GitHub Markdown report for Stage 15."""
    test_status_text = "EVALUATED" if test_metrics else "HELD BACK FOR FINAL COMPARISON"

    md = f"""# CBIS-DDSM Stage 15 — ResNet-50 Transfer Learning Report

**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Compute Device:** `{cfg['hardware']['device']}`  
**Architecture:** `{cfg['model_architecture']['name']}` ({cfg['model_architecture']['parameters']['total_parameters']:,} parameters)  
**Pretrained Weights:** `{cfg['model_architecture']['weight_version']}` (Loaded: `{cfg['model_architecture']['pretrained_status']}`)  
**Clean Test Status:** `{test_status_text}`  

---

## 1. Hardware & Memory Management (RTX 2050 4 GB VRAM)

| Parameter | Value | Rationale |
|---|---|---|
| **Device** | `{cfg['hardware']['device']}` | NVIDIA RTX 2050 (4 GB VRAM) |
| **Batch Size** | `{cfg['hyperparameters']['batch_size']}` | Conservative allocation to prevent CUDA OOM |
| **Gradient Accumulation** | `{cfg['hyperparameters']['accum_steps']}` steps | Yields effective batch size of `{cfg['hyperparameters']['effective_batch_size']}` |
| **Image Resolution** | `{cfg['hyperparameters']['image_size'][0]} × {cfg['hyperparameters']['image_size'][1]}` | Matches Stage 5/14 baseline resolution |
| **Precision** | `AMP (torch.amp.autocast)` | ~50% VRAM memory savings and Tensor Core acceleration |
| **Optimizer** | `AdamW (lr={cfg['hyperparameters']['learning_rate']}, wd={cfg['hyperparameters']['weight_decay']})` | Stable fine-tuning of deep residual weights |

---

## 2. Model Architecture & Grayscale Adaptation

- **Backbone:** ResNet-50 (ImageNet-1K pretrained).
- **Grayscale Strategy:** Single-channel first convolution with channel weight averaging ($W_{{\\text{{gray}}}} = \\frac{1}{3} \\sum_{{c=1}}^3 W_c$). Preserves early edge/texture detector response without 3× VRAM replication.
- **Classification Head:** Global Average Pooling $\\to$ Dropout(0.3) $\\to$ Linear(2048, 1).
- **Parameters:**
  - Total: **{cfg['model_architecture']['parameters']['total_parameters']:,}**
  - Trainable: **{cfg['model_architecture']['parameters']['trainable_parameters']:,}**
  - Backbone: **{cfg['model_architecture']['parameters']['backbone_parameters']:,}**
  - Classifier Head: **{cfg['model_architecture']['parameters']['classifier_parameters']:,}**

---

## 3. Dataset Usage & Leakage Controls

- **Training:** 2,239 records (1,324 benign, 915 malignant). Gradient updates strictly limited to this partition.
- **Validation:** 542 records. Used exclusively for early stopping and model checkpoint selection.
- **Clean Test:** 644 records. Evaluated only when explicitly commanded; otherwise held back pristine for final comparative analysis.
- **Quarantined:** 143 records excluded due to cross-partition patient conflicts (31 patients).

---

## 4. Validation Performance (Model Selection)

- **Best Checkpoint Epoch:** {cfg['training_summary']['best_epoch']}
- **Best Validation Loss:** `{cfg['training_summary']['best_val_loss']:.4f}`
- **Validation Accuracy:** `{val_metrics.get('accuracy', 0.0) * 100:.2f}%`
- **Validation Sensitivity:** `{val_metrics.get('sensitivity', 0.0) * 100:.2f}%`
- **Validation Specificity:** `{val_metrics.get('specificity', 0.0) * 100:.2f}%`
- **Validation F1-Score:** `{val_metrics.get('f1_score', 0.0) * 100:.2f}%`
- **Validation ROC-AUC:** `{val_metrics.get('roc_auc') or 0.0:.4f}`

---

## 5. Comparative Baseline Context (Stage 14 CNN)

| Metric | Stage 14 Baseline CNN | Stage 15 ResNet-50 (Val) | Stage 15 ResNet-50 (Test) |
|---|---|---|---|
| **Parameters** | 421,601 | 23,510,081 | 23,510,081 |
| **Accuracy** | 59.47% | {val_metrics.get('accuracy', 0.0) * 100:.2f}% | {"N/A (Held Back)" if not test_metrics else f"{test_metrics['accuracy'] * 100:.2f}%"} |
| **Sensitivity** | 72.37% | {val_metrics.get('sensitivity', 0.0) * 100:.2f}% | {"N/A (Held Back)" if not test_metrics else f"{test_metrics['sensitivity'] * 100:.2f}%"} |
| **Specificity** | 50.90% | {val_metrics.get('specificity', 0.0) * 100:.2f}% | {"N/A (Held Back)" if not test_metrics else f"{test_metrics['specificity'] * 100:.2f}%"} |
| **F1-Score** | 58.77% | {val_metrics.get('f1_score', 0.0) * 100:.2f}% | {"N/A (Held Back)" if not test_metrics else f"{test_metrics['f1_score'] * 100:.2f}%"} |
| **ROC-AUC** | 0.6669 | {val_metrics.get('roc_auc') or 0.0:.4f} | {"N/A (Held Back)" if not test_metrics else f"{test_metrics.get('roc_auc') or 0.0:.4f}"} |

> [!NOTE]
> *Scientific Protocol Reminder:* Do not claim ResNet-50 outperforms the Stage 14 baseline until both models are evaluated and statistically compared under identical frozen evaluation conditions in Stage 17/18.
"""
    return md


def main():
    """CLI entrypoint for Stage 15 training pipeline."""
    parser = argparse.ArgumentParser(description="CBIS-DDSM Stage 15: ResNet-50 Transfer Learning")
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

    run_stage15_training(
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
