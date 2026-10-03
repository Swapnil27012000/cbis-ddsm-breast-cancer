"""Training execution pipeline with full orchestration."""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from .validate import validate_epoch
from .loss import get_loss_function
from .scheduler import get_lr_scheduler
from .checkpoint import CheckpointManager
from src.models.model import build_model
from src.datasets.cbis_ddsm_dataset import CBISDDSMDataset
from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc_curve import plot_roc_curve
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.gpu.device import get_device, get_device_info

logger = setup_logger("Trainer")

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None
) -> float:
    """Run one training epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    total = 0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += targets.size(0)

    return total_loss / max(total, 1)

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device
) -> float:
    """Full multi-epoch training loop with best-model tracking."""
    opt = torch.optim.AdamW(model.parameters(), lr=config.get("training", {}).get("lr", 1e-4), weight_decay=1e-5)
    criterion = get_loss_function(config.get("training", {}).get("loss", "focal"))
    epochs = config.get("training", {}).get("epochs", 30)
    scheduler = get_lr_scheduler(opt, epochs=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_mgr = CheckpointManager(config.get("paths", {}).get("models_dir", "models") + "/checkpoints")
    res_dir = config.get("paths", {}).get("results_dir", "results") + "/classification"
    os.makedirs(res_dir, exist_ok=True)

    best_val_acc = 0.0
    logger.info(f"Starting model training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, criterion, device, scaler)
        v_loss, v_acc, v_preds, v_targets = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            ckpt_mgr.save(model, opt, epoch, {"val_acc": v_acc, "val_loss": v_loss}, "best_model.pt")

            # Update metrics & plots
            metrics = compute_classification_metrics(v_targets, v_preds)
            pd.DataFrame([metrics]).to_csv(os.path.join(res_dir, "metrics.csv"), index=False)
            plot_confusion_matrix(v_targets, v_preds, save_path=os.path.join(res_dir, "confusion_matrix.png"))

    logger.info(f"Training finished! Best Validation Accuracy: {best_val_acc:.4f}")
    return best_val_acc

def main():
    """Main training entrypoint."""
    config = load_config("config/config.yaml") if os.path.exists("config/config.yaml") else {}
    device = get_device(config.get("gpu", {}).get("device_id", 0))
    logger.info(f"Using device: {device} | Info: {get_device_info()}")

    metadata_path = config.get("dataset", {}).get("master_metadata", "data/metadata/CBIS_DDSM_master_metadata.csv")
    if not os.path.exists(metadata_path):
        logger.error(f"Master metadata not found at {metadata_path}. Please run metadata builder first: python -m src.data.metadata_builder")
        return

    df = pd.read_csv(metadata_path)
    valid_df = df[df["file_exists"] == True].dropna(subset=["label"]).copy()
    if len(valid_df) == 0:
        logger.error("No valid image files found in master metadata.")
        return

    # Train / Val split
    split_col = None
    for candidate in ["split", "dataset_split"]:
        if candidate in valid_df.columns:
            split_col = candidate
            break

    if split_col is not None:
        split_series = valid_df[split_col].astype(str).str.strip().str.lower()
        train_df = valid_df[split_series == "train"].copy()
        val_df = valid_df[split_series.isin(["val", "validation", "test"])].copy()
    else:
        train_df = valid_df.copy()
        val_df = pd.DataFrame()

    if len(train_df) == 0 or len(val_df) == 0:
        logger.info("Split column missing or incomplete; creating 80/20 train/validation partition...")
        val_df = valid_df.sample(frac=0.2, random_state=42)
        train_df = valid_df.drop(val_df.index)

    logger.info(f"Dataset ready: {len(train_df)} train samples, {len(val_df)} validation samples.")

    train_ds = CBISDDSMDataset(train_df, image_col="image_path", label_col="label", transform=get_train_transforms())
    val_ds = CBISDDSMDataset(val_df, image_col="image_path", label_col="label", transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=config.get("training", {}).get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get("training", {}).get("batch_size", 16), shuffle=False)

    model_cfg = config.get("model", {})
    model = build_model(
        arch=model_cfg.get("architecture", "resnet50"),
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3)
    ).to(device)

    train_model(model, train_loader, val_loader, config, device)

if __name__ == "__main__":
    main()
