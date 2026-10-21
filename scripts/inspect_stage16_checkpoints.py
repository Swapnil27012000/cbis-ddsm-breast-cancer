"""Read-only checkpoint inspection utility for Stage 16.2.

Safely loads checkpoint dictionaries for DenseNet-121 and EfficientNet-B0 to
inspect stored metadata, selected epoch, metrics, and distinguish full-training
runs from smoke-test runs.
"""

import os
import sys
import torch

CHECKPOINTS = {
    "densenet121_best_loss": "models/stage16_densenet121/densenet121_best_loss.pt",
    "densenet121_best_auc": "models/stage16_densenet121/densenet121_best_auc.pt",
    "efficientnet_b0_best_loss": "models/stage16_efficientnet_b0/efficientnet_b0_best_loss.pt",
    "efficientnet_b0_best_auc": "models/stage16_efficientnet_b0/efficientnet_b0_best_auc.pt",
}


def inspect_checkpoint(name: str, path: str):
    print("=" * 80)
    print(f"CHECKPOINT: {name}")
    print(f"PATH:       {path}")
    if not os.path.exists(path):
        print(f"STATUS:     MISSING (File not found)")
        return

    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"FILE SIZE:  {size_bytes:,} bytes ({size_mb:.2f} MB)")

    try:
        # Load with weights_only=False since dict contains int, dict, string metadata
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"LOAD ERROR: {e}")
        return

    if not isinstance(ckpt, dict):
        print(f"TYPE:       Raw tensor / state_dict (not a metadata dictionary)")
        return

    print("KEYS:      ", list(ckpt.keys()))
    epoch = ckpt.get("epoch")
    metric = ckpt.get("selection_metric")
    val_loss = ckpt.get("val_loss")
    val_roc_auc = ckpt.get("val_roc_auc")
    metrics = ckpt.get("val_metrics", {})
    cm = metrics.get("confusion_matrix")
    total_val = None
    if cm:
        total_val = sum(sum(row) for row in cm)

    print(f"EPOCH:      {epoch}")
    print(f"SELECTION:  {metric}")
    print(f"VAL LOSS:   {val_loss}")
    print(f"VAL AUC:    {val_roc_auc}")
    print(f"VAL SAMPLES:{total_val}")
    print(f"CM:         {cm}")

    # Determine provenance
    if total_val == 16:
        print("--> RUN ORIGIN: [SMOKE TEST OVERWRITE DETECTED] (16 validation samples)")
    elif total_val == 542:
        print("--> RUN ORIGIN: [FULL TRAINING VERIFIED] (542 validation samples)")
    else:
        print(f"--> RUN ORIGIN: [UNKNOWN SAMPLE COUNT: {total_val}]")


def main():
    print("STAGE 16.2 CHECKPOINT INSPECTION REPORT")
    print("Executing in read-only mode...")
    for name, path in CHECKPOINTS.items():
        inspect_checkpoint(name, path)
    print("=" * 80)


if __name__ == "__main__":
    main()
