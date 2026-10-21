"""Forensic inference audit for Stage 17.3 Baseline CNN discrepancy.

Compares Baseline CNN test inference across:
1. Batch size 32 with CUDA AMP autocast (Historical Stage 14 configuration).
2. Batch size 16 in standard FP32 (Stage 17.1 configuration).
3. Focuses specifically on borderline predictions near probability 0.50 (e.g. P_00457_LEFT_MLO_1).
"""

import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.datasets.stage14_dataloader import CBISDDSMClassificationDataset, load_stage13_manifests
from src.models.baseline_cnn import get_baseline_cnn


def evaluate_baseline_with_config(
    model: nn.Module,
    test_ds: CBISDDSMClassificationDataset,
    batch_size: int,
    use_amp: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()

    all_logits = []
    all_probs = []
    all_targets = []

    use_cuda_amp = use_amp and (device.type == "cuda")

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            if use_cuda_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits)
            all_logits.extend(logits.squeeze(1).cpu().float().numpy().tolist())
            all_probs.extend(probs.squeeze(1).cpu().float().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    return np.array(all_targets, dtype=int), np.array(all_logits, dtype=float), np.array(all_probs, dtype=float)


def main():
    print("=" * 80)
    print("STAGE 17.3 FORENSIC INFERENCE AUDIT: BASELINE CNN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    ckpt_path = os.path.join(_PROJECT_ROOT, "models/stage14_baseline_cnn/baseline_cnn_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Missing checkpoint at {ckpt_path}")
        return

    # Load Stage 13 test manifest
    manifest_dir = os.path.join(_PROJECT_ROOT, "data/metadata/stage13")
    _, _, test_df = load_stage13_manifests(manifest_dir)
    print(f"Loaded test manifest: {len(test_df)} samples")

    test_ds = CBISDDSMClassificationDataset(test_df, split_name="test", image_size=(256, 256), augment=False)

    # Instantiate model and load checkpoint
    model = get_baseline_cnn(in_channels=1, num_classes=1, dropout_rate=0.4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Run Config A: Historical Stage 14 (Batch size 32, AMP if CUDA)
    print("\n[Configuration A] Historical Stage 14 Settings: batch_size=32, use_amp=True")
    y_true_a, logits_a, probs_a = evaluate_baseline_with_config(model, test_ds, batch_size=32, use_amp=True, device=device)
    preds_a = (probs_a >= 0.50).astype(int)
    cm_a = confusion_matrix(y_true_a, preds_a, labels=[0, 1])
    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
    print(f"Result A: TN={tn_a}, FP={fp_a}, FN={fn_a}, TP={tp_a} | Accuracy: {(tp_a+tn_a)/len(y_true_a):.4%}")

    # 2. Run Config B: Stage 17.1 Settings (Batch size 16, FP32 no AMP)
    print("\n[Configuration B] Stage 17.1 Settings: batch_size=16, use_amp=False (FP32)")
    y_true_b, logits_b, probs_b = evaluate_baseline_with_config(model, test_ds, batch_size=16, use_amp=False, device=device)
    preds_b = (probs_b >= 0.50).astype(int)
    cm_b = confusion_matrix(y_true_b, preds_b, labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    print(f"Result B: TN={tn_b}, FP={fp_b}, FN={fn_b}, TP={tp_b} | Accuracy: {(tp_b+tn_b)/len(y_true_b):.4%}")

    # 3. Compare predictions case-by-case
    diff_mask = (preds_a != preds_b)
    diff_indices = np.where(diff_mask)[0]
    print(f"\nNumber of differing predictions between Config A and Config B: {len(diff_indices)}")

    for idx in diff_indices:
        row = test_df.iloc[idx]
        case_key = row["case_key"]
        pid = row["patient_id"]
        true_lbl = row["binary_label"]
        print(f"Discrepant Case: {case_key} ({pid}) | True Label: {true_lbl}")
        print(f"  Config A (Stage 14): Logit={logits_a[idx]:+.7f}, Prob={probs_a[idx]:.7f}, Pred={preds_a[idx]}")
        print(f"  Config B (Stage 17): Logit={logits_b[idx]:+.7f}, Prob={probs_b[idx]:.7f}, Pred={preds_b[idx]}")

    # Inspect P_00457_LEFT_MLO_1 specifically
    target_case = "P_00457_mass_LEFT_MLO_1"
    target_match = test_df[test_df["case_key"] == target_case]
    if len(target_match) > 0:
        idx_t = target_match.index[0]
        print(f"\nInspecting Target Case: {target_case} (Index {idx_t})")
        print(f"  Config A: Logit={logits_a[idx_t]:+.7f}, Prob={probs_a[idx_t]:.7f}, Pred={preds_a[idx_t]}")
        print(f"  Config B: Logit={logits_b[idx_t]:+.7f}, Prob={probs_b[idx_t]:.7f}, Pred={preds_b[idx_t]}")


if __name__ == "__main__":
    main()
