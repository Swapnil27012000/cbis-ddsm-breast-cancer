# CBIS-DDSM Stage 15.1 — ResNet-50 Checkpoint and Metrics Audit

**Audit Date:** 2026-10-21  
**Audit Scope:** Read-only inspection of completed Stage 15 ResNet-50 transfer learning run, checkpoint integrity, metrics divergence analysis, and split immutability.  
**Audit Status:** **PASS** (Zero test leakage, prior stages immutable, checkpoint provenance verified).

---

## 1. Executive Audit Summary

| Audit Item | Verified State | Audit Finding |
|---|---|---|
| **Checkpoint Selection Metric** | Validation Binary Cross-Entropy Loss (`val_loss`, minimization) | **PASS** |
| **Saved Best Checkpoint Epoch** | **Epoch 4** (`val_loss = 0.60488`) | **PASS** |
| **Checkpoint Provenance** | Confirmed from `stage15_resnet50_config.json`, `stage15_epoch_history.csv`, and `stage15_trainer.py` | **PASS** |
| **Clean Test Partition Isolation** | Zero test records evaluated (`status = HELD_BACK_FOR_FINAL_COMPARISON`) | **PASS** |
| **Stages 1–13 Manifest Integrity** | 2,239 Train, 542 Val, 644 Test, 143 Quarantined — Unmodified | **PASS** |
| **Stage 14 Baseline Integrity** | Baseline CNN config, epoch history, and test metrics — Unmodified | **PASS** |
| **Cross-Epoch Metrics Divergence** | Epoch 4 (best loss) vs Epoch 9 (best ROC-AUC / Acc) explained by cross-entropy calibration vs ranking invariance | **DOCUMENTED** |

---

## 2. Checkpoint Selection & Epoch Verification

### 2.1 Selection Policy in Code
In [`src/training/stage15_trainer.py:271`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/src/training/stage15_trainer.py#L271), the checkpoint-saving logic is explicitly defined as:
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_val_metrics = val_metrics
    best_epoch = epoch
    patience_counter = 0
    torch.save(..., best_checkpoint_path)
```
The checkpoint selection criterion is strictly **Validation BCE Loss (`val_loss`) minimization**.

### 2.2 Proof of Saved Epoch
From [`data/metadata/stage15/stage15_epoch_history.csv`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage15/stage15_epoch_history.csv), the validation loss trajectory across all 9 executed epochs is:
- Epoch 1: `val_loss = 0.70192` (Initial best)
- Epoch 2: `val_loss = 0.66373` (New best)
- Epoch 3: `val_loss = 0.63537` (New best)
- **Epoch 4: `val_loss = 0.60488` (Global minimum `val_loss` achieved)**
- Epoch 5: `val_loss = 0.71205` (> 0.60488, patience = 1)
- Epoch 6: `val_loss = 0.61075` (> 0.60488, patience = 2)
- Epoch 7: `val_loss = 0.79679` (> 0.60488, patience = 3, LR reduced to $5 \times 10^{-5}$)
- Epoch 8: `val_loss = 0.77781` (> 0.60488, patience = 4)
- Epoch 9: `val_loss = 0.77841` (> 0.60488, patience = 5 $\to$ **Early stopping triggered**)

**Conclusion:** The saved best checkpoint [`models/stage15_resnet50/resnet50_best.pt`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/models/stage15_resnet50/resnet50_best.pt) corresponds **strictly and unambiguously to Epoch 4**. No subsequent epoch surpassed the 0.60488 loss threshold.

---

## 3. Metric Comparison: Selected Checkpoint vs. Cross-Epoch Peaks

The table below contrasts the metrics of the **actual selected checkpoint (Epoch 4)** against the **independent maximum values observed across all 9 epochs**:

| Metric | Selected Checkpoint (Epoch 4) | Best Observed Across All Epochs | Peak Epoch |
|---|---|---|---|
| **Validation Loss** | **0.60488** | **0.60488** | Epoch 4 |
| **Validation Accuracy** | 75.28% | **78.97%** | Epoch 9 |
| **Validation Sensitivity (Recall)** | 72.61% | **86.09%** | Epoch 3 |
| **Validation Specificity** | 77.24% | **88.14%** | Epoch 9 |
| **Validation Precision** | 70.17% | **80.53%** | Epoch 9 |
| **Validation F1-Score** | 71.37% | **72.86%** | Epoch 9 |
| **Validation ROC-AUC** | 0.8326 | **0.8557** | Epoch 9 |
| **Confusion Matrix ($TN, FP, FN, TP$)** | $241, 71, 63, 167$ | $275, 37, 77, 153$ | Epoch 9 |

---

## 4. Technical Analysis: Why Epoch 9 Differs from Epoch 4

The discrepancy between Epoch 4 (best loss) and Epoch 9 (highest accuracy / ROC-AUC) is explained by four core machine learning principles:

1. **Cross-Entropy Penalization vs. Thresholding**:
   - Binary Cross-Entropy heavily penalizes overconfident incorrect predictions ($-\log(1 - \hat{p})$ as $\hat{p} \to 1$ for a negative case).
   - By Epoch 9, the training loss decreased drastically from 0.4844 down to 0.1233. The network pushed logits to larger absolute magnitudes.
   - For a small fraction of hard validation cases, the model made overconfident mistakes, heavily inflating the average validation loss (from 0.6049 to 0.7784) despite overall high classification accuracy.
2. **Ranking Invariance of ROC-AUC**:
   - ROC-AUC measures pairwise rank concordance across all potential thresholds. It is scale-invariant and completely unaffected by uncalibrated probability margins, as long as positive cases are scored above negative cases.
   - Thus, ROC-AUC improved from 0.8326 (Epoch 4) to 0.8557 (Epoch 9) because global relative separation improved, even though probability calibration deteriorated.
3. **Clinical Operating Point Trade-Off**:
   - In Epoch 4, the model maintained balanced clinical sensitivity (**72.61%**) and specificity (**77.24%**), missing only 63 malignant cases ($FN=63$).
   - In Epoch 9, the model shifted its decision threshold towards conservatism, boosting specificity to **88.14%** and precision to **80.53%** ($FP$ dropped from 71 to 37), but sensitivity deteriorated to **66.52%** ($FN$ increased from 63 to 77).
   - In breast cancer screening, missing 14 additional cancers ($FN=77$) to suppress false positives is clinically suboptimal. Epoch 4 represents a superior clinical balance.

---

## 5. Verification of Test Set Isolation & Prior Stage Immutability

### 5.1 Clean Test Set Isolation
- In [`data/metadata/stage15/stage15_resnet50_config.json`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage15/stage15_resnet50_config.json), `test_results.evaluated` is `false`, with status `"HELD_BACK_FOR_FINAL_COMPARISON"`.
- In [`data/metadata/stage15/stage15_test_metrics.csv`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage15/stage15_test_metrics.csv), the clean test evaluation is formally recorded as deferred.
- No test confusion matrix or ROC curve was created in `results/stage15_resnet50/`.
- **Finding:** Complete compliance with zero-leakage test preservation.

### 5.2 Immutability of Stages 1–14
- Stage 13 manifests verified:
  - `classification_train_manifest.csv`: 2,239 records
  - `classification_val_manifest.csv`: 542 records
  - `classification_test_manifest.csv`: 644 records
  - `quarantined_conflict_manifest.csv`: 143 records
- Stage 14 Baseline CNN artifacts verified:
  - `stage14_baseline_cnn_config.json`: intact (1,832 bytes)
  - `stage14_test_metrics.csv`: intact (Accuracy 59.47%, Sensitivity 72.37%, Specificity 50.90%, F1 58.77%, ROC-AUC 0.6669)
  - `models/stage14_baseline_cnn/baseline_cnn_best.pt`: intact

---

## 6. Recommendations for Final Cross-Model Comparison Policy

When comparing Baseline CNN (Stage 14), ResNet-50 (Stage 15), and subsequent architectures (Stage 16):

1. **Maintain Policy Consistency Across All Models**:
   - Stage 14 Baseline CNN selected its best checkpoint using **`val_loss`** (best epoch 7, val loss 0.7041).
   - Stage 15 ResNet-50 selected its best checkpoint using **`val_loss`** (best epoch 4, val loss 0.6049).
   - *Recommendation:* Keep **`val_loss`** as the primary selection criterion for model selection to preserve direct methodological comparability and well-calibrated probabilities.
2. **Dual Checkpoint Archiving (Recommended for Stage 16+)**:
   - For subsequent architectures, save both:
     - `<model>_best_loss.pt` (Monitors minimal validation BCE loss — calibrated operating point).
     - `<model>_best_auc.pt` (Monitors maximal validation ROC-AUC — discriminative ranking capacity).
   - This provides transparent visibility into both calibration and discriminative potential during final Stage 17/18 synthesis without introducing retrospective bias.
3. **Pristine Test Set Unblinding**:
   - Defer evaluation of the 644 clean test cases until all candidate architectures (Baseline CNN, ResNet-50, DenseNet, EfficientNet) have completed validation-based model selection.
