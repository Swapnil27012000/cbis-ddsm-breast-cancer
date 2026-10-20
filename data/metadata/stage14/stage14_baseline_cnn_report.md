# Stage 14 — Baseline CNN Model Implementation Report

**Project:** CBIS-DDSM Breast Cancer Detection and Diagnosis  
**Generated:** 2026-10-20 16:18:01  
**Compute Device:** `cuda`  
**Model:** `BaselineBreastCancerCNN` (421,601 parameters)  

## 1. Architecture & Design Rationale

To establish a modest scientific baseline prior to deep transfer learning (Stages 15+), a custom 4-stage convolutional neural network was implemented:
- **Feature Extractor:** 4 sequential blocks `Conv2d(3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)` progressing from 1 to 256 feature channels.
- **Global Pooling:** `AdaptiveAvgPool2d((1, 1))` collapses spatial dimensions to prevent fully-connected weight explosion.
- **Classifier Head:** `Linear(256, 128) -> BatchNorm1d -> ReLU -> Dropout(0.4) -> Linear(128, 1)`.
- **Loss Function:** `BCEWithLogitsLoss(pos_weight=1.447)` to account for the 59% benign / 41% malignant training distribution.
- **Memory Budget:** Designed to execute within 4 GB VRAM (RTX 2050), utilizing $\approx 1.1\text{ GB}$ peak memory during backpropagation.

## 2. Dataset Partitioning & Anti-Leakage Protocol

| Partition | Sample Count | Primary Views | Patient Overlap | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Training** | 2239 | 1891 | 0 patients | Optimization with horizontal flips |
| **Validation** | 542 | 465 | 0 patients | Early stopping & model selection |
| **Clean Test** | 644 | 591 | 0 patients | One-time benchmark evaluation |

> [!IMPORTANT]
> **Anti-Leakage Verification:** Zero patient overlap exists between active training, validation, and test manifests. The 31 patients with official cross-split conflicts remain strictly quarantined.

## 3. Test Set Classification Performance

Evaluation performed on the clean test manifest (**644 records**):

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **0.5947** (59.47%) | Overall diagnostic correctness |
| **Sensitivity (Recall)** | **0.7237** (72.37%) | True malignant detection rate |
| **Specificity** | **0.5090** (50.90%) | Correct benign exclusion rate |
| **Precision** | **0.4947** (49.47%) | Positive predictive value |
| **F1-Score** | **0.5877** | Harmonic mean of precision & sensitivity |
| **ROC-AUC** | **0.6669** | Area under the ROC curve |

### Confusion Matrix

| Actual \ Predicted | Predicted Benign (0) | Predicted Malignant (1) |
| :--- | :---: | :---: |
| **Actual Benign (0)** | **TN = 197** | **FP = 190** |
| **Actual Malignant (1)** | **FN = 71** | **TP = 186** |

## 4. Limitations & Scope Notice

1. **Record-Level Evaluation:** This evaluation reports case-abnormality record-level performance. True multi-view patient-level aggregation will be evaluated in Stage 17.
2. **Baseline Role:** This model serves as the scientific control to measure the exact value added by transfer learning (ResNet-50 in Stage 15) and advanced denoising/contrast preprocessing.