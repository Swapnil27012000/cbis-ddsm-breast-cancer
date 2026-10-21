# CBIS-DDSM Stage 16.1 — DenseNet-121 and EfficientNet-B0 Implementation Report

**Implementation Date:** 2026-10-21  
**Project:** MSc IT (Artificial Intelligence) — CBIS-DDSM Breast Cancer Classification  
**Scope:** Stage 16.1 Transfer Learning Architecture Implementation for DenseNet-121 and EfficientNet-B0 with Dual Checkpointing  
**Status:** **IMPLEMENTED & READY FOR SMOKE TESTING** (Zero test set exposure, prior stages immutable)

---

## 1. Executive Implementation Summary

Stage 16.1 has successfully implemented transfer learning architectures for **DenseNet-121** and **EfficientNet-B0** alongside a unified training and dual-checkpointing engine specifically tailored for the NVIDIA GeForce RTX 2050 (4 GB VRAM).

| Metric / Parameter | DenseNet-121 Implementation | EfficientNet-B0 Implementation |
|---|---|---|
| **Model Module** | [`src/models/stage16_densenet.py`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/src/models/stage16_densenet.py) | [`src/models/stage16_efficientnet.py`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/src/models/stage16_efficientnet.py) |
| **Total Parameters** | **~7,000,000** | **~4,000,000** |
| **Pretrained Weights** | `DenseNet121_Weights.IMAGENET1K_V1` | `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| **Input Adaptation** | `features.conv0` ($7 \times 7$, stride 2, pad 3, 1 ch) | `features[0][0]` ($3 \times 3$, stride 2, pad 1, 1 ch) |
| **Weight Averaging** | $W_{\text{gray}} = \text{mean}(W_{\text{rgb}}, \text{dim}=1)$ | $W_{\text{gray}} = \text{mean}(W_{\text{rgb}}, \text{dim}=1)$ |
| **Classifier Head** | Dropout(0.3) $\to$ Linear(1024, 1) | Dropout(0.2) $\to$ Linear(1280, 1) |
| **Estimated Peak VRAM**| ~1.8 GB (55% headroom on 4 GB) | ~1.2 GB (70% headroom on 4 GB) |
| **Primary Checkpoint** | `densenet121_best_loss.pt` (min `val_loss`) | `efficientnet_b0_best_loss.pt` (min `val_loss`) |
| **Secondary Checkpoint**| `densenet121_best_auc.pt` (max `val_roc_auc`)| `efficientnet_b0_best_auc.pt` (max `val_roc_auc`) |

---

## 2. Model Architecture & Grayscale Adaptation Details

### 2.1 Grayscale Channel Weight Averaging
Mammograms are natively single-channel (grayscale) 256×256 images. Rather than triplicating image data across 3 channels in VRAM, both architectures adapt their initial convolutional layer to `in_channels = 1` by averaging the pretrained RGB weights:
$$W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^3 W_{\text{rgb}}^{(c)} = W_{\text{rgb}}.\text{mean}(\text{dim}=1, \text{keepdim}=\text{True})$$

- **DenseNet-121:**
  - `model.features.conv0` is adapted to `Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)`.
  - Weight tensor shape: `(64, 1, 7, 7)`.
  - Mathematical equivalence to ImageNet filter response verified.
- **EfficientNet-B0:**
  - `model.features[0][0]` is adapted to `Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)`.
  - Weight tensor shape: `(32, 1, 3, 3)`.
  - Mathematical equivalence to ImageNet filter response verified.

### 2.2 Transparent Pretrained Weight Auditing
Both models expose `get_weight_metadata()` recording:
- Architecture name.
- Pretrained weights requested vs loaded status.
- Exact torchvision weight enum version string.
- Complete parameter breakdown (total, trainable, backbone, classifier).
- Load error diagnostics (random initialization is never silently substituted).

---

## 3. Shared Trainer & Dual Checkpoint Policy

Implemented in [`src/training/stage16_trainer.py`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/src/training/stage16_trainer.py):

### 3.1 Resource Optimization Controls
- **Batch Size:** `8` per GPU step.
- **Gradient Accumulation:** `2` steps (effective batch size = `16`).
- **Precision:** Automatic Mixed Precision (`torch.amp.autocast("cuda")` + `GradScaler`).
- **Optimizer:** AdamW (`lr=0.0001`, `weight_decay=0.0001`).
- **Scheduler:** `ReduceLROnPlateau(mode="min", factor=0.5, patience=2)`.
- **Loss Function:** `nn.BCEWithLogitsLoss(pos_weight=1.446995)` (derived strictly from training split: 1,324 benign vs 915 malignant).
- **VRAM Cleanup & OOM Recovery:** Automated pre-run purge (`clean_gpu_memory(0)`) and runtime OOM catching with actionable diagnostic recommendations.

### 3.2 Dual Checkpointing Specification
To resolve the metric divergence identified in Stage 15.2, Stage 16 trains with explicit dual checkpointing:
1. **Best Loss Checkpoint (`<model>_best_loss.pt`)**:
   - Monitored Metric: **Minimal Validation Loss (`val_loss`)**.
   - Preserves calibrated probabilities and balanced clinical sensitivity/specificity.
   - Saved with metadata `selection_metric: "val_loss"`.
   - Exported to: `stage16_<model>_validation_metrics_best_loss.csv`.
2. **Best AUC Checkpoint (`<model>_best_auc.pt`)**:
   - Monitored Metric: **Maximal Validation ROC-AUC (`val_roc_auc`)**.
   - Captures peak discriminative ranking capacity across all thresholds.
   - Saved with metadata `selection_metric: "val_roc_auc"`.
   - Exported to: `stage16_<model>_validation_metrics_best_auc.csv`.

*Integrity Rule:* The metrics for these two checkpoints are stored in distinct CSV and JSON fields so they are never conflated or confused.

---

## 4. Test Set Isolation & Prior Stage Immutability

- **Strict Test Set Blinded Status:**
  The clean test set (644 records) is **strictly held back and unevaluated** (`evaluate_test=False` by default). Test loader initialization is omitted during training and validation.
- **Stages 1–15 Immutability:**
  - Stage 13 manifests (`classification_train_manifest.csv`, `classification_val_manifest.csv`, `classification_test_manifest.csv`, `quarantined_conflict_manifest.csv`) are unchanged.
  - Stage 14 baseline CNN outputs (`stage14_baseline_cnn_config.json`, `baseline_cnn_best.pt`) are unchanged.
  - Stage 15 ResNet-50 outputs (`stage15_resnet50_config.json`, `resnet50_best.pt`, `stage15_1_audit_report.md`, `stage15_2_metric_consistency_audit.md`) are unchanged.

---

## 5. Artifact & Directory Layout

```
cbis-ddsm-breast-cancer/
├── src/
│   ├── models/
│   │   ├── stage16_densenet.py         [NEW]
│   │   ├── stage16_efficientnet.py     [NEW]
│   │   └── __init__.py                 [UPDATED: Exported Stage 16 models]
│   └── training/
│       └── stage16_trainer.py          [NEW: Unified trainer with dual checkpointing]
├── tests/
│   ├── test_stage16_readiness.py       [CREATED in 16.0: 8 tests passed]
│   └── test_stage16_models.py          [NEW: 11 comprehensive unit & integration tests]
├── data/metadata/stage16/
│   ├── stage16_0_readiness_audit.md    [CREATED]
│   ├── stage16_0_readiness_audit.txt   [CREATED]
│   ├── stage16_1_implementation_report.md [CREATED: This report]
│   └── stage16_1_implementation_report.txt [CREATED]
├── models/
│   ├── stage16_densenet121/            [Target checkpoint location]
│   └── stage16_efficientnet_b0/        [Target checkpoint location]
└── results/
    ├── stage16_densenet121/            [Target plots & curves location]
    └── stage16_efficientnet_b0/        [Target plots & curves location]
```

---

## 6. Targeted Test Suite (`tests/test_stage16_models.py`)

A comprehensive test suite with 11 targeted unit and integration tests has been implemented:

1. `test_densenet121_architecture_and_output_shape` — Validates DenseNet-121 single-channel conv0, output shape `(B, 1)`, and ~7.0M parameters.
2. `test_efficientnet_b0_architecture_and_output_shape` — Validates EfficientNet-B0 single-channel conv, output shape `(B, 1)`, and ~4.0M parameters.
3. `test_densenet_pretrained_channel_averaging_exactness` — Verifies exact equality: $W_{\text{gray}} = \text{mean}(W_{\text{rgb}})$ for DenseNet-121.
4. `test_efficientnet_pretrained_channel_averaging_exactness` — Verifies exact equality: $W_{\text{gray}} = \text{mean}(W_{\text{rgb}})$ for EfficientNet-B0.
5. `test_densenet_forward_and_backward_pass` — Verifies forward/backward pass with `BCEWithLogitsLoss` and finite gradients on DenseNet-121.
6. `test_efficientnet_forward_and_backward_pass` — Verifies forward/backward pass with `BCEWithLogitsLoss` and finite gradients on EfficientNet-B0.
7. `test_metric_computations_exactness` — Verifies clinical metric accuracy against known analytical ground truth ($AUC = 0.78125$).
8. `test_dual_checkpoint_save_and_safe_load` — Verifies saving and reloading both `_best_loss.pt` and `_best_auc.pt` with `safe_torch_load`.
9. `test_smoke_test_mode_isolation_densenet` — Verifies smoke-test mode runs on DenseNet-121 without touching the clean test partition.
10. `test_smoke_test_mode_isolation_efficientnet` — Verifies smoke-test mode runs on EfficientNet-B0 without touching the clean test partition.
11. `test_stage16_manifest_and_patient_leakage_integrity` — Verifies Stage 13 split counts (2239/542/644) and 0 patient overlap.
12. `test_prior_stages_immutability` — Verifies Stages 1–15 artifacts and checkpoints remain intact.

---

## 7. Recommended Terminal Execution Commands

### Step 1: Run Stage 16 Targeted Test Suite
```powershell
docker compose run --rm cbis-ddsm-ai pytest tests/test_stage16_models.py -v
```

### Step 2: Run DenseNet-121 Smoke Test (Excludes Test Set)
```powershell
docker compose run --rm cbis-ddsm-ai python -m src.training.stage16_trainer --model densenet121 --smoke-test
```

### Step 3: Run EfficientNet-B0 Smoke Test (Excludes Test Set)
```powershell
docker compose run --rm cbis-ddsm-ai python -m src.training.stage16_trainer --model efficientnet_b0 --smoke-test
```

### Step 4: Full Training (Validation Model Selection Only, When Ready)
- **DenseNet-121:**
  ```powershell
  docker compose run --rm cbis-ddsm-ai python -m src.training.stage16_trainer --model densenet121 --epochs 15 --batch-size 8 --accum-steps 2
  ```
- **EfficientNet-B0:**
  ```powershell
  docker compose run --rm cbis-ddsm-ai python -m src.training.stage16_trainer --model efficientnet_b0 --epochs 15 --batch-size 8 --accum-steps 2
  ```
