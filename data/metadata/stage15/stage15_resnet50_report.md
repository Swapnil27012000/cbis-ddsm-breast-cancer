# CBIS-DDSM Stage 15 — ResNet-50 Transfer Learning Report

**Generated:** 2026-10-20 20:29:14  
**Compute Device:** `cuda`  
**Architecture:** `Stage15ResNet50` (23,503,809 parameters)  
**Pretrained Weights:** `ResNet50_Weights.IMAGENET1K_V2` (Loaded: `True`)  
**Clean Test Status:** `HELD BACK FOR FINAL COMPARISON`  

---

## 1. Hardware & Memory Management (RTX 2050 4 GB VRAM)

| Parameter | Value | Rationale |
|---|---|---|
| **Device** | `cuda` | NVIDIA RTX 2050 (4 GB VRAM) |
| **Batch Size** | `8` | Conservative allocation to prevent CUDA OOM |
| **Gradient Accumulation** | `2` steps | Yields effective batch size of `16` |
| **Image Resolution** | `256 × 256` | Matches Stage 5/14 baseline resolution |
| **Precision** | `AMP (torch.amp.autocast)` | ~50% VRAM memory savings and Tensor Core acceleration |
| **Optimizer** | `AdamW (lr=0.0001, wd=0.0001)` | Stable fine-tuning of deep residual weights |

---

## 2. Model Architecture & Grayscale Adaptation

- **Backbone:** ResNet-50 (ImageNet-1K pretrained).
- **Grayscale Strategy:** Single-channel first convolution with channel weight averaging ($W_{\text{gray}} = \frac13 \sum_{c=1}^3 W_c$). Preserves early edge/texture detector response without 3× VRAM replication.
- **Classification Head:** Global Average Pooling $\to$ Dropout(0.3) $\to$ Linear(2048, 1).
- **Parameters:**
  - Total: **23,503,809**
  - Trainable: **23,503,809**
  - Backbone: **23,501,760**
  - Classifier Head: **2,049**

---

## 3. Dataset Usage & Leakage Controls

- **Training:** 2,239 records (1,324 benign, 915 malignant). Gradient updates strictly limited to this partition.
- **Validation:** 542 records. Used exclusively for early stopping and model checkpoint selection.
- **Clean Test:** 644 records. Evaluated only when explicitly commanded; otherwise held back pristine for final comparative analysis.
- **Quarantined:** 143 records excluded due to cross-partition patient conflicts (31 patients).

---

## 4. Validation Performance (Model Selection)

- **Best Checkpoint Epoch:** 4
- **Best Validation Loss:** `0.6049`
- **Validation Accuracy:** `75.28%`
- **Validation Sensitivity:** `72.61%`
- **Validation Specificity:** `77.24%`
- **Validation F1-Score:** `71.37%`
- **Validation ROC-AUC:** `0.8326`

---

## 5. Comparative Baseline Context (Stage 14 CNN)

| Metric | Stage 14 Baseline CNN | Stage 15 ResNet-50 (Val) | Stage 15 ResNet-50 (Test) |
|---|---|---|---|
| **Parameters** | 421,601 | 23,510,081 | 23,510,081 |
| **Accuracy** | 59.47% | 75.28% | N/A (Held Back) |
| **Sensitivity** | 72.37% | 72.61% | N/A (Held Back) |
| **Specificity** | 50.90% | 77.24% | N/A (Held Back) |
| **F1-Score** | 58.77% | 71.37% | N/A (Held Back) |
| **ROC-AUC** | 0.6669 | 0.8326 | N/A (Held Back) |

> [!NOTE]
> *Scientific Protocol Reminder:* Do not claim ResNet-50 outperforms the Stage 14 baseline until both models are evaluated and statistically compared under identical frozen evaluation conditions in Stage 17/18.
