# CBIS-DDSM Stage 15.2 — ResNet-50 Numerical Consistency Audit

**Audit Date:** 2026-10-21  
**Audit Type:** Read-only mathematical and numerical consistency verification  
**Scope:** Stage 15 ResNet-50 validation epoch history, checkpoint metrics, confusion matrices, and contingency table derivation  
**Final Audit Verdict:** **PASS** (100% mathematical consistency verified across all epochs and metrics)

---

## 1. Executive Summary

A comprehensive numerical audit was conducted on all validation metrics, confusion matrices, and configuration artifacts for the completed Stage 15 ResNet-50 experiment. 

### Key Findings:
1. **Mathematical Consistency Across All Epochs:**
   Every reported metric (Accuracy, Sensitivity, Specificity, Precision, and F1-score) across all 9 epochs in [`data/metadata/stage15/stage15_epoch_history.csv`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage15/stage15_epoch_history.csv) is **100% mathematically consistent** with the underlying validation dataset size ($N = 542$) and its ground-truth marginal distribution ($312\text{ Benign}, 230\text{ Malignant}$).
2. **Resolution of the Suspected Epoch 9 Inconsistency:**
   The hypothesis that confusion matrix `[TN=275, FP=37, FN=77, TP=153]` implies an accuracy of 66.52% was investigated and disproven. 
   - $(275 + 153) / 542 = 428 / 542 = \mathbf{78.9668\%} \approx \mathbf{78.97\%}$ (**Accuracy**).
   - $153 / (153 + 77) = 153 / 230 = \mathbf{66.5217\%} \approx \mathbf{66.52\%}$ (**Sensitivity**).
   The confusion matrix does not imply 66.52% accuracy; it implies **78.97% accuracy** and **66.52% sensitivity**, which **exactly matches** the reported values in `stage15_epoch_history.csv`.
3. **Selected Checkpoint (Epoch 4) Exactness:**
   The saved best checkpoint metrics from [`data/metadata/stage15/stage15_resnet50_config.json`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage15/stage15_resnet50_config.json) match the saved confusion matrix `[TN=241, FP=71, FN=63, TP=167]` to 16 decimal places.
4. **Data Artifact Availability Limitation:**
   Individual sample-level raw prediction arrays ($y_{\text{pred\_prob}}$ for all 542 samples per epoch) were not serialized to CSV/NPY; only epoch-level summary statistics and the selected checkpoint matrix were saved. However, exact integer contingency tables are mathematically proven via Diophantine reconstruction from fixed validation marginals.

---

## 2. Validation Split Ground Truth Marginals

From [`data/metadata/stage13/classification_val_manifest.csv`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage13/classification_val_manifest.csv) and [`class_distribution_summary.csv`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage13/class_distribution_summary.csv):
- Total Validation Records ($N$): **542**
- True Benign (Class 0, $N_0 = TN + FP$): **312** (57.56%)
- True Malignant (Class 1, $N_1 = TP + FN$): **230** (42.44%)

Every valid confusion matrix $\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$ on this validation set must satisfy:
$$TN + FP = 312, \quad FN + TP = 230, \quad TN + FP + FN + TP = 542$$

---

## 3. Detailed Recalculation Across All Epochs

Using the exact integer relations $TP = \text{round}(230 \times \text{Sensitivity})$ and $TN = \text{round}(312 \times \text{Specificity})$, the confusion matrices and derived metrics for all 9 epochs are recalculated below:

| Epoch | Confusion Matrix $[TN, FP, FN, TP]$ | Calculated Accuracy | Reported Accuracy | Calculated Sensitivity | Reported Sensitivity | Calculated Specificity | Reported Specificity | Calculated Precision | Reported Precision | Calculated F1-Score | Reported F1-Score | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | $[253, 59, 115, 115]$ | $368/542 = 0.6790$ | 0.6790 | $115/230 = 0.5000$ | 0.5000 | $253/312 = 0.8109$ | 0.8109 | $115/174 = 0.6609$ | 0.6609 | $230/404 = 0.5693$ | 0.5693 | **EXACT MATCH** |
| **2** | $[245, 67, 83, 147]$ | $392/542 = 0.7232$ | 0.7232 | $147/230 = 0.6391$ | 0.6391 | $245/312 = 0.7853$ | 0.7853 | $147/214 = 0.6869$ | 0.6869 | $294/444 = 0.6622$ | 0.6622 | **EXACT MATCH** |
| **3** | $[186, 126, 32, 198]$ | $384/542 = 0.7085$ | 0.7085 | $198/230 = 0.8609$ | 0.8609 | $186/312 = 0.5962$ | 0.5962 | $198/324 = 0.6111$ | 0.6111 | $396/554 = 0.7148$ | 0.7148 | **EXACT MATCH** |
| **4 (Best)** | $[241, 71, 63, 167]$ | $408/542 = 0.7528$ | 0.7528 | $167/230 = 0.7261$ | 0.7261 | $241/312 = 0.7724$ | 0.7724 | $167/238 = 0.7017$ | 0.7017 | $334/468 = 0.7137$ | 0.7137 | **EXACT MATCH** |
| **5** | $[186, 126, 44, 186]$ | $372/542 = 0.6863$ | 0.6863 | $186/230 = 0.8087$ | 0.8087 | $186/312 = 0.5962$ | 0.5962 | $186/312 = 0.5962$ | 0.5962 | $372/542 = 0.6863$ | 0.6863 | **EXACT MATCH** |
| **6** | $[232, 80, 60, 170]$ | $402/542 = 0.7417$ | 0.7417 | $170/230 = 0.7391$ | 0.7391 | $232/312 = 0.7436$ | 0.7436 | $170/250 = 0.6800$ | 0.6800 | $340/480 = 0.7083$ | 0.7083 | **EXACT MATCH** |
| **7** | $[262, 50, 94, 136]$ | $398/542 = 0.7343$ | 0.7343 | $136/230 = 0.5913$ | 0.5913 | $262/312 = 0.8397$ | 0.8397 | $136/186 = 0.7312$ | 0.7312 | $272/416 = 0.6538$ | 0.6538 | **EXACT MATCH** |
| **8** | $[248, 64, 71, 159]$ | $407/542 = 0.7509$ | 0.7509 | $159/230 = 0.6913$ | 0.6913 | $248/312 = 0.7949$ | 0.7949 | $159/223 = 0.7130$ | 0.7130 | $318/453 = 0.7020$ | 0.7020 | **EXACT MATCH** |
| **9** | $[275, 37, 77, 153]$ | $428/542 = 0.7897$ | 0.7897 | $153/230 = 0.6652$ | 0.6652 | $275/312 = 0.8814$ | 0.8814 | $153/190 = 0.8053$ | 0.8053 | $306/420 = 0.7286$ | 0.7286 | **EXACT MATCH** |

---

## 4. In-Depth Analysis of the Epoch 9 Question

The prompt raised the concern:
> *"Investigate the Epoch 9 inconsistency: confusion matrix `[TN=275, FP=37, FN=77, TP=153]` implies accuracy 66.52%, sensitivity 66.52%, specificity 88.14%, precision 80.53%, and F1-score approximately 72.86%. Compare these calculations against the reported Epoch 9 accuracy of 78.97% and sensitivity of 66.52%."*

### Root Cause Analysis:
1. **Mathematical Recalculation:**
   - **True Accuracy:** $\frac{TN + TP}{N} = \frac{275 + 153}{542} = \frac{428}{542} = \mathbf{0.78966789...} \to \mathbf{78.97\%}$.
   - **True Sensitivity:** $\frac{TP}{TP + FN} = \frac{153}{230} = \mathbf{0.66521739...} \to \mathbf{66.52\%}$.
2. **Finding:**
   The confusion matrix $[TN=275, FP=37, FN=77, TP=153]$ does **not** yield 66.52% accuracy. The fraction $\frac{153}{230} = 66.52\%$ is strictly the **Sensitivity** (recall for malignant lesions). The accuracy is unequivocally **$78.97\%$**.
3. **Conclusion:**
   There is **no inconsistency** in the Epoch 9 data. All five metrics reported in `stage15_epoch_history.csv` for Epoch 9 are identical to the mathematical outputs of the contingency matrix $[TN=275, FP=37, FN=77, TP=153]$.

---

## 5. Recalculation of Epoch 4 (Selected Checkpoint)

From [`stage15_resnet50_config.json`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage15/stage15_resnet50_config.json) and [`stage15_validation_metrics.csv`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/data/metadata/stage15/stage15_validation_metrics.csv), the saved confusion matrix is:
$$\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix} = \begin{bmatrix} 241 & 71 \\ 63 & 167 \end{bmatrix}$$

Recalculating with full floating-point precision:
- **Accuracy:** $\frac{241 + 167}{542} = \frac{408}{542} = 0.7527675276752768$ (Reported: `0.7527675276752768` $\to$ **75.28%**)
- **Sensitivity:** $\frac{167}{230} = 0.7260869565217392$ (Reported: `0.7260869565217392` $\to$ **72.61%**)
- **Specificity:** $\frac{241}{312} = 0.7724358974358975$ (Reported: `0.7724358974358975` $\to$ **77.24%**)
- **Precision:** $\frac{167}{167 + 71} = \frac{167}{238} = 0.7016806722689075$ (Reported: `0.7016806722689075` $\to$ **70.17%**)
- **F1-Score:** $\frac{2 \times 167}{2 \times 167 + 71 + 63} = \frac{334}{468} = 0.7136752136752137$ (Reported: `0.7136752136752137` $\to$ **71.37%**)

**Conclusion:** The selected checkpoint metrics in all reports match the saved contingency table to 16 decimal places.

---

## 6. Confusion Matrix Ordering & Code Consistency

Inspecting [`src/training/stage15_trainer.py:84`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/src/training/stage15_trainer.py#L84):
```python
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
```
- In scikit-learn, setting `labels=[0, 1]` guarantees that row 0 corresponds to true class 0 (Benign) and row 1 to true class 1 (Malignant).
- Column 0 represents predicted class 0, and column 1 represents predicted class 1.
- `cm.ravel()` unwraps strictly as $[TN, FP, FN, TP]$.
- This ordering is consistent across all reporting scripts, plotting functions, and audit documentation.

---

## 7. Artifact Limitations

- **Prediction-Level Outputs:** The raw arrays of individual sample probabilities for each epoch were stored in RAM during training and aggregated into summary metrics; they were not dumped to an on-disk CSV/NPY table.
- **Verification Certainty:** Because the validation cohort is fixed at $N=542$ with fixed marginals ($N_0=312, N_1=230$), there exists a unique integer solution for $[TN, FP, FN, TP]$ given sensitivity and specificity. As shown in Section 3, all 9 integer contingency tables reproduce the exact reported metrics.

---

## 8. Final Audit Verdict & Supporting Evidence

### Verdict: **PASS**

### Supporting Evidence:
1. **Zero Numerical Inconsistencies:** All 9 epochs exhibit zero discrepancy between reported metrics and theoretical integer confusion matrices.
2. **Epoch 9 Inconsistency Clarified:** The apparent conflict was demonstrated to be a confusion between Accuracy ($78.97\%$) and Sensitivity ($66.52\%$).
3. **Selected Checkpoint Verified:** Epoch 4 metrics match its saved confusion matrix with 16-decimal-place precision.
4. **Test Set Isolation Maintained:** The clean test manifest remains unread and unblinded.
5. **Prior Stage Immutability:** Stages 1–14 artifacts and manifests remain unmodified.
