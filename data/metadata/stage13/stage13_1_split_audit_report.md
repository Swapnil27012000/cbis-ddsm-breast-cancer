# Stage 13.1 — Independent Dataset Split Audit Report

**Date of Audit:** 2026-10-20  
**Project:** CBIS-DDSM Breast Cancer Classification (MSc IT AI Final-Year Research)  
**Audit Scope:** Independent, read-only audit of Stage 13 classification manifests, patient-level splits, multi-abnormality case auditing, and official train/test cross-partition patient conflicts.  
**Audit Status:** COMPLETE  
**Audit Conclusion:** **REVIEW REQUIRED — POLICY DECISION NEEDED** (Active manifests are technically sound and leakage-free; formal sign-off required on official benchmark test policy).

---

## 1. Executive Summary & Verification Matrix

An exhaustive, item-by-item verification was performed on all artifacts in `data/metadata/stage13/`, the 4 raw case description CSVs, Stage 5 baseline metadata, and Stage 2 reference mapping.

| Check Item | Target Requirement | Audit Finding | Status |
| :--- | :--- | :--- | :---: |
| **Source Record Reconciliation** | Exactly 3,568 records | Exactly 3,568 records (Train: 2,239, Val: 542, Test: 644, Quarantined: 143) | **PASS** |
| **Stage 5 Manifest Alignment** | 100% mapping to Stage 5 baseline PNGs | 3,568 of 3,568 records mapped to verified baseline full mammograms | **PASS** |
| **Image Role & Geometry** | Full/original mammograms only | 100% full mammograms. Zero cropped ROIs, zero lesion masks, zero synthetic noisy/denoised files | **PASS** |
| **Disk Existence & Readability** | All images readable on disk | 3,568 of 3,568 images verified present and readable with native spatial dimensions | **PASS** |
| **Pathology Label Mapping** | BENIGN/WITHOUT_CALLBACK=0, MALIGNANT=1 | 2,111 Benign (Class 0), 1,457 Malignant (Class 1), 0 unmapped or ambiguous | **PASS** |
| **Invalid / Missing Labels** | 0 unknown or missing labels | 0 unknown, missing, or corrupted labels | **PASS** |
| **Unique Patients Accounted** | Exactly 1,566 unique patients | Exactly 1,566 patients (Train: 973, Val: 244, Test: 318, Quarantined: 31) | **PASS** |
| **Patient-Level Zero Leakage** | 0 patient overlap across active splits | **0 patients overlap** between Train, Val, and Test ($\text{Train}\cap\text{Val}=\emptyset, \text{Train}\cap\text{Test}=\emptyset, \text{Val}\cap\text{Test}=\emptyset$) | **PASS** |
| **Official Conflict Isolation** | 31 cross-split patients quarantined | 31 patients (143 records: 83 train + 60 test) quarantined in `cross_partition_patient_conflicts.csv` | **PASS** |
| **Multi-Abnormality Handling** | Canonical views + secondary records | 3,032 primary canonical views (`is_primary_view_record=True`), 536 secondary abnormality records | **PASS** |
| **Deterministic Reproducibility** | Seed 42 stratified split | Deterministic patient assignment; stratified benign/malignant balance preserved | **PASS** |
| **Prior Stages Immutability** | Stages 1–12 outputs unmodified | All raw CSVs and Stages 1–12 outputs remain 100% intact and read-only | **PASS** |
| **Official Test Set Policy** | 644 clean test vs 704 raw test | 60 official test records quarantined with train counterparts; requires supervisor/project sign-off | **REVIEW** |

---

## 2. Full Record & Patient Reconciliation

### A. Case Record Balance Sheet (Exact Single-Digit Reconciliation)

$$\begin{aligned}
\text{Total Source Records} &= 3,568 \\
&= \text{Mass Train (1,318)} + \text{Mass Test (378)} + \text{Calc Train (1,546)} + \text{Calc Test (326)} \\
&= \text{Official Train Pool (2,864)} + \text{Official Test Pool (704)}
\end{aligned}$$

How these 3,568 records are partitioned in Stage 13:
- **Active Training Partition (`classification_train_manifest.csv`):** **2,239 records** (62.75%)
- **Active Validation Partition (`classification_val_manifest.csv`):** **542 records** (15.19%)
- **Active Test Partition (`classification_test_manifest.csv`):** **644 records** (18.05%)
- **Quarantined Conflict Set (`cross_partition_patient_conflicts.csv`):** **143 records** (4.01%)
- **Unassigned / Dropped Records:** **0 records** (0.00%)

$$\mathbf{2,239 + 542 + 644 + 143 = 3,568 \quad (\text{Exact Match})}$$

### B. Reconciliation of the 644 Test Records vs. Expected 704 Records

The official CBIS-DDSM test set defines **704 records**. The Stage 13 active test manifest contains **644 records**.
- **Difference:** $704 - 644 = \mathbf{60\text{ records}}$.
- **Cause:** These exact 60 records belong to the **31 patients who have conflicting records in both official train and official test sets**.
- **Action Taken:** To protect the test benchmark from patient-level data leakage, these 60 test records were quarantined alongside their 83 training counterparts into `cross_partition_patient_conflicts.csv`.
- **Finding:** No test records were lost, corrupted, or dropped due to unreadability. The difference is 100% accounted for by the cross-partition leakage quarantine policy ($704 - 60 = 644$).

### C. Patient Balance Sheet

$$\begin{aligned}
\text{Total Unique Patients} &= 1,566 \\
&= \text{Active Train Patients (973)} \\
&\quad + \text{Active Validation Patients (244)} \\
&\quad + \text{Active Test Patients (318)} \\
&\quad + \text{Quarantined Conflict Patients (31)} \\
&= \mathbf{1,566 \quad (\text{Exact Match})}
\end{aligned}$$

### D. Mammographic View Balance Sheet (Multi-Abnormality Analysis)

A mammogram is defined by `patient_id` + `breast_side` + `image_view` (e.g. `P_00038_RIGHT_CC`).
- **Total Unique Mammographic Views:** **3,032 views**
- **Single-Abnormality Views:** 2,586 views (1 abnormality record per view)
- **Multi-Abnormality Views:** 446 views (representing multiple distinct abnormality records sharing the same physical baseline image)
- **Secondary Abnormality Records:** $3,568 - 3,032 = \mathbf{536\text{ records}}$
- **Audit Verification:** All 3,032 primary canonical views have `is_primary_view_record = True`. All 536 secondary records have `is_primary_view_record = False`. Both view-level deduplicated modeling and lesion-level multi-abnormality analysis can be conducted without ambiguity.

---

## 3. Investigation of the 31 Conflicting Patients

### A. Root Cause of the Conflict in Official CBIS-DDSM
The CBIS-DDSM curators split the mass cohort (`mass_case_description_train_set.csv` and `mass_case_description_test_set.csv`) and the calcification cohort (`calc_case_description_train_set.csv` and `calc_case_description_test_set.csv`) **independently**.
Patients who underwent mammography for both mass lesions and calcifications across multiple visits were not cross-referenced by the original curators. Consequently, 31 patients were assigned to the `train` partition in one CSV and the `test` partition in another.

### B. Summary of Affected Patients & Records
- **Total Conflicting Patients:** **31 patients**
- **Total Quarantined Records:** **143 records**
  - **Originating from Official Train CSVs:** **83 records**
  - **Originating from Official Test CSVs:** **60 records**
  - **Unique Mammographic Views:** **85 views**
  - **Pathology Breakdown:** 88 Benign (61.54%), 55 Malignant (38.46%)

### C. Detailed Roster of All 31 Conflicting Patients

| # | Patient ID | Total Rec | Train Rec | Test Rec | Train Abnormality | Test Abnormality | Train Pathology | Test Pathology | Discordant Pathology? |
| :-: | :--- | :-: | :-: | :-: | :--- | :--- | :--- | :--- | :-: |
| 1 | `P_00016` | 4 | 2 | 2 | Calcification | Mass | MALIGNANT | MALIGNANT | No |
| 2 | `P_00041` | 4 | 2 | 2 | Mass | Calcification | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | No |
| 3 | `P_00159` | 3 | 2 | 1 | Calcification | Mass | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | No |
| 4 | `P_00394` | 3 | 2 | 1 | Calcification | Mass | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | No |
| 5 | `P_00495` | 6 | 2 | 4 | Mass | Calcification | BENIGN | BENIGN_WITHOUT_CALLBACK | No (Both Benign) |
| 6 | `P_00510` | 14 | 12 | 2 | Calcification | Mass | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | No |
| 7 | `P_00591` | 5 | 4 | 1 | Calcification | Mass | MALIGNANT | MALIGNANT | No |
| 8 | `P_00601` | 4 | 2 | 2 | Calcification | Mass | MALIGNANT | BENIGN | **YES (Discrepant)** |
| 9 | `P_00639` | 3 | 2 | 1 | Calcification | Mass | MALIGNANT | BENIGN | **YES (Discrepant)** |
| 10 | `P_00652` | 3 | 2 | 1 | Calcification | Mass | MALIGNANT | BENIGN | **YES (Discrepant)** |
| 11 | `P_00678` | 3 | 2 | 1 | Mass | Calcification | MALIGNANT | MALIGNANT | No |
| 12 | `P_00723` | 4 | 1 | 3 | Mass | Calcification | BENIGN | BENIGN | No |
| 13 | `P_00753` | 4 | 2 | 2 | Mass | Calcification | MALIGNANT | BENIGN | **YES (Discrepant)** |
| 14 | `P_00882` | 4 | 2 | 2 | Calcification | Mass | BENIGN | BENIGN | No |
| 15 | `P_00906` | 5 | 2 | 3 | Mass | Calcification | MALIGNANT | MALIGNANT | No |
| 16 | `P_00924` | 9 | 8 | 1 | Calcification | Mass | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | No |
| 17 | `P_00969` | 8 | 4 | 4 | Calcification | Mass | MALIGNANT | BENIGN, MALIGNANT | Mixed |
| 18 | `P_01108` | 4 | 2 | 2 | Calcification | Mass | MALIGNANT | BENIGN | **YES (Discrepant)** |
| 19 | `P_01152` | 6 | 4 | 2 | Mass | Calcification | BENIGN | BENIGN | No |
| 20 | `P_01179` | 3 | 2 | 1 | Mass | Calcification | MALIGNANT | MALIGNANT | No |
| 21 | `P_01237` | 3 | 1 | 2 | Mass | Calcification | BENIGN | BENIGN | No |
| 22 | `P_01307` | 4 | 2 | 2 | Calcification | Mass | MALIGNANT | BENIGN | **YES (Discrepant)** |
| 23 | `P_01365` | 3 | 2 | 1 | Calcification | Mass | MALIGNANT | BENIGN_WITHOUT_CALLBACK | **YES (Discrepant)** |
| 24 | `P_01467` | 3 | 2 | 1 | Calcification | Mass | MALIGNANT | MALIGNANT | No |
| 25 | `P_01566` | 7 | 2 | 5 | Calcification | Mass | BENIGN | BENIGN | No |
| 26 | `P_01605` | 3 | 2 | 1 | Calcification | Mass | MALIGNANT | BENIGN | **YES (Discrepant)** |
| 27 | `P_01621` | 4 | 2 | 2 | Mass | Calcification | BENIGN | BENIGN | No |
| 28 | `P_01635` | 6 | 4 | 2 | Mass | Calcification | BENIGN_WITHOUT_CALLBACK | MALIGNANT | **YES (Discrepant)** |
| 29 | `P_01741` | 4 | 2 | 2 | Calcification | Mass | BENIGN | MALIGNANT | **YES (Discrepant)** |
| 30 | `P_01803` | 4 | 2 | 2 | Mass | Calcification | MALIGNANT | MALIGNANT | No |
| 31 | `P_01883` | 3 | 1 | 2 | Mass | Calcification | BENIGN | MALIGNANT | **YES (Discrepant)** |

### D. Critical Pathology Discordance Finding
In **11 of the 31 patients** (e.g. `P_00601`, `P_00639`, `P_01635`, `P_01741`), the mass abnormality and the calcification abnormality carry **opposing pathology labels** (one is Malignant, while the other is Benign).
- If a model trained on `P_00601`'s calcification (Malignant) and was tested on `P_00601`'s mass (Benign), the model would not only leak patient identity, but would face contradictory ground-truth targets for the same patient anatomy.
- **Audit Verdict:** Quarantining these 31 patients was scientifically necessary. Allowing them into training/testing without isolation would introduce both patient identity leakage and diagnostic label ambiguity.

---

## 4. Patient-Level Separation & Zero Leakage Verification

An independent set intersection was computed across all assigned partitions:

$$\begin{aligned}
\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Validation}) &= \emptyset \quad (0\text{ overlapping patients}) \\
\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Test}) &= \emptyset \quad (0\text{ overlapping patients}) \\
\text{Patients}(\text{Validation}) \cap \text{Patients}(\text{Test}) &= \emptyset \quad (0\text{ overlapping patients}) \\
\text{Patients}(\text{Active Sets}) \cap \text{Patients}(\text{Conflict Quarantine}) &= \emptyset \quad (0\text{ overlapping patients})
\end{aligned}$$

- **Patient Grouping Integrity:** In all active manifests, 100% of the images belonging to a patient (bilateral breasts, CC and MLO views, multiple abnormalities) reside in exactly one partition.
- **Image Path Duplication:** Zero baseline PNG image paths are shared across partitions.

---

## 5. Label Integrity & Image Traceability

### A. Label Mapping Distribution
- Mapping Rule:
  - `BENIGN` $\rightarrow$ **0** (1,429 records)
  - `BENIGN_WITHOUT_CALLBACK` $\rightarrow$ **0** (682 records)
  - `MALIGNANT` $\rightarrow$ **1** (1,457 records)
- Total Benign: 2,111 (59.16%)
- Total Malignant: 1,457 (40.84%)
- Missing / Unknown / Invalid labels: **0**
- Class Imbalance Ratio: $1.45 : 1.00$

### B. Class Balance Across Partitions

| Partition | Total Records | Benign Records (0) | Malignant Records (1) | Malignant % | Patients | Malignant Patients % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Active Train** | 2,239 | 1,324 | 915 | 40.87% | 973 | 48.82% |
| **Active Validation** | 542 | 312 | 230 | 42.44% | 244 | 48.77% |
| **Active Test** | 644 | 387 | 257 | 39.91% | 318 | 43.71% |
| **Quarantined Conflicts**| 143 | 88 | 55 | 38.46% | 31 | 61.29% |
| **Overall Dataset** | **3,568** | **2,111** | **1,457** | **40.84%** | **1,566** | **48.02%** |

### C. Image Traceability & Source Provenance
- All 3,568 records link directly to `data/processed/baseline/full_mammogram/P_*_*_*_baseline.png`.
- Zero records link to cropped ROIs (`cropped image file path`) or segmentation masks (`ROI mask file path`).
- Zero records link to synthetic noisy (Stage 8) or denoised (Stage 9) images.
- Full provenance (`source_csv`, `csv_row_number`, `case_key`, `view_identifier`, `assigned_split`) is preserved across all manifests.

---

## 6. Reproducibility & Immutability Verification

1. **Seed Reproducibility:** Validation was derived using `train_test_split(..., test_size=0.20, random_state=42, stratify=...)`. Re-executing with seed 42 produces identical patient assignments down to the exact row order.
2. **Immutability of Prior Stages:** All raw CSVs in `data/raw/CBIS_DDSM/csv/` and all metadata files from Stages 1–12 (including `baseline_preprocessing_metadata.csv` and `preprocessing_candidate_comparison.csv`) have identical byte sizes and timestamps, confirming zero modifications.

---

## 7. Policy Assessment & Next Steps for Stage 14

Because 60 official test records were quarantined along with their 83 training records to prevent data leakage, a formal policy decision is required for benchmarking in Stage 14:

### Policy Options for Stage 14:

1. **Option A: Pure Leakage-Free Benchmark (Recommended for MSc AI Thesis)**
   - Train on `classification_train_manifest.csv` (2,239 records).
   - Validate on `classification_val_manifest.csv` (542 records).
   - Test on `classification_test_manifest.csv` (644 records).
   - *Advantage:* Scientifically rigorous, 100% patient-isolated, zero leakage. The MSc thesis can explicitly document this as a methodological correction to published literature that ignored CBIS-DDSM cross-split leakage.

2. **Option B: Dual-Report Benchmark (Recommended for Academic Publication)**
   - Primary metric evaluated on the **Clean Test Set (644 records)**.
   - Secondary metric evaluated on the **Full Official Test Set (704 records)**, noting which 60 test records had cross-split training exposure in literature benchmarks.
   - *Advantage:* Allows direct head-to-head comparison with papers that used the uncorrected 704 test records, while highlighting our leakage-free methodology.

3. **Option C: Reassign Conflicted Patients Entirely to Test**
   - Reassign all 31 patients exclusively to the Test set (making Test = $644 + 143 = 787\text{ records}$).
   - Active Train pool remains 2,781 records (Train: 2,225, Val: 556).
   - *Advantage:* Zero training records leak into test, but test set size diverges from the official 704.

---

## 8. Final Audit Conclusion

### **Status: REVIEW REQUIRED — POLICY DECISION NEEDED**

The Stage 13 classification dataset is **technically sound, fully traceable, and 100% free of patient-level leakage**. 

Once the user/supervisor confirms **Option A (Clean 644 Test Set)** or **Option B (Dual Benchmark)** as the evaluation protocol, the dataset is **READY FOR STAGE 14 (Baseline CNN Modeling)**. No code or manifest modifications are needed to proceed under Option A.
