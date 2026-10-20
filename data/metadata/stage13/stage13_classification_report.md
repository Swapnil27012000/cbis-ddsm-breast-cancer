# Stage 13 — Classification Dataset Construction and Patient-Level Splitting Report

**Generated:** 2026-10-20 12:10:23  
**Dataset:** CBIS-DDSM Full Mammograms  
**Status:** VALIDATION PASSED (Leakage-Free Patient-Level Splitting)  

## 1. Executive Summary & Inventory Reconciliation

| Inventory Metric | Count | Reconciliation Status |
| :--- | :--- | :--- |
| Raw Case Description Records | 3568 | Match (1318 mass train + 378 mass test + 1546 calc train + 326 calc test) |
| Stage 5 Baseline Processed Records | 3568 | 100% matched to raw case records |
| Physical Baseline PNGs Verified on Disk | 3568 | 100% readable with verified dimensions |
| Unique Patients | 1566 | Audited |
| Unique Mammographic Views | 3032 | CC and MLO views preserved per patient |
| Unique Physical Baseline Files | 3568 | Multi-abnormality shared views identified |
| Primary Canonical View Records | 3032 | Deduplicated physical views |
| Secondary Abnormality Records | 536 | Preserved for lesion-level provenance |

## 2. Pathology & Binary Label Mapping

- `BENIGN` $\rightarrow$ **0**
- `BENIGN_WITHOUT_CALLBACK` $\rightarrow$ **0**
- `MALIGNANT` $\rightarrow$ **1**

- **Class 0 (Benign):** 2111 records (59.16%)
- **Class 1 (Malignant):** 1457 records (40.84%)
- **Invalid / Unknown Labels:** 0
- **Class Imbalance:** 1.45 : 1.00

## 3. Patient-Level Zero-Leakage Split Architecture

| Partition | Patients | Records | Primary Views | Benign Records | Malignant Records | Malignant % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Active Train** | 973 | 2239 | 1891 | 1324 | 915 | 40.87% |
| **Active Val** | 244 | 542 | 465 | 312 | 230 | 42.44% |
| **Active Test** | 318 | 644 | 591 | 387 | 257 | 39.91% |
| **Quarantined Conflicts** | 31 | 143 | 85 | 88 | 55 | 38.46% |

> [!IMPORTANT]
> **Patient Isolation Verification**: Zero patient overlap exists between `train`, `val`, and `test` manifests. Cross-split contamination embedded in official CBIS-DDSM is quarantined in `cross_partition_patient_conflicts.csv`.

## 4. Official Cross-Partition Conflict Audit

CBIS-DDSM authors independently split mass and calcification cohorts. As an established artifact, **31 patients** appear in both official train and test sets.

| Patient ID | Total Records | Train Records | Test Records | Train Pathology | Test Pathology | Source CSVs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `P_00016` | 4 | 2 | 2 | MALIGNANT | MALIGNANT | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00041` | 4 | 2 | 2 | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_00159` | 3 | 2 | 1 | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00394` | 3 | 2 | 1 | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00495` | 6 | 2 | 4 | BENIGN | BENIGN_WITHOUT_CALLBACK | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_00510` | 14 | 12 | 2 | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00591` | 5 | 4 | 1 | MALIGNANT | MALIGNANT | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00601` | 4 | 2 | 2 | MALIGNANT | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00639` | 3 | 2 | 1 | MALIGNANT | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00652` | 3 | 2 | 1 | MALIGNANT | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00678` | 3 | 2 | 1 | MALIGNANT | MALIGNANT | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_00723` | 4 | 1 | 3 | BENIGN | BENIGN | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_00753` | 4 | 2 | 2 | MALIGNANT | BENIGN | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_00882` | 4 | 2 | 2 | BENIGN | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00906` | 5 | 2 | 3 | MALIGNANT | MALIGNANT | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_00924` | 9 | 8 | 1 | BENIGN_WITHOUT_CALLBACK | BENIGN_WITHOUT_CALLBACK | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_00969` | 8 | 4 | 4 | MALIGNANT | BENIGN,MALIGNANT | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01108` | 4 | 2 | 2 | MALIGNANT | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01152` | 6 | 4 | 2 | BENIGN | BENIGN | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_01179` | 3 | 2 | 1 | MALIGNANT | MALIGNANT | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_01237` | 3 | 1 | 2 | BENIGN | BENIGN | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_01307` | 4 | 2 | 2 | MALIGNANT | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01365` | 3 | 2 | 1 | MALIGNANT | BENIGN_WITHOUT_CALLBACK | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01467` | 3 | 2 | 1 | MALIGNANT | MALIGNANT | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01566` | 7 | 2 | 5 | BENIGN | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01605` | 3 | 2 | 1 | MALIGNANT | BENIGN | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01621` | 4 | 2 | 2 | BENIGN | BENIGN | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_01635` | 6 | 4 | 2 | BENIGN_WITHOUT_CALLBACK | MALIGNANT | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_01741` | 4 | 2 | 2 | BENIGN | MALIGNANT | calc_case_description_train_set.csv,mass_case_description_test_set.csv |
| `P_01803` | 4 | 2 | 2 | MALIGNANT | MALIGNANT | calc_case_description_test_set.csv,mass_case_description_train_set.csv |
| `P_01883` | 3 | 1 | 2 | BENIGN | MALIGNANT | calc_case_description_test_set.csv,mass_case_description_train_set.csv |

## 5. Stage 12 Preprocessing Candidates & Classification Readiness

| Candidate ID | Strategy Name | Corpus Availability | Classification Readiness |
| :--- | :--- | :--- | :--- |
| `C1_BASELINE` | Baseline Preprocessing (Control) | **Full Corpus (3,568 images)** | **READY FOR EVALUATION** |
| `C2_CLAHE` | CLAHE Contrast Enhancement | Pilot Only (16 images) | Deferred (requires full dataset preprocessing) |
| `C3_UNSHARP_MASK` | Unsharp Masking Sharpening | Pilot Only (16 images) | Deferred (requires full dataset preprocessing) |
| `C4_CLAHE_UNSHARP` | Compound Enhancement | Pilot Only (16 images) | Deferred (requires full dataset preprocessing) |
| `C5_MEDIAN_DENOISING` | Median Filtering Denoising | Pilot Only (16 images) | Deferred (requires full dataset preprocessing) |

## 6. Output Artifacts Directory

All Stage 13 metadata artifacts have been generated in `data/metadata/stage13/` with full provenance.