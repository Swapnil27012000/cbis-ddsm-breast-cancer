# CBIS-DDSM Breast Cancer Detection & Denoising AI Pipeline

A modular, production-ready deep learning pipeline for mammography preprocessing, noise simulation, advanced medical denoising, and high-accuracy breast cancer classification (Benign vs. Malignant) using the CBIS-DDSM dataset.

---

## 📁 Project Architecture Overview

```
breast-cancer-detection-ai-model-project/
├── config/                  # Centralized YAML configuration files
├── data/
│   ├── raw/                 # Original CBIS-DDSM dataset (immutable)
│   ├── metadata/            # Curated CSV records, patient splits, reports
│   └── processed/          # Staged intermediate outputs (validated, ROIs, denoised, etc.)
├── dataset/                 # Raw DICOM/JPEG files & original case description CSVs
├── src/
│   ├── data/                # Dataset audit, CSV loaders, metadata builders, patient-safe splitters
│   ├── preprocessing/       # Mammogram validation, normalization, CLAHE contrast & sharpening
│   ├── gpu/                 # CUDA memory management, streaming, and batch processors
│   ├── noise/               # Synthetic medical noise models (Gaussian, Poisson, Speckle, etc.)
│   ├── denoising/           # Medical filters (Wiener, Bilateral, NLM, Anscombe, Adaptive Median, Kuan)
│   ├── metrics/             # Image fidelity (PSNR, SSIM, SNR, CNR, CII, Entropy)
│   ├── datasets/            # PyTorch Dataset classes & Albumentations transforms
│   ├── models/              # ResNet, DenseNet, EfficientNet backbones for mammograms
│   ├── training/            # Mixed-precision training loops, loss functions, checkpointing
│   ├── evaluation/          # ROC curves, confusion matrices, Grad-CAM visualization
│   └── utils/               # Structured logging, safe I/O, reproducibility
├── models/                  # Checkpoints (best_model.pt), exported ONNX weights, logs
├── experiments/             # Structured ablation and hyperparameter runs
├── results/                 # Metrics (metrics.csv), reports, and confusion matrix plots
└── tests/                   # Automated unit test suite
```

---

## 🚀 Quick Start (Local Environment)

### 1. Environment Setup
```bash
# Clone the repository and navigate to folder
cd cbis-ddsm-breast-cancer

# Create & activate a virtual environment
# On Linux / macOS / Ubuntu:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install GPU/PyTorch requirements
pip install -r requirements-gpu.txt
```

### 2. Verify GPU Acceleration
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Pipeline Execution Stages
```bash
# Stage 1: Dataset structure audit & path linkage verification (read-only)
python -m src.data.dataset_audit

# Stage 2: Image inventory, reference mapping & decodability validation (read-only)
python -m src.data.image_inventory

# Stage 3: Image-role verification and visual inspection (read-only)
python -m src.data.image_role_verification

# Stage 4: Image quality control and validation audit (read-only)
python -m src.preprocessing.image_quality

# Stage 5: Primary baseline image preprocessing pilot & full-dataset execution
python -m src.preprocessing.baseline_preprocessing
python -m src.preprocessing.baseline_visualization
python -m src.preprocessing.verify_optimization_equivalence

# Stage 6: Controlled contrast enhancement experiment (Baseline vs HistEq vs CLAHE)
python -m src.preprocessing.contrast_enhancement
python -m src.preprocessing.contrast_visualization

# Stage 7: Controlled image sharpening experiment (3x2 factorial: Baseline, HistEq, CLAHE x None, Unsharp)
python -m src.preprocessing.sharpening
python -m src.preprocessing.sharpening_visualization

# Stage 8: Controlled artificial noise experiment (5 synthetic noise models on Stage-5 baseline)
python -m src.preprocessing.noise_experiment

# Stage 9: Medical image denoising experiment (8 filters on exact Stage-8 noisy images)
python -m src.preprocessing.denoising_experiment
# Or quick pilot test on 5 images:
python -m src.preprocessing.denoising_experiment --limit 5

# Stage 10: Denoising metrics & quantitative evaluation (MSE, PSNR, SSIM, SNR, CNR, MAE, RMSE, Entropy, Gradient)
python -m src.preprocessing.denoising_metrics

# Stage 11: Preprocessing visualization, comparison panels, difference maps & scientific report
python -m src.preprocessing.stage11_visualization
python -m src.preprocessing.stage11_audit
python -m src.preprocessing.stage11_report

# Stage 12: Evidence-based preprocessing strategy selection (shortlist candidates C1–C5)
python -m src.preprocessing.stage12_strategy_selection

# Stage 13: Classification dataset construction & patient-level zero-leakage splitting
python -m src.data.stage13_dataset

# Stage 14: Baseline CNN training and clean test evaluation
# Quick smoke test:
python -m src.training.stage14_trainer --smoke-test
# Full baseline training:
python -m src.training.stage14_trainer --epochs 15 --batch-size 32

# Stage 15+: Deep transfer learning (ResNet-50, DenseNet, EfficientNet)
python -m src.training.train
```

---

## 🐳 Docker Deployment & Execution (Windows & Linux / Ubuntu)

The Docker Compose configuration is **100% cross-platform** and runs identically on **Ubuntu / Linux, Windows (WSL2 / Docker Desktop), and macOS**.

### Prerequisites for Ubuntu / Linux
1. **Docker & Docker Compose Plugin**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
   ```
2. **NVIDIA Container Toolkit (for GPU passthrough on Linux)**:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

---

### Option A: Using Docker Compose (Recommended)

1. **Build the container image**:
   ```bash
   docker compose build
   ```

2. **Run the full training pipeline with GPU acceleration**:
   ```bash
   docker compose up
   ```

3. **Run individual pipeline stages inside the container**:

   - **Stage 1 — Run dataset structure and path linkage audit**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.data.dataset_audit
     ```

   - **Stage 2 — Run image inventory & reference mapping**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.data.image_inventory
     ```

   - **Stage 3 — Run image-role verification and visual inspection**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.data.image_role_verification
     ```

   - **Stage 4 — Run image quality control and validation audit**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.image_quality
     ```

   - **Stage 5 — Run primary baseline image preprocessing pilot & visual validation**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.baseline_preprocessing
     ```
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.baseline_visualization
     ```

   - **Stage 6 — Controlled contrast enhancement experiment**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.contrast_enhancement
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.contrast_visualization
     ```

   - **Stage 7 — Controlled image sharpening experiment**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.sharpening
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.sharpening_visualization
     ```

   - **Stage 8 — Controlled artificial noise experiment**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.noise_experiment
     ```

   - **Stage 9 — Medical image denoising experiment (8 filters on Stage-8 noisy images)**:
     ```bash
     # Full 16-case pilot run with safe resume (80 noisy inputs -> 640 denoised images):
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.denoising_experiment --cases 16

     # Force overwrite mode (recomputes all outputs):
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.denoising_experiment --cases 16 --overwrite

     # Run Stage 9 unit tests:
     docker compose run --rm cbis-ddsm-ai pytest tests/test_denoising_experiment.py -v
     ```

   - **Stage 10 — Denoising metrics & quantitative evaluation**:
     ```bash
     # Full 16-case pilot evaluation with safe resume (evaluates missing records without duplicating):
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.denoising_metrics

     # Force overwrite mode (recomputes all 640 evaluations from scratch):
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.denoising_metrics --overwrite

     # Run Stage 10 unit tests:
     docker compose run --rm cbis-ddsm-ai pytest tests/test_denoising_metrics.py -v
     ```

   - **Stage 11 — Preprocessing visualization & comparison**:
     ```bash
     # Run Stage 11 visual comparisons & quantitative chart generation:
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.stage11_visualization

     # Run Stage 11 artifact audit & validation checks:
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.stage11_audit

     # Compile Stage 11 scientific comparison report:
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.stage11_report

     # Run Stage 11 unit & integration tests:
     docker compose run --rm cbis-ddsm-ai pytest tests/test_stage11_visualization.py -v
     ```

   - **Stage 12 — Preprocessing strategy selection (evidence-based shortlist)**:
     ```bash
     # Evaluate empirical evidence from Stages 5–11 and produce C1–C5 candidate comparison:
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.stage12_strategy_selection

     # Run Stage 12 unit tests:
     docker compose run --rm cbis-ddsm-ai pytest tests/test_stage12_strategy_selection.py -v
     ```

   - **Stage 13 — Classification dataset construction & patient-level splitting**:
     ```bash
     # Construct auditable classification manifests and zero-leakage patient-level splits:
     docker compose run --rm cbis-ddsm-ai python -m src.data.stage13_dataset

     # Run Stage 13 unit & integration tests:
     docker compose run --rm cbis-ddsm-ai pytest tests/test_stage13_classification_dataset.py -v
     ```

   - **Stage 14 — Baseline CNN training & evaluation**:
     ```bash
     # Quick smoke test (1 epoch on 32 samples):
     docker compose run --rm cbis-ddsm-ai python -m src.training.stage14_trainer --smoke-test

     # Full training (15 epochs) on NVIDIA RTX 2050 GPU:
     docker compose run --rm cbis-ddsm-ai python -m src.training.stage14_trainer --epochs 15 --batch-size 32

     # Run Stage 14 unit and integration tests:
     docker compose run --rm cbis-ddsm-ai pytest tests/test_stage14_baseline_cnn.py -v
     ```

   - **Build metadata and validation reports**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.data.metadata_builder
     ```

   - **Run full preprocessing pipeline**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.pipeline
     ```

   - **Finalize preprocessing dataset & integrity verification**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.finalize_preprocessing
     ```

   - **Run classification training**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.training.train
     ```

   - **Run unit test suite**:
     ```bash
     docker compose run --rm cbis-ddsm-ai pytest tests/
     ```

4. **Stop and clean up containers**:
   ```bash
   docker compose down
   ```

---

### Option B: Using Docker CLI Directly

1. **Build the Docker Image**:
   ```bash
   docker build -t cbis-ddsm-ai:latest .
   ```

2. **Run Pipeline Stages with Docker CLI**:

   **Run Dataset Audit (Windows PowerShell):**
   ```powershell
   docker run --rm `
     -v ${PWD}/data:/app/data `
     -v ${PWD}/dataset:/app/dataset `
     -v ${PWD}/dataset:/app/data/raw/CBIS_DDSM `
     -v ${PWD}/config:/app/config `
     -v ${PWD}/src:/app/src `
     cbis-ddsm-ai:latest python -m src.data.dataset_audit
   ```

   **Run Full Training with NVIDIA GPU support (Windows PowerShell):**
   ```powershell
   docker run --gpus all --rm `
     -v ${PWD}/data:/app/data `
     -v ${PWD}/dataset:/app/dataset `
     -v ${PWD}/dataset:/app/data/raw/CBIS_DDSM `
     -v ${PWD}/config:/app/config `
     -v ${PWD}/src:/app/src `
     -v ${PWD}/models:/app/models `
     -v ${PWD}/results:/app/results `
     cbis-ddsm-ai:latest python -m src.training.train
   ```

   **Run Full Training with NVIDIA GPU support (Ubuntu / Linux / macOS Bash):**
   ```bash
   docker run --gpus all --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/dataset:/app/dataset \
     -v $(pwd)/dataset:/app/data/raw/CBIS_DDSM \
     -v $(pwd)/config:/app/config \
     -v $(pwd)/src:/app/src \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/results:/app/results \
     cbis-ddsm-ai:latest python -m src.training.train
   ```

3. **Run on CPU (fallback without GPU)**:
   ```bash
   docker run --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/dataset:/app/dataset \
     -v $(pwd)/dataset:/app/data/raw/CBIS_DDSM \
     -v $(pwd)/config:/app/config \
     -v $(pwd)/src:/app/src \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/results:/app/results \
     cbis-ddsm-ai:latest python -m src.training.train
   ```

---

## 🧪 Running Tests
```bash
# Run all project tests
pytest tests/

# Run Stage 9 Denoising Experiment unit tests specifically
pytest tests/test_denoising_experiment.py -v

# Run Stage 10 Denoising Metrics unit tests specifically
pytest tests/test_denoising_metrics.py -v

# Run Stage 11 Preprocessing Visualization unit tests specifically
pytest tests/test_stage11_visualization.py -v

# Run Stage 12 Strategy Selection unit tests specifically
pytest tests/test_stage12_strategy_selection.py -v

# Run Stage 13 Classification Dataset & Patient-Level Splitting tests specifically
pytest tests/test_stage13_classification_dataset.py -v

# Run Stage 14 Baseline CNN Model unit & integration tests specifically
pytest tests/test_stage14_baseline_cnn.py -v
```

---

## 📊 Research Pipeline Status & Experiment Registry

| Stage | Name | Description | Status |
|---|---|---|---|
| **Stage 1** | Dataset Structure Audit | Physical CBIS-DDSM file tree, format & DICOM mapping validation | **COMPLETED** |
| **Stage 2** | Image Inventory & Mapping | Resolving exact original mammogram paths vs. ROI/crop labels | **COMPLETED** |
| **Stage 3** | Image Role Verification | Separating full mammograms from crops and lesion masks | **COMPLETED** |
| **Stage 4** | Image Quality Control | Decodability, dynamic range, aspect ratio & artifact checks | **COMPLETED** |
| **Stage 5** | Primary Baseline Preprocessing | Conservative breast segmentation, margin crop & 1st–99th percentile normalization | **COMPLETED** |
| **Stage 6** | Contrast Enhancement Experiment | Controlled evaluation: Baseline vs. Histogram Equalization vs. CLAHE | **COMPLETED** |
| **Stage 7** | Sharpening Experiment | Controlled evaluation: Unsharp Masking vs. Unsharpened across contrast variants | **COMPLETED** |
| **Stage 8** | Artificial Noise Experiment | Controlled synthetic noise perturbation: Gaussian, Salt & Pepper, Speckle, Poisson, Mixed P-G | **COMPLETED** |
| **Stage 9** | Denoising Experiment | Benchmark 8 medical denoising filters across the exact same 80 Stage-8 noisy images | **COMPLETED** |
| **Stage 10** | Denoising Metrics | Quantitative image fidelity analysis (MSE, PSNR, SSIM, SNR, CNR, MAE, RMSE, Entropy, Gradient) | **COMPLETED** |
| **Stage 11** | Preprocessing Visualization & Comparison | Visual contact sheets, difference maps & statistical comparisons | **COMPLETED** |
| **Stage 12** | Preprocessing Strategy Selection | Evidence-based review of Stages 5–11; shortlisted candidates C1–C5; no premature winner | **COMPLETED** |
| **Stage 13** | Classification Dataset & Splitting | Verified 3,568 baseline PNGs; 0 patient leakage; 31 official conflicts quarantined | **COMPLETED** |
| **Stage 14** | Baseline CNN Model | Custom 4-block CNN baseline on Stage 5 C1 images; BCE loss with pos_weight=1.45 | **COMPLETED** |
| **Stage 15** | Transfer Learning (ResNet-50) | Transfer-learning classification benchmark on full mammograms | *Next Stage* |
| **Stage 16–18** | Advanced Models & Evaluation | DenseNet-121, EfficientNet-B0 comparison, multi-view aggregation & clinical ROC-AUC | *Upcoming* |

---

## 🔬 Stage 9: Medical Image Denoising Experiment Details

Stage 9 evaluates 8 medical image denoising algorithms on the exact 80 synthetic noisy mammograms generated during Stage 8 ($16\text{ pilot cases} \times 5\text{ noise models}$), producing $80 \times 8 = 640$ denoised outputs.

### Denoising Algorithms & Configurations

| # | Algorithm | Implementation | Configuration | Primary Target |
|---|---|---|---|---|
| 1 | **Median** | OpenCV SIMD Median Blur | `kernel_size = 3` | Salt & Pepper impulse noise |
| 2 | **Gaussian** | Spatial Gaussian Smoothing | `kernel_size = 5`, `sigma = 1.0` | Additive high-frequency Gaussian noise |
| 3 | **Wiener** | SciPy Adaptive MMSE Local Filter | `window_size = 5` | Stationary additive noise |
| 4 | **Bilateral** | Edge-Preserving Domain/Range Filter | `diameter = 5`, `sigma_color = 0.1`, `sigma_space = 5.0` | Edge-retentive noise reduction |
| 5 | **Non-Local Means** | OpenCV fastNlMeans Patch Matching | `patch_size = 5`, `patch_distance = 6`, `h = 0.05` | Texture and structural self-similarity |
| 6 | **Anscombe-Wiener** | Generalized Anscombe VST + Wiener | `scale = 255.0`, `wiener_window = 5`, `sigma = 1.0` | Quantum Poisson & Mixed shot noise |
| 7 | **Adaptive Median** | Two-Level Dynamic Window Expansion | `min_window = 3`, `max_window = 7` | Dense impulse noise |
| 8 | **Kuan** | Local Statistical MMSE Speckle Filter | `window_size = 5`, `noise_var = 0.04`, `damping = 1.0` | Multiplicative Speckle noise |

### Scientific Fairness & Reproducibility Protocol
- **Zero Noise Re-generation**: All 8 denoisers receive the exact same pre-loaded `float32 [0.0, 1.0]` array per noisy image.
- **SHA-256 Hash Verification**: An array hash (`noisy_image_hash`) is recorded in metadata to guarantee identical inputs.
- **Lossless Storage**: Denoised images are clipped to $[0.0, 1.0]$ and saved as lossless 8-bit grayscale PNGs in `data/processed/denoising/<noise_type>/<method>/`.
- **14-Point Validation Audit**: Automatically verifies dimension match, grayscale format, dynamic range, no NaNs/Infs, and absence of blank or saturated outputs.
- **No Early Winner Selection**: Algorithmic ranking and quantitative metrics (MSE, PSNR, SSIM, SNR, CNR) are reserved strictly for Stage 10.

---

## 🎯 Stage 12: Evidence-Based Preprocessing Strategy Selection

Stage 12 evaluates empirical evidence from Stages 5–11 to establish a defensible shortlist of candidate preprocessing pipelines for downstream classification, preventing premature bias or unverified assumptions.

### Preprocessing Candidate Taxonomy

| Candidate ID | Strategy Name | Family | Experimental Source | Corpus Availability | Recommendation Role |
|---|---|---|---|---|---|
| **C1_BASELINE** | Baseline Preprocessing (Control) | Baseline | Stage 5 Baseline Pipeline | **Full Corpus (3,568 images)** | **Mandatory Scientific Control Reference** |
| **C2_CLAHE** | Contrast Limited Adaptive HistEq | Contrast | Stage 6 Contrast Experiment | Pilot Only (16 images) | Primary Contrast Enhancement Candidate |
| **C3_UNSHARP_MASK** | High-Frequency Sharpening | Edge Enhancement | Stage 7 Sharpening Experiment | Pilot Only (16 images) | Primary Edge Enhancement Candidate |
| **C4_CLAHE_UNSHARP** | Compound (CLAHE + Unsharp) | Compound | Stage 7 Factorial Experiment | Pilot Only (16 images) | Exploratory Compound Candidate |
| **C5_MEDIAN_DENOISING** | Non-Linear Median Filter (3x3) | Denoising | Stages 8–10 Benchmark | Pilot Only (16 images / 80 noisy) | Exploratory Noise Robustness Benchmark |

### Documented Discarded Methods
- **Global Histogram Equalization (HistEq)**: Excluded due to severe dynamic range blowout, background sensor noise amplification, and tissue saturation.
- **Gaussian Spatial Smoothing**: Excluded due to indiscriminate low-pass filtering that blurs microcalcifications and fine spicules.
- **Non-Local Means (NLM)**: Excluded due to prohibitive computational latency (>10 hrs for full corpus) and over-smoothing of fine glandular structures.
- **Adaptive Median Filtering**: Excluded because standard 3x3 median filtering achieved higher general fidelity on continuous medical noise models.
- **Untested Arbitrary Chains**: Strict experimental protocols exclude multi-stage chains not explicitly evaluated in Stages 5–11.

*Policy Constraint*: Winner declaration status is strictly `NONE_DECLARED_SHORTLIST_PRESERVED`. Final preprocessing efficacy is deferred to empirical CNN benchmark evaluation in Stage 14+.

---

## 🧬 Stage 13: Classification Dataset Construction & Patient-Level Splitting

Stage 13 maps verified Stage 5 full/original baseline mammograms to clinical pathology labels, audits multi-abnormality case records, quarantines official cross-split conflicts, and creates 100% patient-isolated, leakage-free train, validation, and test manifests.

### 1. Source Inventory Reconciliation
- **Official Raw Case Description Records**: 3,568 total (1,318 mass train + 378 mass test + 1,546 calc train + 326 calc test).
- **Stage 5 Baseline Processed Images**: 3,568 lossless 8-bit grayscale PNGs (100% verified and readable on disk).
- **Unique Patients**: 1,566 across all 4 description CSVs.
- **Unique Mammographic Views**: 3,032 distinct anatomical views (`patient_id` + `breast_side` + `image_view`).
- **Multi-Abnormality Shared Mammograms**: 536 secondary abnormality records sharing physical mammograms are tracked with full diagnostic provenance; 3,032 canonical primary views are flagged via `is_primary_view_record = True`.

### 2. Reliable Pathology Label Mapping
- `BENIGN` $\rightarrow$ **0** (1,429 records, 40.05%)
- `BENIGN_WITHOUT_CALLBACK` $\rightarrow$ **0** (682 records, 19.11%)
- `MALIGNANT` $\rightarrow$ **1** (1,457 records, 40.84%)
- **Total Class 0 (Benign)**: 2,111 records (59.16%)
- **Total Class 1 (Malignant)**: 1,457 records (40.84%)
- **Unmapped / Unknown Labels**: **0** (Zero unmapped records; unknown strings are isolated and never assumed benign).

### 3. Official Cross-Partition Patient Conflict Audit
CBIS-DDSM creators independently split the mass cohort and calcification cohort. Consequently, **31 patients** were assigned to `train` in one CSV and `test` in another.
- **Quarantine Policy**: Conflicting patients are **not** silently reassigned. All 31 patients (143 records, 85 views) are isolated into a quarantined review set documented in `data/metadata/stage13/cross_partition_patient_conflicts.csv`.
- **Zero Leakage Guarantee**: The active `train`, `val`, and `test` manifests maintain **strict zero patient overlap**:
  $$\text{Train Patients} \cap \text{Val Patients} = \emptyset, \quad \text{Train Patients} \cap \text{Test Patients} = \emptyset, \quad \text{Val Patients} \cap \text{Test Patients} = \emptyset$$

### 4. Patient-Level Split Manifest Summary

| Partition | Patients | Case Records | Primary Views | Benign Records | Malignant Records | Malignant % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Active Train** | 973 | 2,239 | 1,891 | 1,324 | 915 | 40.87% |
| **Active Validation** | 244 | 542 | 465 | 312 | 230 | 42.44% |
| **Active Test** | 318 | 644 | 591 | 387 | 257 | 39.91% |
| **Quarantined Conflicts** | 31 | 143 | 85 | 88 | 55 | 38.46% |
| **Total / Overall** | **1,566** | **3,568** | **3,032** | **2,111** | **1,457** | **40.84%** |

*Split Rules*: Official test assignment is preserved for clean test patients; validation is derived strictly from clean official training patients using a deterministic seed (`seed = 42`) with patient-level stratification.

### 5. Stage 13 Artifacts Directory (`data/metadata/stage13/`)
- `classification_record_audit.csv`: Complete audit of all 3,568 records with view and duplication flags.
- `classification_master_manifest.csv`: Unified master manifest with image paths, labels, and assigned splits.
- `classification_train_manifest.csv`: Active, clean, leakage-free training manifest (2,239 records).
- `classification_val_manifest.csv`: Active, clean, leakage-free validation manifest (542 records).
- `classification_test_manifest.csv`: Active, clean, leakage-free test manifest (644 records).
- `patient_level_split_manifest.csv`: Patient-level split assignment table (1,566 patients).
- `cross_partition_patient_conflicts.csv`: Audit of the 31 official cross-partition conflicting patients.
- `class_distribution_summary.csv`: Class distributions and ratios across all partitions.
- `missing_image_and_invalid_label_report.csv`: Zero-loss integrity audit report.
- `stage13_validation.csv`: Validation checklist with 10 passing verification items.
- `stage13_classification_config.json`: Machine-readable configuration and Stage 12 candidate bindings.
- `stage13_classification_report.txt`: Human-readable executive summary report.
- `stage13_classification_report.md`: Detailed Markdown report for academic documentation.

---

## 🧠 Stage 14: Baseline CNN Model Implementation

Stage 14 establishes an empirical scientific baseline for binary breast cancer classification (Benign vs. Malignant) using a custom 4-block convolutional neural network on Stage 5 baseline mammograms downscaled to $256 \times 256$ grayscale.

### 1. Architecture Specification (`BaselineBreastCancerCNN`)
- **Block 1**: `Conv2d(1, 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)` $\rightarrow$ Output: $32 \times 128 \times 128$
- **Block 2**: `Conv2d(32, 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)` $\rightarrow$ Output: $64 \times 64 \times 64$
- **Block 3**: `Conv2d(64, 128, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)` $\rightarrow$ Output: $128 \times 32 \times 32$
- **Block 4**: `Conv2d(128, 256, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)` $\rightarrow$ Output: $256 \times 16 \times 16$
- **Global Average Pooling**: `AdaptiveAvgPool2d((1, 1))` collapses spatial dimensions to a compact 256-d embedding.
- **Classifier Head**: `Linear(256, 128) -> BatchNorm1d -> ReLU -> Dropout(0.4) -> Linear(128, 1)` (Raw logits).
- **Total Parameters**: $\approx 460,000$ parameters (modest capacity to prevent memorization/overfitting).
- **VRAM Footprint**: Peaks at $\approx 1.1\text{ GB}$ VRAM, operating well within the 4 GB limit of the NVIDIA RTX 2050 GPU.

### 2. Training & Optimization Protocol
- **Loss Function**: `BCEWithLogitsLoss(pos_weight=1.447)` to compensate for the 59% benign vs. 41% malignant class distribution.
- **Optimizer**: `AdamW(lr=0.0003, weight_decay=1e-4)`.
- **LR Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`.
- **Mixed Precision**: `torch.amp.autocast('cuda')` + `GradScaler` for memory efficiency and accelerated training.
- **Early Stopping**: Tracks `val_loss` with a patience of 5 epochs. Best model state is saved to `baseline_cnn_best.pt`.
- **Evaluation**: The selected best checkpoint is evaluated once on the clean test set (644 records).

### 3. Execution Commands
```bash
# Step 1: Run smoke test (1 epoch on 32 samples to verify GPU memory & pipeline):
docker compose run --rm cbis-ddsm-ai python -m src.training.stage14_trainer --smoke-test

# Step 2: Run full training run (15 epochs, batch size 32):
docker compose run --rm cbis-ddsm-ai python -m src.training.stage14_trainer --epochs 15 --batch-size 32

# Step 3: Run Stage 14 unit and integration tests:
docker compose run --rm cbis-ddsm-ai pytest tests/test_stage14_baseline_cnn.py -v
```

### 4. Stage 14 Output Directory
- **Metadata** (`data/metadata/stage14/`): `stage14_baseline_cnn_config.json`, `stage14_dataset_loader_audit.csv`, `stage14_epoch_history.csv`, `stage14_validation_metrics.csv`, `stage14_test_metrics.csv`, `stage14_baseline_cnn_report.md`.
- **Checkpoints** (`models/stage14_baseline_cnn/`): `baseline_cnn_best.pt`, `baseline_cnn_final.pt`.
- **Results & Visualizations** (`results/stage14_baseline_cnn/`): `confusion_matrix_test.png`, `roc_curve_test.png`, `training_curves.png`, `test_metrics_summary.json`.



