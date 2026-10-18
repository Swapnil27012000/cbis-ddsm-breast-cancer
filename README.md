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

# Stage 12+: Preprocessing strategy selection & classification training
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
| **Stage 12** | Preprocessing Strategy Selection | Evidence-based selection of optimal preprocessing pipeline | *Next Stage* |
| **Stage 13–18** | CNN Architecture & Training | Classification modeling with ResNet-50, DenseNet-121, EfficientNet | *Upcoming* |

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

