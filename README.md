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
│   ├── data/                # CSV loaders, metadata builders, patient-safe splitters
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
# Stage 1: Build unified master metadata and patient-stratified splits
python -m src.data.metadata_builder

# Stage 2: Run full image preprocessing pipeline (normalization, CLAHE, sharpening)
python -m src.preprocessing.pipeline

# Stage 3: Run noise simulation & denoising filter benchmarks
python experiments/run_denoising_benchmark.py

# Stage 4: Run preprocessing verification audit & integrity check
python -m src.preprocessing.finalize_preprocessing

# Stage 5: Train ResNet50 classification model (Benign vs. Malignant)
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

   - **Build metadata and validation reports**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.data.metadata_builder
     ```

   - **Run full preprocessing pipeline**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.preprocessing.pipeline
     ```

   - **Run noise simulation & denoising benchmark**:
     ```bash
     docker compose run --rm cbis-ddsm-ai python -m src.experiments.denoising_experiment
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

2. **Run with NVIDIA GPU support**:

   **On Ubuntu / Linux / macOS (Bash):**
   ```bash
   docker run --gpus all --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/dataset:/app/data/raw/CBIS_DDSM \
     -v $(pwd)/config:/app/config \
     -v $(pwd)/src:/app/src \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/results:/app/results \
     cbis-ddsm-ai:latest python -m src.training.train
   ```

   **On Windows (PowerShell):**
   ```powershell
   docker run --gpus all --rm `
     -v ${PWD}/data:/app/data `
     -v ${PWD}/dataset:/app/data/raw/CBIS_DDSM `
     -v ${PWD}/config:/app/config `
     -v ${PWD}/src:/app/src `
     -v ${PWD}/models:/app/models `
     -v ${PWD}/results:/app/results `
     cbis-ddsm-ai:latest python -m src.training.train
   ```

3. **Run on CPU (fallback without GPU)**:
   ```bash
   docker run --rm \
     -v $(pwd)/data:/app/data \
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
pytest tests/
```
