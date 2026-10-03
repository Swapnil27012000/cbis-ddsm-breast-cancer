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
├── src/
│   ├── data/                # CSV loaders, metadata builders, patient-safe splitters
│   ├── preprocessing/       # Mammogram validation, normalization, ROI & muscle removal
│   ├── gpu/                 # CUDA memory management, streaming, and batch processors
│   ├── noise/               # Synthetic medical noise models (Gaussian, Poisson, Speckle, etc.)
│   ├── denoising/           # Medical filters (Wiener, Bilateral, NLM, Anscombe, Adaptive Median, Kuan)
│   ├── metrics/             # Image fidelity (PSNR, SSIM, SNR, CNR, CII, Entropy)
│   ├── datasets/            # PyTorch Dataset classes & Albumentations transforms
│   ├── models/              # ResNet, DenseNet, EfficientNet backbones for mammograms
│   ├── training/            # Mixed-precision training loops, loss functions, checkpointing
│   ├── evaluation/          # ROC curves, confusion matrices, Grad-CAM visualization
│   └── utils/               # Structured logging, safe I/O, reproducibility
├── models/                  # Checkpoints, exported ONNX weights, TensorBoard logs
├── experiments/             # Structured ablation and hyperparameter runs
├── results/                 # Metric tables and publication figures
└── tests/                   # Automated unit test suite
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository and install dependencies
pip install -r requirements-gpu.txt
```

### 2. Verify GPU Acceleration
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Build Metadata
```bash
python -m src.data.metadata_builder
```

### 4. Run Preprocessing Pipeline
```bash
python -m src.preprocessing.pipeline
```

### 5. Run Denoising Benchmark
```bash
python experiments/run_denoising_benchmark.py
```

### 6. Finalize Preprocessing Dataset & Integrity Verification
```bash
python -m src.preprocessing.finalize_preprocessing
```

### 7. Train Classification Model (Future Stage)
```bash
python -m src.training.train
```

---

## 🐳 Docker Deployment & Execution

Ensure **Docker Desktop** is running (and NVIDIA Container Toolkit is enabled if utilizing GPU acceleration).

### Option A: Using Docker Compose (Recommended)

1. **Build the container image**:
   ```bash
   docker compose build
   ```

2. **Run the default training pipeline with GPU acceleration**:
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

   - **Run unit tests**:
     ```bash
     docker compose run --rm cbis-ddsm-ai pytest tests/
     ```

4. **Stop the containers**:
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
   ```bash
   # On Windows PowerShell:
   docker run --gpus all --rm `
     -v ${PWD}/data:/app/data `
     -v ${PWD}/config:/app/config `
     -v ${PWD}/models:/app/models `
     -v ${PWD}/results:/app/results `
     cbis-ddsm-ai:latest python -m src.training.train
   ```

3. **Run on CPU (fallback)**:
   ```bash
   # On Windows PowerShell:
   docker run --rm `
     -v ${PWD}/data:/app/data `
     -v ${PWD}/config:/app/config `
     -v ${PWD}/models:/app/models `
     -v ${PWD}/results:/app/results `
     cbis-ddsm-ai:latest python -m src.training.train
   ```

---

## 🧪 Running Tests
```bash
pytest tests/
```
