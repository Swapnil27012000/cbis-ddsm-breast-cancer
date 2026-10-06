STAGE 1
│
├── Dataset Structure Audit
│   └── Understand CBIS-DDSM folder/image structure
│
↓
STAGE 2
│
├── Image Inventory & CSV Mapping
│   └── Build reliable physical-image ↔ CSV mapping
│
↓
STAGE 3
│
├── Image Role Verification
│   ├── Full / Original mammograms
│   ├── Cropped images
│   └── ROI / mask images
│
↓
STAGE 4
│
├── Image Quality Control
│   ├── Readability
│   ├── Dimensions
│   ├── Intensity statistics
│   ├── ROI/mask validation
│   └── Quality flags
│
↓
STAGE 5
│
├── PRIMARY BASELINE PREPROCESSING
│   ├── Full/Original mammogram selection
│   ├── Grayscale conversion
│   ├── Conservative breast/background detection
│   ├── Conservative breast crop
│   ├── 1–99 percentile intensity normalization
│   └── Clean baseline mammogram
│
↓
STAGE 6
│
├── CONTRAST ENHANCEMENT EXPERIMENT
│   ├── Baseline / Control
│   ├── Histogram Equalization
│   └── CLAHE
│
↓
STAGE 7
│
├── SHARPENING EXPERIMENT
│   ├── No sharpening / Control
│   └── Unsharp Masking
│
│   Applied independently to:
│   ├── Baseline
│   ├── Histogram Equalization
│   └── CLAHE
│
↓
STAGE 8
│
├── ARTIFICIAL NOISE EXPERIMENT
│   │
│   ├── Clean Stage-5 baseline
│   ├── Gaussian noise
│   ├── Salt & Pepper noise
│   ├── Speckle noise
│   ├── Poisson noise
│   └── Mixed Poisson-Gaussian noise
│
│   ↓
│   Save exact noisy images
│
↓
STAGE 9
│
├── DENOISING EXPERIMENT
│   ├── Median
│   ├── Gaussian
│   ├── Wiener
│   ├── Bilateral
│   ├── Non-Local Means
│   ├── Anscombe-Wiener
│   ├── Adaptive Median
│   └── Kuan
│
↓
STAGE 10
│
├── DENOISING METRICS
│   ├── MSE
│   ├── PSNR
│   ├── SSIM
│   ├── SNR
│   ├── CNR
│   └── Other appropriate metrics
│
↓
STAGE 11
│
├── PREPROCESSING VISUALIZATION & COMPARISON
│   ├── Clean vs noisy
│   ├── Noisy vs denoised
│   ├── Histograms
│   ├── Difference maps
│   ├── Metric comparisons
│   └── Statistical analysis
│
↓
STAGE 12
│
├── FINAL PREPROCESSING STRATEGY
│   ├── Analyze contrast results
│   ├── Analyze sharpening results
│   ├── Analyze denoising results
│   └── Select preprocessing configuration
│
↓
STAGE 13
│
├── CLASSIFICATION DATASET
│   ├── Patient-level split
│   ├── Train
│   ├── Validation
│   └── Test
│
↓
STAGE 14
│
├── BASELINE CNN
│   └── Establish baseline classification performance
│
↓
STAGE 15
│
├── RESNET-50
│   └── Transfer-learning classification
│
↓
STAGE 16
│
├── OTHER MODEL COMPARISON
│   ├── DenseNet
│   └── EfficientNet
│
↓
STAGE 17
│
├── CLASSIFICATION EVALUATION
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   ├── F1-score
│   ├── Specificity
│   ├── Sensitivity
│   ├── ROC-AUC
│   ├── Confusion matrix
│   └── Patient-level evaluation
│
↓
STAGE 18
│
└── FINAL RESEARCH RESULTS
    ├── Compare preprocessing strategies
    ├── Compare models
    ├── Error analysis
    ├── Statistical analysis
    └── Final conclusions