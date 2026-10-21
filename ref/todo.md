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
│   ├── Analyze contrast results [x]
│   ├── Analyze sharpening results [x]
│   ├── Analyze denoising results [x]
│   └── Select preprocessing configuration (candidate shortlist established) [x]
│
↓
STAGE 13
│
├── CLASSIFICATION DATASET
│   ├── Patient-level split [x]
│   ├── Train [x]
│   ├── Validation [x]
│   └── Test [x]
│
↓
STAGE 14
│
├── BASELINE CNN
│   └── Establish baseline classification performance [x]
│
↓
STAGE 15
│
├── RESNET-50
│   └── Transfer-learning classification [x]
│
↓
STAGE 16
│
├── OTHER MODEL COMPARISON
│   ├── DenseNet [x]
│   └── EfficientNet [x]
│
↓
STAGE 17
│
├── CLASSIFICATION EVALUATION
│   ├── Accuracy [x]
│   ├── Precision [x]
│   ├── Recall [x]
│   ├── F1-score [x]
│   ├── Specificity [x]
│   ├── Sensitivity [x]
│   ├── ROC-AUC [x]
│   ├── Confusion matrix [x]
│   └── Patient-level evaluation [x]
│
↓
STAGE 18
│
└── FINAL RESEARCH RESULTS
    ├── Compare preprocessing strategies [x]
    ├── Compare models [x]
    ├── Error analysis [x]
    ├── Statistical analysis [x]
    └── Final conclusions [x]