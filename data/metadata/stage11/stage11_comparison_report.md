# CBIS-DDSM Stage 11 — Preprocessing Visualization & Comparison Report

## 1. Stage Information & Research Scope
- **Stage**: Stage 11 — Preprocessing Visualization and Comparison
- **Academic Context**: MSc Information Technology (Artificial Intelligence) Research Project
- **Dataset**: CBIS-DDSM Mammography
- **Scope**: Reproducible Visual Comparisons and Evidence-Based Summaries across Stages 5–10
- **Policy**: **No Preprocessing Strategy Selected** (Strategy Selection Deferred to Stage 12)

## 2. Artifact and Record Count Audit

| Stage | Name | Actual Records | Expected Records | Unique Pilot Cases | Status | Notes |
|---|---|---|---|---|---|---|
| **Stage 5** | Baseline Preprocessing | 3568 | 3568 | 16 | PASS | Full dataset processed; 16 pilot mammograms authoritative clean references |
| **Stage 6** | Contrast Enhancement | 48 | 48 | 16 | PASS | 16 pilot mammograms x 3 methods (baseline, histeq, clahe) |
| **Stage 7** | Sharpening Experiment | 96 | 96 | 16 | PASS | 16 pilot mammograms x 6 factorial conditions (3 contrast x 2 sharpen) |
| **Stage 8** | Artificial Noise Simulation | 96 | 96 | 16 | PASS | 16 clean references + 80 noisy images across 5 noise models |
| **Stage 9** | Denoising Experiment | 640 | 640 | 16 | PASS | 16 pilot mammograms x 5 noise models x 8 denoising methods = 640 outputs |
| **Stage 10** | Quantitative Denoising Metrics | 640 | 640 | 16 | PASS | All 640 evaluations completed and validated across 16 pilot cases (5 noise x 8 methods); 0 missing |

> [!IMPORTANT]
> **Coverage Audit Finding**: Stage 10 contains complete factorial evaluations across all 16 pilot mammograms (640 valid records: 16 cases x 5 noise types x 8 methods). 0 missing evaluations, 0 duplicate records. All 640 records passed validation.

## 3. Representative Case Selection
Deterministic selection rules ensure broad clinical and morphological representation:
1. **`P_00004_1_LEFT_CC`** (Benign Mass, CC View, 5491x2986): Primary anchor case directly linking visual inspection with Stage 10 quantitative metrics.
2. **`P_00009_1_RIGHT_MLO`** (Malignant Mass, MLO View, 5266x2596): Irregular spiculated mass boundary in dense fibroglandular parenchyma.
3. **`P_00007_1_LEFT_CC`** (Benign Calcification, CC View, 5356x2986): Punctate microcalcification cluster preservation.
4. **`P_00299_1_LEFT_MLO`** (Malignant Calcification, MLO View, 6301x3736): High-resolution pleomorphic microcalcification examination.

## 4. Key Visual and Scientific Observations

### A. Contrast Enhancement
- **Global Histogram Equalization (HistEq)** heavily expands dynamic range but causes severe noise amplification in dark air backgrounds and saturation in dense fibroglandular tissue.
- **CLAHE (clip=2.0, tileGrid=8x8)** adaptively enhances local glandular contrast while effectively limiting background noise amplification.

### B. Sharpening & Edge Enhancement
- **Unsharp Masking (sigma=1.0, amount=1.0)** enhances fine structural edges. When combined with Global HistEq, it severely compounds graininess and noise. When combined with CLAHE, it sharpens parenchymal borders without excessive saturation.

### C. Synthetic Noise Models
- **Gaussian**: Uniform additive sensor noise.
- **Salt & Pepper**: Severe impulse dropout spikes.
- **Speckle**: Signal-dependent multiplicative noise that spares the background air.
- **Poisson & Mixed Poisson-Gaussian**: Quantum photon counting shot noise causing pronounced tissue degradation.

### D. Denoising Algorithms
- **Median / Adaptive Median**: Visually superior suppression of Salt & Pepper impulse noise (PSNR 49.6–52.8 dB).
- **Bilateral & Non-Local Means**: Strong structural edge preservation along fibroglandular boundaries.
- **Anscombe-Wiener**: Effective variance stabilization for quantum Poisson shot degradation.
- **Gaussian Smoothing**: Effective thermal noise reduction, but causes indiscriminate blurring of fine microcalcifications.

## 5. Triangulation: Visual Appearance vs Stage 10 Metrics
- **Corroborations**: Quantitative PSNR and SSIM heavily agree with visual impulse removal by median filters.
- **Discrepancies**: Whole-image metrics overestimate restoration quality because the large, untextured black air background inflates PSNR by 1.5–2.5 dB relative to Breast Foreground-Only metrics.
- **Comparison C Distinctiveness**: Noisy vs Denoised metrics measure filtering magnitude, not ground-truth restoration accuracy.
- **CNR Mask Exclusion**: CNR is unavailable (NaN) for all 640 records because cropped ROI masks do not align with full mammograms. Misleading zero values were strictly avoided.

## 6. Research Limitations
> [!NOTE]
> 1. **Pilot Sample Size**: Stage 10 metrics are based on N=640 evaluations across all 16 pilot mammograms. Findings reflect comprehensive pilot restoration benchmarks.
> 2. **Synthetic Perturbations**: Controlled synthetic noise may differ from raw analog film grain or physical sensor noise.
> 3. **Absence of Clinical Efficacy Claims**: Image sharpness or high PSNR does not prove preserved diagnostic performance or improved cancer detection.
> 4. **No Winning Strategy Declared**: Final preprocessing strategy selection belongs to Stage 12.

---
*Report generated automatically by Stage 11 Preprocessing Visualization & Comparison Module.*