# CBIS-DDSM Stage 12 — Evidence-Based Preprocessing Strategy Selection

> **Research Phase**: Stage 12 — Preprocessing Strategy Selection & Evidence-Based Shortlist  
> **Author**: MSc IT (Artificial Intelligence) Candidate  
> **Authoritative Upstream Evidence**: Stages 5–11 (3,568 baseline, 48 contrast, 96 sharpening, 640 denoising records)  
> **Core Mandate**: Defer definitive single-winner declaration until downstream empirical classification (Stage 14+).  

## 1. Executive Summary & Selection Policy

- **Status**: **SHORTLIST ESTABLISHED — NO PREMATURE WINNER DECLARED**
- **Evidence Foundation**: Comprehensive audit of Stages 5 through 11 outputs.
- **Scientific Guiding Principle**: Preprocessing efficacy in medical AI cannot be inferred from synthetic metrics alone:
  1. High PSNR or SSIM under synthetic noise benchmarks (Stage 10) confirms algorithmic restoration fidelity, **not** clinical diagnostic superiority.
  2. Elevated entropy and gradient magnitudes from contrast/sharpening filters confirm high-frequency boost, but also elevate background sensor noise.
  3. CNR remains unmeasurable (`NaN`) across the full mammogram corpus due to unaligned lesion bounding boxes in CBIS-DDSM.
  4. **Empirical Grounding**: Multiple defensible candidates are preserved alongside the mandatory Baseline Control (C1) to enable rigorous, controlled classification experiments in Stage 14+.

## 2. Recommended Preprocessing Shortlist for Stage 13

| Candidate ID | Strategy Name | Family | Key Metrics (Stage 6/7/10) | Primary Justification | Key Clinical / Technical Risks |
|---|---|---|---|---|---|
| **C1_BASELINE** | **Baseline Preprocessing (Control)** | Baseline | Entropy: 4.6143 b<br>Grad: 9.45<br>Edges: 10.8% | Provides an uncorrupted benchmark for deep learning feature extraction. Guarantees that classification models do not learn preprocessing-induced artifacts. | Dense glandular regions (BI-RADS density C/D) exhibit lower local contrast, potentially challenging early-stage feature extraction in shallow networks. |
| **C2_CLAHE** | **Contrast Limited Adaptive Histogram Equalization (CLAHE)** | Contrast Enhancement | Entropy: 4.9258 b<br>Grad: 16.08<br>Edges: 26.44% | Highest entropy (4.93 bits) among contrast methods; boosts subtle border visibility in dense tissue, which may facilitate mass margin discrimination in CNNs. | May subtly elevate high-frequency sensor noise in low-density adipose regions; tile boundaries could theoretically produce microscopic gradient discontinuities. |
| **C3_UNSHARP_MASK** | **High-Frequency Sharpening (Unsharp Masking)** | Edge Enhancement | Entropy: 4.7529 b<br>Grad: 14.02<br>Edges: 21.98% | Accentuates high-frequency spatial transitions critical for microcalcification cluster and subtle spiculated malignant lesion detection. | Indiscriminate high-frequency gain sharpens background quantum mottle and scanner noise, which may lead CNN feature extractors to attend to spurious noise spikes. |
| **C4_CLAHE_UNSHARP** | **Compound Enhancement (CLAHE + Unsharp Masking)** | Compound Enhancement | Entropy: 4.9748 b<br>Grad: 24.25<br>Edges: 36.15% | Maximizes visual conspicuity by simultaneously expanding local dynamic range and sharpening fine edges. | Compounding risk: CLAHE local noise enhancement is subsequently sharpened by unsharp masking. Highest mean absolute change from baseline (1.32 intensity levels; max diff 111.0). |
| **C5_MEDIAN_DENOISING** | **Non-Linear Denoising (Median Filtering, 3x3)** | Denoising | Entropy: 4.7998 b<br>Grad: 0.0753<br>Edges: nan%<br>PSNR: 36.06 dB<br>SSIM: 0.8037 | Demonstrated top general restoration (36.06 dB PSNR, 0.8037 SSIM) in Stage 10 benchmark; serves as a controlled candidate to evaluate whether noise suppression helps CNN generalization. | Critical Clinical Risk: Denoising raw, uncorrupted clinical scans risks blurring subtle, punctate microcalcifications (< 0.2 mm) and fine spicules, potentially destroying life-saving diagnostic cues. Stage 10 evaluations evaluated synthetic noise, not native scanner noise. |

## 3. Formally Discarded / Excluded Methods

| Discard ID | Strategy Name | Family | Empirical Basis | Rationale for Exclusion |
|---|---|---|---|---|
| **DISC_01_HISTEQ** | Global Histogram Equalization (HistEq) | Contrast Enhancement | Stage 6 Contrast Experiment & Stage 11 Visual Panels | Severely expands global dynamic range across non-breast background areas, amplifying sensor noise. Causes excessive saturation and whiteout in dense fibroglandular parenchyma, obliterating fine internal mass structures. Entropy decreased from 4.6143 to 4.5282 bits. |
| **DISC_02_GAUSSIAN_BLUR** | Gaussian Spatial Smoothing (Denoising) | Denoising | Stage 10 Quantitative Summary & Stage 11 Magnified Crops | Indiscriminate low-pass spatial filtering washes out fine structural details and smooths out microcalcifications. Mean gradient dropped to 0.0642 (lowest of all methods). Inferior SSIM (0.7521) to median filtering. |
| **DISC_03_NLM** | Non-Local Means (NLM Denoising) | Denoising | Stage 10 Quantitative Summary & Stage 9 Runtime Logs | Extreme computational complexity (~10-15 seconds per high-resolution mammogram, requiring > 10 hours for 3,568 images). Low restoration fidelity under Comparison B (PSNR 27.88 dB, SSIM 0.6349), indicating aggressive alteration of native tissue. |
| **DISC_04_ADAPTIVE_MEDIAN** | Adaptive Median Filtering | Denoising | Stage 10 Summary by Method & Noise Type | Highly specialized for salt-and-pepper noise (PSNR 52.77 dB), but inferior to standard median filtering on continuous Gaussian, Poisson, and Speckle noise models (overall PSNR 33.82 dB vs 36.06 dB for standard median). |
| **DISC_05_UNTESTED_CHAINS** | Arbitrary Multi-Stage Pipelines (e.g. Denoise -> HistEq -> Sharpen) | Arbitrary Chains | Strict Experimental Boundary Constraints | Scientific boundary: Chaining operations that were never tested in Stages 5–11 violates reproducible research protocols. Only factorial combinations explicitly tested in Stage 7 (CLAHE + Unsharp Mask) are eligible for shortlisting. |

## 4. Evidence Synthesis & Critical Gaps

### A. Contrast Enhancement (Stage 6)
- **Global Histogram Equalization** is ruled out due to non-breast background noise blowout and extreme over-saturation in glandular tissue.
- **CLAHE (clip=2.0, tileGrid=8x8)** is the sole defensible contrast candidate, elevating entropy to 4.93 bits without global dynamic range clipping.

### B. Sharpening (Stage 7)
- **Unsharp Masking** effectively enhances high-frequency microcalcification boundaries and spicules (+103% strong edge pixels).
- **Compound CLAHE + Unsharp Masking** achieves maximum edge conspicuity (gradient 24.25), but carries compounded noise risks.

### C. Denoising Benchmark (Stages 8–10)
- **Median Filtering (3x3)** outperformed all 7 competing denoisers under synthetic noise (36.06 dB PSNR, 0.8037 SSIM).
- **Critical Nuance**: Clinical CBIS-DDSM images do not exhibit synthetic noise; filtering native scans risks blurring subtle microcalcifications. Retained purely as an exploratory benchmark.

### D. Identified Evidence Gaps
1. **Lesion-Specific CNR**: Mask registration limitations in CBIS-DDSM prevent CNR calculation on full mammograms (`NaN`).
2. **Classifier Feature Activation**: Whether CNN attention heatmaps (e.g. Grad-CAM) benefit from pre-sharpened borders or are distracted by amplified noise is unknown prior to Stage 14.
3. **Pilot Scaling**: 16 pilot cases provide strong directional guidance, but the full 3,568-case corpus in Stage 13 is required to confirm population-level robustness.

## 5. Transition Plan for Stage 13

1. **Patient-Level Leakage Prevention**: In Stage 13, partitioning across Train, Validation, and Test sets must strictly occur at the unique `patient_id` level.
2. **Core Benchmark Dataset**: Stage 13 will prepare the full 3,568-image dataset using **Candidate C1 (Baseline Control)** as the primary gold-standard dataset.
3. **Candidate Branches**: Candidate configurations (C2 CLAHE, C3 Unsharp Mask, C4 CLAHE+Unsharp, and C5 Median Denoising) will be generated as controlled parallel branches to enable empirical comparative training in Stages 14–17.
