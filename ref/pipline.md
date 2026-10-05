                 CBIS-DDSM
                     │
                     ▼
              Data validation
                     │
                     ▼
             Image-type mapping
                     │
          ┌──────────┴───────────┐
          │                      │
    Full mammogram         ROI / cropped
          │                      │
          ▼                      ▼
      Grayscale              Grayscale
          │                      │
          ▼                      ▼
   Breast/background       ROI alignment
      handling                   │
          │                      ▼
          ▼                 ROI extraction
      Normalize                  │
          │                      │
          └──────────┬───────────┘
                     ▼
             Contrast experiment
              ├── None
              ├── HE
              └── CLAHE
                     │
                     ▼
          Sharpening experiment
              ├── None
              └── Unsharp
                     │
                     ▼
              CLEAN REFERENCE
                     │
          ┌──────────┴───────────┐
          │                      │
      Clean image          Artificial noise
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                 Gaussian     S&P         Speckle
                    │
                    ├── Poisson
                    └── Mixed PG
                                 │
                                 ▼
                         8 Denoising Methods
                                 │
                                 ▼
                      Quality Metrics
                                 │
                                 ▼
                    Best/selected configurations
                                 │
                                 ▼
                         ResNet-50
                                 │
                                 ▼
                     Benign / Malignant
                                 │
                                 ▼
                   Final Classification Metrics