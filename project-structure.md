breast-cancer-detection-ai-model-project/
│
├── .env
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements-gpu.txt
│
├── config/
│   ├── config.yaml
│   ├── gpu_config.yaml
│   └── preprocessing_config.yaml
│
├── data/
│   │
│   ├── raw/                                  # NEVER modify original dataset
│   │   └── CBIS_DDSM/
│   │       ├── csv/
│   │       │   ├── mass_case_description_train_set.csv
│   │       │   ├── mass_case_description_test_set.csv
│   │       │   ├── calc_case_description_train_set.csv
│   │       │   ├── calc_case_description_test_set.csv
│   │       │   ├── dicom_info.csv
│   │       │   └── meta.csv
│   │       │
│   │       └── jpeg/
│   │           └── <CBIS-DDSM image folders>
│   │
│   ├── metadata/
│   │   ├── CBIS_DDSM_master_metadata.csv
│   │   ├── image_validation_report.csv
│   │   ├── dataset_statistics.csv
│   │   └── patient_split_report.csv
│   │
│   └── processed/
│       │
│       ├── validated/
│       │
│       ├── normalized/
│       │
│       ├── roi/
│       │
│       ├── cropped/
│       │
│       ├── contrast/
│       │
│       ├── sharpened/
│       │
│       ├── noisy/
│       │   ├── gaussian/
│       │   ├── salt_pepper/
│       │   ├── speckle/
│       │   ├── poisson/
│       │   └── mixed_poisson_gaussian/
│       │
│       ├── denoised/
│       │   ├── median/
│       │   ├── gaussian/
│       │   ├── wiener/
│       │   ├── bilateral/
│       │   ├── non_local_means/
│       │   ├── anscombe_wiener/
│       │   ├── adaptive_median/
│       │   └── kuan/
│       │
│       └── final/
│           ├── benign/
│           └── malignant/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── csv_loader.py
│   │   ├── metadata_builder.py
│   │   ├── image_index.py
│   │   ├── path_resolver.py
│   │   └── dataset_split.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   ├── normalization.py
│   │   ├── roi_processing.py
│   │   ├── contrast.py
│   │   └── sharpening.py
│   │
│   ├── gpu/
│   │   ├── __init__.py
│   │   ├── device.py
│   │   ├── gpu_utils.py
│   │   ├── tensor_utils.py
│   │   ├── batch_processor.py
│   │   └── memory_manager.py
│   │
│   ├── noise/
│   │   ├── __init__.py
│   │   ├── gaussian.py
│   │   ├── salt_pepper.py
│   │   ├── speckle.py
│   │   ├── poisson.py
│   │   └── mixed_poisson_gaussian.py
│   │
│   ├── denoising/
│   │   ├── __init__.py
│   │   ├── median.py
│   │   ├── gaussian.py
│   │   ├── wiener.py
│   │   ├── bilateral.py
│   │   ├── non_local_means.py
│   │   ├── anscombe_wiener.py
│   │   ├── adaptive_median.py
│   │   └── kuan.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── mse.py
│   │   ├── psnr.py
│   │   ├── ssim.py
│   │   ├── snr.py
│   │   ├── cnr.py
│   │   ├── cii.py
│   │   └── entropy.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── cbis_ddsm_dataset.py
│   │   ├── mammogram_dataset.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── resnet.py
│   │   ├── densenet.py
│   │   └── efficientnet.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── validate.py
│   │   ├── loss.py
│   │   ├── scheduler.py
│   │   └── checkpoint.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── classification_metrics.py
│   │   ├── confusion_matrix.py
│   │   ├── roc_curve.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── file_utils.py
│       ├── image_utils.py
│       ├── reproducibility.py
│       └── config_loader.py
│
├── models/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   │
│   ├── exported/
│   │   ├── model.pt
│   │   └── model.onnx
│   │
│   └── logs/
│       ├── training_history.csv
│       └── tensorboard/
│
├── experiments/
│   │
│   ├── denoising/
│   │   ├── experiment_config.yaml
│   │   ├── results.csv
│   │   └── visual_comparisons/
│   │
│   ├── preprocessing/
│   │   ├── results.csv
│   │   └── visual_comparisons/
│   │
│   └── classification/
│       ├── experiment_config.yaml
│       ├── results.csv
│       └── figures/
│
├── results/
│   ├── preprocessing/
│   │   ├── metrics.csv
│   │   ├── summary.csv
│   │   └── comparison_plots/
│   │
│   ├── denoising/
│   │   ├── metrics.csv
│   │   ├── method_comparison.csv
│   │   └── comparison_plots/
│   │
│   └── classification/
│       ├── metrics.csv
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── training_curves.png
│
└── tests/
    ├── test_data_loader.py
    ├── test_metadata.py
    ├── test_preprocessing.py
    ├── test_noise.py
    ├── test_denoising.py
    ├── test_metrics.py
    └── test_model.py