"""Preprocessing modules and pipeline for CBIS-DDSM mammography."""

__all__ = [
    "validate_mammogram",
    "robust_min_max_normalize",
    "min_max_normalize",
    "z_score_normalize",
    "segment_breast",
    "remove_pectoral_muscle",
    "crop_to_roi",
    "apply_clahe",
    "apply_histogram_equalization",
    "stretch_contrast",
    "run_contrast_stretching",
    "unsharp_mask",
    "apply_unsharp_mask",
    "apply_laplacian_sharpening",
    "run_image_sharpening",
    "run_basic_preprocessing",
    "validate_and_visualize_samples",
    "build_final_preprocessing_metadata",
    "run_integrity_checks",
    "generate_final_processing_report",
    "run_final_preprocessing_stage",
    "run_stage4_image_quality_control",
    "audit_single_image",
    "run_baseline_preprocessing_pilot",
    "run_baseline_preprocessing",
    "load_full_dataset_samples",
    "run_baseline_visualization",
    "run_contrast_enhancement",
    "run_contrast_visualization",
    "load_contrast_config",
    "compute_image_statistics",
    "run_image_sharpening_stage7",
    "run_sharpening_visualization",
    "load_sharpening_config",
    "load_stage6_inputs",
    "compute_edge_statistics",
    "compute_difference_metrics",
    "run_stage8_pilot",
    "load_stage8_config",
    "derive_deterministic_seed",
    "load_clean_image",
    "calculate_noise_statistics",
    "save_noisy_image",
    "validate_noisy_image",
    "generate_noise_visualizations",
]

def __getattr__(name: str):
    if name in (
        "run_stage8_pilot",
        "load_stage8_config",
        "derive_deterministic_seed",
        "load_clean_image",
        "calculate_noise_statistics",
        "save_noisy_image",
        "validate_noisy_image",
        "generate_noise_visualizations",
    ):
        from . import noise_experiment
        return getattr(noise_experiment, name)

    if name in ("run_baseline_preprocessing_pilot", "run_baseline_preprocessing", "load_full_dataset_samples"):
        from . import baseline_preprocessing
        return getattr(baseline_preprocessing, name)
    elif name in ("run_contrast_enhancement", "load_contrast_config"):
        from . import contrast_enhancement
        return getattr(contrast_enhancement, name)
    elif name == "run_contrast_visualization":
        from . import contrast_visualization
        return getattr(contrast_visualization, name)
    elif name == "run_baseline_visualization":
        from . import baseline_visualization
        return getattr(baseline_visualization, name)
    elif name == "run_equivalence_verification":
        from . import verify_optimization_equivalence
        return getattr(verify_optimization_equivalence, name)
    elif name in ("run_stage4_image_quality_control", "audit_single_image"):
        from . import image_quality
        return getattr(image_quality, name)
    elif name == "validate_mammogram":
        from .validation import validate_mammogram
        return validate_mammogram
    elif name in ("robust_min_max_normalize", "min_max_normalize", "z_score_normalize"):
        from . import normalization
        return getattr(normalization, name)
    elif name in ("segment_breast", "remove_pectoral_muscle", "crop_to_roi"):
        from . import roi_processing
        return getattr(roi_processing, name)
    elif name in ("apply_clahe", "apply_histogram_equalization", "stretch_contrast", "run_contrast_stretching"):
        from . import contrast
        return getattr(contrast, name)
    elif name in (
        "unsharp_mask",
        "apply_unsharp_mask",
        "apply_laplacian_sharpening",
        "run_image_sharpening",
        "run_image_sharpening_stage7",
        "load_sharpening_config",
        "load_stage6_inputs",
        "compute_image_statistics",
        "compute_edge_statistics",
        "compute_difference_metrics",
    ):
        from . import sharpening
        return getattr(sharpening, name)
    elif name == "run_sharpening_visualization":
        from . import sharpening_visualization
        return getattr(sharpening_visualization, name)
    elif name == "run_basic_preprocessing":
        from .basic_preprocessing import run_basic_preprocessing
        return run_basic_preprocessing
    elif name == "validate_and_visualize_samples":
        from .visualize import validate_and_visualize_samples
        return validate_and_visualize_samples
    elif name in (
        "build_final_preprocessing_metadata",
        "run_integrity_checks",
        "generate_final_processing_report",
        "run_final_preprocessing_stage",
    ):
        from . import finalize_preprocessing
        return getattr(finalize_preprocessing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
