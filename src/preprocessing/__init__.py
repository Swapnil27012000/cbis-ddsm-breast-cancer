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
]

def __getattr__(name: str):
    if name in ("run_stage4_image_quality_control", "audit_single_image"):
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
    elif name in ("unsharp_mask", "apply_unsharp_mask", "apply_laplacian_sharpening", "run_image_sharpening"):
        from . import sharpening
        return getattr(sharpening, name)
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
