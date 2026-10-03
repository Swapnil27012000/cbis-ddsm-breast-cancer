"""Model evaluation and benchmarking utilities for CBIS-DDSM."""

__all__ = [
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "visualize_mammogram_predictions",
    "run_denoising_visualization",
]


def __getattr__(name: str):
    if name == "compute_classification_metrics":
        from .classification_metrics import compute_classification_metrics
        return compute_classification_metrics
    elif name == "plot_confusion_matrix":
        from .confusion_matrix import plot_confusion_matrix
        return plot_confusion_matrix
    elif name == "plot_roc_curve":
        from .roc_curve import plot_roc_curve
        return plot_roc_curve
    elif name == "visualize_mammogram_predictions":
        from .visualization import visualize_mammogram_predictions
        return visualize_mammogram_predictions
    elif name == "run_denoising_visualization":
        from .denoising_visualization import run_denoising_visualization
        return run_denoising_visualization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
