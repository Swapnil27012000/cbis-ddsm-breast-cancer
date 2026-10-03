"""Medical image denoising benchmark filters for CBIS-DDSM mammography."""

__all__ = [
    "denoise_median",
    "denoise_gaussian",
    "denoise_wiener",
    "denoise_bilateral",
    "denoise_nlm",
    "denoise_anscombe_wiener",
    "denoise_adaptive_median",
    "denoise_kuan",
    "run_denoising_pipeline",
]


def __getattr__(name: str):
    if name == "denoise_median":
        from .median import denoise_median
        return denoise_median
    elif name == "denoise_gaussian":
        from .gaussian import denoise_gaussian
        return denoise_gaussian
    elif name == "denoise_wiener":
        from .wiener import denoise_wiener
        return denoise_wiener
    elif name == "denoise_bilateral":
        from .bilateral import denoise_bilateral
        return denoise_bilateral
    elif name == "denoise_nlm":
        from .non_local_means import denoise_nlm
        return denoise_nlm
    elif name == "denoise_anscombe_wiener":
        from .anscombe_wiener import denoise_anscombe_wiener
        return denoise_anscombe_wiener
    elif name == "denoise_adaptive_median":
        from .adaptive_median import denoise_adaptive_median
        return denoise_adaptive_median
    elif name == "denoise_kuan":
        from .kuan import denoise_kuan
        return denoise_kuan
    elif name == "run_denoising_pipeline":
        from .run_denoising import run_denoising_pipeline
        return run_denoising_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
