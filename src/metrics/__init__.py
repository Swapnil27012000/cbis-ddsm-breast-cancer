"""Image quality evaluation metrics for CBIS-DDSM breast cancer research."""

__all__ = [
    "compute_mse",
    "compute_psnr",
    "compute_ssim",
    "compute_snr",
    "compute_cnr",
    "extract_roi_and_background_pixels",
    "compute_cii",
    "compute_contrast",
    "compute_entropy",
]


def __getattr__(name: str):
    if name == "compute_mse":
        from .mse import compute_mse
        return compute_mse
    elif name == "compute_psnr":
        from .psnr import compute_psnr
        return compute_psnr
    elif name == "compute_ssim":
        from .ssim import compute_ssim
        return compute_ssim
    elif name == "compute_snr":
        from .snr import compute_snr
        return compute_snr
    elif name in ("compute_cnr", "extract_roi_and_background_pixels"):
        from . import cnr
        return getattr(cnr, name)
    elif name in ("compute_cii", "compute_contrast"):
        from . import cii
        return getattr(cii, name)
    elif name == "compute_entropy":
        from .entropy import compute_entropy
        return compute_entropy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
