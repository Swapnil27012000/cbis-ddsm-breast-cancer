"""Artificial noise simulation models for experimental denoising benchmark study.

IMPORTANT RESEARCH DISTINCTION:
--------------------------------
The noise simulated by this module is ARTIFICIALLY GENERATED for an experimental
medical image denoising research benchmark. It does NOT represent naturally occurring
physical noise in the CBIS-DDSM mammography dataset.
"""

__all__ = [
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    "add_speckle_noise",
    "add_poisson_noise",
    "add_mixed_poisson_gaussian_noise",
    "run_noise_generation",
    "run_noise_visualizations",
]

def __getattr__(name: str):
    if name == "add_gaussian_noise":
        from .gaussian import add_gaussian_noise
        return add_gaussian_noise
    elif name == "add_salt_pepper_noise":
        from .salt_pepper import add_salt_pepper_noise
        return add_salt_pepper_noise
    elif name == "add_speckle_noise":
        from .speckle import add_speckle_noise
        return add_speckle_noise
    elif name == "add_poisson_noise":
        from .poisson import add_poisson_noise
        return add_poisson_noise
    elif name == "add_mixed_poisson_gaussian_noise":
        from .mixed_poisson_gaussian import add_mixed_poisson_gaussian_noise
        return add_mixed_poisson_gaussian_noise
    elif name == "run_noise_generation":
        from .generate_noise import run_noise_generation
        return run_noise_generation
    elif name == "run_noise_visualizations":
        from .visualize_noise import run_noise_visualizations
        return run_noise_visualizations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
