"""Benchmark experiment modules for CBIS-DDSM."""

__all__ = ["run_denoising_experiment"]


def __getattr__(name: str):
    if name == "run_denoising_experiment":
        from .denoising_experiment import run_denoising_experiment
        return run_denoising_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
