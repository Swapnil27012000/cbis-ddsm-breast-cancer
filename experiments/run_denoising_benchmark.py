"""Comparative benchmark runner for noise simulation and medical image denoising.

Delegates to src.experiments.denoising_experiment.run_denoising_experiment.
"""
import sys
from src.experiments.denoising_experiment import main

if __name__ == "__main__":
    main()
