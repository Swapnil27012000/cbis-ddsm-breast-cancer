"""GPU Memory manager context for profiling peak VRAM usage."""
import torch

class CUDAMemoryManager:
    """Context manager for measuring and limiting CUDA memory allocations."""

    def __init__(self, device: torch.device = None):
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.start_mem = 0
        self.peak_mem = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_mem = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_mem = torch.cuda.max_memory_allocated(self.device)
            print(f"[CUDA Memory] Peak Allocated: {self.peak_mem / (1024**2):.2f} MB")
