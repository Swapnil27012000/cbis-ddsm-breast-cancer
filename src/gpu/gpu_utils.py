"""CUDA memory management and hardware utility functions."""
import gc
from typing import Dict, Any
import torch

def empty_cache(device_id: int = 0) -> None:
    """Run Python garbage collection and release cached PyTorch CUDA VRAM.

    Safe to call on CPU-only systems (will not crash).
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def clean_gpu_memory(device_id: int = 0) -> Dict[str, float]:
    """Perform aggressive, safe CUDA memory cleanup and return released memory delta.

    Args:
        device_id: Target GPU device index (default: 0).

    Returns:
        Dict with memory before and after cleanup in MB.
    """
    stats = {"before_mb": 0.0, "after_mb": 0.0, "freed_mb": 0.0}

    if not torch.cuda.is_available():
        gc.collect()
        return stats

    try:
        valid_id = device_id if 0 <= device_id < torch.cuda.device_count() else 0
        before = torch.cuda.memory_reserved(valid_id)
        stats["before_mb"] = round(before / (1024 ** 2), 2)

        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        after = torch.cuda.memory_reserved(valid_id)
        stats["after_mb"] = round(after / (1024 ** 2), 2)
        stats["freed_mb"] = round(max(0.0, stats["before_mb"] - stats["after_mb"]), 2)
    except Exception as e:
        stats["error"] = str(e)

    return stats

def get_memory_stats(device_id: int = 0) -> Dict[str, float]:
    """Retrieve current allocated, reserved, and peak CUDA VRAM in MB.

    Safe on CPU (returns zeros without raising an exception).
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        }

    try:
        valid_id = device_id if 0 <= device_id < torch.cuda.device_count() else 0
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(valid_id) / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(valid_id) / (1024 ** 2), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(valid_id) / (1024 ** 2), 2),
        }
    except Exception:
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        }

def set_reproducible_gpu(benchmark: bool = False, deterministic: bool = True) -> None:
    """Configure cuDNN benchmark and deterministic flags for experimental reproducibility.

    Args:
        benchmark: If True, uses cuDNN heuristic to find fastest convolution algorithms.
                   Set to False for exact reproducibility.
        deterministic: If True, forces deterministic convolution algorithms.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
