"""GPU infrastructure and acceleration utilities for CBIS-DDSM."""

__all__ = [
    "get_device",
    "get_device_info",
    "print_device_summary",
    "empty_cache",
    "clean_gpu_memory",
    "get_memory_stats",
    "set_reproducible_gpu",
]

def __getattr__(name: str):
    if name in ("get_device", "get_device_info", "print_device_summary"):
        from . import device
        return getattr(device, name)
    elif name in ("empty_cache", "clean_gpu_memory", "get_memory_stats", "set_reproducible_gpu"):
        from . import gpu_utils
        return getattr(gpu_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
