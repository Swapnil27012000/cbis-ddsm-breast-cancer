from typing import Dict, Any, Optional
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    TORCH_AVAILABLE = False

def get_device(device_id: int = 0):
    """Return torch.device for CUDA if available, otherwise safely fall back to CPU.

    Args:
        device_id: Target GPU device index (default: 0).

    Returns:
        torch.device instance ('cuda:<device_id>' or 'cpu') or 'cpu' string if torch unavailable.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return torch.device("cpu") if TORCH_AVAILABLE else "cpu"

    num_devices = torch.cuda.device_count()
    if device_id < 0 or device_id >= num_devices:
        device_id = 0

    return torch.device(f"cuda:{device_id}")

def get_device_info(device_id: int = 0) -> Dict[str, Any]:
    """Get hardware details for the active compute device without crashing on CPU-only machines.

    Args:
        device_id: Target GPU device index (default: 0).

    Returns:
        Dictionary containing hardware details:
            - cuda_available (bool)
            - device (str)
            - gpu_name (str)
            - cuda_version (str)
            - device_count (int)
            - total_memory_bytes (int)
            - total_memory_mb (float)
            - total_memory_gb (float)
            - allocated_memory_mb (float)
            - cached_memory_mb (float)
    """
    cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
    info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "device": "cpu",
        "gpu_name": "None (Running on CPU)",
        "cuda_version": "N/A",
        "device_count": 0,
        "total_memory_bytes": 0,
        "total_memory_mb": 0.0,
        "total_memory_gb": 0.0,
        "allocated_memory_mb": 0.0,
        "cached_memory_mb": 0.0,
    }

    if not cuda_available:
        return info

    try:
        num_devices = torch.cuda.device_count()
        info["device_count"] = num_devices
        valid_id = device_id if 0 <= device_id < num_devices else 0
        info["device"] = f"cuda:{valid_id}"
        info["gpu_name"] = torch.cuda.get_device_name(valid_id)
        info["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else "N/A"

        props = torch.cuda.get_device_properties(valid_id)
        total_bytes = props.total_memory
        info["total_memory_bytes"] = total_bytes
        info["total_memory_mb"] = round(total_bytes / (1024 ** 2), 2)
        info["total_memory_gb"] = round(total_bytes / (1024 ** 3), 2)

        info["allocated_memory_mb"] = round(torch.cuda.memory_allocated(valid_id) / (1024 ** 2), 2)
        info["cached_memory_mb"] = round(torch.cuda.memory_reserved(valid_id) / (1024 ** 2), 2)
    except Exception as e:
        info["error"] = str(e)

    return info

def print_device_summary(device_id: int = 0) -> None:
    """Print structured hardware report to stdout for verification."""
    info = get_device_info(device_id)

    print("=" * 50)
    print("CBIS-DDSM GPU Infrastructure Check")
    print("=" * 50)
    print(f"CUDA available : {info['cuda_available']}")
    print(f"Device         : {info['device']}")
    print(f"GPU name       : {info['gpu_name']}")
    print(f"CUDA version   : {info['cuda_version']}")
    if info["cuda_available"]:
        print(f"GPU memory     : {info['total_memory_gb']} GB ({info['total_memory_mb']} MB)")
    else:
        print("GPU memory     : N/A (CPU)")
    print("=" * 50)

if __name__ == "__main__":
    print_device_summary()
