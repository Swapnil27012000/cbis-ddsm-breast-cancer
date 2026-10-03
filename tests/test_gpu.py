"""Unit tests for GPU device detection and utility functions."""
import torch
import pytest
from src.gpu.device import get_device, get_device_info
from src.gpu.gpu_utils import empty_cache, clean_gpu_memory, get_memory_stats

def test_get_device_returns_torch_device():
    device = get_device(0)
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "cpu")

def test_get_device_info_structure():
    info = get_device_info(0)
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "device" in info
    assert "gpu_name" in info
    assert "cuda_version" in info
    assert "total_memory_gb" in info
    assert isinstance(info["cuda_available"], bool)

def test_memory_cleanup_safe():
    # Must not raise on CPU or GPU
    empty_cache()
    stats = clean_gpu_memory(0)
    assert isinstance(stats, dict)
    mem_stats = get_memory_stats(0)
    assert isinstance(mem_stats, dict)
    assert "allocated_mb" in mem_stats
