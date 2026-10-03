"""NumPy and PyTorch tensor conversion helpers."""
import torch
import numpy as np
from typing import Union

def to_tensor(
    arr: np.ndarray,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert numpy array (H, W) or (H, W, C) to (1, H, W) or (C, H, W) PyTorch Tensor."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    
    if len(arr.shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)
    elif len(arr.shape) == 3 and arr.shape[-1] in (1, 3):
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        tensor = torch.from_numpy(arr)

    return tensor.to(device=device, dtype=dtype)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch Tensor on any device to CPU NumPy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    t = tensor.detach().cpu()
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0).squeeze()
    return t.numpy()
