"""Batch processing helper on GPU."""
import torch
from typing import Callable, List

class BatchProcessor:
    """Applies a GPU tensor function in batches to avoid out-of-memory errors."""

    def __init__(self, batch_size: int = 16, device: torch.device = None):
        self.batch_size = batch_size
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    def process(self, items: List[torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i in range(0, len(items), self.batch_size):
            batch = torch.stack(items[i:i + self.batch_size]).to(self.device)
            with torch.no_grad():
                res = func(batch)
            outputs.extend([t.cpu() for t in res])
        return outputs
