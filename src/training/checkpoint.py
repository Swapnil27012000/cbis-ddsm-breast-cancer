"""Model checkpointing and state persistence."""
import os
import torch

class CheckpointManager:
    """Handles saving and loading of model weights and optimizer states."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, filename: str = "checkpoint.pt"):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }, path)

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, filename: str = "best_model.pt"):
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
