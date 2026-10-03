"""Learning rate schedulers."""
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

def get_lr_scheduler(optimizer, scheduler_type: str = "cosine", epochs: int = 50, **kwargs):
    """Initialize learning rate scheduler."""
    if scheduler_type.lower() == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
