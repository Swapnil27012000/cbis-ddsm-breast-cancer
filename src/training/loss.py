"""Loss functions for class-imbalanced breast cancer classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss to address benign vs malignant class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

def get_loss_function(loss_name: str = "focal", weights: torch.Tensor = None, **kwargs) -> nn.Module:
    """Return requested criterion."""
    if loss_name.lower() == "focal":
        return FocalLoss(gamma=kwargs.get("gamma", 2.0))
    elif loss_name.lower() == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()
