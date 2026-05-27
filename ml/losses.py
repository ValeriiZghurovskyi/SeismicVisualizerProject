"""BCE + Dice combined loss for binary segmentation.

Dice handles class imbalance (horizons/faults are thin features).
BCE provides stable gradients at the start of training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCEWithLogitsLoss and Dice loss.

    Args:
        bce_weight: Weight for BCE term.
        dice_weight: Weight for Dice term.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self._bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self._bce(logits, targets) + self.dice_weight * dice_loss(logits, targets)
