"""Saves a grid of model predictions during training for visual inspection.

Layout per row (one row = one sample):
  seismic | hint | ground truth | predicted probability | predicted mask (>0.5)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.unet import UNet


def save_sample_grid(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    n_samples: int = 4,
) -> None:
    """Run inference on the first n_samples from loader and save a PNG grid.

    Args:
        model: Trained (or partially trained) UNet in eval mode.
        loader: DataLoader to pull samples from (typically val_loader).
        device: Device to run inference on.
        output_path: Where to write the PNG file.
        n_samples: Number of rows in the grid.
    """
    model.eval()
    inputs, targets, preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"]
            logits = model(x)
            probs = torch.sigmoid(logits).cpu()

            for i in range(len(x)):
                if len(inputs) >= n_samples:
                    break
                inputs.append(x[i].cpu())
                targets.append(y[i])
                preds.append(probs[i])
            if len(inputs) >= n_samples:
                break

    n = len(inputs)
    n_cols = 5  # seismic | hint | GT | probability | binary mask
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 3, n * 3), squeeze=False)
    col_titles = ["Seismic", "Hint", "Ground Truth", "Prob. Map", "Prediction"]

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9)

    for row in range(n):
        seismic = inputs[row][0].numpy()   # [-1, 1]
        hint = inputs[row][1].numpy()      # [0, 1]
        gt = targets[row][0].numpy()       # {0, 1}
        prob = preds[row][0].numpy()       # [0, 1]
        binary = (prob > 0.5).astype(np.float32)

        axes[row][0].imshow(seismic, cmap="gray", vmin=-1, vmax=1, aspect="auto")
        axes[row][1].imshow(hint, cmap="hot", vmin=0, vmax=1, aspect="auto")
        axes[row][2].imshow(gt, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[row][3].imshow(prob, cmap="plasma", vmin=0, vmax=1, aspect="auto")
        axes[row][4].imshow(binary, cmap="gray", vmin=0, vmax=1, aspect="auto")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
