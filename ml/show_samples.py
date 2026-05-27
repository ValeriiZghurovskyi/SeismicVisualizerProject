"""Show training samples as they are fed to the model.

Usage:
    python -m ml.show_samples
    python -m ml.show_samples --n 8 --output ml/train_samples.png --augment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml.dataset import SeismicSliceDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubes_dir", type=Path, default=Path("data/cubes"))
    parser.add_argument("--label_type", choices=["horizons", "faults"], default="horizons")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--n", type=int, default=6, help="Number of samples to show")
    parser.add_argument("--output", type=Path, default=Path("ml/train_samples.png"))
    parser.add_argument("--augment", action="store_true", help="Apply augmentation")
    args = parser.parse_args()

    ds = SeismicSliceDataset(
        cubes_dir=args.cubes_dir,
        label_type=args.label_type,
        tile_size=args.tile_size,
        augment=args.augment,
    )
    print(f"Dataset size: {len(ds)}")

    # Pick evenly spaced samples across the dataset
    indices = np.linspace(0, len(ds) - 1, args.n, dtype=int)

    n_cols = 3  # seismic+mask overlay | hint | mask
    fig, axes = plt.subplots(args.n, n_cols, figsize=(n_cols * 4, args.n * 4), squeeze=False)

    col_titles = ["Seismic + GT overlay", "Hint map", "Ground truth mask"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for row, idx in enumerate(indices):
        sample = ds[idx]
        meta = ds._index[idx]

        seismic = sample["input"][0].numpy()   # [-1, 1]
        hint    = sample["input"][1].numpy()   # [0, 1]
        mask    = sample["target"][0].numpy()  # {0, 1}

        # Col 0: seismic in gray, horizon overlay in red
        axes[row][0].imshow(seismic, cmap="gray", vmin=-1, vmax=1, aspect="auto")
        if mask.max() > 0:
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask > 0] = [1.0, 0.0, 0.0, 0.7]  # red, semi-transparent
            axes[row][0].imshow(overlay, aspect="auto")
        axes[row][0].set_ylabel(
            f"idx={idx}\naxis={'IL' if meta.axis==0 else 'XL'}  sl={meta.slice_idx}\nentity={meta.entity_id}  px={int(mask.sum())}",
            fontsize=7, rotation=0, labelpad=60, va="center",
        )

        # Col 1: hint map
        axes[row][1].imshow(hint, cmap="hot", vmin=0, vmax=1, aspect="auto")

        # Col 2: binary mask
        axes[row][2].imshow(mask, cmap="gray", vmin=0, vmax=1, aspect="auto")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        f"{args.label_type.capitalize()} training tiles  |  tile={args.tile_size}px  |  augment={args.augment}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
