"""Plot training curves from a training_log.csv produced by ml.train.

Usage:
    python -m ml.plot_training
    python -m ml.plot_training --log ml/checkpoints/training_log.csv
    python -m ml.plot_training --log ml/checkpoints/training_log.csv --out curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _read_log(path: Path) -> dict[str, list[float]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if headers is None:
                headers = line.split(",")
                continue
            rows.append(dict(zip(headers, line.split(","))))

    if not rows or headers is None:
        raise ValueError(f"No epoch data found in {path}")

    result: dict[str, list[float]] = {h: [] for h in headers}
    for row in rows:
        for k, v in row.items():
            result[k].append(float(v))
    return result


def _read_meta(path: Path) -> list[str]:
    meta = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                meta.append(line[1:].strip())
            else:
                break
    return meta


def plot(log_path: Path, out_path: Path | None = None) -> None:
    data = _read_log(log_path)
    meta = _read_meta(log_path)

    epochs = [int(e) for e in data["epoch"]]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    lr = data["lr"]
    epoch_time = data["epoch_time_s"]
    cumul_time = data["cumul_time_s"]
    is_best = data["is_best"]

    best_indices = [i for i, b in enumerate(is_best) if b]
    best_epoch = epochs[best_indices[-1]] if best_indices else None
    best_val = val_loss[best_indices[-1]] if best_indices else None

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    title = f"Training — {log_path.parent.name}"
    if best_epoch is not None:
        title += f"   |   best val={best_val:.4f} @ ep {best_epoch}"
    fig.suptitle(title, fontsize=12)

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, train_loss, label="train", linewidth=1.5)
    ax.plot(epochs, val_loss, label="val", linewidth=1.5)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
                   label=f"best ep {best_epoch}")
        ax.scatter([best_epoch], [best_val], color="green", zorder=5, s=40)
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    # --- LR ---
    ax = axes[1]
    ax.plot(epochs, lr, color="darkorange", linewidth=1.5)
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    # --- Epoch time ---
    ax = axes[2]
    ax.bar(epochs, epoch_time, color="steelblue", alpha=0.7, width=0.8, label="epoch time (s)")
    ax2 = ax.twinx()
    cumul_min = [t / 60 for t in cumul_time]
    ax2.plot(epochs, cumul_min, color="crimson", linewidth=1.5, label="cumulative (min)")
    ax2.set_ylabel("Cumulative time (min)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Training Time per Epoch")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # Metadata annotation
    meta_text = "\n".join(m for m in meta if m.startswith(("data:", "train_cfg:", "model:")))
    if meta_text:
        fig.text(0.01, 0.01, meta_text, fontsize=7, va="bottom",
                 family="monospace", color="gray")

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if out_path is None:
        out_path = log_path.parent / "training_curves.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Print summary table
    print(f"\n{'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'LR':>10}  {'Time(s)':>8}  Best")
    print("-" * 54)
    for i, ep in enumerate(epochs):
        marker = " <--" if is_best[i] else ""
        print(f"{ep:>6}  {train_loss[i]:>8.4f}  {val_loss[i]:>8.4f}  "
              f"{lr[i]:>10.2e}  {epoch_time[i]:>8.0f}{marker}")
    total_min = cumul_time[-1] / 60 if cumul_time else 0
    print(f"\nTotal: {total_min:.1f} min   Best val: {best_val:.4f} @ epoch {best_epoch}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from training_log.csv")
    parser.add_argument(
        "--log", type=Path,
        default=Path("ml/checkpoints/training_log.csv"),
        help="Path to training_log.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path (default: <log_dir>/training_curves.png)",
    )
    args = parser.parse_args()
    plot(args.log, args.out)


if __name__ == "__main__":
    main()
