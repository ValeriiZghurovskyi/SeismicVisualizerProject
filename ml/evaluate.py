"""Evaluation metrics for trained U-Net models.

Usage:
    python -m ml.evaluate
    python -m ml.evaluate --checkpoint ml/checkpoints/best.pt --label_type faults
    python -m ml.evaluate --tolerance 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader

from ml.config import Config
from ml.dataset import SeismicSliceDataset
from ml.unet import UNet


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum().item()
    union = ((preds + targets) > 0).float().sum().item()
    return intersection / union if union > 0 else 1.0


def f1_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    denom = 2.0 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 1.0


def relaxed_f1_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    tolerance: int = 5,
) -> tuple[float, float, float]:
    """Relaxed precision/recall/F1: a predicted pixel counts as TP if it falls
    within `tolerance` pixels of any ground-truth pixel, and vice-versa.

    Standard approach for thin-feature evaluation (BSDS, seismic horizons).
    """
    preds_np = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy().astype(bool)
    targets_np = targets.squeeze().cpu().numpy().astype(bool)

    if not targets_np.any():
        # No ground truth — skip sample
        return float("nan"), float("nan"), float("nan")

    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    dilated_targets = binary_dilation(targets_np, structure=struct)
    dilated_preds = binary_dilation(preds_np, structure=struct)

    pred_sum = preds_np.sum()
    target_sum = targets_np.sum()

    # relaxed precision: predicted pixels that land within tolerance of GT
    prec = float((preds_np & dilated_targets).sum()) / pred_sum if pred_sum > 0 else 0.0
    # relaxed recall: GT pixels covered by predictions within tolerance
    rec = float((targets_np & dilated_preds).sum()) / target_sum if target_sum > 0 else 0.0

    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate(cfg: Config, checkpoint_path: Path, tolerance: int = 0) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SeismicSliceDataset(
        cubes_dir=cfg.data.cubes_dir,
        label_type=cfg.data.label_type,
        tile_size=cfg.data.tile_size,
        augment=False,
        min_labels=cfg.data.min_labels_per_slice,
        min_tile_labels=cfg.data.min_tile_labels,
        n_seeds=cfg.data.n_seeds,
        seed_sigma=cfg.data.seed_sigma,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        depth=cfg.model.depth,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    iou_list: list[float] = []
    f1_list: list[float] = []
    rel_prec_list: list[float] = []
    rel_rec_list: list[float] = []
    rel_f1_list: list[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            logits = model(x)
            for i in range(len(logits)):
                lg = logits[i : i + 1]
                tg = y[i : i + 1]
                iou_list.append(iou_score(lg, tg))
                f1_list.append(f1_score(lg, tg))
                if tolerance > 0:
                    prec, rec, rf1 = relaxed_f1_score(lg, tg, tolerance=tolerance)
                    if not np.isnan(rf1):
                        rel_prec_list.append(prec)
                        rel_rec_list.append(rec)
                        rel_f1_list.append(rf1)

    results: dict[str, float] = {
        "n_samples": len(iou_list),
        "mean_iou": float(np.mean(iou_list)),
        "std_iou": float(np.std(iou_list)),
        "mean_f1": float(np.mean(f1_list)),
        "std_f1": float(np.std(f1_list)),
    }

    print(f"Samples      : {results['n_samples']}")
    print(f"Mean IoU     : {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
    print(f"Mean F1      : {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")

    if tolerance > 0 and rel_f1_list:
        results.update({
            "tolerance_px": float(tolerance),
            "relaxed_precision": float(np.mean(rel_prec_list)),
            "relaxed_recall": float(np.mean(rel_rec_list)),
            "relaxed_f1": float(np.mean(rel_f1_list)),
            "std_relaxed_f1": float(np.std(rel_f1_list)),
        })
        print(f"\n--- Relaxed evaluation (tolerance={tolerance}px) ---")
        print(f"Relaxed Precision: {results['relaxed_precision']:.4f}")
        print(f"Relaxed Recall   : {results['relaxed_recall']:.4f}")
        print(f"Relaxed F1       : {results['relaxed_f1']:.4f} ± {results['std_relaxed_f1']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate seismic U-Net")
    parser.add_argument("--label_type", choices=["horizons", "faults"], default="horizons")
    parser.add_argument("--cubes_dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("ml/checkpoints/best.pt"))
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Pixel tolerance for relaxed precision/recall/F1 (0 = strict, 5 = relaxed ±5px).",
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.data.label_type = args.label_type
    if args.cubes_dir:
        cfg.data.cubes_dir = args.cubes_dir
    if args.tile_size:
        cfg.data.tile_size = args.tile_size

    evaluate(cfg, args.checkpoint, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
