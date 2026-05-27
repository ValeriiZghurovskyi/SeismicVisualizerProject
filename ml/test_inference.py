"""Quick inference test on the training cube.

Picks N labeled inline slices from delf, runs the ONNX model with a seed hint,
computes IoU vs ground truth, and saves a multi-panel PNG.

Usage:
    python -m ml.test_inference
    python -m ml.test_inference --n_slices 6 --entity_id 1 --output results/test.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------
# Constants (must match training / OnnxTracker)
# --------------------------------------------------------------------------
PAD_MULTIPLE = 16
HINT_SIGMA = 8.0   # use training sigma here — we have ground-truth seeds
THRESHOLD = 0.5
N_SEEDS = 3        # seed points sampled from ground-truth labels


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def pad_to_multiple(arr: np.ndarray, factor: int = PAD_MULTIPLE):
    h, w = arr.shape
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return np.pad(arr, ((0, ph), (0, pw)), mode="reflect"), h, w


def make_hint(mask_2d: np.ndarray, n_seeds: int, sigma: float) -> np.ndarray:
    coords = np.argwhere(mask_2d > 0)
    hint = np.zeros_like(mask_2d, dtype=np.float32)
    if len(coords) == 0:
        return hint
    chosen = coords[np.random.choice(len(coords), min(n_seeds, len(coords)), replace=False)]
    for r, c in chosen:
        hint[r, c] = 1.0
    hint = gaussian_filter(hint, sigma=sigma)
    mx = hint.max()
    if mx > 0:
        hint /= mx
    return hint


def run_inference(session, seismic_2d: np.ndarray, hint_2d: np.ndarray):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    seismic_norm = (seismic_2d.astype(np.float32) - 128.0) / 128.0
    s_pad, h_orig, w_orig = pad_to_multiple(seismic_norm)
    hint_pad, _, _ = pad_to_multiple(hint_2d.astype(np.float32))

    x = np.stack([s_pad, hint_pad], axis=0)[np.newaxis]  # (1, 2, H_pad, W_pad)
    logits = session.run([output_name], {input_name: x})[0]
    probs = 1.0 / (1.0 + np.exp(-logits[0, 0].astype(np.float64)))
    return probs[:h_orig, :w_orig].astype(np.float32)


def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = THRESHOLD) -> float:
    p = pred > threshold
    g = gt > 0
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else 1.0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seismic",    type=Path, default=Path("data/cubes/delf_seismic.npz"))
    parser.add_argument("--horizons",   type=Path, default=Path("data/cubes/delf_horizons.npz"))
    parser.add_argument("--model",      type=Path, default=Path("models/horizon_tracker.onnx"))
    parser.add_argument("--entity_id",  type=int,  default=1)
    parser.add_argument("--n_slices",   type=int,  default=6,
                        help="Number of slices to visualise (evenly spaced)")
    parser.add_argument("--axis",       type=int,  default=0,
                        help="0=inline, 1=crossline")
    parser.add_argument("--output",     type=Path, default=Path("results/inference_test.png"))
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading seismic : {args.seismic}")
    seismic = np.load(args.seismic)["cube"].astype(np.uint8)
    print(f"Loading horizons: {args.horizons}")
    labels  = np.load(args.horizons)["labels"].astype(np.uint8)
    print(f"Cube shape: {seismic.shape}")

    print(f"Loading model   : {args.model}")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    # Find slices that have enough labels for entity_id
    entity_mask_3d = (labels == args.entity_id)
    labeled_slices = [
        i for i in range(seismic.shape[args.axis])
        if np.take(entity_mask_3d, i, axis=args.axis).sum() >= 10
    ]
    if not labeled_slices:
        print(f"No labeled slices found for entity_id={args.entity_id}, axis={args.axis}")
        return

    # Pick evenly spaced slices
    indices = np.linspace(0, len(labeled_slices) - 1, args.n_slices, dtype=int)
    chosen = [labeled_slices[i] for i in indices]
    print(f"Testing on {len(chosen)} slices: {chosen}")

    # Build figure
    fig, axes = plt.subplots(len(chosen), 4, figsize=(20, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis]

    ZOOM = 80  # half-height of zoom window around horizon (pixels)

    ious = []
    for row, sl_idx in enumerate(chosen):
        seismic_2d = np.take(seismic, sl_idx, axis=args.axis)       # (H, W) uint8
        gt_mask    = np.take(entity_mask_3d, sl_idx, axis=args.axis).astype(np.float32)

        hint = make_hint(gt_mask, N_SEEDS, HINT_SIGMA)
        probs = run_inference(session, seismic_2d, hint)
        score = iou(probs, gt_mask)
        ious.append(score)

        # Zoom window: centre on GT centroid (time axis = col)
        gt_coords = np.argwhere(gt_mask > 0)
        if len(gt_coords) > 0:
            t_centre = int(gt_coords[:, 1].mean())
        else:
            t_centre = seismic_2d.shape[1] // 2
        t0 = max(0, t_centre - ZOOM)
        t1 = min(seismic_2d.shape[1], t_centre + ZOOM)

        s_zoom  = seismic_2d[:, t0:t1]
        gt_zoom = gt_mask[:, t0:t1]
        pr_zoom = probs[:, t0:t1]
        h_zoom  = hint[:, t0:t1]

        ax_s, ax_h, ax_p, ax_g = axes[row]

        ax_s.imshow(s_zoom.T, cmap="gray", aspect="auto", origin="lower")
        ax_s.set_title(f"Seismic (slice {sl_idx}, zoom ±{ZOOM}t)")

        ax_h.imshow(s_zoom.T, cmap="gray", aspect="auto", origin="lower")
        ax_h.imshow(h_zoom.T, cmap="hot", aspect="auto", origin="lower",
                    alpha=0.6, vmin=0, vmax=1)
        ax_h.set_title("Hint overlay")

        ax_p.imshow(s_zoom.T, cmap="gray", aspect="auto", origin="lower")
        ax_p.imshow((pr_zoom > THRESHOLD).T, cmap="Reds", aspect="auto",
                    origin="lower", alpha=0.7, vmin=0, vmax=1)
        ax_p.set_title(f"Prediction  IoU={score:.3f}")

        ax_g.imshow(s_zoom.T, cmap="gray", aspect="auto", origin="lower")
        ax_g.imshow(gt_zoom.T, cmap="Greens", aspect="auto",
                    origin="lower", alpha=0.7, vmin=0, vmax=1)
        ax_g.set_title("Ground truth")

        for ax in (ax_s, ax_h, ax_p, ax_g):
            ax.axis("off")

        print(f"  slice {sl_idx:4d}  IoU={score:.3f}  t_centre={t_centre}")

    mean_iou = float(np.mean(ious))
    fig.suptitle(
        f"Horizon tracker — entity {args.entity_id}, axis={'inline' if args.axis == 0 else 'crossline'}\n"
        f"mean IoU = {mean_iou:.3f}  ({len(chosen)} slices)",
        fontsize=14,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMean IoU: {mean_iou:.3f}")
    print(f"Saved  → {args.output}")


if __name__ == "__main__":
    main()
