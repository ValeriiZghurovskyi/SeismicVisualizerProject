"""Diagnostic script — run before training to inspect dataset pipeline.

Usage:
    python -m ml.debug_dataset
    python -m ml.debug_dataset --cubes_dir data/cubes --label_type horizons
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ml.dataset import SeismicSliceDataset, get_cube_stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubes_dir", type=Path, default=Path("data/cubes"))
    parser.add_argument("--label_type", choices=["horizons", "faults"], default="horizons")
    parser.add_argument("--tile_size", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("1. RAW FILES")
    print("=" * 60)
    stems = get_cube_stems(args.cubes_dir, args.label_type)
    print(f"Found cubes: {stems}")

    for stem in stems:
        s_path = args.cubes_dir / f"{stem}_seismic.npz"
        l_path = args.cubes_dir / f"{stem}_{args.label_type}.npz"

        s_data = np.load(s_path)
        l_data = np.load(l_path)
        print(f"\n  Cube: {stem}")
        print(f"    seismic keys  : {s_data.files}")
        print(f"    labels  keys  : {l_data.files}")

        seismic = s_data["cube"].astype(np.uint8)
        labels = l_data["labels"].astype(np.uint8)
        unique = np.unique(labels)
        print(f"    seismic shape : {seismic.shape}  dtype={seismic.dtype}")
        print(f"    labels  shape : {labels.shape}   dtype={labels.dtype}")
        print(f"    unique labels : {unique}  (0=empty, rest=entity IDs)")
        print(f"    labeled voxels: {(labels > 0).sum():,} / {labels.size:,}")

        for eid in unique:
            if eid == 0:
                continue
            cnt = int((labels == eid).sum())
            print(f"      entity {eid:3d}: {cnt:,} voxels")

    print("\n" + "=" * 60)
    print("2. DATASET INDEX")
    print("=" * 60)
    ds = SeismicSliceDataset(
        cubes_dir=args.cubes_dir,
        label_type=args.label_type,
        tile_size=args.tile_size,
        augment=False,
    )
    print(f"Total samples in index: {len(ds)}")

    if len(ds) == 0:
        print("ERROR: dataset is empty — no valid slices found.")
        return

    # Inspect first 5 index entries
    print("\nFirst 5 index entries:")
    for i, meta in enumerate(ds._index[:5]):
        labels_3d = ds._labels[meta.cube_stem]
        full_mask = (np.take(labels_3d, meta.slice_idx, axis=meta.axis) == meta.entity_id).astype(np.float32)
        print(f"  [{i}] cube={meta.cube_stem}  axis={meta.axis}  slice={meta.slice_idx:4d}  "
              f"entity={meta.entity_id}  labeled_px={full_mask.sum():.0f}  "
              f"slice_shape={full_mask.shape}")

    print("\n" + "=" * 60)
    print("3. PIPELINE CHECK (first sample)")
    print("=" * 60)
    meta = ds._index[0]
    labels_3d = ds._labels[meta.cube_stem]
    seismic_3d = ds._seismic[meta.cube_stem]

    seismic_2d = np.take(seismic_3d, meta.slice_idx, axis=meta.axis).astype(np.float32)
    mask_2d = (np.take(labels_3d, meta.slice_idx, axis=meta.axis) == meta.entity_id).astype(np.float32)

    print(f"  Full seismic slice : {seismic_2d.shape}  min={seismic_2d.min():.1f}  max={seismic_2d.max():.1f}")
    print(f"  Full mask          : {mask_2d.shape}  labeled_px={mask_2d.sum():.0f}")

    coords = np.argwhere(mask_2d > 0)
    if len(coords) > 0:
        cr = int(coords[:, 0].mean())
        cc = int(coords[:, 1].mean())
        print(f"  Label centroid     : row={cr}  col={cc}")
        print(f"  Label row range    : [{coords[:,0].min()}, {coords[:,0].max()}]")
        print(f"  Label col range    : [{coords[:,1].min()}, {coords[:,1].max()}]")
    else:
        print("  ERROR: mask_2d has NO labeled pixels (but index said it should)")

    sample = ds[0]
    mask_tile = sample["target"][0].numpy()
    hint_tile = sample["input"][1].numpy()
    print(f"\n  Tile shape         : {mask_tile.shape}")
    print(f"  mask_tile labeled  : {mask_tile.sum():.0f} px")
    print(f"  hint_tile max      : {hint_tile.max():.4f}")

    # Manual crop simulation to diagnose
    print("\n--- Manual crop simulation ---")
    H, W = mask_2d.shape
    ts = args.tile_size
    print(f"  Slice: H={H}, W={W}, tile_size={ts}")

    coords = np.argwhere(mask_2d > 0)
    print(f"  All labeled pixel coords (row, col):")
    for r, c in coords:
        print(f"    ({r}, {c})")

    cr = int(coords[:, 0].mean())
    cc = int(coords[:, 1].mean())
    r0 = int(np.clip(cr - ts // 2, 0, max(0, H - ts)))
    c0 = int(np.clip(cc - ts // 2, 0, max(0, W - ts)))
    print(f"  Crop window: rows [{r0}:{r0+ts}], cols [{c0}:{c0+ts}]")

    manual_tile = mask_2d[r0:r0+ts, c0:c0+ts]
    print(f"  Manual crop labeled px: {manual_tile.sum():.0f}")

    in_window = [(r, c) for r, c in coords if r0 <= r < r0+ts and c0 <= c < c0+ts]
    print(f"  Pixels inside window: {in_window}")

    if mask_tile.sum() == 0:
        print("\n  *** PROBLEM: mask_tile is empty after crop! ***")
        if manual_tile.sum() > 0:
            print("  Manual crop HAS pixels but ds[0] does not — dataset.py is using OLD cached code.")
            print("  Try: find . -name '*.pyc' -delete  then re-run")
        else:
            print("  All labeled pixels are OUTSIDE the crop window.")
    else:
        print("\n  Pipeline OK — labeled pixels survive crop.")


if __name__ == "__main__":
    main()
