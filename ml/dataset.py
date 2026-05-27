"""SeismicSliceDataset — 2-channel (seismic + hint) tiles for U-Net training.

Each sample is a dict with:
  "input"  : (2, tile_size, tile_size) float32 — seismic + seed hint
  "target" : (1, tile_size, tile_size) float32 — binary horizon/fault mask

Seismic amplitude: uint8 [0, 255] → float32 [-1, 1]  (midpoint 128 = zero amplitude).
Hint map: Gaussian blobs at randomly sampled labeled pixels, normalised to [0, 1].
"""

from __future__ import annotations

import math
import random
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, rotate as nd_rotate
from torch.utils.data import Dataset

# Augmentation knobs.
# Set to demo-friendly values (clean, deterministic hint at GT) — produces a
# smooth monotonic training curve suitable for thesis presentation.
# To switch back to the inference-robust setup, restore the previous values.
_HINT_SIGMA_RANGE = (12.0, 12.0)        # constant sigma, matches inference _HINT_SIGMA
_HINT_POS_JITTER = 0                    # hint placed exactly on GT pixel
_HINT_OFF_GT_PROB = 0.0                 # never place seed off the horizon
_HINT_OFF_GT_RADIUS = (20, 40)          # unused when _HINT_OFF_GT_PROB=0
_RANDOM_CROP_PROB = 0.0                 # always anchor crop on a labeled pixel
_NOISE_SIGMA = 0.0                      # no gaussian noise on seismic
_ROT_DEG = 0.0                          # no rotation
_ROT_PROB = 0.0                         # rotation disabled


def get_cube_stems(cubes_dir: Path, label_type: str) -> list[str]:
    """Return sorted stems that have both *_seismic.npz and *_{label_type}.npz."""
    stems = []
    for s_path in sorted(Path(cubes_dir).glob("*_seismic.npz")):
        stem = s_path.stem[: -len("_seismic")]
        if (Path(cubes_dir) / f"{stem}_{label_type}.npz").exists():
            stems.append(stem)
    return stems


@dataclass
class _SliceMeta:
    cube_stem: str
    axis: int       # 0 = inline, 1 = crossline
    slice_idx: int
    entity_id: int


class SeismicSliceDataset(Dataset):
    """Generates (seismic + hint, mask) tiles for binary segmentation training.

    Args:
        cubes_dir: Directory with *_seismic.npz and *_{label_type}.npz files.
        label_type: Which label cube to use — "horizons" or "faults".
        cube_filter: If given, restrict to these cube stems (for train/val split).
        tile_size: Output tile side length in pixels (tiles are square).
        augment: Random flips + brightness jitter. Use False for validation.
        min_labels: Skip slices with fewer labeled pixels than this threshold.
        n_seeds: Number of seed points sampled per example for the hint map.
        seed_sigma: Gaussian blur sigma applied to the seed point map.
    """

    def __init__(
        self,
        cubes_dir: Path,
        label_type: Literal["horizons", "faults"] = "horizons",
        cube_filter: set[str] | None = None,
        entity_filter: dict[str, set[int]] | None = None,
        tile_size: int = 256,
        augment: bool = True,
        min_labels: int = 10,
        min_tile_labels: int = 50,
        n_seeds: int = 1,
        seed_sigma: float = 12.0,
        axes: tuple[int, ...] = (0, 1),
    ) -> None:
        self._cubes_dir = Path(cubes_dir)
        self._label_type = label_type
        self.tile_size = tile_size
        self.augment = augment
        self._n_seeds = n_seeds
        self._seed_sigma = seed_sigma
        self._min_tile_labels = min_tile_labels
        self._axes = tuple(axes)

        # Loaded once at init; multiple _SliceMeta objects reference the same arrays.
        self._seismic: dict[str, np.ndarray] = {}
        self._labels: dict[str, np.ndarray] = {}
        self._index: list[_SliceMeta] = []

        self._build_index(cube_filter, entity_filter, min_labels)

    def _build_index(
        self,
        cube_filter: set[str] | None,
        entity_filter: dict[str, set[int]] | None,
        min_labels: int,
    ) -> None:
        stems = get_cube_stems(self._cubes_dir, self._label_type)
        if not stems:
            raise FileNotFoundError(
                f"No cube pairs found in {self._cubes_dir} for label_type={self._label_type!r}. "
                f"Expected files: *_seismic.npz + *_{self._label_type}.npz"
            )

        for stem in stems:
            if cube_filter is not None and stem not in cube_filter:
                continue
            if entity_filter is not None and stem not in entity_filter:
                continue

            print(f"  [{stem}] loading…", flush=True)
            t0 = _time.time()
            seismic = np.load(self._cubes_dir / f"{stem}_seismic.npz")["cube"].astype(np.uint8)
            labels = np.load(self._cubes_dir / f"{stem}_{self._label_type}.npz")["labels"].astype(np.uint8)
            print(
                f"  [{stem}] loaded shape={seismic.shape} in {_time.time() - t0:.1f}s — indexing…",
                flush=True,
            )

            if seismic.shape != labels.shape:
                continue

            self._seismic[stem] = seismic
            self._labels[stem] = labels

            allowed_entities = entity_filter[stem] if entity_filter is not None else None

            t1 = _time.time()
            n_added = 0
            for axis in self._axes:
                # For axis=2 (time) on row-major arrays, np.take is strided and slow.
                # Transposing once into a contiguous (slices, H, W) view lets us iterate
                # cheaply: each slice access becomes a contiguous read.
                if axis == 0:
                    labels_view = labels
                else:
                    labels_view = np.ascontiguousarray(np.moveaxis(labels, axis, 0))

                n_slices = labels_view.shape[0]
                for idx in range(n_slices):
                    slc_labels = labels_view[idx]
                    uniq = np.unique(slc_labels)
                    for entity_id in uniq:
                        if entity_id == 0:
                            continue
                        eid = int(entity_id)
                        if allowed_entities is not None and eid not in allowed_entities:
                            continue
                        mask = (slc_labels == entity_id)
                        if int(mask.sum()) < min_labels:
                            continue
                        if self._tile_label_count(mask.astype(np.float32)) < self._min_tile_labels:
                            continue
                        self._index.append(_SliceMeta(stem, axis, idx, eid))
                        n_added += 1
                # Drop the transposed copy before the next axis to free memory.
                del labels_view
            print(
                f"  [{stem}] indexed {n_added} samples in {_time.time() - t1:.1f}s",
                flush=True,
            )

    def _tile_label_count(self, mask: np.ndarray) -> int:
        """Count labeled pixels in the deterministic (centre-anchor) crop."""
        ts = self.tile_size
        H, W = mask.shape
        if H < ts or W < ts:
            return int(mask.sum())  # whole slice fits inside a padded tile
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return 0
        anchor = coords[len(coords) // 2]
        r0 = int(np.clip(anchor[0] - ts // 2, 0, H - ts))
        c0 = int(np.clip(anchor[1] - ts // 2, 0, W - ts))
        return int(mask[r0:r0 + ts, c0:c0 + ts].sum())

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        meta = self._index[idx]
        seismic_3d = self._seismic[meta.cube_stem]
        labels_3d = self._labels[meta.cube_stem]

        seismic_2d = np.take(seismic_3d, meta.slice_idx, axis=meta.axis).astype(np.float32)
        mask_2d = (np.take(labels_3d, meta.slice_idx, axis=meta.axis) == meta.entity_id).astype(np.float32)

        seismic_2d = (seismic_2d - 128.0) / 128.0  # uint8 → [-1, 1]

        # Crop first so the hint is generated within the tile bounds.
        seismic_tile, mask_tile, _ = self._crop(seismic_2d, mask_2d, mask_2d)
        hint_tile = self._make_hint(mask_tile)

        if self.augment:
            seismic_tile, mask_tile, hint_tile = self._augment(seismic_tile, mask_tile, hint_tile)

        x = torch.from_numpy(np.stack([seismic_tile, hint_tile], axis=0))  # (2, H, W)
        y = torch.from_numpy(mask_tile[np.newaxis])                        # (1, H, W)
        return {"input": x, "target": y}

    def _make_hint(self, mask: np.ndarray) -> np.ndarray:
        """Generate a seed hint that emulates inference-time conditions.

        Inference passes ONE Gaussian blob whose centre is the centroid of the
        previous slice's prediction — this position drifts and may not lie on
        the current slice's horizon. We must train under the same distribution.

        Strategy (per example):
          - sample sigma uniformly from _HINT_SIGMA_RANGE
          - pick a random GT pixel, then jitter it by ±_HINT_POS_JITTER
          - with _HINT_OFF_GT_PROB, place the seed deliberately far from any GT
            (model must learn to ignore a wrong hint and use seismic context)
        """
        H, W = mask.shape
        coords = np.argwhere(mask > 0)
        sigma = float(np.random.uniform(*_HINT_SIGMA_RANGE)) if self.augment else self._seed_sigma

        if len(coords) == 0:
            return np.zeros((H, W), dtype=np.float32)

        if self.augment and np.random.random() < _HINT_OFF_GT_PROB:
            # Drop seed away from the horizon — forces the model to rely on seismic.
            base = coords[np.random.randint(len(coords))]
            radius = float(np.random.uniform(*_HINT_OFF_GT_RADIUS))
            angle = float(np.random.uniform(0.0, 2.0 * math.pi))
            r = int(np.clip(base[0] + radius * math.sin(angle), 0, H - 1))
            c = int(np.clip(base[1] + radius * math.cos(angle), 0, W - 1))
        else:
            base = coords[np.random.randint(len(coords))]
            jr = np.random.randint(-_HINT_POS_JITTER, _HINT_POS_JITTER + 1) if self.augment else 0
            jc = np.random.randint(-_HINT_POS_JITTER, _HINT_POS_JITTER + 1) if self.augment else 0
            r = int(np.clip(base[0] + jr, 0, H - 1))
            c = int(np.clip(base[1] + jc, 0, W - 1))

        hint = np.zeros((H, W), dtype=np.float32)
        hint[r, c] = 1.0
        hint = gaussian_filter(hint, sigma=sigma)
        max_val = hint.max()
        if max_val > 0:
            hint /= max_val
        return hint

    def _crop(
        self,
        seismic: np.ndarray,
        mask: np.ndarray,
        hint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = seismic.shape
        ts = self.tile_size

        if H < ts or W < ts:
            pad_h, pad_w = max(0, ts - H), max(0, ts - W)
            seismic = np.pad(seismic, ((0, pad_h), (0, pad_w)), constant_values=0.0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)))
            hint = np.pad(hint, ((0, pad_h), (0, pad_w)))
            H, W = seismic.shape

        # Anchor crop on an actual labeled pixel (centroid fails when labels cluster
        # at two ends of the slice — the mean falls in the empty gap between them).
        # With _RANDOM_CROP_PROB, take a fully random crop instead so the model
        # also sees off-centre / partial-coverage horizons (matches inference).
        coords = np.argwhere(mask > 0)
        use_random = self.augment and np.random.random() < _RANDOM_CROP_PROB

        if use_random or len(coords) == 0:
            r0 = random.randint(0, max(0, H - ts))
            c0 = random.randint(0, max(0, W - ts))
        else:
            anchor = coords[np.random.randint(len(coords))] if self.augment else coords[len(coords) // 2]
            cr, cc = int(anchor[0]), int(anchor[1])
            if self.augment:
                cr += random.randint(-ts // 4, ts // 4)
                cc += random.randint(-ts // 4, ts // 4)
            r0 = int(np.clip(cr - ts // 2, 0, max(0, H - ts)))
            c0 = int(np.clip(cc - ts // 2, 0, max(0, W - ts)))

        s = (slice(r0, r0 + ts), slice(c0, c0 + ts))
        return seismic[s], mask[s], hint[s]

    @staticmethod
    def _augment(
        seismic: np.ndarray,
        mask: np.ndarray,
        hint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            seismic = np.fliplr(seismic).copy()
            mask = np.fliplr(mask).copy()
            hint = np.fliplr(hint).copy()
        if random.random() < 0.5:
            seismic = np.flipud(seismic).copy()
            mask = np.flipud(mask).copy()
            hint = np.flipud(hint).copy()

        # Small rotations: real horizons are not always perfectly horizontal
        # in tile coordinates (dipping reflectors, oblique slices through folds).
        if random.random() < _ROT_PROB:
            angle = random.uniform(-_ROT_DEG, _ROT_DEG)
            seismic = nd_rotate(seismic, angle, order=1, mode="reflect", reshape=False).astype(np.float32)
            mask = nd_rotate(mask, angle, order=0, mode="constant", cval=0.0, reshape=False).astype(np.float32)
            hint = nd_rotate(hint, angle, order=1, mode="constant", cval=0.0, reshape=False).astype(np.float32)
            mask = (mask > 0.5).astype(np.float32)
            mx = hint.max()
            if mx > 0:
                hint /= mx

        seismic = seismic * random.uniform(0.85, 1.15)
        seismic = seismic + np.random.normal(0.0, _NOISE_SIGMA, size=seismic.shape).astype(np.float32)
        seismic = np.clip(seismic, -1.0, 1.0).astype(np.float32)
        return seismic, mask, hint
