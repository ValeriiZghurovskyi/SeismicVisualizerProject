"""OnnxFaultTracker — ONNX Runtime implementation of FaultTracker."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label as nd_label

from seismic_visualizer.domain.cube import SeismicCube
from seismic_visualizer.domain.geometry import Axis, Point3D
from seismic_visualizer.domain.ml.fault_tracker import FaultTracker

_log = logging.getLogger(__name__)

_HINT_SIGMA = 12.0
_THRESHOLD = 0.4
_CC_MAX_DIST = 100
_PAD_MULTIPLE = 16
_TILE_SIZE = 256
_TILE_OVERLAP = 128
_SEED_ANCHOR_WEIGHT = 0.3
_HINT_SAMPLE_POINTS = 6
_STOP_MAX_PROB = 0.5
_STOP_MIN_PIXELS = 8
_MAX_PROPAGATION = 25
_USER_HINT_SPACING = 24
_USER_HINT_MAX_POINTS = 12


class OnnxFaultTracker:
    """Runs a fault-tracking ONNX model.

    Implements FaultTracker structurally (no explicit inheritance needed).
    Unlike horizon tracking (one depth pick per row), fault tracking accepts
    entire connected-component regions — faults are planar volumes, not surfaces.

    Args:
        model_path: Path to the .onnx model file.
    """

    def __init__(self, model_path: Path, debug_dir: Path | None = None) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if "DmlExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        env_dir = os.environ.get("SEISMIC_FAULT_DEBUG_DIR")
        self._debug_dir: Path | None = (
            Path(debug_dir) if debug_dir is not None
            else (Path(env_dir) if env_dir else None)
        )
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_limit = int(os.environ.get("SEISMIC_FAULT_DEBUG_LIMIT", "5"))
        self._debug_count = 0

    def track(
        self,
        seismic: SeismicCube,
        seed_points: list[Point3D],
        axis: Axis,
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Predict fault voxels propagating from seed points along *axis*.

        Args:
            seismic: Read-only seismic amplitude cube (uint8, midpoint=128).
            seed_points: User-provided seed voxel coordinates.
            axis: INLINE, CROSSLINE, or TIME — slice direction to iterate over.
                For faults, TIME is the natural choice: a fault plane intersects
                a horizontal cut as a planar trace.

        Returns:
            Integer array of shape (N, 3): (inline, crossline, time) per voxel.
            Empty (0, 3) when no predictions are made.
        """
        n_slices = seismic.shape[axis.value]
        seed_map = self._build_seed_map(seed_points, axis, n_slices)
        user_points = self._build_user_points(seed_points, axis)

        if not seed_map or not user_points:
            return np.empty((0, 3), dtype=np.int32)

        user_keys = sorted(user_points)
        min_idx = user_keys[0]
        max_idx = user_keys[-1]

        print(f"[OnnxFaultTracker] axis={axis.name}, n_slices={n_slices}, "
              f"user_seed_range=[{min_idx},{max_idx}]")
        print(f"[OnnxFaultTracker] slice shape={np.take(seismic.data, 0, axis=axis.value).shape}")
        print(f"[OnnxFaultTracker] user-drawn slices: {user_keys}")

        voxels: list[tuple[int, int, int]] = []

        def _hint_for(sl: int, prev_hint: list[tuple[int, int]]) -> list[tuple[int, int]]:
            if sl in user_points:
                return self._subsample_user_hint(user_points[sl])
            return prev_hint

        fwd_stop = min(n_slices, max_idx + _MAX_PROPAGATION + 1)
        print(f"[OnnxFaultTracker] forward: {min_idx} → {fwd_stop - 1}")
        prev_centroid: tuple[int, int] = seed_map[min_idx]
        prev_hint: list[tuple[int, int]] = self._subsample_user_hint(user_points[min_idx])
        for sl in range(min_idx, fwd_stop):
            if cancel_check is not None and cancel_check():
                print(f"[OnnxFaultTracker] cancelled at forward slice {sl}")
                break
            sl_data = np.take(seismic.data, sl, axis=axis.value)
            probs = self._run_model(
                sl_data, _hint_for(sl, prev_hint), _HINT_SIGMA,
                debug_tag=f"fwd_{sl:04d}",
            )
            if sl not in user_points and self._should_stop(probs):
                print(f"[OnnxFaultTracker] forward stopped at slice {sl} (low confidence)")
                break
            new_voxels, new_centroid, pixels_2d = self._pick_fault(probs, sl, axis, prev_centroid)
            if new_centroid is None:
                print(f"[OnnxFaultTracker] forward stopped at slice {sl} (no pick)")
                break
            prev_centroid = self._anchor_to_seed(new_centroid, seed_map.get(sl))
            prev_hint = self._sample_hint_from_pixels(pixels_2d, prev_centroid)
            voxels.extend(new_voxels)
            if sl % 10 == 0:
                print(f"[OnnxFaultTracker] fwd {sl}/{n_slices - 1}  voxels={len(voxels)}  hint_pts={len(prev_hint)}")

        bwd_stop = max(-1, min_idx - _MAX_PROPAGATION - 1)
        print(f"[OnnxFaultTracker] backward: {min_idx - 1} → {bwd_stop + 1}")
        prev_centroid = seed_map[min_idx]
        prev_hint = self._subsample_user_hint(user_points[min_idx])
        for sl in range(min_idx - 1, bwd_stop, -1):
            if cancel_check is not None and cancel_check():
                print(f"[OnnxFaultTracker] cancelled at backward slice {sl}")
                break
            sl_data = np.take(seismic.data, sl, axis=axis.value)
            probs = self._run_model(
                sl_data, _hint_for(sl, prev_hint), _HINT_SIGMA,
                debug_tag=f"bwd_{sl:04d}",
            )
            if sl not in user_points and self._should_stop(probs):
                print(f"[OnnxFaultTracker] backward stopped at slice {sl} (low confidence)")
                break
            new_voxels, new_centroid, pixels_2d = self._pick_fault(probs, sl, axis, prev_centroid)
            if new_centroid is None:
                print(f"[OnnxFaultTracker] backward stopped at slice {sl} (no pick)")
                break
            prev_centroid = self._anchor_to_seed(new_centroid, seed_map.get(sl))
            prev_hint = self._sample_hint_from_pixels(pixels_2d, prev_centroid)
            voxels.extend(new_voxels)
            if (min_idx - 1 - sl) % 10 == 0:
                print(f"[OnnxFaultTracker] bwd {sl}  voxels={len(voxels)}  hint_pts={len(prev_hint)}")

        print(f"[OnnxFaultTracker] done — total voxels: {len(voxels)}")
        if not voxels:
            return np.empty((0, 3), dtype=np.int32)
        return np.array(voxels, dtype=np.int32)


    def _build_seed_map(
        self,
        seed_points: list[Point3D],
        axis: Axis,
        n_slices: int,
    ) -> dict[int, tuple[int, int]]:
        """Map every slice index to an (row, col) centroid via linear interpolation.

        For INLINE axis:    key=inline,    value=(crossline, time) centroid.
        For CROSSLINE axis: key=crossline, value=(inline,    time) centroid.
        For TIME axis:      key=time,      value=(inline,    crossline) centroid.
        Extrapolates boundary values to cube edges.
        """
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        if axis == Axis.INLINE:
            for pt in seed_points:
                groups[pt.inline].append((pt.crossline, pt.time))
        elif axis == Axis.CROSSLINE:
            for pt in seed_points:
                groups[pt.crossline].append((pt.inline, pt.time))
        else:
            for pt in seed_points:
                groups[pt.time].append((pt.inline, pt.crossline))

        raw: dict[int, tuple[int, int]] = {
            k: (round(sum(r for r, _ in v) / len(v)), round(sum(c for _, c in v) / len(v)))
            for k, v in groups.items()
        }
        if not raw:
            return {}

        keys = sorted(raw)
        result: dict[int, tuple[int, int]] = {}
        for k in keys:
            result[k] = raw[k]
        for i in range(len(keys) - 1):
            k0, k1 = keys[i], keys[i + 1]
            r0, c0 = raw[k0]
            r1, c1 = raw[k1]
            for sl in range(k0 + 1, k1):
                t = (sl - k0) / (k1 - k0)
                result[sl] = (round(r0 + t * (r1 - r0)), round(c0 + t * (c1 - c0)))
        for sl in range(0, keys[0]):
            result[sl] = raw[keys[0]]
        for sl in range(keys[-1] + 1, n_slices):
            result[sl] = raw[keys[-1]]
        return result

    @staticmethod
    def _build_user_points(
        seed_points: list[Point3D],
        axis: Axis,
    ) -> dict[int, list[tuple[int, int]]]:
        """Group user seeds by slice index → list of (row, col) in slice coords."""
        pts: dict[int, list[tuple[int, int]]] = defaultdict(list)
        if axis == Axis.INLINE:
            for pt in seed_points:
                pts[pt.inline].append((pt.crossline, pt.time))
        elif axis == Axis.CROSSLINE:
            for pt in seed_points:
                pts[pt.crossline].append((pt.inline, pt.time))
        else:
            for pt in seed_points:
                pts[pt.time].append((pt.inline, pt.crossline))
        return dict(pts)

    def _run_model(
        self,
        seismic_2d: np.ndarray,
        hint_points: list[tuple[int, int]],
        sigma: float,
        debug_tag: str | None = None,
    ) -> np.ndarray:
        """Run ONNX model on one slice via sliding-window tiling, multi-point hint."""
        h_orig, w_orig = seismic_2d.shape
        seismic_norm = (seismic_2d.astype(np.float32) - 128.0) / 128.0
        hint_full = self._make_hint_multi(hint_points, (h_orig, w_orig), sigma)

        if h_orig <= _TILE_SIZE and w_orig <= _TILE_SIZE:
            probs = self._run_tile(seismic_norm, hint_full, h_orig, w_orig)
            self._dump_debug(seismic_norm, hint_full, probs, debug_tag)
            return probs

        stride = _TILE_SIZE - _TILE_OVERLAP
        prob_acc = np.zeros((h_orig, w_orig), dtype=np.float32)
        weight_acc = np.zeros((h_orig, w_orig), dtype=np.float32)

        rows = self._tile_starts(h_orig, _TILE_SIZE, stride)
        cols = self._tile_starts(w_orig, _TILE_SIZE, stride)

        for r0 in rows:
            for c0 in cols:
                r1 = min(r0 + _TILE_SIZE, h_orig)
                c1 = min(c0 + _TILE_SIZE, w_orig)
                s_tile = seismic_norm[r0:r1, c0:c1]
                tile_h, tile_w = s_tile.shape
                h_tile = hint_full[r0:r1, c0:c1].astype(np.float32, copy=True)

                if hint_points and h_tile.max() < 0.05:
                    cy = r0 + tile_h / 2
                    cx = c0 + tile_w / 2
                    closest = min(
                        hint_points,
                        key=lambda p, _cy=cy, _cx=cx: (p[0] - _cy) ** 2 + (p[1] - _cx) ** 2,
                    )
                    local_pos = (
                        int(np.clip(closest[0] - r0, 0, tile_h - 1)),
                        int(np.clip(closest[1] - c0, 0, tile_w - 1)),
                    )
                    h_tile = self._make_hint(local_pos, (tile_h, tile_w), sigma)

                probs_tile = self._run_tile(s_tile, h_tile, tile_h, tile_w)
                prob_acc[r0:r1, c0:c1] += probs_tile
                weight_acc[r0:r1, c0:c1] += 1.0

        probs = (prob_acc / np.maximum(weight_acc, 1e-6)).astype(np.float32)
        self._dump_debug(seismic_norm, hint_full, probs, debug_tag)
        return probs

    def _run_tile(
        self,
        seismic_norm: np.ndarray,
        hint: np.ndarray,
        h_orig: int,
        w_orig: int,
    ) -> np.ndarray:
        s_pad, _, _ = self._pad_to_multiple(seismic_norm)
        h_pad, _, _ = self._pad_to_multiple(hint)
        x = np.stack([s_pad, h_pad], axis=0)[np.newaxis]
        logits = self._session.run([self._output_name], {self._input_name: x})[0]
        probs = 1.0 / (1.0 + np.exp(-logits[0, 0].astype(np.float64)))
        return probs[:h_orig, :w_orig].astype(np.float32)

    def _pad_to_multiple(
        self, arr: np.ndarray, factor: int = _PAD_MULTIPLE
    ) -> tuple[np.ndarray, int, int]:
        h, w = arr.shape
        ph = (factor - h % factor) % factor
        pw = (factor - w % factor) % factor
        return np.pad(arr, ((0, ph), (0, pw)), mode="reflect"), h, w

    def _dump_debug(
        self,
        seismic_norm: np.ndarray,
        hint: np.ndarray,
        probs: np.ndarray,
        tag: str | None,
    ) -> None:
        if self._debug_dir is None or tag is None:
            return
        if self._debug_count >= self._debug_limit:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(seismic_norm, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[0].set_title("seismic")
        axes[1].imshow(hint, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title("hint")
        im = axes[2].imshow(probs, cmap="hot", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"probs (max={probs.max():.2f}, >0.4={int((probs > 0.4).sum())}px)")
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        fig.tight_layout()
        out = self._debug_dir / f"{tag}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        self._debug_count += 1
        print(f"[OnnxFaultTracker] debug dump → {out}")

    @staticmethod
    def _tile_starts(size: int, tile: int, stride: int) -> list[int]:
        if size <= tile:
            return [0]
        starts = list(range(0, size - tile + 1, stride))
        if starts[-1] != size - tile:
            starts.append(size - tile)
        return starts

    def _make_hint_multi(
        self,
        points: list[tuple[int, int]],
        shape: tuple[int, int],
        sigma: float,
    ) -> np.ndarray:
        if not points:
            return np.zeros(shape, dtype=np.float32)
        H, W = shape
        hint = np.zeros(shape, dtype=np.float32)
        for r, c in points:
            rr = int(np.clip(r, 0, H - 1))
            cc = int(np.clip(c, 0, W - 1))
            hint[rr, cc] = 1.0
        hint = gaussian_filter(hint, sigma=sigma).astype(np.float32)
        mx = hint.max()
        if mx > 0:
            hint /= mx
        return hint

    def _make_hint(
        self, pos: tuple[int, int] | None, shape: tuple[int, int], sigma: float
    ) -> np.ndarray:
        hint = np.zeros(shape, dtype=np.float32)
        if pos is None:
            return hint
        r = int(np.clip(pos[0], 0, shape[0] - 1))
        c = int(np.clip(pos[1], 0, shape[1] - 1))
        hint[r, c] = 1.0
        hint = gaussian_filter(hint, sigma=sigma).astype(np.float32)
        mx = hint.max()
        if mx > 0:
            hint /= mx
        return hint

    @staticmethod
    def _subsample_user_hint(
        points: list[tuple[int, int]],
        spacing: int = _USER_HINT_SPACING,
        max_points: int = _USER_HINT_MAX_POINTS,
    ) -> list[tuple[int, int]]:
        """Pick points spaced ~`spacing` px along the seed line's principal axis.

        Training uses one Gaussian-blob hint per tile; dense seeds (hundreds
        of pixels) produce an overly thick hint blob that pushes the model to
        over-predict. We subsample to one point every ~2σ along the seed's
        principal direction so the hint covers the seed's full extent without
        bloating each Gaussian into the neighbours.
        """
        if len(points) <= 1:
            return list(points)
        arr = np.asarray(points, dtype=np.int32)
        centred = arr.astype(np.float32) - arr.mean(axis=0)
        cov = np.cov(centred, rowvar=False)
        _, vecs = np.linalg.eigh(cov)
        proj = centred @ vecs[:, -1]
        extent = float(proj.max() - proj.min())
        n = max(2, min(max_points, int(extent // spacing) + 1))
        if len(points) <= n:
            return list(points)
        order = np.argsort(proj)
        sorted_arr = arr[order]
        idx = np.linspace(0, len(sorted_arr) - 1, n, dtype=np.int32)
        return [(int(sorted_arr[i, 0]), int(sorted_arr[i, 1])) for i in idx]

    @staticmethod
    def _should_stop(probs: np.ndarray) -> bool:
        """Heuristic: the fault has ended when the model has no confident pixel."""
        return (
            float(probs.max()) < _STOP_MAX_PROB
            or int((probs > _THRESHOLD).sum()) < _STOP_MIN_PIXELS
        )

    @staticmethod
    def _sample_hint_from_pixels(
        pixels: np.ndarray,
        fallback_centroid: tuple[int, int],
        n: int = _HINT_SAMPLE_POINTS,
    ) -> list[tuple[int, int]]:
        """Pick *n* points spread along the predicted segment's principal axis.

        Replaces single-centroid propagation: a multi-point hint conveys the
        previous slice's prediction *shape*, not just its centre, so the model
        receives a directional cue ("the fault was a line through these
        points") instead of "the fault was a blob at this point".
        """
        if len(pixels) == 0:
            return [fallback_centroid]
        if len(pixels) <= n:
            return [(int(r), int(c)) for r, c in pixels]

        centred = pixels.astype(np.float32) - pixels.mean(axis=0)
        cov = np.cov(centred, rowvar=False)
        _, vecs = np.linalg.eigh(cov)
        principal = vecs[:, -1]
        proj = centred @ principal
        order = np.argsort(proj)
        sorted_pixels = pixels[order]
        idx = np.linspace(0, len(sorted_pixels) - 1, n, dtype=np.int32)
        return [(int(sorted_pixels[i, 0]), int(sorted_pixels[i, 1])) for i in idx]

    @staticmethod
    def _anchor_to_seed(
        predicted: tuple[int, int],
        seed_pos: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if seed_pos is None:
            return predicted
        w = _SEED_ANCHOR_WEIGHT
        r = round((1.0 - w) * predicted[0] + w * seed_pos[0])
        c = round((1.0 - w) * predicted[1] + w * seed_pos[1])
        return (int(r), int(c))

    def _pick_fault(
        self,
        probs: np.ndarray,
        slice_idx: int,
        axis: Axis,
        prev_centroid: tuple[int, int],
    ) -> tuple[list[tuple[int, int, int]], tuple[int, int] | None, np.ndarray]:
        """Accept all pixels in connected components near prev_centroid.

        Unlike horizon picking (one depth pick per row), faults are planar
        volumes — we keep every pixel in qualifying CCs above threshold.

        Returns (voxels, new_centroid, pixels_2d). Empty → ([], None, empty array).
        """
        probs_filtered = self._select_fault_components(probs, prev_centroid)
        pixels = np.argwhere(probs_filtered > _THRESHOLD)

        if len(pixels) == 0:
            return [], None, np.empty((0, 2), dtype=np.int32)

        if axis == Axis.INLINE:
            voxels = [(slice_idx, int(r), int(c)) for r, c in pixels]
        elif axis == Axis.CROSSLINE:
            voxels = [(int(r), slice_idx, int(c)) for r, c in pixels]
        else:
            voxels = [(int(r), int(c), slice_idx) for r, c in pixels]

        new_centroid = (int(np.mean(pixels[:, 0])), int(np.mean(pixels[:, 1])))
        return voxels, new_centroid, pixels

    @staticmethod
    def _select_fault_components(
        probs: np.ndarray,
        prev_centroid: tuple[int, int],
    ) -> np.ndarray:
        """Keep CCs whose centroid is within _CC_MAX_DIST of prev_centroid.

        Falls back to the largest CC when none qualifies — preserves propagation
        signal rather than producing an empty slice and breaking continuity.
        """
        binary = (probs > _THRESHOLD).astype(np.uint8)
        if binary.sum() == 0:
            return probs
        labeled, n = nd_label(binary)
        if n == 0:
            return probs
        if n == 1:
            return probs

        centroids = center_of_mass(binary, labeled, range(1, n + 1))
        pr, pc = prev_centroid

        keep_mask = np.zeros(probs.shape, dtype=np.float32)
        for cc_idx, (cy, cx) in enumerate(centroids):
            dist = ((cy - pr) ** 2 + (cx - pc) ** 2) ** 0.5
            if dist <= _CC_MAX_DIST:
                keep_mask[labeled == cc_idx + 1] = 1.0

        if keep_mask.sum() == 0:
            sizes = np.bincount(labeled.ravel())[1:]
            best = int(np.argmax(sizes)) + 1
            keep_mask[labeled == best] = 1.0

        return probs * keep_mask


assert isinstance(OnnxFaultTracker.__new__(OnnxFaultTracker), FaultTracker)
