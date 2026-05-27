"""OnnxTracker — ONNX Runtime implementation of HorizonTracker."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label as nd_label

from seismic_visualizer.domain.cube import SeismicCube
from seismic_visualizer.domain.geometry import Axis, Point3D
from seismic_visualizer.domain.ml.horizon_tracker import HorizonTracker

_HINT_SIGMA = 12.0
_THRESHOLD = 0.4
_CC_MAX_DIST = 150
_PAD_MULTIPLE = 16
_TILE_SIZE = 256
_TILE_OVERLAP = 128
_TRACKING_WINDOW = 50
_CONSISTENCY = 50
_CC_TIME_TOL = 40
_MAX_DRIFT_FROM_USER = 25
_SEED_ANCHOR_WEIGHT = 0.3


class OnnxTracker:
    """Runs a horizon-tracking ONNX model.

    Implements HorizonTracker structurally (no explicit inheritance needed).

    Args:
        model_path: Path to the .onnx model file.
    """

    def __init__(self, model_path: Path) -> None:
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

    def track(
        self,
        seismic: SeismicCube,
        seed_points: list[Point3D],
        axis: Axis,
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Predict horizon voxels propagating from seed points along *axis*.

        Args:
            seismic: Read-only seismic amplitude cube (uint8, midpoint=128).
            seed_points: User-provided seed voxel coordinates.
            axis: INLINE or CROSSLINE — slice direction to iterate over.

        Returns:
            Integer array of shape (N, 3): (inline, crossline, time) per voxel.
            Empty (0, 3) when no predictions are made.
        """
        n_slices = seismic.shape[axis.value]
        seed_map = self._build_seed_map(seed_points, axis, n_slices)
        user_curves = self._build_user_curves(seed_points, axis)

        if not seed_map:
            return np.empty((0, 3), dtype=np.int32)

        sorted_keys = sorted(seed_map)
        min_idx = sorted_keys[0]

        slice_shape = np.take(seismic.data, 0, axis=axis.value).shape
        n_rows_in_slice = slice_shape[0]

        interp_curves = self._build_interp_curves(
            user_curves, n_slices, n_rows_in_slice, self._extend_curve,
        )

        print(f"[OnnxTracker] axis={axis.name}, n_slices={n_slices}, min_seed_idx={min_idx}")
        print(f"[OnnxTracker] slice shape={slice_shape}")
        print(f"[OnnxTracker] user-drawn slices: {sorted(user_curves)}")

        voxels: list[tuple[int, int, int]] = []

        def _curve_for(sl: int, fallback_curve: list[tuple[int, int]]) -> list[tuple[int, int]]:
            """User's drawn curve (extended to full slice width) takes priority;
            otherwise the propagated predicted curve from the previous slice."""
            if sl in user_curves:
                return self._extend_curve(user_curves[sl], n_rows_in_slice)
            return fallback_curve

        print(f"[OnnxTracker] forward: {min_idx} → {n_slices - 1}")
        prev_pos: tuple[int, int] = seed_map[min_idx]
        initial_pts = user_curves.get(min_idx, [prev_pos])
        prev_curve: list[tuple[int, int]] = self._extend_curve(initial_pts, n_rows_in_slice)
        for sl in range(min_idx, n_slices):
            if cancel_check is not None and cancel_check():
                print(f"[OnnxTracker] cancelled at forward slice {sl}")
                break
            sl_data = np.take(seismic.data, sl, axis=axis.value)
            hint_curve = _curve_for(sl, prev_curve)
            probs = self._run_model(sl_data, hint_curve, _HINT_SIGMA)
            probs = self._select_curve_components(probs, hint_curve)
            new_voxels, new_pos = self._pick_horizon(probs, sl, axis, hint_curve)
            if new_pos is None:
                new_pos = seed_map.get(sl)
            if new_pos is None:
                print(f"[OnnxTracker] forward stopped at slice {sl}")
                break
            prev_pos = self._anchor_to_seed(new_pos, seed_map.get(sl))
            if new_voxels:
                raw_curve = self._voxels_to_curve(new_voxels, axis)
                prev_curve = self._extend_curve(raw_curve, n_rows_in_slice)
                if sl in interp_curves:
                    prev_curve = self._clamp_curve_to_anchor(
                        prev_curve, interp_curves[sl], _MAX_DRIFT_FROM_USER,
                    )
            voxels.extend(new_voxels)
            if sl % 10 == 0:
                print(f"[OnnxTracker] fwd {sl}/{n_slices - 1}  voxels={len(voxels)}  curve={len(prev_curve)}")

        print(f"[OnnxTracker] backward: {min_idx - 1} → 0")
        prev_pos = seed_map[min_idx]
        initial_pts = user_curves.get(min_idx, [prev_pos])
        prev_curve = self._extend_curve(initial_pts, n_rows_in_slice)
        for sl in range(min_idx - 1, -1, -1):
            if cancel_check is not None and cancel_check():
                print(f"[OnnxTracker] cancelled at backward slice {sl}")
                break
            sl_data = np.take(seismic.data, sl, axis=axis.value)
            hint_curve = _curve_for(sl, prev_curve)
            probs = self._run_model(sl_data, hint_curve, _HINT_SIGMA)
            probs = self._select_curve_components(probs, hint_curve)
            new_voxels, new_pos = self._pick_horizon(probs, sl, axis, hint_curve)
            if new_pos is None:
                new_pos = seed_map.get(sl)
            if new_pos is None:
                print(f"[OnnxTracker] backward stopped at slice {sl}")
                break
            prev_pos = self._anchor_to_seed(new_pos, seed_map.get(sl))
            if new_voxels:
                raw_curve = self._voxels_to_curve(new_voxels, axis)
                prev_curve = self._extend_curve(raw_curve, n_rows_in_slice)
                if sl in interp_curves:
                    prev_curve = self._clamp_curve_to_anchor(
                        prev_curve, interp_curves[sl], _MAX_DRIFT_FROM_USER,
                    )
            voxels.extend(new_voxels)
            if (min_idx - 1 - sl) % 10 == 0:
                print(f"[OnnxTracker] bwd {sl}  voxels={len(voxels)}  curve={len(prev_curve)}")

        print(f"[OnnxTracker] done — total voxels: {len(voxels)}")
        if not voxels:
            return np.empty((0, 3), dtype=np.int32)
        return np.array(voxels, dtype=np.int32)


    def _build_seed_map(
        self,
        seed_points: list[Point3D],
        axis: Axis,
        n_slices: int,
    ) -> dict[int, tuple[int, int]]:
        """Map every slice index to an (row, col) position via linear interpolation.

        Uses centroid per slice to handle multiple labeled voxels on the same slice.
        For INLINE axis: key=inline, value=(crossline, time).
        For CROSSLINE axis: key=crossline, value=(inline, time).
        Extrapolates the first/last known seed to the cube edges.
        """
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        if axis == Axis.INLINE:
            for pt in seed_points:
                groups[pt.inline].append((pt.crossline, pt.time))
        else:
            for pt in seed_points:
                groups[pt.crossline].append((pt.inline, pt.time))

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

    def _pad_to_multiple(
        self, arr: np.ndarray, factor: int = _PAD_MULTIPLE
    ) -> tuple[np.ndarray, int, int]:
        h, w = arr.shape
        ph = (factor - h % factor) % factor
        pw = (factor - w % factor) % factor
        return np.pad(arr, ((0, ph), (0, pw)), mode="reflect"), h, w

    def _run_model(
        self,
        seismic_2d: np.ndarray,
        hint_points: list[tuple[int, int]],
        sigma: float,
    ) -> np.ndarray:
        """Run ONNX model on one slice via sliding-window tiling, multi-point hint.

        Model was trained on _TILE_SIZE × _TILE_SIZE crops with single-point hint.
        At inference we build a multi-point hint over the full slice from
        hint_points (typically the predicted curve from the previous slice or
        the user's drawing on this slice). Each tile then crops the hint map.

        Tiles whose crop has no hint signal (the curve doesn't pass through this
        tile) get a fallback single-point hint at the closest curve point clamped
        to tile bounds — keeps every tile in-distribution as model expects.

        Falls back to a single padded forward pass when the slice fits one tile.
        """
        h_orig, w_orig = seismic_2d.shape
        seismic_norm = (seismic_2d.astype(np.float32) - 128.0) / 128.0

        hint_full = self._make_hint_multi(hint_points, (h_orig, w_orig), sigma)

        if h_orig <= _TILE_SIZE and w_orig <= _TILE_SIZE:
            return self._run_tile(seismic_norm, hint_full, h_orig, w_orig)

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

        return (prob_acc / np.maximum(weight_acc, 1e-6)).astype(np.float32)

    def _run_tile(
        self,
        seismic_norm: np.ndarray,
        hint: np.ndarray,
        h_orig: int,
        w_orig: int,
    ) -> np.ndarray:
        """Single forward pass on one (already-normalised) tile."""
        s_pad, _, _ = self._pad_to_multiple(seismic_norm)
        h_pad, _, _ = self._pad_to_multiple(hint)
        x = np.stack([s_pad, h_pad], axis=0)[np.newaxis]
        logits = self._session.run([self._output_name], {self._input_name: x})[0]
        probs = 1.0 / (1.0 + np.exp(-logits[0, 0].astype(np.float64)))
        return probs[:h_orig, :w_orig].astype(np.float32)

    @staticmethod
    def _tile_starts(size: int, tile: int, stride: int) -> list[int]:
        """Return tile start offsets covering [0, size) with the last tile flush right."""
        if size <= tile:
            return [0]
        starts = list(range(0, size - tile + 1, stride))
        if starts[-1] != size - tile:
            starts.append(size - tile)
        return starts

    @staticmethod
    def _build_user_curves(
        seed_points: list[Point3D],
        axis: Axis,
    ) -> dict[int, list[tuple[int, int]]]:
        """Group user seeds by slice index → list of (row, col) in slice coords.

        Unlike _build_seed_map (which collapses to one centroid per slice),
        this preserves every drawn pixel so a user-seeded slice gets a full
        multi-point hint, not a single point.
        """
        curves: dict[int, list[tuple[int, int]]] = defaultdict(list)
        if axis == Axis.INLINE:
            for pt in seed_points:
                curves[pt.inline].append((pt.crossline, pt.time))
        else:
            for pt in seed_points:
                curves[pt.crossline].append((pt.inline, pt.time))
        return dict(curves)

    @staticmethod
    def _build_interp_curves(
        user_curves: dict[int, list[tuple[int, int]]],
        n_slices: int,
        n_rows: int,
        extend_fn,
    ) -> dict[int, list[tuple[int, int]]]:
        """Per-slice expected curve via linear interp between user-drawn slices.

        For every slice index in [0, n_slices), produces a per-row (row, time)
        curve obtained by:
          - extending each user-drawn slice's curve to all rows
          - linearly interpolating between adjacent user slices' extended curves
          - holding boundary values for slices before the first / after the last
            user slice

        This is a stable "rail" that does not drift over slices and reflects the
        user's intent. Used both as a soft anchor for the propagated curve and
        as the reference for per-row consistency checks.
        """
        if not user_curves:
            return {}
        user_extended = {sl: extend_fn(pts, n_rows) for sl, pts in user_curves.items()}
        keys = sorted(user_extended)
        result: dict[int, list[tuple[int, int]]] = {}

        for sl in range(n_slices):
            if sl in user_extended:
                result[sl] = user_extended[sl]
                continue
            lower = max((k for k in keys if k < sl), default=None)
            upper = min((k for k in keys if k > sl), default=None)
            if lower is None and upper is None:
                continue
            if lower is None:
                result[sl] = user_extended[upper]
                continue
            if upper is None:
                result[sl] = user_extended[lower]
                continue
            t = (sl - lower) / (upper - lower)
            cl = user_extended[lower]
            cu = user_extended[upper]
            result[sl] = [
                (r, int(round((1 - t) * tl + t * tu)))
                for (r, tl), (_, tu) in zip(cl, cu)
            ]
        return result

    @staticmethod
    def _clamp_curve_to_anchor(
        curve: list[tuple[int, int]],
        anchor: list[tuple[int, int]],
        max_drift: int,
    ) -> list[tuple[int, int]]:
        """Clamp each row's time in `curve` to be within ±max_drift of `anchor`.

        Caps how far the propagated prediction curve is allowed to deviate from
        the user-anchored interpolated curve. Drift to a wrong reflector beyond
        max_drift is impossible — the next slice's hint will be pulled back.
        """
        if not anchor or not curve:
            return curve
        anchor_by_row = dict(anchor)
        result: list[tuple[int, int]] = []
        for r, t in curve:
            ta = anchor_by_row.get(r)
            if ta is None:
                result.append((r, t))
            else:
                result.append((r, int(np.clip(t, ta - max_drift, ta + max_drift))))
        return result

    @staticmethod
    def _extend_curve(
        points: list[tuple[int, int]],
        n_rows: int,
    ) -> list[tuple[int, int]]:
        """Extend a partial curve across all rows via linear interp + edge-constant
        extrapolation.

        Users typically draw on only part of a slice's width. Without extension,
        the multi-point hint lives only where the user drew → the model only
        predicts there → the horizon never propagates beyond the drawn extent.

        We linearly interpolate between drawn points and hold the boundary
        values for rows outside the drawn extent (np.interp default). The
        extrapolated portion of the hint is a "soft suggestion" — if there is
        a real horizon in the seismic at that location, the model predicts it;
        if not, low confidence falls below threshold and it produces nothing,
        so wrong extrapolation is harmless.
        """
        if not points:
            return []
        if len(points) == 1:
            col = points[0][1]
            return [(r, col) for r in range(n_rows)]
        sorted_pts = sorted(points, key=lambda p: p[0])
        rows = np.array([p[0] for p in sorted_pts], dtype=np.float32)
        cols = np.array([p[1] for p in sorted_pts], dtype=np.float32)
        out_rows = np.arange(n_rows)
        out_cols = np.interp(out_rows, rows, cols)
        return [(int(r), int(round(c))) for r, c in zip(out_rows, out_cols)]

    @staticmethod
    def _voxels_to_curve(
        new_voxels: list[tuple[int, int, int]],
        axis: Axis,
    ) -> list[tuple[int, int]]:
        """Project (inline, crossline, time) voxels to slice-coord (row, col)."""
        if axis == Axis.INLINE:
            return [(v[1], v[2]) for v in new_voxels]
        return [(v[0], v[2]) for v in new_voxels]

    def _make_hint_multi(
        self,
        points: list[tuple[int, int]],
        shape: tuple[int, int],
        sigma: float,
    ) -> np.ndarray:
        """Multi-point Gaussian hint: place 1.0 at each point, blur, normalise.

        The model was trained on single-point hints, but multi-point is just a
        sum of single-point Gaussians and locally (within the receptive field)
        looks identical to the training distribution at each peak.
        """
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

    @staticmethod
    def _select_curve_components(
        probs: np.ndarray,
        prev_curve: list[tuple[int, int]],
    ) -> np.ndarray:
        """Keep all CCs whose centroid lies near the expected curve at that row.

        The model often produces a horizon broken into several CCs by small
        per-pixel gaps in confidence. Keeping only one CC (closest to a single
        prev_pos) discards the rest of the line. We instead build a per-row
        expected time map from prev_curve and keep every CC whose centroid is
        within _CC_TIME_TOL of the expected time at the centroid's row.

        CCs from a parallel reflector at a different time are still rejected
        because their centroid time differs from the curve.
        """
        binary = (probs > _THRESHOLD).astype(np.uint8)
        if binary.sum() == 0:
            return probs
        labeled, n = nd_label(binary)
        if n == 0:
            return probs
        if not prev_curve:
            return probs

        H = probs.shape[0]
        curve_time = np.full(H, -1.0, dtype=np.float32)
        for r, c in prev_curve:
            if 0 <= r < H:
                curve_time[r] = c

        centroids = center_of_mass(binary, labeled, range(1, n + 1))

        keep_mask = np.zeros(probs.shape, dtype=np.float32)
        for cc_idx, (cy, cx) in enumerate(centroids):
            cy_int = int(round(cy))
            if 0 <= cy_int < H and curve_time[cy_int] >= 0:
                if abs(cx - curve_time[cy_int]) <= _CC_TIME_TOL:
                    keep_mask[labeled == cc_idx + 1] = 1.0
        if keep_mask.sum() == 0:
            sizes = np.bincount(labeled.ravel())[1:]
            best = int(np.argmax(sizes)) + 1
            keep_mask[labeled == best] = 1.0
        return probs * keep_mask

    @staticmethod
    def _select_main_component(
        probs: np.ndarray, prev_pos: tuple[int, int]
    ) -> np.ndarray:
        """Keep only the connected component closest to prev_pos.

        After thresholding, the model often produces several thin horizontal
        blobs — the target horizon plus weaker neighbouring reflectors. Picking
        only the CC nearest to the propagated hint position eliminates the
        scatter that would otherwise pass through the median consistency filter
        and produce vertical noise around the surface in 3D.

        If no CC's centroid is within _CC_MAX_DIST of prev_pos, returns zeros
        so _pick_horizon falls back to the seed_map for this slice.
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
        sizes = np.bincount(labeled.ravel())[1:]
        pr, pc = prev_pos
        dists = np.array([(c[0] - pr) ** 2 + (c[1] - pc) ** 2 for c in centroids])
        best_idx = int(np.argmin(dists))
        if dists[best_idx] > _CC_MAX_DIST ** 2:
            best_idx = int(np.argmax(sizes))
        keep = (labeled == best_idx + 1).astype(np.float32)
        return probs * keep

    @staticmethod
    def _anchor_to_seed(
        predicted: tuple[int, int],
        seed_pos: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Pull predicted hint position toward the user-seed-interpolated position.

        Pure prediction-based propagation accumulates drift across many slices.
        Linear interpolation of user seed points is the most reliable anchor we
        have, so blend it in with weight _SEED_ANCHOR_WEIGHT every slice.
        """
        if seed_pos is None:
            return predicted
        w = _SEED_ANCHOR_WEIGHT
        r = round((1.0 - w) * predicted[0] + w * seed_pos[0])
        c = round((1.0 - w) * predicted[1] + w * seed_pos[1])
        return (int(r), int(c))

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

    def _pick_horizon(
        self,
        probs: np.ndarray,
        slice_idx: int,
        axis: Axis,
        prev_curve: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int, int]], tuple[int, int] | None]:
        """Per-row anchor pick using prev_curve as the per-row expected time.

        For each row r:
          - Search window is ±_TRACKING_WINDOW around prev_curve[r].time (not a
            single global prev_pos[1]) — handles dipping/varying horizons.
          - Pick = weighted centroid of probs in that window.
          - Accept only if |pick - prev_curve[r].time| <= _CONSISTENCY.

        Anchoring to the curve per row prevents the global-median trap: when
        the model produces picks on two reflectors, the global median used to
        fall between them and accept both, letting drift creep in.

        Returns (voxels, new_pos) where new_pos is the centroid of accepted
        picks (used to anchor seed_map blending). Empty result → ([], None).
        """
        n_rows, n_cols = probs.shape

        expected_t = np.full(n_rows, -1, dtype=np.int32)
        for r, t in prev_curve:
            if 0 <= r < n_rows:
                expected_t[r] = int(t)

        voxels: list[tuple[int, int, int]] = []
        accepted_rows: list[int] = []
        accepted_times: list[int] = []

        for r in range(n_rows):
            ex_t = int(expected_t[r])
            if ex_t < 0:
                continue
            t0 = max(0, ex_t - _TRACKING_WINDOW)
            t1 = min(n_cols, ex_t + _TRACKING_WINDOW)
            if t1 <= t0:
                continue

            row_probs = probs[r, t0:t1]
            if row_probs.max() < _THRESHOLD:
                continue

            t_range = np.arange(t0, t1, dtype=np.float32)
            weight_sum = float(row_probs.sum())
            if weight_sum < 1e-6:
                continue
            t_centroid = int(round((row_probs * t_range).sum() / weight_sum))

            if abs(t_centroid - ex_t) > _CONSISTENCY:
                continue

            if axis == Axis.INLINE:
                voxels.append((slice_idx, r, t_centroid))
            else:
                voxels.append((r, slice_idx, t_centroid))
            accepted_rows.append(r)
            accepted_times.append(t_centroid)

        if not voxels:
            return [], None

        new_r = int(np.median(accepted_rows))
        new_t = int(np.median(accepted_times))
        return voxels, (new_r, new_t)


assert isinstance(OnnxTracker.__new__(OnnxTracker), HorizonTracker)
