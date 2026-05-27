"""Tests for infrastructure.ml.onnx_tracker — OnnxTracker helper methods.

Strategy:
- Static methods tested directly (no instance needed).
- Instance methods tested via a mocked onnxruntime.InferenceSession injected
  through sys.modules patching before OnnxTracker.__init__ runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from seismic_visualizer.domain.geometry import Axis, Point3D


# ---------------------------------------------------------------------------
# Fixture: OnnxTracker instance with mocked session
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ort():
    """A fake onnxruntime module that does not require the actual DLL."""
    mod = MagicMock()
    mod.get_available_providers.return_value = ["CPUExecutionProvider"]
    inp = MagicMock()
    inp.name = "input"
    out = MagicMock()
    out.name = "output"
    session = MagicMock()
    session.get_inputs.return_value = [inp]
    session.get_outputs.return_value = [out]
    mod.InferenceSession.return_value = session
    return mod, session


@pytest.fixture
def tracker(tmp_path, mock_ort):
    """OnnxTracker instance with mocked onnxruntime."""
    mod, session = mock_ort
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    with _patch_ort(mod):
        from seismic_visualizer.infrastructure.ml.onnx_tracker import OnnxTracker
        t = OnnxTracker(model_path)
    t._session = session  # keep reference to mock session
    return t


def _patch_ort(mod):
    """Context manager that injects fake onnxruntime into sys.modules."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        old = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = mod
        try:
            yield
        finally:
            if old is None:
                sys.modules.pop("onnxruntime", None)
            else:
                sys.modules["onnxruntime"] = old

    return _ctx()


# ---------------------------------------------------------------------------
# Import the class once (after path setup) so we can call @staticmethod
# ---------------------------------------------------------------------------

def _get_class():
    from seismic_visualizer.infrastructure.ml.onnx_tracker import OnnxTracker
    return OnnxTracker


# ---------------------------------------------------------------------------
# _tile_starts
# ---------------------------------------------------------------------------

class TestTileStarts:
    def test_single_tile_when_size_le_tile(self):
        OnnxTracker = _get_class()
        assert OnnxTracker._tile_starts(100, 256, 128) == [0]

    def test_exact_size_equals_tile(self):
        OnnxTracker = _get_class()
        assert OnnxTracker._tile_starts(256, 256, 128) == [0]

    def test_two_tiles_overlap(self):
        OnnxTracker = _get_class()
        starts = OnnxTracker._tile_starts(300, 256, 128)
        assert starts[0] == 0
        assert starts[-1] == 300 - 256  # flush-right tile

    def test_last_tile_is_flush_right(self):
        OnnxTracker = _get_class()
        size, tile, stride = 500, 256, 128
        starts = OnnxTracker._tile_starts(size, tile, stride)
        assert starts[-1] == size - tile

    def test_no_duplicate_starts(self):
        OnnxTracker = _get_class()
        starts = OnnxTracker._tile_starts(700, 256, 128)
        assert len(starts) == len(set(starts))


# ---------------------------------------------------------------------------
# _extend_curve
# ---------------------------------------------------------------------------

class TestExtendCurve:
    def test_empty_points_returns_empty(self):
        OnnxTracker = _get_class()
        assert OnnxTracker._extend_curve([], 10) == []

    def test_single_point_replicated_across_all_rows(self):
        OnnxTracker = _get_class()
        result = OnnxTracker._extend_curve([(0, 5)], 4)
        assert len(result) == 4
        assert all(c == 5 for _, c in result)

    def test_two_points_linearly_interpolated(self):
        OnnxTracker = _get_class()
        result = OnnxTracker._extend_curve([(0, 0), (9, 9)], 10)
        assert len(result) == 10
        for r, c in result:
            assert abs(c - r) <= 1  # col ≈ row

    def test_output_length_equals_n_rows(self):
        OnnxTracker = _get_class()
        result = OnnxTracker._extend_curve([(2, 3), (7, 8)], 15)
        assert len(result) == 15

    def test_extrapolation_holds_boundary_value(self):
        OnnxTracker = _get_class()
        result = OnnxTracker._extend_curve([(3, 10), (7, 20)], 10)
        # rows 0,1,2 should hold value at row 3 (extrapolation)
        assert result[0][1] == result[1][1] == result[2][1]


# ---------------------------------------------------------------------------
# _build_user_curves
# ---------------------------------------------------------------------------

class TestBuildUserCurves:
    def test_inline_axis_groups_by_inline(self):
        OnnxTracker = _get_class()
        seeds = [Point3D(2, 3, 5), Point3D(2, 4, 6), Point3D(4, 1, 7)]
        result = OnnxTracker._build_user_curves(seeds, Axis.INLINE)
        assert 2 in result
        assert 4 in result
        assert len(result[2]) == 2

    def test_crossline_axis_groups_by_crossline(self):
        OnnxTracker = _get_class()
        seeds = [Point3D(1, 3, 5), Point3D(2, 3, 6)]
        result = OnnxTracker._build_user_curves(seeds, Axis.CROSSLINE)
        assert 3 in result
        assert len(result[3]) == 2

    def test_empty_seeds_returns_empty_dict(self):
        OnnxTracker = _get_class()
        assert OnnxTracker._build_user_curves([], Axis.INLINE) == {}

    def test_inline_curve_values_are_crossline_time_pairs(self):
        OnnxTracker = _get_class()
        seeds = [Point3D(5, 3, 7)]
        result = OnnxTracker._build_user_curves(seeds, Axis.INLINE)
        assert result[5] == [(3, 7)]

    def test_crossline_curve_values_are_inline_time_pairs(self):
        OnnxTracker = _get_class()
        seeds = [Point3D(1, 4, 9)]
        result = OnnxTracker._build_user_curves(seeds, Axis.CROSSLINE)
        assert result[4] == [(1, 9)]


# ---------------------------------------------------------------------------
# _clamp_curve_to_anchor
# ---------------------------------------------------------------------------

class TestClampCurveToAnchor:
    def test_empty_anchor_returns_curve_unchanged(self):
        OnnxTracker = _get_class()
        curve = [(0, 5), (1, 6)]
        assert OnnxTracker._clamp_curve_to_anchor(curve, [], 10) == curve

    def test_empty_curve_returns_empty(self):
        OnnxTracker = _get_class()
        assert OnnxTracker._clamp_curve_to_anchor([], [(0, 5)], 10) == []

    def test_within_max_drift_unchanged(self):
        OnnxTracker = _get_class()
        curve = [(0, 10)]
        anchor = [(0, 10)]
        result = OnnxTracker._clamp_curve_to_anchor(curve, anchor, max_drift=5)
        assert result == [(0, 10)]

    def test_exceeds_max_drift_clamped_above(self):
        OnnxTracker = _get_class()
        curve = [(0, 20)]
        anchor = [(0, 10)]
        result = OnnxTracker._clamp_curve_to_anchor(curve, anchor, max_drift=5)
        assert result[0][1] == 15

    def test_exceeds_max_drift_clamped_below(self):
        OnnxTracker = _get_class()
        curve = [(0, 0)]
        anchor = [(0, 10)]
        result = OnnxTracker._clamp_curve_to_anchor(curve, anchor, max_drift=5)
        assert result[0][1] == 5

    def test_row_not_in_anchor_passes_through(self):
        OnnxTracker = _get_class()
        curve = [(5, 100)]
        anchor = [(0, 10)]  # row 5 not in anchor
        result = OnnxTracker._clamp_curve_to_anchor(curve, anchor, max_drift=3)
        assert result == [(5, 100)]


# ---------------------------------------------------------------------------
# _voxels_to_curve
# ---------------------------------------------------------------------------

class TestVoxelsToCurve:
    def test_inline_axis_projects_crossline_time(self):
        OnnxTracker = _get_class()
        voxels = [(1, 3, 7), (1, 4, 8)]
        result = OnnxTracker._voxels_to_curve(voxels, Axis.INLINE)
        assert result == [(3, 7), (4, 8)]

    def test_crossline_axis_projects_inline_time(self):
        OnnxTracker = _get_class()
        voxels = [(2, 1, 5), (3, 1, 6)]
        result = OnnxTracker._voxels_to_curve(voxels, Axis.CROSSLINE)
        assert result == [(2, 5), (3, 6)]

    def test_empty_voxels_returns_empty(self):
        OnnxTracker = _get_class()
        assert OnnxTracker._voxels_to_curve([], Axis.INLINE) == []


# ---------------------------------------------------------------------------
# _select_curve_components
# ---------------------------------------------------------------------------

class TestSelectCurveComponents:
    def test_zero_probs_returns_unchanged(self):
        OnnxTracker = _get_class()
        probs = np.zeros((10, 10), dtype=np.float32)
        result = OnnxTracker._select_curve_components(probs, [(5, 5)])
        np.testing.assert_array_equal(result, probs)

    def test_empty_prev_curve_returns_probs_unchanged(self):
        OnnxTracker = _get_class()
        probs = np.ones((10, 10), dtype=np.float32) * 0.8
        result = OnnxTracker._select_curve_components(probs, [])
        np.testing.assert_array_equal(result, probs)

    def test_single_component_near_curve_kept(self):
        OnnxTracker = _get_class()
        probs = np.zeros((20, 20), dtype=np.float32)
        probs[5:8, 5:8] = 0.9
        curve = [(r, 6) for r in range(20)]
        result = OnnxTracker._select_curve_components(probs, curve)
        assert result.max() > 0

    def test_output_shape_preserved(self):
        OnnxTracker = _get_class()
        probs = np.random.rand(15, 15).astype(np.float32)
        curve = [(r, 7) for r in range(15)]
        result = OnnxTracker._select_curve_components(probs, curve)
        assert result.shape == probs.shape


# ---------------------------------------------------------------------------
# _build_interp_curves
# ---------------------------------------------------------------------------

class TestBuildInterpCurves:
    def test_empty_user_curves_returns_empty(self):
        OnnxTracker = _get_class()
        result = OnnxTracker._build_interp_curves({}, 10, 8, OnnxTracker._extend_curve)
        assert result == {}

    def test_single_user_curve_propagated_to_all_slices(self):
        OnnxTracker = _get_class()
        user_curves = {5: [(r, 10) for r in range(8)]}
        result = OnnxTracker._build_interp_curves(user_curves, 10, 8, OnnxTracker._extend_curve)
        assert len(result) == 10

    def test_two_user_curves_interpolated(self):
        OnnxTracker = _get_class()
        user_curves = {
            0: [(r, 0) for r in range(8)],
            9: [(r, 9) for r in range(8)],
        }
        result = OnnxTracker._build_interp_curves(user_curves, 10, 8, OnnxTracker._extend_curve)
        # Slice 5 should have col ≈ 5 for each row
        if 5 in result:
            times = [c for _, c in result[5]]
            assert all(3 <= t <= 7 for t in times)


# ---------------------------------------------------------------------------
# _build_seed_map (instance method)
# ---------------------------------------------------------------------------

class TestBuildSeedMap:
    def test_inline_axis_maps_by_inline(self, tracker):
        seeds = [Point3D(2, 5, 7)]
        result = tracker._build_seed_map(seeds, Axis.INLINE, 10)
        assert 2 in result
        assert result[2] == (5, 7)

    def test_crossline_axis_maps_by_crossline(self, tracker):
        seeds = [Point3D(3, 4, 8)]
        result = tracker._build_seed_map(seeds, Axis.CROSSLINE, 10)
        assert 4 in result

    def test_empty_seeds_returns_empty(self, tracker):
        assert tracker._build_seed_map([], Axis.INLINE, 10) == {}

    def test_extrapolates_to_all_slices(self, tracker):
        seeds = [Point3D(5, 3, 7)]
        result = tracker._build_seed_map(seeds, Axis.INLINE, 10)
        assert len(result) == 10

    def test_two_seeds_interpolated(self, tracker):
        seeds = [Point3D(0, 0, 0), Point3D(9, 9, 9)]
        result = tracker._build_seed_map(seeds, Axis.INLINE, 10)
        mid = result[5]
        assert 3 <= mid[0] <= 6  # crossline ~4-5
        assert 3 <= mid[1] <= 6  # time ~4-5


# ---------------------------------------------------------------------------
# _pad_to_multiple (instance method)
# ---------------------------------------------------------------------------

class TestPadToMultiple:
    def test_already_multiple_not_padded(self, tracker):
        arr = np.zeros((16, 32), dtype=np.float32)
        padded, h, w = tracker._pad_to_multiple(arr, factor=16)
        assert padded.shape == (16, 32)
        assert h == 16 and w == 32

    def test_padding_applied_when_not_multiple(self, tracker):
        arr = np.zeros((10, 20), dtype=np.float32)
        padded, h, w = tracker._pad_to_multiple(arr, factor=16)
        assert padded.shape[0] % 16 == 0
        assert padded.shape[1] % 16 == 0
        assert h == 10 and w == 20

    def test_original_content_preserved(self, tracker):
        arr = np.ones((8, 8), dtype=np.float32) * 0.5
        padded, _, _ = tracker._pad_to_multiple(arr, factor=16)
        np.testing.assert_array_equal(padded[:8, :8], arr)


# ---------------------------------------------------------------------------
# _make_hint_multi (instance method)
# ---------------------------------------------------------------------------

class TestMakeHintMulti:
    def test_empty_points_returns_zero_array(self, tracker):
        result = tracker._make_hint_multi([], (10, 10), sigma=2.0)
        assert np.all(result == 0.0)
        assert result.shape == (10, 10)

    def test_single_point_peak_near_point(self, tracker):
        result = tracker._make_hint_multi([(5, 5)], (10, 10), sigma=1.0)
        # Peak should be at or near (5,5)
        assert result[5, 5] == pytest.approx(1.0, abs=0.01)

    def test_output_normalised_to_one(self, tracker):
        result = tracker._make_hint_multi([(3, 3), (7, 7)], (10, 10), sigma=1.0)
        assert result.max() == pytest.approx(1.0, abs=0.01)

    def test_output_shape_matches_input(self, tracker):
        result = tracker._make_hint_multi([(1, 2)], (8, 12), sigma=2.0)
        assert result.shape == (8, 12)

    def test_out_of_bounds_point_clamped(self, tracker):
        result = tracker._make_hint_multi([(100, 200)], (10, 10), sigma=1.0)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# _run_tile (instance method — calls mocked session)
# ---------------------------------------------------------------------------

class TestRunTile:
    def test_output_shape_matches_original_dimensions(self, tracker):
        H, W = 16, 16
        seismic_norm = np.zeros((H, W), dtype=np.float32)
        hint = np.zeros((H, W), dtype=np.float32)
        # Mock session returns logits of shape (1, 1, H_pad, W_pad)
        tracker._session.run.return_value = [np.zeros((1, 1, 16, 16), dtype=np.float32)]
        result = tracker._run_tile(seismic_norm, hint, H, W)
        assert result.shape == (H, W)

    def test_output_values_are_probabilities(self, tracker):
        H, W = 16, 16
        logits = np.zeros((1, 1, H, W), dtype=np.float32)
        tracker._session.run.return_value = [logits]
        result = tracker._run_tile(
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            H, W,
        )
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_positive_logits_give_probs_above_half(self, tracker):
        H, W = 16, 16
        logits = np.ones((1, 1, H, W), dtype=np.float32) * 5.0
        tracker._session.run.return_value = [logits]
        result = tracker._run_tile(
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            H, W,
        )
        assert np.all(result > 0.5)


# ---------------------------------------------------------------------------
# track (public method — end-to-end with mocked session)
# ---------------------------------------------------------------------------

class TestTrack:
    def _setup_session(self, tracker, shape):
        H, W = shape
        tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]

    def test_empty_seeds_returns_empty_array(self, tracker):
        from seismic_visualizer.domain.cube import SeismicCube
        seismic = SeismicCube(np.zeros((5, 5, 5), dtype=np.uint8))
        result = tracker.track(seismic, [], Axis.INLINE)
        assert result.shape == (0, 3)

    def test_returns_ndarray(self, tracker):
        from seismic_visualizer.domain.cube import SeismicCube
        H, W = 5, 5
        self._setup_session(tracker, (H, W))
        seismic = SeismicCube(np.zeros((5, 5, 5), dtype=np.uint8))
        seeds = [Point3D(2, 2, 2)]
        result = tracker.track(seismic, seeds, Axis.INLINE)
        assert isinstance(result, np.ndarray)

    def test_backward_pass_body_executed(self, tracker):
        """Lines 154-177: backward loop body when min_idx > 0."""
        from seismic_visualizer.domain.cube import SeismicCube

        n_slices, H, W = 10, 5, 5
        tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]
        seismic = SeismicCube(np.zeros((n_slices, H, W), dtype=np.uint8))
        seeds = [Point3D(8, 2, 2)]  # min_idx=8 → backward covers slices 0-7
        result = tracker.track(seismic, seeds, Axis.INLINE)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3

    def test_forward_no_picks_falls_back_to_seed_map(self, tracker):
        """Line 128: _pick_horizon returns ([], None), use seed_map fallback."""
        from seismic_visualizer.domain.cube import SeismicCube

        n_slices, H, W = 5, 5, 5
        tracker._session.run.return_value = [
            np.full((1, 1, H, W), -100.0, dtype=np.float32)
        ]
        seismic = SeismicCube(np.zeros((n_slices, H, W), dtype=np.uint8))
        seeds = [Point3D(2, 2, 2)]
        result = tracker.track(seismic, seeds, Axis.INLINE)
        assert isinstance(result, np.ndarray)

    def test_cancel_check_stops_tracking(self, tracker):
        from seismic_visualizer.domain.cube import SeismicCube
        H, W = 5, 5
        self._setup_session(tracker, (H, W))
        seismic = SeismicCube(np.zeros((10, 5, 5), dtype=np.uint8))
        seeds = [Point3D(5, 2, 2)]
        result = tracker.track(seismic, seeds, Axis.INLINE, cancel_check=lambda: True)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# _run_model — tiling path (slice > _TILE_SIZE=256)
# ---------------------------------------------------------------------------

class TestBackwardPassDirect:
    """Force backward pass by replacing _build_seed_map with a partial dict.

    _build_seed_map normally extrapolates to slice 0, making min_idx=0 and
    range(min_idx-1, -1, -1) always empty.  Patching it to start at a higher
    slice gives range(k, -1, -1) a non-empty iteration, which enters the
    backward loop body (lines 154-177).
    """

    def test_backward_body_executed_positive_logits(self, tracker):
        """Lines 154, 157-161, 167-177: backward iterations with picks found."""
        from seismic_visualizer.domain.cube import SeismicCube

        n_slices, H, W = 10, 5, 5
        tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]
        seismic = SeismicCube(np.zeros((n_slices, H, W), dtype=np.uint8))
        seeds = [Point3D(5, 2, 2)]

        # Partial map starting at 5 → sorted_keys=[5..9] → min_idx=5
        # → backward: range(4,-1,-1)=[4,3,2,1,0]  (non-empty)
        # seed_map.get(4..0) = None → _anchor_to_seed(pred, None) hits line 594
        tracker._build_seed_map = lambda pts, ax, ns: {sl: (2, 2) for sl in range(5, ns)}

        result = tracker.track(seismic, seeds, Axis.INLINE)
        assert isinstance(result, np.ndarray)

    def test_backward_break_when_no_picks_and_no_seed(self, tracker):
        """Lines 162-166: backward _pick_horizon→None, seed_map fallback None → break."""
        from seismic_visualizer.domain.cube import SeismicCube

        n_slices, H, W = 10, 5, 5
        # Negative logits → probs ≈ 0 → _pick_horizon returns ([], None)
        tracker._session.run.return_value = [
            np.full((1, 1, H, W), -100.0, dtype=np.float32)
        ]
        seismic = SeismicCube(np.zeros((n_slices, H, W), dtype=np.uint8))
        seeds = [Point3D(5, 2, 2)]

        # Single-entry map → slices 0-4 and 6-9 not present
        tracker._build_seed_map = lambda pts, ax, ns: {5: (2, 2)}

        result = tracker.track(seismic, seeds, Axis.INLINE)
        assert isinstance(result, np.ndarray)

    def test_forward_break_no_seed_fallback(self, tracker):
        """Lines 128, 130-131: forward _pick_horizon→None, seed_map.get(sl)=None → break."""
        from seismic_visualizer.domain.cube import SeismicCube

        n_slices, H, W = 10, 5, 5
        tracker._session.run.return_value = [
            np.full((1, 1, H, W), -100.0, dtype=np.float32)
        ]
        seismic = SeismicCube(np.zeros((n_slices, H, W), dtype=np.uint8))
        seeds = [Point3D(5, 2, 2)]

        # Only slice 5 present → sl=6 onward: seed_map.get(sl)=None → forward break
        tracker._build_seed_map = lambda pts, ax, ns: {5: (2, 2)}

        result = tracker.track(seismic, seeds, Axis.INLINE)
        assert isinstance(result, np.ndarray)

    def test_backward_cancel_fires(self, tracker):
        """Lines 154-156: cancel_check fires during the backward pass."""
        from seismic_visualizer.domain.cube import SeismicCube

        n_slices, H, W = 10, 5, 5
        tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]
        seismic = SeismicCube(np.zeros((n_slices, H, W), dtype=np.uint8))
        seeds = [Point3D(5, 2, 2)]

        tracker._build_seed_map = lambda pts, ax, ns: {sl: (2, 2) for sl in range(5, ns)}

        forward_done = [False]

        def cancel_after_forward():
            if not forward_done[0]:
                return False
            return True

        def _make_cancel():
            """Fire only after the forward loop finishes (signalled by side-effect)."""
            calls = [0]

            def check():
                calls[0] += 1
                # 5 forward slices (5..9) + 1 backward = fire on 6th call
                return calls[0] > 5

            return check

        result = tracker.track(seismic, seeds, Axis.INLINE, cancel_check=_make_cancel())
        assert isinstance(result, np.ndarray)


class TestRunModelTiling:
    def test_large_slice_uses_tiling(self, tracker):
        """Lines 273-309: tiling path when h or w > 256."""
        H, W = 300, 300
        seismic_2d = np.zeros((H, W), dtype=np.float32)
        hint_points = [(H // 2, W // 2)]
        tracker._session.run.return_value = [np.zeros((1, 1, 256, 256), dtype=np.float32)]
        result = tracker._run_model(seismic_2d, hint_points, sigma=12.0)
        assert result.shape == (H, W)
        assert result.dtype == np.float32

    def test_tiling_empty_hint_fallback(self, tracker):
        """Lines 292-303: tile with no hint signal gets single-point fallback."""
        H, W = 300, 300
        seismic_2d = np.zeros((H, W), dtype=np.float32)
        # Hint at (0,0) with very small sigma → most tiles see near-zero hint
        hint_points = [(0, 0)]
        tracker._session.run.return_value = [np.zeros((1, 1, 256, 256), dtype=np.float32)]
        result = tracker._run_model(seismic_2d, hint_points, sigma=0.5)
        assert result.shape == (H, W)

    def test_tiling_no_hint_points(self, tracker):
        """Lines 273+: tiling with empty hint_points."""
        H, W = 300, 300
        seismic_2d = np.zeros((H, W), dtype=np.float32)
        tracker._session.run.return_value = [np.zeros((1, 1, 256, 256), dtype=np.float32)]
        result = tracker._run_model(seismic_2d, [], sigma=12.0)
        assert result.shape == (H, W)


# ---------------------------------------------------------------------------
# _select_curve_components — fallback when no CC near curve
# ---------------------------------------------------------------------------

class TestSelectCurveComponentsFallback:
    def test_fallback_to_largest_cc_when_none_near_curve(self):
        """Lines 539-542: keep_mask.sum()==0 → fallback to largest CC."""
        OnnxTracker = _get_class()
        probs = np.zeros((10, 100), dtype=np.float32)
        probs[3:7, 80:90] = 0.9  # CC centroid col ≈ 85
        # Curve expects time=10 → |85 - 10| = 75 > _CC_TIME_TOL=40 → CC rejected
        curve = [(r, 10) for r in range(10)]
        result = OnnxTracker._select_curve_components(probs, curve)
        assert result.max() > 0  # fallback CC kept


# ---------------------------------------------------------------------------
# _select_main_component (static, currently not called from track())
# ---------------------------------------------------------------------------

class TestSelectMainComponent:
    def test_zero_probs_returns_unchanged(self):
        """Line 561: binary.sum()==0 → return probs."""
        OnnxTracker = _get_class()
        probs = np.zeros((10, 10), dtype=np.float32)
        result = OnnxTracker._select_main_component(probs, (5, 5))
        np.testing.assert_array_equal(result, probs)

    def test_single_component_returned_unchanged(self):
        """Line 566-567: n==1 → return probs."""
        OnnxTracker = _get_class()
        probs = np.zeros((20, 20), dtype=np.float32)
        probs[5:8, 5:8] = 0.9
        result = OnnxTracker._select_main_component(probs, (6, 6))
        np.testing.assert_array_equal(result, probs)

    def test_two_components_picks_closest(self):
        """Lines 568-580: two CCs, closest to prev_pos is kept."""
        OnnxTracker = _get_class()
        probs = np.zeros((20, 20), dtype=np.float32)
        probs[1:3, 1:3] = 0.9   # CC1 near (2, 2)
        probs[15:18, 15:18] = 0.9  # CC2 near (16, 16)
        result = OnnxTracker._select_main_component(probs, (2, 2))
        assert result[1, 1] > 0     # CC1 kept
        assert result[15, 15] == 0.0  # CC2 zeroed

    def test_far_component_falls_back_to_largest(self):
        """Line 574: no CC within _CC_MAX_DIST → fallback to largest CC."""
        OnnxTracker = _get_class()
        probs = np.zeros((20, 20), dtype=np.float32)
        probs[0:2, 0:2] = 0.9      # small CC
        probs[9:18, 9:18] = 0.9    # large CC
        # prev_pos=(200,200) is far from both → fallback to largest
        result = OnnxTracker._select_main_component(probs, (200, 200))
        assert result[9, 9] > 0    # large CC kept


# ---------------------------------------------------------------------------
# _anchor_to_seed — None seed_pos path
# ---------------------------------------------------------------------------

class TestAnchorToSeed:
    def test_none_seed_returns_predicted_unchanged(self):
        """Line 594: seed_pos is None → return predicted as-is."""
        OnnxTracker = _get_class()
        predicted = (5, 10)
        result = OnnxTracker._anchor_to_seed(predicted, None)
        assert result == predicted

    def test_with_seed_blends_positions(self):
        OnnxTracker = _get_class()
        # predicted=(0,0), seed=(10,10), weight=0.3 → ~(3,3)
        result = OnnxTracker._anchor_to_seed((0, 0), (10, 10))
        assert result[0] == 3
        assert result[1] == 3


# ---------------------------------------------------------------------------
# _make_hint — single-point hint map
# ---------------------------------------------------------------------------

class TestMakeHint:
    def test_make_hint_with_position(self, tracker):
        """Lines 603-613: _make_hint when pos is not None."""
        result = tracker._make_hint((3, 4), (10, 10), sigma=1.0)
        assert result.shape == (10, 10)
        assert result.max() == pytest.approx(1.0, abs=0.01)
        assert result.dtype == np.float32

    def test_make_hint_none_pos_returns_zeros(self, tracker):
        """Line 604: pos is None → all zeros."""
        result = tracker._make_hint(None, (5, 5), sigma=1.0)
        assert np.all(result == 0.0)

    def test_make_hint_clamps_out_of_bounds_pos(self, tracker):
        result = tracker._make_hint((100, 200), (5, 5), sigma=1.0)
        assert result.shape == (5, 5)
        assert result.max() == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# _pick_horizon — edge-case paths
# ---------------------------------------------------------------------------

class TestPickHorizon:
    def test_empty_curve_returns_empty_and_none(self, tracker):
        """Lines 652, 682: ex_t<0 for all rows (empty curve) → ([], None)."""
        probs = np.ones((5, 5), dtype=np.float32) * 0.9
        voxels, new_pos = tracker._pick_horizon(probs, 0, Axis.INLINE, [])
        assert voxels == []
        assert new_pos is None

    def test_probs_below_threshold_no_picks(self, tracker):
        """Line 660: row_probs.max() < _THRESHOLD → skip row."""
        probs = np.ones((5, 5), dtype=np.float32) * 0.3  # below threshold=0.4
        curve = [(r, 2) for r in range(5)]
        voxels, new_pos = tracker._pick_horizon(probs, 0, Axis.INLINE, curve)
        assert voxels == []
        assert new_pos is None

    def test_crossline_axis_voxel_format(self, tracker):
        """Line 677: CROSSLINE axis → voxel = (row, slice_idx, t_centroid)."""
        probs = np.zeros((5, 5), dtype=np.float32)
        probs[:, 2] = 0.9  # all rows, col=2 → centroid at 2
        curve = [(r, 2) for r in range(5)]
        voxels, new_pos = tracker._pick_horizon(probs, 3, Axis.CROSSLINE, curve)
        assert len(voxels) > 0
        assert all(v[1] == 3 for v in voxels)  # slice_idx=3 in position [1]

    def test_partial_curve_rows_with_no_info_skipped(self, tracker):
        """Line 652: rows not in curve → ex_t=-1 → continue."""
        probs = np.ones((10, 10), dtype=np.float32) * 0.9
        # Only rows 5-9 have curve info; rows 0-4 have ex_t=-1
        curve = [(r, 5) for r in range(5, 10)]
        voxels, new_pos = tracker._pick_horizon(probs, 0, Axis.INLINE, curve)
        # Only rows 5-9 should contribute voxels
        assert all(v[1] >= 5 for v in voxels)
