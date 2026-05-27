"""Tests for infrastructure.ml.onnx_fault_tracker — OnnxFaultTracker helper methods.

Strategy identical to test_onnx_tracker.py:
- @staticmethod methods tested directly on the class.
- Instance methods tested via mocked onnxruntime injected through sys.modules.
"""

from __future__ import annotations

import sys
import contextlib
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from seismic_visualizer.domain.geometry import Axis, Point3D


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_ort(mod):
    old = sys.modules.get("onnxruntime")
    sys.modules["onnxruntime"] = mod
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = old


def _make_mock_ort():
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
def fault_tracker(tmp_path):
    mod, session = _make_mock_ort()
    model_path = tmp_path / "fault_model.onnx"
    model_path.touch()
    with _patch_ort(mod):
        from seismic_visualizer.infrastructure.ml.onnx_fault_tracker import OnnxFaultTracker
        t = OnnxFaultTracker(model_path)
    t._session = session
    return t


def _cls():
    from seismic_visualizer.infrastructure.ml.onnx_fault_tracker import OnnxFaultTracker
    return OnnxFaultTracker


# ---------------------------------------------------------------------------
# _tile_starts  (same logic as onnx_tracker but lives on OnnxFaultTracker)
# ---------------------------------------------------------------------------

class TestFaultTileStarts:
    def test_small_size_returns_single_zero(self):
        assert _cls()._tile_starts(100, 256, 128) == [0]

    def test_exact_tile_size_returns_zero(self):
        assert _cls()._tile_starts(256, 256, 128) == [0]

    def test_last_start_is_flush_right(self):
        size, tile, stride = 400, 256, 128
        starts = _cls()._tile_starts(size, tile, stride)
        assert starts[-1] == size - tile

    def test_no_duplicates(self):
        starts = _cls()._tile_starts(600, 256, 128)
        assert len(starts) == len(set(starts))


# ---------------------------------------------------------------------------
# _build_user_points
# ---------------------------------------------------------------------------

class TestBuildUserPoints:
    def test_inline_axis_maps_crossline_time(self):
        seeds = [Point3D(3, 5, 7)]
        result = _cls()._build_user_points(seeds, Axis.INLINE)
        assert 3 in result
        assert result[3] == [(5, 7)]

    def test_crossline_axis_maps_inline_time(self):
        seeds = [Point3D(1, 4, 9)]
        result = _cls()._build_user_points(seeds, Axis.CROSSLINE)
        assert 4 in result
        assert result[4] == [(1, 9)]

    def test_time_axis_maps_inline_crossline(self):
        seeds = [Point3D(2, 3, 8)]
        result = _cls()._build_user_points(seeds, Axis.TIME)
        assert 8 in result
        assert result[8] == [(2, 3)]

    def test_empty_seeds_returns_empty(self):
        assert _cls()._build_user_points([], Axis.INLINE) == {}

    def test_multiple_seeds_same_slice_grouped(self):
        seeds = [Point3D(5, 1, 2), Point3D(5, 3, 4)]
        result = _cls()._build_user_points(seeds, Axis.INLINE)
        assert len(result[5]) == 2


# ---------------------------------------------------------------------------
# _should_stop
# ---------------------------------------------------------------------------

class TestShouldStop:
    def test_zero_probs_should_stop(self):
        probs = np.zeros((10, 10), dtype=np.float32)
        assert _cls()._should_stop(probs) is True

    def test_high_confidence_should_not_stop(self):
        probs = np.ones((10, 10), dtype=np.float32) * 0.9
        assert _cls()._should_stop(probs) is False

    def test_max_below_threshold_should_stop(self):
        probs = np.ones((10, 10), dtype=np.float32) * 0.3
        assert _cls()._should_stop(probs) is True

    def test_too_few_pixels_above_threshold(self):
        probs = np.zeros((20, 20), dtype=np.float32)
        probs[0, 0] = 0.9  # only 1 pixel > 0.4, less than _STOP_MIN_PIXELS=8
        assert _cls()._should_stop(probs) is True


# ---------------------------------------------------------------------------
# _subsample_user_hint
# ---------------------------------------------------------------------------

class TestSubsampleUserHint:
    def test_empty_returns_empty(self):
        assert _cls()._subsample_user_hint([]) == []

    def test_single_point_returns_it(self):
        pts = [(5, 3)]
        assert _cls()._subsample_user_hint(pts) == [(5, 3)]

    def test_output_length_bounded_by_max_points(self):
        pts = [(r, r) for r in range(100)]
        result = _cls()._subsample_user_hint(pts, spacing=6, max_points=12)
        assert len(result) <= 12

    def test_output_is_subset_of_input(self):
        pts = [(r, r * 2) for r in range(50)]
        result = _cls()._subsample_user_hint(pts, spacing=10, max_points=8)
        for p in result:
            assert p in pts

    def test_two_points_returns_both(self):
        pts = [(0, 0), (10, 10)]
        result = _cls()._subsample_user_hint(pts)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# _sample_hint_from_pixels
# ---------------------------------------------------------------------------

class TestSampleHintFromPixels:
    def test_empty_pixels_returns_fallback(self):
        fallback = (5, 5)
        result = _cls()._sample_hint_from_pixels(np.empty((0, 2), dtype=np.int32), fallback)
        assert fallback in result

    def test_output_length_bounded_by_n(self):
        pixels = np.array([[r, c] for r in range(20) for c in range(5)], dtype=np.int32)
        result = _cls()._sample_hint_from_pixels(pixels, (10, 2), n=4)
        assert len(result) <= 4

    def test_single_pixel_returns_it(self):
        pixels = np.array([[3, 7]], dtype=np.int32)
        result = _cls()._sample_hint_from_pixels(pixels, (3, 7))
        assert (3, 7) in result


# ---------------------------------------------------------------------------
# _build_seed_map (instance method)
# ---------------------------------------------------------------------------

class TestFaultBuildSeedMap:
    def test_inline_axis(self, fault_tracker):
        seeds = [Point3D(3, 5, 7)]
        result = fault_tracker._build_seed_map(seeds, Axis.INLINE, 10)
        assert 3 in result
        assert result[3] == (5, 7)

    def test_crossline_axis(self, fault_tracker):
        seeds = [Point3D(2, 4, 8)]
        result = fault_tracker._build_seed_map(seeds, Axis.CROSSLINE, 10)
        assert 4 in result

    def test_time_axis(self, fault_tracker):
        seeds = [Point3D(1, 2, 6)]
        result = fault_tracker._build_seed_map(seeds, Axis.TIME, 10)
        assert 6 in result
        assert result[6] == (1, 2)

    def test_empty_returns_empty(self, fault_tracker):
        assert fault_tracker._build_seed_map([], Axis.INLINE, 10) == {}

    def test_extrapolates_to_all_slices(self, fault_tracker):
        seeds = [Point3D(5, 3, 7)]
        result = fault_tracker._build_seed_map(seeds, Axis.INLINE, 10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _pad_to_multiple (instance method)
# ---------------------------------------------------------------------------

class TestFaultPadToMultiple:
    def test_already_multiple_no_padding(self, fault_tracker):
        arr = np.zeros((16, 32), dtype=np.float32)
        padded, h, w = fault_tracker._pad_to_multiple(arr, factor=16)
        assert padded.shape == (16, 32)

    def test_padding_rounds_up(self, fault_tracker):
        arr = np.zeros((10, 10), dtype=np.float32)
        padded, h, w = fault_tracker._pad_to_multiple(arr, factor=16)
        assert padded.shape[0] % 16 == 0
        assert padded.shape[1] % 16 == 0

    def test_original_dims_returned(self, fault_tracker):
        arr = np.zeros((7, 13), dtype=np.float32)
        _, h, w = fault_tracker._pad_to_multiple(arr, factor=16)
        assert h == 7 and w == 13


# ---------------------------------------------------------------------------
# _make_hint_multi (instance method)
# ---------------------------------------------------------------------------

class TestFaultMakeHintMulti:
    def test_empty_points_zero_array(self, fault_tracker):
        result = fault_tracker._make_hint_multi([], (10, 10), sigma=2.0)
        assert np.all(result == 0.0)

    def test_single_point_normalised(self, fault_tracker):
        result = fault_tracker._make_hint_multi([(5, 5)], (10, 10), sigma=1.0)
        assert result.max() == pytest.approx(1.0, abs=0.01)

    def test_output_shape(self, fault_tracker):
        result = fault_tracker._make_hint_multi([(3, 3)], (8, 12), sigma=1.5)
        assert result.shape == (8, 12)


# ---------------------------------------------------------------------------
# _make_hint (instance method)
# ---------------------------------------------------------------------------

class TestFaultMakeHint:
    def test_none_pos_returns_zeros(self, fault_tracker):
        result = fault_tracker._make_hint(None, (10, 10), sigma=2.0)
        assert np.all(result == 0.0)

    def test_valid_pos_peak_at_pos(self, fault_tracker):
        result = fault_tracker._make_hint((5, 5), (10, 10), sigma=1.0)
        assert result[5, 5] == pytest.approx(1.0, abs=0.01)

    def test_out_of_bounds_clamped(self, fault_tracker):
        result = fault_tracker._make_hint((100, 200), (10, 10), sigma=1.0)
        assert result.shape == (10, 10)


# ---------------------------------------------------------------------------
# _run_tile (instance method)
# ---------------------------------------------------------------------------

class TestFaultRunTile:
    def test_output_shape_matches(self, fault_tracker):
        H, W = 16, 16
        fault_tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]
        result = fault_tracker._run_tile(
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            H, W,
        )
        assert result.shape == (H, W)

    def test_output_probabilities_in_range(self, fault_tracker):
        H, W = 16, 16
        fault_tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]
        result = fault_tracker._run_tile(
            np.zeros((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
            H, W,
        )
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# track (public — end-to-end with mocked session)
# ---------------------------------------------------------------------------

class TestFaultTrack:
    def test_empty_seeds_returns_empty(self, fault_tracker):
        from seismic_visualizer.domain.cube import SeismicCube
        seismic = SeismicCube(np.zeros((5, 5, 5), dtype=np.uint8))
        result = fault_tracker.track(seismic, [], Axis.INLINE)
        assert result.shape == (0, 3)

    def test_returns_ndarray(self, fault_tracker):
        from seismic_visualizer.domain.cube import SeismicCube
        H, W = 5, 5
        fault_tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]
        seismic = SeismicCube(np.zeros((5, 5, 5), dtype=np.uint8))
        seeds = [Point3D(2, 2, 2)]
        result = fault_tracker.track(seismic, seeds, Axis.INLINE)
        assert isinstance(result, np.ndarray)

    def test_cancel_check_stops_tracking(self, fault_tracker):
        from seismic_visualizer.domain.cube import SeismicCube
        H, W = 5, 5
        fault_tracker._session.run.return_value = [np.zeros((1, 1, H, W), dtype=np.float32)]
        seismic = SeismicCube(np.zeros((10, 5, 5), dtype=np.uint8))
        seeds = [Point3D(5, 2, 2)]
        result = fault_tracker.track(seismic, seeds, Axis.INLINE, cancel_check=lambda: True)
        assert isinstance(result, np.ndarray)

    def test_time_axis_accepted(self, fault_tracker):
        from seismic_visualizer.domain.cube import SeismicCube
        seismic = SeismicCube(np.zeros((5, 5, 5), dtype=np.uint8))
        result = fault_tracker.track(seismic, [], Axis.TIME)
        assert result.shape == (0, 3)
