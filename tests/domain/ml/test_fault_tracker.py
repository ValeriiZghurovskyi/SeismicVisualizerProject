"""Tests for domain.ml.fault_tracker — FaultTracker Protocol."""

from __future__ import annotations

import numpy as np

from seismic_visualizer.domain.cube import SeismicCube
from seismic_visualizer.domain.geometry import Axis, Point3D
from seismic_visualizer.domain.ml.fault_tracker import FaultTracker


class _ConcreteFaultTracker:
    """Minimal structural implementation of FaultTracker."""

    def track(
        self,
        seismic: SeismicCube,
        seed_points: list[Point3D],
        axis: Axis,
        cancel_check=None,
    ) -> np.ndarray:
        return np.empty((0, 3), dtype=np.int32)


def test_concrete_class_is_instance_of_protocol() -> None:
    tracker = _ConcreteFaultTracker()
    assert isinstance(tracker, FaultTracker)


def test_protocol_is_runtime_checkable() -> None:
    class NotATracker:
        pass

    assert not isinstance(NotATracker(), FaultTracker)


def test_track_returns_empty_array_for_no_seeds() -> None:
    tracker = _ConcreteFaultTracker()
    seismic = SeismicCube(np.zeros((4, 4, 4), dtype=np.uint8))
    result = tracker.track(seismic, [], Axis.INLINE)
    assert result.shape == (0, 3)


def test_track_accepts_cancel_check_callable() -> None:
    tracker = _ConcreteFaultTracker()
    seismic = SeismicCube(np.zeros((4, 4, 4), dtype=np.uint8))
    cancelled = False
    result = tracker.track(seismic, [], Axis.INLINE, cancel_check=lambda: cancelled)
    assert result.shape == (0, 3)


def test_tracker_with_seed_points_runs_without_error() -> None:
    tracker = _ConcreteFaultTracker()
    seismic = SeismicCube(np.zeros((5, 5, 5), dtype=np.uint8))
    seeds = [Point3D(inline=1, crossline=2, time=3)]
    result = tracker.track(seismic, seeds, Axis.CROSSLINE)
    assert isinstance(result, np.ndarray)
