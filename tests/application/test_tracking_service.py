"""Tests for TrackingService — voxel writing and dirty-flag behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.services.tracking_service import TrackingService
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityRegistry
from seismic_visualizer.domain.exceptions import EntityNotFoundError
from seismic_visualizer.domain.geometry import Axis, Point3D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPE = (10, 10, 10)


def make_project() -> tuple[Project, int]:
    """Return a minimal project with one registered horizon."""
    seismic = SeismicCube(np.zeros(SHAPE, dtype=np.uint8))
    horizons = LabelCube.empty(SHAPE)
    faults = LabelCube.empty(SHAPE)
    horizon_registry = EntityRegistry()
    fault_registry = EntityRegistry()
    from seismic_visualizer.domain.entities import EntityKind
    entity = horizon_registry.create_new(EntityKind.HORIZON, "H1")
    project = Project(
        seismic=seismic,
        horizons=horizons,
        faults=faults,
        horizon_registry=horizon_registry,
        fault_registry=fault_registry,
        seismic_path=None,
        horizons_path=None,
        faults_path=None,
    )
    return project, entity.id


class MockTracker:
    """Returns a fixed set of voxel coordinates."""

    def __init__(self, voxels: np.ndarray) -> None:
        self._voxels = voxels

    def track(
        self,
        seismic: SeismicCube,
        seed_points: list[Point3D],
        axis: Axis,
        cancel_check=None,
    ) -> np.ndarray:
        return self._voxels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tracking_writes_voxels_to_horizon_cube() -> None:
    project, entity_id = make_project()
    fixed_voxels = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int64)
    tracker = MockTracker(fixed_voxels)

    TrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    for il, xl, t in fixed_voxels:
        assert project.horizons.data[il, xl, t] == entity_id


def test_tracking_sets_dirty_flag() -> None:
    project, entity_id = make_project()
    tracker = MockTracker(np.array([[0, 0, 0]], dtype=np.int64))

    assert not project.is_dirty
    TrackingService().run(project, entity_id, tracker, [], Axis.INLINE)
    assert project.is_dirty


def test_tracking_skips_out_of_bounds_voxels() -> None:
    project, entity_id = make_project()
    # mix of in-bounds and out-of-bounds
    voxels = np.array([[1, 1, 1], [99, 99, 99]], dtype=np.int64)
    tracker = MockTracker(voxels)

    TrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    assert project.horizons.data[1, 1, 1] == entity_id
    # out-of-bounds row was discarded — cube unchanged elsewhere is 0
    assert project.horizons.data[0, 0, 0] == 0


def test_tracking_empty_result_does_not_set_dirty() -> None:
    project, entity_id = make_project()
    tracker = MockTracker(np.empty((0, 3), dtype=np.int64))

    TrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    assert not project.is_dirty


def test_tracking_raises_for_unknown_entity() -> None:
    project, _ = make_project()
    tracker = MockTracker(np.array([[1, 1, 1]], dtype=np.int64))

    with pytest.raises(EntityNotFoundError):
        TrackingService().run(project, 99, tracker, [], Axis.INLINE)


def test_tracking_does_not_write_to_faults_cube() -> None:
    project, entity_id = make_project()
    tracker = MockTracker(np.array([[2, 2, 2]], dtype=np.int64))

    TrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    assert project.faults.data[2, 2, 2] == 0
