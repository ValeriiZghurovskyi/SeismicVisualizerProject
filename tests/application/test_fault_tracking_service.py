"""Tests for application.services.fault_tracking_service — FaultTrackingService."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.services.fault_tracking_service import FaultTrackingService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityKind, EntityRegistry
from seismic_visualizer.domain.exceptions import EntityNotFoundError
from seismic_visualizer.domain.geometry import Axis, Point3D

SHAPE = (10, 10, 10)


def make_project() -> tuple[Project, int]:
    seismic = SeismicCube(np.zeros(SHAPE, dtype=np.uint8))
    horizons = LabelCube.empty(SHAPE)
    faults = LabelCube.empty(SHAPE)
    horizon_registry = EntityRegistry()
    fault_registry = EntityRegistry()
    entity = fault_registry.create_new(EntityKind.FAULT, "F1")
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


class MockFaultTracker:
    def __init__(self, voxels: np.ndarray) -> None:
        self._voxels = voxels

    def track(self, seismic, seed_points, axis, cancel_check=None) -> np.ndarray:
        return self._voxels


def test_fault_tracking_writes_voxels_to_faults_cube() -> None:
    project, entity_id = make_project()
    voxels = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int64)
    tracker = MockFaultTracker(voxels)

    FaultTrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    for il, xl, t in voxels:
        assert project.faults.data[il, xl, t] == entity_id


def test_fault_tracking_sets_dirty_flag() -> None:
    project, entity_id = make_project()
    tracker = MockFaultTracker(np.array([[0, 0, 0]], dtype=np.int64))

    assert not project.is_dirty
    FaultTrackingService().run(project, entity_id, tracker, [], Axis.INLINE)
    assert project.is_dirty


def test_fault_tracking_does_not_touch_horizons_cube() -> None:
    project, entity_id = make_project()
    tracker = MockFaultTracker(np.array([[2, 2, 2]], dtype=np.int64))

    FaultTrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    assert project.horizons.data[2, 2, 2] == 0


def test_fault_tracking_skips_out_of_bounds_voxels() -> None:
    project, entity_id = make_project()
    voxels = np.array([[1, 1, 1], [99, 99, 99]], dtype=np.int64)
    tracker = MockFaultTracker(voxels)

    FaultTrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    assert project.faults.data[1, 1, 1] == entity_id
    assert project.faults.data[0, 0, 0] == 0


def test_fault_tracking_all_out_of_bounds_does_not_set_dirty() -> None:
    project, entity_id = make_project()
    voxels = np.array([[99, 99, 99]], dtype=np.int64)
    tracker = MockFaultTracker(voxels)

    FaultTrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    assert not project.is_dirty


def test_fault_tracking_empty_result_does_not_set_dirty() -> None:
    project, entity_id = make_project()
    tracker = MockFaultTracker(np.empty((0, 3), dtype=np.int64))

    FaultTrackingService().run(project, entity_id, tracker, [], Axis.INLINE)

    assert not project.is_dirty


def test_fault_tracking_raises_for_unknown_entity() -> None:
    project, _ = make_project()
    tracker = MockFaultTracker(np.array([[1, 1, 1]], dtype=np.int64))

    with pytest.raises(EntityNotFoundError):
        FaultTrackingService().run(project, 99, tracker, [], Axis.INLINE)


def test_fault_tracking_cancel_check_prevents_write() -> None:
    project, entity_id = make_project()
    voxels = np.array([[1, 1, 1]], dtype=np.int64)
    tracker = MockFaultTracker(voxels)

    FaultTrackingService().run(
        project, entity_id, tracker, [], Axis.INLINE, cancel_check=lambda: True
    )

    assert project.faults.data[1, 1, 1] == 0
    assert not project.is_dirty


def test_fault_tracking_cancel_check_false_allows_write() -> None:
    project, entity_id = make_project()
    voxels = np.array([[1, 1, 1]], dtype=np.int64)
    tracker = MockFaultTracker(voxels)

    FaultTrackingService().run(
        project, entity_id, tracker, [], Axis.INLINE, cancel_check=lambda: False
    )

    assert project.faults.data[1, 1, 1] == entity_id


def test_fault_tracking_crossline_axis_works() -> None:
    project, entity_id = make_project()
    voxels = np.array([[3, 4, 5]], dtype=np.int64)
    tracker = MockFaultTracker(voxels)

    FaultTrackingService().run(project, entity_id, tracker, [], Axis.CROSSLINE)

    assert project.faults.data[3, 4, 5] == entity_id
