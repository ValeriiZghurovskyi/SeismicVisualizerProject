"""Tests for InterpolationService — two-step interpolation API."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.dto import InterpolationParams, InterpolationResult
from seismic_visualizer.application.services.interpolation_service import InterpolationService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityKind, EntityRegistry
from seismic_visualizer.domain.exceptions import EntityNotFoundError

_SHAPE = (20, 20, 20)
_PARAMS = InterpolationParams(grid_step=2.0, median_size=1, max_radius=8.0)


@pytest.fixture
def project() -> Project:
    return Project(
        seismic=SeismicCube(np.zeros(_SHAPE, dtype=np.uint8)),
        horizons=LabelCube.empty(_SHAPE),
        faults=LabelCube.empty(_SHAPE),
        horizon_registry=EntityRegistry(),
        fault_registry=EntityRegistry(),
        seismic_path=None,
        horizons_path=None,
        faults_path=None,
    )


def _write_sparse_horizon(project: Project, entity_id: int, time: int = 10) -> None:
    """Write a sparse regular grid of horizon labels at a fixed time slice."""
    for i in range(0, 20, 4):
        for j in range(0, 20, 4):
            project.horizons.data[i, j, time] = entity_id


def _write_sparse_fault(project: Project, entity_id: int, inline: int = 10) -> None:
    """Write a sparse vertical fault at a fixed inline."""
    for j in range(0, 20, 4):
        for t in range(0, 20, 4):
            project.faults.data[inline, j, t] = entity_id


class TestInterpolateEntity:
    def test_entity_not_found_raises(self, project: Project) -> None:
        service = InterpolationService(project)
        with pytest.raises(EntityNotFoundError):
            service.interpolate_entity(EntityKind.HORIZON, 99, _PARAMS)

    def test_empty_entity_returns_empty_points(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        assert result.points.shape == (0, 3)

    def test_horizon_result_has_correct_kind_and_id(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        _write_sparse_horizon(project, entity.id)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        assert result.entity_kind == EntityKind.HORIZON
        assert result.entity_id == entity.id

    def test_horizon_interpolation_returns_points(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        _write_sparse_horizon(project, entity.id, time=10)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        assert len(result.points) > 0
        assert result.points.shape[1] == 3

    def test_fault_interpolation_returns_points(self, project: Project) -> None:
        entity = project.fault_registry.create_new(EntityKind.FAULT)
        _write_sparse_fault(project, entity.id, inline=10)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.FAULT, entity.id, _PARAMS)
        assert len(result.points) > 0
        assert result.points.shape[1] == 3

    def test_result_does_not_modify_project(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        _write_sparse_horizon(project, entity.id)
        before = project.horizons.data.copy()
        service = InterpolationService(project)
        service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        np.testing.assert_array_equal(project.horizons.data, before)


class TestApplyInterpolation:
    def test_apply_writes_to_horizon_cube(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        _write_sparse_horizon(project, entity.id)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        labeled_before = int(np.sum(project.horizons.data == entity.id))
        service.apply_interpolation(result)
        labeled_after = int(np.sum(project.horizons.data == entity.id))
        assert labeled_after >= labeled_before

    def test_apply_horizon_does_not_touch_faults(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        _write_sparse_horizon(project, entity.id)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        service.apply_interpolation(result)
        assert np.all(project.faults.data == 0)

    def test_apply_fault_does_not_touch_horizons(self, project: Project) -> None:
        entity = project.fault_registry.create_new(EntityKind.FAULT)
        _write_sparse_fault(project, entity.id)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.FAULT, entity.id, _PARAMS)
        service.apply_interpolation(result)
        assert np.all(project.horizons.data == 0)

    def test_apply_sets_project_dirty(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        _write_sparse_horizon(project, entity.id)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        assert project.is_dirty is False
        service.apply_interpolation(result)
        assert project.is_dirty is True

    def test_apply_empty_result_does_not_set_dirty(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        service = InterpolationService(project)
        result = service.interpolate_entity(EntityKind.HORIZON, entity.id, _PARAMS)
        service.apply_interpolation(result)
        assert project.is_dirty is False

    def test_apply_clips_out_of_bounds_voxels(self, project: Project) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        service = InterpolationService(project)
        # Points entirely outside the cube — nothing should be written
        oob_points = np.array([[999, 999, 999], [-1, -1, -1]], dtype=np.int64)
        result = InterpolationResult(EntityKind.HORIZON, entity.id, oob_points)
        service.apply_interpolation(result)
        assert project.is_dirty is False
        assert np.all(project.horizons.data == 0)
