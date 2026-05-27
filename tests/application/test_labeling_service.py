"""Tests for application.services.labeling_service — LabelingService."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.services.labeling_service import LabelingService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.state import SliceIndex
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.exceptions import EntityNotFoundError
from seismic_visualizer.domain.geometry import Axis, Point2D


@pytest.fixture
def service() -> LabelingService:
    return LabelingService()


# inline slice 0 of (5,5,10) cube → shape (5, 10); points lie within bounds
_SLICE = SliceIndex(Axis.INLINE, 0)
_POINTS = [Point2D(1, 1), Point2D(1, 4)]


class TestApplyHorizon:
    def test_labels_written_to_horizons_cube(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.horizon_registry.create_new(kind=EntityKind.HORIZON)
        service.apply(project, entity.id, EntityKind.HORIZON, _SLICE, _POINTS)
        assert np.any(project.horizons.data != 0)

    def test_label_value_matches_entity_id(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.horizon_registry.create_new(kind=EntityKind.HORIZON)
        service.apply(project, entity.id, EntityKind.HORIZON, _SLICE, _POINTS)
        assert np.any(project.horizons.data == entity.id)

    def test_horizon_apply_does_not_touch_faults_cube(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.horizon_registry.create_new(kind=EntityKind.HORIZON)
        service.apply(project, entity.id, EntityKind.HORIZON, _SLICE, _POINTS)
        assert np.all(project.faults.data == 0)


class TestApplyFault:
    def test_labels_written_to_faults_cube(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.fault_registry.create_new(kind=EntityKind.FAULT)
        service.apply(project, entity.id, EntityKind.FAULT, _SLICE, _POINTS)
        assert np.any(project.faults.data != 0)

    def test_fault_apply_does_not_touch_horizons_cube(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.fault_registry.create_new(kind=EntityKind.FAULT)
        service.apply(project, entity.id, EntityKind.FAULT, _SLICE, _POINTS)
        assert np.all(project.horizons.data == 0)


class TestIsDirty:
    def test_apply_sets_is_dirty(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.horizon_registry.create_new(kind=EntityKind.HORIZON)
        assert project.is_dirty is False
        service.apply(project, entity.id, EntityKind.HORIZON, _SLICE, _POINTS)
        assert project.is_dirty is True


class TestUnknownEntity:
    def test_unknown_entity_id_raises(
        self, project: Project, service: LabelingService
    ) -> None:
        with pytest.raises(EntityNotFoundError):
            service.apply(project, 99, EntityKind.HORIZON, _SLICE, _POINTS)

    def test_is_dirty_unchanged_on_error(
        self, project: Project, service: LabelingService
    ) -> None:
        with pytest.raises(EntityNotFoundError):
            service.apply(project, 99, EntityKind.HORIZON, _SLICE, _POINTS)
        assert project.is_dirty is False


class TestTimeAxisRejected:
    def test_apply_on_time_axis_raises_labeling_not_allowed(
        self, project: Project, service: LabelingService
    ) -> None:
        from seismic_visualizer.domain.exceptions import LabelingNotAllowedError

        entity = project.horizon_registry.create_new(kind=EntityKind.HORIZON)
        time_slice = SliceIndex(Axis.TIME, 5)
        with pytest.raises(LabelingNotAllowedError):
            service.apply(project, entity.id, EntityKind.HORIZON, time_slice, _POINTS)

    def test_apply_on_time_axis_does_not_set_dirty(
        self, project: Project, service: LabelingService
    ) -> None:
        from seismic_visualizer.domain.exceptions import LabelingNotAllowedError

        entity = project.horizon_registry.create_new(kind=EntityKind.HORIZON)
        time_slice = SliceIndex(Axis.TIME, 5)
        with pytest.raises(LabelingNotAllowedError):
            service.apply(project, entity.id, EntityKind.HORIZON, time_slice, _POINTS)
        assert project.is_dirty is False


class TestErase:
    def test_erase_removes_horizon_label(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.horizon_registry.create_new(kind=EntityKind.HORIZON)
        service.apply(project, entity.id, EntityKind.HORIZON, _SLICE, _POINTS)
        assert np.any(project.horizons.data != 0)

        project.is_dirty = False
        center = Point2D(1, 2)
        service.erase(project, EntityKind.HORIZON, _SLICE, center, radius=5)
        assert project.is_dirty is True

    def test_erase_removes_fault_label(
        self, project: Project, service: LabelingService
    ) -> None:
        entity = project.fault_registry.create_new(kind=EntityKind.FAULT)
        service.apply(project, entity.id, EntityKind.FAULT, _SLICE, _POINTS)
        assert np.any(project.faults.data != 0)

        project.is_dirty = False
        center = Point2D(1, 2)
        service.erase(project, EntityKind.FAULT, _SLICE, center, radius=5)
        assert project.is_dirty is True

    def test_erase_on_time_axis_is_no_op(
        self, project: Project, service: LabelingService
    ) -> None:
        time_slice = SliceIndex(Axis.TIME, 5)
        project.is_dirty = False
        service.erase(project, EntityKind.HORIZON, time_slice, Point2D(1, 1), radius=3)
        assert project.is_dirty is False

    def test_erase_sets_dirty_on_inline_axis(
        self, project: Project, service: LabelingService
    ) -> None:
        project.is_dirty = False
        service.erase(project, EntityKind.HORIZON, _SLICE, Point2D(1, 1), radius=2)
        assert project.is_dirty is True
