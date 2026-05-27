"""Tests for application.presenters.view_2d_presenter — View2DPresenter."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.application.presenters.slice_presenter import SlicePresenter
from seismic_visualizer.application.presenters.view_2d_presenter import View2DPresenter
from seismic_visualizer.application.services.attribute_service import AttributeService
from seismic_visualizer.application.services.labeling_service import LabelingService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.state import SliceIndex
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.geometry import Axis


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

class MockView2D:
    """Duck-typed stub implementing View2DProtocol."""

    def __init__(self) -> None:
        self.canvas_updates: int = 0
        self.entity_list_updates: int = 0
        self.status_calls: list[str] = []
        self.stroke_color_calls: list[tuple | None] = []
        self.last_point_removed: int = 0
        self.stroke_cleared: int = 0

    def update_canvas(self, rgba: np.ndarray) -> None:
        self.canvas_updates += 1

    def update_entity_list(self, entities: list[EntityInfo]) -> None:
        self.entity_list_updates += 1

    def update_status(self, text: str) -> None:
        self.status_calls.append(text)

    def set_stroke_color(self, color: tuple[int, int, int] | None) -> None:
        self.stroke_color_calls.append(color)

    def remove_last_stroke_point(self) -> None:
        self.last_point_removed += 1

    def clear_stroke_preview(self) -> None:
        self.stroke_cleared += 1

    def confirm_entity_delete(self, name: str) -> bool:
        return True


class MockToolbar:
    """Duck-typed stub implementing ToolbarProtocol."""

    def __init__(self) -> None:
        self.last_name: str = ""
        self.last_color: tuple | None = None

    def set_active_entity(self, name: str, color: tuple[int, int, int] | None) -> None:
        self.last_name = name
        self.last_color = color


class MockView3DPresenter:
    """Minimal stub for View3DPresenter — only the methods View2DPresenter calls."""

    def __init__(self) -> None:
        self.invalidated: list[int | None] = []
        self.thumbnail_axes: list[Axis] = []

    def invalidate_label_cache(self, entity_id: int | None = None, kind=None) -> None:
        self.invalidated.append(entity_id)

    def push_thumbnail(self, axis: Axis, rgba: np.ndarray) -> None:
        self.thumbnail_axes.append(axis)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def view() -> MockView2D:
    return MockView2D()


@pytest.fixture
def toolbar() -> MockToolbar:
    return MockToolbar()


@pytest.fixture
def last_index() -> dict[Axis, int]:
    return {Axis.INLINE: 0, Axis.CROSSLINE: 0, Axis.TIME: 0}


@pytest.fixture
def slice_presenter(project: Project) -> SlicePresenter:
    return SlicePresenter(project, AttributeService())


@pytest.fixture
def presenter(
    project: Project,
    view: MockView2D,
    toolbar: MockToolbar,
    last_index: dict[Axis, int],
    slice_presenter: SlicePresenter,
) -> View2DPresenter:
    return View2DPresenter(
        project=project,
        view=view,
        view3d_presenter=None,
        slice_presenter=slice_presenter,
        labeling_service=LabelingService(),
        last_index=last_index,
        toolbar=toolbar,
    )


# ---------------------------------------------------------------------------
# TestBuildEntityList
# ---------------------------------------------------------------------------

class TestBuildEntityList:
    def test_empty_with_no_entities(self, presenter: View2DPresenter) -> None:
        assert presenter.build_entity_list() == []

    def test_returns_horizons_and_faults(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        project.horizon_registry.create_new(EntityKind.HORIZON)
        project.fault_registry.create_new(EntityKind.FAULT)
        result = presenter.build_entity_list()
        assert len(result) == 2
        kinds = {info.kind for info in result}
        assert EntityKind.HORIZON in kinds
        assert EntityKind.FAULT in kinds

    def test_entity_info_fields_populated(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        result = presenter.build_entity_list()
        assert result[0].id == entity.id
        assert result[0].name == entity.name
        assert result[0].visible is True


# ---------------------------------------------------------------------------
# TestRefreshCanvas
# ---------------------------------------------------------------------------

class TestRefreshCanvas:
    def test_calls_update_canvas(
        self, presenter: View2DPresenter, view: MockView2D
    ) -> None:
        presenter.refresh_canvas()
        assert view.canvas_updates == 1

    def test_calls_update_entity_list(
        self, presenter: View2DPresenter, view: MockView2D
    ) -> None:
        presenter.refresh_canvas()
        assert view.entity_list_updates == 1

    def test_status_contains_slice_keyword(
        self, presenter: View2DPresenter, view: MockView2D
    ) -> None:
        presenter.refresh_canvas()
        assert len(view.status_calls) == 1
        assert "Slice" in view.status_calls[0]


# ---------------------------------------------------------------------------
# TestClearStroke
# ---------------------------------------------------------------------------

class TestClearStroke:
    def test_clears_internal_buffer(
        self, presenter: View2DPresenter, slice_presenter: SlicePresenter
    ) -> None:
        slice_presenter.navigate(SliceIndex(axis=Axis.INLINE, index=0))
        presenter.on_stroke_point(0, 0)
        presenter.clear_stroke()
        assert presenter._stroke_points == []

    def test_calls_view_clear_preview(
        self, presenter: View2DPresenter, view: MockView2D
    ) -> None:
        presenter.clear_stroke()
        assert view.stroke_cleared == 1


# ---------------------------------------------------------------------------
# TestStroke
# ---------------------------------------------------------------------------

class TestStroke:
    def test_stroke_point_on_inline_appended(
        self, presenter: View2DPresenter, slice_presenter: SlicePresenter
    ) -> None:
        slice_presenter.navigate(SliceIndex(axis=Axis.INLINE, index=0))
        presenter.on_stroke_point(2, 3)
        assert len(presenter._stroke_points) == 1

    def test_stroke_point_on_time_axis_rejected(
        self, presenter: View2DPresenter, slice_presenter: SlicePresenter, view: MockView2D
    ) -> None:
        slice_presenter.navigate(SliceIndex(axis=Axis.TIME, index=0))
        presenter.on_stroke_point(2, 3)
        assert presenter._stroke_points == []
        assert view.last_point_removed == 1

    def test_stroke_on_time_axis_shows_status(
        self, presenter: View2DPresenter, slice_presenter: SlicePresenter, view: MockView2D
    ) -> None:
        slice_presenter.navigate(SliceIndex(axis=Axis.TIME, index=0))
        presenter.on_stroke_point(0, 0)
        assert any("Time" in s for s in view.status_calls)

    def test_stroke_finished_no_entity_clears(
        self, presenter: View2DPresenter, slice_presenter: SlicePresenter
    ) -> None:
        slice_presenter.navigate(SliceIndex(axis=Axis.INLINE, index=0))
        presenter.on_stroke_point(0, 0)
        presenter.on_stroke_finished()
        assert presenter._stroke_points == []

    def test_stroke_finished_applies_label(
        self, presenter: View2DPresenter, project: Project, slice_presenter: SlicePresenter
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        slice_presenter.navigate(SliceIndex(axis=Axis.INLINE, index=0))
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        presenter.on_stroke_point(0, 0)
        presenter.on_stroke_finished()
        assert project.horizons.data.any()

    def test_stroke_finished_invalidates_3d_cache(
        self,
        project: Project,
        view: MockView2D,
        toolbar: MockToolbar,
        last_index: dict[Axis, int],
        slice_presenter: SlicePresenter,
    ) -> None:
        v3d = MockView3DPresenter()
        pres = View2DPresenter(
            project=project, view=view, view3d_presenter=v3d,
            slice_presenter=slice_presenter, labeling_service=LabelingService(),
            last_index=last_index, toolbar=toolbar,
        )
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        slice_presenter.navigate(SliceIndex(axis=Axis.INLINE, index=0))
        pres.on_entity_selected(entity.id, EntityKind.HORIZON)
        pres.on_stroke_point(0, 0)
        pres.on_stroke_finished()
        assert entity.id in v3d.invalidated

    def test_stroke_point_removed_pops_buffer(
        self, presenter: View2DPresenter, slice_presenter: SlicePresenter, view: MockView2D
    ) -> None:
        slice_presenter.navigate(SliceIndex(axis=Axis.INLINE, index=0))
        presenter.on_stroke_point(0, 0)
        presenter.on_stroke_point(1, 1)
        presenter.on_stroke_point_removed()
        assert len(presenter._stroke_points) == 1
        assert view.last_point_removed == 1

    def test_stroke_point_removed_on_empty_buffer_does_not_raise(
        self, presenter: View2DPresenter, view: MockView2D
    ) -> None:
        presenter.on_stroke_point_removed()
        assert presenter._stroke_points == []
        assert view.last_point_removed == 1


# ---------------------------------------------------------------------------
# TestEraseAt
# ---------------------------------------------------------------------------

class TestEraseAt:
    def test_erase_without_entity_kind_is_noop(
        self, presenter: View2DPresenter, project: Project, slice_presenter: SlicePresenter
    ) -> None:
        project.horizons.data[0, 0, 0] = 1
        slice_presenter.navigate(SliceIndex(axis=Axis.INLINE, index=0))
        presenter.on_erase_at(0, 0)
        assert project.horizons.data[0, 0, 0] == 1

    def test_erase_on_time_axis_is_noop(
        self, presenter: View2DPresenter, project: Project, slice_presenter: SlicePresenter
    ) -> None:
        project.horizons.data[0, 0, 0] = 1
        slice_presenter.navigate(SliceIndex(axis=Axis.TIME, index=0))
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        presenter.on_erase_at(0, 0)
        assert project.horizons.data[0, 0, 0] == 1


# ---------------------------------------------------------------------------
# TestEntityManagement
# ---------------------------------------------------------------------------

class TestEntityManagement:
    def test_new_entity_creates_in_registry(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        presenter.on_new_entity(EntityKind.HORIZON)
        assert len(list(project.horizon_registry.all())) == 1

    def test_new_entity_sets_current(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        presenter.on_new_entity(EntityKind.HORIZON)
        assert presenter._current_entity_kind == EntityKind.HORIZON
        assert presenter._current_entity_id is not None

    def test_new_entity_updates_toolbar(
        self, presenter: View2DPresenter, toolbar: MockToolbar
    ) -> None:
        presenter.on_new_entity(EntityKind.HORIZON)
        assert toolbar.last_name != ""
        assert toolbar.last_color is not None

    def test_entity_selected_sets_state(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        assert presenter._current_entity_id == entity.id
        assert presenter._current_entity_kind == EntityKind.HORIZON

    def test_entity_selected_updates_toolbar(
        self, presenter: View2DPresenter, project: Project, toolbar: MockToolbar
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        assert toolbar.last_name == entity.name

    def test_entity_delete_clears_current_when_matches(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        presenter.on_entity_delete(entity.id, EntityKind.HORIZON)
        assert presenter._current_entity_id is None
        assert presenter._current_entity_kind is None

    def test_entity_delete_zeros_label_cube(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        project.horizons.data[0, 0, 0] = entity.id
        presenter.on_entity_delete(entity.id, EntityKind.HORIZON)
        assert project.horizons.data[0, 0, 0] == 0

    def test_entity_delete_does_not_clear_current_when_different(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        e1 = project.horizon_registry.create_new(EntityKind.HORIZON)
        e2 = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(e1.id, EntityKind.HORIZON)
        presenter.on_entity_delete(e2.id, EntityKind.HORIZON)
        assert presenter._current_entity_id == e1.id

    def test_visibility_changed_updates_registry(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        assert entity.visible is True
        presenter.on_visibility_changed(entity.id, EntityKind.HORIZON, False)
        assert project.horizon_registry.get(entity.id).visible is False

    def test_entity_rename_reflected_in_list(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_rename(entity.id, EntityKind.HORIZON, "NewName")
        result = presenter.build_entity_list()
        assert result[0].name == "NewName"


# ---------------------------------------------------------------------------
# TestPushThumbnail
# ---------------------------------------------------------------------------

class TestPushThumbnail:
    def test_no_v3d_presenter_is_noop(self, presenter: View2DPresenter) -> None:
        presenter.push_thumbnail(Axis.INLINE)  # must not raise

    def test_delegates_to_view3d_presenter(
        self,
        project: Project,
        view: MockView2D,
        toolbar: MockToolbar,
        last_index: dict[Axis, int],
        slice_presenter: SlicePresenter,
    ) -> None:
        v3d = MockView3DPresenter()
        pres = View2DPresenter(
            project=project, view=view, view3d_presenter=v3d,
            slice_presenter=slice_presenter, labeling_service=LabelingService(),
            last_index=last_index, toolbar=toolbar,
        )
        pres.push_thumbnail(Axis.INLINE)
        assert Axis.INLINE in v3d.thumbnail_axes

    def test_set_view3d_presenter_enables_thumbnails(
        self,
        presenter: View2DPresenter,
        project: Project,
        view: MockView2D,
        toolbar: MockToolbar,
        last_index: dict[Axis, int],
        slice_presenter: SlicePresenter,
    ) -> None:
        v3d = MockView3DPresenter()
        presenter.set_view3d_presenter(v3d)
        presenter.push_thumbnail(Axis.CROSSLINE)
        assert Axis.CROSSLINE in v3d.thumbnail_axes


# ---------------------------------------------------------------------------
# TestSetProject
# ---------------------------------------------------------------------------

class TestSetProject:
    def test_set_project_clears_entity(
        self, presenter: View2DPresenter, project: Project, slice_presenter: SlicePresenter
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        assert presenter._current_entity_id is not None
        presenter.set_project(project, slice_presenter)
        assert presenter._current_entity_id is None

    def test_set_project_clears_stroke_color(
        self, presenter: View2DPresenter, project: Project, slice_presenter: SlicePresenter,
        view: MockView2D
    ) -> None:
        presenter.set_project(project, slice_presenter)
        assert None in view.stroke_color_calls


# ---------------------------------------------------------------------------
# TestClearActiveEntityIfKind
# ---------------------------------------------------------------------------

class TestClearActiveEntityIfKind:
    def test_clears_when_kind_matches(
        self, presenter: View2DPresenter, project: Project, toolbar: MockToolbar
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        presenter.clear_active_entity_if_kind(EntityKind.HORIZON)
        assert presenter._current_entity_id is None
        assert toolbar.last_name == "—"

    def test_does_not_clear_when_kind_differs(
        self, presenter: View2DPresenter, project: Project
    ) -> None:
        entity = project.horizon_registry.create_new(EntityKind.HORIZON)
        presenter.on_entity_selected(entity.id, EntityKind.HORIZON)
        presenter.clear_active_entity_if_kind(EntityKind.FAULT)
        assert presenter._current_entity_id == entity.id
