"""View2DPresenter — handles all 2D slice view interactions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.application.presenters.slice_presenter import SlicePresenter
from seismic_visualizer.application.rendering.overlay import entity_color
from seismic_visualizer.application.services.labeling_service import LabelingService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.state import SliceIndex
from seismic_visualizer.application.view_interfaces import ToolbarProtocol, View2DProtocol
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.exceptions import EntityNotFoundError
from seismic_visualizer.domain.geometry import Axis, Point2D

if TYPE_CHECKING:
    from seismic_visualizer.application.presenters.view_3d_presenter import View3DPresenter

_log = logging.getLogger(__name__)


class View2DPresenter:
    """Handles 2D labeling, entity management, and canvas refresh.

    Args:
        project: The open seismic project.
        view: 2D view implementing View2DProtocol.
        view3d_presenter: 3D presenter for cache invalidation and thumbnails (None until 3D activated).
        slice_presenter: Produces RGBA frames for the current slice.
        labeling_service: Domain service for stroke/erase operations.
        last_index: Shared mutable dict of current slice indices per axis.
        toolbar: Toolbar implementing ToolbarProtocol for entity indicator updates.
    """

    def __init__(
        self,
        project: Project,
        view: View2DProtocol,
        view3d_presenter: View3DPresenter | None,
        slice_presenter: SlicePresenter,
        labeling_service: LabelingService,
        last_index: dict[Axis, int],
        toolbar: ToolbarProtocol,
    ) -> None:
        self._project = project
        self._view = view
        self._view3d_presenter = view3d_presenter
        self._slice_presenter = slice_presenter
        self._labeling_service = labeling_service
        self._last_index = last_index
        self._toolbar = toolbar
        self._stroke_points: list[Point2D] = []
        self._current_entity_id: int | None = None
        self._current_entity_kind: EntityKind | None = None
        self._line_thickness: int = 1
        self._eraser_radius: int = 5


    @property
    def active_entity_id(self) -> int | None:
        """Currently selected entity ID, or None if nothing is selected."""
        return self._current_entity_id

    @property
    def current_axis(self) -> Axis:
        """Axis of the slice currently displayed in the 2D view."""
        return self._slice_presenter.state.current_slice.axis


    def set_project(self, project: Project, slice_presenter: SlicePresenter) -> None:
        """Replace the active project and reset entity selection."""
        self._project = project
        self._slice_presenter = slice_presenter
        self._current_entity_id = None
        self._current_entity_kind = None
        self._view.set_stroke_color(None)

    def set_view3d_presenter(self, v3d: View3DPresenter | None) -> None:
        self._view3d_presenter = v3d

    def clear_active_entity_if_kind(self, kind: EntityKind) -> None:
        """Clear entity selection when the loaded label cube matches the current kind."""
        if self._current_entity_kind == kind:
            self._current_entity_id = None
            self._current_entity_kind = None
            self._view.set_stroke_color(None)
            self._toolbar.set_active_entity("—", None)


    def clear_stroke(self) -> None:
        self._stroke_points.clear()
        self._view.clear_stroke_preview()

    def refresh_canvas(self) -> None:
        if self._project is None:
            return
        slice_index = self._slice_presenter.state.current_slice
        rgba = self._slice_presenter.get_rgba(slice_index)
        max_index = self._project.seismic.shape[slice_index.axis.value] - 1
        self._view.update_canvas(rgba)
        self._view.update_status(
            f"Slice {slice_index.index}/{max_index} | {slice_index.axis.display_name}"
        )
        self._view.update_entity_list(self.build_entity_list())

    def push_thumbnail(self, axis: Axis) -> None:
        if self._view3d_presenter is None:
            return
        rgba = self._slice_presenter.get_rgba(
            SliceIndex(axis=axis, index=self._last_index[axis])
        )
        self._view3d_presenter.push_thumbnail(axis, rgba)

    def build_entity_list(self) -> list[EntityInfo]:
        if self._project is None:
            return []
        result: list[EntityInfo] = []
        for entity in self._project.horizon_registry.all():
            result.append(EntityInfo(
                id=entity.id, name=entity.name, kind=entity.kind,
                visible=entity.visible, color=entity_color(entity.id, entity.kind),
            ))
        for entity in self._project.fault_registry.all():
            result.append(EntityInfo(
                id=entity.id, name=entity.name, kind=entity.kind,
                visible=entity.visible, color=entity_color(entity.id, entity.kind),
            ))
        return result


    def on_stroke_point(self, row: int, col: int) -> None:
        axis = self._slice_presenter.state.current_slice.axis
        if axis == Axis.TIME:
            self._view.remove_last_stroke_point()
            self._view.update_status("Labeling not allowed on Time axis")
            return
        self._stroke_points.append(Point2D(row=col, col=row))

    def on_stroke_finished(self) -> None:
        if (
            self._current_entity_id is None
            or self._current_entity_kind is None
            or not self._stroke_points
        ):
            self._stroke_points.clear()
            return
        try:
            self._labeling_service.apply(
                self._project,
                self._current_entity_id,
                self._current_entity_kind,
                self._slice_presenter.state.current_slice,
                self._stroke_points,
                thickness=self._line_thickness,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught  # labeler raises various numpy/domain errors
            _log.warning("Labeling failed: %s", exc)
        self._stroke_points.clear()
        self._view.clear_stroke_preview()
        if self._view3d_presenter is not None:
            self._view3d_presenter.invalidate_label_cache(self._current_entity_id, self._current_entity_kind)
        self.refresh_canvas()

    def on_erase_at(self, row: int, col: int) -> None:
        if self._current_entity_kind is None:
            return
        axis = self._slice_presenter.state.current_slice.axis
        if axis == Axis.TIME:
            return
        self._labeling_service.erase(
            self._project,
            self._current_entity_kind,
            self._slice_presenter.state.current_slice,
            Point2D(row=col, col=row),
            self._eraser_radius,
        )
        self.refresh_canvas()

    def on_stroke_point_removed(self) -> None:
        if self._stroke_points:
            self._stroke_points.pop()
        self._view.remove_last_stroke_point()

    def on_new_entity(self, kind: EntityKind) -> None:
        if self._project is None:
            return
        registry = (
            self._project.horizon_registry
            if kind == EntityKind.HORIZON
            else self._project.fault_registry
        )
        entity = registry.create_new(kind)
        self._current_entity_id = entity.id
        self._current_entity_kind = entity.kind
        color = entity_color(entity.id, entity.kind)
        self._view.set_stroke_color(color)
        self._toolbar.set_active_entity(entity.name, color)
        self._project.is_dirty = True
        self._view.update_entity_list(self.build_entity_list())

    def on_entity_selected(self, entity_id: int, kind: object) -> None:
        self._current_entity_id = entity_id
        self._current_entity_kind = kind  # type: ignore[assignment]
        color = entity_color(entity_id, kind)  # type: ignore[arg-type]
        self._view.set_stroke_color(color)
        self._toolbar.set_active_entity(
            self._entity_name(entity_id, kind),  # type: ignore[arg-type]
            color,
        )

    def on_entity_delete(self, entity_id: int, kind: object) -> None:
        if self._project is None:
            return
        registry = (
            self._project.horizon_registry
            if kind == EntityKind.HORIZON
            else self._project.fault_registry
        )
        label_cube = (
            self._project.horizons
            if kind == EntityKind.HORIZON
            else self._project.faults
        )
        if np.any(label_cube.data == entity_id):
            name = self._entity_name(entity_id, kind)  # type: ignore[arg-type]
            if not self._view.confirm_entity_delete(name):
                return
        registry.remove(entity_id)
        label_cube.data[label_cube.data == entity_id] = 0
        if self._current_entity_id == entity_id and self._current_entity_kind == kind:
            self._current_entity_id = None
            self._current_entity_kind = None
            self._view.set_stroke_color(None)
            self._toolbar.set_active_entity("—", None)
        self._project.is_dirty = True
        self.refresh_canvas()

    def on_entity_rename(self, entity_id: int, kind: object, name: str) -> None:
        if self._project is None:
            return
        registry = (
            self._project.horizon_registry
            if kind == EntityKind.HORIZON
            else self._project.fault_registry
        )
        registry.rename(entity_id, name)
        self.refresh_canvas()

    def on_visibility_changed(self, entity_id: int, kind: object, visible: bool) -> None:
        if self._project is None:
            return
        registry = (
            self._project.horizon_registry
            if kind == EntityKind.HORIZON
            else self._project.fault_registry
        )
        try:
            registry.set_visibility(entity_id, visible)
        except EntityNotFoundError:
            _log.warning("entity_visibility_changed: id %d not found", entity_id)
        self.refresh_canvas()

    def on_thickness_changed(self, thickness: int) -> None:
        self._line_thickness = thickness

    def on_eraser_radius_changed(self, radius: int) -> None:
        self._eraser_radius = radius


    def _entity_name(self, entity_id: int, kind: EntityKind) -> str:
        if self._project is None:
            return "—"
        registry = (
            self._project.horizon_registry
            if kind == EntityKind.HORIZON
            else self._project.fault_registry
        )
        try:
            return registry.get(entity_id).name
        except EntityNotFoundError:
            return "—"
