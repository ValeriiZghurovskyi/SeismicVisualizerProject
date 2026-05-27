"""View3D — PyVista QtInteractor implementing View3DProtocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QFrame, QLabel, QSplitter, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from seismic_visualizer.application.dto import EntityInfo, RandomLinePlaneDTO
from seismic_visualizer.application.view_interfaces import LabelMesh
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.application.rendering.colormap import apply_seismic_colormap
from seismic_visualizer.domain.slice import Slice
from seismic_visualizer.ui.i18n import LanguageManager, tr
from seismic_visualizer.ui.widgets.entity_sidebar import EntitySidebar
from seismic_visualizer.ui.widgets.thumbnail_panel import ThumbnailSidePanel

if TYPE_CHECKING:
    from seismic_visualizer.application.services.project import Project


class View3D(QWidget):
    """Three orthogonal seismic slice planes in an interactive 3D viewport.

    Implements View3DProtocol via duck typing.
    World-space coordinates map directly to data indices:
        X = inline,  Y = crossline,  Z = time (depth).
    """

    plane_moved = pyqtSignal(object, int)
    entity_visibility_changed = pyqtSignal(int, object, bool)
    plane_visibility_changed = pyqtSignal(object, bool)
    clip_labels_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._plotter = QtInteractor(self)
        self._plotter.set_background("white")
        self._plotter.add_axes()

        self._plane_actors: dict[Axis, Any] = {}
        self._label_actors: dict[tuple[str, int], Any] = {}
        self._random_line_actor: Any | None = None
        self._preview_actor: Any | None = None
        self._project: Project | None = None
        self._cmap_name: str = "seismic"

        self._thumb_side = ThumbnailSidePanel()

        self._sidebar = EntitySidebar()
        self._sidebar.set_view_only(True)
        self._sidebar.entity_visibility_changed.connect(self.entity_visibility_changed)

        self._plane_checkboxes: dict[Axis, QCheckBox] = {}
        self._planes_label: QLabel | None = None
        self._clip_cb: QCheckBox | None = None
        self._thumb_cb: QCheckBox | None = None
        planes_panel = self._build_planes_panel()

        LanguageManager.instance().language_changed.connect(lambda _: self.retranslate_ui())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(planes_panel)
        right_layout.addWidget(self._sidebar, stretch=1)

        self._splitter = QSplitter(Qt.Horizontal)  # type: ignore[arg-type]
        self._splitter.addWidget(self._plotter)
        self._splitter.addWidget(self._thumb_side)
        self._splitter.addWidget(right)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setSizes([540, 160, 200])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)


    def update_seismic_planes(
        self, project: Project, slice_indices: dict[Axis, int]
    ) -> None:
        self._project = project
        for axis, index in slice_indices.items():
            self._render_plane(project, axis, index)

    def set_plane_position(self, axis: Axis, index: int) -> None:
        if self._project is None:
            return
        self._render_plane(self._project, axis, index)

    def update_label_meshes(self, meshes: list[LabelMesh]) -> None:
        new_keys = {(info.kind.value, info.id) for _, info in meshes}
        for key in list(self._label_actors.keys()):
            if key not in new_keys:
                self._plotter.remove_actor(self._label_actors.pop(key))

        for coords, info in meshes:
            key = (info.kind.value, info.id)
            cloud = pv.PolyData(coords.astype(float))
            actor = self._plotter.add_mesh(
                cloud,
                color=info.color,
                point_size=3,
                render_points_as_spheres=True,
                name=f"label_{info.kind.value}_{info.id}",
                reset_camera=False,
            )
            actor.SetVisibility(info.visible)
            self._label_actors[key] = actor

    def set_entity_visible(self, entity_id: int, kind: object, visible: bool) -> None:
        kind_str = kind.value if hasattr(kind, "value") else str(kind)
        actor = self._label_actors.get((kind_str, entity_id))
        if actor is not None:
            actor.SetVisibility(visible)
            self._plotter.render()

    def update_entity_list(self, entities: list[EntityInfo]) -> None:
        self._sidebar.update_entities(entities)

    def reset_camera(self) -> None:
        self._plotter.reset_camera()
        self._plotter.camera.zoom(0.8)

    def set_plane_visible(self, axis: Axis, visible: bool) -> None:
        if axis in self._plane_actors:
            self._plane_actors[axis].SetVisibility(visible)
            self._plotter.render()
        cb = self._plane_checkboxes.get(axis)
        if cb is not None and cb.isChecked() != visible:
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)

    def show_random_line_plane(self, dto: RandomLinePlaneDTO) -> None:
        if self._random_line_actor is not None:
            self._plotter.remove_actor(self._random_line_actor)

        plane = pv.Plane(
            center=dto.origin,
            direction=dto.normal,
            i_size=dto.width,
            j_size=dto.height,
        )
        self._random_line_actor = self._plotter.add_mesh(
            plane,
            color="yellow",
            opacity=0.4,
            name="random_line",
        )


    def preview_interpolation(
        self, points: np.ndarray, color: tuple[int, int, int]
    ) -> None:
        self.clear_interpolation_preview()
        cloud = pv.PolyData(points.astype(float))
        self._preview_actor = self._plotter.add_mesh(
            cloud,
            color=color,
            opacity=0.35,
            point_size=4,
            render_points_as_spheres=True,
            name="interp_preview",
            reset_camera=False,
        )

    def clear_interpolation_preview(self) -> None:
        if self._preview_actor is not None:
            self._plotter.remove_actor(self._preview_actor)
            self._preview_actor = None
            self._plotter.render()

    def set_colormap(self, name: str) -> None:
        self._cmap_name = name

    def set_theme(self, theme: str) -> None:
        bg = "black" if theme == "dark" else "white"
        self._plotter.set_background(bg)
        self._plotter.render()

    def update_slice_thumbnail(self, axis: Axis, rgba: np.ndarray) -> None:
        self._thumb_side.set_thumbnail(axis, rgba)

    def retranslate_ui(self) -> None:
        if self._planes_label is not None:
            self._planes_label.setText(tr("3d.planes"))
        if self._clip_cb is not None:
            self._clip_cb.setText(tr("3d.clip_labels"))
        if self._thumb_cb is not None:
            self._thumb_cb.setText(tr("3d.thumbnails"))
        for axis, key in (
            (Axis.INLINE, "toolbar.inline"),
            (Axis.CROSSLINE, "toolbar.crossline"),
            (Axis.TIME, "toolbar.time"),
        ):
            cb = self._plane_checkboxes.get(axis)
            if cb is not None:
                cb.setText(tr(key))
        self._thumb_side.retranslate_ui()

    def _build_planes_panel(self) -> QFrame:
        """Build the fixed-height panel with per-plane visibility checkboxes."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)  # type: ignore[attr-defined]
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        self._planes_label = QLabel(tr("3d.planes"))
        layout.addWidget(self._planes_label)

        for axis, key in (
            (Axis.INLINE, "toolbar.inline"),
            (Axis.CROSSLINE, "toolbar.crossline"),
            (Axis.TIME, "toolbar.time"),
        ):
            cb = QCheckBox(tr(key))
            cb.setChecked(True)
            cb.toggled.connect(
                lambda checked, ax=axis: self.plane_visibility_changed.emit(ax, checked)
            )
            self._plane_checkboxes[axis] = cb
            layout.addWidget(cb)

        layout.addSpacing(4)
        self._clip_cb = QCheckBox(tr("3d.clip_labels"))
        self._clip_cb.setChecked(False)
        self._clip_cb.toggled.connect(self.clip_labels_changed)
        layout.addWidget(self._clip_cb)

        layout.addSpacing(4)
        self._thumb_cb = QCheckBox(tr("3d.thumbnails"))
        self._thumb_cb.setChecked(True)
        self._thumb_cb.toggled.connect(self._on_thumb_toggle)
        layout.addWidget(self._thumb_cb)

        return frame

    def _on_thumb_toggle(self, checked: bool) -> None:
        self._thumb_side.setVisible(checked)


    def _render_plane(self, project: Project, axis: Axis, index: int) -> None:
        data_2d = Slice(project.seismic.data, axis, index).data
        rgba = apply_seismic_colormap(data_2d, self._cmap_name)
        texture = pv.numpy_to_texture(rgba)

        ni, nxl, nt = project.seismic.shape
        plane = _make_plane(axis, index, ni, nxl, nt)

        if axis in self._plane_actors:
            self._plotter.remove_actor(self._plane_actors[axis])

        actor = self._plotter.add_mesh(
            plane,
            texture=texture,
            show_scalar_bar=False,
            name=f"plane_{axis.name}",
            reset_camera=False,
        )
        self._plane_actors[axis] = actor


def _make_plane(
    axis: Axis, index: int, ni: int, nxl: int, nt: int
) -> pv.PolyData:
    """Build an axis-aligned quad at the exact slice position.

    World axes: X=inline, Y=crossline, Z=time.
    Uses explicit vertex positions instead of pv.Plane to avoid ambiguity
    in tangent-vector computation from the normal direction.
    Vertices are ordered CCW (bottom-left → bottom-right → top-right → top-left).
    """
    x = float(index)
    fni, fnxl, fnt = float(ni), float(nxl), float(nt)

    match axis:
        case Axis.INLINE:
            pts = np.array([
                [x, 0., 0.],
                [x, 0., fnt],
                [x, fnxl, fnt],
                [x, fnxl, 0.],
            ])
        case Axis.CROSSLINE:
            pts = np.array([
                [0., x, 0.],
                [0., x, fnt],
                [fni, x, fnt],
                [fni, x, 0.],
            ])
        case _:
            pts = np.array([
                [0., 0., x],
                [0., fnxl, x],
                [fni, fnxl, x],
                [fni, 0., x],
            ])

    mesh = pv.PolyData(pts, faces=np.array([4, 0, 1, 2, 3]))
    mesh.active_texture_coordinates = np.array([
        [0., 1.],
        [1., 1.],
        [1., 0.],
        [0., 0.],
    ])
    return mesh
