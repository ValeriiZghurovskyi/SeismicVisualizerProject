"""View3DPresenter — coordinates 3D visualization without touching PyVista."""

from __future__ import annotations

import math

import numpy as np

from seismic_visualizer.application.dto import EntityInfo, RandomLinePlaneDTO
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.view_interfaces import LabelMesh, View3DProtocol
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.geometry import Axis


class View3DPresenter:
    """Bridges Project data and View3DProtocol without PyVista knowledge.

    Args:
        project: The currently open project.
        view: 3D view implementing View3DProtocol.
    """

    def __init__(self, project: Project, view: View3DProtocol) -> None:
        self._project = project
        self._view = view
        self._slice_indices: dict[Axis, int] = {
            Axis.INLINE: 0,
            Axis.CROSSLINE: 0,
            Axis.TIME: 0,
        }
        self._plane_visible: dict[Axis, bool] = {
            Axis.INLINE: True,
            Axis.CROSSLINE: True,
            Axis.TIME: True,
        }
        self._coords_cache: dict[tuple[int, EntityKind], np.ndarray] = {}

    def set_plane_visible(self, axis: Axis, visible: bool) -> None:
        """Toggle one plane's visibility in the stored state and notify the view."""
        self._plane_visible[axis] = visible
        self._view.set_plane_visible(axis, visible)

    def sync_indices(self, indices: dict[Axis, int]) -> None:
        """Silently update stored indices without touching the view."""
        self._slice_indices.update(indices)

    def on_plane_position_changed(self, axis: Axis, index: int) -> None:
        """Update the stored index for one axis and notify the view."""
        self._slice_indices[axis] = index
        self._view.set_plane_position(axis, index)

    def refresh_planes(self) -> None:
        """Tell the view to re-render all three seismic slice planes."""
        self._view.update_seismic_planes(self._project, dict(self._slice_indices))

    def invalidate_label_cache(self, entity_id: int | None = None, kind: EntityKind | None = None) -> None:
        """Drop cached argwhere results. Call after each labeling stroke."""
        if entity_id is None:
            self._coords_cache.clear()
        else:
            self._coords_cache.pop((entity_id, kind), None)

    def refresh_labels(
        self,
        entities: list[EntityInfo],
        clip_to_slices: bool = False,
    ) -> None:
        """Extract voxel coords for each entity and push to the view.

        Hidden entities are still included so the view can keep their actors
        around and toggle visibility cheaply (no geometry rebuild on toggle).

        Args:
            entities: Entity list to render.
            clip_to_slices: If True, only voxels that lie on at least one of
                the three current slice planes are shown.
        """
        missing_by_kind: dict[EntityKind, list[int]] = {}
        for info in entities:
            if (info.id, info.kind) not in self._coords_cache:
                missing_by_kind.setdefault(info.kind, []).append(info.id)
        for kind, missing_ids in missing_by_kind.items():
            cube_data = (
                self._project.horizons.data
                if kind == EntityKind.HORIZON
                else self._project.faults.data
            )
            self._populate_cache(cube_data, kind, missing_ids)

        meshes: list[LabelMesh] = []
        for info in entities:
            coords = self._coords_cache[(info.id, info.kind)]
            if clip_to_slices and coords.size > 0:
                mask = np.zeros(len(coords), dtype=bool)
                for axis, idx in self._slice_indices.items():
                    mask |= coords[:, axis.value] == idx
                coords = coords[mask]
            if coords.size > 0:
                meshes.append((coords, info))
        self._view.update_label_meshes(meshes)

    def set_entity_visible(self, entity_id: int, kind: EntityKind, visible: bool) -> None:
        """Fast-path visibility toggle: flips one actor without rebuild."""
        self._view.set_entity_visible(entity_id, kind, visible)

    def _populate_cache(
        self, cube_data: np.ndarray, kind: EntityKind, target_ids: list[int]
    ) -> None:
        """Fill cache for all target_ids with one cube scan (vs. one per id)."""
        nz = np.argwhere(cube_data > 0)
        if nz.size == 0:
            empty = np.empty((0, 3), dtype=np.int64)
            for eid in target_ids:
                self._coords_cache[(eid, kind)] = empty
            return
        values = cube_data[nz[:, 0], nz[:, 1], nz[:, 2]]
        for eid in target_ids:
            self._coords_cache[(eid, kind)] = nz[values == eid]

    def compute_random_line(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
    ) -> None:
        """Compute the random-line plane geometry and push it to the view.

        The plane is vertical (spans the full TIME axis) and passes through
        both seed points in the inline-crossline plane.

        Args:
            p1: First seed point as (inline, crossline).
            p2: Second seed point as (inline, crossline).
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return

        normal = (-dy / length, dx / length, 0.0)

        shape = self._project.seismic.shape
        origin = (
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
            shape[2] / 2.0,
        )

        dto = RandomLinePlaneDTO(
            origin=origin,
            normal=normal,
            width=length,
            height=float(shape[2]),
        )
        self._view.show_random_line_plane(dto)

    def push_thumbnail(self, axis: Axis, rgba: np.ndarray) -> None:
        """Delegate a 2D slice thumbnail update to the 3D view."""
        self._view.update_slice_thumbnail(axis, rgba)
