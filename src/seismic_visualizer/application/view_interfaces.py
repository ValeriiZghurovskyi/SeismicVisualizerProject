"""View protocols — View2DProtocol and View3DProtocol.

Presenter depends on these interfaces, never on concrete Qt classes.
See CLAUDE.md §4.6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from seismic_visualizer.application.dto import EntityInfo, RandomLinePlaneDTO
from seismic_visualizer.domain.geometry import Axis

LabelMesh = tuple[np.ndarray, EntityInfo]

if TYPE_CHECKING:
    from seismic_visualizer.application.services.project import Project


class View2DProtocol(Protocol):
    """Interface the 2D view must satisfy.

    Presenter calls these methods; the concrete Qt widget implements them.
    No Qt symbols appear here — domain and application layers stay Qt-free.
    """

    def update_canvas(self, rgba: np.ndarray) -> None:
        """Replace the displayed image with a new RGBA frame.

        Args:
            rgba: uint8 array of shape (H, W, 4).
        """
        ...

    def update_entity_list(self, entities: list[EntityInfo]) -> None:
        """Rebuild the sidebar entity list from the given snapshot.

        Args:
            entities: Ordered list of entity display data.
        """
        ...

    def update_status(self, text: str) -> None:
        """Set the status bar text.

        Args:
            text: Formatted status string, e.g. "Slice 42/499 | Inline".
        """
        ...

    def clear_stroke_preview(self) -> None:
        """Remove the point-click preview overlay from the canvas."""
        ...

    def set_stroke_color(self, color: tuple[int, int, int] | None) -> None:
        """Set the active entity color for canvas preview drawing.

        Args:
            color: RGB tuple, or None to disable drawing.
        """
        ...

    def remove_last_stroke_point(self) -> None:
        """Remove the most recently placed preview point from the canvas."""
        ...

    def confirm_entity_delete(self, name: str) -> bool:
        """Show a confirmation dialog before irreversibly deleting an entity.

        Args:
            name: Display name of the entity to delete.

        Returns:
            True if the user confirmed, False if cancelled.
        """
        ...


class View3DProtocol(Protocol):
    """Interface the 3D view must satisfy (M8).

    All methods are Qt/PyVista-free so the presenter stays testable without
    a display server.
    """

    def update_seismic_planes(
        self, project: Project, slice_indices: dict[Axis, int]
    ) -> None:
        """Re-render all three orthogonal seismic slice planes.

        Args:
            project: The currently open project (provides seismic data).
            slice_indices: Current position for each axis.
        """
        ...

    def set_plane_position(self, axis: Axis, index: int) -> None:
        """Move a single slice plane to a new position without full re-render.

        Args:
            axis: Which plane to move.
            index: New voxel index along that axis.
        """
        ...

    def update_label_meshes(self, meshes: list[LabelMesh]) -> None:
        """Re-build 3D label geometry from pre-extracted voxel coordinates.

        Args:
            meshes: Each entry is (coords, entity_info) where coords is an
                (N, 3) int array of (inline, crossline, time) voxel positions.
                Entities are included regardless of visibility; the view applies
                ``info.visible`` to the actor so toggles can flip visibility
                without rebuilding geometry.
        """
        ...

    def set_entity_visible(self, entity_id: int, kind: object, visible: bool) -> None:
        """Toggle a single entity label actor's visibility (no geometry rebuild).

        Args:
            entity_id: Entity id whose label cloud should be shown/hidden.
            kind: EntityKind — horizon and fault id spaces overlap, so kind
                is needed to disambiguate.
            visible: True = show, False = hide.
        """
        ...

    def update_entity_list(self, entities: list[EntityInfo]) -> None:
        """Rebuild the sidebar entity list (mirrors View2DProtocol.update_entity_list).

        Args:
            entities: Ordered list of entity display data.
        """
        ...

    def show_random_line_plane(self, dto: RandomLinePlaneDTO) -> None:
        """Render a vertical random-line plane computed from two seed points.

        Args:
            dto: Pre-computed plane geometry in data-space coordinates.
        """
        ...

    def set_plane_visible(self, axis: Axis, visible: bool) -> None:
        """Show or hide a single seismic slice plane actor.

        Args:
            axis: Which plane to toggle.
            visible: True = show, False = hide.
        """
        ...

    def update_slice_thumbnail(self, axis: Axis, rgba: np.ndarray) -> None:
        """Update the mini 2D thumbnail overlay for a given axis.

        Args:
            axis: Which slice thumbnail to update.
            rgba: uint8 RGBA array of shape (H, W, 4) — the current slice image.
        """
        ...

    def preview_interpolation(
        self, points: np.ndarray, color: tuple[int, int, int]
    ) -> None:
        """Show interpolated voxels as a semi-transparent preview cloud.

        Replaces any existing preview.  Call clear_interpolation_preview() to
        remove it without adding a new one.

        Args:
            points: (M, 3) int array of (inline, crossline, time) voxels.
            color: RGB tuple (0-255) for the preview point cloud.
        """
        ...

    def clear_interpolation_preview(self) -> None:
        """Remove the interpolation preview mesh, if any."""
        ...


class ToolbarProtocol(Protocol):
    """Interface the main toolbar must satisfy for presenter updates.

    Only the subset of toolbar methods used by presenters is included here.
    """

    def set_active_entity(self, name: str, color: tuple[int, int, int] | None) -> None:
        """Update the active entity indicator.

        Args:
            name: Display name of the active entity.
            color: RGB color, or None to show placeholder.
        """
        ...
