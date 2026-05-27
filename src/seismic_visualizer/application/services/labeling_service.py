"""LabelingService — dispatches polyline labeling to the correct cube and labeler."""

from __future__ import annotations

from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.state import SliceIndex
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.exceptions import LabelingNotAllowedError
from seismic_visualizer.domain.geometry import Point2D
from seismic_visualizer.domain.labeling.base import Labeler
from seismic_visualizer.domain.labeling.fault import fault_labeler
from seismic_visualizer.domain.labeling.horizon import horizon_labeler
from seismic_visualizer.domain.slice import Slice


class LabelingService:
    """Applies polyline labels to the correct cube based on entity kind.

    Callers pass an entity_id; the service resolves which registry and cube
    to use, selects the appropriate labeler, and marks the project dirty.
    """

    def apply(
        self,
        project: Project,
        entity_id: int,
        kind: EntityKind,
        slice_index: SliceIndex,
        points: list[Point2D],
        thickness: int = 1,
    ) -> None:
        """Draw a polyline of entity labels onto the appropriate label cube.

        Args:
            project: The open project (mutated in place).
            entity_id: ID of the entity to label.
            kind: Horizon or fault — determines which cube and registry to use.
            slice_index: Axis + position of the target slice.
            points: Ordered polyline vertices in slice pixel coordinates.

        Raises:
            LabelingNotAllowedError: If slice_index.axis is Time.
            EntityNotFoundError: If entity_id is not found in the correct registry.
        """
        if not slice_index.axis.is_labelable:
            raise LabelingNotAllowedError(
                f"Labeling is not allowed on {slice_index.axis.display_name} axis"
            )

        if kind == EntityKind.HORIZON:
            project.horizon_registry.get(entity_id)
            label_cube = project.horizons
            labeler = horizon_labeler(entity_id, thickness=thickness)
        else:
            project.fault_registry.get(entity_id)
            label_cube = project.faults
            labeler = fault_labeler(entity_id, thickness=thickness)

        sl = Slice(label_cube.data, slice_index.axis, slice_index.index)
        labeler.label_polyline(sl, points)
        project.is_dirty = True

    def erase(
        self,
        project: Project,
        kind: EntityKind,
        slice_index: SliceIndex,
        center: Point2D,
        radius: int,
    ) -> None:
        """Erase all labels within a circle on the given slice.

        Args:
            project: The open project (mutated in place).
            kind: Which cube to erase from (horizons or faults).
            slice_index: Axis + position of the target slice.
            center: Centre of the eraser in slice coordinates.
            radius: Eraser radius in pixels.
        """
        if not slice_index.axis.is_labelable:
            return
        label_cube = project.horizons if kind == EntityKind.HORIZON else project.faults
        sl = Slice(label_cube.data, slice_index.axis, slice_index.index)
        Labeler.erase_at(sl, center, radius)
        project.is_dirty = True
