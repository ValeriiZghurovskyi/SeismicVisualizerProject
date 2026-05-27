"""InterpolationService — two-step API: preview then apply."""

from __future__ import annotations

import numpy as np

from seismic_visualizer.application.dto import InterpolationParams, InterpolationResult
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.interpolation.fault_interpolator import interpolate_fault_surface
from seismic_visualizer.domain.interpolation.horizon_interpolator import interpolate_horizon_surface


class InterpolationService:
    """Orchestrates surface interpolation for horizons and faults.

    Two-step workflow:
        result = service.interpolate_entity(kind, entity_id, params)  # preview
        service.apply_interpolation(result)                            # commit

    Args:
        project: The open project whose label cubes are read and written.
    """

    def __init__(self, project: Project) -> None:
        self._project = project

    def interpolate_entity(
        self,
        entity_kind: EntityKind,
        entity_id: int,
        params: InterpolationParams,
    ) -> InterpolationResult:
        """Compute interpolated surface voxels without modifying the project.

        Reads all labeled voxels for *entity_id* from the appropriate cube,
        runs the domain interpolator, and returns the result for preview or
        subsequent apply.

        Args:
            entity_kind: Horizon or fault — determines which cube and algorithm.
            entity_id: ID of the entity to interpolate.
            params: Interpolation parameters.

        Returns:
            InterpolationResult with (M, 3) int64 point array.
            points is empty (shape (0, 3)) when fewer than 3 labeled voxels exist.

        Raises:
            EntityNotFoundError: If entity_id is not registered.
        """
        if entity_kind == EntityKind.HORIZON:
            self._project.horizon_registry.get(entity_id)
            cube = self._project.horizons
            interpolate = interpolate_horizon_surface
        else:
            self._project.fault_registry.get(entity_id)
            cube = self._project.faults
            interpolate = interpolate_fault_surface

        labeled_voxels = np.argwhere(cube.data == entity_id)

        points = interpolate(
            labeled_voxels,
            grid_step=params.grid_step,
            median_size=params.median_size,
            max_radius=params.max_radius,
        )

        return InterpolationResult(
            entity_kind=entity_kind,
            entity_id=entity_id,
            points=points,
        )

    def apply_interpolation(self, result: InterpolationResult) -> None:
        """Write interpolated voxels into the label cube.

        Out-of-bounds voxels are silently discarded.
        Only sets is_dirty when at least one voxel is written.

        Args:
            result: The result previously returned by interpolate_entity.
        """
        if len(result.points) == 0:
            return

        if result.entity_kind == EntityKind.HORIZON:
            cube = self._project.horizons
        else:
            cube = self._project.faults

        shape = cube.data.shape
        pts = result.points

        in_bounds = (
            (pts[:, 0] >= 0) & (pts[:, 0] < shape[0])
            & (pts[:, 1] >= 0) & (pts[:, 1] < shape[1])
            & (pts[:, 2] >= 0) & (pts[:, 2] < shape[2])
        )
        pts = pts[in_bounds]

        if len(pts) == 0:
            return

        cube.data[pts[:, 0], pts[:, 1], pts[:, 2]] = result.entity_id
        self._project.is_dirty = True
