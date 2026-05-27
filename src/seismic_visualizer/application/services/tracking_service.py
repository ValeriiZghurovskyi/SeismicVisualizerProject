"""TrackingService — applies ML horizon tracking results to the label cube."""

from __future__ import annotations

from collections.abc import Callable


from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.geometry import Axis, Point3D
from seismic_visualizer.domain.ml.horizon_tracker import HorizonTracker


class TrackingService:
    """Orchestrates ML auto-tracking for horizons.

    Calls tracker.track(), then writes the returned voxel array into
    project.horizons exactly like InterpolationService.apply_interpolation().
    Out-of-bounds voxels are silently discarded.
    """

    def run(
        self,
        project: Project,
        entity_id: int,
        tracker: HorizonTracker,
        seed_points: list[Point3D],
        axis: Axis,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        """Auto-track a horizon and write results to the label cube.

        Args:
            project: The open project (mutated in place).
            entity_id: Horizon entity to label — must exist in horizon_registry.
            tracker: ML tracker implementation.
            seed_points: User seed voxels in 3D space.
            axis: Tracking propagation axis.

        Raises:
            EntityNotFoundError: If entity_id is not in horizon_registry.
        """
        project.horizon_registry.get(entity_id)

        voxels = tracker.track(project.seismic, seed_points, axis, cancel_check)

        if cancel_check is not None and cancel_check():
            return

        if len(voxels) == 0:
            return

        shape = project.horizons.data.shape
        in_bounds = (
            (voxels[:, 0] >= 0) & (voxels[:, 0] < shape[0])
            & (voxels[:, 1] >= 0) & (voxels[:, 1] < shape[1])
            & (voxels[:, 2] >= 0) & (voxels[:, 2] < shape[2])
        )
        pts = voxels[in_bounds]

        if len(pts) == 0:
            return

        project.horizons.data[pts[:, 0], pts[:, 1], pts[:, 2]] = entity_id
        project.is_dirty = True
