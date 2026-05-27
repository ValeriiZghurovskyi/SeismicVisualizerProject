"""FaultTracker — domain Protocol for ML-based fault auto-tracking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from seismic_visualizer.domain.cube import SeismicCube
from seismic_visualizer.domain.geometry import Axis, Point3D


@runtime_checkable
class FaultTracker(Protocol):
    """Contract for an ML model that auto-tracks a fault in a seismic cube.

    Implementations live in infrastructure/ml/ (ONNX, PyTorch, etc.).
    Domain and application layers depend only on this Protocol.
    """

    def track(
        self,
        seismic: SeismicCube,
        seed_points: list[Point3D],
        axis: Axis,
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Predict fault voxels starting from seed points.

        Args:
            seismic: Read-only seismic amplitude cube.
            seed_points: User-provided seed voxel coordinates in 3D space.
            axis: Axis along which tracking propagates (INLINE or CROSSLINE).
            cancel_check: Optional callable; tracking stops between slices when it
                returns True. Partial results collected up to that point are returned.

        Returns:
            Integer array of shape (N, 3) where each row is (inline, crossline, time).
            May be empty (shape (0, 3)) when tracking yields no result.
        """
        ...
