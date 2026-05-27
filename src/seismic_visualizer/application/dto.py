"""Application-layer data transfer objects.

These are plain Python dataclasses that cross the boundary between
Application and UI layers — no Qt, no domain mutation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seismic_visualizer.domain.entities import EntityKind


@dataclass(frozen=True)
class EntityInfo:
    """Read-only snapshot of an entity for display in the UI.

    Args:
        id: Entity identifier (1..200).
        name: Human-readable label.
        kind: Horizon or fault.
        visible: Current visibility flag.
        color: RGB tuple derived from the domain palette (0-255 per channel).
    """

    id: int
    name: str
    kind: EntityKind
    visible: bool
    color: tuple[int, int, int]


@dataclass(frozen=True)
class RandomLinePlaneDTO:
    """Geometry of a random-line vertical plane in data-space coordinates.

    All values use voxel units (1 unit = 1 voxel).

    Args:
        origin: Centre of the plane (inline, crossline, time).
        normal: Unit normal vector of the plane (perpendicular to line direction).
        width: Horizontal extent — distance between the two seed points.
        height: Vertical extent — full time dimension of the cube.
    """

    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    width: float
    height: float


@dataclass(frozen=True)
class InterpolationParams:
    """Parameters controlling the interpolation algorithm.

    Args:
        grid_step: Sampling step for the output grid (voxel units).
        median_size: Kernel size for median smoothing filter (1 = no smoothing).
        max_radius: Rolling-ball radius for alpha shape clipping (voxel units).
    """

    grid_step: float = 1.0
    median_size: int = 3
    max_radius: float = 10.0


@dataclass
class InterpolationResult:
    """Output of a surface interpolation operation.

    Not frozen — holds a mutable numpy array.

    Args:
        entity_kind: Horizon or fault.
        entity_id: ID of the interpolated entity.
        points: (M, 3) int64 array of (inline, crossline, time) voxels.
    """

    entity_kind: EntityKind
    entity_id: int
    points: np.ndarray = field(repr=False)
