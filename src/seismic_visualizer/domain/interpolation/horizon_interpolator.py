"""3D horizon surface interpolation from sparse labeled voxels."""

import numpy as np

from seismic_visualizer.domain.interpolation._alpha_shape import compute_alpha_shape
from seismic_visualizer.domain.interpolation._grid_interp import grid_interpolate


def interpolate_horizon_surface(
    points: np.ndarray,
    grid_step: float = 1.0,
    median_size: int = 3,
    max_radius: float = 10.0,
) -> np.ndarray:
    """Interpolate a horizon surface from sparse 3D labeled voxels.

    Treats (inline, crossline) as the 2D domain and interpolates the time
    value. Alpha shape limits the fill to the annotated region.

    Args:
        points: (N, 3) int array — (inline, crossline, time) voxels.
        grid_step: Sampling step for the output grid.
        median_size: Smoothing kernel size (1 = no smoothing).
        max_radius: Rolling-ball radius for alpha shape clipping.

    Returns:
        (M, 3) int64 array of interpolated (inline, crossline, time) voxels.
        Empty (0, 3) array if fewer than 3 input points.
    """
    if len(points) < 3:
        return np.empty((0, 3), dtype=np.int64)

    points = np.unique(points, axis=0)
    if len(points) < 3:
        return np.empty((0, 3), dtype=np.int64)

    xy = points[:, :2].astype(float)
    z = points[:, 2].astype(float)

    alpha = 1.0 / max_radius
    try:
        boundary = compute_alpha_shape(xy, alpha)
    except Exception:  # pylint: disable=broad-exception-caught  # shapely/scipy failure modes are not typed
        boundary = None

    xi, yi, zi = grid_interpolate(xy, z, grid_step, median_size, boundary)

    if len(xi) == 0:
        return np.empty((0, 3), dtype=np.int64)

    result = np.stack([xi, yi, zi], axis=1)
    return np.round(result).astype(np.int64)
