"""3D fault surface interpolation via PCA-based plane alignment."""

import numpy as np

from seismic_visualizer.domain.interpolation._alpha_shape import compute_alpha_shape
from seismic_visualizer.domain.interpolation._grid_interp import grid_interpolate


def interpolate_fault_surface(
    points: np.ndarray,
    grid_step: float = 1.0,
    median_size: int = 3,
    max_radius: float = 10.0,
) -> np.ndarray:
    """Interpolate a fault surface from sparse 3D labeled voxels.

    Uses PCA to rotate the point cloud so the fault plane aligns with the
    first two principal components, interpolates in that plane, then rotates
    back to the original coordinate system.

    Args:
        points: (N, 3) int array — (inline, crossline, time) voxels.
        grid_step: Sampling step for the output grid (in PCA space).
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

    pts = points.astype(float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    projected = centered @ Vt.T
    xy = projected[:, :2]
    depth = projected[:, 2]

    alpha = 1.0 / max_radius
    try:
        boundary = compute_alpha_shape(xy, alpha)
    except Exception:  # pylint: disable=broad-exception-caught  # shapely/scipy failure modes are not typed
        boundary = None

    xi, yi, zi = grid_interpolate(xy, depth, grid_step, median_size, boundary)

    if len(xi) == 0:
        return np.empty((0, 3), dtype=np.int64)

    proj_interp = np.stack([xi, yi, zi], axis=1)
    original = proj_interp @ Vt + centroid

    return np.round(original).astype(np.int64)
