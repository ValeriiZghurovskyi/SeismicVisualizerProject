"""Scatter-to-grid interpolation with optional alpha-shape clipping."""

import numpy as np
import shapely
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import median_filter
from shapely.geometry.base import BaseGeometry


def grid_interpolate(
    xy: np.ndarray,
    z: np.ndarray,
    grid_step: float,
    median_size: int,
    boundary: BaseGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered points onto a regular grid.

    Args:
        xy: (N, 2) 2D coordinates of known points.
        z: (N,) values at those points.
        grid_step: Spacing of the output grid.
        median_size: Kernel size for scipy.ndimage.median_filter (1 = no filter).
        boundary: Optional shapely geometry; points outside are discarded.

    Returns:
        (xi, yi, zi) 1D float arrays of valid interpolated points.
    """
    _empty = (np.empty(0), np.empty(0), np.empty(0))
    if len(np.unique(xy, axis=0)) < 3:
        return _empty

    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())

    if (x_max - x_min) < 1e-9 or (y_max - y_min) < 1e-9:
        return _empty

    xi_1d = np.arange(x_min, x_max + grid_step, grid_step)
    yi_1d = np.arange(y_min, y_max + grid_step, grid_step)
    grid_x, grid_y = np.meshgrid(xi_1d, yi_1d)

    interp = LinearNDInterpolator(xy, z)
    grid_z = interp(grid_x, grid_y)

    valid_mask = ~np.isnan(grid_z)

    if median_size > 1:
        fill_val = float(np.nanmean(grid_z)) if np.any(valid_mask) else 0.0
        grid_z_filled = np.where(valid_mask, grid_z, fill_val)
        grid_z_filtered = median_filter(grid_z_filled, size=median_size)
        grid_z = np.where(valid_mask, grid_z_filtered, np.nan)

    if boundary is not None:
        padded = boundary.buffer(1e-9)
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        inside = shapely.contains_xy(padded, flat_x, flat_y).reshape(grid_x.shape)
        grid_z[~inside] = np.nan
        valid_mask = ~np.isnan(grid_z)

    return grid_x[valid_mask], grid_y[valid_mask], grid_z[valid_mask]
