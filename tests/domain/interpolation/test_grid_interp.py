import numpy as np
import pytest
from shapely.geometry import box

from seismic_visualizer.domain.interpolation._grid_interp import grid_interpolate


def _regular_grid(n: int = 5, z_val: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([[float(i), float(j)] for i in range(n) for j in range(n)])
    z = np.full(len(xy), z_val)
    return xy, z


def test_flat_plane_z_const() -> None:
    xy, z = _regular_grid(6, z_val=5.0)
    xi, yi, zi = grid_interpolate(xy, z, grid_step=1.0, median_size=1)
    assert len(zi) > 0
    np.testing.assert_allclose(zi, 5.0, atol=1e-6)


def test_linear_gradient_recovers_analytical() -> None:
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10, (40, 2))
    z = xy[:, 0] + xy[:, 1]
    xi, yi, zi = grid_interpolate(xy, z, grid_step=0.5, median_size=1)
    expected = xi + yi
    np.testing.assert_allclose(zi, expected, atol=0.5)


def test_boundary_clips_output() -> None:
    xy = np.array([[float(i), float(j)] for i in range(10) for j in range(10)])
    z = np.ones(len(xy))
    region = box(2.0, 2.0, 6.0, 6.0)
    xi, yi, zi = grid_interpolate(xy, z, grid_step=1.0, median_size=1, boundary=region)
    assert len(xi) > 0
    # All returned points must lie within the bounding box (with float tolerance)
    assert np.all(xi >= 2.0 - 1e-6)
    assert np.all(xi <= 6.0 + 1e-6)
    assert np.all(yi >= 2.0 - 1e-6)
    assert np.all(yi <= 6.0 + 1e-6)


def test_median_size_1_flat_z_unchanged() -> None:
    xy, z = _regular_grid(6, z_val=3.0)
    xi, yi, zi = grid_interpolate(xy, z, grid_step=1.0, median_size=1)
    np.testing.assert_allclose(zi, 3.0, atol=1e-10)


def test_fewer_than_3_unique_points_returns_empty() -> None:
    """Line 31: < 3 unique points → empty result."""
    xy = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])  # all duplicates
    z = np.array([5.0, 5.0, 5.0])
    xi, yi, zi = grid_interpolate(xy, z, grid_step=1.0, median_size=1)
    assert len(xi) == 0


def test_collinear_x_returns_empty() -> None:
    """Line 38: x_max - x_min < 1e-9 (all same x) → empty result."""
    xy = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # all x=1
    z = np.array([1.0, 2.0, 3.0])
    xi, yi, zi = grid_interpolate(xy, z, grid_step=1.0, median_size=1)
    assert len(xi) == 0


def test_median_filter_applied_with_size_3() -> None:
    """Lines 50-53: median_size > 1 branch runs."""
    xy, z = _regular_grid(6, z_val=3.0)
    xi, yi, zi = grid_interpolate(xy, z, grid_step=1.0, median_size=3)
    assert len(zi) > 0


def test_no_boundary_returns_interior_points() -> None:
    xy, z = _regular_grid(5, z_val=1.0)
    xi, yi, zi = grid_interpolate(xy, z, grid_step=1.0, median_size=1)
    # Without boundary, all grid points inside convex hull should be returned
    assert len(xi) >= len(xy)
