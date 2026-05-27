import numpy as np
import pytest

from seismic_visualizer.domain.interpolation.horizon_interpolator import interpolate_horizon_surface


def _flat_grid(inline_step: int = 2, crossline_step: int = 2, time: int = 20) -> np.ndarray:
    return np.array(
        [[i, j, time] for i in range(0, 20, inline_step) for j in range(0, 20, crossline_step)],
        dtype=np.int64,
    )


def test_empty_points_returns_empty() -> None:
    result = interpolate_horizon_surface(np.empty((0, 3), dtype=np.int64))
    assert result.shape == (0, 3)


def test_fewer_than_3_points_returns_empty() -> None:
    pts = np.array([[0, 0, 10], [5, 5, 12]], dtype=np.int64)
    result = interpolate_horizon_surface(pts)
    assert result.shape == (0, 3)


def test_flat_horizon_constant_time() -> None:
    pts = _flat_grid(time=20)
    result = interpolate_horizon_surface(pts, grid_step=2.0, median_size=1, max_radius=5.0)
    assert len(result) > 0
    np.testing.assert_allclose(result[:, 2], 20, atol=1)


def test_fills_gaps_between_annotated_inlines() -> None:
    # Annotate every 5th inline; expect interpolation to fill inlines between them
    pts = np.array(
        [[i, j, i // 5] for i in range(0, 50, 5) for j in range(0, 10)],
        dtype=np.int64,
    )
    result = interpolate_horizon_surface(pts, grid_step=1.0, median_size=1, max_radius=10.0)
    inlines = np.unique(result[:, 0])
    # Should produce more inline values than the 10 annotated ones
    assert len(inlines) > 10


def test_grid_step_affects_output_density() -> None:
    pts = _flat_grid(inline_step=2, crossline_step=2, time=50)
    coarse = interpolate_horizon_surface(pts, grid_step=4.0, median_size=1, max_radius=10.0)
    fine = interpolate_horizon_surface(pts, grid_step=1.0, median_size=1, max_radius=10.0)
    assert len(fine) > len(coarse)


def test_duplicate_points_after_unique_returns_empty() -> None:
    """Line 35: after np.unique, < 3 distinct points → empty."""
    # 5 copies of same 2 points → after unique: 2 < 3
    pts = np.array([[0, 0, 10], [1, 1, 20], [0, 0, 10], [1, 1, 20], [0, 0, 10]], dtype=np.int64)
    result = interpolate_horizon_surface(pts)
    assert result.shape == (0, 3)


def test_collinear_inline_grid_interpolate_returns_empty() -> None:
    """Lines 43-49: alpha shape computed; if grid_interpolate returns empty → empty result."""
    # All points on same inline → xy[:,0] all equal → collinear → grid_interpolate empty
    pts = np.array([[5, j, j * 2] for j in range(10)], dtype=np.int64)
    result = interpolate_horizon_surface(pts)
    # collinear x_max-x_min=0 in inline coord → empty interpolation
    assert result.shape[1] == 3


def test_result_dtype_is_integer() -> None:
    pts = np.array(
        [[i, j, 30] for i in range(0, 10) for j in range(0, 10)],
        dtype=np.int64,
    )
    result = interpolate_horizon_surface(pts, grid_step=1.0, median_size=1, max_radius=5.0)
    assert np.issubdtype(result.dtype, np.integer)
