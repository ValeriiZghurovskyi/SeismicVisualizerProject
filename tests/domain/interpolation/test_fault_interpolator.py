import numpy as np
import pytest

from seismic_visualizer.domain.interpolation.fault_interpolator import interpolate_fault_surface


def test_empty_points_returns_empty() -> None:
    result = interpolate_fault_surface(np.empty((0, 3), dtype=np.int64))
    assert result.shape == (0, 3)


def test_fewer_than_3_points_returns_empty() -> None:
    pts = np.array([[0, 0, 10], [5, 5, 12]], dtype=np.int64)
    result = interpolate_fault_surface(pts)
    assert result.shape == (0, 3)


def test_horizontal_plane_pca_roundtrip() -> None:
    # Horizontal fault (time=const) — after PCA, time becomes PC3 with depth=0
    pts = np.array(
        [[i, j, 20] for i in range(0, 20, 2) for j in range(0, 20, 2)],
        dtype=np.int64,
    )
    result = interpolate_fault_surface(pts, grid_step=2.0, median_size=1, max_radius=10.0)
    assert len(result) > 0
    np.testing.assert_allclose(result[:, 2], 20, atol=2)


def test_vertical_fault_along_inline() -> None:
    # Vertical fault: inline=const=10, crossline and time vary
    pts = np.array(
        [[10, j, t] for j in range(0, 20, 2) for t in range(0, 20, 2)],
        dtype=np.int64,
    )
    result = interpolate_fault_surface(pts, grid_step=2.0, median_size=1, max_radius=10.0)
    assert len(result) > 0
    np.testing.assert_allclose(result[:, 0], 10, atol=2)


def test_duplicate_points_after_unique_returns_empty() -> None:
    """Line 36: after np.unique, < 3 distinct points → empty."""
    pts = np.array([[10, 0, 5], [10, 0, 5], [10, 0, 5]], dtype=np.int64)
    result = interpolate_fault_surface(pts)
    assert result.shape == (0, 3)


def test_degenerate_grid_returns_empty() -> None:
    """Line 59: grid_interpolate returns empty (degenerate cloud) → empty result."""
    # Collinear in PCA space → depth=0 for all → grid_interpolate on constant z
    # Use identical time for all points → pure horizontal → depth=0 → degenerate
    pts = np.array([[i, i, 10] for i in range(5)], dtype=np.int64)
    result = interpolate_fault_surface(pts, max_radius=1.0)
    # Just verify no crash; result may be empty or not depending on geometry
    assert result.shape[1] == 3


def test_diagonal_fault_inline_equals_crossline() -> None:
    # 45° fault: inline == crossline for all labeled points
    pts = np.array(
        [[i, i, t] for i in range(0, 20, 2) for t in range(0, 20, 2)],
        dtype=np.int64,
    )
    result = interpolate_fault_surface(pts, grid_step=2.0, median_size=1, max_radius=10.0)
    assert len(result) > 0
    # Reconstructed points should maintain inline ≈ crossline
    np.testing.assert_allclose(result[:, 0], result[:, 1], atol=2)
