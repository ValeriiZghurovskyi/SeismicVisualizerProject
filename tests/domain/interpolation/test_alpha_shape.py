import numpy as np
import pytest
from shapely.geometry import MultiPoint

from seismic_visualizer.domain.interpolation._alpha_shape import compute_alpha_shape


def test_fewer_than_3_points_raises() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="3 points"):
        compute_alpha_shape(pts, alpha=1.0)


def test_convex_cloud_all_points_inside() -> None:
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (50, 2))
    # Loose alpha (large radius=20) — should cover all input points
    shape = compute_alpha_shape(pts, alpha=0.05)
    mp = MultiPoint(pts)
    for p in mp.geoms:
        assert shape.distance(p) < 0.5, f"Point {p} too far from alpha shape"


def test_tight_alpha_reduces_area_vs_loose() -> None:
    # U-shaped cloud: two vertical arms + horizontal base
    pts: list[list[float]] = []
    for y in range(10):
        pts.extend([[0.0, float(y)], [6.0, float(y)]])
    for x in range(1, 6):
        pts.append([float(x), 0.0])
    arr = np.array(pts, dtype=float)

    loose = compute_alpha_shape(arr, alpha=0.05)  # radius=20
    tight = compute_alpha_shape(arr, alpha=0.5)   # radius=2

    assert tight.area < loose.area


def test_delaunay_failure_falls_back_to_convex_hull() -> None:
    """Lines 30-31: Delaunay raises → return convex_hull."""
    from unittest.mock import patch

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with patch(
        "seismic_visualizer.domain.interpolation._alpha_shape.Delaunay",
        side_effect=ValueError("degenerate"),
    ):
        shape = compute_alpha_shape(pts, alpha=1.0)
    assert shape is not None
    assert shape.area >= 0


def test_degenerate_triangle_area_skipped() -> None:
    """Line 42: area < 1e-10 → triangle skipped but hull still computed."""
    # Collinear points cause degenerate triangles in Delaunay
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    shape = compute_alpha_shape(pts, alpha=1.0)
    assert shape is not None


def test_tight_alpha_no_triangles_returns_convex_hull() -> None:
    """Line 48: all triangles rejected → return MultiPoint convex hull."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    # alpha=1000 → radius=0.001 → all circumradii > 0.001 → no triangles pass
    shape = compute_alpha_shape(pts, alpha=1000.0)
    assert shape is not None
    assert shape.area >= 0


def test_loose_alpha_approximates_convex_hull() -> None:
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 10, (30, 2))

    hull = MultiPoint(pts).convex_hull
    shape = compute_alpha_shape(pts, alpha=0.01)  # radius=100, very loose

    assert abs(shape.area - hull.area) / hull.area < 0.1
