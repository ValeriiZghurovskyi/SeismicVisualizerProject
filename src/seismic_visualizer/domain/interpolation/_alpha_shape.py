"""Alpha shape (concave hull) via rolling-ball on Delaunay triangulation."""

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def compute_alpha_shape(points: np.ndarray, alpha: float) -> BaseGeometry:
    """Concave hull of a 2D point cloud.

    Args:
        points: (N, 2) float array of 2D coordinates.
        alpha: Reciprocal of the rolling-ball radius. Larger = tighter hull.

    Returns:
        shapely Polygon or MultiPolygon representing the alpha shape.

    Raises:
        ValueError: If fewer than 3 points are provided.
    """
    if len(points) < 3:
        raise ValueError(f"Need at least 3 points, got {len(points)}")

    radius = 1.0 / alpha

    try:
        tri = Delaunay(points)
    except (ValueError, RuntimeError):
        return MultiPoint(points).convex_hull

    triangles: list[Polygon] = []
    for simplex in tri.simplices:
        pts = points[simplex]
        a = np.linalg.norm(pts[1] - pts[0])
        b = np.linalg.norm(pts[2] - pts[1])
        c = np.linalg.norm(pts[0] - pts[2])
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        if area < 1e-10:
            continue
        circumradius = (a * b * c) / (4.0 * area)
        if circumradius <= radius:
            triangles.append(Polygon(pts))

    if not triangles:
        return MultiPoint(points).convex_hull

    return unary_union(triangles)
