"""Horizon interpolation — fills gaps between annotated points on a slice."""

from seismic_visualizer.domain.geometry import Point2D


def interpolate_horizon(points: list[Point2D]) -> list[Point2D]:
    """Return one pixel per column spanning the full col range of points.

    Algorithm:
        1. Deduplicate by col (last point per col wins).
        2. Sort by col ascending.
        3. For each consecutive pair, linearly interpolate row for every
           integer col in [col1, col2].

    Args:
        points: Annotated pixels on a single 2D slice.

    Returns:
        Ordered list of Point2D (one per column, no gaps, no duplicates).
    """
    if len(points) <= 1:
        return list(points)

    by_col: dict[int, int] = {p.col: p.row for p in points}
    sorted_pts = sorted(by_col.items())

    result: list[Point2D] = []
    for (c1, r1), (c2, r2) in zip(sorted_pts, sorted_pts[1:], strict=False):
        span = c2 - c1
        for step in range(span):
            t = step / span
            row = round(r1 + t * (r2 - r1))
            result.append(Point2D(row, c1 + step))

    result.append(Point2D(sorted_pts[-1][1], sorted_pts[-1][0]))
    return result
