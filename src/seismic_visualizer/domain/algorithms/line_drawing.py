"""Bresenham line-drawing algorithm for rasterizing polylines onto label slices."""

from skimage.draw import line as sk_line

from seismic_visualizer.domain.geometry import Point2D


def draw_line(p1: Point2D, p2: Point2D) -> list[Point2D]:
    """Return all pixels on the Bresenham line from p1 to p2 (inclusive).

    Args:
        p1: Start pixel.
        p2: End pixel.

    Returns:
        Ordered list of Point2D from p1 to p2 with no duplicates.
    """
    rows, cols = sk_line(p1.row, p1.col, p2.row, p2.col)  # type: ignore[no-untyped-call]
    return [Point2D(int(r), int(c)) for r, c in zip(rows, cols, strict=False)]
