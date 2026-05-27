"""Tests for draw_line — pixel-level correctness."""

import pytest

from seismic_visualizer.domain.algorithms.line_drawing import draw_line
from seismic_visualizer.domain.geometry import Point2D


# ---------------------------------------------------------------------------
# Trivial / degenerate cases
# ---------------------------------------------------------------------------


def test_same_point_returns_single_pixel() -> None:
    result = draw_line(Point2D(3, 5), Point2D(3, 5))
    assert result == [Point2D(3, 5)]


def test_horizontal_line() -> None:
    result = draw_line(Point2D(0, 0), Point2D(0, 4))
    assert result == [Point2D(0, c) for c in range(5)]


def test_vertical_line() -> None:
    result = draw_line(Point2D(0, 2), Point2D(4, 2))
    assert result == [Point2D(r, 2) for r in range(5)]


# ---------------------------------------------------------------------------
# Connectivity: every returned pixel is 8-connected to its neighbours
# ---------------------------------------------------------------------------


def _max_chebyshev_step(points: list[Point2D]) -> int:
    """Return the largest Chebyshev distance between consecutive points."""
    return max(
        max(abs(b.row - a.row), abs(b.col - a.col))
        for a, b in zip(points, points[1:])
    )


def test_diagonal_line_8_connected() -> None:
    result = draw_line(Point2D(0, 0), Point2D(5, 5))
    assert len(result) >= 6
    assert _max_chebyshev_step(result) == 1


def test_steep_line_8_connected() -> None:
    result = draw_line(Point2D(0, 0), Point2D(10, 3))
    assert _max_chebyshev_step(result) == 1


def test_shallow_line_8_connected() -> None:
    result = draw_line(Point2D(0, 0), Point2D(3, 10))
    assert _max_chebyshev_step(result) == 1


# ---------------------------------------------------------------------------
# Endpoints included
# ---------------------------------------------------------------------------


def test_endpoints_always_included() -> None:
    p1, p2 = Point2D(2, 7), Point2D(9, 1)
    result = draw_line(p1, p2)
    assert result[0] == p1
    assert result[-1] == p2


def test_reverse_direction_same_pixels() -> None:
    """draw_line(A→B) and draw_line(B→A) must cover identical pixel sets."""
    p1, p2 = Point2D(1, 1), Point2D(6, 4)
    forward = set(draw_line(p1, p2))
    backward = set(draw_line(p2, p1))
    assert forward == backward


# ---------------------------------------------------------------------------
# No duplicates
# ---------------------------------------------------------------------------


def test_no_duplicate_pixels() -> None:
    result = draw_line(Point2D(0, 0), Point2D(7, 3))
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_list_of_point2d() -> None:
    result = draw_line(Point2D(0, 0), Point2D(2, 2))
    assert isinstance(result, list)
    assert all(isinstance(p, Point2D) for p in result)
