"""Tests for interpolate_horizon — edge cases first."""

import pytest

from seismic_visualizer.domain.algorithms.interpolation import interpolate_horizon
from seismic_visualizer.domain.geometry import Point2D


# ---------------------------------------------------------------------------
# Edge cases: 0 and 1 point
# ---------------------------------------------------------------------------


def test_empty_returns_empty() -> None:
    assert interpolate_horizon([]) == []


def test_single_point_returns_that_point() -> None:
    result = interpolate_horizon([Point2D(5, 3)])
    assert result == [Point2D(5, 3)]


# ---------------------------------------------------------------------------
# Edge cases: duplicates and ordering
# ---------------------------------------------------------------------------


def test_two_points_same_col_dedup() -> None:
    """Two points in the same column — only one survives (last wins by col)."""
    result = interpolate_horizon([Point2D(2, 4), Point2D(7, 4)])
    cols = [p.col for p in result]
    assert cols.count(4) == 1


def test_reverse_order_same_as_forward() -> None:
    """Points given in descending col order must produce same result as ascending."""
    forward = interpolate_horizon([Point2D(0, 0), Point2D(4, 8)])
    backward = interpolate_horizon([Point2D(4, 8), Point2D(0, 0)])
    assert set(forward) == set(backward)


def test_unsorted_three_points() -> None:
    """Points out of col order must be sorted before interpolation."""
    # col order: 0, 5, 10 — row should be 0, 5, 10 (diagonal)
    pts = [Point2D(10, 10), Point2D(0, 0), Point2D(5, 5)]
    result = interpolate_horizon(pts)
    cols = [p.col for p in result]
    assert cols == sorted(cols)
    assert result[0].col == 0
    assert result[-1].col == 10


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------


def test_two_points_fills_all_columns() -> None:
    """Every column between p1 and p2 must have exactly one pixel."""
    result = interpolate_horizon([Point2D(0, 0), Point2D(0, 5)])
    cols = [p.col for p in result]
    assert cols == list(range(6))


def test_linear_interpolation_row_values() -> None:
    """Row must be round(lerp(row1, row2, t)) for each intermediate col."""
    result = interpolate_horizon([Point2D(0, 0), Point2D(10, 10)])
    for p in result:
        assert p.row == p.col  # perfect diagonal: row == col


def test_three_points_spans_full_range() -> None:
    pts = [Point2D(0, 0), Point2D(5, 5), Point2D(0, 10)]
    result = interpolate_horizon(pts)
    cols = [p.col for p in result]
    assert cols[0] == 0
    assert cols[-1] == 10
    assert cols == sorted(cols)


def test_no_duplicate_columns_in_output() -> None:
    pts = [Point2D(0, 0), Point2D(3, 6), Point2D(1, 12)]
    result = interpolate_horizon(pts)
    cols = [p.col for p in result]
    assert len(cols) == len(set(cols))


def test_output_is_list_of_point2d() -> None:
    result = interpolate_horizon([Point2D(0, 0), Point2D(4, 4)])
    assert isinstance(result, list)
    assert all(isinstance(p, Point2D) for p in result)
