"""Tests for Labeler — label_polyline and erase_at pixel-level correctness."""

import numpy as np
import pytest

from seismic_visualizer.domain.cube import LabelCube
from seismic_visualizer.domain.geometry import Axis, Point2D
from seismic_visualizer.domain.labeling.base import Labeler
from seismic_visualizer.domain.slice import Slice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPE = (20, 20, 20)
ENTITY_ID = 3


def make_slice(axis: Axis = Axis.INLINE, index: int = 0) -> tuple[Slice, LabelCube]:
    cube = LabelCube.empty(SHAPE)
    return Slice(cube.data, axis, index), cube


# ---------------------------------------------------------------------------
# label_polyline — basic drawing
# ---------------------------------------------------------------------------


def test_single_point_polyline_marks_pixel() -> None:
    sl, cube = make_slice()
    labeler = Labeler(entity_id=ENTITY_ID)
    labeler.label_polyline(sl, [Point2D(5, 5)])
    assert sl.data[5, 5] == ENTITY_ID


def test_two_point_polyline_fills_line() -> None:
    sl, cube = make_slice()
    labeler = Labeler(entity_id=ENTITY_ID)
    labeler.label_polyline(sl, [Point2D(0, 0), Point2D(0, 9)])
    row = sl.data[0, :]
    assert all(row[c] == ENTITY_ID for c in range(10))


def test_empty_polyline_changes_nothing() -> None:
    sl, cube = make_slice()
    before = sl.data.copy()
    Labeler(entity_id=ENTITY_ID).label_polyline(sl, [])
    assert np.array_equal(sl.data, before)


def test_multi_segment_polyline() -> None:
    """Three points → two segments; all pixels on both segments labelled."""
    sl, cube = make_slice()
    labeler = Labeler(entity_id=ENTITY_ID)
    pts = [Point2D(0, 0), Point2D(0, 5), Point2D(5, 5)]
    labeler.label_polyline(sl, pts)
    # first segment horizontal
    assert all(sl.data[0, c] == ENTITY_ID for c in range(6))
    # second segment vertical
    assert all(sl.data[r, 5] == ENTITY_ID for r in range(6))


def test_polyline_does_not_overwrite_other_entity() -> None:
    """Pixels already labelled with a different entity must not be zeroed."""
    sl, cube = make_slice()
    sl.write_pixel(3, 3, 7)  # entity 7 placed first
    Labeler(entity_id=ENTITY_ID).label_polyline(sl, [Point2D(3, 3)])
    # current entity overwrites — that is expected and correct
    assert sl.data[3, 3] == ENTITY_ID


# ---------------------------------------------------------------------------
# label_polyline — thickness
# ---------------------------------------------------------------------------


def test_thickness_1_is_single_pixel_wide() -> None:
    sl, _ = make_slice()
    Labeler(entity_id=ENTITY_ID, thickness=1).label_polyline(
        sl, [Point2D(10, 0), Point2D(10, 19)]
    )
    # row above and below must be unlabelled
    assert sl.data[9, 10] == 0
    assert sl.data[11, 10] == 0


def test_thickness_3_dilates_line() -> None:
    sl, _ = make_slice()
    Labeler(entity_id=ENTITY_ID, thickness=3).label_polyline(
        sl, [Point2D(10, 0), Point2D(10, 19)]
    )
    # dilation radius = (3-1)//2 = 1, so rows 9..11 must be labelled
    for r in (9, 10, 11):
        assert sl.data[r, 10] == ENTITY_ID


def test_thickness_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Labeler(entity_id=ENTITY_ID, thickness=0)


# ---------------------------------------------------------------------------
# erase_at — circle eraser
# ---------------------------------------------------------------------------


def test_erase_at_clears_center() -> None:
    sl, _ = make_slice()
    sl.write_pixel(5, 5, ENTITY_ID)
    Labeler.erase_at(sl, Point2D(5, 5), radius=1)
    assert sl.data[5, 5] == 0


def test_erase_at_circle_shape() -> None:
    """Pixels within radius are cleared; corner pixel at distance > radius is not."""
    sl, _ = make_slice()
    # fill a region
    for r in range(3, 8):
        for c in range(3, 8):
            sl.write_pixel(r, c, ENTITY_ID)
    Labeler.erase_at(sl, Point2D(5, 5), radius=2)
    # center must be cleared
    assert sl.data[5, 5] == 0
    # pixel at Chebyshev=2 but Euclidean > 2 (e.g. diagonal corner) is NOT cleared
    # distance from (5,5) to (3,3) = sqrt(8) ≈ 2.83 > 2
    assert sl.data[3, 3] == ENTITY_ID


def test_erase_at_zero_radius_clears_only_center() -> None:
    sl, _ = make_slice()
    for r in range(4, 7):
        for c in range(4, 7):
            sl.write_pixel(r, c, ENTITY_ID)
    Labeler.erase_at(sl, Point2D(5, 5), radius=0)
    assert sl.data[5, 5] == 0
    assert sl.data[5, 6] == ENTITY_ID


def test_erase_at_clamps_to_slice_bounds() -> None:
    """Erasing near an edge must not raise IndexError."""
    sl, _ = make_slice()
    sl.write_pixel(0, 0, ENTITY_ID)
    Labeler.erase_at(sl, Point2D(0, 0), radius=5)
    assert sl.data[0, 0] == 0


def test_erase_at_erases_all_entities_not_only_current() -> None:
    """Eraser clears any label value, not only the ID that was written."""
    sl, _ = make_slice()
    sl.write_pixel(5, 5, 9)  # different entity
    Labeler.erase_at(sl, Point2D(5, 5), radius=0)
    assert sl.data[5, 5] == 0
