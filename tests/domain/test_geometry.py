"""Tests for domain.geometry — Axis, Point2D, Point3D."""

import pytest

from seismic_visualizer.domain.geometry import Axis, Point2D, Point3D


class TestAxisValues:
    def test_inline_is_zero(self) -> None:
        assert Axis.INLINE == 0

    def test_crossline_is_one(self) -> None:
        assert Axis.CROSSLINE == 1

    def test_time_is_two(self) -> None:
        assert Axis.TIME == 2

    def test_is_int_subtype(self) -> None:
        assert isinstance(Axis.INLINE, int)

    def test_construct_from_int(self) -> None:
        assert Axis(0) is Axis.INLINE
        assert Axis(1) is Axis.CROSSLINE
        assert Axis(2) is Axis.TIME

    def test_invalid_int_raises(self) -> None:
        with pytest.raises(ValueError):
            Axis(3)

    def test_all_axes_count(self) -> None:
        assert len(list(Axis)) == 3


class TestAxisDisplayName:
    def test_inline_display_name(self) -> None:
        assert Axis.INLINE.display_name == "Inline"

    def test_crossline_display_name(self) -> None:
        assert Axis.CROSSLINE.display_name == "Crossline"

    def test_time_display_name(self) -> None:
        assert Axis.TIME.display_name == "Time"

    def test_all_names_are_nonempty_strings(self) -> None:
        for axis in Axis:
            assert isinstance(axis.display_name, str)
            assert len(axis.display_name) > 0


class TestAxisIsLabelable:
    def test_inline_is_labelable(self) -> None:
        assert Axis.INLINE.is_labelable is True

    def test_crossline_is_labelable(self) -> None:
        assert Axis.CROSSLINE.is_labelable is True

    def test_time_is_not_labelable(self) -> None:
        assert Axis.TIME.is_labelable is False

    def test_exactly_two_labelable_axes(self) -> None:
        labelable = [a for a in Axis if a.is_labelable]
        assert len(labelable) == 2


class TestAxisOtherAxes:
    def test_inline_other_axes(self) -> None:
        assert Axis.INLINE.other_axes == (Axis.CROSSLINE, Axis.TIME)

    def test_crossline_other_axes(self) -> None:
        assert Axis.CROSSLINE.other_axes == (Axis.INLINE, Axis.TIME)

    def test_time_other_axes(self) -> None:
        assert Axis.TIME.other_axes == (Axis.INLINE, Axis.CROSSLINE)

    def test_other_axes_excludes_self(self) -> None:
        for axis in Axis:
            a, b = axis.other_axes
            assert a != axis
            assert b != axis

    def test_other_axes_are_distinct(self) -> None:
        for axis in Axis:
            a, b = axis.other_axes
            assert a != b

    def test_other_axes_lower_index_first(self) -> None:
        for axis in Axis:
            a, b = axis.other_axes
            assert int(a) < int(b)

    def test_other_axes_plus_self_covers_all(self) -> None:
        for axis in Axis:
            a, b = axis.other_axes
            assert set((axis, a, b)) == set(Axis)


class TestPoint2D:
    def test_creation_positional(self) -> None:
        p = Point2D(5, 10)
        assert p.row == 5
        assert p.col == 10

    def test_creation_keyword(self) -> None:
        p = Point2D(row=3, col=7)
        assert p.row == 3
        assert p.col == 7

    def test_immutable_row(self) -> None:
        p = Point2D(1, 2)
        with pytest.raises(AttributeError):
            p.row = 99  # type: ignore[misc]

    def test_immutable_col(self) -> None:
        p = Point2D(1, 2)
        with pytest.raises(AttributeError):
            p.col = 99  # type: ignore[misc]

    def test_equality_same(self) -> None:
        assert Point2D(3, 7) == Point2D(3, 7)

    def test_equality_different(self) -> None:
        assert Point2D(3, 7) != Point2D(7, 3)

    def test_hashable(self) -> None:
        points = {Point2D(0, 0), Point2D(1, 2), Point2D(0, 0)}
        assert len(points) == 2

    def test_usable_as_dict_key(self) -> None:
        mapping = {Point2D(1, 2): "hit"}
        assert mapping[Point2D(1, 2)] == "hit"

    def test_zero_coordinates_allowed(self) -> None:
        p = Point2D(0, 0)
        assert p.row == 0
        assert p.col == 0

    def test_negative_coordinates_allowed(self) -> None:
        p = Point2D(-1, -5)
        assert p.row == -1
        assert p.col == -5


class TestPoint3D:
    def test_creation_positional(self) -> None:
        p = Point3D(10, 20, 30)
        assert p.inline == 10
        assert p.crossline == 20
        assert p.time == 30

    def test_creation_keyword(self) -> None:
        p = Point3D(inline=1, crossline=2, time=3)
        assert p.inline == 1

    def test_immutable(self) -> None:
        p = Point3D(1, 2, 3)
        with pytest.raises(AttributeError):
            p.inline = 99  # type: ignore[misc]

    def test_equality_same(self) -> None:
        assert Point3D(1, 2, 3) == Point3D(1, 2, 3)

    def test_equality_different(self) -> None:
        assert Point3D(1, 2, 3) != Point3D(3, 2, 1)

    def test_hashable(self) -> None:
        points = {Point3D(0, 0, 0), Point3D(1, 2, 3), Point3D(0, 0, 0)}
        assert len(points) == 2

    def test_usable_as_dict_key(self) -> None:
        mapping = {Point3D(1, 2, 3): "voxel"}
        assert mapping[Point3D(1, 2, 3)] == "voxel"
