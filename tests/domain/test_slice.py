"""Tests for domain.slice — Slice (axis-aware 2D view into a 3D array).

Cube shape used throughout: (inline=4, crossline=5, time=6).

Expected slice shapes:
  INLINE    slice → data.shape == (5, 6)   [crossline × time]
  CROSSLINE slice → data.shape == (4, 6)   [inline    × time]
  TIME      slice → data.shape == (4, 5)   [inline    × crossline]
"""

import numpy as np
import pytest

from seismic_visualizer.domain.cube import SeismicCube
from seismic_visualizer.domain.exceptions import IndexOutOfBoundsError
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.domain.slice import Slice

N_INLINE, N_CROSS, N_TIME = 4, 5, 6
SHAPE = (N_INLINE, N_CROSS, N_TIME)


@pytest.fixture
def seismic() -> np.ndarray:
    return np.arange(120, dtype=np.float32).reshape(SHAPE)


@pytest.fixture
def labels() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction and bounds
# ---------------------------------------------------------------------------


class TestSliceCreation:
    def test_create_inline(self, seismic: np.ndarray) -> None:
        s = Slice(seismic, Axis.INLINE, 0)
        assert s.axis is Axis.INLINE
        assert s.index == 0

    def test_create_crossline(self, seismic: np.ndarray) -> None:
        s = Slice(seismic, Axis.CROSSLINE, 0)
        assert s.axis is Axis.CROSSLINE

    def test_create_time(self, seismic: np.ndarray) -> None:
        s = Slice(seismic, Axis.TIME, 0)
        assert s.axis is Axis.TIME

    def test_max_valid_index(self, seismic: np.ndarray) -> None:
        s = Slice(seismic, Axis.INLINE, N_INLINE - 1)
        assert s.index == N_INLINE - 1

    def test_negative_index_raises(self, seismic: np.ndarray) -> None:
        with pytest.raises(IndexOutOfBoundsError):
            Slice(seismic, Axis.INLINE, -1)

    def test_index_equal_to_size_raises(self, seismic: np.ndarray) -> None:
        with pytest.raises(IndexOutOfBoundsError):
            Slice(seismic, Axis.INLINE, N_INLINE)  # max valid = N_INLINE-1

    def test_index_far_out_of_range_raises(self, seismic: np.ndarray) -> None:
        with pytest.raises(IndexOutOfBoundsError):
            Slice(seismic, Axis.TIME, 9999)

    @pytest.mark.parametrize("axis,size", [
        (Axis.INLINE, N_INLINE),
        (Axis.CROSSLINE, N_CROSS),
        (Axis.TIME, N_TIME),
    ])
    def test_last_valid_index_each_axis(
        self, seismic: np.ndarray, axis: Axis, size: int
    ) -> None:
        s = Slice(seismic, axis, size - 1)
        assert s.index == size - 1


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class TestSliceShape:
    def test_inline_slice_shape(self, seismic: np.ndarray) -> None:
        assert Slice(seismic, Axis.INLINE, 0).shape == (N_CROSS, N_TIME)

    def test_crossline_slice_shape(self, seismic: np.ndarray) -> None:
        assert Slice(seismic, Axis.CROSSLINE, 0).shape == (N_INLINE, N_TIME)

    def test_time_slice_shape(self, seismic: np.ndarray) -> None:
        assert Slice(seismic, Axis.TIME, 0).shape == (N_INLINE, N_CROSS)

    def test_data_ndim_is_2(self, seismic: np.ndarray) -> None:
        for axis in Axis:
            s = Slice(seismic, axis, 0)
            assert s.data.ndim == 2


# ---------------------------------------------------------------------------
# Data correctness
# ---------------------------------------------------------------------------


class TestSliceData:
    def test_inline_values_match(self, seismic: np.ndarray) -> None:
        s = Slice(seismic, Axis.INLINE, 2)
        np.testing.assert_array_equal(s.data, seismic[2, :, :])

    def test_crossline_values_match(self, seismic: np.ndarray) -> None:
        s = Slice(seismic, Axis.CROSSLINE, 3)
        np.testing.assert_array_equal(s.data, seismic[:, 3, :])

    def test_time_values_match(self, seismic: np.ndarray) -> None:
        s = Slice(seismic, Axis.TIME, 4)
        np.testing.assert_array_equal(s.data, seismic[:, :, 4])

    def test_data_reflects_external_changes(self) -> None:
        cube = np.zeros(SHAPE, dtype=np.float32)
        s = Slice(cube, Axis.INLINE, 0)
        cube[0, 2, 3] = 99.0
        assert s.data[2, 3] == 99.0


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


class TestWritePixel:
    def test_inline_write_pixel(self, labels: np.ndarray) -> None:
        Slice(labels, Axis.INLINE, 1).write_pixel(2, 3, 7)
        assert labels[1, 2, 3] == 7

    def test_crossline_write_pixel(self, labels: np.ndarray) -> None:
        Slice(labels, Axis.CROSSLINE, 2).write_pixel(1, 4, 5)
        assert labels[1, 2, 4] == 5

    def test_time_write_pixel(self, labels: np.ndarray) -> None:
        Slice(labels, Axis.TIME, 3).write_pixel(0, 1, 3)
        assert labels[0, 1, 3] == 3

    def test_write_does_not_affect_adjacent_slice(self, labels: np.ndarray) -> None:
        Slice(labels, Axis.INLINE, 0).write_pixel(0, 0, 5)
        s2 = Slice(labels, Axis.INLINE, 1)
        assert s2.data[0, 0] == 0

    def test_write_pixel_all_axes(self) -> None:
        cube = np.zeros(SHAPE, dtype=np.uint8)
        Slice(cube, Axis.INLINE,    1).write_pixel(2, 3, 11)
        Slice(cube, Axis.CROSSLINE, 2).write_pixel(1, 4, 22)
        Slice(cube, Axis.TIME,      3).write_pixel(0, 1, 33)

        assert cube[1, 2, 3] == 11  # INLINE=1,    row=cross=2,  col=time=3
        assert cube[1, 2, 4] == 22  # CROSSLINE=2, row=inline=1, col=time=4
        assert cube[0, 1, 3] == 33  # TIME=3,      row=inline=0, col=cross=1


class TestWriteMask:
    def test_mask_sets_targeted_pixels(self, labels: np.ndarray) -> None:
        s = Slice(labels, Axis.INLINE, 0)
        mask = np.zeros((N_CROSS, N_TIME), dtype=bool)
        mask[1, 2] = True
        mask[3, 4] = True
        s.write_mask(mask, 9)
        assert labels[0, 1, 2] == 9
        assert labels[0, 3, 4] == 9

    def test_mask_leaves_unset_pixels_untouched(self, labels: np.ndarray) -> None:
        s = Slice(labels, Axis.INLINE, 0)
        mask = np.zeros((N_CROSS, N_TIME), dtype=bool)
        mask[0, 0] = True
        s.write_mask(mask, 5)
        assert labels[0, 1, 1] == 0

    def test_full_mask_sets_all(self, labels: np.ndarray) -> None:
        s = Slice(labels, Axis.INLINE, 0)
        s.write_mask(np.ones((N_CROSS, N_TIME), dtype=bool), 1)
        assert np.all(labels[0, :, :] == 1)
        assert np.all(labels[1:, :, :] == 0)


# ---------------------------------------------------------------------------
# to_point3d
# ---------------------------------------------------------------------------


class TestToPoint3D:
    """Verify that (row, col) on a slice maps to correct 3D voxel coords.

    Cube shape: (inline=4, crossline=5, time=6).

    INLINE   slice: other_axes = (CROSSLINE, TIME)  → row=crossline, col=time
    CROSSLINE slice: other_axes = (INLINE,    TIME)  → row=inline,    col=time
    TIME      slice: other_axes = (INLINE,    CROSSLINE) → row=inline, col=crossline
    """

    def test_inline_slice_point3d(self, seismic: np.ndarray) -> None:
        p = Slice(seismic, Axis.INLINE, 2).to_point3d(3, 4)
        assert (p.inline, p.crossline, p.time) == (2, 3, 4)

    def test_crossline_slice_point3d(self, seismic: np.ndarray) -> None:
        p = Slice(seismic, Axis.CROSSLINE, 2).to_point3d(1, 4)
        assert (p.inline, p.crossline, p.time) == (1, 2, 4)

    def test_time_slice_point3d(self, seismic: np.ndarray) -> None:
        p = Slice(seismic, Axis.TIME, 3).to_point3d(0, 1)
        assert (p.inline, p.crossline, p.time) == (0, 1, 3)

    @pytest.mark.parametrize("axis,idx,row,col,expected", [
        (Axis.INLINE,     2, 3, 4, (2, 3, 4)),
        (Axis.CROSSLINE,  2, 1, 4, (1, 2, 4)),
        (Axis.TIME,       3, 0, 1, (0, 1, 3)),
    ])
    def test_data_roundtrip(
        self,
        seismic: np.ndarray,
        axis: Axis,
        idx: int,
        row: int,
        col: int,
        expected: tuple[int, int, int],
    ) -> None:
        """Point3D coords index into the same voxel as slice.data[row, col]."""
        s = Slice(seismic, axis, idx)
        p = s.to_point3d(row, col)
        assert seismic[p.inline, p.crossline, p.time] == s.data[row, col]

    def test_corner_origin(self, seismic: np.ndarray) -> None:
        p = Slice(seismic, Axis.INLINE, 0).to_point3d(0, 0)
        assert (p.inline, p.crossline, p.time) == (0, 0, 0)

    def test_corner_max(self, seismic: np.ndarray) -> None:
        p = Slice(seismic, Axis.TIME, N_TIME - 1).to_point3d(N_INLINE - 1, N_CROSS - 1)
        assert (p.inline, p.crossline, p.time) == (N_INLINE - 1, N_CROSS - 1, N_TIME - 1)


class TestWriteToReadOnlyRaises:
    def test_seismic_slice_write_pixel_raises(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float32))
        s = Slice(cube.data, Axis.INLINE, 0)
        with pytest.raises(ValueError):
            s.write_pixel(0, 0, 1)

    def test_seismic_slice_write_mask_raises(self) -> None:
        cube = SeismicCube(np.zeros(SHAPE, dtype=np.float32))
        s = Slice(cube.data, Axis.INLINE, 0)
        mask = np.ones((N_CROSS, N_TIME), dtype=bool)
        with pytest.raises(ValueError):
            s.write_mask(mask, 1)
