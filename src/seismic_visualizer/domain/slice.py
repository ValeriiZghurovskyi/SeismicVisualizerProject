"""Slice — axis-aware 2D view into a 3D cube.

All index arithmetic is encapsulated here so callers never write
`if axis == 0/1/2` conditionals. See CLAUDE.md §4.2.
"""

from __future__ import annotations

import numpy as np

from seismic_visualizer.domain.exceptions import IndexOutOfBoundsError
from seismic_visualizer.domain.geometry import Axis, Point3D


class Slice:
    """Axis-aware 2D view into a raw 3D numpy array.

    Construct via SeismicCube.data or LabelCube.data:

        s = Slice(label_cube.data, Axis.INLINE, 42)
        s.write_pixel(row, col, entity_id)

    Args:
        cube: Raw 3D numpy array from a SeismicCube or LabelCube.
        axis: The axis along which to slice.
        index: Position along that axis.

    Raises:
        IndexOutOfBoundsError: If index is outside [0, cube.shape[axis] - 1].
    """

    def __init__(self, cube: np.ndarray, axis: Axis, index: int) -> None:
        max_idx = int(cube.shape[axis]) - 1
        if not 0 <= index <= max_idx:
            raise IndexOutOfBoundsError(
                f"Index {index} out of range [0, {max_idx}] "
                f"for {axis.display_name} axis"
            )
        self._cube = cube
        self._axis = axis
        self._index = index

    @property
    def axis(self) -> Axis:
        """The axis along which this slice was taken."""
        return self._axis

    @property
    def index(self) -> int:
        """Position of this slice along its axis."""
        return self._index

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the 2D slice: (row_count, col_count).

        Row and column axes follow numpy convention (lower-index axis first).
        """
        row_axis, col_axis = self._axis.other_axes
        return int(self._cube.shape[row_axis]), int(self._cube.shape[col_axis])

    @property
    def data(self) -> np.ndarray:
        """2D view into the underlying 3D array at this slice position.

        The view is read-only when the backing array is read-only
        (i.e. SeismicCube.data). Use write_pixel / write_mask to
        mutate LabelCube-backed slices.
        """
        return self._view()

    def write_pixel(self, row: int, col: int, value: int) -> None:
        """Write a single pixel into the underlying 3D array via this slice."""
        self._view()[row, col] = value

    def write_mask(self, mask: np.ndarray, value: int) -> None:
        """Write `value` to every True position in `mask`."""
        self._view()[mask] = value

    def to_point3d(self, row: int, col: int) -> Point3D:
        """Convert 2D slice coordinates to a 3D voxel position.

        Uses ``other_axes`` so no axis-index conditionals are needed.
        """
        row_ax, col_ax = self._axis.other_axes
        coords = [0, 0, 0]
        coords[int(self._axis)] = self._index
        coords[int(row_ax)] = row
        coords[int(col_ax)] = col
        return Point3D(inline=coords[0], crossline=coords[1], time=coords[2])

    def _view(self) -> np.ndarray:
        idx: list[int | slice] = [slice(None), slice(None), slice(None)]
        idx[int(self._axis)] = self._index
        return self._cube[tuple(idx)]
