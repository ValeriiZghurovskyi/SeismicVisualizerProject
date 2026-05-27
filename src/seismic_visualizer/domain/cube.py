"""SeismicCube (read-only wrapper) and LabelCube (mutable uint8).

See CLAUDE.md §2.2 — three separate cubes so a voxel can hold
both a horizon label and a fault label simultaneously.
"""

from __future__ import annotations

import numpy as np


class SeismicCube:
    """Immutable 3D seismic amplitude array.

    Exposes a read-only view so downstream code cannot accidentally
    modify raw seismic data during interpretation.

    Args:
        data: 3-dimensional numeric numpy array (float32 / int8 / uint8 typical).

    Raises:
        ValueError: If data is not 3D or has a non-numeric dtype.
    """

    def __init__(self, data: np.ndarray) -> None:
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
        if data.dtype.kind not in "fiu":
            raise ValueError(
                f"Expected numeric dtype (float / int / uint), got {data.dtype}"
            )
        self._data: np.ndarray = data.view()
        self._data.flags.writeable = False

    @property
    def shape(self) -> tuple[int, int, int]:
        """Cube dimensions as (n_inline, n_crossline, n_time)."""
        h, w, d = self._data.shape
        return int(h), int(w), int(d)

    @property
    def data(self) -> np.ndarray:
        """Read-only 3D view of the seismic amplitude values."""
        return self._data


class LabelCube:
    """Mutable 3D label array (uint8).

    Voxel encoding: 0 = unlabeled, 1..200 = entity ID.
    A separate LabelCube exists for horizons and for faults, allowing
    both to occupy the same voxel without collision.

    Args:
        data: 3-dimensional uint8 numpy array, same shape as SeismicCube.

    Raises:
        ValueError: If data is not 3D or dtype is not uint8.
    """

    MAX_ENTITY_ID: int = 200

    def __init__(self, data: np.ndarray) -> None:
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
        if data.dtype != np.uint8:
            raise ValueError(f"LabelCube requires uint8 dtype, got {data.dtype}")
        self._data = data

    @classmethod
    def empty(cls, shape: tuple[int, int, int]) -> LabelCube:
        """Create an all-zero (fully unlabeled) cube of the given shape."""
        return cls(np.zeros(shape, dtype=np.uint8))

    @property
    def shape(self) -> tuple[int, int, int]:
        """Cube dimensions as (n_inline, n_crossline, n_time)."""
        h, w, d = self._data.shape
        return int(h), int(w), int(d)

    @property
    def data(self) -> np.ndarray:
        """Mutable reference to the underlying uint8 label array."""
        return self._data
