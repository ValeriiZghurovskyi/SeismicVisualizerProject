"""CubeReader Protocol and reader factory."""

from pathlib import Path
from typing import Protocol

import numpy as np


class CubeReader(Protocol):
    def read_seismic(self, path: Path) -> np.ndarray: ...

    def read_labels(
        self, path: Path, reference_shape: tuple[int, int, int] | None = None
    ) -> np.ndarray: ...


_SEGY_SUFFIXES = {".segy", ".sgy"}


def get_reader(path: Path) -> CubeReader:
    """Return the appropriate reader for *path* based on file extension."""
    if Path(path).suffix.lower() in _SEGY_SUFFIXES:
        from seismic_visualizer.infrastructure.io.segy_reader import SegyReader
        return SegyReader()
    from seismic_visualizer.infrastructure.io.npz_reader import NpzReader
    return NpzReader()
