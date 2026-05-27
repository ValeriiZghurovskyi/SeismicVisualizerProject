"""NpzReader — reads seismic and label cubes from .npz files."""

from pathlib import Path

import numpy as np

from seismic_visualizer.domain.exceptions import ShapeMismatchError


class NpzReader:
    def read_seismic(self, path: Path) -> np.ndarray:
        """Load seismic cube from *path*; key must be ``"cube"``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        if "cube" not in data.files:
            raise KeyError(f"'cube' not found in {path}; available: {data.files}")
        arr = data["cube"]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return np.asarray(arr, dtype=np.uint8)

    def read_labels(
        self,
        path: Path,
        reference_shape: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """Load label cube from *path*; key must be ``"labels"``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        if "labels" not in data.files:
            raise KeyError(f"'labels' not found in {path}; available: {data.files}")
        arr = data["labels"]
        if reference_shape is not None and arr.shape != reference_shape:
            raise ShapeMismatchError(
                f"Label shape {arr.shape} != seismic shape {reference_shape}"
            )
        return np.asarray(arr)
