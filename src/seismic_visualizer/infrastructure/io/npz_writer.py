"""NpzWriter — saves label cubes to .npz files."""

from pathlib import Path

import numpy as np


class NpzWriter:
    def write_seismic(self, path: Path, data: np.ndarray) -> None:
        """Save *data* under key ``"cube"``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, cube=data)

    def write_labels(self, path: Path, data: np.ndarray) -> None:
        """Save *data* under key ``"labels"``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=data)
