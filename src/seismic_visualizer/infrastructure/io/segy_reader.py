"""SegyReader — reads seismic data from SEG-Y files via segyio."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize float amplitude data to uint8 [0, 255] with midpoint 128 = zero."""
    max_abs = float(np.max(np.abs(data)))
    if max_abs == 0.0:
        return np.full(data.shape, 128, dtype=np.uint8)
    normalized = data / max_abs * 127.5 + 127.5
    return np.clip(np.round(normalized), 0, 255).astype(np.uint8)


class SegyReader:
    """Reads SEG-Y files and converts seismic data to the internal uint8 cube format."""

    def read_seismic(self, path: Path) -> np.ndarray:
        """Load a SEG-Y file and return a uint8 ndarray of shape (inlines, crosslines, samples).

        Attempts structured geometry first; falls back to unstructured (single inline).
        """
        import segyio

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        try:
            with segyio.open(str(path), ignore_geometry=False) as f:
                cube = segyio.tools.cube(f)
        except Exception:  # pylint: disable=broad-exception-caught  # segyio raises undocumented internal types
            with segyio.open(str(path), ignore_geometry=True) as f:
                n_samples = f.samples.size
                n_traces = f.tracecount
                traces = np.stack([f.trace[i] for i in range(n_traces)], axis=0)
                cube = traces.reshape(1, n_traces, n_samples)

        return _normalize_to_uint8(cube)

    def read_labels(
        self,
        path: Path,
        reference_shape: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "SEG-Y files do not contain label data. "
            "Load horizons/faults from separate .npz files after converting."
        )
