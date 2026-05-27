"""Integration: write → read → arrays must be identical."""

from pathlib import Path

import numpy as np

from seismic_visualizer.infrastructure.io.npz_reader import NpzReader
from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter


SHAPE = (8, 10, 12)
RNG = np.random.default_rng(0)


def test_seismic_roundtrip(tmp_path: Path) -> None:
    original = RNG.integers(0, 256, SHAPE, dtype=np.uint8)
    path = tmp_path / "s_seismic.npz"
    NpzWriter().write_seismic(path, original)
    loaded = NpzReader().read_seismic(path)
    np.testing.assert_array_equal(loaded, original)


def test_labels_roundtrip(tmp_path: Path) -> None:
    original = RNG.integers(0, 201, SHAPE, dtype=np.uint8)
    path = tmp_path / "s_horizons.npz"
    NpzWriter().write_labels(path, original)
    loaded = NpzReader().read_labels(path)
    np.testing.assert_array_equal(loaded, original)


def test_seismic_and_labels_roundtrip_independent(tmp_path: Path) -> None:
    seismic = RNG.integers(0, 256, SHAPE, dtype=np.uint8)
    horizons = RNG.integers(0, 201, SHAPE, dtype=np.uint8)
    faults = RNG.integers(0, 201, SHAPE, dtype=np.uint8)

    writer = NpzWriter()
    writer.write_seismic(tmp_path / "cube_seismic.npz", seismic)
    writer.write_labels(tmp_path / "cube_horizons.npz", horizons)
    writer.write_labels(tmp_path / "cube_faults.npz", faults)

    reader = NpzReader()
    np.testing.assert_array_equal(reader.read_seismic(tmp_path / "cube_seismic.npz"), seismic)
    np.testing.assert_array_equal(reader.read_labels(tmp_path / "cube_horizons.npz"), horizons)
    np.testing.assert_array_equal(reader.read_labels(tmp_path / "cube_faults.npz"), faults)
