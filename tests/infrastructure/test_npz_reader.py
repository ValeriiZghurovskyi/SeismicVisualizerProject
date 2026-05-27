"""Tests for NpzReader."""

from pathlib import Path

import numpy as np
import pytest

from seismic_visualizer.domain.exceptions import ShapeMismatchError
from seismic_visualizer.infrastructure.io.npz_reader import NpzReader


SHAPE = (4, 5, 6)


@pytest.fixture()
def seismic_file(tmp_path: Path) -> Path:
    path = tmp_path / "data_seismic.npz"
    np.savez_compressed(path, cube=np.zeros(SHAPE, dtype=np.uint8))
    return path


@pytest.fixture()
def labels_file(tmp_path: Path) -> Path:
    path = tmp_path / "data_horizons.npz"
    np.savez_compressed(path, labels=np.zeros(SHAPE, dtype=np.uint8))
    return path


# --- valid cases ---


def test_read_seismic_returns_correct_array(seismic_file: Path) -> None:
    reader = NpzReader()
    arr = reader.read_seismic(seismic_file)
    assert arr.shape == SHAPE
    assert arr.dtype == np.uint8


def test_read_labels_returns_correct_array(labels_file: Path) -> None:
    reader = NpzReader()
    arr = reader.read_labels(labels_file)
    assert arr.shape == SHAPE
    assert arr.dtype == np.uint8


def test_read_labels_validates_shape_against_reference(
    tmp_path: Path, labels_file: Path
) -> None:
    reader = NpzReader()
    reader.read_labels(labels_file, reference_shape=SHAPE)  # must not raise


# --- invalid cases ---


def test_read_seismic_missing_file_raises(tmp_path: Path) -> None:
    reader = NpzReader()
    with pytest.raises(FileNotFoundError):
        reader.read_seismic(tmp_path / "ghost.npz")


def test_read_seismic_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, wrong_key=np.zeros(SHAPE, dtype=np.uint8))
    reader = NpzReader()
    with pytest.raises(KeyError):
        reader.read_seismic(path)


def test_read_labels_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, wrong_key=np.zeros(SHAPE, dtype=np.uint8))
    reader = NpzReader()
    with pytest.raises(KeyError):
        reader.read_labels(path)


def test_read_seismic_wrong_dtype_is_cast(tmp_path: Path) -> None:
    """float32 data stored by external tools must be loadable (cast to uint8)."""
    path = tmp_path / "float.npz"
    np.savez_compressed(path, cube=np.zeros(SHAPE, dtype=np.float32))
    reader = NpzReader()
    arr = reader.read_seismic(path)
    assert arr.dtype == np.uint8


def test_read_labels_shape_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "labels.npz"
    np.savez_compressed(path, labels=np.zeros((3, 3, 3), dtype=np.uint8))
    reader = NpzReader()
    with pytest.raises(ShapeMismatchError):
        reader.read_labels(path, reference_shape=SHAPE)
