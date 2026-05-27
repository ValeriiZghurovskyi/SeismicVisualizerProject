"""Tests for NpzWriter."""

from pathlib import Path

import numpy as np
import pytest

from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter


SHAPE = (4, 5, 6)


def test_write_seismic_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out_seismic.npz"
    data = np.zeros(SHAPE, dtype=np.uint8)
    NpzWriter().write_seismic(path, data)
    assert path.exists()


def test_write_labels_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out_horizons.npz"
    data = np.zeros(SHAPE, dtype=np.uint8)
    NpzWriter().write_labels(path, data)
    assert path.exists()


def test_write_creates_intermediate_dirs(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c_seismic.npz"
    NpzWriter().write_seismic(path, np.zeros(SHAPE, dtype=np.uint8))
    assert path.exists()


def test_write_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "out_seismic.npz"
    writer = NpzWriter()
    first = np.ones(SHAPE, dtype=np.uint8)
    second = np.full(SHAPE, 42, dtype=np.uint8)
    writer.write_seismic(path, first)
    writer.write_seismic(path, second)
    result = np.load(path)["cube"]
    assert (result == 42).all()


def test_write_labels_stores_correct_key(tmp_path: Path) -> None:
    path = tmp_path / "out_horizons.npz"
    NpzWriter().write_labels(path, np.zeros(SHAPE, dtype=np.uint8))
    assert "labels" in np.load(path).files


def test_write_seismic_stores_correct_key(tmp_path: Path) -> None:
    path = tmp_path / "out_seismic.npz"
    NpzWriter().write_seismic(path, np.zeros(SHAPE, dtype=np.uint8))
    assert "cube" in np.load(path).files
