"""Tests for infrastructure.io.readers — get_reader factory."""

from __future__ import annotations

from pathlib import Path

from seismic_visualizer.infrastructure.io.npz_reader import NpzReader
from seismic_visualizer.infrastructure.io.readers import get_reader
from seismic_visualizer.infrastructure.io.segy_reader import SegyReader


def test_get_reader_returns_npz_reader_for_npz(tmp_path: Path) -> None:
    path = tmp_path / "data.npz"
    reader = get_reader(path)
    assert isinstance(reader, NpzReader)


def test_get_reader_returns_segy_reader_for_segy(tmp_path: Path) -> None:
    path = tmp_path / "data.segy"
    reader = get_reader(path)
    assert isinstance(reader, SegyReader)


def test_get_reader_returns_segy_reader_for_sgy(tmp_path: Path) -> None:
    path = tmp_path / "data.sgy"
    reader = get_reader(path)
    assert isinstance(reader, SegyReader)


def test_get_reader_case_insensitive_segy(tmp_path: Path) -> None:
    reader = get_reader(tmp_path / "data.SEGY")
    assert isinstance(reader, SegyReader)


def test_get_reader_case_insensitive_sgy(tmp_path: Path) -> None:
    reader = get_reader(tmp_path / "data.SGY")
    assert isinstance(reader, SegyReader)


def test_get_reader_returns_npz_reader_for_unknown_ext(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    reader = get_reader(path)
    assert isinstance(reader, NpzReader)


def test_get_reader_accepts_string_path(tmp_path: Path) -> None:
    reader = get_reader(str(tmp_path / "data.npz"))
    assert isinstance(reader, NpzReader)
