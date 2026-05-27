"""Tests for SegyReader using a synthetic SEG-Y fixture."""

from pathlib import Path

import numpy as np
import pytest

segyio = pytest.importorskip("segyio")

from seismic_visualizer.infrastructure.io.segy_reader import SegyReader, _normalize_to_uint8


SHAPE = (4, 5, 20)  # inlines × crosslines × samples


@pytest.fixture()
def segy_file(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.segy"
    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = segyio.SegySampleFormat.IBM_FLOAT_4_BYTE
    spec.ilines = np.arange(SHAPE[0], dtype=np.int32)
    spec.xlines = np.arange(SHAPE[1], dtype=np.int32)
    spec.samples = np.arange(SHAPE[2], dtype=np.float32)
    with segyio.create(str(path), spec) as f:
        for il_idx, il in enumerate(spec.ilines):
            for xl_idx, xl in enumerate(spec.xlines):
                tr_idx = il_idx * SHAPE[1] + xl_idx
                data = np.sin(np.linspace(-np.pi, np.pi, SHAPE[2])).astype(np.float32)
                f.trace[tr_idx] = data
                f.header[tr_idx] = {
                    segyio.TraceField.INLINE_3D: int(il),
                    segyio.TraceField.CROSSLINE_3D: int(xl),
                }
    return path


def test_read_seismic_shape(segy_file: Path) -> None:
    arr = SegyReader().read_seismic(segy_file)
    assert arr.shape == SHAPE


def test_read_seismic_dtype_uint8(segy_file: Path) -> None:
    arr = SegyReader().read_seismic(segy_file)
    assert arr.dtype == np.uint8


def test_read_seismic_range(segy_file: Path) -> None:
    arr = SegyReader().read_seismic(segy_file)
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_read_seismic_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SegyReader().read_seismic(tmp_path / "ghost.segy")


def test_read_labels_raises(segy_file: Path) -> None:
    with pytest.raises(NotImplementedError):
        SegyReader().read_labels(segy_file)


def test_normalize_zero_data() -> None:
    data = np.zeros((2, 3, 4), dtype=np.float32)
    result = _normalize_to_uint8(data)
    assert result.shape == data.shape
    assert np.all(result == 128)


def test_read_seismic_unstructured_fallback(tmp_path: Path) -> None:
    """Lines 36-42: fallback to ignore_geometry=True when structured open fails."""
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    path = tmp_path / "fallback.segy"
    path.touch()

    mock_file = MagicMock()
    mock_file.samples.size = 4
    mock_file.tracecount = 2
    trace_data = {0: np.zeros(4, dtype=np.float32), 1: np.ones(4, dtype=np.float32)}
    mock_file.trace = trace_data

    calls = [0]

    @contextmanager
    def fake_open(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise Exception("no geometry")
        yield mock_file

    mock_segyio = MagicMock()
    mock_segyio.open.side_effect = fake_open

    with patch.dict("sys.modules", {"segyio": mock_segyio}):
        reader = SegyReader()
        result = reader.read_seismic(path)

    assert result.dtype == np.uint8
    assert result.shape == (1, 2, 4)


def test_normalize_preserves_midpoint() -> None:
    data = np.array([[[0.0]]], dtype=np.float32)
    result = _normalize_to_uint8(np.array([[[0.0], [1.0], [-1.0]]], dtype=np.float32))
    # zero amplitude → 128, max positive → 255, max negative → 0
    assert result[0, 0, 0] == 128
    assert result[0, 1, 0] == 255
    assert result[0, 2, 0] == 0
