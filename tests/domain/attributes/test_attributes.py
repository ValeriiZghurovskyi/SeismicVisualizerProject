"""Tests for domain.attributes — all four attribute processors.

Each processor accepts a Slice (Inline or Crossline) and returns a 2D float32
array of the same shape. On Time slice they raise AttributeNotApplicableError.

Synthetic signal: a 10×20 array where each column is a sine wave (axis=1 is time).
"""

import numpy as np
import pytest

from seismic_visualizer.domain.attributes.envelope import compute_envelope
from seismic_visualizer.domain.attributes.phase import compute_instantaneous_phase
from seismic_visualizer.domain.attributes.frequency import compute_instantaneous_frequency
from seismic_visualizer.domain.attributes.rms import compute_rms
from seismic_visualizer.domain.exceptions import AttributeNotApplicableError
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.domain.slice import Slice

# Cube shape: (inline=10, crossline=20, time=30)
SHAPE = (10, 20, 200)

_ALL = [compute_envelope, compute_instantaneous_phase, compute_instantaneous_frequency, compute_rms]


def _make_sine_cube() -> np.ndarray:
    """Cube where each trace (time axis) is a pure sine wave."""
    t = np.linspace(0, 2 * np.pi, SHAPE[2], dtype=np.float32)
    cube = np.tile(np.sin(t), (*SHAPE[:2], 1))
    return cube.astype(np.float32)


def _inline_slice(idx: int = 0) -> Slice:
    return Slice(_make_sine_cube(), Axis.INLINE, idx)


def _crossline_slice(idx: int = 0) -> Slice:
    return Slice(_make_sine_cube(), Axis.CROSSLINE, idx)


def _time_slice(idx: int = 0) -> Slice:
    return Slice(_make_sine_cube(), Axis.TIME, idx)


# ---------------------------------------------------------------------------
# output shape and dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", _ALL)
def test_inline_output_shape(fn) -> None:
    s = _inline_slice()
    result = fn(s)
    assert result.shape == s.shape


@pytest.mark.parametrize("fn", _ALL)
def test_crossline_output_shape(fn) -> None:
    s = _crossline_slice()
    result = fn(s)
    assert result.shape == s.shape


@pytest.mark.parametrize("fn", _ALL)
def test_output_dtype_float32(fn) -> None:
    result = fn(_inline_slice())
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Time slice raises AttributeNotApplicableError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", _ALL)
def test_time_slice_raises(fn) -> None:
    with pytest.raises(AttributeNotApplicableError):
        fn(_time_slice())


# ---------------------------------------------------------------------------
# Envelope — |analytic signal|; always >= 0
# ---------------------------------------------------------------------------

def test_envelope_non_negative() -> None:
    result = compute_envelope(_inline_slice())
    assert np.all(result >= 0)


def test_envelope_sine_approx_amplitude() -> None:
    # Hilbert edge effects are significant in the outer ~25%; use only the central 50%
    result = compute_envelope(_inline_slice())
    quarter = result.shape[1] // 4
    interior = result[:, quarter:-quarter]
    assert np.allclose(interior, 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Phase — in [-π, π]
# ---------------------------------------------------------------------------

def test_phase_range() -> None:
    result = compute_instantaneous_phase(_inline_slice())
    assert np.all(result >= -np.pi - 1e-5)
    assert np.all(result <= np.pi + 1e-5)


# ---------------------------------------------------------------------------
# Frequency — instantaneous; reasonable range for synthetic
# ---------------------------------------------------------------------------

def test_frequency_finite() -> None:
    result = compute_instantaneous_frequency(_inline_slice())
    assert np.all(np.isfinite(result))


def test_frequency_non_negative_interior() -> None:
    # Pure sine → constant positive instantaneous frequency
    result = compute_instantaneous_frequency(_inline_slice())
    interior = result[:, 2:-2]
    assert np.all(interior >= 0)


# ---------------------------------------------------------------------------
# RMS — root mean square in sliding window; always >= 0
# ---------------------------------------------------------------------------

def test_rms_non_negative() -> None:
    result = compute_rms(_inline_slice())
    assert np.all(result >= 0)


def test_rms_sine_approx() -> None:
    # window=9 must cover many full cycles for RMS ≈ 1/√2.
    # One cycle per 200 samples → window covers 4.5% of a cycle → RMS ≠ 0.707.
    # Use 50 cycles over 200 samples (period≈4 samples): window=9 ≈ 2 full cycles.
    t = np.linspace(0, 50 * 2 * np.pi, SHAPE[2], dtype=np.float32)
    cube = np.tile(np.sin(t), (*SHAPE[:2], 1)).astype(np.float32)
    slc = Slice(cube, Axis.INLINE, 0)
    result = compute_rms(slc, window=9)
    interior = result[:, 5:-5]
    assert np.allclose(interior, 1 / np.sqrt(2), atol=0.1)


def test_rms_window_parameter_accepted() -> None:
    s = _inline_slice()
    r1 = compute_rms(s, window=5)
    r2 = compute_rms(s, window=11)
    assert r1.shape == s.shape
    assert r2.shape == s.shape
    assert not np.array_equal(r1, r2)
