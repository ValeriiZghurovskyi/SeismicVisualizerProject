"""Tests for SliceCanvas — coordinate math and state, no rendering."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.ui.widgets.slice_canvas import SliceCanvas


def _rgba(h: int = 10, w: int = 20) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _canvas_with_image(h: int = 10, w: int = 20) -> SliceCanvas:
    """Create a SliceCanvas with a loaded image and 1:1 zoom (no fit-to-window distortion)."""
    canvas = SliceCanvas()
    canvas.set_image(_rgba(h, w))
    # Override fit_to_window result: force 1:1 zoom with no pan
    canvas._zoom_x = 1.0
    canvas._zoom_y = 1.0
    canvas._pan_x = 0.0
    canvas._pan_y = 0.0
    return canvas


# ---------------------------------------------------------------------------
# set_image
# ---------------------------------------------------------------------------

def test_set_image_does_not_raise(qtbot: object) -> None:
    canvas = SliceCanvas()
    canvas.set_image(_rgba(10, 20))  # must not raise


def test_set_image_loads_pixmap(qtbot: object) -> None:
    canvas = SliceCanvas()
    assert canvas._pixmap is None
    canvas.set_image(_rgba(10, 20))
    assert canvas._pixmap is not None
    assert canvas._pixmap.width() == 20
    assert canvas._pixmap.height() == 10


# ---------------------------------------------------------------------------
# pixel_to_data — no image
# ---------------------------------------------------------------------------

def test_pixel_to_data_without_image_returns_none(qtbot: object) -> None:
    canvas = SliceCanvas()
    assert canvas.pixel_to_data(0, 0) is None


# ---------------------------------------------------------------------------
# pixel_to_data — zoom=1, pan=0
# ---------------------------------------------------------------------------

def test_pixel_to_data_origin_at_zoom1(qtbot: object) -> None:
    canvas = _canvas_with_image()
    assert canvas.pixel_to_data(0, 0) == (0, 0)


def test_pixel_to_data_mid_pixel(qtbot: object) -> None:
    canvas = _canvas_with_image()
    # px=5, py=3 → col=5, row=3
    assert canvas.pixel_to_data(5, 3) == (3, 5)


def test_pixel_to_data_last_valid_pixel(qtbot: object) -> None:
    canvas = _canvas_with_image()
    # width=20, height=10 → last valid col=19, row=9
    assert canvas.pixel_to_data(19, 9) == (9, 19)


# ---------------------------------------------------------------------------
# pixel_to_data — out of bounds
# ---------------------------------------------------------------------------

def test_pixel_to_data_negative_x_returns_none(qtbot: object) -> None:
    canvas = SliceCanvas()
    canvas.set_image(_rgba(10, 20))
    assert canvas.pixel_to_data(-1, 0) is None


def test_pixel_to_data_negative_y_returns_none(qtbot: object) -> None:
    canvas = SliceCanvas()
    canvas.set_image(_rgba(10, 20))
    assert canvas.pixel_to_data(0, -1) is None


def test_pixel_to_data_beyond_width_returns_none(qtbot: object) -> None:
    canvas = _canvas_with_image()
    assert canvas.pixel_to_data(20, 0) is None


def test_pixel_to_data_beyond_height_returns_none(qtbot: object) -> None:
    canvas = _canvas_with_image()
    assert canvas.pixel_to_data(0, 10) is None


# ---------------------------------------------------------------------------
# pixel_to_data — zoom changes mapping
# ---------------------------------------------------------------------------

def test_pixel_to_data_zoom2_halves_coords(qtbot: object) -> None:
    canvas = _canvas_with_image()
    canvas._zoom_x = 2.0
    canvas._zoom_y = 2.0
    # screen (10, 6) → data col=5, row=3
    assert canvas.pixel_to_data(10, 6) == (3, 5)


def test_pixel_to_data_zoom_half_doubles_coords(qtbot: object) -> None:
    canvas = _canvas_with_image()
    canvas._zoom_x = 0.5
    canvas._zoom_y = 0.5
    # screen (4, 2) → data col=8, row=4
    assert canvas.pixel_to_data(4, 2) == (4, 8)


# ---------------------------------------------------------------------------
# paintEvent with no image must not raise
# ---------------------------------------------------------------------------

def test_paint_event_without_image_does_not_raise(qtbot: object) -> None:
    canvas = SliceCanvas()
    canvas.resize(100, 100)
    canvas.repaint()  # triggers paintEvent synchronously
