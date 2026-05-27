"""Tests for ThumbnailSidePanel widget."""

from __future__ import annotations

import numpy as np

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.widgets.thumbnail_panel import ThumbnailSidePanel


def test_set_thumbnail_updates_label(qtbot) -> None:
    """Lines 64-66: set_thumbnail converts RGBA array to pixmap."""
    panel = ThumbnailSidePanel()
    qtbot.addWidget(panel)
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    panel.set_thumbnail(Axis.INLINE, rgba)
    assert not panel._labels[Axis.INLINE].pixmap().isNull()


def test_set_thumbnail_all_axes(qtbot) -> None:
    panel = ThumbnailSidePanel()
    qtbot.addWidget(panel)
    for axis in (Axis.INLINE, Axis.CROSSLINE, Axis.TIME):
        rgba = np.full((8, 8, 4), 128, dtype=np.uint8)
        panel.set_thumbnail(axis, rgba)
        assert not panel._labels[axis].pixmap().isNull()
