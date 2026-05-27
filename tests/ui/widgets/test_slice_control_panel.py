"""Tests for SliceControlPanel and MultiAxisSliderPanel."""

from __future__ import annotations

import pytest

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.widgets.slice_control_panel import SliceControlPanel
from seismic_visualizer.ui.widgets.multi_axis_slider_panel import MultiAxisSliderPanel


# ---------------------------------------------------------------------------
# SliceControlPanel
# ---------------------------------------------------------------------------


def test_initial_label(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.INLINE, 100, 0)
    assert "Inline" in panel._label.text()
    assert "0" in panel._label.text()
    assert "100" in panel._label.text()


def test_set_range_emits_nothing(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    received: list[int] = []
    panel.index_changed.connect(received.append)
    panel.set_range(Axis.CROSSLINE, 50, 25)
    assert received == []


def test_set_value_no_emit(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.INLINE, 100, 0)
    received: list[int] = []
    panel.index_changed.connect(received.append)
    panel.set_value(42)
    assert received == []
    assert "42" in panel._label.text()


def test_slider_change_emits(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.INLINE, 100, 0)
    received: list[int] = []
    panel.index_changed.connect(received.append)
    panel._slider.setValue(10)
    assert received == [10]


def test_prev_button_decrements(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.INLINE, 100, 5)
    received: list[int] = []
    panel.index_changed.connect(received.append)
    panel._prev_btn.click()
    assert received == [4]


def test_next_button_increments(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.INLINE, 100, 5)
    received: list[int] = []
    panel.index_changed.connect(received.append)
    panel._next_btn.click()
    assert received == [6]


def test_prev_clamps_at_zero(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.INLINE, 100, 0)
    received: list[int] = []
    panel.index_changed.connect(received.append)
    panel._prev_btn.click()
    assert received == []


def test_next_clamps_at_max(qtbot: object) -> None:
    panel = SliceControlPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.INLINE, 10, 10)
    received: list[int] = []
    panel.index_changed.connect(received.append)
    panel._next_btn.click()
    assert received == []


# ---------------------------------------------------------------------------
# MultiAxisSliderPanel
# ---------------------------------------------------------------------------


def test_multi_panel_forwards_axis(qtbot: object) -> None:
    panel = MultiAxisSliderPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.CROSSLINE, 80, 0)
    received: list[tuple[Axis, int]] = []
    panel.axis_index_changed.connect(lambda a, v: received.append((a, v)))
    panel._panels[Axis.CROSSLINE]._slider.setValue(15)
    assert received == [(Axis.CROSSLINE, 15)]


def test_multi_set_value_no_emit(qtbot: object) -> None:
    panel = MultiAxisSliderPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.set_range(Axis.TIME, 200, 0)
    received: list[tuple[Axis, int]] = []
    panel.axis_index_changed.connect(lambda a, v: received.append((a, v)))
    panel.set_value(Axis.TIME, 100)
    assert received == []


def test_multi_set_range_all_axes(qtbot: object) -> None:
    panel = MultiAxisSliderPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    for axis, max_idx in ((Axis.INLINE, 50), (Axis.CROSSLINE, 80), (Axis.TIME, 200)):
        panel.set_range(axis, max_idx, 0)
    assert panel._panels[Axis.INLINE]._max_index == 50
    assert panel._panels[Axis.CROSSLINE]._max_index == 80
    assert panel._panels[Axis.TIME]._max_index == 200
