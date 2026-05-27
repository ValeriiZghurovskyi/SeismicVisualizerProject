"""Smoke tests for DrawingPanel."""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from seismic_visualizer.ui.widgets.drawing_panel import DrawingPanel


@pytest.fixture()
def panel(qtbot: QtBot) -> DrawingPanel:
    w = DrawingPanel()
    qtbot.addWidget(w)
    w.show()
    return w


def test_initial_state(panel: DrawingPanel) -> None:
    assert panel._brush_btn.isChecked()
    assert not panel._eraser_btn.isChecked()
    assert panel._thickness_row.isVisible()
    assert not panel._radius_row.isVisible()


def test_switch_to_eraser_emits_signal(panel: DrawingPanel, qtbot: QtBot) -> None:
    with qtbot.waitSignal(panel.mode_changed) as blocker:
        panel._eraser_btn.click()
    assert blocker.args == ["eraser"]
    assert not panel._thickness_row.isVisible()
    assert panel._radius_row.isVisible()


def test_switch_back_to_brush(panel: DrawingPanel, qtbot: QtBot) -> None:
    panel._eraser_btn.click()
    with qtbot.waitSignal(panel.mode_changed) as blocker:
        panel._brush_btn.click()
    assert blocker.args == ["brush"]
    assert panel._thickness_row.isVisible()
    assert not panel._radius_row.isVisible()


def test_thickness_signal(panel: DrawingPanel, qtbot: QtBot) -> None:
    with qtbot.waitSignal(panel.thickness_changed) as blocker:
        panel._thickness_slider.setValue(7)
    assert blocker.args == [7]


def test_eraser_radius_signal(panel: DrawingPanel, qtbot: QtBot) -> None:
    panel._eraser_btn.click()
    with qtbot.waitSignal(panel.eraser_radius_changed) as blocker:
        panel._radius_slider.setValue(15)
    assert blocker.args == [15]


def test_set_thickness_public(panel: DrawingPanel) -> None:
    panel.set_thickness(10)
    assert panel._thickness_slider.value() == 10


def test_set_eraser_radius_public(panel: DrawingPanel) -> None:
    panel.set_eraser_radius(20)
    assert panel._radius_slider.value() == 20
