"""Tests for SeismicToolbar — signal emissions and state."""

from __future__ import annotations

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.widgets.toolbar import SeismicToolbar


# ---------------------------------------------------------------------------
# Axis buttons
# ---------------------------------------------------------------------------


def test_inline_button_emits_axis_inline(qtbot: object) -> None:
    toolbar = SeismicToolbar()
    received: list[Axis] = []
    toolbar.axis_changed.connect(received.append)

    toolbar._axis_btn_map[Axis.INLINE].click()

    assert received == [Axis.INLINE]


def test_crossline_button_emits_axis_crossline(qtbot: object) -> None:
    toolbar = SeismicToolbar()
    received: list[Axis] = []
    toolbar.axis_changed.connect(received.append)

    toolbar._axis_btn_map[Axis.CROSSLINE].click()

    assert received == [Axis.CROSSLINE]


def test_time_button_emits_axis_time(qtbot: object) -> None:
    toolbar = SeismicToolbar()
    received: list[Axis] = []
    toolbar.axis_changed.connect(received.append)

    toolbar._axis_btn_map[Axis.TIME].click()

    assert received == [Axis.TIME]


# ---------------------------------------------------------------------------
# View mode toggle
# ---------------------------------------------------------------------------


def test_3d_button_emits_3d(qtbot: object) -> None:
    toolbar = SeismicToolbar()
    received: list[str] = []
    toolbar.view_mode_changed.connect(received.append)

    toolbar._3d_btn.click()

    assert received == ["3d"]


def test_3d_button_toggle_emits_2d(qtbot: object) -> None:
    toolbar = SeismicToolbar()
    toolbar._3d_btn.setChecked(True)
    received: list[str] = []
    toolbar.view_mode_changed.connect(received.append)

    toolbar._3d_btn.click()

    assert received == ["2d"]


# ---------------------------------------------------------------------------
# Active entity indicator
# ---------------------------------------------------------------------------


def test_set_active_entity_updates_label(qtbot: object) -> None:
    toolbar = SeismicToolbar()
    toolbar.set_active_entity("H1: Horizon_1", (255, 0, 0))
    assert toolbar._entity_label.text() == "H1: Horizon_1"


def test_set_active_entity_none_color_resets_style(qtbot: object) -> None:
    toolbar = SeismicToolbar()
    toolbar.set_active_entity("—", None)
    assert "#bbbbbb" in toolbar._entity_color_box.styleSheet()
