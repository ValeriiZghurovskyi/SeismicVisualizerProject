"""Tests for HUD widget — position update and clear."""

from __future__ import annotations

import numpy as np

from seismic_visualizer.ui.widgets.hud import HUD


def test_hud_update_position_shows_values(qtbot) -> None:
    """Lines 31-34: update_position sets all label texts."""
    hud = HUD()
    qtbot.addWidget(hud)
    hud.update_position(10, 20, 30, 128)
    assert "10" in hud._inline.text()
    assert "20" in hud._xline.text()
    assert "30" in hud._time.text()
    assert "128" in hud._amp.text()


def test_hud_update_position_none_amp_shows_placeholder(qtbot) -> None:
    """Line 34: amp=None → shows placeholder '—'."""
    hud = HUD()
    qtbot.addWidget(hud)
    hud.update_position(0, 0, 0, None)
    assert "—" in hud._amp.text()


def test_hud_clear_resets_to_placeholder(qtbot) -> None:
    hud = HUD()
    qtbot.addWidget(hud)
    hud.update_position(5, 6, 7, 100)
    hud.clear()
    assert "—" in hud._inline.text()
    assert "—" in hud._amp.text()
