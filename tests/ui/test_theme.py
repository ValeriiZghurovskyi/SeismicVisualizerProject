"""Tests for ui.theme — palette construction and title-bar helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget

from seismic_visualizer.ui.theme import (
    _make_dark_palette,
    _set_titlebar_dark,
    apply_dark_theme,
    apply_light_theme,
    apply_theme,
    set_titlebar_dark,
)


# ---------------------------------------------------------------------------
# _make_dark_palette (lines 15-33)
# ---------------------------------------------------------------------------

def test_make_dark_palette_returns_qpalette(qtbot) -> None:
    palette = _make_dark_palette()
    assert isinstance(palette, QPalette)


def test_make_dark_palette_window_color_is_dark(qtbot) -> None:
    palette = _make_dark_palette()
    color = palette.color(QPalette.Window)
    # #2b2b2b ≈ R=43 G=43 B=43
    assert color.red() < 80
    assert color.green() < 80
    assert color.blue() < 80


# ---------------------------------------------------------------------------
# _set_titlebar_dark (lines 37-46)
# ---------------------------------------------------------------------------

def test_set_titlebar_dark_non_win32_returns_without_ctypes() -> None:
    """Line 38: non-win32 → immediate return, no ctypes call."""
    with patch("seismic_visualizer.ui.theme.sys.platform", "linux"):
        with patch("seismic_visualizer.ui.theme.ctypes") as mock_ctypes:
            _set_titlebar_dark(12345, True)
            mock_ctypes.windll.dwmapi.DwmSetWindowAttribute.assert_not_called()


def test_set_titlebar_dark_win32_handles_oserror() -> None:
    """Lines 39-46: OSError from DWM is silently swallowed."""
    with patch("seismic_visualizer.ui.theme.sys.platform", "win32"):
        with patch("seismic_visualizer.ui.theme.ctypes") as mock_ctypes:
            mock_ctypes.c_int.return_value = MagicMock()
            mock_ctypes.windll.dwmapi.DwmSetWindowAttribute.side_effect = OSError("DWM error")
            _set_titlebar_dark(0, True)  # Should not raise


def test_set_titlebar_dark_win32_success() -> None:
    """Lines 39-44: successful DWM call on win32."""
    with patch("seismic_visualizer.ui.theme.sys.platform", "win32"):
        with patch("seismic_visualizer.ui.theme.ctypes") as mock_ctypes:
            mock_ctypes.c_int.return_value = MagicMock()
            _set_titlebar_dark(0, False)
            mock_ctypes.windll.dwmapi.DwmSetWindowAttribute.assert_called_once()


def test_set_titlebar_dark_win32_attribute_error_swallowed() -> None:
    """Line 45-46: AttributeError (missing DWM API) is also swallowed."""
    with patch("seismic_visualizer.ui.theme.sys.platform", "win32"):
        with patch("seismic_visualizer.ui.theme.ctypes") as mock_ctypes:
            mock_ctypes.c_int.return_value = MagicMock()
            mock_ctypes.windll.dwmapi.DwmSetWindowAttribute.side_effect = AttributeError
            _set_titlebar_dark(0, True)


# ---------------------------------------------------------------------------
# set_titlebar_dark (lines 51-52)
# ---------------------------------------------------------------------------

def test_set_titlebar_dark_invisible_widget_noop(qtbot) -> None:
    """Widget not visible → _set_titlebar_dark is never called."""
    w = QWidget()
    qtbot.addWidget(w)
    with patch("seismic_visualizer.ui.theme._set_titlebar_dark") as mock_fn:
        set_titlebar_dark(w, True)
        mock_fn.assert_not_called()


def test_set_titlebar_dark_visible_widget_calls_helper(qtbot) -> None:
    """Lines 51-52: visible widget → calls _set_titlebar_dark with winId."""
    w = QWidget()
    qtbot.addWidget(w)
    w.show()
    with patch("seismic_visualizer.ui.theme._set_titlebar_dark") as mock_fn:
        set_titlebar_dark(w, True)
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# apply_light_theme / apply_dark_theme (lines 56-68)
# ---------------------------------------------------------------------------

def test_apply_light_theme_no_visible_widgets(qapp) -> None:
    """Lines 56-57: runs without errors even with no visible widgets."""
    apply_light_theme(qapp)


def test_apply_dark_theme_no_visible_widgets(qapp) -> None:
    """Lines 64-65: applies dark palette even with no visible widgets."""
    apply_dark_theme(qapp)


def test_apply_light_theme_with_visible_widget(qtbot, qapp) -> None:
    """Lines 58-60: iterates topLevelWidgets and calls _set_titlebar_dark."""
    w = QWidget()
    qtbot.addWidget(w)
    w.show()
    with patch("seismic_visualizer.ui.theme._set_titlebar_dark") as mock_fn:
        apply_light_theme(qapp)
        assert mock_fn.call_count >= 1


def test_apply_dark_theme_with_visible_widget(qtbot, qapp) -> None:
    """Lines 66-68: iterates topLevelWidgets for dark theme."""
    w = QWidget()
    qtbot.addWidget(w)
    w.show()
    with patch("seismic_visualizer.ui.theme._set_titlebar_dark") as mock_fn:
        apply_dark_theme(qapp)
        assert mock_fn.call_count >= 1


# ---------------------------------------------------------------------------
# apply_theme (lines 72-75)
# ---------------------------------------------------------------------------

def test_apply_theme_dark_delegates(qapp) -> None:
    with patch("seismic_visualizer.ui.theme.apply_dark_theme") as mock_fn:
        apply_theme(qapp, "dark")
        mock_fn.assert_called_once_with(qapp)


def test_apply_theme_light_delegates(qapp) -> None:
    with patch("seismic_visualizer.ui.theme.apply_light_theme") as mock_fn:
        apply_theme(qapp, "light")
        mock_fn.assert_called_once_with(qapp)


def test_apply_theme_unknown_falls_back_to_light(qapp) -> None:
    with patch("seismic_visualizer.ui.theme.apply_light_theme") as mock_fn:
        apply_theme(qapp, "neon")
        mock_fn.assert_called_once_with(qapp)
