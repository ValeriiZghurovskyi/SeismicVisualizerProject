"""Theme management — QPalette-based light/dark themes for Fusion style."""

from __future__ import annotations

import ctypes
import sys

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication, QWidget

_ACCENT = QColor("#0078D4")


def _make_dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#2b2b2b"))
    p.setColor(QPalette.WindowText, QColor("#e8e8e8"))
    p.setColor(QPalette.Base, QColor("#1e1e1e"))
    p.setColor(QPalette.AlternateBase, QColor("#2b2b2b"))
    p.setColor(QPalette.ToolTipBase, QColor("#2b2b2b"))
    p.setColor(QPalette.ToolTipText, QColor("#e8e8e8"))
    p.setColor(QPalette.Text, QColor("#e8e8e8"))
    p.setColor(QPalette.Button, QColor("#3c3c3c"))
    p.setColor(QPalette.ButtonText, QColor("#e8e8e8"))
    p.setColor(QPalette.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.Link, _ACCENT)
    p.setColor(QPalette.Highlight, _ACCENT)
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#666666"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#666666"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#666666"))
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor("#555555"))
    return p


def _set_titlebar_dark(hwnd: int, dark: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        value = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
        )
    except (OSError, AttributeError):
        pass


def set_titlebar_dark(widget: QWidget, dark: bool) -> None:
    """Apply dark/light native title bar to a specific top-level window."""
    if widget.isVisible():
        _set_titlebar_dark(int(widget.winId()), dark)


def apply_light_theme(app: QApplication) -> None:
    app.setStyleSheet("")
    app.setPalette(app.style().standardPalette())
    for w in app.topLevelWidgets():
        if w.isVisible():
            _set_titlebar_dark(int(w.winId()), False)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyleSheet("")
    app.setPalette(_make_dark_palette())
    for w in app.topLevelWidgets():
        if w.isVisible():
            _set_titlebar_dark(int(w.winId()), True)


def apply_theme(app: QApplication, theme: str) -> None:
    if theme == "dark":
        apply_dark_theme(app)
    else:
        apply_light_theme(app)
