"""Smoke tests for App — composition root."""

from __future__ import annotations

from PyQt5.QtWidgets import QApplication

from seismic_visualizer.infrastructure.config import AppConfig
from seismic_visualizer.ui.app import App
from seismic_visualizer.ui.main_window import MainWindow


def _make_app() -> App:
    return App(QApplication.instance())


def test_app_instantiates(qtbot: object) -> None:
    app = _make_app()
    assert app is not None


def test_project_is_none_on_start(qtbot: object) -> None:
    app = _make_app()
    assert app._project is None


def test_presenter_is_none_before_open(qtbot: object) -> None:
    app = _make_app()
    assert app._main_presenter is None


def test_config_initialized_on_start(qtbot: object) -> None:
    app = _make_app()
    assert isinstance(app._config, AppConfig)  # type: ignore[attr-defined]
