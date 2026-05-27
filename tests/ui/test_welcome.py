"""Smoke tests for WelcomeScreen."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtWidgets import QPushButton

from seismic_visualizer.ui.welcome import WelcomeScreen


@pytest.fixture
def screen(qtbot):  # type: ignore[no-untyped-def]
    w = WelcomeScreen()
    qtbot.addWidget(w)
    return w


def test_recent_section_hidden_by_default(screen: WelcomeScreen) -> None:
    assert screen._recent_container is not None
    assert not screen._recent_container.isVisible()


def test_set_recent_files_shows_buttons(screen: WelcomeScreen, tmp_path: Path) -> None:
    paths = [tmp_path / f"{i}.npz" for i in range(3)]
    screen.set_recent_files(paths)
    assert screen._recent_container is not None
    # isVisibleTo checks widget's own visibility flag, not the full parent chain
    assert screen._recent_container.isVisibleTo(screen)
    buttons = screen._recent_container.findChildren(QPushButton)
    assert len(buttons) == 3
    assert buttons[0].text() == paths[0].name


def test_set_recent_files_empty_hides_section(screen: WelcomeScreen, tmp_path: Path) -> None:
    screen.set_recent_files([tmp_path / "a.npz"])
    screen.set_recent_files([])
    assert screen._recent_container is not None
    assert not screen._recent_container.isVisible()


def test_set_recent_files_replaces_buttons(screen: WelcomeScreen, tmp_path: Path) -> None:
    screen.set_recent_files([tmp_path / "a.npz", tmp_path / "b.npz"])
    screen.set_recent_files([tmp_path / "c.npz"])
    buttons = screen._recent_container.findChildren(QPushButton)  # type: ignore[union-attr]
    assert len(buttons) == 1
    assert buttons[0].text() == "c.npz"


def test_button_click_emits_signal(screen: WelcomeScreen, tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "seismic.npz"
    screen.set_recent_files([path])
    received: list[Path] = []
    screen.open_recent_requested.connect(received.append)
    buttons = screen._recent_container.findChildren(QPushButton)  # type: ignore[union-attr]
    buttons[0].click()
    assert received == [path]


def test_set_recent_files_caps_at_five(screen: WelcomeScreen, tmp_path: Path) -> None:
    paths = [tmp_path / f"{i}.npz" for i in range(7)]
    screen.set_recent_files(paths)
    buttons = screen._recent_container.findChildren(QPushButton)  # type: ignore[union-attr]
    assert len(buttons) == 5


def test_button_tooltip_is_full_path(screen: WelcomeScreen, tmp_path: Path) -> None:
    path = tmp_path / "seismic.npz"
    screen.set_recent_files([path])
    buttons = screen._recent_container.findChildren(QPushButton)  # type: ignore[union-attr]
    assert buttons[0].toolTip() == str(path)
