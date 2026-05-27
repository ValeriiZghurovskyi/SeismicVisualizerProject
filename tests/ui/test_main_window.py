"""Smoke tests for MainWindow — layout and signal routing."""

from __future__ import annotations

from PyQt5.QtWidgets import QMenuBar, QStackedWidget, QStatusBar

from seismic_visualizer.ui.main_window import MainWindow
from seismic_visualizer.ui.views.view_2d import View2D
from seismic_visualizer.ui.views.view_3d import View3D


def test_main_window_instantiates(qtbot: object) -> None:
    win = MainWindow()
    assert win is not None


def test_main_window_has_status_bar(qtbot: object) -> None:
    win = MainWindow()
    assert isinstance(win.statusBar(), QStatusBar)


def test_main_window_has_menu_bar(qtbot: object) -> None:
    win = MainWindow()
    assert isinstance(win.menuBar(), QMenuBar)


def test_view_returns_view_2d_instance(qtbot: object) -> None:
    win = MainWindow()
    assert isinstance(win.view(), View2D)


def test_file_menu_has_open_save(qtbot: object) -> None:
    win = MainWindow()
    file_menu = win.menuBar().actions()[0].menu()
    assert file_menu is not None
    action_texts = [a.text() for a in file_menu.actions() if not a.isSeparator()]
    assert any(t.startswith("Open") for t in action_texts)
    assert any("Save" in t for t in action_texts)


def test_status_changed_updates_status_bar(qtbot: object) -> None:
    win = MainWindow()
    win.view().update_status("Slice 10/499 | Inline")
    assert win.statusBar().currentMessage() == "Slice 10/499 | Inline"


def test_open_svp_requested_emitted_from_menu(qtbot: object) -> None:
    win = MainWindow()
    received: list[bool] = []
    win.open_svp_requested.connect(lambda: received.append(True))

    file_menu = win.menuBar().actions()[0].menu()
    assert file_menu is not None
    open_act = next(a for a in file_menu.actions() if "svp" in a.text().lower())
    open_act.triggered.emit(False)

    assert received == [True]


def test_save_svp_requested_emitted_from_menu(qtbot: object) -> None:
    win = MainWindow()
    received: list[bool] = []
    win.save_svp_requested.connect(lambda: received.append(True))

    file_menu = win.menuBar().actions()[0].menu()
    assert file_menu is not None
    save_act = next(a for a in file_menu.actions() if "Save Project" in a.text())
    save_act.triggered.emit(False)

    assert received == [True]


# ------------------------------------------------------------------
# QStackedWidget / 2D-3D switching
# ------------------------------------------------------------------


def test_central_widget_has_view_stack(qtbot: object) -> None:
    win = MainWindow()
    assert isinstance(win._stack, QStackedWidget)


def test_switch_to_2d_sets_stack_index_0(qtbot: object) -> None:
    win = MainWindow()
    win.switch_to_3d()
    win.switch_to_2d()
    assert win._stack.currentIndex() == 0


def test_switch_to_3d_sets_stack_index_1(qtbot: object) -> None:
    win = MainWindow()
    win.switch_to_3d()
    assert win._stack.currentIndex() == 1


def test_view_3d_returns_view3d_instance(qtbot: object) -> None:
    win = MainWindow()
    assert isinstance(win.view_3d(), View3D)


def test_view_mode_changed_emitted_by_toolbar_toggle(qtbot: object) -> None:
    win = MainWindow()
    received: list[str] = []
    win.view_mode_changed.connect(received.append)

    toolbar = win._toolbar  # type: ignore[attr-defined]
    toolbar._3d_btn.setChecked(True)  # type: ignore[attr-defined]
    toolbar._3d_btn.clicked.emit(True)  # type: ignore[attr-defined]
    assert received == ["3d"]

    toolbar._3d_btn.setChecked(False)  # type: ignore[attr-defined]
    toolbar._3d_btn.clicked.emit(False)  # type: ignore[attr-defined]
    assert received == ["3d", "2d"]
