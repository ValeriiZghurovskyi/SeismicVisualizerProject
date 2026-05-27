"""View2D — passive 2D slice viewer: canvas + sidebar only."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QMessageBox, QShortcut, QSplitter, QVBoxLayout, QWidget

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.ui.widgets.drawing_panel import DrawingPanel
from seismic_visualizer.ui.widgets.entity_sidebar import EntitySidebar
from seismic_visualizer.ui.widgets.slice_canvas import SliceCanvas


class View2D(QWidget):
    """Canvas + sidebar. Toolbar lives in MainWindow.

    Signals forwarded to MainWindow/App; no business logic here.
    """

    status_changed = pyqtSignal(str)

    stroke_point = pyqtSignal(int, int)
    stroke_finished = pyqtSignal()
    erase_requested = pyqtSignal(int, int)
    stroke_point_removed = pyqtSignal()
    canvas_hovered = pyqtSignal(int, int)

    new_entity_requested = pyqtSignal(EntityKind)
    entity_visibility_changed = pyqtSignal(int, object, bool)
    entity_selected = pyqtSignal(int, object)
    entity_delete_requested = pyqtSignal(int, object)
    entity_rename_requested = pyqtSignal(int, object, str)

    drawing_mode_changed = pyqtSignal(str)
    thickness_changed = pyqtSignal(int)
    eraser_radius_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._canvas = SliceCanvas()
        self._sidebar = EntitySidebar()
        self._drawing_panel = DrawingPanel()

        self._build_layout()
        self._connect_signals()


    def update_canvas(self, rgba: np.ndarray) -> None:
        self._canvas.set_image(rgba)

    def update_entity_list(self, entities: list[EntityInfo]) -> None:
        self._sidebar.update_entities(entities)

    def update_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def clear_stroke_preview(self) -> None:
        self._canvas.clear_preview()

    def remove_last_stroke_point(self) -> None:
        self._canvas.remove_last_preview_point()

    def set_stroke_color(self, color: tuple[int, int, int] | None) -> None:
        self._canvas.set_stroke_color(color)

    def confirm_entity_delete(self, name: str) -> bool:
        result = QMessageBox.question(
            self,
            "Delete Entity",
            f'Entity "{name}" contains labels that will be permanently deleted. Continue?',
            QMessageBox.Yes | QMessageBox.No,  # type: ignore[attr-defined]
            QMessageBox.No,  # type: ignore[attr-defined]
        )
        return result == QMessageBox.Yes  # type: ignore[attr-defined]


    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._sidebar, stretch=1)
        right_layout.addWidget(self._drawing_panel)

        splitter = QSplitter(Qt.Horizontal)  # type: ignore[arg-type]
        splitter.addWidget(self._canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self._canvas.stroke_point.connect(self.stroke_point)
        self._canvas.stroke_finished.connect(self.stroke_finished)
        self._canvas.erase_requested.connect(self.erase_requested)
        self._canvas.mouse_hovered.connect(self.canvas_hovered)

        commit_shortcut = QShortcut(QKeySequence("S"), self)
        commit_shortcut.activated.connect(self.stroke_finished)

        undo_point_shortcut = QShortcut(QKeySequence("0"), self)
        undo_point_shortcut.activated.connect(self.stroke_point_removed)

        self._sidebar.new_entity_requested.connect(self.new_entity_requested)
        self._sidebar.entity_visibility_changed.connect(self.entity_visibility_changed)
        self._sidebar.entity_selected.connect(self.entity_selected)
        self._sidebar.entity_delete_requested.connect(self.entity_delete_requested)
        self._sidebar.entity_rename_requested.connect(self.entity_rename_requested)

        self._drawing_panel.mode_changed.connect(self.drawing_mode_changed)
        self._drawing_panel.thickness_changed.connect(self.thickness_changed)
        self._drawing_panel.eraser_radius_changed.connect(self.eraser_radius_changed)
