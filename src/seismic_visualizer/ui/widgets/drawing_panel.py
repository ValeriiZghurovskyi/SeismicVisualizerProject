"""DrawingPanel — compact brush/eraser controls for the right sidebar."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from seismic_visualizer.ui.i18n import LanguageManager, tr


class DrawingPanel(QWidget):
    """Compact panel with mode toggle and thickness/radius sliders.

    Emits signals on user interaction; no business logic.
    """

    mode_changed = pyqtSignal(str)
    thickness_changed = pyqtSignal(int)
    eraser_radius_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # type: ignore[arg-type]

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 8)
        root.setSpacing(6)

        root.addWidget(self._make_separator())
        root.addLayout(self._make_mode_row())
        self._thickness_row, self._thickness_slider, self._thickness_val_lbl, self._thickness_name_lbl = (
            self._make_slider_row(tr("drawing.thickness"), 1, 20, 1)
        )
        self._radius_row, self._radius_slider, self._radius_val_lbl, self._radius_name_lbl = (
            self._make_slider_row(tr("drawing.radius"), 1, 50, 5)
        )
        root.addWidget(self._thickness_row)
        root.addWidget(self._radius_row)

        self._radius_row.setVisible(False)

        self._thickness_slider.valueChanged.connect(self._on_thickness)
        self._radius_slider.valueChanged.connect(self._on_radius)

        LanguageManager.instance().language_changed.connect(lambda _: self.retranslate_ui())


    def set_thickness(self, v: int) -> None:
        self._thickness_slider.setValue(v)

    def set_eraser_radius(self, v: int) -> None:
        self._radius_slider.setValue(v)

    def retranslate_ui(self) -> None:
        self._drawing_section_lbl.setText(tr("drawing.section"))
        self._brush_btn.setText(tr("drawing.brush"))
        self._eraser_btn.setText(tr("drawing.eraser"))
        self._thickness_name_lbl.setText(f"{tr('drawing.thickness')}:")
        self._radius_name_lbl.setText(f"{tr('drawing.radius')}:")


    def _make_separator(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.HLine)  # type: ignore[attr-defined]
        frame.setFrameShadow(QFrame.Sunken)  # type: ignore[attr-defined]
        self._drawing_section_lbl = QLabel(tr("drawing.section"))
        self._drawing_section_lbl.setAlignment(Qt.AlignCenter)  # type: ignore[arg-type]
        self._drawing_section_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 2, 0, 0)
        vl.setSpacing(2)
        vl.addWidget(frame)
        vl.addWidget(self._drawing_section_lbl)
        return container

    def _make_mode_row(self) -> QHBoxLayout:
        self._brush_btn = QPushButton(tr("drawing.brush"))
        self._eraser_btn = QPushButton(tr("drawing.eraser"))
        for btn in (self._brush_btn, self._eraser_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(24)

        self._brush_btn.setChecked(True)

        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self._brush_btn)
        group.addButton(self._eraser_btn)
        group.buttonClicked.connect(self._on_mode_clicked)

        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self._brush_btn)
        row.addWidget(self._eraser_btn)
        return row

    @staticmethod
    def _make_slider_row(
        label: str, min_v: int, max_v: int, default: int
    ) -> tuple[QWidget, QSlider, QLabel, QLabel]:
        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 4, 0, 4)
        hl.setSpacing(6)

        name_lbl = QLabel(f"{label}:")
        name_lbl.setFixedWidth(62)

        slider = QSlider(Qt.Horizontal)  # type: ignore[arg-type]
        slider.setMinimum(min_v)
        slider.setMaximum(max_v)
        slider.setValue(default)
        slider.setFixedHeight(20)

        val_lbl = QLabel(str(default))
        val_lbl.setFixedWidth(26)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore[arg-type]
        slider.valueChanged.connect(lambda v, lbl=val_lbl: lbl.setText(str(v)))

        hl.addWidget(name_lbl)
        hl.addWidget(slider)
        hl.addWidget(val_lbl)
        return container, slider, val_lbl, name_lbl


    def _on_mode_clicked(self, btn: QPushButton) -> None:
        is_brush = btn is self._brush_btn
        self._thickness_row.setVisible(is_brush)
        self._radius_row.setVisible(not is_brush)
        self.mode_changed.emit("brush" if is_brush else "eraser")

    def _on_thickness(self, v: int) -> None:
        self.thickness_changed.emit(v)

    def _on_radius(self, v: int) -> None:
        self.eraser_radius_changed.emit(v)
