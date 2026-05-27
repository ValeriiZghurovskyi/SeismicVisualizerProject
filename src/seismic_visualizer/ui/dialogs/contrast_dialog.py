"""ContrastDialog — non-modal dialog with vmin/vmax sliders."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from seismic_visualizer.ui.i18n import tr


class ContrastDialog(QDialog):
    contrast_changed = pyqtSignal(int, int)

    def __init__(
        self,
        parent: QWidget | None = None,
        vmin: int = 0,
        vmax: int = 255,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("contrast.title"))
        self.setFixedWidth(360)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)  # type: ignore[arg-type]

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        self._min_slider = self._add_row(layout, tr("contrast.min"), 0, 254, vmin)
        self._max_slider = self._add_row(layout, tr("contrast.max"), 1, 255, vmax)

        self._min_slider.valueChanged.connect(self._on_change)
        self._max_slider.valueChanged.connect(self._on_change)

    def _add_row(
        self, parent: QVBoxLayout, title: str, min_v: int, max_v: int, value: int
    ) -> QSlider:
        parent.addWidget(QLabel(title))

        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)  # type: ignore[arg-type]
        slider.setMinimum(min_v)
        slider.setMaximum(max_v)
        slider.setValue(value)

        val_lbl = QLabel(str(value))
        val_lbl.setFixedWidth(32)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore[arg-type]

        slider.valueChanged.connect(lambda v, lbl=val_lbl: lbl.setText(str(v)))

        row.addWidget(slider)
        row.addWidget(val_lbl)
        parent.addLayout(row)
        return slider

    def _on_change(self) -> None:
        vmin = self._min_slider.value()
        vmax = self._max_slider.value()
        if vmin < vmax:
            self.contrast_changed.emit(vmin, vmax)
