"""SliceControlPanel — single-axis slider with ◀/▶ navigation."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.i18n import LanguageManager, tr

_AXIS_TR_KEY: dict[Axis, str] = {
    Axis.INLINE: "toolbar.inline",
    Axis.CROSSLINE: "toolbar.crossline",
    Axis.TIME: "toolbar.time",
}


class SliceControlPanel(QWidget):
    """Horizontal slider with prev/next buttons and a text label."""

    index_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._axis = Axis.INLINE
        self._max_index = 0
        self._current_value = 0

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setMaximumWidth(28)
        self._prev_btn.setStyleSheet("padding: 2px 4px; font-size: 13px;")
        self._prev_btn.clicked.connect(self._on_prev)

        self._slider = QSlider(Qt.Horizontal)  # type: ignore[arg-type]
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)

        self._next_btn = QPushButton("▶")
        self._next_btn.setMaximumWidth(28)
        self._next_btn.setStyleSheet("padding: 2px 4px; font-size: 13px;")
        self._next_btn.clicked.connect(self._on_next)

        self._label = QLabel(self._format_label(0))
        self._label.setMinimumWidth(120)
        self._label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)  # type: ignore[arg-type]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.addWidget(self._prev_btn)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._next_btn)
        layout.addWidget(self._label)

        LanguageManager.instance().language_changed.connect(
            lambda _: self._label.setText(self._format_label(self._current_value))
        )


    def set_range(self, axis: Axis, max_index: int, value: int) -> None:
        self._axis = axis
        self._max_index = max_index
        self._current_value = value
        self._slider.blockSignals(True)
        self._slider.setMaximum(max_index)
        self._slider.setValue(value)
        self._slider.blockSignals(False)
        self._label.setText(self._format_label(value))

    def set_value(self, value: int) -> None:
        """Update position without emitting index_changed."""
        self._current_value = value
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)
        self._label.setText(self._format_label(value))


    def _on_prev(self) -> None:
        v = self._slider.value()
        if v <= 0:
            return
        self._slider.setValue(v - 1)

    def _on_next(self) -> None:
        v = self._slider.value()
        if v >= self._max_index:
            return
        self._slider.setValue(v + 1)

    def _on_slider_changed(self, value: int) -> None:
        self._current_value = value
        self._label.setText(self._format_label(value))
        self.index_changed.emit(value)


    def _format_label(self, value: int) -> str:
        axis_name = tr(_AXIS_TR_KEY.get(self._axis, "toolbar.inline"))
        return f"{axis_name}  {value} / {self._max_index}"
