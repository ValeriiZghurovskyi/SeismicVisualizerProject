"""MultiAxisSliderPanel — three SliceControlPanels stacked vertically."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.widgets.slice_control_panel import SliceControlPanel

_AXES = (Axis.INLINE, Axis.CROSSLINE, Axis.TIME)


class MultiAxisSliderPanel(QWidget):
    """One SliceControlPanel per axis; forwards changes as (Axis, int)."""

    axis_index_changed = pyqtSignal(Axis, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panels: dict[Axis, SliceControlPanel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)

        for axis in _AXES:
            panel = SliceControlPanel(self)
            panel.set_range(axis, 0, 0)
            panel.index_changed.connect(
                lambda value, a=axis: self.axis_index_changed.emit(a, value)
            )
            layout.addWidget(panel)
            self._panels[axis] = panel


    def set_range(self, axis: Axis, max_index: int, value: int) -> None:
        self._panels[axis].set_range(axis, max_index, value)

    def set_value(self, axis: Axis, value: int) -> None:
        """Update position without emitting axis_index_changed."""
        self._panels[axis].set_value(value)
