"""SeismicToolbar — axis selector, active entity indicator, 2D/3D toggle."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QLabel,
    QPushButton,
    QToolBar,
    QWidget,
)

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.i18n import LanguageManager, tr

_AXIS_ENTRIES: list[tuple[str, Axis]] = [
    ("toolbar.inline", Axis.INLINE),
    ("toolbar.crossline", Axis.CROSSLINE),
    ("toolbar.time", Axis.TIME),
]


class SeismicToolbar(QToolBar):
    """Axis toggle buttons, active entity indicator, and 2D/3D view toggle."""

    axis_changed = pyqtSignal(Axis)
    view_mode_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMovable(False)

        self._axis_btn_map: dict[Axis, QPushButton] = {}
        self._axis_btn_map = self._build_axis_buttons()
        self.addSeparator()
        self._build_entity_indicator()
        self.addSeparator()
        self._3d_btn = self._build_view_toggle()

        LanguageManager.instance().language_changed.connect(lambda _: self.retranslate_ui())


    def set_active_entity(self, name: str, color: tuple[int, int, int] | None) -> None:
        self._entity_label.setText(name)
        if color is None:
            self._entity_color_box.setStyleSheet("background: #bbbbbb; border: 1px solid #aaa;")
        else:
            hex_color = QColor(*color).name()
            self._entity_color_box.setStyleSheet(
                f"background: {hex_color}; border: 1px solid #aaa;"
            )

    def retranslate_ui(self) -> None:
        for key, axis in _AXIS_ENTRIES:
            btn = self._axis_btn_map.get(axis)
            if btn is not None:
                btn.setText(tr(key))


    def _build_axis_buttons(self) -> dict[Axis, QPushButton]:
        group = QButtonGroup(self)
        group.setExclusive(True)
        btn_map: dict[Axis, QPushButton] = {}
        first = True
        for key, axis in _AXIS_ENTRIES:
            btn = QPushButton(tr(key))
            btn.setCheckable(True)
            group.addButton(btn)
            self.addWidget(btn)
            btn.clicked.connect(lambda checked, a=axis: self.axis_changed.emit(a))
            btn_map[axis] = btn
            if first:
                btn.setChecked(True)
                first = False
        return btn_map

    def _build_entity_indicator(self) -> None:
        self._entity_color_box = QFrame()
        self._entity_color_box.setFixedSize(14, 14)
        self._entity_color_box.setStyleSheet("background: #bbbbbb; border: 1px solid #aaa;")
        self.addWidget(self._entity_color_box)

        self._entity_label = QLabel("—")
        self._entity_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)  # type: ignore[arg-type]
        self._entity_label.setFixedWidth(100)
        self._entity_label.setStyleSheet("padding-left: 4px;")
        self.addWidget(self._entity_label)

    def _build_view_toggle(self) -> QPushButton:
        btn = QPushButton("3D")
        btn.setCheckable(True)
        btn.clicked.connect(
            lambda checked: self.view_mode_changed.emit("3d" if checked else "2d")
        )
        self.addWidget(btn)
        return btn
