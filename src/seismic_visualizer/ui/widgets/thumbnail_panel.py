"""ThumbnailSidePanel — static sidebar panel with per-axis 2D slice thumbnails."""

from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.i18n import LanguageManager, tr

_AXIS_TR_KEY: dict[Axis, str] = {
    Axis.INLINE: "toolbar.inline",
    Axis.CROSSLINE: "toolbar.crossline",
    Axis.TIME: "toolbar.time",
}


class ThumbnailSidePanel(QWidget):
    """Vertical panel with three resizable slice thumbnails.

    Intended as a QSplitter child between the 3D plotter and the right sidebar.
    Each axis has an individual checkbox to show/hide that thumbnail.
    The panel itself is toggled by the 'Thumbnails' checkbox in the planes panel.
    Height is distributed equally among visible thumbnails via stretch factors.
    Width is controlled by the parent QSplitter divider (horizontal only).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        self._title = QLabel(tr("3d.slices"))
        self._title.setStyleSheet("font-weight: bold; color: #444; font-size: 11px;")
        layout.addWidget(self._title)

        self._labels: dict[Axis, QLabel] = {}
        self._checks: dict[Axis, QCheckBox] = {}

        for axis in (Axis.INLINE, Axis.CROSSLINE, Axis.TIME):
            cb = QCheckBox(tr(_AXIS_TR_KEY[axis]))
            cb.setChecked(True)
            layout.addWidget(cb)
            self._checks[axis] = cb

            lbl = QLabel()
            lbl.setScaledContents(True)
            lbl.setMinimumHeight(30)
            lbl.setStyleSheet("background: #f5f5f5; border: 1px solid #cccccc;")
            layout.addWidget(lbl, stretch=1)
            self._labels[axis] = lbl

            cb.toggled.connect(lambda checked, ax=axis: self._labels[ax].setVisible(checked))

        LanguageManager.instance().language_changed.connect(lambda _: self.retranslate_ui())


    def set_thumbnail(self, axis: Axis, rgba: np.ndarray) -> None:
        """Update one axis thumbnail from a uint8 RGBA (H, W, 4) array."""
        h, w = rgba.shape[:2]
        img = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
        self._labels[axis].setPixmap(QPixmap.fromImage(img))

    def retranslate_ui(self) -> None:
        self._title.setText(tr("3d.slices"))
        for axis, key in _AXIS_TR_KEY.items():
            cb = self._checks.get(axis)
            if cb is not None:
                cb.setText(tr(key))
