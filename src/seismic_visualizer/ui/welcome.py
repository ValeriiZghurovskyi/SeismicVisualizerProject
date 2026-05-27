"""WelcomeScreen — launcher shown at startup before any project is open."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from seismic_visualizer.ui.i18n import tr


class WelcomeScreen(QWidget):
    open_seismic_npz_requested = pyqtSignal()
    open_seismic_segy_requested = pyqtSignal()
    open_svp_requested = pyqtSignal()
    open_recent_requested = pyqtSignal(Path)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Seismic Visualizer")
        self.setMinimumWidth(360)
        self._recent_container: QWidget | None = None
        self._recent_layout: QVBoxLayout | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)  # type: ignore[arg-type]
        layout.setSpacing(0)
        layout.setContentsMargins(40, 32, 40, 32)

        title = QLabel("Seismic Visualizer")
        title.setAlignment(Qt.AlignCenter)  # type: ignore[arg-type]
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(tr("welcome.subtitle"))
        subtitle.setAlignment(Qt.AlignCenter)  # type: ignore[arg-type]
        subtitle.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)  # type: ignore[attr-defined]
        sep.setStyleSheet("color: #888; margin-top: 16px; margin-bottom: 16px;")
        layout.addWidget(sep)

        btn_seismic = QPushButton(tr("welcome.open_npz"))
        btn_seismic.setFixedHeight(36)
        btn_seismic.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        btn_seismic.setToolTip("Open a single seismic cube (.npz)")
        btn_seismic.clicked.connect(self.open_seismic_npz_requested)
        layout.addWidget(btn_seismic)

        layout.addSpacing(6)

        btn_segy = QPushButton(tr("welcome.open_segy"))
        btn_segy.setFixedHeight(36)
        btn_segy.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        btn_segy.setToolTip("Open a SEG-Y seismic file — will offer to convert to .npz")
        btn_segy.clicked.connect(self.open_seismic_segy_requested)
        layout.addWidget(btn_segy)

        layout.addSpacing(6)

        btn_svp = QPushButton(tr("welcome.open_svp"))
        btn_svp.setFixedHeight(36)
        btn_svp.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        btn_svp.setToolTip("Open a complete project saved as a single .svp file")
        btn_svp.clicked.connect(self.open_svp_requested)
        layout.addWidget(btn_svp)

        self._recent_container = QWidget()
        self._recent_layout = QVBoxLayout(self._recent_container)
        self._recent_layout.setSpacing(4)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)

        recent_sep = QFrame()
        recent_sep.setFrameShape(QFrame.HLine)  # type: ignore[attr-defined]
        recent_sep.setStyleSheet("color: #888; margin-top: 12px; margin-bottom: 4px;")
        self._recent_layout.addWidget(recent_sep)

        recent_label = QLabel(tr("welcome.recent"))
        recent_label.setStyleSheet("font-size: 11px; color: #888;")
        self._recent_layout.addWidget(recent_label)

        self._recent_container.setVisible(False)
        layout.addWidget(self._recent_container)

        self.adjustSize()

    def set_recent_files(self, paths: list[Path]) -> None:
        assert self._recent_container is not None
        assert self._recent_layout is not None

        while self._recent_layout.count() > 2:
            item = self._recent_layout.takeAt(2)
            if item.widget():
                item.widget().setParent(None)  # type: ignore[call-overload]

        for path in paths[:5]:
            btn = QPushButton(path.name)
            btn.setFixedHeight(30)
            btn.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
            btn.setToolTip(str(path))
            btn.clicked.connect(partial(self.open_recent_requested.emit, path))
            self._recent_layout.addWidget(btn)

        self._recent_container.setVisible(bool(paths))
        self.adjustSize()
