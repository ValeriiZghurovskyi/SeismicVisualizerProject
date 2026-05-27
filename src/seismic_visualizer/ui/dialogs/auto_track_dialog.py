"""AutoTrackDialog — UI for ML horizon auto-tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.geometry import Axis, Point3D
from seismic_visualizer.ui.i18n import tr
from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker


class AutoTrackDialog(QDialog):
    """Dialog for running ML auto-tracking on a horizon entity.

    Seeds are taken from already-labeled voxels of the selected entity.
    Tracking runs in a background QThread to avoid freezing the UI.

    Args:
        parent: Owner widget.
        project: The open project (mutated in place on success).
        current_entity_id: Pre-selected entity (from active sidebar item).
        current_axis: Pre-selected axis (from active 2D view).
        model_path: Path to the .onnx model file.
    """

    tracking_applied = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None,
        project: Project,
        current_entity_id: int | None,
        current_axis: Axis,
        model_path: Path,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._model_path = model_path
        self._worker: TrackingWorker | None = None
        self._entity_ids: list[int] = []

        self.setWindowTitle(tr("autotrack_h.title"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)  # type: ignore[arg-type]
        self.setFixedWidth(380)

        self._build_ui(current_entity_id, current_axis)


    def _build_ui(self, current_entity_id: int | None, current_axis: Axis) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)  # type: ignore[arg-type]

        self._entity_combo = QComboBox()
        entities = self._project.horizon_registry.all()
        self._entity_ids = [e.id for e in entities]
        for e in entities:
            self._entity_combo.addItem(e.name, e.id)
        if current_entity_id is not None and current_entity_id in self._entity_ids:
            self._entity_combo.setCurrentIndex(self._entity_ids.index(current_entity_id))
        form.addRow(tr("autotrack_h.entity"), self._entity_combo)

        self._axis_combo = QComboBox()
        self._axis_combo.addItem(tr("toolbar.inline"), Axis.INLINE)
        self._axis_combo.addItem(tr("toolbar.crossline"), Axis.CROSSLINE)
        if current_axis == Axis.CROSSLINE:
            self._axis_combo.setCurrentIndex(1)
        form.addRow(tr("autotrack.axis"), self._axis_combo)

        self._seed_label = QLabel()
        form.addRow(tr("autotrack.seeds"), self._seed_label)

        layout.addLayout(form)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignCenter)  # type: ignore[arg-type]
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton(tr("autotrack.run"))
        cancel_btn = QPushButton(tr("autotrack.cancel"))

        self._run_btn.clicked.connect(self._on_run)
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._entity_combo.currentIndexChanged.connect(self._update_seed_info)
        self._update_seed_info()

        if not entities:
            self._run_btn.setEnabled(False)
            self._status_label.setText(tr("autotrack_h.no_entities"))


    def _selected_entity_id(self) -> int | None:
        idx = self._entity_combo.currentIndex()
        if idx < 0 or idx >= len(self._entity_ids):
            return None
        return self._entity_ids[idx]

    def _selected_axis(self) -> Axis:
        return self._axis_combo.currentData()

    def _collect_seeds(self) -> list[Point3D]:
        entity_id = self._selected_entity_id()
        if entity_id is None:
            return []
        data = self._project.horizons.data
        coords = np.argwhere(data == entity_id)
        return [
            Point3D(inline=int(r[0]), crossline=int(r[1]), time=int(r[2]))
            for r in coords
        ]

    def _update_seed_info(self) -> None:
        seeds = self._collect_seeds()
        n = len(seeds)
        key = "autotrack.seeds_one" if n == 1 else "autotrack.seeds_many"
        self._seed_label.setText(tr(key).format(n=n))
        self._run_btn.setEnabled(n > 0 and self._worker is None)


    def _on_run(self) -> None:
        entity_id = self._selected_entity_id()
        if entity_id is None:
            return

        seeds = self._collect_seeds()
        if not seeds:
            QMessageBox.warning(
                self,
                tr("autotrack_h.no_seeds_title"),
                tr("autotrack_h.no_seeds_msg"),
            )
            return

        self._progress_bar.setVisible(True)
        self._status_label.setText(tr("autotrack.loading_model"))
        self._run_btn.setEnabled(False)
        self._entity_combo.setEnabled(False)
        self._axis_combo.setEnabled(False)

        try:
            from seismic_visualizer.infrastructure.ml.onnx_tracker import OnnxTracker
            tracker = OnnxTracker(self._model_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_error(str(exc))
            return

        self._status_label.setText(tr("autotrack.tracking"))

        self._worker = TrackingWorker(
            project=self._project,
            entity_id=entity_id,
            tracker=tracker,
            seed_points=seeds,
            axis=self._selected_axis(),
            parent=self,
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self) -> None:
        self.tracking_applied.emit()
        self.accept()

    def _on_error(self, message: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Error: {message}")
        self._run_btn.setEnabled(True)
        self._entity_combo.setEnabled(True)
        self._axis_combo.setEnabled(True)
        self._worker = None

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.finished.disconnect(self._on_finished)
            self._worker.error.disconnect(self._on_error)
            self._worker.cancel()
            self._worker.setParent(None)
        super().reject()
