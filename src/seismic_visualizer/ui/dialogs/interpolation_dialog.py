"""InterpolationDialog — preview-first UX for horizon/fault surface interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from seismic_visualizer.application.dto import InterpolationParams, InterpolationResult
from seismic_visualizer.application.services.interpolation_service import InterpolationService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.ui.i18n import tr

if TYPE_CHECKING:
    from seismic_visualizer.ui.views.view_3d import View3D


class InterpolationDialog(QDialog):
    """Modal dialog for surface interpolation with 3D preview.

    Workflow:
        1. User picks entity + tunes parameters.
        2. Press Preview → semi-transparent cloud appears in 3D view.
        3. Press Apply → voxels written, 3D view refreshed, dialog closes.

    Args:
        parent: Owner widget.
        kind: "horizon" or "fault" — controls which registry is shown.
        project: The open project (read-only within this dialog).
        service: InterpolationService bound to the same project.
        view_3d: The live 3D view used for preview rendering.
    """

    interpolation_applied = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None,
        kind: Literal["horizon", "fault"],
        project: Project,
        service: InterpolationService,
        view_3d: "View3D",
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._entity_kind = EntityKind.HORIZON if kind == "horizon" else EntityKind.FAULT
        self._project = project
        self._service = service
        self._view_3d = view_3d
        self._result: InterpolationResult | None = None
        self._entity_ids: list[int] = []

        title_key = "interp.title_horizons" if kind == "horizon" else "interp.title_faults"
        self.setWindowTitle(tr(title_key))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)  # type: ignore[arg-type]
        self.setFixedWidth(360)

        self._build_ui()


    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)  # type: ignore[arg-type]

        self._entity_combo = QComboBox()
        registry = (
            self._project.horizon_registry
            if self._entity_kind == EntityKind.HORIZON
            else self._project.fault_registry
        )
        entities = registry.all()
        self._entity_ids = [e.id for e in entities]
        for e in entities:
            self._entity_combo.addItem(e.name, e.id)
        form.addRow(tr("interp.entity"), self._entity_combo)

        self._grid_step = QDoubleSpinBox()
        self._grid_step.setRange(0.1, 50.0)
        self._grid_step.setSingleStep(0.5)
        self._grid_step.setValue(1.0)
        self._grid_step.setSuffix(" vx")
        form.addRow(tr("interp.grid_step"), self._grid_step)

        self._median_size = QSpinBox()
        self._median_size.setRange(1, 15)
        self._median_size.setSingleStep(2)
        self._median_size.setValue(3)
        form.addRow(tr("interp.median_size"), self._median_size)

        self._max_radius = QDoubleSpinBox()
        self._max_radius.setRange(1.0, 500.0)
        self._max_radius.setSingleStep(5.0)
        self._max_radius.setValue(10.0)
        self._max_radius.setSuffix(" vx")
        form.addRow(tr("interp.max_radius"), self._max_radius)

        layout.addLayout(form)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setAlignment(Qt.AlignCenter)  # type: ignore[arg-type]
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._preview_btn = QPushButton(tr("interp.preview"))
        self._apply_btn = QPushButton(tr("interp.apply"))
        cancel_btn = QPushButton(tr("interp.cancel"))
        self._apply_btn.setEnabled(False)

        if not entities:
            self._preview_btn.setEnabled(False)
            self._status.setText(tr("interp.no_entities"))

        self._preview_btn.clicked.connect(self._on_preview)
        self._apply_btn.clicked.connect(self._on_apply)
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._apply_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._entity_combo.currentIndexChanged.connect(self._on_entity_changed)


    def _current_entity_id(self) -> int | None:
        idx = self._entity_combo.currentIndex()
        if idx < 0 or idx >= len(self._entity_ids):
            return None
        return self._entity_ids[idx]

    def _current_params(self) -> InterpolationParams:
        return InterpolationParams(
            grid_step=self._grid_step.value(),
            median_size=self._median_size.value(),
            max_radius=self._max_radius.value(),
        )


    def _on_entity_changed(self) -> None:
        self._result = None
        self._apply_btn.setEnabled(False)
        self._view_3d.clear_interpolation_preview()
        self._status.setText("")

    def _on_preview(self) -> None:
        entity_id = self._current_entity_id()
        if entity_id is None:
            return

        self._status.setText(tr("interp.computing"))
        self._preview_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            result = self._service.interpolate_entity(
                self._entity_kind, entity_id, self._current_params()
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._status.setText(f"Error: {exc}")
            self._preview_btn.setEnabled(True)
            return
        finally:
            self._preview_btn.setEnabled(True)

        self._result = result

        if len(result.points) == 0:
            self._status.setText(tr("interp.no_voxels"))
            self._apply_btn.setEnabled(False)
            return

        self._view_3d.preview_interpolation(result.points, (0, 255, 255))
        self._status.setText(tr("interp.preview_result").format(n=len(result.points)))
        self._apply_btn.setEnabled(True)

    def _on_apply(self) -> None:
        if self._result is None:
            return
        self._service.apply_interpolation(self._result)
        self._view_3d.clear_interpolation_preview()
        self._result = None
        self.interpolation_applied.emit()
        self.accept()

    def reject(self) -> None:
        self._view_3d.clear_interpolation_preview()
        super().reject()
