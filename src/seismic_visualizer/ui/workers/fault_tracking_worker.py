"""FaultTrackingWorker — QThread wrapper for ML fault auto-tracking."""

from __future__ import annotations

import threading

from PyQt5.QtCore import QThread, pyqtSignal

from seismic_visualizer.application.dto import InterpolationParams
from seismic_visualizer.application.services.fault_tracking_service import FaultTrackingService
from seismic_visualizer.application.services.interpolation_service import InterpolationService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.geometry import Axis, Point3D
from seismic_visualizer.domain.ml.fault_tracker import FaultTracker


class FaultTrackingWorker(QThread):
    """Runs FaultTrackingService in a background thread to avoid UI freezing.

    The tracker must be created in the main thread before passing here —
    onnxruntime DLLs fail to initialize when first imported inside a QThread
    on Windows.

    Args:
        project: Open project (mutated in place on success).
        entity_id: Fault entity to label.
        tracker: Pre-initialized FaultTracker (created in main thread).
        seed_points: User-provided seed voxels.
        axis: INLINE or CROSSLINE propagation axis.

    Signals:
        finished: Emitted when tracking completes successfully.
        error: Emitted with an error message string on failure.
    """

    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        project: Project,
        entity_id: int,
        tracker: FaultTracker,
        seed_points: list[Point3D],
        axis: Axis,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._entity_id = entity_id
        self._tracker = tracker
        self._seed_points = seed_points
        self._axis = axis
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            FaultTrackingService().run(
                self._project,
                self._entity_id,
                self._tracker,
                self._seed_points,
                self._axis,
                cancel_check=self._cancel_event.is_set,
            )
            if not self._cancel_event.is_set():
                svc = InterpolationService(self._project)
                params = InterpolationParams(max_radius=50.0)
                result = svc.interpolate_entity(EntityKind.FAULT, self._entity_id, params)
                svc.apply_interpolation(result)
            self.finished.emit()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.error.emit(str(exc))
