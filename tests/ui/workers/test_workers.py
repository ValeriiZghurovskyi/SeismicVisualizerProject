"""Tests for ui.workers.tracking_worker and ui.workers.fault_tracking_worker.

QThread workers are tested by calling run() directly — no event loop needed
for the core tracking/error-emit logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityKind, EntityRegistry
from seismic_visualizer.domain.geometry import Axis, Point3D

SHAPE = (5, 5, 5)


def make_project() -> tuple[Project, int, int]:
    seismic = SeismicCube(np.zeros(SHAPE, dtype=np.uint8))
    horizons = LabelCube.empty(SHAPE)
    faults = LabelCube.empty(SHAPE)
    hreg = EntityRegistry()
    freg = EntityRegistry()
    h = hreg.create_new(EntityKind.HORIZON, "H1")
    f = freg.create_new(EntityKind.FAULT, "F1")
    project = Project(
        seismic=seismic,
        horizons=horizons,
        faults=faults,
        horizon_registry=hreg,
        fault_registry=freg,
        seismic_path=None,
        horizons_path=None,
        faults_path=None,
    )
    return project, h.id, f.id


# ===========================================================================
# TrackingWorker
# ===========================================================================

class TestTrackingWorkerConstruct:
    def test_constructs_without_error(self):
        from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker
        project, entity_id, _ = make_project()
        w = TrackingWorker(project, entity_id, MagicMock(), [], Axis.INLINE)
        assert w is not None

    def test_cancel_sets_event(self):
        from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker
        project, entity_id, _ = make_project()
        w = TrackingWorker(project, entity_id, MagicMock(), [], Axis.INLINE)
        assert not w._cancel_event.is_set()
        w.cancel()
        assert w._cancel_event.is_set()

    def test_stores_axis(self):
        from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker
        project, entity_id, _ = make_project()
        w = TrackingWorker(project, entity_id, MagicMock(), [], Axis.CROSSLINE)
        assert w._axis == Axis.CROSSLINE

    def test_stores_seed_points(self):
        from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker
        project, entity_id, _ = make_project()
        seeds = [Point3D(1, 2, 3)]
        w = TrackingWorker(project, entity_id, MagicMock(), seeds, Axis.INLINE)
        assert w._seed_points == seeds


class TestTrackingWorkerRun:
    def test_run_emits_finished_on_success(self):
        from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker
        project, entity_id, _ = make_project()

        with (
            patch("seismic_visualizer.ui.workers.tracking_worker.TrackingService") as MockTS,
            patch("seismic_visualizer.ui.workers.tracking_worker.InterpolationService") as MockIS,
        ):
            MockTS.return_value.run.return_value = None
            mock_result = MagicMock()
            MockIS.return_value.interpolate_entity.return_value = mock_result
            MockIS.return_value.apply_interpolation.return_value = None

            w = TrackingWorker(project, entity_id, MagicMock(), [], Axis.INLINE)
            finished_calls = []
            w.finished.connect(lambda: finished_calls.append(True))
            w.run()
            assert len(finished_calls) == 1

    def test_run_emits_error_on_exception(self):
        from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker
        project, entity_id, _ = make_project()

        with patch("seismic_visualizer.ui.workers.tracking_worker.TrackingService") as MockTS:
            MockTS.return_value.run.side_effect = RuntimeError("boom")

            w = TrackingWorker(project, entity_id, MagicMock(), [], Axis.INLINE)
            errors = []
            w.error.connect(errors.append)
            w.run()
            assert len(errors) == 1
            assert "boom" in errors[0]

    def test_run_skips_interpolation_when_cancelled(self):
        from seismic_visualizer.ui.workers.tracking_worker import TrackingWorker
        project, entity_id, _ = make_project()

        with (
            patch("seismic_visualizer.ui.workers.tracking_worker.TrackingService") as MockTS,
            patch("seismic_visualizer.ui.workers.tracking_worker.InterpolationService") as MockIS,
        ):
            MockTS.return_value.run.return_value = None

            w = TrackingWorker(project, entity_id, MagicMock(), [], Axis.INLINE)
            w.cancel()

            finished_calls = []
            w.finished.connect(lambda: finished_calls.append(True))
            w.run()

            MockIS.return_value.interpolate_entity.assert_not_called()
            assert len(finished_calls) == 1


# ===========================================================================
# FaultTrackingWorker
# ===========================================================================

class TestFaultTrackingWorkerConstruct:
    def test_constructs_without_error(self):
        from seismic_visualizer.ui.workers.fault_tracking_worker import FaultTrackingWorker
        project, _, fault_id = make_project()
        w = FaultTrackingWorker(project, fault_id, MagicMock(), [], Axis.INLINE)
        assert w is not None

    def test_cancel_sets_event(self):
        from seismic_visualizer.ui.workers.fault_tracking_worker import FaultTrackingWorker
        project, _, fault_id = make_project()
        w = FaultTrackingWorker(project, fault_id, MagicMock(), [], Axis.INLINE)
        assert not w._cancel_event.is_set()
        w.cancel()
        assert w._cancel_event.is_set()

    def test_stores_axis(self):
        from seismic_visualizer.ui.workers.fault_tracking_worker import FaultTrackingWorker
        project, _, fault_id = make_project()
        w = FaultTrackingWorker(project, fault_id, MagicMock(), [], Axis.CROSSLINE)
        assert w._axis == Axis.CROSSLINE

    def test_stores_entity_id(self):
        from seismic_visualizer.ui.workers.fault_tracking_worker import FaultTrackingWorker
        project, _, fault_id = make_project()
        w = FaultTrackingWorker(project, fault_id, MagicMock(), [], Axis.INLINE)
        assert w._entity_id == fault_id


class TestFaultTrackingWorkerRun:
    def test_run_emits_finished_on_success(self):
        from seismic_visualizer.ui.workers.fault_tracking_worker import FaultTrackingWorker
        project, _, fault_id = make_project()

        with (
            patch("seismic_visualizer.ui.workers.fault_tracking_worker.FaultTrackingService") as MockFTS,
            patch("seismic_visualizer.ui.workers.fault_tracking_worker.InterpolationService") as MockIS,
        ):
            MockFTS.return_value.run.return_value = None
            mock_result = MagicMock()
            MockIS.return_value.interpolate_entity.return_value = mock_result
            MockIS.return_value.apply_interpolation.return_value = None

            w = FaultTrackingWorker(project, fault_id, MagicMock(), [], Axis.INLINE)
            finished_calls = []
            w.finished.connect(lambda: finished_calls.append(True))
            w.run()
            assert len(finished_calls) == 1

    def test_run_emits_error_on_exception(self):
        from seismic_visualizer.ui.workers.fault_tracking_worker import FaultTrackingWorker
        project, _, fault_id = make_project()

        with patch("seismic_visualizer.ui.workers.fault_tracking_worker.FaultTrackingService") as MockFTS:
            MockFTS.return_value.run.side_effect = ValueError("fail")

            w = FaultTrackingWorker(project, fault_id, MagicMock(), [], Axis.INLINE)
            errors = []
            w.error.connect(errors.append)
            w.run()
            assert "fail" in errors[0]

    def test_run_skips_interpolation_when_cancelled(self):
        from seismic_visualizer.ui.workers.fault_tracking_worker import FaultTrackingWorker
        project, _, fault_id = make_project()

        with (
            patch("seismic_visualizer.ui.workers.fault_tracking_worker.FaultTrackingService") as MockFTS,
            patch("seismic_visualizer.ui.workers.fault_tracking_worker.InterpolationService") as MockIS,
        ):
            MockFTS.return_value.run.return_value = None

            w = FaultTrackingWorker(project, fault_id, MagicMock(), [], Axis.INLINE)
            w.cancel()

            finished_calls = []
            w.finished.connect(lambda: finished_calls.append(True))
            w.run()

            MockIS.return_value.interpolate_entity.assert_not_called()
