"""Smoke tests for UI dialogs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from seismic_visualizer.ui.dialogs.new_project import NewProjectDialog
from seismic_visualizer.ui.dialogs.save_prompt import ask_save_before_close


# ---------------------------------------------------------------------------
# NewProjectDialog
# ---------------------------------------------------------------------------

def test_new_project_dialog_instantiates(qtbot: object) -> None:
    dlg = NewProjectDialog()
    assert dlg is not None


def test_get_paths_empty_fields_returns_empty_seismic(qtbot: object) -> None:
    dlg = NewProjectDialog()
    seismic, horizons, faults = dlg.get_paths()
    assert str(seismic) == "."
    assert horizons is None
    assert faults is None


def test_get_paths_seismic_only(qtbot: object) -> None:
    dlg = NewProjectDialog()
    dlg._seismic_edit.setText("/data/cube_seismic.npz")
    seismic, horizons, faults = dlg.get_paths()
    assert seismic == Path("/data/cube_seismic.npz")
    assert horizons is None
    assert faults is None


def test_get_paths_all_fields_set(qtbot: object) -> None:
    dlg = NewProjectDialog()
    dlg._seismic_edit.setText("/data/cube_seismic.npz")
    dlg._horizons_edit.setText("/data/cube_horizons.npz")
    dlg._faults_edit.setText("/data/cube_faults.npz")
    seismic, horizons, faults = dlg.get_paths()
    assert seismic == Path("/data/cube_seismic.npz")
    assert horizons == Path("/data/cube_horizons.npz")
    assert faults == Path("/data/cube_faults.npz")


# ---------------------------------------------------------------------------
# ask_save_before_close
# ---------------------------------------------------------------------------

def test_ask_save_before_close_is_callable(qtbot: object) -> None:
    assert callable(ask_save_before_close)


# ---------------------------------------------------------------------------
# NewProjectDialog._browse (lines 84-87)
# ---------------------------------------------------------------------------

def test_browse_sets_text_when_file_selected(qtbot: object) -> None:
    dlg = NewProjectDialog()
    with patch(
        "seismic_visualizer.ui.dialogs.new_project.QFileDialog.getOpenFileName",
        return_value=("/data/cube_seismic.npz", ""),
    ):
        dlg._browse(dlg._seismic_edit)
    assert dlg._seismic_edit.text() == "/data/cube_seismic.npz"


def test_browse_does_not_change_text_on_cancel(qtbot: object) -> None:
    dlg = NewProjectDialog()
    with patch(
        "seismic_visualizer.ui.dialogs.new_project.QFileDialog.getOpenFileName",
        return_value=("", ""),
    ):
        dlg._browse(dlg._seismic_edit)
    assert dlg._seismic_edit.text() == ""


# ---------------------------------------------------------------------------
# InterpolationDialog
# ---------------------------------------------------------------------------

def _make_interpolation_dialog(kind: str = "horizon", n_entities: int = 1):
    """Helper that builds an InterpolationDialog with fully mocked dependencies."""
    from seismic_visualizer.ui.dialogs.interpolation_dialog import InterpolationDialog

    entities = []
    for i in range(1, n_entities + 1):
        e = MagicMock()
        e.id = i
        e.name = f"Entity{i}"
        entities.append(e)

    registry = MagicMock()
    registry.all.return_value = entities

    project = MagicMock()
    project.horizon_registry = registry
    project.fault_registry = registry

    service = MagicMock()
    view_3d = MagicMock()

    dlg = InterpolationDialog(None, kind, project, service, view_3d)
    return dlg, service, view_3d


def test_interpolation_dialog_horizon_title(qtbot: object) -> None:
    dlg, _, _ = _make_interpolation_dialog("horizon")
    assert dlg.windowTitle() == "Interpolate Horizons"


def test_interpolation_dialog_fault_title(qtbot: object) -> None:
    dlg, _, _ = _make_interpolation_dialog("fault")
    assert dlg.windowTitle() == "Interpolate Faults"


def test_interpolation_dialog_no_entities_disables_preview(qtbot: object) -> None:
    dlg, _, _ = _make_interpolation_dialog(n_entities=0)
    assert not dlg._preview_btn.isEnabled()


def test_interpolation_dialog_current_params(qtbot: object) -> None:
    from seismic_visualizer.application.dto import InterpolationParams

    dlg, _, _ = _make_interpolation_dialog()
    params = dlg._current_params()
    assert isinstance(params, InterpolationParams)


def test_interpolation_dialog_current_entity_id(qtbot: object) -> None:
    dlg, _, _ = _make_interpolation_dialog()
    assert dlg._current_entity_id() == 1


def test_interpolation_dialog_current_entity_id_none_when_empty(qtbot: object) -> None:
    dlg, _, _ = _make_interpolation_dialog(n_entities=0)
    assert dlg._current_entity_id() is None


def test_on_entity_changed_clears_result(qtbot: object) -> None:
    dlg, _, view_3d = _make_interpolation_dialog()
    dlg._result = MagicMock()
    dlg._on_entity_changed()
    assert dlg._result is None
    view_3d.clear_interpolation_preview.assert_called()


def test_on_preview_success_enables_apply(qtbot: object) -> None:
    dlg, service, view_3d = _make_interpolation_dialog()
    mock_result = MagicMock()
    mock_result.points = [(1, 2, 3), (4, 5, 6)]
    service.interpolate_entity.return_value = mock_result
    dlg._on_preview()
    assert dlg._apply_btn.isEnabled()
    view_3d.preview_interpolation.assert_called()


def test_on_preview_empty_points_disables_apply(qtbot: object) -> None:
    dlg, service, view_3d = _make_interpolation_dialog()
    mock_result = MagicMock()
    mock_result.points = []
    service.interpolate_entity.return_value = mock_result
    dlg._on_preview()
    assert not dlg._apply_btn.isEnabled()


def test_on_preview_error_shows_message(qtbot: object) -> None:
    dlg, service, _ = _make_interpolation_dialog()
    service.interpolate_entity.side_effect = ValueError("bad data")
    dlg._on_preview()
    assert "Error" in dlg._status.text()
    assert dlg._preview_btn.isEnabled()


def test_on_apply_calls_service_and_closes(qtbot: object) -> None:
    dlg, service, view_3d = _make_interpolation_dialog()
    dlg._result = MagicMock()
    dlg._on_apply()
    service.apply_interpolation.assert_called()
    view_3d.clear_interpolation_preview.assert_called()
    assert dlg._result is None


def test_on_apply_noop_when_result_is_none(qtbot: object) -> None:
    dlg, service, _ = _make_interpolation_dialog()
    dlg._result = None
    dlg._on_apply()
    service.apply_interpolation.assert_not_called()


def test_reject_clears_interpolation_preview(qtbot: object) -> None:
    dlg, _, view_3d = _make_interpolation_dialog()
    dlg.reject()
    view_3d.clear_interpolation_preview.assert_called()
