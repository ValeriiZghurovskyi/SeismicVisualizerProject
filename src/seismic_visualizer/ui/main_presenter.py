"""MainPresenter — orchestrates top-level window events: file I/O, navigation, view modes.

Lives in ui/ because it depends on Qt (QFileDialog, QMessageBox, MainWindow) directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from seismic_visualizer.application.presenters.slice_presenter import SlicePresenter
from seismic_visualizer.application.presenters.view_2d_presenter import View2DPresenter
from seismic_visualizer.application.presenters.view_3d_presenter import View3DPresenter
from seismic_visualizer.application.services.attribute_service import AttributeService
from seismic_visualizer.application.services.interpolation_service import InterpolationService
from seismic_visualizer.application.services.persistence_service import PersistenceService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.state import SliceIndex
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.exceptions import EntityNotFoundError
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.infrastructure.config import AppConfig
from seismic_visualizer.ui.dialogs.interpolation_dialog import InterpolationDialog
from seismic_visualizer.ui.dialogs.save_prompt import ask_save_before_close
from seismic_visualizer.ui.i18n import tr
from seismic_visualizer.ui.main_window import MainWindow
from seismic_visualizer.ui.theme import apply_theme

_log = logging.getLogger(__name__)

_NPZ_FILTER = "NumPy archive (*.npz);;All files (*)"
_SVP_FILTER = "Seismic Visualizer Project (*.svp);;All files (*)"
_SEGY_FILTER = "SEG-Y (*.segy *.sgy);;All files (*)"


class MainPresenter:
    """Orchestrates window-level events: file I/O, navigation, view-mode switching.

    Created by App.create_window() with an already-loaded project.  Holds all
    navigation state and delegates 2D labeling interactions to View2DPresenter.

    Args:
        qt_app: The QApplication instance.
        persistence: Project persistence service.
        config: Mutable application configuration.
        attr_service: Attribute computation and cache (shared with SlicePresenter).
        window: The main application window.
        view2d_presenter: Presenter for 2D labeling interactions.
        project: Already-loaded project supplied by App on first open.
        slice_presenter: Already-created slice presenter for the loaded project.
        last_index: Shared mutable dict of current slice indices per axis.
    """

    def __init__(
        self,
        qt_app: QApplication,
        persistence: PersistenceService,
        config: AppConfig,
        attr_service: AttributeService,
        window: MainWindow,
        view2d_presenter: View2DPresenter,
        project: Project,
        slice_presenter: SlicePresenter,
        last_index: dict[Axis, int],
    ) -> None:
        self._qt_app = qt_app
        self._persistence = persistence
        self._config = config
        self._attr_service = attr_service
        self._window = window
        self._view2d_presenter = view2d_presenter
        self._view3d_presenter: View3DPresenter | None = None
        self._project: Project = project
        self._slice_presenter: SlicePresenter = slice_presenter
        self._last_index = last_index
        self._clip_labels: bool = False
        self._current_view_mode: str = "2d"
        self._last_2d_axis: Axis = Axis.INLINE
        self._interp_dlg: InterpolationDialog | None = None


    def apply_loaded_project(self) -> None:
        """Update the entire view after a project has been loaded."""
        self._window.view().set_stroke_color(None)
        axis = self._last_2d_axis
        self._slice_presenter.navigate(SliceIndex(axis=axis, index=self._last_index[axis]))
        max_index = self._project.seismic.shape[axis.value] - 1
        self._window.update_slice_range(axis, max_index, self._last_index[axis])
        self._window.update_all_slice_ranges({
            a: (self._project.seismic.shape[a.value] - 1, self._last_index[a])
            for a in (Axis.INLINE, Axis.CROSSLINE, Axis.TIME)
        })
        if self._current_view_mode == "3d":
            self._view3d_presenter = View3DPresenter(self._project, self._window.view_3d())
            self._view2d_presenter.set_view3d_presenter(self._view3d_presenter)
            entity_list = self._view2d_presenter.build_entity_list()
            self._view3d_presenter.sync_indices(self._last_index)
            self._view3d_presenter.refresh_planes()
            self._view3d_presenter.refresh_labels(entity_list, clip_to_slices=self._clip_labels)
            self._window.view_3d().update_entity_list(entity_list)
            self._window.view_3d().reset_camera()
            for a in (Axis.INLINE, Axis.CROSSLINE, Axis.TIME):
                self._view2d_presenter.push_thumbnail(a)
        self._view2d_presenter.refresh_canvas()


    def _check_unsaved_changes(self) -> bool:
        """Return True if the caller may proceed; False if the user cancelled.

        When the project is dirty, shows a Save / Discard / Cancel dialog.
        Save → calls _on_save_svp() and returns True only if is_dirty was cleared.
        Discard → returns True.
        Cancel → returns False.
        """
        if not self._project.is_dirty:
            return True
        choice = ask_save_before_close(self._window)
        if choice is None:
            return False
        if choice:
            self._on_save_svp()
            return not self._project.is_dirty
        return True


    def _on_open_seismic_segy(self) -> None:
        if not self._check_unsaved_changes():
            return
        npz_path = self._convert_segy_to_npz(self._window)
        if npz_path is None:
            return
        self._window.statusBar().showMessage(tr("status.loading_seismic"))
        QApplication.processEvents()
        if not self._load_project(npz_path, None, None, self._window):
            self._window.statusBar().clearMessage()
            return
        self._window.refresh_recent_menu(self._config.get_recent())
        self._window.statusBar().clearMessage()
        self.apply_loaded_project()

    def _on_open_seismic_npz(self) -> None:
        if not self._check_unsaved_changes():
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self._window, tr("fdlg.open_seismic_npz"), "", _NPZ_FILTER
        )
        if not path_str:
            return
        self._window.statusBar().showMessage(tr("status.loading_seismic"))
        QApplication.processEvents()
        if not self._load_project(Path(path_str), None, None, self._window):
            self._window.statusBar().clearMessage()
            return
        self._window.refresh_recent_menu(self._config.get_recent())
        self._window.statusBar().clearMessage()
        self.apply_loaded_project()

    def _on_open_svp(self) -> None:
        if not self._check_unsaved_changes():
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self._window, tr("fdlg.open_project"), "", _SVP_FILTER
        )
        if not path_str:
            return
        self._window.statusBar().showMessage(tr("status.loading_project"))
        QApplication.processEvents()
        if not self._load_svp(Path(path_str), self._window):
            self._window.statusBar().clearMessage()
            return
        self._window.refresh_recent_menu(self._config.get_recent())
        self._window.statusBar().clearMessage()
        self.apply_loaded_project()

    def _on_open_recent_from_window(self, path: Path) -> None:
        if not self._check_unsaved_changes():
            return
        if not path.exists():
            QMessageBox.warning(
                self._window, tr("msg.file_not_found"),
                tr("msg.file_not_found_detail").format(path=path),
            )
            self._window.refresh_recent_menu(self._config.get_recent())
            return
        self._window.statusBar().showMessage(tr("status.loading"))
        QApplication.processEvents()
        ok = (
            self._load_svp(path, self._window)
            if path.suffix.lower() == ".svp"
            else self._load_project(path, None, None, self._window)
        )
        if not ok:
            self._window.statusBar().clearMessage()
            return
        self._window.refresh_recent_menu(self._config.get_recent())
        self._window.statusBar().clearMessage()
        self.apply_loaded_project()

    def _on_load_horizons(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self._window, tr("fdlg.load_horizons"), "", _NPZ_FILTER
        )
        if not path_str:
            return
        self._window.statusBar().showMessage(tr("status.loading_horizons"))
        QApplication.processEvents()
        try:
            self._persistence.load_horizons(self._project, Path(path_str))
        except Exception as exc:  # pylint: disable=broad-exception-caught  # persistence raises numpy/zip/OS errors
            _log.error("Failed to load horizons: %s", exc)
            self._window.statusBar().clearMessage()
            QMessageBox.critical(self._window, tr("msg.error"), tr("msg.failed_load_horizons").format(exc=exc))
            return
        self._view2d_presenter.clear_active_entity_if_kind(EntityKind.HORIZON)
        self._refresh_labels_3d_if_active()
        self._window.statusBar().clearMessage()
        self._view2d_presenter.refresh_canvas()

    def _on_load_faults(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self._window, tr("fdlg.load_faults"), "", _NPZ_FILTER
        )
        if not path_str:
            return
        self._window.statusBar().showMessage(tr("status.loading_faults"))
        QApplication.processEvents()
        try:
            self._persistence.load_faults(self._project, Path(path_str))
        except Exception as exc:  # pylint: disable=broad-exception-caught  # persistence raises numpy/zip/OS errors
            _log.error("Failed to load faults: %s", exc)
            self._window.statusBar().clearMessage()
            QMessageBox.critical(self._window, tr("msg.error"), tr("msg.failed_load_faults").format(exc=exc))
            return
        self._view2d_presenter.clear_active_entity_if_kind(EntityKind.FAULT)
        self._refresh_labels_3d_if_active()
        self._window.statusBar().clearMessage()
        self._view2d_presenter.refresh_canvas()

    def _on_save_seismic(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self._window, tr("fdlg.save_seismic"),
            str(self._project.seismic_path or "seismic.npz"), _NPZ_FILTER,
        )
        if not path_str:
            return
        self._project.seismic_path = Path(path_str)
        self._window.statusBar().showMessage(tr("status.saving_seismic"))
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            self._persistence.save_seismic(self._project)
            self._window.view().update_status(tr("status.seismic_saved"))
        except Exception as exc:  # pylint: disable=broad-exception-caught  # numpy/OS errors on save
            _log.error("Save seismic failed: %s", exc)
            self._window.view().update_status(tr("status.save_failed").format(exc=exc))
        finally:
            QApplication.restoreOverrideCursor()
            self._window.statusBar().clearMessage()

    def _on_save_horizons(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self._window, tr("fdlg.save_horizons"),
            str(self._project.horizons_path or "horizons.npz"), _NPZ_FILTER,
        )
        if not path_str:
            return
        self._project.horizons_path = Path(path_str)
        self._window.statusBar().showMessage(tr("status.saving_horizons"))
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            self._persistence.save_horizons(self._project)
            self._window.view().update_status(tr("status.horizons_saved"))
        except Exception as exc:  # pylint: disable=broad-exception-caught  # numpy/OS errors on save
            _log.error("Save horizons failed: %s", exc)
            self._window.view().update_status(tr("status.save_failed").format(exc=exc))
        finally:
            QApplication.restoreOverrideCursor()
            self._window.statusBar().clearMessage()

    def _on_save_faults(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self._window, tr("fdlg.save_faults"),
            str(self._project.faults_path or "faults.npz"), _NPZ_FILTER,
        )
        if not path_str:
            return
        self._project.faults_path = Path(path_str)
        self._window.statusBar().showMessage(tr("status.saving_faults"))
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            self._persistence.save_faults(self._project)
            self._window.view().update_status(tr("status.faults_saved"))
        except Exception as exc:  # pylint: disable=broad-exception-caught  # numpy/OS errors on save
            _log.error("Save faults failed: %s", exc)
            self._window.view().update_status(tr("status.save_failed").format(exc=exc))
        finally:
            QApplication.restoreOverrideCursor()
            self._window.statusBar().clearMessage()

    def _on_save_svp(self) -> None:
        default = str(self._project.project_path or "project.svp")
        path_str, _ = QFileDialog.getSaveFileName(
            self._window, tr("fdlg.save_project"), default, _SVP_FILTER
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != ".svp":
            path = path.with_suffix(".svp")
        self._window.statusBar().showMessage(tr("status.saving_project"))
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            self._persistence.save_svp(self._project, path)
            self._config.add_recent(path)
            try:
                self._config.save(AppConfig.config_path())
            except Exception as exc:  # pylint: disable=broad-exception-caught  # config save OS/json errors
                _log.warning("Could not save config: %s", exc)
            self._window.refresh_recent_menu(self._config.get_recent())
            self._window.view().update_status(tr("status.project_saved"))
        except Exception as exc:  # pylint: disable=broad-exception-caught  # numpy/zip/OS errors on save
            _log.error("Save .svp failed: %s", exc)
            self._window.view().update_status(tr("status.save_failed").format(exc=exc))
        finally:
            QApplication.restoreOverrideCursor()
            self._window.statusBar().clearMessage()

    def _on_clear_recent(self) -> None:
        self._config.clear_recent()
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught  # config save OS/json errors
            _log.warning("Could not save config: %s", exc)
        self._window.refresh_recent_menu([])


    def _on_axis_changed(self, axis: Axis) -> None:
        self._last_2d_axis = axis
        self._view2d_presenter.clear_stroke()
        self._slice_presenter.navigate(SliceIndex(axis=axis, index=self._last_index[axis]))
        max_index = self._project.seismic.shape[axis.value] - 1
        self._window.update_slice_range(axis, max_index, self._last_index[axis])
        self._view2d_presenter.refresh_canvas()

    def _on_slice_changed(self, axis: Axis, index: int) -> None:
        self._view2d_presenter.clear_stroke()
        self._last_index[axis] = index
        self._slice_presenter.navigate(SliceIndex(axis=axis, index=index))
        if self._view3d_presenter is not None and self._current_view_mode == "3d":
            self._view3d_presenter.on_plane_position_changed(axis, index)
            if self._clip_labels:
                self._view3d_presenter.refresh_labels(
                    self._view2d_presenter.build_entity_list(), clip_to_slices=True
                )
            self._view2d_presenter.push_thumbnail(axis)
        self._view2d_presenter.refresh_canvas()

    def _on_view_mode_changed(self, mode: str) -> None:
        self._current_view_mode = mode
        if mode == "3d":
            if self._view3d_presenter is None:
                self._view3d_presenter = View3DPresenter(
                    self._project, self._window.view_3d()
                )
                self._view2d_presenter.set_view3d_presenter(self._view3d_presenter)
            self._window.switch_to_3d()
            self._window.statusBar().showMessage("Rendering 3D view…")
            QApplication.processEvents()
            entity_list = self._view2d_presenter.build_entity_list()
            self._view3d_presenter.sync_indices(self._last_index)
            self._view3d_presenter.refresh_planes()
            self._view3d_presenter.refresh_labels(entity_list, clip_to_slices=self._clip_labels)
            self._window.view_3d().update_entity_list(entity_list)
            self._window.view_3d().reset_camera()
            for axis in (Axis.INLINE, Axis.CROSSLINE, Axis.TIME):
                self._view2d_presenter.push_thumbnail(axis)
            self._window.statusBar().clearMessage()
            self._window.update_all_slice_ranges({
                a: (self._project.seismic.shape[a.value] - 1, self._last_index[a])
                for a in (Axis.INLINE, Axis.CROSSLINE, Axis.TIME)
            })
        else:
            self._window.switch_to_2d()
            axis = self._last_2d_axis
            self._slice_presenter.navigate(SliceIndex(axis=axis, index=self._last_index[axis]))
            max_index = self._project.seismic.shape[axis.value] - 1
            self._window.update_slice_range(axis, max_index, self._last_index[axis])
            self._view2d_presenter.refresh_canvas()

    def _on_3d_plane_moved(self, axis: Axis, index: int) -> None:
        self._last_index[axis] = index
        self._slice_presenter.navigate(SliceIndex(axis=axis, index=index))
        self._window.set_slice_panel_value(axis, index)
        self._view2d_presenter.refresh_canvas()
        if self._view3d_presenter is not None and self._clip_labels:
            self._view3d_presenter.refresh_labels(
                self._view2d_presenter.build_entity_list(), clip_to_slices=True
            )
        self._view2d_presenter.push_thumbnail(axis)

    def _on_3d_plane_visibility_changed(self, axis: Axis, visible: bool) -> None:
        if self._view3d_presenter is not None:
            self._view3d_presenter.set_plane_visible(axis, visible)

    def _on_3d_visibility_changed(self, entity_id: int, kind: object, visible: bool) -> None:
        from seismic_visualizer.domain.entities import EntityKind as _EK
        registry = (
            self._project.horizon_registry
            if kind == _EK.HORIZON
            else self._project.fault_registry
        )
        try:
            registry.set_visibility(entity_id, visible)
        except EntityNotFoundError:
            _log.warning("3d_visibility_changed: id %d not found", entity_id)
        if self._view3d_presenter is not None:
            self._view3d_presenter.set_entity_visible(entity_id, kind, visible)  # type: ignore[arg-type]
        self._window.view_3d().update_entity_list(self._view2d_presenter.build_entity_list())

    def _on_clip_labels_changed(self, checked: bool) -> None:
        self._clip_labels = checked
        if self._view3d_presenter is not None:
            self._view3d_presenter.refresh_labels(
                self._view2d_presenter.build_entity_list(), clip_to_slices=checked
            )


    def _on_attribute_changed(self, name: str) -> None:
        self._attr_service.clear()
        self._slice_presenter.state.current_attribute = name if name else None
        self._view2d_presenter.refresh_canvas()

    def _on_contrast_changed(self, vmin: int, vmax: int) -> None:
        self._slice_presenter.state.contrast_vmin = vmin
        self._slice_presenter.state.contrast_vmax = vmax
        self._view2d_presenter.refresh_canvas()

    def _on_colormap_changed(self, name: str) -> None:
        self._config.colormap = name
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught  # config save OS/json errors
            _log.warning("Could not save config: %s", exc)
        self._slice_presenter.set_colormap(name)
        self._view2d_presenter.refresh_canvas()
        self._window.view_3d().set_colormap(name)
        if self._view3d_presenter is not None:
            self._view3d_presenter.refresh_planes()
            for axis in (Axis.INLINE, Axis.CROSSLINE, Axis.TIME):
                self._view2d_presenter.push_thumbnail(axis)

    def _on_theme_changed(self, theme: str) -> None:
        self._config.theme = theme
        apply_theme(self._qt_app, theme)
        self._window.view_3d().set_theme(theme)
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught  # config save OS/json errors
            _log.warning("Could not save config: %s", exc)

    def _on_canvas_hover(self, row: int, col: int) -> None:
        hud = self._window.hud()
        if row == -1:
            hud.clear()
            return
        p = self._slice_presenter.display_to_point3d(row, col)
        shape = self._project.seismic.data.shape
        if not (
            0 <= p.inline < shape[0]
            and 0 <= p.crossline < shape[1]
            and 0 <= p.time < shape[2]
        ):
            hud.clear()
            return
        amp = int(self._project.seismic.data[p.inline, p.crossline, p.time])
        hud.update_position(p.inline, p.crossline, p.time, amp)

    def _on_drawing_mode_changed(self, mode: str) -> None:
        _log.debug("drawing mode: %s", mode)


    def _on_auto_track_horizon(self) -> None:
        if self._project is None:
            QMessageBox.warning(self._window, "No project", "Open a project first.")
            return

        model_path = Path(__file__).parents[3] / "models" / "horizon_tracker.onnx"
        if not model_path.exists():
            QMessageBox.warning(
                self._window,
                "Model not found",
                f"ONNX model not found at:\n{model_path}\n\n"
                "Train and export the model first:\n"
                "  python -m ml.train\n"
                "  python -m ml.export_onnx",
            )
            return

        from seismic_visualizer.ui.dialogs.auto_track_dialog import AutoTrackDialog

        dlg = AutoTrackDialog(
            parent=self._window,
            project=self._project,
            current_entity_id=self._view2d_presenter.active_entity_id,
            current_axis=self._view2d_presenter.current_axis,
            model_path=model_path,
        )
        dlg.tracking_applied.connect(self._on_tracking_applied)
        dlg.exec()

    def _on_tracking_applied(self) -> None:
        self._view2d_presenter.refresh_canvas()
        self._refresh_labels_3d_if_active()

    def _on_auto_track_fault(self) -> None:
        if self._project is None:
            QMessageBox.warning(self._window, "No project", "Open a project first.")
            return

        model_path = Path(__file__).parents[3] / "models" / "fault_tracker.onnx"
        if not model_path.exists():
            QMessageBox.warning(
                self._window,
                "Model not found",
                f"ONNX model not found at:\n{model_path}\n\n"
                "Train and export the model first:\n"
                "  python -m ml.train --label_type faults\n"
                "  python -m ml.export_onnx"
                " --checkpoint ml/checkpoints_faults/best.pt --output models/fault_tracker.onnx",
            )
            return

        from seismic_visualizer.ui.dialogs.auto_track_fault_dialog import AutoTrackFaultDialog

        dlg = AutoTrackFaultDialog(
            parent=self._window,
            project=self._project,
            current_entity_id=self._view2d_presenter.active_entity_id,
            current_axis=self._view2d_presenter.current_axis,
            model_path=model_path,
        )
        dlg.tracking_applied.connect(self._on_tracking_applied)
        dlg.exec()

    def _on_interpolate_horizons(self) -> None:
        self._open_interpolation_dialog("horizon")

    def _on_interpolate_faults(self) -> None:
        self._open_interpolation_dialog("fault")

    def _open_interpolation_dialog(self, kind: str) -> None:
        if self._interp_dlg is not None:
            self._interp_dlg.close()
        service = InterpolationService(self._project)
        dlg = InterpolationDialog(
            self._window, kind, self._project, service, self._window.view_3d()  # type: ignore[arg-type]
        )
        dlg.interpolation_applied.connect(self._on_interpolation_applied)
        dlg.finished.connect(self._on_interp_dlg_closed)
        self._interp_dlg = dlg
        dlg.show()

    def _on_interp_dlg_closed(self) -> None:
        self._window.view_3d().clear_interpolation_preview()
        self._interp_dlg = None

    def _on_interpolation_applied(self) -> None:
        self._refresh_labels_3d_if_active()
        self._view2d_presenter.refresh_canvas()


    def _load_project(
        self,
        seismic_path: Path,
        horizons_path: Path | None,
        faults_path: Path | None,
        parent: QWidget | None,
    ) -> bool:
        """Load project from NPZ. Returns True on success."""
        try:
            project = self._persistence.load(seismic_path, horizons_path, faults_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # persistence raises numpy/zip/OS errors
            _log.error("Failed to load project: %s", exc)
            QMessageBox.critical(parent, tr("msg.error"), tr("msg.failed_load_project").format(exc=exc))
            return False
        self._project = project
        self._view3d_presenter = None
        self._view2d_presenter.set_view3d_presenter(None)
        self._slice_presenter = SlicePresenter(
            project, self._attr_service, cmap_name=self._config.colormap
        )
        self._view2d_presenter.set_project(project, self._slice_presenter)
        self._config.add_recent(seismic_path)
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught  # config save OS/json errors
            _log.warning("Could not save config: %s", exc)
        return True

    def _load_svp(self, path: Path, parent: QWidget | None) -> bool:
        """Load project from .svp file. Returns True on success."""
        try:
            project = self._persistence.load_svp(path)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # persistence raises numpy/zip/OS errors
            _log.error("Failed to load .svp: %s", exc)
            QMessageBox.critical(parent, tr("msg.error"), tr("msg.failed_load_svp").format(exc=exc))
            return False
        self._project = project
        self._view3d_presenter = None
        self._view2d_presenter.set_view3d_presenter(None)
        self._slice_presenter = SlicePresenter(
            project, self._attr_service, cmap_name=self._config.colormap
        )
        self._view2d_presenter.set_project(project, self._slice_presenter)
        self._config.add_recent(path)
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught  # config save OS/json errors
            _log.warning("Could not save config: %s", exc)
        return True

    def _convert_segy_to_npz(self, parent: QWidget | None) -> Path | None:
        """Open a SEG-Y file, convert it to .npz, and return the saved path."""
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter
        from seismic_visualizer.infrastructure.io.segy_reader import SegyReader

        segy_str, _ = QFileDialog.getOpenFileName(parent, tr("fdlg.open_segy"), "", _SEGY_FILTER)
        if not segy_str:
            return None
        segy_path = Path(segy_str)
        npz_str, _ = QFileDialog.getSaveFileName(
            parent, tr("fdlg.save_npz_from_segy"), str(segy_path.with_suffix(".npz")), _NPZ_FILTER
        )
        if not npz_str:
            return None
        npz_path = Path(npz_str)
        if npz_path.suffix.lower() != ".npz":
            npz_path = npz_path.with_suffix(".npz")
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            cube = SegyReader().read_seismic(segy_path)
            NpzWriter().write_seismic(npz_path, cube)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # segyio/numpy/OS errors
            _log.error("SEG-Y conversion failed: %s", exc)
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(parent, tr("msg.conversion_error"), tr("msg.failed_segy").format(exc=exc))
            return None
        finally:
            QApplication.restoreOverrideCursor()
        return npz_path


    def _refresh_labels_3d_if_active(self) -> None:
        """Rebuild 3D label meshes when labels changed outside labeling (load/interpolation)."""
        if self._view3d_presenter is None or self._current_view_mode != "3d":
            return
        self._window.statusBar().showMessage("Rendering 3D labels…")
        QApplication.processEvents()
        self._view3d_presenter.invalidate_label_cache()
        entity_list = self._view2d_presenter.build_entity_list()
        self._view3d_presenter.refresh_labels(entity_list, clip_to_slices=self._clip_labels)
        self._window.view_3d().update_entity_list(entity_list)
        self._window.statusBar().clearMessage()
