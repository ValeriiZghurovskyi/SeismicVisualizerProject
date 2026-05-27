"""App — composition root. Creates services, shows welcome, wires presenters."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from seismic_visualizer.ui.main_presenter import MainPresenter
from seismic_visualizer.application.presenters.slice_presenter import SlicePresenter
from seismic_visualizer.application.presenters.view_2d_presenter import View2DPresenter
from seismic_visualizer.application.services.attribute_service import AttributeService
from seismic_visualizer.application.services.labeling_service import LabelingService
from seismic_visualizer.application.services.persistence_service import PersistenceService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.infrastructure.config import AppConfig
from seismic_visualizer.ui.i18n import LanguageManager, tr
from seismic_visualizer.ui.main_window import MainWindow
from seismic_visualizer.ui.theme import apply_theme, set_titlebar_dark
from seismic_visualizer.ui.welcome import WelcomeScreen

_log = logging.getLogger(__name__)

_NPZ_FILTER = "NumPy archive (*.npz);;All files (*)"
_SVP_FILTER = "Seismic Visualizer Project (*.svp);;All files (*)"
_SEGY_FILTER = "SEG-Y (*.segy *.sgy);;All files (*)"


class App:
    """Composition root — creates services, shows welcome screen, builds MainWindow."""

    def __init__(self, qt_app: QApplication) -> None:
        self._qt_app = qt_app
        self._persistence = PersistenceService()
        self._attr_service = AttributeService()
        self._labeling_service = LabelingService()
        self._config = AppConfig.load(AppConfig.config_path())
        self._project: Project | None = None
        self._slice_presenter: SlicePresenter | None = None
        self._welcome: WelcomeScreen | None = None
        self._main_presenter: MainPresenter | None = None


    def start(self) -> None:
        """Show the welcome screen. Entry point for the application."""
        LanguageManager.instance().set_language(self._config.language)
        LanguageManager.instance().language_changed.connect(self._on_language_changed)
        apply_theme(self._qt_app, self._config.theme)
        self._welcome = WelcomeScreen()
        self._welcome.open_seismic_npz_requested.connect(self._on_open_seismic_npz_from_welcome)
        self._welcome.open_seismic_segy_requested.connect(self._on_open_seismic_segy_from_welcome)
        self._welcome.open_svp_requested.connect(self._on_open_svp_from_welcome)
        self._welcome.open_recent_requested.connect(self._on_open_recent)
        self._welcome.set_recent_files(self._config.get_recent()[:5])
        self._welcome.show()
        set_titlebar_dark(self._welcome, self._config.theme == "dark")

    def create_window(self) -> MainWindow:
        """Build MainWindow, create presenters, and wire all signals."""
        assert self._project is not None
        assert self._slice_presenter is not None

        window = MainWindow()
        view = window.view()
        toolbar = window.toolbar()

        last_index: dict[Axis, int] = {
            Axis.INLINE: 0, Axis.CROSSLINE: 0, Axis.TIME: 0
        }

        v2dp = View2DPresenter(
            project=self._project,
            view=view,
            view3d_presenter=None,
            slice_presenter=self._slice_presenter,
            labeling_service=self._labeling_service,
            last_index=last_index,
            toolbar=toolbar,
        )

        mp = MainPresenter(
            qt_app=self._qt_app,
            persistence=self._persistence,
            config=self._config,
            attr_service=self._attr_service,
            window=window,
            view2d_presenter=v2dp,
            project=self._project,
            slice_presenter=self._slice_presenter,
            last_index=last_index,
        )
        self._main_presenter = mp
        window.set_close_guard(mp._check_unsaved_changes)

        view.stroke_point.connect(v2dp.on_stroke_point)
        view.stroke_finished.connect(v2dp.on_stroke_finished)
        view.erase_requested.connect(v2dp.on_erase_at)
        view.stroke_point_removed.connect(v2dp.on_stroke_point_removed)
        view.new_entity_requested.connect(v2dp.on_new_entity)
        view.entity_visibility_changed.connect(v2dp.on_visibility_changed)
        view.entity_selected.connect(v2dp.on_entity_selected)
        view.entity_delete_requested.connect(v2dp.on_entity_delete)
        view.entity_rename_requested.connect(v2dp.on_entity_rename)
        view.thickness_changed.connect(v2dp.on_thickness_changed)
        view.eraser_radius_changed.connect(v2dp.on_eraser_radius_changed)

        view.canvas_hovered.connect(mp._on_canvas_hover)
        view.drawing_mode_changed.connect(mp._on_drawing_mode_changed)

        window.open_seismic_npz_requested.connect(mp._on_open_seismic_npz)
        window.open_seismic_segy_requested.connect(mp._on_open_seismic_segy)
        window.open_svp_requested.connect(mp._on_open_svp)
        window.load_horizons_requested.connect(mp._on_load_horizons)
        window.load_faults_requested.connect(mp._on_load_faults)
        window.save_seismic_requested.connect(mp._on_save_seismic)
        window.save_horizons_requested.connect(mp._on_save_horizons)
        window.save_faults_requested.connect(mp._on_save_faults)
        window.save_svp_requested.connect(mp._on_save_svp)

        window.axis_changed.connect(mp._on_axis_changed)
        window.slice_index_changed.connect(mp._on_slice_changed)
        window.attribute_changed.connect(mp._on_attribute_changed)
        window.contrast_changed.connect(mp._on_contrast_changed)
        window.view_mode_changed.connect(mp._on_view_mode_changed)
        window.theme_changed.connect(mp._on_theme_changed)
        window.colormap_changed.connect(mp._on_colormap_changed)
        window.recent_file_selected.connect(mp._on_open_recent_from_window)
        window.recent_cleared.connect(mp._on_clear_recent)
        window.interpolate_horizons_requested.connect(mp._on_interpolate_horizons)
        window.interpolate_faults_requested.connect(mp._on_interpolate_faults)
        window.auto_track_horizon_requested.connect(mp._on_auto_track_horizon)
        window.auto_track_fault_requested.connect(mp._on_auto_track_fault)

        window.view_3d().plane_moved.connect(mp._on_3d_plane_moved)
        window.view_3d().entity_visibility_changed.connect(mp._on_3d_visibility_changed)
        window.view_3d().plane_visibility_changed.connect(mp._on_3d_plane_visibility_changed)
        window.view_3d().clip_labels_changed.connect(mp._on_clip_labels_changed)

        window.set_theme(self._config.theme)
        window.view_3d().set_theme(self._config.theme)
        window.set_colormap(self._config.colormap)
        window.set_language_checked(self._config.language)
        window.refresh_recent_menu(self._config.get_recent())

        return window


    def _on_language_changed(self, lang: str) -> None:
        self._config.language = lang
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _log.warning("Could not save config after language change: %s", exc)

    def _on_open_seismic_npz_from_welcome(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self._welcome, tr("fdlg.open_seismic_npz"), "", _NPZ_FILTER
        )
        if not path_str:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            ok = self._load_project(Path(path_str), None, None, self._welcome)
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            self._open_main_window_from_welcome()

    def _on_open_seismic_segy_from_welcome(self) -> None:
        npz_path = self._convert_segy_to_npz(self._welcome)
        if npz_path is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            ok = self._load_project(npz_path, None, None, self._welcome)
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            self._open_main_window_from_welcome()

    def _on_open_svp_from_welcome(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self._welcome, tr("fdlg.open_project"), "", _SVP_FILTER
        )
        if not path_str:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            ok = self._load_svp(Path(path_str), self._welcome)
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            self._open_main_window_from_welcome()

    def _on_open_recent(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(
                self._welcome, tr("msg.file_not_found"),
                tr("msg.file_not_found_detail").format(path=path),
            )
            if self._welcome is not None:
                self._welcome.set_recent_files(self._config.get_recent()[:5])
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[attr-defined]
        QApplication.processEvents()
        try:
            ok = (
                self._load_svp(path, self._welcome)
                if path.suffix.lower() == ".svp"
                else self._load_project(path, None, None, self._welcome)
            )
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            self._open_main_window_from_welcome()

    def _open_main_window_from_welcome(self) -> None:
        window = self.create_window()
        window.show()
        set_titlebar_dark(window, self._config.theme == "dark")
        if self._welcome is not None:
            self._welcome.close()
            self._welcome = None
        assert self._main_presenter is not None
        self._main_presenter.apply_loaded_project()


    def _load_project(
        self,
        seismic_path: Path,
        horizons_path: Path | None,
        faults_path: Path | None,
        parent: QWidget | None,
    ) -> bool:
        try:
            self._project = self._persistence.load(seismic_path, horizons_path, faults_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # persistence raises numpy/zip/OS errors
            _log.error("Failed to load project: %s", exc)
            QMessageBox.critical(parent, tr("msg.error"), tr("msg.failed_load_project").format(exc=exc))
            return False
        self._slice_presenter = SlicePresenter(
            self._project, self._attr_service, cmap_name=self._config.colormap
        )
        self._config.add_recent(seismic_path)
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught  # config save may fail with OS/json errors
            _log.warning("Could not save config: %s", exc)
        return True

    def _load_svp(self, path: Path, parent: QWidget | None) -> bool:
        try:
            self._project = self._persistence.load_svp(path)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # persistence raises numpy/zip/OS errors
            _log.error("Failed to load .svp: %s", exc)
            QMessageBox.critical(parent, tr("msg.error"), tr("msg.failed_load_svp").format(exc=exc))
            return False
        self._slice_presenter = SlicePresenter(
            self._project, self._attr_service, cmap_name=self._config.colormap
        )
        self._config.add_recent(path)
        try:
            self._config.save(AppConfig.config_path())
        except Exception as exc:  # pylint: disable=broad-exception-caught  # config save may fail with OS/json errors
            _log.warning("Could not save config: %s", exc)
        return True

    def _convert_segy_to_npz(self, parent: QWidget | None) -> Path | None:
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter
        from seismic_visualizer.infrastructure.io.segy_reader import SegyReader

        segy_str, _ = QFileDialog.getOpenFileName(parent, tr("fdlg.open_segy"), "", _SEGY_FILTER)
        if not segy_str:
            return None
        segy_path = Path(segy_str)
        npz_str, _ = QFileDialog.getSaveFileName(
            parent, tr("fdlg.save_npz_from_segy"),
            str(segy_path.with_suffix(".npz")), _NPZ_FILTER,
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
