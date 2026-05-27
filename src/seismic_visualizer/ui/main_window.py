"""MainWindow — top-level window. Menu bar + slim toolbar + central view + bottom slice panel."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.dialogs.contrast_dialog import ContrastDialog
from seismic_visualizer.ui.i18n import LanguageManager, tr
from seismic_visualizer.ui.views.view_2d import View2D
from seismic_visualizer.ui.views.view_3d import View3D
from seismic_visualizer.ui.widgets.hud import HUD
from seismic_visualizer.ui.widgets.multi_axis_slider_panel import MultiAxisSliderPanel
from seismic_visualizer.ui.widgets.slice_control_panel import SliceControlPanel
from seismic_visualizer.ui.widgets.toolbar import SeismicToolbar

_ATTRIBUTES: list[str] = ["None", "envelope", "phase", "frequency", "rms"]
_COLORMAPS: list[tuple[str, str]] = [
    ("Seismic (RWB)", "seismic"),
    ("RdBu", "RdBu"),
    ("Gray", "gray"),
    ("Viridis", "viridis"),
    ("Coolwarm", "coolwarm"),
]


class MainWindow(QMainWindow):
    """Menu bar + toolbar + stacked views + bottom slice navigation panel.

    All signals are forwarded upward to App; no business logic here.
    """

    open_seismic_npz_requested = pyqtSignal()
    open_seismic_segy_requested = pyqtSignal()
    open_svp_requested = pyqtSignal()
    load_horizons_requested = pyqtSignal()
    load_faults_requested = pyqtSignal()
    save_seismic_requested = pyqtSignal()
    save_horizons_requested = pyqtSignal()
    save_faults_requested = pyqtSignal()
    save_svp_requested = pyqtSignal()

    axis_changed = pyqtSignal(Axis)
    slice_index_changed = pyqtSignal(Axis, int)
    attribute_changed = pyqtSignal(str)
    contrast_changed = pyqtSignal(int, int)
    view_mode_changed = pyqtSignal(str)
    theme_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    recent_file_selected = pyqtSignal(Path)
    recent_cleared = pyqtSignal()
    interpolate_horizons_requested = pyqtSignal()
    interpolate_faults_requested = pyqtSignal()
    auto_track_horizon_requested = pyqtSignal()
    auto_track_fault_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Seismic Visualizer")
        self.resize(1280, 720)

        self._contrast_dlg: ContrastDialog | None = None
        self._current_vmin: int = 0
        self._current_vmax: int = 255
        self._current_axis: Axis = Axis.INLINE
        self._theme_actions: dict[str, QAction] = {}
        self._colormap_actions: dict[str, QAction] = {}
        self._lang_actions: dict[str, QAction] = {}
        self._recent_menu: QMenu | None = None
        self._recent_paths: list[Path] = []
        self._close_guard: Callable[[], bool] | None = None

        self._file_menu: QMenu | None = None
        self._view_menu: QMenu | None = None
        self._attr_menu: QMenu | None = None
        self._theme_menu: QMenu | None = None
        self._cmap_menu: QMenu | None = None
        self._lang_menu: QMenu | None = None
        self._interp_menu: QMenu | None = None
        self._tools_menu: QMenu | None = None
        self._help_menu: QMenu | None = None

        self._act_open_npz: QAction | None = None
        self._act_open_segy: QAction | None = None
        self._act_open_svp: QAction | None = None
        self._act_load_horizons: QAction | None = None
        self._act_load_faults: QAction | None = None
        self._act_save_seismic: QAction | None = None
        self._act_save_horizons: QAction | None = None
        self._act_save_faults: QAction | None = None
        self._act_save_svp: QAction | None = None
        self._act_exit: QAction | None = None
        self._act_contrast: QAction | None = None
        self._act_theme_light: QAction | None = None
        self._act_theme_dark: QAction | None = None
        self._act_interp_horizons: QAction | None = None
        self._act_interp_faults: QAction | None = None
        self._act_track_horizon: QAction | None = None
        self._act_track_fault: QAction | None = None
        self._act_about: QAction | None = None

        self._build_menubar()

        self._toolbar = SeismicToolbar(self)
        self.addToolBar(Qt.TopToolBarArea, self._toolbar)  # type: ignore[arg-type]

        self._view = View2D()
        self._view_3d = View3D()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._view)
        self._stack.addWidget(self._view_3d)

        self._slice_panel = SliceControlPanel()
        self._multi_panel = MultiAxisSliderPanel()

        self._hud = HUD()
        self.statusBar().addPermanentWidget(self._hud)
        self.statusBar().setSizeGripEnabled(False)

        self._bottom_panel = QStackedWidget()
        self._bottom_panel.addWidget(self._slice_panel)
        self._bottom_panel.addWidget(self._multi_panel)
        self._bottom_panel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed  # type: ignore[arg-type]
        )

        self._handle_strip = QPushButton("▾")
        self._handle_strip.setFlat(True)
        self._handle_strip.setFixedSize(120, 20)
        self._handle_strip.setCursor(Qt.PointingHandCursor)  # type: ignore[arg-type]
        self._handle_strip.setToolTip(tr("slice.hide_panel"))
        from PyQt5.QtGui import QFont
        f = QFont()
        f.setPointSize(12)
        self._handle_strip.setFont(f)
        self._handle_strip.clicked.connect(self._toggle_slice_panel)

        handle_container = QWidget()
        handle_container.setFixedHeight(20)
        handle_container.setStyleSheet("")
        handle_layout = QHBoxLayout(handle_container)
        handle_layout.setContentsMargins(0, 0, 0, 0)
        handle_layout.setSpacing(0)
        handle_layout.addStretch()
        handle_layout.addWidget(self._handle_strip)
        handle_layout.addStretch()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stack, stretch=1)
        layout.addWidget(handle_container)
        layout.addWidget(self._bottom_panel)
        self._bottom_panel.setFixedHeight(self._slice_panel.sizeHint().height())
        self.setCentralWidget(central)

        self._connect_signals()
        LanguageManager.instance().language_changed.connect(lambda _: self.retranslate_ui())


    def set_theme(self, theme: str) -> None:
        """Check the correct Theme menu item without emitting theme_changed."""
        act = self._theme_actions.get(theme)
        if act is not None:
            act.setChecked(True)

    def set_colormap(self, name: str) -> None:
        """Check the correct Colormap menu item without emitting colormap_changed."""
        act = self._colormap_actions.get(name)
        if act is not None:
            act.setChecked(True)

    def set_language_checked(self, lang: str) -> None:
        """Check the correct Language menu item without triggering set_language."""
        act = self._lang_actions.get(lang)
        if act is not None:
            act.setChecked(True)

    def refresh_recent_menu(self, paths: list[Path]) -> None:
        """Rebuild the Recent Files submenu from the given path list."""
        self._recent_paths = list(paths)
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        if not paths:
            no_act = self._recent_menu.addAction(tr("menu.recent_empty"))
            no_act.setEnabled(False)
        else:
            for p in paths:
                act = self._recent_menu.addAction(str(p))
                act.triggered.connect(lambda checked, path=p: self.recent_file_selected.emit(path))
            self._recent_menu.addSeparator()
            clear_act = self._recent_menu.addAction(tr("menu.clear_recent"))
            clear_act.triggered.connect(self.recent_cleared)

    def toolbar(self) -> SeismicToolbar:
        return self._toolbar

    def view(self) -> View2D:
        return self._view

    def view_3d(self) -> View3D:
        return self._view_3d

    def switch_to_2d(self) -> None:
        self._stack.setCurrentIndex(0)
        self._bottom_panel.setCurrentIndex(0)
        self._bottom_panel.setFixedHeight(self._slice_panel.sizeHint().height())
        self._hud.show()

    def switch_to_3d(self) -> None:
        self._stack.setCurrentIndex(1)
        self._bottom_panel.setCurrentIndex(1)
        self._bottom_panel.setFixedHeight(self._multi_panel.sizeHint().height())
        self._hud.hide()

    def update_slice_range(self, axis: Axis, max_index: int, value: int) -> None:
        """Update the 2D slice panel for the given axis."""
        self._current_axis = axis
        self._slice_panel.set_range(axis, max_index, value)

    def update_all_slice_ranges(self, ranges: dict[Axis, tuple[int, int]]) -> None:
        """Update all three axes in the 3D multi-panel. ranges: {axis: (max_index, value)}"""
        for axis, (max_index, value) in ranges.items():
            self._multi_panel.set_range(axis, max_index, value)

    def hud(self) -> HUD:
        return self._hud

    def set_slice_panel_value(self, axis: Axis, value: int) -> None:
        """Silently update slider positions in both panels (no signal emitted)."""
        self._multi_panel.set_value(axis, value)
        if axis == self._current_axis:
            self._slice_panel.set_value(value)

    def set_close_guard(self, guard: Callable[[], bool]) -> None:
        """Register a callable invoked on close; returning False cancels it."""
        self._close_guard = guard

    def retranslate_ui(self) -> None:
        """Update all translatable strings in the menu bar and window chrome."""
        if self._file_menu:
            self._file_menu.setTitle(tr("menu.file"))
        if self._view_menu:
            self._view_menu.setTitle(tr("menu.view"))
        if self._attr_menu:
            self._attr_menu.setTitle(tr("menu.attributes"))
        if self._theme_menu:
            self._theme_menu.setTitle(tr("menu.theme"))
        if self._cmap_menu:
            self._cmap_menu.setTitle(tr("menu.colormap"))
        if self._lang_menu:
            self._lang_menu.setTitle(tr("menu.language"))
        if self._interp_menu:
            self._interp_menu.setTitle(tr("menu.interpolation"))
        if self._tools_menu:
            self._tools_menu.setTitle(tr("menu.tools"))
        if self._help_menu:
            self._help_menu.setTitle(tr("menu.help"))
        if self._recent_menu:
            self._recent_menu.setTitle(tr("menu.recent_files"))

        _safe_set(self._act_open_npz, tr("menu.open_npz"))
        _safe_set(self._act_open_segy, tr("menu.open_segy"))
        _safe_set(self._act_open_svp, tr("menu.open_svp"))
        _safe_set(self._act_load_horizons, tr("menu.load_horizons"))
        _safe_set(self._act_load_faults, tr("menu.load_faults"))
        _safe_set(self._act_save_seismic, tr("menu.save_seismic"))
        _safe_set(self._act_save_horizons, tr("menu.save_horizons"))
        _safe_set(self._act_save_faults, tr("menu.save_faults"))
        _safe_set(self._act_save_svp, tr("menu.save_svp"))
        _safe_set(self._act_exit, tr("menu.exit"))
        _safe_set(self._act_contrast, tr("menu.contrast"))
        _safe_set(self._act_theme_light, tr("menu.theme_light"))
        _safe_set(self._act_theme_dark, tr("menu.theme_dark"))
        _safe_set(self._act_interp_horizons, tr("menu.interp_horizons"))
        _safe_set(self._act_interp_faults, tr("menu.interp_faults"))
        _safe_set(self._act_track_horizon, tr("menu.track_horizon"))
        _safe_set(self._act_track_fault, tr("menu.track_fault"))
        _safe_set(self._act_about, tr("menu.about"))

        self.refresh_recent_menu(self._recent_paths)

        visible = self._bottom_panel.isVisible()
        self._handle_strip.setToolTip(
            tr("slice.hide_panel") if visible else tr("slice.show_panel")
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if self._close_guard is not None and not self._close_guard():
            event.ignore()
            return
        super().closeEvent(event)


    def _build_menubar(self) -> None:
        menubar: QMenuBar = self.menuBar()

        self._file_menu = menubar.addMenu(tr("menu.file"))

        self._act_open_npz = self._file_menu.addAction(tr("menu.open_npz"))
        self._act_open_npz.triggered.connect(self.open_seismic_npz_requested)

        self._act_open_segy = self._file_menu.addAction(tr("menu.open_segy"))
        self._act_open_segy.triggered.connect(self.open_seismic_segy_requested)

        self._act_open_svp = self._file_menu.addAction(tr("menu.open_svp"))
        self._act_open_svp.triggered.connect(self.open_svp_requested)

        self._recent_menu = QMenu(tr("menu.recent_files"), self)
        self._file_menu.addMenu(self._recent_menu)

        self._file_menu.addSeparator()

        self._act_load_horizons = self._file_menu.addAction(tr("menu.load_horizons"))
        self._act_load_horizons.triggered.connect(self.load_horizons_requested)

        self._act_load_faults = self._file_menu.addAction(tr("menu.load_faults"))
        self._act_load_faults.triggered.connect(self.load_faults_requested)

        self._file_menu.addSeparator()

        self._act_save_seismic = self._file_menu.addAction(tr("menu.save_seismic"))
        self._act_save_seismic.triggered.connect(self.save_seismic_requested)

        self._act_save_horizons = self._file_menu.addAction(tr("menu.save_horizons"))
        self._act_save_horizons.triggered.connect(self.save_horizons_requested)

        self._act_save_faults = self._file_menu.addAction(tr("menu.save_faults"))
        self._act_save_faults.triggered.connect(self.save_faults_requested)

        self._act_save_svp = self._file_menu.addAction(tr("menu.save_svp"))
        self._act_save_svp.triggered.connect(self.save_svp_requested)

        self._file_menu.addSeparator()
        self._act_exit = self._file_menu.addAction(tr("menu.exit"))
        self._act_exit.triggered.connect(self.close)

        self._view_menu = menubar.addMenu(tr("menu.view"))

        self._attr_menu = self._view_menu.addMenu(tr("menu.attributes"))
        attr_group = QActionGroup(self)
        attr_group.setExclusive(True)
        for name in _ATTRIBUTES:
            act: QAction = self._attr_menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name == "None")
            attr_group.addAction(act)
            act.triggered.connect(
                lambda checked, n=name: self.attribute_changed.emit("" if n == "None" else n)
            )

        self._view_menu.addSeparator()

        self._act_contrast = self._view_menu.addAction(tr("menu.contrast"))
        self._act_contrast.triggered.connect(self._open_contrast_dialog)

        self._view_menu.addSeparator()

        self._theme_menu = self._view_menu.addMenu(tr("menu.theme"))
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for label_key, key in [("menu.theme_light", "light"), ("menu.theme_dark", "dark")]:
            act_t: QAction = self._theme_menu.addAction(tr(label_key))
            act_t.setCheckable(True)
            act_t.setData(key)
            theme_group.addAction(act_t)
            self._theme_actions[key] = act_t
            if label_key == "menu.theme_light":
                self._act_theme_light = act_t
            else:
                self._act_theme_dark = act_t
        theme_group.triggered.connect(lambda a: self.theme_changed.emit(a.data()))

        self._cmap_menu = self._view_menu.addMenu(tr("menu.colormap"))
        cmap_group = QActionGroup(self)
        cmap_group.setExclusive(True)
        for label, key in _COLORMAPS:
            act_c = self._cmap_menu.addAction(label)
            act_c.setCheckable(True)
            act_c.setData(key)
            cmap_group.addAction(act_c)
            self._colormap_actions[key] = act_c
        self._colormap_actions["seismic"].setChecked(True)
        cmap_group.triggered.connect(lambda a: self.colormap_changed.emit(a.data()))

        self._lang_menu = self._view_menu.addMenu(tr("menu.language"))
        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)
        for label, key in [("English", "en"), ("Українська", "uk")]:
            act_l: QAction = self._lang_menu.addAction(label)
            act_l.setCheckable(True)
            act_l.setData(key)
            lang_group.addAction(act_l)
            self._lang_actions[key] = act_l
        self._lang_actions["en"].setChecked(True)
        lang_group.triggered.connect(lambda a: LanguageManager.instance().set_language(a.data()))

        self._interp_menu = menubar.addMenu(tr("menu.interpolation"))
        self._act_interp_horizons = self._interp_menu.addAction(tr("menu.interp_horizons"))
        self._act_interp_horizons.triggered.connect(self.interpolate_horizons_requested)
        self._act_interp_faults = self._interp_menu.addAction(tr("menu.interp_faults"))
        self._act_interp_faults.triggered.connect(self.interpolate_faults_requested)

        self._tools_menu = menubar.addMenu(tr("menu.tools"))
        self._act_track_horizon = self._tools_menu.addAction(tr("menu.track_horizon"))
        self._act_track_horizon.triggered.connect(self.auto_track_horizon_requested)
        self._act_track_fault = self._tools_menu.addAction(tr("menu.track_fault"))
        self._act_track_fault.triggered.connect(self.auto_track_fault_requested)

        self._help_menu = menubar.addMenu(tr("menu.help"))
        self._act_about = self._help_menu.addAction(tr("menu.about"))


    def _open_contrast_dialog(self) -> None:
        if self._contrast_dlg is None:
            self._contrast_dlg = ContrastDialog(
                self, self._current_vmin, self._current_vmax
            )
            self._contrast_dlg.contrast_changed.connect(self._on_contrast)
        self._contrast_dlg.show()
        self._contrast_dlg.raise_()
        self._contrast_dlg.activateWindow()

    def _on_contrast(self, vmin: int, vmax: int) -> None:
        self._current_vmin = vmin
        self._current_vmax = vmax
        self.contrast_changed.emit(vmin, vmax)


    def _connect_signals(self) -> None:
        self._toolbar.axis_changed.connect(self._on_axis_changed_internal)
        self._toolbar.axis_changed.connect(self.axis_changed)
        self._toolbar.view_mode_changed.connect(self.view_mode_changed)
        self._slice_panel.index_changed.connect(
            lambda v: self.slice_index_changed.emit(self._current_axis, v)
        )
        self._multi_panel.axis_index_changed.connect(self.slice_index_changed)
        self._view.status_changed.connect(self.statusBar().showMessage)

    def _on_axis_changed_internal(self, axis: Axis) -> None:
        self._current_axis = axis

    def _toggle_slice_panel(self) -> None:
        visible = not self._bottom_panel.isVisible()
        self._bottom_panel.setVisible(visible)
        self._handle_strip.setText("▴" if not visible else "▾")
        self._handle_strip.setToolTip(
            tr("slice.show_panel") if not visible else tr("slice.hide_panel")
        )


def _safe_set(action: QAction | None, text: str) -> None:
    if action is not None:
        action.setText(text)
