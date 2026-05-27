"""EntitySidebar — list of horizons/faults with visibility toggles and color markers."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.ui.i18n import LanguageManager, tr

_SWATCH_SIZE: int = 14
_ENTITY_ID_ROLE: int = Qt.UserRole  # type: ignore[attr-defined]
_ENTITY_KIND_ROLE: int = Qt.UserRole + 1  # type: ignore[attr-defined]


def _make_color_icon(rgb: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(_SWATCH_SIZE, _SWATCH_SIZE)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


class EntitySidebar(QWidget):
    """Side panel with mode selector and entity list."""

    mode_changed = pyqtSignal(EntityKind)
    entity_visibility_changed = pyqtSignal(int, object, bool)
    new_entity_requested = pyqtSignal(EntityKind)
    entity_selected = pyqtSignal(int, object)
    entity_delete_requested = pyqtSignal(int, object)
    entity_rename_requested = pyqtSignal(int, object, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._selected_key: tuple[int, EntityKind] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._radio_horizon, self._radio_fault, self._mode_group = (
            self._build_mode_selector(layout)
        )
        self._list = self._build_entity_list(layout)
        self._new_btn = self._build_new_button(layout)

        LanguageManager.instance().language_changed.connect(lambda _: self.retranslate_ui())


    def update_entities(self, entities: list[EntityInfo]) -> None:
        """Rebuild the list from a fresh snapshot, preserving selection highlight.

        Args:
            entities: Ordered list of entity display data.
        """
        self._list.blockSignals(True)
        self._list.clear()
        for info in entities:
            item = QListWidgetItem(_make_color_icon(info.color), info.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # type: ignore[attr-defined]
            item.setCheckState(
                Qt.Checked if info.visible else Qt.Unchecked  # type: ignore[attr-defined]
            )
            item.setData(_ENTITY_ID_ROLE, info.id)
            item.setData(_ENTITY_KIND_ROLE, info.kind)
            self._list.addItem(item)

        if self._selected_key is not None:
            sel_id, sel_kind = self._selected_key
            for i in range(self._list.count()):
                it = self._list.item(i)
                if (
                    it
                    and it.data(_ENTITY_ID_ROLE) == sel_id
                    and it.data(_ENTITY_KIND_ROLE) == sel_kind
                ):
                    self._list.setCurrentItem(it)
                    break

        self._list.blockSignals(False)

    def set_view_only(self, view_only: bool) -> None:
        """Hide mode selector and New button (use in contexts where editing is not allowed)."""
        self._radio_horizon.setVisible(not view_only)
        self._radio_fault.setVisible(not view_only)
        self._new_btn.setVisible(not view_only)

    def current_mode(self) -> EntityKind:
        """Return the currently selected entity kind."""
        return EntityKind.FAULT if self._radio_fault.isChecked() else EntityKind.HORIZON

    def retranslate_ui(self) -> None:
        self._radio_horizon.setText(tr("sidebar.horizons"))
        self._radio_fault.setText(tr("sidebar.faults"))
        self._new_btn.setText(tr("sidebar.new"))


    def _build_mode_selector(
        self, parent_layout: QVBoxLayout
    ) -> tuple[QRadioButton, QRadioButton, QButtonGroup]:
        row = QHBoxLayout()
        horizon_radio = QRadioButton(tr("sidebar.horizons"))
        fault_radio = QRadioButton(tr("sidebar.faults"))
        horizon_radio.setChecked(True)

        group = QButtonGroup(self)
        group.addButton(horizon_radio)
        group.addButton(fault_radio)

        row.addWidget(horizon_radio)
        row.addWidget(fault_radio)
        parent_layout.addLayout(row)

        horizon_radio.toggled.connect(
            lambda checked: self.mode_changed.emit(EntityKind.HORIZON) if checked else None
        )
        fault_radio.toggled.connect(
            lambda checked: self.mode_changed.emit(EntityKind.FAULT) if checked else None
        )

        return horizon_radio, fault_radio, group

    def _build_entity_list(self, parent_layout: QVBoxLayout) -> QListWidget:
        lst = QListWidget()
        lst.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        lst.itemChanged.connect(self._on_item_changed)
        lst.itemClicked.connect(self._on_item_clicked)
        lst.customContextMenuRequested.connect(self._on_context_menu)
        parent_layout.addWidget(lst)
        return lst

    def _build_new_button(self, parent_layout: QVBoxLayout) -> QPushButton:
        btn = QPushButton(tr("sidebar.new"))
        btn.clicked.connect(lambda: self.new_entity_requested.emit(self.current_mode()))
        parent_layout.addWidget(btn)
        return btn


    def _on_item_changed(self, item: QListWidgetItem) -> None:
        entity_id: int = item.data(_ENTITY_ID_ROLE)
        kind: EntityKind = item.data(_ENTITY_KIND_ROLE)
        visible = item.checkState() == Qt.Checked  # type: ignore[attr-defined]
        self.entity_visibility_changed.emit(entity_id, kind, visible)

    def _on_context_menu(self, pos: object) -> None:
        from PyQt5.QtCore import QPoint as _QPoint
        p: _QPoint = pos  # type: ignore[assignment]
        item = self._list.itemAt(p)
        if item is None:
            return
        entity_id: int = item.data(_ENTITY_ID_ROLE)
        kind: EntityKind = item.data(_ENTITY_KIND_ROLE)

        menu = QMenu(self)
        rename_action = menu.addAction(tr("sidebar.rename"))
        delete_action = menu.addAction(tr("sidebar.delete"))
        action = menu.exec_(self._list.viewport().mapToGlobal(p))

        if action == rename_action:
            new_name, ok = QInputDialog.getText(
                self, tr("sidebar.rename_title"), tr("sidebar.rename_label"), text=item.text()
            )
            if ok and new_name.strip():
                self.entity_rename_requested.emit(entity_id, kind, new_name.strip())
        elif action == delete_action:
            self.entity_delete_requested.emit(entity_id, kind)

    def _on_item_clicked(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        entity_id: int = item.data(_ENTITY_ID_ROLE)
        kind: EntityKind = item.data(_ENTITY_KIND_ROLE)
        self._selected_key = (entity_id, kind)

        self._radio_horizon.blockSignals(True)
        self._radio_fault.blockSignals(True)
        self._radio_horizon.setChecked(kind == EntityKind.HORIZON)
        self._radio_fault.setChecked(kind == EntityKind.FAULT)
        self._radio_horizon.blockSignals(False)
        self._radio_fault.blockSignals(False)

        self.entity_selected.emit(entity_id, kind)
