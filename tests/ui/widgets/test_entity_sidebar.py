"""Tests for EntitySidebar — mode selector, entity list, signals."""

from __future__ import annotations

import pytest
from PyQt5.QtCore import Qt

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.ui.widgets.entity_sidebar import EntitySidebar


def _info(entity_id: int, kind: EntityKind = EntityKind.HORIZON, visible: bool = True) -> EntityInfo:
    return EntityInfo(id=entity_id, name=f"Entity {entity_id}", kind=kind, visible=visible, color=(255, 0, 0))


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

def test_default_mode_is_horizon(qtbot: object) -> None:
    sidebar = EntitySidebar()
    assert sidebar.current_mode() is EntityKind.HORIZON


def test_default_list_is_empty(qtbot: object) -> None:
    sidebar = EntitySidebar()
    assert sidebar._list.count() == 0


# ---------------------------------------------------------------------------
# Mode radio
# ---------------------------------------------------------------------------

def test_fault_radio_emits_mode_changed_fault(qtbot: object) -> None:
    sidebar = EntitySidebar()
    received: list[EntityKind] = []
    sidebar.mode_changed.connect(received.append)

    sidebar._radio_fault.setChecked(True)

    assert EntityKind.FAULT in received


def test_horizon_radio_emits_mode_changed_horizon(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar._radio_fault.setChecked(True)
    received: list[EntityKind] = []
    sidebar.mode_changed.connect(received.append)

    sidebar._radio_horizon.setChecked(True)

    assert EntityKind.HORIZON in received


def test_current_mode_returns_fault_when_fault_selected(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar._radio_fault.setChecked(True)
    assert sidebar.current_mode() is EntityKind.FAULT


# ---------------------------------------------------------------------------
# update_entities
# ---------------------------------------------------------------------------

def test_update_entities_populates_list(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar.update_entities([_info(1), _info(2)])
    assert sidebar._list.count() == 2


def test_update_entities_empty_clears_list(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar.update_entities([_info(1), _info(2)])
    sidebar.update_entities([])
    assert sidebar._list.count() == 0


def test_update_entities_checked_state_matches_visible(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar.update_entities([_info(1, visible=True), _info(2, visible=False)])
    assert sidebar._list.item(0).checkState() == Qt.Checked
    assert sidebar._list.item(1).checkState() == Qt.Unchecked


# ---------------------------------------------------------------------------
# "+ New" button
# ---------------------------------------------------------------------------

def test_new_button_emits_current_mode_horizon(qtbot: object) -> None:
    sidebar = EntitySidebar()
    received: list[EntityKind] = []
    sidebar.new_entity_requested.connect(received.append)

    sidebar._new_btn.click()

    assert received == [EntityKind.HORIZON]


def test_new_button_emits_current_mode_fault(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar._radio_fault.setChecked(True)
    received: list[EntityKind] = []
    sidebar.new_entity_requested.connect(received.append)

    sidebar._new_btn.click()

    assert received == [EntityKind.FAULT]


# ---------------------------------------------------------------------------
# Checkbox toggle → entity_visibility_changed
# ---------------------------------------------------------------------------

def test_checkbox_uncheck_emits_visibility_false(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar.update_entities([_info(7, visible=True)])
    received: list[tuple] = []
    sidebar.entity_visibility_changed.connect(lambda i, k, v: received.append((i, v)))

    sidebar._list.item(0).setCheckState(Qt.Unchecked)

    assert (7, False) in received


def test_checkbox_check_emits_visibility_true(qtbot: object) -> None:
    sidebar = EntitySidebar()
    sidebar.update_entities([_info(3, visible=False)])
    received: list[tuple] = []
    sidebar.entity_visibility_changed.connect(lambda i, k, v: received.append((i, v)))

    sidebar._list.item(0).setCheckState(Qt.Checked)

    assert (3, True) in received
