"""Tests for application-layer DTOs."""

from __future__ import annotations

import pytest

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.domain.entities import EntityKind


def test_entity_info_creates_with_horizon_kind() -> None:
    info = EntityInfo(id=1, name="Horizon 1", kind=EntityKind.HORIZON, visible=True, color=(255, 100, 100))
    assert info.id == 1
    assert info.name == "Horizon 1"
    assert info.kind is EntityKind.HORIZON
    assert info.visible is True
    assert info.color == (255, 100, 100)


def test_entity_info_creates_with_fault_kind() -> None:
    info = EntityInfo(id=2, name="Fault A", kind=EntityKind.FAULT, visible=False, color=(255, 200, 0))
    assert info.kind is EntityKind.FAULT
    assert info.visible is False


def test_entity_info_is_frozen() -> None:
    info = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(0, 0, 0))
    with pytest.raises((AttributeError, TypeError)):
        info.id = 99  # type: ignore[misc]
