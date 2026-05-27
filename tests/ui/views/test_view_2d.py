"""Smoke tests for View2D — instantiation and protocol method calls."""

from __future__ import annotations

import numpy as np

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.ui.views.view_2d import View2D


def _rgba(h: int = 10, w: int = 20) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def test_view_2d_instantiates(qtbot: object) -> None:
    view = View2D()
    assert view is not None


def test_update_canvas_does_not_raise(qtbot: object) -> None:
    view = View2D()
    view.update_canvas(_rgba(10, 20))


def test_update_entity_list_empty_does_not_raise(qtbot: object) -> None:
    view = View2D()
    view.update_entity_list([])


def test_update_entity_list_with_items_does_not_raise(qtbot: object) -> None:
    view = View2D()
    entities = [
        EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 100, 100)),
        EntityInfo(id=2, name="F1", kind=EntityKind.FAULT, visible=False, color=(255, 200, 0)),
    ]
    view.update_entity_list(entities)


def test_update_status_emits_status_changed(qtbot: object) -> None:
    view = View2D()
    received: list[str] = []
    view.status_changed.connect(received.append)

    view.update_status("Slice 42/499 | Inline")

    assert received == ["Slice 42/499 | Inline"]


