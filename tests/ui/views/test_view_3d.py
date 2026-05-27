"""Smoke tests for View3D — instantiation and protocol method calls."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.dto import EntityInfo
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.view_interfaces import LabelMesh
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityKind, EntityRegistry
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.ui.views.view_3d import View3D

_SHAPE: tuple[int, int, int] = (5, 6, 10)


@pytest.fixture
def small_project() -> Project:
    data = np.arange(5 * 6 * 10, dtype=np.uint8).reshape(_SHAPE)
    return Project(
        seismic=SeismicCube(data),
        horizons=LabelCube.empty(_SHAPE),
        faults=LabelCube.empty(_SHAPE),
        horizon_registry=EntityRegistry(),
        fault_registry=EntityRegistry(),
        seismic_path=None,
        horizons_path=None,
        faults_path=None,
    )


@pytest.fixture
def view(qtbot: object) -> View3D:
    w = View3D()
    return w


def test_view_3d_instantiates(view: View3D) -> None:
    assert view is not None


def test_update_seismic_planes_does_not_raise(
    view: View3D, small_project: Project
) -> None:
    indices = {Axis.INLINE: 0, Axis.CROSSLINE: 0, Axis.TIME: 0}
    view.update_seismic_planes(small_project, indices)


def test_set_plane_position_without_project_does_not_raise(view: View3D) -> None:
    # no project loaded yet — should silently do nothing
    view.set_plane_position(Axis.INLINE, 0)


def test_set_plane_position_after_load_does_not_raise(
    view: View3D, small_project: Project
) -> None:
    view.update_seismic_planes(
        small_project, {Axis.INLINE: 0, Axis.CROSSLINE: 0, Axis.TIME: 0}
    )
    view.set_plane_position(Axis.INLINE, 2)


def test_update_label_meshes_empty_does_not_raise(view: View3D) -> None:
    view.update_label_meshes([])


def test_update_label_meshes_with_coords_does_not_raise(view: View3D) -> None:
    coords = np.array([[0, 0, 0], [1, 2, 3], [2, 3, 5]], dtype=np.int32)
    info = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 0, 0))
    meshes: list[LabelMesh] = [(coords, info)]
    view.update_label_meshes(meshes)


def test_update_entity_list_empty_does_not_raise(view: View3D) -> None:
    view.update_entity_list([])


def test_update_entity_list_with_items_does_not_raise(view: View3D) -> None:
    entities = [
        EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 0, 0)),
    ]
    view.update_entity_list(entities)


def test_update_seismic_planes_registers_all_three_axes(
    view: View3D, small_project: Project
) -> None:
    view.update_seismic_planes(
        small_project, {Axis.INLINE: 0, Axis.CROSSLINE: 0, Axis.TIME: 0}
    )
    assert set(view._plane_actors.keys()) == {Axis.INLINE, Axis.CROSSLINE, Axis.TIME}


def test_set_plane_position_replaces_actor(
    view: View3D, small_project: Project
) -> None:
    view.update_seismic_planes(
        small_project, {Axis.INLINE: 0, Axis.CROSSLINE: 0, Axis.TIME: 0}
    )
    actor_before = view._plane_actors[Axis.INLINE]
    view.set_plane_position(Axis.INLINE, 2)
    actor_after = view._plane_actors[Axis.INLINE]
    # actor object must be refreshed (new mesh at new position)
    assert actor_after is not actor_before
