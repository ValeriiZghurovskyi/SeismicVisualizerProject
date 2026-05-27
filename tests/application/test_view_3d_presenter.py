"""Tests for application.presenters.view_3d_presenter — View3DPresenter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seismic_visualizer.application.dto import EntityInfo, RandomLinePlaneDTO
from seismic_visualizer.application.presenters.view_3d_presenter import View3DPresenter
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.view_interfaces import LabelMesh
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.geometry import Axis


# ---------------------------------------------------------------------------
# Mock view
# ---------------------------------------------------------------------------

class MockView3D:
    """Duck-typed stub implementing View3DProtocol."""

    def __init__(self) -> None:
        self.seismic_planes_calls: list[tuple[Project, dict[Axis, int]]] = []
        self.plane_position_calls: list[tuple[Axis, int]] = []
        self.label_meshes_calls: list[list[LabelMesh]] = []
        self.entity_list_calls: list[list[EntityInfo]] = []
        self.random_line_calls: list[RandomLinePlaneDTO] = []
        self.plane_visible_calls: list[tuple[Axis, bool]] = []
        self.thumbnail_calls: list[tuple[Axis, np.ndarray]] = []
        self.entity_visible_calls: list[tuple[int, EntityKind, bool]] = []

    def update_seismic_planes(
        self, project: Project, slice_indices: dict[Axis, int]
    ) -> None:
        self.seismic_planes_calls.append((project, slice_indices))

    def set_plane_position(self, axis: Axis, index: int) -> None:
        self.plane_position_calls.append((axis, index))

    def update_label_meshes(self, meshes: list[LabelMesh]) -> None:
        self.label_meshes_calls.append(meshes)

    def update_entity_list(self, entities: list[EntityInfo]) -> None:
        self.entity_list_calls.append(entities)

    def show_random_line_plane(self, dto: RandomLinePlaneDTO) -> None:
        self.random_line_calls.append(dto)

    def set_plane_visible(self, axis: Axis, visible: bool) -> None:
        self.plane_visible_calls.append((axis, visible))

    def update_slice_thumbnail(self, axis: Axis, rgba: np.ndarray) -> None:
        self.thumbnail_calls.append((axis, rgba))

    def set_entity_visible(self, entity_id: int, kind: EntityKind, visible: bool) -> None:
        self.entity_visible_calls.append((entity_id, kind, visible))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def view() -> MockView3D:
    return MockView3D()


@pytest.fixture
def presenter(project: Project, view: MockView3D) -> View3DPresenter:
    return View3DPresenter(project=project, view=view)


# ---------------------------------------------------------------------------
# on_plane_position_changed
# ---------------------------------------------------------------------------

class TestOnPlanePositionChanged:
    def test_calls_set_plane_position(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.on_plane_position_changed(Axis.INLINE, 5)
        assert view.plane_position_calls == [(Axis.INLINE, 5)]

    def test_updates_stored_index(self, presenter: View3DPresenter) -> None:
        presenter.on_plane_position_changed(Axis.CROSSLINE, 3)
        assert presenter._slice_indices[Axis.CROSSLINE] == 3

    def test_each_axis_independent(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.on_plane_position_changed(Axis.INLINE, 1)
        presenter.on_plane_position_changed(Axis.TIME, 9)
        assert presenter._slice_indices[Axis.INLINE] == 1
        assert presenter._slice_indices[Axis.TIME] == 9
        assert presenter._slice_indices[Axis.CROSSLINE] == 0


# ---------------------------------------------------------------------------
# refresh_planes
# ---------------------------------------------------------------------------

class TestRefreshPlanes:
    def test_calls_update_seismic_planes(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        presenter.refresh_planes()
        assert len(view.seismic_planes_calls) == 1
        called_project, _ = view.seismic_planes_calls[0]
        assert called_project is project

    def test_passes_current_slice_indices(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.on_plane_position_changed(Axis.INLINE, 2)
        presenter.refresh_planes()
        _, indices = view.seismic_planes_calls[0]
        assert indices[Axis.INLINE] == 2

    def test_passes_copy_not_reference(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.refresh_planes()
        _, indices = view.seismic_planes_calls[0]
        presenter.on_plane_position_changed(Axis.INLINE, 4)
        # snapshot passed to the view must not change retroactively
        assert indices[Axis.INLINE] == 0


# ---------------------------------------------------------------------------
# refresh_labels
# ---------------------------------------------------------------------------

class TestRefreshLabels:
    def test_calls_update_label_meshes(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.refresh_labels([])
        assert len(view.label_meshes_calls) == 1

    def test_invisible_entity_included_with_visible_flag(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        # Hidden entities are still passed to the view so the view can keep
        # their actors and flip visibility without geometry rebuild.
        project.horizons.data[0, 0, 0] = 1
        entity = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=False, color=(255, 0, 0))
        presenter.refresh_labels([entity])
        meshes = view.label_meshes_calls[0]
        assert len(meshes) == 1
        _, info = meshes[0]
        assert info.visible is False

    def test_visible_entity_with_data_included(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        project.horizons.data[0, 0, 0] = 1
        project.horizons.data[1, 2, 3] = 1
        entity = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 0, 0))
        presenter.refresh_labels([entity])
        meshes = view.label_meshes_calls[0]
        assert len(meshes) == 1

    def test_coords_have_shape_n_by_3(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        project.horizons.data[0, 0, 0] = 1
        project.horizons.data[1, 2, 3] = 1
        entity = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 0, 0))
        presenter.refresh_labels([entity])
        coords, _ = view.label_meshes_calls[0][0]
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert coords.shape[0] == 2  # two labeled voxels

    def test_clip_to_slices_keeps_only_voxels_on_any_plane(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        # inline=0 slice: voxels at (0,0,0) and (0,1,2)
        # crossline=0 slice: voxel at (1,0,0) — different inline but same crossline
        # voxel at (1,2,3): on no current plane → excluded
        project.horizons.data[0, 0, 0] = 1
        project.horizons.data[0, 1, 2] = 1
        project.horizons.data[1, 0, 0] = 1
        project.horizons.data[1, 2, 3] = 1
        entity = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 0, 0))
        presenter.on_plane_position_changed(Axis.INLINE, 0)
        presenter.on_plane_position_changed(Axis.CROSSLINE, 0)
        presenter.on_plane_position_changed(Axis.TIME, 0)
        presenter.refresh_labels([entity], clip_to_slices=True)
        coords, _ = view.label_meshes_calls[0][0]
        # (0,0,0): inline=0 ✓; (0,1,2): inline=0 ✓; (1,0,0): crossline=0 ✓; (1,2,3): none ✗
        assert coords.shape[0] == 3

    def test_clip_to_slices_false_returns_all_voxels(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        project.horizons.data[0, 0, 0] = 1
        project.horizons.data[1, 2, 3] = 1
        entity = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 0, 0))
        presenter.refresh_labels([entity], clip_to_slices=False)
        coords, _ = view.label_meshes_calls[0][0]
        assert coords.shape[0] == 2

    def test_clip_to_slices_no_match_excludes_mesh(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        # voxel at (4,4,4): no plane passes through it when all indices=0
        project.horizons.data[4, 4, 4] = 1
        entity = EntityInfo(id=1, name="H1", kind=EntityKind.HORIZON, visible=True, color=(255, 0, 0))
        presenter.on_plane_position_changed(Axis.INLINE, 0)
        presenter.on_plane_position_changed(Axis.CROSSLINE, 0)
        presenter.on_plane_position_changed(Axis.TIME, 0)
        presenter.refresh_labels([entity], clip_to_slices=True)
        assert view.label_meshes_calls[0] == []


# ---------------------------------------------------------------------------
# compute_random_line
# ---------------------------------------------------------------------------

class TestComputeRandomLine:
    def test_calls_show_random_line_plane(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.compute_random_line((0, 0), (4, 4))
        assert len(view.random_line_calls) == 1

    def test_dto_normal_is_unit_vector(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.compute_random_line((0, 0), (4, 4))
        dto = view.random_line_calls[0]
        nx, ny, nz = dto.normal
        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
        assert math.isclose(magnitude, 1.0, abs_tol=1e-9)

    def test_dto_normal_is_unit_vector_axis_aligned(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        # purely inline direction → normal points along crossline
        presenter.compute_random_line((0, 0), (4, 0))
        dto = view.random_line_calls[0]
        nx, ny, nz = dto.normal
        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
        assert math.isclose(magnitude, 1.0, abs_tol=1e-9)

    def test_dto_width_equals_distance(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.compute_random_line((0, 0), (3, 4))
        dto = view.random_line_calls[0]
        assert math.isclose(dto.width, 5.0, abs_tol=1e-9)

    def test_dto_height_equals_time_dimension(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        presenter.compute_random_line((0, 0), (4, 4))
        dto = view.random_line_calls[0]
        assert dto.height == float(project.seismic.shape[Axis.TIME])

    def test_dto_origin_is_midpoint(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        presenter.compute_random_line((2, 4), (6, 8))
        dto = view.random_line_calls[0]
        assert math.isclose(dto.origin[0], 4.0, abs_tol=1e-9)  # inline mid
        assert math.isclose(dto.origin[1], 6.0, abs_tol=1e-9)  # crossline mid
        assert math.isclose(dto.origin[2], project.seismic.shape[Axis.TIME] / 2.0, abs_tol=1e-9)

    def test_zero_length_does_not_call_view(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.compute_random_line((3, 3), (3, 3))
        assert len(view.random_line_calls) == 0

    def test_coincident_at_origin_does_not_raise(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.compute_random_line((0, 0), (0, 0))
        assert len(view.random_line_calls) == 0

    def test_diagonal_width_approx_4_sqrt2(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.compute_random_line((0, 0), (4, 4))
        dto = view.random_line_calls[0]
        assert math.isclose(dto.width, 4 * math.sqrt(2), rel_tol=1e-9)

    def test_height_equals_shape_index_2(
        self, presenter: View3DPresenter, view: MockView3D, project: Project
    ) -> None:
        presenter.compute_random_line((0, 0), (4, 4))
        dto = view.random_line_calls[0]
        assert dto.height == float(project.seismic.shape[2])


# ---------------------------------------------------------------------------
# set_plane_visible
# ---------------------------------------------------------------------------

class TestSetPlaneVisible:
    def test_delegates_to_view(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.set_plane_visible(Axis.INLINE, False)
        assert view.plane_visible_calls == [(Axis.INLINE, False)]

    def test_updates_stored_state(self, presenter: View3DPresenter) -> None:
        presenter.set_plane_visible(Axis.CROSSLINE, False)
        assert presenter._plane_visible[Axis.CROSSLINE] is False

    def test_default_all_visible(self, presenter: View3DPresenter) -> None:
        for axis in Axis:
            assert presenter._plane_visible[axis] is True

    def test_toggle_back_to_visible(
        self, presenter: View3DPresenter, view: MockView3D
    ) -> None:
        presenter.set_plane_visible(Axis.TIME, False)
        presenter.set_plane_visible(Axis.TIME, True)
        assert presenter._plane_visible[Axis.TIME] is True
        assert view.plane_visible_calls[-1] == (Axis.TIME, True)

    def test_each_axis_independent(self, presenter: View3DPresenter) -> None:
        presenter.set_plane_visible(Axis.INLINE, False)
        assert presenter._plane_visible[Axis.CROSSLINE] is True
        assert presenter._plane_visible[Axis.TIME] is True
