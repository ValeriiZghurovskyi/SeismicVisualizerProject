"""Tests for application.presenters.slice_presenter — SlicePresenter."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.presenters.slice_presenter import SlicePresenter
from seismic_visualizer.application.services.attribute_service import AttributeService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.state import SliceIndex
from seismic_visualizer.domain.geometry import Axis


@pytest.fixture
def attr_service() -> AttributeService:
    return AttributeService()


@pytest.fixture
def presenter(project: Project, attr_service: AttributeService) -> SlicePresenter:
    return SlicePresenter(project=project, attr_service=attr_service)


class TestGetRgba:
    def test_returns_uint8(self, presenter: SlicePresenter) -> None:
        rgba = presenter.get_rgba(SliceIndex(Axis.INLINE, 0))
        assert rgba.dtype == np.uint8

    def test_shape_inline_slice(self, presenter: SlicePresenter) -> None:
        # inline slice (5,5,10) → після транспонування (time=10, crossline=5, 4)
        rgba = presenter.get_rgba(SliceIndex(Axis.INLINE, 0))
        assert rgba.shape == (10, 5, 4)

    def test_shape_crossline_slice(self, presenter: SlicePresenter) -> None:
        # crossline slice → (time=10, inline=5, 4)
        rgba = presenter.get_rgba(SliceIndex(Axis.CROSSLINE, 0))
        assert rgba.shape == (10, 5, 4)

    def test_shape_time_slice(self, presenter: SlicePresenter) -> None:
        # time slice → (inline=5, crossline=5, 4)
        rgba = presenter.get_rgba(SliceIndex(Axis.TIME, 0))
        assert rgba.shape == (5, 5, 4)

    def test_no_attribute_does_not_raise(self, presenter: SlicePresenter) -> None:
        assert presenter.state.current_attribute is None
        presenter.get_rgba(SliceIndex(Axis.INLINE, 2))  # must not raise

    def test_all_valid_indices_inline(self, presenter: SlicePresenter) -> None:
        for i in range(5):  # inline axis size = 5
            presenter.get_rgba(SliceIndex(Axis.INLINE, i))


class TestNavigate:
    def test_navigate_updates_current_slice(self, presenter: SlicePresenter) -> None:
        target = SliceIndex(Axis.CROSSLINE, 3)
        presenter.navigate(target)
        assert presenter.state.current_slice == target

    def test_navigate_changes_axis(self, presenter: SlicePresenter) -> None:
        presenter.navigate(SliceIndex(Axis.TIME, 5))
        assert presenter.state.current_slice.axis == Axis.TIME

    def test_navigate_clears_attribute_cache(
        self, presenter: SlicePresenter, attr_service: AttributeService
    ) -> None:
        # pre-populate cache to verify clear() is called
        attr_service._cache[(Axis.INLINE, 0, "envelope")] = np.zeros((5, 10))
        assert len(attr_service._cache) == 1

        presenter.navigate(SliceIndex(Axis.INLINE, 1))

        assert len(attr_service._cache) == 0

    def test_initial_state_is_inline_0(self, presenter: SlicePresenter) -> None:
        assert presenter.state.current_slice == SliceIndex(Axis.INLINE, 0)
