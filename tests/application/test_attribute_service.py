"""Tests for application.services.attribute_service — AttributeService."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.services.attribute_service import AttributeService
from seismic_visualizer.domain.cube import SeismicCube
from seismic_visualizer.domain.exceptions import AttributeNotApplicableError
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.domain.slice import Slice


@pytest.fixture
def service() -> AttributeService:
    return AttributeService()


@pytest.fixture
def inline_slice(small_seismic: SeismicCube) -> Slice:
    return Slice(small_seismic.data, Axis.INLINE, 0)


@pytest.fixture
def time_slice(small_seismic: SeismicCube) -> Slice:
    return Slice(small_seismic.data, Axis.TIME, 0)


class TestCompute:
    def test_returns_float32(
        self, service: AttributeService, inline_slice: Slice
    ) -> None:
        result = service.compute(inline_slice, "envelope")
        assert result.dtype == np.float32

    def test_shape_matches_slice(
        self, service: AttributeService, inline_slice: Slice
    ) -> None:
        result = service.compute(inline_slice, "envelope")
        assert result.shape == inline_slice.shape

    @pytest.mark.parametrize("attr_name", ["envelope", "phase", "frequency", "rms"])
    def test_all_attributes_return_correct_shape(
        self,
        service: AttributeService,
        small_seismic: SeismicCube,
        attr_name: str,
    ) -> None:
        slc = Slice(small_seismic.data, Axis.INLINE, 0)
        result = service.compute(slc, attr_name)
        assert result.shape == slc.shape
        assert result.dtype == np.float32

    def test_unknown_attribute_raises(
        self, service: AttributeService, inline_slice: Slice
    ) -> None:
        with pytest.raises(ValueError, match="Unknown attribute"):
            service.compute(inline_slice, "nonexistent")

    def test_time_slice_raises(
        self, service: AttributeService, time_slice: Slice
    ) -> None:
        with pytest.raises(AttributeNotApplicableError):
            service.compute(time_slice, "envelope")


class TestCache:
    def test_second_call_returns_same_object(
        self, service: AttributeService, small_seismic: SeismicCube
    ) -> None:
        slc1 = Slice(small_seismic.data, Axis.INLINE, 0)
        slc2 = Slice(small_seismic.data, Axis.INLINE, 0)
        result1 = service.compute(slc1, "rms")
        result2 = service.compute(slc2, "rms")
        assert result1 is result2

    def test_different_index_not_cached(
        self, service: AttributeService, small_seismic: SeismicCube
    ) -> None:
        slc0 = Slice(small_seismic.data, Axis.INLINE, 0)
        slc1 = Slice(small_seismic.data, Axis.INLINE, 1)
        result0 = service.compute(slc0, "envelope")
        result1 = service.compute(slc1, "envelope")
        assert result0 is not result1

    def test_different_attr_not_cached(
        self, service: AttributeService, inline_slice: Slice
    ) -> None:
        r1 = service.compute(inline_slice, "envelope")
        r2 = service.compute(inline_slice, "rms")
        assert r1 is not r2


class TestClear:
    def test_clear_forces_recompute(
        self, service: AttributeService, small_seismic: SeismicCube
    ) -> None:
        slc1 = Slice(small_seismic.data, Axis.INLINE, 0)
        result1 = service.compute(slc1, "envelope")
        service.clear()
        slc2 = Slice(small_seismic.data, Axis.INLINE, 0)
        result2 = service.compute(slc2, "envelope")
        assert result1 is not result2

    def test_clear_empties_cache(
        self, service: AttributeService, inline_slice: Slice
    ) -> None:
        service.compute(inline_slice, "rms")
        assert len(service._cache) == 1
        service.clear()
        assert len(service._cache) == 0
