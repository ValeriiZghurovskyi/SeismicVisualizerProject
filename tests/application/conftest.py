"""Shared fixtures for application-layer tests."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityRegistry
from seismic_visualizer.application.services.project import Project

_SHAPE: tuple[int, int, int] = (5, 5, 10)


@pytest.fixture
def small_seismic() -> SeismicCube:
    data = np.arange(5 * 5 * 10, dtype=np.uint8).reshape(_SHAPE)
    return SeismicCube(data)


@pytest.fixture
def small_horizons() -> LabelCube:
    return LabelCube.empty(_SHAPE)


@pytest.fixture
def small_faults() -> LabelCube:
    return LabelCube.empty(_SHAPE)


@pytest.fixture
def project(
    small_seismic: SeismicCube,
    small_horizons: LabelCube,
    small_faults: LabelCube,
) -> Project:
    return Project(
        seismic=small_seismic,
        horizons=small_horizons,
        faults=small_faults,
        horizon_registry=EntityRegistry(),
        fault_registry=EntityRegistry(),
        seismic_path=None,
        horizons_path=None,
        faults_path=None,
        is_dirty=False,
    )
