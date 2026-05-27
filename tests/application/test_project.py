"""Tests for application.services.project — Project dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityRegistry


class TestProjectCreation:
    def test_created_from_fixture(self, project: Project) -> None:
        assert isinstance(project, Project)

    def test_is_dirty_default_false(self, project: Project) -> None:
        assert project.is_dirty is False

    def test_paths_default_none(self, project: Project) -> None:
        assert project.seismic_path is None
        assert project.horizons_path is None
        assert project.faults_path is None

    def test_is_dirty_can_be_set_true(self, project: Project) -> None:
        project.is_dirty = True
        assert project.is_dirty is True


class TestProjectReferences:
    def test_seismic_is_same_object(
        self, project: Project, small_seismic: SeismicCube
    ) -> None:
        assert project.seismic is small_seismic

    def test_horizons_is_same_object(
        self, project: Project, small_horizons: LabelCube
    ) -> None:
        assert project.horizons is small_horizons

    def test_faults_is_same_object(
        self, project: Project, small_faults: LabelCube
    ) -> None:
        assert project.faults is small_faults

    def test_horizon_registry_is_same_object(self, project: Project) -> None:
        assert isinstance(project.horizon_registry, EntityRegistry)

    def test_label_mutation_visible_through_project(
        self, project: Project, small_horizons: LabelCube
    ) -> None:
        small_horizons.data[0, 0, 0] = 42
        assert project.horizons.data[0, 0, 0] == 42


class TestProjectShape:
    def test_seismic_shape(self, project: Project) -> None:
        assert project.seismic.shape == (5, 5, 10)

    def test_horizons_shape_matches_seismic(self, project: Project) -> None:
        assert project.horizons.shape == project.seismic.shape

    def test_faults_shape_matches_seismic(self, project: Project) -> None:
        assert project.faults.shape == project.seismic.shape
