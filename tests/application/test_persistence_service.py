"""Tests for application.services.persistence_service — PersistenceService."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from seismic_visualizer.application.services.persistence_service import PersistenceService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityRegistry


@pytest.fixture
def service() -> PersistenceService:
    return PersistenceService()


@pytest.fixture
def saved_project(
    tmp_path: Path,
    small_seismic: SeismicCube,
    small_horizons: LabelCube,
    small_faults: LabelCube,
) -> Project:
    """Project with all paths set, ready for save/load tests."""
    return Project(
        seismic=small_seismic,
        horizons=small_horizons,
        faults=small_faults,
        horizon_registry=EntityRegistry(),
        fault_registry=EntityRegistry(),
        seismic_path=tmp_path / "test_seismic.npz",
        horizons_path=tmp_path / "test_horizons.npz",
        faults_path=tmp_path / "test_faults.npz",
        is_dirty=True,
    )


class TestSave:
    def test_seismic_file_exists_after_save(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        assert saved_project.seismic_path.exists()  # type: ignore[union-attr]

    def test_horizons_file_exists_after_save(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        assert saved_project.horizons_path.exists()  # type: ignore[union-attr]

    def test_faults_file_exists_after_save(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        assert saved_project.faults_path.exists()  # type: ignore[union-attr]

    def test_save_clears_is_dirty(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        assert saved_project.is_dirty is True
        service.save(saved_project)
        assert saved_project.is_dirty is False

    def test_save_seismic_path_none_raises(
        self,
        service: PersistenceService,
        saved_project: Project,
    ) -> None:
        saved_project.seismic_path = None
        with pytest.raises(ValueError, match="seismic cube"):
            service.save(saved_project)

    def test_save_horizons_path_none_raises(
        self,
        service: PersistenceService,
        saved_project: Project,
    ) -> None:
        saved_project.horizons_path = None
        with pytest.raises(ValueError, match="horizons"):
            service.save(saved_project)

    def test_save_faults_path_none_raises(
        self,
        service: PersistenceService,
        saved_project: Project,
    ) -> None:
        saved_project.faults_path = None
        with pytest.raises(ValueError, match="faults"):
            service.save(saved_project)


class TestLoad:
    def test_load_is_dirty_false(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        loaded = service.load(
            saved_project.seismic_path,  # type: ignore[arg-type]
            saved_project.horizons_path,
            saved_project.faults_path,
        )
        assert loaded.is_dirty is False

    def test_load_seismic_data_matches(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        loaded = service.load(
            saved_project.seismic_path,  # type: ignore[arg-type]
            saved_project.horizons_path,
            saved_project.faults_path,
        )
        np.testing.assert_array_equal(
            loaded.seismic.data, saved_project.seismic.data
        )

    def test_load_horizons_data_matches(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        loaded = service.load(
            saved_project.seismic_path,  # type: ignore[arg-type]
            saved_project.horizons_path,
            saved_project.faults_path,
        )
        np.testing.assert_array_equal(
            loaded.horizons.data, saved_project.horizons.data
        )

    def test_load_faults_data_matches(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        loaded = service.load(
            saved_project.seismic_path,  # type: ignore[arg-type]
            saved_project.horizons_path,
            saved_project.faults_path,
        )
        np.testing.assert_array_equal(
            loaded.faults.data, saved_project.faults.data
        )

    def test_load_without_label_paths_creates_empty_cubes(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        loaded = service.load(
            saved_project.seismic_path,  # type: ignore[arg-type]
            horizons_path=None,
            faults_path=None,
        )
        assert np.all(loaded.horizons.data == 0)
        assert np.all(loaded.faults.data == 0)

    def test_load_paths_stored_on_project(
        self, service: PersistenceService, saved_project: Project
    ) -> None:
        service.save(saved_project)
        loaded = service.load(
            saved_project.seismic_path,  # type: ignore[arg-type]
            saved_project.horizons_path,
            saved_project.faults_path,
        )
        assert loaded.seismic_path == saved_project.seismic_path


# ---------------------------------------------------------------------------
# Helpers for SVP validation tests
# ---------------------------------------------------------------------------

def _make_svp(tmp_path: Path, *, extra_names: list[str] | None = None, skip: str | None = None) -> Path:
    """Build a minimal valid .svp archive, optionally injecting extra entries or omitting one required file."""
    arr = np.zeros((2, 2, 2), dtype=np.uint8)

    def _npy_bytes(a: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, a)
        return buf.getvalue()

    required = {
        "seismic.npy": _npy_bytes(arr),
        "horizons.npy": _npy_bytes(arr),
        "faults.npy": _npy_bytes(arr),
        "meta.json": b'{"version": 1, "horizons": [], "faults": []}',
    }
    if skip:
        required.pop(skip, None)

    svp = tmp_path / "test.svp"
    with zipfile.ZipFile(svp, "w") as zf:
        for name, data in required.items():
            zf.writestr(name, data)
        for name in (extra_names or []):
            zf.writestr(name, b"x")
    return svp


class TestValidateSvpArchive:
    def test_valid_archive_loads_without_error(self, tmp_path: Path) -> None:
        svp = _make_svp(tmp_path)
        service = PersistenceService()
        project = service.load_svp(svp)
        assert project is not None

    def test_missing_meta_json_raises(self, tmp_path: Path) -> None:
        svp = _make_svp(tmp_path, skip="meta.json")
        service = PersistenceService()
        with pytest.raises(ValueError, match="meta.json"):
            service.load_svp(svp)

    def test_missing_seismic_raises(self, tmp_path: Path) -> None:
        svp = _make_svp(tmp_path, skip="seismic.npy")
        service = PersistenceService()
        with pytest.raises(ValueError, match="seismic.npy"):
            service.load_svp(svp)

    def test_path_traversal_dotdot_raises(self, tmp_path: Path) -> None:
        svp = _make_svp(tmp_path, extra_names=["../etc/passwd"])
        service = PersistenceService()
        with pytest.raises(ValueError, match="path traversal"):
            service.load_svp(svp)

    def test_path_traversal_subdir_dotdot_raises(self, tmp_path: Path) -> None:
        svp = _make_svp(tmp_path, extra_names=["subdir/../../escape.txt"])
        service = PersistenceService()
        with pytest.raises(ValueError, match="path traversal"):
            service.load_svp(svp)

    def test_absolute_posix_path_raises(self, tmp_path: Path) -> None:
        svp = _make_svp(tmp_path, extra_names=["/abs/path"])
        service = PersistenceService()
        with pytest.raises(ValueError, match="absolute path"):
            service.load_svp(svp)

    def test_absolute_windows_path_raises(self, tmp_path: Path) -> None:
        svp = _make_svp(tmp_path, extra_names=["C:/windows/system32/evil.dll"])
        service = PersistenceService()
        with pytest.raises(ValueError, match="absolute path"):
            service.load_svp(svp)


# ---------------------------------------------------------------------------
# save_svp / load_svp roundtrip
# ---------------------------------------------------------------------------


class TestSaveSvp:
    def test_save_svp_creates_file(self, tmp_path: Path, project: Project) -> None:
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        assert svp.exists()

    def test_save_svp_clears_dirty_flag(self, tmp_path: Path, project: Project) -> None:
        project.is_dirty = True
        svc = PersistenceService()
        svc.save_svp(project, tmp_path / "out.svp")
        assert project.is_dirty is False

    def test_save_svp_sets_project_path(self, tmp_path: Path, project: Project) -> None:
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        assert project.project_path == svp

    def test_save_svp_roundtrip_seismic(
        self, tmp_path: Path, project: Project
    ) -> None:
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        loaded = svc.load_svp(svp)
        np.testing.assert_array_equal(loaded.seismic.data, project.seismic.data)

    def test_save_svp_roundtrip_horizons(
        self, tmp_path: Path, project: Project
    ) -> None:
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        loaded = svc.load_svp(svp)
        np.testing.assert_array_equal(loaded.horizons.data, project.horizons.data)

    def test_save_svp_roundtrip_faults(
        self, tmp_path: Path, project: Project
    ) -> None:
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        loaded = svc.load_svp(svp)
        np.testing.assert_array_equal(loaded.faults.data, project.faults.data)

    def test_save_svp_preserves_entity_names(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.domain.entities import EntityKind
        project.horizon_registry.create_new(EntityKind.HORIZON, "MyHorizon")
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        loaded = svc.load_svp(svp)
        names = [e.name for e in loaded.horizon_registry.all()]
        assert "MyHorizon" in names

    def test_save_svp_preserves_entity_visibility(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.domain.entities import EntityKind
        e = project.horizon_registry.create_new(EntityKind.HORIZON, "H1")
        e.visible = False
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        loaded = svc.load_svp(svp)
        h = loaded.horizon_registry.all()[0]
        assert h.visible is False

    def test_load_svp_restores_fault_entities(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.domain.entities import EntityKind
        project.fault_registry.create_new(EntityKind.FAULT, "MyFault")
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        loaded = svc.load_svp(svp)
        names = [e.name for e in loaded.fault_registry.all()]
        assert "MyFault" in names

    def test_load_svp_restores_fault_visibility(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.domain.entities import EntityKind
        e = project.fault_registry.create_new(EntityKind.FAULT, "F1")
        e.visible = False
        svc = PersistenceService()
        svp = tmp_path / "out.svp"
        svc.save_svp(project, svp)
        loaded = svc.load_svp(svp)
        f = loaded.fault_registry.all()[0]
        assert f.visible is False


# ---------------------------------------------------------------------------
# save_seismic / save_horizons / save_faults
# ---------------------------------------------------------------------------


class TestSavePartial:
    def test_save_seismic_creates_file(
        self, tmp_path: Path, project: Project
    ) -> None:
        project.seismic_path = tmp_path / "seis.npz"
        svc = PersistenceService()
        svc.save_seismic(project)
        assert project.seismic_path.exists()

    def test_save_seismic_raises_when_path_none(self, project: Project) -> None:
        with pytest.raises(ValueError, match="seismic"):
            PersistenceService().save_seismic(project)

    def test_save_horizons_creates_file(
        self, tmp_path: Path, project: Project
    ) -> None:
        project.horizons_path = tmp_path / "hor.npz"
        svc = PersistenceService()
        svc.save_horizons(project)
        assert project.horizons_path.exists()

    def test_save_horizons_raises_when_path_none(self, project: Project) -> None:
        with pytest.raises(ValueError, match="horizons"):
            PersistenceService().save_horizons(project)

    def test_save_faults_creates_file(
        self, tmp_path: Path, project: Project
    ) -> None:
        project.faults_path = tmp_path / "flt.npz"
        svc = PersistenceService()
        svc.save_faults(project)
        assert project.faults_path.exists()

    def test_save_faults_raises_when_path_none(self, project: Project) -> None:
        with pytest.raises(ValueError, match="faults"):
            PersistenceService().save_faults(project)


# ---------------------------------------------------------------------------
# load_horizons / load_faults
# ---------------------------------------------------------------------------


class TestLoadPartial:
    def test_load_horizons_replaces_cube(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.domain.entities import EntityKind
        project.seismic_path = tmp_path / "seis.npz"
        project.horizons_path = tmp_path / "hor.npz"
        project.faults_path = tmp_path / "flt.npz"
        svc = PersistenceService()
        svc.save(project)

        # create a separate horizons file with a labeled voxel
        labeled = np.zeros(project.seismic.data.shape, dtype=np.uint8)
        labeled[0, 0, 0] = 1
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter
        hor_path = tmp_path / "new_hor.npz"
        NpzWriter().write_labels(hor_path, labeled)

        svc.load_horizons(project, hor_path)
        assert project.horizons.data[0, 0, 0] == 1
        assert project.horizons_path == hor_path

    def test_load_horizons_rebuilds_registry(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.domain.entities import EntityKind
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter

        labeled = np.zeros(project.seismic.data.shape, dtype=np.uint8)
        labeled[1, 1, 1] = 2
        hor_path = tmp_path / "hor.npz"
        NpzWriter().write_labels(hor_path, labeled)

        svc = PersistenceService()
        svc.load_horizons(project, hor_path)

        ids = [e.id for e in project.horizon_registry.all()]
        assert 2 in ids

    def test_load_faults_replaces_cube(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter

        labeled = np.zeros(project.seismic.data.shape, dtype=np.uint8)
        labeled[2, 2, 2] = 3
        flt_path = tmp_path / "flt.npz"
        NpzWriter().write_labels(flt_path, labeled)

        svc = PersistenceService()
        svc.load_faults(project, flt_path)
        assert project.faults.data[2, 2, 2] == 3
        assert project.faults_path == flt_path

    def test_load_faults_rebuilds_registry(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter

        labeled = np.zeros(project.seismic.data.shape, dtype=np.uint8)
        labeled[0, 1, 2] = 5
        flt_path = tmp_path / "flt.npz"
        NpzWriter().write_labels(flt_path, labeled)

        svc = PersistenceService()
        svc.load_faults(project, flt_path)
        ids = [e.id for e in project.fault_registry.all()]
        assert 5 in ids


# ---------------------------------------------------------------------------
# load with labeled data → registry.restore
# ---------------------------------------------------------------------------


class TestLoadWithLabeledData:
    def test_load_restores_horizon_registry_from_labeled_cube(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter

        seis_path = tmp_path / "seis.npz"
        hor_path = tmp_path / "hor.npz"
        flt_path = tmp_path / "flt.npz"

        seismic = project.seismic.data
        labeled_hor = np.zeros_like(seismic)
        labeled_hor[0, 0, 0] = 1
        labeled_hor[1, 1, 1] = 2

        NpzWriter().write_seismic(seis_path, seismic)
        NpzWriter().write_labels(hor_path, labeled_hor)
        NpzWriter().write_labels(flt_path, np.zeros_like(seismic))

        svc = PersistenceService()
        loaded = svc.load(seis_path, hor_path, flt_path)
        ids = {e.id for e in loaded.horizon_registry.all()}
        assert {1, 2} <= ids

    def test_load_restores_fault_registry_from_labeled_cube(
        self, tmp_path: Path, project: Project
    ) -> None:
        from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter

        seis_path = tmp_path / "seis.npz"
        hor_path = tmp_path / "hor.npz"
        flt_path = tmp_path / "flt.npz"

        seismic = project.seismic.data
        labeled_flt = np.zeros_like(seismic)
        labeled_flt[0, 0, 0] = 3

        NpzWriter().write_seismic(seis_path, seismic)
        NpzWriter().write_labels(hor_path, np.zeros_like(seismic))
        NpzWriter().write_labels(flt_path, labeled_flt)

        svc = PersistenceService()
        loaded = svc.load(seis_path, hor_path, flt_path)
        ids = {e.id for e in loaded.fault_registry.all()}
        assert 3 in ids
