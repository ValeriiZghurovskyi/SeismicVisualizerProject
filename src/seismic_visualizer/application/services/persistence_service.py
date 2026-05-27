"""PersistenceService — save and load Project data to/from .npz files or .svp archives."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from seismic_visualizer.application.services.project import Project
from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityKind, EntityRegistry
from seismic_visualizer.infrastructure.io.npz_reader import NpzReader
from seismic_visualizer.infrastructure.io.npz_writer import NpzWriter

_SVP_VERSION = 1
_SVP_SEISMIC = "seismic.npy"
_SVP_HORIZONS = "horizons.npy"
_SVP_FAULTS = "faults.npy"
_SVP_META = "meta.json"
_SVP_REQUIRED_FILES: frozenset[str] = frozenset({_SVP_SEISMIC, _SVP_HORIZONS, _SVP_FAULTS, _SVP_META})


class PersistenceService:
    """Reads and writes project cube data using NpzReader / NpzWriter."""

    def __init__(self) -> None:
        self._reader = NpzReader()
        self._writer = NpzWriter()


    def save_svp(self, project: Project, path: Path) -> None:
        """Save all three cubes + entity metadata to a single .svp archive."""
        def _to_npy_bytes(arr: np.ndarray) -> bytes:
            buf = io.BytesIO()
            np.save(buf, arr)
            return buf.getvalue()

        meta = {
            "version": _SVP_VERSION,
            "horizons": [
                {"id": e.id, "name": e.name, "visible": e.visible}
                for e in project.horizon_registry.all()
            ],
            "faults": [
                {"id": e.id, "name": e.name, "visible": e.visible}
                for e in project.fault_registry.all()
            ],
        }

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(_SVP_SEISMIC, _to_npy_bytes(project.seismic.data))
            zf.writestr(_SVP_HORIZONS, _to_npy_bytes(project.horizons.data))
            zf.writestr(_SVP_FAULTS, _to_npy_bytes(project.faults.data))
            zf.writestr(_SVP_META, json.dumps(meta, indent=2))

        project.project_path = path
        project.is_dirty = False

    def load_svp(self, path: Path) -> Project:
        """Load a project from a .svp archive. Returns a new Project."""
        with zipfile.ZipFile(path, "r") as zf:
            self._validate_svp_archive(zf)
            seismic_arr: np.ndarray = np.load(io.BytesIO(zf.read(_SVP_SEISMIC)))
            horizons_arr: np.ndarray = np.load(io.BytesIO(zf.read(_SVP_HORIZONS)))
            faults_arr: np.ndarray = np.load(io.BytesIO(zf.read(_SVP_FAULTS)))
            meta: dict = json.loads(zf.read(_SVP_META))

        horizon_registry = EntityRegistry()
        for entry in meta.get("horizons", []):
            e = horizon_registry.restore(entry["id"], EntityKind.HORIZON, entry["name"])
            e.visible = entry.get("visible", True)

        fault_registry = EntityRegistry()
        for entry in meta.get("faults", []):
            e = fault_registry.restore(entry["id"], EntityKind.FAULT, entry["name"])
            e.visible = entry.get("visible", True)

        return Project(
            seismic=SeismicCube(seismic_arr),
            horizons=LabelCube(horizons_arr),
            faults=LabelCube(faults_arr),
            horizon_registry=horizon_registry,
            fault_registry=fault_registry,
            seismic_path=None,
            horizons_path=None,
            faults_path=None,
            project_path=path,
            is_dirty=False,
        )


    @staticmethod
    def _validate_svp_archive(zf: zipfile.ZipFile) -> None:
        """Validate a .svp ZIP archive before reading its contents.

        Checks performed:
        1. Path traversal: every entry name must not contain ``..`` path
           components and must not be an absolute path (POSIX ``/`` prefix
           or Windows drive letter such as ``C:``).
        2. Required files: ``seismic.npy``, ``horizons.npy``, ``faults.npy``,
           and ``meta.json`` must all be present.

        Args:
            zf: An already-opened ``zipfile.ZipFile`` in read mode.

        Raises:
            ValueError: If a path traversal attempt is detected (message
                includes the offending entry name), or if one or more required
                files are absent (message lists the missing names).
        """
        names = zf.namelist()

        for name in names:
            forward = name.replace("\\", "/")
            parts = [p for p in forward.split("/") if p]

            if ".." in parts:
                raise ValueError(
                    f"Unsafe archive entry (path traversal detected): {name!r}"
                )

            if forward.startswith("/") or (len(forward) >= 2 and forward[1] == ":"):
                raise ValueError(
                    f"Unsafe archive entry (absolute path): {name!r}"
                )

            normalised = os.path.normpath(forward).replace("\\", "/")
            if normalised != forward.rstrip("/"):
                raise ValueError(
                    f"Unsafe archive entry (non-normalised path): {name!r}"
                )

        present = set(names)
        missing = _SVP_REQUIRED_FILES - present
        if missing:
            raise ValueError(
                "Invalid .svp archive — missing required files: "
                + ", ".join(sorted(missing))
            )


    def save(self, project: Project) -> None:
        """Write all three cubes to their respective paths.

        Raises:
            ValueError: If any cube path on the project is None.
        """
        for attr, label in (
            ("seismic_path", "seismic cube"),
            ("horizons_path", "horizons"),
            ("faults_path", "faults"),
        ):
            if getattr(project, attr) is None:
                raise ValueError(f"No path set for {label}")

        self._writer.write_seismic(project.seismic_path, project.seismic.data)  # type: ignore[arg-type]
        self._writer.write_labels(project.horizons_path, project.horizons.data)  # type: ignore[arg-type]
        self._writer.write_labels(project.faults_path, project.faults.data)  # type: ignore[arg-type]
        project.is_dirty = False

    def save_seismic(self, project: Project) -> None:
        """Write only the seismic cube. Raises ValueError if seismic_path is None."""
        if project.seismic_path is None:
            raise ValueError("No seismic path set")
        self._writer.write_seismic(project.seismic_path, project.seismic.data)

    def save_horizons(self, project: Project) -> None:
        """Write only the horizons cube. Raises ValueError if horizons_path is None."""
        if project.horizons_path is None:
            raise ValueError("No horizons path set")
        self._writer.write_labels(project.horizons_path, project.horizons.data)

    def save_faults(self, project: Project) -> None:
        """Write only the faults cube. Raises ValueError if faults_path is None."""
        if project.faults_path is None:
            raise ValueError("No faults path set")
        self._writer.write_labels(project.faults_path, project.faults.data)


    def load_horizons(self, project: Project, path: Path) -> None:
        """Replace the horizons cube in an existing project from disk.

        Validates that the loaded array shape matches the seismic cube.
        Rebuilds horizon_registry from non-zero voxels.
        """
        arr = self._reader.read_labels(path, reference_shape=project.seismic.data.shape)
        registry = EntityRegistry()
        for eid in np.unique(arr[arr > 0]):
            registry.restore(int(eid), EntityKind.HORIZON, f"Horizon {int(eid)}")
        project.horizons = LabelCube(arr)
        project.horizon_registry = registry
        project.horizons_path = path

    def load_faults(self, project: Project, path: Path) -> None:
        """Replace the faults cube in an existing project from disk.

        Validates that the loaded array shape matches the seismic cube.
        Rebuilds fault_registry from non-zero voxels.
        """
        arr = self._reader.read_labels(path, reference_shape=project.seismic.data.shape)
        registry = EntityRegistry()
        for eid in np.unique(arr[arr > 0]):
            registry.restore(int(eid), EntityKind.FAULT, f"Fault {int(eid)}")
        project.faults = LabelCube(arr)
        project.fault_registry = registry
        project.faults_path = path

    def load(
        self,
        seismic_path: Path,
        horizons_path: Path | None,
        faults_path: Path | None,
    ) -> Project:
        """Load a project from disk.

        Args:
            seismic_path: Path to the seismic .npz file (required).
            horizons_path: Path to the horizons .npz file, or None → empty cube.
            faults_path: Path to the faults .npz file, or None → empty cube.

        Returns:
            New Project with is_dirty=False and empty entity registries.
        """
        seismic_arr = self._reader.read_seismic(seismic_path)
        shape: tuple[int, int, int] = seismic_arr.shape

        if horizons_path is not None:
            horizons_arr = self._reader.read_labels(horizons_path, reference_shape=shape)
        else:
            horizons_arr = np.zeros(shape, dtype=np.uint8)

        if faults_path is not None:
            faults_arr = self._reader.read_labels(faults_path, reference_shape=shape)
        else:
            faults_arr = np.zeros(shape, dtype=np.uint8)

        horizon_registry = EntityRegistry()
        for eid in np.unique(horizons_arr[horizons_arr > 0]):
            horizon_registry.restore(int(eid), EntityKind.HORIZON, f"Horizon {int(eid)}")

        fault_registry = EntityRegistry()
        for eid in np.unique(faults_arr[faults_arr > 0]):
            fault_registry.restore(int(eid), EntityKind.FAULT, f"Fault {int(eid)}")

        return Project(
            seismic=SeismicCube(seismic_arr),
            horizons=LabelCube(horizons_arr),
            faults=LabelCube(faults_arr),
            horizon_registry=horizon_registry,
            fault_registry=fault_registry,
            seismic_path=seismic_path,
            horizons_path=horizons_path,
            faults_path=faults_path,
            is_dirty=False,
        )
