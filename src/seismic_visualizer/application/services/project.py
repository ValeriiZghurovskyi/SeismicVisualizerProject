"""Project dataclass — root of all open data (seismic + label cubes + registries).

See CLAUDE.md §4.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from seismic_visualizer.domain.cube import LabelCube, SeismicCube
from seismic_visualizer.domain.entities import EntityRegistry


@dataclass
class Project:
    """All data associated with an open seismic interpretation session.

    Holds the three cubes, their registries, the file paths they were loaded
    from (None for unsaved projects), and a dirty flag that tracks unsaved
    label changes.

    Args:
        seismic: Read-only seismic amplitude cube.
        horizons: Mutable horizon label cube.
        faults: Mutable fault label cube.
        horizon_registry: Registry of all horizon entities.
        fault_registry: Registry of all fault entities.
        seismic_path: Path to the seismic .npz file, or None if not yet saved.
        horizons_path: Path to the horizons .npz file, or None if not yet saved.
        faults_path: Path to the faults .npz file, or None if not yet saved.
        is_dirty: True when label cubes have unsaved changes.
    """

    seismic: SeismicCube
    horizons: LabelCube
    faults: LabelCube
    horizon_registry: EntityRegistry
    fault_registry: EntityRegistry
    seismic_path: Path | None
    horizons_path: Path | None
    faults_path: Path | None
    project_path: Path | None = field(default=None)
    is_dirty: bool = field(default=False)
