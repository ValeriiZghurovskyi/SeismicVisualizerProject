"""Smoke tests — Milestone 0 health check.

These tests verify only that:
- Python 3.10+ is in use
- The package and all its sub-packages can be imported without errors
- The entry point module exists
"""

import importlib
import sys


def test_python_version() -> None:
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_package_importable() -> None:
    import seismic_visualizer  # noqa: F401

    assert seismic_visualizer.__version__ == "0.1.0"


def test_main_module_exists() -> None:
    spec = importlib.util.find_spec("seismic_visualizer.__main__")
    assert spec is not None, "__main__.py not found"


def test_domain_subpackages_importable() -> None:
    packages = [
        "seismic_visualizer.domain",
        "seismic_visualizer.domain.labeling",
        "seismic_visualizer.domain.algorithms",
        "seismic_visualizer.domain.attributes",
        "seismic_visualizer.application.rendering",
    ]
    for pkg in packages:
        importlib.import_module(pkg)


def test_infrastructure_subpackages_importable() -> None:
    packages = [
        "seismic_visualizer.infrastructure",
        "seismic_visualizer.infrastructure.io",
    ]
    for pkg in packages:
        importlib.import_module(pkg)


def test_application_subpackages_importable() -> None:
    packages = [
        "seismic_visualizer.application",
        "seismic_visualizer.application.presenters",
        "seismic_visualizer.application.services",
    ]
    for pkg in packages:
        importlib.import_module(pkg)
