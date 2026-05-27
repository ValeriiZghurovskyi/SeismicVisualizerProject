# Seismic Visualizer

Desktop application for viewing and interpreting 3D seismic data (Python + PyQt5 + PyVista).

## Features

- Load 3D seismic cubes (`.npz` format; `.segy` coming in Milestone 9)
- 2D slice viewer — Inline / Crossline / Time axes, zoom & pan
- 3D visualization with interactive slice planes and random-line plane
- Geological **horizon** and **fault** annotation with polyline drawing
- Seismic attribute computation — RMS, Instantaneous Phase, Frequency, Envelope
- Save/load annotations to `.npz` files
- Light and dark themes (via `qdarkstyle`)

## Requirements

- Python 3.10+
- Windows (primary target); other platforms untested

## Installation

```bash
# Production
pip install -e .

# Development (includes pytest, ruff, mypy)
pip install -e ".[dev]"
```

## Running

```bash
python -m seismic_visualizer
# or after installation:
seismic-visualizer
```

## Development commands

```bash
# Run tests
pytest

# Lint
ruff check src/

# Auto-fix lint issues
ruff check src/ --fix

# Format
ruff format src/

# Type check
mypy src/
```

## Architecture

MVP (Model-View-Presenter) with 4 strict layers:

```
UI (PyQt5 / PyVista)
    ↕  signals + protocol calls
Application (Presenters, Services)
    ↕
Domain (pure Python — no Qt, no PyVista)
    ↕
Infrastructure (I/O, logging, config)
```

See `CLAUDE.md` for the full architecture specification.

## Project structure

```
src/seismic_visualizer/
├── domain/          # Pure Python — business logic
├── infrastructure/  # I/O, logging, config
├── application/     # Presenters, services, view protocols
└── ui/              # PyQt5 widgets, dialogs, 3D view
```
