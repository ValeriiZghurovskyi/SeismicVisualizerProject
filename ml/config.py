"""Training configuration for seismic U-Net."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DataConfig:
    cubes_dir: Path = Path("data/cubes")
    label_type: Literal["horizons", "faults"] = "horizons"
    tile_size: int = 256
    val_fraction: float = 0.2
    seed: int = 42
    min_labels_per_slice: int = 10
    min_tile_labels: int = 50
    n_seeds: int = 1
    seed_sigma: float = 12.0
    # Axes to iterate when building the slice index.
    # Both horizons and faults: (0, 1) — inline + crossline vertical sections.
    axes: tuple[int, ...] = (0, 1)


@dataclass
class ModelConfig:
    in_channels: int = 2  # seismic + hint
    base_filters: int = 32
    depth: int = 4
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 5e-4
    warmup_epochs: int = 3
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    checkpoint_dir: Path = Path("ml/checkpoints")
    log_every: int = 10
    save_samples_every: int = 1
    early_stop_patience: int = 8


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
