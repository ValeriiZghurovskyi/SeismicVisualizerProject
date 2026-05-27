"""Label overlay renderer — composites horizon/fault masks onto a seismic slice image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seismic_visualizer.application.rendering.colormap import apply_seismic_colormap
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.slice import Slice

_HORIZON_PALETTE: list[tuple[int, int, int]] = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 180, 255),
    (255, 220, 80),
    (220, 100, 255),
    (100, 255, 220),
    (255, 160, 60),
    (180, 255, 100),
]

_FAULT_PALETTE: list[tuple[int, int, int]] = [
    (255, 200, 0),
    (255, 120, 0),
    (200, 50, 255),
    (0, 220, 255),
    (255, 50, 150),
    (150, 255, 50),
    (255, 50, 50),
    (50, 150, 255),
]


@dataclass
class RenderOptions:
    horizon_alpha: float = 0.6
    fault_alpha: float = 0.7


class SliceRenderer:
    """Composites seismic amplitude + label masks into a single RGBA image.

    Usage:
        renderer = SliceRenderer()
        rgba = renderer.render(seismic_slice, horizon_slice, fault_slice)
    """

    def render(
        self,
        seismic: Slice,
        horizons: Slice,
        faults: Slice,
        options: RenderOptions | None = None,
    ) -> np.ndarray:
        """Render seismic slice with horizon and fault overlays.

        Args:
            seismic: Seismic amplitude slice (float).
            horizons: Horizon label slice (uint8, 0 = empty, 1..200 = entity ID).
            faults: Fault label slice (uint8, 0 = empty, 1..200 = entity ID).
            options: Alpha blend settings; uses defaults if None.

        Returns:
            uint8 RGBA array of shape (rows, cols, 4).
        """
        opts = options or RenderOptions()
        canvas = apply_seismic_colormap(seismic.data).astype(np.float32)

        self._blend_labels(canvas, horizons.data, opts.horizon_alpha, kind="horizon")
        self._blend_labels(canvas, faults.data, opts.fault_alpha, kind="fault")

        return np.clip(canvas, 0, 255).astype(np.uint8)


    def _blend_labels(
        self,
        canvas: np.ndarray,
        label_data: np.ndarray,
        alpha: float,
        kind: str,
    ) -> None:
        mask = label_data != 0
        if not np.any(mask):
            return

        entity_ids: np.ndarray = np.unique(label_data[mask])
        for eid in entity_ids:
            pixel_mask = label_data == eid
            color = self._color_for(int(eid), kind)
            for ch, val in enumerate(color):
                canvas[:, :, ch] = np.where(
                    pixel_mask,
                    canvas[:, :, ch] * (1 - alpha) + val * alpha,
                    canvas[:, :, ch],
                )

    @staticmethod
    def _color_for(entity_id: int, kind: str) -> tuple[int, int, int]:
        if kind == "fault":
            return _FAULT_PALETTE[(entity_id - 1) % len(_FAULT_PALETTE)]
        return _HORIZON_PALETTE[(entity_id - 1) % len(_HORIZON_PALETTE)]


def entity_color(entity_id: int, kind: EntityKind) -> tuple[int, int, int]:
    """Return the RGB display color for an entity.

    Args:
        entity_id: Entity identifier (1..200; 0 is also safe due to Python modulo).
        kind: Horizon or fault.

    Returns:
        RGB tuple with each channel in [0, 255].
    """
    return SliceRenderer._color_for(entity_id, kind.value)


def overlay_labels(
    canvas: np.ndarray,
    horizons_data: np.ndarray,
    faults_data: np.ndarray,
    options: RenderOptions | None = None,
) -> np.ndarray:
    """Blend horizon and fault label masks onto a pre-colormapped RGBA canvas.

    Accepts the output of apply_seismic_colormap (or attribute-processed data)
    so callers can apply a seismic attribute before overlaying labels.

    Args:
        canvas: Base RGBA image, uint8 or float32, shape (H, W, 4).
        horizons_data: Horizon label slice, uint8 shape (H, W). 0 = unlabeled.
        faults_data: Fault label slice, uint8 shape (H, W). 0 = unlabeled.
        options: Alpha blend settings; uses defaults if None.

    Returns:
        uint8 RGBA array of shape (H, W, 4).
    """
    opts = options or RenderOptions()
    result = canvas.astype(np.float32)
    for label_data, alpha, kind in (
        (horizons_data, opts.horizon_alpha, "horizon"),
        (faults_data, opts.fault_alpha, "fault"),
    ):
        mask = label_data != 0
        if not np.any(mask):
            continue
        for eid in np.unique(label_data[mask]):
            pixel_mask = label_data == eid
            color = SliceRenderer._color_for(int(eid), kind)
            for ch, val in enumerate(color):
                result[:, :, ch] = np.where(
                    pixel_mask,
                    result[:, :, ch] * (1 - alpha) + val * alpha,
                    result[:, :, ch],
                )
    return np.clip(result, 0, 255).astype(np.uint8)
