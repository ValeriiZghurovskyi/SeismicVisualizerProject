"""Labeler — draws and erases entity labels on a Slice."""

from __future__ import annotations

import numpy as np
from skimage.morphology import dilation, disk

from seismic_visualizer.domain.algorithms.line_drawing import draw_line
from seismic_visualizer.domain.geometry import Point2D
from seismic_visualizer.domain.slice import Slice


class Labeler:
    """Writes entity labels onto a Slice via polyline drawing and circle erasing.

    Args:
        entity_id: Label value written to pixels (1..200).
        thickness: Line width in pixels (must be >= 1). Values > 1 apply
            morphological dilation with disk(radius=(thickness-1)//2).
    """

    def __init__(self, entity_id: int, thickness: int = 1) -> None:
        if thickness < 1:
            raise ValueError(f"thickness must be >= 1, got {thickness}")
        self.entity_id = entity_id
        self.thickness = thickness

    def label_polyline(self, sl: Slice, points: list[Point2D]) -> None:
        """Draw a polyline through points and write entity_id to each pixel.

        Consecutive point pairs are rasterized with Bresenham. If thickness > 1
        the resulting mask is dilated before writing.

        Args:
            sl: Target label slice (must be writable).
            points: Ordered vertices of the polyline.
        """
        if len(points) == 0:
            return

        rows, cols = sl.shape
        mask = np.zeros((rows, cols), dtype=bool)

        if len(points) == 1:
            p = points[0]
            mask[p.row, p.col] = True
        else:
            for p1, p2 in zip(points, points[1:], strict=False):
                for px in draw_line(p1, p2):
                    mask[px.row, px.col] = True

        if self.thickness > 1:
            radius = (self.thickness - 1) // 2
            mask = dilation(mask, disk(radius))  # type: ignore[no-untyped-call]

        sl.write_mask(mask, self.entity_id)

    @staticmethod
    def erase_at(sl: Slice, center: Point2D, radius: int) -> None:
        """Erase all labels within a circular area around center.

        Sets any non-zero pixel whose Euclidean distance to center is <= radius
        to 0. Erases all entity IDs regardless of entity_id.

        Args:
            sl: Target label slice.
            center: Centre of the eraser circle.
            radius: Eraser radius in pixels (0 = single pixel).
        """
        rows, cols = sl.shape
        r0, c0 = center.row, center.col

        r_min = max(0, r0 - radius)
        r_max = min(rows - 1, r0 + radius)
        c_min = max(0, c0 - radius)
        c_max = min(cols - 1, c0 + radius)

        rs = np.arange(r_min, r_max + 1)
        cs = np.arange(c_min, c_max + 1)
        rr, cc = np.meshgrid(rs, cs, indexing="ij")
        circle_mask = np.hypot(rr - r0, cc - c0) <= radius

        patch = sl.data[r_min: r_max + 1, c_min: c_max + 1]
        patch[circle_mask] = 0
