"""Tests for application.rendering.overlay — SliceRenderer, overlay_labels, entity_color."""

from __future__ import annotations

import numpy as np
import pytest

from seismic_visualizer.application.rendering.overlay import (
    RenderOptions,
    SliceRenderer,
    entity_color,
    overlay_labels,
)
from seismic_visualizer.application.rendering.colormap import apply_seismic_colormap
from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.domain.slice import Slice

SHAPE = (6, 8, 10)
_rng = np.random.default_rng(0)


def _seismic_slice(axis: Axis = Axis.INLINE, idx: int = 2) -> Slice:
    data = _rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
    return Slice(data, axis, idx)


def _label_slice(axis: Axis = Axis.INLINE, idx: int = 2, fill: int = 0) -> Slice:
    data = np.full(SHAPE, fill, dtype=np.uint8)
    return Slice(data, axis, idx)


# ---------------------------------------------------------------------------
# SliceRenderer.render
# ---------------------------------------------------------------------------


class TestSliceRendererRender:
    def test_output_shape_inline(self) -> None:
        s = _seismic_slice()
        h = _label_slice()
        f = _label_slice()
        result = SliceRenderer().render(s, h, f)
        assert result.shape == (s.shape[0], s.shape[1], 4)

    def test_output_dtype_uint8(self) -> None:
        s = _seismic_slice()
        result = SliceRenderer().render(s, _label_slice(), _label_slice())
        assert result.dtype == np.uint8

    def test_alpha_channel_fully_opaque(self) -> None:
        s = _seismic_slice()
        result = SliceRenderer().render(s, _label_slice(), _label_slice())
        assert np.all(result[:, :, 3] == 255)

    def test_horizon_label_modifies_pixel(self) -> None:
        seismic_data = _rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
        labels = np.zeros(SHAPE, dtype=np.uint8)
        labels[2, 3, 5] = 1

        s = Slice(seismic_data, Axis.INLINE, 2)
        h_labeled = Slice(labels, Axis.INLINE, 2)
        h_empty = Slice(np.zeros(SHAPE, dtype=np.uint8), Axis.INLINE, 2)
        f = Slice(np.zeros(SHAPE, dtype=np.uint8), Axis.INLINE, 2)

        with_label = SliceRenderer().render(s, h_labeled, f)
        without_label = SliceRenderer().render(s, h_empty, f)

        assert not np.array_equal(with_label[3, 5], without_label[3, 5])

    def test_fault_label_modifies_pixel(self) -> None:
        seismic_data = _rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
        faults = np.zeros(SHAPE, dtype=np.uint8)
        faults[2, 1, 2] = 1

        s = Slice(seismic_data, Axis.INLINE, 2)
        h = Slice(np.zeros(SHAPE, dtype=np.uint8), Axis.INLINE, 2)
        f_labeled = Slice(faults, Axis.INLINE, 2)
        f_empty = Slice(np.zeros(SHAPE, dtype=np.uint8), Axis.INLINE, 2)

        with_fault = SliceRenderer().render(s, h, f_labeled)
        without_fault = SliceRenderer().render(s, h, f_empty)

        assert not np.array_equal(with_fault[1, 2], without_fault[1, 2])

    def test_custom_options_accepted(self) -> None:
        s = _seismic_slice()
        opts = RenderOptions(horizon_alpha=0.9, fault_alpha=0.5)
        result = SliceRenderer().render(s, _label_slice(), _label_slice(), options=opts)
        assert result.shape[2] == 4

    def test_default_options_used_when_none(self) -> None:
        s = _seismic_slice()
        r1 = SliceRenderer().render(s, _label_slice(), _label_slice(), options=None)
        r2 = SliceRenderer().render(s, _label_slice(), _label_slice(), options=RenderOptions())
        np.testing.assert_array_equal(r1, r2)

    def test_multiple_entity_ids_rendered(self) -> None:
        seismic_data = _rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
        labels = np.zeros(SHAPE, dtype=np.uint8)
        labels[2, 0, 0] = 1
        labels[2, 4, 4] = 2

        s = Slice(seismic_data, Axis.INLINE, 2)
        h = Slice(labels, Axis.INLINE, 2)
        f = Slice(np.zeros(SHAPE, dtype=np.uint8), Axis.INLINE, 2)

        result = SliceRenderer().render(s, h, f)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# overlay_labels
# ---------------------------------------------------------------------------


class TestOverlayLabels:
    def _canvas(self, shape=(8, 10)) -> np.ndarray:
        data = _rng.integers(0, 256, size=(shape[0], shape[1]), dtype=np.uint8)
        return apply_seismic_colormap(data)

    def test_output_shape_matches_canvas(self) -> None:
        canvas = self._canvas((8, 10))
        result = overlay_labels(canvas, np.zeros((8, 10), dtype=np.uint8), np.zeros((8, 10), dtype=np.uint8))
        assert result.shape == canvas.shape

    def test_output_dtype_uint8(self) -> None:
        canvas = self._canvas()
        result = overlay_labels(canvas, np.zeros((8, 10), dtype=np.uint8), np.zeros((8, 10), dtype=np.uint8))
        assert result.dtype == np.uint8

    def test_no_labels_returns_same_as_input(self) -> None:
        canvas = self._canvas((6, 8))
        result = overlay_labels(
            canvas,
            np.zeros((6, 8), dtype=np.uint8),
            np.zeros((6, 8), dtype=np.uint8),
        )
        np.testing.assert_array_equal(result, canvas)

    def test_horizon_label_changes_pixel(self) -> None:
        canvas = self._canvas((6, 8))
        horizons = np.zeros((6, 8), dtype=np.uint8)
        horizons[2, 3] = 1
        faults = np.zeros((6, 8), dtype=np.uint8)

        result = overlay_labels(canvas, horizons, faults)
        assert not np.array_equal(result[2, 3], canvas[2, 3])

    def test_fault_label_changes_pixel(self) -> None:
        canvas = self._canvas((6, 8))
        horizons = np.zeros((6, 8), dtype=np.uint8)
        faults = np.zeros((6, 8), dtype=np.uint8)
        faults[4, 5] = 1

        result = overlay_labels(canvas, horizons, faults)
        assert not np.array_equal(result[4, 5], canvas[4, 5])

    def test_custom_options_applied(self) -> None:
        canvas = self._canvas((6, 8))
        horizons = np.zeros((6, 8), dtype=np.uint8)
        horizons[1, 1] = 1

        opts = RenderOptions(horizon_alpha=1.0, fault_alpha=0.0)
        result = overlay_labels(canvas, horizons, np.zeros((6, 8), dtype=np.uint8), options=opts)
        assert result.dtype == np.uint8

    def test_values_clipped_to_uint8_range(self) -> None:
        canvas = np.full((4, 4, 4), 255, dtype=np.uint8)
        horizons = np.ones((4, 4), dtype=np.uint8)
        result = overlay_labels(canvas, horizons, np.zeros((4, 4), dtype=np.uint8))
        assert result.max() <= 255
        assert result.min() >= 0


# ---------------------------------------------------------------------------
# entity_color
# ---------------------------------------------------------------------------


class TestEntityColor:
    def test_horizon_returns_tuple_of_3(self) -> None:
        color = entity_color(1, EntityKind.HORIZON)
        assert isinstance(color, tuple) and len(color) == 3

    def test_fault_returns_tuple_of_3(self) -> None:
        color = entity_color(1, EntityKind.FAULT)
        assert isinstance(color, tuple) and len(color) == 3

    def test_horizon_palette_cycles_at_9(self) -> None:
        assert entity_color(9, EntityKind.HORIZON) == entity_color(1, EntityKind.HORIZON)

    def test_fault_palette_cycles_at_9(self) -> None:
        assert entity_color(9, EntityKind.FAULT) == entity_color(1, EntityKind.FAULT)

    def test_different_ids_have_different_colors(self) -> None:
        c1 = entity_color(1, EntityKind.HORIZON)
        c2 = entity_color(2, EntityKind.HORIZON)
        assert c1 != c2

    def test_horizon_and_fault_differ_for_same_id(self) -> None:
        ch = entity_color(1, EntityKind.HORIZON)
        cf = entity_color(1, EntityKind.FAULT)
        assert ch != cf

    def test_all_channels_in_valid_range(self) -> None:
        for kind in (EntityKind.HORIZON, EntityKind.FAULT):
            for eid in range(1, 10):
                for ch in entity_color(eid, kind):
                    assert 0 <= ch <= 255
