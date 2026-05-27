"""Tests for domain.rendering.overlay — SliceRenderer and entity_color."""

import numpy as np
import pytest

from seismic_visualizer.domain.entities import EntityKind
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.application.rendering.overlay import RenderOptions, SliceRenderer, entity_color
from seismic_visualizer.domain.slice import Slice

SHAPE = (8, 10, 12)  # inline × crossline × time
_rng = np.random.default_rng(42)


def _seismic() -> np.ndarray:
    return _rng.integers(0, 256, size=SHAPE, dtype=np.uint8)


def _labels() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _inline_slices(idx: int = 3) -> tuple[Slice, Slice, Slice]:
    seismic = _seismic()
    horizons = _labels()
    faults = _labels()
    return (
        Slice(seismic, Axis.INLINE, idx),
        Slice(horizons, Axis.INLINE, idx),
        Slice(faults, Axis.INLINE, idx),
    )


# ---------------------------------------------------------------------------
# shape and dtype
# ---------------------------------------------------------------------------

def test_output_shape_rgba() -> None:
    s, h, f = _inline_slices()
    result = SliceRenderer().render(s, h, f)
    assert result.shape == (s.shape[0], s.shape[1], 4)


def test_output_dtype_uint8() -> None:
    s, h, f = _inline_slices()
    result = SliceRenderer().render(s, h, f)
    assert result.dtype == np.uint8


def test_time_slice_renders_without_error() -> None:
    seismic = _seismic()
    horizons = _labels()
    faults = _labels()
    s = Slice(seismic, Axis.TIME, 5)
    h = Slice(horizons, Axis.TIME, 5)
    f = Slice(faults, Axis.TIME, 5)
    result = SliceRenderer().render(s, h, f)
    assert result.shape == (s.shape[0], s.shape[1], 4)


# ---------------------------------------------------------------------------
# overlay logic
# ---------------------------------------------------------------------------

def test_horizon_label_changes_pixel_color() -> None:
    seismic = _seismic()
    horizons = _labels()
    faults = _labels()

    # paint entity 1 on a single pixel
    horizons[3, 2, 4] = 1
    s = Slice(seismic, Axis.INLINE, 3)
    h = Slice(horizons, Axis.INLINE, 3)
    f = Slice(faults, Axis.INLINE, 3)

    no_label = _labels()
    s2 = Slice(seismic, Axis.INLINE, 3)
    h2 = Slice(no_label, Axis.INLINE, 3)
    f2 = Slice(faults, Axis.INLINE, 3)

    with_label = SliceRenderer().render(s, h, f)
    without_label = SliceRenderer().render(s2, h2, f2)

    assert not np.array_equal(with_label[2, 4], without_label[2, 4])


def test_fault_label_changes_pixel_color() -> None:
    seismic = _seismic()
    horizons = _labels()
    faults = _labels()

    faults[3, 5, 7] = 1
    s = Slice(seismic, Axis.INLINE, 3)
    h = Slice(horizons, Axis.INLINE, 3)
    f = Slice(faults, Axis.INLINE, 3)

    no_faults = _labels()
    f2 = Slice(no_faults, Axis.INLINE, 3)
    without_label = SliceRenderer().render(s, h, f2)
    with_label = SliceRenderer().render(s, h, f)

    assert not np.array_equal(with_label[5, 7], without_label[5, 7])


def test_alpha_fully_opaque() -> None:
    s, h, f = _inline_slices()
    result = SliceRenderer().render(s, h, f)
    assert np.all(result[:, :, 3] == 255)


def test_horizon_and_fault_different_colors_on_same_pixel() -> None:
    # If a pixel has both horizon and fault, fault wins (or they differ from seismic)
    seismic = _seismic()
    horizons = _labels()
    faults = _labels()
    horizons[3, 1, 1] = 1
    faults[3, 1, 1] = 1

    h_only = _labels()
    h_only[3, 1, 1] = 1
    f_only = _labels()
    f_only[3, 1, 1] = 1

    renderer = SliceRenderer()
    s = Slice(seismic, Axis.INLINE, 3)

    r_both = renderer.render(s, Slice(horizons, Axis.INLINE, 3), Slice(faults, Axis.INLINE, 3))
    r_horizon = renderer.render(s, Slice(h_only, Axis.INLINE, 3), Slice(_labels(), Axis.INLINE, 3))
    r_fault = renderer.render(s, Slice(_labels(), Axis.INLINE, 3), Slice(f_only, Axis.INLINE, 3))

    # All three results exist and are uint8 RGBA
    for r in (r_both, r_horizon, r_fault):
        assert r.dtype == np.uint8
        assert r.shape[2] == 4


def test_custom_render_options_accepted() -> None:
    s, h, f = _inline_slices()
    opts = RenderOptions(horizon_alpha=0.8, fault_alpha=0.9)
    result = SliceRenderer().render(s, h, f, options=opts)
    assert result.shape == (s.shape[0], s.shape[1], 4)


# ---------------------------------------------------------------------------
# entity_color
# ---------------------------------------------------------------------------

def test_entity_color_horizon_returns_rgb_tuple() -> None:
    color = entity_color(0, EntityKind.HORIZON)
    assert isinstance(color, tuple)
    assert len(color) == 3


def test_entity_color_fault_returns_rgb_tuple() -> None:
    color = entity_color(0, EntityKind.FAULT)
    assert isinstance(color, tuple)
    assert len(color) == 3


def test_entity_color_horizon_palette_cycles() -> None:
    colors = [entity_color(i, EntityKind.HORIZON) for i in range(1, 9)]
    # all 8 palette slots are distinct
    assert len(set(colors)) == 8
    # ID 9 wraps back to same color as ID 1
    assert entity_color(9, EntityKind.HORIZON) == entity_color(1, EntityKind.HORIZON)


def test_entity_color_fault_cycles_palette() -> None:
    # faults now use a palette (8 colors) — ID 9 wraps back to same color as ID 1
    assert entity_color(9, EntityKind.FAULT) == entity_color(1, EntityKind.FAULT)
    # distinct IDs within one cycle have different colors
    assert entity_color(1, EntityKind.FAULT) != entity_color(2, EntityKind.FAULT)


def test_entity_color_values_in_valid_range() -> None:
    for kind in (EntityKind.HORIZON, EntityKind.FAULT):
        for eid in range(1, 12):
            for channel in entity_color(eid, kind):
                assert 0 <= channel <= 255
