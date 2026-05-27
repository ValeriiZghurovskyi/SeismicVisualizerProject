"""Tests for domain.rendering.colormap — apply_seismic_colormap.

Seismic data is uint8 [0, 255]; midpoint 128 = zero amplitude.
"""

import numpy as np

from seismic_visualizer.application.rendering.colormap import apply_seismic_colormap


ROWS, COLS = 10, 12


def _make_slice(fill: int) -> np.ndarray:
    return np.full((ROWS, COLS), fill, dtype=np.uint8)


def test_output_shape_rgba() -> None:
    result = apply_seismic_colormap(_make_slice(128))
    assert result.shape == (ROWS, COLS, 4)


def test_output_dtype_uint8() -> None:
    result = apply_seismic_colormap(_make_slice(128))
    assert result.dtype == np.uint8


def test_midpoint_maps_to_neutral() -> None:
    # Value 128 = zero amplitude → white (near 255,255,255)
    result = apply_seismic_colormap(_make_slice(128))
    pixel = result[0, 0]
    assert int(pixel[0]) > 200 and int(pixel[2]) > 200


def test_max_value_maps_to_red_dominant() -> None:
    result = apply_seismic_colormap(_make_slice(255))
    pixel = result[0, 0]
    assert int(pixel[0]) > int(pixel[2])  # R > B


def test_min_value_maps_to_blue_dominant() -> None:
    result = apply_seismic_colormap(_make_slice(0))
    pixel = result[0, 0]
    assert int(pixel[2]) > int(pixel[0])  # B > R


def test_all_same_value_no_crash() -> None:
    apply_seismic_colormap(_make_slice(128))


def test_alpha_channel_fully_opaque() -> None:
    result = apply_seismic_colormap(_make_slice(200))
    assert np.all(result[:, :, 3] == 255)


def test_symmetric_around_midpoint() -> None:
    # 128+d and 128-d should map to red-dominant and blue-dominant respectively
    r_high = apply_seismic_colormap(_make_slice(255))
    r_low = apply_seismic_colormap(_make_slice(0))
    assert int(r_high[0, 0, 0]) > int(r_high[0, 0, 2])  # R > B for high
    assert int(r_low[0, 0, 2]) > int(r_low[0, 0, 0])    # B > R for low


def test_accepts_2d_uint8_array() -> None:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(ROWS, COLS), dtype=np.uint8)
    result = apply_seismic_colormap(data)
    assert result.shape == (ROWS, COLS, 4)
    assert result.dtype == np.uint8


def test_default_cmap_name_backward_compatible() -> None:
    # Explicit "seismic" must produce the same result as the default.
    data = _make_slice(200)
    assert np.array_equal(
        apply_seismic_colormap(data),
        apply_seismic_colormap(data, cmap_name="seismic"),
    )


def test_gray_maps_to_grayscale() -> None:
    # gray colormap: R == G == B for every pixel.
    result = apply_seismic_colormap(_make_slice(100), cmap_name="gray")
    r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
    assert np.all(r == g) and np.all(g == b)


def test_invalid_cmap_name_raises_key_error() -> None:
    import pytest
    with pytest.raises(KeyError):
        apply_seismic_colormap(_make_slice(128), cmap_name="__nonexistent_xyz__")
