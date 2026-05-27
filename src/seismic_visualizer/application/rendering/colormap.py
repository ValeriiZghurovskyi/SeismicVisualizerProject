"""Seismic colormap — maps uint8 amplitude to RGBA for display."""

import matplotlib.colors as _colors
import numpy as np
from matplotlib import colormaps as _colormaps

_NORM = _colors.Normalize(vmin=0, vmax=255)


def apply_seismic_colormap(data: np.ndarray, cmap_name: str = "seismic") -> np.ndarray:
    """Map a 2D uint8 seismic slice to RGBA using the given matplotlib colormap.

    Args:
        data: 2D array of seismic amplitudes, uint8 [0, 255].
        cmap_name: matplotlib colormap name. Raises KeyError for unknown names.

    Returns:
        uint8 RGBA array of shape (*data.shape, 4).
    """
    cmap = _colormaps[cmap_name]
    rgba_float = cmap(_NORM(data))
    return np.asarray((rgba_float * 255), dtype=np.uint8)
