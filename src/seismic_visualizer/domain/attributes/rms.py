"""RMS amplitude attribute processor."""

import numpy as np
from scipy.ndimage import uniform_filter1d

from seismic_visualizer.domain.attributes._guard import require_vertical
from seismic_visualizer.domain.slice import Slice


def compute_rms(slc: Slice, window: int = 9) -> np.ndarray:
    """Compute RMS amplitude in a sliding window along the time axis.

    Args:
        slc: Vertical seismic slice (Inline or Crossline).
        window: Full window size in samples (rounded up to odd if even).

    Returns:
        float32 array of shape slc.shape, values >= 0.

    Raises:
        AttributeNotApplicableError: If slc.axis is TIME.
    """
    require_vertical(slc)
    if window % 2 == 0:
        window += 1
    squared = slc.data.astype(np.float64) ** 2
    mean_sq = uniform_filter1d(squared, size=window, axis=1, mode="nearest")
    return np.asarray(np.sqrt(mean_sq), dtype=np.float32)
