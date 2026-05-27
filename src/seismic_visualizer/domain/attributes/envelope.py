"""Envelope (instantaneous amplitude) attribute processor."""

import numpy as np
from scipy.signal import hilbert

from seismic_visualizer.domain.attributes._guard import require_vertical
from seismic_visualizer.domain.slice import Slice


def compute_envelope(slc: Slice) -> np.ndarray:
    """Compute instantaneous amplitude (envelope) of the analytic signal.

    Applies 1D Hilbert transform along the time axis (last axis of the slice).
    Valid only for Inline and Crossline slices.

    Args:
        slc: Vertical seismic slice (Inline or Crossline).

    Returns:
        float32 array of shape slc.shape, values >= 0.

    Raises:
        AttributeNotApplicableError: If slc.axis is TIME.
    """
    require_vertical(slc)
    analytic = hilbert(slc.data, axis=1)
    return np.asarray(np.abs(analytic), dtype=np.float32)
