"""Instantaneous phase attribute processor."""

import numpy as np
from scipy.signal import hilbert

from seismic_visualizer.domain.attributes._guard import require_vertical
from seismic_visualizer.domain.slice import Slice


def compute_instantaneous_phase(slc: Slice) -> np.ndarray:
    """Compute instantaneous phase of the analytic signal in radians.

    Args:
        slc: Vertical seismic slice (Inline or Crossline).

    Returns:
        float32 array of shape slc.shape, values in [-π, π].

    Raises:
        AttributeNotApplicableError: If slc.axis is TIME.
    """
    require_vertical(slc)
    analytic = hilbert(slc.data, axis=1)
    return np.asarray(np.angle(analytic), dtype=np.float32)
