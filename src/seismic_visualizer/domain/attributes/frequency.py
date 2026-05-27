"""Instantaneous frequency attribute processor."""

import numpy as np
from scipy.signal import hilbert

from seismic_visualizer.domain.attributes._guard import require_vertical
from seismic_visualizer.domain.slice import Slice


def compute_instantaneous_frequency(slc: Slice) -> np.ndarray:
    """Compute instantaneous frequency via derivative of unwrapped phase.

    Frequency is estimated as dφ/dt / (2π), normalised to samples⁻¹.
    Edge samples are replicated from their nearest interior neighbour.

    Args:
        slc: Vertical seismic slice (Inline or Crossline).

    Returns:
        float32 array of shape slc.shape, values >= 0 in the interior.

    Raises:
        AttributeNotApplicableError: If slc.axis is TIME.
    """
    require_vertical(slc)
    analytic = hilbert(slc.data, axis=1)
    phase = np.unwrap(np.angle(analytic), axis=1)
    diff = np.diff(phase, axis=1) / (2 * np.pi)
    freq = np.concatenate([diff[:, :1], diff], axis=1)
    return np.asarray(np.abs(freq), dtype=np.float32)
