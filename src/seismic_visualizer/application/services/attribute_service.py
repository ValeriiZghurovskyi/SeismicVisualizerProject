"""AttributeService — computes and caches seismic attributes per slice."""

from __future__ import annotations

import numpy as np

from seismic_visualizer.domain.attributes import (
    AttributeProcessor,
    compute_envelope,
    compute_instantaneous_frequency,
    compute_instantaneous_phase,
    compute_rms,
)
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.domain.slice import Slice

_PROCESSORS: dict[str, AttributeProcessor] = {
    "envelope": compute_envelope,
    "phase": compute_instantaneous_phase,
    "frequency": compute_instantaneous_frequency,
    "rms": compute_rms,
}


class AttributeService:
    """Computes seismic attributes on demand and caches results by slice position.

    Cache key: (axis, index, attribute_name).
    Call clear() whenever slice navigation invalidates cached results.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[Axis, int, str], np.ndarray] = {}

    def compute(self, seismic_slice: Slice, attr_name: str) -> np.ndarray:
        """Return attribute data for the given slice, using cache when available.

        Args:
            seismic_slice: The seismic slice to process.
            attr_name: One of "envelope", "phase", "frequency", "rms".

        Returns:
            float32 ndarray of the same shape as seismic_slice.

        Raises:
            AttributeNotApplicableError: If seismic_slice.axis is TIME.
            ValueError: If attr_name is not a known attribute.
        """
        key = (seismic_slice.axis, seismic_slice.index, attr_name)
        if key in self._cache:
            return self._cache[key]

        processor = _PROCESSORS.get(attr_name)
        if processor is None:
            raise ValueError(f"Unknown attribute: {attr_name!r}")

        result = processor(seismic_slice)
        self._cache[key] = result
        return result

    def clear(self) -> None:
        """Invalidate all cached attribute results."""
        self._cache.clear()
