"""Shared guard for attribute axis validation."""

from seismic_visualizer.domain.exceptions import AttributeNotApplicableError
from seismic_visualizer.domain.geometry import Axis
from seismic_visualizer.domain.slice import Slice


def require_vertical(slc: Slice) -> None:
    """Raise AttributeNotApplicableError if slc is a Time slice."""
    if slc.axis == Axis.TIME:
        raise AttributeNotApplicableError(
            f"Attribute requires a vertical slice (Inline or Crossline), "
            f"got axis={slc.axis.display_name}"
        )
