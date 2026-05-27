"""ApplicationState and PresenterState — distributed state (no God Object).

See CLAUDE.md §3.2 rule 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seismic_visualizer.domain.geometry import Axis


@dataclass(frozen=True)
class SliceIndex:
    """Immutable identifier for a position in the seismic cube.

    Frozen so it can be used as a dict key (e.g. in AttributeService cache).

    Args:
        axis: Which axis the slice runs along.
        index: Position along that axis.
    """

    axis: Axis
    index: int


@dataclass
class PresenterState:
    """Mutable view state for View2DPresenter.

    Separate from Project so UI navigation state never pollutes domain data.

    Args:
        current_slice: The slice currently displayed.
        current_attribute: Name of the active seismic attribute, or None for raw.
    """

    current_slice: SliceIndex = field(
        default_factory=lambda: SliceIndex(axis=Axis.INLINE, index=0)
    )
    current_attribute: str | None = None
    contrast_vmin: int = 0
    contrast_vmax: int = 255
