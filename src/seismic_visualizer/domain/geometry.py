"""Core geometry primitives: Axis enum, Point2D, Point3D.

Implements CLAUDE.md §4.1 — Axis replaces magic 0/1/2 constants everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Axis(IntEnum):
    """Seismic cube axis identifier.

    Replaces magic integers 0/1/2 throughout the codebase. Use this enum
    wherever an axis must be specified; never compare raw ints against axes.
    """

    INLINE = 0
    CROSSLINE = 1
    TIME = 2

    @property
    def display_name(self) -> str:
        """Human-readable label used in status bars and dialogs."""
        return {0: "Inline", 1: "Crossline", 2: "Time"}[self.value]

    @property
    def is_labelable(self) -> bool:
        """True for Inline and Crossline; False for Time (view-only axis)."""
        return self in (Axis.INLINE, Axis.CROSSLINE)

    @property
    def other_axes(self) -> tuple[Axis, Axis]:
        """The two remaining axes that form the 2D plane of a slice along this axis.

        Order follows numpy convention (lower index first), so the result
        matches the shape of np.take(cube, idx, axis=self).

        Examples:
            Axis.INLINE.other_axes     == (Axis.CROSSLINE, Axis.TIME)
            Axis.CROSSLINE.other_axes  == (Axis.INLINE,    Axis.TIME)
            Axis.TIME.other_axes       == (Axis.INLINE,    Axis.CROSSLINE)
        """
        rest = [a for a in Axis if a != self]
        return rest[0], rest[1]


@dataclass(frozen=True)
class Point2D:
    """Immutable integer pixel coordinate on a 2D slice.

    Attributes:
        row: Vertical position (first dimension of the 2D slice array).
        col: Horizontal position (second dimension of the 2D slice array).
    """

    row: int
    col: int


@dataclass(frozen=True)
class Point3D:
    """Immutable voxel coordinate in the 3D seismic cube.

    Attributes:
        inline: Position along the Inline axis.
        crossline: Position along the Crossline axis.
        time: Position along the Time axis.
    """

    inline: int
    crossline: int
    time: int
