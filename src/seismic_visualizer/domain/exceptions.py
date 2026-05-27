"""Domain-layer custom exceptions.

All errors raised by domain objects are subclasses of SeismicError so callers
can catch the base type without accidentally masking unrelated bugs.
"""


class SeismicError(Exception):
    """Base class for all domain-layer errors."""


class IndexOutOfBoundsError(SeismicError):
    """Slice index is outside the valid range for that cube axis."""


class LabelingNotAllowedError(SeismicError):
    """Attempt to draw labels on a view-only axis (e.g. Time)."""


class EntityNotFoundError(SeismicError):
    """Requested entity ID does not exist in the registry."""


class EntityLimitReachedError(SeismicError):
    """All 200 entity slots are occupied; no new entity can be created."""


class ShapeMismatchError(SeismicError):
    """Label cube shape does not match the seismic cube shape."""


class AttributeNotApplicableError(SeismicError):
    """Attribute cannot be computed for the given slice axis (e.g. Phase on Time slice)."""
