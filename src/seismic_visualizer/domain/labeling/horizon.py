"""Convenience factory for horizon labeling (thickness=1 by default)."""

from seismic_visualizer.domain.labeling.base import Labeler


def horizon_labeler(entity_id: int, thickness: int = 1) -> Labeler:
    """Return a Labeler configured for horizon annotation."""
    return Labeler(entity_id=entity_id, thickness=thickness)
