"""Convenience factory for fault labeling (thickness=3 by default)."""

from seismic_visualizer.domain.labeling.base import Labeler


def fault_labeler(entity_id: int, thickness: int = 3) -> Labeler:
    """Return a Labeler configured for fault annotation."""
    return Labeler(entity_id=entity_id, thickness=thickness)
