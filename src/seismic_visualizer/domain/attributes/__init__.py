"""Seismic attribute processors (RMS, Phase, Frequency, Envelope)."""

from seismic_visualizer.domain.attributes.base import AttributeProcessor
from seismic_visualizer.domain.attributes.envelope import compute_envelope
from seismic_visualizer.domain.attributes.frequency import compute_instantaneous_frequency
from seismic_visualizer.domain.attributes.phase import compute_instantaneous_phase
from seismic_visualizer.domain.attributes.rms import compute_rms

__all__ = [
    "AttributeProcessor",
    "compute_envelope",
    "compute_instantaneous_frequency",
    "compute_instantaneous_phase",
    "compute_rms",
]
