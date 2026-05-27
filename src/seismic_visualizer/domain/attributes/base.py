"""AttributeProcessor type alias — callable contract for seismic attributes."""

from collections.abc import Callable

import numpy as np

from seismic_visualizer.domain.slice import Slice

AttributeProcessor = Callable[[Slice], np.ndarray]
