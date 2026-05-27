"""SlicePresenter — produces RGBA frames and handles slice navigation."""

from __future__ import annotations

import numpy as np

from seismic_visualizer.application.services.attribute_service import AttributeService
from seismic_visualizer.application.services.project import Project
from seismic_visualizer.application.state import PresenterState, SliceIndex
from seismic_visualizer.domain.entities import EntityRegistry
from seismic_visualizer.domain.geometry import Axis, Point3D
from seismic_visualizer.application.rendering.colormap import apply_seismic_colormap
from seismic_visualizer.application.rendering.overlay import overlay_labels
from seismic_visualizer.domain.exceptions import AttributeNotApplicableError
from seismic_visualizer.domain.slice import Slice


class SlicePresenter:
    """Converts project data into renderable RGBA frames for the 2D view.

    Args:
        project: The open seismic project (cubes + registries).
        attr_service: Attribute computation and cache.
    """

    def __init__(
        self, project: Project, attr_service: AttributeService, cmap_name: str = "seismic"
    ) -> None:
        self._project = project
        self._attr_service = attr_service
        self._state = PresenterState()
        self._cmap_name: str = cmap_name

    def set_colormap(self, name: str) -> None:
        self._cmap_name = name

    @property
    def state(self) -> PresenterState:
        return self._state

    def get_rgba(self, slice_index: SliceIndex) -> np.ndarray:
        """Render the requested slice as a uint8 RGBA image.

        Applies the active attribute (if any), seismic colormap, and label overlays.

        Args:
            slice_index: Axis + position to render.

        Returns:
            uint8 RGBA array of shape (H, W, 4).
        """
        axis, index = slice_index.axis, slice_index.index
        seismic_slice = Slice(self._project.seismic.data, axis, index)

        if self._state.current_attribute is not None:
            try:
                attr_data = self._attr_service.compute(
                    seismic_slice, self._state.current_attribute
                )
                amin, amax = float(attr_data.min()), float(attr_data.max())
                span = amax - amin if amax > amin else 1.0
                seismic_data = ((attr_data - amin) * 255.0 / span).astype(np.uint8)
            except AttributeNotApplicableError:
                seismic_data = seismic_slice.data
        else:
            seismic_data = seismic_slice.data

        vmin, vmax = self._state.contrast_vmin, self._state.contrast_vmax
        if vmin > 0 or vmax < 255:
            data_f = np.clip(seismic_data.astype(np.float32), vmin, vmax)
            span = max(vmax - vmin, 1)
            seismic_data = ((data_f - vmin) * 255.0 / span).astype(np.uint8)

        h_data = Slice(self._project.horizons.data, axis, index).data
        f_data = Slice(self._project.faults.data, axis, index).data

        if axis != Axis.TIME:
            seismic_data = seismic_data.T
            h_data = h_data.T
            f_data = f_data.T

        h_data = self._apply_visibility(h_data, self._project.horizon_registry)
        f_data = self._apply_visibility(f_data, self._project.fault_registry)

        rgba = apply_seismic_colormap(seismic_data, self._cmap_name)
        return overlay_labels(rgba, h_data, f_data)

    @staticmethod
    def _apply_visibility(data: np.ndarray, registry: EntityRegistry) -> np.ndarray:
        hidden = {e.id for e in registry.all() if not e.visible}
        if not hidden:
            return data
        result = data.copy()
        for eid in hidden:
            result[result == eid] = 0
        return result

    def display_to_point3d(self, display_row: int, display_col: int) -> Point3D:
        """Convert display pixel coordinates to a 3D voxel position.

        Inverts the transpose applied in get_rgba: for INLINE/CROSSLINE slices
        the image is transposed so time runs vertically, so display_row=time,
        display_col=spatial. TIME slices are not transposed.
        """
        si = self._state.current_slice
        if si.axis != Axis.TIME:
            data_row, data_col = display_col, display_row
        else:
            data_row, data_col = display_row, display_col
        return Slice(self._project.seismic.data, si.axis, si.index).to_point3d(
            data_row, data_col
        )

    def navigate(self, slice_index: SliceIndex) -> None:
        """Update current slice position and invalidate the attribute cache.

        Args:
            slice_index: New slice to navigate to.
        """
        self._state.current_slice = slice_index
        self._attr_service.clear()
