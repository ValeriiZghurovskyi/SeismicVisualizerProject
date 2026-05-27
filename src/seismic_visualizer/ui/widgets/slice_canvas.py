"""SliceCanvas — central canvas with zoom (Ctrl+wheel) and pan (middle/space+drag)."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QWidget

_ZOOM_MIN: float = 0.1
_ZOOM_MAX: float = 50.0
_ZOOM_FACTOR: float = 1.15
_PAN_MARGIN: int = 50

_RULER_TICKS: int = 10
_RULER_TICK_LEN: int = 8
_RULER_FONT_PX: int = 10


class SliceCanvas(QWidget):
    """Displays a seismic slice as a zoomable/pannable pixmap.

    Emits slice-space coordinates — callers never see screen pixels.
    """

    stroke_point = pyqtSignal(int, int)
    stroke_finished = pyqtSignal()
    erase_requested = pyqtSignal(int, int)
    mouse_hovered = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

        self._pixmap: QPixmap | None = None
        self._zoom_x: float = 1.0
        self._zoom_y: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._pixmap_size: tuple[int, int] = (0, 0)
        self._last_mouse: QPoint | None = None
        self._panning: bool = False
        self._preview_points: list[tuple[int, int]] = []
        self._stroke_color: tuple[int, int, int] | None = None
        self._auto_fit: bool = True
        self._cursor_data_pos: tuple[int, int] | None = None


    def set_stroke_color(self, color: tuple[int, int, int] | None) -> None:
        """Set the active entity color for preview drawing.

        Args:
            color: RGB tuple to use for preview, or None to disable drawing.
        """
        self._stroke_color = color

    def clear_preview(self) -> None:
        """Remove all preview points and connecting lines."""
        self._preview_points.clear()
        self.update()

    def remove_last_preview_point(self) -> None:
        """Remove the most recently added preview point."""
        if self._preview_points:
            self._preview_points.pop()
            self.update()

    def set_image(self, rgba: np.ndarray) -> None:
        """Load a uint8 RGBA array (H, W, 4) as the displayed frame.

        Args:
            rgba: uint8 array of shape (H, W, 4).
        """
        h, w = rgba.shape[:2]
        img = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(img)
        if (w, h) != self._pixmap_size:
            self._pixmap_size = (w, h)
            self._auto_fit = True
            self.fit_to_window()
        else:
            self.update()

    def fit_to_window(self) -> None:
        """Scale the pixmap to fill the entire widget (non-proportional stretch)."""
        if self._pixmap is None or self.width() == 0 or self.height() == 0:
            return
        self._zoom_x = self.width() / self._pixmap.width()
        self._zoom_y = self.height() / self._pixmap.height()
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def _clamp_pan(self) -> None:
        if self._pixmap is None:
            return
        img_w = self._pixmap.width() * self._zoom_x
        img_h = self._pixmap.height() * self._zoom_y
        w, h = self.width(), self.height()
        m = _PAN_MARGIN
        self._pan_x = max(-(img_w - m), min(self._pan_x, w - m))
        self._pan_y = max(-(img_h - m), min(self._pan_y, h - m))

    def pixel_to_data(self, px: int, py: int) -> tuple[int, int] | None:
        """Convert screen coordinates to data-array indices.

        Args:
            px: X screen coordinate (column direction).
            py: Y screen coordinate (row direction).

        Returns:
            (row, col) clamped to valid data bounds, or None if no image loaded
            or the point falls outside the data extent.
        """
        if self._pixmap is None:
            return None

        col = (px - self._pan_x) / self._zoom_x
        row = (py - self._pan_y) / self._zoom_y

        w = self._pixmap.width()
        h = self._pixmap.height()

        if col < 0 or col >= w or row < 0 or row >= h:
            return None

        return int(row), int(col)


    def paintEvent(self, event: object) -> None:  # type: ignore[override]
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.translate(self._pan_x, self._pan_y)
        painter.scale(self._zoom_x, self._zoom_y)
        painter.drawPixmap(0, 0, self._pixmap)

        if self._preview_points and self._stroke_color is not None:
            qcolor = QColor(*self._stroke_color)
            pen = QPen(qcolor)
            pen.setCosmetic(True)
            pen.setWidth(2)
            painter.setPen(pen)
            for i in range(len(self._preview_points) - 1):
                r0, c0 = self._preview_points[i]
                r1, c1 = self._preview_points[i + 1]
                painter.drawLine(QPointF(c0, r0), QPointF(c1, r1))
            painter.resetTransform()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(qcolor))
            dot_r = 4.0
            for row, col in self._preview_points:
                sx = col * self._zoom_x + self._pan_x
                sy = row * self._zoom_y + self._pan_y
                painter.drawEllipse(QPointF(sx, sy), dot_r, dot_r)

        if self._cursor_data_pos is not None:
            row, col = self._cursor_data_pos
            sx = col * self._zoom_x + self._pan_x
            sy = row * self._zoom_y + self._pan_y
            painter.resetTransform()
            painter.setRenderHint(QPainter.Antialiasing, False)
            crosshair_pen = QPen(QColor(180, 180, 180, 160), 0)
            painter.setPen(crosshair_pen)
            painter.drawLine(0, int(sy), self.width(), int(sy))
            painter.drawLine(int(sx), 0, int(sx), self.height())

        self._draw_rulers(painter)

    def _draw_rulers(self, painter: QPainter) -> None:
        """Draw axis tick marks with data-index labels along bottom and left edges."""
        if self._pixmap is None:
            return

        pw, ph = self._pixmap.width(), self._pixmap.height()
        painter.resetTransform()

        font = painter.font()
        font.setPixelSize(_RULER_FONT_PX)
        painter.setFont(font)
        fm = painter.fontMetrics()
        fh = fm.height()

        img_l = self._pan_x
        img_t = self._pan_y
        img_r = pw * self._zoom_x + self._pan_x
        img_b = ph * self._zoom_y + self._pan_y
        vx0 = max(img_l, 0.0)
        vx1 = min(img_r, float(self.width()))
        vy0 = max(img_t, 0.0)
        vy1 = min(img_b, float(self.height()))
        if vx1 <= vx0 or vy1 <= vy0:
            return

        tick_pen = QPen(QColor(200, 200, 200, 180))
        tick_pen.setWidth(1)
        text_pen = QPen(QColor(235, 235, 235))
        bg_brush = QBrush(QColor(0, 0, 0, 155))

        for i in range(_RULER_TICKS + 1):
            data_col = round(i * (pw - 1) / _RULER_TICKS)
            sx = data_col * self._zoom_x + self._pan_x
            if not vx0 <= sx <= vx1:
                continue
            label = str(data_col)
            lw = fm.horizontalAdvance(label)
            lx = int(max(vx0, min(sx - lw / 2, vx1 - lw)))
            ly = int(vy1) - 2

            painter.setPen(tick_pen)
            painter.drawLine(int(sx), int(vy1) - _RULER_TICK_LEN, int(sx), int(vy1))

            painter.setPen(Qt.NoPen)  # type: ignore[arg-type]
            painter.setBrush(bg_brush)
            painter.drawRect(lx - 1, ly - fh + 1, lw + 2, fh)
            painter.setPen(text_pen)
            painter.drawText(lx, ly, label)

        for i in range(_RULER_TICKS + 1):
            data_row = round(i * (ph - 1) / _RULER_TICKS)
            sy = data_row * self._zoom_y + self._pan_y
            if not vy0 <= sy <= vy1:
                continue
            label = str(data_row)
            lw = fm.horizontalAdvance(label)
            lx = int(vx0) + 3
            ly = int(max(vy0 + fh, min(sy + fh / 2, vy1)))

            painter.setPen(tick_pen)
            painter.drawLine(int(vx0), int(sy), int(vx0) + _RULER_TICK_LEN, int(sy))

            painter.setPen(Qt.NoPen)  # type: ignore[arg-type]
            painter.setBrush(bg_brush)
            painter.drawRect(lx - 1, ly - fh + 1, lw + 2, fh)
            painter.setPen(text_pen)
            painter.drawText(lx, ly, label)

    def mousePressEvent(self, event: object) -> None:  # type: ignore[override]
        from PyQt5.QtGui import QMouseEvent

        e: QMouseEvent = event  # type: ignore[assignment]
        if e.button() == Qt.MiddleButton:
            self._panning = True
            self._last_mouse = e.pos()
        elif e.button() == Qt.LeftButton and self._stroke_color is not None:
            coords = self.pixel_to_data(e.pos().x(), e.pos().y())
            if coords is not None:
                row, col = coords
                self._preview_points.append((row, col))
                self.update()
                self.stroke_point.emit(row, col)
        elif e.button() == Qt.RightButton and self._stroke_color is not None:
            coords = self.pixel_to_data(e.pos().x(), e.pos().y())
            if coords is not None:
                self.erase_requested.emit(*coords)

    def mouseMoveEvent(self, event: object) -> None:  # type: ignore[override]
        from PyQt5.QtGui import QMouseEvent

        e: QMouseEvent = event  # type: ignore[assignment]
        coords = self.pixel_to_data(e.pos().x(), e.pos().y())
        if coords is not None:
            self.mouse_hovered.emit(*coords)
        else:
            self.mouse_hovered.emit(-1, -1)
        self._cursor_data_pos = coords
        self.update()

        if self._panning and self._last_mouse is not None:
            delta = e.pos() - self._last_mouse
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._last_mouse = e.pos()
            self._clamp_pan()
            self.update()

    def leaveEvent(self, event: object) -> None:  # type: ignore[override]
        self.mouse_hovered.emit(-1, -1)
        self._cursor_data_pos = None
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:  # type: ignore[override]
        from PyQt5.QtGui import QMouseEvent

        e: QMouseEvent = event  # type: ignore[assignment]
        if e.button() == Qt.MiddleButton:
            self._panning = False
        self._last_mouse = None

    def resizeEvent(self, event: object) -> None:  # type: ignore[override]
        if self._auto_fit:
            self.fit_to_window()

    def wheelEvent(self, event: object) -> None:  # type: ignore[override]
        from PyQt5.QtGui import QWheelEvent as _QWE

        e: _QWE = event  # type: ignore[assignment]
        delta = e.angleDelta().y()

        if e.modifiers() & Qt.ControlModifier:
            self._auto_fit = False
            cursor_x = e.pos().x()
            cursor_y = e.pos().y()
            factor = _ZOOM_FACTOR if delta > 0 else 1.0 / _ZOOM_FACTOR
            new_zoom_x = max(_ZOOM_MIN, min(self._zoom_x * factor, _ZOOM_MAX))
            new_zoom_y = max(_ZOOM_MIN, min(self._zoom_y * factor, _ZOOM_MAX))
            self._pan_x = cursor_x - (new_zoom_x / self._zoom_x) * (cursor_x - self._pan_x)
            self._pan_y = cursor_y - (new_zoom_y / self._zoom_y) * (cursor_y - self._pan_y)
            self._zoom_x = new_zoom_x
            self._zoom_y = new_zoom_y
        elif e.modifiers() & Qt.ShiftModifier:
            self._pan_x += delta / 120 * 60
        else:
            self._pan_y += delta / 120 * 60

        self._clamp_pan()
        self.update()
