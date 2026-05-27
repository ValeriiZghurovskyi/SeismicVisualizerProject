"""HUD — heads-up display showing cursor coordinates and seismic value."""

from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

from seismic_visualizer.ui.i18n import LanguageManager, tr

_PLACEHOLDER = "—"


class HUD(QWidget):
    """Fixed-height bar displaying inline/crossline/time/amplitude under cursor."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._inline = QLabel()
        self._xline = QLabel()
        self._time = QLabel()
        self._amp = QLabel()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 0, 2)
        layout.setSpacing(16)
        for label in (self._inline, self._xline, self._time, self._amp):
            layout.addWidget(label)

        self._cur_inline: int | None = None
        self._cur_xline: int | None = None
        self._cur_time: int | None = None
        self._cur_amp: int | None = None

        self.clear()
        LanguageManager.instance().language_changed.connect(lambda _: self._retranslate())

    def update_position(
        self, inline: int, xline: int, time: int, amp: int | None
    ) -> None:
        self._cur_inline = inline
        self._cur_xline = xline
        self._cur_time = time
        self._cur_amp = amp
        self._inline.setText(f"{tr('hud.inline')}: {inline}")
        self._xline.setText(f"{tr('hud.xline')}: {xline}")
        self._time.setText(f"{tr('hud.time')}: {time}")
        self._amp.setText(f"{tr('hud.amp')}: {amp if amp is not None else _PLACEHOLDER}")

    def clear(self) -> None:
        self._cur_inline = None
        self._inline.setText(f"{tr('hud.inline')}: {_PLACEHOLDER}")
        self._xline.setText(f"{tr('hud.xline')}: {_PLACEHOLDER}")
        self._time.setText(f"{tr('hud.time')}: {_PLACEHOLDER}")
        self._amp.setText(f"{tr('hud.amp')}: {_PLACEHOLDER}")

    def _retranslate(self) -> None:
        if self._cur_inline is None:
            self.clear()
        else:
            self.update_position(
                self._cur_inline,
                self._cur_xline or 0,
                self._cur_time or 0,
                self._cur_amp,
            )
