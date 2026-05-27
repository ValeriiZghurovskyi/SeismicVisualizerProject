"""SavePromptDialog — "Save changes?" with Save / Discard / Cancel options."""

from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox, QWidget

from seismic_visualizer.ui.i18n import tr


def ask_save_before_close(parent: QWidget | None = None) -> bool | None:
    """Show a "Save changes before closing?" modal dialog.

    Args:
        parent: Parent widget for dialog centering.

    Returns:
        True if the user chose Save, False if they chose Discard, None if Cancel.
    """
    box = QMessageBox(parent)
    box.setWindowTitle(tr("save_prompt.title"))
    box.setText(tr("save_prompt.text"))
    box.setInformativeText(tr("save_prompt.info"))
    box.setStandardButtons(
        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel  # type: ignore[attr-defined]
    )
    box.setDefaultButton(QMessageBox.Save)  # type: ignore[attr-defined]

    result = box.exec_()
    if result == QMessageBox.Save:  # type: ignore[attr-defined]
        return True
    if result == QMessageBox.Discard:  # type: ignore[attr-defined]
        return False
    return None
