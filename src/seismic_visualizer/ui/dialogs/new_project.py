"""NewProjectDialog — file pickers for seismic, horizons, and faults .npz files."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from seismic_visualizer.ui.i18n import tr

_NPZ_FILTER = "NumPy archive (*.npz);;All files (*)"


class NewProjectDialog(QDialog):
    """Three-file picker: seismic (required), horizons and faults (optional)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dlg.open_project"))
        self.setMinimumWidth(480)

        self._seismic_edit = QLineEdit()
        self._horizons_edit = QLineEdit()
        self._faults_edit = QLineEdit()

        self._build_layout()


    def get_paths(self) -> tuple[Path, Path | None, Path | None]:
        """Return the selected file paths.

        Returns:
            (seismic_path, horizons_path, faults_path) where horizons and faults
            are None when their fields are left empty.
        """
        seismic = Path(self._seismic_edit.text())
        horizons_text = self._horizons_edit.text().strip()
        faults_text = self._faults_edit.text().strip()
        horizons: Path | None = Path(horizons_text) if horizons_text else None
        faults: Path | None = Path(faults_text) if faults_text else None
        return seismic, horizons, faults


    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.addRow(tr("dlg.seismic"), self._file_row(self._seismic_edit))
        form.addRow(tr("dlg.horizons"), self._file_row(self._horizons_edit))
        form.addRow(tr("dlg.faults"), self._file_row(self._faults_edit))
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel  # type: ignore[attr-defined]
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _file_row(self, edit: QLineEdit) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        browse = QPushButton(tr("dlg.browse"))
        browse.clicked.connect(lambda: self._browse(edit))
        row.addWidget(browse)
        return container

    def _browse(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("dlg.select_file"), "", _NPZ_FILTER)
        if path:
            edit.setText(path)
