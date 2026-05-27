from __future__ import annotations

import sys
import onnxruntime
from PyQt5.QtWidgets import QApplication
from seismic_visualizer.ui.app import App



def main() -> None:

    qt_app = QApplication(sys.argv)
    qt_app.setStyle("Fusion")
    app = App(qt_app)
    app.start()
    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
