"""Desktop GUI entry point for PaintWeave Studio."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from paintweave.gui.main_window import PaintWeaveMainWindow
from paintweave.nodes import register_all_nodes


def main(argv: list[str] | None = None) -> int:
    """Launch the PaintWeave Studio desktop application."""
    register_all_nodes()

    app = QApplication.instance() or QApplication(
        argv if argv is not None else sys.argv
    )
    app.setApplicationName("PaintWeave Studio")
    app.setOrganizationName("PaintWeave")
    app.setStyle("Fusion")

    window = PaintWeaveMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
