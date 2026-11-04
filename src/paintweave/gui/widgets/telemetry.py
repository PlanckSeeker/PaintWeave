"""pyqtgraph Cook Profiler & Telemetry Workbench displaying per-node execution latency."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from paintweave.core.graph import Graph
from paintweave.engine.geometry.compute_cuda import get_compute_summary


class CookTelemetryPanel(QWidget):
    """Visualizes per-node cook latency (ms) in a live pyqtgraph bar chart."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.total_lbl = QLabel("Total Network Cook Time: 0.00 ms")
        self.total_lbl.setStyleSheet("color: #38bdf8; font-weight: 700;")

        self.gpu_lbl = QLabel(get_compute_summary())
        self.gpu_lbl.setStyleSheet("color: #34d399; font-weight: 600;")

        header.addWidget(self.total_lbl)
        header.addStretch(1)
        header.addWidget(self.gpu_lbl)
        layout.addLayout(header)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#0d1016")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Cook Time", units="ms")
        self.plot_widget.setLabel("bottom", "Graph Operators")
        layout.addWidget(self.plot_widget, stretch=1)

    def refresh_telemetry(self, graph: Graph) -> None:
        self.plot_widget.clear()
        if not graph.nodes:
            self.total_lbl.setText("Total Network Cook Time: 0.00 ms")
            return

        try:
            ordered = graph.get_evaluation_order()
        except Exception:
            ordered = list(graph.nodes.values())

        names = [n.name for n in ordered]
        times = np.array(
            [float(n.last_cook_time_ms) for n in ordered], dtype=np.float32
        )
        total_ms = float(np.sum(times))
        self.total_lbl.setText(
            f"Total Network Cook Time: {total_ms:.2f} ms  ({len(ordered)} active nodes)"
        )

        x = np.arange(len(ordered), dtype=np.float32)
        bar_item = pg.BarGraphItem(
            x=x,
            height=times,
            width=0.55,
            brush=pg.mkBrush("#0284c7"),
            pen=pg.mkPen("#38bdf8", width=1.2),
        )
        self.plot_widget.addItem(bar_item)

        ax = self.plot_widget.getAxis("bottom")
        ax.setTicks([[(int(i), name) for i, name in enumerate(names)]])
