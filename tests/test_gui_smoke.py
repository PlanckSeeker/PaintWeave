"""Smoke test for PaintWeaveMainWindow, NodeGraphQt canvas sync, and example .pw projects."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from paintweave.core.graph import Graph
from paintweave.gui.main_window import PaintWeaveMainWindow
from paintweave.nodes import register_all_nodes


def test_examples_cook_and_main_window_initializes() -> None:
    register_all_nodes()
    app = QApplication.instance() or QApplication([])
    assert app is not None

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    pw_files = sorted(examples_dir.glob("*.pw"))
    assert len(pw_files) >= 3

    for pw_file in pw_files:
        graph = Graph.load_from_file(str(pw_file))
        sop = graph.cook_display_sop(force=True)
        assert sop is not None
        geo = sop.outputs["Geo"].value
        assert geo is not None
        assert geo.num_points > 0
        assert geo.num_primitives > 0
        assert len(geo.detail["materials"]) >= 1

    window = PaintWeaveMainWindow()
    assert len(window.canvas_ctrl.core_graph.nodes) > 0
    assert len(window.canvas_ctrl.core_graph.links) == 12
    assert len(window.canvas_ctrl.node_graph.viewer().all_pipes()) == 12

    # Verify interactive node connection and single-input replacement on the canvas
    window._build_default_showcase_network()
    assert len(window.canvas_ctrl.core_graph.nodes) >= 10
    initial_pipes = len(window.canvas_ctrl.node_graph.viewer().all_pipes())
    assert initial_pipes == len(window.canvas_ctrl.core_graph.links) > 0

    ctrl = window.canvas_ctrl
    g1 = ctrl.spawn_node("SOP_Grid", pos=(100, 100))
    g2 = ctrl.spawn_node("SOP_Box", pos=(100, 220))
    mtn = ctrl.spawn_node("SOP_Mountain", pos=(340, 160))

    v_g1 = ctrl._uid_to_visual[g1.id]
    v_g2 = ctrl._uid_to_visual[g2.id]
    v_mtn = ctrl._uid_to_visual[mtn.id]

    # Simulate user connecting g1.Geo -> mtn.Geo in NodeGraphQt
    v_g1.get_output("Geo").connect_to(v_mtn.get_input("Geo"))
    assert v_mtn.get_input("Geo") in v_g1.get_output("Geo").connected_ports()
    assert mtn.inputs["Geo"].is_connected
    assert mtn.inputs["Geo"].links[0].source_pin is g1.outputs["Geo"]

    # Simulate user replacing mtn.Geo connection with g2.Geo (single-input replacement)
    v_g2.get_output("Geo").connect_to(v_mtn.get_input("Geo"))
    assert v_mtn.get_input("Geo") in v_g2.get_output("Geo").connected_ports()
    assert mtn.inputs["Geo"].links[0].source_pin is g2.outputs["Geo"]
    assert len(ctrl.node_graph.viewer().all_pipes()) == len(ctrl.core_graph.links)

    window.close()


def test_artificers_atelier_ui_components() -> None:
    """Verify Artificer's Atelier theme titles, swatches, and ArtificerCodeEditor."""
    from paintweave.gui.widgets.code_studio import ArtificerCodeEditor
    from paintweave.gui.widgets.inspector_dock import ARTIFICER_SWATCHES

    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = PaintWeaveMainWindow()
    assert "Artificer's Atelier" in window.windowTitle()
    assert "Operator Catalog" in window.dock_palette.windowTitle()
    assert "Transmutation Matrix" in window.dock_inspector.windowTitle()
    assert "Atelier Workbench" in window.dock_workbench.windowTitle()

    # Code Studio: ArtificerCodeEditor with Line Numbers & Error Highlighting
    editor = window.code_studio.source_editor
    assert isinstance(editor, ArtificerCodeEditor)
    assert editor.line_number_area_width() >= 32

    editor.setPlainText("line 1\nline 2\nline 3\nline 4\nline 5\n")
    editor.highlight_error_line(3)
    assert editor._error_line == 3
    # Typing in editor clears error line
    editor.insertPlainText(" ")
    assert editor._error_line is None

    # Inspector Swatches
    assert len(ARTIFICER_SWATCHES) >= 8
    for hex_code, title in ARTIFICER_SWATCHES:
        assert hex_code.startswith("#")
        assert len(hex_code) == 7
        assert len(title) > 0

    window.close()
