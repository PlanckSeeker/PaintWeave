"""Main studio IDE window for PaintWeave: Dockable Palette, NodeGraphQt Canvas, ModernGL 3D Viewport, Inspector, and Tabbed Workbench."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
)

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import NodeContext
from paintweave.core.graph import Graph
from paintweave.engine.geometry.compute_cuda import get_compute_summary
from paintweave.gui.canvas.controller import PaintWeaveCanvasController
from paintweave.gui.widgets.code_studio import CodeStudioPanel
from paintweave.gui.widgets.inspector_dock import NodePropertyInspector
from paintweave.gui.widgets.palette_dock import NodePaletteDock
from paintweave.gui.widgets.spreadsheet import GeometrySpreadsheet
from paintweave.gui.widgets.telemetry import CookTelemetryPanel
from paintweave.gui.widgets.viewport_3d import Viewport3D


class PaintWeaveMainWindow(QMainWindow):
    """Main desktop IDE window for PaintWeave Studio v1.0.0."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "PaintWeave Studio v1.0.0 — Procedural Geometry (SOP) & Multi-Material Shader (VOP) IDE"
        )
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(
                min(1680, max(1200, int(avail.width() * 0.94))),
                min(980, max(720, int(avail.height() * 0.92))),
            )
        else:
            self.resize(1600, 920)

        self.current_file_path: str | None = None
        self._selected_node: NodeBase | None = None
        self._spawn_offset: float = 0.0
        self._timeline_playing: bool = True
        self._is_modified: bool = False

        self._load_stylesheet()
        self._init_components()
        self._init_docks()
        self._init_menus_and_toolbar()
        self._init_statusbar()

        # 60 FPS Timeline Timer — only re-cooks graph when timeline is playing AND animated nodes exist
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_timeline_tick)
        self.play_timer.start(16)

        # Load default example if present, otherwise build default multi-material diorama
        default_example = self._examples_dir() / "02_multi_material_crystal_shrine.pw"
        if default_example.exists():
            self.load_project_file(default_example)
        else:
            self._build_default_showcase_network()
        self._is_modified = False

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent.parent

    def _examples_dir(self) -> Path:
        return self._project_root() / "examples"

    def _load_stylesheet(self) -> None:
        qss_path = self._project_root() / "assets" / "styles" / "dark_studio.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _init_components(self) -> None:
        # Central split workspace: NodeGraphQt Canvas (Left) + ModernGL 3D Viewport (Right)
        self.central_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self.canvas_ctrl = PaintWeaveCanvasController(
            on_node_selected=self._on_node_selected,
            on_graph_modified=self._on_canvas_graph_modified,
            on_connection_error=self._on_connection_error,
        )
        self.viewport = Viewport3D(self)
        self.viewport.shader_error.connect(self._on_shader_error)

        self.central_splitter.addWidget(self.canvas_ctrl.widget)
        self.central_splitter.addWidget(self.viewport)
        self.central_splitter.setSizes([660, 740])
        self.setCentralWidget(self.central_splitter)

        # Left Dock: Operator Palette
        self.palette_widget = NodePaletteDock(self)
        self.palette_widget.node_spawn_requested.connect(
            self._on_spawn_node_from_palette
        )

        # Right Dock: Property Inspector
        self.inspector_widget = NodePropertyInspector(self)
        self.inspector_widget.undo_snapshot_requested.connect(
            self.canvas_ctrl.push_undo_snapshot
        )
        self.inspector_widget.parameter_changed.connect(self._on_node_param_changed)
        self.inspector_widget.display_flag_toggled.connect(
            self._on_display_flag_changed
        )
        self.inspector_widget.template_flag_toggled.connect(
            self._on_template_flag_changed
        )
        self.inspector_widget.bypass_flag_toggled.connect(self._on_bypass_flag_changed)

        # Bottom Tabbed Workbench
        self.workbench_tabs = QTabWidget(self)
        self.spreadsheet_panel = GeometrySpreadsheet(self)
        self.code_studio = CodeStudioPanel(self)
        self.telemetry_panel = CookTelemetryPanel(self)

        self.code_studio.undo_snapshot_requested.connect(
            self.canvas_ctrl.push_undo_snapshot
        )
        self.code_studio.code_applied.connect(self._on_node_param_changed)

        self.workbench_tabs.addTab(
            self.spreadsheet_panel, "📊 Geometry & Material Spreadsheet"
        )
        self.workbench_tabs.addTab(self.code_studio, "💻 Wrangle & NWShader Studio")
        self.workbench_tabs.addTab(
            self.telemetry_panel, "⏱️ Cook Profiler & CUDA Telemetry"
        )

    def _init_docks(self) -> None:
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )

        # Left Dock: Operator Palette
        self.dock_palette = QDockWidget("Operator Catalog (SOP / VOP / Math)", self)
        self.dock_palette.setWidget(self.palette_widget)
        self.dock_palette.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_palette)

        # Right Dock: Parameter Inspector
        self.dock_inspector = QDockWidget("Node Parameter Inspector", self)
        self.dock_inspector.setWidget(self.inspector_widget)
        self.dock_inspector.setMinimumWidth(290)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_inspector)

        # Bottom Dock: Spreadsheet, Code Studio & Telemetry
        self.dock_workbench = QDockWidget(
            "Geometry Spreadsheet, Shader Studio & Telemetry", self
        )
        self.dock_workbench.setWidget(self.workbench_tabs)
        self.dock_workbench.setMinimumHeight(250)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_workbench)

    def _init_menus_and_toolbar(self) -> None:
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        act_new = QAction("New Procedural Graph", self)
        act_new.setIconText("New")
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self.file_new)
        file_menu.addAction(act_new)

        act_open = QAction("Open Project (.pw / .nw)...", self)
        act_open.setIconText("Open")
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.file_open)
        file_menu.addAction(act_open)

        act_save = QAction("Save Project (.pw)...", self)
        act_save.setIconText("Save")
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.file_save)
        file_menu.addAction(act_save)

        file_menu.addSeparator()
        act_export_obj = QAction("Export Display Mesh (.obj)...", self)
        act_export_obj.setIconText("Export OBJ")
        act_export_obj.triggered.connect(self.export_mesh_obj)
        file_menu.addAction(act_export_obj)

        act_render_png = QAction("📸 Render Viewport to Image (.png)... (F12)", self)
        act_render_png.setIconText("📸 Render")
        act_render_png.setShortcut(QKeySequence("F12"))
        act_render_png.triggered.connect(self.export_render_png)
        file_menu.addAction(act_render_png)

        file_menu.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Edit Menu (Undo / Redo / Copy / Paste / Duplicate / Delete)
        edit_menu = menubar.addMenu("&Edit")

        self.act_undo = QAction("↩ Undo (Ctrl+Z)", self)
        self.act_undo.setIconText("↩ Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.edit_undo)
        edit_menu.addAction(self.act_undo)

        self.act_redo = QAction("↪ Redo (Ctrl+Y)", self)
        self.act_redo.setIconText("↪ Redo")
        self.act_redo.setShortcuts(
            [QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")]
        )
        self.act_redo.triggered.connect(self.edit_redo)
        edit_menu.addAction(self.act_redo)

        edit_menu.addSeparator()

        act_copy = QAction("Copy Selected Nodes (Ctrl+C)", self)
        act_copy.setShortcut(QKeySequence.StandardKey.Copy)
        act_copy.triggered.connect(self.edit_copy)
        edit_menu.addAction(act_copy)

        act_paste = QAction("Paste Nodes (Ctrl+V)", self)
        act_paste.setShortcut(QKeySequence.StandardKey.Paste)
        act_paste.triggered.connect(self.edit_paste)
        edit_menu.addAction(act_paste)

        act_dup = QAction("Duplicate Selected Nodes (Ctrl+D)", self)
        act_dup.setIconText("Duplicate")
        act_dup.setShortcut(QKeySequence("Ctrl+D"))
        act_dup.triggered.connect(self.edit_duplicate)
        edit_menu.addAction(act_dup)

        edit_menu.addSeparator()
        act_del = QAction("Delete Selected Nodes (Del)", self)
        act_del.setShortcut(QKeySequence.StandardKey.Delete)
        act_del.triggered.connect(self.canvas_ctrl.delete_selected_nodes)
        edit_menu.addAction(act_del)

        # Examples Menu
        self.examples_menu = menubar.addMenu("&Examples")
        self._populate_examples_menu()

        # Graph & Simulation Menu
        sim_menu = menubar.addMenu("&Network")
        act_cook = QAction("⚡ Force Cook Display Network (F5)", self)
        act_cook.setIconText("⚡ Cook")
        act_cook.setShortcut(QKeySequence("F5"))
        act_cook.triggered.connect(lambda: self.cook_active_network(force=True))
        sim_menu.addAction(act_cook)

        act_fit = QAction("🔍 Fit Node Canvas", self)
        act_fit.setIconText("🔍 Fit")
        act_fit.triggered.connect(self.canvas_ctrl.node_graph.fit_to_selection)
        sim_menu.addAction(act_fit)

        act_frame_geo = QAction("🎯 Frame 3D Geometry (F)", self)
        act_frame_geo.triggered.connect(self.viewport.frame_active_geometry)
        sim_menu.addAction(act_frame_geo)

        # Quick-Add Menu
        nodes_menu = menubar.addMenu("&Quick Add")
        for label, nid in [
            ("SOP: Grid", "SOP_Grid"),
            ("SOP: Curve (Helix / Arc / Wave)", "SOP_Curve"),
            ("SOP: Sweep Along Curve", "SOP_Sweep"),
            ("SOP: Mountain (CUDA)", "SOP_Mountain"),
            ("SOP: Smooth (Laplacian)", "SOP_Smooth"),
            ("SOP: UV Project (Planar/Cyl/Sph/Box)", "SOP_UVProject"),
            ("SOP: Raycast / Project to Surface", "SOP_Raycast"),
            ("SOP: Copy to Points (Normal Aligned)", "SOP_CopyToPoints"),
            ("SOP: Assign Material (SOP_Material)", "SOP_Material"),
            ("SOP: Point Wrangle (Python)", "SOP_Wrangle"),
            ("SOP: Ambient Gradient (Blizzard)", "SOP_AmbientGradient"),
            ("SOP: Curvature & Cavity", "SOP_Curvature"),
            ("SOP: Stylized Damage (Chunky Notches)", "SOP_StylizedDamage"),
            ("SOP: Merge", "SOP_Merge"),
            ("VOP: NWShader (Spatial GLSL)", "VOP_NWShader"),
            ("VOP: Simplex Noise 3D (FBM)", "VOP_SimplexNoise3D"),
            ("VOP: UV Checkerboard", "VOP_UVChecker"),
            ("VOP: Iso Sun Light", "VOP_IsoLight"),
            ("VOP: Rim Light", "VOP_RimLight"),
            ("VOP: Bayer 4x4 Dither", "VOP_BayerDither"),
            ("VOP: Arcane Painterly Ramp", "VOP_ArcaneRamp"),
            ("VOP: Hand-Painted Edge Glint", "VOP_EdgeGlint"),
            ("VOP: Brush Stroke Impasto", "VOP_BrushImpasto"),
            ("VOP: MatCap & Stylized Metal/Gem", "VOP_MatCap"),
        ]:
            act = QAction(label, self)
            act.triggered.connect(
                lambda _=False, t=nid: self._on_spawn_node_from_palette(t)
            )
            nodes_menu.addAction(act)

        # Help Menu
        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About PaintWeave Studio", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

        # Main Studio Toolbar
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        toolbar.addAction(act_new)
        toolbar.addAction(act_open)
        toolbar.addAction(act_save)
        toolbar.addSeparator()
        toolbar.addAction(self.act_undo)
        toolbar.addAction(self.act_redo)
        toolbar.addAction(act_dup)
        toolbar.addSeparator()
        toolbar.addAction(act_cook)

        self.act_play = QAction("⏸ Pause Timeline", self)
        self.act_play.setIconText("⏸ Pause")
        self.act_play.setCheckable(True)
        self.act_play.setChecked(True)
        self.act_play.toggled.connect(self._on_toggle_timeline_play)
        toolbar.addAction(self.act_play)

        toolbar.addSeparator()
        act_mat_quick = QAction("🎨 + Assign Material SOP", self)
        act_mat_quick.setIconText("🎨 + Material")
        act_mat_quick.triggered.connect(
            lambda: self._on_spawn_node_from_palette("SOP_Material")
        )
        toolbar.addAction(act_mat_quick)

        act_shader_quick = QAction("✨ + NWShader VOP", self)
        act_shader_quick.setIconText("✨ + NWShader")
        act_shader_quick.triggered.connect(
            lambda: self._on_spawn_node_from_palette("VOP_NWShader")
        )
        toolbar.addAction(act_shader_quick)

        toolbar.addSeparator()
        toolbar.addAction(act_fit)
        toolbar.addAction(act_export_obj)
        toolbar.addAction(act_render_png)

    def _populate_examples_menu(self) -> None:
        self.examples_menu.clear()
        ex_dir = self._examples_dir()
        if ex_dir.exists():
            for ex_file in sorted(ex_dir.glob("*.pw")):
                act = QAction(ex_file.stem.replace("_", " ").title(), self)
                act.triggered.connect(
                    lambda _=False, p=ex_file: self.load_project_file(p)
                )
                self.examples_menu.addAction(act)

    def _init_statusbar(self) -> None:
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.device_badge = QLabel(
            f"  {get_compute_summary()}  |  OpenGL 4.3 Core MSAAx4  "
        )
        self.device_badge.setStyleSheet("color: #38bdf8; font-weight: 600;")
        self.status_bar.addPermanentWidget(self.device_badge)
        self.status_bar.showMessage(
            "Ready — Double-click any node on the canvas to set its Display Flag, press Tab on the canvas for quick search, or double-click in the left Palette."
        )

    # -------------------------------------------------------------------------
    # Default Multi-Material Showcase Network
    # -------------------------------------------------------------------------
    def _build_default_showcase_network(self) -> None:
        """Build a multi-material procedural diorama showcasing SOP_Material + SOP_Merge."""
        self.canvas_ctrl.clear()

        # Stream 1: Procedural Canyon Terrain + Stylized VOP Material
        grid = self.canvas_ctrl.spawn_node(
            "SOP_Grid", pos=(-320, -240), size_x=8.0, size_z=8.0, rows=48, cols=48
        )
        mountain = self.canvas_ctrl.spawn_node(
            "SOP_Mountain", pos=(-320, -80), frequency=1.6, octaves=3, roughness=0.45
        )
        mountain.inputs["Amplitude"].value = 1.15

        iso_light = self.canvas_ctrl.spawn_node(
            "VOP_IsoLight", pos=(-620, -220), intensity=1.4
        )
        rim_light = self.canvas_ctrl.spawn_node(
            "VOP_RimLight", pos=(-620, -90), rim_intensity=1.9
        )
        dither = self.canvas_ctrl.spawn_node(
            "VOP_BayerDither", pos=(-620, 40), strength=0.10
        )

        mat_terrain = self.canvas_ctrl.spawn_node(
            "SOP_Material", pos=(-320, 90), material_label="mat_terrain_dither"
        )

        # Stream 2: Scattered Bioluminescent Crystal Monoliths + NWShader Material
        scatter = self.canvas_ctrl.spawn_node(
            "SOP_Scatter", pos=(80, -180), num_points=28, seed=108
        )
        rand_scale = self.canvas_ctrl.spawn_node(
            "SOP_AttributeRandomize",
            pos=(80, -40),
            attribute="pscale",
            min_value=0.35,
            max_value=0.95,
        )
        crystal_mesh = self.canvas_ctrl.spawn_node(
            "SOP_Cylinder", pos=(340, -180), radius=0.22, height=1.35, cols=6, rows=2
        )
        copy_pts = self.canvas_ctrl.spawn_node(
            "SOP_CopyToPoints", pos=(200, 90), master_scale=1.0
        )
        nw_shader = self.canvas_ctrl.spawn_node("VOP_NWShader", pos=(480, -20))
        mat_crystal = self.canvas_ctrl.spawn_node(
            "SOP_Material", pos=(200, 230), material_label="mat_crystal_bio"
        )

        # Final Multi-Material Merge
        merge = self.canvas_ctrl.spawn_node("SOP_Merge", pos=(-60, 380))

        # Wire Stream 1
        self.canvas_ctrl.connect_nodes(grid, "Geo", mountain, "Geo")
        self.canvas_ctrl.connect_nodes(iso_light, "Shader", rim_light, "Shader")
        self.canvas_ctrl.connect_nodes(rim_light, "Shader", dither, "Shader")
        self.canvas_ctrl.connect_nodes(mountain, "Geo", mat_terrain, "Geo")
        self.canvas_ctrl.connect_nodes(dither, "Shader", mat_terrain, "Shader")

        # Wire Stream 2
        self.canvas_ctrl.connect_nodes(mountain, "Geo", scatter, "Geo")
        self.canvas_ctrl.connect_nodes(scatter, "Geo", rand_scale, "Geo")
        self.canvas_ctrl.connect_nodes(crystal_mesh, "Geo", copy_pts, "Source")
        self.canvas_ctrl.connect_nodes(rand_scale, "Geo", copy_pts, "TargetPoints")
        self.canvas_ctrl.connect_nodes(copy_pts, "Geo", mat_crystal, "Geo")
        self.canvas_ctrl.connect_nodes(nw_shader, "Shader", mat_crystal, "Shader")

        # Wire into Merge & set Display Flag
        self.canvas_ctrl.connect_nodes(mat_terrain, "Geo", merge, "Geo1")
        self.canvas_ctrl.connect_nodes(mat_crystal, "Geo", merge, "Geo2")

        self.canvas_ctrl.core_graph.set_display_node(merge.id)
        self.canvas_ctrl.core_graph.set_display_node(dither.id)
        self.canvas_ctrl.refresh_visual_badges()
        self.canvas_ctrl.node_graph.fit_to_selection()
        self.cook_active_network(force=True)

    # -------------------------------------------------------------------------
    # Edit / Undo / Redo / Clipboard Operations
    # -------------------------------------------------------------------------
    def _rebind_selected_after_snapshot(self, prev_selected_id: str | None) -> None:
        if prev_selected_id is not None:
            restored = self.canvas_ctrl.core_graph.nodes.get(prev_selected_id)
            self._on_node_selected(restored)
        else:
            self.inspector_widget.inspect_node(None)
            self.code_studio.bind_node(None)

    def edit_undo(self) -> None:
        prev_selected_id = (
            self._selected_node.id if self._selected_node is not None else None
        )
        if self.canvas_ctrl.undo():
            self._rebind_selected_after_snapshot(prev_selected_id)
            self.status_bar.showMessage("Undo applied.", 2500)
        else:
            self.status_bar.showMessage("Nothing to undo.", 2000)

    def edit_redo(self) -> None:
        prev_selected_id = (
            self._selected_node.id if self._selected_node is not None else None
        )
        if self.canvas_ctrl.redo():
            self._rebind_selected_after_snapshot(prev_selected_id)
            self.status_bar.showMessage("Redo applied.", 2500)
        else:
            self.status_bar.showMessage("Nothing to redo.", 2000)

    def edit_copy(self) -> None:
        count = self.canvas_ctrl.copy_selected_nodes()
        if count > 0:
            self.status_bar.showMessage(f"Copied {count} node(s) to clipboard.", 2500)

    def edit_paste(self) -> None:
        pasted = self.canvas_ctrl.paste_nodes()
        if pasted:
            self.status_bar.showMessage(f"Pasted {len(pasted)} node(s).", 2500)

    def edit_duplicate(self) -> None:
        duped = self.canvas_ctrl.duplicate_selected_nodes()
        if duped:
            self.status_bar.showMessage(f"Duplicated {len(duped)} node(s).", 2500)

    # -------------------------------------------------------------------------
    # Event Handlers & Cooking
    # -------------------------------------------------------------------------
    def _on_canvas_graph_modified(self) -> None:
        self._is_modified = True
        self.cook_active_network()

    def _on_toggle_timeline_play(self, playing: bool) -> None:
        self._timeline_playing = playing
        self.act_play.setText("⏸ Pause Timeline" if playing else "▶ Play Timeline")
        self.act_play.setIconText("⏸ Pause" if playing else "▶ Play")

    def _on_timeline_tick(self) -> None:
        if not self._timeline_playing:
            return
        graph = self.canvas_ctrl.core_graph
        if graph.has_animated_nodes():
            graph.mark_time_nodes_dirty()
            self.cook_active_network()
        elif self.viewport.has_animated_shaders():
            # Only redraw on idle ticks if an active shader references the TIME uniform
            self.viewport.update()

    def _on_spawn_node_from_palette(self, type_id: str) -> None:
        self._spawn_offset = (self._spawn_offset + 45.0) % 270.0
        node = self.canvas_ctrl.spawn_node(
            type_id, pos=(self._spawn_offset, self._spawn_offset), record_undo=True
        )
        self._is_modified = True
        self._on_node_selected(node)
        self.status_bar.showMessage(f"Spawned operator '{node.name}' ({type_id})")

    def _on_node_selected(self, node: NodeBase | None) -> None:
        self._selected_node = node
        if node is not None:
            node.cook()
        self.inspector_widget.inspect_node(node)
        self.code_studio.bind_node(node)

        if (
            node is not None
            and node.context == NodeContext.SOP
            and "Geo" in node.outputs
        ):
            self.spreadsheet_panel.set_geometry(node.outputs["Geo"].value)

    def _on_node_param_changed(self, node: NodeBase) -> None:
        self._is_modified = True
        prev_dyn = dict(node.dynamic_param_specs)
        node.cook()
        self.cook_active_network()
        if node.dynamic_param_specs != prev_dyn:
            self.inspector_widget.inspect_node(node)
        else:
            self.inspector_widget.update_cook_badge(node)
        self.code_studio.refresh_compiled_preview()

    def _on_display_flag_changed(self, node_id: str) -> None:
        self.canvas_ctrl.push_undo_snapshot()
        self._is_modified = True
        self.canvas_ctrl.core_graph.set_display_node(node_id)
        self.canvas_ctrl.refresh_visual_badges()
        self.cook_active_network()
        if self._selected_node is not None:
            self.inspector_widget.inspect_node(self._selected_node)

    def _on_template_flag_changed(self, node_id: str, enabled: bool) -> None:
        self.canvas_ctrl.push_undo_snapshot()
        self._is_modified = True
        self.canvas_ctrl.core_graph.set_template_node(node_id, enabled)
        self.canvas_ctrl.refresh_visual_badges()
        self.cook_active_network()

    def _on_bypass_flag_changed(self, _node_id: str, _bypassed: bool) -> None:
        self._is_modified = True
        self.canvas_ctrl.refresh_visual_badges()
        self.cook_active_network()

    def _on_shader_error(self, err_msg: str) -> None:
        if err_msg:
            self.status_bar.setStyleSheet(
                "color: #ffffff; background-color: #991b1b; font-weight: bold;"
            )
            self.status_bar.showMessage(err_msg)

    def _on_connection_error(self, err_msg: str) -> None:
        if err_msg:
            self.status_bar.showMessage(f"Connection rejected: {err_msg}", 4000)

    def cook_active_network(self, force: bool = False) -> None:
        """Pull execution from active Display SOP, Template SOP, and fallback Display VOP."""
        graph = self.canvas_ctrl.core_graph
        try:
            sop_node = graph.cook_display_sop(force=force)
            template_node = graph.cook_template_sop()
            vop_node = graph.cook_display_vop(force=force)

            geo = (
                sop_node.outputs["Geo"].value
                if (sop_node and "Geo" in sop_node.outputs)
                else None
            )
            template_geo = (
                template_node.outputs["Geo"].value
                if (template_node and "Geo" in template_node.outputs)
                else None
            )

            fallback_shader = None
            if (
                sop_node
                and "Shader" in sop_node.outputs
                and sop_node.outputs["Shader"].value is not None
            ):
                fallback_shader = sop_node.outputs["Shader"].value
            elif vop_node and "Shader" in vop_node.outputs:
                fallback_shader = vop_node.outputs["Shader"].value

            error_msg = None
            for n in (sop_node, vop_node, self._selected_node):
                if n is not None and n.error_message:
                    error_msg = n.error_message
                    break

            self.viewport.update_scene(geo, fallback_shader, template_geo=template_geo)

            # Show selected SOP geometry in spreadsheet if a SOP is selected, otherwise Display SOP
            if self._selected_node and self._selected_node.context == NodeContext.SOP:
                self._selected_node.cook()
                self.spreadsheet_panel.set_geometry(
                    self._selected_node.outputs["Geo"].value
                )
            else:
                self.spreadsheet_panel.set_geometry(geo)

            self.code_studio.refresh_compiled_preview()
            self.telemetry_panel.refresh_telemetry(graph)

            if error_msg:
                self.status_bar.setStyleSheet(
                    "color: #ffffff; background-color: #991b1b; font-weight: bold;"
                )
                self.status_bar.showMessage(error_msg)
            else:
                self.status_bar.setStyleSheet("")
                if sop_node:
                    self.status_bar.showMessage(
                        f"Cooked Display SOP '{sop_node.name}' ({sop_node.last_cook_time_ms:.2f} ms)",
                        3000,
                    )
        except Exception as exc:
            self.status_bar.setStyleSheet(
                "color: #ffffff; background-color: #991b1b; font-weight: bold;"
            )
            self.status_bar.showMessage(f"Cook Error: {exc}")

    # -------------------------------------------------------------------------
    # File I/O Actions
    # -------------------------------------------------------------------------
    def _maybe_save_changes(self) -> bool:
        """If the graph has unsaved modifications and window is visible, prompt user to Save/Discard/Cancel."""
        if not (self._is_modified and self.isVisible()):
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Procedural Network Changes",
            "The current procedural graph has unsaved modifications.\nDo you want to save before proceeding?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            self.file_save()
            return not self._is_modified
        return reply != QMessageBox.StandardButton.Cancel

    def file_new(self) -> None:
        if not self._maybe_save_changes():
            return
        self.current_file_path = None
        self._is_modified = False
        self.setWindowTitle("PaintWeave Studio v1.0.0 — [Untitled_Project.pw]")
        self.canvas_ctrl.clear()
        self.viewport.update_scene(None, None)
        self.spreadsheet_panel.set_geometry(None)
        self.inspector_widget.inspect_node(None)
        self.code_studio.bind_node(None)
        self.telemetry_panel.refresh_telemetry(self.canvas_ctrl.core_graph)
        self.status_bar.showMessage("Created new empty procedural graph.")

    def file_open(self) -> None:
        if not self._maybe_save_changes():
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open PaintWeave / NodeWeave Project",
            str(self._examples_dir()),
            "Procedural Graphs (*.pw *.nw *.json);;All Files (*)",
        )
        if filepath:
            self.load_project_file(Path(filepath), check_unsaved=False)

    def load_project_file(self, filepath: Path, check_unsaved: bool = True) -> None:
        if check_unsaved and not self._maybe_save_changes():
            return
        try:
            loaded_graph = Graph.load_from_file(str(filepath))
            self.current_file_path = str(filepath)
            self._is_modified = False
            self.setWindowTitle(f"PaintWeave Studio v1.0.0 — [{filepath.name}]")
            self.canvas_ctrl.load_graph(loaded_graph)
            self.cook_active_network(force=True)
            self.status_bar.showMessage(
                f"Loaded project '{filepath.name}' ({len(loaded_graph.nodes)} nodes)."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error Loading Project", str(exc))

    def file_save(self) -> None:
        if not self.current_file_path:
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save PaintWeave Project",
                str(self._examples_dir() / "untitled_scene.pw"),
                "PaintWeave Projects (*.pw);;NodeWeave Legacy (*.nw);;All Files (*)",
            )
            if not filepath:
                return
            if not (filepath.endswith(".pw") or filepath.endswith(".nw")):
                filepath += ".pw"
            self.current_file_path = filepath
            self.setWindowTitle(
                f"PaintWeave Studio v1.0.0 — [{os.path.basename(filepath)}]"
            )

        try:
            self.canvas_ctrl.sync_positions_to_engine()
            self.canvas_ctrl.core_graph.save_to_file(self.current_file_path)
            self._is_modified = False
            self._populate_examples_menu()
            self.status_bar.showMessage(f"Saved project to {self.current_file_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def export_mesh_obj(self) -> None:
        graph = self.canvas_ctrl.core_graph
        sop_node = graph.cook_display_sop()
        if not sop_node or "Geo" not in sop_node.outputs:
            self.status_bar.showMessage("No active Display SOP geometry to export.")
            return

        geo = sop_node.outputs["Geo"].value
        if geo is None or geo.num_primitives == 0:
            self.status_bar.showMessage("Display geometry is empty.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Display Mesh as Wavefront OBJ",
            "",
            "Wavefront OBJ (*.obj);;All Files (*)",
        )
        if not filepath:
            return
        if not filepath.endswith(".obj"):
            filepath += ".obj"

        try:
            geo.export_to_obj(filepath)
            self.status_bar.showMessage(
                f"Exported mesh ({geo.num_primitives:,} triangles) to {filepath}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "OBJ Export Error", str(exc))

    def export_render_png(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Render 3D Viewport to High-Resolution PNG",
            "paintweave_render_1080p.png",
            "PNG Image (*.png);;All Files (*)",
        )
        if not filepath:
            return
        if not filepath.endswith(".png"):
            filepath += ".png"

        try:
            self.viewport.render_to_image(
                filepath, width=1920, height=1080, include_grid=False
            )
            self.status_bar.showMessage(
                f"Rendered 1920x1080 frame to {filepath}", 4000
            )
        except Exception as exc:
            QMessageBox.warning(self, "Render Export Error", str(exc))

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._maybe_save_changes():
            event.ignore()
            return
        super().closeEvent(event)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About PaintWeave Studio",
            "<b>PaintWeave Studio v1.0.0</b><br><br>"
            "An open-source procedural geometry (SOP) and multi-material shader (VOP / NWShader) "
            "desktop IDE inspired by SideFX Houdini.<br><br>"
            f"<b>Backend:</b> {get_compute_summary()}<br>"
            "<b>UI Stack:</b> PySide6, NodeGraphQt, ModernGL, PyGLM, pyqtgraph.",
        )
