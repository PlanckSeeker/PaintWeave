"""Unit tests for PaintWeave v2.0 Chunky Painterly SOPs: SOP_AmbientGradient, SOP_Curvature, and SOP_StylizedDamage."""

import numpy as np

from paintweave.core.registry import NodeRegistry
from paintweave.nodes import register_all_nodes


def test_ambient_gradient_sop_vertical_fade() -> None:
    """Verify SOP_AmbientGradient computes top-to-bottom @vert_fade and tints @Cd."""
    register_all_nodes()
    box = NodeRegistry.create_node("SOP_Box")
    box.set_param("scale_y", 4.0)
    box.cook()

    grad = NodeRegistry.create_node("SOP_AmbientGradient")
    box.outputs["Geo"].connect(grad.inputs["Geo"])
    grad.set_param("blend_mode", "Multiply")
    grad.set_param("tint_top", (1.0, 1.0, 1.0))
    grad.set_param("tint_bottom", (0.2, 0.2, 0.2))
    grad.cook()

    out_geo = grad.outputs["Geo"].value
    assert out_geo is not None
    assert "vert_fade" in out_geo.points
    vf = out_geo.points["vert_fade"]
    assert np.min(vf) == 0.0
    assert np.max(vf) == 1.0

    assert "Cd" in out_geo.points
    cd = out_geo.points["Cd"]
    # Top vertices (Y=+2.0) must have brighter @Cd than bottom vertices (Y=-2.0)
    top_mask = out_geo.points["P"][:, 1] > 1.5
    bottom_mask = out_geo.points["P"][:, 1] < -1.5
    assert np.mean(cd[top_mask]) > np.mean(cd[bottom_mask])


def test_curvature_sop_highlights_and_cavities() -> None:
    """Verify SOP_Curvature isolates convex edges (@highlight) and concave valleys (@cavity)."""
    register_all_nodes()
    # A box has 12 sharp convex 90-degree edges
    box = NodeRegistry.create_node("SOP_Box")
    subdiv = NodeRegistry.create_node("SOP_Subdivide")
    subdiv.set_param("iterations", 1)
    box.outputs["Geo"].connect(subdiv.inputs["Geo"])
    subdiv.cook()

    curv = NodeRegistry.create_node("SOP_Curvature")
    subdiv.outputs["Geo"].connect(curv.inputs["Geo"])
    curv.set_param("tint_cd", True)
    curv.set_param("highlight_intensity", 2.0)
    curv.cook()

    out_geo = curv.outputs["Geo"].value
    assert out_geo is not None
    assert "highlight" in out_geo.points
    assert "cavity" in out_geo.points

    high = out_geo.points["highlight"]
    assert np.max(high) > 0.3
    assert np.min(high) >= 0.0


def test_stylized_damage_sop_notches() -> None:
    """Verify SOP_StylizedDamage carves chunky gouges into convex corners and marks @damaged."""
    register_all_nodes()
    box = NodeRegistry.create_node("SOP_Box")
    subdiv = NodeRegistry.create_node("SOP_Subdivide")
    subdiv.set_param("iterations", 2)
    box.outputs["Geo"].connect(subdiv.inputs["Geo"])
    subdiv.cook()

    in_p = subdiv.outputs["Geo"].value.points["P"].copy()

    dmg = NodeRegistry.create_node("SOP_StylizedDamage")
    subdiv.outputs["Geo"].connect(dmg.inputs["Geo"])
    dmg.set_param("chip_count", 4)
    dmg.set_param("chip_radius", 0.6)
    dmg.set_param("chip_depth", 0.3)
    dmg.cook()

    out_geo = dmg.outputs["Geo"].value
    assert out_geo is not None
    assert "damaged" in out_geo.points
    d_mask = out_geo.points["damaged"]

    # Some points must be affected by damage
    assert np.any(d_mask > 0.0)
    # Positions of damaged points must have moved inward
    assert not np.allclose(out_geo.points["P"], in_p)
    # Vertex normals must be recomputed and valid unit vectors
    normals = out_geo.points["N"]
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def test_v2_arcane_vops_compilation() -> None:
    """Verify VOP_ArcaneRamp, VOP_EdgeGlint, and VOP_BrushImpasto chain and compile cleanly."""
    import moderngl
    register_all_nodes()

    arcane = NodeRegistry.create_node("VOP_ArcaneRamp")
    glint = NodeRegistry.create_node("VOP_EdgeGlint")
    impasto = NodeRegistry.create_node("VOP_BrushImpasto")

    arcane.outputs["Shader"].connect(glint.inputs["Shader"])
    glint.outputs["Shader"].connect(impasto.inputs["Shader"])
    impasto.cook()

    payload = impasto.outputs["Shader"].value
    assert payload is not None
    assert "Arcane Painterly Terminator Ramp" in payload.fragment_body
    assert "Hand-Painted Edge Glint" in payload.fragment_body
    assert "Brush Stroke Impasto" in payload.fragment_body

    ctx = moderngl.create_context(standalone=True)
    vert_src = payload.build_vertex_shader()
    frag_src = payload.build_fragment_shader()
    prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    assert prog is not None
    prog.release()


def test_v2_matcap_vop_presets_and_compilation() -> None:
    """Verify VOP_MatCap generates valid GLSL across all stylized presets and blend modes."""
    import moderngl

    register_all_nodes()
    ctx = moderngl.create_context(standalone=True)

    for preset in ("Chunky Fantasy Gold", "Arcane Gem", "Toon Metal", "Chalky Clay"):
        for blend in ("Overlay", "Multiply", "Add", "Replace", "Mix"):
            matcap = NodeRegistry.create_node("VOP_MatCap")
            matcap.set_param("preset", preset)
            matcap.set_param("blend_mode", blend)
            matcap.set_param("intensity", 1.2)
            matcap.set_param("roughness", 0.2)
            matcap.cook()

            payload = matcap.outputs["Shader"].value
            assert payload is not None
            assert "MatCap & Stylized Metal/Gem Shading" in payload.fragment_body

            vert_src = payload.build_vertex_shader()
            frag_src = payload.build_fragment_shader()
            prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            assert prog is not None
            prog.release()



def test_05_chunky_arcane_diorama_example_load_and_cook() -> None:
    """Verify examples/05_chunky_arcane_diorama.pw deserializes, cooks all SOPs & VOPs, and preserves materials."""
    from pathlib import Path

    from paintweave.core.graph import Graph

    example_path = Path("examples/05_chunky_arcane_diorama.pw")
    assert example_path.exists(), "05_chunky_arcane_diorama.pw must exist in examples/"

    graph = Graph.load_from_file(str(example_path))
    assert graph is not None
    assert graph.display_sop_id is not None

    display_node = graph.cook_display_sop(force=True)
    assert display_node is not None
    assert "Geo" in display_node.outputs
    geo = display_node.outputs["Geo"].value
    assert geo is not None
    assert geo.num_vertices > 0
    assert geo.num_primitives > 0

    # Ensure chunky painterly attributes exist on merged mesh
    for attr in ("vert_fade", "damaged", "highlight", "cavity", "Cd", "N", "P"):
        assert attr in geo.points, f"Expected '{attr}' in point attributes"

    # Ensure 4 multi-material shader assignments were preserved across hierarchical merges
    materials = geo.detail.get("materials", {})
    assert len(materials) == 4
    mat_names = {m.name for m in materials.values()}
    assert "mat_ancient_stone_base" in mat_names
    assert "mat_chunky_monolith" in mat_names
    assert "mat_runic_gold_trim" in mat_names
    assert "mat_arcane_crystal_gem" in mat_names


def test_05_diorama_main_window_viewport_compositor_smoke() -> None:
    """Verify loading 05_chunky_arcane_diorama.pw in PaintWeaveMainWindow cooks and renders through the 3-pass compositor."""
    import os
    from pathlib import Path

    import moderngl
    import numpy as np
    from pyglm import glm
    from PySide6.QtWidgets import QApplication

    from paintweave.engine.renderer.compositor import FrameCompositor
    from paintweave.engine.renderer.mesh_buffer import GPUMeshBuffer
    from paintweave.engine.shader.payload import (
        DEFAULT_FRAGMENT_SHADER,
        DEFAULT_VERTEX_SHADER,
    )
    from paintweave.gui.main_window import PaintWeaveMainWindow

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    assert app is not None

    window = PaintWeaveMainWindow()
    diorama_path = Path("examples/05_chunky_arcane_diorama.pw").resolve()
    window.load_project_file(diorama_path, check_unsaved=False)

    # 1. Verify node graph synchronization
    graph = window.canvas_ctrl.core_graph
    assert len(graph.nodes) >= 40
    assert len(graph.links) >= 30

    display_sop = graph.nodes.get(graph.display_sop_id)
    assert display_sop is not None
    geo = display_sop.outputs["Geo"].value
    assert geo is not None
    assert geo.num_vertices > 0

    # 2. Verify complete 3-Pass Deferred/Hybrid Compositor Render
    ctx = moderngl.create_context(standalone=True)
    default_prog = ctx.program(
        vertex_shader=DEFAULT_VERTEX_SHADER,
        fragment_shader=DEFAULT_FRAGMENT_SHADER,
    )
    comp = FrameCompositor(ctx)
    comp.resize(512, 512)

    buf = GPUMeshBuffer(ctx)
    errors = buf.upload(geo, default_prog, depth_program=comp.shadow_prog)
    assert errors == []

    bmin = geo.detail.get("bounds_min")
    bmax = geo.detail.get("bounds_max")
    lsm = comp.get_light_space_matrix(bmin, bmax)
    lsm_bytes = lsm.to_bytes()
    model_bytes = glm.mat4(1.0).to_bytes()

    # Pass 1: Directional Shadow Pass
    comp.begin_shadow_pass()
    buf.render_depth(lsm_bytes, model_bytes)

    # Pass 2: G-Buffer MRT Pass
    comp.begin_geometry_pass(512, 512)
    comp.shadow_tex.use(location=0)
    mvp = glm.perspective(glm.radians(45.0), 1.0, 0.1, 100.0) * glm.lookAt(
        glm.vec3(12, 14, 12), glm.vec3(0, 3, 0), glm.vec3(0, 1, 0)
    )
    buf.render(
        mvp.to_bytes(),
        model_bytes,
        0.0,
        camera_pos=(12.0, 14.0, 12.0),
        light_space_bytes=lsm_bytes,
        shadow_enabled=True,
    )

    # Pass 3: Sobel Ink Outline & Composite Pass
    target_fbo = ctx.simple_framebuffer((512, 512))
    comp.composite_to_target(target_fbo, 512, 512)

    raw = target_fbo.read(components=4)
    arr = np.frombuffer(raw, dtype=np.uint8)
    assert np.count_nonzero(arr) > 1000
    assert np.mean(arr) > 10.0
    window.close()



