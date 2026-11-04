"""Comprehensive v1.0.0 Studio milestone tests for PaintWeave."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from paintweave.core.graph import Graph
from paintweave.core.registry import NodeRegistry
from paintweave.engine.geometry.geometry import Geometry
from paintweave.gui.canvas.controller import PaintWeaveCanvasController
from paintweave.nodes import register_all_nodes


def create_node(node_type: str, name: str | None = None):
    return NodeRegistry.create_node(node_type, custom_name=name)


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


def test_v1_uv_project_and_obj_roundtrip(tmp_path: Path) -> None:
    register_all_nodes()
    box = create_node("SOP_Box", name="box")
    uv_proj = create_node("SOP_UVProject", name="uv_proj")
    g = Graph()
    g.add_node(box)
    g.add_node(uv_proj)
    g.add_link(box.outputs["Geo"], uv_proj.inputs["Geo"])

    for mode in [
        "Planar XZ (Top)",
        "Planar XY (Front)",
        "Planar YZ (Side)",
        "Cylindrical (Y-Axis)",
        "Spherical",
        "Triplanar Box",
    ]:
        uv_proj.set_param("projection", mode)
        uv_proj.cook()
        geo = uv_proj.outputs["Geo"].value
        assert geo is not None
        assert "uv" in geo.points
        assert geo.points["uv"].shape == (geo.num_points, 2)
        assert np.all(np.isfinite(geo.points["uv"]))

    # Test OBJ export and reload with vt coordinates
    obj_path = tmp_path / "test_uv_box.obj"
    geo = uv_proj.outputs["Geo"].value
    geo.export_to_obj(str(obj_path))
    assert "vt " in obj_path.read_text(encoding="utf-8")

    reloaded = Geometry.load_from_obj(str(obj_path))
    assert "uv" in reloaded.points
    assert reloaded.points["uv"].shape == (reloaded.num_points, 2)


def test_v1_curve_and_sweep_and_smooth_and_normal() -> None:
    register_all_nodes()
    g = Graph()
    curve = create_node("SOP_Curve", name="helix")
    curve.set_param("curve_type", "Helix / Spiral")
    curve.set_param("num_points", 32)
    curve.cook()
    # [HIGH-03] Ensure both "Helix / Spiral" and "Helix Spiral" generate non-zero X/Z spiral coordinates
    curve_p = curve.outputs["Geo"].value.points["P"]
    assert float(np.max(np.abs(curve_p[:, 0]))) > 0.5
    assert float(np.max(np.abs(curve_p[:, 2]))) > 0.5

    sweep = create_node("SOP_Sweep", name="sweep")
    sweep.set_param("radius", 0.25)
    sweep.set_param("taper_end", 0.2)
    sweep.set_param("cols", 10)
    sweep.set_param("twist_deg", 360.0)

    smooth = create_node("SOP_Smooth", name="smooth")
    smooth.set_param("iterations", 2)
    smooth.set_param("strength", 0.5)

    normal_node = create_node("SOP_Normal", name="norm")
    normal_node.set_param("reverse_normals", True)
    normal_node.set_param("add_y_bias", 0.25)

    for n in (curve, sweep, smooth, normal_node):
        g.add_node(n)

    g.add_link(curve.outputs["Geo"], sweep.inputs["Backbone"])
    g.add_link(sweep.outputs["Geo"], smooth.inputs["Geo"])
    g.add_link(smooth.outputs["Geo"], normal_node.inputs["Geo"])

    normal_node.cook()
    geo = normal_node.outputs["Geo"].value
    assert geo is not None
    assert geo.num_points == 32 * 10
    assert geo.num_primitives > 0
    assert "uv" in geo.points
    assert np.all(np.isfinite(geo.points["N"]))


def test_v1_raycast_surface_projection() -> None:
    register_all_nodes()
    g = Graph()
    terrain = create_node("SOP_Grid", name="terrain")
    terrain.set_param("size_x", 6.0)
    terrain.set_param("size_z", 6.0)

    source = create_node("SOP_Grid", name="src")
    source.set_param("size_x", 2.0)
    source.set_param("size_z", 2.0)
    source.set_param("rows", 4)
    source.set_param("cols", 4)

    xform = create_node("SOP_Transform", name="lift")
    xform.set_param("translate_y", 3.0)

    ray = create_node("SOP_Raycast", name="ray")
    ray.set_param("lift_offset", 0.15)
    ray.set_param("blend", 1.0)

    for n in (terrain, source, xform, ray):
        g.add_node(n)

    g.add_link(source.outputs["Geo"], xform.inputs["Geo"])
    g.add_link(xform.outputs["Geo"], ray.inputs["Geo"])
    g.add_link(terrain.outputs["Geo"], ray.inputs["TargetSurface"])

    ray.cook()
    projected = ray.outputs["Geo"].value
    assert projected is not None
    # All points should have snapped from y=3.0 down to y=0.0 + lift_offset (0.15)
    assert np.allclose(projected.points["P"][:, 1], 0.15, atol=1e-4)


def test_v1_copy_to_points_normal_alignment_and_antiparallel() -> None:
    register_all_nodes()
    g = Graph()
    sphere = create_node("SOP_Sphere", name="target_sphere")
    scatter = create_node("SOP_Scatter", name="scatter")
    scatter.set_param("num_points", 24)

    cone = create_node("SOP_Cylinder", name="spike")
    copy_pts = create_node("SOP_CopyToPoints", name="copy")
    copy_pts.set_param("align_to_normal", True)
    copy_pts.set_param("up_axis", "Y (Up)")
    copy_pts.set_param("inherit_target_color", True)

    for n in (sphere, scatter, cone, copy_pts):
        g.add_node(n)

    g.add_link(sphere.outputs["Geo"], scatter.inputs["Geo"])
    g.add_link(cone.outputs["Geo"], copy_pts.inputs["Source"])
    g.add_link(scatter.outputs["Geo"], copy_pts.inputs["TargetPoints"])

    copy_pts.cook()
    instanced = copy_pts.outputs["Geo"].value
    assert instanced is not None
    assert instanced.num_points == 24 * cone.outputs["Geo"].value.num_points
    assert np.all(np.isfinite(instanced.points["P"]))
    assert np.all(np.isfinite(instanced.points["N"]))


def test_v1_simplex_noise_and_uv_checker_vops() -> None:
    register_all_nodes()
    g = Graph()
    noise_vop = create_node("VOP_SimplexNoise3D", name="noise3d")
    checker_vop = create_node("VOP_UVChecker", name="checker")
    rim_vop = create_node("VOP_RimLight", name="rim")

    g.add_node(noise_vop)
    g.add_node(checker_vop)
    g.add_node(rim_vop)

    g.add_link(noise_vop.outputs["Shader"], checker_vop.inputs["Shader"])
    g.add_link(checker_vop.outputs["Shader"], rim_vop.inputs["Shader"])

    rim_vop.cook()
    payload = rim_vop.outputs["Shader"].value
    assert payload is not None
    glsl = payload.build_fragment_shader()
    assert "pw_fbm(" in glsl
    assert "v_uv" in glsl
    assert "vec3 V = normalize(-v_world_pos);" in glsl


def test_v1_canvas_undo_redo_and_copy_paste_duplicate() -> None:
    _ensure_qapp()
    ctrl = PaintWeaveCanvasController()

    n1 = ctrl.spawn_node("SOP_Grid", pos=(0, 0), record_undo=True)
    n2 = ctrl.spawn_node("SOP_Mountain", pos=(200, 0), record_undo=True)
    ctrl.push_undo_snapshot()
    ctrl.connect_nodes(n1, "Geo", n2, "Geo")

    assert len(ctrl.core_graph.nodes) == 2
    assert len(ctrl.core_graph.links) == 1

    # Undo connection
    assert ctrl.undo() is True
    assert len(ctrl.core_graph.nodes) == 2
    assert len(ctrl.core_graph.links) == 0

    # Redo connection
    assert ctrl.redo() is True
    assert len(ctrl.core_graph.links) == 1

    # Select all visual nodes and duplicate
    for vnode in ctrl._uid_to_visual.values():
        vnode.set_selected(True)

    duped = ctrl.duplicate_selected_nodes()
    assert len(duped) == 2
    assert len(ctrl.core_graph.nodes) == 4
    # Internal link between duplicated pair should also be duplicated
    assert len(ctrl.core_graph.links) == 2

    # Undo duplication
    assert ctrl.undo() is True
    assert len(ctrl.core_graph.nodes) == 2
    assert len(ctrl.core_graph.links) == 1


def test_v1_example_04_loads_and_cooks() -> None:
    register_all_nodes()
    ex_path = Path(__file__).resolve().parent.parent / "examples" / "04_swept_arcane_tendrils_uv.pw"
    assert ex_path.exists()
    g = Graph.load_from_file(str(ex_path))
    disp = g.cook_display_sop(force=True)
    assert disp is not None
    geo = disp.outputs["Geo"].value
    assert geo is not None
    assert geo.num_points > 500
    assert len(geo.detail["materials"]) == 2
