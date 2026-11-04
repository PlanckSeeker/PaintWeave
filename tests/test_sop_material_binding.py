"""Tests for SOP_Material primitive shader binding and multi-material SOP_Merge."""

from __future__ import annotations

import numpy as np

from paintweave.core.graph import Graph
from paintweave.core.registry import NodeRegistry
from paintweave.nodes import register_all_nodes


def test_sop_material_binds_vop_to_primitives_and_merges() -> None:
    register_all_nodes()
    graph = Graph()

    # Stream 1: Grid + RimLight Shader
    grid = NodeRegistry.create_node("SOP_Grid")
    grid.set_param("rows", 4)
    grid.set_param("cols", 4)
    rim = NodeRegistry.create_node("VOP_RimLight")
    mat1 = NodeRegistry.create_node("SOP_Material")
    mat1.set_param("material_label", "terrain_mat")

    # Stream 2: Box + NWShader
    box = NodeRegistry.create_node("SOP_Box")
    nw = NodeRegistry.create_node("VOP_NWShader")
    mat2 = NodeRegistry.create_node("SOP_Material")
    mat2.set_param("material_label", "crystal_mat")

    # Merge
    merge = NodeRegistry.create_node("SOP_Merge")

    for n in (grid, rim, mat1, box, nw, mat2, merge):
        graph.add_node(n)

    graph.add_link(grid.outputs["Geo"], mat1.inputs["Geo"])
    graph.add_link(rim.outputs["Shader"], mat1.inputs["Shader"])

    graph.add_link(box.outputs["Geo"], mat2.inputs["Geo"])
    graph.add_link(nw.outputs["Shader"], mat2.inputs["Shader"])

    graph.add_link(mat1.outputs["Geo"], merge.inputs["Geo1"])
    graph.add_link(mat2.outputs["Geo"], merge.inputs["Geo2"])

    graph.set_display_node(merge.id)
    graph.cook_display_sop(force=True)

    merged_geo = merge.outputs["Geo"].value
    assert merged_geo is not None
    materials = merged_geo.detail["materials"]
    assert len(materials) == 2
    assert materials[0].name == "terrain_mat"
    assert materials[1].name == "crystal_mat"

    mat_ids = merged_geo.primitives["material_id"]
    assert np.sum(mat_ids == 0) == 18  # 3x3 quads * 2 tris = 18
    assert np.sum(mat_ids == 1) == 12  # Box has 12 tris


def test_group_selective_material_assignment() -> None:
    register_all_nodes()
    grid = NodeRegistry.create_node("SOP_Grid")
    grid.set_param("size_x", 6.0)
    grid.set_param("size_z", 6.0)
    grid.set_param("rows", 10)
    grid.set_param("cols", 10)

    grp = NodeRegistry.create_node("SOP_Group")
    grp.set_param("group_name", "center_patch")
    grp.set_param("size_x", 2.5)
    grp.set_param("size_z", 2.5)

    shader = NodeRegistry.create_node("VOP_BayerDither")
    mat = NodeRegistry.create_node("SOP_Material")
    mat.set_param("group", "center_patch")

    grid.outputs["Geo"].connect(grp.inputs["Geo"])
    grp.outputs["Geo"].connect(mat.inputs["Geo"])
    shader.outputs["Shader"].connect(mat.inputs["Shader"])

    mat.cook()
    out_geo = mat.outputs["Geo"].value
    mat_ids = out_geo.primitives["material_id"]
    # Some primitives inside center_patch get material 0, outer primitives stay -1
    assert np.any(mat_ids == 0)
    assert np.any(mat_ids == -1)
