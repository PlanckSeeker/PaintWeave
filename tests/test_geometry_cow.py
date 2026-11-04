"""Tests for columnar Geometry Copy-on-Write (CoW), normals, and Wavefront OBJ round-trip."""

from __future__ import annotations

import numpy as np

from paintweave.core.registry import NodeRegistry
from paintweave.engine.geometry.geometry import Geometry
from paintweave.nodes import register_all_nodes


def test_geometry_copy_on_write_pointer_sharing() -> None:
    register_all_nodes()
    grid_node = NodeRegistry.create_node("SOP_Grid")
    grid_node.set_param("rows", 16)
    grid_node.set_param("cols", 16)
    grid_node.cook()

    geo1: Geometry = grid_node.outputs["Geo"].value
    geo2: Geometry = geo1.clone()

    # Unmodified attribute columns share underlying memory pointers
    assert np.shares_memory(geo1.points["P"], geo2.points["P"])
    assert np.shares_memory(geo1.points["uv"], geo2.points["uv"])
    assert np.shares_memory(geo1.primitives["triangles"], geo2.primitives["triangles"])

    # Replacing @P on geo2 breaks pointer sharing for @P while keeping @uv shared
    new_p = geo2.points["P"] + 1.0
    geo2.set_point_attrib("P", new_p)
    assert not np.shares_memory(geo1.points["P"], geo2.points["P"])
    assert np.shares_memory(geo1.points["uv"], geo2.points["uv"])


def test_obj_export_and_import_roundtrip(tmp_path) -> None:
    register_all_nodes()
    box = NodeRegistry.create_node("SOP_Box")
    box.cook()
    geo: Geometry = box.outputs["Geo"].value

    obj_file = tmp_path / "test_box.obj"
    geo.export_to_obj(str(obj_file))
    assert obj_file.exists()

    loaded = Geometry.load_from_obj(str(obj_file))
    assert loaded.num_points == geo.num_points
    assert loaded.num_primitives == geo.num_primitives
