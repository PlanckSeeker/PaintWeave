"""Tests for Graph topological sorting, cycle prevention, bypass flag, and .pw JSON persistence."""

from __future__ import annotations

import pytest

from paintweave.core.graph import Graph
from paintweave.core.registry import NodeRegistry
from paintweave.nodes import register_all_nodes


def test_graph_cycle_detection_and_bypass_flag(tmp_path) -> None:
    register_all_nodes()
    g = Graph("TestGraph")

    n1 = g.add_node(NodeRegistry.create_node("Math_Add"), 10, 20)
    n2 = g.add_node(NodeRegistry.create_node("Math_Multiply"), 100, 120)

    g.add_link(n1.outputs["Result"], n2.inputs["A"])
    with pytest.raises(ValueError, match="cyclic"):
        g.add_link(n2.outputs["Result"], n1.inputs["A"])

    # Test SOP Bypass flag passthrough
    grid = g.add_node(NodeRegistry.create_node("SOP_Grid"), 0, 0)
    extrude = g.add_node(NodeRegistry.create_node("SOP_Extrude"), 0, 100)
    g.add_link(grid.outputs["Geo"], extrude.inputs["Geo"])

    extrude.cook()
    extruded_pts = extrude.outputs["Geo"].value.num_points

    extrude.set_bypassed(True)
    extrude.cook()
    bypassed_pts = extrude.outputs["Geo"].value.num_points
    assert bypassed_pts == grid.outputs["Geo"].value.num_points
    assert extruded_pts > bypassed_pts

    # Save and reload
    save_path = tmp_path / "scene.pw"
    g.save_to_file(str(save_path))
    loaded = Graph.load_from_file(str(save_path))
    assert len(loaded.nodes) == len(g.nodes)
    assert len(loaded.links) == len(g.links)
