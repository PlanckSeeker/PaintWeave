"""Directed Acyclic Graph (DAG) engine with pull-based lazy cooking, flags, and serialization."""

from __future__ import annotations

import json
import os
from typing import Any

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import NodeContext
from paintweave.core.pin import Link, Pin
from paintweave.core.registry import NodeRegistry


class Graph:
    """Manages the network of nodes, cycle validation, pull-based cooking, and JSON (.pw/.nw) persistence."""

    def __init__(self, name: str = "Untitled_Project") -> None:
        self.name: str = name
        self.nodes: dict[str, NodeBase] = {}
        self.links: dict[str, Link] = {}
        self.node_positions: dict[str, tuple[float, float]] = {}

        self.display_sop_id: str | None = None
        self.template_sop_id: str | None = None
        self.display_vop_id: str | None = None

    def add_node(self, node: NodeBase, x: float = 0.0, y: float = 0.0) -> NodeBase:
        self.nodes[node.id] = node
        self.node_positions[node.id] = (float(x), float(y))
        if node.context == NodeContext.SOP and self.display_sop_id is None or node.context == NodeContext.VOP and self.display_vop_id is None:
            self.set_display_node(node.id)
        return node

    def set_display_node(self, node_id: str) -> None:
        """Set the Houdini blue Display Flag on a node within its context (SOP or VOP)."""
        if node_id not in self.nodes:
            return
        target = self.nodes[node_id]
        for n in self.nodes.values():
            if n.context == target.context:
                n.is_display_node = n.id == node_id

        if target.context == NodeContext.SOP:
            self.display_sop_id = node_id
        elif target.context == NodeContext.VOP:
            self.display_vop_id = node_id

    def set_template_node(self, node_id: str | None, enabled: bool = True) -> None:
        """Set or clear the Houdini pink Template/Ghost wireframe flag on a SOP node."""
        for n in self.nodes.values():
            if n.context == NodeContext.SOP:
                n.is_template_node = False

        if enabled and node_id and node_id in self.nodes:
            target = self.nodes[node_id]
            if target.context == NodeContext.SOP:
                target.is_template_node = True
                self.template_sop_id = node_id
                return
        self.template_sop_id = None

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        for pin in node.inputs.values():
            pin.disconnect_all()
        for pin in node.outputs.values():
            pin.disconnect_all()
        self.links = {
            k: v
            for k, v in self.links.items()
            if v.source_pin.node.id != node_id and v.target_pin.node.id != node_id
        }
        del self.nodes[node_id]
        self.node_positions.pop(node_id, None)

        if self.display_sop_id == node_id:
            # Auto-promote another SOP node if available
            remaining_sops = [
                n.id for n in self.nodes.values() if n.context == NodeContext.SOP
            ]
            self.display_sop_id = None
            if remaining_sops:
                self.set_display_node(remaining_sops[-1])
        if self.template_sop_id == node_id:
            self.template_sop_id = None
        if self.display_vop_id == node_id:
            remaining_vops = [
                n.id for n in self.nodes.values() if n.context == NodeContext.VOP
            ]
            self.display_vop_id = None
            if remaining_vops:
                self.set_display_node(remaining_vops[-1])

    def add_link(self, source_pin: Pin, target_pin: Pin) -> Link:
        from paintweave.core.datatypes import PinDirection

        src, tgt = (
            (source_pin, target_pin)
            if source_pin.direction == PinDirection.OUTPUT
            else (target_pin, source_pin)
        )

        # If this exact (src, tgt) link already exists, return it idempotently
        for existing in list(tgt.links):
            if existing.source_pin is src and existing.target_pin is tgt:
                self.links[existing.id] = existing
                return existing

        # Remember any existing links on the single-input target pin so we purge them from self.links
        replaced_links = list(tgt.links)
        for old_link in replaced_links:
            self.links.pop(old_link.id, None)

        link = src.connect(tgt)
        self.links[link.id] = link
        try:
            self.get_evaluation_order()
        except RecursionError as exc:
            # Roll back only the newly created link
            if link in src.links:
                src.links.remove(link)
            if link in tgt.links:
                tgt.links.remove(link)
            self.links.pop(link.id, None)

            # Restore any replaced links
            for old_link in replaced_links:
                if old_link not in old_link.source_pin.links:
                    old_link.source_pin.links.append(old_link)
                if old_link not in old_link.target_pin.links:
                    old_link.target_pin.links.append(old_link)
                self.links[old_link.id] = old_link

            raise ValueError(
                "Invalid connection: creates a cyclic dependency loop."
            ) from exc
        return link

    def disconnect_pin(self, target_node_id: str, target_pin_name: str) -> None:
        if target_node_id not in self.nodes:
            return
        node = self.nodes[target_node_id]
        if target_pin_name in node.inputs:
            pin = node.inputs[target_pin_name]
            removed_ids = {lnk.id for lnk in pin.links}
            pin.disconnect_all()
            for lid in removed_ids:
                self.links.pop(lid, None)

    def get_evaluation_order(self) -> list[NodeBase]:
        """Kahn's Algorithm for topological sorting and cycle detection."""
        in_degree = {node_id: 0 for node_id in self.nodes}
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}

        # Build active edge list directly from node output pins so in_degree and traversal always match
        active_links: dict[str, Link] = {}
        for node_id, node in self.nodes.items():
            for out_pin in node.outputs.values():
                for link in out_pin.links:
                    tgt_id = link.target_pin.node.id
                    if tgt_id in in_degree:
                        in_degree[tgt_id] += 1
                        adjacency[node_id].append(tgt_id)
                        active_links[link.id] = link

        self.links = active_links

        queue = [self.nodes[n_id] for n_id, degree in in_degree.items() if degree == 0]
        sorted_nodes: list[NodeBase] = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for target_node_id in adjacency[node.id]:
                in_degree[target_node_id] -= 1
                if in_degree[target_node_id] == 0:
                    queue.append(self.nodes[target_node_id])

        if len(sorted_nodes) != len(self.nodes):
            raise RecursionError("Cycle detected in node graph!")

        return sorted_nodes

    def has_animated_nodes(self) -> bool:
        """Check whether the graph contains active Time/Oscillator nodes that need frame ticks."""
        for node in self.nodes.values():
            if node.registry_type == "Math_Time" and bool(
                node.params.get("playing", True)
            ):
                return True
        return False

    def mark_time_nodes_dirty(self) -> None:
        """Mark active Math_Time nodes dirty so downstream dependents re-cook on frame tick."""
        for node in self.nodes.values():
            if node.registry_type == "Math_Time" and bool(
                node.params.get("playing", True)
            ):
                node.mark_dirty()

    def evaluate_all(self) -> None:
        for node in self.get_evaluation_order():
            node.cook()

    def cook_display_sop(self, force: bool = False) -> NodeBase | None:
        """Pull-based cook starting strictly from the active Display SOP node."""
        if self.display_sop_id and self.display_sop_id in self.nodes:
            node = self.nodes[self.display_sop_id]
            if force:
                node.mark_dirty()
            node.cook()
            return node
        return None

    def cook_template_sop(self) -> NodeBase | None:
        """Pull-based cook for the optional pink Template/Ghost wireframe SOP node."""
        if self.template_sop_id and self.template_sop_id in self.nodes:
            node = self.nodes[self.template_sop_id]
            node.cook()
            return node
        return None

    def cook_display_vop(self, force: bool = False) -> NodeBase | None:
        """Pull-based cook starting from the active Display VOP node (used as fallback shader)."""
        if self.display_vop_id and self.display_vop_id in self.nodes:
            node = self.nodes[self.display_vop_id]
            if force:
                node.mark_dirty()
            node.cook()
            return node
        return None

    def serialize(self, base_dir: str | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": "1.0.0",
            "nodes": [
                {
                    **node.serialize(base_dir),
                    "x": self.node_positions.get(node.id, (0.0, 0.0))[0],
                    "y": self.node_positions.get(node.id, (0.0, 0.0))[1],
                }
                for node in self.nodes.values()
            ],
            "links": [link.serialize() for link in self.links.values()],
            "display_sop_id": self.display_sop_id,
            "template_sop_id": self.template_sop_id,
            "display_vop_id": self.display_vop_id,
        }

    def save_to_file(self, filepath: str) -> None:
        base_dir = os.path.dirname(os.path.abspath(filepath))
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.serialize(base_dir), f, indent=2)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        base_dir: str | None = None,
        preserve_ids: bool = False,
    ) -> Graph:
        from paintweave.nodes import register_all_nodes

        register_all_nodes()
        graph = cls(name=data.get("name", "Untitled_Project"))
        id_map: dict[str, NodeBase] = {}

        for n_data in data.get("nodes", []):
            node = NodeRegistry.create_node(
                n_data["type"], custom_name=n_data.get("name")
            )
            if preserve_ids and "id" in n_data:
                node.id = str(n_data["id"])
            node.is_bypassed = bool(n_data.get("is_bypassed", False))
            for p_name, p_val in n_data.get("params", {}).items():
                if (
                    base_dir
                    and n_data["type"] == "SOP_File"
                    and p_name == "filepath"
                    and p_val
                    and not os.path.isabs(p_val)
                ):
                    p_val = os.path.normpath(os.path.join(base_dir, p_val)).replace(
                        "\\", "/"
                    )
                if isinstance(p_val, list) and len(p_val) == 3:
                    p_val = tuple(p_val)
                if p_name in node.params:
                    node.set_param(p_name, p_val)
                else:
                    node.add_param(p_name, p_val)
            for p_name, p_val in n_data.get("inputs", {}).items():
                if p_name in node.inputs:
                    if isinstance(p_val, list) and len(p_val) == 3:
                        p_val = tuple(p_val)
                    node.inputs[p_name].value = p_val
            id_map[n_data["id"]] = node
            graph.add_node(
                node, float(n_data.get("x", 0.0)), float(n_data.get("y", 0.0))
            )

        for l_data in data.get("links", []):
            src_node = id_map.get(l_data["source_node"])
            tgt_node = id_map.get(l_data["target_node"])
            if src_node and tgt_node:
                src_pin = src_node.outputs.get(l_data["source_pin"])
                tgt_pin = tgt_node.inputs.get(l_data["target_pin"])
                if src_pin and tgt_pin:
                    try:
                        graph.add_link(src_pin, tgt_pin)
                    except (ValueError, TypeError):
                        pass

        if data.get("display_sop_id") in id_map:
            graph.set_display_node(id_map[data["display_sop_id"]].id)
        if data.get("template_sop_id") in id_map:
            graph.set_template_node(id_map[data["template_sop_id"]].id, True)
        if data.get("display_vop_id") in id_map:
            graph.set_display_node(id_map[data["display_vop_id"]].id)

        return graph

    @classmethod
    def load_from_file(cls, filepath: str) -> Graph:
        base_dir = os.path.dirname(os.path.abspath(filepath))
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if "name" not in data:
            data["name"] = os.path.splitext(os.path.basename(filepath))[0]
        return cls.from_dict(data, base_dir=base_dir, preserve_ids=False)
