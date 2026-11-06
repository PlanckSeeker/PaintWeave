"""NodeGraphQt canvas customization, typed colored ports, Houdini flag badges, and bidirectional Graph sync."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from NodeGraphQt import BaseNode, NodeGraph
from NodeGraphQt.constants import PipeLayoutEnum

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext
from paintweave.core.graph import Graph
from paintweave.core.registry import (
    GLOBAL_NODE_REGISTRY,
    NodeRegistration,
    NodeRegistry,
)
from paintweave.nodes import register_all_nodes

# Node header colors (R, G, B) by (Context, Category) - Artificer's Atelier Palette
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "Primitives": (18, 115, 96),  # Alchemical Malachite
    "Modifiers": (28, 96, 160),  # Cobalt Brass
    "Attributes & Groups": (36, 122, 142),  # Ether Slate
    "Instancing & Topology": (22, 128, 84),  # Emerald
    "Materials & Shading": (138, 43, 226),  # Arcane Amethyst
    "Lighting": (124, 58, 237),  # Ether Violet
    "Patterns": (168, 85, 247),  # Radiant Orchid
    "Post & Stylized": (192, 38, 211),  # Magenta Glaze
    "Programmable Shaders": (190, 24, 93),  # Philosopher's Crimson
    "Scalars & Vectors": (55, 68, 88),  # Polished Steel
    "Arithmetic": (75, 88, 108),  # Slate
    "Animation & Oscillators": (217, 119, 6),  # Celestial Amber
}

# Port colors (R, G, B) by DataType - Glowing Conduit Spectrum
PORT_TYPE_COLORS: dict[DataType, tuple[int, int, int]] = {
    DataType.GEOMETRY: (52, 211, 153),  # Mint Emerald (SOP Geometry stream)
    DataType.SHADER: (192, 132, 252),  # Glowing Amethyst (VOP Shader stream)
    DataType.FLOAT: (56, 189, 248),  # Laser Cyan
    DataType.INT: (96, 165, 250),  # Sky Blue
    DataType.VEC3: (245, 158, 11),  # Warm Amber / Solar Gold
    DataType.ANY: (160, 174, 192),  # Polished Silver
}


def _make_visual_node_class(reg: NodeRegistration) -> type[BaseNode]:
    """Dynamically construct a `NodeGraphQt.BaseNode` subclass wrapping a registered `NodeBase`."""
    ctx_slug = reg.context.name.lower()
    cls_name = f"Visual_{reg.node_id}"

    def __init__(self: BaseNode) -> None:
        BaseNode.__init__(self)
        self.set_name(reg.title)
        r, g, b = CATEGORY_COLORS.get(reg.category, (45, 55, 72))
        self.set_color(r, g, b)

        for in_spec in reg.node_cls.INPUT_PORTS:
            pr, pg_c, pb = PORT_TYPE_COLORS.get(in_spec.data_type, (148, 163, 184))
            self.add_input(in_spec.name, color=(pr, pg_c, pb), multi_input=False)

        for out_spec in reg.node_cls.OUTPUT_PORTS:
            pr, pg_c, pb = PORT_TYPE_COLORS.get(out_spec.data_type, (56, 189, 248))
            self.add_output(out_spec.name, color=(pr, pg_c, pb), multi_output=True)

    attrs = {
        "__identifier__": f"paintweave.{ctx_slug}",
        "NODE_NAME": reg.title,
        "_pw_type_id": reg.node_id,
        "__init__": __init__,
    }
    return type(cls_name, (BaseNode,), attrs)


class PaintWeaveCanvasController:
    """Manages the `NodeGraphQt.NodeGraph` canvas and keeps it synchronized with a PaintWeave `Graph`."""

    def __init__(
        self,
        on_node_selected: Callable[[NodeBase | None], None] | None = None,
        on_graph_modified: Callable[[], None] | None = None,
        on_connection_error: Callable[[str], None] | None = None,
    ) -> None:
        register_all_nodes()
        self.node_graph = NodeGraph()
        self.node_graph.set_pipe_style(PipeLayoutEnum.CURVED.value)
        self.node_graph.set_background_color(10, 13, 20)
        self.node_graph.set_grid_color(21, 30, 46)

        self.core_graph = Graph(name="Untitled_Diorama")
        self._visual_to_uid: dict[str, str] = {}
        self._uid_to_visual: dict[str, BaseNode] = {}
        self._type_to_identifier: dict[str, str] = {}
        self._sync_depth: int = 0

        self._undo_history: list[dict[str, Any]] = []
        self._redo_history: list[dict[str, Any]] = []
        self._clipboard: dict[str, Any] | None = None
        self._max_undo_steps: int = 64

        self.on_node_selected = on_node_selected
        self.on_graph_modified = on_graph_modified
        self.on_connection_error = on_connection_error

        self._register_all_visual_nodes()
        self._connect_signals()
        self._setup_context_menus()

    @property
    def _syncing(self) -> bool:
        return self._sync_depth > 0

    @contextmanager
    def _sync_guard(self) -> Iterator[None]:
        self._sync_depth += 1
        try:
            yield
        finally:
            self._sync_depth = max(0, self._sync_depth - 1)

    @property
    def widget(self) -> Any:
        return self.node_graph.widget

    def _register_all_visual_nodes(self) -> None:
        for type_id, reg in GLOBAL_NODE_REGISTRY.all_nodes().items():
            vis_cls = _make_visual_node_class(reg)
            self.node_graph.register_node(vis_cls)
            full_ident = f"{vis_cls.__identifier__}.{vis_cls.__name__}"
            self._type_to_identifier[type_id] = full_ident

    def _connect_signals(self) -> None:
        self.node_graph.node_created.connect(self._on_visual_node_created)
        self.node_graph.node_selected.connect(self._on_visual_node_selected)
        self.node_graph.port_connected.connect(self._on_visual_port_connected)
        self.node_graph.port_disconnected.connect(self._on_visual_port_disconnected)
        self.node_graph.node_double_clicked.connect(self._on_visual_node_double_clicked)

    def _setup_context_menus(self) -> None:
        """Configure right-click context menus and Tab quick-search on the NodeGraphQt canvas."""
        try:
            graph_menu = self.node_graph.get_context_menu("graph")
            graph_menu.add_command(
                "Quick Search Operator (Tab)",
                lambda graph: graph.toggle_node_search(),
                "Tab",
            )
            graph_menu.add_separator()
            graph_menu.add_command(
                "Duplicate Selected",
                lambda _g: self.duplicate_selected_nodes(),
            )
            graph_menu.add_command(
                "Copy Selected",
                lambda _g: self.copy_selected_nodes(),
            )
            graph_menu.add_command(
                "Paste Nodes",
                lambda _g: self.paste_nodes(),
            )
            graph_menu.add_command(
                "Delete Selected",
                lambda _g: self.delete_selected_nodes(),
            )
            graph_menu.add_separator()
            graph_menu.add_command(
                "Fit Canvas Selection",
                lambda graph: graph.fit_to_selection(),
            )
        except Exception:
            pass

    @property
    def can_undo(self) -> bool:
        return len(self._undo_history) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_history) > 0

    def clear_undo_history(self) -> None:
        self._undo_history.clear()
        self._redo_history.clear()

    def push_undo_snapshot(self) -> None:
        """Record the current graph state onto the Undo stack prior to a user mutation."""
        if self._syncing:
            return
        self.sync_positions_to_engine()
        snap = copy.deepcopy(self.core_graph.serialize())
        if self._undo_history and self._undo_history[-1] == snap:
            return
        self._undo_history.append(snap)
        if len(self._undo_history) > self._max_undo_steps:
            self._undo_history.pop(0)
        self._redo_history.clear()

    def undo(self) -> bool:
        """Restore the previous graph state from the Undo stack."""
        if not self._undo_history:
            return False
        self.sync_positions_to_engine()
        current_snap = copy.deepcopy(self.core_graph.serialize())
        self._redo_history.append(current_snap)

        prev_snap = self._undo_history.pop()
        restored = Graph.from_dict(prev_snap, preserve_ids=True)
        self.load_graph(restored, fit_view=False, clear_history=False)
        if self.on_node_selected:
            self.on_node_selected(None)
        if self.on_graph_modified:
            self.on_graph_modified()
        return True

    def redo(self) -> bool:
        """Re-apply the next graph state from the Redo stack."""
        if not self._redo_history:
            return False
        self.sync_positions_to_engine()
        current_snap = copy.deepcopy(self.core_graph.serialize())
        self._undo_history.append(current_snap)

        next_snap = self._redo_history.pop()
        restored = Graph.from_dict(next_snap, preserve_ids=True)
        self.load_graph(restored, fit_view=False, clear_history=False)
        if self.on_node_selected:
            self.on_node_selected(None)
        if self.on_graph_modified:
            self.on_graph_modified()
        return True

    def _on_visual_node_created(self, vis_node: BaseNode) -> None:
        """Triggered when a user spawns a node directly via the canvas Tab quick-search menu."""
        if self._syncing or vis_node.id in self._visual_to_uid:
            return
        type_id = getattr(vis_node, "_pw_type_id", None)
        if not type_id:
            return
        self.push_undo_snapshot()
        pos = vis_node.pos()
        core_node = NodeRegistry.create_node(type_id)
        self.core_graph.add_node(core_node, float(pos[0]), float(pos[1]))
        self._visual_to_uid[vis_node.id] = core_node.id
        self._uid_to_visual[core_node.id] = vis_node
        self.refresh_visual_badges()
        if self.on_node_selected:
            self.on_node_selected(core_node)
        if self.on_graph_modified:
            self.on_graph_modified()

    def _on_visual_node_selected(self, vis_node: BaseNode) -> None:
        if self._syncing:
            return
        uid = self._visual_to_uid.get(vis_node.id)
        if uid and uid in self.core_graph.nodes and self.on_node_selected:
            self.on_node_selected(self.core_graph.nodes[uid])

    def _on_visual_node_double_clicked(self, vis_node: BaseNode) -> None:
        """Double-clicking a node on the canvas sets it as the active Display node in its context."""
        if self._syncing:
            return
        uid = self._visual_to_uid.get(vis_node.id)
        if uid and uid in self.core_graph.nodes:
            node = self.core_graph.nodes[uid]
            if node.context in (NodeContext.SOP, NodeContext.VOP):
                self.core_graph.set_display_node(uid)
                self.refresh_visual_badges()
                if self.on_node_selected:
                    self.on_node_selected(node)
                if self.on_graph_modified:
                    self.on_graph_modified()

    def _on_visual_port_connected(self, in_port: Any, out_port: Any) -> None:
        if self._syncing:
            return
        src_vis = out_port.node()
        dst_vis = in_port.node()
        src_uid = self._visual_to_uid.get(src_vis.id)
        dst_uid = self._visual_to_uid.get(dst_vis.id)
        if not src_uid or not dst_uid:
            return

        src_node = self.core_graph.nodes.get(src_uid)
        dst_node = self.core_graph.nodes.get(dst_uid)
        if not src_node or not dst_node:
            return

        src_pin = src_node.outputs.get(out_port.name())
        dst_pin = dst_node.inputs.get(in_port.name())
        if not src_pin or not dst_pin:
            return

        try:
            self.push_undo_snapshot()
            self.core_graph.add_link(src_pin, dst_pin)
            if self.on_graph_modified:
                self.on_graph_modified()
        except Exception as exc:
            with self._sync_guard():
                out_port.disconnect_from(
                    in_port, push_undo=False, emit_signal=False
                )
            if self.on_connection_error:
                self.on_connection_error(str(exc))

    def _on_visual_port_disconnected(self, in_port: Any, _out_port: Any) -> None:
        if self._syncing:
            return
        dst_vis = in_port.node()
        dst_uid = self._visual_to_uid.get(dst_vis.id)
        if dst_uid:
            self.push_undo_snapshot()
            self.core_graph.disconnect_pin(dst_uid, in_port.name())
            if self.on_graph_modified:
                self.on_graph_modified()

    def clear(self, clear_history: bool = True) -> None:
        with self._sync_guard():
            self.node_graph.clear_session()
            self.core_graph = Graph(name="Untitled_Diorama")
            self._visual_to_uid.clear()
            self._uid_to_visual.clear()
        if clear_history:
            self.clear_undo_history()

    def spawn_node(
        self,
        type_id: str,
        pos: tuple[float, float] = (0.0, 0.0),
        custom_name: str | None = None,
        record_undo: bool = True,
        **params: Any,
    ) -> NodeBase:
        """Create a node in both the engine `Graph` and the `NodeGraphQt` canvas."""
        if record_undo:
            self.push_undo_snapshot()
        core_node = NodeRegistry.create_node(type_id, custom_name=custom_name)
        for k, v in params.items():
            if k in core_node.params or k in core_node.inputs:
                core_node.set_param(k, v)

        self.core_graph.add_node(core_node, pos[0], pos[1])
        self._spawn_visual_for_core_node(core_node, pos)
        self.refresh_visual_badges()
        if self.on_graph_modified:
            self.on_graph_modified()
        return core_node

    def _spawn_visual_for_core_node(
        self, core_node: NodeBase, pos: tuple[float, float]
    ) -> BaseNode:
        ident = self._type_to_identifier[core_node.registry_type]
        with self._sync_guard():
            vis_node: BaseNode = self.node_graph.create_node(
                ident,
                name=core_node.name,
                pos=[float(pos[0]), float(pos[1])],
                push_undo=False,
            )
            self._visual_to_uid[vis_node.id] = core_node.id
            self._uid_to_visual[core_node.id] = vis_node
            return vis_node

    def connect_nodes(
        self,
        src_node: NodeBase,
        src_pin_name: str,
        dst_node: NodeBase,
        dst_pin_name: str,
    ) -> None:
        """Programmatically wire two nodes in both core Graph and NodeGraphQt canvas."""
        self.core_graph.add_link(
            src_node.outputs[src_pin_name], dst_node.inputs[dst_pin_name]
        )
        src_vis = self._uid_to_visual.get(src_node.id)
        dst_vis = self._uid_to_visual.get(dst_node.id)
        if src_vis is not None and dst_vis is not None:
            with self._sync_guard():
                out_p = src_vis.get_output(src_pin_name)
                in_p = dst_vis.get_input(dst_pin_name)
                if out_p is not None and in_p is not None:
                    out_p.connect_to(
                        in_p, push_undo=False, emit_signal=False
                    )

    def refresh_visual_badges(self) -> None:
        """Update visual node titles and colors to reflect Display, Template, and Bypass flags."""
        with self._sync_guard():
            for uid, vis_node in self._uid_to_visual.items():
                core_node = self.core_graph.nodes.get(uid)
                if core_node is None:
                    continue

                reg = GLOBAL_NODE_REGISTRY.get_registration(core_node.registry_type)
                base_r, base_g, base_b = CATEGORY_COLORS.get(reg.category, (45, 55, 72))

                badges: list[str] = []
                if core_node.is_bypassed:
                    badges.append("BYP")
                    vis_node.set_color(133, 77, 14)  # Amber-Yellow bypass tint
                elif core_node.is_display_node:
                    badges.append("DISP")
                    # Brighten header slightly for active Display node
                    vis_node.set_color(
                        min(255, int(base_r * 1.35 + 15)),
                        min(255, int(base_g * 1.35 + 25)),
                        min(255, int(base_b * 1.35 + 35)),
                    )
                elif core_node.is_template_node:
                    badges.append("TMPL")
                    vis_node.set_color(157, 23, 77)  # Pink template tint
                else:
                    vis_node.set_color(base_r, base_g, base_b)

                if core_node.is_template_node and "TMPL" not in badges:
                    badges.append("TMPL")

                suffix = f" [{' | '.join(badges)}]" if badges else ""
                vis_node.set_name(f"{core_node.name}{suffix}")

    def load_graph(
        self,
        graph: Graph,
        fit_view: bool = True,
        clear_history: bool = True,
    ) -> None:
        """Populate the visual canvas from a loaded `Graph`."""
        self.clear(clear_history=clear_history)
        self.core_graph = graph
        with self._sync_guard():
            for core_node in graph.nodes.values():
                pos = graph.node_positions.get(core_node.id, (0.0, 0.0))
                self._spawn_visual_for_core_node(core_node, pos)

            for link in list(graph.links.values()):
                src_vis = self._uid_to_visual.get(link.source_pin.node.id)
                dst_vis = self._uid_to_visual.get(link.target_pin.node.id)
                if src_vis is not None and dst_vis is not None:
                    out_p = src_vis.get_output(link.source_pin.name)
                    in_p = dst_vis.get_input(link.target_pin.name)
                    if out_p is not None and in_p is not None:
                        out_p.connect_to(
                            in_p, push_undo=False, emit_signal=False
                        )

        self.refresh_visual_badges()
        if fit_view:
            self.node_graph.fit_to_selection()

    def sync_positions_to_engine(self) -> None:
        """Copy current visual node `(x, y)` coordinates on the canvas back into `core_graph.node_positions`."""
        for uid, vis_node in self._uid_to_visual.items():
            if uid in self.core_graph.nodes:
                pos = vis_node.pos()
                self.core_graph.node_positions[uid] = (float(pos[0]), float(pos[1]))

    def copy_selected_nodes(self) -> int:
        """Copy currently selected visual nodes and their internal links into the clipboard."""
        selected = list(self.node_graph.selected_nodes())
        if not selected:
            return 0
        self.sync_positions_to_engine()
        sel_uids = {
            self._visual_to_uid[v.id]
            for v in selected
            if v.id in self._visual_to_uid
        }
        nodes_payload = []
        for uid in sel_uids:
            node = self.core_graph.nodes.get(uid)
            if node is not None:
                pos = self.core_graph.node_positions.get(uid, (0.0, 0.0))
                nodes_payload.append(
                    {
                        **node.serialize(),
                        "x": pos[0],
                        "y": pos[1],
                    }
                )
        links_payload = [
            link.serialize()
            for link in self.core_graph.links.values()
            if link.source_pin.node.id in sel_uids
            and link.target_pin.node.id in sel_uids
        ]
        self._clipboard = {
            "nodes": copy.deepcopy(nodes_payload),
            "links": copy.deepcopy(links_payload),
        }
        return len(nodes_payload)

    def paste_nodes(self, offset: tuple[float, float] = (50.0, 50.0)) -> list[NodeBase]:
        """Paste nodes and internal links from the clipboard onto the canvas."""
        if not self._clipboard or not self._clipboard.get("nodes"):
            return []
        self.push_undo_snapshot()
        pasted_nodes: list[NodeBase] = []
        id_map: dict[str, NodeBase] = {}

        with self._sync_guard():
            self.node_graph.clear_selection()
            for n_data in self._clipboard["nodes"]:
                new_node = NodeRegistry.create_node(n_data["type"])
                new_node.is_bypassed = bool(n_data.get("is_bypassed", False))
                for p_name, p_val in n_data.get("params", {}).items():
                    if isinstance(p_val, list) and len(p_val) == 3:
                        p_val = tuple(p_val)
                    if p_name in new_node.params:
                        new_node.set_param(p_name, p_val)
                    else:
                        new_node.add_param(p_name, p_val)
                for p_name, p_val in n_data.get("inputs", {}).items():
                    if p_name in new_node.inputs:
                        if isinstance(p_val, list) and len(p_val) == 3:
                            p_val = tuple(p_val)
                        new_node.inputs[p_name].value = p_val

                nx = float(n_data.get("x", 0.0)) + offset[0]
                ny = float(n_data.get("y", 0.0)) + offset[1]
                self.core_graph.add_node(new_node, nx, ny)
                vis_node = self._spawn_visual_for_core_node(new_node, (nx, ny))
                vis_node.set_selected(True)
                id_map[n_data["id"]] = new_node
                pasted_nodes.append(new_node)

            for l_data in self._clipboard.get("links", []):
                src_node = id_map.get(l_data["source_node"])
                dst_node = id_map.get(l_data["target_node"])
                if src_node and dst_node:
                    self.connect_nodes(
                        src_node,
                        l_data["source_pin"],
                        dst_node,
                        l_data["target_pin"],
                    )

        self.refresh_visual_badges()
        if pasted_nodes and self.on_node_selected:
            self.on_node_selected(pasted_nodes[-1])
        if self.on_graph_modified:
            self.on_graph_modified()
        return pasted_nodes

    def duplicate_selected_nodes(self) -> list[NodeBase]:
        """Duplicate currently selected nodes and their internal links (`Ctrl+D`)."""
        if self.copy_selected_nodes() == 0:
            return []
        return self.paste_nodes(offset=(55.0, 55.0))

    def delete_selected_nodes(self) -> None:
        """Remove currently selected visual nodes from both canvas and engine."""
        selected = list(self.node_graph.selected_nodes())
        if not selected:
            return
        self.push_undo_snapshot()
        with self._sync_guard():
            for vis_node in selected:
                uid = self._visual_to_uid.pop(vis_node.id, None)
                if uid:
                    self._uid_to_visual.pop(uid, None)
                    self.core_graph.remove_node(uid)
            self.node_graph.delete_nodes(selected, push_undo=False)

        self.refresh_visual_badges()
        if self.on_node_selected:
            self.on_node_selected(None)
        if self.on_graph_modified:
            self.on_graph_modified()
