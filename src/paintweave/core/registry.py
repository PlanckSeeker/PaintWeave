"""Central node catalog supporting both decorator registration and NodeGraphQt generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import NodeContext


@dataclass(frozen=True)
class NodeRegistration:
    """Metadata record for a registered operator class."""

    node_id: str
    title: str
    category: str
    context: NodeContext
    description: str
    node_cls: type[NodeBase]


class NodeRegistry:
    """Central registry mapping string operator IDs (e.g. 'SOP_Grid') to NodeBase classes."""

    _nodes: dict[str, NodeRegistration] = {}
    _node_counter: dict[str, int] = {}

    @classmethod
    def register(
        cls,
        node_type: str,
        node_class: type[NodeBase],
        title: str | None = None,
        category: str | None = None,
    ) -> None:
        resolved_title = (
            title or getattr(node_class, "title", "") or node_type.replace("_", " ")
        )
        resolved_cat = category or getattr(node_class, "category", "General")
        resolved_ctx = getattr(node_class, "context", NodeContext.MATH)
        desc = (node_class.__doc__ or "").strip()

        cls._nodes[node_type] = NodeRegistration(
            node_id=node_type,
            title=resolved_title,
            category=resolved_cat,
            context=resolved_ctx,
            description=desc,
            node_cls=node_class,
        )
        cls._node_counter.setdefault(node_type, 0)

    @classmethod
    def get_registration(cls, node_type: str) -> NodeRegistration:
        if node_type not in cls._nodes:
            raise ValueError(f"Node type '{node_type}' is not registered.")
        return cls._nodes[node_type]

    @classmethod
    def get_node_class(cls, node_type: str) -> type[NodeBase]:
        return cls.get_registration(node_type).node_cls

    @classmethod
    def create_node(cls, node_type: str, custom_name: str | None = None) -> NodeBase:
        """Instantiate a node with a unique sequential name."""
        reg = cls.get_registration(node_type)
        cls._node_counter[node_type] = cls._node_counter.get(node_type, 0) + 1
        instance_name = custom_name or f"{node_type}_{cls._node_counter[node_type]}"
        node = reg.node_cls(instance_name)
        node.registry_type = node_type
        return node

    @classmethod
    def all_nodes(cls) -> dict[str, NodeRegistration]:
        return dict(cls._nodes)

    @classmethod
    def get_all_registered(cls) -> list[str]:
        return sorted(cls._nodes.keys())

    @classmethod
    def by_category(cls) -> dict[str, list[NodeRegistration]]:
        """Group registrations by 'Context — Category' for the studio NodePaletteDock."""
        grouped: dict[str, list[NodeRegistration]] = {}
        for _, reg in sorted(
            cls._nodes.items(),
            key=lambda kv: (kv[1].context.value, kv[1].category, kv[0]),
        ):
            ctx_prefix = reg.context.name
            header = f"{ctx_prefix} : {reg.category}"
            grouped.setdefault(header, []).append(reg)
        return grouped

    @classmethod
    def get_by_context(cls) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for name, reg in sorted(cls._nodes.items()):
            label = (
                f"{reg.context.value} - {reg.category}"
                if reg.category != "General"
                else reg.context.value
            )
            grouped.setdefault(label, []).append(name)
        return grouped


def register_node(
    node_id: str,
    title: str | None = None,
    category: str | None = None,
) -> Callable[[type[NodeBase]], type[NodeBase]]:
    """Class decorator to register a NodeBase subclass into NodeRegistry."""

    def _decorator(cls: type[NodeBase]) -> type[NodeBase]:
        NodeRegistry.register(node_id, cls, title=title, category=category)
        return cls

    return _decorator


GLOBAL_NODE_REGISTRY = NodeRegistry
