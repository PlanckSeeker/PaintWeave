"""Typed input/output pins and directed graph links with pull-based lazy evaluation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from paintweave.core.datatypes import DataType, PinDirection

if TYPE_CHECKING:
    from paintweave.core.base_node import NodeBase


class Link:
    """Directed connection bridging an output pin to an input pin."""

    def __init__(self, source_pin: Pin, target_pin: Pin) -> None:
        self.id: str = str(uuid.uuid4())
        self.source_pin: Pin = source_pin
        self.target_pin: Pin = target_pin

    def serialize(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source_node": self.source_pin.node.id,
            "source_pin": self.source_pin.name,
            "target_node": self.target_pin.node.id,
            "target_pin": self.target_pin.name,
        }


class Pin:
    """A socket on a node that accepts or provides typed data."""

    def __init__(
        self,
        node: NodeBase,
        name: str,
        direction: PinDirection,
        data_type: DataType,
        default_value: Any = None,
        description: str = "",
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.node: NodeBase = node
        self.name: str = name
        self.direction: PinDirection = direction
        self.data_type: DataType = data_type
        self.default_value: Any = default_value
        self.value: Any = default_value
        self.description: str = description
        self.links: list[Link] = []

    @property
    def is_connected(self) -> bool:
        return len(self.links) > 0

    def connect(self, other_pin: Pin) -> Link:
        """Validate types/directions and create a directed Link."""
        if self.direction == other_pin.direction:
            raise ValueError(
                f"Cannot connect {self.direction.name} to {other_pin.direction.name}"
            )

        numeric_types = {DataType.FLOAT, DataType.INT}
        types_compatible = (
            self.data_type == other_pin.data_type
            or self.data_type == DataType.ANY
            or other_pin.data_type == DataType.ANY
            or (self.data_type in numeric_types and other_pin.data_type in numeric_types)
        )
        if not types_compatible:
            raise TypeError(
                f"Port type mismatch: {self.data_type.name} cannot connect to {other_pin.data_type.name}"
            )

        source, target = (
            (self, other_pin)
            if self.direction == PinDirection.OUTPUT
            else (other_pin, self)
        )

        # Single-input rule: replace any existing connection on target input pin
        if len(target.links) > 0:
            target.disconnect_all()

        link = Link(source, target)
        source.links.append(link)
        target.links.append(link)

        target.node.mark_dirty()
        return link

    def disconnect_all(self) -> None:
        """Remove all links connected to this pin."""
        for link in list(self.links):
            other_pin = (
                link.target_pin
                if self.direction == PinDirection.OUTPUT
                else link.source_pin
            )
            if link in other_pin.links:
                other_pin.links.remove(link)
            if link in self.links:
                self.links.remove(link)

        if self.direction == PinDirection.INPUT:
            self.value = self.default_value
            self.node.mark_dirty()

    def get_data(self) -> Any:
        """
        Pull-based data retrieval (Houdini cooking model).
        If connected to an upstream output pin, ensures the upstream node is cooked first.
        """
        if self.direction == PinDirection.INPUT and self.is_connected:
            source_pin = self.links[0].source_pin
            source_pin.node.cook()
            return source_pin.value
        return self.value
