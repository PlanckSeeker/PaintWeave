"""Declarative & reactive base node class supporting SOP, VOP, and Math operators."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, ClassVar

from paintweave.core.datatypes import (
    CodeString,
    DataType,
    NodeContext,
    ParamSpec,
    PinDirection,
    PortSpec,
)
from paintweave.core.pin import Pin


class NodeBase:
    """
    Base class for all SOP, VOP, and Math nodes in PaintWeave.

    Supports both:
      1. Declarative class-level `INPUT_PORTS`, `OUTPUT_PORTS`, and `PARAM_SPECS`
         (used by NodeGraphQt visual factory and schema-driven Property Inspector)
      2. Imperative `add_input`, `add_output`, `add_param` (for dynamic shader uniform reflection)
    """

    context: ClassVar[NodeContext] = NodeContext.MATH
    category: ClassVar[str] = "General"
    title: ClassVar[str] = ""

    INPUT_PORTS: ClassVar[list[PortSpec]] = []
    OUTPUT_PORTS: ClassVar[list[PortSpec]] = []
    PARAM_SPECS: ClassVar[list[ParamSpec]] = []

    def __init__(self, name: str) -> None:
        self.id: str = str(uuid.uuid4())
        self.name: str = name
        self.type_name: str = self.__class__.__name__
        self.registry_type: str = self.type_name

        self.inputs: dict[str, Pin] = {}
        self.outputs: dict[str, Pin] = {}
        self.params: dict[str, Any] = {}
        self.dynamic_param_specs: dict[str, ParamSpec] = {}

        # Houdini Node Flags
        self._is_dirty: bool = True
        self.is_display_node: bool = False
        self.is_template_node: bool = False
        self.is_bypassed: bool = False

        # Telemetry & Diagnostics
        self.last_cook_time_ms: float = 0.0
        self.error_message: str | None = None

        # Instantiate declarative ports & parameters
        for in_spec in self.INPUT_PORTS:
            self.add_input(
                in_spec.name, in_spec.data_type, in_spec.default, in_spec.description
            )
        for out_spec in self.OUTPUT_PORTS:
            self.add_output(
                out_spec.name,
                out_spec.data_type,
                out_spec.default,
                out_spec.description,
            )
        for p_spec in self.PARAM_SPECS:
            self.add_param(p_spec.name, p_spec.default)

    def all_param_specs(self) -> list[ParamSpec]:
        """Return class-level `PARAM_SPECS` combined with any runtime-reflected uniform specs."""
        specs = list(self.PARAM_SPECS)
        known = {s.name for s in specs}
        for name, dspec in self.dynamic_param_specs.items():
            if name not in known:
                specs.append(dspec)
                known.add(name)
        # Fallback for any raw add_param calls without explicit ParamSpec
        for name, val in self.params.items():
            if name not in known:
                specs.append(self._infer_param_spec(name, val))
        return specs

    @staticmethod
    def _infer_param_spec(name: str, val: Any) -> ParamSpec:
        if isinstance(val, bool):
            return ParamSpec(name=name, param_type="bool", default=val)
        if isinstance(val, int):
            return ParamSpec(
                name=name, param_type="int", default=val, min_val=1, max_val=2048
            )
        if isinstance(val, float):
            return ParamSpec(
                name=name,
                param_type="float",
                default=val,
                min_val=-1000.0,
                max_val=1000.0,
            )
        if isinstance(val, CodeString):
            return ParamSpec(name=name, param_type="code", default=val)
        if isinstance(val, tuple) and len(val) == 3:
            return ParamSpec(name=name, param_type="color", default=val)
        if name == "filepath":
            return ParamSpec(name=name, param_type="filepath", default=str(val))
        return ParamSpec(name=name, param_type="str", default=str(val))

    def add_input(
        self,
        name: str,
        data_type: DataType,
        default_value: Any = None,
        description: str = "",
    ) -> Pin:
        pin = Pin(self, name, PinDirection.INPUT, data_type, default_value, description)
        self.inputs[name] = pin
        return pin

    def add_output(
        self,
        name: str,
        data_type: DataType,
        default_value: Any = None,
        description: str = "",
    ) -> Pin:
        pin = Pin(
            self, name, PinDirection.OUTPUT, data_type, default_value, description
        )
        self.outputs[name] = pin
        return pin

    def add_param(
        self, name: str, default_value: Any, spec: ParamSpec | None = None
    ) -> None:
        self.params[name] = default_value
        if spec is not None:
            self.dynamic_param_specs[name] = spec

    def set_param(self, name: str, value: Any) -> None:
        """Update a parameter or unconnected input pin and propagate dirty flags downstream."""
        if name in self.params:
            if self.params[name] != value:
                self.params[name] = value
                self.mark_dirty()
        elif name in self.inputs:
            self.inputs[name].value = value
            self.mark_dirty()
        else:
            raise KeyError(
                f"Parameter or input '{name}' not found on node '{self.name}'."
            )

    def get_param(self, name: str) -> Any:
        return self.params[name]

    def set_bypassed(self, bypassed: bool) -> None:
        if self.is_bypassed != bypassed:
            self.is_bypassed = bypassed
            self.mark_dirty()

    def mark_dirty(self) -> None:
        """Push Phase: flag this node and all downstream dependents as dirty."""
        if not self._is_dirty:
            self._is_dirty = True
            for pin in self.outputs.values():
                for link in pin.links:
                    link.target_pin.node.mark_dirty()

    def evaluate(self) -> None:
        """Overridden by subclasses to perform node-specific computation."""

    def _bypass_passthrough(self) -> None:
        """When bypassed (Houdini yellow flag), pass first matching input data straight to output."""
        for out_pin in self.outputs.values():
            matched = False
            for in_pin in self.inputs.values():
                if in_pin.data_type == out_pin.data_type:
                    out_pin.value = in_pin.get_data()
                    matched = True
                    break
            if not matched:
                out_pin.value = out_pin.default_value

    def cook(self) -> None:
        """
        Pull Phase: lazily cooks upstream connected nodes first, then evaluates
        this node only if its dirty flag is set.
        """
        if not self._is_dirty:
            return

        for in_pin in self.inputs.values():
            if in_pin.is_connected:
                in_pin.links[0].source_pin.node.cook()

        t0 = time.perf_counter()
        if self.is_bypassed:
            self._bypass_passthrough()
            self.error_message = None
        else:
            self.evaluate()
        self.last_cook_time_ms = (time.perf_counter() - t0) * 1000.0
        self._is_dirty = False

    def compute(self) -> None:
        """Alias for cook()."""
        self.cook()

    def serialize(self, base_dir: str | None = None) -> dict[str, Any]:
        params_dict = dict(self.params)

        if self.registry_type == "SOP_File" and "filepath" in params_dict and base_dir:
            path = str(params_dict["filepath"])
            if path and os.path.isabs(path):
                try:
                    params_dict["filepath"] = os.path.relpath(path, base_dir).replace(
                        "\\", "/"
                    )
                except ValueError:
                    pass

        return {
            "id": self.id,
            "type": self.registry_type,
            "name": self.name,
            "is_bypassed": self.is_bypassed,
            "params": params_dict,
            "inputs": {
                name: pin.value
                for name, pin in self.inputs.items()
                if not pin.is_connected
                and isinstance(pin.value, (int, float, str, list, tuple))
            },
        }
