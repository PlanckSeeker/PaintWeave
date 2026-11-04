"""Math, Scalar, Vector, and Procedural Time Operators."""

from __future__ import annotations

import numpy as np

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.core.registry import NodeRegistry


class FloatGeneratorNode(NodeBase):
    """Outputs a scalar float value."""

    context = NodeContext.MATH
    category = "Scalars & Vectors"
    title = "Float Constant"

    OUTPUT_PORTS = [
        PortSpec("Out", DataType.FLOAT, 1.0, "Scalar float output."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "value",
            "float",
            1.0,
            label="Value",
            min_val=-1000.0,
            max_val=1000.0,
            step=0.1,
        ),
    ]

    def evaluate(self) -> None:
        self.outputs["Out"].value = float(self.get_param("value"))


class AddNode(NodeBase):
    """Adds two scalar floats together (A + B)."""

    context = NodeContext.MATH
    category = "Arithmetic"
    title = "Add (A + B)"

    INPUT_PORTS = [
        PortSpec("A", DataType.FLOAT, 0.0, "First operand."),
        PortSpec("B", DataType.FLOAT, 0.0, "Second operand."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Result", DataType.FLOAT, 0.0, "Sum (A + B)."),
    ]

    def evaluate(self) -> None:
        val_a = float(self.inputs["A"].get_data())
        val_b = float(self.inputs["B"].get_data())
        self.outputs["Result"].value = val_a + val_b


class MultiplyNode(NodeBase):
    """Multiplies two scalar floats together (A * B)."""

    context = NodeContext.MATH
    category = "Arithmetic"
    title = "Multiply (A * B)"

    INPUT_PORTS = [
        PortSpec("A", DataType.FLOAT, 1.0, "First operand."),
        PortSpec("B", DataType.FLOAT, 1.0, "Second operand."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Result", DataType.FLOAT, 1.0, "Product (A * B)."),
    ]

    def evaluate(self) -> None:
        val_a = float(self.inputs["A"].get_data())
        val_b = float(self.inputs["B"].get_data())
        self.outputs["Result"].value = val_a * val_b


class TimeNode(NodeBase):
    """Outputs a continuous time scalar with deterministic 60 FPS frame stepping."""

    context = NodeContext.MATH
    category = "Animation & Oscillators"
    title = "Time ($T)"

    OUTPUT_PORTS = [
        PortSpec("Time", DataType.FLOAT, 0.0, "Elapsed procedural time in seconds."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "speed_multiplier",
            "float",
            1.0,
            label="Speed Multiplier",
            min_val=-10.0,
            max_val=10.0,
            step=0.1,
        ),
        ParamSpec(
            "current_time",
            "float",
            0.0,
            label="Current Time (s)",
            min_val=0.0,
            max_val=100000.0,
            step=0.1,
        ),
        ParamSpec("playing", "bool", True, label="Active on Timeline Play"),
    ]

    def evaluate(self) -> None:
        speed = float(self.get_param("speed_multiplier"))
        is_playing = bool(self.get_param("playing"))
        curr_t = float(self.get_param("current_time"))

        if is_playing:
            curr_t += (1.0 / 60.0) * speed
            self.params["current_time"] = curr_t

        self.outputs["Time"].value = curr_t


class Vec3GeneratorNode(NodeBase):
    """Outputs a 3-component vector (X, Y, Z)."""

    context = NodeContext.MATH
    category = "Scalars & Vectors"
    title = "Vec3 Constant"

    OUTPUT_PORTS = [
        PortSpec("Vec3", DataType.VEC3, (0.0, 1.0, 0.0), "3D vector output."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "x", "float", 0.0, label="X", min_val=-100.0, max_val=100.0, step=0.1
        ),
        ParamSpec(
            "y", "float", 1.0, label="Y", min_val=-100.0, max_val=100.0, step=0.1
        ),
        ParamSpec(
            "z", "float", 0.0, label="Z", min_val=-100.0, max_val=100.0, step=0.1
        ),
    ]

    def evaluate(self) -> None:
        self.outputs["Vec3"].value = (
            float(self.get_param("x")),
            float(self.get_param("y")),
            float(self.get_param("z")),
        )


class FloatToVec3Node(NodeBase):
    """Packs 3 scalar float inputs (X, Y, Z) into a single Vec3."""

    context = NodeContext.MATH
    category = "Scalars & Vectors"
    title = "Float to Vec3"

    INPUT_PORTS = [
        PortSpec("X", DataType.FLOAT, 0.0, "X component."),
        PortSpec("Y", DataType.FLOAT, 0.0, "Y component."),
        PortSpec("Z", DataType.FLOAT, 0.0, "Z component."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Vec3", DataType.VEC3, (0.0, 0.0, 0.0), "Packed Vec3."),
    ]

    def evaluate(self) -> None:
        self.outputs["Vec3"].value = (
            float(self.inputs["X"].get_data()),
            float(self.inputs["Y"].get_data()),
            float(self.inputs["Z"].get_data()),
        )


class ArcaneOscillatorNode(NodeBase):
    """Computes a harmonic wave-interference Vec3 pulse for dynamic lighting and procedural motion."""

    context = NodeContext.MATH
    category = "Animation & Oscillators"
    title = "Harmonic Oscillator"

    INPUT_PORTS = [
        PortSpec("Time", DataType.FLOAT, 0.0, "Time input in seconds."),
    ]
    OUTPUT_PORTS = [
        PortSpec(
            "PulseVec3", DataType.VEC3, (1.0, 1.0, 1.0), "Oscillating RGB / XYZ vector."
        ),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "base_frequency",
            "float",
            2.0,
            label="Base Frequency",
            min_val=0.05,
            max_val=20.0,
            step=0.1,
        ),
        ParamSpec(
            "chaos_factor",
            "float",
            1.5,
            label="Harmonic Ratio",
            min_val=0.1,
            max_val=10.0,
            step=0.1,
        ),
    ]

    def evaluate(self) -> None:
        t = float(self.inputs["Time"].get_data())
        freq = float(self.get_param("base_frequency"))
        chaos = float(self.get_param("chaos_factor"))

        r = 0.5 + 0.5 * np.sin(t * freq) * np.cos(t * freq * chaos * 0.61803)
        g = 0.5 + 0.5 * np.sin(t * freq * 1.33) * np.cos(t * freq * chaos * 1.41421)
        b = 0.5 + 0.5 * np.sin(t * freq * 0.8) * np.cos(t * freq * chaos * 2.71828)

        self.outputs["PulseVec3"].value = (float(r), float(g), float(b))


def register_math_nodes() -> None:
    NodeRegistry.register("Math_Float", FloatGeneratorNode)
    NodeRegistry.register("Math_Add", AddNode)
    NodeRegistry.register("Math_Multiply", MultiplyNode)
    NodeRegistry.register("Math_Time", TimeNode)
    NodeRegistry.register("Math_Vec3", Vec3GeneratorNode)
    NodeRegistry.register("Math_FloatToVec3", FloatToVec3Node)
    NodeRegistry.register("Math_ArcaneOscillator", ArcaneOscillatorNode)
