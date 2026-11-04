"""Data types, port specifications, and parameter schemas for PaintWeave nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class DataType(Enum):
    """Defines typed data flowing across node ports."""

    FLOAT = auto()
    INT = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    MAT4 = auto()
    GEOMETRY = (
        auto()
    )  # Columnar Geometry container (Points, Vertices, Primitives, Detail)
    SHADER = auto()  # GLSL ShaderPayload (fragment/vertex AST + uniforms)
    ANY = auto()


class PinDirection(Enum):
    """Port direction."""

    INPUT = auto()
    OUTPUT = auto()


class NodeContext(Enum):
    """Houdini-inspired operator contexts."""

    SOP = "SOP (Geometry)"
    VOP = "VOP (Shader)"
    MATH = "Math & Utility"


class AttributeDomain(Enum):
    """Houdini geometry spreadsheet domains."""

    POINT = "Points"
    VERTEX = "Vertices"
    PRIMITIVE = "Primitives"
    DETAIL = "Detail"
    MATERIALS = "Materials"


class CodeString(str):
    """String subtype signalling a multi-line syntax-highlighted code editor (Python/GLSL)."""


@dataclass(frozen=True)
class PortSpec:
    """Declarative specification for an input or output pin on a node."""

    name: str
    data_type: DataType
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class ParamSpec:
    """Declarative specification for an editable parameter in the Node Property Inspector."""

    name: str
    param_type: str  # "float", "int", "bool", "str", "choice", "enum", "color", "vec3", "filepath", "code"
    default: Any
    label: str = ""
    min_val: float | int | None = None
    max_val: float | int | None = None
    step: float | int = 0.1
    unit: str = ""
    choices: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    description: str = ""
    code_lang: str = "python"  # "python" or "glsl"
