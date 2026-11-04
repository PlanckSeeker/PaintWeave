"""SOP Attribute & Group Operators: Group, AttributeRandomize, Color, and Python/PyTorch Wrangle."""

from __future__ import annotations

import traceback

import numpy as np

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import (
    CodeString,
    DataType,
    NodeContext,
    ParamSpec,
    PortSpec,
)
from paintweave.engine.geometry.compute_cuda import HAS_TORCH, torch
from paintweave.engine.geometry.geometry import Geometry


class GroupSOPNode(NodeBase):
    """Creates a boolean point group mask (`@group_<name>`) based on a 3D bounding box."""

    context = NodeContext.SOP
    category = "Attributes & Groups"
    title = "Group (Bounding Box)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input geometry."),
    ]
    OUTPUT_PORTS = [
        PortSpec(
            "Geo", DataType.GEOMETRY, None, "Geometry with @group_<name> point mask."
        ),
    ]
    PARAM_SPECS = [
        ParamSpec("group_name", "str", "group1", label="Group Name"),
        ParamSpec(
            "size_x",
            "float",
            3.0,
            label="Box Size X",
            min_val=0.01,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "size_y",
            "float",
            3.0,
            label="Box Size Y",
            min_val=0.01,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "size_z",
            "float",
            3.0,
            label="Box Size Z",
            min_val=0.01,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "center_x",
            "float",
            0.0,
            label="Center X",
            min_val=-100.0,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "center_y",
            "float",
            0.0,
            label="Center Y",
            min_val=-100.0,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "center_z",
            "float",
            0.0,
            label="Center Z",
            min_val=-100.0,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()

        g_name = str(self.get_param("group_name")).strip() or "group1"
        cx = float(self.get_param("center_x"))
        cy = float(self.get_param("center_y"))
        cz = float(self.get_param("center_z"))
        sx = float(self.get_param("size_x")) * 0.5
        sy = float(self.get_param("size_y")) * 0.5
        sz = float(self.get_param("size_z")) * 0.5

        p = in_geo.points["P"]
        mask_x = (p[:, 0] >= cx - sx) & (p[:, 0] <= cx + sx)
        mask_y = (p[:, 1] >= cy - sy) & (p[:, 1] <= cy + sy)
        mask_z = (p[:, 2] >= cz - sz) & (p[:, 2] <= cz + sz)

        group_mask = (mask_x & mask_y & mask_z).astype(np.float32).reshape(-1, 1)
        out_geo.set_point_attrib(f"group_{g_name}", group_mask)
        self.outputs["Geo"].value = out_geo


class ColorSOPNode(NodeBase):
    """Assigns a constant or group-masked RGB color (`@Cd`) to geometry points."""

    context = NodeContext.SOP
    category = "Attributes & Groups"
    title = "Color (@Cd)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input geometry."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Colored geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "group",
            "str",
            "",
            label="Group Mask",
            description="Optional group name (empty = all points).",
        ),
        ParamSpec("color", "color", (0.22, 0.74, 0.97), label="Color (@Cd)"),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        col = self.get_param("color")
        rgb = np.array(
            [[float(col[0]), float(col[1]), float(col[2])]], dtype=np.float32
        )
        target_cd = np.tile(rgb, (in_geo.num_points, 1))

        group_name = str(self.get_param("group")).strip()
        group_key = f"group_{group_name}"
        if group_name and group_key in in_geo.points and "Cd" in in_geo.points:
            mask = in_geo.points[group_key]
            base_cd = in_geo.points["Cd"]
            new_cd = base_cd + (target_cd - base_cd) * mask
        else:
            new_cd = target_cd

        out_geo.set_point_attrib("Cd", new_cd)
        self.outputs["Geo"].value = out_geo


class AttributeRandomizeSOPNode(NodeBase):
    """Houdini Attribute Randomize SOP: injects randomized values for pscale, Cd, or N."""

    context = NodeContext.SOP
    category = "Attributes & Groups"
    title = "Attribute Randomize"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input geometry."),
    ]
    OUTPUT_PORTS = [
        PortSpec(
            "Geo", DataType.GEOMETRY, None, "Geometry with randomized point attribute."
        ),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "attribute",
            "str",
            "pscale",
            label="Attribute Name",
            description="e.g. pscale, Cd, N",
        ),
        ParamSpec(
            "dimensions", "int", 1, label="Dimensions", min_val=1, max_val=4, step=1
        ),
        ParamSpec(
            "min_value",
            "float",
            0.5,
            label="Min Value",
            min_val=-100.0,
            max_val=100.0,
            step=0.1,
        ),
        ParamSpec(
            "max_value",
            "float",
            1.5,
            label="Max Value",
            min_val=-100.0,
            max_val=100.0,
            step=0.1,
        ),
        ParamSpec(
            "seed", "int", 42, label="Random Seed", min_val=0, max_val=999999, step=1
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        attr = str(self.get_param("attribute")).strip() or "pscale"
        dims = max(1, int(self.get_param("dimensions")))
        min_v = float(self.get_param("min_value"))
        max_v = float(self.get_param("max_value"))
        seed = int(self.get_param("seed"))

        rng = np.random.default_rng(seed)
        shape = (in_geo.num_points, dims) if dims > 1 else (in_geo.num_points, 1)
        rand_vals = rng.uniform(min_v, max_v, shape).astype(np.float32)

        if attr == "N" and dims == 3:
            lengths = np.linalg.norm(rand_vals, axis=1, keepdims=True)
            rand_vals = np.where(lengths < 1e-8, 1.0, rand_vals / lengths)

        out_geo.set_point_attrib(attr, rand_vals)
        self.outputs["Geo"].value = out_geo


DEFAULT_SOP_WRANGLE = CodeString(
    "# Available: P, N, Cd, uv, np, torch, num_points\n"
    "# Example: radial ripple wave\n"
    "r = np.sqrt(P[:, 0]**2 + P[:, 2]**2)\n"
    "P[:, 1] += np.sin(r * 3.5) * 0.35\n"
)


class WrangleSOPNode(NodeBase):
    """Executes vectorized Python / NumPy / PyTorch code to manipulate geometry attributes (Houdini Point Wrangle)."""

    context = NodeContext.SOP
    category = "Attributes & Groups"
    title = "Point Wrangle (Python)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input geometry."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Modified geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "code",
            "code",
            DEFAULT_SOP_WRANGLE,
            label="Python / NumPy Wrangle",
            code_lang="python",
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        code_str = str(self.get_param("code"))

        sandbox = {k: v.copy() for k, v in in_geo.points.items()}
        sandbox["np"] = np
        if HAS_TORCH:
            sandbox["torch"] = torch
        sandbox["num_points"] = in_geo.num_points

        self.error_message = None
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "len": len,
            "range": range,
            "float": float,
            "int": int,
        }
        try:
            exec(code_str, {"__builtins__": safe_builtins}, sandbox)
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            line_no = tb[-1].lineno if tb else "?"
            self.error_message = (
                f"Wrangle Error: {type(exc).__name__} at line {line_no} — {exc}"
            )
            self.outputs["Geo"].value = out_geo
            return

        for k, v in sandbox.items():
            if k in ("np", "torch", "num_points") or k.startswith("__"):
                continue
            if HAS_TORCH and torch is not None and isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 0 or arr.size == 1:
                arr = np.tile(arr, (in_geo.num_points, 1))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] == in_geo.num_points:
                out_geo.set_point_attrib(k, arr)

        out_geo.compute_vertex_normals()
        self.outputs["Geo"].value = out_geo


class UVProjectSOPNode(NodeBase):
    """Houdini UVProject SOP: computes `@uv` texture coordinates via Planar, Cylindrical, Spherical, or Box projection."""

    context = NodeContext.SOP
    category = "Attributes & Groups"
    title = "UV Project (@uv)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input geometry."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Geometry with projected @uv attribute."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "projection",
            "enum",
            "Planar XZ (Top)",
            label="Projection Type",
            options=[
                "Planar XZ (Top)",
                "Planar XY (Front)",
                "Planar YZ (Side)",
                "Cylindrical (Y-Axis)",
                "Spherical",
                "Triplanar Box",
            ],
        ),
        ParamSpec(
            "scale_u",
            "float",
            1.0,
            label="Scale U",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
        ),
        ParamSpec(
            "scale_v",
            "float",
            1.0,
            label="Scale V",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
        ),
        ParamSpec(
            "offset_u",
            "float",
            0.0,
            label="Offset U",
            min_val=-10.0,
            max_val=10.0,
            step=0.05,
        ),
        ParamSpec(
            "offset_v",
            "float",
            0.0,
            label="Offset V",
            min_val=-10.0,
            max_val=10.0,
            step=0.05,
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        p = in_geo.points["P"]
        proj = str(self.get_param("projection"))
        su = float(self.get_param("scale_u"))
        sv = float(self.get_param("scale_v"))
        ou = float(self.get_param("offset_u"))
        ov = float(self.get_param("offset_v"))

        b_min = np.min(p, axis=0)
        b_max = np.max(p, axis=0)
        span = np.where((b_max - b_min) < 1e-6, 1.0, b_max - b_min)
        norm_p = (p - b_min) / span

        if proj == "Planar XZ (Top)":
            u = norm_p[:, 0]
            v = norm_p[:, 2]
        elif proj == "Planar XY (Front)":
            u = norm_p[:, 0]
            v = norm_p[:, 1]
        elif proj == "Planar YZ (Side)":
            u = norm_p[:, 2]
            v = norm_p[:, 1]
        elif proj == "Cylindrical (Y-Axis)":
            centered = p - 0.5 * (b_min + b_max)
            theta = np.arctan2(centered[:, 2], centered[:, 0])
            u = (theta + np.pi) / (2.0 * np.pi)
            v = norm_p[:, 1]
        elif proj == "Spherical":
            centered = p - 0.5 * (b_min + b_max)
            r = np.linalg.norm(centered, axis=1)
            r_safe = np.where(r < 1e-6, 1.0, r)
            theta = np.arctan2(centered[:, 2], centered[:, 0])
            phi = np.arcsin(np.clip(centered[:, 1] / r_safe, -1.0, 1.0))
            u = (theta + np.pi) / (2.0 * np.pi)
            v = (phi + 0.5 * np.pi) / np.pi
        else:  # Triplanar Box
            n = (
                in_geo.points["N"]
                if "N" in in_geo.points and in_geo.points["N"].shape == p.shape
                else np.tile(
                    np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (p.shape[0], 1)
                )
            )
            abs_n = np.abs(n)
            dom = np.argmax(abs_n, axis=1)
            u = np.where(
                dom == 0, norm_p[:, 2], np.where(dom == 1, norm_p[:, 0], norm_p[:, 0])
            )
            v = np.where(
                dom == 0, norm_p[:, 1], np.where(dom == 1, norm_p[:, 2], norm_p[:, 1])
            )

        uv = np.column_stack([u * su + ou, v * sv + ov]).astype(np.float32)
        out_geo.set_point_attrib("uv", uv)
        self.outputs["Geo"].value = out_geo
