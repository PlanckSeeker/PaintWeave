"""SOP Primitive Generators: Grid, Box, Sphere, Cylinder, and Wavefront .obj File Loader."""

from __future__ import annotations

import os

import numpy as np

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.engine.geometry.geometry import Geometry


class GridSOPNode(NodeBase):
    """Generates a planar XZ grid mesh (Houdini Grid SOP equivalent)."""

    context = NodeContext.SOP
    category = "Primitives"
    title = "Grid"

    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Generated XZ planar grid geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "size_x",
            "float",
            6.0,
            label="Size X",
            min_val=0.1,
            max_val=100.0,
            step=0.5,
            unit="m",
        ),
        ParamSpec(
            "size_z",
            "float",
            6.0,
            label="Size Z",
            min_val=0.1,
            max_val=100.0,
            step=0.5,
            unit="m",
        ),
        ParamSpec("rows", "int", 32, label="Rows", min_val=2, max_val=512, step=1),
        ParamSpec("cols", "int", 32, label="Columns", min_val=2, max_val=512, step=1),
    ]

    def evaluate(self) -> None:
        sx = float(self.get_param("size_x"))
        sz = float(self.get_param("size_z"))
        rows = max(2, int(self.get_param("rows")))
        cols = max(2, int(self.get_param("cols")))

        xs = np.linspace(-sx * 0.5, sx * 0.5, cols, dtype=np.float32)
        zs = np.linspace(-sz * 0.5, sz * 0.5, rows, dtype=np.float32)
        gx, gz = np.meshgrid(xs, zs)
        gy = np.zeros_like(gx, dtype=np.float32)

        p = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)
        n = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (p.shape[0], 1))
        cd = np.tile(np.array([[0.38, 0.62, 0.85]], dtype=np.float32), (p.shape[0], 1))

        u = np.linspace(0.0, 1.0, cols, dtype=np.float32)
        v = np.linspace(0.0, 1.0, rows, dtype=np.float32)
        gu, gv = np.meshgrid(u, v)
        uv = np.column_stack([gu.ravel(), gv.ravel()]).astype(np.float32)

        r_idx = np.arange(rows - 1, dtype=np.uint32)[:, None]
        c_idx = np.arange(cols - 1, dtype=np.uint32)[None, :]
        i0 = (r_idx * cols + c_idx).ravel()
        i1 = i0 + 1
        i2 = i0 + cols
        i3 = i2 + 1

        tris = np.vstack(
            [
                np.column_stack([i0, i2, i1]),
                np.column_stack([i1, i2, i3]),
            ]
        ).astype(np.uint32)

        geo = Geometry()
        geo.set_point_attrib("P", p)
        geo.set_point_attrib("N", n)
        geo.set_point_attrib("Cd", cd)
        geo.set_point_attrib("uv", uv)
        geo.set_triangles(tris)

        self.outputs["Geo"].value = geo


class BoxSOPNode(NodeBase):
    """Generates a 3D box/cube mesh (Houdini Box SOP equivalent)."""

    context = NodeContext.SOP
    category = "Primitives"
    title = "Box"

    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Generated 3D box geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "scale_x",
            "float",
            1.5,
            label="Scale X",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
            unit="m",
        ),
        ParamSpec(
            "scale_y",
            "float",
            1.5,
            label="Scale Y",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
            unit="m",
        ),
        ParamSpec(
            "scale_z",
            "float",
            1.5,
            label="Scale Z",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
            unit="m",
        ),
    ]

    def evaluate(self) -> None:
        hx = float(self.get_param("scale_x")) * 0.5
        hy = float(self.get_param("scale_y")) * 0.5
        hz = float(self.get_param("scale_z")) * 0.5

        p = np.array(
            [
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
            ],
            dtype=np.float32,
        )

        tris = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 5, 6],
                [1, 6, 2],
                [5, 4, 7],
                [5, 7, 6],
                [4, 0, 3],
                [4, 3, 7],
                [3, 2, 6],
                [3, 6, 7],
                [4, 5, 1],
                [4, 1, 0],
            ],
            dtype=np.uint32,
        )

        cd = np.tile(np.array([[0.82, 0.52, 0.36]], dtype=np.float32), (8, 1))
        uv = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        geo = Geometry()
        geo.set_point_attrib("P", p)
        geo.set_point_attrib("Cd", cd)
        geo.set_point_attrib("uv", uv)
        geo.set_triangles(tris)
        geo.compute_vertex_normals()
        self.outputs["Geo"].value = geo


class CylinderSOPNode(NodeBase):
    """Generates a parametric 3D cylinder/tube mesh (Houdini Tube SOP equivalent)."""

    context = NodeContext.SOP
    category = "Primitives"
    title = "Cylinder"

    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Generated cylinder geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "radius",
            "float",
            1.0,
            label="Radius",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
            unit="m",
        ),
        ParamSpec(
            "height",
            "float",
            2.0,
            label="Height",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
            unit="m",
        ),
        ParamSpec("cols", "int", 20, label="Columns", min_val=3, max_val=256, step=1),
        ParamSpec("rows", "int", 4, label="Rows", min_val=1, max_val=256, step=1),
    ]

    def evaluate(self) -> None:
        r = float(self.get_param("radius"))
        h = float(self.get_param("height"))
        cols = max(3, int(self.get_param("cols")))
        rows = max(1, int(self.get_param("rows")))

        theta = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False, dtype=np.float32)
        ys = np.linspace(-h * 0.5, h * 0.5, rows + 1, dtype=np.float32)
        th_grid, y_grid = np.meshgrid(theta, ys)

        x_grid = r * np.cos(th_grid)
        z_grid = r * np.sin(th_grid)
        p = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).astype(
            np.float32
        )

        nx = np.cos(th_grid)
        nz = np.sin(th_grid)
        ny = np.zeros_like(nx)
        n = np.column_stack([nx.ravel(), ny.ravel(), nz.ravel()]).astype(np.float32)

        cd = np.tile(np.array([[0.55, 0.58, 0.65]], dtype=np.float32), (p.shape[0], 1))
        u_grid, v_grid = np.meshgrid(
            np.linspace(0, 1, cols, endpoint=False),
            np.linspace(0, 1, rows + 1),
        )
        uv = np.column_stack([u_grid.ravel(), v_grid.ravel()]).astype(np.float32)

        tris = []
        for r_idx in range(rows):
            for c_idx in range(cols):
                next_c = (c_idx + 1) % cols
                i0 = r_idx * cols + c_idx
                i1 = r_idx * cols + next_c
                i2 = (r_idx + 1) * cols + c_idx
                i3 = (r_idx + 1) * cols + next_c
                tris.append([i0, i2, i1])
                tris.append([i1, i2, i3])

        tri_arr = np.array(tris, dtype=np.uint32)

        geo = Geometry()
        geo.set_point_attrib("P", p)
        geo.set_point_attrib("N", n)
        geo.set_point_attrib("Cd", cd)
        geo.set_point_attrib("uv", uv)
        geo.set_triangles(tri_arr)
        self.outputs["Geo"].value = geo


class SphereSOPNode(NodeBase):
    """Generates a parametric UV sphere mesh (Houdini Sphere SOP equivalent)."""

    context = NodeContext.SOP
    category = "Primitives"
    title = "Sphere"

    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Generated UV sphere geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "radius",
            "float",
            1.2,
            label="Radius",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
            unit="m",
        ),
        ParamSpec("rows", "int", 16, label="Rows", min_val=3, max_val=256, step=1),
        ParamSpec("cols", "int", 24, label="Columns", min_val=3, max_val=256, step=1),
    ]

    def evaluate(self) -> None:
        rad = float(self.get_param("radius"))
        rows = max(3, int(self.get_param("rows")))
        cols = max(3, int(self.get_param("cols")))

        phi = np.linspace(-np.pi / 2, np.pi / 2, rows + 1, dtype=np.float32)
        theta = np.linspace(0, 2 * np.pi, cols, endpoint=False, dtype=np.float32)
        th_grid, phi_grid = np.meshgrid(theta, phi)

        x = rad * np.cos(phi_grid) * np.cos(th_grid)
        y = rad * np.sin(phi_grid)
        z = rad * np.cos(phi_grid) * np.sin(th_grid)
        p = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

        lengths = np.linalg.norm(p, axis=1, keepdims=True)
        n = (p / np.where(lengths < 1e-8, 1.0, lengths)).astype(np.float32)

        cd = np.tile(np.array([[0.78, 0.65, 0.45]], dtype=np.float32), (p.shape[0], 1))
        u_grid, v_grid = np.meshgrid(
            np.linspace(0, 1, cols, endpoint=False),
            np.linspace(0, 1, rows + 1),
        )
        uv = np.column_stack([u_grid.ravel(), v_grid.ravel()]).astype(np.float32)

        tris = []
        for r_idx in range(rows):
            for c_idx in range(cols):
                next_c = (c_idx + 1) % cols
                i0 = r_idx * cols + c_idx
                i1 = r_idx * cols + next_c
                i2 = (r_idx + 1) * cols + c_idx
                i3 = (r_idx + 1) * cols + next_c
                tris.append([i0, i2, i1])
                tris.append([i1, i2, i3])

        tri_arr = np.array(tris, dtype=np.uint32)

        geo = Geometry()
        geo.set_point_attrib("P", p)
        geo.set_point_attrib("N", n)
        geo.set_point_attrib("Cd", cd)
        geo.set_point_attrib("uv", uv)
        geo.set_triangles(tri_arr)
        self.outputs["Geo"].value = geo


class FileSOPNode(NodeBase):
    """Loads external 3D geometry from a Wavefront .obj file (Houdini File SOP equivalent)."""

    context = NodeContext.SOP
    category = "Primitives"
    title = "File (.obj)"

    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Loaded Wavefront .obj geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "filepath",
            "filepath",
            "",
            label="OBJ File Path",
            description="Path to external .obj mesh.",
        ),
    ]

    def evaluate(self) -> None:
        path = str(self.get_param("filepath"))
        if path and os.path.exists(path):
            self.outputs["Geo"].value = Geometry.load_from_obj(path)
        else:
            self.outputs["Geo"].value = Geometry()


class CurveSOPNode(NodeBase):
    """Generates a parametric 3D backbone curve (Helix, Circle, Sine Wave, or Line) with tangent @N and @uv."""

    context = NodeContext.SOP
    category = "Primitives"
    title = "Curve (Spline/Helix)"

    OUTPUT_PORTS = [
        PortSpec(
            "Geo",
            DataType.GEOMETRY,
            None,
            "Ordered 3D curve points with tangent @N and @uv (ready for Sweep or CopyToPoints).",
        ),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "curve_type",
            "enum",
            "Helix Spiral",
            label="Curve Shape",
            options=["Helix Spiral", "Circle / Arc", "Sine Wave", "Straight Line"],
        ),
        ParamSpec(
            "num_points",
            "int",
            48,
            label="Point Count",
            min_val=2,
            max_val=2048,
            step=4,
        ),
        ParamSpec(
            "radius",
            "float",
            1.8,
            label="Radius / Amplitude",
            min_val=0.05,
            max_val=50.0,
            step=0.1,
            unit="m",
        ),
        ParamSpec(
            "height",
            "float",
            3.5,
            label="Height / Length",
            min_val=0.0,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "turns",
            "float",
            2.0,
            label="Turns / Cycles",
            min_val=0.1,
            max_val=20.0,
            step=0.25,
        ),
    ]

    def evaluate(self) -> None:
        shape = str(self.get_param("curve_type"))
        n_pts = max(2, int(self.get_param("num_points")))
        rad = float(self.get_param("radius"))
        height = float(self.get_param("height"))
        turns = float(self.get_param("turns"))

        t = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
        theta = t * turns * 2.0 * np.pi

        if shape in ("Helix Spiral", "Helix / Spiral"):
            x = rad * np.cos(theta)
            y = (t - 0.5) * height
            z = rad * np.sin(theta)
        elif shape == "Circle / Arc":
            x = rad * np.cos(theta)
            y = np.zeros_like(t)
            z = rad * np.sin(theta)
        elif shape == "Sine Wave":
            x = (t - 0.5) * height
            y = rad * np.sin(theta)
            z = np.zeros_like(t)
        else:  # Straight Line
            x = np.zeros_like(t)
            y = (t - 0.5) * height
            z = np.zeros_like(t)

        p = np.column_stack([x, y, z]).astype(np.float32)

        # Compute forward finite-difference tangents along the curve and store in @N
        tangents = np.zeros_like(p)
        tangents[:-1] = p[1:] - p[:-1]
        tangents[-1] = tangents[-2]
        t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.where(t_len < 1e-8, 1.0, t_len)

        cd = np.column_stack(
            [
                0.25 + 0.65 * t,
                0.75 - 0.35 * t,
                np.full_like(t, 0.90),
            ]
        ).astype(np.float32)
        uv = np.column_stack([t, np.zeros_like(t)]).astype(np.float32)

        geo = Geometry()
        geo.set_point_attrib("P", p)
        geo.set_point_attrib("N", tangents.astype(np.float32))
        geo.set_point_attrib("Cd", cd)
        geo.set_point_attrib("uv", uv)
        self.outputs["Geo"].value = geo
