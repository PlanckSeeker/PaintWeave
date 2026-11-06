"""SOP Modifiers: Transform, Mountain (CUDA/CPU), Extrude, Subdivide, and Facet."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.engine.geometry.compute_cuda import mountain_displace
from paintweave.engine.geometry.geometry import Geometry


class TransformSOPNode(NodeBase):
    """Translates and scales @P using Copy-on-Write, with optional Group masking."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Transform"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input geometry to transform."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Transformed geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "group",
            "str",
            "",
            label="Group Mask",
            description="Optional point group name (empty = all).",
        ),
        ParamSpec(
            "translate_x",
            "float",
            0.0,
            label="Translate X",
            min_val=-100.0,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "translate_y",
            "float",
            0.0,
            label="Translate Y",
            min_val=-100.0,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "translate_z",
            "float",
            0.0,
            label="Translate Z",
            min_val=-100.0,
            max_val=100.0,
            step=0.25,
            unit="m",
        ),
        ParamSpec(
            "scale_x",
            "float",
            1.0,
            label="Scale X",
            min_val=0.01,
            max_val=50.0,
            step=0.1,
        ),
        ParamSpec(
            "scale_y",
            "float",
            1.0,
            label="Scale Y",
            min_val=0.01,
            max_val=50.0,
            step=0.1,
        ),
        ParamSpec(
            "scale_z",
            "float",
            1.0,
            label="Scale Z",
            min_val=0.01,
            max_val=50.0,
            step=0.1,
        ),
        ParamSpec(
            "uniform_scale",
            "float",
            1.0,
            label="Uniform Scale",
            min_val=0.01,
            max_val=50.0,
            step=0.1,
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        if out_geo.num_points == 0:
            self.outputs["Geo"].value = out_geo
            return

        u_scale = float(self.get_param("uniform_scale"))
        scale = np.array(
            [
                float(self.get_param("scale_x")) * u_scale,
                float(self.get_param("scale_y")) * u_scale,
                float(self.get_param("scale_z")) * u_scale,
            ],
            dtype=np.float32,
        )

        offset = np.array(
            [
                float(self.get_param("translate_x")),
                float(self.get_param("translate_y")),
                float(self.get_param("translate_z")),
            ],
            dtype=np.float32,
        )

        p = in_geo.points["P"]
        target_p = p * scale + offset

        group_name = str(self.get_param("group")).strip()
        group_key = f"group_{group_name}"

        if group_name and group_key in in_geo.points:
            mask = in_geo.points[group_key]
            new_p = p + (target_p - p) * mask
        else:
            new_p = target_p

        out_geo.set_point_attrib("P", new_p)
        self.outputs["Geo"].value = out_geo


class MountainSOPNode(NodeBase):
    """Houdini Mountain SOP: displaces @P along Y via multi-octave procedural wave interference (CUDA accelerated)."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Mountain (CUDA)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input mesh to displace."),
        PortSpec("Amplitude", DataType.FLOAT, 0.85, "Peak height displacement."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Displaced terrain geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "frequency",
            "float",
            1.8,
            label="Frequency",
            min_val=0.05,
            max_val=25.0,
            step=0.1,
        ),
        ParamSpec(
            "phase",
            "float",
            0.0,
            label="Phase Offset",
            min_val=-50.0,
            max_val=50.0,
            step=0.1,
        ),
        ParamSpec(
            "octaves", "int", 2, label="FBM Octaves", min_val=1, max_val=8, step=1
        ),
        ParamSpec(
            "roughness",
            "float",
            0.45,
            label="Roughness",
            min_val=0.1,
            max_val=0.95,
            step=0.05,
        ),
        ParamSpec(
            "use_cuda",
            "bool",
            True,
            label="Use CUDA (cu134)",
            description="Accelerate on RTX GPU if available.",
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        amp = float(self.inputs["Amplitude"].get_data())
        freq = float(self.get_param("frequency"))
        phase = float(self.get_param("phase"))
        octaves = int(self.get_param("octaves"))
        roughness = float(self.get_param("roughness"))
        use_cuda = bool(self.get_param("use_cuda"))

        out_geo = in_geo.clone()
        new_p, cd = mountain_displace(
            in_geo.points["P"],
            amplitude=amp,
            frequency=freq,
            phase=phase,
            octaves=octaves,
            roughness=roughness,
            use_cuda=use_cuda,
        )

        out_geo.set_point_attrib("P", new_p)
        out_geo.set_point_attrib("Cd", cd)
        out_geo.compute_vertex_normals()
        self.outputs["Geo"].value = out_geo


class ExtrudeSOPNode(NodeBase):
    """Extrudes geometry along point normals or a specific global axis, creating side walls and caps."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "PolyExtrude"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input mesh to extrude."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Extruded mesh."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "distance",
            "float",
            1.0,
            label="Distance",
            min_val=-25.0,
            max_val=25.0,
            step=0.1,
            unit="m",
        ),
        ParamSpec("use_normal", "bool", True, label="Extrude Along Normals"),
        ParamSpec(
            "dir_x",
            "float",
            0.0,
            label="Direction X",
            min_val=-1.0,
            max_val=1.0,
            step=0.1,
        ),
        ParamSpec(
            "dir_y",
            "float",
            1.0,
            label="Direction Y",
            min_val=-1.0,
            max_val=1.0,
            step=0.1,
        ),
        ParamSpec(
            "dir_z",
            "float",
            0.0,
            label="Direction Z",
            min_val=-1.0,
            max_val=1.0,
            step=0.1,
        ),
        ParamSpec("output_base", "bool", False, label="Output Back/Base Cap"),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_primitives == 0:
            self.outputs["Geo"].value = Geometry()
            return

        dist = float(self.get_param("distance"))
        use_normal = bool(self.get_param("use_normal"))
        output_base = bool(self.get_param("output_base"))

        p = in_geo.points["P"]

        if use_normal:
            n = in_geo.points["N"] if "N" in in_geo.points else np.zeros_like(p)
            top_p = p + n * dist
        else:
            dir_vec = np.array(
                [
                    float(self.get_param("dir_x")),
                    float(self.get_param("dir_y")),
                    float(self.get_param("dir_z")),
                ],
                dtype=np.float32,
            )
            length = np.linalg.norm(dir_vec)
            if length > 1e-8:
                dir_vec /= length
            top_p = p + dir_vec * dist

        num_base_pts = p.shape[0]
        out_geo = Geometry()
        out_geo.detail["materials"] = in_geo.detail.get("materials", {}).copy()

        for k, v in in_geo.points.items():
            if k == "P":
                out_geo.set_point_attrib("P", np.vstack([p, top_p]))
            else:
                out_geo.set_point_attrib(k, np.vstack([v, v]))

        tris = in_geo.primitives["triangles"]
        base_mat_ids = in_geo.ensure_material_ids()
        new_tris_list = [tris + num_base_pts]
        new_mat_list = [base_mat_ids]

        if output_base:
            new_tris_list.append(tris[:, [0, 2, 1]])
            new_mat_list.append(base_mat_ids)

        e0 = tris[:, [0, 1]]
        e1 = tris[:, [1, 2]]
        e2 = tris[:, [2, 0]]
        edges = np.vstack([e0, e1, e2])
        edge_mats = np.concatenate([base_mat_ids, base_mat_ids, base_mat_ids])

        sorted_edges = np.sort(edges, axis=1).astype(np.int64)
        max_vtx = int(num_base_pts) + 1
        undirected_hash = sorted_edges[:, 0] * max_vtx + sorted_edges[:, 1]
        unq, counts = np.unique(undirected_hash, return_counts=True)

        boundary_hashes = unq[counts == 1]
        mask = np.isin(undirected_hash, boundary_hashes)
        boundary_edges = edges[mask]
        boundary_mats = edge_mats[mask]

        if boundary_edges.shape[0] > 0:
            v0 = boundary_edges[:, 0]
            v1 = boundary_edges[:, 1]
            v0_top = v0 + num_base_pts
            v1_top = v1 + num_base_pts

            side_tris_1 = np.column_stack([v0, v1, v1_top])
            side_tris_2 = np.column_stack([v0, v1_top, v0_top])

            new_tris_list.append(side_tris_1)
            new_tris_list.append(side_tris_2)
            new_mat_list.append(boundary_mats)
            new_mat_list.append(boundary_mats)

        final_tris = np.vstack(new_tris_list).astype(np.uint32)
        final_mats = np.concatenate(new_mat_list).astype(np.int32)
        out_geo.set_triangles(final_tris, final_mats)
        out_geo.compute_vertex_normals()
        self.outputs["Geo"].value = out_geo


class SubdivideSOPNode(NodeBase):
    """Houdini Subdivide SOP: splits triangle primitives into 4 sub-triangles to increase mesh resolution."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Subdivide"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input triangle mesh."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Subdivided high-resolution mesh."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "iterations", "int", 1, label="Iterations", min_val=1, max_val=4, step=1
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_primitives == 0:
            self.outputs["Geo"].value = Geometry()
            return

        iterations = max(1, min(4, int(self.get_param("iterations"))))
        curr_geo = in_geo.clone()

        for _ in range(iterations):
            p = curr_geo.points["P"]
            tris = curr_geo.primitives["triangles"]
            mat_ids = curr_geo.ensure_material_ids()

            e0 = tris[:, [0, 1]]
            e1 = tris[:, [1, 2]]
            e2 = tris[:, [2, 0]]
            all_edges = np.vstack([e0, e1, e2])
            sorted_edges = np.sort(all_edges, axis=1)

            unique_edges, edge_inverse = np.unique(
                sorted_edges, axis=0, return_inverse=True
            )
            num_orig_pts = p.shape[0]
            edge_map = edge_inverse.reshape(-1, 3)

            a = tris[:, 0]
            b = tris[:, 1]
            c = tris[:, 2]

            ab = edge_map[:, 0] + num_orig_pts
            bc = edge_map[:, 1] + num_orig_pts
            ca = edge_map[:, 2] + num_orig_pts

            sub_tris = np.vstack(
                [
                    np.column_stack([a, ab, ca]),
                    np.column_stack([b, bc, ab]),
                    np.column_stack([c, ca, bc]),
                    np.column_stack([ab, bc, ca]),
                ]
            ).astype(np.uint32)
            sub_mats = np.tile(mat_ids, 4).astype(np.int32)

            out_geo = Geometry()
            out_geo.detail["materials"] = curr_geo.detail.get("materials", {}).copy()

            for k, val in curr_geo.points.items():
                if val.shape[0] == num_orig_pts:
                    val_v0 = val[unique_edges[:, 0]]
                    val_v1 = val[unique_edges[:, 1]]
                    mid_vals = (val_v0 + val_v1) * 0.5
                    out_geo.set_point_attrib(k, np.vstack([val, mid_vals]))
                else:
                    out_geo.set_point_attrib(k, val)

            out_geo.set_triangles(sub_tris, sub_mats)
            out_geo.compute_vertex_normals()
            curr_geo = out_geo

        self.outputs["Geo"].value = curr_geo


class FacetSOPNode(NodeBase):
    """Houdini Facet SOP: unwraps shared points per primitive for crisp flat/low-poly shading."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Facet (Flat Shade)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input smooth-shaded mesh."),
    ]
    OUTPUT_PORTS = [
        PortSpec(
            "Geo",
            DataType.GEOMETRY,
            None,
            "Flat-shaded mesh with unique vertices per triangle.",
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_primitives == 0:
            self.outputs["Geo"].value = Geometry()
            return

        tris = in_geo.primitives["triangles"]
        mat_ids = in_geo.ensure_material_ids().copy()
        flat_indices = tris.ravel()

        out_geo = Geometry()
        out_geo.detail["materials"] = in_geo.detail.get("materials", {}).copy()

        for k, v in in_geo.points.items():
            out_geo.set_point_attrib(k, v[flat_indices])

        n_vtx = flat_indices.shape[0]
        new_tris = np.arange(n_vtx, dtype=np.uint32).reshape(-1, 3)

        out_geo.set_triangles(new_tris, mat_ids)
        out_geo.compute_vertex_normals()
        self.outputs["Geo"].value = out_geo


class SmoothSOPNode(NodeBase):
    """Houdini Smooth SOP: iterative Laplacian mesh relaxation over triangle adjacency."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Smooth (Laplacian)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input triangle mesh to smooth."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Smoothed geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "iterations",
            "int",
            4,
            label="Iterations",
            min_val=1,
            max_val=50,
            step=1,
        ),
        ParamSpec(
            "strength",
            "float",
            0.5,
            label="Filter Strength",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        if in_geo.num_primitives == 0:
            self.outputs["Geo"].value = out_geo
            return

        iters = max(1, int(self.get_param("iterations")))
        lam = float(np.clip(self.get_param("strength"), 0.0, 1.0))

        p = in_geo.points["P"].copy()
        tris = in_geo.primitives["triangles"]
        n_pts = p.shape[0]

        # Build directed edge pairs (u -> v and v -> u) from triangle indices
        u_idx = np.concatenate(
            [
                tris[:, 0],
                tris[:, 1],
                tris[:, 1],
                tris[:, 2],
                tris[:, 2],
                tris[:, 0],
            ]
        )
        v_idx = np.concatenate(
            [
                tris[:, 1],
                tris[:, 0],
                tris[:, 2],
                tris[:, 1],
                tris[:, 0],
                tris[:, 2],
            ]
        )

        deg = np.zeros((n_pts, 1), dtype=np.float32)
        np.add.at(deg[:, 0], u_idx, 1.0)
        deg_safe = np.where(deg < 1.0, 1.0, deg)
        has_neighbors = (deg[:, 0] > 0.0)[:, None]

        for _ in range(iters):
            neighbor_sum = np.zeros_like(p, dtype=np.float32)
            np.add.at(neighbor_sum, u_idx, p[v_idx])
            avg_p = neighbor_sum / deg_safe
            p = np.where(has_neighbors, p + lam * (avg_p - p), p)

        out_geo.set_point_attrib("P", p.astype(np.float32))
        out_geo.compute_vertex_normals()
        self.outputs["Geo"].value = out_geo


class NormalSOPNode(NodeBase):
    """Houdini Normal SOP: recomputes or reverses vertex/point normals (@N) with optional flat shading."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Normal (@N)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input mesh."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Mesh with updated @N attribute."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "reverse_normals",
            "bool",
            False,
            label="Reverse Normals",
            description="Flip @N direction (and optionally triangle winding).",
        ),
        ParamSpec(
            "add_y_bias",
            "float",
            0.0,
            label="Upward Y Bias",
            min_val=-2.0,
            max_val=2.0,
            step=0.1,
            description="Stylized foliage/toon normal upward bias.",
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        if out_geo.num_primitives > 0:
            out_geo.compute_vertex_normals()

        n = out_geo.points["N"].copy()
        y_bias = float(self.get_param("add_y_bias"))
        if abs(y_bias) > 1e-5:
            n[:, 1] += y_bias
            lengths = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.where(lengths < 1e-8, 1.0, lengths)

        if bool(self.get_param("reverse_normals")):
            n = -n

        out_geo.set_point_attrib("N", n.astype(np.float32))
        self.outputs["Geo"].value = out_geo


class SweepSOPNode(NodeBase):
    """Houdini Sweep SOP: sweeps a tubular cross-section along an ordered curve backbone with taper and twist."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Sweep (Curve Tube)"

    INPUT_PORTS = [
        PortSpec(
            "Backbone",
            DataType.GEOMETRY,
            None,
            "Ordered curve points (e.g. from SOP_Curve).",
        ),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Swept 3D tube mesh with @uv and @N."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "radius",
            "float",
            0.35,
            label="Base Radius",
            min_val=0.01,
            max_val=25.0,
            step=0.05,
            unit="m",
        ),
        ParamSpec(
            "taper_end",
            "float",
            0.25,
            label="End Taper Ratio",
            min_val=0.0,
            max_val=5.0,
            step=0.05,
            description="Radius multiplier at the tip of the curve (0 = sharp horn/tendril).",
        ),
        ParamSpec(
            "cols",
            "int",
            12,
            label="Cross-Section Sides",
            min_val=3,
            max_val=128,
            step=1,
        ),
        ParamSpec(
            "twist_deg",
            "float",
            0.0,
            label="Twist (Degrees)",
            min_val=-1440.0,
            max_val=1440.0,
            step=15.0,
        ),
    ]

    def evaluate(self) -> None:
        backbone: Geometry | None = self.inputs["Backbone"].get_data()
        if backbone is None or backbone.num_points < 2:
            self.outputs["Geo"].value = Geometry()
            return

        curve_p = backbone.points["P"]
        n_rings = curve_p.shape[0]
        base_rad = float(self.get_param("radius"))
        taper_end = float(self.get_param("taper_end"))
        cols = max(3, int(self.get_param("cols")))
        twist_rad = np.deg2rad(float(self.get_param("twist_deg")))

        # Compute tangents along backbone
        tangents = np.zeros_like(curve_p)
        tangents[:-1] = curve_p[1:] - curve_p[:-1]
        tangents[-1] = tangents[-2]
        t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.where(t_len < 1e-8, 1.0, t_len)

        v_param = np.linspace(0.0, 1.0, n_rings, dtype=np.float32)
        radii = base_rad * (1.0 + (taper_end - 1.0) * v_param)
        if (
            "pscale" in backbone.points
            and backbone.points["pscale"].shape[0] == n_rings
        ):
            radii = radii * backbone.points["pscale"].ravel()

        angles = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False, dtype=np.float32)
        u_param = np.linspace(0.0, 1.0, cols, endpoint=False, dtype=np.float32)

        all_p: list[np.ndarray] = []
        all_uv: list[np.ndarray] = []
        all_cd: list[np.ndarray] = []

        has_cd = "Cd" in backbone.points and backbone.points["Cd"].shape == (n_rings, 3)

        # Parallel transport frame along backbone
        prev_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(tangents[0], prev_up))) > 0.95:
            prev_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        for i in range(n_rings):
            T = tangents[i]
            B = np.cross(T, prev_up)
            b_norm = float(np.linalg.norm(B))
            if b_norm < 1e-6:
                alt = (
                    np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    if abs(T[0]) < 0.9
                    else np.array([0.0, 0.0, 1.0], dtype=np.float32)
                )
                B = np.cross(T, alt)
                b_norm = float(np.linalg.norm(B))
            B = B / max(b_norm, 1e-8)
            N_frame = np.cross(B, T)
            prev_up = N_frame

            ang = angles + v_param[i] * twist_rad
            ring_offsets = radii[i] * (
                np.cos(ang)[:, None] * N_frame[None, :]
                + np.sin(ang)[:, None] * B[None, :]
            )
            ring_p = curve_p[i][None, :] + ring_offsets
            all_p.append(ring_p)

            ring_uv = np.column_stack([u_param, np.full(cols, v_param[i])])
            all_uv.append(ring_uv)

            if has_cd:
                all_cd.append(np.tile(backbone.points["Cd"][i], (cols, 1)))
            else:
                all_cd.append(
                    np.tile(
                        np.array([[0.32, 0.78, 0.72]], dtype=np.float32), (cols, 1)
                    )
                )

        p_arr = np.vstack(all_p).astype(np.float32)
        uv_arr = np.vstack(all_uv).astype(np.float32)
        cd_arr = np.vstack(all_cd).astype(np.float32)

        tris: list[list[int]] = []
        for r_idx in range(n_rings - 1):
            for c_idx in range(cols):
                next_c = (c_idx + 1) % cols
                i0 = r_idx * cols + c_idx
                i1 = r_idx * cols + next_c
                i2 = (r_idx + 1) * cols + c_idx
                i3 = (r_idx + 1) * cols + next_c
                tris.append([i0, i2, i1])
                tris.append([i1, i2, i3])

        out_geo = Geometry()
        out_geo.set_point_attrib("P", p_arr)
        out_geo.set_point_attrib("Cd", cd_arr)
        out_geo.set_point_attrib("uv", uv_arr)
        out_geo.set_triangles(np.array(tris, dtype=np.uint32))
        out_geo.compute_vertex_normals()
        self.outputs["Geo"].value = out_geo


class RaycastSOPNode(NodeBase):
    """Houdini Ray SOP: projects points of Input 1 onto the closest surface height/position of Input 2."""

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Ray (Project to Surface)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Geometry or points to project."),
        PortSpec(
            "TargetSurface",
            DataType.GEOMETRY,
            None,
            "Reference surface mesh to snap onto.",
        ),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Surface-projected geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "lift_offset",
            "float",
            0.0,
            label="Surface Offset",
            min_val=-20.0,
            max_val=20.0,
            step=0.05,
            unit="m",
        ),
        ParamSpec(
            "blend",
            "float",
            1.0,
            label="Projection Blend",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        ),
        ParamSpec(
            "transfer_normal",
            "bool",
            True,
            label="Transfer Target @N",
            description="Copy surface normal @N from closest target point.",
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        target: Geometry | None = self.inputs["TargetSurface"].get_data()

        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return
        if target is None or target.num_points == 0:
            self.outputs["Geo"].value = in_geo.clone()
            return

        out_geo = in_geo.clone()
        lift = float(self.get_param("lift_offset"))
        blend = float(np.clip(self.get_param("blend"), 0.0, 1.0))
        transfer_n = bool(self.get_param("transfer_normal"))

        src_p = in_geo.points["P"].copy()
        tgt_p = target.points["P"]

        # Find nearest surface sample in XZ plane using cKDTree for O((N + M) log M) scaling
        src_xz = src_p[:, [0, 2]]
        tgt_xz = tgt_p[:, [0, 2]]
        tree = cKDTree(tgt_xz)
        _, nearest_idx = tree.query(src_xz, k=1)

        snapped_y = tgt_p[nearest_idx, 1] + lift
        src_p[:, 1] = src_p[:, 1] + blend * (snapped_y - src_p[:, 1])
        out_geo.set_point_attrib("P", src_p.astype(np.float32))

        if (
            transfer_n
            and "N" in target.points
            and target.points["N"].shape[0] == target.num_points
        ):
            tgt_n = target.points["N"][nearest_idx]
            out_geo.set_point_attrib("N", tgt_n.astype(np.float32))
        elif out_geo.num_primitives > 0:
            out_geo.compute_vertex_normals()

        self.outputs["Geo"].value = out_geo


class StylizedDamageSOPNode(NodeBase):
    """
    Procedurally carves chunky triangular notches, chipped corners, and chisel dents into geometry edges,
    mimicking the hand-carved stone monoliths and weathered wood props of Hearthstone, Dofus, and Arcane.
    """

    context = NodeContext.SOP
    category = "Modifiers"
    title = "Stylized Damage (Chunky Notches)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Geometry to chisel and chip."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Chipped and chiseled geometry with @damaged mask."),
    ]
    PARAM_SPECS = [
        ParamSpec("seed", "int", 42, label="Random Seed", min_val=0, max_val=9999),
        ParamSpec("chip_count", "int", 6, label="Chip Count", min_val=1, max_val=48),
        ParamSpec("chip_radius", "float", 0.45, label="Chip Radius", min_val=0.05, max_val=3.0, step=0.05),
        ParamSpec("chip_depth", "float", 0.22, label="Chip Depth", min_val=0.01, max_val=2.0, step=0.02),
        ParamSpec("angularity", "float", 1.6, label="Angularity / Bevel Falloff", min_val=0.5, max_val=4.0, step=0.1),
        ParamSpec(
            "tint_damage",
            "bool",
            True,
            label="Tint Raw Chipped Core",
            description="Colors chipped gouges with freshly exposed core stone/wood pigment.",
        ),
        ParamSpec("damage_color", "vec3", (0.88, 0.84, 0.78), label="Exposed Core Color"),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        p = in_geo.points["P"].copy()
        n_pts = p.shape[0]

        seed = int(self.get_param("seed"))
        chip_count = max(1, int(self.get_param("chip_count")))
        chip_radius = max(0.01, float(self.get_param("chip_radius")))
        chip_depth = max(0.001, float(self.get_param("chip_depth")))
        angularity = max(0.1, float(self.get_param("angularity")))
        tint_damage = bool(self.get_param("tint_damage"))
        dmg_col = np.array(self.get_param("damage_color"), dtype=np.float32)

        rng = np.random.default_rng(seed)

        if "N" in in_geo.points and in_geo.points["N"].shape == p.shape:
            normals = in_geo.points["N"].copy()
        else:
            normals = np.zeros_like(p)
            normals[:, 1] = 1.0

        tree = cKDTree(p)
        damage_mask = np.zeros(n_pts, dtype=np.float32)

        # Candidate selection: pick from exterior/corner points preferentially
        center = np.mean(p, axis=0)
        dist_from_center = np.linalg.norm(p - center, axis=1)
        weights = dist_from_center / (np.sum(dist_from_center) + 1e-6)
        num_candidates = min(chip_count, n_pts)
        chosen_indices = rng.choice(n_pts, size=num_candidates, replace=False, p=weights)

        for c_idx in chosen_indices:
            c_pos = p[c_idx]
            c_norm = normals[c_idx]
            r = chip_radius * float(rng.uniform(0.65, 1.35))
            d_amt = chip_depth * float(rng.uniform(0.7, 1.4))

            # Find local neighborhood within chip radius
            neighbor_indices = tree.query_ball_point(c_pos, r)
            if not neighbor_indices:
                continue

            sub_p = p[neighbor_indices]
            dists = np.linalg.norm(sub_p - c_pos, axis=1)
            falloff = np.power(np.maximum(1.0 - (dists / r), 0.0), angularity)

            # Inward chisel displacement (opposite normal)
            inward_dir = -c_norm
            p[neighbor_indices] += inward_dir * (falloff[:, None] * d_amt)
            damage_mask[neighbor_indices] = np.maximum(damage_mask[neighbor_indices], falloff)

        out_geo.set_point_attrib("P", p.astype(np.float32))
        out_geo.set_point_attrib("damaged", damage_mask.astype(np.float32))

        # Recompute surface normals for affected geometry
        if out_geo.num_primitives > 0:
            out_geo.compute_vertex_normals()

        if tint_damage:
            if "Cd" in in_geo.points and in_geo.points["Cd"].shape[0] == n_pts:
                cd = in_geo.points["Cd"].copy()
            else:
                cd = np.full((n_pts, 3), 0.72, dtype=np.float32)

            cd = cd * (1.0 - damage_mask[:, None]) + dmg_col * damage_mask[:, None]
            out_geo.set_point_attrib("Cd", np.clip(cd, 0.0, 1.0).astype(np.float32))

        self.outputs["Geo"].value = out_geo
