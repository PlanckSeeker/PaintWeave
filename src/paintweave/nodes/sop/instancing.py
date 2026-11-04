"""SOP Instancing & Composition: Scatter, CopyToPoints, and Multi-Material Merge."""

from __future__ import annotations

import numpy as np

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.engine.geometry.geometry import Geometry


class ScatterSOPNode(NodeBase):
    """Houdini Scatter SOP: scatters points procedurally across primitive surfaces using area-weighted probability."""

    context = NodeContext.SOP
    category = "Instancing & Topology"
    title = "Scatter"

    INPUT_PORTS = [
        PortSpec(
            "Geo", DataType.GEOMETRY, None, "Surface mesh to scatter points onto."
        ),
    ]
    OUTPUT_PORTS = [
        PortSpec(
            "Geo",
            DataType.GEOMETRY,
            None,
            "Point cloud with interpolated @P, @N, and @Cd.",
        ),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "num_points",
            "int",
            150,
            label="Point Count",
            min_val=1,
            max_val=50000,
            step=10,
        ),
        ParamSpec(
            "seed", "int", 42, label="Random Seed", min_val=0, max_val=999999, step=1
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None or in_geo.num_primitives == 0:
            self.outputs["Geo"].value = Geometry()
            return

        n_pts = max(1, int(self.get_param("num_points")))
        seed = int(self.get_param("seed"))
        rng = np.random.default_rng(seed)

        pts = in_geo.points["P"]
        tris = in_geo.primitives["triangles"]

        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]

        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) * 0.5
        total_area = np.sum(areas)

        if total_area < 1e-8:
            self.outputs["Geo"].value = Geometry()
            return

        probs = areas / total_area
        chosen_tris = rng.choice(len(tris), size=n_pts, p=probs)
        cv0, cv1, cv2 = v0[chosen_tris], v1[chosen_tris], v2[chosen_tris]

        r1, r2 = rng.random(n_pts), rng.random(n_pts)
        sqrt_r1 = np.sqrt(r1)
        u = (1.0 - sqrt_r1)[:, None]
        v = (sqrt_r1 * (1.0 - r2))[:, None]
        w = (sqrt_r1 * r2)[:, None]

        scattered_p = u * cv0 + v * cv1 + w * cv2

        out_geo = Geometry()
        out_geo.set_point_attrib("P", scattered_p.astype(np.float32))

        if "N" in in_geo.points:
            n_arr = in_geo.points["N"]
            cn0 = n_arr[tris[:, 0]][chosen_tris]
            cn1 = n_arr[tris[:, 1]][chosen_tris]
            cn2 = n_arr[tris[:, 2]][chosen_tris]
            scattered_n = u * cn0 + v * cn1 + w * cn2
            lengths = np.linalg.norm(scattered_n, axis=1, keepdims=True)
            scattered_n = np.where(lengths < 1e-8, 1.0, scattered_n / lengths)
            out_geo.set_point_attrib("N", scattered_n.astype(np.float32))

        if "Cd" in in_geo.points:
            cd_arr = in_geo.points["Cd"]
            cd0 = cd_arr[tris[:, 0]][chosen_tris]
            cd1 = cd_arr[tris[:, 1]][chosen_tris]
            cd2 = cd_arr[tris[:, 2]][chosen_tris]
            scattered_cd = u * cd0 + v * cd1 + w * cd2
            out_geo.set_point_attrib("Cd", scattered_cd.astype(np.float32))

        if "uv" in in_geo.points and in_geo.points["uv"].shape[0] == in_geo.num_points:
            uv_arr = in_geo.points["uv"]
            uv0 = uv_arr[tris[:, 0]][chosen_tris]
            uv1 = uv_arr[tris[:, 1]][chosen_tris]
            uv2 = uv_arr[tris[:, 2]][chosen_tris]
            scattered_uv = u * uv0 + v * uv1 + w * uv2
            out_geo.set_point_attrib("uv", scattered_uv.astype(np.float32))

        self.outputs["Geo"].value = out_geo


class CopyToPointsSOPNode(NodeBase):
    """Houdini Copy to Points SOP: instances source geometry onto target points with scale and normal alignment."""

    context = NodeContext.SOP
    category = "Instancing & Topology"
    title = "Copy to Points"

    INPUT_PORTS = [
        PortSpec("Source", DataType.GEOMETRY, None, "Prototype mesh to instance."),
        PortSpec(
            "TargetPoints", DataType.GEOMETRY, None, "Target points (@P, @N, @pscale)."
        ),
    ]
    OUTPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Instanced combined geometry."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "master_scale",
            "float",
            1.0,
            label="Master Scale",
            min_val=0.01,
            max_val=25.0,
            step=0.1,
        ),
        ParamSpec(
            "align_to_normal",
            "bool",
            True,
            label="Align to Target @N",
            description="Rotate instances so their Up Axis aligns with target point normals @N.",
        ),
        ParamSpec(
            "up_axis",
            "enum",
            "Y (Up)",
            label="Instance Up Axis",
            options=["Y (Up)", "Z (Forward)"],
        ),
        ParamSpec(
            "inherit_target_color",
            "bool",
            False,
            label="Inherit Target @Cd",
            description="Tint or replace instance @Cd with the target point's @Cd.",
        ),
    ]

    def evaluate(self) -> None:
        src: Geometry | None = self.inputs["Source"].get_data()
        tgt: Geometry | None = self.inputs["TargetPoints"].get_data()

        if src is None or tgt is None or tgt.num_points == 0 or src.num_points == 0:
            self.outputs["Geo"].value = Geometry()
            return

        pts = tgt.points["P"]
        n_pts = pts.shape[0]
        n_src = src.num_points
        master_scale = float(self.get_param("master_scale"))
        align_to_normal = bool(self.get_param("align_to_normal"))
        up_axis_mode = str(self.get_param("up_axis"))
        inherit_cd = bool(self.get_param("inherit_target_color"))

        out_geo = Geometry()
        out_geo.detail["materials"] = src.detail.get("materials", {}).copy()

        if "pscale" in tgt.points:
            pscale = tgt.points["pscale"]
            if pscale.ndim == 1:
                pscale = pscale.reshape(-1, 1)
        else:
            pscale = np.ones((n_pts, 1), dtype=np.float32)

        pscale = pscale * master_scale

        R = None
        if (
            align_to_normal
            and "N" in tgt.points
            and tgt.points["N"].shape[0] == n_pts
        ):
            raw_n = tgt.points["N"]
            n_len = np.linalg.norm(raw_n, axis=1, keepdims=True)
            tgt_n = raw_n / np.where(n_len < 1e-8, 1.0, n_len)

            up = (
                np.array([0.0, 0.0, 1.0], dtype=np.float32)
                if up_axis_mode == "Z (Forward)"
                else np.array([0.0, 1.0, 0.0], dtype=np.float32)
            )
            axis = np.cross(up, tgt_n)
            axis_len = np.linalg.norm(axis, axis=1, keepdims=True)
            dot_val = np.clip(np.sum(tgt_n * up, axis=1, keepdims=True), -1.0, 1.0)
            valid_mask = (axis_len > 1e-6).flatten()

            R = np.zeros((n_pts, 3, 3), dtype=np.float32)
            R[~valid_mask] = np.eye(3, dtype=np.float32)

            # Handle antiparallel 180-degree flip cleanly
            anti_mask = (~valid_mask) & (dot_val.flatten() < -0.999)
            if np.any(anti_mask):
                flip_mat = np.eye(3, dtype=np.float32)
                flip_mat[1, 1] = -1.0
                flip_mat[2, 2] = -1.0
                R[anti_mask] = flip_mat

            if np.any(valid_mask):
                axis_norm = axis[valid_mask] / axis_len[valid_mask]
                angle = np.arccos(dot_val[valid_mask, 0])

                kx, ky, kz = (
                    axis_norm[:, 0],
                    axis_norm[:, 1],
                    axis_norm[:, 2],
                )
                c, s = np.cos(angle), np.sin(angle)
                omc = 1.0 - c

                R[valid_mask, 0, 0] = c + kx * kx * omc
                R[valid_mask, 0, 1] = kx * ky * omc - kz * s
                R[valid_mask, 0, 2] = kx * kz * omc + ky * s
                R[valid_mask, 1, 0] = ky * kx * omc + kz * s
                R[valid_mask, 1, 1] = c + ky * ky * omc
                R[valid_mask, 1, 2] = ky * kz * omc - kx * s
                R[valid_mask, 2, 0] = kz * kx * omc - ky * s
                R[valid_mask, 2, 1] = kz * ky * omc + kx * s
                R[valid_mask, 2, 2] = c + kz * kz * omc

        for k, v in src.points.items():
            if k == "P":
                if R is not None:
                    rot_p = np.einsum("tij,sj->tsi", R, v)
                else:
                    rot_p = np.tile(v, (n_pts, 1, 1))
                scaled_p = rot_p * pscale[:, np.newaxis, :]
                final_p = scaled_p + pts[:, np.newaxis, :]
                out_geo.set_point_attrib("P", final_p.reshape(-1, 3))
            elif k == "N":
                if R is not None:
                    rot_n = np.einsum("tij,sj->tsi", R, v)
                    out_geo.set_point_attrib("N", rot_n.reshape(-1, 3))
                else:
                    tiled_n = np.tile(v, (n_pts,) + (1,) * (v.ndim - 1))
                    out_geo.set_point_attrib("N", tiled_n)
            elif (
                k == "Cd"
                and inherit_cd
                and "Cd" in tgt.points
                and tgt.points["Cd"].shape == (n_pts, 3)
            ):
                tgt_cd = np.repeat(tgt.points["Cd"], n_src, axis=0)
                out_geo.set_point_attrib("Cd", tgt_cd.astype(np.float32))
            else:
                tiled_attr = np.tile(v, (n_pts,) + (1,) * (v.ndim - 1))
                out_geo.set_point_attrib(k, tiled_attr)

        src_tris = src.primitives["triangles"]
        src_mats = src.ensure_material_ids()
        offsets = np.arange(n_pts, dtype=np.uint32) * n_src
        tiled_tris = (src_tris[None, :, :] + offsets[:, None, None]).reshape(-1, 3)
        tiled_mats = np.tile(src_mats, n_pts).astype(np.int32)

        out_geo.set_triangles(tiled_tris, tiled_mats)
        self.outputs["Geo"].value = out_geo


class MergeSOPNode(NodeBase):
    """Houdini Merge SOP: concatenates two geometry streams while preserving multi-material shader bindings."""

    context = NodeContext.SOP
    category = "Instancing & Topology"
    title = "Merge"

    INPUT_PORTS = [
        PortSpec("Geo1", DataType.GEOMETRY, None, "First geometry stream."),
        PortSpec("Geo2", DataType.GEOMETRY, None, "Second geometry stream."),
    ]
    OUTPUT_PORTS = [
        PortSpec(
            "Geo",
            DataType.GEOMETRY,
            None,
            "Merged geometry with unified material table.",
        ),
    ]

    def evaluate(self) -> None:
        g1: Geometry | None = self.inputs["Geo1"].get_data()
        g2: Geometry | None = self.inputs["Geo2"].get_data()

        if g1 is None and g2 is None:
            self.outputs["Geo"].value = Geometry()
            return
        if g1 is None or g1.num_points == 0:
            self.outputs["Geo"].value = g2.clone() if g2 is not None else Geometry()
            return
        if g2 is None or g2.num_points == 0:
            self.outputs["Geo"].value = g1.clone()
            return

        out_geo = Geometry()

        # 1. Merge Point Attributes
        all_pt_keys = set(g1.points.keys()).union(set(g2.points.keys()))
        for k in all_pt_keys:
            if k in g1.points and g1.points[k].shape[0] == g1.num_points:
                arr1 = g1.points[k]
            else:
                ref_shape = g2.points[k].shape[1:] if k in g2.points else (3,)
                arr1 = np.zeros((g1.num_points,) + ref_shape, dtype=np.float32)

            if k in g2.points and g2.points[k].shape[0] == g2.num_points:
                arr2 = g2.points[k]
            else:
                ref_shape = g1.points[k].shape[1:] if k in g1.points else (3,)
                arr2 = np.zeros((g2.num_points,) + ref_shape, dtype=np.float32)

            out_geo.set_point_attrib(k, np.vstack([arr1, arr2]))

        # 2. Merge Primitives & Re-index Material IDs so both streams keep their distinct VOP shaders!
        t1 = g1.primitives["triangles"]
        t2 = g2.primitives["triangles"] + g1.num_points
        tris = np.vstack([t1, t2]).astype(np.uint32)

        mats1_dict = g1.detail.get("materials", {})
        mats2_dict = g2.detail.get("materials", {})
        merged_materials = mats1_dict.copy()

        m1 = g1.ensure_material_ids().copy()
        m2 = g2.ensure_material_ids().copy()

        next_mat_id = (max(merged_materials.keys()) + 1) if merged_materials else 0
        remap_m2: dict[int, int] = {}
        for old_id, payload in mats2_dict.items():
            new_id = next_mat_id
            next_mat_id += 1
            merged_materials[new_id] = payload
            remap_m2[old_id] = new_id

        if remap_m2 and len(m2) > 0:
            m2_remapped = m2.copy()
            for old_id, new_id in remap_m2.items():
                m2_remapped[m2 == old_id] = new_id
            m2 = m2_remapped

        merged_mat_ids = np.concatenate([m1, m2]).astype(np.int32)
        out_geo.detail["materials"] = merged_materials
        out_geo.set_triangles(tris, merged_mat_ids)

        self.outputs["Geo"].value = out_geo
