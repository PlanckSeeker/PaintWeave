"""Houdini-inspired columnar Geometry container with Copy-on-Write (CoW) and Multi-Material primitives."""

from __future__ import annotations

from typing import Any

import numpy as np


class Geometry:
    """
    Columnar Geometry container storing attributes across four domains:
      - points:     dict[str, np.ndarray] of shape (N_pts, k)
      - vertices:   dict[str, np.ndarray] of shape (N_vtx, k) including 'point_idx'
      - primitives: dict[str, np.ndarray] of shape (N_prims, k) including 'triangles' (N_prims, 3)
                    and 'material_id' (N_prims,)
      - detail:     dict[str, Any] global metadata including 'bounds_min', 'bounds_max',
                    and 'materials' (dict[int, ShaderPayload])
    """

    def __init__(self) -> None:
        self.points: dict[str, np.ndarray] = {
            "P": np.empty((0, 3), dtype=np.float32),
            "N": np.empty((0, 3), dtype=np.float32),
            "Cd": np.empty((0, 3), dtype=np.float32),
            "uv": np.empty((0, 2), dtype=np.float32),
        }
        self.vertices: dict[str, np.ndarray] = {
            "point_idx": np.empty(0, dtype=np.uint32),
        }
        self.primitives: dict[str, np.ndarray] = {
            "triangles": np.empty((0, 3), dtype=np.uint32),
            "material_id": np.empty(0, dtype=np.int32),
        }
        self.detail: dict[str, Any] = {
            "bounds_min": np.zeros(3, dtype=np.float32),
            "bounds_max": np.zeros(3, dtype=np.float32),
            "materials": {},  # dict[int, ShaderPayload]
        }

    @property
    def num_points(self) -> int:
        return int(self.points["P"].shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices["point_idx"].shape[0])

    @property
    def num_primitives(self) -> int:
        return int(self.primitives["triangles"].shape[0])

    def clone(self) -> Geometry:
        """
        Houdini Copy-on-Write (CoW): creates a new Geometry shell sharing
        underlying NumPy array memory pointers until an operator replaces a column.
        """
        new_geo = Geometry()
        new_geo.points = self.points.copy()
        new_geo.vertices = self.vertices.copy()
        new_geo.primitives = self.primitives.copy()
        new_geo.detail = self.detail.copy()
        if "materials" in self.detail and isinstance(self.detail["materials"], dict):
            new_geo.detail["materials"] = self.detail["materials"].copy()
        return new_geo

    def set_point_attrib(self, name: str, data: np.ndarray) -> None:
        """Assign a newly allocated or copied array to a point attribute column."""
        arr = np.asarray(data, dtype=np.float32)
        if self.num_points > 0 and name != "P" and arr.shape[0] != self.num_points:
            raise ValueError(
                f"Point attribute '{name}' length {arr.shape[0]} != {self.num_points}"
            )
        self.points[name] = arr
        if name == "P":
            self.update_bounds()

    def set_triangles(
        self, tris: np.ndarray, material_ids: np.ndarray | None = None
    ) -> None:
        """Assign triangle primitive topology and synchronize vertex indices + material_id array."""
        tri_arr = np.asarray(tris, dtype=np.uint32).reshape(-1, 3)
        self.primitives["triangles"] = tri_arr
        self.vertices["point_idx"] = tri_arr.ravel()
        if material_ids is not None and len(material_ids) == len(tri_arr):
            self.primitives["material_id"] = np.asarray(material_ids, dtype=np.int32)
        elif "material_id" not in self.primitives or len(
            self.primitives["material_id"]
        ) != len(tri_arr):
            self.primitives["material_id"] = np.full(len(tri_arr), -1, dtype=np.int32)

    def ensure_material_ids(self) -> np.ndarray:
        """Ensure `primitives['material_id']` matches `num_primitives`."""
        n_prims = self.num_primitives
        if (
            "material_id" not in self.primitives
            or len(self.primitives["material_id"]) != n_prims
        ):
            self.primitives["material_id"] = np.full(n_prims, -1, dtype=np.int32)
        return self.primitives["material_id"]

    def assign_material(self, shader_payload: Any, group_name: str = "") -> int:
        """
        Bind a `ShaderPayload` to all primitives or primitives matching a point/primitive group.
        Returns the allocated integer material_id.
        """
        self.ensure_material_ids()
        materials: dict[int, Any] = self.detail.setdefault("materials", {})

        # Allocate next material ID
        mat_id = (max(materials.keys()) + 1) if materials else 0
        materials[mat_id] = shader_payload

        n_prims = self.num_primitives
        if n_prims == 0:
            return mat_id

        mat_arr = self.primitives["material_id"].copy()
        g_clean = group_name.strip()
        if not g_clean:
            mat_arr[:] = mat_id
        else:
            prim_group_key = f"group_{g_clean}"
            if (
                prim_group_key in self.primitives
                and len(self.primitives[prim_group_key]) == n_prims
            ):
                mask = self.primitives[prim_group_key].ravel() > 0.5
                mat_arr[mask] = mat_id
            elif prim_group_key in self.points:
                # Promote point group to primitive group (any or all vertices in group)
                pt_mask = self.points[prim_group_key].ravel() > 0.5
                tris = self.primitives["triangles"]
                prim_mask = (
                    pt_mask[tris[:, 0]] & pt_mask[tris[:, 1]] & pt_mask[tris[:, 2]]
                )
                if not np.any(prim_mask):
                    # Fallback to any vertex in group if strict all-vertex match is empty
                    prim_mask = (
                        pt_mask[tris[:, 0]] | pt_mask[tris[:, 1]] | pt_mask[tris[:, 2]]
                    )
                mat_arr[prim_mask] = mat_id

        self.primitives["material_id"] = mat_arr
        return mat_id

    def update_bounds(self) -> None:
        """Recompute Detail domain bounding box from @P."""
        p = self.points["P"]
        if p.shape[0] > 0:
            self.detail["bounds_min"] = np.min(p, axis=0)
            self.detail["bounds_max"] = np.max(p, axis=0)
        else:
            self.detail["bounds_min"] = np.zeros(3, dtype=np.float32)
            self.detail["bounds_max"] = np.zeros(3, dtype=np.float32)

    def compute_vertex_normals(self) -> None:
        """Vectorized recalculation of smooth point normals (@N) from triangle primitives."""
        p = self.points["P"]
        tris = self.primitives["triangles"]
        if p.shape[0] == 0 or tris.shape[0] == 0:
            return

        normals = np.zeros_like(p, dtype=np.float32)
        v0 = p[tris[:, 0]]
        v1 = p[tris[:, 1]]
        v2 = p[tris[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)

        np.add.at(normals, tris[:, 0], face_normals)
        np.add.at(normals, tris[:, 1], face_normals)
        np.add.at(normals, tris[:, 2], face_normals)

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths < 1e-8, 1.0, lengths)
        self.points["N"] = (normals / lengths).astype(np.float32)

    def export_to_obj(self, filepath: str) -> None:
        """Export columnar geometry into a standard Wavefront .obj file."""
        lines = ["# Exported from PaintWeave Studio v1.0.0\n"]

        if "P" in self.points:
            for p in self.points["P"]:
                lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        if "uv" in self.points and self.points["uv"].shape[0] > 0:
            for uv in self.points["uv"]:
                lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        if "N" in self.points and self.points["N"].shape[0] > 0:
            for n in self.points["N"]:
                lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        if "triangles" in self.primitives:
            has_uv = (
                "uv" in self.points and self.points["uv"].shape[0] == self.num_points
            )
            has_n = "N" in self.points and self.points["N"].shape[0] == self.num_points

            for tri in self.primitives["triangles"]:
                i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                if has_uv and has_n:
                    lines.append(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")
                elif has_n:
                    lines.append(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
                elif has_uv:
                    lines.append(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
                else:
                    lines.append(f"f {i0} {i1} {i2}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def load_from_obj(cls, filepath: str) -> Geometry:
        """Load a Wavefront .obj file into a columnar Geometry instance (supports negative indices & seam UVs)."""
        geo = cls()
        verts: list[list[float]] = []
        uvs: list[list[float]] = []
        norms: list[list[float]] = []
        faces: list[list[int]] = []
        corner_vt_pairs: list[tuple[int, int]] = []
        corner_vn_pairs: list[tuple[int, int]] = []

        def _resolve_idx(raw_token: str, current_len: int) -> int:
            idx = int(raw_token)
            return (current_len + idx) if idx < 0 else (idx - 1)

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "vt" and len(parts) >= 3:
                    uvs.append([float(parts[1]), float(parts[2])])
                elif parts[0] == "vn" and len(parts) >= 4:
                    norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f":
                    face_indices: list[int] = []
                    for p in parts[1:]:
                        sub = p.split("/")
                        v_idx = _resolve_idx(sub[0], len(verts))
                        face_indices.append(v_idx)
                        if len(sub) >= 2 and sub[1]:
                            vt_idx = _resolve_idx(sub[1], len(uvs))
                            corner_vt_pairs.append((v_idx, vt_idx))
                        if len(sub) >= 3 and sub[2]:
                            vn_idx = _resolve_idx(sub[2], len(norms))
                            corner_vn_pairs.append((v_idx, vn_idx))
                    for i in range(1, len(face_indices) - 1):
                        faces.append(
                            [face_indices[0], face_indices[i], face_indices[i + 1]]
                        )

        n_verts = len(verts)
        if n_verts > 0:
            p_arr = np.array(verts, dtype=np.float32)
            geo.set_point_attrib("P", p_arr)
            geo.set_point_attrib(
                "Cd",
                np.tile(
                    np.array([[0.72, 0.76, 0.82]], dtype=np.float32), (n_verts, 1)
                ),
            )

            uv_arr = np.zeros((n_verts, 2), dtype=np.float32)
            if uvs:
                raw_uvs = np.array(uvs, dtype=np.float32)
                if len(uvs) == n_verts:
                    uv_arr = raw_uvs.copy()
                for v_i, vt_i in corner_vt_pairs:
                    if 0 <= v_i < n_verts and 0 <= vt_i < len(uvs):
                        uv_arr[v_i] = raw_uvs[vt_i]
            geo.set_point_attrib("uv", uv_arr)

            has_valid_normals = False
            if norms:
                raw_norms = np.array(norms, dtype=np.float32)
                if len(norms) == n_verts and not corner_vn_pairs:
                    geo.set_point_attrib("N", raw_norms)
                    has_valid_normals = True
                elif corner_vn_pairs:
                    n_accum = np.zeros((n_verts, 3), dtype=np.float32)
                    for v_i, vn_i in corner_vn_pairs:
                        if 0 <= v_i < n_verts and 0 <= vn_i < len(norms):
                            n_accum[v_i] += raw_norms[vn_i]
                    lengths = np.linalg.norm(n_accum, axis=1, keepdims=True)
                    if np.any(lengths > 1e-8):
                        n_arr = np.where(
                            lengths > 1e-8,
                            n_accum / np.where(lengths > 1e-8, lengths, 1.0),
                            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                        )
                        geo.set_point_attrib("N", n_arr.astype(np.float32))
                        has_valid_normals = True

            if not has_valid_normals:
                n_arr = np.zeros((n_verts, 3), dtype=np.float32)
                n_arr[:, 1] = 1.0
                geo.set_point_attrib("N", n_arr)

        if faces:
            geo.set_triangles(np.array(faces, dtype=np.uint32))
            if not norms:
                geo.compute_vertex_normals()

        return geo
