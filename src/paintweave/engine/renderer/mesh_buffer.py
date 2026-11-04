"""Multi-Material GPU Mesh Buffer manager supporting per-primitive ShaderPayload batches and GLSL program caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import moderngl
import numpy as np

from paintweave.engine.geometry.geometry import Geometry
from paintweave.engine.shader.payload import ShaderPayload


@dataclass
class MaterialDrawBatch:
    """Single ModernGL index buffer + VAO bound to a specific ShaderPayload or default program."""

    mat_id: int
    ebo: moderngl.Buffer
    vao: moderngl.VertexArray
    program: moderngl.Program
    shader_payload: ShaderPayload | None
    owns_program: bool
    index_count: int


class GPUMeshBuffer:
    """
    Manages dynamic ModernGL VBOs and multi-material EBO/VAO draw batches for a columnar Geometry table.
    Caches compiled ModernGL programs by (vertex_shader, fragment_shader) source pair to eliminate
    per-frame GLSL recompilation during 60 FPS timeline playback.
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx: moderngl.Context = ctx
        self.vbo_p: moderngl.Buffer | None = None
        self.vbo_n: moderngl.Buffer | None = None
        self.vbo_cd: moderngl.Buffer | None = None
        self.vbo_uv: moderngl.Buffer | None = None

        self.batches: list[MaterialDrawBatch] = []
        self._program_cache: dict[tuple[str, str], moderngl.Program] = {}
        self.point_vao: moderngl.VertexArray | None = None
        self.point_count: int = 0
        self.index_count: int = 0

    def _build_vao(
        self,
        program: moderngl.Program,
        ebo: moderngl.Buffer | None = None,
    ) -> moderngl.VertexArray | None:
        content: list[tuple[Any, str, str]] = []
        if self.vbo_p is not None and "in_position" in program:
            content.append((self.vbo_p, "3f", "in_position"))
        if self.vbo_n is not None and "in_normal" in program:
            content.append((self.vbo_n, "3f", "in_normal"))
        if self.vbo_cd is not None and "in_color" in program:
            content.append((self.vbo_cd, "3f", "in_color"))
        if self.vbo_uv is not None and "in_uv" in program:
            content.append((self.vbo_uv, "2f", "in_uv"))

        if not content:
            return None
        if ebo is not None:
            return self.ctx.vertex_array(
                program, content, index_buffer=ebo, index_element_size=4
            )
        return self.ctx.vertex_array(program, content)

    def _get_or_compile_program(
        self, vert_src: str, frag_src: str
    ) -> moderngl.Program:
        key = (vert_src, frag_src)
        cached = self._program_cache.get(key)
        if cached is not None:
            return cached
        prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        self._program_cache[key] = prog
        return prog

    def upload(
        self,
        geo: Geometry | None,
        default_program: moderngl.Program,
        default_shader: ShaderPayload | None = None,
    ) -> list[str]:
        """
        Upload geometry attributes and partition primitive draw batches by `@material_id`.
        Reuses cached ModernGL shader programs across frames when GLSL source is unchanged.
        Returns a list of any shader compilation errors encountered on material batches.
        """
        self.release(release_programs=False)
        errors: list[str] = []

        if geo is None or geo.num_points == 0:
            return errors

        n_pts = geo.num_points
        p_data = np.ascontiguousarray(geo.points["P"], dtype=np.float32)

        if "N" in geo.points and geo.points["N"].shape[0] == n_pts:
            n_data = np.ascontiguousarray(geo.points["N"], dtype=np.float32)
        else:
            n_data = np.zeros((n_pts, 3), dtype=np.float32)
            n_data[:, 1] = 1.0

        if "Cd" in geo.points and geo.points["Cd"].shape[0] == n_pts:
            cd_data = np.ascontiguousarray(geo.points["Cd"], dtype=np.float32)
        else:
            cd_data = np.full((n_pts, 3), 0.72, dtype=np.float32)

        if "uv" in geo.points and geo.points["uv"].shape[0] == n_pts:
            uv_data = np.ascontiguousarray(geo.points["uv"], dtype=np.float32)
        else:
            uv_data = np.zeros((n_pts, 2), dtype=np.float32)

        self.vbo_p = self.ctx.buffer(p_data.tobytes())
        self.vbo_n = self.ctx.buffer(n_data.tobytes())
        self.vbo_cd = self.ctx.buffer(cd_data.tobytes())
        self.vbo_uv = self.ctx.buffer(uv_data.tobytes())
        self.point_count = n_pts

        # Point-cloud fallback (e.g., SOP_Scatter output with 0 triangles)
        if geo.num_primitives == 0:
            self.point_vao = self._build_vao(default_program, ebo=None)
            return errors

        tris = np.ascontiguousarray(geo.primitives["triangles"], dtype=np.uint32)
        mat_ids = geo.ensure_material_ids()
        materials_map: dict[int, ShaderPayload] = geo.detail.get("materials", {})

        unique_mats = np.unique(mat_ids)
        total_indices = 0

        for mat_id_val in unique_mats:
            mid = int(mat_id_val)
            sub_tris = np.ascontiguousarray(tris[mat_ids == mid], dtype=np.uint32)
            if sub_tris.size == 0:
                continue

            ebo = self.ctx.buffer(sub_tris.tobytes())
            payload = materials_map.get(mid) if mid >= 0 else None

            prog = default_program
            owns_prog = False
            active_payload = default_shader

            if payload is not None:
                try:
                    vert_src = payload.build_vertex_shader()
                    frag_src = payload.build_fragment_shader()
                    prog = self._get_or_compile_program(vert_src, frag_src)
                    active_payload = payload
                except Exception as exc:
                    err_line = str(exc).splitlines()[0] if str(exc) else str(exc)
                    errors.append(f"Material[{mid}] Shader Error: {err_line}")
                    prog = default_program

            vao = self._build_vao(prog, ebo=ebo)
            if vao is not None:
                self.batches.append(
                    MaterialDrawBatch(
                        mat_id=mid,
                        ebo=ebo,
                        vao=vao,
                        program=prog,
                        shader_payload=active_payload,
                        owns_program=owns_prog,
                        index_count=int(sub_tris.size),
                    )
                )
                total_indices += int(sub_tris.size)
            else:
                ebo.release()

        self.index_count = total_indices
        return errors

    def render(self, mvp_bytes: bytes, model_bytes: bytes, time_val: float) -> None:
        """Render all multi-material primitive batches (or point cloud if 0 triangles)."""
        if self.batches:
            for batch in self.batches:
                prog = batch.program
                if "u_mvp" in prog:
                    prog["u_mvp"].write(mvp_bytes)
                if "u_model" in prog:
                    prog["u_model"].write(model_bytes)
                if "TIME" in prog:
                    prog["TIME"].value = time_val

                if batch.shader_payload is not None:
                    for u_name, (
                        _u_type,
                        u_val,
                    ) in batch.shader_payload.uniforms.items():
                        if u_name in prog:
                            try:
                                prog[u_name].value = u_val
                            except Exception:
                                pass

                batch.vao.render(moderngl.TRIANGLES)
        elif self.point_vao is not None and self.point_count > 0:
            self.ctx.point_size = 5.0
            self.point_vao.render(moderngl.POINTS)

    def clear_program_cache(self) -> None:
        """Release all cached ModernGL shader programs."""
        for prog in self._program_cache.values():
            prog.release()
        self._program_cache.clear()

    def release(self, release_programs: bool = True) -> None:
        for batch in self.batches:
            batch.vao.release()
            batch.ebo.release()
            if batch.owns_program and release_programs:
                batch.program.release()
        self.batches.clear()

        if release_programs:
            self.clear_program_cache()

        if self.point_vao is not None:
            self.point_vao.release()
            self.point_vao = None

        for buf in (self.vbo_p, self.vbo_n, self.vbo_cd, self.vbo_uv):
            if buf is not None:
                buf.release()
        self.vbo_p = self.vbo_n = self.vbo_cd = self.vbo_uv = None
        self.point_count = 0
        self.index_count = 0
