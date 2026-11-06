"""Stylized 3-Pass Deferred/Hybrid Frame Compositor with Directional Shadows and Screen-Space Outlines."""

from __future__ import annotations

import moderngl
import numpy as np
from pyglm import glm

SHADOW_DEPTH_VERTEX_SHADER = """#version 330 core
in vec3 in_position;
uniform mat4 u_light_space_matrix;
uniform mat4 u_model;

void main() {
    gl_Position = u_light_space_matrix * u_model * vec4(in_position, 1.0);
}
"""

SHADOW_DEPTH_FRAGMENT_SHADER = """#version 330 core
void main() {
    // ModernGL writes depth automatically
}
"""

COMPOSITE_VERTEX_SHADER = """#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

COMPOSITE_FRAGMENT_SHADER = """#version 330 core
in vec2 v_uv;

uniform sampler2D u_color_tex;
uniform sampler2D u_normal_depth_tex;
uniform vec2 u_texel_size;
uniform float u_outline_width = 1.25;
uniform float u_depth_threshold = 0.0025;
uniform float u_normal_threshold = 0.28;
uniform vec4 u_outline_color = vec4(0.07, 0.08, 0.11, 1.0);
uniform int u_outline_enabled = 1;

out vec4 frag_color;

void main() {
    vec4 base_color = texture(u_color_tex, v_uv);
    if (u_outline_enabled == 0) {
        frag_color = base_color;
        return;
    }

    vec2 offset = u_texel_size * u_outline_width;

    // 3x3 Sobel edge detection on Normal (RGB) and Linear Depth (A)
    float d_gx = 0.0;
    float d_gy = 0.0;
    vec3 n_gx = vec3(0.0);
    vec3 n_gy = vec3(0.0);

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 sample_uv = v_uv + vec2(float(x), float(y)) * offset;
            vec4 nd = texture(u_normal_depth_tex, sample_uv);
            vec3 n = nd.rgb * 2.0 - 1.0;
            float d = nd.a;

            float kx = (x != 0 && y == 0) ? float(x) * 2.0 : float(x);
            float ky = (y != 0 && x == 0) ? float(y) * 2.0 : float(y);

            d_gx += d * kx;
            d_gy += d * ky;
            n_gx += n * kx;
            n_gy += n * ky;
        }
    }

    float depth_edge = sqrt(d_gx * d_gx + d_gy * d_gy);
    float norm_edge = length(n_gx) + length(n_gy);

    float is_edge = 0.0;
    if (depth_edge > u_depth_threshold || norm_edge > u_normal_threshold) {
        is_edge = 1.0;
    }

    frag_color = mix(base_color, u_outline_color, is_edge * u_outline_color.a);
}
"""


class FrameCompositor:
    """
    Orchestrates the 3-pass stylized rendering pipeline:
      Pass 1: Directional Light Shadow Map (2048x2048 Depth FBO)
      Pass 2: Geometry + G-Buffer MRT Pass (Color + Normal/Depth)
      Pass 3: Screen-Space Stylized Outline & Composite Pass (Target FBO)
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx: moderngl.Context = ctx

        # Pass 1: Shadow Map (fixed 2048x2048)
        self.shadow_size: tuple[int, int] = (2048, 2048)
        self.shadow_tex: moderngl.Texture = self.ctx.depth_texture(self.shadow_size)
        self.shadow_fbo: moderngl.Framebuffer = self.ctx.framebuffer(
            depth_attachment=self.shadow_tex
        )
        self.shadow_prog: moderngl.Program = self.ctx.program(
            vertex_shader=SHADOW_DEPTH_VERTEX_SHADER,
            fragment_shader=SHADOW_DEPTH_FRAGMENT_SHADER,
        )

        # Pass 2: G-Buffer Offscreen MRT
        self.width: int = 1
        self.height: int = 1
        self.color_tex: moderngl.Texture | None = None
        self.normal_depth_tex: moderngl.Texture | None = None
        self.depth_rb: moderngl.Renderbuffer | None = None
        self.mrt_fbo: moderngl.Framebuffer | None = None

        # Pass 3: Fullscreen Composite Quad
        self.composite_prog: moderngl.Program = self.ctx.program(
            vertex_shader=COMPOSITE_VERTEX_SHADER,
            fragment_shader=COMPOSITE_FRAGMENT_SHADER,
        )
        quad_verts = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 0.0,
                 1.0,  1.0, 1.0, 1.0,
            ],
            dtype=np.float32,
        )
        self.quad_vbo: moderngl.Buffer = self.ctx.buffer(quad_verts.tobytes())
        self.quad_vao: moderngl.VertexArray = self.ctx.vertex_array(
            self.composite_prog,
            [(self.quad_vbo, "2f 2f", "in_vert", "in_uv")],
        )

        # Stylized Rendering Options
        self.shadows_enabled: bool = True
        self.outlines_enabled: bool = True
        self.outline_width: float = 1.25
        self.depth_threshold: float = 0.0025
        self.normal_threshold: float = 0.28
        self.outline_color: tuple[float, float, float, float] = (0.07, 0.08, 0.11, 1.0)

    def resize(self, width: int, height: int) -> None:
        w = max(width, 1)
        h = max(height, 1)
        if self.mrt_fbo is not None and w == self.width and h == self.height:
            return

        self.width = w
        self.height = h

        if self.mrt_fbo is not None:
            self.mrt_fbo.release()
            self.mrt_fbo = None
        if self.color_tex is not None:
            self.color_tex.release()
            self.color_tex = None
        if self.normal_depth_tex is not None:
            self.normal_depth_tex.release()
            self.normal_depth_tex = None
        if self.depth_rb is not None:
            self.depth_rb.release()
            self.depth_rb = None

        self.color_tex = self.ctx.texture((w, h), 4, dtype="f1")
        self.normal_depth_tex = self.ctx.texture((w, h), 4, dtype="f2")
        self.depth_rb = self.ctx.depth_renderbuffer((w, h))
        self.mrt_fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex, self.normal_depth_tex],
            depth_attachment=self.depth_rb,
        )

    def get_light_space_matrix(
        self,
        bounds_min: np.ndarray | None = None,
        bounds_max: np.ndarray | None = None,
    ) -> glm.mat4:
        """Compute an orthographic light view-projection matrix fitted to scene bounds."""
        light_dir = glm.normalize(glm.vec3(0.6, 0.85, 0.5))

        if bounds_min is not None and bounds_max is not None:
            center_arr = (bounds_min + bounds_max) * 0.5
            extent = max(float(np.linalg.norm(bounds_max - bounds_min)), 6.0)
            center = glm.vec3(float(center_arr[0]), float(center_arr[1]), float(center_arr[2]))
        else:
            center = glm.vec3(0.0, 0.0, 0.0)
            extent = 12.0

        radius = extent * 0.75
        light_pos = center + light_dir * (radius * 1.8)
        light_view = glm.lookAt(light_pos, center, glm.vec3(0.0, 1.0, 0.0))
        light_proj = glm.ortho(-radius, radius, -radius, radius, 1.0, radius * 3.5)
        return light_proj * light_view

    def begin_shadow_pass(self) -> None:
        """Pass 1: Bind 2048x2048 Depth FBO and clear depth."""
        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_size[0], self.shadow_size[1])
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(depth=1.0)

    def begin_geometry_pass(
        self,
        width: int,
        height: int,
        clear_color: tuple[float, float, float, float] = (0.055, 0.065, 0.088, 1.0),
    ) -> None:
        """Pass 2: Bind G-Buffer MRT FBO, bind Shadow Map to Texture Unit 0, clear buffers."""
        self.resize(width, height)
        assert self.mrt_fbo is not None
        self.mrt_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.clear(clear_color[0], clear_color[1], clear_color[2], clear_color[3], depth=1.0)

        # Bind Pass 1 shadow depth texture to Texture Unit 0
        self.shadow_tex.use(location=0)

    def composite_to_target(
        self,
        target_fbo: moderngl.Framebuffer,
        width: int,
        height: int,
    ) -> None:
        """Pass 3: Screen-Space Stylized Outline & Color Composite onto target FBO."""
        if self.color_tex is None or self.normal_depth_tex is None:
            return

        target_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Bind G-Buffer attachments
        self.color_tex.use(location=0)
        self.normal_depth_tex.use(location=1)

        prog = self.composite_prog
        if "u_color_tex" in prog:
            prog["u_color_tex"].value = 0
        if "u_normal_depth_tex" in prog:
            prog["u_normal_depth_tex"].value = 1
        if "u_texel_size" in prog:
            prog["u_texel_size"].value = (1.0 / max(width, 1), 1.0 / max(height, 1))
        if "u_outline_width" in prog:
            prog["u_outline_width"].value = float(self.outline_width)
        if "u_depth_threshold" in prog:
            prog["u_depth_threshold"].value = float(self.depth_threshold)
        if "u_normal_threshold" in prog:
            prog["u_normal_threshold"].value = float(self.normal_threshold)
        if "u_outline_color" in prog:
            prog["u_outline_color"].value = self.outline_color
        if "u_outline_enabled" in prog:
            prog["u_outline_enabled"].value = 1 if self.outlines_enabled else 0

        self.quad_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def release(self) -> None:
        """Release all allocated ModernGL textures, framebuffers, and programs."""
        if self.mrt_fbo is not None:
            self.mrt_fbo.release()
            self.mrt_fbo = None
        if self.color_tex is not None:
            self.color_tex.release()
            self.color_tex = None
        if self.normal_depth_tex is not None:
            self.normal_depth_tex.release()
            self.normal_depth_tex = None
        if self.depth_rb is not None:
            self.depth_rb.release()
            self.depth_rb = None

        self.shadow_fbo.release()
        self.shadow_tex.release()
        self.shadow_prog.release()

        self.quad_vao.release()
        self.quad_vbo.release()
        self.composite_prog.release()
