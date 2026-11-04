"""Composable GLSL ShaderPayload for VOP operator chains and multi-material primitive bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_VERTEX_SHADER = """#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_color = in_color;
    v_uv = in_uv;
    gl_Position = u_mvp * world;
}
"""

DEFAULT_FRAGMENT_SHADER = """#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 light_dir = normalize(vec3(0.6, 0.85, 0.5));
    vec3 fill_dir = normalize(vec3(-0.5, 0.25, -0.6));
    vec3 N = normalize(v_normal);
    float key = max(dot(N, light_dir), 0.0);
    float fill = max(dot(N, fill_dir), 0.0) * 0.25;
    float amb = 0.18;
    vec3 shaded = v_color * (amb + key * 0.85 + fill);
    frag_color = vec4(shaded, 1.0);
}
"""


VOP_GLSL_HELPERS = """
float pw_hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float pw_snoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(pw_hash(i + vec3(0,0,0)), pw_hash(i + vec3(1,0,0)), f.x),
                   mix(pw_hash(i + vec3(0,1,0)), pw_hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(pw_hash(i + vec3(0,0,1)), pw_hash(i + vec3(1,0,1)), f.x),
                   mix(pw_hash(i + vec3(0,1,1)), pw_hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

float pw_fbm(vec3 p, int octaves, float lacunarity, float gain) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 8; ++i) {
        if (i >= octaves) break;
        v += a * pw_snoise(p);
        p = p * lacunarity + vec3(13.1, 7.7, 3.3);
        a *= gain;
    }
    return clamp(v, 0.0, 1.0);
}
"""


@dataclass
class ShaderPayload:
    """Represents a compiled or composable GLSL program produced by a VOP chain."""

    name: str = "VOP_Shader"
    vertex_src: str = DEFAULT_VERTEX_SHADER
    fragment_body: str = "    vec3 final_rgb = v_color;\n"
    uniforms: dict[str, tuple[str, Any]] = field(default_factory=dict)
    is_complete_program: bool = False

    def build_vertex_shader(self) -> str:
        return self.vertex_src

    def build_fragment_shader(self) -> str:
        if self.is_complete_program:
            return self.fragment_body

        uniform_decls = "".join(
            f"uniform {glsl_type} {k};\n" for k, (glsl_type, _) in self.uniforms.items()
        )
        return (
            "#version 330 core\n"
            "in vec3 v_world_pos;\n"
            "in vec3 v_normal;\n"
            "in vec3 v_color;\n"
            "in vec2 v_uv;\n"
            "uniform float TIME;\n"
            f"{uniform_decls}"
            "out vec4 frag_color;\n"
            f"{VOP_GLSL_HELPERS}\n"
            "void main() {\n"
            "    vec3 N = normalize(v_normal);\n"
            "    vec3 V = normalize(-v_world_pos);\n"
            f"{self.fragment_body}\n"
            "    frag_color = vec4(final_rgb, 1.0);\n"
            "}\n"
        )
