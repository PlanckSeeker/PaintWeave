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
uniform mat4 u_light_space_matrix = mat4(1.0);

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out vec4 v_light_space_pos;

void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_color = in_color;
    v_uv = in_uv;
    v_light_space_pos = u_light_space_matrix * world;
    gl_Position = u_mvp * world;
}
"""

DEFAULT_FRAGMENT_SHADER = """#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
in vec4 v_light_space_pos;

uniform sampler2D u_shadow_map;
uniform int u_shadow_enabled = 1;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal_depth;

float calc_shadow(vec4 light_space_pos, vec3 normal, vec3 light_dir) {
    if (u_shadow_enabled == 0) return 1.0;
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0)
        return 1.0;

    float bias = max(0.003 * (1.0 - dot(normal, light_dir)), 0.0008);
    float current_depth = proj_coords.z;
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(2048.0, 2048.0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcf_depth = texture(u_shadow_map, proj_coords.xy + vec2(x, y) * texel_size).r;
            shadow += (current_depth - bias > pcf_depth) ? 0.35 : 1.0;
        }
    }
    return shadow / 9.0;
}

void main() {
    vec3 light_dir = normalize(vec3(0.6, 0.85, 0.5));
    vec3 fill_dir = normalize(vec3(-0.5, 0.25, -0.6));
    vec3 N = normalize(v_normal);
    float key = max(dot(N, light_dir), 0.0);
    float fill = max(dot(N, fill_dir), 0.0) * 0.25;
    float amb = 0.18;
    float s_factor = calc_shadow(v_light_space_pos, N, light_dir);
    vec3 shaded = v_color * (amb + key * 0.85 * s_factor + fill);
    out_color = vec4(shaded, 1.0);
    out_normal_depth = vec4(N * 0.5 + 0.5, gl_FragCoord.z);
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
            "in vec4 v_light_space_pos;\n"
            "uniform float TIME;\n"
            "uniform vec3 u_camera_pos = vec3(11.5, 11.5, 11.5);\n"
            "uniform sampler2D u_shadow_map;\n"
            "uniform int u_shadow_enabled = 1;\n"
            f"{uniform_decls}"
            "layout(location = 0) out vec4 out_color;\n"
            "layout(location = 1) out vec4 out_normal_depth;\n"
            f"{VOP_GLSL_HELPERS}\n"
            """
float calc_shadow(vec4 light_space_pos, vec3 normal, vec3 light_dir) {
    if (u_shadow_enabled == 0) return 1.0;
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0)
        return 1.0;

    float bias = max(0.003 * (1.0 - dot(normal, light_dir)), 0.0008);
    float current_depth = proj_coords.z;
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(2048.0, 2048.0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcf_depth = texture(u_shadow_map, proj_coords.xy + vec2(x, y) * texel_size).r;
            shadow += (current_depth - bias > pcf_depth) ? 0.35 : 1.0;
        }
    }
    return shadow / 9.0;
}
"""
            "void main() {\n"
            "    vec3 N = normalize(v_normal);\n"
            "    vec3 V = normalize(u_camera_pos - v_world_pos);\n"
            "    vec3 light_dir = normalize(vec3(0.6, 0.85, 0.5));\n"
            f"{self.fragment_body}\n"
            "    float s_factor = calc_shadow(v_light_space_pos, N, light_dir);\n"
            "    vec3 shaded_final = mix(final_rgb * 0.35, final_rgb, s_factor);\n"
            "    out_color = vec4(shaded_final, 1.0);\n"
            "    out_normal_depth = vec4(N * 0.5 + 0.5, gl_FragCoord.z);\n"
            "}\n"
        )
