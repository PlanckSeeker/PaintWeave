"""Godot-inspired spatial shader compiler (`shader_type spatial;`) with uniform hint reflection."""

from __future__ import annotations

import re
from typing import Any

# Matches: uniform <type> <name> [: hint_range(min, max) | : source_color] [= default];
UNIFORM_REGEX = re.compile(
    r"uniform\s+(float|int|vec2|vec3|vec4)\s+([a-zA-Z_]\w*)"
    r"(?:\s*:\s*(hint_range\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)|source_color))?"
    r"(?:\s*=\s*([^;]+))?\s*;",
    re.MULTILINE,
)

STANDARD_LIBRARY_GLSL = """
// --- PaintWeave Procedural Noise & SDF Library ---
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float snoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                   mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                   mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

float fbm(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; ++i) {
        v += a * snoise(p);
        p = p * 2.02 + vec3(13.1, 7.7, 3.3);
        a *= 0.5;
    }
    return v;
}
"""

MASTER_VERTEX_TEMPLATE = """#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_light_space_matrix = mat4(1.0);
uniform float TIME;
uniform vec3 u_camera_pos = vec3(11.5, 11.5, 11.5);

{uniform_declarations}

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out vec3 v_view;
out vec4 v_light_space_pos;

{global_functions}

void main() {{
    vec3 VERTEX = in_position;
    vec3 NORMAL = in_normal;
    vec3 COLOR = in_color;
    vec2 UV = in_uv;

    // --- User Vertex Function ---
{vertex_body}

    vec4 world = u_model * vec4(VERTEX, 1.0);
    v_world_pos = world.xyz;
    v_normal = normalize(mat3(u_model) * NORMAL);
    v_color = COLOR;
    v_uv = UV;
    v_view = normalize(u_camera_pos - v_world_pos);
    v_light_space_pos = u_light_space_matrix * world;
    gl_Position = u_mvp * world;
}}
"""

MASTER_FRAGMENT_TEMPLATE = """#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
in vec3 v_view;
in vec4 v_light_space_pos;

uniform float TIME;
uniform sampler2D u_shadow_map;
uniform int u_shadow_enabled = 1;

{uniform_declarations}

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal_depth;

{global_functions}

float calc_shadow(vec4 light_space_pos, vec3 normal, vec3 light_dir) {{
    if (u_shadow_enabled == 0) return 1.0;
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0)
        return 1.0;

    float bias = max(0.003 * (1.0 - dot(normal, light_dir)), 0.0008);
    float current_depth = proj_coords.z;
    float shadow = 0.0;
    vec2 texel_size = 1.0 / vec2(2048.0, 2048.0);
    for(int x = -1; x <= 1; ++x) {{
        for(int y = -1; y <= 1; ++y) {{
            float pcf_depth = texture(u_shadow_map, proj_coords.xy + vec2(x, y) * texel_size).r;
            shadow += (current_depth - bias > pcf_depth) ? 0.35 : 1.0;
        }}
    }}
    return shadow / 9.0;
}}

void main() {{
    vec3 VERTEX = v_world_pos;
    vec3 NORMAL = normalize(v_normal);
    vec3 COLOR = v_color;
    vec2 UV = v_uv;
    vec3 VIEW = normalize(v_view);

    vec3 ALBEDO = v_color;
    vec3 EMISSION = vec3(0.0);
    float ROUGHNESS = 0.5;
    float METALLIC = 0.0;

    // --- User Fragment Function ---
{fragment_body}

    vec3 L = normalize(vec3(0.6, 0.85, 0.5));
    float ndotl = max(dot(NORMAL, L), 0.15);
    float s_factor = calc_shadow(v_light_space_pos, NORMAL, L);
    vec3 diffuse = ALBEDO * ndotl * s_factor;

    out_color = vec4(diffuse + EMISSION, 1.0);
    out_normal_depth = vec4(NORMAL * 0.5 + 0.5, gl_FragCoord.z);
}}
"""


def _extract_function_block(script_text: str, func_name: str) -> tuple[str, str]:
    """
    Locate `void <func_name>() { ... }` using balanced brace counting (ignoring comments).
    Returns `(inner_body, remaining_script_without_block)`.
    """
    pattern = re.compile(rf"\bvoid\s+{func_name}\s*\(\s*\)\s*\{{")
    match = pattern.search(script_text)
    if not match:
        return "", script_text

    open_brace_idx = match.end() - 1
    depth = 0
    n = len(script_text)
    i = open_brace_idx

    while i < n:
        ch = script_text[i]
        # Skip single-line comments //
        if ch == "/" and i + 1 < n and script_text[i + 1] == "/":
            newline_idx = script_text.find("\n", i + 2)
            if newline_idx == -1:
                break
            i = newline_idx + 1
            continue
        # Skip block comments /* ... */
        if ch == "/" and i + 1 < n and script_text[i + 1] == "*":
            close_comment = script_text.find("*/", i + 2)
            if close_comment == -1:
                raise ValueError(f"Unclosed block comment in void {func_name}().")
            i = close_comment + 2
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                body = script_text[open_brace_idx + 1 : i].strip("\r\n")
                remaining = script_text[: match.start()] + script_text[i + 1 :]
                return body, remaining
        i += 1

    raise ValueError(f"Unclosed brace '{{' in void {func_name}() block.")


class NWShaderCompiler:
    """Transpiles Godot-like spatial shaders into GLSL 330 Core vertex/fragment programs + uniform specs."""

    @classmethod
    def compile_script(cls, script_text: str) -> dict[str, Any]:
        uniform_metadata: dict[str, dict[str, Any]] = {}
        clean_uniform_lines: list[str] = []

        for match in UNIFORM_REGEX.finditer(script_text):
            u_type, u_name, hint_full, h_min, h_max, default_val = match.groups()

            meta: dict[str, Any] = {"type": u_type, "hint": None, "default": None}
            if hint_full:
                if "hint_range" in hint_full:
                    meta["hint"] = "range"
                    meta["range"] = (float(h_min), float(h_max))
                elif "source_color" in hint_full:
                    meta["hint"] = "color"

            if default_val:
                meta["default"] = default_val.strip()

            uniform_metadata[u_name] = meta
            clean_uniform_lines.append(f"uniform {u_type} {u_name};")

        uniform_block = "\n".join(clean_uniform_lines)

        temp = UNIFORM_REGEX.sub("", script_text)
        v_body, temp = _extract_function_block(temp, "vertex")
        f_body, temp = _extract_function_block(temp, "fragment")
        temp = re.sub(r"shader_type\s+\w+\s*;", "", temp)

        global_helpers = f"{STANDARD_LIBRARY_GLSL}\n{temp.strip()}"

        vertex_glsl = MASTER_VERTEX_TEMPLATE.format(
            uniform_declarations=uniform_block,
            global_functions=global_helpers,
            vertex_body=v_body,
        )
        fragment_glsl = MASTER_FRAGMENT_TEMPLATE.format(
            uniform_declarations=uniform_block,
            global_functions=global_helpers,
            fragment_body=f_body,
        )

        return {
            "uniforms": uniform_metadata,
            "vertex_glsl": vertex_glsl,
            "fragment_glsl": fragment_glsl,
        }
