"""VOP Programmable Shaders: VOP_Wrangle (inline GLSL) and VOP_NWShader (Godot-style spatial shader)."""

from __future__ import annotations

import re
from typing import Any

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import (
    CodeString,
    DataType,
    NodeContext,
    ParamSpec,
    PortSpec,
)
from paintweave.engine.shader.nwshader import NWShaderCompiler
from paintweave.engine.shader.payload import ShaderPayload

DEFAULT_VOP_WRANGLE = CodeString(
    "// Available: v_world_pos, v_normal, N, v_color, v_uv, TIME, gl_FragCoord\n"
    "// Modify final_rgb directly:\n"
    "float bands = sin(v_world_pos.y * 6.0 - TIME * 2.5) * 0.5 + 0.5;\n"
    "final_rgb = mix(final_rgb, vec3(0.15, 0.88, 0.72), bands * 0.45);\n"
)

_NUMBER_TOKEN_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_numeric_default(
    raw: str | None, count: int, fallback: tuple[float, ...]
) -> tuple[float, ...]:
    """Parse float components from a GLSL literal (supporting scalar broadcast like `vec3(0.5)`)."""
    if not raw:
        return fallback
    inner = raw.strip()
    if "(" in inner and ")" in inner:
        inner = inner[inner.find("(") + 1 : inner.rfind(")")]
    tokens = _NUMBER_TOKEN_RE.findall(inner)
    if len(tokens) >= count:
        return tuple(float(tokens[i]) for i in range(count))
    if len(tokens) == 1:
        val = float(tokens[0])
        return tuple(val for _ in range(count))
    return fallback


class WrangleVOPNode(NodeBase):
    """Executes custom inline GLSL fragment snippets to manipulate color and lighting."""

    context = NodeContext.VOP
    category = "Programmable Shaders"
    title = "GLSL Fragment Wrangle"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Optional upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Custom GLSL ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "code", "code", DEFAULT_VOP_WRANGLE, label="GLSL Snippet", code_lang="glsl"
        ),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        user_code = str(self.get_param("code"))
        indented_code = "\n".join(
            f"        {line}" for line in user_code.splitlines() if line.strip()
        )

        wrangle_snippet = (
            f"{base_body}\n"
            "    // --- VOP Wrangle Start ---\n"
            "    {\n"
            f"{indented_code}\n"
            "    }\n"
            "    // --- VOP Wrangle End ---\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=wrangle_snippet,
            uniforms=uniforms,
        )


DEFAULT_NWSHADER = CodeString(
    """shader_type spatial;

uniform float bend : hint_range(-2.0, 2.0) = 0.0;
uniform float pulse_speed : hint_range(0.0, 10.0) = 2.5;
uniform vec3 bio_color : source_color = vec3(0.14, 0.88, 0.98);

void vertex() {
    float curve = pow(max(VERTEX.y, 0.0) * 0.35, 2.0);
    VERTEX.x += curve * bend;
}

void fragment() {
    float fresnel = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 2.6);
    float pulse = 0.55 + 0.45 * sin(TIME * pulse_speed + VERTEX.y * 4.0);
    float noise_vein = smoothstep(0.45, 0.85, fbm(VERTEX * 2.5));

    ALBEDO = mix(vec3(0.08, 0.11, 0.18), bio_color * 0.45, noise_vein);
    EMISSION = bio_color * (fresnel * pulse * 1.8 + noise_vein * 0.35);
}
"""
)


class NWShaderNode(NodeBase):
    """Full Godot-style spatial shader (`vertex()` + `fragment()`) with automatic uniform UI reflection."""

    context = NodeContext.VOP
    category = "Programmable Shaders"
    title = "NWShader (Spatial GLSL)"

    OUTPUT_PORTS = [
        PortSpec(
            "Shader", DataType.SHADER, None, "Complete vertex + fragment ShaderPayload."
        ),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "code", "code", DEFAULT_NWSHADER, label="NWShader Source", code_lang="glsl"
        ),
    ]

    def evaluate(self) -> None:
        code_str = str(self.get_param("code"))
        try:
            result = NWShaderCompiler.compile_script(code_str)
            self.error_message = None

            # 1. Prune stale dynamic uniforms that were removed or renamed in the shader script
            active_uniforms = set(result["uniforms"].keys())
            for stale_key in list(self.dynamic_param_specs.keys()):
                if stale_key not in active_uniforms:
                    del self.dynamic_param_specs[stale_key]
                    self.params.pop(stale_key, None)

            # 2. Reflect uniform hints & defaults into dynamic ParamSpecs
            for u_name, meta in result["uniforms"].items():
                u_type = meta["type"]
                raw_default = meta.get("default")
                spec: ParamSpec

                if u_type == "float":
                    min_v, max_v = (
                        meta.get("range", (-10.0, 10.0))
                        if meta.get("hint") == "range"
                        else (-100.0, 100.0)
                    )
                    default_f = _parse_numeric_default(raw_default, 1, (1.0,))[0]
                    spec = ParamSpec(
                        name=u_name,
                        param_type="float",
                        default=default_f,
                        label=f"[Uniform] {u_name}",
                        min_val=min_v,
                        max_val=max_v,
                        step=0.05,
                    )
                elif u_type == "int":
                    min_v, max_v = (
                        meta.get("range", (0.0, 32.0))
                        if meta.get("hint") == "range"
                        else (-100.0, 100.0)
                    )
                    default_i = int(
                        round(_parse_numeric_default(raw_default, 1, (1.0,))[0])
                    )
                    spec = ParamSpec(
                        name=u_name,
                        param_type="int",
                        default=default_i,
                        label=f"[Uniform] {u_name}",
                        min_val=int(min_v),
                        max_val=int(max_v),
                        step=1,
                    )
                elif u_type == "vec2":
                    default_v2 = _parse_numeric_default(raw_default, 2, (0.0, 0.0))
                    spec = ParamSpec(
                        name=u_name,
                        param_type="vec2",
                        default=default_v2,
                        label=f"[Uniform] {u_name}",
                    )
                elif u_type == "vec3":
                    default_col = _parse_numeric_default(
                        raw_default, 3, (0.14, 0.88, 0.98)
                    )
                    spec = ParamSpec(
                        name=u_name,
                        param_type="color",
                        default=default_col,
                        label=f"[Uniform] {u_name}",
                    )
                else:  # vec4
                    default_v4 = _parse_numeric_default(
                        raw_default, 4, (0.0, 0.0, 0.0, 1.0)
                    )
                    spec = ParamSpec(
                        name=u_name,
                        param_type="vec4",
                        default=default_v4,
                        label=f"[Uniform] {u_name}",
                    )

                prev_spec = self.dynamic_param_specs.get(u_name)
                if u_name not in self.params:
                    self.add_param(u_name, spec.default, spec=spec)
                else:
                    # Update parameter value if type changed or if user edited the default literal in shader code
                    if (
                        prev_spec is not None
                        and (
                            prev_spec.param_type != spec.param_type
                            or (
                                prev_spec.default != spec.default
                                and self.params[u_name] == prev_spec.default
                            )
                        )
                    ):
                        self.params[u_name] = spec.default
                    self.dynamic_param_specs[u_name] = spec

            runtime_uniforms: dict[str, tuple[str, Any]] = {}
            for u_name, meta in result["uniforms"].items():
                spec = self.dynamic_param_specs[u_name]
                val = self.params.get(u_name, spec.default)
                runtime_uniforms[u_name] = (meta["type"], val)

            payload = ShaderPayload(
                name=self.name,
                vertex_src=result["vertex_glsl"],
                fragment_body=result["fragment_glsl"],
                uniforms=runtime_uniforms,
                is_complete_program=True,
            )
            self.outputs["Shader"].value = payload
        except Exception as exc:
            self.error_message = f"NWShader Syntax Error: {exc}"
