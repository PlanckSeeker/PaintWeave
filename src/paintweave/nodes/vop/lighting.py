"""VOP Lighting & Surface Shading Operators: IsoLight, RimLight, Fresnel, and CelTooner."""

from __future__ import annotations

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.engine.shader.payload import ShaderPayload


class IsoDirectionalLightVOPNode(NodeBase):
    """Generates directional isometric sun lighting GLSL snippet using dynamic uniforms."""

    context = NodeContext.VOP
    category = "Lighting"
    title = "Iso Sun Light"

    INPUT_PORTS = [
        PortSpec(
            "LightDir",
            DataType.VEC3,
            (0.6, 0.85, 0.5),
            "Light direction / tint vector.",
        ),
        PortSpec("Ambient", DataType.FLOAT, 0.22, "Ambient floor intensity."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Lit GLSL ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "intensity",
            "float",
            1.35,
            label="Sun Intensity",
            min_val=0.0,
            max_val=10.0,
            step=0.1,
        ),
    ]

    def evaluate(self) -> None:
        light_dir = self.inputs["LightDir"].get_data()
        ambient = float(self.inputs["Ambient"].get_data())
        intensity = float(self.get_param("intensity"))

        u_dir = f"u_LightDir_{self.id[:6]}"
        u_amb = f"u_Ambient_{self.id[:6]}"
        body = (
            "    vec3 final_rgb = v_color;\n"
            "    {\n"
            "        vec3 L = normalize(vec3(0.6, 0.8, 0.5));\n"
            f"        float lum = max(dot(N, L), {u_amb});\n"
            f"        vec3 pulse_rgb = {u_dir};\n"
            f"        final_rgb = v_color * lum * (pulse_rgb * {intensity:.3f});\n"
            "    }\n"
        )

        payload = ShaderPayload(name=self.name, fragment_body=body)
        payload.uniforms[u_dir] = ("vec3", light_dir)
        payload.uniforms[u_amb] = ("float", ambient)
        self.outputs["Shader"].value = payload


class RimLightVOPNode(NodeBase):
    """Adds an arcanepunk glowing rim or edge light effect based on surface normal angles."""

    context = NodeContext.VOP
    category = "Lighting"
    title = "Rim Light"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Shader with rim lighting."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "rim_intensity",
            "float",
            1.8,
            label="Rim Intensity",
            min_val=0.0,
            max_val=10.0,
            step=0.1,
        ),
        ParamSpec("rim_color", "color", (0.20, 0.90, 0.82), label="Rim Color"),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        intensity = float(self.get_param("rim_intensity"))
        col = self.get_param("rim_color")
        r, g, b = float(col[0]), float(col[1]), float(col[2])

        snippet = (
            f"{base_body}\n"
            "    // --- Rim Light ---\n"
            "    {\n"
            "        vec3 view_dir = vec3(0.0, 0.0, 1.0);\n"
            "        float rim = 1.0 - max(dot(N, view_dir), 0.0);\n"
            f"        rim = pow(rim, 2.2) * {intensity:.3f};\n"
            f"        final_rgb += vec3({r:.3f}, {g:.3f}, {b:.3f}) * rim;\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class FresnelVOPNode(NodeBase):
    """Computes a view-dependent Fresnel falloff for magical shields and iridescent auras."""

    context = NodeContext.VOP
    category = "Lighting"
    title = "Fresnel Aura"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Shader with Fresnel aura."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "exponent",
            "float",
            3.0,
            label="Fresnel Power",
            min_val=0.2,
            max_val=12.0,
            step=0.2,
        ),
        ParamSpec("aura_color", "color", (0.60, 0.22, 0.92), label="Aura Color"),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}
        exp = float(self.get_param("exponent"))
        col = self.get_param("aura_color")
        r, g, b = float(col[0]), float(col[1]), float(col[2])

        snippet = (
            f"{base_body}\n"
            "    // --- Fresnel Aura ---\n"
            "    {\n"
            "        vec3 v_dir = vec3(0.0, 0.0, 1.0);\n"
            f"        float fresnel = pow(1.0 - clamp(dot(N, v_dir), 0.0, 1.0), {exp:.2f});\n"
            f"        final_rgb = mix(final_rgb, vec3({r:.3f}, {g:.3f}, {b:.3f}), fresnel * 0.75);\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class CelToonerVOPNode(NodeBase):
    """Quantizes lighting intensity into discrete cel-shaded bands."""

    context = NodeContext.VOP
    category = "Lighting"
    title = "Cel Toon Quantizer"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Cel-shaded ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec("bands", "int", 4, label="Toon Bands", min_val=2, max_val=16, step=1),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        bands = max(2, int(self.get_param("bands")))
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        quant_snippet = (
            f"{base_body}\n"
            "    {\n"
            f"        final_rgb = floor(final_rgb * {float(bands):.1f}) / {float(bands):.1f};\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=quant_snippet,
            uniforms=uniforms,
        )
