"""VOP Post-Processing & Color Grading Operators: BayerDither, Blend, and Gamma."""

from __future__ import annotations

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.engine.shader.payload import ShaderPayload


class BayerDitherVOPNode(NodeBase):
    """Applies screen-space 4x4 Bayer matrix ordered dithering for stylized arcanepunk shading."""

    context = NodeContext.VOP
    category = "Post & Stylized"
    title = "Bayer 4x4 Dither"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Dithered ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "strength",
            "float",
            0.12,
            label="Dither Spread",
            min_val=0.0,
            max_val=1.0,
            step=0.02,
        ),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}
        strength = float(self.get_param("strength"))

        dither_snippet = (
            f"{base_body}\n"
            "    {\n"
            "        float bayer[16] = float[16](\n"
            "            0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,\n"
            "           12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,\n"
            "            3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,\n"
            "           15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0\n"
            "        );\n"
            "        int bx = int(gl_FragCoord.x) % 4;\n"
            "        int by = int(gl_FragCoord.y) % 4;\n"
            "        float limit = bayer[bx + by * 4];\n"
            f"        final_rgb += (limit - 0.5) * {strength:.3f};\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=dither_snippet,
            uniforms=uniforms,
        )


class BlendVOPNode(NodeBase):
    """Blends current fragment colors with a secondary color tint based on a mix factor."""

    context = NodeContext.VOP
    category = "Post & Stylized"
    title = "Color Blend"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Blended ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "blend_factor",
            "float",
            0.35,
            label="Blend Factor",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        ),
        ParamSpec("tint_color", "color", (0.85, 0.55, 0.15), label="Tint Color"),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}
        factor = float(self.get_param("blend_factor"))
        col = self.get_param("tint_color")

        snippet = (
            f"{base_body}\n"
            "    {\n"
            f"        vec3 secondary_tint = vec3({float(col[0]):.3f}, {float(col[1]):.3f}, {float(col[2]):.3f});\n"
            f"        final_rgb = mix(final_rgb, secondary_tint, {factor:.3f});\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class GammaVOPNode(NodeBase):
    """Applies gamma correction curve to the fragment output."""

    context = NodeContext.VOP
    category = "Post & Stylized"
    title = "Gamma Curve"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Gamma-corrected ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "gamma_value",
            "float",
            2.2,
            label="Gamma",
            min_val=0.2,
            max_val=5.0,
            step=0.1,
        ),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}
        g = max(0.1, float(self.get_param("gamma_value")))

        snippet = (
            f"{base_body}\n"
            "    {\n"
            f"        final_rgb = pow(clamp(final_rgb, 0.0, 1.0), vec3(1.0 / {g:.3f}));\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )
