"""VOP Procedural Patterns: Checkerboard, ProceduralNoise, and Height ColorRamp."""

from __future__ import annotations

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.engine.shader.payload import ShaderPayload


class CheckerboardVOPNode(NodeBase):
    """Generates a 3D world-space procedural checkerboard pattern for floors and tiles."""

    context = NodeContext.VOP
    category = "Patterns"
    title = "3D Checkerboard"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Optional upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Checkerboard ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "scale",
            "float",
            4.0,
            label="Tile Frequency",
            min_val=0.1,
            max_val=64.0,
            step=0.5,
        ),
        ParamSpec("color_a", "color", (0.12, 0.14, 0.18), label="Tile Color A"),
        ParamSpec("color_b", "color", (0.68, 0.72, 0.80), label="Tile Color B"),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}
        scale = float(self.get_param("scale"))
        ca = self.get_param("color_a")
        cb = self.get_param("color_b")

        snippet = (
            f"{base_body}\n"
            "    // --- Checkerboard Pattern ---\n"
            "    {\n"
            f"        vec3 check_pos = v_world_pos * {scale:.2f};\n"
            "        vec3 c_mod = floor(mod(check_pos, 2.0));\n"
            "        float check = mod(c_mod.x + c_mod.y + c_mod.z, 2.0);\n"
            f"        vec3 col_a = vec3({float(ca[0]):.3f}, {float(ca[1]):.3f}, {float(ca[2]):.3f});\n"
            f"        vec3 col_b = vec3({float(cb[0]):.3f}, {float(cb[1]):.3f}, {float(cb[2]):.3f});\n"
            "        final_rgb = mix(col_a, col_b, check) * max(dot(N, normalize(vec3(0.6, 0.85, 0.5))), 0.25);\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class ProceduralNoiseVOPNode(NodeBase):
    """Generates procedural pseudo-noise for organic weathering and magical field patterns."""

    context = NodeContext.VOP
    category = "Patterns"
    title = "Procedural Noise"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Noise-modulated ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "frequency",
            "float",
            5.0,
            label="Frequency",
            min_val=0.1,
            max_val=100.0,
            step=0.5,
        ),
        ParamSpec(
            "mix_amount",
            "float",
            0.5,
            label="Mix Strength",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        ),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}
        freq = float(self.get_param("frequency"))
        mix_amt = float(self.get_param("mix_amount"))

        snippet = (
            f"{base_body}\n"
            "    // --- Procedural Noise ---\n"
            "    {\n"
            f"        vec3 p_noise = v_world_pos * {freq:.2f};\n"
            "        float n_val = fract(sin(dot(p_noise, vec3(12.9898, 78.233, 45.543))) * 43758.5453);\n"
            "        vec3 noise_col = mix(vec3(0.08, 0.20, 0.35), vec3(0.75, 0.90, 0.95), n_val);\n"
            f"        final_rgb = mix(final_rgb, noise_col, {mix_amt:.2f});\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class ColorRampVOPNode(NodeBase):
    """Maps world-space Y elevation to a two-stop gradient ramp."""

    context = NodeContext.VOP
    category = "Patterns"
    title = "Elevation Ramp"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Optional upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Elevation-mapped ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "min_y",
            "float",
            -1.0,
            label="Min Height Y",
            min_val=-50.0,
            max_val=50.0,
            step=0.2,
        ),
        ParamSpec(
            "max_y",
            "float",
            1.5,
            label="Max Height Y",
            min_val=-50.0,
            max_val=50.0,
            step=0.2,
        ),
        ParamSpec(
            "low_color", "color", (0.10, 0.26, 0.45), label="Low Elevation Color"
        ),
        ParamSpec(
            "high_color", "color", (0.92, 0.78, 0.48), label="High Elevation Color"
        ),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        min_y = float(self.get_param("min_y"))
        max_y = float(self.get_param("max_y"))
        span = max(abs(max_y - min_y), 1e-4)
        c_low = self.get_param("low_color")
        c_high = self.get_param("high_color")

        snippet = (
            f"{base_body}\n"
            "    {\n"
            f"        float h_t = clamp((v_world_pos.y - ({min_y:.3f})) / {span:.3f}, 0.0, 1.0);\n"
            f"        vec3 ramp_rgb = mix(vec3({float(c_low[0]):.3f}, {float(c_low[1]):.3f}, {float(c_low[2]):.3f}), "
            f"vec3({float(c_high[0]):.3f}, {float(c_high[1]):.3f}, {float(c_high[2]):.3f}), h_t);\n"
            "        final_rgb = ramp_rgb * max(dot(N, normalize(vec3(0.6, 0.85, 0.5))), 0.2);\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class SimplexNoise3DVOPNode(NodeBase):
    """Multi-octave 3D Fractal Brownian Motion (FBM) smooth noise VOP supporting world-space or UV coordinates."""

    context = NodeContext.VOP
    category = "Patterns"
    title = "3D FBM Noise"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Optional upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "3D FBM Noise ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "coord_space",
            "enum",
            "World Position",
            label="Coordinate Space",
            options=["World Position", "UV Coordinates"],
        ),
        ParamSpec(
            "frequency",
            "float",
            2.5,
            label="Frequency",
            min_val=0.05,
            max_val=50.0,
            step=0.25,
        ),
        ParamSpec(
            "octaves",
            "int",
            4,
            label="Octaves",
            min_val=1,
            max_val=8,
            step=1,
        ),
        ParamSpec(
            "lacunarity",
            "float",
            2.02,
            label="Lacunarity",
            min_val=1.1,
            max_val=4.0,
            step=0.1,
        ),
        ParamSpec(
            "gain",
            "float",
            0.5,
            label="Roughness / Gain",
            min_val=0.1,
            max_val=0.9,
            step=0.05,
        ),
        ParamSpec("color_a", "color", (0.09, 0.16, 0.28), label="Valley Color"),
        ParamSpec("color_b", "color", (0.32, 0.85, 0.68), label="Peak Color"),
        ParamSpec(
            "mix_amount",
            "float",
            0.75,
            label="Blend Factor",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        ),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        space = str(self.get_param("coord_space"))
        freq = float(self.get_param("frequency"))
        octaves = max(1, min(8, int(self.get_param("octaves"))))
        lac = float(self.get_param("lacunarity"))
        gain = float(self.get_param("gain"))
        ca = self.get_param("color_a")
        cb = self.get_param("color_b")
        mix_amt = float(self.get_param("mix_amount"))

        coord_expr = (
            f"vec3(v_uv * {freq:.3f}, 0.0)"
            if space == "UV Coordinates"
            else f"v_world_pos * {freq:.3f}"
        )

        snippet = (
            f"{base_body}\n"
            "    {\n"
            f"        float fbm_val = pw_fbm({coord_expr}, {octaves}, {lac:.3f}, {gain:.3f});\n"
            f"        vec3 col_a = vec3({float(ca[0]):.3f}, {float(ca[1]):.3f}, {float(ca[2]):.3f});\n"
            f"        vec3 col_b = vec3({float(cb[0]):.3f}, {float(cb[1]):.3f}, {float(cb[2]):.3f});\n"
            "        float lit = max(dot(N, normalize(vec3(0.6, 0.85, 0.5))), 0.22);\n"
            "        vec3 fbm_rgb = mix(col_a, col_b, fbm_val) * lit;\n"
            f"        final_rgb = mix(final_rgb, fbm_rgb, {mix_amt:.3f});\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class UVCheckerVOPNode(NodeBase):
    """Generates a 2D UV-space diagnostic and stylized grid/checkerboard pattern from `@uv`."""

    context = NodeContext.VOP
    category = "Patterns"
    title = "UV Checker Grid"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Optional upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "UV Checker ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "tiles_u",
            "float",
            8.0,
            label="Tiles U",
            min_val=1.0,
            max_val=64.0,
            step=1.0,
        ),
        ParamSpec(
            "tiles_v",
            "float",
            8.0,
            label="Tiles V",
            min_val=1.0,
            max_val=64.0,
            step=1.0,
        ),
        ParamSpec("color_a", "color", (0.14, 0.18, 0.24), label="Cell Color A"),
        ParamSpec("color_b", "color", (0.24, 0.78, 0.62), label="Cell Color B"),
        ParamSpec(
            "mix_amount",
            "float",
            0.85,
            label="Blend Factor",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        ),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = (
            in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        )
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        tu = float(self.get_param("tiles_u"))
        tv = float(self.get_param("tiles_v"))
        ca = self.get_param("color_a")
        cb = self.get_param("color_b")
        mix_amt = float(self.get_param("mix_amount"))

        snippet = (
            f"{base_body}\n"
            "    {\n"
            f"        vec2 uv_scaled = v_uv * vec2({tu:.3f}, {tv:.3f});\n"
            "        vec2 uv_cell = floor(mod(uv_scaled, 2.0));\n"
            "        float uv_chk = mod(uv_cell.x + uv_cell.y, 2.0);\n"
            f"        vec3 col_a = vec3({float(ca[0]):.3f}, {float(ca[1]):.3f}, {float(ca[2]):.3f});\n"
            f"        vec3 col_b = vec3({float(cb[0]):.3f}, {float(cb[1]):.3f}, {float(cb[2]):.3f});\n"
            "        float lit = max(dot(N, normalize(vec3(0.6, 0.85, 0.5))), 0.25);\n"
            "        vec3 chk_rgb = mix(col_a, col_b, uv_chk) * lit;\n"
            f"        final_rgb = mix(final_rgb, chk_rgb, {mix_amt:.3f});\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )
