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


class ArcaneRampVOPNode(NodeBase):
    """
    Applies stylized Arcane / Fortiche painterly lighting with a hyper-saturated warm terminator band,
    bridging the transition between key sunlight and cool ambient bounce shadow.
    """

    context = NodeContext.VOP
    category = "Lighting"
    title = "Arcane Painterly Ramp"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Painterly Arcane-lit ShaderPayload."),
    ]
    PARAM_SPECS = [
        ParamSpec("key_tint", "vec3", (1.05, 0.98, 0.90), label="Sun / Key Color"),
        ParamSpec("terminator_tint", "vec3", (0.98, 0.35, 0.12), label="Saturated Terminator Bridge"),
        ParamSpec("shadow_tint", "vec3", (0.18, 0.28, 0.42), label="Deep Ambient Shadow"),
        ParamSpec("terminator_pos", "float", 0.45, label="Terminator Position", min_val=0.05, max_val=0.95, step=0.05),
        ParamSpec("terminator_width", "float", 0.22, label="Terminator Width", min_val=0.02, max_val=0.6, step=0.02),
        ParamSpec("feather", "float", 0.12, label="Gouache Feathering", min_val=0.01, max_val=0.4, step=0.02),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        key_c = self.get_param("key_tint")
        term_c = self.get_param("terminator_tint")
        shd_c = self.get_param("shadow_tint")
        t_pos = float(self.get_param("terminator_pos"))
        t_width = float(self.get_param("terminator_width"))
        feather = max(0.001, float(self.get_param("feather")))

        u_key = f"u_ArcaneKey_{self.id[:6]}"
        u_term = f"u_ArcaneTerm_{self.id[:6]}"
        u_shd = f"u_ArcaneShd_{self.id[:6]}"

        uniforms[u_key] = ("vec3", key_c)
        uniforms[u_term] = ("vec3", term_c)
        uniforms[u_shd] = ("vec3", shd_c)

        snippet = (
            f"{base_body}\n"
            "    // --- Arcane Painterly Terminator Ramp ---\n"
            "    {\n"
            "        vec3 L = normalize(vec3(0.6, 0.85, 0.5));\n"
            "        float ndotl = dot(N, L) * 0.5 + 0.5;\n"
            f"        float t_start = {t_pos:.3f} - {t_width:.3f} * 0.5;\n"
            f"        float t_end = {t_pos:.3f} + {t_width:.3f} * 0.5;\n"
            f"        float in_sun = smoothstep(t_end - {feather:.3f}, t_end + {feather:.3f}, ndotl);\n"
            f"        float in_term = smoothstep(t_start - {feather:.3f}, t_start + {feather:.3f}, ndotl) * (1.0 - in_sun);\n"
            f"        float in_shadow = 1.0 - smoothstep(t_start - {feather:.3f}, t_start + {feather:.3f}, ndotl);\n"
            f"        vec3 ramp_color = {u_key} * in_sun + {u_term} * in_term + {u_shd} * in_shadow;\n"
            "        final_rgb = final_rgb * ramp_color;\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class EdgeGlintVOPNode(NodeBase):
    """
    Renders hand-painted gouache/chalk edge wear and specular glints along facets and beveled edges,
    popularized by Hearthstone stone monoliths and Dofus stylized crystal props.
    """

    context = NodeContext.VOP
    category = "Lighting"
    title = "Hand-Painted Edge Glint"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "ShaderPayload with gouache edge glints."),
    ]
    PARAM_SPECS = [
        ParamSpec("glint_color", "vec3", (1.0, 0.96, 0.88), label="Chalky Glint Color"),
        ParamSpec("intensity", "float", 1.4, label="Glint Intensity", min_val=0.0, max_val=5.0, step=0.1),
        ParamSpec("glint_width", "float", 2.5, label="Edge Sensitivity", min_val=0.5, max_val=8.0, step=0.25),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        col = self.get_param("glint_color")
        intensity = float(self.get_param("intensity"))
        width = float(self.get_param("glint_width"))

        u_col = f"u_GlintCol_{self.id[:6]}"
        uniforms[u_col] = ("vec3", col)

        snippet = (
            f"{base_body}\n"
            "    // --- Hand-Painted Edge Glint ---\n"
            "    {\n"
            "        vec3 f_dx = dFdx(N);\n"
            "        vec3 f_dy = dFdy(N);\n"
            "        float edge_deriv = length(f_dx) + length(f_dy);\n"
            "        float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);\n"
            f"        float glint = clamp(edge_deriv * {width:.2f} + rim * 0.5, 0.0, 1.0);\n"
            f"        final_rgb = mix(final_rgb, {u_col}, glint * {intensity:.2f} * 0.65);\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class BrushImpastoVOPNode(NodeBase):
    """
    Overlays directional hog-bristle brush drag and impasto canvas texture onto surface shading,
    giving 3D surfaces the tactile physical texture of oil and gouache paintings.
    """

    context = NodeContext.VOP
    category = "Lighting"
    title = "Brush Stroke Impasto"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "ShaderPayload with impasto oil bristle texture."),
    ]
    PARAM_SPECS = [
        ParamSpec("stroke_scale", "float", 18.0, label="Stroke Scale", min_val=2.0, max_val=100.0, step=1.0),
        ParamSpec("stroke_stretch", "float", 3.5, label="Bristle Anisotropy", min_val=1.0, max_val=10.0, step=0.25),
        ParamSpec("intensity", "float", 0.25, label="Impasto Depth", min_val=0.0, max_val=1.0, step=0.05),
        ParamSpec("angle_deg", "float", 35.0, label="Stroke Angle (°)", min_val=0.0, max_val=180.0, step=5.0),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        scale = float(self.get_param("stroke_scale"))
        stretch = float(self.get_param("stroke_stretch"))
        intensity = float(self.get_param("intensity"))
        angle = float(self.get_param("angle_deg"))

        snippet = (
            f"{base_body}\n"
            "    // --- Brush Stroke Impasto ---\n"
            "    {\n"
            f"        float rad = radians({angle:.1f});\n"
            "        mat2 rot = mat2(cos(rad), -sin(rad), sin(rad), cos(rad));\n"
            f"        vec2 stroke_uv = rot * (v_uv * vec2({scale:.1f}, {scale * stretch:.1f}));\n"
            "        float bristle = pw_snoise(vec3(stroke_uv, 0.0));\n"
            f"        float stroke_mod = 1.0 + bristle * {intensity:.3f};\n"
            "        final_rgb = clamp(final_rgb * stroke_mod, 0.0, 2.0);\n"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )


class MatCapVOPNode(NodeBase):
    """
    Renders view-space spherical MatCap reflections and stylized lighting (chunky fantasy gold,
    arcane gems, toon metals, and chalky clay), quintessential to Hearthstone and Dofus 3 art styles.
    """

    context = NodeContext.VOP
    category = "Lighting"
    title = "MatCap & Stylized Metal/Gem"

    INPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "Upstream shader chain."),
    ]
    OUTPUT_PORTS = [
        PortSpec("Shader", DataType.SHADER, None, "ShaderPayload with MatCap reflections."),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "preset",
            "str",
            "Chunky Fantasy Gold",
            label="Stylized Preset",
            description="Built-in procedural hand-painted sphere preset.",
            options=["Chunky Fantasy Gold", "Arcane Gem", "Toon Metal", "Chalky Clay"],
        ),
        ParamSpec(
            "blend_mode",
            "str",
            "Overlay",
            label="Blend Mode",
            description="Layering mode over upstream surface color.",
            options=["Overlay", "Multiply", "Add", "Replace", "Mix"],
        ),
        ParamSpec("intensity", "float", 1.0, label="MatCap Intensity", min_val=0.0, max_val=2.0, step=0.05),
        ParamSpec("rim_power", "float", 2.5, label="Fresnel Rim Power", min_val=0.5, max_val=6.0, step=0.25),
        ParamSpec("roughness", "float", 0.15, label="Surface Roughness", min_val=0.0, max_val=1.0, step=0.05),
        ParamSpec("primary_tint", "vec3", (1.0, 0.82, 0.32), label="Primary / Core Tint"),
        ParamSpec("secondary_tint", "vec3", (0.22, 0.65, 0.95), label="Accent / Reflection Tint"),
    ]

    def evaluate(self) -> None:
        in_shader: ShaderPayload | None = self.inputs["Shader"].get_data()
        base_body = in_shader.fragment_body if in_shader else "    vec3 final_rgb = v_color;\n"
        uniforms = in_shader.uniforms.copy() if in_shader else {}

        preset = str(self.get_param("preset"))
        blend_mode = str(self.get_param("blend_mode"))
        intensity = float(self.get_param("intensity"))
        rim_power = float(self.get_param("rim_power"))
        roughness = float(self.get_param("roughness"))
        p_tint = tuple(float(x) for x in self.get_param("primary_tint")[:3])
        s_tint = tuple(float(x) for x in self.get_param("secondary_tint")[:3])

        glsl_primary = f"vec3({p_tint[0]:.3f}, {p_tint[1]:.3f}, {p_tint[2]:.3f})"
        glsl_secondary = f"vec3({s_tint[0]:.3f}, {s_tint[1]:.3f}, {s_tint[2]:.3f})"

        # Procedural MatCap GLSL generator
        if preset == "Arcane Gem":
            sphere_code = (
                f"        float _mc_caustic = pow(clamp(1.0 - _mc_rad, 0.0, 1.0), 1.6);\n"
                f"        float _mc_glint = smoothstep(0.78 - {roughness:.2f} * 0.25, 0.96, max(dot(_mc_uv, normalize(vec2(0.4, 0.7))), 0.0));\n"
                f"        vec3 _mc_chroma = mix({glsl_secondary}, {glsl_primary}, smoothstep(-0.8, 0.8, _mc_uv.y));\n"
                f"        _mc_sphere = {glsl_primary} * 0.35 + {glsl_primary} * _mc_caustic * 1.2;\n"
                f"        _mc_sphere = mix(_mc_sphere, _mc_chroma, _mc_fresnel * 0.9);\n"
                f"        _mc_sphere += vec3(1.0) * _mc_glint * 2.0;\n"
            )
        elif preset == "Toon Metal":
            sphere_code = (
                f"        float _mc_split = smoothstep(-0.06, 0.06, _mc_uv.y + 0.08 * sin(_mc_uv.x * 3.1415));\n"
                f"        vec3 _mc_sky = mix(vec3(0.45, 0.55, 0.70), vec3(0.92, 0.96, 1.0), clamp(_mc_uv.y, 0.0, 1.0));\n"
                f"        vec3 _mc_ground = mix(vec3(0.12, 0.10, 0.15), vec3(0.38, 0.32, 0.28), clamp(-_mc_uv.y, 0.0, 1.0));\n"
                f"        _mc_sphere = mix(_mc_ground, _mc_sky, _mc_split);\n"
                f"        _mc_sphere = mix(_mc_sphere, {glsl_primary}, 0.35);\n"
                f"        float _mc_hspec = step(0.86 - {roughness:.2f} * 0.3, dot(_mc_uv, normalize(vec2(0.6, 0.6))));\n"
                f"        _mc_sphere += vec3(1.0) * _mc_hspec;\n"
            )
        elif preset == "Chalky Clay":
            sphere_code = (
                f"        float _mc_diff = clamp(dot(_mc_uv, normalize(vec2(0.4, 0.6))) * 0.5 + 0.5, 0.0, 1.0);\n"
                f"        _mc_sphere = mix({glsl_secondary}, {glsl_primary}, _mc_diff);\n"
                f"        _mc_sphere += vec3(0.95, 0.92, 0.86) * pow(_mc_diff, 3.0) * 0.45;\n"
            )
        else:  # Chunky Fantasy Gold
            sphere_code = (
                f"        vec2 _mc_sdir = normalize(vec2(0.55, 0.65));\n"
                f"        float _mc_sdot = max(dot(_mc_uv, _mc_sdir), 0.0);\n"
                f"        float _mc_spec = smoothstep(0.72 - {roughness:.2f} * 0.3, 0.94, _mc_sdot);\n"
                f"        float _mc_horiz = smoothstep(0.05, -0.22, _mc_uv.y) * smoothstep(-0.55, -0.22, _mc_uv.y);\n"
                f"        float _mc_bounce = smoothstep(-0.2, -0.9, _mc_uv.y) * 0.65;\n"
                f"        vec3 _mc_bronze = vec3(0.32, 0.16, 0.05);\n"
                f"        _mc_sphere = mix({glsl_primary} * 0.6, {glsl_primary}, smoothstep(-0.5, 0.6, _mc_uv.y));\n"
                f"        _mc_sphere = mix(_mc_sphere, _mc_bronze, _mc_horiz * 0.75);\n"
                f"        _mc_sphere += {glsl_primary} * _mc_bounce;\n"
                f"        _mc_sphere = mix(_mc_sphere, vec3(1.0, 0.98, 0.88), _mc_spec * 1.6);\n"
                f"        _mc_sphere += vec3(1.0, 0.95, 0.75) * _mc_fresnel * 0.85;\n"
            )

        # Blend GLSL
        if blend_mode == "Multiply":
            blend_code = f"        final_rgb = mix(final_rgb, final_rgb * _mc_sphere, {intensity:.3f});\n"
        elif blend_mode == "Add":
            blend_code = f"        final_rgb = final_rgb + _mc_sphere * {intensity:.3f};\n"
        elif blend_mode in ("Replace", "Mix"):
            blend_code = f"        final_rgb = mix(final_rgb, _mc_sphere, {intensity:.3f});\n"
        else:  # Overlay
            blend_code = (
                "        vec3 _mc_base = final_rgb;\n"
                "        vec3 _mc_over = vec3(\n"
                "            _mc_base.r < 0.5 ? (2.0 * _mc_base.r * _mc_sphere.r) : (1.0 - 2.0 * (1.0 - _mc_base.r) * (1.0 - _mc_sphere.r)),\n"
                "            _mc_base.g < 0.5 ? (2.0 * _mc_base.g * _mc_sphere.g) : (1.0 - 2.0 * (1.0 - _mc_base.g) * (1.0 - _mc_sphere.g)),\n"
                "            _mc_base.b < 0.5 ? (2.0 * _mc_base.b * _mc_sphere.b) : (1.0 - 2.0 * (1.0 - _mc_base.b) * (1.0 - _mc_sphere.b))\n"
                "        );\n"
                f"        final_rgb = mix(final_rgb, _mc_over, {intensity:.3f});\n"
            )

        snippet = (
            f"{base_body}\n"
            "    // --- MatCap & Stylized Metal/Gem Shading ---\n"
            "    {\n"
            "        vec3 _mc_up_ref = abs(V.y) > 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);\n"
            "        vec3 _mc_right = normalize(cross(_mc_up_ref, V));\n"
            "        vec3 _mc_up = cross(V, _mc_right);\n"
            "        vec2 _mc_uv = vec2(dot(N, _mc_right), dot(N, _mc_up));\n"
            "        float _mc_rad = clamp(length(_mc_uv), 0.0, 1.0);\n"
            f"        float _mc_fresnel = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), {rim_power:.2f});\n"
            "        vec3 _mc_sphere = vec3(0.0);\n"
            f"{sphere_code}"
            f"{blend_code}"
            "    }\n"
        )
        self.outputs["Shader"].value = ShaderPayload(
            name=self.name,
            fragment_body=snippet,
            uniforms=uniforms,
        )
