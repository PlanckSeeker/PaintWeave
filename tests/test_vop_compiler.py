"""Tests for VOP GLSL shader composition, scoping, and NWShader uniform reflection."""

from __future__ import annotations

import moderngl

from paintweave.core.datatypes import CodeString
from paintweave.core.registry import NodeRegistry
from paintweave.engine.renderer.mesh_buffer import GPUMeshBuffer
from paintweave.engine.shader.nwshader import NWShaderCompiler
from paintweave.nodes import register_all_nodes
from paintweave.nodes.sop.primitives import GridSOPNode


def test_vop_chain_and_nwshader_reflection() -> None:
    register_all_nodes()

    iso = NodeRegistry.create_node("VOP_IsoLight")
    toon = NodeRegistry.create_node("VOP_CelToon")
    dither = NodeRegistry.create_node("VOP_BayerDither")

    iso.outputs["Shader"].connect(toon.inputs["Shader"])
    toon.outputs["Shader"].connect(dither.inputs["Shader"])
    dither.cook()

    payload = dither.outputs["Shader"].value
    frag_glsl = payload.build_fragment_shader()
    assert "#version 330 core" in frag_glsl
    assert "floor(final_rgb *" in frag_glsl
    assert "float bayer[16]" in frag_glsl

    # NWShader uniform reflection
    nw = NodeRegistry.create_node("VOP_NWShader")
    nw.cook()
    assert nw.error_message is None
    assert "bend" in nw.params
    assert "pulse_speed" in nw.params
    assert "bio_color" in nw.params
    assert "bend" in nw.dynamic_param_specs


def test_high01_chaining_identical_vops_compiles_without_redefinition() -> None:
    """[HIGH-01] Chaining identical VOP operators in series must wrap local variables in `{ ... }` blocks."""
    register_all_nodes()
    ctx = moderngl.create_standalone_context()
    try:
        vop_types = [
            "VOP_RimLight",
            "VOP_Fresnel",
            "VOP_CelToon",
            "VOP_Checkerboard",
            "VOP_Noise",
            "VOP_ColorRamp",
            "VOP_SimplexNoise3D",
            "VOP_UVChecker",
            "VOP_BayerDither",
            "VOP_Blend",
            "VOP_Gamma",
            "VOP_Wrangle",
        ]
        for vtype in vop_types:
            n1 = NodeRegistry.create_node(vtype)
            n2 = NodeRegistry.create_node(vtype)
            n1.outputs["Shader"].connect(n2.inputs["Shader"])
            n2.cook()
            payload = n2.outputs["Shader"].value
            assert payload is not None
            vert_src = payload.build_vertex_shader()
            frag_src = payload.build_fragment_shader()
            # Compile on actual OpenGL driver to verify zero variable redefinition errors
            prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            prog.release()
    finally:
        ctx.release()


def test_high02_gpu_mesh_buffer_program_cache() -> None:
    """[HIGH-02] GPUMeshBuffer must cache compiled ModernGL programs across repeated upload() calls."""
    register_all_nodes()
    ctx = moderngl.create_standalone_context()
    try:
        grid_node = GridSOPNode("grid")
        grid_node.cook()
        geo = grid_node.outputs["Geo"].value

        rim = NodeRegistry.create_node("VOP_RimLight")
        rim.cook()
        shader_payload = rim.outputs["Shader"].value
        geo.assign_material(shader_payload)

        default_prog = ctx.program(
            vertex_shader=shader_payload.build_vertex_shader(),
            fragment_shader=shader_payload.build_fragment_shader(),
        )
        mesh_buf = GPUMeshBuffer(ctx)

        errors1 = mesh_buf.upload(geo, default_prog, None)
        assert not errors1
        assert len(mesh_buf._program_cache) == 1
        cached_prog = next(iter(mesh_buf._program_cache.values()))

        # Second upload (simulating 60 FPS timeline tick) must reuse the exact same compiled Program object
        errors2 = mesh_buf.upload(geo, default_prog, None)
        assert not errors2
        assert len(mesh_buf._program_cache) == 1
        assert next(iter(mesh_buf._program_cache.values())) is cached_prog
        assert mesh_buf.batches[0].program is cached_prog

        mesh_buf.release(release_programs=True)
        assert len(mesh_buf._program_cache) == 0
        default_prog.release()
    finally:
        ctx.release()


def test_high05_and_high06_nwshader_uniforms_and_balanced_braces() -> None:
    """[HIGH-05 & HIGH-06] NWShader custom vec3/int/vec2/vec4 defaults, stale uniform pruning, and nested braces."""
    register_all_nodes()
    nw = NodeRegistry.create_node("VOP_NWShader")

    script_v1 = CodeString(
        """shader_type spatial;
uniform int steps : hint_range(1, 16) = 6;
uniform vec2 uv_tile = vec2(3.5, 4.5);
uniform vec3 custom_tint : source_color = vec3(0.95, 0.25, 0.15);
uniform vec4 rim_rgba = vec4(0.1, 0.2, 0.3, 0.8);

void vertex() { VERTEX.y += 0.1; }

void fragment() {
    if (steps > 2) {
        ALBEDO = custom_tint * vec3(uv_tile.x, uv_tile.y, rim_rgba.w);
}
    }
"""
    )
    nw.set_param("code", script_v1)
    nw.cook()
    assert nw.error_message is None
    assert nw.params["steps"] == 6
    assert nw.params["uv_tile"] == (3.5, 4.5)
    assert nw.params["custom_tint"] == (0.95, 0.25, 0.15)
    assert nw.params["rim_rgba"] == (0.1, 0.2, 0.3, 0.8)
    # Old defaults from DEFAULT_NWSHADER must be pruned
    assert "bend" not in nw.params
    assert "bend" not in nw.dynamic_param_specs
    assert "bio_color" not in nw.params

    # Verify nested braces and indented closing brace in fragment() extracted properly
    compiled = NWShaderCompiler.compile_script(str(script_v1))
    assert "VERTEX.y += 0.1;" in compiled["vertex_glsl"]
    assert "if (steps > 2)" in compiled["fragment_glsl"]
    assert "ALBEDO = custom_tint" in compiled["fragment_glsl"]
