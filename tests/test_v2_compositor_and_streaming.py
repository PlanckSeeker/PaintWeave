"""Unit tests for PaintWeave v2.0 In-Place VBO Streaming and 3-Pass Deferred/Hybrid Compositor."""

import moderngl
import numpy as np
from pyglm import glm

from paintweave.engine.geometry.geometry import Geometry
from paintweave.engine.renderer.compositor import FrameCompositor
from paintweave.engine.renderer.mesh_buffer import GPUMeshBuffer
from paintweave.engine.shader.payload import (
    DEFAULT_FRAGMENT_SHADER,
    DEFAULT_VERTEX_SHADER,
    ShaderPayload,
)


def _create_standalone_context() -> moderngl.Context:
    return moderngl.create_context(standalone=True)


def test_frame_compositor_lifecycle_and_passes() -> None:
    """Verify FrameCompositor initializes, resizes MRT buffers, and executes 3 passes."""
    ctx = _create_standalone_context()
    comp = FrameCompositor(ctx)

    assert comp.shadow_tex is not None
    assert comp.shadow_fbo is not None
    assert comp.shadow_prog is not None
    assert comp.shadows_enabled is True
    assert comp.outlines_enabled is True

    # 1. Test Resize creates G-Buffer MRT attachments
    comp.resize(640, 480)
    assert comp.color_tex is not None
    assert comp.normal_depth_tex is not None
    assert comp.mrt_fbo is not None
    assert comp.width == 640
    assert comp.height == 480

    # 2. Test Light Space Matrix computation
    bmin = np.array([-5.0, 0.0, -5.0], dtype=np.float32)
    bmax = np.array([5.0, 4.0, 5.0], dtype=np.float32)
    lsm = comp.get_light_space_matrix(bmin, bmax)
    assert isinstance(lsm, glm.mat4)

    # 3. Test Pass 1: Shadow Pass
    comp.begin_shadow_pass()
    assert ctx.fbo == comp.shadow_fbo

    # 4. Test Pass 2: Geometry MRT Pass
    comp.begin_geometry_pass(640, 480)
    assert ctx.fbo == comp.mrt_fbo

    # 5. Test Pass 3: Composite Pass into target FBO
    target_fbo = ctx.simple_framebuffer((640, 480))
    comp.composite_to_target(target_fbo, 640, 480)
    assert ctx.fbo == target_fbo

    # Verify toggle options
    comp.shadows_enabled = False
    comp.outlines_enabled = False
    comp.composite_to_target(target_fbo, 640, 480)

    target_fbo.release()
    comp.release()
    assert comp.mrt_fbo is None


def test_gpu_mesh_buffer_in_place_vbo_streaming() -> None:
    """Verify GPUMeshBuffer performs zero-reallocation in-place buffer streaming on identical topology."""
    ctx = _create_standalone_context()
    mesh_buf = GPUMeshBuffer(ctx)
    comp = FrameCompositor(ctx)
    default_prog = ctx.program(
        vertex_shader=DEFAULT_VERTEX_SHADER,
        fragment_shader=DEFAULT_FRAGMENT_SHADER,
    )

    # 1. Create initial quad geometry (4 points, 2 triangles)
    geo = Geometry()
    geo.points["P"] = np.array(
        [
            [-1.0, 0.0, -1.0],
            [ 1.0, 0.0, -1.0],
            [ 1.0, 0.0,  1.0],
            [-1.0, 0.0,  1.0],
        ],
        dtype=np.float32,
    )
    geo.points["N"] = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    geo.primitives["triangles"] = np.array(
        [[0, 1, 2], [0, 2, 3]], dtype=np.uint32
    )
    geo.ensure_material_ids()

    # Initial upload: allocates VBOs
    errors = mesh_buf.upload(
        geo, default_prog, depth_program=comp.shadow_prog
    )
    assert errors == []
    assert mesh_buf.point_count == 4
    assert len(mesh_buf.batches) == 1
    assert mesh_buf.vbo_p is not None

    vbo_p_inst = mesh_buf.vbo_p
    vbo_n_inst = mesh_buf.vbo_n
    batch_inst = mesh_buf.batches[0]
    ebo_inst = batch_inst.ebo
    vao_inst = batch_inst.vao

    # 2. Deform @P (e.g. Wrangle / wave displacement with identical topology)
    geo_deformed = geo.clone()
    geo_deformed.points["P"] = geo.points["P"] + np.array([0.0, 2.5, 0.0], dtype=np.float32)

    errors2 = mesh_buf.upload(
        geo_deformed, default_prog, depth_program=comp.shadow_prog
    )
    assert errors2 == []

    # In-place streaming MUST retain the exact same VBO, EBO, and VAO instances (zero reallocation)
    assert mesh_buf.vbo_p is vbo_p_inst
    assert mesh_buf.vbo_n is vbo_n_inst
    assert mesh_buf.batches[0].ebo is ebo_inst
    assert mesh_buf.batches[0].vao is vao_inst

    # 3. Test Pass 1 render_depth and Pass 2 render
    lsm = comp.get_light_space_matrix()
    lsm_bytes = lsm.to_bytes()
    model_bytes = glm.mat4(1.0).to_bytes()

    comp.begin_shadow_pass()
    mesh_buf.render_depth(lsm_bytes, model_bytes)

    comp.begin_geometry_pass(512, 512)
    mesh_buf.render(
        lsm_bytes,
        model_bytes,
        time_val=1.0,
        light_space_bytes=lsm_bytes,
        shadow_enabled=True,
    )

    # 4. Topology change (different point count): MUST safely reallocate
    geo_rebuilt = Geometry()
    geo_rebuilt.points["P"] = np.zeros((8, 3), dtype=np.float32)
    geo_rebuilt.points["N"] = np.zeros((8, 3), dtype=np.float32)
    geo_rebuilt.primitives["triangles"] = np.zeros((4, 3), dtype=np.uint32)
    geo_rebuilt.ensure_material_ids()

    errors3 = mesh_buf.upload(
        geo_rebuilt, default_prog, depth_program=comp.shadow_prog
    )
    assert errors3 == []
    assert mesh_buf.point_count == 8
    # New VBO allocated
    assert mesh_buf.vbo_p is not vbo_p_inst

    mesh_buf.release()
    comp.release()
    default_prog.release()


def test_vop_shader_payload_mrt_and_shadow_integration() -> None:
    """Verify ShaderPayload generates GLSL fragment source that binds to MRT outputs and shadow maps."""
    payload = ShaderPayload(
        name="CustomCel",
        fragment_body="vec3 final_rgb = vec3(0.2, 0.8, 0.4);\n",
    )
    frag_src = payload.build_fragment_shader()
    assert "layout(location = 0) out vec4 out_color;" in frag_src
    assert "layout(location = 1) out vec4 out_normal_depth;" in frag_src
    assert "uniform sampler2D u_shadow_map;" in frag_src
    assert "calc_shadow" in frag_src

    ctx = _create_standalone_context()
    vert_src = payload.build_vertex_shader()
    prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    assert "u_shadow_map" in prog
    assert "u_light_space_matrix" in prog
    prog.release()
