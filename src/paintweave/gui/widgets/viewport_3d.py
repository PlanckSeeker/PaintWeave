"""ModernGL 3D Viewport with HUD overlay bar, Multi-Material rendering, and Pink Template wireframe."""

from __future__ import annotations

import time

import moderngl
import numpy as np
from pyglm import glm
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from paintweave.engine.geometry.geometry import Geometry
from paintweave.engine.renderer.camera import IsometricCamera
from paintweave.engine.renderer.mesh_buffer import GPUMeshBuffer
from paintweave.engine.shader.payload import (
    DEFAULT_FRAGMENT_SHADER,
    DEFAULT_VERTEX_SHADER,
    ShaderPayload,
)

GRID_VERTEX_SHADER = """#version 330 core
in vec3 in_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

GRID_FRAGMENT_SHADER = """#version 330 core
uniform vec4 u_line_color;
out vec4 frag_color;
void main() {
    frag_color = u_line_color;
}
"""

AXIS_VERTEX_SHADER = """#version 330 core
in vec3 in_position;
in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

AXIS_FRAGMENT_SHADER = """#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""


class Viewport3D(QOpenGLWidget):
    """Interactive Isometric/Perspective ModernGL Viewport with Multi-Material Draw Batching & HUD."""

    shader_error = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.ctx: moderngl.Context | None = None
        self.program: moderngl.Program | None = None
        self.grid_program: moderngl.Program | None = None
        self.axis_program: moderngl.Program | None = None

        self.mesh_buffer: GPUMeshBuffer | None = None
        self.template_buffer: GPUMeshBuffer | None = None

        self.grid_vbo: moderngl.Buffer | None = None
        self.grid_vao: moderngl.VertexArray | None = None
        self.axis_vbo: moderngl.Buffer | None = None
        self.axis_vao: moderngl.VertexArray | None = None
        self.gizmo_vbo: moderngl.Buffer | None = None
        self.gizmo_vao: moderngl.VertexArray | None = None
        self.normals_vbo: moderngl.Buffer | None = None
        self.normals_vao: moderngl.VertexArray | None = None
        self.normals_vertex_count: int = 0

        self.camera: IsometricCamera = IsometricCamera(zoom=3.4)
        self._pending_geo: Geometry | None = None
        self._pending_template_geo: Geometry | None = None
        self._pending_shader: ShaderPayload | None = None
        self._current_frag_src: str = DEFAULT_FRAGMENT_SHADER

        self.wireframe_enabled: bool = False
        self.show_grid: bool = True
        self.show_normals: bool = False

        fmt = QSurfaceFormat()
        fmt.setVersion(4, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSamples(4)
        self.setFormat(fmt)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.last_mouse_pos = None

        self._init_hud_overlay()

    def _init_hud_overlay(self) -> None:
        """Build floating translucent HUD bar at the top of the 3D viewport."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        hud_bar = QWidget(self)
        hud_bar.setStyleSheet(
            "background-color: rgba(14, 16, 21, 200); border: 1px solid #1e2430; border-radius: 6px;"
        )
        hbox = QHBoxLayout(hud_bar)
        hbox.setContentsMargins(8, 4, 8, 4)
        hbox.setSpacing(6)

        self.btn_proj = QPushButton("📐 Ortho (Iso)")
        self.btn_proj.setCheckable(True)
        self.btn_proj.setFixedHeight(24)
        self.btn_proj.setToolTip(
            "Toggle between Isometric Orthographic and Perspective Camera"
        )
        self.btn_proj.toggled.connect(self._on_toggle_projection)

        self.btn_wire = QPushButton("🕸️ Wireframe (W)")
        self.btn_wire.setCheckable(True)
        self.btn_wire.setFixedHeight(24)
        self.btn_wire.toggled.connect(self._on_toggle_wireframe)

        self.btn_grid = QPushButton("🌐 Grid")
        self.btn_grid.setCheckable(True)
        self.btn_grid.setChecked(True)
        self.btn_grid.setFixedHeight(24)
        self.btn_grid.toggled.connect(self._on_toggle_grid)

        self.btn_normals = QPushButton("📏 Normals (N)")
        self.btn_normals.setCheckable(True)
        self.btn_normals.setFixedHeight(24)
        self.btn_normals.setToolTip("Toggle point normal (@N) hair vectors")
        self.btn_normals.toggled.connect(self._on_toggle_normals)

        btn_frame = QPushButton("🎯 Frame (F)")
        btn_frame.setFixedHeight(24)
        btn_frame.clicked.connect(self.frame_active_geometry)

        btn_reset = QPushButton("↺ Reset Cam")
        btn_reset.setFixedHeight(24)
        btn_reset.clicked.connect(self.reset_camera)

        self.stats_lbl = QLabel("Pts: 0  |  Tris: 0  |  Mats: 0")
        self.stats_lbl.setStyleSheet(
            "color: #38bdf8; font-weight: 600; border: none; background: transparent;"
        )

        hbox.addWidget(self.btn_proj)
        hbox.addWidget(self.btn_wire)
        hbox.addWidget(self.btn_grid)
        hbox.addWidget(self.btn_normals)
        hbox.addWidget(btn_frame)
        hbox.addWidget(btn_reset)
        hbox.addStretch(1)
        hbox.addWidget(self.stats_lbl)

        root_layout.addWidget(hud_bar)
        root_layout.addStretch(1)

    def _on_toggle_projection(self, checked: bool) -> None:
        self.camera.perspective = checked
        self.btn_proj.setText("🎥 Perspective" if checked else "📐 Ortho (Iso)")
        self.update()

    def _on_toggle_wireframe(self, checked: bool) -> None:
        self.wireframe_enabled = checked
        self.update()

    def _on_toggle_grid(self, checked: bool) -> None:
        self.show_grid = checked
        self.update()

    def _on_toggle_normals(self, checked: bool) -> None:
        self.show_normals = checked
        self.update()

    def reset_camera(self) -> None:
        self.camera.reset()
        self.update()

    def frame_active_geometry(self) -> None:
        if self._pending_geo is not None and self._pending_geo.num_points > 0:
            self.camera.frame_bounds(
                self._pending_geo.detail["bounds_min"],
                self._pending_geo.detail["bounds_max"],
            )
            self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self.last_mouse_pos = event.position()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.last_mouse_pos is not None and (
            event.buttons()
            & (
                Qt.MouseButton.LeftButton
                | Qt.MouseButton.MiddleButton
                | Qt.MouseButton.RightButton
            )
        ):
            pos = event.position()
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()
            self.last_mouse_pos = pos

            if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) or (
                event.buttons() & Qt.MouseButton.RightButton
            ):
                self.camera.pan(dx, dy)
            else:
                self.camera.orbit(dx, dy)
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self.last_mouse_pos = None
        event.accept()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.camera.zoom = max(0.3, min(self.camera.zoom * factor, 35.0))
        self.update()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_A:
            self.camera.rotate_yaw(-45.0)
            self.update()
            event.accept()
        elif event.key() == Qt.Key.Key_E:
            self.camera.rotate_yaw(45.0)
            self.update()
            event.accept()
        elif event.key() == Qt.Key.Key_W:
            self.btn_wire.setChecked(not self.wireframe_enabled)
            event.accept()
        elif event.key() == Qt.Key.Key_N:
            self.btn_normals.setChecked(not self.show_normals)
            event.accept()
        elif event.key() in (Qt.Key.Key_F, Qt.Key.Key_H):
            self.frame_active_geometry()
            event.accept()
        else:
            super().keyPressEvent(event)

    def initializeGL(self) -> None:
        ctx = moderngl.create_context()
        self.ctx = ctx
        if self.ctx is None:
            return

        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.program = self.ctx.program(
            vertex_shader=DEFAULT_VERTEX_SHADER,
            fragment_shader=DEFAULT_FRAGMENT_SHADER,
        )
        self.mesh_buffer = GPUMeshBuffer(self.ctx)
        self.template_buffer = GPUMeshBuffer(self.ctx)

        self.grid_program = self.ctx.program(
            vertex_shader=GRID_VERTEX_SHADER,
            fragment_shader=GRID_FRAGMENT_SHADER,
        )
        self._init_grid_buffer()

        self.axis_program = self.ctx.program(
            vertex_shader=AXIS_VERTEX_SHADER,
            fragment_shader=AXIS_FRAGMENT_SHADER,
        )
        self._init_axis_buffer()
        self._init_gizmo_buffer()

        if self._pending_shader is not None:
            self._compile_default_shader(self._pending_shader)

        if (
            self._pending_geo is not None
            and self.mesh_buffer is not None
            and self.program is not None
        ):
            self.mesh_buffer.upload(
                self._pending_geo, self.program, self._pending_shader
            )

        if (
            self._pending_template_geo is not None
            and self.template_buffer is not None
            and self.grid_program is not None
        ):
            self.template_buffer.upload(
                self._pending_template_geo, self.grid_program, None
            )

    def _init_grid_buffer(self) -> None:
        if self.ctx is None or self.grid_program is None:
            return
        size = 10.0
        step = 1.0
        lines = []
        floor_y = 0.0
        for i in np.arange(-size, size + step, step):
            lines.append([i, floor_y, -size])
            lines.append([i, floor_y, size])
            lines.append([-size, floor_y, i])
            lines.append([size, floor_y, i])
        grid_data = np.array(lines, dtype=np.float32)
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.grid_program, [(self.grid_vbo, "3f", "in_position")]
        )

    def _init_axis_buffer(self) -> None:
        if self.ctx is None or self.axis_program is None:
            return
        floor_y = 0.0
        axis_data = np.array(
            [
                [0.0, floor_y, 0.0, 0.95, 0.25, 0.30],
                [2.2, floor_y, 0.0, 0.95, 0.25, 0.30],
                [0.0, floor_y, 0.0, 0.25, 0.92, 0.45],
                [0.0, floor_y + 2.2, 0.0, 0.25, 0.92, 0.45],
                [0.0, floor_y, 0.0, 0.25, 0.65, 0.98],
                [0.0, floor_y, 2.2, 0.25, 0.65, 0.98],
            ],
            dtype=np.float32,
        )
        self.axis_vbo = self.ctx.buffer(axis_data.tobytes())
        self.axis_vao = self.ctx.vertex_array(
            self.axis_program,
            [(self.axis_vbo, "3f 3f", "in_position", "in_color")],
        )

    def _init_gizmo_buffer(self) -> None:
        if self.ctx is None or self.axis_program is None:
            return
        gizmo_data = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.2, 0.2],
                [1.0, 0.0, 0.0, 1.0, 0.2, 0.2],
                [0.0, 0.0, 0.0, 0.2, 1.0, 0.3],
                [0.0, 1.0, 0.0, 0.2, 1.0, 0.3],
                [0.0, 0.0, 0.0, 0.2, 0.6, 1.0],
                [0.0, 0.0, 1.0, 0.2, 0.6, 1.0],
            ],
            dtype=np.float32,
        )
        self.gizmo_vbo = self.ctx.buffer(gizmo_data.tobytes())
        self.gizmo_vao = self.ctx.vertex_array(
            self.axis_program,
            [(self.gizmo_vbo, "3f 3f", "in_position", "in_color")],
        )

    def resizeGL(self, w: int, h: int) -> None:
        if self.ctx:
            self.ctx.viewport = (0, 0, max(w, 1), max(h, 1))
            self.camera.aspect = max(w, 1) / float(max(h, 1))

    def update_scene(
        self,
        geo: Geometry | None,
        fallback_shader: ShaderPayload | None = None,
        template_geo: Geometry | None = None,
    ) -> None:
        self._pending_geo = geo
        self._pending_shader = fallback_shader
        self._pending_template_geo = template_geo

        if geo is not None:
            n_mats = len(geo.detail.get("materials", {}))
            self.stats_lbl.setText(
                f"Pts: {geo.num_points:,}  |  Tris: {geo.num_primitives:,}  |  Bound Mats: {n_mats}"
            )
        else:
            self.stats_lbl.setText("Pts: 0  |  Tris: 0  |  Bound Mats: 0")

        if self.ctx is not None:
            self.makeCurrent()
            if fallback_shader is not None:
                self._compile_default_shader(fallback_shader)
            else:
                self._reset_to_studio_shader()

            if self.mesh_buffer is not None and self.program is not None:
                batch_errors = self.mesh_buffer.upload(
                    geo, self.program, fallback_shader
                )
                if batch_errors:
                    self.shader_error.emit(batch_errors[0])

            if self.template_buffer is not None and self.grid_program is not None:
                self.template_buffer.upload(template_geo, self.grid_program, None)

            self._upload_normals_buffer(geo)
            self.doneCurrent()
            self.update()

    def _upload_normals_buffer(self, geo: Geometry | None) -> None:
        if self.normals_vao is not None:
            self.normals_vao.release()
            self.normals_vao = None
        if self.normals_vbo is not None:
            self.normals_vbo.release()
            self.normals_vbo = None
        self.normals_vertex_count = 0

        if (
            self.ctx is None
            or self.grid_program is None
            or geo is None
            or geo.num_points == 0
            or "N" not in geo.points
            or geo.points["N"].shape != geo.points["P"].shape
        ):
            return

        p = geo.points["P"]
        n = geo.points["N"]
        tips = p + n * 0.22
        segments = np.empty((p.shape[0] * 2, 3), dtype=np.float32)
        segments[0::2] = p
        segments[1::2] = tips
        self.normals_vbo = self.ctx.buffer(segments.tobytes())
        self.normals_vao = self.ctx.vertex_array(
            self.grid_program, [(self.normals_vbo, "3f", "in_position")]
        )
        self.normals_vertex_count = segments.shape[0]

    def _reset_to_studio_shader(self) -> None:
        if self.ctx is None:
            return
        if self._current_frag_src != DEFAULT_FRAGMENT_SHADER:
            try:
                new_prog = self.ctx.program(
                    vertex_shader=DEFAULT_VERTEX_SHADER,
                    fragment_shader=DEFAULT_FRAGMENT_SHADER,
                )
                if self.program is not None:
                    self.program.release()
                self.program = new_prog
                self._current_frag_src = DEFAULT_FRAGMENT_SHADER
            except Exception:
                pass

    def _compile_default_shader(self, shader: ShaderPayload) -> None:
        if self.ctx is None:
            return

        vert_src = shader.build_vertex_shader()
        frag_src = shader.build_fragment_shader()

        if frag_src != self._current_frag_src:
            try:
                new_prog = self.ctx.program(
                    vertex_shader=vert_src, fragment_shader=frag_src
                )
                if self.program is not None:
                    self.program.release()
                self.program = new_prog
                self._current_frag_src = frag_src
                self.shader_error.emit("")
            except Exception as exc:
                err_line = str(exc).split("\n")[0]
                self.shader_error.emit(f"GLSL Link/Compile Error: {err_line}")

    def paintGL(self) -> None:
        if not self.ctx or not self.mesh_buffer:
            return

        dpr = self.devicePixelRatio()
        w = int(self.width() * dpr)
        h = int(self.height() * dpr)

        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.055, 0.065, 0.088, 1.0)

        mvp_mat = self.camera.get_mvp_matrix()
        mvp_bytes = mvp_mat.to_bytes()
        model_bytes = glm.mat4(1.0).to_bytes()
        now_t = time.perf_counter() % 10000.0

        # 1. Ground Grid & World Origin Axes
        if self.show_grid:
            if self.grid_vao is not None and self.grid_program is not None:
                if "u_mvp" in self.grid_program:
                    self.grid_program["u_mvp"].write(mvp_bytes)
                if "u_line_color" in self.grid_program:
                    self.grid_program["u_line_color"].value = (0.18, 0.22, 0.30, 0.65)
                self.grid_vao.render(moderngl.LINES)

            if self.axis_vao is not None and self.axis_program is not None:
                if "u_mvp" in self.axis_program:
                    self.axis_program["u_mvp"].write(mvp_bytes)
                self.axis_vao.render(moderngl.LINES)

        # 2. Render Active SOP Mesh (Multi-Material Batches or Wireframe)
        if self.wireframe_enabled:
            self.ctx.wireframe = True
            self.mesh_buffer.render(mvp_bytes, model_bytes, now_t)
            self.ctx.wireframe = False
        else:
            self.mesh_buffer.render(mvp_bytes, model_bytes, now_t)

        # 3. Render Pink Template / Ghost Reference SOP Wireframe if active
        if (
            self.template_buffer is not None
            and self.template_buffer.index_count > 0
            and self.grid_program is not None
        ):
            self.ctx.wireframe = True
            if "u_line_color" in self.grid_program:
                self.grid_program["u_line_color"].value = (0.95, 0.28, 0.62, 0.55)
            self.template_buffer.render(mvp_bytes, model_bytes, now_t)
            self.ctx.wireframe = False

        # 3b. Render Point Normal (@N) Hair Vectors if enabled
        if (
            self.show_normals
            and self.normals_vao is not None
            and self.normals_vertex_count > 0
            and self.grid_program is not None
        ):
            if "u_mvp" in self.grid_program:
                self.grid_program["u_mvp"].write(mvp_bytes)
            if "u_line_color" in self.grid_program:
                self.grid_program["u_line_color"].value = (0.22, 0.88, 0.98, 0.85)
            self.normals_vao.render(moderngl.LINES)

        # 4. Bottom-Right Corner Orientation Gizmo Pass
        gizmo_size = int(96 * dpr)
        margin = int(14 * dpr)
        self.ctx.viewport = (w - gizmo_size - margin, margin, gizmo_size, gizmo_size)

        self.ctx.disable(moderngl.DEPTH_TEST)
        gizmo_cam = IsometricCamera(zoom=2.0, aspect=1.0)
        gizmo_cam.pitch_deg = self.camera.pitch_deg
        gizmo_cam.yaw_deg = self.camera.yaw_deg

        gizmo_mat = gizmo_cam.get_mvp_matrix()
        if self.gizmo_vao is not None and self.axis_program is not None:
            if "u_mvp" in self.axis_program:
                self.axis_program["u_mvp"].write(gizmo_mat.to_bytes())
            self.gizmo_vao.render(moderngl.LINES)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.viewport = (0, 0, w, h)

    def render_to_image(
        self,
        filepath: str,
        width: int = 1920,
        height: int = 1080,
        include_grid: bool = False,
    ) -> None:
        """Render the active 3D scene offscreen at high resolution and save as a PNG/JPEG image."""
        from PIL import Image

        if self.ctx is None or self.mesh_buffer is None:
            raise RuntimeError("OpenGL context is not initialized.")

        self.makeCurrent()
        fbo = self.ctx.simple_framebuffer((width, height), components=4)
        prev_aspect = self.camera.aspect
        try:
            fbo.use()
            self.ctx.viewport = (0, 0, width, height)
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
            self.ctx.clear(0.055, 0.065, 0.088, 1.0)

            self.camera.aspect = width / float(max(height, 1))
            mvp_bytes = self.camera.get_mvp_matrix().to_bytes()
            model_bytes = glm.mat4(1.0).to_bytes()
            now_t = time.perf_counter() % 10000.0

            if include_grid and self.grid_vao is not None and self.grid_program is not None:
                if "u_mvp" in self.grid_program:
                    self.grid_program["u_mvp"].write(mvp_bytes)
                if "u_line_color" in self.grid_program:
                    self.grid_program["u_line_color"].value = (0.18, 0.22, 0.30, 0.65)
                self.grid_vao.render(moderngl.LINES)

            self.mesh_buffer.render(mvp_bytes, model_bytes, now_t)

            raw_bytes = fbo.read(components=4)
            img = Image.frombytes("RGBA", (width, height), raw_bytes)
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            img.save(filepath)
        finally:
            self.camera.aspect = prev_aspect
            fbo.release()
            self.doneCurrent()
            self.update()
