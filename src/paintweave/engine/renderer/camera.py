"""Hybrid Isometric / Perspective 3D studio camera supporting orbit, pan, zoom, and bounds framing."""

from __future__ import annotations

import math

import numpy as np
from pyglm import glm


class IsometricCamera:
    """3D Viewport camera supporting both Orthographic (Isometric) and Perspective projections."""

    def __init__(
        self, zoom: float = 3.2, aspect: float = 1.0, perspective: bool = False
    ) -> None:
        self.zoom: float = zoom
        self.aspect: float = aspect
        self.perspective: bool = perspective
        self.fov_deg: float = 45.0
        self.pitch_deg: float = math.degrees(
            math.asin(1.0 / math.sqrt(3.0))
        )  # ~35.264°
        self.yaw_deg: float = 45.0
        self.roll_deg: float = 0.0
        self.target: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.pitch_deg = math.degrees(math.asin(1.0 / math.sqrt(3.0)))
        self.yaw_deg = 45.0
        self.roll_deg = 0.0
        self.zoom = 3.2
        self.target = glm.vec3(0.0, 0.0, 0.0)

    def frame_bounds(self, bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
        """Center camera target on geometry bounding box and fit zoom."""
        center = (bounds_min + bounds_max) * 0.5
        extent = np.linalg.norm(bounds_max - bounds_min)
        self.target = glm.vec3(float(center[0]), float(center[1]), float(center[2]))
        if extent > 1e-4:
            self.zoom = max(0.8, min(float(extent) * 0.65, 25.0))

    def pan(self, dx: float, dy: float) -> None:
        yaw_rad = glm.radians(self.yaw_deg)
        right = glm.vec3(math.cos(yaw_rad), 0.0, -math.sin(yaw_rad))
        forward = glm.vec3(-math.sin(yaw_rad), 0.0, -math.cos(yaw_rad))

        scale = self.zoom * 0.003
        self.target -= right * (dx * scale)
        self.target += forward * (dy * scale)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw_deg = (self.yaw_deg - dx * 0.5) % 360.0
        self.pitch_deg = max(-85.0, min(85.0, self.pitch_deg - dy * 0.5))

    def rotate_yaw(self, delta_deg: float) -> None:
        self.yaw_deg = (self.yaw_deg + delta_deg) % 360.0

    def rotate_pitch(self, delta_deg: float) -> None:
        self.pitch_deg = max(-85.0, min(85.0, self.pitch_deg + delta_deg))

    def rotate_roll(self, delta_deg: float) -> None:
        self.roll_deg = (self.roll_deg + delta_deg) % 360.0

    def get_mvp_matrix(self) -> glm.mat4:
        aspect_safe = max(self.aspect, 0.01)
        pitch_rad = glm.radians(self.pitch_deg)
        yaw_rad = glm.radians(self.yaw_deg)

        if self.perspective:
            proj = glm.perspective(glm.radians(self.fov_deg), aspect_safe, 0.1, 200.0)
            dist = max(0.5, self.zoom * 2.6)
        else:
            half_h = self.zoom
            half_w = self.zoom * aspect_safe
            proj = glm.ortho(-half_w, half_w, -half_h, half_h, -80.0, 80.0)
            dist = 20.0

        eye_offset = glm.vec3(
            dist * math.cos(pitch_rad) * math.sin(yaw_rad),
            dist * math.sin(pitch_rad),
            dist * math.cos(pitch_rad) * math.cos(yaw_rad),
        )
        camera_pos = eye_offset + self.target

        forward = glm.normalize(self.target - camera_pos)
        world_up = glm.vec3(0.0, 1.0, 0.0)
        right = glm.normalize(glm.cross(forward, world_up))
        up = glm.normalize(glm.cross(right, forward))

        if self.roll_deg != 0.0:
            roll_mat = glm.rotate(glm.mat4(1.0), glm.radians(self.roll_deg), forward)
            up = glm.vec3(roll_mat * glm.vec4(up, 0.0))

        view = glm.lookAt(camera_pos, self.target, up)
        return proj * view
