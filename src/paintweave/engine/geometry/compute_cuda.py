"""PyTorch CUDA (cu134) accelerated geometry kernels with automatic CPU fallback."""

from __future__ import annotations

import numpy as np

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
    DEVICE_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU (NumPy/PyTorch)"
except Exception:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False
    HAS_CUDA = False
    DEVICE = None
    DEVICE_NAME = "CPU (NumPy)"


def get_compute_summary() -> str:
    """Return human-readable compute backend telemetry string for the studio status bar."""
    if HAS_CUDA and torch is not None:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"GPU: {DEVICE_NAME} ({vram_gb:.1f} GB VRAM | CUDA {torch.version.cuda})"
    if HAS_TORCH:
        return "Compute: PyTorch CPU + NumPy"
    return "Compute: NumPy Vectorized CPU"


def mountain_displace(
    p: np.ndarray,
    amplitude: float,
    frequency: float,
    phase: float,
    octaves: int = 1,
    roughness: float = 0.5,
    use_cuda: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute multi-octave procedural wave/fbm terrain displacement and height-based @Cd.
    Uses PyTorch on CUDA (RTX 4060 cu134) when available and enabled.
    """
    if HAS_TORCH and torch is not None and (use_cuda and HAS_CUDA):
        with torch.no_grad():
            pt = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32)).to(DEVICE)
            x = pt[:, 0]
            z = pt[:, 2]

            disp = torch.zeros_like(x)
            cur_amp = float(amplitude)
            cur_freq = float(frequency)

            for oct_idx in range(max(1, int(octaves))):
                shift = float(phase) + oct_idx * 1.732
                wave = torch.sin(x * cur_freq + shift) * torch.cos(z * cur_freq + shift)
                if oct_idx > 0:
                    # Cross-diagonal harmonic for organic ridges
                    wave = 0.7 * wave + 0.3 * torch.sin(
                        (x + z) * cur_freq * 0.7071 - shift
                    )
                disp = disp + cur_amp * wave
                cur_amp *= float(roughness)
                cur_freq *= 2.03

            pt_out = pt.clone()
            pt_out[:, 1] = pt_out[:, 1] + disp

            denom = max(abs(float(amplitude)), 1e-4)
            norm_h = torch.clamp((disp / denom) * 0.5 + 0.5, 0.0, 1.0)
            cd = torch.stack(
                [
                    0.25 + 0.55 * norm_h,
                    0.45 + 0.35 * norm_h,
                    0.75 - 0.30 * norm_h,
                ],
                dim=1,
            )
            return pt_out.cpu().numpy().astype(np.float32), cd.cpu().numpy().astype(
                np.float32
            )

    # Vectorized NumPy fallback
    p_out = p.copy()
    x = p_out[:, 0]
    z = p_out[:, 2]
    disp = np.zeros_like(x, dtype=np.float32)
    cur_amp = float(amplitude)
    cur_freq = float(frequency)

    for oct_idx in range(max(1, int(octaves))):
        shift = float(phase) + oct_idx * 1.732
        wave = np.sin(x * cur_freq + shift) * np.cos(z * cur_freq + shift)
        if oct_idx > 0:
            wave = 0.7 * wave + 0.3 * np.sin((x + z) * cur_freq * 0.7071 - shift)
        disp += cur_amp * wave
        cur_amp *= float(roughness)
        cur_freq *= 2.03

    p_out[:, 1] += disp
    norm_h = np.clip((disp / max(abs(float(amplitude)), 1e-4)) * 0.5 + 0.5, 0.0, 1.0)
    cd = np.column_stack(
        [
            0.25 + 0.55 * norm_h,
            0.45 + 0.35 * norm_h,
            0.75 - 0.30 * norm_h,
        ]
    ).astype(np.float32)
    return p_out, cd
