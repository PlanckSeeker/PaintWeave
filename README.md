# PaintWeave Studio 🔮

**PaintWeave** is a modular, Houdini-inspired desktop IDE and visual programming studio for procedural geometry (**SOPs**), multi-material primitive shading (**VOPs** & **NWShader**), and GPU/CUDA-accelerated field operations.

## ✨ Key Architectural Features

* **Columnar Copy-on-Write (CoW) Geometry Engine**: Geometry attributes across `Points`, `Vertices`, `Primitives`, and `Detail` domains backed by contiguous `float32`/`uint32` NumPy arrays and optional **PyTorch CUDA (`cu134`)** tensor acceleration.
* **Modular SOP + VOP Material Binding (`SOP_Material`)**: Assign independent VOP shader graphs or Godot-style `NWShader` programs directly to primitive groups (`@material_id`) and merge multi-material geometry streams seamlessly.
* **Studio-Grade Dockable UI (`PySide6` + `NodeGraphQt`)**:
  * Interactive `NodeGraphQt` node canvas with color-coded typed ports, curved/angled pipes, and Display/Template/Bypass flags.
  * **ModernGL 3D Viewport** with isometric/perspective camera, MSAAx4, wireframe overlay, orientation gizmo, and multi-material draw batching.
  * **Schema-Driven Parameter Inspector** (`ParamSpec`) with sliders, spinboxes, color pickers, file dialogs, and live uniform reflection.
  * **Tabbed Analytics & Code Workbench**: Virtualized `QAbstractTableModel` Geometry Spreadsheet, Wrangle/NWShader Code Studio with live GLSL preview, and `pyqtgraph` Cook Telemetry Profiler.

## 🚀 Quickstart

```bash
uv sync
uv run paintweave
```
