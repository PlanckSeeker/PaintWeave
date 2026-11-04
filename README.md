# PaintWeave Studio

**PaintWeave** is a modular, Houdini-inspired desktop IDE and visual programming studio for procedural geometry (**SOPs**), multi-material primitive shading (**VOPs** & **NWShader**), and GPU/CUDA-accelerated field operations[cite: 1, 2].

---

## Key Architectural Features

* **Columnar Copy-on-Write (CoW) Geometry Engine (`Geometry`)**:
  * Attributes across `Points`, `Vertices`, `Primitives`, and `Detail` domains backed by contiguous `float32`/`uint32`/`int32` NumPy arrays and optional **PyTorch CUDA** tensor acceleration[cite: 1, 2].
  * Zero-copy attribute sharing via `Geometry.clone()` across downstream SOP nodes until a specific attribute column is replaced[cite: 2].
* **Multi-Material Primitive Shading (`SOP_Material` + `SOP_Merge`)**:
  * Bind distinct VOP shader chains or full `NWShader` programs to specific point/primitive groups (`@group_<name>` $\rightarrow$ `@material_id`)[cite: 1, 2].
  * `SOP_Merge` automatically remaps `@material_id` buffers and merges `detail["materials"]` tables so multiple materials render simultaneously in a single mesh[cite: 1, 2].
* **Studio-Grade Dockable Desktop IDE (`PySide6` + `NodeGraphQt` + `ModernGL`)**:
  * **NodeGraphQt Canvas**: Color-coded typed ports (`Geometry`, `Shader`, `Float`, `Int`, `Vec3`), curved pipes, `Tab` quick-search, 64-step Undo/Redo, Copy/Paste/Duplicate, and Houdini **Display (`DISP`)**, **Template (`TMPL`)**, and **Bypass (`BYP`)** flags[cite: 1, 2].
  * **ModernGL 3D Viewport**: OpenGL 4.3 Core context with MSAA×4, hybrid Orthographic (Isometric) / Perspective camera, pink ghost Template wireframe overlay, `@N` normal hair visualizer, orientation gizmo, and `F12` 1080p offscreen PNG rendering[cite: 1, 2].
  * **Schema-Driven Parameter Inspector (`ParamSpec`)**: Auto-generated sliders, spinboxes, enum dropdowns, RGB color pickers, file dialogs, and live `NWShader` uniform reflection[cite: 1, 2].
  * **Tabbed Workbench**: Virtualized zero-copy `QAbstractTableModel` Geometry & Material Spreadsheet, dual-pane Wrangle/NWShader Code Studio with live compiled GLSL preview, and `pyqtgraph` Cook Profiler[cite: 1, 2].

---

## Quickstart & CLI Usage

### 1. Launch the Desktop Studio
```bash
uv sync
uv run paintweave
```

### 2. Headless CLI Evaluation & Wavefront `.obj` Export
PaintWeave includes a zero-GUI command-line runner (`paintweave-cli`) for batch cooking `.pw` / `.nw` graphs and exporting meshes[cite: 1, 2]:

```bash
# Inspect compute backend (CUDA / PyTorch / NumPy)
uv run paintweave-cli --info

# Cook the active Display SOP of a project and export to Wavefront OBJ
uv run paintweave-cli examples/02_multi_material_crystal_shrine.pw -o shrine.obj
```

---

## Operator Catalog (45 Built-In Nodes)

| Context | Category | Registered Operators (`type_id`) |
| :--- | :--- | :--- |
| **SOP** *(Geometry)* | **Primitives** | `SOP_Grid`, `SOP_Box`, `SOP_Cylinder`, `SOP_Sphere`, `SOP_Curve` *(Helix/Arc/Wave/Line)*, `SOP_File` *(Wavefront `.obj`)*[cite: 2] |
| **SOP** *(Geometry)* | **Modifiers** | `SOP_Transform`, `SOP_Mountain` *(CUDA/CPU FBM)*, `SOP_Extrude` *(PolyExtrude)*, `SOP_Subdivide`, `SOP_Facet`, `SOP_Smooth` *(Laplacian)*, `SOP_Normal`, `SOP_Sweep` *(Curve Tube)*, `SOP_Raycast` *(Project to Surface)*[cite: 2] |
| **SOP** *(Geometry)* | **Attributes & Groups** | `SOP_Group` *(Bounding Box)*, `SOP_Color`, `SOP_AttributeRandomize`, `SOP_UVProject` *(Planar/Cyl/Sph/Triplanar)*, `SOP_Wrangle` *(Python/NumPy/PyTorch)*[cite: 2] |
| **SOP** *(Geometry)* | **Instancing & Topology** | `SOP_Scatter`, `SOP_CopyToPoints` *(with `@N` alignment & `@pscale`)*, `SOP_Merge` *(Multi-Material)*[cite: 2] |
| **SOP** *(Geometry)* | **Materials & Shading** | `SOP_Material` *(Group-aware `ShaderPayload` binder)*[cite: 2] |
| **VOP** *(Shader)* | **Lighting** | `VOP_IsoLight`, `VOP_RimLight`, `VOP_Fresnel`, `VOP_CelToon`[cite: 2] |
| **VOP** *(Shader)* | **Patterns** | `VOP_Checkerboard`, `VOP_Noise`, `VOP_SimplexNoise3D` *(3D FBM)*, `VOP_UVChecker`, `VOP_ColorRamp`[cite: 2] |
| **VOP** *(Shader)* | **Post & Stylized** | `VOP_BayerDither` *(4×4 Ordered Dither)*, `VOP_Blend`, `VOP_Gamma`[cite: 2] |
| **VOP** *(Shader)* | **Programmable Shaders** | `VOP_Wrangle` *(Inline GLSL)*, `VOP_NWShader` *(Spatial Vertex + Fragment GLSL)*[cite: 2] |
| **MATH** *(Utility)* | **Scalars, Vectors & Time** | `Math_Float`, `Math_Add`, `Math_Multiply`, `Math_Vec3`, `Math_FloatToVec3`, `Math_Time` *(`$T`)*, `Math_ArcaneOscillator`[cite: 2] |

---

## End-to-End Walkthroughs

### 1. Multi-Material Crystal Shrine Diorama (`SOP` + `VOP` + `NWShader`)
This network builds the default multi-material diorama (`_build_default_showcase_network`), combining a dithered canyon terrain with instanced bioluminescent crystal monoliths[cite: 2]:

```text
Stream 1 (Canyon Terrain):
  [SOP_Grid] ──(Geo)──► [SOP_Mountain (CUDA)] ─────────────────────(Geo)──► [SOP_Material: mat_terrain_dither]
                               │                                                        ▲ (Shader)
                               │                     [VOP_IsoLight] ──► [VOP_RimLight] ──► [VOP_BayerDither]
                               │                                                                │
                               ▼ (Geo)                                                          ▼ (Geo1)
Stream 2 (Crystal Monoliths): [SOP_Scatter] ──► [SOP_AttributeRandomize (@pscale)]         [SOP_Merge] (DISP)
                                                         │                                      ▲ (Geo2)
                              [SOP_Cylinder] ──(Source)──┴──(TargetPoints)──► [SOP_CopyToPoints] │
                                                                                     │ (Geo)    │
                                               [VOP_NWShader] ──(Shader)──► [SOP_Material: mat_crystal_bio]
```

1. **Terrain Generation & Shading**:
   * Spawn `SOP_Grid` (`size_x=8.0, size_z=8.0, rows=48, cols=48`) $\rightarrow$ `SOP_Mountain` (`frequency=1.6, octaves=3, roughness=0.45, Amplitude=1.15`)[cite: 2].
   * Chain `VOP_IsoLight` $\rightarrow$ `VOP_RimLight` $\rightarrow$ `VOP_BayerDither` and wire both the terrain and shader into `SOP_Material` (`material_label="mat_terrain_dither"`)[cite: 2].
2. **Normal-Aligned Crystal Instancing**:
   * Branch `SOP_Mountain.Geo` into `SOP_Scatter` (`num_points=28`) $\rightarrow$ `SOP_AttributeRandomize` (`attribute="pscale", min_value=0.35, max_value=0.95`)[cite: 2].
   * Wire a 6-sided `SOP_Cylinder` into `SOP_CopyToPoints.Source` and the scattered points into `SOP_CopyToPoints.TargetPoints` (`align_to_normal=True` rotates each crystal to match the terrain slope `@N`)[cite: 2].
   * Bind a custom `VOP_NWShader` to the crystals via a second `SOP_Material` (`material_label="mat_crystal_bio"`)[cite: 2].
3. **Multi-Material Merge**:
   * Connect both `SOP_Material` outputs to `SOP_Merge` (`Geo1`, `Geo2`) and double-click `SOP_Merge` to set its **Display Flag (`DISP`)**[cite: 2].
   * Open the **Geometry & Material Spreadsheet** tab and switch the domain dropdown to **`Materials`** to inspect the assigned triangle counts and uniforms per `material_id`[cite: 2].

---

### 2. Python / PyTorch Geometry Wrangle (`SOP_Wrangle`)
Select any `SOP_Wrangle` node and open the **Wrangle & NWShader Studio** tab (`Ctrl+Enter` to apply and recook)[cite: 2]. All point attributes (`P`, `N`, `Cd`, `uv`, `pscale`, or custom attributes) are injected into the sandbox scope as `(num_points, k)` NumPy arrays alongside `np`, `torch`, and `num_points`[cite: 2]:

```python
# Available in scope: P, N, Cd, uv, np, torch, num_points
# Radial ripple wave with height-based color ramp
r = np.sqrt(P[:, 0]**2 + P[:, 2]**2)
P[:, 1] += np.sin(r * 3.5) * np.exp(-0.2 * r) * 0.65

# Compute normalized elevation and write to @Cd
h = (P[:, 1] - P[:, 1].min()) / (P[:, 1].max() - P[:, 1].min() + 1e-6)
Cd[:, 0] = 0.12 + 0.25 * h
Cd[:, 1] = 0.45 + 0.45 * h
Cd[:, 2] = 0.85 - 0.30 * h
```
Any new or modified array matching `(num_points, k)` in scope is automatically written back to the output `Geometry` and vertex normals (`@N`) are recomputed[cite: 2].

---

### 3. Programmable Spatial Shaders (`VOP_NWShader`)
`VOP_NWShader` transpiles Godot-style `shader_type spatial;` scripts into GLSL 330 Core vertex and fragment programs[cite: 2]. Built-in noise functions (`hash(vec3)`, `snoise(vec3)`, `fbm(vec3)`) and stage globals (`VERTEX`, `NORMAL`, `COLOR`, `UV`, `VIEW`, `TIME`, `ALBEDO`, `EMISSION`) are available out of the box, and `uniform` declarations with `: hint_range(min, max)` or `: source_color` are automatically reflected into sliders and color pickers in the **Node Parameter Inspector**[cite: 2]:

```glsl
shader_type spatial;

uniform float bend : hint_range(-2.0, 2.0) = 0.35;
uniform float pulse_speed : hint_range(0.0, 10.0) = 2.5;
uniform vec3 bio_color : source_color = vec3(0.14, 0.88, 0.98);

void vertex() {
    float curve = pow(max(VERTEX.y, 0.0) * 0.35, 2.0);
    VERTEX.x += curve * bend;
}

void fragment() {
    float fresnel = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), 2.6);
    float pulse = 0.55 + 0.45 * sin(TIME * pulse_speed + VERTEX.y * 4.0);
    float noise_vein = smoothstep(0.45, 0.85, fbm(VERTEX * 2.5));
    ALBEDO = mix(vec3(0.08, 0.11, 0.18), bio_color * 0.45, noise_vein);
    EMISSION = bio_color * (fresnel * pulse * 1.8 + noise_vein * 0.35);
}
```

---

## Studio Shortcuts & Controls

| Context | Shortcut / Action | Description |
| :--- | :--- | :--- |
| **Node Canvas** | `Tab` | Open operator quick-search popup[cite: 2] |
| **Node Canvas** | `Double-Click Node` | Set node as active **Display (`DISP`)** operator[cite: 2] |
| **Node Canvas** | `Ctrl+Z` / `Ctrl+Y` | Undo / Redo graph modifications[cite: 2] |
| **Node Canvas** | `Ctrl+C` / `Ctrl+V` / `Ctrl+D` | Copy / Paste / Duplicate selected nodes & internal links[cite: 2] |
| **3D Viewport** | `LMB / MMB Drag` | Orbit camera around target[cite: 2] |
| **3D Viewport** | `RMB Drag` or `Shift + Drag` | Pan camera target[cite: 2] |
| **3D Viewport** | `Mouse Wheel` | Zoom camera in / out[cite: 2] |
| **3D Viewport** | `A` / `E` | Rotate camera yaw by $-45^\circ$ / $+45^\circ$[cite: 2] |
| **3D Viewport** | `F` or `H` | Frame active Display SOP geometry bounds[cite: 2] |
| **3D Viewport** | `W` / `N` | Toggle Wireframe / Point Normal (`@N`) hair vectors[cite: 2] |
| **Code Studio** | `Ctrl+Enter` | Apply Wrangle / NWShader script and recook[cite: 2] |
| **Global** | `F5` / `F12` | Force recook network / Render 1920×1080 PNG frame[cite: 2] |

---

## License

MIT License — see `pyproject.toml` for details[cite: 1].
