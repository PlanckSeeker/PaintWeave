# 🎨 PaintWeave Studio (v2.0)

> **The Procedural Stylized Art Studio**  
> *Where Houdini procedural graph mechanics meet the chunky, hand-painted aesthetic of Arcane, Dofus, and Hearthstone.*

---

## 🌟 What is PaintWeave?

**PaintWeave Studio** is an artist-centric procedural 3D modeling and shading environment engineered specifically for stylized, painterly artwork.

While traditional 3D DCC tools focus on physically based photorealism (PBR) and complex engineering CAD workflows, PaintWeave treats 3D models as interactive digital paintings:
* **Chunky, Exaggerated Silhouettes**: Procedural stone chisel notches, chipped wooden edges, and beveled facets that evoke the physical miniature props of *Hearthstone* and *Dofus*.
* **Painterly Lighting**: Warm-to-cool lighting models replacing harsh photorealism with saturated vermilion terminator bands inspired by *Arcane* (Fortiche).
* **Hand-Painted Finishes**: Real-time gouache edge wear, directional hog-bristle impasto brush drag, and camera-space spherical MatCaps for gleaming gold trims and magical gems.
* **Graphic Ink Outlines**: Real-time screen-space Sobel edge detection over geometry normals and linear depth, framing silhouettes in comic ink.

---

## ⚡ What's New in v2.0

### 1. In-Place Zero-Reallocation VBO Streaming
PaintWeave v2.0 replaces per-frame GPU buffer reallocation with a smart topology-hash cache:
* **Fast-Path Streaming**: When topology remains stable during noise modulation, interactive parameter scrubbing, or Python/CUDA point wrangles, vertex and normal buffers stream directly into existing GPU VBOs via in-place memory writes (`.write()`).
* **Zero Frame Stutter**: Eliminates GPU heap reallocation and VAO teardowns, providing smooth real-time procedural sculpting.

### 2. 3-Pass Deferred / Hybrid Stylized Compositor
A dedicated multi-pass frame pipeline designed for graphic, non-photorealistic rendering:

```
[CPU / CUDA Scene Cook]
        │
        ▼ (In-Place VBO Streaming)
┌────────────────────────────────────────────────────────────────────────┐
│ PASS 1: Directional Shadow Map (2048x2048 Depth FBO)                   │
│  • Renders depth using lightweight hardware shadow-caster shader       │
│  • Outputs: tex_shadow_depth (16-tap PCF soft shadows)                 │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PASS 2: Scene Geometry + G-Buffer MRT Pass (Offscreen FBO)             │
│  • Attachment 0 (RGBA8): Shaded Color + Arcane Ramps + PCF Shadows     │
│  • Attachment 1 (RGBA16F): World/View Normal (RGB) + Linear Depth (A)  │
│  • Renders Grid, Multi-Material Batches, Template Ghost, Normal Hairs  │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PASS 3: Screen-Space Stylized Outline & Composite Pass (Screen FBO)    │
│  • Fullscreen quad sampling Color + Normal/Depth G-Buffer textures     │
│  • 3x3 Sobel edge detection over Normal & Depth gradients              │
│  • Composites ink outlines over shaded color + renders Corner Gizmo    │
└────────────────────────────────────────────────────────────────────────┘
```

### 3. The "Chunky Painterly Fantasy" Operator Suite
* [`SOP_AmbientGradient`](src/paintweave/nodes/sop/attributes.py): Bounding-box top-to-bottom value fade (`@vert_fade` $\in [0, 1]$), applying Blizzard-style value grounding and vertex tinting (`Multiply`, `Tint`, `Replace`) directly onto `@Cd`.
* [`SOP_Curvature`](src/paintweave/nodes/sop/attributes.py): Discrete Laplacian curvature estimator isolating convex ridges (`@highlight`) for chalky edge wear and concave crevices (`@cavity`) for pooled grime.
* [`SOP_StylizedDamage`](src/paintweave/nodes/sop/modifiers.py): Procedurally carves chunky triangular notches, chipped corners, and chisel dents into geometry edges with angular falloff and normal recalculation.
* [`VOP_ArcaneRamp`](src/paintweave/nodes/vop/lighting.py): 3-stop painterly lighting model featuring a hyper-saturated warm vermilion terminator band bridging sunlit peach light and deep teal ambient bounce.
* [`VOP_EdgeGlint`](src/paintweave/nodes/vop/lighting.py): Gouache/chalk edge wear and specular glints along facets and beveled edges using screen-space normal derivatives (`dFdx`/`dFdy`) and Fresnel response.
* [`VOP_BrushImpasto`](src/paintweave/nodes/vop/lighting.py): Overlays directional hog-bristle brush drag and oil-paint impasto canvas texture onto surface shading.
* [`VOP_MatCap`](src/paintweave/nodes/vop/lighting.py): View-space spherical reflections with zero matrix overhead, featuring built-in procedural presets for **Chunky Fantasy Gold**, **Arcane Gem**, **Toon Metal**, and **Chalky Clay**.

---

## 🏛️ Showcase Presets (`examples/`)

| File | Theme | Description |
| :--- | :--- | :--- |
| **`05_chunky_arcane_diorama.pw`** | **Chunky Arcane Diorama** | **(v2.0 Flagship)** Complete 4-tier fantasy altar featuring weathered stone pedestal steps, a chiseled rune monolith, beveled gold trims (`VOP_MatCap` Gold), and hovering arcane power crystals (`VOP_MatCap` Gem) bound across 4 distinct multi-material shaders. |
| **`01_arcanepunk_canyon_diorama.pw`** | **Arcanepunk Canyon** | Swept curved canyon cliffs, elevation ramp shading, and floating crystals. |
| **`02_multi_material_crystal_shrine.pw`** | **Crystal Shrine** | Multi-material scattering, normal-aligned point instancing, and Bayer dither terrain shading. |
| **`03_cuda_ripple_wrangle.pw`** | **CUDA Wave Simulation** | PyTorch GPU tensor wrangle calculating a 2D dynamic wave ripple on a subdivided grid. |
| **`04_swept_arcane_tendrils_uv.pw`** | **Swept Tendrils & UV** | 3D spiral curve sweep with procedural UV coordinate projection and UV checker validation. |

---

## 🚀 Quickstart & Desktop Workflow

### Prerequisites
* **Python 3.10+** (64-bit)
* **OpenGL 4.3 Core Profile** capable graphics card (NVIDIA / AMD / Intel)
* *(Optional)* **NVIDIA CUDA Toolkit & PyTorch** for hardware-accelerated tensor wrangles.

### Installation via `uv`
```bash
# Clone the repository
git clone https://github.com/ayoub-ben/paintweave.git
cd paintweave

# Sync dependencies into isolated environment
uv sync

# Launch PaintWeave Studio
uv run paintweave
```

### Headless CLI Runner (`paintweave-cli`)
PaintWeave includes a headless CLI runner for automation pipelines, batch network cooking, and Wavefront `.obj` export:

```bash
# Display compute backend summary (PyTorch CUDA / NumPy)
uv run paintweave-cli --info

# Cook the active Display SOP of a project and export to Wavefront OBJ
uv run paintweave-cli examples/05_chunky_arcane_diorama.pw -o diorama.obj
```

---

## 📚 Complete Operator Catalog (50+ Operators)

### SOP Context *(Surface Geometry Operators)*

| Category | Operator (`type_id`) | Description |
| :--- | :--- | :--- |
| **Primitives** | `SOP_Grid` | Parametric quad mesh grid with customizable row/col density. |
| | `SOP_Box` | 3D box/cube primitive with dimensional scales. |
| | `SOP_Cylinder` | Parametric cylinder and faceted polygonal prism with caps. |
| | `SOP_Sphere` | UV sphere primitive with adjustable latitude/longitude rings. |
| | `SOP_Curve` | Parametric 3D curves (Helix, Spiral, Arc, Sine Wave, Line). |
| | `SOP_File` | Wavefront `.obj` mesh file loader with auto normal computation. |
| **Modifiers** | `SOP_Transform` | Translates, rotates, and scales `@P` with optional point group masking. |
| | `SOP_Mountain` | Fractal Brownian Motion (FBM) displacement (PyTorch CUDA / CPU). |
| | `SOP_Subdivide` | Loop/Catmull-style midpoint mesh subdivision. |
| | `SOP_Smooth` | Discrete Laplacian mesh smoothing and feature softening. |
| | `SOP_Normal` | Computes or recalculates unit vertex normals (`@N`). |
| | `SOP_Sweep` | Sweeps cross-section profiles along 3D backbone curves. |
| | `SOP_Raycast` | Projects points along ray directions onto collision target surfaces. |
| | `SOP_StylizedDamage` | Procedurally chips triangular corner notches and chisel gouges along sharp edges. |
| **Attributes & Groups** | `SOP_AmbientGradient` | Top-to-bottom bounding box value wash (`@vert_fade`) and color blending. |
| | `SOP_Curvature` | Isolates convex ridges (`@highlight`) and concave crevices (`@cavity`). |
| | `SOP_Group` | Creates point and primitive groups via bounding box volumes. |
| | `SOP_Color` | Applies constant RGB colors or attribute color sweeps onto `@Cd`. |
| | `SOP_AttributeRandomize` | Randomizes 1D, 2D, or 3D point attributes (`pscale`, `Cd`, `N`). |
| | `SOP_UVProject` | Planar, Cylindrical, Spherical, and Box UV projection with transform gizmo. |
| | `SOP_Wrangle` | High-performance Python/NumPy/PyTorch vector scripting on `@P`, `@N`, and `@Cd`. |
| **Materials & Merging** | `SOP_Material` | Binds upstream VOP/NWShader chains to primitive groups via `@material_id`. |
| | `SOP_Merge` | Concatenates geometry streams while preserving multi-material bindings and 1D/3D attributes. |
| | `SOP_Scatter` | Distributes random Poisson/uniform points across surface primitives. |
| | `SOP_CopyToPoints` | Instantiates source geometry onto target points with automatic normal alignment. |

### VOP Context *(Visual & Programmable Shading Operators)*

| Category | Operator (`type_id`) | Description |
| :--- | :--- | :--- |
| **Stylized Lighting** | `VOP_ArcaneRamp` | 3-stop lighting with hyper-saturated warm vermilion terminator bridge. |
| | `VOP_EdgeGlint` | Hand-painted gouache bevel glints using screen-space normal derivatives. |
| | `VOP_BrushImpasto` | Directional hog-bristle brush drag and oil-paint impasto canvas texture. |
| | `VOP_MatCap` | View-space spherical reflections (Fantasy Gold, Arcane Gem, Toon Metal, Clay). |
| | `VOP_IsoLight` | Dynamic isometric directional sunlight with ambient floor control. |
| | `VOP_RimLight` | Grazing-angle Fresnel rim glow with customizable falloff power. |
| | `VOP_Fresnel` | View-angle Fresnel reflection and refraction factor. |
| | `VOP_CelToon` | Quantized stepped toon lighting bands. |
| **Procedural Patterns** | `VOP_SimplexNoise3D` | 3D spatial Simplex noise generating FBM procedural textures. |
| | `VOP_UVChecker` | High-visibility UV alignment checkerboard pattern. |
| | `VOP_Checkerboard` | 2D procedural checker pattern with dual color inputs. |
| | `VOP_ColorRamp` | Linear coordinate-to-color gradient ramp. |
| | `VOP_Noise` | 2D procedural Perlin/Value noise. |
| **Post & Composite** | `VOP_BayerDither` | Retro 4x4 Bayer matrix ordered cross-hatch dithering. |
| | `VOP_Blend` | Layer blending modes (Mix, Multiply, Screen, Overlay, Add). |
| | `VOP_Gamma` | Color space gamma correction and contrast curves. |
| **Programmable Code** | `VOP_NWShader` | Godot-style spatial shader transpiler with live uniform reflection in Inspector. |
| | `VOP_Wrangle` | Inline GLSL fragment snippet wrangle. |

---

## 🎨 Programmable Spatial Shaders (`VOP_NWShader`)

`VOP_NWShader` allows writing concise, high-level spatial shaders with automatic Inspector UI generation:

```glsl
shader_type spatial;

uniform float pulse_speed : hint_range(0.0, 10.0) = 2.5;
uniform vec3 crystal_glow : source_color = vec3(0.25, 0.88, 1.0);
uniform float rim_power : hint_range(0.5, 6.0) = 2.2;

void fragment() {
    float fresnel = pow(1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0), rim_power);
    float pulse = 0.65 + 0.35 * sin(TIME * pulse_speed + VERTEX.y * 3.0);
    ALBEDO = mix(vec3(0.08, 0.12, 0.20), crystal_glow * 0.45, fresnel);
    EMISSION = crystal_glow * (fresnel * pulse * 2.0);
}
```

---

## ⌨️ Studio Controls & Shortcuts

| Context | Shortcut / Action | Description |
| :--- | :--- | :--- |
| **Node Canvas** | `Tab` | Open operator quick-search palette popup |
| **Node Canvas** | `Double-Click Node` | Set node as active **Display (`DISP`)** operator |
| **Node Canvas** | `Ctrl + Z` / `Ctrl + Y` | Undo / Redo network modifications |
| **Node Canvas** | `Ctrl + C` / `Ctrl + V` | Copy / Paste selected operators |
| **Node Canvas** | `Ctrl + D` | Duplicate selected operators with input connections |
| **Node Canvas** | `Delete` | Remove selected nodes |
| **3D Viewport** | `LMB / MMB Drag` | Orbit camera around center of interest |
| **3D Viewport** | `RMB Drag` or `Shift + Drag` | Pan camera target |
| **3D Viewport** | `Mouse Wheel` | Zoom camera in / out |
| **3D Viewport** | `A` / `E` | Rotate camera yaw by $-45^\circ$ / $+45^\circ$ |
| **3D Viewport** | `F` or `H` | Frame camera to active Display SOP bounding box |
| **3D Viewport** | `W` | Toggle Wireframe display overlay |
| **3D Viewport** | `N` | Toggle Point Normal (`@N`) hair vector visualizer |
| **Code Studio** | `Ctrl + Enter` | Recompile and cook active Wrangle / NWShader code |
| **Global** | `F5` | Force re-cook active network from scratch |
| **Global** | `F12` | Render active viewport to 1080p offscreen PNG |

---

## 🧪 Automated Test Suite

PaintWeave is fully tested with unit and GUI integration coverage:

```bash
# Run the complete test suite
uv run pytest -v

# Run code linter
uv run ruff check src tests
```

---

## 📄 License

MIT License — see `pyproject.toml` for details.
