"""Dedicated Wrangle & NWShader Code Studio with live compiled GLSL Vertex/Fragment preview."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QShortcut,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import CodeString
from paintweave.engine.shader.payload import ShaderPayload


class PaintWeaveSyntaxHighlighter(QSyntaxHighlighter):
    """Dual GLSL / NWShader and Python / NumPy / PyTorch syntax highlighter."""

    def __init__(self, parent: QTextDocument) -> None:
        super().__init__(parent)
        self._rules: list[tuple[re.Pattern[str], QTextCharFormat]] = []

        # 1. Keywords (GLSL + Python)
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#c084fc"))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        keywords = (
            r"\b(shader_type|spatial|uniform|void|vertex|fragment|main|in|out|"
            r"return|if|else|for|while|break|continue|def|import|from|as|"
            r"float|int|bool|vec2|vec3|vec4|mat3|mat4|hint_range|source_color)\b"
        )
        self._rules.append((re.compile(keywords), kw_fmt))

        # 2. Built-in Attributes & Shader Globals
        builtin_fmt = QTextCharFormat()
        builtin_fmt.setForeground(QColor("#34d399"))
        builtin_fmt.setFontWeight(QFont.Weight.Bold)
        builtins = (
            r"\b(VERTEX|NORMAL|COLOR|UV|VIEW|TIME|ALBEDO|EMISSION|ROUGHNESS|METALLIC|"
            r"P|N|Cd|uv|pscale|np|torch|num_points|gl_Position|frag_color|"
            r"v_world_pos|v_normal|v_color|v_uv|final_rgb|snoise|fbm|pw_fbm|pw_snoise)\b"
        )
        self._rules.append((re.compile(builtins), builtin_fmt))

        # 3. Math & GLSL Functions
        fn_fmt = QTextCharFormat()
        fn_fmt.setForeground(QColor("#38bdf8"))
        funcs = (
            r"\b(sin|cos|tan|asin|acos|atan|sqrt|pow|exp|log|abs|min|max|clamp|"
            r"mix|smoothstep|step|length|distance|dot|cross|normalize|reflect|fract|floor|mod)\b"
        )
        self._rules.append((re.compile(funcs), fn_fmt))

        # 4. Numeric Literals
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("#fb923c"))
        self._rules.append((re.compile(r"\b\d+(?:\.\d+)?\b"), num_fmt))

        # 5. Comments (# and //)
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#64748b"))
        comment_fmt.setFontItalic(True)
        self._rules.append((re.compile(r"(?:#|//).*$"), comment_fmt))

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, fmt)


class CodeStudioPanel(QWidget):
    """
    Side-by-side Code Workbench for `SOP_Wrangle`, `VOP_Wrangle`, and `VOP_NWShader`:
      - Left pane: Editable Python/NumPy/PyTorch or GLSL/NWShader script
      - Right pane: Read-only live compiled GLSL Vertex & Fragment shader output or status diagnostics
    """

    code_applied = Signal(object)  # Emits modified NodeBase
    undo_snapshot_requested = Signal()  # Emits before mutating script code

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_node: NodeBase | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header Bar
        top_bar = QHBoxLayout()
        self.title_lbl = QLabel(
            "Select a Wrangle or NWShader node to edit code, or inspect any VOP's generated GLSL"
        )
        self.title_lbl.setStyleSheet("color: #38bdf8; font-weight: 700;")

        self.apply_btn = QPushButton("⚡ Apply & Cook (Ctrl+Enter)")
        self.apply_btn.setObjectName("PrimaryActionBtn")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_current_code)

        top_bar.addWidget(self.title_lbl)
        top_bar.addStretch(1)
        top_bar.addWidget(self.apply_btn)
        layout.addLayout(top_bar)

        # Split Editor & Compiled GLSL Output
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_box = QWidget()
        left_lay = QVBoxLayout(left_box)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(QLabel("✏️ Source Script (Python / NWShader / GLSL):"))
        self.source_editor = QPlainTextEdit()
        self.source_editor.setStyleSheet(
            "font-family: Consolas, 'JetBrains Mono', monospace; font-size: 12px; background-color: #0b0d13;"
        )
        self.source_editor.setPlaceholderText(
            "Select SOP_Wrangle, VOP_Wrangle, or VOP_NWShader to edit source code..."
        )
        self.source_editor.setReadOnly(True)
        left_lay.addWidget(self.source_editor, stretch=1)

        right_box = QWidget()
        right_lay = QVBoxLayout(right_box)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(QLabel("🔍 Compiled GLSL / Execution Output:"))
        self.glsl_preview = QPlainTextEdit()
        self.glsl_preview.setReadOnly(True)
        self.glsl_preview.setStyleSheet(
            "font-family: Consolas, 'JetBrains Mono', monospace; font-size: 11px; color: #a5f3fc; background-color: #090b10;"
        )
        right_lay.addWidget(self.glsl_preview, stretch=1)

        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        splitter.setSizes([520, 480])
        layout.addWidget(splitter, stretch=1)

        self._src_highlighter = PaintWeaveSyntaxHighlighter(
            self.source_editor.document()
        )
        self._glsl_highlighter = PaintWeaveSyntaxHighlighter(
            self.glsl_preview.document()
        )

        # Ctrl+Enter shortcut to apply code
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut.activated.connect(self._apply_current_code)

    def bind_node(self, node: NodeBase | None) -> None:
        self._current_node = node
        if node is None:
            self.title_lbl.setText(
                "No Node Selected — Click a Wrangle or VOP node on the canvas"
            )
            self.source_editor.clear()
            self.source_editor.setReadOnly(True)
            self.glsl_preview.clear()
            self.apply_btn.setEnabled(False)
            return

        has_code = "code" in node.params
        self.apply_btn.setEnabled(has_code)
        self.source_editor.setReadOnly(not has_code)

        if has_code:
            self.title_lbl.setText(f"Editing {node.name} [{node.registry_type}]")
            self.source_editor.setPlainText(str(node.params["code"]))
        else:
            self.title_lbl.setText(
                f"Inspecting {node.name} [{node.registry_type}] (Read-Only)"
            )
            self.source_editor.setPlainText(
                f"// Node '{node.name}' does not have an inline script parameter."
            )

        self.refresh_compiled_preview()

    def refresh_compiled_preview(self) -> None:
        node = self._current_node
        if node is None:
            self.glsl_preview.clear()
            return

        if node.error_message:
            self.glsl_preview.setPlainText(f"❌ ERROR:\n{node.error_message}")
            return

        if "Shader" in node.outputs and isinstance(
            node.outputs["Shader"].value, ShaderPayload
        ):
            payload: ShaderPayload = node.outputs["Shader"].value
            preview_txt = (
                "// ==========================================\n"
                f"// COMPILED FRAGMENT SHADER ({node.name})\n"
                "// ==========================================\n"
                f"{payload.build_fragment_shader()}\n\n"
                "// ==========================================\n"
                f"// COMPILED VERTEX SHADER ({node.name})\n"
                "// ==========================================\n"
                f"{payload.build_vertex_shader()}"
            )
            self.glsl_preview.setPlainText(preview_txt)
        elif "Geo" in node.outputs and node.outputs["Geo"].value is not None:
            geo = node.outputs["Geo"].value
            mats = geo.detail.get("materials", {})
            if mats:
                blocks = []
                for mid, payload in sorted(mats.items()):
                    blocks.append(
                        f"// --- Bound Material [{mid}]: {payload.name} ---\n"
                        f"{payload.build_fragment_shader()}"
                    )
                self.glsl_preview.setPlainText("\n\n".join(blocks))
            else:
                pt_attrs = ", ".join(geo.points.keys())
                self.glsl_preview.setPlainText(
                    f"✅ SOP Cooked in {node.last_cook_time_ms:.2f} ms\n"
                    f"Points: {geo.num_points:,}\n"
                    f"Triangles: {geo.num_primitives:,}\n"
                    f"Point Attributes: {pt_attrs}"
                )
        else:
            self.glsl_preview.setPlainText(
                f"Cook time: {node.last_cook_time_ms:.2f} ms"
            )

    def _apply_current_code(self) -> None:
        if self._current_node is not None and "code" in self._current_node.params:
            new_code = CodeString(self.source_editor.toPlainText())
            if self._current_node.params.get("code") != new_code:
                self.undo_snapshot_requested.emit()
            self._current_node.set_param("code", new_code)
            self.code_applied.emit(self._current_node)
            self.refresh_compiled_preview()
