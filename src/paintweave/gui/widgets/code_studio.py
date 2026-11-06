"""Dedicated Wrangle & NWShader Code Studio with live compiled GLSL Vertex/Fragment preview."""

from __future__ import annotations

import re

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QShortcut,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
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


class LineNumberArea(QWidget):
    """Gutter widget displaying line numbers for ArtificerCodeEditor."""

    def __init__(self, editor: ArtificerCodeEditor) -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        self.editor.line_number_area_paint_event(event)


class ArtificerCodeEditor(QPlainTextEdit):
    """
    Precision code editor for Wrangle & NWShader scripts with:
      - Line numbers gutter painted in antique brass gold
      - Smart 4-space Tab and Shift+Tab indentation
      - Auto-indentation on Return
      - Inline error line indicator highlighting
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.line_number_area = LineNumberArea(self)
        self._error_line: int | None = None

        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self.textChanged.connect(self._on_text_changed)

        self._update_line_number_area_width(0)
        self._highlight_current_line()

    def _on_text_changed(self) -> None:
        if self._error_line is not None:
            self._error_line = None
            self._highlight_current_line()
            self.line_number_area.update()

    def highlight_error_line(self, line_num: int | None) -> None:
        self._error_line = line_num
        self._highlight_current_line()
        self.line_number_area.update()
        if line_num is not None:
            doc = self.document()
            block = doc.findBlockByLineNumber(max(0, line_num - 1))
            if block.isValid():
                cursor = self.textCursor()
                cursor.setPosition(block.position())
                self.setTextCursor(cursor)
                self.centerCursor()

    def line_number_area_width(self) -> int:
        digits = max(1, len(str(max(1, self.blockCount()))))
        space = 14 + self.fontMetrics().horizontalAdvance("9") * digits
        return max(32, space)

    def _update_line_number_area_width(self, _new_block_count: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect: QRect, dy: int) -> None:
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width(0)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def _highlight_current_line(self) -> None:
        extra_selections: list[QTextEdit.ExtraSelection] = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            line_color = QColor("#1e293b")
            line_color.setAlpha(60)
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)

        if self._error_line is not None and self._error_line > 0:
            err_selection = QTextEdit.ExtraSelection()
            err_color = QColor("#7f1d1d")
            err_color.setAlpha(120)
            err_selection.format.setBackground(err_color)
            err_selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            doc = self.document()
            block = doc.findBlockByLineNumber(self._error_line - 1)
            if block.isValid():
                err_cursor = self.textCursor()
                err_cursor.setPosition(block.position())
                err_selection.cursor = err_cursor
                extra_selections.append(err_selection)

        self.setExtraSelections(extra_selections)

    def line_number_area_paint_event(self, event) -> None:
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#090d14"))

        # Hairline brass separator rule
        painter.setPen(QColor(198, 146, 60, 60))
        painter.drawLine(event.rect().topRight(), event.rect().bottomRight())

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                is_error = self._error_line == (block_number + 1)
                if is_error:
                    painter.fillRect(2, top + 2, 4, bottom - top - 4, QColor("#ef4444"))
                    painter.setPen(QColor("#fca5a5"))
                else:
                    painter.setPen(QColor("#c6923c") if not self.isReadOnly() else QColor("#64748b"))

                painter.setFont(self.font())
                painter.drawText(
                    0,
                    top,
                    self.line_number_area.width() - 8,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    number,
                )

            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Tab:
            self.insertPlainText("    ")
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backtab:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.StartOfLine, cursor.MoveMode.MoveAnchor)
            line_text = cursor.block().text()
            if line_text.startswith("    "):
                cursor.movePosition(cursor.MoveOperation.Right, cursor.MoveMode.KeepAnchor, 4)
                cursor.removeSelectedText()
            elif line_text.startswith("\t"):
                cursor.movePosition(cursor.MoveOperation.Right, cursor.MoveMode.KeepAnchor, 1)
                cursor.removeSelectedText()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            cursor = self.textCursor()
            line_text = cursor.block().text()
            leading_spaces = len(line_text) - len(line_text.lstrip(" "))
            super().keyPressEvent(event)
            if leading_spaces > 0:
                self.insertPlainText(" " * leading_spaces)
            event.accept()
            return
        super().keyPressEvent(event)


class CodeStudioPanel(QWidget):
    """
    Side-by-side Code Workbench for `SOP_Wrangle`, `VOP_Wrangle`, and `VOP_NWShader`:
      - Left pane: Editable Python/NumPy/PyTorch or GLSL/NWShader script with line numbers & error pinning
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
        self.title_lbl.setStyleSheet("color: #e5b869; font-weight: 700; font-size: 12px;")

        self.apply_btn = QPushButton("⚡ Attune & Cook (Ctrl+Enter)")
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
        lbl_src = QLabel("📜 Source Script (Python Wrangle / NWShader / GLSL):")
        lbl_src.setStyleSheet("color: #e5b869; font-weight: 600;")
        left_lay.addWidget(lbl_src)
        self.source_editor = ArtificerCodeEditor()
        self.source_editor.setStyleSheet(
            "font-family: Consolas, 'JetBrains Mono', monospace; font-size: 12px; background-color: #080b12; border: 1px solid rgba(198, 146, 60, 0.35);"
        )
        self.source_editor.setPlaceholderText(
            "Select SOP_Wrangle, VOP_Wrangle, or VOP_NWShader to edit source code..."
        )
        self.source_editor.setReadOnly(True)
        left_lay.addWidget(self.source_editor, stretch=1)

        right_box = QWidget()
        right_lay = QVBoxLayout(right_box)
        right_lay.setContentsMargins(0, 0, 0, 0)
        lbl_out = QLabel("🔍 Compiled Ledger / Output:")
        lbl_out.setStyleSheet("color: #38bdf8; font-weight: 600;")
        right_lay.addWidget(lbl_out)
        self.glsl_preview = ArtificerCodeEditor()
        self.glsl_preview.setReadOnly(True)
        self.glsl_preview.setStyleSheet(
            "font-family: Consolas, 'JetBrains Mono', monospace; font-size: 11px; color: #a5f3fc; background-color: #06080e; border: 1px solid rgba(56, 189, 248, 0.25);"
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
            self.source_editor.highlight_error_line(None)
            return

        if node.error_message:
            self.glsl_preview.setPlainText(f"❌ ERROR:\n{node.error_message}")
            m = re.search(
                r"(?:line\s*(\d+)|\b0\((\d+)\)\s*:|\b(\d+)\s*:\s*(?:error|warning))",
                node.error_message,
                re.IGNORECASE,
            )
            if m:
                line_str = m.group(1) or m.group(2) or m.group(3)
                if line_str:
                    self.source_editor.highlight_error_line(int(line_str))
            return

        self.source_editor.highlight_error_line(None)

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
