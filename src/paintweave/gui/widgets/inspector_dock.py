"""Schema-driven Parameter Inspector with Houdini Display/Template/Bypass flag controls."""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import CodeString, NodeContext, ParamSpec


class NodePropertyInspector(QWidget):
    """Dynamic property editor auto-generated from `NodeBase.all_param_specs()`."""

    parameter_changed = Signal(object)  # Emits modified NodeBase
    undo_snapshot_requested = Signal()  # Emits before mutating parameter/flag state
    display_flag_toggled = Signal(str)  # Emits node_id
    template_flag_toggled = Signal(str, bool)  # Emits (node_id, enabled)
    bypass_flag_toggled = Signal(str, bool)  # Emits (node_id, bypassed)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_node: NodeBase | None = None
        self._last_undo_key: tuple[str, str] | None = None
        self._last_undo_time: float = 0.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header Block
        self.header_lbl = QLabel("No Node Selected")
        self.header_lbl.setStyleSheet(
            "color: #38bdf8; font-weight: 700; font-size: 13px;"
        )
        layout.addWidget(self.header_lbl)

        self.meta_lbl = QLabel(
            "Click any SOP, VOP, or Math node on the canvas to edit parameters."
        )
        self.meta_lbl.setWordWrap(True)
        self.meta_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.meta_lbl)

        # Houdini Node Flags Bar (Display / Template / Bypass)
        self.flags_bar = QWidget()
        flags_layout = QHBoxLayout(self.flags_bar)
        flags_layout.setContentsMargins(0, 2, 0, 2)
        flags_layout.setSpacing(6)

        self.btn_display = QPushButton("🔵 Display")
        self.btn_display.setObjectName("DisplayFlagBtn")
        self.btn_display.setCheckable(True)
        self.btn_display.setToolTip("Set as active Display / Output node in viewport")
        self.btn_display.clicked.connect(self._on_display_clicked)

        self.btn_template = QPushButton("🟣 Template")
        self.btn_template.setObjectName("TemplateFlagBtn")
        self.btn_template.setCheckable(True)
        self.btn_template.setToolTip(
            "Overlay this SOP as a pink ghost reference wireframe in the viewport"
        )
        self.btn_template.clicked.connect(self._on_template_clicked)

        self.btn_bypass = QPushButton("🟡 Bypass")
        self.btn_bypass.setObjectName("BypassFlagBtn")
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setToolTip(
            "Bypass this operator and pass input directly to output"
        )
        self.btn_bypass.clicked.connect(self._on_bypass_clicked)

        flags_layout.addWidget(self.btn_display)
        flags_layout.addWidget(self.btn_template)
        flags_layout.addWidget(self.btn_bypass)
        self.flags_bar.setVisible(False)
        layout.addWidget(self.flags_bar)

        # Scrollable Form Container
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.form_container = QWidget()
        self.form_layout = QFormLayout(self.form_container)
        self.form_layout.setContentsMargins(8, 8, 8, 8)
        self.form_layout.setSpacing(8)
        self.scroll.setWidget(self.form_container)
        layout.addWidget(self.scroll, stretch=1)

    def inspect_node(self, node: NodeBase | None) -> None:
        self._current_node = node
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)

        if node is None:
            self.header_lbl.setText("No Node Selected")
            self.meta_lbl.setText(
                "Click any SOP, VOP, or Math node on the canvas to edit parameters."
            )
            self.flags_bar.setVisible(False)
            return

        self.header_lbl.setText(f"{node.name}  [{node.registry_type}]")
        doc_str = (node.__doc__ or "").strip()
        cook_str = (
            f"Context: {node.context.value}  |  Cook: {node.last_cook_time_ms:.2f} ms"
        )
        self.meta_lbl.setText(f"{cook_str}\n{doc_str}" if doc_str else cook_str)

        # Configure Houdini Flag buttons
        self.flags_bar.setVisible(True)
        self.btn_display.setEnabled(node.context in (NodeContext.SOP, NodeContext.VOP))
        self.btn_display.setChecked(node.is_display_node)

        self.btn_template.setEnabled(node.context == NodeContext.SOP)
        self.btn_template.setChecked(node.is_template_node)

        self.btn_bypass.setChecked(node.is_bypassed)

        # 1. Render all schema-driven ParamSpecs
        for spec in node.all_param_specs():
            editor = self._create_editor(node, spec)
            base_lbl = spec.label or spec.name
            label_txt = f"{base_lbl} ({spec.unit})" if spec.unit else base_lbl
            lbl = QLabel(label_txt)
            if spec.description:
                lbl.setToolTip(spec.description)
            self.form_layout.addRow(lbl, editor)

        # 2. Render unconnected scalar input pins
        for pin_name, pin in node.inputs.items():
            if not pin.is_connected and isinstance(pin.value, (int, float)):
                dspin = QDoubleSpinBox()
                dspin.setRange(-1000.0, 1000.0)
                dspin.setSingleStep(0.1)
                dspin.setDecimals(3)
                dspin.setValue(float(pin.value))
                dspin.valueChanged.connect(
                    lambda v, p=pin_name: self._update_param(p, float(v))
                )
                in_lbl = QLabel(f"[In] {pin_name}")
                in_lbl.setStyleSheet("color: #38bdf8;")
                self.form_layout.addRow(in_lbl, dspin)

    def _create_editor(self, node: NodeBase, spec: ParamSpec) -> QWidget:
        val = node.params.get(spec.name, spec.default)

        if spec.param_type in ("choice", "enum"):
            cb = QComboBox()
            cb.addItems(spec.choices or spec.options)
            cb.setCurrentText(str(val))
            cb.currentTextChanged.connect(
                lambda v, k=spec.name: self._update_param(k, v)
            )
            return cb

        if spec.param_type == "bool":
            chk = QCheckBox()
            chk.setChecked(bool(val))
            chk.toggled.connect(lambda v, k=spec.name: self._update_param(k, bool(v)))
            return chk

        if spec.param_type == "int":
            container = QWidget()
            hbox = QHBoxLayout(container)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(6)

            min_i = int(spec.min_val if spec.min_val is not None else 0)
            max_i = int(spec.max_val if spec.max_val is not None else 1024)

            sp = QSpinBox()
            sp.setRange(min_i, max_i)
            sp.setSingleStep(int(spec.step or 1))
            sp.setValue(int(val))

            if max_i - min_i <= 512:
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(min_i, max_i)
                slider.setValue(int(val))
                slider.valueChanged.connect(sp.setValue)
                sp.valueChanged.connect(slider.setValue)
                hbox.addWidget(slider, stretch=1)

            sp.valueChanged.connect(
                lambda v, k=spec.name: self._update_param(k, int(v))
            )
            hbox.addWidget(sp)
            return container

        if spec.param_type == "float":
            container = QWidget()
            hbox = QHBoxLayout(container)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(6)

            min_f = float(spec.min_val if spec.min_val is not None else -1000.0)
            max_f = float(spec.max_val if spec.max_val is not None else 1000.0)

            dsp = QDoubleSpinBox()
            dsp.setDecimals(3)
            dsp.setRange(min_f, max_f)
            dsp.setSingleStep(float(spec.step or 0.1))
            dsp.setValue(float(val))
            dsp.valueChanged.connect(
                lambda v, k=spec.name: self._update_param(k, float(v))
            )
            hbox.addWidget(dsp, stretch=1)
            return container

        if spec.param_type in ("vec2", "vec3", "vec4"):
            dims = int(spec.param_type[-1])
            container = QWidget()
            hbox = QHBoxLayout(container)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(4)

            tup = (
                tuple(float(x) for x in val)
                if (isinstance(val, (tuple, list)) and len(val) == dims)
                else tuple(0.0 for _ in range(dims))
            )
            spinboxes: list[QDoubleSpinBox] = []
            for idx in range(dims):
                dsp = QDoubleSpinBox()
                dsp.setDecimals(3)
                dsp.setRange(-1000.0, 1000.0)
                dsp.setSingleStep(0.1)
                dsp.setValue(tup[idx])
                spinboxes.append(dsp)
                hbox.addWidget(dsp, stretch=1)

            def _on_vec_changed(
                _v: float,
                k: str = spec.name,
                sbs: list[QDoubleSpinBox] = spinboxes,
            ) -> None:
                self._update_param(k, tuple(float(sb.value()) for sb in sbs))

            for sb in spinboxes:
                sb.valueChanged.connect(_on_vec_changed)
            return container

        if spec.param_type == "color":
            container = QWidget()
            hbox = QHBoxLayout(container)
            hbox.setContentsMargins(0, 0, 0, 0)

            tup = (
                val
                if (isinstance(val, (tuple, list)) and len(val) == 3)
                else (0.5, 0.5, 0.5)
            )
            current_color = QColor(
                int(max(0.0, min(1.0, float(tup[0]))) * 255),
                int(max(0.0, min(1.0, float(tup[1]))) * 255),
                int(max(0.0, min(1.0, float(tup[2]))) * 255),
            )

            color_btn = QPushButton(current_color.name())
            color_btn.setStyleSheet(
                f"background-color: {current_color.name()}; color: #0e1015; font-weight: bold; border: 1px solid #cbd5e1;"
            )
            color_btn.setFixedHeight(26)

            def pick_color(
                _checked: bool = False, btn: QPushButton = color_btn, k: str = spec.name
            ) -> None:
                cur_val = node.params.get(k, (0.5, 0.5, 0.5))
                qcol = QColor(
                    int(cur_val[0] * 255), int(cur_val[1] * 255), int(cur_val[2] * 255)
                )
                new_color = QColorDialog.getColor(qcol, self, "Select Color")
                if new_color.isValid():
                    new_tup = (new_color.redF(), new_color.greenF(), new_color.blueF())
                    btn.setText(new_color.name())
                    btn.setStyleSheet(
                        f"background-color: {new_color.name()}; color: #0e1015; font-weight: bold; border: 1px solid #cbd5e1;"
                    )
                    self._update_param(k, new_tup)

            color_btn.clicked.connect(pick_color)
            hbox.addWidget(color_btn)
            return container

        if spec.param_type == "filepath":
            container = QWidget()
            hbox = QHBoxLayout(container)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(4)

            le = QLineEdit(str(val))
            btn = QPushButton("...")
            btn.setFixedWidth(32)
            hbox.addWidget(le, stretch=1)
            hbox.addWidget(btn)

            le.editingFinished.connect(
                lambda ed=le, k=spec.name: self._update_param(k, ed.text())
            )

            def pick_file(
                _checked: bool = False, ed: QLineEdit = le, k: str = spec.name
            ) -> None:
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Wavefront OBJ Asset",
                    "",
                    "Wavefront OBJ (*.obj);;All Files (*)",
                )
                if path:
                    ed.setText(path)
                    self._update_param(k, path)

            btn.clicked.connect(pick_file)
            return container

        if spec.param_type == "code":
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(4)

            text_edit = QPlainTextEdit(str(val))
            text_edit.setMinimumHeight(135)
            text_edit.setStyleSheet(
                "font-family: Consolas, 'JetBrains Mono', monospace; font-size: 11px;"
            )

            apply_btn = QPushButton("⚡ Apply Script")
            apply_btn.setObjectName("PrimaryActionBtn")
            apply_btn.clicked.connect(
                lambda _=False, k=spec.name, te=text_edit: self._update_param(
                    k, CodeString(te.toPlainText())
                )
            )

            vbox.addWidget(text_edit)
            vbox.addWidget(apply_btn)
            return container

        le = QLineEdit(str(val))
        le.editingFinished.connect(
            lambda ed=le, k=spec.name: self._update_param(k, ed.text())
        )
        return le

    def _update_param(self, key: str, value: Any) -> None:
        if self._current_node is None:
            return
        cur_val = (
            self._current_node.params.get(key)
            if key in self._current_node.params
            else (
                self._current_node.inputs[key].value
                if key in self._current_node.inputs
                else None
            )
        )
        if cur_val != value:
            now = time.monotonic()
            if (
                (self._current_node.id, key) != self._last_undo_key
                or (now - self._last_undo_time) > 0.6
            ):
                self.undo_snapshot_requested.emit()
                self._last_undo_key = (self._current_node.id, key)
            self._last_undo_time = now

        prev_keys = set(self._current_node.params.keys())
        self._current_node.set_param(key, value)
        self.parameter_changed.emit(self._current_node)
        # If compiling NWShader added new reflected uniform params, refresh inspector rows
        if set(self._current_node.params.keys()) != prev_keys:
            self.inspect_node(self._current_node)

    def update_cook_badge(self, node: NodeBase | None = None) -> None:
        """Refresh the header cook time readout without destroying active editor widgets."""
        target = node or self._current_node
        if target is None:
            return
        doc_str = (target.__doc__ or "").strip()
        cook_str = (
            f"Context: {target.context.value}  |  Cook: {target.last_cook_time_ms:.2f} ms"
        )
        self.meta_lbl.setText(f"{cook_str}\n{doc_str}" if doc_str else cook_str)
        self.btn_display.setChecked(target.is_display_node)
        self.btn_template.setChecked(target.is_template_node)
        self.btn_bypass.setChecked(target.is_bypassed)

    def _on_display_clicked(self, checked: bool) -> None:
        if self._current_node is not None and checked:
            self.display_flag_toggled.emit(self._current_node.id)
        elif self._current_node is not None:
            self.btn_display.setChecked(self._current_node.is_display_node)

    def _on_template_clicked(self, checked: bool) -> None:
        if self._current_node is not None:
            self.template_flag_toggled.emit(self._current_node.id, checked)

    def _on_bypass_clicked(self, checked: bool) -> None:
        if self._current_node is not None and self._current_node.is_bypassed != checked:
            self.undo_snapshot_requested.emit()
            self._current_node.set_bypassed(checked)
            self.bypass_flag_toggled.emit(self._current_node.id, checked)
