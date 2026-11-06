"""Virtualized Houdini Geometry Spreadsheet inspecting Points, Primitives, Detail, and Bound Materials."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from paintweave.engine.geometry.geometry import Geometry


class ColumnarGeoTableModel(QAbstractTableModel):
    """Zero-copy virtualized Qt table model over NumPy geometry attribute columns."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._headers: list[str] = []
        self._columns: list[tuple[np.ndarray, int | None, str]] = []
        self._static_rows: list[list[str]] = []
        self._num_rows: int = 0
        self._is_static: bool = False

    def clear_data(self) -> None:
        self.beginResetModel()
        self._headers = []
        self._columns = []
        self._static_rows = []
        self._num_rows = 0
        self._is_static = False
        self.endResetModel()

    def set_columnar_data(
        self,
        headers: list[str],
        columns: list[tuple[np.ndarray, int | None, str]],
        num_rows: int,
    ) -> None:
        self.beginResetModel()
        self._headers = headers
        self._columns = columns
        self._static_rows = []
        self._num_rows = num_rows
        self._is_static = False
        self.endResetModel()

    def set_static_rows(self, headers: list[str], rows: list[list[str]]) -> None:
        self.beginResetModel()
        self._headers = headers
        self._columns = []
        self._static_rows = rows
        self._num_rows = len(rows)
        self._is_static = True
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        if parent.isValid():
            return 0
        return self._num_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        if parent.isValid():
            return 0
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        r = index.row()
        c = index.column()
        if self._is_static:
            return self._static_rows[r][c]

        if c == 0:
            return str(r)

        arr, sub_idx, fmt = self._columns[c - 1]
        val = arr[r] if sub_idx is None else arr[r, sub_idx]
        if fmt == "int":
            return str(int(val))
        return f"{float(val):.3f}"

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(
            self._headers
        ):
            return self._headers[section]
        return str(section)


class GeometrySpreadsheet(QWidget):
    """Houdini-style Geometry Spreadsheet inspecting Points, Primitives, Detail, and Bound Materials."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_bar = QHBoxLayout()
        self.domain_combo = QComboBox()
        self.domain_combo.addItems(["Points", "Primitives", "Detail", "Materials"])
        self.domain_combo.currentTextChanged.connect(self._refresh_table)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter attributes (e.g. P, Cd, group)...")
        self.filter_edit.setMaximumWidth(240)
        self.filter_edit.textChanged.connect(self._refresh_table)

        self.summary_label = QLabel("Pts: 0  |  Prims: 0  |  Mats: 0")
        self.summary_label.setStyleSheet("color: #e5b869; font-weight: 700; font-family: Consolas, monospace;")

        top_bar.addWidget(QLabel("Domain:"))
        top_bar.addWidget(self.domain_combo)
        top_bar.addSpacing(8)
        top_bar.addWidget(self.filter_edit)
        top_bar.addStretch(1)
        top_bar.addWidget(self.summary_label)
        layout.addLayout(top_bar)

        self.model = ColumnarGeoTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=1)

        self._current_geo: Geometry | None = None

    def set_geometry(self, geo: Geometry | None) -> None:
        self._current_geo = geo
        if geo is None:
            self.summary_label.setText("Pts: 0  |  Prims: 0  |  Mats: 0")
            self.model.clear_data()
            return

        n_mats = len(geo.detail.get("materials", {}))
        self.summary_label.setText(
            f"Pts: {geo.num_points:,}  |  Prims: {geo.num_primitives:,}  |  Mats: {n_mats}"
        )
        self._refresh_table()

    def _refresh_table(self) -> None:
        geo = self._current_geo
        if geo is None:
            self.model.clear_data()
            return

        domain = self.domain_combo.currentText()
        attr_filter = self.filter_edit.text().strip().lower()

        if domain == "Points":
            pt_keys = list(geo.points.keys())
            standard = ["P", "N", "Cd", "uv"]
            display_keys: list[str] = []

            for k in standard:
                if k in pt_keys and geo.points[k].shape[0] == geo.num_points:
                    if not attr_filter or attr_filter in k.lower():
                        display_keys.append(k)
            for k in pt_keys:
                if k not in standard and geo.points[k].shape[0] == geo.num_points:
                    if not attr_filter or attr_filter in k.lower():
                        display_keys.append(k)

            headers = ["ptnum"]
            columns: list[tuple[np.ndarray, int | None, str]] = []

            for k in display_keys:
                arr = geo.points[k]
                if arr.ndim == 1:
                    headers.append(k)
                    columns.append((arr, None, "float"))
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    headers.append(k)
                    columns.append((arr, 0, "float"))
                elif arr.shape[1] == 2:
                    headers.extend([f"{k}.u", f"{k}.v"])
                    columns.extend([(arr, 0, "float"), (arr, 1, "float")])
                elif arr.shape[1] == 3:
                    suffixes = ("r", "g", "b") if k == "Cd" else ("x", "y", "z")
                    for idx, sfx in enumerate(suffixes):
                        headers.append(f"{k}.{sfx}")
                        columns.append((arr, idx, "float"))
                elif arr.shape[1] == 4:
                    for idx, sfx in enumerate(("x", "y", "z", "w")):
                        headers.append(f"{k}.{sfx}")
                        columns.append((arr, idx, "float"))

            self.model.set_columnar_data(headers, columns, geo.num_points)

        elif domain == "Primitives":
            tris = geo.primitives["triangles"]
            mat_ids = geo.ensure_material_ids()
            headers = ["primnum", "vtx_0", "vtx_1", "vtx_2", "material_id"]
            columns = [
                (tris, 0, "int"),
                (tris, 1, "int"),
                (tris, 2, "int"),
                (mat_ids, None, "int"),
            ]
            self.model.set_columnar_data(headers, columns, geo.num_primitives)

        elif domain == "Detail":
            headers = ["Detail Attribute", "Value"]
            rows: list[list[str]] = []
            for k, v in geo.detail.items():
                if k == "materials":
                    rows.append([k, f"{len(v)} bound ShaderPayload(s)"])
                elif isinstance(v, np.ndarray):
                    rows.append([k, np.array2string(v, precision=3, separator=", ")])
                else:
                    rows.append([str(k), str(v)])
            self.model.set_static_rows(headers, rows)

        elif domain == "Materials":
            headers = [
                "material_id",
                "Shader Name",
                "Complete Program",
                "Uniforms",
                "Assigned Triangles",
            ]
            mats = geo.detail.get("materials", {})
            mat_ids = geo.ensure_material_ids()
            rows = []
            for mid, payload in sorted(mats.items()):
                tri_cnt = int(np.sum(mat_ids == mid)) if len(mat_ids) > 0 else 0
                u_str = (
                    ", ".join(f"{uk}:{ut}" for uk, (ut, _) in payload.uniforms.items())
                    or "None"
                )
                rows.append(
                    [
                        str(mid),
                        str(getattr(payload, "name", f"mat_{mid}")),
                        str(bool(getattr(payload, "is_complete_program", False))),
                        u_str,
                        f"{tri_cnt:,}",
                    ]
                )
            if not rows:
                rows.append(
                    [
                        "-1",
                        "Default Studio Clay Shader",
                        "True",
                        "u_mvp, u_model",
                        f"{geo.num_primitives:,}",
                    ]
                )
            self.model.set_static_rows(headers, rows)
