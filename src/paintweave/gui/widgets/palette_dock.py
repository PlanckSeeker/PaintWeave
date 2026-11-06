"""Searchable categorized tree of SOP, VOP, and Math operators."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from paintweave.core.datatypes import NodeContext
from paintweave.core.registry import GLOBAL_NODE_REGISTRY

CONTEXT_ACCENT_COLORS = {
    NodeContext.SOP: "#34d399",  # Alchemical Emerald
    NodeContext.VOP: "#c084fc",  # Arcane Amethyst
    NodeContext.MATH: "#f59e0b",  # Celestial Amber
}


class NodePaletteDock(QWidget):
    """Searchable categorized tree of SOP, VOP, and Math VPL operators."""

    node_spawn_requested = Signal(str)  # Emits node_id (e.g. 'SOP_Grid')

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search SOP, VOP, or Math nodes...")
        self.search_input.textChanged.connect(self._filter_tree)
        layout.addWidget(self.search_input)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Operator Catalog"])
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree, stretch=1)

        add_btn = QPushButton("+ Spawn Selected Operator")
        add_btn.setObjectName("PrimaryActionBtn")
        add_btn.clicked.connect(self._on_add_clicked)
        layout.addWidget(add_btn)

        self.refresh_library()

    def refresh_library(self) -> None:
        self.tree.clear()
        for category_header, regs in GLOBAL_NODE_REGISTRY.by_category().items():
            cat_item = QTreeWidgetItem(self.tree, [category_header])
            cat_item.setExpanded(True)
            if regs:
                accent = CONTEXT_ACCENT_COLORS.get(regs[0].context, "#94a3b8")
                cat_item.setForeground(0, QColor(accent))

            for reg in regs:
                child = QTreeWidgetItem(cat_item, [f"{reg.title}  ({reg.node_id})"])
                child.setData(0, Qt.ItemDataRole.UserRole, reg.node_id)
                child.setToolTip(
                    0, f"<b>{reg.title}</b> [{reg.node_id}]<br>{reg.description}"
                )
        self.tree.expandAll()

    def _filter_tree(self, query: str) -> None:
        q = query.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            if cat is None:
                continue
            visible_children = 0
            for j in range(cat.childCount()):
                item = cat.child(j)
                if item is None:
                    continue
                match = q in item.text(0).lower()
                item.setHidden(not match)
                if match:
                    visible_children += 1
            cat.setHidden(visible_children == 0 and bool(q))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        type_id = item.data(0, Qt.ItemDataRole.UserRole)
        if type_id:
            self.node_spawn_requested.emit(str(type_id))

    def _on_add_clicked(self) -> None:
        item = self.tree.currentItem()
        if item is not None:
            type_id = item.data(0, Qt.ItemDataRole.UserRole)
            if type_id:
                self.node_spawn_requested.emit(str(type_id))
