"""Master registration entry point for all SOP, VOP, and Math nodes in PaintWeave."""

from paintweave.nodes.math import register_math_nodes
from paintweave.nodes.sop import register_sop_nodes
from paintweave.nodes.vop import register_vop_nodes


def register_all_nodes() -> None:
    """Register all SOP, VOP, and Math nodes into the global NodeRegistry."""
    register_math_nodes()
    register_sop_nodes()
    register_vop_nodes()
