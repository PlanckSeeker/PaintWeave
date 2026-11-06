"""Surface Operator (SOP) registration."""

from paintweave.core.registry import NodeRegistry
from paintweave.nodes.sop.attributes import (
    AmbientGradientSOPNode,
    AttributeRandomizeSOPNode,
    ColorSOPNode,
    CurvatureSOPNode,
    GroupSOPNode,
    UVProjectSOPNode,
    WrangleSOPNode,
)
from paintweave.nodes.sop.instancing import (
    CopyToPointsSOPNode,
    MergeSOPNode,
    ScatterSOPNode,
)
from paintweave.nodes.sop.material import MaterialSOPNode
from paintweave.nodes.sop.modifiers import (
    ExtrudeSOPNode,
    FacetSOPNode,
    MountainSOPNode,
    NormalSOPNode,
    RaycastSOPNode,
    SmoothSOPNode,
    StylizedDamageSOPNode,
    SubdivideSOPNode,
    SweepSOPNode,
    TransformSOPNode,
)
from paintweave.nodes.sop.primitives import (
    BoxSOPNode,
    CurveSOPNode,
    CylinderSOPNode,
    FileSOPNode,
    GridSOPNode,
    SphereSOPNode,
)


def register_sop_nodes() -> None:
    NodeRegistry.register("SOP_Grid", GridSOPNode)
    NodeRegistry.register("SOP_Box", BoxSOPNode)
    NodeRegistry.register("SOP_Cylinder", CylinderSOPNode)
    NodeRegistry.register("SOP_Sphere", SphereSOPNode)
    NodeRegistry.register("SOP_Curve", CurveSOPNode)
    NodeRegistry.register("SOP_File", FileSOPNode)

    NodeRegistry.register("SOP_Transform", TransformSOPNode)
    NodeRegistry.register("SOP_Mountain", MountainSOPNode)
    NodeRegistry.register("SOP_Extrude", ExtrudeSOPNode)
    NodeRegistry.register("SOP_Subdivide", SubdivideSOPNode)
    NodeRegistry.register("SOP_Facet", FacetSOPNode)
    NodeRegistry.register("SOP_Smooth", SmoothSOPNode)
    NodeRegistry.register("SOP_Normal", NormalSOPNode)
    NodeRegistry.register("SOP_Sweep", SweepSOPNode)
    NodeRegistry.register("SOP_Raycast", RaycastSOPNode)
    NodeRegistry.register("SOP_StylizedDamage", StylizedDamageSOPNode)

    NodeRegistry.register("SOP_Group", GroupSOPNode)
    NodeRegistry.register("SOP_Color", ColorSOPNode)
    NodeRegistry.register("SOP_AttributeRandomize", AttributeRandomizeSOPNode)
    NodeRegistry.register("SOP_UVProject", UVProjectSOPNode)
    NodeRegistry.register("SOP_Wrangle", WrangleSOPNode)
    NodeRegistry.register("SOP_AmbientGradient", AmbientGradientSOPNode)
    NodeRegistry.register("SOP_Curvature", CurvatureSOPNode)

    NodeRegistry.register("SOP_Scatter", ScatterSOPNode)
    NodeRegistry.register("SOP_CopyToPoints", CopyToPointsSOPNode)
    NodeRegistry.register("SOP_Merge", MergeSOPNode)

    NodeRegistry.register("SOP_Material", MaterialSOPNode)
