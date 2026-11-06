"""VOP Operator registration."""

from paintweave.core.registry import NodeRegistry
from paintweave.nodes.vop.code import NWShaderNode, WrangleVOPNode
from paintweave.nodes.vop.lighting import (
    ArcaneRampVOPNode,
    BrushImpastoVOPNode,
    CelToonerVOPNode,
    EdgeGlintVOPNode,
    FresnelVOPNode,
    IsoDirectionalLightVOPNode,
    MatCapVOPNode,
    RimLightVOPNode,
)
from paintweave.nodes.vop.patterns import (
    CheckerboardVOPNode,
    ColorRampVOPNode,
    ProceduralNoiseVOPNode,
    SimplexNoise3DVOPNode,
    UVCheckerVOPNode,
)
from paintweave.nodes.vop.post import BayerDitherVOPNode, BlendVOPNode, GammaVOPNode


def register_vop_nodes() -> None:
    NodeRegistry.register("VOP_IsoLight", IsoDirectionalLightVOPNode)
    NodeRegistry.register("VOP_RimLight", RimLightVOPNode)
    NodeRegistry.register("VOP_Fresnel", FresnelVOPNode)
    NodeRegistry.register("VOP_CelToon", CelToonerVOPNode)
    NodeRegistry.register("VOP_ArcaneRamp", ArcaneRampVOPNode)
    NodeRegistry.register("VOP_EdgeGlint", EdgeGlintVOPNode)
    NodeRegistry.register("VOP_BrushImpasto", BrushImpastoVOPNode)
    NodeRegistry.register("VOP_MatCap", MatCapVOPNode)

    NodeRegistry.register("VOP_Checkerboard", CheckerboardVOPNode)
    NodeRegistry.register("VOP_Noise", ProceduralNoiseVOPNode)
    NodeRegistry.register("VOP_SimplexNoise3D", SimplexNoise3DVOPNode)
    NodeRegistry.register("VOP_UVChecker", UVCheckerVOPNode)
    NodeRegistry.register("VOP_ColorRamp", ColorRampVOPNode)

    NodeRegistry.register("VOP_BayerDither", BayerDitherVOPNode)
    NodeRegistry.register("VOP_Blend", BlendVOPNode)
    NodeRegistry.register("VOP_Gamma", GammaVOPNode)

    NodeRegistry.register("VOP_Wrangle", WrangleVOPNode)
    NodeRegistry.register("VOP_NWShader", NWShaderNode)
