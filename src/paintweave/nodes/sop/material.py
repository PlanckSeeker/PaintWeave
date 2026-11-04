"""SOP Material Operator: binds a VOP / NWShader chain directly onto geometry primitives or groups."""

from __future__ import annotations

from paintweave.core.base_node import NodeBase
from paintweave.core.datatypes import DataType, NodeContext, ParamSpec, PortSpec
from paintweave.engine.geometry.geometry import Geometry
from paintweave.engine.shader.payload import ShaderPayload


class MaterialSOPNode(NodeBase):
    """
    Houdini Material SOP equivalent: binds a VOP/Shader chain (`PortType.SHADER`)
    onto all primitives or a specific primitive/point group (`@material_id`).
    Multiple `SOP_Material` nodes can be stacked or merged for multi-material rendering!
    """

    context = NodeContext.SOP
    category = "Materials & Shading"
    title = "Assign Material (SOP)"

    INPUT_PORTS = [
        PortSpec("Geo", DataType.GEOMETRY, None, "Input geometry to assign shader to."),
        PortSpec("Shader", DataType.SHADER, None, "VOP or NWShader chain to bind."),
    ]
    OUTPUT_PORTS = [
        PortSpec(
            "Geo",
            DataType.GEOMETRY,
            None,
            "Geometry with bound @material_id & ShaderPayload.",
        ),
    ]
    PARAM_SPECS = [
        ParamSpec(
            "group",
            "str",
            "",
            label="Primitive / Point Group",
            description="Group name from SOP_Group (leave empty to assign shader to all primitives).",
        ),
        ParamSpec(
            "material_label",
            "str",
            "mat_surface",
            label="Material Tag",
            description="Human-readable label stored on the bound ShaderPayload.",
        ),
    ]

    def evaluate(self) -> None:
        in_geo: Geometry | None = self.inputs["Geo"].get_data()
        if in_geo is None:
            self.outputs["Geo"].value = Geometry()
            return

        out_geo = in_geo.clone()
        shader: ShaderPayload | None = self.inputs["Shader"].get_data()

        if shader is not None and out_geo.num_primitives > 0:
            group_name = str(self.get_param("group")).strip()
            mat_label = str(self.get_param("material_label")).strip() or "mat_surface"
            # Shallow clone shader metadata so label is specific to this assignment
            bound_shader = ShaderPayload(
                name=mat_label,
                vertex_src=shader.vertex_src,
                fragment_body=shader.fragment_body,
                uniforms=shader.uniforms.copy(),
                is_complete_program=shader.is_complete_program,
            )
            out_geo.assign_material(bound_shader, group_name=group_name)

        self.outputs["Geo"].value = out_geo
