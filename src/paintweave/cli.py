"""Headless CLI for cooking PaintWeave (.pw / .nw) graphs and exporting Wavefront .obj meshes."""

from __future__ import annotations

import argparse
import sys

from paintweave.core.graph import Graph
from paintweave.engine.geometry.compute_cuda import get_compute_summary
from paintweave.nodes import register_all_nodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paintweave-cli",
        description="Headless procedural graph evaluator and OBJ exporter for PaintWeave Studio.",
    )
    parser.add_argument(
        "project", nargs="?", help="Path to .pw or .nw project file to cook."
    )
    parser.add_argument(
        "-o", "--export-obj", help="Output path for exported Wavefront .obj mesh."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print registered operators and compute backend info.",
    )

    args = parser.parse_args(argv)
    register_all_nodes()

    if args.info:
        print(f"PaintWeave Studio CLI — {get_compute_summary()}")
        return 0

    if not args.project:
        parser.print_help()
        return 0

    graph = Graph.load_from_file(args.project)
    sop_node = graph.cook_display_sop(force=True)
    if (
        sop_node is None
        or "Geo" not in sop_node.outputs
        or sop_node.outputs["Geo"].value is None
    ):
        print("Error: Project has no active Display SOP geometry.", file=sys.stderr)
        return 1

    geo = sop_node.outputs["Geo"].value
    n_mats = len(geo.detail.get("materials", {}))
    print(
        f"Cooked '{sop_node.name}' in {sop_node.last_cook_time_ms:.2f} ms -> "
        f"Points: {geo.num_points:,} | Triangles: {geo.num_primitives:,} | Materials: {n_mats}"
    )

    if args.export_obj:
        geo.export_to_obj(args.export_obj)
        print(f"Exported Wavefront OBJ to: {args.export_obj}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
