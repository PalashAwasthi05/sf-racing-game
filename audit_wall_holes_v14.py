"""Find buildings you can see through.

A wall that "goes invisible and reveals the building behind" is, in a scene with
no transparent materials and consistent winding (both measured: 53/53 materials
OPAQUE, 6 inward faces in the whole scene and none on a building), a HOLE. The
surface simply is not there, so the rasteriser draws whatever stands behind it.

The general heuristic already refuses to create one: detail_compiler cuts the
proxy solid with a wall envelope and rejects the result unless it comes back
watertight and a positive volume (detail_compiler.py:173). The shipped detailed
world was not built by that path — it is the inherited chain
detailed-v11 -> surface-final-v12 -> material-final-v13 — so nothing enforced it.

This audits the shipped scene the way the general rule would. Per building, all
its meshes (host shell, facades, roofs, foundations) are concatenated and merged
by distance, and the result is checked for BOUNDARY EDGES: an edge used by only
one triangle is the rim of a hole. Their total length and the number of separate
loops say how much of the building is missing and in how many places.

Intentional openings are not a confound here: a modelled window or door is a
closed aperture ring, whose edges are shared by two faces. Only an unfinished cut
leaves a free rim.

No location-specific logic: every building group in the scene is judged the same
way, and none is selected or exempted by identity.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

# Vertices this close are the same vertex. Separate meshes that share a corner
# store it twice; without merging, every seam reads as a hole.
MERGE_TOL_M = 1e-4
# A rim shorter than this is a modelling sliver, not something you can see a
# building through.
MIN_LOOP_LENGTH_M = 0.5

GROUP = re.compile(r"^(?:buildings|facades|roofs|foundations)__(osm_[0-9]+)")


def boundary_report(mesh: trimesh.Trimesh) -> dict:
    mesh = mesh.copy()
    mesh.merge_vertices(digits_vertex=int(-np.log10(MERGE_TOL_M)))
    edges = mesh.edges_sorted
    if len(edges) == 0:
        return {"openEdges": 0, "loops": 0, "boundaryLengthM": 0.0, "watertight": True}
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = unique[counts == 1]
    if len(open_edges) == 0:
        return {"openEdges": 0, "loops": 0, "boundaryLengthM": 0.0, "watertight": True}
    v = mesh.vertices
    lengths = np.linalg.norm(v[open_edges[:, 0]] - v[open_edges[:, 1]], axis=1)

    # Group the free edges into loops so "one big missing wall" is told apart
    # from "lots of tiny seams".
    adj = defaultdict(list)
    for i, (a, b) in enumerate(open_edges):
        adj[a].append(i)
        adj[b].append(i)
    seen = np.zeros(len(open_edges), dtype=bool)
    loops = []
    for start in range(len(open_edges)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        total = 0.0
        count = 0
        while stack:
            e = stack.pop()
            total += float(lengths[e])
            count += 1
            for node in open_edges[e]:
                for nxt in adj[node]:
                    if not seen[nxt]:
                        seen[nxt] = True
                        stack.append(nxt)
        loops.append({"edges": count, "lengthM": round(total, 3)})
    loops.sort(key=lambda l: -l["lengthM"])
    big = [l for l in loops if l["lengthM"] >= MIN_LOOP_LENGTH_M]
    return {
        "openEdges": int(len(open_edges)),
        "loops": len(loops),
        "significantLoops": len(big),
        "boundaryLengthM": round(float(lengths.sum()), 2),
        "significantLengthM": round(sum(l["lengthM"] for l in big), 2),
        "largestLoopM": big[0]["lengthM"] if big else 0.0,
        "watertight": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("work/material-final-v13/candidate.glb"))
    ap.add_argument("--report", type=Path, default=Path("outputs/wall-hole-audit-v14.json"))
    args = ap.parse_args()

    scene = trimesh.load(args.scene, process=False)
    geometries = scene.geometry

    groups: dict[str, list] = defaultdict(list)
    for name, mesh in geometries.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        m = GROUP.match(name)
        if m:
            groups[m.group(1)].append((name, mesh))

    rows = []
    for building, parts in groups.items():
        merged = trimesh.util.concatenate([p for _, p in parts])
        rep = boundary_report(merged)
        rows.append({
            "building": building,
            "meshes": len(parts),
            "faces": int(len(merged.faces)),
            **rep,
        })

    rows.sort(key=lambda r: -r["significantLengthM"])
    holed = [r for r in rows if r["significantLengthM"] > 0]

    report = {
        "schema": "wall-hole-audit/v1",
        "scene": str(args.scene),
        "parameters": {"MERGE_TOL_M": MERGE_TOL_M, "MIN_LOOP_LENGTH_M": MIN_LOOP_LENGTH_M},
        "buildings": len(rows),
        "buildingsWithHoles": len(holed),
        "totalSignificantBoundaryM": round(sum(r["significantLengthM"] for r in rows), 2),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: report[k] for k in ("buildings", "buildingsWithHoles", "totalSignificantBoundaryM")}, indent=2))
    print()
    print("  building            meshes   faces   loops  largest   total open rim")
    for r in holed[:20]:
        print(f"  {r['building']:20} {r['meshes']:5} {r['faces']:8} {r['significantLoops']:6}"
              f" {r['largestLoopM']:8.1f} {r['significantLengthM']:12.1f} m")


if __name__ == "__main__":
    main()
