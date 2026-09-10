"""Audit inward-facing wall triangles across the scene.

The renderer sets no material side override for buildings (world-view.ts only
double-sides the study-radius ring), so glTF single-sided materials apply and a
triangle wound inward is INVISIBLE from outside. You then see straight through
the wall to whatever stands behind it.

v0.12 fixed this class for roof caps only (surface_orientation.audit_downward_caps,
24 faces across ten meshes). Nothing audits walls.

This measures the population, per mesh, before anything is changed. Two
independent signals so a single heuristic cannot manufacture the answer:

  windingInconsistent   trimesh's own edge-parity check: a closed surface whose
                        shared edges are traversed in opposite directions by
                        their two faces has mixed winding somewhere.
  outwardDisagreement   for near-vertical faces, whether the face normal points
                        away from the component's own vertical axis. A wall's
                        outside faces away from the mass it encloses. Reported
                        per connected component, so a courtyard or a wing is
                        judged against its own centre rather than the whole
                        building's.

No location-specific logic: every mesh in the scene is judged the same way and
nothing is selected by name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

# A face is a WALL when its normal is within this of horizontal. Roof caps and
# soffits are the other class and already have an audit.
VERTICAL_NORMAL_MAX = 0.35
# Ignore slivers: a triangle this small cannot be the hole you see from a street.
MIN_FACE_AREA_M2 = 0.05


def component_report(mesh: trimesh.Trimesh) -> dict:
    faces = mesh.faces
    if len(faces) == 0:
        return {}
    normals = mesh.face_normals
    areas = mesh.area_faces
    centres = mesh.triangles_center

    up = np.abs(normals[:, 2])
    wall = (up <= VERTICAL_NORMAL_MAX) & (areas >= MIN_FACE_AREA_M2)
    if not wall.any():
        return {
            "wallFaces": 0,
            "inwardFaces": 0,
            "inwardAreaM2": 0.0,
            "windingConsistent": bool(mesh.is_winding_consistent),
        }

    # Per connected component, so a courtyard ring is judged against its own
    # centre and not the building's.
    try:
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(faces))
    except Exception:
        labels = np.zeros(len(faces), dtype=np.int64)

    inward = np.zeros(len(faces), dtype=bool)
    for label in np.unique(labels[wall]):
        sel = (labels == label) & wall
        if sel.sum() < 3:
            continue
        centre_xy = centres[sel][:, :2].mean(axis=0)
        radial = centres[sel][:, :2] - centre_xy
        norm = np.linalg.norm(radial, axis=1)
        keep = norm > 1e-6
        if keep.sum() < 3:
            continue
        radial_unit = np.zeros_like(radial)
        radial_unit[keep] = radial[keep] / norm[keep, None]
        dot = (normals[sel][:, :2] * radial_unit).sum(axis=1)
        idx = np.where(sel)[0]
        inward[idx[keep & (dot < -0.2)]] = True

    return {
        "wallFaces": int(wall.sum()),
        "inwardFaces": int(inward.sum()),
        "inwardAreaM2": float(areas[inward].sum()),
        "windingConsistent": bool(mesh.is_winding_consistent),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("work/material-final-v13/candidate.glb"))
    ap.add_argument("--report", type=Path, default=Path("outputs/wall-orientation-audit-v14.json"))
    args = ap.parse_args()

    scene = trimesh.load(args.scene, process=False)
    geometries = scene.geometry if hasattr(scene, "geometry") else {"scene": scene}

    rows = []
    for name, mesh in geometries.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        rep = component_report(mesh)
        if not rep:
            continue
        rows.append({"mesh": name, **rep})

    affected = [r for r in rows if r["inwardFaces"] > 0]
    affected.sort(key=lambda r: -r["inwardAreaM2"])
    inconsistent = [r for r in rows if not r["windingConsistent"]]

    report = {
        "schema": "wall-orientation-audit/v1",
        "scene": str(args.scene),
        "parameters": {
            "VERTICAL_NORMAL_MAX": VERTICAL_NORMAL_MAX,
            "MIN_FACE_AREA_M2": MIN_FACE_AREA_M2,
        },
        "meshes": len(rows),
        "meshesWithInwardWalls": len(affected),
        "meshesWithInconsistentWinding": len(inconsistent),
        "inwardFacesTotal": sum(r["inwardFaces"] for r in rows),
        "inwardAreaM2Total": round(sum(r["inwardAreaM2"] for r in rows), 2),
        "worst": affected[:40],
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "meshes",
                    "meshesWithInwardWalls",
                    "meshesWithInconsistentWinding",
                    "inwardFacesTotal",
                    "inwardAreaM2Total",
                )
            },
            indent=2,
        )
    )
    for r in affected[:15]:
        print(
            f"  {r['mesh'][:52]:54} {r['inwardFaces']:5} inward of {r['wallFaces']:6} wall faces"
            f"   {r['inwardAreaM2']:9.1f} m2   winding_ok={r['windingConsistent']}"
        )


if __name__ == "__main__":
    main()
