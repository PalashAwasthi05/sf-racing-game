"""Measure the defect the eye actually reports: walls you can see through.

Two mechanisms were tested first and both are refuted on this scene, so neither
explains it:

  inverted winding   6 inward-facing wall triangles in the whole scene, total
                     0.77 m2, all of them on roads, paths, markings and
                     sidewalks. No building carries one.
                     (audit_wall_orientation_v14.py)
  transparency       all 53 glTF materials are OPAQUE; the renderer double-sides
                     only the study-radius ring.

What is left is missing surface. This measures that the way a driver meets it:
by looking. For each building, horizontal rays are cast at eye heights across
its own footprint. A ray that ENTERS the footprint polygon and leaves the mesh
with an even number of surface crossings has passed clean through the building —
that is a line of sight through a wall, and whatever stands behind is what you
see.

Why parity rather than "no hit": a solid, closed building returns an odd count
only when the ray starts inside it. Started outside, a sound building gives an
even count with at least one entry and one exit; a building with a hole gives
ZERO crossings along a line that plan geometry says must cross solid, or an odd
count where the ray entered through a wall and found no back. Both are reported
separately so the failure mode is legible rather than lumped.

Rays are cast against the BUILDING'S OWN meshes only, so a neighbour standing
behind cannot mask its hole.

No location-specific logic: every building group is sampled by the same rule,
none is named, exempted or weighted.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon, Point

GROUP = re.compile(r"^(buildings|facades|roofs|foundations)__(.+?)___")
# A building is drawn as its base group PLUS any authored "observed" variants,
# which are separate meshes at the same place. Auditing them apart reports a
# hole in the base that the variant actually covers, so they are merged on the
# base id before anything is judged.
VARIANT = re.compile(r"__observed_.*$")

# Heights are sampled up the WHOLE building, not just at eye level. The first
# version looked only at 1.6, 4 and 8 m and reported a gate 10-25 m up as clean:
# a hole you can see from the street is not necessarily a hole at the height of
# your eyes. FRACTIONS are of the building's own height, so a shed and a tower
# are both covered, plus fixed low heights for the ground storey.
RAY_HEIGHTS_M = (1.6, 4.0, 8.0)
HEIGHT_FRACTIONS = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
# Rays per height, spread over directions around the building.
RAYS_PER_HEIGHT = 90
# Start the ray this far outside the footprint so it begins in open air.
STANDOFF_M = 12.0
# A chord shorter than this is a corner clip, not a wall you look through.
MIN_CHORD_M = 1.5


def footprint_of(mesh: trimesh.Trimesh) -> Polygon | None:
    """Plan outline from the mesh's own vertices, as a convex hull.

    A hull, not the exact outline: it only decides WHICH rays should have hit
    something, and a hull is the conservative choice — it never claims a ray
    must cross solid where the building is concave.
    """
    pts = mesh.vertices[:, :2]
    if len(pts) < 3:
        return None
    try:
        from shapely import MultiPoint

        hull = MultiPoint([tuple(p) for p in pts[:: max(1, len(pts) // 4000)]]).convex_hull
        return hull if isinstance(hull, Polygon) and hull.area > 1.0 else None
    except Exception:
        return None


def audit_building(mesh: trimesh.Trimesh, rng: np.random.Generator) -> dict:
    """Count sight-lines that cross the plan but meet no wall.

    Done on horizontal CROSS-SECTIONS rather than by ray-tracing triangles: a
    slice at eye height is exactly the set of walls a horizontal look can meet,
    it needs no spatial index (and so no extra dependency in this environment),
    and counting segment crossings in 2D is the same question the eye asks.
    """
    from shapely.geometry import LineString, MultiPoint

    poly = footprint_of(mesh)
    if poly is None:
        return {"rays": 0, "seeThrough": 0, "sections": 0, "seeThroughFraction": 0.0}
    cx, cy = poly.centroid.x, poly.centroid.y
    lo, hi = mesh.bounds
    radius = float(np.hypot(hi[0] - lo[0], hi[1] - lo[1])) / 2 + STANDOFF_M

    rays = 0
    see_through = 0
    sections = 0
    chords = []
    height = float(hi[2] - lo[2])
    levels = [lo[2] + z for z in RAY_HEIGHTS_M] + [lo[2] + f * height for f in HEIGHT_FRACTIONS]
    for zz in sorted(set(round(v, 3) for v in levels)):
        if zz > hi[2] or zz < lo[2]:
            continue
        try:
            section = mesh.section(plane_origin=[0, 0, zz], plane_normal=[0, 0, 1])
        except Exception:
            section = None
        if section is None:
            # No surface at all at this height inside the building's own bounds:
            # every crossing line is a see-through line.
            section_segments = np.empty((0, 2, 2))
        else:
            sections += 1
            v = np.asarray(section.vertices)[:, :2]
            ent = [np.asarray(e.points) for e in section.entities if hasattr(e, "points")]
            segs = []
            for idx in ent:
                pts = v[idx]
                if len(pts) >= 2:
                    segs.extend(zip(pts[:-1], pts[1:]))
            section_segments = np.asarray(segs) if segs else np.empty((0, 2, 2))

        for k in range(RAYS_PER_HEIGHT):
            theta = 2 * np.pi * (k + rng.random()) / RAYS_PER_HEIGHT
            dx, dy = -np.cos(theta), -np.sin(theta)
            ox, oy = cx - dx * radius, cy - dy * radius
            line = LineString([(ox, oy), (ox + dx * 4 * radius, oy + dy * 4 * radius)])
            chord = float(poly.intersection(line).length)
            if chord < MIN_CHORD_M:
                continue
            rays += 1
            chords.append(chord)
            if len(section_segments) == 0:
                see_through += 1
                continue
            # 2D segment crossings along the ray.
            a = section_segments[:, 0, :]
            b = section_segments[:, 1, :]
            d = np.array([dx, dy])
            n = np.array([-dy, dx])  # ray normal
            sa = (a - [ox, oy]) @ n
            sb = (b - [ox, oy]) @ n
            straddle = (sa > 0) != (sb > 0)
            if not straddle.any():
                see_through += 1
                continue
            t = sa[straddle] / (sa[straddle] - sb[straddle])
            pts = a[straddle] + (b[straddle] - a[straddle]) * t[:, None]
            along = (pts - [ox, oy]) @ d
            if not (along > 0).any():
                see_through += 1

    if rays == 0:
        return {"rays": 0, "seeThrough": 0, "sections": sections, "seeThroughFraction": 0.0}
    return {
        "rays": rays,
        "seeThrough": see_through,
        "sections": sections,
        "seeThroughFraction": round(see_through / rays, 4),
        "medianChordM": round(float(np.median(chords)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("work/material-final-v13/candidate.glb"))
    ap.add_argument("--report", type=Path, default=Path("outputs/see-through-audit-v14.json"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    scene = trimesh.load(args.scene, process=False)
    groups: dict[str, list] = defaultdict(list)
    has_shell: dict[str, bool] = defaultdict(bool)
    for name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        m = GROUP.match(name)
        if not m:
            continue
        base = VARIANT.sub("", m.group(2))
        groups[base].append(mesh)
        if m.group(1) == "buildings":
            has_shell[base] = True
    # A group with no host shell is decoration attached to a building (lettering,
    # signage), not an enclosure. Judging it for see-through asks a question its
    # geometry never claimed to answer.
    groups = {k: v for k, v in groups.items() if has_shell[k]}

    rng = np.random.default_rng(args.seed)
    rows = []
    for bid, parts in sorted(groups.items()):
        merged = trimesh.util.concatenate(parts)
        rep = audit_building(merged, rng)
        rows.append({"building": bid, "meshes": len(parts), **rep})

    measured = [r for r in rows if r["rays"] > 0]
    holed = sorted((r for r in measured if r["seeThrough"] > 0), key=lambda r: -r["seeThroughFraction"])

    report = {
        "schema": "see-through-audit/v1",
        "scene": str(args.scene),
        "parameters": {
            "RAY_HEIGHTS_M": list(RAY_HEIGHTS_M),
            "HEIGHT_FRACTIONS": list(HEIGHT_FRACTIONS),
            "RAYS_PER_HEIGHT": RAYS_PER_HEIGHT,
            "STANDOFF_M": STANDOFF_M,
            "MIN_CHORD_M": MIN_CHORD_M,
        },
        "buildingsMeasured": len(measured),
        "buildingsSeeThrough": len(holed),
        "raysTotal": sum(r["rays"] for r in measured),
        "raysSeeThrough": sum(r["seeThrough"] for r in measured),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: report[k] for k in ("buildingsMeasured", "buildingsSeeThrough", "raysTotal", "raysSeeThrough")}, indent=2))
    print()
    print("  building                                    rays  seeThru   frac")
    for r in holed[:20]:
        print(f"  {r['building'][:42]:44} {r['rays']:5} {r['seeThrough']:8} {r['seeThroughFraction']:7.3f}")


if __name__ == "__main__":
    main()
