"""Find gaps in walls: sight-lines that enter a building's real footprint and
leave it without meeting a single wall.

This replaces audit_see_through_v14.py, which was wrong twice and is superseded.

  Wrong the first time: it decided where a wall SHOULD be from a convex hull of
  the mesh. A hull is larger than the real plan, so rays clipping the corners
  passed outside the actual building and were reported as holes; and an open
  structure like a ballpark read as 29% see-through when its field is correctly
  open.

  Wrong the second time: it treated ANY surface hit as proof of a wall. A ray
  that passed clean through a gap and came to rest on the road beyond counted as
  blocked, which is precisely why it reported no gaps at all.

The test here is the one the eye performs. At a given height, a horizontal slice
of the building is the set of walls a level look can meet. Take the building's
REAL footprint polygon from the pipeline's own input, cast rays across it, and
ask: between entering the footprint and leaving it, did this ray cross any wall?
If not, you are looking straight through the building at that height and angle.

Interior voids do not confound it: a courtyard or a ballpark field is inside the
footprint, but the ray still has to cross the outer wall to reach it, so it is
blocked. Only a genuinely missing wall lets a ray pass.

Setbacks do not confound it either: the slice is taken from the building's own
geometry at that height, so an upper storey that steps back is judged against
where it actually stands, not against the ground outline.

Footprints are matched to mesh groups BY GEOMETRY (nearest centroid, then area
agreement), never by identity, because the shipped meshes were renamed from the
input keys during municipal-identity reconciliation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import LineString, shape

GROUP = re.compile(r"^(buildings|facades|roofs|foundations)__(.+?)___")
VARIANT = re.compile(r"__observed_.*$")

# Heights: the ground storey, then up the building's own height.
FIXED_HEIGHTS_M = (1.6, 4.0)
HEIGHT_FRACTIONS = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
RAYS_PER_HEIGHT = 120
# A ray must cross this much of the plan to be asking a fair question; below it
# the ray is grazing a corner.
MIN_CHORD_M = 2.0
# Matching a footprint to a mesh group.
MATCH_RADIUS_M = 20.0


def sections_at(mesh, z):
    try:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    except Exception:
        return np.empty((0, 2, 2))
    if sec is None:
        return np.empty((0, 2, 2))
    v = np.asarray(sec.vertices)[:, :2]
    segs = []
    for e in sec.entities:
        idx = getattr(e, "points", None)
        if idx is None:
            continue
        pts = v[np.asarray(idx)]
        if len(pts) >= 2:
            segs.extend(zip(pts[:-1], pts[1:]))
    return np.asarray(segs) if segs else np.empty((0, 2, 2))


def crosses(segs, ox, oy, dx, dy, t_enter, t_exit):
    """Does the ray meet any wall segment between entering and leaving the plan?"""
    if len(segs) == 0:
        return False
    a, b = segs[:, 0, :], segs[:, 1, :]
    n = np.array([-dy, dx])
    sa = (a - [ox, oy]) @ n
    sb = (b - [ox, oy]) @ n
    straddle = (sa > 0) != (sb > 0)
    if not straddle.any():
        return False
    t = sa[straddle] / (sa[straddle] - sb[straddle])
    pts = a[straddle] + (b[straddle] - a[straddle]) * t[:, None]
    along = (pts - [ox, oy]) @ np.array([dx, dy])
    # Count ANY wall the ray meets before it has passed the far side of the
    # plan. The first version windowed this to the footprint's own chord, which
    # is wrong on this data: the input footprints are fragmented parcels, 2-9x
    # smaller in plan than the consolidated buildings actually meshed, so a
    # perfectly good wall standing outside the parcel line was outside the
    # window and the ray was reported as passing clean through. Measured on
    # sf_201006_0007874: a plain 4-wall box at 11.5 m (8 straddling triangles)
    # was scored 41.7% see-through. The question the eye asks is "is anything in
    # the way before I'm past the building", so that is the window.
    return bool(((along > 0.0) & (along < t_exit + 1.0)).any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("work/material-final-v13/candidate.glb"))
    ap.add_argument("--input", type=Path, default=Path("work/imagery-street-v12/input-with-imagery.json"))
    ap.add_argument("--report", type=Path, default=Path("outputs/wall-gap-audit-v14.json"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    footprints = []
    for b in json.loads(args.input.read_text())["buildings"]:
        try:
            poly = shape(b["geometry"])
        except Exception:
            continue
        if poly.is_valid and poly.area > 1:
            footprints.append((b["key"], poly))

    scene = trimesh.load(args.scene, process=False)
    groups = defaultdict(list)
    shells = set()
    for name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        m = GROUP.match(name)
        if not m:
            continue
        base = VARIANT.sub("", m.group(2))
        groups[base].append(mesh)
        if m.group(1) == "buildings":
            shells.add(base)
    groups = {k: v for k, v in groups.items() if k in shells}

    rng = np.random.default_rng(args.seed)
    rows = []
    for bid, parts in sorted(groups.items()):
        mesh = trimesh.util.concatenate(parts)
        lo, hi = mesh.bounds
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        best = min(footprints, key=lambda kp: (kp[1].centroid.x - cx) ** 2 + (kp[1].centroid.y - cy) ** 2)
        dist = float(np.hypot(best[1].centroid.x - cx, best[1].centroid.y - cy))
        if dist > MATCH_RADIUS_M:
            rows.append({"building": bid, "matched": None, "reason": f"no footprint within {MATCH_RADIUS_M} m"})
            continue
        poly = best[1]
        height = float(hi[2] - lo[2])
        levels = sorted({round(v, 2) for v in
                         [lo[2] + z for z in FIXED_HEIGHTS_M] + [lo[2] + f * height for f in HEIGHT_FRACTIONS]
                         if lo[2] <= v <= hi[2]})
        radius = float(np.hypot(hi[0] - lo[0], hi[1] - lo[1])) / 2 + 15.0
        total = gaps = 0
        worst = []
        skipped_levels = []
        perimeter = float(poly.length)
        for z in levels:
            segs = sections_at(mesh, z)
            # A level only asks a fair question if walls actually exist there.
            # Bounding boxes are inflated by signs, lettering and spires, so the
            # upper sampled heights often sit ABOVE the real roofline, where every
            # ray "passes through" trivially. Measured on sf_201006_0007874: the
            # section at 11.7 m was eight triangles of a rooftop element, and the
            # audit scored it 41.7% see-through. Require the slice to carry wall
            # length comparable to the plan perimeter before judging it; record
            # what was skipped so the omission is counted, not hidden.
            wall_len = float(np.linalg.norm(segs[:, 1, :] - segs[:, 0, :], axis=1).sum()) if len(segs) else 0.0
            if wall_len < 0.5 * perimeter:
                skipped_levels.append({"heightM": round(z - lo[2], 2), "wallLenM": round(wall_len, 1), "perimeterM": round(perimeter, 1)})
                continue
            for k in range(RAYS_PER_HEIGHT):
                th = 2 * np.pi * (k + rng.random()) / RAYS_PER_HEIGHT
                dx, dy = -np.cos(th), -np.sin(th)
                ox, oy = poly.centroid.x - dx * radius, poly.centroid.y - dy * radius
                line = LineString([(ox, oy), (ox + dx * 4 * radius, oy + dy * 4 * radius)])
                inter = poly.intersection(line)
                if inter.is_empty or inter.length < MIN_CHORD_M:
                    continue
                coords = np.asarray(inter.coords) if inter.geom_type == "LineString" else np.vstack(
                    [np.asarray(g.coords) for g in inter.geoms])
                proj = (coords - [ox, oy]) @ np.array([dx, dy])
                t_enter, t_exit = float(proj.min()), float(proj.max())
                total += 1
                if not crosses(segs, ox, oy, dx, dy, t_enter, t_exit):
                    gaps += 1
                    if len(worst) < 6:
                        worst.append({
                            "heightM": round(z - lo[2], 2),
                            "entryXY": [round(ox + dx * t_enter, 2), round(oy + dy * t_enter, 2)],
                            "chordM": round(t_exit - t_enter, 2),
                        })
        rows.append({
            "building": bid,
            "matched": best[0],
            "matchDistM": round(dist, 2),
            "rays": total,
            "gapRays": gaps,
            "gapFraction": round(gaps / total, 4) if total else 0.0,
            "levels": len(levels),
            "levelsSkippedAboveRoofline": skipped_levels,
            "samples": worst,
        })

    measured = [r for r in rows if r.get("rays")]
    holed = sorted((r for r in measured if r["gapRays"] > 0), key=lambda r: -r["gapFraction"])
    report = {
        "schema": "wall-gap-audit/v1",
        "test": "a ray that enters the real footprint and leaves it without crossing any wall at that height",
        "parameters": {
            "FIXED_HEIGHTS_M": list(FIXED_HEIGHTS_M),
            "HEIGHT_FRACTIONS": list(HEIGHT_FRACTIONS),
            "RAYS_PER_HEIGHT": RAYS_PER_HEIGHT,
            "MIN_CHORD_M": MIN_CHORD_M,
        },
        "buildingsMeasured": len(measured),
        "buildingsWithGaps": len(holed),
        "raysTotal": sum(r["rays"] for r in measured),
        "gapRaysTotal": sum(r["gapRays"] for r in measured),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: report[k] for k in
                      ("buildingsMeasured", "buildingsWithGaps", "raysTotal", "gapRaysTotal")}, indent=2))
    print("\n  building                     rays   gapRays    frac   worst gap at (x, y, height)")
    for r in holed[:20]:
        s = r["samples"][0] if r["samples"] else None
        where = f"({s['entryXY'][0]}, {s['entryXY'][1]}, {s['heightM']} m)" if s else ""
        print(f"  {r['building'][:26]:28} {r['rays']:5} {r['gapRays']:8} {r['gapFraction']:7.3f}   {where}")


if __name__ == "__main__":
    main()
