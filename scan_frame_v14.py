"""Cast a grid of sight-lines over one camera frame and map what each one meets.

The single-pixel probe answered "is there a wall at the pixel I guessed". This
answers the question that actually matters: across the WHOLE frame the user is
looking at, which sight-lines meet no surface at all, and which meet a surface
that belongs to a building far behind the one that should be in the way.

Output is a coarse ASCII map aligned to the screenshot, so a suspicious region
can be read straight off it and then probed at full precision with
probe_sightline_v14.py.

Camera is the viewer's: PerspectiveCamera(58, aspect, 0.18, 1100), raw metric
ENU, up = +z. All 951 glb nodes share one rotation (verified), so raw mesh
coordinates and the raw camera are one frame.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

FOV_DEG = 58.0
CHUNK = 20_000  # triangles per batch, to keep peak memory sane


def basis(cam, target):
    f = np.asarray(target, float) - np.asarray(cam, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return f, right, up


def nearest_hits(mesh, origins, dirs):
    """Nearest positive t per ray against one mesh, vectorised over both."""
    tri = mesh.triangles
    best = np.full(len(origins), np.inf)
    for s in range(0, len(tri), CHUNK):
        t3 = tri[s : s + CHUNK]
        v0 = t3[:, 0][None, :, :]
        e1 = (t3[:, 1] - t3[:, 0])[None, :, :]
        e2 = (t3[:, 2] - t3[:, 0])[None, :, :]
        D = dirs[:, None, :]
        P = np.cross(D, e2)
        det = (e1 * P).sum(-1)
        ok = np.abs(det) > 1e-9
        inv = np.zeros_like(det)
        np.divide(1.0, det, out=inv, where=ok)
        T = origins[:, None, :] - v0
        u = (T * P).sum(-1) * inv
        Q = np.cross(T, e1)
        v = (D * Q).sum(-1) * inv
        t = (e2 * Q).sum(-1) * inv
        good = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-3)
        t = np.where(good, t, np.inf)
        best = np.minimum(best, t.min(axis=1))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("work/material-final-v13/candidate.glb"))
    ap.add_argument("--from", dest="cam", default="52,-34,2")
    ap.add_argument("--at", dest="target", default="76,-12,20")
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--rows", type=int, default=36)
    ap.add_argument("--aspect", type=float, default=1280 / 720)
    ap.add_argument("--box", default="", help="x0,y0,x1,y1 in screenshot pixels; blank = whole frame")
    ap.add_argument("--size", default="800,450")
    ap.add_argument("--report", type=Path, default=Path("outputs/frame-scan-v14.json"))
    args = ap.parse_args()

    cam = np.array([float(x) for x in args.cam.split(",")])
    target = np.array([float(x) for x in args.target.split(",")])
    f, right, up = basis(cam, target)
    t = np.tan(np.radians(FOV_DEG) / 2)

    if args.box:
        x0, y0, x1, y1 = (float(v) for v in args.box.split(","))
        W, H = (float(v) for v in args.size.split(","))
        xs = (x0 + (np.arange(args.cols) + 0.5) / args.cols * (x1 - x0)) / W * 2 - 1
        ys = 1 - (y0 + (np.arange(args.rows) + 0.5) / args.rows * (y1 - y0)) / H * 2
    else:
        xs = (np.arange(args.cols) + 0.5) / args.cols * 2 - 1
        ys = 1 - (np.arange(args.rows) + 0.5) / args.rows * 2
    dirs = []
    for y in ys:
        for x in xs:
            d = f + right * (x * t * args.aspect) + up * (y * t)
            dirs.append(d / np.linalg.norm(d))
    dirs = np.asarray(dirs)
    origins = np.repeat(cam[None, :], len(dirs), axis=0)

    scene = trimesh.load(args.scene, process=False)
    best = np.full(len(dirs), np.inf)
    owner = np.array([""] * len(dirs), dtype=object)

    meshes = [(n, m) for n, m in scene.geometry.items() if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    for i, (name, mesh) in enumerate(meshes):
        lo, hi = mesh.bounds
        # Cheap frustum reject: is any bbox corner in front of the camera and
        # within the cone? Conservative, so nothing in view is skipped.
        corners = np.array(np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]])).T.reshape(-1, 3)
        rel = corners - cam
        depth = rel @ f
        if depth.max() <= 0:
            continue
        tvals = nearest_hits(mesh, origins, dirs)
        closer = tvals < best
        best[closer] = tvals[closer]
        owner[closer] = name

    grid = best.reshape(args.rows, args.cols)
    names = owner.reshape(args.rows, args.cols)

    # Legend: '.' background (no surface), digits = distance decade, so a
    # near wall and a far one read differently at a glance.
    print(f"\n  frame scan {args.cols}x{args.rows} from {cam.tolist()} to {target.tolist()}")
    print("  '.' = NOTHING HIT (sky or see-through)   digits = nearest surface, tens of metres\n")
    # Symbol per ray = WHAT IT LANDED ON, not merely whether it landed.
    # The first version printed distance only and treated any hit as fine, so a
    # ray that passed clean through a gap and came to rest on the ground beyond
    # counted as a wall. That is why it reported no gaps.
    def sym(name):
        if not name:
            return "."
        if name.startswith(("terrain", "roads", "markings", "sidewalks", "cycleways", "paths", "water", "landscape")):
            return "_"   # GROUND: the ray reached the floor, so it missed every wall
        if name.startswith("buildings"):
            return "B"
        if name.startswith("facades"):
            return "f"
        if name.startswith("roofs"):
            return "r"
        if name.startswith("foundations"):
            return "n"
        return "o"
    for r in range(args.rows):
        print("   " + "".join(sym(names[r, c]) for c in range(args.cols)))
    print("\n   '.' nothing   '_' GROUND (a gap: the ray reached the floor)   B shell  f facade  r roof  n foundation  o other")

    finite = np.isfinite(best)
    counts = Counter(owner[finite])
    print(f"\n  {int((~finite).sum())} of {len(dirs)} sight-lines hit nothing")
    print("  nearest-surface owners, most rays first:")
    for name, n in counts.most_common(10):
        print(f"    {n:5}  {name}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "camera": {"from": cam.tolist(), "at": target.tolist(), "fovDeg": FOV_DEG, "aspect": args.aspect},
        "grid": {"cols": args.cols, "rows": args.rows},
        "distanceM": [[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in grid],
        "owner": [[str(v) for v in row] for row in names],
    }, indent=1))
    print(f"\n  report {args.report}")


if __name__ == "__main__":
    main()
