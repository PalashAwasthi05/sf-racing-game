"""Cast one sight-line and report every surface it meets, in order.

Settles the question the eye cannot: at a pixel that looks wrong, is there a
wall wearing the wrong material, or is there no wall at all?

Camera parameters are the viewer's own: PerspectiveCamera(58, aspect, 0.18,
1100) in world-view.ts:13, and ?viewFrom/?viewAt read raw metric ENU and are
mapped to viewer axes by reviewVector (world-view.ts:98) as (x, z, -y). Every
node in the glb carries the SAME matrix — the z-up to y-up rotation, verified,
951 of 951 — so raw mesh coordinates and the raw camera live in one consistent
frame and the whole test can be done in raw ENU.

Triangle intersection is Möller-Trumbore, done in numpy against the meshes whose
bounding box the ray actually pierces. No spatial index, so no new dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

FOV_DEG = 58.0  # world-view.ts:13


def ray_dir(cam, target, ndcx, ndcy, aspect, fov_deg=FOV_DEG):
    f = np.asarray(target, float) - np.asarray(cam, float)
    f /= np.linalg.norm(f)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    t = np.tan(np.radians(fov_deg) / 2)
    d = f + right * (ndcx * t * aspect) + up * (ndcy * t)
    return d / np.linalg.norm(d)


def slab_hit(origin, direction, lo, hi):
    inv = np.divide(1.0, direction, out=np.full(3, np.inf), where=direction != 0)
    t1 = (lo - origin) * inv
    t2 = (hi - origin) * inv
    tmin = np.max(np.minimum(t1, t2))
    tmax = np.min(np.maximum(t1, t2))
    return tmax >= max(tmin, 0.0)


def hits(mesh, origin, direction):
    tri = mesh.triangles
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(direction, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-9
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tvec = origin - v0
    u = (tvec * p).sum(1) * inv
    q = np.cross(tvec, e1)
    v = (direction * q).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    good = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-4)
    return t[good]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("work/material-final-v13/candidate.glb"))
    ap.add_argument("--from", dest="cam", default="52,-34,2")
    ap.add_argument("--at", dest="target", default="76,-12,20")
    ap.add_argument("--pixel", default="535,232", help="pixel in the screenshot")
    ap.add_argument("--size", default="800,450", help="screenshot size")
    ap.add_argument("--aspect", type=float, default=1280 / 720)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cam = np.array([float(x) for x in args.cam.split(",")])
    target = np.array([float(x) for x in args.target.split(",")])
    px, py = (float(x) for x in args.pixel.split(","))
    W, H = (float(x) for x in args.size.split(","))
    ndcx = 2 * (px / W) - 1
    ndcy = 1 - 2 * (py / H)
    d = ray_dir(cam, target, ndcx, ndcy, args.aspect)

    scene = trimesh.load(args.scene, process=False)
    found = []
    for name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        lo, hi = mesh.bounds
        if not slab_hit(cam, d, lo, hi):
            continue
        for t in hits(mesh, cam, d):
            found.append((float(t), name))
    found.sort()

    print(f"\n  ray {args.label or args.pixel}: from {cam.tolist()} dir {[round(x,4) for x in d]}")
    if not found:
        print("  NOTHING HIT — the sight-line leaves the scene without meeting a surface")
    else:
        seen = []
        for t, name in found[:12]:
            print(f"    {t:8.2f} m   {name}")
            seen.append(name)
        print(f"  first surface: {found[0][1]} at {found[0][0]:.2f} m; {len(found)} crossings total")


if __name__ == "__main__":
    main()
