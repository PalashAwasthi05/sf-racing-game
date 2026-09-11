"""Frame scan that renders the way the viewer renders: back faces are culled.

scan_frame_v14.py accepted a hit from either side of a triangle, so a wall
wound to face into its own building counted as present. The viewer keeps
three.js FrontSide on 50 of 53 materials (glTF doubleSided is false), so that
wall is invisible on screen. This scan reports, per sight-line, the nearest
FRONT-facing surface (what the eye sees) beside the nearest surface of either
facing (what the geometry contains). The difference is the set of walls that
exist but cannot be seen from this camera.

Legend of the printed map (front-facing hits only):
  '.' nothing   '_' ground   'B' buildings   'f' facades   'r' roofs   'o' other
  '#' cell where the front-facing hit lies >8 m behind the either-facing hit:
      a surface stands there but is culled — an invisible wall.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np, trimesh

FOV_DEG = 58.0
CHUNK = 20_000

def basis(cam, target):
    f = np.asarray(target, float) - np.asarray(cam, float); f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    return f, right, np.cross(right, f)

def hits(mesh, origins, dirs):
    """Nearest t per ray, twice: any facing, and front facing (det>0 <=> D.N<0)."""
    tri = mesh.triangles
    best_any = np.full(len(origins), np.inf); best_front = np.full(len(origins), np.inf)
    for s in range(0, len(tri), CHUNK):
        t3 = tri[s:s + CHUNK]
        v0 = t3[:, 0][None]; e1 = (t3[:, 1] - t3[:, 0])[None]; e2 = (t3[:, 2] - t3[:, 0])[None]
        D = dirs[:, None, :]; P = np.cross(D, e2); det = (e1 * P).sum(-1)
        ok = np.abs(det) > 1e-9; inv = np.zeros_like(det); np.divide(1.0, det, out=inv, where=ok)
        T = origins[:, None, :] - v0; u = (T * P).sum(-1) * inv; Q = np.cross(T, e1)
        v = (D * Q).sum(-1) * inv; t = (e2 * Q).sum(-1) * inv
        good = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-3)
        ta = np.where(good, t, np.inf); best_any = np.minimum(best_any, ta.min(axis=1))
        tf = np.where(good & (det > 0), t, np.inf); best_front = np.minimum(best_front, tf.min(axis=1))
    return best_any, best_front

def sym(name):
    if not name: return "."
    if name.startswith(("terrain","roads","markings","sidewalks","cycleways","paths","water","landscape")): return "_"
    return {"buildings":"B","facades":"f","roofs":"r"}.get(name.split("__")[0], "o")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=Path("work/material-final-v13/candidate.glb"))
    ap.add_argument("--from", dest="cam", default="52,-34,2"); ap.add_argument("--at", dest="target", default="76,-12,20")
    ap.add_argument("--cols", type=int, default=64); ap.add_argument("--rows", type=int, default=36)
    ap.add_argument("--aspect", type=float, default=1280/720); ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--owner", default="", help="restrict to meshes whose name contains this")
    ap.add_argument("--max-distance", type=float, default=0, help="skip meshes whose bounding box is farther than this from the camera (0 = no limit)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cam = np.array([float(x) for x in a.cam.split(",")]); target = np.array([float(x) for x in a.target.split(",")])
    f, right, up = basis(cam, target); t = np.tan(np.radians(FOV_DEG)/2)
    xs = (np.arange(a.cols)+.5)/a.cols*2-1; ys = 1-(np.arange(a.rows)+.5)/a.rows*2
    dirs = np.array([ (f + right*(x*t*a.aspect) + up*(y*t)) for y in ys for x in xs]); dirs /= np.linalg.norm(dirs,axis=1,keepdims=True)
    origins = np.repeat(cam[None], len(dirs), axis=0)
    scene = trimesh.load(a.scene, process=False)
    best_any = np.full(len(dirs), np.inf); best_front = np.full(len(dirs), np.inf)
    own_any = np.array([""]*len(dirs), dtype=object); own_front = own_any.copy()
    for name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces): continue
        if a.owner and a.owner not in name: continue
        lo, hi = mesh.bounds; corners = np.array(np.meshgrid([lo[0],hi[0]],[lo[1],hi[1]],[lo[2],hi[2]])).T.reshape(-1,3)
        if ((corners-cam)@f).max() <= 0: continue
        if a.max_distance and np.linalg.norm(np.clip(cam, lo, hi) - cam) > a.max_distance: continue
        ta, tf = hits(mesh, origins, dirs)
        c = ta < best_any; best_any[c] = ta[c]; own_any[c] = name
        c = tf < best_front; best_front[c] = tf[c]; own_front[c] = name
    culled = np.isfinite(best_any) & (best_front > best_any + 8)
    gap_geom = ~np.isfinite(best_any) | np.array([sym(n)=="_" for n in own_any])
    rows_front = own_front.reshape(a.rows, a.cols); cul = culled.reshape(a.rows, a.cols)
    print(f"\n culling-aware frame scan {a.cols}x{a.rows} from {cam.tolist()} to {target.tolist()}")
    print(" '#' = a surface is there but faces away (culled)   '.'/'_' = nothing / ground (geometric gap)\n")
    for r in range(a.rows):
        if not a.quiet: print("  " + "".join("#" if cul[r,c] else sym(rows_front[r,c]) for c in range(a.cols)))
    print(f"\n cells: {len(dirs)}  culled-through: {int(culled.sum())}  geometric gap (nothing/ground either-facing): {int(gap_geom.sum())}")
    print(" culled surfaces (the wall that should have been seen), by mesh:")
    for n, k in Counter(own_any[culled]).most_common(12): print(f"   {k:4d}  {n}")
    print(" what is seen instead behind culled walls, by mesh:")
    for n, k in Counter(own_front[culled]).most_common(8): print(f"   {k:4d}  {n or '(nothing)'}")
    if a.report:
        a.report.write_text(json.dumps({"camera":{"from":cam.tolist(),"at":target.tolist()},"culledThrough":int(culled.sum()),"geometricGap":int(gap_geom.sum()),
            "culledSurfaces":Counter(own_any[culled]).most_common(),"seenInstead":Counter(own_front[culled]).most_common(),
            "distanceAny":np.where(np.isfinite(best_any),best_any,None).reshape(a.rows,a.cols).tolist(),"distanceFront":np.where(np.isfinite(best_front),best_front,None).reshape(a.rows,a.cols).tolist(),
            "ownerFront":rows_front.tolist(),"ownerAny":own_any.reshape(a.rows,a.cols).tolist()}, indent=None))
main()
