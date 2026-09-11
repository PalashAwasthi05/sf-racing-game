"""See-through audit over every envelope street view, rendered the way the viewer renders.

For each street view the pilot generates (one per envelope frontage and per grammar profile) plus any
extra cameras given, cast a ray grid and report two counts per view:

  culledThrough  rays whose nearest surface of either facing is >8 m nearer than the nearest FRONT-facing
                 surface: a wall stands there but faces into its own building and is culled on screen.
  seeThrough     rays that pass through the target building's footprint prism (footprint x [ground,
                 ground+height]) and whose first visible surface lies beyond where they leave that prism,
                 or nowhere: the eye reaches the far side or the sky through the building.

Both counts are mechanism-independent measurements of the scene; nothing is selected by name. Run on the
previous and the regenerated final scenes with identical cameras to compare.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, trimesh
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from shapely import contains_xy

FOV_DEG = 58.0; CHUNK = 20_000

def basis(cam, target):
    f = np.asarray(target, float) - np.asarray(cam, float); f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    return f, right, np.cross(right, f)

def hits(mesh, origins, dirs, cam=None, f=None, cone=None, max_distance=None):
    tri = mesh.triangles
    if cam is not None:
        # Frustum pre-filter: keep triangles whose centroid lies inside the view cone (with a margin for size).
        c = tri.mean(axis=1); rel = c - cam; dist = np.linalg.norm(rel, axis=1); size = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1) + np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
        ang = np.arccos(np.clip((rel @ f) / np.maximum(dist, 1e-9), -1, 1)); keep = (ang < cone + np.arctan2(size, np.maximum(dist, 1e-3))) & (dist - size < max_distance)
        tri = tri[keep]
    best_any = np.full(len(origins), np.inf); best_front = np.full(len(origins), np.inf)
    if not len(tri): return best_any, best_front
    for s in range(0, len(tri), CHUNK):
        t3 = tri[s:s + CHUNK]; v0 = t3[:, 0][None]; e1 = (t3[:, 1] - t3[:, 0])[None]; e2 = (t3[:, 2] - t3[:, 0])[None]
        D = dirs[:, None, :]; P = np.cross(D, e2); det = (e1 * P).sum(-1)
        ok = np.abs(det) > 1e-9; inv = np.zeros_like(det); np.divide(1.0, det, out=inv, where=ok)
        T = origins[:, None, :] - v0; u = (T * P).sum(-1) * inv; Q = np.cross(T, e1); v = (D * Q).sum(-1) * inv; t = (e2 * Q).sum(-1) * inv
        good = ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-3)
        best_any = np.minimum(best_any, np.where(good, t, np.inf).min(axis=1)); best_front = np.minimum(best_front, np.where(good & (det > 0), t, np.inf).min(axis=1))
    return best_any, best_front

def scan(meshes, cam, target, cols, rows, aspect, max_distance):
    f, right, up = basis(cam, target); t = np.tan(np.radians(FOV_DEG) / 2)
    xs = (np.arange(cols) + .5) / cols * 2 - 1; ys = 1 - (np.arange(rows) + .5) / rows * 2
    dirs = np.array([(f + right * (x * t * aspect) + up * (y * t)) for y in ys for x in xs]); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.repeat(cam[None], len(dirs), axis=0)
    best_any = np.full(len(dirs), np.inf); best_front = np.full(len(dirs), np.inf); own = np.array([""] * len(dirs), dtype=object)
    for name, mesh, lo, hi in meshes:
        corners = np.array(np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]])).T.reshape(-1, 3)
        if ((corners - cam) @ f).max() <= 0 or np.linalg.norm(np.clip(cam, lo, hi) - cam) > max_distance: continue
        ta, tf = hits(mesh, origins, dirs, cam, f, np.arctan(t * np.hypot(aspect, 1.0)), max_distance)
        c = ta < best_any; best_any[c] = ta[c]; own[c] = name
        best_front = np.minimum(best_front, tf)
    return dirs, best_any, best_front, own

def see_through(cam, dirs, best_front, footprint, z0, top, step=.5, reach=220.):
    ts = np.arange(step, reach, step); pts = cam[None, None, :] + dirs[:, None, :] * ts[None, :, None]
    inside = contains_xy(footprint, pts[..., 0].ravel(), pts[..., 1].ravel()).reshape(pts.shape[:2]) & (pts[..., 2] > z0 - 1) & (pts[..., 2] < top + 1)
    enters = inside.any(axis=1); t_exit = np.where(enters, (inside * ts[None, :]).max(axis=1), np.nan)
    return enters & (best_front > t_exit + 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--world", required=True); ap.add_argument("--footprints", default="work/support-footprints-v11.json")
    ap.add_argument("--extra", action="append", default=[], help="label|fromx,fromy,fromz|atx,aty,atz|buildingId")
    ap.add_argument("--cols", type=int, default=40); ap.add_argument("--rows", type=int, default=23); ap.add_argument("--aspect", type=float, default=1280 / 720)
    ap.add_argument("--max-distance", type=float, default=260); ap.add_argument("--report", required=True)
    a = ap.parse_args(); start = time.time()
    world = json.loads(Path(a.world).read_text()); feats = {b['id']: b for b in world['buildings']}
    grid = world['grid']; ground = np.asarray(world['ground'], float).reshape(grid['n'], grid['n'])
    def eye(x, y):
        # Street cameras stand 1.8 m above the pilot's own ground raster, like the viewer's street mode.
        ix = int(np.clip(round((x - grid['min']) / grid['step']), 0, grid['n'] - 1)); iy = int(np.clip(round((y - grid['min']) / grid['step']), 0, grid['n'] - 1))
        return float(ground[iy, ix]) + 1.8
    fps = {r['id']: (shape(r['geometry']) if 'geometry' in r else unary_union([Polygon(rg) for rg in r['rings']])).buffer(0) for r in json.loads(Path(a.footprints).read_text())}
    profiles = json.loads(Path('work/pipeline/envelope-observations.json').read_text())['profiles']
    by_profile = {}
    for p in profiles:
        m = p['match']; by_profile[p['id']] = m.get('featureId') or next((b['id'] for b in world['buildings'] if b.get('id', '').endswith(str(m.get('osmWayId')))), None)
    views = []
    for v in world.get('streetViews', []):
        pid = v['label'].split(' / ')[-1]; bid = by_profile.get(pid)
        if bid is None:
            gid = pid; bid = next((b['id'] for b in world['buildings'] if b['id'] == 'osm_' + str(next((p['match']['osmWayId'] for p in json.loads(Path('work/pipeline/facade-profiles.json').read_text())['profiles'] if p['id'] == gid), ''))), None)
        views.append((v['label'], np.array([*v['position'][:2], eye(*v['position'][:2])]), np.array(v['lookAt'], float), bid))
    for e in a.extra:
        label, fr, at, bid = e.split('|'); views.append((label, np.array([float(x) for x in fr.split(',')]), np.array([float(x) for x in at.split(',')]), bid))
    scene = trimesh.load(a.scene, process=False)
    meshes = [(n, m, *m.bounds) for n, m in scene.geometry.items() if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    rows = []
    for label, cam, target, bid in views:
        dirs, best_any, best_front, own = scan(meshes, cam, target, a.cols, a.rows, a.aspect, a.max_distance)
        culled = np.isfinite(best_any) & (best_front > best_any + 8)
        row = {"view": label, "building": bid, "from": cam.round(2).tolist(), "at": target.round(2).tolist(), "rays": len(dirs), "culledThrough": int(culled.sum()),
               "culledMeshes": sorted({n for n in own[culled]})[:6]}
        if bid in fps and bid in feats:
            z0 = feats[bid]['center'][2]; top = z0 + feats[bid]['height']
            st = see_through(cam, dirs, best_front, fps[bid], z0, top)
            entering = see_through(cam, dirs, np.full(len(dirs), np.inf), fps[bid], z0, top)   # every ray that crosses the prism
            row.update(seeThrough=int(st.sum()), raysEnteringBuilding=int(entering.sum()))
        rows.append(row); print(f"  {label[:58]:58s} culled {row['culledThrough']:4d}  seeThrough {row.get('seeThrough','-'):>4}  of {row.get('raysEnteringBuilding','-'):>4} entering", flush=True)
    totals = {"culledThrough": sum(r['culledThrough'] for r in rows), "seeThrough": sum(r.get('seeThrough', 0) for r in rows), "views": len(rows), "elapsedSeconds": round(time.time() - start, 1)}
    print("TOTAL", json.dumps(totals))
    Path(a.report).write_text(json.dumps({"scene": a.scene, "world": a.world, "grid": [a.cols, a.rows], "totals": totals, "views": rows}, indent=1))
main()
