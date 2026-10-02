"""Stage 1c: correct each side photograph's placement on its wall against the LiDAR.

A camera's provider pose is off by metres, so a rectified photograph shows the wall shifted and scaled. Per view the
skyline over the wall (topmost non-sky pixel per column, from SegFormer; columns topped by a tree or pole skipped) is
compared with the skyline the LiDAR predicts from the same nominal camera (building returns only: points inside the
footprints, as a 0.5 m height grid). A shift along the wall, a shift up, and a scale about the camera fit the two; the
fit is kept only when it clearly beats no correction and the skyline has enough shape to fix the shift. Geometry only."""
import argparse, json, pathlib, sys, time, os
import numpy as np
from pyproj import Transformer
from shapely import contains_xy
from shapely.ops import unary_union
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import frontage_crops_v15 as fc, auto_read_frontages_v24 as ar
from panorama_balance import road_gain

PPM, EXT_U, EXT_Z, CELL = 12., 6., 12., .5

def dsm(frame):
    pts = np.load(fc.DATA_DIR / 'points.npz')['points']
    web = Transformer.from_crs(3857, 32610, always_xy=True); x, y = web.transform(pts[:, 0], pts[:, 1]); x = x - frame.cx; y = y - frame.cy; z = pts[:, 2]; c = pts[:, 3]
    g = (c == 2) & (z > -5) & (z < 20); base = float(np.median(z[g & (np.hypot(x, y) < 40)])); z = z - base
    fp = unary_union(list(fc.load_footprints(frame).values())).buffer(.3)
    keep = (c == 1); keep[keep] = contains_xy(fp, x[keep], y[keep])
    x, y, z = x[keep], y[keep], z[keep]
    x0, y0 = x.min(), y.min(); ix = ((x - x0) / CELL).astype(int); iy = ((y - y0) / CELL).astype(int)
    G = np.full((ix.max() + 1, iy.max() + 1), -np.inf, np.float32); np.maximum.at(G, (ix, iy), z.astype(np.float32))
    return G, x0, y0

def predicted(G, x0, y0, C, eye, W, dist):
    """Skyline height on the wall plane at each wall point W (n,2) seen from C at eye height."""
    d = (W - C) / dist[:, None]; s = np.arange(-1., 140., CELL / 2)
    P = C[None, None, :] + d[:, None, :] * (dist[:, None, None] + s[None, :, None])
    ix = ((P[..., 0] - x0) / CELL).astype(int); iy = ((P[..., 1] - y0) / CELL).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < G.shape[0]) & (iy < G.shape[1])
    h = np.where(ok, G[np.clip(ix, 0, G.shape[0] - 1), np.clip(iy, 0, G.shape[1] - 1)], -np.inf)
    t = (h - eye) / (dist[:, None] + s[None, :])
    tm = t.max(1); return np.where(np.isfinite(tm), eye + tm * dist, np.nan)

def fit(u, zi, zp, uc, eye):
    """Best (dx, dz, s) mapping true (u, zp) to image (uc + (u-uc)/s + dx, eye + (zp-eye)/s + dz) onto observed zi(u_img)."""
    good = np.isfinite(zp)
    if good.sum() < 20: return None
    def cost(dx, dz, s):
        ui = uc + (u - uc) / s + dx; zq = eye + (zp - eye) / s + dz
        obs = np.interp(ui, u, zi, left=np.nan, right=np.nan)
        e = np.abs(obs - zq); e = e[np.isfinite(e) & good]
        if len(e) < 15: return 9., 0
        return float(np.mean(np.minimum(e, 2.))), len(e)
    c0, n0 = cost(0, 0, 1.)
    best = (c0, 0., 0., 1.)
    for s in np.arange(.9, 1.101, .02):
        for dx in np.arange(-5, 5.01, .15):
            for dz in np.arange(-2.5, 2.51, .1):
                c, n = cost(dx, dz, s)
                if c < best[0] and n >= .6 * n0: best = (c, dx, dz, s)
    c, dx, dz, s = best
    for s2 in np.arange(s - .02, s + .021, .005):          # refine
        for dx2 in np.arange(dx - .15, dx + .151, .03):
            for dz2 in np.arange(dz - .1, dz + .101, .02):
                cc, n = cost(dx2, dz2, s2)
                if cc < best[0] and n >= .6 * n0: best = (cc, dx2, dz2, s2)
    # the height alone (no shift along, no scale): for a level skyline that cannot fix the rest
    zonly = min(((cost(0, dz, 1.)[0], dz) for dz in np.arange(-2.5, 2.51, .05)), key=lambda q: q[0])
    c, dx, dz, s = best
    atbound = abs(dx) > 4.9 or abs(dz) > 2.45 or s < .905 or s > 1.095
    return {'cost0': round(c0, 3), 'zOnlyCost': round(float(zonly[0]), 3), 'zOnlyDz': round(float(zonly[1]), 3), 'atBound': bool(atbound), 'cost': round(best[0], 3), 'dx': round(float(best[1]), 3), 'dz': round(float(best[2]), 3), 's': round(float(best[3]), 4),
            'columns': int(n0), 'shape': round(float(np.nanstd(zp)), 2)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.environ.get('REGEN_ROOT', 'regen/v1')); ap.add_argument('--min-area', type=float, default=30.); ap.add_argument('--ids', nargs='*'); ap.add_argument('--out', default='alignment.json'); a = ap.parse_args()
    root = pathlib.Path(a.root); P = [p for p in json.load(open(root / 'sides/sides.json'))['pieces'] if p['views'] and p['lengthM'] * p['heightM'] >= a.min_area and (not a.ids or p['id'] in a.ids)]
    world, frame = fc.load_world(); cams = fc.load_cameras(frame); t0 = time.time()
    G, x0, y0 = dsm(frame); print('dsm', G.shape, round(time.time() - t0), 's', flush=True)
    from facade_segment_v27 import Segmenter
    seg = Segmenter(); out = {}; pano = {}
    for k, p in enumerate(P):
        n = np.asarray(p['normal'], float); t = np.array([-n[1], n[0]]); u0, u1 = p['frameU']; L = u1 - u0; H = p['heightM']; z0 = p['groundZ']
        mid = n * p['planeV'] + t * (u0 + u1) / 2
        ext = {'midpoint': mid.tolist(), 'normal': p['normal'], 'lengthM': L + 2 * EXT_U, 'heightM': H + EXT_Z}
        for vi, v in enumerate(p['views']):
            cam = cams[v['image']]
            if v['image'] not in pano:
                if len(pano) > 2: pano.pop(next(iter(pano)))
                q = np.asarray(fc.load_pano(cam['path']), dtype=np.float32); pano[v['image']] = np.clip(q * road_gain(q), 0, 255).astype(np.uint8)
            ar.PPM = PPM; img, Hh, Lh = ar.rectify(ext, cam, z0, pano[v['image']])
            codes = seg.labels((img * 255).astype(np.uint8))
            h, w = codes.shape; sky = codes == 4
            top = np.argmax(~sky, 0)                          # first non-sky row from the top
            valid = sky[0] & (~sky).any(0)                    # sky at the frame top, building below it
            hit = codes[np.clip(top, 0, h - 1), np.arange(w)]; valid &= (hit != 3)
            u_img = (np.arange(w) + .5) / PPM - EXT_U          # piece frame: 0 at the piece's left end
            z_img = np.where(valid, Hh - (top + .5) / PPM, np.nan)
            Wp = (mid[None, :] + t[None, :] * (u_img[:, None] - L / 2))
            C = np.asarray(cam['xy'], float); dist = np.linalg.norm(Wp - C, axis=1)
            eye_abs = z0 + fc.EYE_HEIGHT_M
            zp = predicted(G, x0, y0, C, eye_abs, Wp, dist) - z0
            uc = float((C - mid) @ t + L / 2)
            f = fit(u_img, z_img, zp, uc, fc.EYE_HEIGHT_M)
            if f: f.update(uc=round(uc, 2), eye=fc.EYE_HEIGHT_M, validShare=round(float(valid.mean()), 2))
            out.setdefault(p['id'], {})[str(vi)] = f
            if a.ids: np.savez(root / f"align_{p['id']}_{vi}.npz", u=u_img, zi=z_img, zp=zp, img=img)
        if k % 20 == 0: print(k, len(P), round(time.time() - t0), 's', flush=True); json.dump(out, open(root / a.out, 'w'))
    json.dump(out, open(root / a.out, 'w')); print('done', round(time.time() - t0), 's')
