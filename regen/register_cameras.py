"""Register each panorama's camera against the LiDAR: its true position, height and heading.

The provider poses are off by metres (Mapillary's computed positions, never registered against this world), so a photograph
seen from the nominal camera shows its building at the wrong distance: a render from the nominal pose frames it bigger or
smaller than the photograph, and a rectified side photograph measures its wall too short or too long. Per panorama, the
whole 360-degree skyline (the highest non-sky elevation in each compass direction, from SegFormer) is compared with the
skyline the LiDAR surface predicts from a camera at a candidate position: everything the returns hold counts (buildings,
trees, the hills), thin poles and wires are opened out of both. A grid search over position, height and heading, then a
fine search with a small tilt, picks the camera; it is accepted only when it clearly beats the nominal pose, does not sit
at the search edge and no camera metres away fits nearly as well. Geometry only, location-free.

Usage (SFW_AREA = the area dir; run from work/):
  register_cameras.py [--poses poses.json] [--sides <area>/regen/sides/sides.json] [--images id ...]
                      [--out <area>/imagery/registered-cameras.json]
                      [--plots dir]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_opening, binary_opening
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frontage_crops_v15 as fc
from panorama_balance import road_gain

CELL = .5                                   # surface model cell (m)
AZ = np.radians(np.arange(0., 360., .5))    # compass directions sampled (from east, counter-clockwise)
EL = np.radians(np.arange(80., -10.01, -.25))   # elevations, top down
# distances sampled along each ray: fine near the camera, where a rooftop's angle changes fastest, coarser far out
RS = np.concatenate([np.arange(1., 20., .5), np.arange(20., 60., 1.), np.arange(60., 150., 2.)]).astype(np.float32)
EYE = fc.EYE_HEIGHT_M
CLIP = np.radians(6.)                       # a direction worse than this costs this much (trees changed, a bus passing)


def _rz(a):
    """Turns world vectors by `a` (radians) about the vertical, counter-clockwise from above."""
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _tilt(a, b):
    """Raises a direction at compass angle t by a cos t + b sin t (radians, first order): a rotation by hypot(a, b) about
    the level axis square to the direction it raises most."""
    tau = float(np.hypot(a, b))
    if tau < 1e-9: return np.eye(3)
    p0 = np.arctan2(b, a); k = np.array([np.sin(p0), -np.cos(p0), 0.]); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(tau) * K + (1 - np.cos(tau)) * K @ K


def apply_registration(cams, path, bias_fallback=True):
    """Camera records (frontage_crops_v15.load_cameras) moved to where register_cameras placed them: position, rotation
    (the provider's rotation turned by the fitted heading and tilt: a direction the provider called compass angle t is
    truly t + dyaw) and `heightOffsetM` (the fitted camera height less the fixed eye height, for the ground the
    rectifier stands the camera on). A panorama the skyline could not place moves by the placed ones' median shift (the
    provider's shared bias) when `bias_fallback`. Returns new records; `registered` says which way each was placed."""
    import json as _json
    reg = _json.load(open(path)); ok = {k: r for k, r in reg.items() if r.get('accepted')}
    bias = np.median([[r['dx'], r['dy']] for r in ok.values()], 0) if bias_fallback and len(ok) >= 10 else None
    out = {}
    for k, c in cams.items():
        r = ok.get(k); c = dict(c)
        if r is not None:
            ty = np.radians(r.get('tiltDeg') or [0., 0.])
            c['R'] = np.asarray(c['R']) @ _rz(-np.radians(r['dyawDeg'])) @ _tilt(ty[0], ty[1])
            c['xy'] = np.asarray(r['xy'], float); c['heightOffsetM'] = float(r['cameraHeightM']) - EYE; c['registered'] = 'skyline'
        elif bias is not None:
            c['xy'] = np.asarray(c['xy'], float) + bias; c['registered'] = 'bias'
        else: c['registered'] = False
        out[k] = c
    return out


def surfaces(frame):
    """(surface max grid, ground grid, origin, datum): the returns' highest point per 0.5 m cell with wires and thin poles
    opened out, and the ground under them (class-2 lowest per cell, gaps filled from the nearest measured cell)."""
    pts = np.load(Path(os.environ['SFW_AREA']) / 'data/points.npz')['points']
    t = Transformer.from_crs(3857, 32610, always_xy=True); x, y = t.transform(pts[:, 0], pts[:, 1])
    x = x - frame.cx; y = y - frame.cy; z = pts[:, 2]; c = pts[:, 3]
    datum = float(np.median(z[(c == 2) & (np.hypot(x, y) < 40)]))     # the world's datum (as the generator sets it)
    keep = (c != 7) & (c != 18); x, y, z, c = x[keep], y[keep], z[keep] - datum, c[keep]
    x0, y0 = x.min(), y.min(); ix = ((x - x0) / CELL).astype(int); iy = ((y - y0) / CELL).astype(int)
    G = np.full((ix.max() + 1, iy.max() + 1), np.inf, np.float32); g = c == 2; np.minimum.at(G, (ix[g], iy[g]), z[g])
    known = np.isfinite(G); _, (ii, jj) = distance_transform_edt(~known, return_indices=True); G = gaussian_filter(G[ii, jj], 2.)
    S = np.full(G.shape, -np.inf, np.float32); np.maximum.at(S, (ix, iy), z.astype(np.float32))
    S = np.maximum(S, G)                                                # empty cells stand at the ground
    S = grey_opening(S, size=(3, 3))                                    # a wire or a pole is narrower than 1.5 m
    return S, G, np.array([x0, y0]), datum


def lookup(A, o, px, py):
    i = ((px - o[0]) / CELL).astype(np.int32); j = ((py - o[1]) / CELL).astype(np.int32)
    ok = (i >= 0) & (j >= 0) & (i < A.shape[0]) & (j < A.shape[1])
    return np.where(ok, A[np.clip(i, 0, A.shape[0] - 1), np.clip(j, 0, A.shape[1] - 1)], -np.inf)


def predicted_heights(S, o, xy):
    """Surface height (m above datum) along every compass ray from each camera position: (n, azimuths, distances)."""
    r = RS; c, s = np.cos(AZ).astype(np.float32), np.sin(AZ).astype(np.float32)
    xy = np.asarray(xy, np.float32)
    px = xy[:, 0, None, None] + c[None, :, None] * r[None, None, :]; py = xy[:, 1, None, None] + s[None, :, None] * r[None, None, :]
    return lookup(S, o, px, py), r


def skyline_of(h, r, zc):
    """Predicted skyline elevation (rad) per compass direction from heights h (n, az, r) seen from camera heights zc (n,)."""
    return np.arctan(((h - np.asarray(zc, np.float32)[:, None, None]) / r[None, None, :]).max(2))   # the steepest slope: one arctan a direction


def observed(cam, seg):
    """The photograph's skyline elevation per compass direction under the nominal rotation, and which directions are usable
    (sky overhead, something standing below it)."""
    pano = np.asarray(fc.load_pano(cam['path']), np.float32); pano = np.clip(pano * road_gain(pano), 0, 255).astype(np.uint8)
    pano = np.asarray(Image.fromarray(pano).resize((2048, 1024), Image.BILINEAR))
    solid = seg.labels(pano) != 4                                       # 4 = sky
    solid = binary_opening(solid, structure=np.ones((5, 13), bool))     # thin poles, wires and sign posts out
    d = np.stack(np.meshgrid(AZ, EL, indexing='xy'), -1)                # (el, az, 2)
    dirs = np.stack([np.cos(d[..., 1]) * np.cos(d[..., 0]), np.cos(d[..., 1]) * np.sin(d[..., 0]), np.sin(d[..., 1])], -1)
    u, v = fc.world_dirs_to_pano_uv(dirs, cam['R'])
    M = solid[np.clip((v * 1024).astype(int), 0, 1023), (u * 2048).astype(int) % 2048]   # (el, az), top row first
    top = np.argmax(M, 0)                                               # first solid row from the top
    valid = ~M[0] & M.any(0)
    return np.where(valid, EL[top], np.nan), valid


def costs(O, P):
    """Mean clipped skyline error (rad) and directions compared, for every observed variant O (m, az) against every
    prediction P (n, az): (n, m) each."""
    E = np.abs(O[None] - P[:, None]); ok = np.isfinite(E)
    return np.where(ok, np.minimum(E, CLIP), 0.).sum(2) / np.maximum(ok.sum(2), 1), ok.sum(2)


def register(cam, obs, S, G, o):
    """Best camera for one panorama; heights in metres above the datum."""
    xy0 = cam['xy']
    # coarse: 1 m over +-14 m, three heights, headings +-6 degrees (whole half-degree steps: a roll of the skyline)
    D = np.arange(-14., 14.01, 1.); cand = np.array([[a, b] for a in D for b in D]); shifts = np.arange(-12, 13)
    O = np.stack([np.roll(obs, k) for k in shifts]); best = np.full(len(cand), 9.); arg = [None] * len(cand)
    for k in range(0, len(cand), 30):
        xy = xy0[None] + cand[k:k + 30]; h, r = predicted_heights(S, o, xy); gz = lookup(G, o, xy[:, 0], xy[:, 1])
        for dz in (-.8, 0., .8):
            C, N = costs(O, skyline_of(h, r, gz + EYE + dz))
            C = np.where(N > .3 * len(AZ), C, 9.)
            for n in range(len(xy)):
                m = int(np.argmin(C[n]))
                if C[n, m] < best[k + n]: best[k + n] = C[n, m]; arg[k + n] = (dz, int(shifts[m]))
    i0 = int(np.flatnonzero((cand == 0).all(1))[0]); h, r = predicted_heights(S, o, xy0[None]); g0 = lookup(G, o, xy0[None, 0], xy0[None, 1])
    c0 = float(costs(obs[None], skyline_of(h, r, g0 + EYE))[0][0, 0])
    b = int(np.argmin(best)); cb = float(best[b])
    if arg[b] is None:      # no camera sees enough skyline to compare (sky hidden by trees overhead, a washed-out sky)
        return {'nominalXY': [round(float(v), 3) for v in xy0], 'costDeg0': round(float(np.degrees(c0)), 3), 'accepted': False, 'why': 'too little skyline'}, None
    (bx, by), (bz, bs) = cand[b], arg[b]
    # a rival: the best camera more than 3 m from the winner (a second basin = the skyline cannot pin the camera)
    far = np.hypot(cand[:, 0] - bx, cand[:, 1] - by) > 3.; rival = float(best[far].min()) if far.any() else 9.
    # fine: 0.25 m, 0.2 m, 0.1 degree, then a tilt (the rotation's pitch and roll) fitted to what is left
    F = np.arange(-1.5, 1.51, .25); cand = np.array([[bx + a, by + b2] for a in F for b2 in F])
    A = np.arange(bs * .5 - 1., bs * .5 + 1.01, .1); deg = np.degrees(AZ)
    O = np.stack([np.interp((deg - a) % 360., deg, obs, period=360.) for a in A])
    fine = (9., 0, 0, 0., 0.)
    for k in range(0, len(cand), 30):
        xy = xy0[None] + cand[k:k + 30]; h, r = predicted_heights(S, o, xy); gz = lookup(G, o, xy[:, 0], xy[:, 1])
        for dz in np.arange(bz - .8, bz + .81, .2):
            C, N = costs(O, skyline_of(h, r, gz + EYE + dz)); C = np.where(N > .3 * len(AZ), C, 9.)
            n, m = np.unravel_index(int(np.argmin(C)), C.shape)
            if C[n, m] < fine[0]: fine = (float(C[n, m]), k + n, m, float(dz), int(N[n, m]))
    cf, fn, fm, fz, nv = fine; fx, fy = cand[fn]; fa = float(A[fm])
    xy = xy0[None] + cand[fn:fn + 1]; h, r = predicted_heights(S, o, xy); gz = lookup(G, o, xy[:, 0], xy[:, 1])
    P = skyline_of(h, r, gz + EYE + fz)[0]; ob = O[fm]
    res = ob - P; m = np.isfinite(res) & (np.abs(res) < CLIP)
    tilt = np.linalg.lstsq(np.c_[np.cos(AZ), np.sin(AZ)][m], res[m], rcond=None)[0] if m.sum() > 50 else np.zeros(2)
    tilt = np.clip(tilt, -np.radians(3), np.radians(3)); obt = ob - (np.cos(AZ) * tilt[0] + np.sin(AZ) * tilt[1])
    ct = float(costs(obt[None], P[None])[0][0, 0])
    if ct < cf: cf, ob = ct, obt
    else: tilt = np.zeros(2)
    edge = max(abs(fx), abs(fy)) > 13.
    out = {'nominalXY': [round(float(v), 3) for v in xy0], 'dx': round(float(fx), 2), 'dy': round(float(fy), 2),
           'cameraHeightM': round(EYE + fz, 2), 'dyawDeg': round(fa, 2), 'tiltDeg': [round(float(np.degrees(t)), 2) for t in tilt],
           'costDeg0': round(float(np.degrees(c0)), 3), 'costDeg': round(float(np.degrees(cf)), 3),
           'rivalCostDeg': round(float(np.degrees(rival)), 3), 'directions': nv, 'atEdge': bool(edge)}
    p = xy0 + np.array([fx, fy]); out['xy'] = [round(float(v), 3) for v in p]
    out['groundZ'] = round(float(lookup(G, o, p[None, 0], p[None, 1])[0]), 3); out['z'] = round(out['groundZ'] + out['cameraHeightM'], 3)
    better = cf < .75 * c0 or c0 - cf > np.radians(.4); unique = rival > 1.12 * cb
    out['accepted'] = bool(better and unique and not edge and nv > .3 * len(AZ))
    out['why'] = 'ok' if out['accepted'] else ', '.join(w for w, bad in (('no clear gain', not better), ('ambiguous', not unique), ('at the search edge', edge), ('too little skyline', nv <= .3 * len(AZ))) if bad)
    return out, (obs, P, ob, c0)


def plot(path, name, obs, pred_best, ob_best, S, G, o, cam):
    """Skyline elevation against compass direction: photograph (black: nominal heading, blue: registered) and the LiDAR's
    (red: from the nominal camera, green: from the registered one)."""
    from PIL import ImageDraw
    xy0 = cam['xy'][None]; h, r = predicted_heights(S, o, xy0); g0 = lookup(G, o, xy0[:, 0], xy0[:, 1])
    P0 = skyline_of(h, r, g0 + EYE)[0]; W, H = 1440, 420
    im = Image.new('RGB', (W, H + 30), (255, 255, 255)); d = ImageDraw.Draw(im)
    Y = lambda e: 30 + (80. - np.degrees(e)) / 90. * H
    for el in range(-10, 81, 10): d.line([(0, Y(np.radians(el))), (W, Y(np.radians(el)))], fill=(225, 225, 225)); d.text((2, Y(np.radians(el)) - 10), str(el), fill=(120, 120, 120))
    for series, col, line in ((obs, (0, 0, 0), False), (P0, (220, 40, 40), True), (ob_best, (40, 90, 230), False), (pred_best, (30, 160, 60), True)):
        pts = [(2 * i + 1, Y(e)) for i, e in enumerate(series) if np.isfinite(e)]
        if line:
            for a, b in zip(pts, pts[1:]): d.line([a, b], fill=col, width=1)
        else:
            for x, y in pts: d.point((x, y), fill=col)
    d.text((6, 6), name, fill=(0, 0, 0)); im.save(path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--poses'); ap.add_argument('--sides', help='a sides.json: every panorama its pieces were photographed from'); ap.add_argument('--images', nargs='*'); ap.add_argument('--out'); ap.add_argument('--plots')
    a = ap.parse_args()
    area = Path(os.environ['SFW_AREA']); out_path = Path(a.out or area / 'imagery/registered-cameras.json')
    ids = list(a.images or [])
    if a.poses: ids += [p['image'] for p in json.load(open(a.poses))['poses'] if p['image'] not in ids]
    if a.sides: ids += [v['image'] for p in json.load(open(a.sides))['pieces'] for v in p.get('views') or []]
    world, frame = fc.load_world(); cams = fc.load_cameras(frame); t0 = time.time()
    S, G, o, datum = surfaces(frame); print('surfaces', S.shape, round(time.time() - t0), 's', flush=True)
    done = json.load(open(out_path)) if out_path.exists() else {}
    ids = [i for i in dict.fromkeys(ids) if i in cams and i not in done]; print('to register', len(ids), 'already', len(done), flush=True)
    from facade_segment_v27 import Segmenter
    seg = Segmenter()
    if a.plots: Path(a.plots).mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(ids):
        t1 = time.time()
        try:
            obs, valid = observed(cams[i], seg); rec, fit = register(cams[i], obs, S, G, o)
        except Exception as e:      # one unreadable panorama is recorded and skipped, not the end of the run
            rec, fit = {'accepted': False, 'why': f'failed: {type(e).__name__}: {e}'[:200]}, None
        done[i] = rec
        if fit is None:
            json.dump(done, open(out_path, 'w'), indent=1); print(k + 1, len(ids), i, rec['why'], flush=True); continue
        obs, P, ob, c0 = fit
        if a.plots: plot(Path(a.plots) / f'{i}.png', f"{i}  shift {rec['dx']:+.1f},{rec['dy']:+.1f} m  heading {rec['dyawDeg']:+.1f}  cost {rec['costDeg0']}->{rec['costDeg']} deg  {rec['why']}", obs, P, ob, S, G, o, cams[i])
        json.dump(done, open(out_path, 'w'), indent=1)
        print(k + 1, len(ids), i, rec['dx'], rec['dy'], rec['dyawDeg'], rec['costDeg0'], '->', rec['costDeg'], rec['why'], round(time.time() - t1, 1), 's', flush=True)
