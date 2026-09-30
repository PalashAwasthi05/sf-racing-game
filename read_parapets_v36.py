"""Measured parapet outlines of photographed frontages (stepped fronts, false fronts, shaped gables).

LiDAR at this spacing reads a thin parapet as a straight or smoothed top: a stepped false front came out as a rake cut
by the roof planes. Its outline is in the photographs. For each photographed wall (a hand or automatic reading's wall of
at least 8 m), up to five photographs are rectified onto the wall plane with 4 m of headroom above it, only from registered
panorama poses (review/panorama-registration.json; register_panoramas.py): a provider pose moves the top by metres, and in every 0.25 m column the wall's top is the
highest row with its own material under it: at least 60 % of the metre below matching the wall's reading colour
(colour_agreement.same_material: hue and chroma kept, lightness free). Sky, glass, a set-back block or a neighbour above
the parapet do not match. Columns are kept where two or more photographs lie within 0.4 m of the photographs' median (the median of those).

Scale: rectification onto the footprint plane puts heights off by the plane's depth error (the metre grid of a camera
metres off the wall), so the outline is scaled to the LiDAR along the same wall (the 90th percentile of the building
surface raster 0.5-1.5 m inside it, per column): one factor for the whole outline, the 95th percentile of the ratios (a
false front stands over the roof it hides; the median left the roof's rake above the steps), within 0.75-1.25
(the "two cues" of the frontage readers). The outline is then cut into steps: runs of columns within 0.2 m of their
median, runs under 0.5 m merged into a neighbour. A wall is written only when its measured top is not flat (0.8 m or
more between its lowest and highest step) and stands above the LiDAR top somewhere by 0.3 m (what the raster misses).

Writes work/pipeline/frontage-parapets-v36.json: per wall its world end points a, b (the frame's x=0 and x=L), the
ground z0 and the top as [metres along a->b, height above z0] step corners. The generator cuts that wall to the outline
(facades.measured_parapet). Carries no location.
Usage: read_parapets_v36.py --candidate work/<dir> [--only <source id> ...] [--out work/pipeline/frontage-parapets-v36.json]"""
import argparse, json, pathlib, re, sys
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
import frontage_evidence_v34 as E
from pipeline.colour_agreement import lab, MAX_DL, GREY_C, MAX_DH, MAX_C_RATIO

ROOT = pathlib.Path(__file__).resolve().parent
PPM, HEADROOM_M, COL_M = 20, 4., .25
MIN_WALL_M, MIN_RISE_M, ABOVE_LIDAR_M, AGREE_M = 8., .8, .3, .4


def lab_img(img):
    c = np.clip(img, 0, 255) / 255.; c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    X = c @ np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]]) / np.array([.9505, 1., 1.089])
    f = np.where(X > .008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def wall_like(img, rgb):
    """Pixels that can be the wall's material under other light (colour_agreement.same_material, vectorised)."""
    L = lab_img(img.astype(np.float64)); w = lab(rgb)
    C = np.hypot(L[..., 1], L[..., 2]); Cw = float(np.hypot(w[1], w[2]))
    ok = np.abs(L[..., 0] - w[0]) <= MAX_DL
    if Cw < GREY_C: return ok & (C < GREY_C + 6)
    h = np.degrees(np.arctan2(L[..., 2], L[..., 1])); hw = float(np.degrees(np.arctan2(w[2], w[1])))
    dh = np.abs((h - hw + 180) % 360 - 180)
    return ok & (C > 6) & (dh <= MAX_DH) & (np.maximum(C, Cw) <= MAX_C_RATIO * np.minimum(C, Cw) + 1e-6)


def rectify(wall, eye, R, pano, z0, H):
    m = np.asarray(wall['midpoint'], float); n = np.asarray(wall['normal'], float)[:2]; n /= np.linalg.norm(n); t = np.array([-n[1], n[0]])
    L = float(wall['lengthM']); Wp = max(8, int(L * PPM)); Hp = int(H * PPM)
    xs = (np.arange(Wp) + .5) / PPM - L / 2; zs = H - (np.arange(Hp) + .5) / PPM
    P = np.empty((Hp, Wp, 3)); P[..., 0] = (m[0] + t[0] * xs)[None] + n[0] * .05; P[..., 1] = (m[1] + t[1] * xs)[None] + n[1] * .05; P[..., 2] = (z0 + zs)[:, None]
    u, v = fc.world_dirs_to_pano_uv(P - eye, R); ph, pw = pano.shape[:2]
    return np.stack([map_coordinates(pano[..., k], [v * (ph - 1), u * (pw - 1)], order=1, mode='wrap') for k in range(3)], -1), m - t * L / 2, m + t * L / 2, t


def column_tops(img, rgb, H, lo_m):
    B = wall_like(img, rgb); step = int(COL_M * PPM); out = []
    for c0 in range(0, B.shape[1], step):
        f = B[:, c0:c0 + step].mean(1); run = np.convolve(f, np.ones(PPM) / PPM, 'valid')
        k = np.flatnonzero(run[: max(1, int((H - lo_m) * PPM))] >= .6)
        out.append(H - (k[0] + .5) / PPM if len(k) else np.nan)
    return np.array(out)


def steps(z, min_len=2):
    runs, start = [], 0
    for i in range(1, len(z) + 1):
        if i == len(z) or abs(z[i] - np.median(z[start:i])) > .2: runs.append([start, i, float(np.median(z[start:i]))]); start = i
    merged = []
    for r in runs:
        if merged and r[1] - r[0] < min_len: merged[-1][1] = r[1]
        else: merged.append(r)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_len: merged[1][0] = 0; merged.pop(0)
    return merged


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True); ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', type=pathlib.Path, default=ROOT / 'pipeline/frontage-parapets-v36.json'); ap.add_argument('--max-photos', type=int, default=5)
    a = ap.parse_args(); only = set(a.only) if a.only else None
    world = json.load(open(a.candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}
    ev = json.load(open(a.candidate / 'public/facade-evidence.json'))
    reg_p = ROOT / 'review/panorama-registration.json'; reg = json.load(open(reg_p))['panoramas'] if reg_p.exists() else {}
    cams = fc.load_cameras(frame); occ = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame)
    surf = E.building_surface(world, frame); top, (x0, y1, st) = surf
    profiles = {}
    for f in sorted((ROOT / 'pipeline').glob('frontage-observations*.json')): profiles.update({p['id']: p for p in json.load(open(f)).get('profiles', [])})
    out = {p['key']: p for p in (json.load(open(a.out)).get('walls', []) if a.out.exists() else []) if only is None or re.sub(r'_p\d+$', '', p['buildingId']) not in only}
    for w in ev['walls']:
        bid = w.get('buildingId'); src = re.sub(r'_p\d+$', '', bid or '')
        if only is not None and src not in only: continue
        if w.get('lengthM', 0) < MIN_WALL_M or not (w.get('grammarId') or w.get('archetype') in profiles): continue
        prof = profiles.get(w.get('grammarId') or w.get('archetype')); rgb = (prof or {}).get('materialEvidence', {}).get('colorRgb')
        if not rgb or bid not in buildings: continue
        z0 = float(buildings[bid]['center'][2]); H = float(w['heightM']) + HEADROOM_M
        cands = fc.wall_candidates(w, cams, occ, trees, fc.MAX_DISTANCE_M)[: a.max_photos * 2]
        tops, used = [], []
        for c in cands:
            if len(used) >= a.max_photos: break
            cam = cams[c['image']]; r = reg.get(c['image'])
            if r is None: continue          # an outline needs a registered pose: a coarse one moves the top by metres
            eye, R = (np.array([*r['xy'], r['z']]), Rotation.from_rotvec(r['rotationAngleAxis']).as_matrix()) if r else (np.array([*cam['xy'], z0 + fc.EYE_HEIGHT_M]), cam['R'])
            img, pa, pb, t = rectify(w, eye, R, np.asarray(fc.load_pano(cam['path'])).astype(np.float32), z0, H)
            tops.append(column_tops(img, rgb, H, .5 * float(w['heightM']))); used.append((c['image'], bool(r)))
        if len(tops) < 2: print('skip', bid, 'fewer than two photographs', flush=True); continue
        # a column counts where two or more photographs lie within 0.4 m of their median (one photograph with a tree or
        # pole across the top does not veto the others); its height is the median of those that agree
        T = np.array(tops); med0 = np.nanmedian(T, 0); near_ = np.abs(T - med0[None]) <= AGREE_M
        ok = np.isfinite(med0) & (near_.sum(0) >= 2)
        med = np.array([np.median(T[near_[:, i], i]) if ok[i] else np.nan for i in range(T.shape[1])])
        if ok.mean() < .6: print('skip', bid, 'photographs agree on', round(float(ok.mean()), 2), 'of the columns', flush=True); continue
        xs = (np.arange(len(med)) + .5) * COL_M; med = np.interp(xs, xs[ok], med[ok])
        # LiDAR along the wall, 0.5-1.5 m inside it
        nrm = np.asarray(w['normal'], float)[:2]; nrm /= np.linalg.norm(nrm); lid = []
        for s in xs:
            vals = []
            for inset in (.5, 1., 1.5):
                x, y = pa + t * s - nrm * inset; cc = int((x - x0) // st); rr = int((y1 - y) // st)
                if 0 <= rr < top.shape[0] and 0 <= cc < top.shape[1] and np.isfinite(top[rr, cc]): vals.append(top[rr, cc] - z0)
            lid.append(np.quantile(vals, .9) if vals else np.nan)
        lid = np.array(lid); both = np.isfinite(lid) & ok
        if both.sum() < 8: print('skip', bid, 'no LiDAR along the wall', flush=True); continue
        # a false front stands above the roof it hides: the scale that puts the outline over the LiDAR returns along the
        # wall (95th percentile of the ratios; the median sat the steps under the roof's rake between them)
        scale = float(np.clip(np.quantile(lid[both] / med[both], .95), .75, 1.25)); z = med * scale
        runs = steps(z, int(round(.5 / COL_M)))
        zr = np.array([r[2] for r in runs])
        if zr.max() - zr.min() < MIN_RISE_M or not np.nanmax(z - np.nan_to_num(lid, nan=-9)) > ABOVE_LIDAR_M: print('skip', bid, 'flat or at the LiDAR top', round(float(zr.max() - zr.min()), 2), flush=True); continue
        top_pts = []
        for r in runs: top_pts += [[round(r[0] * COL_M, 2), round(r[2], 2)], [round(min(r[1] * COL_M, float(w['lengthM'])), 2), round(r[2], 2)]]
        key = f"{bid}|{round(w['midpoint'][0], 1)}|{round(w['midpoint'][1], 1)}"
        out[key] = {'key': key, 'buildingId': bid, 'readingId': prof['id'], 'a': [round(float(v), 3) for v in pa], 'b': [round(float(v), 3) for v in pb],
                    'normal': [round(float(v), 4) for v in nrm], 'z0': round(z0, 3), 'top': top_pts, 'scale': round(scale, 3), 'agreedShare': round(float(ok.mean()), 2),
                    'photographs': [{'image': i, 'registered': g} for i, g in used], 'evidence': 'photographed wall top, LiDAR-scaled'}
        print(key, 'steps', len(runs), 'rise %.2f' % (zr.max() - zr.min()), 'scale', round(scale, 3), 'photos', len(used), flush=True)
    a.out.write_text(json.dumps({'schema': 1, 'method': __doc__.split('\n')[0], 'walls': list(out.values())}, indent=1)); print('wrote', a.out, len(out))


if __name__ == '__main__':
    main()
