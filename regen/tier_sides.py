"""Stage 1 for set-back tiers: the sides of walls that stand on a lower roof (zBaseM above the building's ground).

sides.py rectified every side from the building's ground, so a 5 m tier wall standing on a 22 m podium was read off
the bottom 5 m of the street front. Here a tier side is the run of one building's tier walls facing the same way in
one plane with the same foot (within 1.5 m); it is rectified from its foot (rectify's `base`). A street camera sees a
tier only over the podium's parapet: cameras are ranked by the share of the wall that LiDAR sight lines reach (rays
from the camera's eye to a grid on the wall, through the building-return height grid, stopping 1.5 m short of the
wall). Ids are <building>__t<side>p<piece>; the pieces are written as their own sides file. Geometric rules only."""
import argparse, json, math, pathlib, sys, time
import numpy as np
from PIL import Image, ImageDraw
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import frontage_crops_v15 as fc, auto_read_frontages_v24 as ar, align
from auto_read_frontages_v24 import rectify
from panorama_balance import road_gain

ANGLE_DEG, PLANE_M, BASE_M, PIECE_M, PPM_MAX, MAX_PX, MIN_SEEN = 12., 1.5, 1.5, 25., 45., 1400, .35

def sides_of(walls):
    out = []
    for w in sorted(walls, key=lambda w: -w['lengthM']):
        n = np.asarray(w['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]]); m = np.asarray(w['midpoint'], float)
        zb = float(w['zBaseM'])
        for s in out:
            if float(n @ s['n']) > math.cos(math.radians(ANGLE_DEG)) and abs(float(m @ s['n']) - s['v']) < PLANE_M and abs(zb - s['zb']) < BASE_M:
                u = float(m @ s['t']); s['u0'] = min(s['u0'], u - w['lengthM'] / 2); s['u1'] = max(s['u1'], u + w['lengthM'] / 2)
                s['h'] = max(s['h'], w['heightM']); s['walls'].append(w); break
        else:
            u = float(m @ t)
            out.append({'n': n, 't': t, 'v': float(m @ n), 'zb': zb, 'u0': u - w['lengthM'] / 2, 'u1': u + w['lengthM'] / 2, 'h': w['heightM'], 'walls': [w]})
    return out

def seen_share(G, x0, y0, cam_xy, piece, base):
    """Share of a 7 x 5 grid on the wall that the camera's eye sees over the LiDAR building returns. Heights are taken
    relative to the podium roof just in front of the wall, which stands at the tier's foot."""
    m, n, t, a, b = fc.wall_geometry(piece)
    def h(p):
        i, j = int((p[0] - x0) / align.CELL), int((p[1] - y0) / align.CELL)
        return float(G[i, j]) if 0 <= i < G.shape[0] and 0 <= j < G.shape[1] else -np.inf
    front = [h(m + n * k + t * s) for k in (1., 2., 3.) for s in (-2., 0., 2.)]
    front = [f for f in front if np.isfinite(f)]
    if not front: return 0.
    ground = float(np.median(front)) - base                       # the DSM height of the building's ground
    eye = np.r_[cam_xy, ground + fc.EYE_HEIGHT_M]
    su, sz = np.meshgrid(np.linspace(.1, .9, 7), np.linspace(.1, .9, 5)); su, sz = su.ravel(), sz.ravel()
    P = np.c_[a[None] + (b - a)[None] * su[:, None] + n * .3, ground + base + sz * piece['heightM']]
    d = P - eye; L = np.linalg.norm(d[:, :2], axis=1); f = np.linspace(0, 1, 400)[None, :]
    Q = eye[None, None] + d[:, None, :] * f[..., None]                      # 400 samples per ray
    near = (f * L[:, None] < 1.) | (f * L[:, None] > L[:, None] - 1.5)        # skip the camera's own cell and the wall's foot
    i = ((Q[..., 0] - x0) / align.CELL).astype(int); j = ((Q[..., 1] - y0) / align.CELL).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < G.shape[0]) & (j < G.shape[1])
    hg = np.where(ok, G[np.clip(i, 0, G.shape[0] - 1), np.clip(j, 0, G.shape[1] - 1)], -np.inf)
    blocked = ((hg > Q[..., 2]) & ~near).any(1)
    return float((~blocked)[L >= 2].mean()) if (L >= 2).any() else 0.

ap = argparse.ArgumentParser(); ap.add_argument('--evidence', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
out = pathlib.Path(a.out); (out / 'img').mkdir(parents=True, exist_ok=True); t0 = time.time()
world, frame = fc.load_world(); buildings = {b['id']: b for b in world['buildings']}
walls = [w for w in json.load(open(a.evidence))['facadeWalls'] if w['buildingId'] in buildings and (w.get('zBaseM') or 0) > .5]
occ = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = {k: c for k, c in fc.load_cameras(frame).items() if c['exists']}
G, x0, y0 = align.dsm(frame); print('dsm', G.shape, round(time.time() - t0), 's', flush=True)

class FeatureOcc(fc.Occluders):
    """2D occlusion that lets sight lines cross the tier's own building (its podium): only the 3D test can say whether
    the podium hides the wall."""
    def __init__(self, base, bid):
        self.ids, self.polys, self.tree = base.ids, base.polys, base.tree; f = bid.split('_p')[0]
        self.own = {k for k, i in enumerate(base.ids) if str(i).split('_p')[0] == f}
    def own_indices(self, inside_point): return fc.Occluders.own_indices(self, inside_point) | self.own

by_b = {}
for w in walls: by_b.setdefault(w['buildingId'], []).append(w)
records, pano_cache = [], {}
for bid, ws in by_b.items():
    z0 = buildings[bid]['center'][2]; occ_b = FeatureOcc(occ, bid)
    for si, s in enumerate(sides_of(ws)):
        L = s['u1'] - s['u0']
        if L < 3 or s['h'] < 2.5: continue
        k = max(1, int(round(L / PIECE_M)))
        for pi in range(k):
            u0 = s['u0'] + L * pi / k; u1 = s['u0'] + L * (pi + 1) / k; um = (u0 + u1) / 2
            mid = s['n'] * s['v'] + s['t'] * um
            piece = {'buildingId': bid, 'midpoint': mid.tolist(), 'normal': s['n'].tolist(), 'lengthM': u1 - u0, 'heightM': s['h']}
            cands = fc.wall_candidates(piece, cams, occ_b, trees, fc.MAX_DISTANCE_M)
            for c in cands:
                c['seenShare'] = round(seen_share(G, x0, y0, cams[c['image']]['xy'], piece, s['zb']), 3)
                c['rank'] = c['score'] * c['incidence'] ** 2 * c['seenShare']
            cands = sorted([c for c in cands if c['seenShare'] >= MIN_SEEN], key=lambda c: -c['rank'])
            rec = {'id': f'{bid}__t{si}p{pi}', 'building': bid, 'side': si, 'piece': pi, 'pieces': k, 'normal': [round(x, 4) for x in s['n']],
                   'planeV': round(s['v'], 3), 'frameU': [round(u0, 3), round(u1, 3)], 'lengthM': round(u1 - u0, 2), 'heightM': round(s['h'], 2),
                   'groundZ': z0, 'baseM': round(s['zb'], 2), 'currentStatus': sorted({w['status'] for w in s['walls']}),
                   'streetFacing': any(w.get('streetFacing') for w in s['walls']), 'cameras': [c['image'] for c in cands[:3]], 'views': []}
            for vi, c in enumerate(cands[:2]):
                cam = cams[c['image']]
                if c['image'] not in pano_cache:
                    if len(pano_cache) > 2: pano_cache.pop(next(iter(pano_cache)))
                    p = np.asarray(fc.load_pano(cam['path']), dtype=np.float32); pano_cache[c['image']] = np.clip(p * road_gain(p), 0, 255).astype(np.uint8)
                pano = pano_cache[c['image']]
                ppm = min(PPM_MAX, MAX_PX / max(piece['lengthM'], 1e-3), MAX_PX / max(piece['heightM'], 1e-3)); ar.PPM = ppm
                img, H, Lp = rectify(piece, cam, z0, pano, base=s['zb'])
                im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); W, Hh = im.size
                pad = 26; can = Image.new('RGB', (W + 2 * pad, Hh + 2 * pad), (255, 255, 255)); can.paste(im, (pad, pad)); d = ImageDraw.Draw(can)
                for mtr in range(int(Lp) + 1):
                    x = pad + mtr * ppm; ln = 10 if mtr % 5 == 0 else 4
                    d.line([(x, pad + Hh), (x, pad + Hh + ln)], fill=(200, 0, 0)); d.line([(x, pad - ln), (x, pad)], fill=(200, 0, 0))
                    if mtr % 5 == 0: d.text((x + 2, pad + Hh + 10), f'{mtr}m', fill=(200, 0, 0))
                for mtr in range(int(H) + 1):
                    y = pad + Hh - mtr * ppm; ln = 10 if mtr % 5 == 0 else 4
                    d.line([(pad - ln, y), (pad, y)], fill=(0, 0, 200)); d.line([(pad + W, y), (pad + W + ln, y)], fill=(0, 0, 200))
                    if mtr % 5 == 0: d.text((1, y - 12), f'{mtr}', fill=(0, 0, 200))
                rname = f"{rec['id']}__v{vi}_rect.jpg"; can.save(out / 'img' / rname, quality=85)
                # the perspective view shows the tier with the podium below it, for context
                vp = fc.view_parameters(cam['xy'], {**piece, 'heightM': s['zb'] + s['h']}, z0)
                pimg, vfov = fc.render_perspective(pano, cam['R'], vp['forward'], min(100, vp['hfovDeg'] * 1.25), min(100, vp['vfovDeg'] * 1.15), width=1000, max_height=1000)
                pname = f"{rec['id']}__v{vi}_persp.jpg"; Image.fromarray(pimg).save(out / 'img' / pname, quality=82)
                rec['views'].append({'image': c['image'], 'captureDate': cam['captureDate'], 'distanceM': c['distanceM'], 'incidence': c['incidence'],
                                     'visibleFraction': c['seenShare'], 'treesNearSightLine': c['treesNearSightLine'], 'ppm': round(ppm, 2),
                                     'rect': rname, 'persp': pname, 'rectPad': pad})
            records.append(rec)
summ = {'buildings': len(by_b), 'pieces': len(records), 'seen': sum(bool(r['views']) for r in records),
        'seenStreetFacing': sum(bool(r['views']) and r['streetFacing'] for r in records), 'seconds': round(time.time() - t0)}
json.dump({'summary': summ, 'pieces': records}, open(out / 'sides-tiers.json', 'w'), indent=1); print(json.dumps(summ, indent=1))
