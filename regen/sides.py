"""Stage 1: building sides and their photographs.

A side is the run of one building's wall pieces that face the same way in one plane (the current build splits walls
where the roof steps; a facade reads across those cuts). Long sides are cut into pieces of about 25 m, each seen from
its own best camera. For every piece: a square-on rectified photograph (fractions across it are the spec's frame),
annotated with a metre scale on its edges, and the ordinary perspective view for context. Geometric rules only.
Cameras are the area's registered ones (regen/register_cameras.py: each panorama placed against the LiDAR's skyline)
when the area has them: a provider pose metres off cut crops shifted along the wall or at the wrong scale, and the
readings measured on them landed beside their windows (--nominal uses the provider poses).
"""
import argparse, json, math, pathlib, sys, time
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import frontage_crops_v15 as fc
from auto_read_frontages_v24 import rectify
import auto_read_frontages_v24 as ar
from panorama_balance import road_gain

ANGLE_DEG, PLANE_M, PIECE_M, PPM_MAX, MAX_PX = 12., 1.5, 25., 45., 1400

def sides_of(walls):
    out = []
    for w in sorted(walls, key=lambda w: -w['lengthM']):
        n = np.asarray(w['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]]); m = np.asarray(w['midpoint'], float)
        for s in out:
            if float(n @ s['n']) > math.cos(math.radians(ANGLE_DEG)) and abs(float(m @ s['n']) - s['v']) < PLANE_M:
                u = float(m @ s['t']); s['u0'] = min(s['u0'], u - w['lengthM'] / 2); s['u1'] = max(s['u1'], u + w['lengthM'] / 2)
                s['h'] = max(s['h'], w['heightM']); s['walls'].append(w); break
        else:
            u = float(m @ t)
            out.append({'n': n, 't': t, 'v': float(m @ n), 'u0': u - w['lengthM'] / 2, 'u1': u + w['lengthM'] / 2, 'h': w['heightM'], 'walls': [w]})
    return out

ap = argparse.ArgumentParser(); ap.add_argument('--evidence', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--shard', default=None, help='k/n: only the buildings of shard k of n (zlib.crc32 of the id), written to sides-k.json')
ap.add_argument('--keep', default=None, help='an existing sides file: its pieces that have photographs are left as they are (no images rewritten); the rest are redone (more panoramas fetched)')
ap.add_argument('--nominal', action='store_true', help='the provider camera poses, not the registered ones')
ap.add_argument('--only', default=None, help='a JSON list of building ids: only their sides')
a = ap.parse_args()
OLD = {p['id']: p for p in json.load(open(a.keep))['pieces']} if a.keep else {}
KEEP = {i for i, p in OLD.items() if p['views']}
out = pathlib.Path(a.out); (out / 'img').mkdir(parents=True, exist_ok=True); t0 = time.time()
world, frame = fc.load_world(); buildings = {b['id']: b for b in world['buildings']}
# set-back tier walls (standing on a lower roof) are framed from their own foot by tier_sides.py, not from the street
walls = [w for w in json.load(open(a.evidence))['facadeWalls'] if w['buildingId'] in buildings and (w.get('zBaseM') or 0) <= .5]
occ = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = {k: c for k, c in fc.load_cameras(frame).items() if c['exists']}
import os
_reg = pathlib.Path(os.environ['SFW_AREA']) / 'imagery/registered-cameras.json' if os.environ.get('SFW_AREA') else None
if not a.nominal and _reg is not None and _reg.exists():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from register_cameras import apply_registration
    cams = apply_registration(cams, _reg); print('registered cameras:', sum(c.get('registered') == 'skyline' for c in cams.values()), 'of', len(cams), flush=True)
ONLY = set(json.load(open(a.only))) if a.only else None
by_b = {}
for w in walls: by_b.setdefault(w['buildingId'], []).append(w)
records, pano_cache = [], {}
import zlib
_sk, _sn = (int(x) for x in a.shard.split('/')) if a.shard else (0, 1)
for bid, ws in by_b.items():
    if zlib.crc32(bid.encode()) % _sn != _sk: continue
    if ONLY is not None and bid not in ONLY and bid.split('_p')[0] not in ONLY: continue
    z0 = buildings[bid]['center'][2]
    for si, s in enumerate(sides_of(ws)):
        L = s['u1'] - s['u0']
        if L < 3 or s['h'] < 2.5: continue
        k = max(1, int(round(L / PIECE_M)))
        for pi in range(k):
            if f'{bid}__s{si}p{pi}' in KEEP: records.append(OLD[f'{bid}__s{si}p{pi}']); continue
            u0 = s['u0'] + L * pi / k; u1 = s['u0'] + L * (pi + 1) / k; um = (u0 + u1) / 2
            mid = s['n'] * s['v'] + s['t'] * um
            piece = {'buildingId': bid, 'midpoint': mid.tolist(), 'normal': s['n'].tolist(), 'lengthM': u1 - u0, 'heightM': s['h']}
            cands = fc.wall_candidates(piece, cams, occ, trees, fc.MAX_DISTANCE_M)
            # a camera must stand back far enough to see the whole height square-ish (a close camera sees the ground
            # storey from under its canopy) and face the wall: re-rank by vertical coverage and incidence
            need = max(4., .75 * (piece['heightM'] - fc.EYE_HEIGHT_M))
            # a photograph older than the freshness window ranks lower: it is used where no recent drive passed
            for c in cands: c['rank'] = c['score'] * min(1., c['distanceM'] / need) ** 2 * c['incidence'] ** 2 * (1. if str(cams[c['image']]['captureDate']) >= '2023' else .6)
            cands.sort(key=lambda c: -c['rank'])
            rec = {'id': f'{bid}__s{si}p{pi}', 'building': bid, 'side': si, 'piece': pi, 'pieces': k, 'normal': [round(x, 4) for x in s['n']],
                   'planeV': round(s['v'], 3), 'frameU': [round(u0, 3), round(u1, 3)], 'lengthM': round(u1 - u0, 2), 'heightM': round(s['h'], 2),
                   'groundZ': z0, 'currentStatus': sorted({w['status'] for w in s['walls']}), 'streetFacing': any(w.get('streetFacing') for w in s['walls']),
                   'cameras': [c['image'] for c in cands[:3]], 'views': []}
            for vi, c in enumerate(cands[:2]):
                cam = cams[c['image']]
                if c['image'] not in pano_cache:
                    if len(pano_cache) > 11: pano_cache.pop(next(iter(pano_cache)))
                    p = np.asarray(fc.load_pano(cam['path']), dtype=np.float32); pano_cache[c['image']] = np.clip(p * road_gain(p), 0, 255).astype(np.uint8)
                pano = pano_cache[c['image']]
                ppm = min(PPM_MAX, MAX_PX / max(piece['lengthM'], 1e-3), MAX_PX / max(piece['heightM'], 1e-3))
                ar.PPM = ppm
                gz = fc.ground_at(cam['xy']); gz = None if gz is None else gz + cam.get('heightOffsetM', 0.)   # a registered camera's own height
                img, H, Lp = rectify(piece, {**cam, 'groundZ': gz}, z0, pano)
                im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); W, Hh = im.size
                # a metre scale on the edges: a tick every metre, longer every 5 m, labelled every 5 m from the left and the ground
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
                vp = fc.view_parameters(cam['xy'], piece, z0)
                pimg, vfov = fc.render_perspective(pano, cam['R'], vp['forward'], min(100, vp['hfovDeg'] * 1.25), min(100, vp['vfovDeg'] * 1.15), width=1000, max_height=1000)
                pname = f"{rec['id']}__v{vi}_persp.jpg"; Image.fromarray(pimg).save(out / 'img' / pname, quality=82)
                rec['views'].append({'image': c['image'], 'registered': cam.get('registered', False), 'captureDate': cam['captureDate'], 'distanceM': c['distanceM'], 'incidence': c['incidence'],
                                     'visibleFraction': c['visibleFraction'], 'treesNearSightLine': c['treesNearSightLine'], 'ppm': round(ppm, 2),
                                     'rect': rname, 'persp': pname, 'rectPad': pad, 'cameraGroundZ': None if gz is None else round(gz, 2)})
            records.append(rec)
summ = {'buildings': len(by_b), 'pieces': len(records), 'seen': sum(bool(r['views']) for r in records),
        'seenStreetFacing': sum(bool(r['views']) and r['streetFacing'] for r in records), 'images': sum(len(r['views']) * 2 for r in records),
        'buildingsUnseen': sorted({r['building'] for r in records} - {r['building'] for r in records if r['views']}), 'seconds': round(time.time() - t0)}
json.dump({'summary': summ, 'pieces': records}, open(out / (f'sides-{_sk}.json' if a.shard else 'sides.json'), 'w'), indent=1); print(json.dumps(summ, indent=1))
