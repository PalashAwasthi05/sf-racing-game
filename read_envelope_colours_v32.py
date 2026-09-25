"""Photographed colours for envelope-designed frontages. Envelope profiles (work/pipeline/envelope-observations.json)
design a frontage's parts (plinths, towers, bays) with a material family but rarely a colour, and were drawn without
any photograph: their walls had the second-worst colour error of any source (judge median 0.28). For each envelope
wall of a candidate, the closest clear panorama is rectified over the whole frontage (the envelope's own frame: x
along its tangent from its left edge, z above the building base), openings, signs, trees, vehicles, people and sky
are masked (v27 detections, cached), and the median colour of each 2 x 2 m cell is kept when at least 30 % of the cell
is wall and it is not near-black (L* 22: shade or trees); greenish pixels are foliage, not wall, even where the
segmenter missed a shaded canopy (the rule photo_openings uses). Each wall is read from up to five photographs (v33's choice:
the best view, then others taken on another day or 10 m apart); pipeline/envelopes.py takes, per part, each
photograph's median colour of the cells over its visible face (when they cover 30 % of it) and recolours the part only
with a colour two or more photographs agree on, keeping the part's shape and material family.
Usage: read_envelope_colours_v32.py --candidate work/<dir> [--out work/pipeline/envelope-colours-v32.json] [--max-photos 5]"""
import argparse, json, math, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
import auto_read_frontages_v27 as R27
import auto_read_frontages_v31 as V
from auto_read_frontages_v33 import pick_photos
from auto_read_frontages_v24 import rectify, PPM
from judge_facades_v27 import to_lab

CELL_M, MIN_COVER, DARK = 2., .3, 22.


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('work/pipeline/envelope-colours-v32.json'))
    ap.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('work/frontage-reading-cache-v27')); ap.add_argument('--max-photos', type=int, default=5); a = ap.parse_args()
    world = json.load(open(a.candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}
    walls = [w for w in json.load(open(a.candidate / 'public/facade-evidence.json'))['walls'] if w.get('status') == 'envelope-graph' and w.get('buildingId') in buildings]
    occ_fp = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    seg = det = None; pano_cache = {}; out = {}
    for w in walls:
        pid = w.get('grammarId'); bid = w['buildingId']
        if not pid or buildings[bid].get('landmark'): continue
        cands = fc.wall_candidates(w, cams, occ_fp, trees, fc.MAX_DISTANCE_M)
        if not cands: print(pid, 'no clear panorama'); continue
        z0 = buildings[bid]['center'][2]; photos = []
        for c in pick_photos(cands, cams, a.max_photos):
            cam = cams[c['image']]
            if c['image'] not in pano_cache:
                if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
                pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
                road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
                gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
                pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
            img, H, L = rectify(w, cam, z0, pano_cache[c['image']])
            ck = a.cache / f"env_{re.sub(r'[^A-Za-z0-9_]', '', pid)}_{c['image']}"
            if ck.with_suffix('.json').exists():
                j = json.loads(ck.with_suffix('.json').read_text()); occ = np.load(ck.with_suffix('.npz'))['occ']
                found = ([tuple(o) for o in j['ops']], [tuple(o) for o in j['signs']], occ, j['visible'], (img * 255).astype(np.uint8), [tuple(g) for g in j['glass']])
            else:
                if seg is None:
                    from facade_segment_v27 import Segmenter, Detector
                    seg, det = Segmenter(), Detector()
                found = R27.detect(img, seg, det)
                ck.with_suffix('.json').write_text(json.dumps({'ops': [[v if isinstance(v, str) else float(v) for v in o] for o in found[0]], 'signs': [[v if isinstance(v, str) else float(v) for v in o] for o in found[1]],
                                                              'visible': found[3], 'glass': [list(map(float, g)) for g in found[5]]}))
                np.savez_compressed(ck.with_suffix('.npz'), occ=found[2])
            ops, signs, occ, visible, u8, glass = found
            M = V.wall_mask(u8, occ, {'openings': [], 'groups': []}, ops, signs, glass, H, L); lab = to_lab(u8.astype(float))
            r_, g_, b_ = (u8[..., k].astype(float) for k in range(3)); mx = u8.max(2).astype(float)
            M &= ~((g_ > r_ * 1.04) & (g_ > b_ * 1.02) & ((mx - u8.min(2)) / np.maximum(mx, 1) > .12))   # foliage the segmenter missed (shaded canopy)
            cells = []
            for x0 in np.arange(0., L, CELL_M):
                for zb in np.arange(1., H, CELL_M):
                    x1, zt = min(L, x0 + CELL_M), min(H, zb + CELL_M)
                    r0, r1, c0, c1 = int((H - zt) * PPM), int((H - zb) * PPM), int(x0 * PPM), int(x1 * PPM); m = M[r0:r1, c0:c1]
                    if m.size == 0 or m.mean() < MIN_COVER or m.sum() < 60: continue
                    med = np.median(lab[r0:r1, c0:c1][m], 0)
                    if med[0] < DARK: continue
                    cells.append([round(float(x0), 2), round(float(zb), 2), round(float(x1), 2), round(float(zt), 2), [int(v) for v in np.clip(V.lab_to_rgb(med), 0, 255)], round(float(m.mean()), 2)])
            photos.append({'camera': c['image'], 'distanceM': round(c['distanceM'], 1), 'visibleShare': round(float(visible), 2), 'cells': cells})
        n = np.asarray(w['normal'], float); n /= np.linalg.norm(n)
        key = f'{pid}|{bid}'                                      # one design can serve several buildings
        if key in out and sum(len(q['cells']) for q in out[key]['photos']) >= sum(len(q['cells']) for q in photos): continue
        out[key] = {'buildingId': bid, 'normal': n.round(4).tolist(), 'tangent': [round(-n[1], 4), round(n[0], 4)], 'lengthM': round(L, 2), 'heightM': round(H, 2), 'photos': photos}
        print(pid, bid, f'{L:.0f}x{H:.0f} m', 'photos', len(photos), 'cells', [len(q['cells']) for q in photos], flush=True)
    a.out.write_text(json.dumps({'schemaVersion': 2, 'method': __doc__.split('\n')[0], 'envelopes': out}, indent=1)); print(len(out), 'envelope frontages read')


if __name__ == '__main__':
    main()
