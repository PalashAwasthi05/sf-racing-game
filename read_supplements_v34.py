"""Trims, bands, colour zones, sign boards and window surrounds for hand-read frontages, from photographs that agree.

v31 (`auto_read_frontages_v31.py --supplements`) read each hand facing's components from its single best-scored
photograph, and they were not re-read when many more panoramas arrived. Here each hand facing is read from up to five
photographs (auto_read_frontages_v33.pick_photos; the published supplement's photograph first), with what other
buildings hide masked (frontage_evidence_v34) and photographs hiding more than 60 % of the wall skipped. A component
is kept when two or more photographs show it: piers within 0.5 m along the wall, bands within 0.35 m in height, sign
boards overlapping by a third, colour zones overlapping by half in height with colours that can be one material, window
surrounds within 0.1 m in width. Its measure is the median of the agreeing photographs and its material the
best-viewed one's. A published component no photograph confirms stays unless three or more photographs that see its
place clearly (60 % of it unoccluded) show nothing there. Components are written, as v31's, in the hand reading's
height frame to work/pipeline/frontage-supplements-v31.json, which the generator merges only for kinds the hand
reading lacks.
Usage: read_supplements_v34.py --candidate work/<dir> [--incumbent <published supplements>] [--out ...]"""
import argparse, json, math, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
import auto_read_frontages_v27 as R27
import auto_read_frontages_v31 as V
import frontage_evidence_v34 as E
from auto_read_frontages_v24 import rectify, PPM
from auto_read_frontages_v33 import pick_photos
from judge_facades_v27 import to_lab
from pipeline.colour_agreement import same_material

MAX_HIDDEN, CLEAR = .6, .6


def rgb_of(material):
    base, _, hx = str(material).partition('#')
    return [int(hx[i:i + 2], 16) for i in (0, 2, 4)] if len(hx) == 6 else None


def clear_share(occ, box, H):
    """Unoccluded share of a box (x0, z0, x1, z1 in wall metres)."""
    return 1. - E.box_hidden(box, occ, H)


def components(p, L, H):
    """A photograph's components in wall metres: {kind: [(key, box, item)]}."""
    out = {'piers': [], 'bands': [], 'signs': [], 'zones': []}
    for q in p.get('piers', []):
        x = q['x'] * L; out['piers'].append((x, (x - q['widthM'] / 2, q['y'][0] * H, x + q['widthM'] / 2, q['y'][1] * H), q))
    for q in p.get('bands', []):
        z = q['height'] * H; out['bands'].append((z, (0., z - q['thicknessM'] / 2, L, z + q['thicknessM'] / 2), q))
    for q in p.get('panels', []):
        r = q['rect']; b = (r[0] * L, r[1] * H, r[2] * L, r[3] * H)
        out['zones' if q.get('evidence') == 'image-measured colour zone' else 'signs'].append(((b[1] + b[3]) / 2, b, q))
    return out


def agree_boxes(per, same, min_photos=2):
    """per: [(photo, key, box, item)]. Greedy groups of items that `same` pairs; kept with members from min_photos photos."""
    groups = []
    for v in sorted(per, key=lambda v: v[0]):
        g = next((g for g in groups if all(same(v, w) for w in g) and v[0] not in {w[0] for w in g}), None)
        if g is not None: g.append(v)
        else: groups.append([v])
    return [g for g in groups if len({w[0] for w in g}) >= min_photos]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True)
    root = pathlib.Path(__file__).resolve().parent
    ap.add_argument('--incumbent', type=pathlib.Path, default=root / 'frontage-supplements-incumbent-v032.json')
    ap.add_argument('--out', type=pathlib.Path, default=root / 'pipeline/frontage-supplements-v31.json')
    ap.add_argument('--cache', type=pathlib.Path, default=root / 'frontage-reading-cache-v27'); ap.add_argument('--max-photos', type=int, default=5); a = ap.parse_args()
    world = json.load(open(a.candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}; ev = json.load(open(a.candidate / 'public/facade-evidence.json'))['walls']
    hand = {p['id']: p for f in sorted((root / 'pipeline').glob('frontage-observations*.json')) if 'auto' not in f.name for p in json.load(open(f)).get('profiles', [])}
    walls = {}
    for w in ev:
        g = w.get('grammarId')
        if g in hand and w.get('lengthM', 0) >= 4 and w['buildingId'] in buildings and not buildings[w['buildingId']].get('landmark'):
            if g not in walls or w['lengthM'] > walls[g]['lengthM']: walls[g] = w
    inc = {s['supplementOf']: s for s in json.load(open(a.incumbent)).get('supplements', [])} if a.incumbent.exists() else {}
    occ_fp = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    surf = E.building_surface(world, frame); ground = E.ground_grid(world, frame)
    seg = det = None; out = []; pano_cache = {}; stats = {'facings': 0, 'photos': 0, 'agreed': 0, 'publishedKept': 0, 'publishedDropped': 0}
    for gid, w in sorted(walls.items()):
        cands = fc.wall_candidates(w, cams, occ_fp, trees, fc.MAX_DISTANCE_M)
        if not cands: continue
        bid = w['buildingId']; z0 = buildings[bid]['center'][2]; H, L = float(w['heightM']), float(w['lengthM']); own = E.own_footprints(w, occ_fp)
        masks = {c['image']: E.hidden_mask(w, cams[c['image']], z0, H, L, surf, own, ground) for c in cands[:16]}
        usable = [c for c in cands[:16] if masks[c['image']].mean() <= MAX_HIDDEN]
        first = (inc.get(gid) or {}).get('camera')
        photos = []
        for c in pick_photos(usable, cams, a.max_photos, first=first):
            cam = cams[c['image']]
            if c['image'] not in pano_cache:
                if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
                pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
                road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
                gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
                pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
            img, H_, L_ = rectify(w, cam, z0, pano_cache[c['image']])
            ck = a.cache / f"sup_{re.sub(r'[^A-Za-z0-9_]', '', gid)}_{c['image']}"
            if ck.with_suffix('.json').exists():
                j = json.loads(ck.with_suffix('.json').read_text()); occ = np.load(ck.with_suffix('.npz'))['occ']
                found = ([tuple(o) for o in j['ops']], [tuple(o) for o in j['signs']], occ, j['visible'], (img * 255).astype(np.uint8), [tuple(g) for g in j['glass']])
            else:
                if seg is None: seg, det = R27.Segmenter(), R27.Detector()
                found = R27.detect(img, seg, det)
                ck.with_suffix('.json').write_text(json.dumps({'ops': [[v if isinstance(v, str) else float(v) for v in o] for o in found[0]], 'signs': [[v if isinstance(v, str) else float(v) for v in o] for o in found[1]],
                                                              'visible': found[3], 'glass': [list(map(float, g)) for g in found[5]]}))
                np.savez_compressed(ck.with_suffix('.npz'), occ=found[2])
            found = E.mask_found(found, masks[c['image']], H_)
            if found[3] < .4: continue
            p = V.enrich({'openings': [], 'groups': [], 'panels': [], 'reading': {'groundStoreyM': min(4.5, H * .35)}}, found, w)
            photos.append({'image': c['image'], 'comp': components(p, L, H), 'occ': found[2].astype(bool), 'frame': p.get('windowFrame')})
        stats['photos'] += len(photos)
        if not photos: continue
        stats['facings'] += 1; per = {k: [(i, key, box, item) for i, ph in enumerate(photos) for key, box, item in ph['comp'][k]] for k in ('piers', 'bands', 'signs', 'zones')}
        near = {'piers': lambda v, u: abs(v[1] - u[1]) <= .5, 'bands': lambda v, u: abs(v[1] - u[1]) <= .35,
                'signs': lambda v, u: E.iou(v[2], u[2]) >= 1 / 3,
                'zones': lambda v, u: (min(v[2][3], u[2][3]) - max(v[2][1], u[2][1])) >= .5 * max(v[2][3] - v[2][1], u[2][3] - u[2][1])
                         and same_material(rgb_of(v[3]['material']) or [128] * 3, rgb_of(u[3]['material']) or [128] * 3)}
        kept = {k: [] for k in per}
        for k in per:
            for g in agree_boxes(per[k], near[k]):
                lead = min(g, key=lambda v: v[0]); item = dict(lead[3]); stats['agreed'] += 1
                if k == 'piers': item['x'] = round(float(np.median([v[1] for v in g])) / L, 4); item['widthM'] = round(float(np.median([v[3]['widthM'] for v in g])), 2)
                elif k == 'bands': item['height'] = round(float(np.median([v[1] for v in g])) / H, 4)
                else: item['rect'] = [round(float(np.median([v[2][0] for v in g])) / L, 4), round(float(np.median([v[2][1] for v in g])) / H, 4),
                                      round(float(np.median([v[2][2] for v in g])) / L, 4), round(float(np.median([v[2][3] for v in g])) / H, 4)]
                kept[k].append((item, g))
        frames = [ph['frame'] for ph in photos if ph.get('frame')]
        frame_ = None
        if len(frames) >= 2:
            fw = sorted(f['frameWidthM'] for f in frames); m = float(np.median(fw))
            agreeing = [f for f in frames if abs(f['frameWidthM'] - m) <= .1]
            if len(agreeing) >= 2: frame_ = dict(agreeing[0]); frame_['frameWidthM'] = round(m, 2); frame_['photographs'] = len(agreeing)
        # published components no photograph confirms: dropped only when three clear photographs show nothing there
        ref = float(hand[gid].get('heightRefM') or H); kz = H / ref
        pub = inc.get(gid) or {}
        for k, src in (('piers', 'piers'), ('bands', 'bands'), ('signs', 'panels')):
            for q in pub.get(src, []):
                if k == 'signs' and q.get('evidence') == 'image-measured colour zone': continue
                if k == 'piers':
                    x = q['x'] * L; box = (x - q['widthM'] / 2, q['y'][0] / kz * H if False else q['y'][0] * H / kz, x + q['widthM'] / 2, q['y'][1] * H / kz); key = x
                elif k == 'bands':
                    z = q['height'] * H / kz; box = (0., z - q['thicknessM'] / 2, L, z + q['thicknessM'] / 2); key = z
                else:
                    r = q['rect']; box = (r[0] * L, r[1] * H / kz, r[2] * L, r[3] * H / kz); key = (box[1] + box[3]) / 2
                v = (-1, key, box, q)
                if any(near[k](v, (0, g[0][1], g[0][2], g[0][3])) for _, g in kept[k]): continue
                clear = [ph for ph in photos if clear_share(ph['occ'], box, H) >= CLEAR]
                absent = [ph for ph in clear if not any(near[k](v, (0, kk, bb, it)) for kk, bb, it in ph['comp'][k])]
                if len(absent) >= 3: stats['publishedDropped'] += 1; continue
                it = dict(q)
                if k == 'piers': it['y'] = [q['y'][0] / kz, q['y'][1] / kz]
                elif k == 'bands': it['height'] = q['height'] / kz
                else: it['rect'] = [q['rect'][0], q['rect'][1] / kz, q['rect'][2], q['rect'][3] / kz]
                kept[k].append((it, [])); stats['publishedKept'] += 1
        if not frame_ and pub.get('windowFrame'): frame_ = pub['windowFrame']
        def zf(v): return round(float(v) * kz, 4)
        sup = {'supplementOf': gid, 'wallMidpoint': [round(v, 2) for v in w['midpoint']], 'camera': photos[0]['image'], 'photographs': [ph['image'] for ph in photos],
               'panels': [{**q, 'rect': [q['rect'][0], zf(q['rect'][1]), q['rect'][2], zf(q['rect'][3])]} for q, _ in kept['zones'] + kept['signs']],
               'piers': [{**q, 'y': [zf(q['y'][0]), zf(q['y'][1])]} for q, _ in kept['piers']],
               'bands': [{**q, 'height': zf(q['height'])} for q, _ in kept['bands']],
               'reading': {k: len(v) for k, v in kept.items() if v}}
        if frame_: sup['windowFrame'] = frame_
        if sup['panels'] or sup['piers'] or sup['bands'] or sup.get('windowFrame'): out.append(sup)
        print(gid, 'photographs', len(photos), sup['reading'], flush=True)
    a.out.write_text(json.dumps({'schemaVersion': 1, 'method': 'read_supplements_v34.py: components two or more photographs agree on, building occlusion masked; published ones kept unless three clear photographs show nothing',
                                 'supplements': out, 'stats': stats}, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    print(len(out), 'supplements', json.dumps(stats))


if __name__ == '__main__':
    main()
