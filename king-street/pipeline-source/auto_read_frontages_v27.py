"""Automatic frontage reading v27: what the photograph shows, not a fitted grid.

For every building facing a cached panorama sees well (the frontage module's camera choice), the wall is rectified
square-on and read by detectors: windows, doors, shop signs and awnings (Grounding DINO over zoomed tiles), window
blobs by colour and frame (openings the detector misses on dark masonry), trees, cars, people and sky (SegFormer).
The detections are regularised as a person drawing the elevation would: windows fall into storeys (one sill and head
per storey), their centres snap to columns shared by at least two storeys, and irregular runs stay irregular. A
stretch a tree or a car hides continues a storey's windows only where that storey's rhythm is regular. The ground
storey carries shopfronts and entries with the sign boards above them as fascia panels (their own colour) and
awnings as projecting bands. A clearly seen wall with no openings is a blank wall; a mostly hidden one keeps only
its measured material. Profiles carry the rectified image's metric frame (`frameU`), so openings land where the
photograph shows them. Hand readings of the same facing take precedence. Nothing here names a place.

Usage: auto_read_frontages_v27.py --candidate <build dir> --out work/pipeline/frontage-observations-auto.json
"""
import argparse, json, math, pathlib, re, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
from auto_read_frontages_v24 import rectify, read, profile_for, PPM, MIN_WALL_M
from facade_segment_v27 import Segmenter, Detector, glazed_boxes
from judge_facades_v27 import photo_openings

STREET_M = 3.2           # storey rows are found above the street clutter band


def detect(img, seg, det):
    """Candidate openings [(x0, z0, x1, z1, kind)] in wall metres (x from the image's left, z up from the base), sign
    and awning boxes, the occluded mask (trees, cars, people, sky) and the usable share of the wall."""
    u8 = (img * 255).astype(np.uint8); Hp, W = u8.shape[:2]; H = Hp / PPM
    codes = seg.labels(u8); occ = (codes == 3) | (codes == 4)
    boxes = det.boxes(u8); ops, signs = [], []
    def metres(x0, y0, x1, y1): return (max(0., x0 / PPM), max(0., H - y1 / PPM), min(W / PPM, x1 / PPM), min(H, H - y0 / PPM))
    for x0, y0, x1, y1, kind, sc in boxes:
        m = metres(x0, y0, x1, y1)
        if kind in ('window', 'door'): ops.append((*m, kind))
        else: signs.append((*m, kind))
    blobs = photo_openings(u8.astype(float))[0]
    lab, n = ndi.label(blobs)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        m = metres(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
        if not any(min(m[2], o[2]) - max(m[0], o[0]) > .3 * (m[2] - m[0]) and min(m[3], o[3]) - max(m[1], o[1]) > .3 * (m[3] - m[1]) for o in ops):
            ops.append((*m, 'window'))
    glass = [(*metres(g[0], g[1], g[2], g[3]), g[4]) for g in glazed_boxes(det, u8)]
    return ops, signs, occ, float(1 - occ[: max(1, Hp - int(STREET_M * PPM))].mean()), u8, glass


def regularise(ops, occ, H, L):
    """Storeys of windows and the ground storey's openings. Returns (ground height, rows, ground openings); a row is
    (bottom, top, [(x0, x1)])."""
    # Window-sized boxes; glazing that starts at the pavement may be a whole shopfront (up to 60 m2, 20 m wide).
    ops = [o for o in ops if .25 <= (o[2] - o[0]) * (o[3] - o[1]) <= (60 if o[1] < 1.5 else 30) and o[3] - o[1] <= 6.5 and o[2] - o[0] <= (20 if o[1] < 1.5 else 12)]
    # The ground storey ends under the first regular upper storey: the lowest row of two or more windows whose rise
    # to the next row is the storey pitch. Transoms, signs and mezzanine lights below it belong to the ground storey.
    probe = sorted([o for o in ops if (o[1] + o[3]) / 2 > 2.2], key=lambda o: (o[1] + o[3]) / 2); bands = []
    for o in probe:
        cz = (o[1] + o[3]) / 2
        if bands and abs(cz - bands[-1][0]) < .9: bands[-1][1].append(o); bands[-1][0] = float(np.mean([(q[1] + q[3]) / 2 for q in bands[-1][1]]))
        else: bands.append([cz, [o]])
    full = [b for b in bands if len(b[1]) >= 2]
    pitch = float(np.median(np.diff([b[0] for b in full]))) if len(full) >= 3 else None
    first = next((b for i, b in enumerate(full[:-1]) if pitch is None or abs((full[i + 1][0] - b[0]) - pitch) <= .3 * pitch), full[0] if full else None)
    gh = float(np.clip(min(q[1] for q in first[1]) - .6, 3., 7.)) if first else min(4.5, H * .45)
    if first:
        # Below the first detected storey the wall may be hidden (trees): then the ground storey is not measured and
        # takes the common height, and the storeys between continue from the lowest one seen.
        Hp_, W_ = occ.shape; r0 = int(max(0, (H - (min(q[1] for q in first[1]) - .3)) * PPM)); r1 = int(min(Hp_, (H - 4.5) * PPM))
        if r1 > r0 and occ[r0:r1].mean() > .5: gh = min(4.5, H * .45)
    if H <= 6.5 and not (first and min(q[1] for q in first[1]) > 3.):
        gh = H - .4                                                       # a single storey: the whole wall is the ground storey
    upper = [o for o in ops if (o[1] + o[3]) / 2 > gh]; ground = [o for o in ops if (o[1] + o[3]) / 2 <= gh]
    upper.sort(key=lambda o: (o[1] + o[3]) / 2); groups = []
    for o in upper:
        cz = (o[1] + o[3]) / 2
        if groups and abs(cz - np.mean([(g[1] + g[3]) / 2 for g in groups[-1]])) < .9: groups[-1].append(o)
        else: groups.append([o])
    rows = []
    for g in groups:
        zb = float(np.median([o[1] for o in g])); zt = float(np.median([o[3] for o in g]))
        if zt - zb < .35: continue                                    # ribbon glazing can be a thin band
        spans = sorted((o[0], o[2]) for o in g); merged = []
        for a, b in spans:
            if merged and a < merged[-1][1] - .3 * (b - a): merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else: merged.append((a, b))
        rows.append([zb, zt, merged])
    # Columns shared by at least two storeys: centres within 0.45 m snap to their mean, widths to the column median.
    centres = [((a + b) / 2, b - a, ri) for ri, r in enumerate(rows) for a, b in r[2]]
    centres.sort(); clusters = []
    for c in centres:
        if clusters and c[0] - clusters[-1][-1][0] < .45: clusters[-1].append(c)
        else: clusters.append([c])
    snap = {}
    for cl in clusters:
        if len({c[2] for c in cl}) >= 2:
            mx = float(np.mean([c[0] for c in cl])); mw = float(np.median([c[1] for c in cl]))
            for c in cl: snap[(c[2], round(c[0], 3))] = (mx - mw / 2, mx + mw / 2)
    for ri, r in enumerate(rows): r[2] = [snap.get((ri, round((a + b) / 2, 3)), (a, b)) for a, b in r[2]]
    # A hidden stretch continues a storey only where its windows keep a steady pitch.
    Hp, W = occ.shape
    def hidden(x0, x1, z0, z1):
        c0, c1 = int(max(0, x0 * PPM)), int(min(W, x1 * PPM)); r0, r1 = int(max(0, (H - z1) * PPM)), int(min(Hp, (H - z0) * PPM))
        return c1 > c0 and r1 > r0 and occ[r0:r1, c0:c1].mean() > .5
    for r in rows:
        xs = [(a + b) / 2 for a, b in r[2]]
        if len(xs) < 3: continue
        d = np.diff(xs); pitch = float(np.median(d))
        if pitch < .8 or np.std(d[d < 1.5 * pitch]) > .15 * pitch: continue
        w = float(np.median([b - a for a, b in r[2]])); add = []
        for x_a, x_b in zip(xs, xs[1:]):
            k = int(round((x_b - x_a) / pitch))
            for j in range(1, k):
                cx = x_a + j * pitch
                if hidden(cx - w / 2, cx + w / 2, r[0], r[1]): add.append((cx - w / 2, cx + w / 2))
        r[2] = sorted(r[2] + add)
    # Storeys hidden behind trees continue the lowest regular storey downward at the storey pitch (never into the
    # ground storey, never where the wall is seen and plain).
    rows.sort(key=lambda r: r[0])
    if len(rows) >= 2:
        pitch = float(np.median(np.diff([r[0] for r in rows]))); low = rows[0]
        if 2.6 <= pitch <= 5.5 and len(low[2]) >= 2:
            zb = low[0] - pitch
            while zb >= gh + .3:
                zt = zb + (low[1] - low[0]); spans = [(a, b) for a, b in low[2] if hidden(a, b, zb, zt)]
                if len(spans) < .5 * len(low[2]): break
                rows.insert(0, [zb, zt, list(low[2])]); zb -= pitch
    gops = []
    for o in sorted(ground, key=lambda o: o[1]):
        a, z0_, b, z1_, kind = o
        if z0_ > 2.:
            # A transom or clerestory over a shopfront raises the opening under it; alone it is not a ground opening.
            under = next((i for i, g in enumerate(gops) if min(b, g[2]) - max(a, g[0]) > .5 * (b - a)), None)
            if under is not None: g = gops[under]; gops[under] = (g[0], g[1], g[2], max(g[3], min(gh - .3, z1_)), g[4])
            continue
        top = min(gh - .3, z1_); bottom = max(.1, min(z0_, .6))
        if top - bottom > 1.2 and b - a > .6: gops.append((a, bottom, b, top, 'entry' if kind == 'door' or b - a < 1.6 else 'shopfront'))
    gops.sort()
    return gh, rows, gops


def colour_of(u8, box, H):
    x0, z0, x1, z1 = box[:4]; r0, r1 = int((H - z1) * PPM), int((H - z0) * PPM); c0, c1 = int(x0 * PPM), int(x1 * PPM)
    px = u8[max(0, r0):max(r0 + 1, r1), max(0, c0):max(c0 + 1, c1)].reshape(-1, 3)
    return np.median(px, 0).astype(int) if len(px) else np.array([60, 60, 60])


def profile(bid, building, wall, img, reading, cam_image, found):
    ops, signs, occ, visible, u8, glass = found
    glass = [g for g in glass if g[4] >= .45]                         # only strong glazing detections are believed
    # A box larger than a window above the ground storey is either a box round a group of windows (smaller detections
    # inside it: those are the windows) or a glazed area in itself (nothing inside: a window wall or curtain wall).
    def inside(o, q): return q is not o and q[0] >= o[0] - .3 and q[2] <= o[2] + .3 and q[1] >= o[1] - .3 and q[3] <= o[3] + .3 and (q[2] - q[0]) * (q[3] - q[1]) < .6 * (o[2] - o[0]) * (o[3] - o[1])
    big = [o for o in ops if (o[2] - o[0]) * (o[3] - o[1]) > 30 and o[1] >= 1.5]
    glazed = [o for o in big if not any(inside(o, q) for q in ops)]
    ops = [o for o in ops if o not in big]
    # Glazing that starts at the pavement and stays in the ground storey is a shopfront the tiled pass may have split.
    ops = list(ops) + [(g[0], g[1], g[2], g[3], 'window') for g in glass if g[1] < 1.5 and g[3] - g[1] <= 6.5]
    H = float(wall['heightM']); L = float(wall['lengthM'])
    p = profile_for(bid, building, wall, reading, cam_image)
    p.pop('groups', None); p.pop('materialOnly', None); p['status'] = 'image-measured-auto-v27'
    n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n); u = np.array([-n[1], n[0]]); m = np.asarray(wall['midpoint'], float)
    p['frameU'] = [round(float(m @ u - L / 2), 3), round(float(m @ u + L / 2), 3)]
    if visible < .4:
        p['materialOnly'] = True; p['reading'] = {'visibleShare': round(visible, 2)}; return p
    gh, rows, gops = regularise(ops, occ, H, L)
    frac = lambda a, b, z0, z1: [round(float(a) / L, 4), round(float(z0) / H, 4), round(float(b) / L, 4), round(float(z1) / H, 4)]
    openings = []; groups = []
    # A second, independent opinion: the periodic-edge reading. Where it finds a confident storey and bay grid and the
    # detector found under 40 % of the windows that grid implies (a dark or screened photograph), the detections are
    # not the facade: the upper storeys take the grid (the detected ground storey and signs are kept).
    grid_fallback = False
    if reading.get('floorPitchM') and reading.get('bayPitchM') and reading.get('gridConfidence', 0) >= .2:
        expected = max(1., (H - gh) / reading['floorPitchM']) * max(1., L / reading['bayPitchM'])
        if sum(len(r[2]) for r in rows) < .4 * expected and not glazed:
            grid_fallback = True; rows = []
            style = {'kind': 'window', 'openingRatio': reading['openingRatio'], 'openingHeightRatio': reading['openingHeightRatio'],
                     'mullionFractions': [.5] if reading['bayPitchM'] * reading['openingRatio'] > 1.8 else [], 'transomFractions': []}
            groups.append({'region': frac(0., L, gh, H * .97), 'columnPitchM': reading['bayPitchM'], 'rowPitchM': reading['floorPitchM'], 'style': style})
    for o in glazed:
        # a glazed area: lights between mullions (1.5 m) and transoms at a storey
        groups.append({'region': frac(max(0., o[0]), min(L, o[2]), max(0., o[1]), min(H, o[3])), 'columnPitchM': 1.5, 'rowPitchM': 3.6,
                       'style': {'kind': 'window', 'openingRatio': .92, 'openingHeightRatio': .8, 'mullionFractions': [], 'transomFractions': []}})
    win_area = sum((b - a) * (zt - zb) for zb, zt, spans in rows for a, b in spans)
    big = max(glass, key=lambda g: (g[2] - g[0]) * (g[3] - g[1]), default=None)
    if big is not None and (big[2] - big[0]) * (big[3] - big[1]) >= .5 * L * H and win_area < .2 * L * H:
        # A curtain wall: the glazed area is one bank of lights between mullions (1.5 m) and spandrels (one storey).
        pitch = float(np.median(np.diff([(r[0] + r[1]) / 2 for r in rows]))) if len(rows) >= 2 else 3.8
        z_lo = max(big[1], gh if gops else 0.); groups.append({'region': frac(big[0], big[2], z_lo, min(H, big[3])), 'columnPitchM': 1.5, 'rowPitchM': round(float(np.clip(pitch, 3., 5.)), 2),
                                                               'style': {'kind': 'window', 'openingRatio': .94, 'openingHeightRatio': .78, 'mullionFractions': [], 'transomFractions': []}})
        rows = [r for r in rows if r[1] <= z_lo or r[0] >= big[3]]      # windows inside the curtain wall are its lights
    for zb, zt, spans in rows:
        for a, b in spans:
            if b - a < .35: continue
            openings.append({'rect': frac(a, b, zb, zt), 'style': {'kind': 'window', 'mullionFractions': [.5] if b - a > 2.2 else [], 'transomFractions': []},
                             'evidence': 'image-detected'})
    for a, z0, b, z1, kind in gops:
        openings.append({'rect': frac(a, b, z0, z1), 'style': {'kind': kind, 'mullionFractions': [.5] if kind == 'shopfront' and b - a > 2.4 else [], 'transomFractions': [.8] if kind == 'shopfront' else []},
                         'evidence': 'image-detected'})
    panels, bands = [], []
    for s in signs:
        a, z0, b, z1, kind = s
        if z1 < 1.2 or z0 > gh + 2. or b - a < .8: continue            # a fascia belongs to the ground storey's top
        rgb = colour_of(u8, s, H); hexc = '%02x%02x%02x' % tuple(int(v) for v in rgb)
        if kind == 'awning': bands.append({'height': round(min(z1, gh) / H, 4), 'thicknessM': .12, 'depthM': 1.2, 'projectionM': 1.0, 'material': f'panel_dark#{hexc}', 'xRange': [round(a / L, 4), round(b / L, 4)]})
        else: panels.append({'rect': frac(a, b, max(z0, gh - 1.2), min(z1, gh + .9)), 'material': f'panel_dark#{hexc}', 'evidence': 'image-detected sign'})
    p['openings'] = openings; p['panels'] = panels; p['bands'] = bands; p['groups'] = groups
    p['reading'] = {'visibleShare': round(visible, 2), 'groundStoreyM': round(gh, 2), 'storeys': len(rows), 'windows': sum(len(r[2]) for r in rows),
                    'groundOpenings': len(gops), 'signs': len(panels), 'awnings': len(bands), 'wallRgb': reading.get('wallRgb'), 'family': reading.get('family')}
    p['reading']['curtainWall'] = bool(groups) and not grid_fallback; p['reading']['glazedAreas'] = len(glazed); p['reading']['gridFallback'] = grid_fallback
    # The measured character, in the terms other walls borrow (storey and bay pitch, opening proportions, ground).
    if len(rows) >= 2:
        fp = float(np.median(np.diff([r[0] for r in rows])))
        if 2.6 <= fp <= 5.5: p['reading']['floorPitchM'] = round(fp, 2)
    gaps = [d for r in rows if len(r[2]) >= 3 for d in np.diff([(a + b) / 2 for a, b in r[2]])]
    if gaps and p['reading'].get('floorPitchM'):
        bp = float(np.median(gaps)); ws = [b - a for r in rows for a, b in r[2]]; hs = [r[1] - r[0] for r in rows]
        if 1.2 <= bp <= 8.:
            p['reading'].update(bayPitchM=round(bp, 2), openingRatio=round(float(np.clip(np.median(ws) / bp, .25, .9)), 2),
                                openingHeightRatio=round(float(np.clip(np.median(hs) / p['reading']['floorPitchM'], .3, .9)), 2))
    p['reading']['shopfront'] = bool(sum(g[2] - g[0] for g in gops if g[4] == 'shopfront') >= .3 * L)
    if not openings and not groups: p['reading']['blank'] = True       # a clearly seen wall without openings is blank
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True); ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--max-walls', type=int, default=400); ap.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('work/frontage-reading-cache-v27'))
    a = ap.parse_args(); a.cache.mkdir(parents=True, exist_ok=True)
    world = json.load(open(a.candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}
    ev = json.load(open(a.candidate / 'public/facade-evidence.json'))
    walls = [w for w in ev['walls'] if w.get('lengthM', 0) >= MIN_WALL_M and w.get('heightM', 0) >= 3 and w.get('buildingId') in buildings and w.get('status') != 'envelope-graph'
             and not buildings[w['buildingId']].get('landmark')]
    occ_fp = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    best = {}
    for w in walls:
        cands = fc.wall_candidates(w, cams, occ_fp, trees, fc.MAX_DISTANCE_M)
        if not cands: continue
        c = cands[0]; key = (w['buildingId'], int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0])) / 10)))
        s = c['score'] * w['lengthM']
        if key not in best or s > best[key][0]: best[key] = (s, w, c)
    hand = []
    for f in sorted((pathlib.Path(__file__).resolve().parent / 'pipeline').glob('frontage-observations*.json')) + [pathlib.Path(__file__).resolve().parent / 'pipeline/facade-profiles.json']:
        if f.resolve() == a.out.resolve(): continue
        hand += json.load(open(f)).get('profiles', [])
    def covered(bid, normal):
        fid = re.sub(r'_p\d+$', '', bid)
        return any(p['match'].get('featureId') == fid and np.dot(normal, p['normal']) > .95 for p in hand)
    chosen = sorted([v for v in best.values() if not covered(v[1]['buildingId'], v[1]['normal'])], key=lambda x: -x[0])[:a.max_walls]
    seg = Segmenter(); det = Detector(); profiles = []; pano_cache = {}
    stats = {'facings': len(chosen), 'read': 0, 'blank': 0, 'materialOnly': 0, 'windows': 0, 'signs': 0}
    for i, (score, w, c) in enumerate(chosen):
        cam = cams[c['image']]; bid = w['buildingId']; z0 = buildings[bid]['center'][2]
        if c['image'] not in pano_cache:
            if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
            pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
            road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
            gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
            pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
        img, H, L = rectify(w, cam, z0, pano_cache[c['image']]); reading = read(img, H, L)
        # Detections are cached per facing and image: changing the layout rules replays them in seconds.
        ck = a.cache / f"{re.sub(r'[^A-Za-z0-9_]', '', bid)}_{int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0]))))%360}_{c['image']}"
        if (ck.with_suffix('.json')).exists():
            j = json.loads(ck.with_suffix('.json').read_text()); occ = np.load(ck.with_suffix('.npz'))['occ']
            found = ([tuple(o) for o in j['ops']], [tuple(o) for o in j['signs']], occ, j['visible'], (img * 255).astype(np.uint8), [tuple(g) for g in j['glass']])
        else:
            found = detect(img, seg, det)
            ck.with_suffix('.json').write_text(json.dumps({'ops': [list(map(lambda v: v if isinstance(v, str) else float(v), o)) for o in found[0]], 'signs': [list(map(lambda v: v if isinstance(v, str) else float(v), o)) for o in found[1]],
                                                          'visible': found[3], 'glass': [list(map(float, g)) for g in found[5]]}))
            np.savez_compressed(ck.with_suffix('.npz'), occ=found[2])
        p = profile(bid, buildings[bid], w, img, reading, c['image'], found)
        p['camera'] = {'image': c['image'], 'distanceM': round(c['distanceM'], 1), 'incidence': round(c['incidence'], 2)}; profiles.append(p)
        stats['read'] += 1; stats['materialOnly'] += bool(p.get('materialOnly')); stats['blank'] += bool(p.get('reading', {}).get('blank'))
        stats['windows'] += len([o for o in p.get('openings', []) if o['style']['kind'] == 'window']); stats['signs'] += len(p.get('panels', []))
        print(i, bid, round(__import__('time').time() % 100000, 1), p.get('reading'), flush=True)
    a.out.write_text(json.dumps({'schemaVersion': 1, 'method': 'automatic rectified reading, detected layout (auto_read_frontages_v27)', 'profiles': profiles, 'stats': stats}, indent=1,
                                default=lambda v: v.item() if hasattr(v, 'item') else list(v) if isinstance(v, tuple) else str(v)))
    print(json.dumps(stats))


if __name__ == '__main__': main()
