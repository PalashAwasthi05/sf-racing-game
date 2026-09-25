"""Frontage reading v31: v27's detected layout, plus what makes a facade look like itself in the photograph and was
being lost to one wall colour: colour zones by height (a dark upper storey over a pale shopfront storey), trims (piers
between bays, string courses, fascia bands, a cornice) and sign boards above the ground storey.

Everything is measured on the rectified wall (white-balanced against the road as in v27) after masking what is not
wall: detected openings, glazing and signs (with a margin), trees, vehicles, people and sky.
- Zones: the wall's median Lab colour in 0.5 m rows, split where a row differs from the running zone by more than
  dE 12; zones under 1 m join their closer neighbour; emitted when two neighbouring zones differ by dE 14 or more.
  The zone with most wall above the ground storey is the body (the wall's material colour); the others are panels.
- Piers: 0.25 m columns inside the body zone that are mostly wall (not window) and differ from the body by dE 16 or
  more in at least 85 % of the zone's rows (spandrels break a column of windows), away from the wall ends; runs
  0.35-1.6 m wide (a pole is thinner) become piers over the zone's height, at most six, strongest first.
- Bands: 0.2 m rows that are mostly wall across the width and differ from their zone by dE 16 or more; runs 0.2-1.0 m
  tall become projecting bands (the topmost within 1.2 m of the wall top is a cornice).
- Signs: sign boxes the detector found (up to 4.5 m above the ground storey) become panels in their colour, unless
  their colour is the wall's (lettering on the wall, not a board).
Components carry materials named `recipe#rrggbb`: the family is read from the colour as the wall's is, the recipe
keeps its surface (joints, relief, roughness) and takes the measured colour.
Run like v27 (same cache, no model call when the detections are cached):
  auto_read_frontages_v31.py --candidate work/<dir> --out work/pipeline/frontage-observations-auto.json"""
import sys
import numpy as np
import auto_read_frontages_v27 as R27
from auto_read_frontages_v24 import family, PPM
from judge_facades_v27 import to_lab

FAMILIES = {'brick': {'default': 'brick_modular', 'dark': 'brick_dark'}, 'render': {'default': 'stucco', 'red': 'stucco_red', 'salmon': 'render_salmon', 'gray': 'render_gray', 'white': 'render_white'},
            'metal-panel': {'default': 'panel_silver', 'dark': 'panel_dark'}, 'terracotta': {'default': 'terracotta_panel'}, 'timber': {'default': 'timber_dark'}, 'pale-trim': {'default': 'stone'}}
ZONE_DE, ZONE_SPLIT_DE, TRIM_DE = 12., 14., 16.


def material_for(rgb):
    fam, fin = family(np.asarray(rgb, float) / 255.)
    base = FAMILIES.get(fam, FAMILIES['render']); base = base.get(fin, base['default'])
    r, g, b = (int(np.clip(round(v), 0, 255)) for v in rgb)
    return f'{base}#{r:02x}{g:02x}{b:02x}'


def lab_to_rgb(lab):
    L, a, b = lab; fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    f = lambda t: np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))
    xyz = np.array([f(fx) * .9505, f(fy), f(fz) * 1.089])
    M = np.array([[3.2406, -1.5372, -.4986], [-.9689, 1.8758, .0415], [.0557, -.2040, 1.0570]])
    lin = np.clip(M @ xyz, 0, 1); srgb = np.where(lin <= .0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - .055)
    return srgb * 255


def de(p, q): return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))


def wall_mask(u8, occ, p, ops, signs, glass, H, L):
    Hp, Wp = u8.shape[:2]; M = ~occ.astype(bool) if occ is not None and occ.shape == (Hp, Wp) else np.ones((Hp, Wp), bool)
    r, g, b = (u8[..., k].astype(int) for k in range(3)); M &= ~((b > r + 20) & (b > 140))                    # sky
    def cut(x0, z0, x1, z1, pad=.15):
        c0, c1 = int(max(0, (x0 - pad) * PPM)), int(min(Wp, (x1 + pad) * PPM)); r0, r1 = int(max(0, (H - z1 - pad) * PPM)), int(min(Hp, (H - z0 + pad) * PPM))
        if c1 > c0 and r1 > r0: M[r0:r1, c0:c1] = False
    for o in list(ops) + list(signs) + [g_[:4] for g_ in glass]: cut(*o[:4])
    for op in p.get('openings', []):
        x0, z0, x1, z1 = op['rect']; cut(x0 * L, z0 * H, x1 * L, z1 * H)
    for grp in p.get('groups', []):
        if grp.get('style', {}).get('openingRatio', 0) > .9 and 'region' in grp:
            x0, z0, x1, z1 = grp['region']; cut(x0 * L, z0 * H, x1 * L, z1 * H, 0.)
    return M


def zones_of(lab, M, H):
    rows = []
    for z0 in np.arange(1., H, .5):                         # the first metre is kerb-side clutter, shadow and plinth
        z1 = min(H, z0 + .5); r0, r1 = int((H - z1) * PPM), int((H - z0) * PPM); m = M[r0:r1]
        if m.size and m.mean() > .4 and m.sum() > 40: rows.append([z0, z1, np.median(lab[r0:r1][m], 0), int(m.sum())])
    if not rows: return []
    zones = [[rows[0][0], rows[0][1], [rows[0][2]], rows[0][3]]]
    for z0, z1, c, n in rows[1:]:
        cur = zones[-1]
        if de(c, np.median(cur[2], 0)) < ZONE_DE and z0 - cur[1] < 1.01: cur[1] = z1; cur[2].append(c); cur[3] += n
        else: zones.append([z0, z1, [c], n])
    zs = [[a, b, np.median(cs, 0), n] for a, b, cs, n in zones]
    changed = True
    while changed and len(zs) > 1:                        # thin zones join the closer neighbour; alike neighbours merge
        changed = False
        for i, z in enumerate(zs):
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(zs)]
            j = min(nb, key=lambda j: de(zs[j][2], z[2]))
            if z[1] - z[0] < 2. or de(zs[j][2], z[2]) < ZONE_DE:
                a, b = sorted((i, j)); w = (zs[a][3], zs[b][3]); c = (zs[a][2] * w[0] + zs[b][2] * w[1]) / max(sum(w), 1)
                zs[a:b + 1] = [[zs[a][0], zs[b][1], c, sum(w)]]; changed = True; break
    while len(zs) > 3:                                      # at most three zones: the closest neighbours merge first
        i = min(range(len(zs) - 1), key=lambda i: de(zs[i][2], zs[i + 1][2])); a, b = zs[i], zs[i + 1]; w = (a[3], b[3])
        zs[i:i + 2] = [[a[0], b[1], (a[2] * w[0] + b[2] * w[1]) / max(sum(w), 1), sum(w)]]
    if zs: zs[0][0] = 0.                                    # the lowest zone reaches the ground
    return zs


def guard_dark(p):
    """A near-black wall reading (L* under 22) is shade or occlusion (trees, a dark street), not paint: it proves neither
    the wall's colour nor that the wall is blank. The reading keeps its layout; the building's other evidence colours it."""
    ev = p.get('materialEvidence')
    if not ev or not ev.get('colorRgb'): return p
    if to_lab(np.asarray(ev['colorRgb'], float)[None, None])[0, 0][0] >= 22.: return p
    p.pop('materialEvidence', None); p.setdefault('reading', {})['darkPhoto'] = True
    if p.get('reading', {}).get('blank') and not p.get('openings') and not p.get('groups'):
        p['materialOnly'] = True; p['reading'].pop('blank', None)
    return p


def enrich(p, found, wall):
    if p.get('materialOnly'): return guard_dark(p)
    ops, signs, occ, visible, u8, glass = found
    H = float(wall['heightM']); L = float(wall['lengthM']); gh = float(p.get('reading', {}).get('groundStoreyM') or min(4.5, H * .35))
    M = wall_mask(u8, occ, p, ops, signs, glass, H, L)
    lab = to_lab(u8.astype(float))
    add = {'zones': [], 'piers': [], 'bands': [], 'signs': [], 'surrounds': []}
    zs = zones_of(lab, M, H)
    if not zs: return guard_dark(p)
    body = max(zs, key=lambda z: (max(0., min(z[1], H) - max(z[0], gh)), z[3]))
    brgb = lab_to_rgb(body[2])
    DARK = 22.                                                # L*: below this a reading is shade or occlusion, not paint
    if len(zs) >= 2 and max(de(a[2], b[2]) for a, b in zip(zs, zs[1:])) >= ZONE_SPLIT_DE and all(z[2][0] >= DARK for z in zs):
        zone_panels = [{'rect': [0., round(z[0] / H, 4), 1., round(min(z[1], H) / H, 4)], 'material': material_for(lab_to_rgb(z[2])), 'evidence': 'image-measured colour zone'}
                       for z in zs if z is not body]
        p['panels'] = zone_panels + p.get('panels', [])          # sign panels stay on top (the last panel wins)
        add['zones'] = [(round(z[0], 1), round(min(z[1], H), 1)) for z in zs]
    # the wall's own colour is the body zone's (masked median above the ground storey), not the dominant cluster
    ev = p.get('materialEvidence')
    if ev is not None:
        old_l = to_lab(np.asarray(ev.get('colorRgb') or brgb, float)[None, None])[0, 0][0]
        support = body[3] / max(1, (min(body[1], H) - body[0]) * L * PPM * PPM)
        if body[2][0] >= DARK and support >= .35 and body[2][0] >= old_l - 15:
            ev['colorRgb'] = [int(v) for v in np.clip(brgb, 0, 255)]; ev['family'], ev['finish'] = family(np.asarray(brgb, float) / 255.)
    guard_dark(p)
    # piers inside the body zone
    zb0, zb1 = max(body[0], gh if body[1] - gh > 2 else body[0]), min(body[1], H)
    r0, r1 = int((H - zb1) * PPM), int((H - zb0) * PPM); cols = []
    if r1 - r0 >= 3 * PPM:
        for x0 in np.arange(0., L - .2, .25):
            c0, c1 = int(x0 * PPM), int(min(L, x0 + .25) * PPM); m = M[r0:r1, c0:c1]
            if m.size == 0 or m.mean() < .5: cols.append(None); continue
            rows_ok = []
            for rr in range(r0, r1, int(.5 * PPM)):
                mm = M[rr:rr + int(.5 * PPM), c0:c1]
                if mm.sum() > 3: rows_ok.append(de(np.median(lab[rr:rr + int(.5 * PPM), c0:c1][mm], 0), body[2]) >= TRIM_DE)
            c = np.median(lab[r0:r1, c0:c1][m], 0)
            cols.append((x0, c) if rows_ok and np.mean(rows_ok) >= .85 and .4 <= x0 <= L - .65 else None)   # spandrels break a window column; wall ends are corners
        runs, run, gap = [], [], 0
        for item in cols + [None, None]:
            if item is not None and (not run or de(item[1], run[-1][1]) < ZONE_DE * 1.5): run.append(item); gap = 0; continue
            if item is None and run and gap == 0: gap = 1; continue                  # a one-column gap does not end a pier
            if run: runs.append(run)
            run = [item] if item is not None else []; gap = 0
        found_p = []
        for run in runs:
            x0, x1 = run[0][0], run[-1][0] + .25; w = x1 - x0
            if .35 <= w <= 1.6:                                               # a pole or a frame is thinner
                c = np.median([q[1] for q in run], 0); found_p.append((de(c, body[2]), (x0 + x1) / 2, w, c))
        for strength, x, w, c in sorted(found_p, key=lambda t: -t[0])[:6]:
            p.setdefault('piers', []).append({'x': round(x / L, 4), 'widthM': round(w, 2), 'projectionM': .1, 'y': [round(zb0 / H, 4), round(zb1 / H, 4)],
                                              'material': material_for(lab_to_rgb(c)), 'evidence': 'image-measured trim'})
            add['piers'].append(round(x, 1))
    # bands: thin full-width rows contrasting with their zone
    zone_of = lambda z: next((q for q in zs if q[0] <= z < q[1]), body)
    cand = []
    for z0 in np.arange(0., H - .1, .2):
        z1 = min(H, z0 + .2); ra, rb = int((H - z1) * PPM), int((H - z0) * PPM); m = M[ra:rb]
        if m.size == 0 or m.mean() < .55: cand.append(None); continue
        c = np.median(lab[ra:rb][m], 0); cand.append((z0, c) if de(c, zone_of(z0 + .1)[2]) >= TRIM_DE else None)
    run = []
    for item in cand + [None]:
        if item is not None and (not run or de(item[1], run[-1][1]) < ZONE_DE): run.append(item); continue
        if run:
            h = len(run) * .2; zc = run[0][0] + h / 2
            if .2 <= h <= 1.0 and zc > .8 and len(add['bands']) < 6:
                top = zc > H - 1.2; c = np.median([q[1] for q in run], 0)
                p.setdefault('bands', []).append({'height': round(zc / H, 4), 'thicknessM': round(h, 2), 'depthM': .35 if top else .25, 'projectionM': .25 if top else .08,
                                                  'material': material_for(lab_to_rgb(c)), 'evidence': 'image-measured cornice' if top else 'image-measured band'})
                add['bands'].append(round(zc, 1))
        run = [item] if item is not None else []
    # Window surrounds: rings 0.05-0.45 m outside each detected window above the ground storey, compared with the body.
    # A ring that contrasts with the body (dE 16 or more) while the ring beyond it matches is the window's frame or
    # surround; when most windows show one, the windows take its colour and width (a thick dark surround round each bay
    # read as thin generic frames).
    Hp, Wp = u8.shape[:2]; widths, colours, n_win = [], [], 0
    for o in ops:
        if o[4] != 'window' or o[1] < gh or o[2] - o[0] < .6 or o[3] - o[1] < .6: continue
        n_win += 1; rings = []
        for d0, d1 in ((.05, .15), (.15, .3), (.3, .45), (.45, .7)):
            mk = np.zeros((Hp, Wp), bool)
            def rc(x0, z0, x1, z1): return (int(max(0, (H - z1) * PPM)), int(min(Hp, (H - z0) * PPM)), int(max(0, x0 * PPM)), int(min(Wp, x1 * PPM)))
            a0, a1, b0, b1 = rc(o[0] - d1, o[1] - d1, o[2] + d1, o[3] + d1); mk[a0:a1, b0:b1] = True
            a0, a1, b0, b1 = rc(o[0] - d0, o[1] - d0, o[2] + d0, o[3] + d0); mk[a0:a1, b0:b1] = False
            mk &= ~occ.astype(bool) if occ is not None and occ.shape == (Hp, Wp) else mk
            rings.append(np.median(lab[mk], 0) if mk.sum() > 12 else None)
        if any(r is None for r in rings): continue
        contrast = [de(r, body[2]) >= TRIM_DE for r in rings]
        if contrast[0] and not contrast[-1]:
            k = contrast.index(False) if False in contrast else len(contrast); widths.append((.15, .3, .45, .7)[k - 1]); colours.append(np.median(rings[:k], 0))
    if n_win >= 3 and len(widths) >= .6 * n_win:
        fw = float(np.median(widths)); fc_ = material_for(lab_to_rgb(np.median(colours, 0)))
        for op in p.get('openings', []):
            if op['style'].get('kind') == 'window': op['style'].update(frameMaterial=fc_, frameWidthM=round(fw, 2))
        for grp in p.get('groups', []): grp.setdefault('style', {}).update(frameMaterial=fc_, frameWidthM=round(fw, 2))
        p['windowFrame'] = {'frameMaterial': fc_, 'frameWidthM': round(fw, 2), 'windows': len(widths), 'of': n_win}
        add['surrounds'] = [round(fw, 2)]
    # Sign boards on their own boxes. v27 clamped every sign to the ground storey's top (gh - 1.2 to gh + 0.9), which
    # inverted the rectangle of any sign higher up (a board over a two-storey shop, a painted panel): those drew as
    # nothing. Rebuilt here from the detections, up to 4.5 m above the ground storey.
    p['panels'] = [q for q in p.get('panels', []) if q.get('evidence') != 'image-detected sign']
    for s in signs:
        a, z0, b, z1, kind = s
        if kind == 'awning' or z1 < 1.2 or z0 > gh + 4.5 or (b - a) < .8 or (b - a) * (z1 - z0) < .6: continue
        if (b - a) * (z1 - z0) > .25 * L * H or z1 - z0 > 3.: continue            # a quarter of the wall is its architecture (relief lettering), not a board
        rgb = R27.colour_of(u8, s, H); z1 = min(z1, H)
        if z1 - z0 < .2: continue
        if de(to_lab(np.asarray(rgb, float)[None, None])[0, 0], body[2]) < 15: continue          # lettering on the wall, not a board
        p['panels'].append({'rect': [round(a / L, 4), round(z0 / H, 4), round(b / L, 4), round(z1 / H, 4)], 'material': f"panel_dark#{'%02x%02x%02x' % tuple(int(v) for v in rgb)}",
                            'evidence': 'image-detected sign'})
        add['signs'].append(round(z0, 1))
    p['status'] = 'image-measured-auto-v31'
    p.setdefault('reading', {})['v31'] = {k: v for k, v in add.items() if v}
    return p


_profile27 = R27.profile
def profile(bid, building, wall, img, reading, cam_image, found):
    return enrich(_profile27(bid, building, wall, img, reading, cam_image, found), found, wall)


if __name__ == '__main__' and '--supplements' not in sys.argv and '--enrich-existing' not in sys.argv:
    R27.profile = profile
    R27.main()


def supplements(candidate, out, cache):
    """Hand-read frontages keep their layout and material; from the photograph of the wall each one draws, add the
    colour zones, trims and sign boards they lack. Written as supplements (heights in the hand reading's frame) that the
    generator merges only where the hand reading has none of that kind."""
    import json, math, pathlib, re
    import frontage_crops_v15 as fc
    from auto_read_frontages_v24 import rectify
    from facade_segment_v27 import Segmenter, Detector
    world = json.load(open(candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}; ev = json.load(open(candidate / 'public/facade-evidence.json'))['walls']
    root = pathlib.Path(__file__).resolve().parent
    hand = {p['id']: p for f in sorted((root / 'pipeline').glob('frontage-observations*.json')) if 'auto' not in f.name for p in json.load(open(f)).get('profiles', [])}
    walls = {}
    for w in ev:
        g = w.get('grammarId')
        if g in hand and w.get('lengthM', 0) >= 4 and w['buildingId'] in buildings and not buildings[w['buildingId']].get('landmark'):
            if g not in walls or w['lengthM'] > walls[g]['lengthM']: walls[g] = w
    occ_fp = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    seg = det = None; outp = []; pano_cache = {}
    for gid, w in sorted(walls.items()):
        cands = fc.wall_candidates(w, cams, occ_fp, trees, fc.MAX_DISTANCE_M)
        if not cands: continue
        c = cands[0]; cam = cams[c['image']]; bid = w['buildingId']; z0 = buildings[bid]['center'][2]
        if c['image'] not in pano_cache:
            if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
            pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
            road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
            gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
            pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
        img, H, L = rectify(w, cam, z0, pano_cache[c['image']])
        ck = cache / f"sup_{re.sub(r'[^A-Za-z0-9_]', '', gid)}_{c['image']}"
        if ck.with_suffix('.json').exists():
            j = json.loads(ck.with_suffix('.json').read_text()); occ = np.load(ck.with_suffix('.npz'))['occ']
            found = ([tuple(o) for o in j['ops']], [tuple(o) for o in j['signs']], occ, j['visible'], (img * 255).astype(np.uint8), [tuple(g) for g in j['glass']])
        else:
            if seg is None: seg, det = Segmenter(), Detector()
            found = R27.detect(img, seg, det)
            ck.with_suffix('.json').write_text(json.dumps({'ops': [[v if isinstance(v, str) else float(v) for v in o] for o in found[0]], 'signs': [[v if isinstance(v, str) else float(v) for v in o] for o in found[1]],
                                                          'visible': found[3], 'glass': [list(map(float, g)) for g in found[5]]}))
            np.savez_compressed(ck.with_suffix('.npz'), occ=found[2])
        if found[3] < .4: continue
        gh = min(4.5, H * .35)
        p = enrich({'openings': [], 'groups': [], 'panels': [], 'reading': {'groundStoreyM': gh}}, found, w)
        ref = float(hand[gid].get('heightRefM') or H); k = H / ref
        def zf(v): return round(float(v) * k, 4)
        sup = {'supplementOf': gid, 'wallMidpoint': [round(v, 2) for v in w['midpoint']], 'camera': c['image'],
               'panels': [{**q, 'rect': [q['rect'][0], zf(q['rect'][1]), q['rect'][2], zf(q['rect'][3])]} for q in p.get('panels', [])],
               'piers': [{**q, 'y': [zf(q['y'][0]), zf(q['y'][1])]} for q in p.get('piers', [])],
               'bands': [{**q, 'height': zf(q['height'])} for q in p.get('bands', [])], 'reading': p.get('reading', {}).get('v31', {})}
        if p.get('windowFrame'): sup['windowFrame'] = p['windowFrame']
        if sup['panels'] or sup['piers'] or sup['bands'] or sup.get('windowFrame'): outp.append(sup)
        print(gid, sup['reading'], flush=True)
    out.write_text(json.dumps({'schemaVersion': 1, 'method': 'auto_read_frontages_v31.py --supplements: colour zones, trims and sign boards measured on the photograph of the wall each hand reading draws; merged only where the reading has none of that kind', 'supplements': outp}, indent=1,
                              default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    print(len(outp), 'supplements')


if __name__ == '__main__' and '--supplements' in sys.argv:
    import pathlib
    i = sys.argv.index('--supplements'); out = pathlib.Path(sys.argv[i + 1]); cand = pathlib.Path(sys.argv[sys.argv.index('--candidate') + 1])
    supplements(cand, out, pathlib.Path('work/frontage-reading-cache-v27')); sys.exit(0)


def enrich_existing(candidate, path, cache):
    """Add v31's components to an existing reading file without changing which facings are read or their colours:
    for each reading, the wall it draws in `candidate` is rectified from the reading's own camera, the cached
    detections are replayed, and only colour zones, piers, bands, window surrounds and sign boards are merged (v27's
    sign panels, clamped to the ground storey and inverted above it, are replaced by boards on their own boxes)."""
    import json, math, pathlib, re, copy
    import frontage_crops_v15 as fc
    from auto_read_frontages_v24 import rectify
    world = json.load(open(candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}; ev = json.load(open(candidate / 'public/facade-evidence.json'))['walls']
    data = json.load(open(path)); cams = fc.load_cameras(frame); pano_cache = {}; done = 0
    for p in data['profiles']:
        cam_id = (p.get('camera') or {}).get('image')
        drawn = [w for w in ev if w.get('grammarId') == p['id']]
        if not cam_id or not drawn or p.get('materialOnly'): continue
        w = max(drawn, key=lambda w: w['lengthM']); bid = w['buildingId']
        if bid not in buildings: continue
        ck = cache / f"{re.sub(r'[^A-Za-z0-9_]', '', bid)}_{int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0]))))%360}_{cam_id}"
        if not ck.with_suffix('.json').exists(): continue
        cam = cams[cam_id]
        if cam_id not in pano_cache:
            if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
            pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
            road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
            gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
            pano_cache[cam_id] = np.clip(pano * gain, 0, 255).astype(np.uint8)
        img, H, L = rectify(w, cam, buildings[bid]['center'][2], pano_cache[cam_id])
        j = json.loads(ck.with_suffix('.json').read_text()); occ = np.load(ck.with_suffix('.npz'))['occ']
        found = ([tuple(o) for o in j['ops']], [tuple(o) for o in j['signs']], occ, j['visible'], (img * 255).astype(np.uint8), [tuple(g) for g in j['glass']])
        q = enrich(copy.deepcopy(p), found, w)
        # frame: enrich measured on the drawn wall; a reading framed on another length keeps its own x fractions only
        # when the drawn wall is its frame (one wall), which is the case for readings made on this wall
        keep_colour = p.get('materialEvidence')
        p['panels'] = [x for x in q.get('panels', []) if 'zone' in x.get('evidence', '') or 'sign' in x.get('evidence', '')] + [x for x in p.get('panels', []) if 'sign' not in str(x.get('evidence', '')) and 'zone' not in str(x.get('evidence', ''))]
        for k in ('piers', 'bands'):
            extra = [x for x in q.get(k, []) if 'image-measured' in x.get('evidence', '')]
            if extra: p[k] = [x for x in p.get(k, []) if 'image-measured' not in str(x.get('evidence', ''))] + extra
        if q.get('windowFrame'): p['windowFrame'] = q['windowFrame']; p['openings'] = q['openings']; p['groups'] = q['groups']
        if keep_colour is not None: p['materialEvidence'] = keep_colour
        p.setdefault('reading', {})['v31'] = q.get('reading', {}).get('v31', {}); done += 1
    data['method'] = data.get('method', '') + '; v31 components merged (auto_read_frontages_v31.py --enrich-existing)'
    path.write_text(json.dumps(data, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else list(v) if isinstance(v, tuple) else str(v))); print(done, 'readings enriched')


if __name__ == '__main__' and '--enrich-existing' in sys.argv:
    import pathlib
    enrich_existing(pathlib.Path(sys.argv[sys.argv.index('--candidate') + 1]), pathlib.Path(sys.argv[sys.argv.index('--enrich-existing') + 1]), pathlib.Path('work/frontage-reading-cache-v27')); sys.exit(0)
