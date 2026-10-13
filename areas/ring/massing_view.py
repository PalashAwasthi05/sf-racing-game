"""The massing of a ring building against the LiDAR, for reading and checking its shape: its walls' height, its roof
form, its parts. Everything is in the world's local frame (metres east and north of the world centre) and in metres
above the part's ground (its foot, z0, as the generator stands it).
Usage (run from work/):
  massing_view.py plan <building> <out.png> [--margin 12]
      top-down: the LiDAR's height above ground (0.5 m cells, the highest return), the model's current parts (outline,
      wall top, roof form) and the side ids of the readings along its walls, neighbours, a 5 m grid with coordinates
  massing_view.py section <building> <x0> <y0> <x1> <y1> <out.png> [--width 1] [--spec spec.json]
      the returns within --width of a line, height above the building's foot against distance along it, with the
      model's current roof (red) and a given massing (green: --spec, else areas/ring/massing-overrides.json)
  massing_view.py stats <building>
      per current part: area, foot, wall top, the returns' heights above the foot (percentiles), the roof form the
      generator's fit reports for them
  massing_view.py fit <building>
      per current part, the roof forms (flat, gable either way, hip, mansard, shed four ways) with the wall top and ridge
      that fit its LiDAR best, scored (share of its roof within 0.6 m, less twice the share built above what the LiDAR sees)
  massing_view.py split <building> <ax> <ay> <bx> <by> [<ax> <ay> <bx> <by> ...]
      the building's whole outline cut by lines through those point pairs: the pieces' outlines (for `parts`)
  massing_view.py rect <cx> <cy> <width> <depth> <angleDeg>
      a rectangle's outline (for a building the model is missing)
  massing_view.py check <building> [<spec.json>] [--out check.png]
      a given massing (the building's entry of areas/ring/massing-overrides.json, or a file holding one) against the
      LiDAR: per part, how much of its roof the returns put within 0.6 m of it, above it, and below it (a wall or roof
      built where the LiDAR sees only lower things fails), and a plan coloured by the difference
  massing_view.py bumps <building> [--rise 0.5]
      what stands on the roof as built: the returns more than --rise above the model's roof, gathered into clusters
      (a chimney, a dormer, a penthouse, plant, or a tree hanging over): per cluster its centre, size and bearing, its
      top above the foot and its rise above the roof, how many returns, how far it is inside the outline
The current parts come from the latest ring build's massing-dump.json (areas/pacific-heights/<build>-raw)."""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point, Polygon, box
from shapely import contains_xy
WORK = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(WORK)); os.environ.setdefault('SFW_AREA', str(WORK / 'areas/pacific-heights'))
import frontage_crops_v15 as fc
from pipeline import roof_forms

AREA = Path(os.environ['SFW_AREA']); RING = WORK / 'areas/ring'
FONT = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13); SMALL = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 11)
L = None


def lidar():
    global L
    if L is None: L = dict(np.load(RING / 'lidar-ring.npz'))
    return L


def dump():
    """The latest ring build's parts: {id: record}."""
    files = sorted(glob.glob(str(AREA / 'ring*-raw/massing-dump.json')), key=os.path.getmtime)
    if not files: sys.exit('no massing-dump.json: run a ring build first')
    return {r['id']: r for r in json.load(open(files[-1]))}, files[-1]


def parts_of(b, D):
    src = b.split('_p')[0]
    out = [r for r in D.values() if r['sourceId'] == src or r['id'] == b]
    if not out: sys.exit(f'{b}: not in the latest ring build')
    return out


def poly(rec): return Polygon(rec['outline'][0], rec['outline'][1:]).buffer(0) if len(rec['outline']) else Polygon()


def foot(P):
    """The building's foot: the terrain at its whole outline's centre, where the generator stands every part of a given
    massing (all heights here are metres above it)."""
    from shapely.ops import unary_union
    c = unary_union([poly(r) for r in P]).centroid; g = fc.ground_at(np.asarray(c.coords[0]))
    return float(g) if g is not None else float(P[0]['z0'])


def load_spec(b, specfile):
    """A given massing: the agent's spec file ({"spec": ..., "adds": [...]}), a bare spec, or the overrides entry."""
    j = json.load(open(specfile)) if specfile else {'spec': json.load(open(RING / 'massing-overrides.json')).get(b.split('_p')[0])}
    if not isinstance(j, dict): return None, []
    sp = j['spec'] if 'spec' in j else j
    return sp, j.get('adds') or []


def roof_z(rec, x, y):
    """The current model's roof height (absolute) at x, y: its patches where they cover, else its base roof."""
    z = np.full(len(x), rec['baseRoofZ'], float)
    for p in rec['patches']: z = np.maximum(np.min([a * x + b * y + c for a, b, c in p['planes']], axis=0), rec['z0'])
    return z


def spec_surface(spec, part_poly, z0, x, y):
    """A given part's top (absolute) at x, y: its roof form over its wall top."""
    wall = float(spec.get('wallTopM', 0.)); eave = z0 + wall; z = np.full(len(x), eave)
    patches, _ = roof_forms.build(part_poly, {**spec.get('roof', {})}, eave, wall)
    for p in patches:
        inside = contains_xy(p['geometry'], x, y)
        z = np.where(inside, np.min([a * x + b * y + c for a, b, c in p['model']['planes']], axis=0), z)
    return z


def dsm(x0, y0, x1, y1, cell=.5):
    """The highest return per cell (absolute z) and the ground under it, over a box."""
    d = lidar(); m = (d['x'] > x0) & (d['x'] < x1) & (d['y'] > y0) & (d['y'] < y1) & (d['cls'] != 2)
    nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1; Z = np.full((ny, nx), -np.inf, np.float32)
    ix = ((d['x'][m] - x0) / cell).astype(int); iy = ((y1 - d['y'][m]) / cell).astype(int)
    np.maximum.at(Z, (iy, ix), d['z'][m]); H = np.full((ny, nx), -np.inf, np.float32); np.maximum.at(H, (iy, ix), d['hag'][m])
    return Z, H


def cmap(h):
    """Height above ground to colour: grey ground, then blue - green - yellow - red - magenta up to 40 m."""
    stops = [(0, (70, 70, 74)), (1.5, (90, 90, 96)), (3, (40, 90, 200)), (8, (30, 170, 170)), (14, (60, 180, 60)), (20, (230, 210, 40)), (28, (230, 120, 30)), (36, (200, 40, 40)), (46, (190, 60, 200))]
    h = np.nan_to_num(h, nan=-1, neginf=-1); out = np.zeros(h.shape + (3,), np.uint8); out[:] = (24, 24, 28)
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        m = (h >= a) & (h < b); t = ((h - a) / (b - a))[m][:, None]; out[m] = (np.array(ca) * (1 - t) + np.array(cb) * t).astype(np.uint8)
    out[h >= stops[-1][0]] = stops[-1][1]; return out


def plan(b, outp, margin):
    D, src = dump(); P = parts_of(b, D); world, frame = fc.load_world(); fps = fc.load_footprints(frame); z0 = foot(P)
    U = Polygon();
    for r in P: U = U.union(poly(r))
    x0, y0, x1, y1 = U.buffer(margin).bounds; cell = .5
    _, H = dsm(x0, y0, x1, y1, cell)
    from scipy.ndimage import maximum_filter
    H = np.where(np.isfinite(H), H, maximum_filter(np.where(np.isfinite(H), H, -1e9), 3)); H[H < -1e8] = -np.inf   # one-cell gaps between returns
    # hill-shading from the smoothed heights: slopes, ridges, hips and steps read where the colour alone is one band
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(np.where(np.isfinite(H), H, 0.), .8); gy, gx = np.gradient(Hs, cell); nz = 1 / np.sqrt(1 + gx ** 2 + gy ** 2)
    shade = np.clip(.55 + .45 * ((-gx * .5 + gy * .6) * nz + nz * .62), .35, 1.25)                      # light from the north-west
    rgb = np.clip(cmap(H).astype(float) * shade[..., None], 0, 255).astype(np.uint8)
    k = max(2, int(round(1000 / H.shape[1]))); img = Image.fromarray(rgb).resize((H.shape[1] * k, H.shape[0] * k), Image.NEAREST)
    s = img.width / (x1 - x0); pad = 46
    can = Image.new('RGB', (img.width + 2 * pad + 190, img.height + 2 * pad), (16, 16, 18)); can.paste(img, (pad, pad)); d = ImageDraw.Draw(can)
    P2 = lambda x, y: (pad + (x - x0) * s, pad + (y1 - y) * s)
    for gx in range(int(np.ceil(x0 / 5)) * 5, int(x1) + 1, 5):
        d.line([P2(gx, y0), P2(gx, y1)], fill=(255, 255, 255, 60) if gx % 10 else (200, 200, 200), width=1); d.text((P2(gx, y0)[0] - 10, pad + img.height + 4), f'{gx}', font=SMALL, fill=(220, 220, 220))
    for gy in range(int(np.ceil(y0 / 5)) * 5, int(y1) + 1, 5):
        d.line([P2(x0, gy), P2(x1, gy)], fill=(200, 200, 200) if gy % 10 == 0 else (110, 110, 110), width=1); d.text((4, P2(x0, gy)[1] - 7), f'{gy}', font=SMALL, fill=(220, 220, 220))
    # neighbours (other parts in the dump and the source outlines) thin; this building's parts bold with their heights
    view = box(x0, y0, x1, y1)
    for k, g in fps.items():
        if g.intersects(view) and k != b.split('_p')[0]:
            for q in ([g] if g.geom_type == 'Polygon' else list(g.geoms)): d.line([P2(*c) for c in q.exterior.coords], fill=(150, 150, 150), width=1)
    for r in D.values():
        g = poly(r)
        if r['sourceId'] != b.split('_p')[0] and g.intersects(view): d.line([P2(*c) for c in g.exterior.coords], fill=(235, 235, 235), width=1)
    cols = [(255, 255, 255), (255, 120, 220), (120, 255, 255), (255, 230, 120), (160, 255, 120)]
    for i, r in enumerate(P):
        g = poly(r); col = cols[i % len(cols)]
        for q in ([g] if g.geom_type == 'Polygon' else list(g.geoms)): d.line([P2(*c) for c in q.exterior.coords], fill=col, width=3)
        c = g.representative_point(); forms = ','.join(p['form'] for p in r['patches']) or 'flat'
        d.text(P2(c.x, c.y), f"{r['id'].split('_')[-1] if r['id'] != r['sourceId'] else 'main'}\nwall {r['baseRoofZ'] - z0:.1f} m\n{forms}" + (f"\ntiers {r['tiers']}" if r['tiers'] else ''), font=FONT, fill=col, stroke_width=2, stroke_fill=(0, 0, 0))
    # the readings' sides along the walls
    for p in json.load(open(AREA / 'regen/sides/sides.json'))['pieces']:
        if p['building'].split('_p')[0] != b.split('_p')[0]: continue
        n = np.asarray(p['normal'], float); t = np.array([-n[1], n[0]]); a = n * p['planeV'] + t * p['frameU'][0]; e = n * p['planeV'] + t * p['frameU'][1]
        d.line([P2(*a), P2(*e)], fill=(255, 60, 60), width=2); m = (a + e) / 2 + n * 1.2
        d.text(P2(*m), p['id'].split('__')[1], font=FONT, fill=(255, 90, 90), stroke_width=2, stroke_fill=(0, 0, 0))
    # legend
    lx = pad + img.width + 14
    for k, h in enumerate([0, 3, 6, 9, 12, 15, 20, 25, 30, 36]):
        d.rectangle([lx, pad + k * 22, lx + 20, pad + k * 22 + 18], fill=tuple(int(v) for v in cmap(np.array([h + .01]))[0])); d.text((lx + 26, pad + k * 22 + 2), f'{h} m above ground', font=SMALL, fill=(230, 230, 230))
    d.text((lx, pad + 240), 'bold: this building\'s parts\n(wall top above its foot,\nroof form)\nred: photographed sides\nthin: neighbours\ngrid: 5 m (local x, y)', font=SMALL, fill=(230, 230, 230))
    can.save(outp); print(json.dumps({'image': outp, 'build': src, 'footZ': round(z0, 2), 'note': 'wall tops in metres above the building foot footZ', 'parts': [{'id': r['id'], 'wallTopM': round(r['baseRoofZ'] - z0, 2), 'forms': [p['form'] for p in r['patches']], 'tiers': r['tiers'], 'areaM2': round(poly(r).area, 1)} for r in P]}))


def section(b, a, e, outp, width, specfile=None):
    D, _ = dump(); P = parts_of(b, D); a, e = np.asarray(a, float), np.asarray(e, float); u = (e - a) / np.linalg.norm(e - a); n = np.array([-u[1], u[0]]); Ln = float(np.linalg.norm(e - a))
    d = lidar(); rel = np.c_[d['x'] - a[0], d['y'] - a[1]]; along = rel @ u; off = rel @ n; m = (np.abs(off) < width) & (along > -1) & (along < Ln + 1)
    z0 = foot(P); W, Hh, pad = 1100, 560, 50
    zs = d['z'][m] - z0; top = max(10., float(np.percentile(zs, 99.5)) + 3 if m.any() else 10.)
    can = Image.new('RGB', (W + 2 * pad, Hh + 2 * pad), (250, 250, 250)); dr = ImageDraw.Draw(can)
    X = lambda t: pad + t / Ln * W; Y = lambda h: pad + Hh - (h + 3) / (top + 3) * Hh
    for h in range(-2, int(top) + 1):
        dr.line([X(0), Y(h), X(Ln), Y(h)], fill=(200, 200, 200) if h % 5 else (140, 140, 140)); (h % 5 == 0) and dr.text((8, Y(h) - 7), f'{h}', font=SMALL, fill=(60, 60, 60))
    for t in range(0, int(Ln) + 1): (t % 5 == 0) and dr.text((X(t) - 4, pad + Hh + 6), f'{t}', font=SMALL, fill=(60, 60, 60))
    for t, h, c in zip(along[m], zs, d['cls'][m]): dr.ellipse([X(t) - 1.5, Y(h) - 1.5, X(t) + 1.5, Y(h) + 1.5], fill=(150, 110, 60) if c == 2 else (40, 70, 160))
    ts = np.linspace(0, Ln, 400); xy = a + ts[:, None] * u
    for r in P:
        g = poly(r); inside = contains_xy(g, xy[:, 0], xy[:, 1]); zz = roof_z(r, xy[:, 0], xy[:, 1]) - z0
        for t0, h0, t1, h1, i0, i1 in zip(ts, zz, ts[1:], zz[1:], inside, inside[1:]):
            if i0 and i1: dr.line([X(t0), Y(h0), X(t1), Y(h1)], fill=(220, 40, 40), width=3)
    spec, adds = load_spec(b, specfile) if (specfile or (RING / 'massing-overrides.json').exists()) else ({}, []); spec = spec or {}
    from shapely.ops import unary_union
    whole = Polygon(spec['outline']).buffer(0) if spec.get('outline') else unary_union([poly(r) for r in P])
    for k, sp in enumerate((spec.get('parts') or ([spec] if spec.get('wallTopM') is not None else [])) + adds):
        g = Polygon(sp['outline']).buffer(0) if sp.get('outline') else whole; inside = contains_xy(g, xy[:, 0], xy[:, 1])
        zz = spec_surface(sp, g, z0, xy[:, 0], xy[:, 1]) - z0
        for t0, h0, t1, h1, i0, i1 in zip(ts, zz, ts[1:], zz[1:], inside, inside[1:]):
            if i0 and i1: dr.line([X(t0), Y(h0), X(t1), Y(h1)], fill=(20, 160, 60), width=3)
    dr.text((pad, 12), f'{b}: section ({a[0]:.1f},{a[1]:.1f}) -> ({e[0]:.1f},{e[1]:.1f}), returns within {width} m; blue returns, brown ground; red: model now; green: given massing; heights above the foot z0={z0:.2f}', font=SMALL, fill=(20, 20, 20))
    can.save(outp); print(json.dumps({'image': outp, 'returns': int(m.sum()), 'lengthM': round(Ln, 2)}))


def stats(b):
    D, src = dump(); P = parts_of(b, D); d = lidar(); out = []; z0 = foot(P)
    for r in P:
        g = poly(r); m = contains_xy(g.buffer(-.5), d['x'], d['y']) & (d['cls'] != 2); h = d['z'][m] - z0
        pts = np.c_[d['x'][m], d['y'][m], d['z'][m]]
        fit = roof_forms.fit(g, pts)[1] if len(pts) > 60 else {'status': 'too-few-returns'}
        out.append({'id': r['id'], 'areaM2': round(g.area, 1), 'wallTopM': round(r['baseRoofZ'] - z0, 2), 'forms': [p['form'] for p in r['patches']], 'tiers': r['tiers'],
                    'returns': int(m.sum()), 'heightAboveFootPercentiles': {q: round(float(np.percentile(h, q)), 2) for q in (10, 25, 50, 75, 90, 98)} if m.any() else None, 'fit': fit})
    print(json.dumps({'build': src, 'footZ': round(z0, 2), 'note': 'heights in metres above the building foot footZ', 'parts': out}, indent=1))


def check(b, specfile, outp):
    D, _ = dump(); P = parts_of(b, D); d = lidar(); spec, adds = load_spec(b, specfile)
    if not spec and not adds: sys.exit(f'no given massing for {b}')
    from shapely.ops import unary_union
    spec = spec or {}; whole = Polygon(spec['outline']).buffer(0) if spec.get('outline') else unary_union([poly(r) for r in P]); zf = foot(P) + float(spec.get('baseOffsetM', 0.))
    parts = (spec.get('parts') or ([spec] if (spec.get('wallTopM') is not None or spec.get('roof')) else [])) + [{**a, '_add': True} for a in adds]; rows = []; U = Polygon()
    for k, sp in enumerate(parts):
        g = Polygon(sp['outline']).buffer(0) if sp.get('outline') else whole
        U = U.union(g); z0 = zf if not sp.get('_add') else float(fc.ground_at(np.asarray(g.centroid.coords[0])) or zf)
        x0, y0, x1, y1 = g.bounds; Z, Hh = dsm(x0 - 1, y0 - 1, x1 + 1, y1 + 1, .5)
        ys, xs = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]; cx = x0 - 1 + (xs + .5) * .5; cy = y1 + 1 - (ys + .5) * .5
        inside = contains_xy(g.buffer(-.4), cx, cy) & np.isfinite(Z)
        surf = spec_surface(sp, g, z0, cx[inside], cy[inside]); res = Z[inside] - surf
        rows.append({'part': sp.get('id', k), 'z0': round(z0, 2), 'wallTopM': sp.get('wallTopM'), 'roof': sp.get('roof', {}).get('kind', 'flat'), 'cells': int(inside.sum()),
                     'within0.6': round(float((np.abs(res) < .6).mean()), 3) if len(res) else None, 'above1': round(float((res > 1.).mean()), 3) if len(res) else None,
                     'below1': round(float((res < -1.).mean()), 3) if len(res) else None, 'medianResidualM': round(float(np.median(res)), 2) if len(res) else None,
                     'pass': bool(len(res) and (res < -1.).mean() <= .15)})
    print(json.dumps({'building': b, 'parts': rows, 'note': 'below1 is the share of the part where the LiDAR sees only lower things more than 1 m under the given top: over 15 % fails (a wall or roof without LiDAR evidence); above1 is things above it (trees, plant, a higher roof)'}, indent=1))
    if outp:
        x0, y0, x1, y1 = U.buffer(6).bounds; Z, Hh = dsm(x0, y0, x1, y1, .5); ys, xs = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]; cx = x0 + (xs + .5) * .5; cy = y1 - (ys + .5) * .5
        img = np.zeros(Z.shape + (3,), np.uint8); img[:] = (40, 40, 44)
        for k, sp in enumerate(parts):
            g = Polygon(sp['outline']).buffer(0) if sp.get('outline') else whole; inside = contains_xy(g, cx, cy) & np.isfinite(Z)
            z0 = rows[k]['z0']; res = Z[inside] - spec_surface(sp, g, z0, cx[inside], cy[inside]); t = np.clip(res / 3., -1, 1)
            img[inside] = np.stack([np.where(t > 0, 255, 255 * (1 + t)), 255 * (1 - np.abs(t)), np.where(t < 0, 255, 255 * (1 - t))], 1).astype(np.uint8)
        Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(outp); print(json.dumps({'image': outp, 'legend': 'white: LiDAR within the given top; red: LiDAR above it; blue: LiDAR below it (3 m = full colour)'}))


def fit_part(g, z0, cells=None):
    """The roof forms the LiDAR supports best over a part outline g (foot z0): for each kind, the wall top and ridge
    (metres above the foot) that put most of the part's highest returns (0.5 m cells) within 0.6 m of the roof, less
    twice the share it would build more than 1 m above what the LiDAR sees."""
    x0, y0, x1, y1 = g.bounds; Z, _ = dsm(x0 - 1, y0 - 1, x1 + 1, y1 + 1, .5)
    ys, xs = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]; cx = x0 - 1 + (xs + .5) * .5; cy = y1 + 1 - (ys + .5) * .5
    m = contains_xy(g.buffer(-.4), cx, cy) & np.isfinite(Z)
    if m.sum() < 20: return []
    X, Y, H = cx[m], cy[m], Z[m] - z0; lo, hi = np.percentile(H, [3, 99.5])
    kinds = [('flat', {}), ('gable', {'along': 'long'}), ('gable', {'along': 'short'}), ('hip', {}), ('mansard', {'pitchDeg': 72})] + \
            [('shed', {'lowSide': d}) for d in ([1, 0], [-1, 0], [0, 1], [0, -1])]
    out = []
    for kind, extra in kinds:
        best = None
        for wall in np.arange(max(2., lo), max(2.2, hi) + .01, .2):
            ridges = [None] if kind == 'flat' else np.arange(wall + .6, hi + .41, .2)
            for ridge in ridges:
                spec = {'wallTopM': round(float(wall), 2), 'roof': {'kind': kind, **extra, **({'ridgeM': round(float(ridge), 2)} if ridge is not None else {}), 'overhangM': 0.}}
                r = H - (spec_surface(spec, g, 0., X, Y))
                sc = float((np.abs(r) < .6).mean()) - 2 * float((r < -1.).mean())
                if best is None or sc > best[0]: best = (sc, spec, float((np.abs(r) < .6).mean()), float((r < -1.).mean()), float((r > 1.).mean()))
        if best: out.append({'score': round(best[0], 3), 'within0.6': round(best[2], 3), 'below1': round(best[3], 3), 'above1': round(best[4], 3), 'spec': best[1]})
    return sorted(out, key=lambda o: -o['score'])


def fit(b):
    D, src = dump(); res = []; P = parts_of(b, D); z0 = foot(P)
    for r in P:
        res.append({'id': r['id'], 'areaM2': round(poly(r).area, 1), 'now': {'wallTopM': round(r['baseRoofZ'] - z0, 2), 'forms': [p['form'] for p in r['patches']], 'tiers': r['tiers']},
                    'best': fit_part(poly(r), z0)[:4]})
    print(json.dumps({'building': b, 'build': src, 'footZ': round(z0, 2), 'note': 'heights in metres above the building foot footZ', 'parts': res}, indent=1))


def split(b, cuts):
    """The building's whole outline (its current parts' union) cut by straight lines, each given by two points it
    passes through (extended across the outline): the pieces, for a given massing's `parts`."""
    from shapely.ops import split as _split, unary_union
    D, _ = dump(); U = unary_union([poly(r) for r in parts_of(b, D)]).buffer(0)
    pieces = [U]
    for (ax, ay, bx, by) in cuts:
        a, e = np.array([ax, ay]), np.array([bx, by]); u = (e - a) / np.linalg.norm(e - a); line = LineString([a - u * 500, e + u * 500])
        nxt = []
        for q in pieces: nxt += [g for g in _split(q, line).geoms if g.area > .5]
        pieces = nxt
    out = [{'outline': [[round(x, 2), round(y, 2)] for x, y in np.asarray(q.exterior.coords)], 'areaM2': round(q.area, 1), 'centroid': [round(q.centroid.x, 2), round(q.centroid.y, 2)]} for q in pieces]
    print(json.dumps({'building': b, 'wholeAreaM2': round(U.area, 1), 'pieces': out}))


def rect(cx, cy, w, d, ang):
    """A rectangle w x d centred on (cx, cy), its w side turned ang degrees from east: an outline for a building the
    model is missing."""
    t = np.radians(ang); u = np.array([np.cos(t), np.sin(t)]); n = np.array([-u[1], u[0]]); c = np.array([cx, cy])
    pts = [c - u * w / 2 - n * d / 2, c + u * w / 2 - n * d / 2, c + u * w / 2 + n * d / 2, c - u * w / 2 + n * d / 2]
    print(json.dumps({'outline': [[round(float(x), 2), round(float(y), 2)] for x, y in pts + [pts[0]]], 'areaM2': round(w * d, 1)}))


def bumps(b, rise):
    """Clusters of returns standing more than `rise` above the model's roof, per part (see the usage)."""
    from scipy import ndimage
    from shapely.geometry import MultiPoint
    D, _ = dump(); P = parts_of(b, D); f0 = foot(P); d = lidar(); out = []
    for rec in P:
        g = poly(rec); m = contains_xy(g.buffer(.3), d['x'], d['y']) & (d['cls'] != 2)
        x, y, z = d['x'][m].astype(float), d['y'][m].astype(float), d['z'][m].astype(float)
        if not len(x): continue
        res = z - roof_z(rec, x, y); up = res > rise
        if up.sum() < 3: continue
        x, y, z, res = x[up], y[up], z[up], res[up]; x0, y0 = x.min(), y.min()
        ix = ((x - x0) / .5).astype(int); iy = ((y - y0) / .5).astype(int)
        G = np.zeros((ix.max() + 1, iy.max() + 1), bool); G[ix, iy] = True
        lab, n = ndimage.label(ndimage.binary_closing(G)); k = lab[ix, iy]
        for i in range(1, n + 1):
            q = k == i
            if q.sum() < 3: continue
            pts = MultiPoint(list(zip(x[q], y[q]))); r = pts.minimum_rotated_rectangle
            if r.geom_type == 'Polygon':
                c = np.asarray(r.exterior.coords)[:4]; e1, e2 = c[1] - c[0], c[2] - c[1]
                a, bb = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
                size = [round(float(np.linalg.norm(a)) + .5, 1), round(float(np.linalg.norm(bb)) + .5, 1)]; ang = round(float(np.degrees(np.arctan2(a[1], a[0]))) % 180, 0)
            else: size, ang = [.5, .5], 0.
            cx, cy = float(x[q].mean()), float(y[q].mean())
            out.append({'part': rec['id'], 'centre': [round(cx, 2), round(cy, 2)], 'sizeM': size, 'bearingDeg': ang,
                        'topM': round(float(z[q].max() - f0), 2), 'riseM': round(float(res[q].max()), 2), 'returns': int(q.sum()),
                        'insideM': round(float(g.exterior.distance(Point(cx, cy))), 1)})
    out.sort(key=lambda o: -o['returns'])
    print(json.dumps({'building': b, 'foot': round(f0, 2), 'bumps': out}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['plan', 'section', 'stats', 'check', 'fit', 'split', 'rect', 'bumps']); ap.add_argument('building'); ap.add_argument('rest', nargs='*')
    ap.add_argument('--margin', type=float, default=12.); ap.add_argument('--width', type=float, default=1.); ap.add_argument('--out'); ap.add_argument('--spec'); ap.add_argument('--rise', type=float, default=.5)
    a = ap.parse_args()
    if a.mode == 'plan': plan(a.building, a.rest[0], a.margin)
    elif a.mode == 'section': section(a.building, list(map(float, a.rest[0:2])), list(map(float, a.rest[2:4])), a.rest[4], a.width, a.spec)
    elif a.mode == 'stats': stats(a.building)
    elif a.mode == 'fit': fit(a.building)
    elif a.mode == 'split': split(a.building, [tuple(map(float, a.rest[i:i + 4])) for i in range(0, len(a.rest), 4)])
    elif a.mode == 'rect': rect(float(a.building), *map(float, a.rest[:4]))
    elif a.mode == 'bumps': bumps(a.building, a.rise)
    else: check(a.building, a.rest[0] if a.rest else None, a.out)


if __name__ == '__main__':
    main()
