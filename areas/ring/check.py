"""Check a facade reading (areas/ring/read-v2.md) against its photographs: the reading drawn over both rectified views,
side by side, and a summary with the photograph's own colour in each material zone. For a reader's edit-and-check loop.
Usage: check.py <side id> <reading.json> <out.jpg>
       check.py grid <side id> <view 0|1> <x0> <z0> <x1> <z1> <out.jpg>   (a zoomed crop with a 0.5 m grid)
       (run from work/)"""
import glob, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent; WORK = HERE.parents[1]; sys.path.insert(0, str(WORK / 'regen')); sys.path.insert(0, str(WORK))
import specs

AREA = WORK / 'areas/pacific-heights'
FONT = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)


def piece_of(side):
    best = None
    for f in sorted(glob.glob(str(AREA / 'regen/sides/sides-*.json'))):
        for p in json.load(open(f))['pieces']:
            if p['id'] == side and p['views'] and (best is None or len(p['views']) > len(best['views'])): best = p
    if best is None: sys.exit(f'no photographed side {side}')
    return best


def draw(r, piece, v):
    im = Image.open(AREA / 'regen/sides/img' / v['rect']).convert('RGB'); photo = np.asarray(im).copy(); d = ImageDraw.Draw(im)
    pad, ppm, H, L = v['rectPad'], v['ppm'], float(piece['heightM']), float(piece['lengthM'])
    P = lambda x, z: (pad + x * ppm, pad + (H - z) * ppm)
    def rect(x0, z0, x1, z1, col, w=2):
        a, b = P(x0, z1), P(x1, z0); d.rectangle([a, b], outline=col, width=w)
    def text(x, z, s, col): d.text(P(x, z), s, fill=col, font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    for z in r.get('storeys') or []:
        a, b = P(0, z), P(L, z)
        for x in np.arange(a[0], b[0], 14): d.line([(x, a[1]), (min(x + 7, b[0]), a[1])], fill=(255, 255, 255), width=1)
        text(-0.0, z + .35, f'{z:g}', (255, 255, 255))
    for k, m in enumerate(r.get('materials') or []):
        rect(m['x0'], m['z0'], m['x1'], m['z1'], (255, 0, 255), 1); text(m['x0'] + .2, m['z1'] - .15, f"M{k} {m.get('family')} {m.get('hex')}", (255, 0, 255))
    for k, q in enumerate(r.get('rustication') or []):
        c = q.get('courseM', .45)
        for z in np.arange(q['z0'], q['z1'], c): d.line([P(q['x0'], z), P(q['x1'], z)], fill=(170, 90, 255), width=1)
        text(q['x0'] + .2, q['z0'] + .5, f'R{k}', (170, 90, 255))
    for q in r.get('quoins') or []:
        c = q.get('courseM', .45); wq = q.get('widthM', .7); left = q['xM'] < L / 2
        for i, z in enumerate(np.arange(q['z0'], q['z1'] - c / 2, c)):
            ww = wq if i % 2 == 0 else wq * .62; xa, xb = (q['xM'], q['xM'] + ww) if left else (q['xM'] - ww, q['xM'])
            rect(xa, z, xb, z + c - .03, (170, 90, 255), 1)
    def window(c, o, col):
        x0, z0, x1, z1 = c; rect(x0, z0, x1, z1, col, 2)
        if o.get('head') in ('arch', 'segmental'):
            sp = z0 + (z1 - z0) * (.68 if o['head'] == 'arch' else .86); r_ = (x1 - x0) / 2
            d.line([P(x0, sp), P(x1, sp)], fill=col, width=1)
        sash = o.get('sash'); top = z1
        if sash == 'double-hung':
            mr = z0 + (z1 - z0) * o.get('meetingRail', .5); d.line([P(x0, mr), P(x1, mr)], fill=(255, 120, 0), width=2)
        lites = o.get('lites') or [1, 1]
        for i in range(1, int(lites[0])): d.line([P(x0 + (x1 - x0) * i / lites[0], z0), P(x0 + (x1 - x0) * i / lites[0], z1)], fill=(255, 200, 120), width=1)
        sr = o.get('surround')
        if sr: rect(x0 - sr.get('widthM', .14), z0 - (0 if o.get('kind') in ('entry', 'shopfront', 'garage') else sr.get('widthM', .14)), x1 + sr.get('widthM', .14), z1 + sr.get('widthM', .14), (60, 255, 120), 1)
        hd = o.get('hood')
        if hd:
            zt = z1 + (sr.get('widthM', .14) if sr else 0); rect(x0 - .2, zt, x1 + .2, zt + hd.get('heightM', .3), (60, 255, 120), 2)
        st = o.get('steps')
        if st and z0 > .1:
            n = max(1, int(st.get('count') or round(z0 / .17)))
            for i in range(n): rect((x0 + x1) / 2 - st.get('widthM', 1.6) / 2, z0 * i / n, (x0 + x1) / 2 + st.get('widthM', 1.6) / 2, z0 * (i + 1) / n, (60, 255, 120), 1)
    for k, b in enumerate(r.get('banks') or []):
        rect(b['x0'], b['z0'], b['x1'], b['z1'], (0, 120, 255), 1); text(b['x0'], b['z1'] + .55, f"B{k} {b.get('columns')}x{b.get('rows')} {b.get('sash', '')}", (0, 160, 255))
        for c in specs.bank_cells(b): window(c, b, (0, 255, 255))
    for k, o in enumerate(r.get('openings') or []):
        if o.get('bayWindow'): window((o['x0'], o['z0'], o['x1'], o['z1']), o, (255, 150, 40)); continue      # laid on the bay's faces (no parallax)
        window((o['x0'], o['z0'], o['x1'], o['z1']), o, (255, 230, 0)); text(o['x0'], o['z1'] + .55, f"O{k} {o.get('kind')}", (255, 230, 0))
    for k, b in enumerate(r.get('bands') or []):
        t = b.get('thicknessM', .3); x0, x1 = b.get('x0', 0), b.get('x1', L)
        rect(x0, b['zM'] - t / 2, x1, b['zM'] + t / 2, (255, 40, 40), 2); text(x0 + .2, b['zM'] + t / 2 + .5, f"L{k} {b.get('kind', '')} {b.get('profile', '')} p{b.get('projectionM', '')}", (255, 60, 60))
    for k, p in enumerate(r.get('piers') or []):
        rect(p['xM'] - p.get('widthM', .5) / 2, 0, p['xM'] + p.get('widthM', .5) / 2, r.get('wallTopM') or H, (40, 255, 40), 2)
    for k, b in enumerate(r.get('bays') or []):
        rect(b['x0'], b['z0'], b['x1'], b['z1'], (255, 140, 0), 3); text(b['x0'] + .1, b['z0'] + .3, f"Y{k} {b.get('shape', 'auto')} p{b.get('projectionM')}", (255, 160, 0))
    for k, f in enumerate(r.get('fireEscapes') or []):
        for z in f['levels']:
            d.line([P(f['x0'], z), P(f['x1'], z)], fill=(200, 30, 30), width=4); rect(f['x0'], z, f['x1'], z + 1., (200, 30, 30), 1)
        text(f['x0'], max(f['levels']) + 1.4, f'FE{k}', (255, 80, 80))
    for k, b in enumerate(r.get('balconies') or []):
        d.line([P(b['x0'], b['zM']), P(b['x1'], b['zM'])], fill=(255, 80, 200), width=4); rect(b['x0'], b['zM'], b['x1'], b['zM'] + b.get('railingHeightM', 1.), (255, 80, 200), 1)
    for k, o in enumerate(r.get('ornaments') or []):
        rect(o['x0'], o['z0'], o['x1'], o['z1'], (255, 150, 220), 1)
    if r.get('wallTopM'): d.line([P(0, r['wallTopM']), P(L, r['wallTopM'])], fill=(255, 255, 255), width=2)
    if r.get('parapetTop'): d.line([P(x, z) for x, z in r['parapetTop']], fill=(255, 255, 255), width=3)
    return im, photo, (pad, ppm, H)


def zone_colours(r, photo, frame):
    """The photograph's median colour in each material zone, openings and bays left out."""
    pad, ppm, H = frame; h, w = photo.shape[:2]; mask = np.ones((h, w), bool)
    def px(x0, z0, x1, z1):
        return max(0, int(pad + x0 * ppm)), max(0, int(pad + (H - z1) * ppm)), min(w, int(pad + x1 * ppm)), min(h, int(pad + (H - z0) * ppm))
    for b in r.get('banks') or []:
        for c in specs.bank_cells(b): a, t, bb, u = px(*c); mask[t:u, a:bb] = False
    for o in (r.get('openings') or []) + (r.get('bays') or []):
        a, t, bb, u = px(o['x0'], o['z0'], o['x1'], o['z1']); mask[t:u, a:bb] = False
    out = []
    for m in r.get('materials') or []:
        a, t, bb, u = px(m['x0'], m['z0'], m['x1'], m['z1']); sel = photo[t:u, a:bb][mask[t:u, a:bb]]
        sel = sel[(sel.sum(axis=1) > 60) & (sel.min(axis=1) < 245)]          # not deep shadow or blown-out sky
        out.append('#%02x%02x%02x' % tuple(int(v) for v in np.median(sel, axis=0)) if len(sel) > 50 else None)
    return out


def grid(side, view, x0, z0, x1, z1, out, scale=3.):
    """A zoomed crop of one rectified view with a labelled half-metre grid, for measuring positions to about 0.1 m."""
    piece = piece_of(side); v = piece['views'][view]; im = Image.open(AREA / 'regen/sides/img' / v['rect']).convert('RGB')
    pad, ppm, H = v['rectPad'], v['ppm'], float(piece['heightM'])
    box = (int(pad + x0 * ppm), int(pad + (H - z1) * ppm), int(pad + x1 * ppm), int(pad + (H - z0) * ppm))
    crop = im.crop(box); s = min(scale, 1800 / max(1, crop.width), 1800 / max(1, crop.height)); crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(crop); P = lambda x, z: ((pad + x * ppm - box[0]) * s, (pad + (H - z) * ppm - box[1]) * s)
    for x in np.arange(math.ceil(x0 * 2) / 2, x1 + 1e-6, .5):
        a, b = P(x, z0), P(x, z1); major = abs(x - round(x)) < 1e-6
        d.line([a, b], fill=(255, 40, 40) if major else (255, 160, 160), width=2 if major else 1)
        if major: d.text((a[0] + 2, b[1] + 2), f'{x:g}', fill=(255, 40, 40), font=FONT, stroke_width=2, stroke_fill=(255, 255, 255))
    for z in np.arange(math.ceil(z0 * 2) / 2, z1 + 1e-6, .5):
        a, b = P(x0, z), P(x1, z); major = abs(z - round(z)) < 1e-6
        d.line([a, b], fill=(40, 40, 255) if major else (160, 160, 255), width=2 if major else 1)
        if major: d.text((a[0] + 2, a[1] - 15), f'{z:g}', fill=(40, 40, 255), font=FONT, stroke_width=2, stroke_fill=(255, 255, 255))
    crop.save(out, quality=90); print(json.dumps({'side': side, 'view': view, 'image': out, 'metresPerGridLine': .5}))


def main():
    if sys.argv[1] == 'grid':
        a = sys.argv[2:]; return grid(a[0], int(a[1]), *map(float, a[2:6]), a[6])
    side, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    piece = piece_of(side); r = specs.lay_bay_windows(specs.clean(json.load(open(path)), piece))
    ims, colours = [], []
    for i, v in enumerate(piece['views'][:2]):
        im, photo, frame = draw(r, piece, v); ims.append(im); colours.append(zone_colours(r, photo, frame))
    W = sum(i.width for i in ims) + 10 * (len(ims) - 1); Hh = max(i.height for i in ims)
    sheet = Image.new('RGB', (W, Hh), (30, 30, 30)); x = 0
    for i in ims: sheet.paste(i, (x, 0)); x += i.width + 10
    sheet.save(out, quality=88)
    cells = sum(len(specs.bank_cells(b)) for b in r.get('banks') or [])
    print(json.dumps({'side': side, 'lengthM': piece['lengthM'], 'frameHeightM': piece['heightM'], 'storeys': r.get('storeys'), 'wallTopM': r.get('wallTopM'),
                      'bankWindows': cells, 'openings': len(r.get('openings') or []), 'bays': len(r.get('bays') or []), 'bands': len(r.get('bands') or []),
                      'fireEscapes': len(r.get('fireEscapes') or []), 'balconies': len(r.get('balconies') or []),
                      'zones': [{'zone': f"M{k}", 'read': m.get('hex'), 'photoMedianByView': [c[k] for c in colours]} for k, m in enumerate(r.get('materials') or [])],
                      'image': out}))


if __name__ == '__main__':
    main()
