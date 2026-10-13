"""Wall colour in each photograph against the same wall in its model render, by wall material: where the model's
surfaces are too bright, too saturated or the wrong hue, measured rather than eyeballed.

Per pose (photograph and a render from the same camera, ideally a registered one): SegFormer's building classes (wall,
building, house, skyscraper) in both images, kept where both agree a wall is (the overlap, so a tree or a car in either
drops out). Over those pixels, the median CIELAB lightness L*, chroma C* and hue of each image, and the same split into
the brighter and darker halves of the photograph's wall (lit and shaded, roughly). Exposure differs between a camera and
the viewer, so each image's road and pavement give a reference: L* is also reported relative to it.
Usage: colour_check.py <poses.json> <render prefix> <profiles.json> <out.json> [--views 0,1]   (run from work/)"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.')

ap = argparse.ArgumentParser(); ap.add_argument('poses'); ap.add_argument('prefix'); ap.add_argument('profiles'); ap.add_argument('out')
ap.add_argument('--views', default='0,1'); a = ap.parse_args()
P = [p for p in json.load(open(a.poses))['poses'] if str(p['view']) in a.views.split(',')]
mat = {}
for x in json.load(open(a.profiles))['profiles']:
    m = x.get('material'); m = m if isinstance(m, str) else json.dumps(m)
    mat[x['id'].replace('regen-', '')] = m.split('#')[0]
from facade_segment_v27 import Segmenter
seg = Segmenter(); names = seg.model.config.id2label; seg.lut = np.arange(len(seg.lut), dtype=np.uint8)
WALL = [int(i) for i, n in names.items() if n in ('wall', 'building', 'house', 'skyscraper')]
GROUND = [int(i) for i, n in names.items() if n in ('road', 'sidewalk', 'floor', 'path')]


def lab(rgb):
    c = rgb.astype(np.float32) / 255.; c = np.where(c > .04045, ((c + .055) / 1.055) ** 2.4, c / 12.92)
    X = c @ np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]], np.float32)
    X = X / np.array([.9505, 1., 1.089], np.float32); f = np.where(X > .008856, np.cbrt(X), 7.787 * X + 16 / 116)
    L = 116 * f[..., 1] - 16; A = 500 * (f[..., 0] - f[..., 1]); B = 200 * (f[..., 1] - f[..., 2])
    return L, np.hypot(A, B), np.degrees(np.arctan2(B, A)) % 360


def stats(L, C, Hh, m):
    if m.sum() < 200: return None
    return {'L': round(float(np.median(L[m])), 1), 'C': round(float(np.median(C[m])), 1), 'hue': round(float(np.median(Hh[m])), 0)}


rows = []
for p in P:
    name = f"{p['side']}__v{p['view']}"; cap = Path('areas/captures') / f'{a.prefix}__{name}.png'
    if not (cap.exists() and Path(p['persp']).exists()): continue
    ph = Image.open(p['persp']).convert('RGB'); W, H = 500, round(500 * ph.height / ph.width)
    ims = [np.asarray(ph.resize((W, H))), np.asarray(Image.open(cap).convert('RGB').resize((W, H)))]
    cls = [seg.labels(i) for i in ims]
    wall = np.isin(cls[0], WALL) & np.isin(cls[1], WALL)
    if wall.sum() < 400: continue
    row = {'side': p['side'], 'view': p['view'], 'material': mat.get(p['side'], '?'), 'wallShare': round(float(wall.mean()), 3)}
    Lp = lab(ims[0])[0]; lit = wall & (Lp >= np.median(Lp[wall])); shade = wall & ~lit
    for k, im, c in (('photo', ims[0], cls[0]), ('model', ims[1], cls[1])):
        L, C, Hh = lab(im); g = np.isin(c, GROUND)
        row[k] = {'all': stats(L, C, Hh, wall), 'lit': stats(L, C, Hh, lit), 'shade': stats(L, C, Hh, shade),
                  'groundL': round(float(np.median(L[g])), 1) if g.sum() > 200 else None}
    rows.append(row); print(name, row['material'], row['photo']['all'], row['model']['all'], flush=True)

summary = collections.defaultdict(dict)
for m in sorted({r['material'] for r in rows}) + ['ALL']:
    R = [r for r in rows if m in ('ALL', r['material'])]
    for part in ('all', 'lit', 'shade'):
        d = [(r['model'][part]['L'] - r['photo'][part]['L'], r['model'][part]['C'] - r['photo'][part]['C'],
              (r['model'][part]['C'] / max(r['photo'][part]['C'], 1.))) for r in R if r['model'][part] and r['photo'][part]]
        if d: summary[m][part] = {'n': len(d), 'dL': round(float(np.median([x[0] for x in d])), 1), 'dC': round(float(np.median([x[1] for x in d])), 1), 'chromaRatio': round(float(np.median([x[2] for x in d])), 2)}
    rel = [(r['model']['all']['L'] - r['model']['groundL']) - (r['photo']['all']['L'] - r['photo']['groundL']) for r in R if r['model']['groundL'] and r['photo']['groundL'] and r['model']['all'] and r['photo']['all']]
    if rel: summary[m]['dL_vs_ground'] = {'n': len(rel), 'median': round(float(np.median(rel)), 1)}
json.dump({'summary': summary, 'rows': rows}, open(a.out, 'w'), indent=1); print(json.dumps(summary, indent=1))
