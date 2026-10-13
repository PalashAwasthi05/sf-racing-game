"""Are the model's windows where the photograph's are? Per photo-matched view (a render from the photograph's own,
registered camera): SegFormer's windows (windowpane, door) in the photograph and in the render, and how well the two
masks overlap (IoU over the wall the two agree is a building), for one or more render prefixes. A change to readings or
to how they are placed is kept when it raises this over the views, not when one building looks better.
Windows are taken as the building's clearly dark features (darker than 72 % of its median brightness) in each image.
Usage: window_match.py <poses.json> <out.json> <prefix> [prefix ...] [--views 0]   (run from work/)"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
sys.path.insert(0, '.')

ap = argparse.ArgumentParser(); ap.add_argument('poses'); ap.add_argument('out'); ap.add_argument('prefixes', nargs='+'); ap.add_argument('--views', default='0'); a = ap.parse_args()
P = [p for p in json.load(open(a.poses))['poses'] if str(p['view']) in a.views.split(',')]
from facade_segment_v27 import Segmenter
seg = Segmenter(); names = seg.model.config.id2label; seg.lut = np.arange(len(seg.lut), dtype=np.uint8)
WIN = [int(i) for i, n in names.items() if n in ('windowpane', 'door', 'screen door')]
BLD = [int(i) for i, n in names.items() if n in ('wall', 'building', 'house', 'skyscraper', 'windowpane', 'door')]
from scipy.ndimage import binary_opening


def dark(img, bld):
    """Window-like features: building pixels clearly darker than the building's own median (glass and openings read
    dark against a wall in a photograph and in a render alike; ADE's windowpane class is not found at street scale)."""
    L = img.astype(np.float32).mean(2)
    if bld.sum() < 500: return np.zeros(bld.shape, bool)
    m = bld & (L < np.median(L[bld]) * .72)
    return binary_opening(m, iterations=1)


rows = []
for p in P:
    name = f"{p['side']}__v{p['view']}"; caps = {k: Path('areas/captures') / f'{k}__{name}.png' for k in a.prefixes}
    if not Path(p['persp']).exists() or not all(c.exists() for c in caps.values()): continue
    ph = Image.open(p['persp']).convert('RGB'); W, H = 640, round(640 * ph.height / ph.width)
    phi = np.asarray(ph.resize((W, H)))
    cp = seg.labels(phi); bp = np.isin(cp, BLD); wp = dark(phi, bp)
    if wp.sum() < 300: continue
    row = {'view': name}
    for k, c in caps.items():
        ri = np.asarray(Image.open(c).convert('RGB').resize((W, H))); cr = seg.labels(ri); br = np.isin(cr, BLD); wr = dark(ri, br)
        area = bp & br                                            # where both see a building
        if area.sum() < 2000: row[k] = None; continue
        a1, b1 = wp & area, wr & area
        # windows within a few pixels count as matched (a frame's width): dilate each before the intersection
        inter = (binary_dilation(a1, iterations=2) & b1).sum() + (binary_dilation(b1, iterations=2) & a1).sum()
        row[k] = round(float(inter / max(a1.sum() + b1.sum(), 1)), 3)
    rows.append(row); print(name, {k: row[k] for k in a.prefixes}, flush=True)
summary = {}
for k in a.prefixes:
    v = [r[k] for r in rows if r.get(k) is not None]; summary[k] = {'views': len(v), 'median': round(float(np.median(v)), 3) if v else None, 'mean': round(float(np.mean(v)), 3) if v else None}
if len(a.prefixes) == 2:
    k0, k1 = a.prefixes; d = [r[k1] - r[k0] for r in rows if r.get(k0) is not None and r.get(k1) is not None]
    summary['better'] = sum(x > .02 for x in d); summary['worse'] = sum(x < -.02 for x in d)
json.dump({'summary': summary, 'rows': rows}, open(a.out, 'w'), indent=1); print(json.dumps(summary))
