"""How well each model render frames what its photograph shows: the sky in the photograph against the sky in the render
(SegFormer on both), per pose: the overlap of the two sky masks (IoU) and the mean gap between the two skylines (the
lowest sky pixel per column, as a share of the image height). A render from the right camera puts the rooftops where
the photograph has them, whatever the model's surfaces look like.
Usage: framing.py <poses.json> <out.json> <capture prefix> [prefix ...] [--views 0,1] [--sides side ...]   (run from work/)"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.')

ap = argparse.ArgumentParser(); ap.add_argument('poses'); ap.add_argument('out'); ap.add_argument('prefixes', nargs='+')
ap.add_argument('--views', default='0,1'); ap.add_argument('--sides', nargs='*'); a = ap.parse_args()
P = [p for p in json.load(open(a.poses))['poses'] if str(p['view']) in a.views.split(',') and (not a.sides or p['side'] in a.sides)]
from facade_segment_v27 import Segmenter
seg = Segmenter(); rows = []


def sky(img):
    return seg.labels(np.asarray(img.convert('RGB'))) == 4


def skyline(m):
    """Per column, the first non-sky row from the top (the column's height when it is all sky)."""
    solid = ~m; top = np.argmax(solid, 0); return np.where(solid.any(0), top, m.shape[0]).astype(float)


for p in P:
    name = f"{p['side']}__v{p['view']}"
    caps = {k: Path('areas/captures') / f'{k}__{name}.png' for k in a.prefixes}
    if not Path(p['persp']).exists() or not all(c.exists() for c in caps.values()): continue
    photo = Image.open(p['persp']); W, H = 500, round(500 * photo.height / photo.width)
    ms = sky(photo.resize((W, H))); ls = skyline(ms); row = {'side': p['side'], 'view': p['view'], 'photoSky': round(float(ms.mean()), 3)}
    for k, c in caps.items():
        mr = sky(Image.open(c).resize((W, H))); lr = skyline(mr)
        inter = (ms & mr).sum(); union = (ms | mr).sum()
        row[k] = {'iou': round(float(inter / union), 3) if union else None, 'skylineGap': round(float(np.mean(np.abs(ls - lr)) / H), 3),
                  'skylineBias': round(float(np.mean(lr - ls) / H), 3)}
    rows.append(row); print(name, {k: row[k] for k in a.prefixes}, flush=True)
summary = {}
for k in a.prefixes:
    iou = [r[k]['iou'] for r in rows if r[k]['iou'] is not None and r['photoSky'] > .03]; gap = [r[k]['skylineGap'] for r in rows if r['photoSky'] > .03]
    summary[k] = {'poses': len(iou), 'iouMedian': round(float(np.median(iou)), 3) if iou else None, 'skylineGapMedian': round(float(np.median(gap)), 3) if gap else None}
json.dump({'summary': summary, 'rows': rows}, open(a.out, 'w'), indent=1); print(json.dumps(summary))
