"""Native-pixel sharpness diagnostics. No resampling or alignment is performed.
python3 sharpness-score.py capture-dir [baseline-dir]
Metrics indicate contrast/detail, not reconstruction accuracy. Inspect paired crops for halos/aliasing.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROIS = {
    'facade-near': (1220, 160, 1450, 470),
    'facade-distant': (760, 395, 900, 490),
    'foliage-near': (160, 175, 500, 390),
    'foliage-distant': (630, 380, 765, 490),
    'street-depth': (810, 450, 1060, 525),
    'overall-scene': (20, 160, 1630, 760),
}

def stats(a):
    gray = a.astype(np.float64) @ np.array([.2126, .7152, .0722])
    mid = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1]+gray[2:, 1:-1]+gray[1:-1, :-2]+gray[1:-1, 2:]-4*mid
    gx = (gray[1:-1, 2:]-gray[1:-1, :-2])/2
    gy = (gray[2:, 1:-1]-gray[:-2, 1:-1])/2
    grad = np.hypot(gx, gy)
    return {'laplacianVariance': float(lap.var()), 'gradientMean': float(grad.mean()),
            'gradientP90': float(np.percentile(grad, 90)), 'lumaStd': float(gray.std())}

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('capture', type=Path)
cli.add_argument('baseline', nargs='?', type=Path)
args = cli.parse_args()
current = json.loads((args.capture/'report.json').read_text())
if not current.get('complete') or not current.get('cameraStable'):
    raise SystemExit('Capture did not pass scene/camera validation')
if args.baseline:
    baseline = json.loads((args.baseline/'report.json').read_text())
    if not baseline.get('complete') or current['camera'] != baseline['camera']:
        raise SystemExit('Baseline is incomplete or camera differs; rerun with its report.json as pose source')
    if current['scope']['hash'] != baseline['scope']['hash']:
        raise SystemExit('Different source maps')
images = [Image.open(args.capture/f'sample-{i}.png').convert('RGB') for i in range(3)]
if any(im.size != (1920,1080) for im in images):
    raise SystemExit('Expected native 1920 x 1080 screenshots')
arrays = [np.asarray(im) for im in images]
prior = np.asarray(Image.open(args.baseline/'sample-0.png').convert('RGB')) if args.baseline else None
if prior is not None and prior.shape != arrays[0].shape:
    raise SystemExit('Baseline pixel dimensions differ')
result = {'rois': {}, 'note': 'Higher contrast/detail metrics alone do not prove better clarity; examine halos, jaggies, leaf breakup and facade ringing in the native crops. Overall includes near car; use facade/foliage ROIs independently.'}
for name, rect in ROIS.items():
    x0,y0,x1,y1=rect
    crops = [a[y0:y1,x0:x1] for a in arrays]
    images[0].crop(rect).save(args.capture/f'{name}.png')
    row = {'rectXYXY': rect, 'current': stats(crops[0]),
           'repeatMeanAbsoluteDifference': [float(np.abs(c.astype(float)-crops[0]).mean()) for c in crops[1:]]}
    if prior is not None:
        before = prior[y0:y1,x0:x1]
        row['baseline'] = stats(before)
        row['baselineMeanAbsoluteDifference'] = float(np.abs(crops[0].astype(float)-before).mean())
        row['metricRatios'] = {k: row['current'][k]/v if v else None for k,v in row['baseline'].items()}
        pair = Image.new('RGB', ((x1-x0)*2,y1-y0))
        pair.paste(Image.fromarray(before),(0,0));pair.paste(Image.fromarray(crops[0]),(x1-x0,0))
        pair.save(args.capture/f'{name}-baseline-left.png')
    result['rois'][name] = row
(args.capture/'sharpness.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
