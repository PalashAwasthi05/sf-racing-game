"""The roof-feature agents' files (areas/ring/massing/features/*.json) into the generator's overrides
(areas/ring/massing-overrides.json, each building's `roofFeatures`), each feature checked first: it stands inside the
building's outline, on the LiDAR bump it names (its centre within 1.5 m of the bump's, inside the bump's extent grown
by 1 m; its top within 1 m of the bump's), and its sizes are a dormer's or a chimney's. A dormer faces down the roof
slope it stands on (the steepest way down the roof plane at its centre); a dormer on a flat roof is refused. A feature
that fails is left out and said so.
Usage (run from work/): merge_features.py [--dry]"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from shapely.geometry import Point
sys.path.insert(0, str(Path(__file__).resolve().parent))
import massing_view as mv

RING = Path('areas/ring'); dry = '--dry' in sys.argv
over = json.load(open(RING / 'massing-overrides.json')); D, _ = mv.dump(); report = []
for f in sorted((RING / 'massing/features').glob('*.json')):
    j = json.load(open(f)); b = j.get('building', f.stem); src = b.split('_p')[0]
    out = subprocess.run([sys.executable, str(RING / 'massing_view.py'), 'bumps', src], capture_output=True, text=True)
    try: bumps = json.loads(out.stdout)['bumps']
    except Exception: report.append({'building': b, 'merged': 0, 'why': 'bumps failed: ' + out.stderr[-200:]}); continue
    P = mv.parts_of(src, D); kept = []
    for i, ft in enumerate(j.get('roofFeatures') or []):
        why = []; xy = np.asarray(ft.get('xy') or [np.nan, np.nan], float); kind = ft.get('kind')
        part = next((r for r in P if mv.poly(r).contains(Point(xy))), None) if np.isfinite(xy).all() else None
        if kind not in ('dormer', 'chimney'): why.append(f'kind {kind}')
        if part is None: why.append('not inside the outline')
        k = ft.get('bump'); bp = bumps[k] if isinstance(k, int) and 0 <= k < len(bumps) else None
        top = ft.get('ridgeM') if kind == 'dormer' else ft.get('topM')
        if bp is None: why.append(f'no bump {k}')
        else:
            dc = float(np.hypot(*(xy - np.asarray(bp['centre']))))
            if dc > max(1.5, max(bp['sizeM']) / 2 + 1.): why.append(f'{dc:.1f} m from bump {k}')
            if top is None or abs(float(top) - bp['topM']) > 1.: why.append(f'top {top} vs bump {bp["topM"]}')
        w, d = float(ft.get('widthM') or 0), float(ft.get('depthM') or 0)
        if kind == 'dormer':
            if not (.8 <= w <= 6 and .8 <= d <= 8): why.append(f'dormer size {w} x {d}')
            if ft.get('eaveM') is None or top is None or float(top) <= float(ft['eaveM']): why.append('dormer eave/ridge')
            if part is not None:
                planes = [q for p in part['patches'] for q in p['planes']]
                if not planes: why.append('on a flat roof')
                else:
                    a, bb, _ = min(planes, key=lambda q: q[0] * xy[0] + q[1] * xy[1] + q[2]); g = np.hypot(a, bb)
                    if g < .05: why.append('on a flat roof')
                    else: ft = {**ft, 'facing': [round(float(-a / g), 4), round(float(-bb / g), 4)]}
        if kind == 'chimney' and not (.3 <= w <= 1.8 and .3 <= d <= 1.8): why.append(f'chimney size {w} x {d} (over a stack\'s 1.8 m)')
        if why: report.append({'building': b, 'feature': i, 'kind': kind, 'merged': False, 'why': '; '.join(why)}); continue
        kept.append({k2: v for k2, v in ft.items() if k2 not in ('seen',)})
        report.append({'building': b, 'feature': i, 'kind': kind, 'merged': True})
    if kept or src in over:
        over.setdefault(src, {})['roofFeatures'] = kept
if not dry: json.dump(over, open(RING / 'massing-overrides.json', 'w'), indent=1)
print(json.dumps(report, indent=1))
