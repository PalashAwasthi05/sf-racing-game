"""Carry photo-read colours over to the road white balance (panorama_balance.road_gain) without reading again.

Until 2026-09-29 every reader balanced a panorama against rows 74-90 % down the frame, which on the roof-mounted rig are
the capture vehicle; road_gain balances against the road below the horizon. The balance is one gain per channel over
the whole panorama, applied before anything is rectified or measured, so a colour measured on the old-balanced image
becomes the new one by the ratio of the two gains for its photograph (exact for a per-channel median or mean of pixels
chosen independently of colour, which is how the readers measure wall, zone and band colours). Everything else in a
reading (which walls, layouts, material families) is kept, so a build from the output differs from its source only by
the white balance. Families chosen from a colour (e.g. dark panel) are not re-decided.

Usage: recolour_readings_wb.py --pipeline <dir with the reading files> [--check <old readings> <new readings>]
Rewrites in place: frontage-observations-auto.json, frontage-supplements-v31.json, envelope-colours-v32.json,
landmarks/outline-readings.json; writes recolour-report.json next to them.
"""
import argparse
import json
import pathlib
import re
import numpy as np
from PIL import Image
import frontage_crops_v15 as fc
from panorama_balance import road_gain

HEX = re.compile(r'#([0-9a-fA-F]{6})')


def old_gain(pano):
    """The balance every reader used before 2026-09-29 (rows 74-90 % down, near-neutral pixels, clipped)."""
    pf = np.asarray(pano, np.float32)
    road = pf[int(pf.shape[0] * .74):int(pf.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
    return np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)


_RATIO = {}


def ratio(image_id):
    """new gain / old gain for one panorama (3,), cached."""
    if image_id not in _RATIO:
        pano = np.asarray(fc.load_pano(fc.IMAGE_DIR / f'{image_id}.jpg'))
        _RATIO[image_id] = (road_gain(pano) / old_gain(pano)).astype(np.float64)
    return _RATIO[image_id]


def scale_rgb(rgb, r):
    return [int(round(min(255., max(0., float(v) * k)))) for v, k in zip(rgb[:3], r)] + list(rgb[3:])


def scale_strings(o, r, n):
    """Every '#rrggbb' inside strings of o, scaled by r; n counts replacements."""
    if isinstance(o, dict): return {k: scale_strings(v, r, n) for k, v in o.items()}
    if isinstance(o, list): return [scale_strings(v, r, n) for v in o]
    if isinstance(o, str) and HEX.search(o):
        def sub(m):
            n[0] += 1; c = scale_rgb([int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)], r)
            return '#' + ''.join(f'{v:02x}' for v in c)
        return HEX.sub(sub, o)
    return o


def recolour_profile(p, n):
    img = (p.get('camera') or {}).get('image')
    if not img: return p, None
    r = ratio(img); p = scale_strings(p, r, n)
    ev = p.get('materialEvidence')
    if ev and ev.get('colorRgb'): ev['colorRgb'] = scale_rgb(ev['colorRgb'], r); n[0] += 1
    rd = p.get('reading')
    if rd and rd.get('wallRgb'): rd['wallRgb'] = scale_rgb(rd['wallRgb'], r); n[0] += 1
    return p, img


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pipeline', type=pathlib.Path, required=True)
    ap.add_argument('--check', nargs=2, type=pathlib.Path, help='old and new frontage readings from one reader run twice, to test the ratio')
    a = ap.parse_args(); P = a.pipeline; report = {}

    if a.check:
        old = {p['id']: p for p in json.load(open(a.check[0]))['profiles']}; new = {p['id']: p for p in json.load(open(a.check[1]))['profiles']}
        errs = []
        for k, p in old.items():
            q = new.get(k)
            if not q or (p.get('camera') or {}).get('image') != (q.get('camera') or {}).get('image'): continue
            c0 = (p.get('materialEvidence') or {}).get('colorRgb'); c1 = (q.get('materialEvidence') or {}).get('colorRgb')
            if not c0 or not c1: continue
            pred = scale_rgb(c0, ratio(p['camera']['image']))
            errs.append((np.abs(np.subtract(pred[:3], c1[:3])).max(), np.abs(np.subtract(c0[:3], c1[:3])).max()))
        e = np.array(errs)
        print(f'{len(e)} walls read from the same photograph both times: |predicted - reread| median {np.median(e[:, 0]):.1f}, '
              f'90th pct {np.percentile(e[:, 0], 90):.1f} (per channel, 0-255); |old - reread| median {np.median(e[:, 1]):.1f}')
        return

    f = P / 'frontage-observations-auto.json'; d = json.load(open(f)); n = [0]; imgs = set()
    out = []
    for p in d['profiles']:
        p, img = recolour_profile(p, n); out.append(p); imgs.add(img)
    d['profiles'] = out; d['whiteBalance'] = 'road below the horizon (panorama_balance.road_gain), carried over by recolour_readings_wb.py'
    f.write_text(json.dumps(d, indent=1)); report['frontages'] = {'colours': n[0], 'photographs': len(imgs - {None})}

    f = P / 'frontage-supplements-v31.json'; d = json.load(open(f)); n = [0]
    d['supplements'] = [scale_strings(s, ratio(s['camera']), n) if s.get('camera') else s for s in d['supplements']]
    d['whiteBalance'] = 'road below the horizon (panorama_balance.road_gain), carried over by recolour_readings_wb.py'
    f.write_text(json.dumps(d, indent=1)); report['supplements'] = {'colours': n[0]}

    f = P / 'envelope-colours-v32.json'; d = json.load(open(f)); n = 0
    for env in d['envelopes'].values():
        for ph in env.get('photos', []):
            r = ratio(ph['camera'])
            for cell in ph.get('cells', []):
                cell[4] = scale_rgb(cell[4], r); n += 1
    d['whiteBalance'] = 'road below the horizon (panorama_balance.road_gain), carried over by recolour_readings_wb.py'
    f.write_text(json.dumps(d, indent=1)); report['envelopes'] = {'cells': n}

    f = P / 'landmarks/outline-readings.json'
    if f.exists():
        d = json.load(open(f)); n = 0
        for lm in d.values():
            for st in lm.get('stretches', []) if isinstance(lm, dict) else []:
                if st.get('colour') and st.get('images'):
                    r = np.mean([ratio(i) for i in st['images']], 0); c = HEX.fullmatch(st['colour'])
                    if c: st['colour'] = '#' + ''.join(f'{v:02x}' for v in scale_rgb([int(c.group(1)[i:i + 2], 16) for i in (0, 2, 4)], r)); n += 1
        f.write_text(json.dumps(d, indent=1)); report['landmarkStretches'] = n

    report['panoramas'] = {k: [round(float(x), 3) for x in v] for k, v in _RATIO.items()}
    (P / 'recolour-report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'panoramas'}), len(_RATIO), 'panoramas')


if __name__ == '__main__':
    main()
