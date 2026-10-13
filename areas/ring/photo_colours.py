"""NOT USED (2026-10-08): measured on the ring, the white-balanced photo crops give pale walls a cool cast (median hue 206
degrees against the readers' 44 on 73 near-neutral walls), so photo hue and saturation are not trusted over the readers'
for pale walls; kept as a diagnostic (its --report) only.

Wall colours from the photographs: each read side's flat wall colour takes its hue and saturation from the sunlit part
of its wall in the rectified photographs, and keeps the reading's lightness.

The readers choose a hex by eye; measured against the photographs (areas/ring/colour_check.py) their stucco and stone
came out about 1.2 times too saturated and their hues a median 27 degrees off. Lightness stays the reader's: a photograph's
exposure says little about how light a wall is, while hue and saturation (chroma over lightness) survive exposure, and the
crops are white-balanced so the road is neutral grey (regen/sides.py). Per side, over its rectified crops (padding cut
off): SegFormer's wall classes, the brighter half of those pixels (the sunlit part, or the less shaded), their median
hue and saturation, the median over the side's photographs. Patterned materials (brick, panels with joints) and sides
with too little wall seen keep the reading's colour. A near-grey wall (saturation under 0.03) keeps the reading's hue.
Usage: photo_colours.py <profiles in> <profiles out> [--report out.json] [--only ids.json]   (SFW_AREA = the area; run from work/)"""
import argparse, colorsys, glob, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.')

FLAT = ('stucco', 'cast_stone', 'venue_concrete', 'concrete', 'stone', 'timber', 'timber_dark', 'clapboard', 'shingle', 'render', 'plaster', 'terracotta')
MIN_PIXELS = 1500


def lab(rgb):
    c = rgb.astype(np.float32) / 255.; c = np.where(c > .04045, ((c + .055) / 1.055) ** 2.4, c / 12.92)
    X = c @ np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]], np.float32) / np.array([.9505, 1., 1.089], np.float32)
    f = np.where(X > .008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])


def hex_of_lab(L, a, b):
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    inv = lambda t: t ** 3 if t ** 3 > .008856 else (t - 16 / 116) / 7.787
    X = np.array([inv(fx) * .9505, inv(fy), inv(fz) * 1.089])
    lin = np.linalg.solve(np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]]).T, X)
    s = np.where(lin <= .0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - .055)
    return '#%02x%02x%02x' % tuple(int(round(v)) for v in np.clip(s, 0, 1) * 255)


def lab_of_hex(h):
    L, a, b = lab(np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]], np.uint8)); return float(L[0]), float(a[0]), float(b[0])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('profiles'); ap.add_argument('out'); ap.add_argument('--report'); ap.add_argument('--only', help='a JSON list of building ids: only their sides'); a = ap.parse_args()
    area = Path(os.environ['SFW_AREA']); prof = json.load(open(a.profiles))
    pieces = {}
    for f in sorted(glob.glob(str(area / 'regen/sides/sides-*.json'))):
        for p in json.load(open(f))['pieces']:
            if p['views'] and (p['id'] not in pieces or len(p['views']) > len(pieces[p['id']]['views'])): pieces[p['id']] = p
    from facade_segment_v27 import Segmenter
    seg = Segmenter(); names = seg.model.config.id2label; seg.lut = np.arange(len(seg.lut), dtype=np.uint8)
    WALL = [int(i) for i, n in names.items() if n in ('wall', 'building', 'house', 'skyscraper')]
    report = {}; changed = 0; only = set(json.load(open(a.only))) if a.only else None
    for x in prof['profiles']:
        sid = x['id'].replace('regen-', ''); m = x.get('material')
        if only is not None and sid.split('__')[0] not in only: continue
        if not isinstance(m, str) or '#' not in m or sid not in pieces: continue
        kind, hx = m.split('#', 1); hx = '#' + hx[:6]
        if not kind.startswith(FLAT) or kind.startswith(('brick', 'panel')): report[sid] = {'kept': f'{kind}: patterned or not a flat wall'}; continue
        per = []
        for v in pieces[sid]['views']:
            f = area / 'regen/sides/img' / v['rect']
            if not f.exists(): continue
            im = np.asarray(Image.open(f).convert('RGB')); pad = int(v.get('rectPad') or 0)
            im = im[pad:im.shape[0] - pad or None, pad:im.shape[1] - pad or None]
            if min(im.shape[:2]) < 40: continue
            s = 640 / max(im.shape[:2]); im = np.asarray(Image.fromarray(im).resize((max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s)))))
            wall = np.isin(seg.labels(im), WALL)
            if wall.sum() < MIN_PIXELS: continue
            L, A, B = lab(im); lit = wall & (L >= np.median(L[wall]))
            C = np.hypot(A[lit], B[lit]); sat = float(np.median(C / np.maximum(L[lit], 1)))
            hue = float(np.degrees(np.arctan2(np.median(B[lit]), np.median(A[lit]))) % 360)
            per.append((sat, hue, int(lit.sum())))
        if not per: report[sid] = {'kept': 'too little wall seen'}; continue
        sat = float(np.median([q[0] for q in per])); hv = np.radians([q[1] for q in per]); hue = float(np.degrees(np.arctan2(np.sin(hv).mean(), np.cos(hv).mean())) % 360)
        L0, a0, b0 = lab_of_hex(hx); C0 = float(np.hypot(a0, b0)); h0 = float(np.degrees(np.arctan2(b0, a0)) % 360)
        h = h0 if sat < .03 else hue; C = sat * L0
        new = hex_of_lab(L0, C * np.cos(np.radians(h)), C * np.sin(np.radians(h)))
        report[sid] = {'kind': kind, 'reading': hx, 'photo': new, 'readingSat': round(C0 / max(L0, 1), 3), 'photoSat': round(sat, 3),
                       'hueShiftDeg': round(((h - h0 + 180) % 360) - 180, 1), 'views': len(per)}
        x['material'] = kind + new; changed += 1
        # panels and bands painted the wall's colour follow it
        for k in ('panels', 'bands', 'piers'):
            for q in x.get(k) or []:
                if isinstance(q, dict) and isinstance(q.get('material'), str) and q['material'].lower() == m.lower(): q['material'] = kind + new
    prof['method'] = (prof.get('method') or '') + ' + wall hue/saturation from the photographs (areas/ring/photo_colours.py)'
    json.dump(prof, open(a.out, 'w'), indent=1)
    if a.report: json.dump(report, open(a.report, 'w'), indent=1)
    ch = [r for r in report.values() if 'photo' in r]
    print(json.dumps({'changed': changed, 'kept': len(report) - len(ch), 'medianSat': [round(float(np.median([r['readingSat'] for r in ch])), 3), round(float(np.median([r['photoSat'] for r in ch])), 3)] if ch else None,
                      'medianAbsHueShift': round(float(np.median([abs(r['hueShiftDeg']) for r in ch])), 1) if ch else None}))
