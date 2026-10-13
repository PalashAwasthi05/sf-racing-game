"""Colours measured on the photograph, in place of the colours a model names.

Each material zone, pier and band takes the median colour of the rectified (road white-balanced) photograph inside its
own rectangle, away from openings, the street clutter band and the frame margin. A secondary zone whose measured colour
is the main wall's (same family, within dE 12) is dropped: it was the same wall in different light."""
import numpy as np
from PIL import Image
import specs

def _lab(rgb):
    c = np.asarray(rgb, float) / 255.; c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    X = c @ np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]]) / np.array([.9505, 1., 1.089])
    f = np.where(X > .008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

def _hex(lab):
    L, a, b = lab; fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    f = np.array([fx, fy, fz]); X = np.where(f ** 3 > .008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([.9505, 1., 1.089])
    rgb = X @ np.linalg.inv(np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]]))
    rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - .055)
    return '#%02x%02x%02x' % tuple(int(round(v)) for v in np.clip(rgb * 255, 0, 255))

def reconcile(model_hex, measured_hex):
    """The photograph's radiance is darker than a surface's own colour (shade, exposure), so a model's lightness is kept;
    its hue is trusted only where the photograph agrees there is colour: a model-vivid accent the photo shows grey
    becomes grey at the model's lightness."""
    if not measured_hex: return model_hex
    if not model_hex or not str(model_hex).startswith('#') or len(model_hex) != 7: return measured_hex
    m = _lab([int(model_hex[i:i + 2], 16) for i in (1, 3, 5)]); q = _lab([int(measured_hex[i:i + 2], 16) for i in (1, 3, 5)])
    cm, cq = np.hypot(m[1], m[2]), np.hypot(q[1], q[2])
    if cm > 25 and cq < 12: return _hex([m[0], q[1], q[2]])
    return model_hex

def measure(r, piece, img_dir, street_m=2.6):
    v = piece['views'][r['use']]; pad, ppm, H, L = v['rectPad'], v['ppm'], float(piece['heightM']), float(piece['lengthM'])
    img = np.asarray(Image.open(f"{img_dir}/{v['rect']}").convert('RGB'), float)[pad:-pad, pad:-pad]
    h, w = img.shape[:2]
    mask = np.ones((h, w), bool)                                   # wall pixels: not an opening of this reading
    for o in [c for b in r.get('banks') or [] for c in specs.bank_cells(b)] + [(o['x0'], o['z0'], o['x1'], o['z1']) for o in r.get('openings') or []]:
        x0, z0, x1, z1 = o; mask[max(0, int((H - z1) * ppm)):int((H - z0) * ppm) + 1, max(0, int(x0 * ppm)):int(x1 * ppm) + 1] = False
    def med(x0, z0, x1, z1, shrink=.15, keep_street=False):
        dx, dz = (x1 - x0) * shrink, (z1 - z0) * shrink
        x0, x1, z0, z1 = x0 + dx, x1 - dx, z0 + dz, z1 - dz
        if not keep_street and z1 > street_m + .5: z0 = max(z0, street_m)
        a, b = max(0, int((H - z1) * ppm)), min(h, int((H - z0) * ppm)); c, d = max(0, int(x0 * ppm)), min(w, int(x1 * ppm))
        if b - a < 3 or d - c < 3: return None
        px = img[a:b, c:d][mask[a:b, c:d]]
        px = px[(px.sum(1) > 30) & (px.sum(1) < 735)]               # no clipped sky or deep shadow
        return None if len(px) < 30 else '#%02x%02x%02x' % tuple(int(x) for x in np.median(px, 0))
    out = dict(r); mats = [dict(m) for m in r.get('materials') or []]
    for m in mats:
        c = med(m['x0'], m['z0'], m['x1'], m['z1'])
        if c: m['hexMeasured'] = c
    if mats:
        # the wall's own material (the one covering most of it, specs.to_profile) leads, whatever order it was read in
        k = max(range(len(mats)), key=lambda i: max(0., mats[i]['x1'] - mats[i]['x0']) * max(0., mats[i]['z1'] - mats[i]['z0']))
        mats = [mats[k]] + mats[:k] + mats[k + 1:]
        mh = lambda m: m.get('hexMeasured') or specs.hexok(m.get('hex'))
        main = _lab([int(mh(mats[0])[i:i + 2], 16) for i in (1, 3, 5)])
        mats = [mats[0]] + [m for m in mats[1:] if not (m.get('family') == mats[0].get('family') and
                np.linalg.norm(_lab([int(mh(m)[i:i + 2], 16) for i in (1, 3, 5)]) - main) < 12)]
        for m in mats[1:]: m['hex'] = reconcile(m.get('hex'), m.get('hexMeasured'))
    out['materials'] = mats
    top = r.get('wallTopM') or H
    out['piers'] = []
    for p in r.get('piers') or []:
        p = dict(p); c = med(p['xM'] - p.get('widthM', .5) / 2, 0, p['xM'] + p.get('widthM', .5) / 2, top, shrink=.2)
        if c: p['hexMeasured'], p['hex'] = c, reconcile(p.get('hex'), c)
        out['piers'].append(p)
    out['bands'] = []
    for b in r.get('bands') or []:
        b = dict(b); t = max(b.get('thicknessM', .3), .2); c = med(0, b['zM'] - t / 2, L, b['zM'] + t / 2, shrink=0, keep_street=True)
        if c: b['hexMeasured'], b['hex'] = c, reconcile(b.get('hex'), c)
        out['bands'].append(b)
    return out
