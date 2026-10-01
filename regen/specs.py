"""Readings (metres, measured on a rectified photograph) -> facade profiles the grammar compiles, and overlays that
draw a reading back onto its photograph for checking. Pure geometry; no place facts."""
import json, pathlib, re
import numpy as np
from PIL import Image, ImageDraw

FAMILY_RECIPE = {'brick': 'brick_modular', 'render': 'stucco', 'stone': 'cast_stone', 'concrete': 'venue_concrete',
                 'metal-panel': 'panel_silver', 'glass': 'panel_dark', 'timber': 'timber_dark', 'tile': 'terracotta_panel',
                 'terracotta': 'terracotta_panel'}
KIND = {'window': 'window', 'shopfront': 'shopfront', 'entry': 'entry', 'garage': 'rollup', 'rollup': 'rollup',
        'louvre': 'louvre', 'curtain': 'window'}


def hexok(h, default='#8a8d8f'):
    return h if isinstance(h, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', h) else default


def clean(r, piece):
    """Tolerate small format slips in a model's reading: a missing view index, malformed outlines, empty rectangles."""
    r = dict(r)
    if not isinstance(r.get('use'), int) or not (0 <= r['use'] < len(piece['views'])): r['use'] = 0
    pt = r.get('parapetTop')
    if pt is not None and not (isinstance(pt, list) and len(pt) >= 2 and all(isinstance(q, (list, tuple)) and len(q) == 2 for q in pt)): r['parapetTop'] = None
    ok = lambda o: all(isinstance(o.get(k), (int, float)) for k in ('x0', 'x1', 'z0', 'z1')) and o['x1'] > o['x0'] and o['z1'] > o['z0']
    for k in ('materials', 'banks', 'openings', 'bays'): r[k] = [o for o in (r.get(k) or []) if isinstance(o, dict) and ok(o)]
    r['bands'] = [b for b in (r.get('bands') or []) if isinstance(b, dict) and isinstance(b.get('zM'), (int, float))]
    r['piers'] = [b for b in (r.get('piers') or []) if isinstance(b, dict) and isinstance(b.get('xM'), (int, float))]
    return r


def material(m):
    base = FAMILY_RECIPE.get(m.get('family'), 'stucco')
    return base + hexok(m.get('hex')).lower()


def bank_cells(b):
    """Opening rectangles (metres) of a bank, as the grammar lays them out: evenly spaced cell centres."""
    nx, ny = max(1, int(b.get('columns', 1))), max(1, int(b.get('rows', 1)))
    dx, dz = (b['x1'] - b['x0']) / nx, (b['z1'] - b['z0']) / ny
    w = min(dx * .98, b.get('openingWidthM') or dx * .7); h = min(dz * .98, b.get('openingHeightM') or dz * .7)
    omit = {tuple(o) for o in b.get('omit', [])}
    xc = b.get('xCentersM') if b.get('xCentersM') and len(b['xCentersM']) == nx else [b['x0'] + (i + .5) * dx for i in range(nx)]
    zc = b.get('zCentersM') if b.get('zCentersM') and len(b['zCentersM']) == ny else [b['z0'] + (j + .5) * dz for j in range(ny)]
    out = []
    for i in range(nx):
        for j in range(ny):
            if (i, j) in omit: continue
            cx, cz = xc[i], zc[j]
            out.append((cx - w / 2, cz - h / 2, cx + w / 2, cz + h / 2))
    return out


def _style(o, dx=None, dz=None, bays=()):
    kind = KIND.get(o.get('kind', 'window'), 'window')
    st = {'kind': kind, 'frameMaterial': hexok(o.get('frame'), '#3a3d40')}
    if o.get('head') in ('arch', 'segmental'): st['head'] = o['head']
    if o.get('recessM') is not None: st['recessM'] = float(np.clip(o['recessM'], .02, 1.5))
    m, t = int(o.get('mullions') or 0), int(o.get('transoms') or 0)
    st['mullionFractions'] = [round((k + 1) / (m + 1), 4) for k in range(m)]
    st['transomFractions'] = [round((k + 1) / (t + 1), 4) for k in range(t)]
    if o.get('kind') == 'curtain': st.update(ribbon=True, mullionFractions=st['mullionFractions'] or [.5])
    elif o.get('ribbon') is not None: st['ribbon'] = bool(o['ribbon'])
    if dx: st['openingRatio'] = round(float(np.clip((o.get('openingWidthM') or dx * .7) / dx, .1, .98)), 4)
    if dz: st['openingHeightRatio'] = round(float(np.clip((o.get('openingHeightM') or dz * .7) / dz, .1, .98)), 4)
    for bay in bays:
        if o['x0'] < bay['x1'] and o['x1'] > bay['x0'] and o['z0'] < bay['z1'] and o['z1'] > bay['z0']:
            st.update(kind='projected_bay', projectionM=float(np.clip(bay.get('projectionM', .5), .1, 1.5)))
    return st


def to_profile(r, piece):
    W, H = float(piece['lengthM']), float(piece['heightM'])
    fx = lambda x: round(float(np.clip(x / W, 0, 1)), 5)
    fz = lambda z: round(float(np.clip(z / H, 0, 1.2)), 5)
    view = piece['views'][r.get('use', 0)]
    src = ['mapillary:' + view['image']]
    mats = r.get('materials') or [{'family': 'render', 'hex': '#b5b2a8', 'x0': 0, 'x1': W, 'z0': 0, 'z1': H}]
    main = mats[0]
    bid = piece['building']
    prof = {'id': 'regen-' + piece['id'], 'status': 'image-measured-auto-regen-v1',
            'match': {'featureId': re.sub(r'_p\d+$', '', bid), 'osmWayId': bid[4:] if bid.startswith('osm_') else None, 'partId': bid},
            'normal': piece['normal'], 'frameU': piece['frameU'], 'planeV': piece['planeV'], 'heightRefM': H, 'sourceIds': src,
            'material': material(main),
            'materialEvidence': {'family': main.get('family', 'render'), 'colorRgb': [int(hexok(main.get('hex'))[i:i + 2], 16) for i in (1, 3, 5)], 'sourceIds': src},
            'panels': [], 'groups': [], 'openings': [], 'bands': [], 'piers': [], 'parapets': [], 'unresolved': [],
            'reading': {'confidence': r.get('confidence'), 'notes': r.get('notes', ''), 'wallTopM': r.get('wallTopM'), 'storeys': r.get('storeys')}}
    if piece.get('baseM'): prof['zBaseM'] = float(piece['baseM'])   # a set-back tier, rectified from the roof it stands on
    prof['materialEvidence'].pop('family') if prof['materialEvidence']['family'] not in ('brick', 'terracotta', 'render', 'timber', 'metal-panel') else None
    if 'family' not in prof['materialEvidence']: prof.pop('materialEvidence')
    for m in mats[1:]:
        prof['panels'].append({'rect': [fx(m['x0']), fz(m['z0']), fx(m['x1']), fz(m['z1'])], 'material': material(m)})
    bays = r.get('bays') or []
    for b in r.get('banks') or []:
        nx, ny = max(1, int(b.get('columns', 1))), max(1, int(b.get('rows', 1)))
        dx, dz = (b['x1'] - b['x0']) / nx, (b['z1'] - b['z0']) / ny
        if dx <= 0 or dz <= 0: continue
        g = {'region': [fx(b['x0']), fz(b['z0']), fx(b['x1']), fz(b['z1'])], 'columns': nx, 'rows': ny,
             'omit': [list(o) for o in b.get('omit', [])], 'style': _style(b, dx, dz, bays)}
        # measured (uneven) cell centres, as fractions of the bank
        if b.get('xCentersM') and len(b['xCentersM']) == nx: g['xCenters'] = [round((x - b['x0']) / (b['x1'] - b['x0']), 5) for x in b['xCentersM']]
        if b.get('zCentersM') and len(b['zCentersM']) == ny: g['yCenters'] = [round((z - b['z0']) / (b['z1'] - b['z0']), 5) for z in b['zCentersM']]
        prof['groups'].append(g)
    for o in r.get('openings') or []:
        if o['x1'] <= o['x0'] or o['z1'] <= o['z0']: continue
        prof['openings'].append({'rect': [fx(o['x0']), fz(o['z0']), fx(o['x1']), fz(o['z1'])], 'style': _style(o, bays=bays), 'sourceIds': src})
    for c in r.get('canopies') or []:
        # a canopy belongs to the opening or bank whose head it shelters
        for target in [(g['region'], g['style']) for g in prof['groups']] + [(o['rect'], o['style']) for o in prof['openings']]:
            rect, st = target
            if rect[0] * W < c['x1'] and rect[2] * W > c['x0'] and abs(rect[3] * H - c['zM']) < 1.2:
                st.update(canopy=True, canopyDepthM=float(np.clip(c.get('depthM', 1.4), .4, 3.)))
    for b in r.get('bands') or []:
        prof['bands'].append({'height': fz(b['zM']), 'thicknessM': float(np.clip(b.get('thicknessM', .3), .05, 1.5)),
                              'depthM': float(np.clip(b.get('projectionM', .1) * 2 + .1, .1, 1.5)), 'projectionM': float(np.clip(b.get('projectionM', .1), 0, 1.)),
                              'material': hexok(b.get('hex'), '#9a9c9a')})
    top = r.get('wallTopM') or H
    for p in r.get('piers') or []:
        prof['piers'].append({'x': fx(p['xM']), 'widthM': float(np.clip(p.get('widthM', .5), .1, 3.)), 'projectionM': float(np.clip(p.get('projectionM', .1), .02, 1.)),
                              'material': hexok(p.get('hex'), '#b8b6ae'), 'y': [0, fz(top)]})
    if r.get('parapetTop'):
        prof['parapets'].append({'top': [[fx(x), fz(z)] for x, z in r['parapetTop']], 'thicknessM': .3, 'material': prof['material']})
    # the measured grid this side lends to the building's unseen walls: its largest upper bank of windows
    upper = [b for b in (r.get('banks') or []) if b.get('kind', 'window') in ('window', 'curtain') and b['z0'] > 2.5 and b.get('columns', 1) >= 1]
    if upper:
        b = max(upper, key=lambda b: (b['x1'] - b['x0']) * (b['z1'] - b['z0']))
        nx, ny = max(1, int(b.get('columns', 1))), max(1, int(b.get('rows', 1)))
        dx, dz = (b['x1'] - b['x0']) / nx, (b['z1'] - b['z0']) / ny
        st = r.get('storeys') or []
        pitch = float(np.median(np.diff(st))) if len(st) >= 3 else dz
        prof['reading'].update(floorPitchM=round(float(np.clip(pitch if ny == 1 else dz, 2.6, 6.)), 3), bayPitchM=round(float(dx), 3),
                               openingRatio=round(float(np.clip((b.get('openingWidthM') or dx * .7) / dx, .1, .98)), 3),
                               openingHeightRatio=round(float(np.clip((b.get('openingHeightM') or dz * .7) / dz, .1, .98)), 3),
                               shopfront=any(x.get('kind') == 'shopfront' for x in (r.get('banks') or []) + (r.get('openings') or [])),
                               ribbon=bool(b.get('ribbon') or b.get('kind') == 'curtain'))
    return prof


def overlay(r, piece, img_dir, out_path):
    """The reading drawn on the photograph it was measured from."""
    v = piece['views'][r.get('use', 0)]
    im = Image.open(pathlib.Path(img_dir) / v['rect']).convert('RGB'); d = ImageDraw.Draw(im)
    pad, ppm, H = v['rectPad'], v['ppm'], float(piece['heightM'])
    P = lambda x, z: (pad + x * ppm, pad + (H - z) * ppm)
    def rect(x0, z0, x1, z1, col, w=2):
        a, b = P(x0, z1), P(x1, z0); d.rectangle([a, b], outline=col, width=w)
    for k, m in enumerate(r.get('materials') or []):
        rect(m['x0'], m['z0'], m['x1'], m['z1'], (255, 0, 255), 1); d.text(P(m['x0'] + .2, m['z1'] - .2), f"M{k} {m.get('family')}", fill=(255, 0, 255))
    for k, b in enumerate(r.get('banks') or []):
        rect(b['x0'], b['z0'], b['x1'], b['z1'], (0, 160, 255), 1); d.text(P(b['x0'], b['z1'] + .5), f"B{k}", fill=(0, 160, 255))
        for c in bank_cells(b): rect(*c, (0, 255, 255), 2)
    for k, o in enumerate(r.get('openings') or []):
        rect(o['x0'], o['z0'], o['x1'], o['z1'], (255, 230, 0), 3); d.text(P(o['x0'], o['z1'] + .5), f"O{k}", fill=(255, 230, 0))
    for k, b in enumerate(r.get('bands') or []):
        rect(0, b['zM'] - b.get('thicknessM', .3) / 2, float(piece['lengthM']), b['zM'] + b.get('thicknessM', .3) / 2, (255, 40, 40), 2)
        d.text(P(.2, b['zM'] + .6), f"L{k} {b.get('kind', '')}", fill=(255, 40, 40))
    for k, p in enumerate(r.get('piers') or []):
        rect(p['xM'] - p.get('widthM', .5) / 2, 0, p['xM'] + p.get('widthM', .5) / 2, r.get('wallTopM') or H, (40, 255, 40), 2)
    for k, b in enumerate(r.get('bays') or []):
        rect(b['x0'], b['z0'], b['x1'], b['z1'], (255, 140, 0), 3)
    if r.get('wallTopM'):
        a, b = P(0, r['wallTopM']), P(float(piece['lengthM']), r['wallTopM']); d.line([a, b], fill=(255, 255, 255), width=2)
    if r.get('parapetTop'): d.line([P(x, z) for x, z in r['parapetTop']], fill=(255, 255, 255), width=3)
    im.save(out_path, quality=85)
    return out_path
