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
    num = lambda v: isinstance(v, (int, float))
    for k in ('rustication', 'ornaments'): r[k] = [o for o in (r.get(k) or []) if isinstance(o, dict) and ok(o)]
    r['quoins'] = [q for q in (r.get('quoins') or []) if isinstance(q, dict) and num(q.get('xM')) and num(q.get('z0')) and num(q.get('z1')) and q['z1'] > q['z0']]
    r['fireEscapes'] = [f for f in (r.get('fireEscapes') or []) if isinstance(f, dict) and num(f.get('x0')) and num(f.get('x1')) and f['x1'] > f['x0']
                        and isinstance(f.get('levels'), list) and all(num(z) for z in f['levels']) and f['levels']]
    r['balconies'] = [b for b in (r.get('balconies') or []) if isinstance(b, dict) and num(b.get('x0')) and num(b.get('x1')) and b['x1'] > b['x0'] and num(b.get('zM'))]
    r['hedges'] = [h for h in (r.get('hedges') or []) if isinstance(h, dict) and num(h.get('x0')) and num(h.get('x1')) and h['x1'] > h['x0'] and num(h.get('heightM'))]
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


def _style(o, dx=None, dz=None):
    kind = KIND.get(o.get('kind', 'window'), 'window')
    st = {'kind': kind, 'frameMaterial': hexok(o.get('frame'), '#3a3d40')}
    if o.get('head') in ('arch', 'segmental'): st['head'] = o['head']
    if isinstance(o.get('springRatio'), (int, float)) and o.get('head') == 'arch': st['springRatio'] = float(np.clip(o['springRatio'], .2, .9))
    if o.get('recessM') is not None: st['recessM'] = float(np.clip(o['recessM'], .02, 1.5))
    m, t = int(o.get('mullions') or 0), int(o.get('transoms') or 0)
    st['mullionFractions'] = [round((k + 1) / (m + 1), 4) for k in range(m)]
    st['transomFractions'] = [round((k + 1) / (t + 1), 4) for k in range(t)]
    # a vehicle door is a panelled door in the colour read for it (its frame), not a zinc roller shutter (drawn black)
    if o.get('kind') == 'garage': st.update(material=st['frameMaterial'], slatMaterial=st['frameMaterial'], slatPitchM=.5)
    if o.get('kind') == 'curtain': st.update(ribbon=True, mullionFractions=st['mullionFractions'] or [.5])
    elif o.get('ribbon') is not None: st['ribbon'] = bool(o['ribbon'])
    # reading v2 (areas/ring/read-v2.md): the window's sashes and panes, its trim, a door and the steps up to it
    if o.get('sash') in ('double-hung', 'casement', 'fixed', 'sliding'): st['sash'] = o['sash']
    if isinstance(o.get('lites'), list) and len(o['lites']) == 2: st['lites'] = [int(np.clip(v, 1, 8)) for v in o['lites']]
    if isinstance(o.get('meetingRail'), (int, float)): st['meetingRail'] = float(np.clip(o['meetingRail'], .3, .75))
    if isinstance(o.get('frameWidthM'), (int, float)): st['frameWidthM'] = float(np.clip(o['frameWidthM'], .03, .2))
    trim = hexok(o.get('trimHex'), st['frameMaterial'])
    if isinstance(o.get('surround'), dict):
        st['surround'] = {'widthM': float(np.clip(o['surround'].get('widthM', .14), .04, .6)), 'projectionM': float(np.clip(o['surround'].get('projectionM', .04), .01, .2)),
                          'material': hexok(o['surround'].get('hex'), trim)}
    if isinstance(o.get('hood'), dict):
        h = o['hood']; st['hood'] = {'kind': h.get('kind') if h.get('kind') in ('cornice', 'pediment', 'segmental') else 'cornice',
                                     'heightM': float(np.clip(h.get('heightM', .3), .08, 1.2)), 'projectionM': float(np.clip(h.get('projectionM', .14), .03, .6)), 'material': hexok(h.get('hex'), trim)}
    if o.get('keystone'): st['keystone'] = True
    if isinstance(o.get('railing'), dict): st['railing'] = {'heightM': float(np.clip(o['railing'].get('heightM', .9), .3, 1.4)), 'material': hexok(o['railing'].get('hex'), '#2b2d2f')}
    if isinstance(o.get('door'), dict) and o.get('kind') == 'entry':
        st['door'] = {'kind': o['door'].get('kind') if o['door'].get('kind') in ('panel', 'glazed', 'double', 'none') else 'panel', 'material': hexok(o['door'].get('hex'), st['frameMaterial'])}
    if isinstance(o.get('transomM'), (int, float)) and o['transomM'] > 0: st['transomM'] = float(np.clip(o['transomM'], .1, 1.6))
    if isinstance(o.get('steps'), dict) and o.get('kind') == 'entry':
        st['steps'] = {'count': int(np.clip(o['steps'].get('count', 0) or 0, 0, 40)) or None, 'widthM': float(np.clip(o['steps'].get('widthM', 1.6), .6, 8.)),
                       'material': material({'family': o['steps'].get('family', 'stone'), 'hex': o['steps'].get('hex')})}
    if dx: st['openingRatio'] = round(float(np.clip((o.get('openingWidthM') or dx * .7) / dx, .1, .98)), 4)
    if dz: st['openingHeightRatio'] = round(float(np.clip((o.get('openingHeightM') or dz * .7) / dz, .1, .98)), 4)
    return st


def lay_bay_windows(r):
    """A copy of a reading with the windows of its bays laid on their faces, for bays read per storey (rows, front,
    side). In a rectified photograph a bay's windows are shifted by parallax (they stand out of the wall plane), so they
    are read as the storeys' sill and head heights and the windows' built widths, not as positions on the image; the
    wall's bank cells inside such a bay are dropped for the laid ones."""
    r = json.loads(json.dumps(r))
    for b in r.get('bays') or []:
        rows = [q for q in (b.get('rows') or []) if isinstance(q, list) and len(q) == 2 and q[1] > q[0]]
        if not rows: continue
        p = float(np.clip(b.get('projectionM', .6), .2, 1.5)); x0, x1 = b['x0'], b['x1']; shape = b.get('shape')
        run = p / np.tan(np.radians(float(np.clip(b.get('sideAngleDeg') or 45., 20, 80))))
        if shape == 'square' or (shape != 'curved' and x1 - x0 < 2 * run + .4): run = 0.
        fa, fb = (x0 + run, x1 - run) if shape != 'curved' else (x0 + .15 * (x1 - x0), x1 - .15 * (x1 - x0))
        win = {k: b[k] for k in ('sash', 'lites', 'meetingRail', 'frame', 'head', 'springRatio') if k in b}; win.setdefault('frame', b.get('trimHex'))
        front = b.get('front') or {'count': 1}; n = max(1, int(front.get('count', 1))); fw = float(front.get('widthM') or (fb - fa) * .7 / n)
        side = b.get('side'); sw = float(side.get('widthM', 0)) if isinstance(side, dict) else 0.
        for z0, z1 in rows:
            gap = (fb - fa - n * fw) / (n + 1)
            for k in range(n):
                xa = fa + gap + k * (fw + gap); r.setdefault('openings', []).append({**win, 'x0': xa, 'x1': xa + fw, 'z0': z0, 'z1': z1, 'kind': 'window', 'bayWindow': True})
            if sw > .2 and run > .2 and shape != 'curved':
                half = sw * np.cos(np.arctan2(p, run)) / 2           # a side window's extent along the wall
                for c in (x0 + run / 2, x1 - run / 2):
                    r['openings'].append({**win, 'x0': c - half, 'x1': c + half, 'z0': z0, 'z1': z1, 'kind': 'window', 'bayWindow': True})
        # the wall's bank cells inside the bay are the bay's windows read on the image: dropped for the laid ones
        for bk in r.get('banks') or []:
            nx, ny = max(1, int(bk.get('columns', 1))), max(1, int(bk.get('rows', 1)))
            xc = bk.get('xCentersM') if bk.get('xCentersM') and len(bk['xCentersM']) == nx else [bk['x0'] + (i + .5) * (bk['x1'] - bk['x0']) / nx for i in range(nx)]
            zc = bk.get('zCentersM') if bk.get('zCentersM') and len(bk['zCentersM']) == ny else [bk['z0'] + (j + .5) * (bk['z1'] - bk['z0']) / ny for j in range(ny)]
            bk['omit'] = list(bk.get('omit', [])) + [[i, j] for i in range(nx) for j in range(ny) if x0 <= xc[i] <= x1 and b['z0'] <= zc[j] <= b['z1'] and [i, j] not in bk.get('omit', [])]
    return r


def to_profile(r, piece):
    r = lay_bay_windows(r)
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
    floors = [z for z in (r.get('storeys') or []) if isinstance(z, (int, float))]
    for m in mats[1:]:
        prof['panels'].append({'rect': [fx(m['x0']), fz(m['z0']), fx(m['x1']), fz(m['z1'])], 'material': material(m)})
    bays = r.get('bays') or []
    # a bay is drawn as one volume standing out from the wall, its windows set in its faces (pipeline/bays.py)
    prof['bays'] = [{'rect': [fx(b['x0']), fz(b['z0']), fx(b['x1']), fz(b['z1'])], 'projectionM': float(np.clip(b.get('projectionM', .6), .2, 1.5)),
                     **({'shape': b['shape']} if b.get('shape') in ('angled', 'square', 'curved') else {}),
                     **({'sideAngleDeg': float(np.clip(b['sideAngleDeg'], 20, 80))} if isinstance(b.get('sideAngleDeg'), (int, float)) else {}),
                     **({'trim': hexok(b['trimHex'])} if b.get('trimHex') else {}),
                     'storeys': [fz(z) for z in floors if b['z0'] + .4 < z < b['z1'] - .4] if b.get('storeyTrim', True) else [],
                     'corbel': bool(b.get('corbel')), 'cap': b.get('cap') if b.get('cap') in ('cornice', 'flat') else 'cornice'}
                    for b in bays if b['x1'] > b['x0'] and b['z1'] > b['z0']]
    for b in r.get('banks') or []:
        nx, ny = max(1, int(b.get('columns', 1))), max(1, int(b.get('rows', 1)))
        dx, dz = (b['x1'] - b['x0']) / nx, (b['z1'] - b['z0']) / ny
        if dx <= 0 or dz <= 0: continue
        g = {'region': [fx(b['x0']), fz(b['z0']), fx(b['x1']), fz(b['z1'])], 'columns': nx, 'rows': ny,
             'omit': [list(o) for o in b.get('omit', [])], 'style': _style(b, dx, dz)}
        # measured (uneven) cell centres, as fractions of the bank
        if b.get('xCentersM') and len(b['xCentersM']) == nx: g['xCenters'] = [round((x - b['x0']) / (b['x1'] - b['x0']), 5) for x in b['xCentersM']]
        if b.get('zCentersM') and len(b['zCentersM']) == ny: g['yCenters'] = [round((z - b['z0']) / (b['z1'] - b['z0']), 5) for z in b['zCentersM']]
        prof['groups'].append(g)
    for o in r.get('openings') or []:
        if o['x1'] <= o['x0'] or o['z1'] <= o['z0']: continue
        prof['openings'].append({'rect': [fx(o['x0']), fz(o['z0']), fx(o['x1']), fz(o['z1'])], 'style': _style(o), 'sourceIds': src})
    for c in r.get('canopies') or []:
        # a canopy belongs to the opening or bank whose head it shelters
        for target in [(g['region'], g['style']) for g in prof['groups']] + [(o['rect'], o['style']) for o in prof['openings']]:
            rect, st = target
            if rect[0] * W < c['x1'] and rect[2] * W > c['x0'] and abs(rect[3] * H - c['zM']) < 1.2:
                st.update(canopy=True, canopyDepthM=float(np.clip(c.get('depthM', 1.4), .4, 3.)))
    for b in r.get('bands') or []:
        T, Pj = float(np.clip(b.get('thicknessM', .3), .05, 2.5)), float(np.clip(b.get('projectionM', .1), 0, 1.5))
        band = {'height': fz(b['zM']), 'thicknessM': T, 'depthM': float(np.clip(Pj * 2 + .1, .1, 3.)), 'projectionM': Pj, 'material': hexok(b.get('hex'), '#9a9c9a')}
        if isinstance(b.get('x0'), (int, float)) and isinstance(b.get('x1'), (int, float)) and b['x1'] > b['x0']: band['xRange'] = [fx(b['x0']), fx(b['x1'])]
        prof['bands'].append(band)
        # a moulded cornice or string course steps out in three members; modillions, dentils or brackets run under it
        kind = b.get('profile')
        if kind in ('moulded', 'modillion', 'dentil', 'bracketed') and Pj > .02:
            band['section'] = [(-T * .35, T * .3, Pj * .35), (-T * .05, T * .3, Pj * .7), (T * .3, T * .4, Pj)]
            row = {'modillion': (0., T * .22, .13, Pj * .8, .5), 'dentil': (-T * .2, min(.12, T * .2), .07, Pj * .45, .14),
                   'bracketed': (-T * .1, T * .75, .2, Pj * .85, float(np.clip(b.get('bracketPitchM') or 1.1, .3, 3.)))}.get(kind)
            if row:
                rise, hh, size, pr, pitch = row
                prof.setdefault('ornaments', []).append({'kind': 'dentil', 'z': fz(b['zM'] + rise), 'xRange': band.get('xRange', [0, 1]), 'pitchM': pitch,
                                                         'sizeM': size, 'heightM': hh, 'projectionM': pr, 'material': band['material']})
    for q in r.get('rustication') or []:
        prof.setdefault('rustication', []).append({'rect': [fx(q['x0']), fz(q['z0']), fx(q['x1']), fz(q['z1'])], 'courseM': float(np.clip(q.get('courseM', .45), .2, 1.2)),
                                                   **({'material': material({'family': q.get('family', 'stone'), 'hex': q['hex']})} if q.get('hex') else {})})
    for q in r.get('quoins') or []:
        prof.setdefault('quoins', []).append({'x': fx(q['xM']), 'y': [fz(q['z0']), fz(q['z1'])], 'widthM': float(np.clip(q.get('widthM', .7), .2, 1.5)),
                                              'courseM': float(np.clip(q.get('courseM', .45), .2, 1.2)), 'material': material({'family': q.get('family', 'stone'), 'hex': q.get('hex')})})
    for f in r.get('fireEscapes') or []:
        prof.setdefault('fireEscapes', []).append({'x': [fx(f['x0']), fx(f['x1'])], 'levels': [fz(z) for z in f['levels']], 'depthM': float(np.clip(f.get('depthM', .9), .5, 1.6)),
                                                   'material': hexok(f.get('hex'), '#2b2d2f')})
    for b in r.get('balconies') or []:
        prof.setdefault('balconies', []).append({'x': [fx(b['x0']), fx(b['x1'])], 'z': fz(b['zM']), 'depthM': float(np.clip(b.get('depthM', .8), .3, 2.5)),
                                                 'railing': b.get('railing') if b.get('railing') in ('metal', 'balustrade', 'solid') else 'metal',
                                                 'railingHeightM': float(np.clip(b.get('railingHeightM', 1.), .5, 1.4)), 'material': hexok(b.get('hex'), '#2b2d2f'),
                                                 **({'slabMaterial': material({'family': b.get('slabFamily', 'render'), 'hex': b['slabHex']})} if b.get('slabHex') else {})})
    for h in r.get('hedges') or []:
        prof.setdefault('hedges', []).append({'x': [fx(h['x0']), fx(h['x1'])], 'heightM': float(np.clip(h['heightM'], .4, 9.)), 'depthM': float(np.clip(h.get('depthM', 1.2), .4, 4.)),
                                              'offsetM': float(np.clip(h.get('offsetM', .5), 0., 25.)), 'material': hexok(h.get('hex'), '#3f5a2c')})
    for o in r.get('ornaments') or []:
        prof.setdefault('reliefs', []).append({'rect': [fx(o['x0']), fz(o['z0']), fx(o['x1']), fz(o['z1'])], 'projectionM': float(np.clip(o.get('projectionM', .04), .01, .3)),
                                               'material': material({'family': o.get('family', 'stone'), 'hex': o.get('hex')})})
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
