"""Readings (metres, measured on a rectified photograph) -> facade profiles the grammar compiles, and overlays that
draw a reading back onto its photograph for checking. Pure geometry; no place facts."""
import json, pathlib, re
import numpy as np
from PIL import Image, ImageDraw

FAMILY_RECIPE = {'brick': 'brick_modular', 'render': 'stucco', 'stone': 'cast_stone', 'concrete': 'venue_concrete',
                 'metal-panel': 'panel_silver', 'glass': 'panel_dark', 'timber': 'timber_dark', 'tile': 'terracotta_ribbed',
                 'terracotta': 'terracotta_panel'}
KIND = {'window': 'window', 'shopfront': 'shopfront', 'entry': 'entry', 'garage': 'rollup', 'rollup': 'rollup',
        'louvre': 'louvre', 'curtain': 'window'}


def hexok(h, default='#8a8d8f'):
    return h if isinstance(h, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', h) else default


def clean(r, piece):
    """Tolerate small format slips in a model's reading: a missing view index, malformed outlines, empty rectangles."""
    r = dict(r)
    from regen.reading_coverage import audit_reading
    r['_sourceCoverage'] = audit_reading(r)
    if not isinstance(r.get('use'), int) or not (0 <= r['use'] < len(piece['views'])): r['use'] = 0
    pt = r.get('parapetTop')
    if pt is not None and not (isinstance(pt, list) and len(pt) >= 2 and all(isinstance(q, (list, tuple)) and len(q) == 2 for q in pt)): r['parapetTop'] = None
    ok = lambda o: all(isinstance(o.get(k), (int, float)) for k in ('x0', 'x1', 'z0', 'z1')) and o['x1'] > o['x0'] and o['z1'] > o['z0']
    for k in ('materials', 'banks', 'bays'):
        r[k] = [o for o in (r.get(k) or []) if isinstance(o, dict) and ok(o)]
    r['openings'] = [{**o, '_sourceIndex': i} for i, o in enumerate(r.get('openings') or [])
                     if isinstance(o, dict) and ok(o)]
    r['bands'] = [b for b in (r.get('bands') or []) if isinstance(b, dict) and isinstance(b.get('zM'), (int, float))]
    r['piers'] = [b for b in (r.get('piers') or []) if isinstance(b, dict) and isinstance(b.get('xM'), (int, float))]
    num = lambda v: isinstance(v, (int, float))
    for k in ('rustication', 'ornaments', 'timbering'): r[k] = [o for o in (r.get(k) or []) if isinstance(o, dict) and ok(o)]
    r['quoins'] = [q for q in (r.get('quoins') or []) if isinstance(q, dict) and num(q.get('xM')) and num(q.get('z0')) and num(q.get('z1')) and q['z1'] > q['z0']]
    r['fireEscapes'] = [f for f in (r.get('fireEscapes') or []) if isinstance(f, dict) and num(f.get('x0')) and num(f.get('x1')) and f['x1'] > f['x0']
                        and isinstance(f.get('levels'), list) and all(num(z) for z in f['levels']) and f['levels']]
    r['balconies'] = [b for b in (r.get('balconies') or []) if isinstance(b, dict) and num(b.get('x0')) and num(b.get('x1')) and b['x1'] > b['x0'] and num(b.get('zM'))]
    r['hedges'] = [h for h in (r.get('hedges') or []) if isinstance(h, dict) and num(h.get('x0')) and num(h.get('x1')) and h['x1'] > h['x0'] and num(h.get('heightM'))]
    r['roofs'] = [q for q in (r.get('roofs') or []) if isinstance(q, dict) and num(q.get('x0')) and num(q.get('x1')) and q['x1'] > q['x0'] and num(q.get('zM'))]
    r['streetEdge'] = [q for q in (r.get('streetEdge') or []) if isinstance(q, dict) and num(q.get('x0')) and num(q.get('x1')) and q['x1'] > q['x0'] and num(q.get('heightM'))]
    r['planting'] = [q for q in (r.get('planting') or []) if isinstance(q, dict) and num(q.get('x0')) and num(q.get('x1')) and q['x1'] > q['x0'] and num(q.get('heightM'))]
    return r


def material(m):
    base = FAMILY_RECIPE.get(m.get('family'), 'stucco')
    if base == 'panel_silver':
        from pipeline.panel_finish import panel_recipe
        base = panel_recipe(m.get('hex'), m.get('finish'), m.get('coating'))
    # Optional typed observations alter only the selected surface recipe. Legacy
    # readings with no descriptors keep their byte-for-byte historical key.
    from pipeline import material_recipes
    return material_recipes.make_observed_key(
        base, hexok(m.get('hex')).lower(), family=m.get('family'),
        pattern=m.get('pattern'), finish=m.get('finish'), moduleM=m.get('moduleM'))


def _ornament_count(value, default, low, high):
    """Normalize explicitly typed ornament counts; never derive them from prose."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        return default
    return int(np.clip(round(float(value)), low, high))


def porch_access_profile(raw, fx):
    """Normalize explicitly observed porch access; return no geometry for unread access."""
    access = raw.get('access')
    if access is None: return None, None
    if not isinstance(access, dict): return None, 'access evidence must be an object'
    status = access.get('status')
    if status == 'occluded': return {'status': 'unresolved'}, 'porch access is occluded'
    if status != 'observed': return {'status': 'unresolved'}, 'access status must be observed or occluded'
    steps = access.get('steps')
    if not isinstance(steps, dict) or not isinstance(steps.get('present'), bool):
        return {'status': 'unresolved'}, 'observed access requires an explicit steps.present decision'
    result = {'status': 'observed', 'steps': {'present': steps['present']}, 'rails': None, 'posts': []}
    if steps['present']:
        count, width, run, center = (steps.get(k) for k in ('count', 'widthM', 'runM', 'centerOffsetM'))
        if (isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 40
                or not isinstance(width, (int, float)) or not .4 <= width <= 10
                or not isinstance(run, (int, float)) or not .1 <= run <= 10
                or not isinstance(center, (int, float)) or not -5 <= center <= 5):
            return {'status': 'unresolved'}, 'observed stairs require count, widthM, runM, and centerOffsetM'
        result['steps'].update({'count': count, 'widthM': float(width), 'runM': float(run),
                                'centerOffsetM': float(center),
                                'material': hexok(steps.get('hex'), '#817d76')})
    rails = access.get('rails')
    if rails is not None:
        sides = rails.get('sides') if isinstance(rails, dict) else None
        if not isinstance(sides, list) or any(s not in ('left', 'right') for s in sides) or len(set(sides)) != len(sides):
            return {'status': 'unresolved'}, 'rails require unique left/right sides'
        if sides:
            height = rails.get('heightM')
            if not isinstance(height, (int, float)) or not .4 <= height <= 1.6:
                return {'status': 'unresolved'}, 'side rails require measured heightM'
        result['rails'] = {'sides': list(sides), 'heightM': float(rails.get('heightM', .9)) if sides else None,
                           'material': hexok(rails.get('hex'), '#5b5a55')}
    posts = access.get('posts', [])
    if not isinstance(posts, list): return {'status': 'unresolved'}, 'posts must be a list'
    for post in posts:
        if not isinstance(post, dict): return {'status': 'unresolved'}, 'each post requires typed placement'
        side, position = post.get('side'), post.get('position')
        height, width = post.get('heightM'), post.get('widthM')
        kind = post.get('kind', 'post')
        if (side not in ('left', 'right') or not isinstance(position, (int, float)) or not 0 <= position <= 1
                or not isinstance(height, (int, float)) or not .2 <= height <= 2.2
                or not isinstance(width, (int, float)) or not .06 <= width <= .6
                or kind not in ('post', 'newel')):
            return {'status': 'unresolved'}, 'posts require side, position [0,1], heightM, widthM, and post/newel kind'
        result['posts'].append({'side': side, 'position': float(position), 'heightM': float(height),
                                'widthM': float(width), 'kind': kind,
                                'finial': bool(post.get('finial', False)),
                                'material': hexok(post.get('hex'), '#e8e4d6')})
    return result, None


def bay_material_zones(bay, fx, fz):
    """Normalize optional, explicitly observed bay cladding bands; return zones and unresolved reasons."""
    raw_zones = bay.get('materialZones')
    if raw_zones is None: return [], []
    if not isinstance(raw_zones, list): return [], ['materialZones must be a list']
    numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
    zones, errors = [], []
    for i, z in enumerate(raw_zones):
        if not isinstance(z, dict):
            errors.append(f'materialZones[{i}] must be an object'); continue
        z0, z1 = z.get('z0'), z.get('z1')
        x0, x1 = z.get('x0', bay['x0']), z.get('x1', bay['x1'])
        if (not numeric(z0) or not numeric(z1) or not bay['z0'] <= z0 < z1 <= bay['z1']
                or not numeric(x0) or not numeric(x1) or not bay['x0'] <= x0 < x1 <= bay['x1']):
            errors.append(f'materialZones[{i}] needs ordered metric bounds inside its bay'); continue
        family = z.get('family')
        color = z.get('hex', z.get('color'))
        if family not in FAMILY_RECIPE or not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
            errors.append(f'materialZones[{i}] needs a supported family and six-digit hex color'); continue
        pattern, finish, unit = z.get('pattern'), z.get('finish'), z.get('unitSizeM')
        if unit is not None and (not numeric(unit) or unit <= 0 or unit > 10):
            errors.append(f'materialZones[{i}].unitSizeM must be a positive finite metre value'); continue
        if unit is not None and pattern is None:
            errors.append(f'materialZones[{i}].unitSizeM requires an explicit pattern'); continue
        module = z.get('moduleM')
        if module is not None and (not isinstance(module, dict) or not module):
            errors.append(f'materialZones[{i}].moduleM must contain measured x/z repeats'); continue
        module = dict(module) if module is not None else None
        # A measured course pitch identifies the vertical repeat, independently
        # of shingle width. Never manufacture the unmeasured second axis.
        if unit is not None:
            axis = ('z' if pattern in ('horizontal_courses', 'straight_shingles', 'scalloped_shingles')
                    else 'x' if pattern == 'vertical_boards' else None)
            if axis is None:
                errors.append(f'materialZones[{i}].unitSizeM has no defined repeat axis for this pattern'); continue
            if module is not None and axis in module and (not numeric(module[axis]) or abs(module[axis]-unit) > 1e-6):
                errors.append(f'materialZones[{i}] has contradictory repeat dimensions'); continue
            module = {**(module or {}), axis: float(unit)}
        try:
            recipe = material({'family': family, 'hex': color, 'pattern': pattern,
                               'finish': finish, **({'moduleM': module} if module else {})})
        except (ValueError, TypeError):
            errors.append(f'materialZones[{i}] has an unsupported pattern, finish, or unit-size combination'); continue
        zone = {'rect': [fx(x0), fz(z0), fx(x1), fz(z1)], 'cladding': recipe,
                'family': family, 'hex': color.lower()}
        if pattern is not None: zone['pattern'] = pattern
        if finish is not None: zone['finish'] = finish
        if unit is not None: zone['unitSizeM'] = float(unit)
        if module is not None: zone['moduleM'] = module
        zones.append(zone)
    return zones, errors


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
    from pipeline.door_leaves import validate_cells
    kind = KIND.get(o.get('kind', 'window'), 'window')
    st = {'kind': kind, 'frameMaterial': hexok(o.get('frame'), '#3a3d40')}
    if o.get('head') in ('arch', 'pointed', 'segmental'): st['head'] = o['head']
    if isinstance(o.get('springRatio'), (int, float)) and o.get('head') in ('arch', 'pointed'): st['springRatio'] = float(np.clip(o['springRatio'], .2, .95))
    if isinstance(o.get('riseRatio'), (int, float)) and o.get('head') == 'segmental': st['riseRatio'] = float(np.clip(o['riseRatio'], .02, .5))
    if o.get('recessM') is not None:
        recess = float(o['recessM'])
        if not np.isfinite(recess) or recess < 0:
            raise ValueError('opening recess must be a finite nonnegative measurement')
        st['recessM'] = max(.02, recess)
    m, t = int(o.get('mullions') or 0), int(o.get('transoms') or 0)
    st['mullionFractions'] = [round((k + 1) / (m + 1), 4) for k in range(m)]
    st['transomFractions'] = [round((k + 1) / (t + 1), 4) for k in range(t)]
    # The door leaf and the surrounding casing can have different finishes.
    # In particular, a dark reveal does not colour every white garage panel.
    if o.get('kind') == 'garage':
        door = o.get('door') or {}
        leaf = hexok(door.get('hex'), st['frameMaterial'])
        st.update(material=leaf, slatMaterial=leaf)
        door_kind = door.get('kind', 'panel')
        if door_kind not in ('panel', 'glazed', 'grille', 'open'):
            door_kind = 'panel'
        construction = door.get('construction')
        if construction not in ('sectional', 'roller'):
            construction = None
        st['doorKind'] = door_kind
        if construction: st['doorConstruction'] = construction
        if 'cells' in door: st['doorCells'] = validate_cells(door['cells'])
        if door_kind == 'glazed': st['glazed'] = True
        if door_kind == 'grille': st['grille'] = True
        if door_kind == 'open': st['openLeaf'] = True
    if o.get('kind') == 'curtain': st.update(ribbon=True, mullionFractions=st['mullionFractions'] or [.5])
    elif o.get('ribbon') is not None: st['ribbon'] = bool(o['ribbon'])
    # reading v2 (areas/ring/read-v2.md): the window's sashes and panes, its trim, a door and the steps up to it
    if o.get('sash') in ('double-hung', 'casement', 'fixed', 'sliding'): st['sash'] = o['sash']
    if isinstance(o.get('lites'), list) and len(o['lites']) == 2: st['lites'] = [int(np.clip(v, 1, 8)) for v in o['lites']]
    if isinstance(o.get('meetingRail'), (int, float)): st['meetingRail'] = float(np.clip(o['meetingRail'], .3, .75))
    if isinstance(o.get('frameWidthM'), (int, float)): st['frameWidthM'] = float(np.clip(o['frameWidthM'], .03, .2))
    # the trim (hood, sill, keystone) takes the surround's colour when the reading gives no trimHex, as window banks never
    # do: falling back to the sash colour put navy hoods over white-cased windows
    trim = hexok(o.get('trimHex'), hexok((o.get('surround') or {}).get('hex') if isinstance(o.get('surround'), dict) else None, st['frameMaterial']))
    if isinstance(o.get('surround'), dict):
        st['surround'] = {'widthM': float(np.clip(o['surround'].get('widthM', .14), .04, .6)), 'projectionM': float(np.clip(o['surround'].get('projectionM', .04), .01, .2)),
                          'material': hexok(o['surround'].get('hex'), trim)}
    if isinstance(o.get('hood'), dict):
        h = o['hood']; st['hood'] = {'kind': h.get('kind') if h.get('kind') in ('cornice', 'pediment', 'segmental') else 'cornice',
                                     'heightM': float(np.clip(h.get('heightM', .3), .08, 1.2)), 'projectionM': float(np.clip(h.get('projectionM', .14), .03, .6)), 'material': hexok(h.get('hex'), trim)}
    if o.get('keystone'): st['keystone'] = True
    if isinstance(o.get('railing'), dict): st['railing'] = {'heightM': float(np.clip(o['railing'].get('heightM', .9), .3, 1.4)), 'material': hexok(o['railing'].get('hex'), '#2b2d2f')}
    if isinstance(o.get('door'), dict) and o.get('kind') == 'entry':
        st['door'] = {'kind': o['door'].get('kind') if o['door'].get('kind') in ('panel', 'glazed', 'double', 'grille', 'none') else 'panel', 'material': hexok(o['door'].get('hex'), st['frameMaterial'])}
        if 'panels' in o['door']:
            panels = o['door']['panels']
            if not isinstance(panels, list) or any(not isinstance(p, (list, tuple)) or len(p) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in p) or
                not (0 <= p[0] < p[2] <= 1 and 0 <= p[1] < p[3] <= 1) for p in panels):
                raise ValueError('door panels must be observed leaf-relative rectangles, or [] when unresolved')
            st['door']['panels'] = [list(p) for p in panels]
        if 'cells' in o['door']:
            st['door']['cells'] = validate_cells(o['door']['cells'])
    if isinstance(o.get('transomM'), (int, float)) and o['transomM'] > 0: st['transomM'] = float(np.clip(o['transomM'], .1, 1.6))
    if isinstance(o.get('steps'), dict) and o.get('kind') == 'entry':
        st['steps'] = {'count': int(np.clip(o['steps'].get('count', 0) or 0, 0, 40)), 'widthM': float(np.clip(o['steps'].get('widthM', 1.6), .6, 8.)),
                       'material': material({'family': o['steps'].get('family', 'stone'), 'hex': o['steps'].get('hex')})}
        for field in ('runM', 'bottomM'):
            value = o['steps'].get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                if field == 'bottomM' or value > 0:
                    st['steps'][field] = float(value)
        if isinstance(o['steps'].get('rails'), dict):
            st['steps']['rails'] = dict(o['steps']['rails'])
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
        # Curved bays are rendered as glazed facets across the whole projected span [x0, x1].
        # Insetting this reading interval reduced pane widths while those facets kept their full measured footprint;
        # the bay renderer then discarded the resulting short per-facet cuts.
        fa, fb = (x0, x1) if shape == 'curved' else (x0 + run, x1 - run)
        # Carry the bay's explicitly read pane divisions through to each laid
        # opening.  Missing fields stay missing: ordinary sash windows must not
        # acquire guessed mullions or transoms.
        win = {k: b[k] for k in ('sash', 'lites', 'meetingRail', 'frame', 'head', 'springRatio', 'riseRatio', 'mullions', 'transoms') if k in b}; win.setdefault('frame', b.get('trimHex'))
        front = b.get('front') or {'count': 1}; n = max(1, int(front.get('count', 1)))
        side = b.get('side'); sw = float(side.get('widthM', 0)) if isinstance(side, dict) else 0.
        pane_fields = ('sash', 'lites', 'meetingRail', 'frame', 'head', 'springRatio', 'riseRatio', 'mullions', 'transoms')
        front_win = {**win, **{k: front[k] for k in pane_fields if k in front}}
        side_win = {**win, **{k: side[k] for k in pane_fields if k in side}} if isinstance(side, dict) else win
        fw = float(front.get('widthM') or (fb - fa) * .7 / n)
        if not np.isfinite(fw) or fw <= 0: fw = max(0., fb - fa) * .7 / n
        for z0, z1 in rows:
            front_width = fb - fa
            if front_width > 0:
                # Keep each pane on the front face, even when the measured widths cannot fit.
                # Preserve the observed count and use a positive, modest gap in the constrained case.
                gap = min(.05, front_width / (4 * (n + 1)))
                fw_eff = min(fw, (front_width - (n + 1) * gap) / n)
                if fw_eff > 0:
                    gap = (front_width - n * fw_eff) / (n + 1)
                    for k in range(n):
                        xa = fa + gap + k * (fw_eff + gap)
                        r.setdefault('openings', []).append({**front_win, 'x0': xa, 'x1': xa + fw_eff, 'z0': z0, 'z1': z1, 'kind': 'window', 'bayWindow': True, 'bayFace': 'front'})
            if sw > .2 and run > .2 and shape != 'curved':
                side_width = min(sw * np.cos(np.arctan2(p, run)), max(0., run - .02))
                if side_width > 0:
                    half = side_width / 2           # a side window's extent along the wall
                    for c in (x0 + run / 2, x1 - run / 2):
                        r['openings'].append({**side_win, 'x0': c - half, 'x1': c + half, 'z0': z0, 'z1': z1, 'kind': 'window', 'bayWindow': True, 'bayFace': 'side'})
        # the wall's bank cells inside the bay are the bay's windows read on the image: dropped for the laid ones
        for bk in r.get('banks') or []:
            nx, ny = max(1, int(bk.get('columns', 1))), max(1, int(bk.get('rows', 1)))
            xc = bk.get('xCentersM') if bk.get('xCentersM') and len(bk['xCentersM']) == nx else [bk['x0'] + (i + .5) * (bk['x1'] - bk['x0']) / nx for i in range(nx)]
            zc = bk.get('zCentersM') if bk.get('zCentersM') and len(bk['zCentersM']) == ny else [bk['z0'] + (j + .5) * (bk['z1'] - bk['z0']) / ny for j in range(ny)]
            bk['omit'] = list(bk.get('omit', [])) + [[i, j] for i in range(nx) for j in range(ny) if x0 <= xc[i] <= x1 and b['z0'] <= zc[j] <= b['z1'] and [i, j] not in bk.get('omit', [])]
    return r


def to_profile(r, piece):
    from regen.reading_coverage import audit_reading, merge_coverage
    coverage = merge_coverage(r.get('_sourceCoverage'), audit_reading(r))
    r = lay_bay_windows(r)
    W, H = float(piece['lengthM']), float(piece['heightM'])
    num = lambda v: isinstance(v, (int, float))
    fx = lambda x: round(float(np.clip(x / W, 0, 1)), 5)
    fz = lambda z: round(float(np.clip(z / H, -.3, 1.2)), 5)   # below 0: a street lower than the building's ground (a garage on a hill)
    view = piece['views'][r.get('use', 0)]
    src = ['mapillary:' + view['image']]
    mats = r.get('materials') or [{'family': 'render', 'hex': '#b5b2a8', 'x0': 0, 'x1': W, 'z0': 0, 'z1': H}]
    # the wall's material is the one covering most of it, whatever order the reader listed them in: readers list a
    # facade bottom up as often as main first, and a third of two-material sides put a base course or a shopfront
    # surround first, which then clad the bays and every unseen wall of the building
    main = max(mats, key=lambda m: max(0., float(m.get('x1', W)) - float(m.get('x0', 0))) * max(0., float(m.get('z1', H)) - float(m.get('z0', 0))))
    bid = piece['building']
    prof = {'id': 'regen-' + piece['id'], 'status': 'image-measured-auto-regen-v1',
            'match': {'featureId': re.sub(r'_p\d+$', '', bid), 'osmWayId': bid[4:] if bid.startswith('osm_') else None, 'partId': bid},
            'normal': piece['normal'], 'frameU': piece['frameU'], 'planeV': piece['planeV'], 'heightRefM': H, 'sourceIds': src,
            'material': material(main),
            'materialEvidence': {'family': main.get('family', 'render'), 'colorRgb': [int(hexok(main.get('hex'))[i:i + 2], 16) for i in (1, 3, 5)], 'sourceIds': src,
                                 **{k: main[k] for k in ('pattern', 'finish', 'moduleM') if k in main}},
            'panels': [], 'groups': [], 'openings': [], 'bands': [], 'piers': [], 'parapets': [],
            'readingCoverage': coverage, 'unresolved': list(coverage['unresolved']) + [
                {'kind': 'reading-gap', **item} for item in coverage['issues']],
            'reading': {'confidence': r.get('confidence'), 'notes': r.get('notes', ''), 'wallTopM': r.get('wallTopM'), 'storeys': r.get('storeys')}}
    # Negative facade volumes are compiled only from explicit measured bounds;
    # ordinary balconies remain separate projecting details.
    from pipeline.facade_recesses import compile_reading as compile_recess_reading
    prof['recesses'] = compile_recess_reading(r, fx, fz)
    from pipeline.facade_signs import compile_reading as compile_sign_reading
    prof['signs'], sign_gaps = compile_sign_reading(r, piece, material)
    prof['unresolved'].extend(sign_gaps)
    if piece.get('baseM'): prof['zBaseM'] = float(piece['baseM'])   # a set-back tier, rectified from the roof it stands on
    _typed_surface = any(k in main for k in ('pattern', 'finish', 'moduleM'))
    _family = prof['materialEvidence']['family']
    if _typed_surface:
        from pipeline.material_recipes import _OBSERVED_PATTERNS
        if _family not in _OBSERVED_PATTERNS:
            raise ValueError(f'unsupported observed material family {_family!r}')
    elif _family not in ('brick', 'terracotta', 'render', 'timber', 'metal-panel'):
        prof['materialEvidence'].pop('family')
    if 'family' not in prof['materialEvidence']: prof.pop('materialEvidence')
    floors = [z for z in (r.get('storeys') or []) if isinstance(z, (int, float))]
    for m in (m for m in mats if m is not main):
        prof['panels'].append({'rect': [fx(m['x0']), fz(m['z0']), fx(m['x1']), fz(m['z1'])], 'material': material(m),
                               **({'toWallTop': True} if m.get('toWallTop') is True else {})})
    bays = r.get('bays') or []
    # a bay is drawn as one volume standing out from the wall, its windows set in its faces (pipeline/bays.py)
    prof['bays'] = []
    for bay_index, b in enumerate(bays):
        if b['x1'] <= b['x0'] or b['z1'] <= b['z0']: continue
        bay_profile = {'rect': [fx(b['x0']), fz(b['z0']), fx(b['x1']), fz(b['z1'])], 'projectionM': float(np.clip(b.get('projectionM', .6), .2, 1.5)),
                       **({'shape': b['shape']} if b.get('shape') in ('angled', 'square', 'curved') else {}),
                       **({'sideAngleDeg': float(np.clip(b['sideAngleDeg'], 20, 80))} if isinstance(b.get('sideAngleDeg'), (int, float)) else {}),
                       **({'trim': hexok(b['trimHex'])} if b.get('trimHex') else {}),
                       **({'cladding': material({'family': b.get('family', main.get('family', 'render')), 'hex': b['hex'],
                                                **{k: b[k] for k in ('pattern', 'finish', 'moduleM') if k in b}})} if b.get('hex') else {}),
                       'storeys': [fz(z) for z in floors if b['z0'] + .4 < z < b['z1'] - .4] if b.get('storeyTrim', True) else [],
                       'corbel': bool(b.get('corbel')), 'cap': b.get('cap') if b.get('cap') in ('cornice', 'flat') else 'cornice'}
        zones, zone_errors = bay_material_zones(b, fx, fz)
        if zones: bay_profile['claddingZones'] = zones
        for reason in zone_errors:
            prof['unresolved'].append({'kind': 'bay_cladding_zone', 'bayIndex': bay_index, 'reason': reason})
        prof['bays'].append(bay_profile)
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
    for opening_source_index, o in enumerate(r.get('openings') or []):
        if o['x1'] <= o['x0'] or o['z1'] <= o['z0']: continue
        prof['openings'].append({'rect': [fx(o['x0']), fz(o['z0']), fx(o['x1']), fz(o['z1'])],
                                 'style': _style(o), 'sourceIds': src,
                                 'sourceIndex': int(o.get('_sourceIndex', opening_source_index))})
    for c in r.get('canopies') or []:
        if not all(num(c.get(k)) for k in ('x0', 'x1', 'zM')) or c['x1'] <= c['x0']: continue
        # a canopy belongs to the opening or bank whose head it shelters; one over no head (a porte-cochere, an awning
        # over a lobby set back behind it) stands on its own
        held = False
        for target in [(g['region'], g['style']) for g in prof['groups']] + [(o['rect'], o['style']) for o in prof['openings']]:
            rect, st = target
            if rect[0] * W < c['x1'] and rect[2] * W > c['x0'] and abs(rect[3] * H - c['zM']) < 1.2:
                st.update(canopy=True, canopyDepthM=float(np.clip(c.get('depthM', 1.4), .4, 3.))); held = True
        if not held:
            prof.setdefault('canopies', []).append({'x': [c['x0'] / W, c['x1'] / W], 'z': fz(c['zM']), 'depthM': float(np.clip(c.get('depthM', 1.4), .4, 6.)),
                                                    'material': material({'family': c.get('family', 'render'), 'hex': c['hex']}) if c.get('hex') else '#e6e3da'})
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
        # Source measurements also describe Juliet guards and shallow pierced
        # window panels. A balcony-sized minimum changes their geometry and
        # invents a walkable slab. Keep finite positive dimensions as measured.
        depth = b.get('depthM', .8)
        rail_height = b.get('railingHeightM', 1.)
        if not num(depth) or not np.isfinite(depth) or depth <= 0: continue
        if not num(rail_height) or not np.isfinite(rail_height) or rail_height <= 0: continue
        has_deck = b.get('hasDeck') if isinstance(b.get('hasDeck'), bool) else float(depth) >= .3
        prof.setdefault('balconies', []).append({'x': [fx(b['x0']), fx(b['x1'])], 'z': fz(b['zM']), 'depthM': float(depth),
                                                 'hasDeck': has_deck,
                                                 **({'layoutEvidence': dict(b['layoutEvidence'])} if isinstance(b.get('layoutEvidence'),dict) else {}),
                                                 'railing': b.get('railing') if b.get('railing') in ('metal', 'balustrade', 'solid') else 'metal',
                                                 'railingHeightM': float(rail_height), 'material': hexok(b.get('hex'), '#2b2d2f'),
                                                 **({'slabMaterial': material({'family': b.get('slabFamily', 'render'), 'hex': b['slabHex']})} if b.get('slabHex') else {})})
    for q in r.get('roofs') or []:
        prof.setdefault('roofs', []).append({'x': [fx(q['x0']), fx(q['x1'])], 'z': fz(q['zM']), 'kind': q.get('kind') if q.get('kind') in ('tile-eave', 'pent', 'tile-roof', 'eave', 'mansard', 'gable') else 'tile-eave',
                                             'projectionM': float(np.clip(q.get('projectionM', .6), .1, 2.)), 'riseM': float(np.clip(q.get('riseM', .3), 0., 6. if q.get('kind') == 'mansard' else 3.)),
                                             **({'pitchDeg': float(q['pitchDeg'])} if num(q.get('pitchDeg')) else {}),
                                             'overhangM': float(np.clip(q.get('overhangM', .2), 0., 1.5)), 'material': material({'family': q.get('family', 'tile'), 'hex': q.get('hex', '#9a4a32')})})
    for q in r.get('streetEdge') or []:
        kind = q.get('kind') if q.get('kind') in ('wall', 'fence', 'gate', 'hedge', 'shrub', 'driveway', 'path') else 'wall'; paved = kind in ('driveway', 'path')
        if not paved and q['heightM'] < .1: continue
        prof.setdefault('streetEdge', []).append({'x': [q['x0'] / W, q['x1'] / W], 'kind': kind,
                                                  'heightM': float(np.clip(q['heightM'], 0. if paved else .2, 6.)), 'offsetM': float(np.clip(q.get('offsetM', 0.), 0., 40.)), 'baseM': float(np.clip(q.get('baseM', 0.), 0., 3.)),
                                                  'depthM': float(np.clip(q.get('depthM', 1.), .3, 25. if kind in ('hedge', 'shrub') else 4.)),   # a hedge may run back from the street
                                                  **({'pierWidthM': float(np.clip(q['pierWidthM'], .15, 1.2)), 'pierHeightM': float(np.clip(q.get('pierHeightM', q['heightM'] + .25), .3, 4.)),
                                                      **({'piers': [float(x) / W for x in q['piersM'] if num(x)]} if isinstance(q.get('piersM'), list) else {}),
                                                      **({'pierPitchM': float(np.clip(q['pierPitchM'], .8, 20.))} if num(q.get('pierPitchM')) else {})} if kind == 'wall' and num(q.get('pierWidthM')) else {}),
                                                  'material': material({'family': q.get('family', 'concrete' if paved else 'render'), 'hex': q.get('hex') or ('#a8a49a' if paved else '#d8d2c4')}) if kind == 'wall' or paved
                                                  else hexok(q.get('hex'), '#1e2220' if kind in ('fence', 'gate') else '#3f5a2a')})
    for q in r.get('planting') or []:
        # a trellis's or a planter's hex is the trellis or the pot; what grows on or in it is foliageHex (green by default)
        kind = q.get('kind') if q.get('kind') in ('shrub', 'hedge', 'trellis', 'climber', 'planter') else 'shrub'; holder = kind in ('trellis', 'planter')
        prof.setdefault('planting', []).append({'x': [q['x0'] / W, q['x1'] / W], 'kind': kind,
                                                'heightM': float(np.clip(q['heightM'], .2, 9.)), 'depthM': float(np.clip(q.get('depthM', .8), .2, 4.)),
                                                'offsetM': float(np.clip(q.get('offsetM', .3), -1., 25.)), 'planter': bool(q.get('planter') or kind == 'planter'),
                                                'material': hexok(q.get('foliageHex') if holder else q.get('hex'), '#4c6638'),
                                                **({'frameMaterial': hexok(q.get('hex'), '#5b4a3a')} if kind == 'trellis' else {}),
                                                **({'planterMaterial': hexok(q.get('planterHex') or q.get('hex'), '#9a9286')} if kind == 'planter' or q.get('planter') else {}),
                                                **({'bare': True} if q.get('climber') is False else {})})
    for h in r.get('hedges') or []:
        prof.setdefault('hedges', []).append({'x': [fx(h['x0']), fx(h['x1'])], 'heightM': float(np.clip(h['heightM'], .4, 9.)), 'depthM': float(np.clip(h.get('depthM', 1.2), .4, 4.)),
                                              'offsetM': float(np.clip(h.get('offsetM', .5), 0., 25.)), 'material': hexok(h.get('hex'), '#3f5a2c')})
    for porch_source_index, q in enumerate(r.get('porches') or []):
        # a porch or portico in front of the wall (pipeline/facade_detail.py): columns, beam, roof, steps
        if not (num(q.get('x0')) and num(q.get('x1')) and q['x1'] - q['x0'] >= .8 and num(q.get('topM'))): continue
        access, access_error = porch_access_profile(q, fx)
        if access_error:
            prof.setdefault('unresolved', []).append({'kind': 'porch_access', 'x': [fx(q['x0']), fx(q['x1'])], 'reason': access_error})
        legacy_steps = bool(q.get('steps', (q.get('floorM') or 0.) > .3))
        if access is not None:
            legacy_steps = bool(access.get('status') == 'observed' and access.get('steps', {}).get('present'))
        values = (q.get('floorM', 0.), q['topM'], q.get('depthM', 1.8))
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
            prof.setdefault('unresolved', []).append({'kind': 'porch', 'reason': 'porch dimensions must be finite measurements'})
            continue
        floor, top, depth = map(float, values)
        columns = q.get('columns', 2)
        if (not all(np.isfinite(v) for v in (floor, top, depth)) or top <= floor or depth <= 0.
                or isinstance(columns, bool) or not isinstance(columns, int) or columns < 0):
            prof.setdefault('unresolved', []).append({'kind': 'porch', 'reason': 'invalid measured porch dimensions'})
            continue
        porch = {'sourceIndex': int(q.get('_sourceIndex', porch_source_index)),
                                               'x': [fx(q['x0']), fx(q['x1'])], 'floorM': floor,
                                               'topM': top, 'depthM': depth,
                                               'columns': columns, 'columnMaterial': hexok(q.get('columnHex'), '#efece4'),
                                               'material': hexok(q.get('hex'), '#e8e4da'), 'roof': q.get('roof') if q.get('roof') in ('flat', 'gable', 'shed', 'balcony') else 'flat',
                                               'steps': legacy_steps}
        if access is not None: porch['access'] = access
        prof.setdefault('porches', []).append(porch)
    if r.get('accessRoutes') is not None:
        from pipeline.access_routes import compile_access_routes
        routes, route_errors = compile_access_routes(r.get('accessRoutes'))
        if routes: prof['accessRoutes'] = routes
        for route_index, route in enumerate(routes):
            if route.get('status') == 'unresolved':
                prof.setdefault('unresolved', []).append({'kind': 'access_route', 'index': route_index,
                                                          'reason': route.get('reason', 'route measurements unresolved')})
        for route_index, reason in route_errors:
            if not any(item.get('kind') == 'access_route' and item.get('index') == route_index
                       for item in prof.get('unresolved', [])):
                prof.setdefault('unresolved', []).append({'kind': 'access_route', 'index': route_index, 'reason': reason})
    for o in r.get('timbering') or []:
        # half-timbering: studs at their read pitch across the region, rails at the read heights (the frame's foot and
        # head are drawn by pipeline/facade_detail.py), cut by the wall's outline there (a gable's rake)
        pitch = float(np.clip(o.get('postPitchM') or 1.2, .3, 4.)); w = float(np.clip(o.get('widthM') or .18, .1, .35))
        n = max(2, int(round((o['x1'] - o['x0']) / pitch)) + 1); xs = np.linspace(o['x0'] + w / 2, o['x1'] - w / 2, n)
        prof.setdefault('timbering', []).append({'rect': [fx(o['x0']), fz(o['z0']), fx(o['x1']), fz(o['z1'])], 'posts': [[fx(x - w / 2), fx(x + w / 2)] for x in xs],
                                                 'rails': [fz(z) for z in (o.get('rails') or []) if num(z) and o['z0'] < z < o['z1']], 'widthM': w,
                                                 'projectionM': .04, 'material': hexok(o.get('hex'), '#2e2320')})
    for o in r.get('ornaments') or []:
        shape = o.get('shape') if o.get('shape') in ('roundel', 'cartouche', 'garland', 'panel') else 'panel'
        relief = {'rect': [fx(o['x0']), fz(o['z0']), fx(o['x1']), fz(o['z1'])],
                  'projectionM': float(np.clip(o.get('projectionM', .04), .01, .3)),
                  'shape': shape,
                  'repeatCount': _ornament_count(o.get('repeatCount'), 1, 1, 20),
                  'material': material({'family': o.get('family', 'stone'), 'hex': o.get('hex')})}
        # Explicit semantic details refine the broad legacy shape. They remain
        # optional so existing readings keep their established rendering.
        if o.get('outerOutline') in ('circle', 'oval', 'pointed'):
            relief['outerOutline'] = o['outerOutline']
        if o.get('motif') in ('radial', 'rosette', 'unknown'):
            relief['motif'] = o['motif']
        if isinstance(o.get('cardinalFlourishes'), bool):
            relief['cardinalFlourishes'] = o['cardinalFlourishes']
        ring_count = o.get('ringCount')
        if ring_count is not None and not isinstance(ring_count, bool) and isinstance(ring_count, (int, float)) and np.isfinite(ring_count):
            relief['ringCount'] = _ornament_count(ring_count, 1, 1, 4)
        curve_depth = o.get('curveDepth')
        if curve_depth is not None and not isinstance(curve_depth, bool) and isinstance(curve_depth, (int, float)) and np.isfinite(curve_depth):
            relief['curveDepth'] = float(np.clip(curve_depth, 0., 1.))
        prof.setdefault('reliefs', []).append(relief)
    top = r.get('wallTopM') or H
    for p in r.get('piers') or []:
        prof['piers'].append({'x': fx(p['xM']), 'widthM': float(np.clip(p.get('widthM', .5), .1, 3.)), 'projectionM': float(np.clip(p.get('projectionM', .1), .02, 1.)),
                              'material': hexok(p.get('hex'), '#b8b6ae'),
                              'y': [fz(p['z0']), fz(p['z1'])] if num(p.get('z0')) and num(p.get('z1')) and p['z1'] > p['z0'] else [0, fz(top)]})   # a column through one storey
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
