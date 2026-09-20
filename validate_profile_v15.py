#!/usr/bin/env python
"""Validate v15 facade profiles and compile each one against a synthetic 20 m x 12 m wall.

The checks are structural and location-free: value ranges, group/opening overlaps, rows/columns >= 1,
material names the material bank accepts, unit-length normal. The compile target is a synthetic
frontage built from the profile's own normal, with a stub ctx ('cube', 'cylinder', 'polys') like the
test files use, so the run needs no LiDAR, no footprints and no network. Profiles are data; their ids
and match keys are join keys only and never select a rule here.

Usage (from the workspace root):
  PYTHONPATH=work work/geoenv/bin/python work/validate_profile_v15.py [PROFILE_JSON ...]
      [--fits work/pipeline/facade-layouts.json] [--boundary-fits work/pipeline/boundary-layouts.json]
      [--no-fits] [--wall 20x12] [--json REPORT.json] [--materials]

A PROFILE_JSON is either {"profiles":[...]} or a single profile object. With no paths the shipped
pipeline/facade-profiles.json is validated. Exit status 1 when any profile has an error or a compile
exception.
"""
_PUT=[]
import argparse, ast, copy, inspect, json, math, pathlib, re, sys, traceback
import numpy as np
from shapely.geometry import Polygon, box
from pipeline import facade_grammar, surface_evidence, material_recipes
import pipeline.materials as materials_module

WORK = pathlib.Path(__file__).resolve().parent
KINDS = ['window', 'entry', 'shopfront', 'rollup', 'terrace_frame', 'projected_bay']   # facade_grammar.compile_wall branches
NEGATIVE_LABELS = ['solid', 'opaque-panel', 'sign-carrier']                             # surface_evidence.admissible vetoes
LABELS = NEGATIVE_LABELS + ['unknown']
POLICIES = ['bounded-repeat', 'observed-only']
PROFILE_KEYS = {'id', 'match', 'normal', 'sourceIds', 'material', 'materialEvidence', 'panels', 'groups', 'bands', 'openings',
                'surfaceEvidence', 'openingPolicy', 'planeToleranceM', 'copingMaterial', 'unresolved', 'description', 'displayName', 'piers', 'observedWallTopM', 'heightRefM', 'parapets', 'ornaments', 'signs'}
GROUP_KEYS = {'region', 'columns', 'rows', 'style', 'omit', 'fitId', 'xCenters', 'yCenters', 'spacingX', 'spacingY', 'id', 'note', 'rowPitchM', 'columnPitchM', 'minHeightM', 'maxHeightM', 'pierWidthM', 'pierProjectionM', 'pierMaterial', 'spandrelMaterial', 'spandrelRecessM'}
STYLE_KEYS = {'kind', 'openingRatio', 'openingHeightRatio', 'frameWidthM', 'frameMaterial', 'projectionM', 'recessM', 'glassMaterial',
              'mullionFractions', 'transomFractions', 'shade', 'canopy', 'canopyDepthM', 'material', 'head', 'springRatio', 'riseRatio', 'material', 'slatMaterial', 'reliefArch', 'slatPitchM'}
STYLE_DEFAULTS = {'kind': 'window', 'openingRatio': .7, 'openingHeightRatio': .7, 'frameWidthM': .06, 'frameMaterial': 'charcoal',
                  'projectionM': .04, 'recessM': .2, 'glassMaterial': 'glass', 'mullionFractions': [.25], 'transomFractions': [.76],
                  'shade': False, 'canopy': False, 'canopyDepthM': 1.6, 'material': 'zinc'}
METRIC_RANGES = {'frameWidthM': (0, .5), 'recessM': (0, 3), 'projectionM': (0, 1.5), 'canopyDepthM': (0, 4)}
SCAN_NAMES = ['brick', 'asphalt', 'concrete']   # Materials.__init__ zips these names onto the first three manifest entries


def known_materials():
    """Every material name Materials.get resolves, keyed to where it resolves. Anything else raises at mesh emission."""
    src = inspect.getsource(materials_module.Materials.get)
    aliases = ast.literal_eval(re.search(r'aliases=(\{.*?\})', src).group(1))
    colors = ast.literal_eval(re.search(r'colors=(\{.*?\})', src, re.S).group(1))
    palette = {}
    gen = WORK / 'generate_detailed_v14.py'
    if gen.exists():
        m = re.search(r'^palette=(\{.*?\})\s*$', gen.read_text(), re.M)
        if m: palette = ast.literal_eval(m.group(1))
    out = {}
    for k, v in aliases.items(): out[k] = 'alias->' + v
    for k in material_recipes.RECIPES: out.setdefault(k, 'recipe')
    for k in SCAN_NAMES: out.setdefault(k, 'scan')
    for k in colors: out.setdefault(k, 'color')
    for k in palette: out.setdefault(k, 'palette')
    return out


def material_families():
    """The family/finish table inside surface_evidence.resolve_material, read from its source so the two cannot drift."""
    src = inspect.getsource(surface_evidence.resolve_material)
    return ast.literal_eval(re.search(r'families=(\{.*?\}\})\s*\n', src, re.S).group(1))


KNOWN = known_materials()
FAMILIES = material_families()


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Report:
    def __init__(self):
        self.errors = [];
        self.warnings = []

    def err(self, where, msg): self.errors.append(where + ': ' + msg)

    def warn(self, where, msg): self.warnings.append(where + ': ' + msg)


def check_material(r, where, name, allow_missing=True):
    if name is None:
        if not allow_missing: r.err(where, 'material is required')
        return
    if not isinstance(name, str):
        r.err(where, 'material must be a string, got %r' % (name,)); return
    if '#' in name:
        base, hexcol = name.split('#', 1)
        from pipeline import material_recipes
        if base in material_recipes.RECIPES and re.fullmatch(r'[0-9a-fA-F]{6}', hexcol): return   # colour variant of a recipe
        r.err(where, 'colour variant %r needs a recipe base and a 6-digit hex colour' % (name,)); return
    if name not in KNOWN:
        hexlike = re.fullmatch(r'#?[0-9a-fA-F]{6}', name)
        r.err(where, 'unknown material %r (not a recipe, scan, colour, palette or alias name%s)' % (name, '; would be read as a hex colour' if hexlike else '; crashes Materials.get'))


def check_rect(r, where, rect, clip_warning=True):
    """Fractional rectangle [x0,y0,x1,y1] of the frontage: 0<=x0<x1<=1, 0<=y0<y1<=1."""
    if not (isinstance(rect, list) and len(rect) == 4 and all(is_num(v) for v in rect)):
        r.err(where, 'rect/region must be four finite numbers [x0,y0,x1,y1], got %r' % (rect,)); return False
    x0, y0, x1, y1 = rect
    ok = True
    if not (x0 < x1 and y0 < y1): r.err(where, 'rect/region must have x0<x1 and y0<y1: %r' % (rect,)); ok = False
    if min(rect) < 0 or max(rect) > 1: r.err(where, 'rect/region fractions must lie in [0,1]: %r' % (rect,)); ok = False
    return ok


def check_polygon(r, where, poly):
    if not (isinstance(poly, list) and len(poly) >= 3 and all(isinstance(p, list) and len(p) == 2 and all(is_num(v) for v in p) for p in poly)):
        r.err(where, 'polygon must be >=3 [x,y] fraction pairs'); return False
    flat = [v for p in poly for v in p]
    if min(flat) < 0 or max(flat) > 1: r.err(where, 'polygon fractions must lie in [0,1]'); return False
    return True


def check_shape(r, where, rec):
    if 'polygon' in rec: return check_polygon(r, where, rec['polygon'])
    if 'rect' in rec: return check_rect(r, where, rec['rect'])
    r.err(where, 'needs rect or polygon'); return False


def check_fraction_list(r, where, key, values):
    if not isinstance(values, list): r.err(where, key + ' must be a list of fractions'); return
    for v in values:
        if not is_num(v) or not (0 < v < 1): r.err(where, key + ' entries must be fractions strictly inside (0,1), got %r' % (v,))


def check_style(r, where, style, explicit=False):
    if not isinstance(style, dict): r.err(where, 'style must be an object'); return
    for k in style:
        if k not in STYLE_KEYS: r.warn(where, 'style key %r is not read by the grammar (ignored)' % k)
    kind = style.get('kind', 'window')
    if not isinstance(kind, str): r.err(where, 'style.kind must be a string')
    elif kind not in KINDS: r.warn(where, 'style.kind %r is not a grammar kind %s; it compiles as a plain window' % (kind, KINDS))
    for k in ['openingRatio', 'openingHeightRatio']:
        if k in style:
            v = style[k]
            if not is_num(v) or not (.05 <= v <= 1): r.err(where, '%s must be in [0.05,1], got %r' % (k, v))
    for k, (lo, hi) in METRIC_RANGES.items():
        if k in style:
            v = style[k]
            if not is_num(v) or not (lo <= v <= hi): r.err(where, '%s must be in [%g,%g] metres, got %r' % (k, lo, hi, v))
            if k in ['frameWidthM', 'canopyDepthM'] and is_num(v) and v <= 0: r.err(where, k + ' must be > 0')
    for k in ['mullionFractions', 'transomFractions']:
        if k in style: check_fraction_list(r, where, k, style[k])
    for k in ['frameMaterial', 'glassMaterial', 'material']:
        if k in style: check_material(r, where + '.' + k, style[k])
    for k in ['shade', 'canopy']:
        if k in style and not isinstance(style[k], bool): r.err(where, k + ' must be true/false')
    if kind == 'rollup' and any(k in style for k in ['frameMaterial', 'glassMaterial', 'mullionFractions', 'transomFractions', 'canopy', 'shade']):
        r.warn(where, 'rollup ignores frame/glass/mullion/transom/canopy/shade fields')


def check_group(r, where, g, fits):
    if not isinstance(g, dict): r.err(where, 'group must be an object'); return False
    for k in g:
        if k not in GROUP_KEYS: r.warn(where, 'group key %r is not read by the grammar (ignored)' % k)
    ok = True
    fit = None
    if 'fitId' in g:
        fit = next((f for f in fits if f['id'] == g['fitId']), None)
        if fit is None:
            if 'region' not in g:
                r.err(where, 'fitId %r not found in the supplied fits and no region fallback: openings() raises KeyError' % g['fitId']); return False
            r.warn(where, 'fitId %r not found in the supplied fits; the region/columns/rows fallback is used' % g['fitId'])
        else:
            r.warn(where, 'group uses numerical fit %r (facade-layouts.json); the vision step does not emit fits' % g['fitId'])
    if fit is None:
        if 'region' not in g: r.err(where, 'region is required'); return False
        ok = check_rect(r, where + '.region', g['region']) and ok
    for k in ['columns', 'rows']:
        v = g.get(k, 1)
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1): r.err(where, '%s must be an integer >= 1, got %r' % (k, v)); ok = False
    nx, ny = g.get('columns', 1), g.get('rows', 1)
    for k, n in [('xCenters', nx), ('yCenters', ny)]:
        if k in g:
            v = g[k]
            if not (isinstance(v, list) and all(is_num(c) and 0 <= c <= 1 for c in v)): r.err(where, k + ' must be a list of fractions in [0,1]'); ok = False
            elif isinstance(n, int) and len(v) != n: r.warn(where, '%s has %d entries but %s=%d; the centres win, %s only sets the default spacing' % (k, len(v), 'columns' if k == 'xCenters' else 'rows', n, 'columns' if k == 'xCenters' else 'rows'))
    for k in ['spacingX', 'spacingY']:
        if k in g:
            v = g[k]
            if not is_num(v) or not (0 < v <= 1): r.err(where, k + ' must be a fraction in (0,1]'); ok = False
    if 'omit' in g:
        om = g['omit']
        if not (isinstance(om, list) and all(isinstance(p, list) and len(p) == 2 and all(isinstance(i, int) for i in p) for p in om)):
            r.err(where, 'omit must be a list of [columnIndex,rowIndex] integer pairs'); ok = False
        else:
            cx = len(g['xCenters']) if isinstance(g.get('xCenters'), list) else nx
            cy = len(g['yCenters']) if isinstance(g.get('yCenters'), list) else ny
            for ix, iy in om:
                if not (0 <= ix < cx and 0 <= iy < cy): r.warn(where, 'omit [%d,%d] is outside the %dx%d grid (ignored)' % (ix, iy, cx, cy))
    if 'style' in g: check_style(r, where + '.style', g['style'])
    return ok


def validate_profile(p, fits, r):
    """Structural rules. Returns True when the group structure is sound enough to compute opening overlaps."""
    if not isinstance(p, dict): r.err('profile', 'must be an object'); return False
    for k in p:
        if k not in PROFILE_KEYS: r.warn('profile', 'key %r is not read by the pipeline (ignored)' % k)
    pid = p.get('id')
    if not (isinstance(pid, str) and pid.strip()): r.err('id', 'required non-empty string')
    m = p.get('match')
    if not isinstance(m, dict): r.err('match', 'required object {"osmWayId": <string|null>, "featureId": <string, optional>}')
    else:
        if 'osmWayId' not in m: r.err('match', 'the osmWayId key must be present (null allowed): generate_detailed_v14.py reads match["osmWayId"] directly')
        elif m['osmWayId'] is not None and not isinstance(m['osmWayId'], (str, int)): r.err('match.osmWayId', 'must be a string (or null)')
        elif isinstance(m['osmWayId'], int): r.warn('match.osmWayId', 'integer id; facades.py compares str() so it matches, but emit a string')
        if m.get('osmWayId') is None and 'featureId' not in m: r.err('match', 'needs osmWayId or featureId')
        for k in m:
            if k not in ['osmWayId', 'featureId']: r.warn('match', 'key %r is not a join key the pipeline reads' % k)
    n = p.get('normal')
    if not (isinstance(n, list) and len(n) == 2 and all(is_num(v) for v in n)): r.err('normal', 'required [nx,ny] with two finite numbers')
    else:
        L = math.hypot(*n)
        if L < 1e-6: r.err('normal', 'zero vector')
        elif abs(L - 1) > .01: r.err('normal', 'must be unit length (|n|=%.4f); facades.py tests dot(n, normal) > .95 on the raw value' % L)
    s = p.get('sourceIds')
    if s is None: r.err('sourceIds', 'required list of source ids; compile_wall reads profile["sourceIds"] without a default')
    elif not (isinstance(s, list) and all(isinstance(x, str) for x in s)): r.err('sourceIds', 'must be a list of strings')
    elif not s: r.warn('sourceIds', 'empty: openings are recorded as "inferred", not image-derived')
    check_material(r, 'material', p.get('material'))
    ev = p.get('materialEvidence')
    if ev is not None:
        if not isinstance(ev, dict): r.err('materialEvidence', 'must be an object {family, finish, sourceIds}')
        else:
            fam = ev.get('family')
            if fam not in FAMILIES: r.err('materialEvidence.family', '%r is not one of %s (resolve_material raises)' % (fam, sorted(FAMILIES)))
            elif ev.get('finish', 'default') not in FAMILIES[fam]: r.warn('materialEvidence.finish', '%r is not a finish of %r %s; the family default %r is used' % (ev.get('finish'), fam, sorted(FAMILIES[fam]), FAMILIES[fam]['default']))
            if not ev.get('sourceIds'): r.err('materialEvidence.sourceIds', 'required non-empty list (resolve_material raises without it)')
    if 'panels' in p:
        if not isinstance(p['panels'], list): r.err('panels', 'must be a list')
        else:
            for i, pn in enumerate(p['panels']):
                w = 'panels[%d]' % i
                if not isinstance(pn, dict): r.err(w, 'must be an object'); continue
                check_rect(r, w + '.rect', pn.get('rect'))
                check_material(r, w + '.material', pn.get('material'), allow_missing=False)
    groups_ok = True
    groups = p.get('groups', [])
    if not isinstance(groups, list): r.err('groups', 'must be a list'); groups_ok = False; groups = []
    for i, g in enumerate(groups):
        groups_ok = check_group(r, 'groups[%d]' % i, g, fits) and groups_ok
    if 'bands' in p:
        if not isinstance(p['bands'], list): r.err('bands', 'must be a list')
        else:
            for i, b in enumerate(p['bands']):
                w = 'bands[%d]' % i
                if not isinstance(b, dict): r.err(w, 'must be an object'); continue
                h = b.get('height')
                if not is_num(h) or not (0 <= h <= 1): r.err(w, 'height (fraction of H) is required in [0,1]')
                for k, lo, hi in [('thicknessM', 0, 2), ('depthM', 0, 3), ('projectionM', -1, 1)]:
                    if k in b and (not is_num(b[k]) or not (lo <= b[k] <= hi)): r.err(w, '%s must be in [%g,%g]' % (k, lo, hi))
                check_material(r, w + '.material', b.get('material'))
    if 'openings' in p:
        if not isinstance(p['openings'], list): r.err('openings', 'must be a list')
        else:
            for i, op in enumerate(p['openings']):
                w = 'openings[%d]' % i
                if not isinstance(op, dict): r.err(w, 'must be an object'); continue
                check_shape(r, w, op)
                if 'style' in op: check_style(r, w + '.style', op['style'], explicit=True)
                if not op.get('sourceIds'): r.warn(w, 'explicit opening without sourceIds; it is admitted anyway (explicit=True) but is untraceable')
                if 'evidence' in op and not isinstance(op['evidence'], str): r.err(w, 'evidence must be a string')
    if 'surfaceEvidence' in p:
        if not isinstance(p['surfaceEvidence'], list): r.err('surfaceEvidence', 'must be a list')
        else:
            for i, se in enumerate(p['surfaceEvidence']):
                w = 'surfaceEvidence[%d]' % i
                if not isinstance(se, dict): r.err(w, 'must be an object'); continue
                check_shape(r, w, se)
                lab = se.get('label')
                if lab not in LABELS: r.warn(w, 'label %r is not one of %s; only %s veto openings, everything else is ignored' % (lab, LABELS, NEGATIVE_LABELS))
                if not se.get('sourceIds'): r.warn(w, 'negative evidence without sourceIds')
    if 'openingPolicy' in p and p['openingPolicy'] not in POLICIES: r.err('openingPolicy', 'must be one of %s' % POLICIES)
    if 'planeToleranceM' in p and (not is_num(p['planeToleranceM']) or not (0 < p['planeToleranceM'] <= 10)): r.err('planeToleranceM', 'must be in (0,10] metres')
    if 'copingMaterial' in p: check_material(r, 'copingMaterial', p['copingMaterial'])
    u = p.get('unresolved')
    if u is None: r.warn('unresolved', 'missing; add the estimate caveats the record should carry (the pipeline writes [] otherwise)')
    elif not (isinstance(u, list) and all(isinstance(x, str) for x in u)): r.err('unresolved', 'must be a list of strings')
    if not groups and not p.get('openings'): r.warn('groups', 'no groups and no explicit openings: the wall compiles blank')
    return groups_ok


def overlap_checks(p, fits, r):
    """Region overlap is legal (a narrow stack may sit inside a wide bank) but overlapping OPENINGS interpenetrate."""
    regions = []
    for i, g in enumerate(p.get('groups', [])):
        reg = g.get('region')
        if reg is None:
            fit = next((f for f in fits if f['id'] == g.get('fitId')), None)
            reg = fit['region'] if fit else None
        if reg: regions.append((i, box(*reg)))
    for a in range(len(regions)):
        for b in range(a + 1, len(regions)):
            ia, ga = regions[a]; ib, gb = regions[b]
            inter = ga.intersection(gb).area
            if inter > 1e-9:
                r.warn('groups[%d]/groups[%d]' % (ia, ib), 'regions overlap (%.1f%% of the smaller); allowed, but check the opening overlap result below' % (100 * inter / min(ga.area, gb.area)))
    try:
        ops = facade_grammar.openings(p, fits)
    except Exception as e:
        r.err('openings()', 'grammar raised while enumerating openings: %r' % (e,)); return None
    # Attribute each admitted opening to the group that produced it (the grammar does not record this).
    origin = {}
    for gi, g in enumerate(p.get('groups', [])):
        sub = dict(p); sub['groups'] = [g]; sub['openings'] = []
        try:
            for op in facade_grammar.openings(sub, fits): origin.setdefault(tuple(round(v, 6) for v in op['rect']), 'groups[%d]' % gi)
        except Exception:
            pass
    def label(op):
        if 'rect' not in op: return 'explicit polygon'
        return origin.get(tuple(round(v, 6) for v in op['rect']), 'explicit')
    shapes = []
    for k, op in enumerate(ops):
        sh = surface_evidence.shape(op)
        shapes.append(sh)
        b = sh.bounds
        if b[0] < -1e-9 or b[1] < -1e-9 or b[2] > 1 + 1e-9 or b[3] > 1 + 1e-9:
            r.warn('opening#%d' % k, 'extends outside the frontage (bounds %s); compile_wall clips it to the wall' % [round(v, 3) for v in b])
    clashes = 0
    # Banks limited to disjoint wall-height ranges never appear on the same wall piece: their openings cannot clash.
    ranges = {'groups[%d]' % gi: (g.get('minHeightM', 0), g.get('maxHeightM', float('inf'))) for gi, g in enumerate(p.get('groups', []))}
    def disjoint(a, b):
        ra, rb = ranges.get(label(ops[a])), ranges.get(label(ops[b]))
        return bool(ra and rb and (ra[1] < rb[0] or rb[1] < ra[0]))
    for a in range(len(shapes)):
        for b in range(a + 1, len(shapes)):
            if disjoint(a, b): continue
            inter = shapes[a].intersection(shapes[b]).area
            if inter > 1e-9 and inter > .01 * min(shapes[a].area, shapes[b].area):
                clashes += 1
                if clashes <= 8:
                    ra = [round(v, 3) for v in shapes[a].bounds]; rb = [round(v, 3) for v in shapes[b].bounds]
                    r.err('opening#%d/opening#%d' % (a, b), 'openings overlap (%.0f%% of the smaller): %s %s vs %s %s; frames and glazing would interpenetrate' % (
                        100 * inter / min(shapes[a].area, shapes[b].area), label(ops[a]), ra, label(ops[b]), rb))
    if clashes > 8: r.err('openings', '%d further overlapping opening pairs not listed' % (clashes - 8))
    return ops


def synthetic_compile(p, fits, W=20., H=12., depth=10.):
    """Compile against one synthetic frontage of width W, height H, oriented by the profile's own normal."""
    n = np.asarray(p['normal'], float); n = n / np.linalg.norm(n); u = np.array([-n[1], n[0]])
    corners = [a * u + b * n for a, b in [(0, 0), (W, 0), (W, -depth), (0, -depth)]]
    g = Polygon([tuple(c) for c in corners])
    if not g.exterior.is_ccw: g = Polygon(list(g.exterior.coords)[::-1])
    c = {'id': 'synthetic', 'g': g, 'osm': None, 'full_area': g.area}
    cubes, cyls = [], []
    ctx = {'cube': lambda center, size, key, mat, angle=0: cubes.append((np.asarray(center, float), np.asarray(size, float), key, mat)),
           'put': lambda mesh, key, material, uv=None: _PUT.append((key, material, int(len(mesh.faces)))),
           'cylinder': lambda a, b, r_, key, mat, sections=8: cyls.append((a, b, r_, key, mat)),
           'polys': lambda geom: [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)}
    a = 0 * u; b = W * u
    fitted = facade_grammar.compile_wall(ctx, c, a, b, 0., H, p, fits)
    out = {'ok': fitted is not None, 'W': W, 'H': H}
    if fitted is None:
        out['exception'] = None; out['note'] = 'compile_wall returned None: no frontage edge parallel to the normal, W<3 m or wall run <0.3 m'; return out
    layer = 'buildings__synthetic__observed_' + p['id']; detail = 'facades__synthetic__observed_' + p['id']
    rec = fitted['record']
    mats = sorted(set([m for *_, m in cubes] + [m for *_, m in cyls]))
    out.update({'openings': fitted['components'], 'wallPieces': sum(k == layer for *_, k, _ in cubes), 'detailPieces': sum(k == detail for *_, k, _ in cubes),
                'cylinders': len(cyls), 'status': rec['status'], 'material': rec['material'], 'materialsUsed': mats,
                'unknownMaterialsUsed': [m for m in mats if m not in KNOWN and not ('#' in m and m.split('#')[0] in __import__('pipeline.material_recipes',fromlist=['RECIPES']).RECIPES and re.fullmatch(r'[0-9a-fA-F]{6}', m.split('#',1)[1]))], 'degenerateCubes': sum(float(np.min(s)) <= 0 for _, s, *_ in cubes),
                'explicitBoundaries': rec['openingEvidence']['explicitBoundaries'], 'negativeRegions': rec['openingEvidence']['negativeRegions'],
                'lengthM': rec['lengthM'], 'heightM': rec['heightM'], 'exception': None})
    return out


def load_profiles(path):
    data = json.load(open(path))
    if isinstance(data, dict) and 'profiles' in data: return data['profiles']
    if isinstance(data, list): return data
    return [data]


def load_fits(args):
    if args.no_fits: return [], []
    fits, bfits = [], []
    fp = pathlib.Path(args.fits) if args.fits else WORK / 'pipeline/facade-layouts.json'
    bp = pathlib.Path(args.boundary_fits) if args.boundary_fits else WORK / 'pipeline/boundary-layouts.json'
    if fp.exists(): fits = json.load(open(fp))['fits']
    if bp.exists(): bfits = json.load(open(bp))['fits']
    return fits, bfits


def run_file(path, fits, bfits, W, H):
    profiles = copy.deepcopy(load_profiles(path))
    for fit in bfits:  # same append the generator performs; on a copy
        target = next((p for p in profiles if isinstance(p, dict) and p.get('id') == fit['profileId']), None)
        if target is not None: target.setdefault('openings', []).extend(fit['openings'])
    ids = [p.get('id') for p in profiles if isinstance(p, dict)]
    results = []
    for idx, p in enumerate(profiles):
        r = Report()
        pid = p.get('id') if isinstance(p, dict) else None
        if pid is not None and ids.count(pid) > 1: r.err('id', 'duplicate id %r in this file' % pid)
        if isinstance(p, dict):
            for q in profiles[:idx]:
                if isinstance(q, dict) and isinstance(q.get('match'), dict) and isinstance(p.get('match'), dict) and q['match'].get('osmWayId') is not None \
                        and str(q['match'].get('osmWayId')) == str(p['match'].get('osmWayId')) and isinstance(q.get('normal'), list) and isinstance(p.get('normal'), list) \
                        and len(q['normal']) == 2 and len(p['normal']) == 2 and all(is_num(v) for v in q['normal'] + p['normal']) and float(np.dot(q['normal'], p['normal'])) > .95:
                    r.warn('match', 'same osmWayId and normal within 18 degrees of profile %r: facades.py takes the first in file order' % q.get('id'))
        structural_ok = validate_profile(p, fits, r)
        ops = overlap_checks(p, fits, r) if structural_ok and isinstance(p, dict) else None
        comp = {'ok': False, 'exception': 'not attempted: profile is not an object'} if not isinstance(p, dict) else None
        if comp is None:
            try:
                comp = synthetic_compile(p, fits, W, H)
            except Exception:
                comp = {'ok': False, 'exception': traceback.format_exc().strip().splitlines()[-1], 'traceback': traceback.format_exc()}
        if comp.get('unknownMaterialsUsed'):
            r.err('compile', 'emitted materials the bank cannot resolve: %s' % comp['unknownMaterialsUsed'])
        if ops is not None and comp.get('ok'):
            clipped = [surface_evidence.shape(o).intersection(box(0, 0, 1, 1)).area for o in ops]
            comp['openingAreaFraction'] = round(float(sum(clipped)), 4)
            if sum(clipped) > .9: r.warn('openings', 'openings cover %.0f%% of the frontage' % (100 * sum(clipped)))
        results.append({'index': idx, 'id': pid, 'errors': r.errors, 'warnings': r.warnings, 'compile': comp})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--fits'); ap.add_argument('--boundary-fits'); ap.add_argument('--no-fits', action='store_true')
    ap.add_argument('--wall', default='20x12', help='synthetic frontage WxH in metres')
    ap.add_argument('--json', help='write the machine-readable report here')
    ap.add_argument('--materials', action='store_true', help='print the accepted material names and exit')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    if args.materials:
        by = {}
        for k, v in KNOWN.items(): by.setdefault(v.split('->')[0], []).append(k)
        for cat in ['recipe', 'scan', 'color', 'palette', 'alias']: print('%-8s %2d  %s' % (cat, len(by.get(cat, [])), ' '.join(sorted(by.get(cat, [])))))
        print('families', json.dumps(FAMILIES)); print('kinds', KINDS); return 0
    W, H = [float(v) for v in args.wall.lower().split('x')]
    paths = args.paths or [str(WORK / 'pipeline/facade-profiles.json')]
    fits, bfits = load_fits(args)
    report = {'wall': [W, H], 'fits': len(fits), 'boundaryFits': len(bfits), 'files': []}
    failed = 0
    for path in paths:
        results = run_file(path, fits, bfits, W, H)
        report['files'].append({'path': str(path), 'profiles': results})
        if not args.quiet: print('== %s  (fits=%d, boundary fits=%d, synthetic wall %gx%g m)' % (path, len(fits), len(bfits), W, H))
        for res in results:
            c = res['compile']; bad = bool(res['errors']) or not c.get('ok')
            failed += bad
            if args.quiet: continue
            head = '%s %s' % ('FAIL' if bad else 'ok  ', res['id'])
            if c.get('ok'):
                head += '  openings=%d wallPieces=%d detailPieces=%d cylinders=%d status=%s material=%s openingArea=%.0f%%' % (
                    c['openings'], c['wallPieces'], c['detailPieces'], c['cylinders'], c['status'], c['material'], 100 * c.get('openingAreaFraction', 0))
            elif c.get('exception'): head += '  EXCEPTION: ' + c['exception']
            else: head += '  ' + c.get('note', 'compile returned None')
            print(head)
            for e in res['errors']: print('     error   ' + e)
            for w in res['warnings']: print('     warning ' + w)
    report['failed'] = failed
    if args.json: pathlib.Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    if not args.quiet: print('-- %d profile(s) failed' % failed)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
