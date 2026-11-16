"""Facade details a photograph shows that the base grammar did not draw: the sashes and panes of a window, the
surround, hood and keystone round it, a railing across a French window, a door with its transom and the steps up to
it, rusticated courses, quoins, fire escapes, balconies and relief panels. Everything is drawn in the wall frame of
facade_grammar.compile_wall (x along the wall from the frame's left edge, z up from the wall's foot, depth out from
the wall's face, metres) with the helpers it passes in `ctx`. Profiles carry these as data; no place facts here.

Profile keys (rectangles as fractions of the frame, like the grammar's openings):
  opening styles: sash ('double-hung' | 'casement' | 'fixed' | 'sliding'), lites [across, down] per sash,
    meetingRail (fraction of the opening's height), surround {widthM, projectionM, material}, hood {kind 'cornice' |
    'pediment' | 'segmental', heightM, projectionM, material}, keystone (bool), railing {heightM, material},
    door {kind 'panel' | 'glazed' | 'double', material}, transomM, steps {count, widthM, material}
  rustication [{rect, courseM, material}], quoins [{x, y, widthM, courseM, material}],
  fireEscapes [{x, levels, depthM, material}], balconies [{x, z, depthM, railing, railingHeightM, material}],
  hedges [{x, heightM, depthM, offsetM, material}], roofs [{x, z, kind, projectionM, riseM, overhangM, material}],
  planting [{x, kind, heightM, depthM, offsetM, planter, material}],
  reliefs [{rect, shape 'panel' | 'roundel' | 'cartouche' | 'garland', projectionM, material,
    repeatCount and ringCount for medallions, outerOutline ('circle' | 'oval' | 'pointed'), motif
    ('radial' | 'rosette' | 'unknown'), cardinalFlourishes, xRange and curveDepth for garlands}]"""
import math
import numpy as np
from shapely.geometry import Polygon, box, LineString, Point
from shapely.ops import unary_union
from .facade_bands import support_depth_at
from .door_leaves import emit_cells, emit_grille, emit_opaque_leaf

SASH_M, MUNTIN_M = .045, .022
_DRAWN = set()          # (detail key, list, index) of the items in front of a wall already drawn in this run


def porch_for_entry(porches, rect, width):
    """An entrance on a porch uses that porch's access, not a second stair flight.

    Identity is spatial: the doorway centre must be on the observed deck and
    its threshold at the deck elevation. Nearby upper/lower doors stay separate.
    """
    x0, z0, x1, _ = rect
    center = (x0 + x1) / 2
    candidates = [p for p in porches if p['x'][0] * width <= center <= p['x'][1] * width
                  and abs(float(p.get('floorM', 0.)) - z0) <= .35]
    return min(candidates, key=lambda p: abs(float(p.get('floorM', 0.)) - z0)) if candidates else None


def route_serves(routes, collection, source_index):
    """Whether a route explicitly owns this source-array porch or opening."""
    return source_index is not None and any(
        {'collection': collection, 'index': source_index} in route.get('serves', [])
        for route in routes or [])


def entrance_access_layout(style, rect, ground, porches=(), width=1.):
    """Compile one evidenced doorstep in the same terrain frame as porch access."""
    steps = style.get('steps') or {}
    count = steps.get('count')
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        return {'status': 'none', 'treads': [], 'rails': [], 'posts': []}
    if porch_for_entry(porches, rect, width) is not None:
        return {'status': 'owned-by-porch', 'treads': [], 'rails': [], 'posts': []}
    x0, floor, x1, _ = rect
    sw = float(steps.get('widthM', x1 - x0 + .2))
    # Old readings supply a count but no depth. Retain that explicitly inferred
    # tread pitch; never reinterpret zero/unknown count as a floor-height count.
    run = float(steps.get('runM', count * .28))
    pc = {'_x0': (x0+x1-sw)/2, '_x1': (x0+x1+sw)/2,
          'floorM': floor, 'depthM': 0.,
          'access': {'status': 'observed', 'steps': {
              'present': True, 'count': count, 'widthM': sw, 'runM': run,
              'centerOffsetM': 0., 'material': steps.get('material', 'stone')},
              'rails': steps.get('rails'), 'posts': []}}
    if 'bottomM' in steps: pc['access']['steps']['bottomM'] = steps['bottomM']
    return porch_access_layout(pc, ground)


def _relief_support_depth(ctx, x, z):
    """The outer face of the most projecting band under a relief point, or the wall face (zero)."""
    depth = None
    for support in ctx.get('reliefSupports', []):
        if support['rect'].covers(Point(float(x), float(z))):
            candidate = (support_depth_at(support['planFootprint'], x) if support.get('planFootprint') is not None
                         else float(support['frontDepth']))
            if candidate is None: continue
            depth = candidate if depth is None else max(depth, candidate)
    return 0. if depth is None else depth


def _relief_prism(ctx, poly, offset, thickness, material):
    """Mount a relief polygon on its backing surface, splitting it at band edges."""
    supports = sorted(ctx.get('reliefSupports', []), key=lambda s: float(s['frontDepth']), reverse=True)
    remaining = poly
    for support in supports:
        region = support['rect']
        piece = remaining.intersection(region)
        if not piece.is_empty and piece.area > 1e-6:
            footprint = support.get('planFootprint')
            if footprint is not None and ctx.get('slopedPrism'):
                # A bay-following band has a piecewise affine front in x/depth.
                # Split on those very vertices, not an unrelated facade grid.
                rings = [footprint.exterior] if footprint.geom_type == 'Polygon' else [p.exterior for p in footprint.geoms]
                cuts = sorted(set([piece.bounds[0], piece.bounds[2]] + [float(x) for ring in rings for x, _ in ring.coords
                                     if piece.bounds[0] < x < piece.bounds[2]]))
                for xa, xb in zip(cuts, cuts[1:]):
                    if xb - xa < 1e-8: continue
                    fragment = piece.intersection(box(xa, piece.bounds[1] - 1, xb, piece.bounds[3] + 1))
                    if fragment.is_empty: continue
                    a, b = xa + (xb-xa)*.25, xa + (xb-xa)*.75
                    da, db = support_depth_at(footprint, a), support_depth_at(footprint, b)
                    if da is None or db is None: continue
                    slope = (db-da)/(b-a)
                    ctx['slopedPrism'](fragment, da-slope*a+offset, slope, thickness, material)
            else:
                ctx['prism'](piece, float(support['frontDepth']) + offset, thickness, material)
            remaining = remaining.difference(region)
            if remaining.is_empty: break
    if not remaining.is_empty and remaining.area > 1e-6:
        ctx['prism'](remaining, offset, thickness, material)


def _relief_beam(ctx, a, b, width, material, offset):
    """Split a relief member at backing steps and support-profile vertices."""
    line = LineString([a[:2], b[:2]])
    if line.length <= 1e-9: return
    supports = ctx.get('reliefSupports', [])
    cuts = [0., 1.]
    delta = np.asarray(b[:2], float) - np.asarray(a[:2], float)
    norm2 = float(delta @ delta)
    for support in supports:
        crossing = line.intersection(support['rect'].boundary)
        geoms = [crossing] if crossing.geom_type == 'Point' else list(getattr(crossing, 'geoms', []))
        for geom in geoms:
            if geom.geom_type != 'Point': continue
            t = float((np.asarray(geom.coords[0]) - np.asarray(a[:2])) @ delta / norm2)
            if 1e-9 < t < 1. - 1e-9: cuts.append(t)

        footprint = support.get('planFootprint')
        if footprint is not None and abs(delta[0]) > 1e-10:
            polys = [footprint] if footprint.geom_type == 'Polygon' else list(getattr(footprint, 'geoms', []))
            for poly in polys:
                if poly.geom_type != 'Polygon': continue
                for ring in [poly.exterior, *poly.interiors]:
                    for x, _ in ring.coords:
                        t = (float(x) - float(a[0])) / delta[0]
                        if 1e-9 < t < 1. - 1e-9: cuts.append(float(t))

    # Within each rectangle/vertex interval, candidate support depths are
    # affine in x. Split where two exposed candidates exchange which one is
    # outermost, so the resulting beam never crosses between support faces.
    base_cuts = sorted(set(round(t, 12) for t in cuts))
    for t0, t1 in zip(base_cuts, base_cuts[1:]):
        if t1 - t0 <= 1e-9: continue
        tq0, tq1 = t0 + (t1 - t0) * .25, t0 + (t1 - t0) * .75
        p0 = np.asarray(a[:2], float) + delta * tq0
        p1 = np.asarray(a[:2], float) + delta * tq1
        active = [s for s in supports if s['rect'].covers(Point(*p0)) and s['rect'].covers(Point(*p1))]
        for i, first in enumerate(active):
            fp = first.get('planFootprint')
            d00 = support_depth_at(fp, p0[0]) if fp is not None else float(first['frontDepth'])
            d01 = support_depth_at(fp, p1[0]) if fp is not None else float(first['frontDepth'])
            if d00 is None or d01 is None: continue
            for second in active[i + 1:]:
                sp = second.get('planFootprint')
                d10 = support_depth_at(sp, p0[0]) if sp is not None else float(second['frontDepth'])
                d11 = support_depth_at(sp, p1[0]) if sp is not None else float(second['frontDepth'])
                if d10 is None or d11 is None: continue
                diff0, diff1 = d00 - d10, d01 - d11
                if diff0 * diff1 < -1e-12:
                    fraction = -diff0 / (diff1 - diff0)
                    cuts.append(tq0 + (tq1 - tq0) * fraction)
    cuts = sorted(set(round(t, 12) for t in cuts))
    for t0, t1 in zip(cuts, cuts[1:]):
        if t1 - t0 <= 1e-9: continue
        p0 = np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t0
        p1 = np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t1
        tm = (t0 + t1) / 2
        mid = np.asarray(a[:2], float) + delta * tm
        active = [s for s in supports if s['rect'].covers(Point(*mid))]
        if not active:
            da = db = 0.
        else:
            selected = max(active, key=lambda s: (support_depth_at(s['planFootprint'], mid[0])
                                                   if s.get('planFootprint') is not None
                                                   else float(s['frontDepth'])) or 0.)
            footprint = selected.get('planFootprint')
            if footprint is None:
                da = db = float(selected['frontDepth'])
            else:
                # Evaluate one-sided values inside this interval and extend
                # their affine segment to the endpoints. This preserves bay
                # slopes while keeping a step's two faces disconnected.
                eps = min((t1 - t0) * .01, 1e-6)
                ta, tb = t0 + eps, t1 - eps
                da0 = support_depth_at(footprint, float(a[0] + delta[0] * ta))
                db0 = support_depth_at(footprint, float(a[0] + delta[0] * tb))
                if da0 is None or db0 is None:
                    da = db = float(selected['frontDepth'])
                elif abs(tb - ta) <= 1e-12:
                    da = db = da0
                else:
                    slope_t = (db0 - da0) / (tb - ta)
                    da, db = da0 - slope_t * eps, db0 + slope_t * eps
        ctx['beam']((float(p0[0]), float(p0[1]), float(da + offset)),
                    (float(p1[0]), float(p1[1]), float(db + offset)), width, material)


def porch_access_layout(porch, ground):
    """Return contiguous porch stair treads and only explicitly observed side rails/posts."""
    access = porch.get('access')
    if access is not None and access.get('status') != 'observed':
        return {'status': 'unresolved', 'treads': [], 'rails': [], 'posts': []}
    steps = access.get('steps', {}) if access is not None else {}
    if access is None:
        if not porch.get('steps'):
            return {'status': 'none', 'treads': [], 'rails': [], 'posts': []}
        explicit = False
    else:
        if not steps.get('present'):
            return {'status': 'none', 'treads': [], 'rails': [], 'posts': []}
        explicit = True

    x0, x1 = porch['_x0'], porch['_x1']
    z0, depth = float(porch.get('floorM', 0.)), float(porch.get('depthM', 1.8))
    if explicit:
        width = float(steps['widthM']); center = (x0 + x1) / 2 + float(steps['centerOffsetM'])
        count = int(steps['count']); run = float(steps['runM']); material = steps.get('material', '#817d76')
    else:
        width = min(2.2, x1 - x0 - .4); center = (x0 + x1) / 2
        count = max(1, int(round((z0 - ground(center, depth + .5)) / .18)))
        run = count * .3; material = porch.get('material', 'stone')
    stair_x0, stair_x1 = center - width / 2, center + width / 2
    if stair_x0 < x0 - 1e-6 or stair_x1 > x1 + 1e-6:
        return {'status': 'unresolved', 'reason': 'stair width/offset falls outside the observed porch extent', 'treads': [], 'rails': [], 'posts': []}
    limit = porch.get('_accessLimitM')
    if limit is not None and depth + run > limit + .01:
        return {'status': 'unresolved', 'reason': 'stair flight crosses the carriageway; source flights and landings must be read',
                'proposedEndDepthM': depth + run, 'carriagewayDepthM': limit,
                'treads': [], 'rails': [], 'posts': []}
    ground_end = float(steps['bottomM']) if explicit and 'bottomM' in steps else float(ground(center, depth + run))
    total_rise = z0 - ground_end
    if total_rise <= .25:
        return {'status': 'unresolved', 'reason': 'porch floor does not stand above terrain at the observed stair endpoint', 'treads': [], 'rails': [], 'posts': []}

    treads = []
    for index in range(count):
        # Treat count as the number of tread courses: the last course reaches
        # the observed ground endpoint, while a single course bridges the full
        # rise from the porch.
        fraction = index / (count - 1) if count > 1 else 0.
        top = z0 - total_rise * fraction
        near = depth + run * index / count
        far = depth + run * (index + 1) / count
        next_fraction = (index + 1) / (count - 1) if count > 1 and index + 1 < count else 1.
        next_top = z0 - total_rise * next_fraction
        # Each tread is a solid wedge-like course represented by a box whose
        # bottom is buried below the lowest sampled ground point. Adjacent
        # courses overlap vertically at their shared depth boundary, so their
        # union is a continuous stair mass rather than thin edge-touching slabs.
        ground_samples = [float(ground(x, d)) for x in (stair_x0, center, stair_x1)
                          for d in (near, (near + far) / 2, far)]
        bottom = min(min(ground_samples) - .03, next_top - .03)
        treads.append({'x0': stair_x0, 'x1': stair_x1, 'topZ': top, 'bottomZ': bottom,
                       'depth0': near, 'depth1': far, 'nextTopZ': next_top,
                       'groundSamples': ground_samples, 'material': material})

    rails = []
    posts = []
    if explicit:
        rail_data = access.get('rails') or {}
        rail_height = rail_data.get('heightM')
        for side in rail_data.get('sides', []):
            x = stair_x0 - .06 if side == 'left' else stair_x1 + .06
            rails.append({'side': side, 'a': (x, z0 + rail_height, depth),
                          'b': (x, ground_end + rail_height, depth + run),
                          'material': rail_data.get('material', '#5b5a55')})
        for post in access.get('posts', []):
            position = float(post['position'])
            x = stair_x0 - .06 if post['side'] == 'left' else stair_x1 + .06
            at_depth = depth + run * position
            if position >= 1.:
                base = float(ground(x, at_depth))
            else:
                tread_index = min(count - 1, int(position * count))
                tread_fraction = tread_index / (count - 1) if count > 1 else 0.
                base = z0 - total_rise * tread_fraction
            posts.append({'x': x, 'depth': at_depth, 'baseZ': base,
                          'heightM': float(post['heightM']), 'widthM': float(post['widthM']),
                          'kind': post.get('kind', 'post'), 'finial': bool(post.get('finial')),
                          'material': post.get('material', '#e8e4d6')})
    return {'status': 'ok', 'treads': treads, 'rails': rails, 'posts': posts}


def opening(ctx, op):
    """The extras of one opening drawn by compile_wall: `op` has its rect in wall metres and its style. Drawn once, by
    the wall piece that holds the opening's centre."""
    s = op.get('style', {})
    if not any(k in s for k in ('sash', 'surround', 'hood', 'keystone', 'railing', 'door', 'steps')): return
    B, P, clip = ctx['block'], ctx['prism'], ctx['clip']
    outline = op.get('_originalOutline'); outline = outline if outline is not None else box(*op['rect'])
    x1, y1, x2, y2 = outline.bounds; w, h = x2 - x1, y2 - y1; cx = (x1 + x2) / 2
    if not ctx['low'] <= cx < ctx['high']: return
    recess = s.get('recessM', .2); frame = s.get('frameMaterial', 'charcoal'); fw = s.get('frameWidthM', .06)
    head = s.get('head', 'flat'); spring = y1 + h * (s.get('springRatio', .68) if head in ('arch', 'pointed') else 1 - s.get('riseRatio', .14) if head == 'segmental' else 1.)
    kind = s.get('kind', 'window')
    # the door: a leaf under the transom, set just in front of the glazing
    door = s.get('door')
    top = y2
    if kind == 'entry' and door and door.get('kind') != 'none':
        tm = float(s.get('transomM') or 0.)
        top = min(spring, y2 - tm) if tm > 0 else spring
        if tm > 0: B(cx, top + .03, w - 2 * fw, .06, .08, frame, -recess + .05)            # the transom bar
        dm = door.get('material', frame); dh = top - y1 - fw; dw = w - 2 * fw
        if dh > .5 and dw > .3:
            leaves = 2 if door.get('kind') == 'double' or dw > 1.4 else 1
            for k in range(leaves):
                lx0 = x1 + fw + k * dw / leaves; lw = dw / leaves
                leaf_bounds = (lx0, y1, lx0+lw, y1+dh)
                if door.get('kind') == 'grille':
                    emit_grille(B, leaf_bounds, depth=-recess+.045, material=dm)
                else:
                    cells = door.get('cells')
                    if door.get('kind') == 'glazed' and not cells:
                        cells = [{'rect': [.08, .16, .92, .9], 'kind': 'glass'}]
                    emit_opaque_leaf(B, leaf_bounds, dm, depth=-recess+.045, glass_cells=cells or ())
                    if cells:
                        emit_cells(B, leaf_bounds, cells, depth=-recess+.045, panel_material=dm)
                    if not door.get('cells') and door.get('kind') != 'glazed':
                        # An empty or absent panel list means the leaf is plain;
                        # do not add a stock two-panel pattern.
                        for px0, py0, px1, py1 in door.get('panels', []):
                            B(lx0 + lw * (px0 + px1) / 2, y1 + dh * (py0 + py1) / 2,
                              lw * (px1 - px0), dh * (py1 - py0), .025, dm, -recess + .085)
                    B(lx0 + lw * (.85 if k == 0 else .15), y1 + dh * .47, .03, .12, .05, 'silver', -recess + .1)
    # sashes and panes (below the springline of a curved or pointed head)
    sash = s.get('sash')
    if sash and kind in ('window', 'entry', 'shopfront') and not (kind == 'entry' and door):
        ix0, ix1, iy0, iy1 = x1 + fw, x2 - fw, y1 + fw, min(spring, y2 - fw)
        d0 = -recess + .03; lites = s.get('lites') or [1, 1]
        if sash == 'double-hung' and iy1 - iy0 > .6:
            mr = iy0 + (iy1 - iy0) * float(s.get('meetingRail', .5))
            parts = [(iy0, mr, d0), (mr, iy1, d0 + .04)]                   # the upper sash runs in front of the lower
            B(cx, mr, ix1 - ix0, .06, .1, frame, d0 + .03)
        elif sash == 'sliding' and ix1 - ix0 > .5:
            parts = [(iy0, iy1, d0)]; B(cx, (iy0 + iy1) / 2, .06, iy1 - iy0, .1, frame, d0 + .03)
        else: parts = [(iy0, iy1, d0)]
        for za, zb, dd in parts:
            # the sash's own frame, then its muntins
            for x in (ix0 + SASH_M / 2, ix1 - SASH_M / 2): B(x, (za + zb) / 2, SASH_M, zb - za, .05, frame, dd)
            for z in (za + SASH_M / 2, zb - SASH_M / 2): B(cx, z, ix1 - ix0, SASH_M, .05, frame, dd)
            na, nd = max(1, int(lites[0])), max(1, int(lites[1]))
            for k in range(1, na): B(ix0 + (ix1 - ix0) * k / na, (za + zb) / 2, MUNTIN_M, zb - za, .03, frame, dd)
            for k in range(1, nd): B(cx, za + (zb - za) * k / nd, ix1 - ix0, MUNTIN_M, .03, frame, dd)
    # the surround: a flat trim standing proud of the wall round the opening (following an arched head)
    sr = s.get('surround'); sw = 0.
    if sr:
        sw = float(sr.get('widthM', .14)); sp = float(sr.get('projectionM', .04)); sm = sr.get('material', frame)
        ring = outline.buffer(sw, join_style=2).difference(outline)
        if kind in ('entry', 'shopfront', 'garage', 'rollup'): ring = ring.difference(box(x1 - sw - .1, y1 - 1., x2 + sw + .1, y1))
        P(ring.intersection(clip.buffer(sw + .5)), sp, sp + .04, sm)
    hd = s.get('hood')
    yt = y2 + sw
    if hd:
        hh = float(hd.get('heightM', .3)); hp = float(hd.get('projectionM', .14)); hm = hd.get('material', (sr or {}).get('material', frame)); hk = hd.get('kind', 'cornice')
        span = w + 2 * sw + .16
        B(cx, yt + hh * .3, span - .08, hh * .6, hp * .6 + .02, hm, hp * .3)               # frieze under the cap
        B(cx, yt + hh * .8, span, hh * .4, hp + .02, hm, hp / 2)                            # the projecting cap
        if hk == 'pediment':
            peak = span * .22
            P(Polygon([(cx - span / 2, yt + hh), (cx + span / 2, yt + hh), (cx, yt + hh + peak)]), hp * .8, hp * .8 + .04, hm)
        elif hk == 'segmental':
            arc = [(cx + span / 2 * math.cos(t), yt + hh + span * .12 * math.sin(t)) for t in np.linspace(0, math.pi, 13)]
            P(Polygon(arc), hp * .8, hp * .8 + .04, hm)
    if s.get('keystone'):
        km = (sr or {}).get('material', hd.get('material') if hd else frame)
        kz = y2 - .04
        P(Polygon([(cx - .09, kz), (cx + .09, kz), (cx + .13, kz + .3 + sw), (cx - .13, kz + .3 + sw)]), .07, .11, km)
    # a railing across the lower part of a French window, just in front of the wall
    rl = s.get('railing')
    if rl:
        rh = float(rl.get('heightM', .9)); rm = rl.get('material', 'charcoal'); dep = .12
        for z in (y1 + .06, y1 + rh):
            B(cx, z, w + .12, .035, .035, rm, dep)
        for x in np.arange(x1 + .06, x2 - .03, .12): B(x, y1 + rh / 2, .016, rh, .016, rm, dep)
    # steps up to a door, from the wall's foot out across the forecourt
    if (kind == 'entry' and s.get('steps')
            and not route_serves(ctx.get('accessRoutes', []), 'openings', op.get('sourceIndex'))):
        access = entrance_access_layout(s, outline.bounds, ctx.get('ground', lambda x, d: 0.),
                                       ctx.get('porches', ()), ctx.get('W', 1.))
        for tread in access['treads']:
            B((tread['x0']+tread['x1'])/2, (tread['topZ']+tread['bottomZ'])/2,
              tread['x1']-tread['x0'], tread['topZ']-tread['bottomZ'],
              tread['depth1']-tread['depth0'], tread['material'],
              (tread['depth0']+tread['depth1'])/2)
        for rail in access['rails']: ctx['beam'](rail['a'], rail['b'], .035, rail['material'])


def wall(ctx, profile, ops):
    """The wall-wide details of a profile, clipped to the wall piece being drawn."""
    B, P, beam, clip = ctx['block'], ctx['prism'], ctx['beam'], ctx['clip']; W, Hs, H = ctx['W'], ctx['Hs'], ctx['H']
    low, high = ctx['low'], ctx['high']
    holes = unary_union([(o['_originalOutline'] if o.get('_originalOutline') is not None else box(*o['rect'])).buffer(float((o.get('style', {}).get('surround') or {}).get('widthM', 0.)) + .03, join_style=2)
                         for o in ops]) if ops else None
    for r in profile.get('rustication', []):
        # banded rustication: courses standing a little proud, a channel between them, kept off the openings
        x0, z0, x1, z1 = r['rect']; x0, x1 = max(low, x0 * W), min(high, x1 * W); z0, z1 = z0 * Hs, min(H, z1 * Hs)
        c = float(np.clip(r.get('courseM', .45), .2, 1.2)); g = min(.04, c * .1); m = r.get('material', ctx['wall_mat']((x0 + x1) / 2, (z0 + z1) / 2))
        if x1 - x0 < .1 or z1 - z0 < .1: continue
        for k in range(int(math.ceil((z1 - z0) / c))):
            band = box(x0, z0 + k * c + g / 2, x1, min(z1, z0 + (k + 1) * c - g / 2))
            if holes is not None: band = band.difference(holes)
            P(band, .03, .06, m)
    for q in profile.get('quoins', []):
        # quoins: alternately long and short blocks up a corner, a little proud of the wall
        x = q['x'] * W; z0, z1 = [v * Hs for v in q.get('y', [0, 1])]; z1 = min(z1, H); wq = float(q.get('widthM', .7)); c = float(q.get('courseM', .45)); m = q.get('material', 'stone')
        left = x < W / 2
        if not (low - wq <= x <= high + wq): continue
        for k in range(int((z1 - z0) / c)):
            ww = wq if k % 2 == 0 else wq * .62
            xa, xb = (x, x + ww) if left else (x - ww, x)
            xa, xb = max(xa, low), min(xb, high)
            if xb - xa > .05: B((xa + xb) / 2, z0 + (k + .5) * c, xb - xa, c - .03, .08, m, .02)
    for fe in profile.get('fireEscapes', []):
        x0, x1 = fe['x'][0] * W, fe['x'][1] * W
        if not low <= (x0 + x1) / 2 < high: continue
        d = float(np.clip(fe.get('depthM', .9), .5, 1.6)); m = fe.get('material', 'charcoal'); levels = sorted(z * Hs for z in fe['levels'])
        cx = (x0 + x1) / 2; L = x1 - x0
        for z in levels:
            B(cx, z, L, .05, d, m, d / 2)                                                       # the grating
            for zz in (z + .5, z + .98):
                beam((x0, zz, d), (x1, zz, d), .03, m); beam((x0, zz, .02), (x0, zz, d), .03, m); beam((x1, zz, .02), (x1, zz, d), .03, m)
            for x in np.arange(x0, x1 + .01, max(.25, L / max(1, round(L / .3)))): beam((x, z, d), (x, z + .98, d), .018, m)
            for x in (x0 + .25, x1 - .25): beam((x, z - .7, .02), (x, z - .03, d * .85), .04, m)   # brackets to the wall
        for k, (za, zb) in enumerate(zip(levels, levels[1:])):
            # a steep stair between the platforms, running along the wall, turning each level
            run = min(L * .7, (zb - za) * .62); xa = x0 + .25 if k % 2 == 0 else x1 - .25 - run; xb = xa + run
            up, down = ((xb, zb), (xa, za + .05)) if k % 2 == 0 else ((xa, zb), (xb, za + .05))
            for dd in (d * .5 - .28, d * .5 + .28): beam((down[0], down[1], dd), (up[0], up[1], dd), .04, m)
            for t in np.linspace(.08, .92, max(3, int((zb - za) / .22))):
                xx, zz = down[0] + (up[0] - down[0]) * t, down[1] + (up[1] - down[1]) * t
                B(xx, zz, .22, .025, .56, m, d * .5)
        if levels and levels[0] > 3.:
            # the drop ladder hanging from the lowest platform
            z = levels[0]; xa = x1 - .55
            for dd in (d * .5 - .2, d * .5 + .2): beam((xa, z - 2.4, dd), (xa, z, dd), .03, m)
            for zz in np.arange(z - 2.3, z, .3): B(xa, zz, .03, .02, .4, m, d * .5)
    for bc in profile.get('balconies', []):
        x0, x1 = bc['x'][0] * W, bc['x'][1] * W; z = bc['z'] * Hs
        if not low <= (x0 + x1) / 2 < high or z > H + .5: continue
        d = float(bc.get('depthM', .8)); m = bc.get('material', 'charcoal'); rh = float(bc.get('railingHeightM', 1.)); kind = bc.get('railing', 'metal')
        if not np.isfinite([d,rh]).all() or min(d,rh)<=0: continue
        has_deck=bc.get('hasDeck',d>=.3)
        cx = (x0 + x1) / 2; L = x1 - x0; slab = bc.get('slabMaterial', ctx['wall_mat'](cx, z))
        if has_deck: B(cx, z - .09, L, .18, d, slab, d / 2)
        if kind == 'solid':
            B(cx, z + rh / 2, L, rh, .1, slab, d - .05)
            for x in (x0 + .05, x1 - .05): B(x, z + rh / 2, .1, rh, d, slab, d / 2)
        else:
            member=min(.05 if kind == 'balustrade' else .035,d*.5,rh*.2)
            front=d-member/2
            beam((x0, z + rh, front), (x1, z + rh, front), member, m)
            if has_deck:
                beam((x0, z + rh, .02), (x0, z + rh, front), .035, m); beam((x1, z + rh, .02), (x1, z + rh, front), .035, m)
            else:
                # A shallow guard is a framed panel fixed to the wall, not a
                # miniature balcony with a slab and three enclosing railings.
                beam((x0, z, front),(x1,z,front),member,m)
                for x in (x0,x1): beam((x,z,front),(x,z+rh,front),member,m)
            pitch = .2 if kind == 'balustrade' else .11; size = .07 if kind == 'balustrade' else .016
            size=min(size,d*.5)
            for x in np.arange(x0 + .05, x1, pitch): B(x, z + rh / 2, size, rh, size, m, front)
            for dd in (np.arange(.1, d - .05, pitch) if has_deck else []):
                for x in (x0 + .03, x1 - .03): B(x, z + rh / 2, size, rh, size, m, dd)
    for rf in profile.get('roofs', []):
        # a roof seen from the street: a tiled eave or pent roof running along the wall's top (or over a bay or a door),
        # sloping down and out from the wall; barrel tiles as ridges down the slope, a fascia at its edge
        oh = min(.3, float(rf.get('overhangM', .2)))
        x0, x1 = max(low - oh, rf['x'][0] * W - oh), min(high + oh, rf['x'][1] * W + oh)
        if x1 - x0 < .3: continue
        z = rf['z'] * Hs; p = float(rf.get('projectionM', .6)); r = float(rf.get('riseM', p * .45)); m = rf.get('material', '#9a4a32')
        # a mansard or gable read on the wall's top stood in for a roof the massing now builds (a given or fitted pitched
        # roof): drawn twice it stands as a wall above the eaves and hides the real roof
        # (anywhere in the wall's upper part: a gable end wall rises to its ridge, so its read gable sits metres under H)
        if ctx.get('roofShell') and rf.get('kind') in ('mansard', 'gable', 'hip') and z >= .6 * H: continue
        if ctx.get('pitchedRoof') and rf.get('kind') in ('mansard', 'gable') and z >= .6 * H: continue
        # an eave read at or above the top of the wall as built sits on that top (the photograph's frame and the LiDAR wall
        # can disagree by a metre on a sloping street): it never floats over the wall it belongs to
        if z > H - .3: z = H - .02
        if rf.get('kind') == 'mansard':
            # a mansard: a steep roof face rising from the wall's top and leaning back over it, capped along its upper edge
            back = r / math.tan(math.radians(float(np.clip(rf.get('pitchDeg', 72.), 45., 85.))))
            quad(ctx, [(x0, z, .14), (x1, z, .14), (x1, z + r, -back), (x0, z + r, -back)], .14, m)
            B((x0 + x1) / 2, z + r + .04, x1 - x0, .08, .3, rf.get('fasciaMaterial', m), -back + .1)
            continue
        if rf.get('kind') == 'gable':
            # a gable end: the triangular face up to the ridge (unless the reading's parapet outline draws it), the roof's
            # overhang following both slopes out over it, bargeboards along its edges
            ax = float(rf.get('apexX', (rf['x'][0] + rf['x'][1]) / 2 * W)); apex = z + r; wm = ctx['wall_mat']((x0 + x1) / 2, z)
            if not profile.get('parapets'): P(Polygon([(x0 + oh, z), (ax, apex - .05), (x1 - oh, z)]), 0., .3, wm)
            for (xa, za, xb, zb) in ((x0, z - .1, ax, apex), (ax, apex, x1, z - .1)):
                quad(ctx, [(xa, za + .1, -.05), (xb, zb + .1, -.05), (xb, zb + .1, p), (xa, za + .1, p)], .1, m)
                beam((xa, za, p), (xb, zb, p), .16, rf.get('fasciaMaterial', ctx['wall_mat'](ax, z)))
            continue
        quad(ctx, [(x0, z + r, -.02), (x1, z + r, -.02), (x1, z, p), (x0, z, p)], .07, m)
        if rf.get('kind', 'tile-eave') in ('tile-eave', 'pent', 'tile-roof'):
            for xx in np.arange(x0 + .12, x1 - .05, .24): beam((xx, z + r + .045, -.02), (xx, z + .045, p), .085, m)
        B((x0 + x1) / 2, z - .02, x1 - x0, .12, .06, rf.get('fasciaMaterial', m), p - .03)
        if r > .25: B((x0 + x1) / 2, z + r - .05, x1 - x0, .1, .2, m, .08)                        # the ridge against the wall
    # Everything in front of the wall stands on the ground where it is (ctx['ground']: the terrain under a point of the
    # wall frame, in metres above the wall's foot), not on the wall's foot: on a hillside a garden wall at the sidewalk
    # stands metres below or above the house. Long pieces step with the slope.
    G = ctx.get('ground') or (lambda x, depth=0.: 0.)

    def mine(kind, i):
        """Each item is drawn once per building and reading, by the first of its wall pieces to be drawn: every piece
        shares the reading's frame, and an item may stand in front of no piece at all (a garden wall either side of a
        drive, beside a garage narrower than the lot)."""
        key = (ctx['detail'], kind, i)
        if key in _DRAWN: return False
        _DRAWN.add(key); return True

    def runs(x0, x1, step=1.5):
        k = max(1, int(math.ceil((x1 - x0) / step))); e = np.linspace(x0, x1, k + 1); return list(zip(e[:-1], e[1:]))

    def clipped(x0, x1, h, d, off, m, base=0.):
        """A clipped hedge: one crinkled mass following the ground (pipeline/facade_detail.hedge)."""
        hedge(ctx, x0, x1, off, off + d, h, m, base)

    # a vehicle door at the foot of the wall needs its driveway clear: garden walls, fences, hedges, shrubs and planters
    # read across it (a planter on the ledge above a garage, a garden wall read along the whole frontage) stop either side
    drives = [(o['rect'][0] - .15, o['rect'][2] + .15) for o in ops
              if o.get('style', {}).get('kind') in ('rollup', 'garage') and o['rect'][1] <= G((o['rect'][0] + o['rect'][2]) / 2, .3) + .6]

    def clear(x0, x1):
        """[x0, x1] less the driveways in front of the wall's vehicle doors: the pieces 0.3 m long or more."""
        pieces = [(x0, x1)]
        for a, b in drives:
            pieces = [q for xa, xb in pieces for q in ((xa, min(xb, a)), (max(xa, b), xb))]
        return [(xa, xb) for xa, xb in pieces if xb - xa >= .3]

    def edge(se, x0, x1, h, off, m, kind):
        """One street-edge item over [x0, x1]: a hedge, shrubs, a garden wall with its piers, a fence or a gate."""
        if kind == 'hedge': clipped(x0, x1, h, float(se.get('depthM', 1.)), off, m, float(se.get('baseM', 0.))); return   # baseM: on a wall's top
        if kind == 'shrub':
            d = float(se.get('depthM', 1.))
            shrubs(x0, x1, 0., h, d, off, m)
            return
        for xa, xb in runs(x0, x1):
            zg = G((xa + xb) / 2, off + .15)
            if kind == 'wall':
                B((xa + xb) / 2, zg + (h - .5) / 2, xb - xa + .02, h + .5, .3, m, off + .15); B((xa + xb) / 2, zg + h + .03, xb - xa + .06, .06, .36, m, off + .15)
            elif kind in ('fence', 'gate'):
                base = zg + float(se.get('baseM', 0.))
                for zz in (base + .08, base + h - .05): beam((xa, zz, off), (xb, zz, off), .035, m)
                for xx in np.arange(xa + .06, xb, .13): beam((xx, base - .1, off), (xx, base + h, off), .018, m)
        if kind == 'wall':
            # square piers standing a little proud of a garden wall, at its ends and at the pitch the photograph shows
            pw, ph = float(se.get('pierWidthM', 0.)), float(se.get('pierHeightM', h + .25))
            if pw > .1:
                xs = [f * W for f in se.get('piers') or []] or ([x0 + pw / 2, x1 - pw / 2] if not se.get('pierPitchM') else list(np.arange(x0 + pw / 2, x1 - pw / 2 + .01, float(se['pierPitchM']))))
                for xp in (q for q in xs if x0 - .01 <= q <= x1 + .01):   # a pier read where a driveway is cut stays out of it
                    zg = G(xp, off + .15); B(xp, zg + (ph - .5) / 2, pw, ph + .5, .42, m, off + .15); B(xp, zg + ph + .04, pw + .08, .08, .5, m, off + .15)

    def shrubs(x0, x1, base, h, d, off, m, sparse=False):
        """Foliage over [x0, x1] standing `h` over the ground in front of the wall, from `base` up (on a planter). Planting
        as tall as a small tree (2.5 m or more) is one: trunk, limbs and leaf cards as the LiDAR's trees are drawn
        (canopy.draw_tree, after the walls, the crown kept off this wall), a tree per run as wide as it is tall; a lumpy
        mass that tall read as a boulder. Lower planting is clumps of foliage."""
        tall = h >= 2.5 and not sparse and ctx.get('trees') is not None
        for xa, xb in runs(x0, x1, max(3., h) if tall else 3.):
            zg = G((xa + xb) / 2, off + d / 2)
            if not tall: blob(ctx, xa, xb, zg + base, zg + h, d, off, m, sparse=sparse); continue
            oc = off + d / 2
            ctx['trees'].append((*(float(v) for v in ctx['point']((xa + xb) / 2, zg + base, oc)), h - base, float(np.clip(min(max(xb - xa, d) / 2, oc - .3), .6, 4.))))
    for i, pl in enumerate(profile.get('planting', [])):
        # planting the photograph shows against the wall: shrubs and planters as clumps of foliage, a trellis as a lattice
        # with a climber on it, a hedge as a clipped mass; in front of the wall (offsetM)
        x0, x1 = pl['x'][0] * W, pl['x'][1] * W; h = float(pl['heightM']); d = float(pl.get('depthM', .8)); off = float(pl.get('offsetM', .3)); m = pl.get('material', '#46612f')
        kind = pl.get('kind', 'shrub')
        if x1 - x0 < .15 or not mine('planting', i): continue
        if kind in ('shrub', 'planter', 'hedge'):
            for xa, xb in clear(x0, x1):
                if kind == 'hedge': clipped(xa, xb, h, d, off, m); continue
                if pl.get('planter'):
                    for xc, xd in runs(xa, xb, 3.): B((xc + xd) / 2, G((xc + xd) / 2, off + d / 2) + .2, xd - xc, .8, d, pl.get('planterMaterial', '#9a9286'), off + d / 2)
                if not pl.get('bare'): shrubs(xa, xb, .6 if pl.get('planter') else 0., h, d, off, m)
            continue
        # a climber is one leafy mass growing up the wall from the ground (a thin crinkled hedge against it), never a few
        # leaf clumps hung in mid-air: scattered small blobs read as stray decals
        if kind == 'climber': clipped(x0, x1, h, float(np.clip(d, .15, .35)), float(np.clip(off, .02, .3)), m); continue
        if kind == 'trellis':
            zg = G((x0 + x1) / 2, off); tm = pl.get('frameMaterial', '#5b4a3a')
            for xx in np.arange(x0, x1 + .01, .3): beam((xx, zg, off), (xx, zg + h, off), .03, tm)
            for zz in np.arange(.3, h, .3): beam((x0, zg + zz, off), (x1, zg + zz, off), .025, tm)
            if not pl.get('bare'): clipped(x0, x1, h * .85, .2, off + .05, m)
            continue
        if pl.get('planter'):
            for xa, xb in runs(x0, x1, 3.): B((xa + xb) / 2, G((xa + xb) / 2, off + d / 2) + .2, xb - xa, .8, d, pl.get('planterMaterial', '#9a9286'), off + d / 2)
        if not pl.get('bare'): shrubs(x0, x1, .6 if pl.get('planter') else 0., h, d, off, m, sparse=kind in ('trellis', 'climber'))
    for i, se in enumerate(profile.get('streetEdge', [])):
        # what stands at the street line in front of a set-back building: garden walls, iron fences and gates, hedges and
        # shrubs, and the paving of a driveway or path running from the wall out to the street
        x0, x1 = se['x'][0] * W, se['x'][1] * W; h = float(se['heightM']); off = float(se.get('offsetM', 0.)); m = se.get('material', '#d8d2c4'); kind = se.get('kind', 'wall')
        if x1 - x0 < .15 or not mine('streetEdge', i): continue
        if kind not in ('driveway', 'path'):
            for xa, xb in clear(x0, x1): edge(se, xa, xb, h, off, m, kind)
            continue
        if kind in ('driveway', 'path'):
            ds = np.linspace(0., max(off, .5), max(2, int(math.ceil(max(off, .5))) + 1))
            for da, db in zip(ds[:-1], ds[1:]):
                quad(ctx, [(x0, G(x0, da) + .07, da), (x1, G(x1, da) + .07, da), (x1, G(x1, db) + .07, db), (x0, G(x0, db) + .07, db)], .14, m)
            continue
    for i, cn in enumerate(profile.get('canopies', [])):
        # a canopy over no opening's head (a porte-cochere, an awning over a lobby set back behind it): a slab with a fascia
        x0, x1 = cn['x'][0] * W, cn['x'][1] * W; z = cn['z'] * Hs; d = float(cn.get('depthM', 1.5)); m = cn.get('material', '#e6e3da')
        if x1 - x0 < .3 or not mine('canopies', i): continue
        B((x0 + x1) / 2, z, x1 - x0, .2, d, m, d / 2); B((x0 + x1) / 2, z - .14, x1 - x0, .12, .07, m, d - .035)
    for i, pc in enumerate(profile.get('porches', [])):
        # a porch or portico before the wall: columns with a base and a capital carrying a beam (entablature) and a roof,
        # its floor a deck at floorM with steps down to the ground in front of it; drawn from the measured extent alone
        full_x0, full_x1 = float(pc['x'][0] * W), float(pc['x'][1] * W)
        x0, x1 = max(low, full_x0), min(high, full_x1)
        if x1 - x0 < .8 or not mine('porches', i): continue
        # Keep access geometry in the measured porch frame when a wall tile
        # clips one end; the stair must not recenter on a partial fragment.
        pc = {**pc, '_x0': full_x0, '_x1': full_x1}
        z0 = float(pc.get('floorM', 0.)); zt = float(pc['topM']); d = float(pc['depthM']); cx = (x0 + x1) / 2
        m = pc.get('material', ctx['wall_mat'](cx, zt)); cm = pc.get('columnMaterial', m); zg = G(cx, d + .5)
        if z0 > .15: B(cx, (z0 + zg) / 2, x1 - x0, max(.2, z0 - zg), d, m, d / 2)          # the deck, on its foundation
        n = int(pc.get('columns', 2)); r = float(np.clip((x1 - x0) / (max(1, n) * 9.), .09, .2))
        for x in np.linspace(x0 + r + .08, x1 - r - .08, n):
            beam((x, z0 + .15, d - r - .08), (x, zt - .45, d - r - .08), 2 * r, cm)
            B(x, z0 + .08, 2.6 * r, .16, 2.6 * r, cm, d - r - .08); B(x, zt - .38, 2.4 * r, .16, 2.4 * r, cm, d - r - .08)   # base, capital
        B(cx, zt - .15, x1 - x0, .3, d, m, d / 2)                                             # the beam over the columns
        roof = pc.get('roof', 'flat')
        if roof == 'gable':
            peak = min(1.6, (x1 - x0) * .28)
            P(Polygon([(x0, zt), (cx, zt + peak), (x1, zt)]), d - .05, .1, m)                # the pediment's face
            for (xa, xb) in ((x0 - .1, cx), (cx, x1 + .1)):
                za, zb = (zt - .02, zt + peak) if xa < cx else (zt + peak, zt - .02)
                quad(ctx, [(xa, za + .1, 0.), (xb, zb + .1, 0.), (xb, zb + .1, d + .15), (xa, za + .1, d + .15)], .08, 'roof')
        elif roof == 'shed':
            quad(ctx, [(x0 - .1, zt + .45, 0.), (x1 + .1, zt + .45, 0.), (x1 + .1, zt + .05, d + .15), (x0 - .1, zt + .05, d + .15)], .08, 'roof')
        else:
            B(cx, zt + .05, x1 - x0 + .2, .1, d + .15, m, (d + .15) / 2)                    # a flat roof slab
            if roof == 'balcony':
                for x in np.arange(x0 + .1, x1, .14): beam((x, zt + .1, d), (x, zt + 1., d), .018, cm)
                B(cx, zt + 1.02, x1 - x0, .06, .12, cm, d)                                    # its handrail
        if ctx.get('accessLimit'):
            access_steps = (pc.get('access') or {}).get('steps') or {}
            sw = float(access_steps.get('widthM', min(2.2, full_x1-full_x0-.4)))
            sc = (full_x0+full_x1)/2 + float(access_steps.get('centerOffsetM', 0.))
            pc['_accessLimitM'] = ctx['accessLimit'](sc, sw, d)
        route_owns_steps = route_serves(ctx.get('accessRoutes', []), 'porches', pc.get('sourceIndex'))
        access = ({'status': 'none', 'treads': [], 'rails': [], 'posts': []}
                  if route_owns_steps else porch_access_layout(pc, G))
        if access['status'] == 'unresolved':
            profile.setdefault('unresolved', []).append({'kind': 'porch_access', 'index': i,
                                                          'reason': access.get('reason', 'access evidence unresolved')})
            continue
        for tread in access['treads']:
            B((tread['x0'] + tread['x1']) / 2, (tread['topZ'] + tread['bottomZ']) / 2,
              tread['x1'] - tread['x0'], tread['topZ'] - tread['bottomZ'],
              tread['depth1'] - tread['depth0'], tread['material'],
              (tread['depth0'] + tread['depth1']) / 2)
        for rail in access['rails']:
            beam(rail['a'], rail['b'], .035, rail['material'])
        for post in access['posts']:
            B(post['x'], post['baseZ'] + post['heightM'] / 2, post['widthM'], post['heightM'],
              post['widthM'], post['material'], post['depth'])
            if post['kind'] == 'newel':
                B(post['x'], post['baseZ'] + post['heightM'] - .05, post['widthM'] * 1.35, .1,
                  post['widthM'] * 1.35, post['material'], post['depth'])
                if post.get('finial'):
                    radius = post['widthM'] * .38; thickness = post['widthM'] * .6
                    P(Point(post['x'], post['baseZ'] + post['heightM'] + radius).buffer(radius, resolution=8),
                      post['depth'] - thickness / 2, thickness, post['material'])
    # A route is owned by its source facade, but may leave that wall's x-range
    # when a measured flight turns beside a terrace or retaining wall. Draw the
    # complete route once in the shared wall frame; do not clip it to this tile.
    for i, route in enumerate(profile.get('accessRoutes', [])):
        if not mine('accessRoutes', i): continue
        if route.get('status') != 'observed':
            reason = route.get('reason', 'route measurements unresolved')
            if not any(item.get('kind') == 'access_route' and item.get('index') == i
                       for item in profile.get('unresolved', [])):
                profile.setdefault('unresolved', []).append({'kind': 'access_route', 'index': i, 'reason': reason})
            continue
        from .access_routes import draw_route
        draw_route(route, ctx['point'], ctx['put'], detail)
    for hg in profile.get('hedges', []):
        # a clipped hedge standing in front of the wall, as the photograph shows it hiding the wall
        x0, x1 = max(low, hg['x'][0] * W), min(high, hg['x'][1] * W)
        if x1 - x0 < .3: continue
        clipped(x0, x1, hg['heightM'], hg.get('depthM', 1.2), hg.get('offsetM', .5), hg.get('material', '#3f5a2c'))
    for tf in profile.get('timbering', []):
        # a timber frame (half-timbering): its posts where they were measured and rails along their foot and head, all
        # on the wall (cut by its outline, a gable's rake included) and stopping at every opening
        x0, z0, x1, z1 = tf['rect']; m = tf.get('material') or ctx['mat']; pr = float(np.clip(tf.get('projectionM', .05), .02, .15))
        w = float(np.clip(tf.get('widthM', .18), .1, .35)); ZT = min(H, z1 * Hs)
        holes = unary_union([box(o['rect'][0] * W - .03, o['rect'][1] * Hs - .03, o['rect'][2] * W + .03, o['rect'][3] * Hs + .03) for o in profile.get('openings', []) if o.get('rect')] or [Polygon()])
        members = [box(a * W, z0 * Hs, e * W, ZT) for a, e in tf.get('posts', [])]
        members += [box(x0 * W, z0 * Hs, x1 * W, z0 * Hs + w), box(x0 * W, ZT - w, x1 * W, ZT)]
        members += [box(x0 * W, zr * Hs - w / 2, x1 * W, zr * Hs + w / 2) for zr in tf.get('rails', []) if z0 * Hs + w < zr * Hs < ZT - w]
        for mb in members:
            poly = mb.intersection(clip).difference(holes)
            if not poly.is_empty: P(poly, pr, pr + .04, m)
    for rf in profile.get('reliefs', []):
        x0, z0, x1, z1 = rf['rect']; x0, x1 = x0 * W, x1 * W; z0, z1 = z0 * Hs, min(H, z1 * Hs)
        if x1 <= x0 or z1 <= z0: continue
        pr = float(np.clip(rf.get('projectionM', .04), .01, .3))
        m = rf.get('material', ctx['mat']); shape = rf.get('shape', 'panel')
        if shape == 'garland':
            xr = rf.get('xRange', [rf['rect'][0], rf['rect'][2]])
            xa, xb = max(x0, float(xr[0]) * W), min(x1, float(xr[1]) * W)
            count = int(np.clip(rf.get('repeatCount', 1), 1, 64))
            if xb - xa < .08: continue
            # Compute all arches from the full authored span before clipping, so neighboring wall pieces preserve phase.
            span = (xb - xa) / count; top = z1
            curve_depth = rf.get('curveDepth')
            drop = (z1 - z0) * float(np.clip(curve_depth, 0., 1.)) if curve_depth is not None else min(z1 - z0, span * .22)
            for j in range(count):
                a, b = xa + j * span, xa + (j + 1) * span
                points = [(a + span * t, top - drop * 4 * t * (1 - t), pr + .02) for t in np.linspace(0., 1., 9)]
                for pa, pb in zip(points, points[1:]):
                    visible = LineString([pa[:2], pb[:2]]).intersection(clip)
                    lines = [visible] if visible.geom_type == 'LineString' else list(getattr(visible, 'geoms', []))
                    for line in lines:
                        coords = list(line.coords)
                        for va, vb in zip(coords, coords[1:]):
                            _relief_beam(ctx, (*va, 0.), (*vb, 0.), min(.045, span / 12), m, pr + .02)
            continue

        bounds = box(x0, z0, x1, z1)
        if shape in ('roundel', 'cartouche'):
            angles = np.linspace(0., 2 * math.pi, 65)
            count = int(np.clip(rf.get('repeatCount', 1), 1, 64))
            cell_w = (x1 - x0) / count; rz = (z1 - z0) * .46
            ring_count = int(np.clip(rf.get('ringCount', 1), 1, 4))
            outline_kind = rf.get('outerOutline')
            motif = rf.get('motif')
            # Keep older observations' cartouche/roundel interpretation unless
            # the newer typed fields explicitly separate frame from ornament.
            if outline_kind is None: outline_kind = 'pointed' if shape == 'cartouche' else 'circle'
            if motif is None: motif = 'rosette'
            cardinal = bool(rf.get('cardinalFlourishes', shape == 'cartouche'))
            for j in range(count):
                cx, cz = x0 + (j + .5) * cell_w, (z0 + z1) / 2
                # Medallions stay within their repeated cell and cannot stretch wider than their height.
                rx = min(cell_w * .46, rz)
                if rx <= .01 or rz <= .01: continue
                if outline_kind == 'circle':
                    # A circle has one metric radius even when its authored box
                    # is wider/taller; it must not inherit cartouche flourishes.
                    radius = min(rx, rz)
                    ox = oz = radius
                    modulation = 0.
                else:
                    ox, oz = rx, rz
                    modulation = .10 if outline_kind == 'pointed' else 0.
                outer = Polygon([(cx + ox * (1 + modulation * math.cos(4 * t)) * math.cos(t),
                                  cz + oz * (1 + modulation * math.cos(4 * t)) * math.sin(t)) for t in angles])
                ring_w = min(.09, rx * .18, rz * .18)
                for ring in range(ring_count):
                    outside = outer.buffer(-ring * ring_w, join_style=1)
                    inside = outer.buffer(-(ring + 1) * ring_w, join_style=1)
                    poly = outside.difference(inside).intersection(bounds).intersection(clip)
                    if not poly.is_empty: _relief_prism(ctx, poly, pr + ring * .006, pr + .04 + ring * .006, m)
                # The center motif is typed independently from the frame. Old
                # records default to their former rosette; unknown means only
                # the observed outline/rings are represented.
                inner_rx = max(.01, ox - ring_w * ring_count - ring_w * .35)
                inner_rz = max(.01, oz - ring_w * ring_count - ring_w * .35)
                if motif == 'rosette':
                    center_shape = Polygon([(cx + inner_rx * (.73 + .27 * math.cos(8 * t)) * math.cos(t),
                                             cz + inner_rz * (.73 + .27 * math.cos(8 * t)) * math.sin(t)) for t in angles])
                    center = center_shape.intersection(bounds).intersection(clip)
                    if not center.is_empty: _relief_prism(ctx, center, pr + .025, .05, m)
                elif motif == 'radial':
                    disk_radius = min(inner_rx, inner_rz) * .18
                    disk = Point(cx, cz).buffer(disk_radius, quad_segs=16).intersection(bounds).intersection(clip)
                    if not disk.is_empty: _relief_prism(ctx, disk, pr + .025, .05, m)
                    radial_edge = outer.buffer(-ring_w * ring_count, join_style=1)
                    ridge_width = max(.012, min(inner_rx, inner_rz) * .035)
                    for k in range(8):
                        theta = 2 * math.pi * k / 8
                        start = (cx + disk_radius * 1.2 * math.cos(theta), cz + disk_radius * 1.2 * math.sin(theta))
                        end = (cx + inner_rx * .88 * math.cos(theta), cz + inner_rz * .88 * math.sin(theta))
                        ridge = LineString([start, end]).buffer(ridge_width, cap_style=1).intersection(radial_edge)
                        ridge = ridge.intersection(bounds).intersection(clip)
                        if not ridge.is_empty: _relief_prism(ctx, ridge, pr + .022, .035, m)
                if cardinal:
                    # Small cardinal leaves decorate the frame without crossing its fitted outline.
                    leaf_rx, leaf_rz = min(.07, ox * .12), min(.12, oz * .12)
                    for dx, dz in ((0., oz * .72), (0., -oz * .72), (ox * .72, 0.), (-ox * .72, 0.)):
                        leaf = Polygon([(cx + dx + leaf_rx * math.cos(t), cz + dz + leaf_rz * math.sin(t)) for t in angles])
                        leaf = leaf.intersection(outer).intersection(clip)
                        if not leaf.is_empty: _relief_prism(ctx, leaf, pr + .025, .05, m)
        else:
            poly = bounds.intersection(clip)
            if not poly.is_empty: _relief_prism(ctx, poly, pr, pr + .04, m)
            inset = poly.buffer(-.06, join_style=2) if not poly.is_empty else poly
            if not inset.is_empty: _relief_prism(ctx, inset, pr + .025, .05, m)


def quad(ctx, corners, t, material):
    """A slab of thickness t under four (x, z, depth) corners in the wall frame (a sloped roof plane)."""
    import trimesh
    top = np.array([ctx['point'](*c) for c in corners]); bot = top - np.array([0, 0, t])
    v = np.vstack([top, bot]); f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    mesh = trimesh.Trimesh(v, f, process=False); mesh.fix_normals(); ctx['put'](mesh, ctx['detail'], material)


def shade(m, k):
    """A hex colour lighter or darker by factor k (other materials as they are)."""
    if not (isinstance(m, str) and len(m) == 7 and m[0] == '#'): return m
    c = [min(255, max(0, round(int(m[i:i + 2], 16) * k))) for i in (1, 3, 5)]; return '#%02x%02x%02x' % tuple(c)


def _noise(v, seed=0.):
    """A smooth-ish deterministic noise in [-1, 1] for points v (n, 3): sums of sines at leaf-cluster scale."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return (np.sin(x * 7.3 + y * 3.1 + z * 5.7 + seed) * .45 + np.sin(x * 13.7 - y * 11.3 + z * 9.1 + 1.7 + seed) * .35
            + np.sin(-x * 23.1 + y * 19.7 - z * 17.3 + 4.1 + seed) * .2)


def _grid_box(L, D, H, step):
    """A closed box L x D x H about the origin whose faces are an even grid of about `step`: subdividing a box's
    triangles until every edge is that short quartered all of them each round, 40 times the faces on a long hedge."""
    import trimesh
    xs, ys, zs = (np.linspace(-s / 2, s / 2, max(1, int(math.ceil(s / step))) + 1) for s in (L, D, H))
    V, F = [], []
    def face(P):
        n = sum(len(v) for v in V); idx = np.arange(P.shape[0] * P.shape[1]).reshape(P.shape[:2]) + n; V.append(P.reshape(-1, 3))
        a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        F.extend([np.c_[a, b, c], np.c_[a, c, d]])
    for z in (zs[0], zs[-1]): X, Y = np.meshgrid(xs, ys, indexing='ij'); face(np.dstack([X, Y, np.full_like(X, z)]))
    for y in (ys[0], ys[-1]): X, Z = np.meshgrid(xs, zs, indexing='ij'); face(np.dstack([X, np.full_like(X, y), Z]))
    for x in (xs[0], xs[-1]): Y, Z = np.meshgrid(ys, zs, indexing='ij'); face(np.dstack([np.full_like(Y, x), Y, Z]))
    mesh = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False); mesh.merge_vertices(); mesh.fix_normals()
    return mesh


def hedge(ctx, x0, x1, d0, d1, h, m, base=0.):
    """A clipped hedge spanning [x0, x1] along the wall and [d0, d1] out from it, h tall: one box subdivided to about
    0.3 m and crinkled along its normals (leaf clusters, not a green wall), its top edges rounded in, every vertex
    standing on the ground under it (plus `base`, a wall it grows on) so the hedge follows a slope, its foot sunk 0.25 m;
    mottled in two shades."""
    import trimesh
    L, D = max(.2, x1 - x0), max(.2, d1 - d0)
    mesh = _grid_box(L, D, h, .3)
    v = mesh.vertices.copy(); nrm = mesh.vertex_normals.copy()
    up = (v[:, 2] + h / 2) / h                                              # 0 at the foot, 1 at the top
    v += nrm * (.055 * _noise(v + [x0, d0, 0.]) * (up > .03))[:, None]
    r = np.clip((up - .82) / .18, 0, 1) * .16                               # the top edges rounded in
    for k, half in ((0, L / 2), (1, D / 2)):
        near = np.abs(v[:, k]) > half - .25; v[near, k] -= np.sign(v[near, k]) * r[near]
    xf, df = x0 + L / 2 + v[:, 0], d0 + D / 2 + v[:, 1]
    gv = ctx['ground_v'](xf, df) if ctx.get('ground_v') else np.array([ctx.get('ground', lambda a, b: 0.)(a, b) for a, b in zip(xf, df)])
    zf = gv + base + (v[:, 2] + h / 2) - .25 * (up < .01)
    mesh.vertices = ctx['points'](xf, zf, df) if ctx.get('points') else np.array([ctx['point'](a, c, b) for a, b, c in zip(xf, df, zf)])
    mesh.fix_normals()
    from .groundcover import emit
    emit(ctx,mesh,m)


def blob(ctx, x0, x1, z0, h, d, off, material, sparse=False):
    """Foliage along [x0, x1]: overlapping rounded clumps, sized and placed by a hash of their position (no randomness:
    the same reading builds the same planting)."""
    import trimesh
    n = max(1, int(round((x1 - x0) / (.9 if not sparse else 1.3))))
    for i in range(n):
        u = (i + .5) / n; seed = (np.sin((x0 + u * (x1 - x0)) * 12.9898 + h * 78.233) * 43758.5453) % 1.
        w = (x1 - x0) / n * (1.15 + .3 * seed); hh = (h - z0) * (.75 + .25 * seed) * (.6 if sparse else 1.)
        c = (x0 + u * (x1 - x0), z0 + hh / 2 + (h - z0 - hh) * (.5 if sparse else 0.), off + d / 2)
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.)
        mesh.vertices *= (1 + .16 * _noise(mesh.vertices * 2.2, seed * 9.))[:, None]   # lumpy, like a shrub's leaf clusters
        mesh.vertices *= [w / 2, hh / 2, d / 2 * (.9 + .2 * seed)]          # (along, up, out) before mapping
        mesh.vertices = np.array([ctx['point'](c[0] + a, c[1] + b, c[2] + e) for a, b, e in mesh.vertices]); mesh.fix_normals()
        from .groundcover import emit
        emit(ctx,mesh,material)
