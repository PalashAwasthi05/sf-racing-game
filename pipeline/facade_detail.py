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
  reliefs [{rect, projectionM, material}]"""
import math
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

SASH_M, MUNTIN_M = .045, .022
_DRAWN = set()          # (detail key, list, index) of the items in front of a wall already drawn in this run


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
    head = s.get('head', 'flat'); spring = y1 + h * (s.get('springRatio', .68) if head == 'arch' else 1 - s.get('riseRatio', .14) if head == 'segmental' else 1.)
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
            B(cx, y1 + dh / 2, dw, dh, .05, dm, -recess + .045)
            leaves = 2 if door.get('kind') == 'double' or dw > 1.4 else 1
            for k in range(leaves):
                lx0 = x1 + fw + k * dw / leaves; lw = dw / leaves
                if leaves == 2: B(lx0 + (lw if k == 0 else 0), y1 + dh / 2, .02, dh, .07, frame, -recess + .06)
                if door.get('kind') == 'glazed':
                    B(lx0 + lw / 2, y1 + dh * .62, lw * .62, dh * .5, .02, 'glass', -recess + .08)
                else:
                    for py, ph in ((.08, .38), (.52, .38)):
                        B(lx0 + lw / 2, y1 + dh * (py + ph / 2), lw * .6, dh * ph, .025, dm, -recess + .085)
                B(lx0 + lw * (.85 if k == 0 else .15), y1 + dh * .47, .03, .12, .05, 'silver', -recess + .1)
    # sashes and panes (below the springline of an arched head)
    sash = s.get('sash')
    if sash and kind in ('window', 'entry', 'shopfront') and not (kind == 'entry' and door):
        ix0, ix1, iy0, iy1 = x1 + fw, x2 - fw, y1 + fw, min(spring, y2 - fw)
        d0 = -recess + .03; lites = s.get('lites') or [1, 1]
        if sash == 'double-hung' and iy1 - iy0 > .6:
            mr = iy0 + (iy1 - iy0) * float(s.get('meetingRail', .5))
            parts = [(iy0, mr, d0), (mr, iy1, d0 + .04)]                   # the upper sash runs in front of the lower
            B(cx, mr, ix1 - ix0, .06, .1, frame, d0 + .03)
        elif sash == 'casement' and ix1 - ix0 > .5:
            parts = [(iy0, iy1, d0)]; B(cx, (iy0 + iy1) / 2, .07, iy1 - iy0, .09, frame, d0 + .02)
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
    st = s.get('steps')
    if st and kind == 'entry' and y1 > .1:
        n = max(1, int(st.get('count') or round(y1 / .17))); rise = y1 / n; tread = .28; sw_ = float(st.get('widthM', w + .2)); sm = st.get('material', 'stone')
        for k in range(n):
            depth = tread * (n - k)
            B(cx, rise * (k + .5), sw_, rise, depth, sm, depth / 2)
        if n >= 3:
            for x in (cx - sw_ / 2 + .04, cx + sw_ / 2 - .04):
                ctx['beam']((x, rise + .9, tread * n - .1), (x, y1 + .9, .05), .04, st.get('railMaterial', 'charcoal'))


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
        d = float(np.clip(bc.get('depthM', .8), .3, 2.5)); m = bc.get('material', 'charcoal'); rh = float(bc.get('railingHeightM', 1.)); kind = bc.get('railing', 'metal')
        cx = (x0 + x1) / 2; L = x1 - x0; slab = bc.get('slabMaterial', ctx['wall_mat'](cx, z))
        B(cx, z - .09, L, .18, d, slab, d / 2)
        if kind == 'solid':
            B(cx, z + rh / 2, L, rh, .1, slab, d - .05)
            for x in (x0 + .05, x1 - .05): B(x, z + rh / 2, .1, rh, d, slab, d / 2)
        else:
            beam((x0, z + rh, d - .03), (x1, z + rh, d - .03), .05 if kind == 'balustrade' else .035, m)
            beam((x0, z + rh, .02), (x0, z + rh, d - .03), .035, m); beam((x1, z + rh, .02), (x1, z + rh, d - .03), .035, m)
            pitch = .2 if kind == 'balustrade' else .11; size = .07 if kind == 'balustrade' else .016
            for x in np.arange(x0 + .05, x1, pitch): B(x, z + rh / 2, size, rh, size, m, d - .03)
            for dd in np.arange(.1, d - .05, pitch):
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
        if kind == 'hedge': clipped(x0, x1, h, d, off, m); continue
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
        if kind in ('driveway', 'path'):
            ds = np.linspace(0., max(off, .5), max(2, int(math.ceil(max(off, .5))) + 1))
            for da, db in zip(ds[:-1], ds[1:]):
                quad(ctx, [(x0, G(x0, da) + .07, da), (x1, G(x1, da) + .07, da), (x1, G(x1, db) + .07, db), (x0, G(x0, db) + .07, db)], .14, m)
            continue
        if kind == 'hedge': clipped(x0, x1, h, float(se.get('depthM', 1.)), off, m, float(se.get('baseM', 0.))); continue   # baseM: on a wall's top
        if kind == 'shrub':
            d = float(se.get('depthM', 1.))
            shrubs(x0, x1, 0., h, d, off, m)
            continue
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
                for xp in xs:
                    zg = G(xp, off + .15); B(xp, zg + (ph - .5) / 2, pw, ph + .5, .42, m, off + .15); B(xp, zg + ph + .04, pw + .08, .08, .5, m, off + .15)
    for i, cn in enumerate(profile.get('canopies', [])):
        # a canopy over no opening's head (a porte-cochere, an awning over a lobby set back behind it): a slab with a fascia
        x0, x1 = cn['x'][0] * W, cn['x'][1] * W; z = cn['z'] * Hs; d = float(cn.get('depthM', 1.5)); m = cn.get('material', '#e6e3da')
        if x1 - x0 < .3 or not mine('canopies', i): continue
        B((x0 + x1) / 2, z, x1 - x0, .2, d, m, d / 2); B((x0 + x1) / 2, z - .14, x1 - x0, .12, .07, m, d - .035)
    for i, pc in enumerate(profile.get('porches', [])):
        # a porch or portico before the wall: columns with a base and a capital carrying a beam (entablature) and a roof,
        # its floor a deck at floorM with steps down to the ground in front of it; drawn from the measured extent alone
        x0, x1 = max(low, pc['x'][0] * W), min(high, pc['x'][1] * W)
        if x1 - x0 < .8 or not mine('porches', i): continue
        z0 = float(pc.get('floorM', 0.)); zt = float(pc['topM']); d = float(pc['depthM']); cx = (x0 + x1) / 2
        m = pc.get('material', ctx['wall_mat'](cx, zt)); cm = pc.get('columnMaterial', m); zg = G(cx, d + .5)
        if z0 > .15: B(cx, (z0 + zg) / 2, x1 - x0, max(.2, z0 - zg), d, m, d / 2)          # the deck, on its foundation
        n = int(pc.get('columns', 2)); r = float(np.clip((x1 - x0) / (n * 9.), .09, .2))
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
        if pc.get('steps') and z0 - zg > .25:
            k = int(round((z0 - zg) / .18)); rise = (z0 - zg) / k; ws = min(2.2, x1 - x0 - .4)
            for j in range(1, k + 1):                                                         # each tread a step lower, a step further out
                top = z0 - (j - 1) * rise
                B(cx, (top + zg) / 2, ws, max(.05, top - zg), .3, m, d + (j - .5) * .3)
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
        x0, z0, x1, z1 = rf['rect']; poly = box(x0 * W, z0 * Hs, x1 * W, min(H, z1 * Hs)).intersection(clip)
        if poly.is_empty: continue
        pr = float(np.clip(rf.get('projectionM', .04), .01, .3))
        P(poly, pr, pr + .04, rf.get('material', ctx['mat']))
        P(poly.buffer(-.06, join_style=2), pr + .025, .05, rf.get('material', ctx['mat']))


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
    pick = _noise(mesh.triangles_center, 2.3) > .15
    for sel, mat in ((pick, shade(m, 1.07)), (~pick, m)):
        if sel.any(): ctx['put'](mesh.submesh([np.nonzero(sel)[0]], append=True, repair=False), ctx['detail'], mat)


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
        ctx['put'](mesh, ctx['detail'], material)
