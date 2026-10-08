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
  hedges [{x, heightM, depthM, offsetM, material}],
  reliefs [{rect, projectionM, material}]"""
import math
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

SASH_M, MUNTIN_M = .045, .022


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
            for x in np.arange(x0, x1 + .01, max(.25, L / round(L / .3))): beam((x, z, d), (x, z + .98, d), .018, m)
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
    for hg in profile.get('hedges', []):
        # a clipped hedge standing in front of the wall (on its ground line), as the photograph shows it hiding the wall
        x0, x1 = max(low, hg['x'][0] * W), min(high, hg['x'][1] * W)
        if x1 - x0 < .3: continue
        h, d, off, m = hg['heightM'], hg.get('depthM', 1.2), hg.get('offsetM', .5), hg.get('material', '#3f5a2c')
        B((x0 + x1) / 2, h * .46, x1 - x0, h * .92, d, m, off + d / 2)
        B((x0 + x1) / 2, h * .96, x1 - x0 - .2, h * .08, d - .25, m, off + d / 2)          # a softened top
    for rf in profile.get('reliefs', []):
        x0, z0, x1, z1 = rf['rect']; poly = box(x0 * W, z0 * Hs, x1 * W, min(H, z1 * Hs)).intersection(clip)
        if poly.is_empty: continue
        pr = float(np.clip(rf.get('projectionM', .04), .01, .3))
        P(poly, pr, pr + .04, rf.get('material', ctx['mat']))
        P(poly.buffer(-.06, join_style=2), pr + .025, .05, rf.get('material', ctx['mat']))
