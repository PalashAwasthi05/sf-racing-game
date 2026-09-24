"""A regulation baseball infield placed by measurement.

The layout is the rulebook's (bases 90 ft apart, rubber 60 ft 6 in from the plate, the grass line an arc of 95 ft from
the rubber, a 13 ft dirt circle round the plate and each base, an 18 ft mound). Only three numbers are measured: the
plate position and the bearing to second base, fitted to an aerial return-intensity raster in which dirt reads darker
than grass. Nothing here names a ground.
"""
import math
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely import affinity

FT = .3048
BASE = 90 * FT                  # base to base
RUBBER = 60.5 * FT              # plate apex to the front of the rubber
MOUND_C = 59 * FT               # plate apex to the mound centre
MOUND_R = 9 * FT
MOUND_H = 10 / 12 * FT
GRASS_ARC = 95 * FT             # rubber front to the infield grass line
CUT_R = 13 * FT                 # dirt circle round the plate and the bases
PATH = 3 * FT                   # half width of a base path
LINE = 4 / 12 * FT              # chalk width


def frame(plate, bearing):
    """Unit vectors: f toward second base, r toward first base (the batter's right)."""
    f = np.array([math.cos(bearing), math.sin(bearing)]); r = np.array([f[1], -f[0]])
    return np.asarray(plate, float), f, r


def local_to_world(plate, bearing, xy):
    p, f, r = frame(plate, bearing); xy = np.atleast_2d(xy)
    return p + xy[:, :1] * r + xy[:, 1:2] * f


def layout(plate, bearing, base_paths=True, foul_cutouts=True):
    """World polygons of the infield parts, keyed by surface: dirt (skin, plate circle, base cutouts, mound),
    grass (the infield square, inside the base paths when the ground has them), bases, the plate, chalk (foul lines
    to 150 m, batter's boxes). Grounds differ in two ways the rulebook leaves open: grass may run to the base lines,
    and the dirt round first and third may stop at the foul line; `variant` reads both from the returns."""
    def world(poly):
        p, f, r = frame(plate, bearing)
        return affinity.affine_transform(poly, [r[0], f[0], r[1], f[1], p[0], p[1]])
    s = BASE / math.sqrt(2)
    first, second, third = (s, s), (0, 2 * s), (-s, s)
    fair = Polygon([(0, 0), (200, 200), (-200, 200)])
    skin = Point(0, RUBBER).buffer(GRASS_ARC, 128).intersection(fair)
    cutouts = unary_union([Point(*b).buffer(CUT_R, 64) for b in (first, second, third)])
    plate_circle = Point(0, 0).buffer(CUT_R, 64)
    mound = Point(0, MOUND_C).buffer(MOUND_R, 64)
    if not foul_cutouts:cutouts = cutouts.intersection(fair)
    square = Polygon([(0, 0), first, second, third])
    grass = (square.buffer(-PATH, join_style=2) if base_paths else square).difference(cutouts).difference(plate_circle).difference(mound)
    dirt = unary_union([skin, cutouts, plate_circle]).difference(grass)
    bag = affinity.rotate(box(-.19, -.19, .19, .19), 45, origin=(0, 0))       # 15 in bags, square to the base lines
    bases = unary_union([affinity.translate(bag, *b) for b in (first, second, third)])
    home = Polygon([(-.216, .216), (.216, .216), (.216, 0), (0, -.216), (-.216, 0)])
    line = Polygon([(-LINE / 2, 0), (LINE / 2, 0), (LINE / 2, 150), (-LINE / 2, 150)])
    foul = unary_union([affinity.rotate(line, -45, origin=(0, 0)), affinity.rotate(line, 45, origin=(0, 0))])
    # Batter's boxes: 4 ft by 6 ft, 6 in clear of the 17 in plate, drawn as chalk outlines.
    boxes = unary_union([box(x0, -.83, x0 + 1.22, 1.0).exterior.buffer(LINE / 2) for x0 in (.368, -.368 - 1.22)])
    rubber = box(-.305, RUBBER, .305, RUBBER + .152)
    return {k: world(v) for k, v in {'dirt': dirt, 'grass': grass, 'mound': mound, 'bases': bases, 'home': home,
                                      'chalk': unary_union([foul, boxes]), 'rubber': rubber}.items()}


def template(plate, bearing, xs, ys, base_paths=True, foul_cutouts=True):
    """1 where the layout puts dirt, 0 on grass, for raster cell centres (xs, ys); compared within the grass arc (plus
    8 m) in fair territory and round the plate and the corner bases, NaN elsewhere."""
    p, f, r = frame(plate, bearing)
    dx, dy = xs - p[0], ys - p[1]; u = dx * r[0] + dy * r[1]; v = dx * f[0] + dy * f[1]
    s = BASE / math.sqrt(2)
    fair = v >= np.abs(u)
    skin = fair & (np.hypot(u, v - RUBBER) <= GRASS_ARC)
    near_base = (np.hypot(u - s, v - s) <= CUT_R) | (np.hypot(u, v - 2 * s) <= CUT_R) | (np.hypot(u + s, v - s) <= CUT_R)
    if not foul_cutouts: near_base &= fair
    plate_c = np.hypot(u, v) <= CUT_R; mound = np.hypot(u, v - MOUND_C) <= MOUND_R
    a = (u + v) / math.sqrt(2); b = (v - u) / math.sqrt(2)          # along the first- and third-base lines
    w = PATH if base_paths else 0.
    grass_sq = (a >= w) & (a <= BASE - w) & (b >= w) & (b <= BASE - w) & ~near_base & ~plate_c & ~mound
    dirt = (skin | near_base | plate_c | mound) & ~grass_sq
    region = ((fair & (np.hypot(u, v - RUBBER) <= GRASS_ARC + 8)) | (np.hypot(u, v) <= CUT_R + 4)
              | (np.hypot(u - s, v - s) <= CUT_R + 2) | (np.hypot(u + s, v - s) <= CUT_R + 2))
    return np.where(region, dirt.astype(float), np.nan)


def fit(intensity, xs, ys, plate0, bearing0, span_m=4., span_deg=25.):
    """Plate and bearing that best separate dark (dirt) from bright (grass) returns: the template's dirt cells should
    be darker than its grass cells. Coarse then fine search round the starting guess. Returns plate, bearing, score."""
    good = np.isfinite(intensity); I = (intensity - np.nanmean(intensity)) / (np.nanstd(intensity) + 1e-9)
    def score(p, b):
        t = template(p, b, xs, ys); m = good & np.isfinite(t)
        if m.sum() < 200: return -9.
        d = t[m] > .5
        if d.sum() < 50 or (~d).sum() < 50: return -9.
        return float(I[m][~d].mean() - I[m][d].mean())          # grass brighter than dirt
    best = (score(plate0, bearing0), np.asarray(plate0, float), bearing0)
    for step_m, step_deg, reach_m, reach_deg in ((1., 2.5, span_m, span_deg), (.25, .5, 1.25, 3.), (.1, .2, .3, .6)):
        c_p, c_b = best[1], best[2]
        for db in np.arange(-reach_deg, reach_deg + 1e-9, step_deg):
            for dx in np.arange(-reach_m, reach_m + 1e-9, step_m):
                for dy in np.arange(-reach_m, reach_m + 1e-9, step_m):
                    p = c_p + (dx, dy); b = c_b + math.radians(db); s = score(p, b)
                    if s > best[0]: best = (s, p, b)
    return best[1], best[2], best[0]


def variant(intensity, xs, ys, plate, bearing):
    """Whether this ground skins its base paths and cuts dirt round first and third into foul ground: each strip is
    dirt if its returns are nearer the fitted skin's than the infield grass's."""
    both = template(plate, bearing, xs, ys, True, True); none = template(plate, bearing, xs, ys, False, False)
    base = template(plate, bearing, xs, ys, False, True)
    ok = np.isfinite(intensity)
    dirt = ok & np.isfinite(none) & (none > .5); grass = ok & np.isfinite(none) & (none < .5)
    if dirt.sum() < 50 or grass.sum() < 50: return True, True
    d_mean, g_mean = intensity[dirt].mean(), intensity[grass].mean()
    def dirt_like(mask):
        m = ok & mask
        return bool(m.sum() >= 20 and abs(intensity[m].mean() - d_mean) < abs(intensity[m].mean() - g_mean))
    paths = np.isfinite(both) & (both > .5) & np.isfinite(base) & (base < .5)            # the base path strips alone
    foul = np.isfinite(base) & (base > .5) & np.isfinite(none) & (none < .5)             # dirt past the foul lines
    return dirt_like(paths), dirt_like(foul)
