"""Stands as floors: seating stays a raked surface, everything else becomes nested floor outlines with vertical walls.
Synthetic, location-free, and rotated off the raster axes on purpose: an axis-aligned test hides the staircase edges a
grid produces on a building that runs at an angle to it."""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely import affinity
from pipeline.landmarks import ballpark as bp

ANG = 30.


def section(d):
    """Height across the stand at distance d from the field edge: a raked bowl, a concourse, a tall back block."""
    return np.where(d < 15, 1. + d * .73, np.where(d < 22, 13.9, np.where(d < 30, 37., np.nan)))


def grid(step=.5):
    stand = affinity.rotate(box(0, 0, 60, 30), ANG, origin=(0, 0))           # the field edge is the side y = 0
    field = affinity.rotate(box(0, -40, 60, 0), ANG, origin=(0, 0))
    x0, y0, x1, y1 = stand.buffer(4).bounds
    xs = np.arange(x0, x1, step) + step / 2; ys = np.arange(y1, y0, -step) - step / 2; X, Y = np.meshgrid(xs, ys)
    c, s = math.cos(math.radians(-ANG)), math.sin(math.radians(-ANG)); d = X * s + Y * c   # distance from the field edge
    H = np.nan_to_num(section(np.clip(d, 0, 29.9)), nan=0.); inside = shapely.contains_xy(stand, X, Y)
    H = np.where(inside, H, 0.)
    return H, (x0, y1, step), inside, stand, field, d


def test_levels_merge_and_order():
    h = np.r_[np.full(400, 13.9), np.full(300, 37.), np.full(200, 21.4), np.full(150, 22.1)]
    lv = bp.levels_of(h + np.random.default_rng(0).normal(0, .05, len(h)), .5)
    assert len(lv) == 3 and abs(lv[0] - 13.9) < .2 and 21.4 < lv[1] < 22.1 and abs(lv[2] - 37.) < .2, lv


def test_seats_only_where_they_rake_toward_the_field():
    H, g, inside, stand, field, d = grid()
    ring = np.asarray(shapely.segmentize(field.exterior, .5).coords)[:-1]
    seat = bp.seat_cells(H, g, inside, ring)
    assert seat[inside & (d > 1.5) & (d < 13.5)].mean() > .9                 # the raked bowl
    assert seat[inside & (d > 16.5)].mean() < .02                            # the concourse and the back block


def test_floor_outlines_are_nested_straight_and_sliver_free():
    H, g, inside, stand, field, d = grid()
    ring = np.asarray(shapely.segmentize(field.exterior, .5).coords)[:-1]
    seat = bp.seat_cells(H, g, inside, ring)
    _, flat = bp.plateaus(H, g[2])
    levels, idx, seat = bp.level_cells(H, flat, g[2], inside, seat)
    assert [round(v, 1) for v in levels][-2:] == [13.9, 37.0], levels
    bands = bp.band_outlines(idx, levels, g, stand)
    k13, k37 = levels.index(min(levels, key=lambda v: abs(v - 13.9))), levels.index(min(levels, key=lambda v: abs(v - 37.)))
    lo, hi = bands[k13], bands[k37]
    assert hi.difference(lo).area < 1.                                       # nested
    assert abs(hi.area - 8 * 60) < 40 and abs(lo.area - 15 * 60) < 60, (lo.area, hi.area)
    for pk in (lo, hi):
        for q in bp._polys(pk):
            assert q.minimum_rotated_rectangle.exterior.length > 0 and q.buffer(-.34).area > 0      # no sliver pieces
            cs = np.asarray(q.exterior.coords); seg = np.diff(cs, axis=0); L = np.hypot(*seg.T)
            ang = np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) % 90
            off_axis = L[np.minimum(np.abs(ang - ANG % 90), 90 - np.abs(ang - ANG % 90)) > 8].sum()
            assert off_axis < .06 * L.sum(), (off_axis, L.sum())            # edges follow the building, not the grid
