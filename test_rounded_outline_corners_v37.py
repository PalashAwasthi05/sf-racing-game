"""Rounded corners of a block standing flush on its outline, and corners the raster draws as a chamfer or jogs (v0.37).

A rounded office block whose front stands on its lot outline kept square corners there: round_tier_corners left every
corner on the outline alone, and the outline's own wall ran on to the corner at the block's height. Its entrance corner
stayed square for a second reason: canopy and tree returns broke the tier outline there into a chamfer and 2-4 m jogs,
so no vertex was a right angle. Location-free synthetic outlines and returns only."""
import math
import numpy as np
from shapely import affinity
from shapely.geometry import box, Point, Polygon
from pipeline import spatial_rules, facades
from test_tier_frame_v36 import tiers_of, rounded_block


def flush_block(step=.5, r=2.5, round_it=True):
    """A 42 x 30 m outline turned 30 degrees: a 4 m podium, and a 30 x 24 m block to 20 m standing 6 m in from three
    sides and flush with the fourth (v = 30), its corners rounded to r (or square)."""
    u, v = np.meshgrid(np.arange(.25, 42, step), np.arange(.25, 30, step)); u, v = u.ravel(), v.ravel()
    x0, x1, y0, y1 = 6., 36., 6., 30.
    inside = (u > x0) & (u < x1) & (v > y0) & (v < y1)
    if round_it:
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            corner = (np.abs(u - cx) <= r) & (np.abs(v - cy) <= r) & ((u < x0 + r) | (u > x1 - r)) & ((v < y0 + r) | (v > y1 - r))
            inside &= ~(corner & ((u - cx) ** 2 + (v - cy) ** 2 > r * r))
    z = np.where(inside, 20., 4.)
    a = math.radians(30.); x = u * math.cos(a) - v * math.sin(a); y = u * math.sin(a) + v * math.cos(a)
    return affinity.rotate(box(0, 0, 42, 30), 30., origin=(0, 0)), np.c_[x, y, z]


def rounded(outline, pts):
    base, kept, frame = tiers_of(outline, pts, budget=3)
    fk = [{**t, 'geometry': frame.geom(t['geometry'])} for t in kept]
    P = np.c_[frame.points(pts[:, :2]), pts[:, 2]]
    records = []
    out, corners = spatial_rules.round_tier_corners(fk, P, base, outer=frame.geom(outline).boundary, records=records)
    return base, fk, out, corners, records


def test_corners_of_a_block_flush_on_its_outline_are_rounded_and_handed_to_the_outline_wall():
    outline, pts = flush_block()
    base, fk, out, corners, records = rounded(outline, pts)
    assert len(corners) == 4, corners                      # the two on the outline too
    assert len(records) == 2, records                      # those two go to the outline's wall
    for rc in records:
        assert 1.75 <= rc['radiusM'] <= 3.25, rc['radiusM']
        assert abs(rc['bottomZ'] - base) < .01
        assert len(rc['outerAxes']) == 1                   # one wall runs on the outline, the other stands inside it
    block = out[-1]['geometry']; square = fk[-1]['geometry']
    assert 3.5 < square.area - block.area < 9.5, square.area - block.area


def test_a_square_block_flush_on_its_outline_stays_square():
    outline, pts = flush_block(round_it=False)
    _, _, _, corners, records = rounded(outline, pts)
    assert corners == [] and records == [], (corners, records)


def test_the_existing_rounded_block_is_unchanged():
    outline, pts = rounded_block()
    _, _, _, corners, records = rounded(outline, pts)
    assert len(corners) == 4 and records == []


def test_a_chamfered_and_jogged_corner_is_found_at_the_crossing_of_its_walls():
    # walls along y = 20 (0..16) and x = 30 (down from 14); between them a chamfer and two small jogs
    q = Polygon([(0, 0), (30, 0), (30, 14), (31.5, 14.5), (29, 16.5), (28.5, 18), (26, 19), (24, 20), (0, 20)])
    found = [(tuple(np.round(b, 2)), L1, L2) for b, ex, ey, L1, L2, G1, G2 in spatial_rules.wall_corners(q)]
    near = [f for f in found if math.hypot(f[0][0] - 30, f[0][1] - 20) < .1]
    assert len(near) == 1, found
    assert max(near[0][1], near[0][2]) > 20              # paired with the long walls, not with a jog


def test_a_lower_terrace_running_on_past_the_corner_is_not_cut():
    # the lower tier runs 8 m past the block's corner along one wall: it is not flush there and keeps its wall
    outline = box(0, 0, 60, 40)
    u, v = np.meshgrid(np.arange(.25, 60, .5), np.arange(.25, 40, .5)); u, v = u.ravel(), v.ravel()
    r = 2.5; block = (u > 10) & (u < 40) & (v > 10) & (v < 40)
    cx, cy = 40 - r, 40 - r
    block &= ~((u > cx) & (v > cy) & ((u - cx) ** 2 + (v - cy) ** 2 > r * r))
    terrace = (u > 5) & (u < 48) & (v > 5) & (v < 40)
    z = np.where(block, 20., np.where(terrace, 10., 4.))
    tiers = [{'bottom': 0., 'top': 6., 'geometry': box(5, 5, 48, 40)}, {'bottom': 6., 'top': 16., 'geometry': box(10, 10, 40, 40)}]
    for i, t in enumerate(tiers): t['cap'] = t['geometry'].difference(tiers[i + 1]['geometry']) if i + 1 < len(tiers) else t['geometry']
    records = []
    out, corners = spatial_rules.round_tier_corners(tiers, np.c_[u, v, z], 4., outer=outline.boundary, records=records)
    assert any(math.hypot(b[0] - 40, b[1] - 40) < .5 for b in corners), corners
    assert abs(out[0]['geometry'].area - tiers[0]['geometry'].area) < .01      # the terrace keeps its corner
    assert out[1]['geometry'].area < tiers[1]['geometry'].area - 1.
    rc = [x for x in records if math.hypot(x['corner'][0] - 40, x['corner'][1] - 40) < .5]
    assert rc and abs(rc[0]['bottomZ'] - 10.) < .01                         # the outline wall stops at the block's foot


def test_the_outline_wall_stops_under_the_rounded_tiers_over_the_arc():
    segments = [((0., 0.), (20., 0.))]; levels = [22.]; gables = [False]
    rc = {'corner': (12., 0.), 'along': [(-1., 0.)], 'radiusM': 3., 'marginM': 1.5, 'bottomZ': 10.}
    seg, lv, gb, parents = facades.cap_rounded_spans(segments, levels, gables, [rc])
    pieces = [(round(a[0], 2), round(b[0], 2), l) for (a, b), l in zip(seg, lv)]
    assert pieces == [(0., 8.95, 22.), (8.95, 16., 10.), (16., 20., 22.)], pieces
    assert gb == [False, False, False]
    assert parents == [20., 20., 20.]                    # every piece remembers the wall it was cut from


def test_a_small_radius_arc_carries_the_facade_round_the_corner():
    # a small rounded corner cut out of a jagged tier came out as two facets (1.16 m and 0.86 m) turning 37 degrees; they
    # stood as blank stubs while the walls either side carried ribbon windows
    from test_tier_frame_v36 import capture_ctx
    from shapely.geometry import Polygon as Poly_
    outline = box(-10, -10, 40, 40); base = 4.
    p1 = np.array([30., 17.8]); a1, a2 = math.radians(116.), math.radians(153.)
    p2 = p1 + 1.16 * np.array([math.cos(a1), math.sin(a1)]); p3 = p2 + .86 * np.array([math.cos(a2), math.sin(a2)])
    g = Poly_([(0, 0), (30, 0), tuple(p1), tuple(p2), tuple(p3), (0, p3[1])])
    ctx, cubes = capture_ctx(outline)
    c = {'id': 'b', 'g': g, 'osm': None, 'full_area': outline.area, 'roofBand': True, 'elevationOffset': base, 'outerBoundary': outline.boundary,
         'tierBase': base, 'tierAbove': None, 'tiersBelow': []}
    ctx.update(building_index=None, all_buildings=Poly_(), street_corridor=outline.buffer(20))
    records, components = facades.compile(ctx, c, base, 16., 20., {}, [])
    arc = [r_ for r_ in records if r_['lengthM'] < 2.5 and abs(r_['normal'][0]) > .2 and abs(r_['normal'][1]) > .2]
    assert len(arc) >= 2 and all(r_.get('status') != 'blank' and r_.get('columns') == 1 for r_ in arc), [(r_['lengthM'], r_.get('status'), r_.get('columns')) for r_ in arc]
