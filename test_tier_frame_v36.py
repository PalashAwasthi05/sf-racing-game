"""Tiers made along the outline's own axes, one wall per flush face, the tiers that matter kept (v0.36).

A block standing on a podium at 45 degrees to the grid came out as scalloped zig-zag walls (a north-up raster's cell
staircase), as three stacked slices of one vertical wall (each with a cornice), and without its set-back top storey (the
slices used the tier budget). Location-free synthetic outlines and returns only."""
import math
import numpy as np
from shapely import affinity
from shapely.geometry import box, Point
from rasterio.transform import from_origin
from pipeline import spatial_rules, roof_envelope, facades

ANGLE = 44.4


def podium_block(step=.5, angle=ANGLE):
    """A 50 x 38 m outline turned `angle` degrees: a 3.2 m planted podium, a 36 x 28 m block to 22.2 m standing 5 m in
    from three sides and flush with the fourth, and a top storey to 26.3 m set back 4 m inside the block."""
    u, v = np.meshgrid(np.arange(.25, 50, step), np.arange(.25, 38, step)); u, v = u.ravel(), v.ravel()
    block = (u > 7) & (u < 43) & (v > 5) & (v < 38)
    top = (u > 11) & (u < 39) & (v > 9) & (v < 34)
    z = np.where(top, 26.3, np.where(block, 22.2, 3.2))
    a = math.radians(angle); x = u * math.cos(a) - v * math.sin(a); y = u * math.sin(a) + v * math.cos(a)
    return affinity.rotate(box(0, 0, 50, 38), angle, origin=(0, 0)), np.c_[x, y, z]


def tiers_of(outline, pts, budget=3):
    """The generator's roof section: surface raster in the outline frame, base roof, tiers, the budget."""
    frame = spatial_rules.OutlineFrame(outline); gF = frame.geom(outline); step = 1.5
    xmin, ymin, xmax, ymax = gF.bounds; nx = math.ceil((xmax - xmin) / step); ny = math.ceil((ymax - ymin) / step)
    roof = pts[pts[:, 2] > 3.]
    rr, _ = spatial_rules.surface_raster(frame.points(roof), xmin, ymax, step, nx, ny)
    rr, known, _, _ = roof_envelope.prepare_grid(rr, gF, xmin, ymax, step)
    base = max(4., roof_envelope.base_level(rr, known, step, outline.area))
    quant = np.maximum(0, np.round((rr - base) / 2.5)).astype(np.int16); tr = from_origin(xmin, ymax, step, step)
    bands = spatial_rules.roof_bands(quant, tr, gF, max(150., outline.area * .15), simplify_m=step * .75, cell_m=step, level_step_m=2.5, measured_mask=known)
    bands = spatial_rules.measured_tiers(bands, rr, known, tr, base)
    bands = [{**b, 'geometry': frame.world(b['geometry']), 'cap': frame.world(b['cap'])} for b in bands]
    return base, spatial_rules.keep_tiers(bands, budget), frame


def test_rotated_block_is_one_tier_with_straight_walls_and_keeps_its_top_storey():
    outline, pts = podium_block()
    base, kept, frame = tiers_of(outline, pts)
    tops = sorted(round(base + t['top'], 1) for t in kept)
    assert tops == [22.2, 26.3], tops                      # the block is one tier, and the set-back storey is kept
    for t in kept:
        g = t['geometry']; assert g.geom_type == 'Polygon' and not g.interiors, g.geom_type
        xy = frame.points(np.asarray(g.exterior.coords)); e = np.diff(xy, axis=0); e = e[np.linalg.norm(e, axis=1) > 1e-6]
        ang = np.degrees(np.arctan2(e[:, 1], e[:, 0])) % 90; off = np.minimum(ang, 90 - ang)
        assert len(e) <= 6 and off.max() < 1., (len(e), off.max())   # a rectangle along the outline, not a staircase


def test_budget_keeps_the_tiers_with_the_largest_roofs():
    big, mid, sliver, top = box(0, 0, 40, 30), box(0, 0, 36, 30), box(0, 0, 35.5, 30), box(5, 5, 30, 25)
    tiers = [{'bottom': 0., 'top': 6., 'geometry': big}, {'bottom': 6., 'top': 12., 'geometry': mid},
             {'bottom': 12., 'top': 18., 'geometry': sliver}, {'bottom': 18., 'top': 22., 'geometry': top}]
    kept = spatial_rules.keep_tiers(tiers, 3)
    assert [k['top'] for k in kept] == [6., 18., 22.], [k['top'] for k in kept]   # the 15 m2 sliver goes, not the top
    assert kept[1]['bottom'] == 6. and abs(kept[1]['cap'].area - (sliver.area - top.area)) < 1e-6


def test_base_is_a_real_lower_wing_but_not_a_ring_of_wall_cells():
    rr = np.full((20, 30), 28.); known = np.ones_like(rr, bool)
    rr[:, :2] = 6.; rr[-2:, :12] = 6.                      # an L-shaped wing two cells wide, 13 % of the roof
    assert abs(roof_envelope.base_level(rr, known, 1.5, 600 * 2.25) - 6.) < .1
    rr = np.full((30, 30), 20.); rr[0, :] = rr[-1, :] = rr[:, 0] = rr[:, -1] = 6.   # one cell of low returns all round
    assert abs(roof_envelope.base_level(rr, np.ones_like(rr, bool), 1.5, 900 * 2.25) - 20.) < .1


def test_a_step_between_two_flat_roofs_is_not_draped_but_a_ramp_is():
    rr = np.full((30, 40), 13.8); rr[8:22, 10:30] = 6.8     # a flat ring round a lower court
    known = np.ones_like(rr, bool)
    keep, _ = roof_envelope.drape_regions(rr, known, box(-1e3, -1e3, 1e3, 1e3), 0., 45., 1.5)
    assert not keep.any()
    ramp = np.tile(np.linspace(0., 12., 40), (30, 1))      # a steady 20 % grade over 60 m
    keep, _ = roof_envelope.drape_regions(ramp, known, box(-1e3, -1e3, 1e3, 1e3), 0., 45., 1.5)
    assert keep.mean() > .8


def test_a_lower_ledge_along_the_edge_stays_out_of_the_tier_and_drift_joins():
    q = np.zeros((20, 30), np.int16); q[2:18, 2:28] = 5
    q[2:18, 2:4] = 1                                       # a ledge along the west edge, 2.25 m inside the outline, clearly lower
    q[17, 4:28] = 1                                        # one row of cells along the south edge reading low: the wall's own cells
    measured = np.ones_like(q, bool); tr = from_origin(0, 30, 1.5, 1.5)
    bands = spatial_rules.roof_bands(q, tr, box(3.75, 3, 42, 27), 25., simplify_m=1.125, cell_m=1.5, level_step_m=2.5, measured_mask=measured)
    top = bands[-1]['geometry']
    assert not top.contains(Point(4.9, 15.)), top.bounds   # a ledge wider than a cell keeps its own lower roof
    assert top.contains(Point(20., 3.75)), top.bounds      # a single row of straddling cells is the same wall


def test_an_unknown_ring_round_a_plant_room_leaves_no_notch_in_the_roof():
    q = np.full((20, 30), 3, np.int16); q[1:6, 20:27] = 6   # a plant room near the north edge
    measured = np.ones_like(q, bool); measured[0:7, 19:28] = False; measured[1:6, 20:27] = True   # cells straddling its walls
    tr = from_origin(0, 30, 1.5, 1.5)
    bands = spatial_rules.roof_bands(q, tr, box(0, 0, 45, 30), 25., simplify_m=1.125, cell_m=1.5, level_step_m=2.5, measured_mask=measured)
    roof = bands[0]['geometry']
    assert roof.area > .97 * 45 * 30, roof.area


def capture_ctx(outline):
    cubes = []
    ctx = {'cube': lambda center, size, key, mat, angle=0: cubes.append((np.asarray(center, float), np.asarray(size, float), key)),
           'cylinder': lambda *a: None, 'put': lambda *a: None, 'polys': lambda g: [g] if g.geom_type == 'Polygon' else list(g.geoms),
           'lines': lambda g: [g] if g.geom_type == 'LineString' else list(getattr(g, 'geoms', [])), 'all_buildings': outline,
           'boundary': Point(0, 0).buffer(1e4)}
    return ctx, cubes


def test_a_flush_face_of_stacked_tiers_is_one_wall_from_the_roof_in_front():
    outline = box(-10, -10, 30, 40); base = 4.
    low, high = box(0, 0, 20, 30), box(0, 0, 20, 20)        # the high tier is flush with the low one on three sides
    walls = {}
    for name, g, bottom, top, above, below in (('low', low, 0., 6., high, []), ('high', high, 6., 18., None, [(low, base + 6.)])):
        ctx, cubes = capture_ctx(outline)
        c = {'id': 'b', 'g': g, 'osm': None, 'full_area': outline.area, 'roofBand': True, 'elevationOffset': base + bottom,
             'outerBoundary': outline.boundary, 'tierBase': base, 'tierAbove': above, 'tiersBelow': below}
        facades.compile(ctx, c, base + bottom, top - bottom, top, {}, [])
        walls[name] = [(ce, sz) for ce, sz, key in cubes if key == 'buildings__b']
    # the low tier draws only its own terrace edge and the sides beside the terrace, never the flush faces
    assert walls['low'] and all(ce[1] > 19.5 for ce, sz in walls['low']), [ce for ce, sz in walls['low']]
    south = [(ce, sz) for ce, sz in walls['high'] if ce[1] < 1.]
    assert south and all(abs((ce[2] - sz[2] / 2) - base) < .05 for ce, sz in south)   # the south face starts on the podium
    assert max(ce[2] + sz[2] / 2 for ce, sz in south) > base + 17.5
    north = [(ce, sz) for ce, sz in walls['high'] if ce[1] > 19.]
    assert north and all(ce[2] - sz[2] / 2 > base + 5.9 for ce, sz in north)          # above the terrace it starts on it


def test_a_blank_planter_wall_framed_on_the_block_behind_keeps_the_podium_blank_with_its_trims():
    outline = box(0, 0, 40, 20)
    u, v = np.meshgrid(np.arange(.25, 40, .5), np.arange(.25, 20, .5)); u, v = u.ravel(), v.ravel()
    samples = np.c_[u, v, np.full(len(u), 4.)]              # a planter wall all round, 4 m
    hand = {'id': 'hand', 'match': {'featureId': 'b'}, 'normal': [0., -1.], 'sourceIds': ['s'], 'heightRefM': 21.,
            'materialEvidence': {'family': 'brick', 'finish': 'default', 'colorRgb': [160, 92, 72], 'sourceIds': ['s']}, 'groups': [],
            'bands': [{'height': .17, 'thicknessM': .4, 'depthM': .35, 'projectionM': .14, 'material': 'redpaint'}],
            'ornaments': [{'kind': 'dentil', 'z': .13, 'pitchM': .34, 'sizeM': .16, 'heightM': .24, 'material': 'render_white'}]}
    ctx, cubes = capture_ctx(outline); ctx.update(PROFILES=[hand], FITS=[], roofSamples=samples)
    c = {'id': 'b', 'g': outline, 'osm': None, 'full_area': outline.area}
    records, components = facades.compile(ctx, c, 0., 4., 21., {}, [])
    east = [r for r in records if r['normal'][0] > .9]
    assert east and all(r.get('status') == 'inferred' and 'trims continued' in ' '.join(r.get('unresolved', [])) for r in east), east
    assert components == 0                                   # no windows, doors or shopfronts on the podium
    dentils = [ce for ce, sz, key in cubes if key.startswith('facades__b') and abs(sz[2] - .24) < 1e-6 and ce[0] > 39.]
    assert len(dentils) > 20                                 # the dentil course runs round the corner onto the east wall


def test_a_face_with_a_small_lower_piece_in_front_stands_on_the_podium_elsewhere():
    outline = box(-10, -10, 30, 40); base = 4.
    block = box(0, 0, 20, 20); low = block.union(box(8, -2, 12, 0))   # a 4 m wide vault in front of the south face's middle
    ctx, cubes = capture_ctx(outline)
    c = {'id': 'b', 'g': block, 'osm': None, 'full_area': outline.area, 'roofBand': True, 'elevationOffset': base + 6.,
         'outerBoundary': outline.boundary, 'tierBase': base, 'tierAbove': None, 'tiersBelow': [(low, base + 6.)]}
    records, _ = facades.compile(ctx, c, base + 6., 12., 18., {}, [])
    south = [r for r in records if r['normal'][1] < -.9]
    feet = sorted((round(r['midpoint'][0], 1), round(r.get('zBaseM', 0.), 2)) for r in south)
    # zBaseM is the wall's foot above the ground: on the vault's roof (10 m) above it, on the podium (4 m) elsewhere
    assert any(abs(x - 10.) < 1. and abs(z - 10.) < .01 for x, z in feet), feet
    assert any(x < 7. and abs(z - 4.) < .01 for x, z in feet) and any(x > 13. and abs(z - 4.) < .01 for x, z in feet), feet


def rounded_block(step=.5, r=2.5):
    """A 30 x 24 m block to 20 m on a 4 m podium (a 6 m ring), its corners rounded to r, turned 30 degrees."""
    u, v = np.meshgrid(np.arange(.25, 42, step), np.arange(.25, 36, step)); u, v = u.ravel(), v.ravel()
    x0, x1, y0, y1 = 6., 36., 6., 30.
    inside = (u > x0) & (u < x1) & (v > y0) & (v < y1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        corner = (np.abs(u - cx) <= r) & (np.abs(v - cy) <= r) & ((u < x0 + r) | (u > x1 - r)) & ((v < y0 + r) | (v > y1 - r))
        inside &= ~(corner & ((u - cx) ** 2 + (v - cy) ** 2 > r * r))
    z = np.where(inside, 20., 4.)
    a = math.radians(30.); x = u * math.cos(a) - v * math.sin(a); y = u * math.sin(a) + v * math.cos(a)
    return affinity.rotate(box(0, 0, 42, 36), 30., origin=(0, 0)), np.c_[x, y, z]


def test_rounded_corners_are_measured_and_filleted_in_every_flush_tier():
    outline, pts = rounded_block()
    base, kept, frame = tiers_of(outline, pts, budget=3)
    fk = [{**t, 'geometry': frame.geom(t['geometry'])} for t in kept]
    P = np.c_[frame.points(pts[:, :2]), pts[:, 2]]
    out, corners = spatial_rules.round_tier_corners(fk, P, base, outer=frame.geom(outline).boundary)
    assert len(corners) == 4, corners                     # all four corners of the block, none square
    block = out[-1]['geometry']
    square = fk[-1]['geometry']
    assert 3.5 < square.area - block.area < 7.5, square.area - block.area   # four cuts of (1 - pi/4) r2 = 1.34 m2
    assert len(block.exterior.coords) > 20                 # arcs, not chamfers


def test_arc_pieces_of_a_rounded_tier_carry_the_facade_round_the_corner():
    from shapely.geometry import Point as P_
    outline = box(-10, -10, 40, 40); base = 4.
    g = box(0, 0, 30, 20).difference(box(27.5, 17.5, 30, 20).difference(P_(27.5, 17.5).buffer(2.5, quad_segs=6)))
    ctx, cubes = capture_ctx(outline)
    c = {'id': 'b', 'g': g, 'osm': None, 'full_area': outline.area, 'roofBand': True, 'elevationOffset': base, 'outerBoundary': outline.boundary,
         'tierBase': base, 'tierAbove': None, 'tiersBelow': []}
    from shapely.geometry import Polygon as Poly_
    ctx.update(building_index=None, all_buildings=Poly_(), street_corridor=outline.buffer(20))   # a street all round, no neighbours
    records, components = facades.compile(ctx, c, base, 16., 20., {}, [])
    arc = [r for r in records if r['lengthM'] < 1.6 and abs(r['normal'][0]) > .2 and abs(r['normal'][1]) > .2]
    assert len(arc) >= 3 and all(r.get('status') != 'blank' and r.get('columns') == 1 for r in arc), [(r['lengthM'], r.get('status'), r.get('columns')) for r in arc]


def gap_roof(lower):
    """A 36 m roof (20 x 20 cells of 1.5 m) with cells at 33 m (`lower`: 'strip' along one side or 'ring' round the edge)
    and a few stragglers between the two, so that the 15th percentile lands in the gap."""
    rng = np.random.default_rng(3)
    rr = np.full((20, 20), 36.) + rng.uniform(-.3, .3, (20, 20)); known = np.ones((20, 20), bool)
    if lower == 'strip': rr[:, :2] = 33.; rr[:12, 2] = 33.                  # 13 % of the cells, two to three cells wide
    else: rr[0, :] = rr[-1, :] = 33.; rr[1:13, 0] = 33.                    # 13 % of the cells, one cell wide at the edge
    rr[2:14, 5] = np.linspace(33.6, 35.0, 12)                               # 3 % between the levels
    return rr, known


def test_a_base_percentile_in_the_gap_moves_to_the_lower_roof_or_to_the_main_roof():
    rr, known = gap_roof('strip')
    q = float(np.quantile(rr[known], .15)); assert 33.5 < q < 35.2, q          # the percentile itself is in the gap
    assert abs(roof_envelope.base_level(rr, known, 1.5, 900.) - 33.) < .2      # a strip of real extent: the base is the lower roof
    rr, known = gap_roof('ring')
    q = float(np.quantile(rr[known], .15)); assert 33.5 < q < 35.2, q
    assert abs(roof_envelope.base_level(rr, known, 1.5, 900.) - 36.) < .5      # a ring of edge cells is no roof: the base is the main roof
