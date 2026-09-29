"""Ribbon glazing and tier caps (v36): lights standing closer than a mullion's width apart are one band per storey, and a
tier's wall ends in a slim cap of its own material rather than a concrete cornice at every step."""
import numpy as np
from shapely.geometry import box, Point
from pipeline import facade_grammar, facades


def bank(pitch, ratio, **style):
    return {'id': 'p', 'groups': [{'region': [0, 0, 1, 1], 'columnPitchM': pitch, 'rowPitchM': 3.6,
                                   'style': {'kind': 'window', 'openingRatio': ratio, 'openingHeightRatio': .6, 'mullionFractions': [], 'transomFractions': [], **style}}]}


def test_lights_a_quarter_metre_apart_are_one_band_of_glass_per_storey():
    W, H = 31.6, 14.4                                   # 20 bays of 1.58 m, 4 storeys
    ops = facade_grammar.openings(bank(1.58, .84), [], W, H, H)
    assert len(ops) == 4, len(ops)
    for op in ops:
        st = op['style']
        assert st['ribbon'] and len(st['jointFractions']) == 19 and op['lights'] == 20
        assert abs((op['rect'][2] - op['rect'][0]) * W - (W - 1.58 * .16)) < .01    # the band runs the frontage
        assert .05 <= st['jointWidthM'] <= .12


def test_punched_windows_and_a_style_that_says_otherwise_stay_separate():
    W, H = 30., 14.4
    assert len(facade_grammar.openings(bank(3., .5), [], W, H, H)) == 40            # 1.5 m of wall between: punched
    assert len(facade_grammar.openings(bank(1.58, .84, ribbon=False), [], 31.6, H, H)) == 80


def test_a_ribbon_skips_an_omitted_light_and_keeps_the_lights_own_mullions():
    p = bank(1.5, .9, mullionFractions=[.5]); p['groups'][0]['omit'] = [[4, 0]]
    ops = [o for o in facade_grammar.openings(p, [], 15., 3.6, 3.6)]
    assert sorted(o.get('lights', 1) for o in ops) == [4, 5]
    five = next(o for o in ops if o.get('lights') == 5)
    assert len(five['style']['mullionFractions']) == 5 and len(five['style']['jointFractions']) == 4


def test_a_borrowed_style_carries_the_ribbon():
    prior = facades.DEFAULTS['commercial'].copy()
    ribbon = {'id': 'd', 'material': 'brick_modular', 'reading': {'floorPitchM': 3.58, 'bayPitchM': 1.58, 'openingRatio': .84, 'openingHeightRatio': .6}}
    punched = {'id': 'd', 'material': 'brick_modular', 'reading': {'floorPitchM': 3.58, 'bayPitchM': 3., 'openingRatio': .5, 'openingHeightRatio': .6}}
    assert facades.lend(prior, ribbon)['ribbon'] and not facades.lend(prior, punched)['ribbon']


def capture_ctx(outline):
    cubes = []
    ctx = {'cube': lambda center, size, key, mat, angle=0: cubes.append((np.asarray(center, float), np.asarray(size, float), key, mat)),
           'cylinder': lambda *a: None, 'put': lambda *a: None, 'polys': lambda g: [g] if g.geom_type == 'Polygon' else list(g.geoms),
           'lines': lambda g: [g] if g.geom_type == 'LineString' else list(getattr(g, 'geoms', [])), 'all_buildings': outline.buffer(-30),
           'boundary': Point(0, 0).buffer(1e4), 'building_index': None, 'street_corridor': outline.buffer(20)}
    return ctx, cubes


def tier_walls(**style):
    outline = box(-10, -10, 40, 40); base = 4.
    ctx, cubes = capture_ctx(outline)
    c = {'id': 'b', 'g': box(0, 0, 30, 20), 'osm': {'id': 1, 'tags': {}}, 'full_area': outline.area, 'roofBand': True, 'elevationOffset': base,
         'outerBoundary': outline.boundary, 'tierBase': base, 'tierAbove': None, 'tiersBelow': []}
    obs = [{'match': {'osmWayId': 1}, 'normal': list(n), **style} for n in ((0, -1), (1, 0), (0, 1), (-1, 0))]
    facades.compile(ctx, c, base, 14.4, 18.4, {}, obs)
    return cubes


def test_a_tier_wall_has_no_concrete_cornice_or_coping():
    cubes = tier_walls()
    long_concrete = [(ce, sz) for ce, sz, key, mat in cubes if mat == 'concrete' and sz[0] > 15.]
    assert not long_concrete, long_concrete
    caps = [(ce, sz) for ce, sz, key, mat in cubes if key == 'buildings__b' and sz[0] > 15. and sz[2] < .2 and ce[2] > 4. + 14.3]
    assert caps                                           # a slim cap of the wall's own material at the tier's top


def test_a_generic_ribbon_wall_has_wall_only_at_the_band_ends():
    ribbon = tier_walls(ribbon=True, bayWidthM=1.58, openingRatio=.84, openingHeightRatio=.6, floorHeightM=3.6)
    punched = tier_walls(ribbon=False, bayWidthM=3., openingRatio=.5, openingHeightRatio=.6, floorHeightM=3.6)
    def piers(cubes):   # wall pieces beside the glass of the second storey of the south face (its wall stands 0.19 m in)
        return [ce for ce, sz, key, mat in cubes if key == "buildings__b" and abs(ce[1] - .19) < .03 and ce[2] - sz[2] / 2 < 4. + 3.6 + 1.8 < ce[2] + sz[2] / 2 and sz[0] < 1.]
    assert len(piers(ribbon)) == 2, len(piers(ribbon))
    assert len(piers(punched)) >= 10


def test_a_reading_taken_a_few_degrees_off_the_wall_far_from_the_origin_still_draws_it():
    # a blank flank 100 m out, read on an older edge 6 degrees off: planeV lies along the reading's own normal
    import math
    from shapely import affinity
    g = affinity.translate(box(0, 0, 8, 12), 60, 80)                      # its south face at y = 80, facing (0, -1)
    outline = g.buffer(15)
    t = math.radians(6.); pn = np.array([math.sin(t), -math.cos(t)])     # the reading's normal, 6 degrees off
    mid = np.array([64., 80.]); pu = np.array([-pn[1], pn[0]]); mu = float(mid @ pu)
    reading = {'id': 'b-auto-264', 'match': {'featureId': 'b', 'partId': 'b'}, 'normal': pn.tolist(), 'planeV': round(float(mid @ pn), 2),
               'frameU': [mu - 4.5, mu + 4.5], 'heightRefM': 8., 'materialEvidence': {'family': 'brick', 'colorRgb': [150, 80, 60], 'sourceIds': ['mapillary:1']}, 'sourceIds': ['mapillary:1'],
               'status': 'image-measured-auto-v27', 'openings': [], 'panels': [], 'bands': [], 'groups': [], 'reading': {'blank': True, 'visibleShare': .9}}
    assert abs(float(mid @ np.array([0., -1.])) - reading['planeV']) > 2.5          # along the wall's normal it looks like another plane
    ctx, cubes = capture_ctx(outline)
    ctx['PROFILES'] = [reading]
    c = {'id': 'b', 'sourceId': 'b', 'g': g, 'osm': None, 'full_area': g.area, 'elevationOffset': 0.}
    records, _ = facades.compile(ctx, c, 0., 8., 8., {}, [])
    south = [r for r in records if r['normal'][1] < -.9]
    assert south and all(r.get('status') != 'inferred' for r in south), [(r.get('status'), r.get('grammarId')) for r in south]
