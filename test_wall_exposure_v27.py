"""Wall exposure: what a wall faces decides whether it carries windows. Location-free synthetic outlines.

Guards the v0.24 regression the photo judge exposed: a wall that rises above a lower neighbour was treated as a
street facade and given windows; it is a party wall and stays blank. A tower face over its own podium keeps its
facade; a wall over a narrow gap is a lot-line wall; a wall to a yard is a lesser facade; a street wall is a full one.
"""
import numpy as np
from shapely.geometry import box, LineString, Point
from shapely.strtree import STRtree
from pipeline.facades import wall_exposure


def ctx_for(outlines, street=None, tops=None):
    cands = [{'id': i, 'g': g, **({'sourceId': s} if s else {})} for i, g, s in outlines]
    ctx = {'candidates': cands, 'building_index': STRtree([c['g'] for c in cands]),
           'building_sources': [c.get('sourceId', c['id']) for c in cands], 'street_corridor': street,
           'all_buildings': None}
    tops = tops or {}
    ctx['surfaceTop'] = lambda x, y: next((z for g, z in tops.items() if g.contains(Point(x, y))), -1e9)
    return ctx, cands


def test_party_wall_above_a_lower_neighbour_is_blank():
    own, nb = box(0, 0, 10, 10), box(10, 0, 20, 10)                    # the neighbour touches the east wall
    ctx, cands = ctx_for([('a', own, None), ('b', nb, None)], tops={nb: 4.})
    assert wall_exposure(ctx, cands[0], np.array([10., 5.]), np.array([1., 0.]), 0., 30.) == 'party'


def test_tower_face_over_its_own_podium_keeps_its_facade():
    tower, podium = box(0, 0, 10, 10), box(10, 0, 20, 10)
    ctx, cands = ctx_for([('t', tower, 'src'), ('t_p1', podium, 'src')], tops={podium: 6.})
    assert wall_exposure(ctx, cands[0], np.array([10., 5.]), np.array([1., 0.]), 0., 30.) == 'own-podium'
    ctx, cands = ctx_for([('t', tower, 'src'), ('t_p1', podium, 'src')], tops={podium: 29.})
    assert wall_exposure(ctx, cands[0], np.array([10., 5.]), np.array([1., 0.]), 0., 30.) == 'hidden'


def test_street_gap_and_yard_walls():
    own = box(0, 0, 10, 10)
    street = LineString([(-50, 22), (50, 22)]).buffer(5.)              # a street 12-32 m north of the wall
    ctx, cands = ctx_for([('a', own, None)], street=street)
    assert wall_exposure(ctx, cands[0], np.array([5., 10.]), np.array([0., 1.]), 0., 12.) == 'street'
    ctx, cands = ctx_for([('a', own, None), ('b', box(12, 0, 20, 10), None)], street=street)
    assert wall_exposure(ctx, cands[0], np.array([10., 5.]), np.array([1., 0.]), 0., 12.) == 'gap'
    ctx, cands = ctx_for([('a', own, None), ('b', box(0, -40, 10, -25), None)], street=street)
    assert wall_exposure(ctx, cands[0], np.array([5., 0.]), np.array([0., -1.]), 0., 12.) == 'secondary'


def test_long_wall_is_cut_at_a_photographed_frame():
    from pipeline.facades import split_at_frames, frame_overlap
    # a 40 m wall facing +y; u = (-ny, nx) = (-1, 0); a reading photographed the stretch x in [10, 18]
    p = {'normal': [0., 1.], 'frameU': [-18., -10.]}
    pieces = split_at_frames([((0., 0.), (40., 0.))], [p])
    xs = sorted(round(min(a[0], b[0]), 3) for a, b in pieces)
    assert xs == [0., 10., 18.], xs
    inside = [(a, b) for a, b in pieces if abs(frame_overlap(p, a, b) - abs(b[0] - a[0])) < 1e-6]
    assert len(inside) == 1 and round(abs(inside[0][1][0] - inside[0][0][0]), 3) == 8.
