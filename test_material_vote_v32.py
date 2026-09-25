"""The material a building's unphotographed walls continue: readings vote by colour likeness, not exact names. Synthetic."""
from pipeline import facades


def reading(fid, family, rgb):
    return {'match': {'featureId': fid}, 'materialEvidence': {'family': family, 'finish': 'default', 'colorRgb': rgb, 'sourceIds': ['x']}}


def test_near_identical_bricks_agree():
    c = {'id': 'b', 'sourceId': 'b', 'osm': None}
    ps = [reading('b', 'brick', [178, 84, 60]), reading('b', 'brick', [191, 118, 107]), reading('b', 'brick', [193, 124, 113])]
    m = facades.observed_material(ps, c)
    assert m is not None and m.startswith('brick_modular#'), m          # an exact-name vote split three ways: None, then the white prior


def test_different_colours_do_not_continue():
    c = {'id': 'b', 'sourceId': 'b', 'osm': None}
    ps = [reading('b', 'brick', [150, 60, 45]), reading('b', 'render', [225, 222, 214]), reading('b', 'render', [60, 70, 80])]
    assert facades.observed_material(ps, c) is None
