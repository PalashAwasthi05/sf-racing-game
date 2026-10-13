"""The material a building's unphotographed walls continue: readings vote by colour likeness, not exact names. Synthetic.
The rule is switched off in builds (VOTE_BY_LIKENESS=False) until readings carry per-part evidence identity."""
from pipeline import facades
import pytest


@pytest.fixture(autouse=True)
def likeness_vote_on(monkeypatch):
    monkeypatch.setattr(facades, 'VOTE_BY_LIKENESS', True)   # the rule is kept but off in builds (see pipeline/facades.py)


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


def test_builds_keep_the_exact_name_vote(monkeypatch):
    monkeypatch.setattr(facades, 'VOTE_BY_LIKENESS', False)
    c = {'id': 'b', 'sourceId': 'b', 'osm': None}
    ps = [reading('b', 'brick', [178, 84, 60]), reading('b', 'brick', [191, 118, 107]), reading('b', 'brick', [193, 124, 113])]
    assert facades.observed_material(ps, c) is None         # three names, no two-thirds majority: the prior
