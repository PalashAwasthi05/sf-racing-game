"""A building no photograph covers keeps its published donor, and the material and grid it lent (facades.donor_for),
until that donor stops qualifying; write_donor_incumbents records them from a published build."""
import pytest
from pipeline import facades
from pipeline.surface_evidence import resolve_material
import write_donor_incumbents as W


def auto(i, h, material='render_gray#707070', normal=(1., 0.), floor=3.2, bay=3.0):
    return {'id': i, 'status': 'image-measured-auto-v27', 'heightRefM': h, 'material': material, 'normal': list(normal), 'match': {'featureId': 'b-' + i},
            'reading': {'floorPitchM': floor, 'bayPitchM': bay, 'openingRatio': .5, 'openingHeightRatio': .6, 'shopfront': False}}


@pytest.fixture
def incumbents(monkeypatch):
    table = {}
    monkeypatch.setattr(facades, '_INCUMBENTS', table)
    return table


POOL = [auto('r1', 12.), auto('r2', 13., 'brick_modular#aa5544'), auto('r3', 14.), auto('r4', 11.), auto('r5', 12.5), auto('r6', 30.)]


def test_without_an_incumbent_it_is_area_donor(incumbents):
    assert facades.donor_for(POOL, 'x', 'x', 12.)['id'] == facades.area_donor(POOL, 'x', 12.)['id']


def test_a_published_donor_survives_a_change_elsewhere_in_the_pool(incumbents):
    first = facades.area_donor(POOL, 'x', 12.)
    incumbents['x'] = {'donorId': first['id'], 'normal': first['normal'], 'material': resolve_material(first), 'grid': {'floorPitchM': 3.2, 'bayPitchM': 3.0, 'openingRatio': .5, 'openingHeightRatio': .6, 'shopfront': False}}
    # readings added closer in height push the published donor out of area_donor's five
    crowded = [auto(f'n{k}', 12. + k * .01) for k in range(8)] + POOL
    assert facades.area_donor(crowded, 'x', 12.)['id'] != first['id']
    assert facades.donor_for(crowded, 'x', 'x', 12.)['id'] == first['id']


def test_the_published_material_and_grid_are_lent_not_the_donor_s_new_reading(incumbents):
    incumbents['x'] = {'donorId': 'r2', 'normal': [1., 0.], 'material': 'brick_modular#aa5544', 'grid': {'floorPitchM': 3.9, 'bayPitchM': 4.1, 'openingRatio': .45, 'openingHeightRatio': .55, 'shopfront': True}}
    reread = [dict(p, material='panel_dark#202020', reading=dict(p['reading'], floorPitchM=2.8)) if p['id'] == 'r2' else p for p in POOL]
    d = facades.donor_for(reread, 'x', 'x', 12.)
    st = facades.lend(facades.DEFAULTS['commercial'].copy(), d)
    assert st['material'] == 'brick_modular#aa5544' and st['floorHeightM'] == 3.9 and st['bayWidthM'] == 4.1 and st['ground'] == 'shopfront' and st['donorId'] == 'r2'
    assert next(p for p in reread if p['id'] == 'r2')['material'] == 'panel_dark#202020'   # the pool itself is untouched


def test_a_donor_that_is_gone_or_whose_building_changed_height_is_chosen_afresh(incumbents):
    incumbents['x'] = {'donorId': 'r6', 'normal': [1., 0.], 'buildingHeightM': 12.}
    assert facades.donor_for(POOL, 'x', 'x', 12.)['id'] == 'r6'                           # published: kept, whatever its height
    assert facades.donor_for(POOL, 'x', 'x', 18.)['id'] == facades.area_donor(POOL, 'x', 18.)['id']   # the building is 50 % taller now
    incumbents['x'] = {'donorId': 'gone', 'normal': [1., 0.]}
    assert facades.donor_for(POOL, 'x', 'x', 12.)['id'] == facades.area_donor(POOL, 'x', 12.)['id']


def test_a_repeated_reading_id_is_matched_by_its_facing(incumbents):
    pool = POOL + [auto('r2', 13., 'stucco#dddddd', normal=(0., 1.))]
    incumbents['x'] = {'donorId': 'r2', 'normal': [0., 1.]}
    assert facades.donor_for(pool, 'x', 'x', 12.)['material'] == 'stucco#dddddd'
    incumbents['x'] = {'donorId': 'r2', 'normal': [-1., 0.]}                                 # no facing matches: afresh
    assert facades.donor_for(pool, 'x', 'x', 12.)['id'] == facades.area_donor(pool, 'x', 12.)['id']


def wall(bid, material, status='inferred', donor=None, grammar=False):
    w = {'buildingId': bid, 'material': material, 'status': 'visual-grammar' if grammar else status, 'midpoint': [0, 0], 'normal': [1, 0], 'lengthM': 10., 'heightM': 8.}
    if grammar: w['edge'] = 'grammar'
    if donor: w['donorId'] = donor
    return w


def test_incumbents_are_written_for_donor_drawn_buildings_only():
    walls = [wall('a', 'brick_modular#aa5544', donor='r2'), wall('a', 'brick_modular#aa5544', 'blank', donor='r2'),
             wall('b', 'render_gray#707070', grammar=True), wall('b', 'render_gray#707070'),
             wall('c', 'stucco')]
    out, skipped = W.donors_of(walls, {'a': {'height': 12.}, 'b': {'height': 12.}, 'c': {'height': 12.}}, POOL)
    assert set(out) == {'a'} and out['a'][0]['id'] == 'r2' and out['a'][1] == 'brick_modular#aa5544' and out['a'][2] == 12.
    assert 'c' in skipped                                                                     # its walls are not any donor's material
    assert W.grid_of(out['a'][0])['floorPitchM'] == 3.2
