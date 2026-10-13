"""An unread building borrows a measured facade of similar height, deterministically by its own id."""
from pipeline import facades

def reading(i, h, rgb):
    src = ['mapillary:%d' % i]
    return {'id': 'r%d' % i, 'status': 'image-measured-auto', 'heightRefM': h, 'match': {'featureId': 'x%d' % i}, 'normal': [0, 1], 'sourceIds': src,
            'materialEvidence': {'family': 'brick', 'finish': 'default', 'colorRgb': rgb, 'sourceIds': src},
            'groups': [{'region': [0, .2, 1, .97]}], 'reading': {'floorPitchM': 3.4, 'bayPitchM': 3.0, 'openingRatio': .5, 'openingHeightRatio': .6, 'shopfront': True}}

def test_donor_is_similar_height_and_deterministic():
    pool = [reading(i, h, [150, 70, 50]) for i, h in enumerate([6, 9, 12, 30, 45, 60, 8, 10])]
    d1 = facades.area_donor(pool, 'building-a', 10.)
    assert d1['heightRefM'] <= 12 and d1 is facades.area_donor(pool, 'building-a', 10.)
    style = facades.lend(facades.DEFAULTS['masonry'].copy(), d1)
    assert style['material'].startswith('brick_modular#') and style['floorHeightM'] == 3.4 and style['ground'] == 'shopfront'
