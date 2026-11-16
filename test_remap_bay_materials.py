import copy

import numpy as np

from regen.remap_readings import remap


class AffineWallMap:
    L = 10.

    def __call__(self, x, z):
        return 2. * np.asarray(x) + 10., 3. * np.asarray(z) - 4.

    def stretch(self, x, z):
        return 2., 3.


def test_secondary_height_transforms_keep_the_original_horizontal_position():
    class ShearedMap:
        L = 10.

        def __call__(self, x, z):
            return np.asarray(x) + 10., np.asarray(z) + .25 * np.asarray(x) + 2.

        def stretch(self, x, z):
            return 1., 1.

    source = {'porches': [{'x0': 1., 'x1': 5., 'floorM': 1., 'topM': 4.}],
              'fireEscapes': [{'x0': 1., 'x1': 5., 'levels': [2., 5.]}],
              'piers': [{'xM': 3., 'widthM': .2, 'z0': 1., 'z1': 6.}]}
    result = remap(source, ShearedMap())
    assert result['porches'][0]['floorM'] == 3.75
    assert result['porches'][0]['topM'] == 6.75
    assert result['fireEscapes'][0]['levels'] == [4.75, 7.75]
    assert result['piers'][0]['z0'] == 3.75
    assert result['piers'][0]['z1'] == 8.75
    assert result['piers'][0]['xM'] == 13.


def test_bay_material_zones_inherit_and_remap_bounds_and_measured_sizes_once():
    reading = {
        'bays': [{
            'x0': 1., 'x1': 5., 'z0': 2., 'z1': 8., 'shape': 'angled',
            'materialZones': [
                {'z0': 3., 'z1': 4., 'family': 'brick', 'hex': '#aabbcc',
                 'pattern': 'horizontal_courses', 'finish': 'weathered',
                 'moduleM': {'x': .2, 'z': .08}, 'unitSizeM': .1},
                {'x0': 2., 'x1': 4., 'z0': 4., 'z1': 6., 'family': 'timber',
                 'hex': '#123456', 'pattern': 'vertical_boards', 'finish': 'matte',
                 'moduleM': {'x': .15, 'z': .3}, 'unitSizeM': .12},
                {'z0': 6., 'z1': 7., 'family': 'stone', 'hex': '#654321',
                 'pattern': 'unknown', 'unitSizeM': .25},
            ],
        }]
    }
    original = copy.deepcopy(reading)

    result = remap(reading, AffineWallMap())
    bay = result['bays'][0]
    zones = bay['materialZones']

    assert (bay['x0'], bay['x1'], bay['z0'], bay['z1']) == (12., 20., 2., 20.)
    assert (zones[0]['x0'], zones[0]['x1'], zones[0]['z0'], zones[0]['z1']) == (12., 20., 5., 8.)
    assert zones[0]['moduleM'] == {'x': .4, 'z': .24}
    assert np.isclose(zones[0]['unitSizeM'], .3)  # horizontal courses use the z scale
    assert zones[0]['pattern'] == 'horizontal_courses'
    assert zones[0]['finish'] == 'weathered' and zones[0]['hex'] == '#aabbcc'

    assert (zones[1]['x0'], zones[1]['x1'], zones[1]['z0'], zones[1]['z1']) == (14., 18., 8., 14.)
    assert np.isclose(zones[1]['moduleM']['x'], .3)
    assert np.isclose(zones[1]['moduleM']['z'], .9)
    assert np.isclose(zones[1]['unitSizeM'], .24)  # vertical boards use the x scale
    assert zones[1]['pattern'] == 'vertical_boards'
    assert zones[2]['unitSizeM'] == .25  # an unidentified pattern supplies no pitch axis
    assert reading == original


def test_bay_zone_with_incomplete_vertical_bounds_is_not_given_invented_geometry():
    reading = {'bays': [{'x0': 1., 'x1': 5., 'z0': 2., 'z1': 8.,
                         'materialZones': [{'z0': 3., 'family': 'brick'}]}]}

    result = remap(reading, AffineWallMap())
    zone = result['bays'][0]['materialZones'][0]
    assert (zone['x0'], zone['x1']) == (1., 5.)  # original parent bounds inherited
    assert zone['z0'] == 3. and 'z1' not in zone


def test_bay_rows_and_panes_follow_camera_transform_with_their_shell():
    source = {'bays': [{'x0': 1., 'x1': 5., 'z0': 2., 'z1': 8.,
                       'rows': [[3., 4.], [5., 7.]],
                       'front': {'count': 2, 'widthM': .8},
                       'side': {'widthM': .4}}]}
    result = remap(source, AffineWallMap())['bays'][0]
    assert result['rows'] == [[5., 8.], [11., 17.]]
    assert result['front'] == {'count': 2, 'widthM': 1.6}
    assert result['side'] == {'widthM': .8}
    assert source['bays'][0]['rows'] == [[3., 4.], [5., 7.]]
