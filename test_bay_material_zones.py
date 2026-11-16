from regen.specs import bay_material_zones, to_profile


def _piece():
    return {'id': 'generic-fixture-side', 'building': 'generic-fixture', 'lengthM': 10., 'heightM': 10.,
            'normal': [0., 1.], 'frameU': [0., 10.], 'planeV': 0.,
            'views': [{'image': 'fixture-image'}]}


def _reading(bay):
    return {'use': 0,
            'materials': [{'family': 'render', 'hex': '#aaaaaa',
                           'x0': 0., 'x1': 10., 'z0': 0., 'z1': 10.}],
            'bays': [bay]}


def _bay():
    return {'x0': 1., 'x1': 5., 'z0': 0., 'z1': 8., 'hex': '#cccccc',
            'family': 'render', 'shape': 'angled'}


def test_observed_material_zones_normalize_metric_bounds_and_material_pattern():
    bay = _bay()
    bay['materialZones'] = [{
        'x0': 1.5, 'x1': 3.5, 'z0': 2., 'z1': 4.,
        'family': 'timber', 'color': '#AABBCC', 'pattern': 'horizontal_courses',
        'finish': 'weathered', 'unitSizeM': .2,
    }]
    zones, errors = bay_material_zones(bay, lambda x: x / 10, lambda z: z / 10)
    assert errors == []
    assert zones[0]['rect'] == [.15, .2, .35, .4]
    assert zones[0]['family'] == 'timber'
    assert zones[0]['hex'] == '#aabbcc'
    assert zones[0]['pattern'] == 'horizontal_courses'
    assert zones[0]['finish'] == 'weathered'
    assert zones[0]['unitSizeM'] == .2
    assert 'pat=horizontal_courses' in zones[0]['cladding']
    assert 'mz=0.2' in zones[0]['cladding']


def test_adapter_keeps_legacy_bay_without_zones_and_emits_zone_when_present():
    legacy = to_profile(_reading(_bay()), _piece())['bays'][0]
    assert 'claddingZones' not in legacy

    bay = _bay()
    bay['materialZones'] = [{'z0': 4., 'z1': 6., 'family': 'timber',
                             'hex': '#aabbcc', 'pattern': 'scalloped_shingles'}]
    prof = to_profile(_reading(bay), _piece())
    assert prof['bays'][0]['claddingZones'][0]['rect'] == [0.1, .4, .5, .6]
    assert 'pat=scalloped_shingles' in prof['bays'][0]['claddingZones'][0]['cladding']


def test_invalid_or_out_of_bay_zone_remains_unresolved_without_fallback_zone():
    bay = _bay()
    bay['materialZones'] = [
        {'z0': 7., 'z1': 9., 'family': 'timber', 'hex': '#aabbcc'},
        {'z0': 2., 'z1': 4., 'family': 'timber', 'hex': '#aabbcc', 'pattern': 'invented'},
    ]
    prof = to_profile(_reading(bay), _piece())
    assert 'claddingZones' not in prof['bays'][0]
    assert [x['kind'] for x in prof['unresolved']].count('bay_cladding_zone') == 2


def test_scalar_unit_size_requires_pattern_and_stays_unresolved():
    zones, errors = bay_material_zones(
        {**_bay(), 'materialZones': [{'z0': 2., 'z1': 4., 'family': 'timber',
                                       'hex': '#aabbcc', 'unitSizeM': .18}]},
        lambda x: x / 10, lambda z: z / 10)
    assert zones == []
    assert 'requires an explicit pattern' in errors[0]


def test_vertical_board_pitch_uses_horizontal_repeat_axis():
    zones, errors = bay_material_zones(
        {**_bay(), 'materialZones': [{'z0': 2., 'z1': 4., 'family': 'timber',
                                       'hex': '#aabbcc', 'pattern': 'vertical_boards',
                                       'unitSizeM': .12}]},
        lambda x: x / 10, lambda z: z / 10)
    assert errors == []
    assert 'pat=vertical_boards' in zones[0]['cladding']
    assert 'mx=0.12' in zones[0]['cladding']


def test_both_measured_shingle_axes_reach_texture_key():
    zones, errors = bay_material_zones(
        {**_bay(), 'materialZones': [{'z0': 2., 'z1': 4., 'family': 'timber',
            'hex': '#aabbcc', 'pattern': 'scalloped_shingles', 'moduleM': {'x': .14, 'z': .19}}]},
        lambda x: x/10, lambda z: z/10)
    assert errors == []
    assert 'mx=0.14' in zones[0]['cladding'] and 'mz=0.19' in zones[0]['cladding']
