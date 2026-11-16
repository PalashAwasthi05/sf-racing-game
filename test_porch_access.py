from regen.specs import porch_access_profile
from pipeline.facade_detail import porch_access_layout, entrance_access_layout, porch_for_entry
from shapely.geometry import box
from shapely.ops import unary_union


def test_legacy_porch_keeps_unrailed_default_and_absent_access_draws_nothing():
    porch = {'_x0': 0.0, '_x1': 4.0, 'floorM': 1.2, 'depthM': 1.0, 'steps': True}
    layout = porch_access_layout(porch, lambda x, d: 0.0)
    assert layout['status'] == 'ok'
    assert layout['treads']
    assert layout['rails'] == []
    assert layout['posts'] == []

    no_access = {**porch, 'steps': False}
    assert porch_access_layout(no_access, lambda x, d: 0.0) == {
        'status': 'none', 'treads': [], 'rails': [], 'posts': []}


def test_observed_access_maps_explicit_step_dimensions_rails_and_posts():
    raw = {'access': {
        'status': 'observed',
        'steps': {'present': True, 'count': 4, 'widthM': 1.4, 'runM': 1.2,
                  'centerOffsetM': 0.2, 'hex': '#777777'},
        'rails': {'sides': ['left', 'right'], 'heightM': 0.9, 'hex': '#333333'},
        'posts': [{'side': 'left', 'position': 0.0, 'kind': 'newel', 'heightM': 1.0,
                   'widthM': 0.22, 'finial': True}],
    }}
    access, error = porch_access_profile(raw, lambda x: x / 4)
    assert error is None
    assert access['steps'] == {'present': True, 'count': 4, 'widthM': 1.4, 'runM': 1.2,
                               'centerOffsetM': 0.2, 'material': '#777777'}
    assert access['rails']['sides'] == ['left', 'right']
    assert access['posts'][0]['kind'] == 'newel'
    assert access['posts'][0]['finial'] is True


def test_occluded_or_incomplete_access_is_unresolved_without_stairs():
    occluded, reason = porch_access_profile({'access': {'status': 'occluded'}}, lambda x: x)
    assert occluded == {'status': 'unresolved'}
    assert 'occluded' in reason

    incomplete, reason = porch_access_profile({'access': {'status': 'observed', 'steps': {'present': True}}}, lambda x: x)
    assert incomplete == {'status': 'unresolved'}
    assert 'count, widthM, runM, and centerOffsetM' in reason

    layout = porch_access_layout({'_x0': 0, '_x1': 4, 'floorM': 1.0, 'depthM': 1.0,
                                  'steps': False, 'access': occluded}, lambda x, d: 0.0)
    assert layout['status'] == 'unresolved'
    assert layout['treads'] == []
    assert layout['rails'] == []


def test_stair_courses_form_supported_connected_mass_and_posts_land_on_tread():
    porch = {
        '_x0': 0.0, '_x1': 4.0, 'floorM': 1.4, 'depthM': 1.0, 'steps': True,
        'access': {'status': 'observed',
                   'steps': {'present': True, 'count': 4, 'widthM': 1.2,
                             'runM': 1.6, 'centerOffsetM': -0.2, 'material': '#777777'},
                   'rails': {'sides': ['left', 'right'], 'heightM': 0.9, 'material': '#333333'},
                   'posts': [{'side': 'left', 'position': 0.0, 'kind': 'newel', 'heightM': 1.0,
                              'widthM': 0.2, 'material': '#eeeeee'}]},
    }
    ground = lambda x, d: .04 * x + .1 * d
    layout = porch_access_layout(porch, ground)
    treads = layout['treads']
    assert len(treads) == 4
    assert treads[0]['depth0'] == porch['depthM']
    assert treads[-1]['depth1'] == porch['depthM'] + 1.6
    assert abs(treads[0]['topZ'] - porch['floorM']) < 1e-9
    stair_center = (porch['_x0'] + porch['_x1']) / 2 - .2
    assert abs(treads[-1]['topZ'] - ground(stair_center, porch['depthM'] + 1.6)) < 1e-9
    # The solid course bottoms are below all sampled terrain points, so the
    # stair mass is buried into its support rather than hovering above it.
    assert all(t['bottomZ'] < min(t['groundSamples']) for t in treads)
    assert all(t['bottomZ'] < ground(x, d)
               for t in treads for x in (t['x0'], (t['x0'] + t['x1']) / 2, t['x1'])
               for d in (t['depth0'], (t['depth0'] + t['depth1']) / 2, t['depth1']))
    for left, right in zip(treads, treads[1:]):
        assert abs(left['depth1'] - right['depth0']) < 1e-9
        assert min(left['topZ'], right['topZ']) > max(left['bottomZ'], right['bottomZ'])

    # In a depth/elevation section, the solid-course union has one component;
    # adjacent blocks share a nonzero vertical face, not only an edge.
    section = [box(t['depth0'], t['bottomZ'], t['depth1'], t['topZ']) for t in treads]
    stair_mass = unary_union(section)
    assert stair_mass.geom_type == 'Polygon'
    assert stair_mass.is_valid
    assert stair_mass.area > sum(p.area for p in section) * .75

    assert {r['side'] for r in layout['rails']} == {'left', 'right'}
    assert layout['rails'][0]['a'][2] == porch['depthM']
    assert layout['rails'][0]['b'][2] == porch['depthM'] + 1.6
    assert all(r['a'][0] <= treads[0]['x0'] or r['a'][0] >= treads[0]['x1'] for r in layout['rails'])
    assert layout['posts'][0]['depth'] == porch['depthM']
    assert layout['posts'][0]['baseZ'] == porch['floorM']


def test_mid_stair_post_uses_actual_tread_level_not_interpolated_slope():
    porch = {'_x0': 0., '_x1': 4., 'floorM': 1.2, 'depthM': 1., 'steps': False,
             'access': {'status': 'observed',
                        'steps': {'present': True, 'count': 4, 'widthM': 1., 'runM': 1.2,
                                  'centerOffsetM': 0.},
                        'posts': [{'side': 'right', 'position': .5, 'heightM': .8,
                                   'widthM': .15}]}}
    layout = porch_access_layout(porch, lambda x, d: 0.)
    # position .5 falls at the third course for four treads, whose top is a
    # real horizontal surface, not the continuously interpolated slope value.
    assert layout['posts'][0]['baseZ'] == layout['treads'][2]['topZ']


def test_ground_end_post_uses_terrain_at_its_own_side_position():
    porch = {'_x0': 0., '_x1': 4., 'floorM': 1.2, 'depthM': 1., 'steps': False,
             'access': {'status': 'observed',
                        'steps': {'present': True, 'count': 3, 'widthM': 1., 'runM': 1.2,
                                  'centerOffsetM': 0.},
                        'posts': [{'side': 'right', 'position': 1., 'heightM': .8,
                                   'widthM': .15}]}}
    ground = lambda x, d: .1 * d + .05 * x
    layout = porch_access_layout(porch, ground)
    post = layout['posts'][0]
    assert post['baseZ'] == ground(post['x'], post['depth'])


def test_explicit_zero_steps_stays_absent_through_compiler_and_renderer():
    from regen.specs import _style
    style = _style({'kind': 'entry', 'steps': {'count': 0, 'widthM': 1.2}})
    assert style['steps']['count'] == 0
    # A high threshold must not turn an explicit zero into a guessed staircase.
    assert entrance_access_layout(style, [1., 4., 2., 6.], lambda x, d: 0.)['treads'] == []
    style['steps']['count'] = None
    assert entrance_access_layout(style, [1., 4., 2., 6.], lambda x, d: 0.)['treads'] == []


def test_door_on_porch_does_not_generate_a_second_staircase():
    porch = {'x': [.2, .6], 'floorM': 2.6, 'depthM': 1.5, 'steps': True}
    rect = [2.5, 2.7, 4., 5.]
    assert porch_for_entry([porch], rect, 10.) is porch
    style = {'steps': {'count': 14, 'widthM': 2.4}}
    result = entrance_access_layout(style, rect, lambda x, d: 0., [porch], 10.)
    assert result['status'] == 'owned-by-porch'
    assert result['treads'] == []
    assert porch_for_entry([porch], [2.5, -.5, 4., 1.8], 10.) is None
    assert porch_for_entry([porch], [7., 2.6, 8., 5.], 10.) is None


def test_hillside_entry_below_building_datum_still_has_its_observed_steps():
    style = {'steps': {'count': 3, 'widthM': 1.4, 'runM': .9}}
    result = entrance_access_layout(style, [1., -2.95, 2.4, -.7], lambda x, d: -3.5)
    assert len(result['treads']) == 3
    assert result['treads'][0]['topZ'] == -2.95
    assert result['treads'][-1]['topZ'] == -3.5
    assert result['rails'] == []  # a count alone never proves rails


def test_observed_stair_landing_is_not_forced_down_to_the_street_datum():
    style = {'steps': {'count': 7, 'widthM': 1.4, 'runM': 2.1, 'bottomM': 2.1}}
    result = entrance_access_layout(style, [1., 3.3, 2.4, 5.6], lambda x, d: -1.4)
    assert result['treads'][-1]['topZ'] == 2.1


def test_inferred_flight_must_not_cross_the_road_even_on_a_steep_lot():
    from pipeline.access_constraints import carriageway_limit
    import numpy as np
    point = lambda x, z, d: np.array([x, -d, z])
    road = box(-20., -15., 20., -7.)
    limit = carriageway_limit(point, road, 2., 1.5, 1.8)
    assert limit == 7.
    porch = {'_x0': 0., '_x1': 4., 'floorM': 1.37, 'depthM': 1.8,
             'steps': True, '_accessLimitM': limit}
    layout = porch_access_layout(porch, lambda x, d: -4.5)
    assert layout['status'] == 'unresolved'
    assert layout['treads'] == []
    assert layout['proposedEndDepthM'] > limit
    assert carriageway_limit(point, box(10., -15., 20., -7.), 2., 1.5, 1.8) is None


def test_porch_compiler_preserves_measured_hill_heights_and_zero_posts():
    from regen.specs import to_profile
    piece = {'id': 'synthetic-side', 'building': 'synthetic', 'lengthM': 10., 'heightM': 14.,
             'normal': [0., -1.], 'frameU': [0., 10.], 'planeV': 0., 'views': [{'image': 'test'}]}
    reading = {'use': 0, 'banks': [], 'openings': [], 'porches': [
        {'x0': 1., 'x1': 4., 'floorM': 4.8, 'topM': 10.2, 'depthM': 1.6, 'columns': 0, 'steps': False},
        {'x0': 5., 'x1': 8., 'floorM': -1.5, 'topM': 1.6, 'depthM': 1.2, 'columns': 2, 'steps': False}]}
    result = to_profile(reading, piece)['porches']
    assert result[0]['floorM'] == 4.8
    assert result[0]['topM'] == 10.2
    assert result[0]['columns'] == 0
    assert result[1]['floorM'] == -1.5
