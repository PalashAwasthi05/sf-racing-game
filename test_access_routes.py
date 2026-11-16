import pytest
from shapely.geometry import Polygon

from pipeline.access_routes import compile_access_routes, route_layout, draw_route
from pipeline.facade_detail import route_serves
from regen.specs import clean, to_profile


def route():
    return {'status': 'observed', 'continuation': 'occluded',
            'serves': {'collection': 'porches', 'index': 0}, 'segments': [
        {'kind': 'flight', 'from': [2.4, 3.3, .2], 'to': [2.4, 1.9, 2.2],
         'steps': 7, 'widthM': 1.7, 'material': '#9a8b70'},
        {'kind': 'landing', 'outline': [[1.7, 2.2], [3.1, 2.2], [3.1, 3.0], [1.7, 3.0]],
         'zM': 1.9, 'thicknessM': .12},
        {'kind': 'flight', 'from': [3.1, 1.9, 2.6], 'to': [4.7, .2, 2.6],
         'steps': 8, 'widthM': 1.2},
    ]}


def test_compile_keeps_explicit_turns_heights_and_out_of_wall_x():
    compiled, errors = compile_access_routes([route()])
    assert errors == []
    result = compiled[0]
    assert result['status'] == 'observed'
    assert result['continuation'] == 'occluded'
    assert result['segments'][0]['from'] == [2.4, 3.3, .2]
    assert result['segments'][1]['outline'][0] == [1.7, 2.2]
    # The second flight leaves the nominal wall interval; it remains measured
    # in the same facade frame rather than being clipped to a frontage box.
    assert result['segments'][2]['to'][0] == 4.7


def test_closed_flights_are_connected_core_and_landing_uses_measured_height():
    compiled, errors = compile_access_routes([route()])
    assert not errors
    layout = route_layout(compiled[0])
    cores = [part for part in layout if part['kind'] == 'flight-core']
    assert len(cores) == 2
    assert all(Polygon(part['crossSection']).is_valid and Polygon(part['crossSection']).area > 0 for part in cores)
    assert sum(part['kind'] == 'tread' for part in layout) == 0
    landing = next(part for part in layout if part['kind'] == 'landing')
    assert landing['topZ'] == pytest.approx(1.9)
    assert landing['bottomZ'] == pytest.approx(1.78)
    first = Polygon(cores[0]['crossSection'])
    assert first.bounds == pytest.approx((0., 1.8, 2.0, 3.3))
    assert cores[0]['widthM'] == pytest.approx(1.7)
    emitted = []
    draw_route(compiled[0], lambda x, z, d: (x, d, z),
               lambda mesh, key, material: emitted.append(mesh))
    flight_mesh = emitted[0]
    assert flight_mesh.is_watertight and flight_mesh.body_count == 1


def test_open_risers_require_and_use_measured_tread_thickness():
    raw = {'status': 'observed', 'serves': {'collection': 'openings', 'index': 2},
           'segments': [{'kind': 'flight', 'from': [1., 2., 0.], 'to': [1., .8, 1.8],
                         'steps': 6, 'widthM': 1., 'construction': 'open_risers', 'treadThicknessM': .08}]}
    compiled, errors = compile_access_routes([raw])
    assert not errors
    layout = route_layout(compiled[0])
    assert len(layout) == 6
    assert all(part['topZ'] - part['bottomZ'] == pytest.approx(.08) for part in layout)


def test_unresolved_or_invalid_routes_emit_no_guessed_geometry():
    unresolved, errors = compile_access_routes([{
        'status': 'unresolved', 'serves': {'collection': 'porches', 'index': 0},
        'note': 'visible flight ends at a landing whose measured elevation is unknown'}])
    assert unresolved[0]['status'] == 'unresolved'
    assert route_layout(unresolved[0]) == []
    assert errors == []

    invalid, errors = compile_access_routes([{
        'status': 'observed', 'serves': {'collection': 'porches', 'index': 0},
        'segments': [{'kind': 'flight', 'from': [0, 2, 0],
                                            'steps': 7, 'widthM': 1.2}]}])
    assert invalid[0]['status'] == 'unresolved'
    assert invalid[0]['segments'] == []
    assert errors and 'from/to' in errors[0][1]


def test_camera_remapped_route_requires_fresh_depth_aware_source_coordinates():
    compiled, errors = compile_access_routes([{
        'status': 'observed', 'serves': {'collection': 'porches', 'index': 0},
        'needsFreshSourceCoordinates': True,
        'segments': [{'kind': 'flight', 'from': [0, 2, .2], 'to': [0, 0, 2.],
                     'steps': 8, 'widthM': 1.2}],
    }])
    assert errors == []
    assert compiled[0]['status'] == 'unresolved'
    assert compiled[0]['needsFreshSourceCoordinates'] is True
    assert route_layout(compiled[0]) == []


def test_remapping_marks_depth_aware_route_stale_without_mutating_source():
    from copy import deepcopy
    from regen.remap_readings import remap

    class AffineMap:
        L = 10.
        def __call__(self, x, z): return x, z
        def stretch(self, x, z): return 1., 1.

    source = {'accessRoutes': [{'status': 'observed',
        'serves': {'collection': 'porches', 'index': 0},
        'segments': [{'kind': 'flight', 'from': [1., 2., .2], 'to': [1., 0., 2.],
                      'steps': 8, 'widthM': 1.2}]}]}
    original = deepcopy(source)
    remapped = remap(source, AffineMap())
    assert source == original
    assert remapped['accessRoutes'][0]['segments'] == original['accessRoutes'][0]['segments']
    assert remapped['accessRoutes'][0]['needsFreshSourceCoordinates'] is True
    compiled, _ = compile_access_routes(remapped['accessRoutes'])
    assert route_layout(compiled[0]) == []


def test_route_extrusion_maps_facade_x_depth_and_vertical_axes():
    compiled, errors = compile_access_routes([{
        'status': 'observed',
        'serves': {'collection': 'openings', 'index': 0},
        'segments': [{'kind': 'landing', 'outline': [[-1, 2], [1, 2], [1, 3], [-1, 3]],
                      'zM': 1., 'thicknessM': .2}],
    }])
    assert not errors
    emitted = []
    def point(x, z, d): return (10. + x, 20. + d, 30. + z)
    count = draw_route(compiled[0], point, lambda mesh, key, material: emitted.append((mesh, key, material)))
    assert count == 1 and len(emitted) == 1
    mesh, key, material = emitted[0]
    assert key == 'detail' and material == '#817d76'
    assert mesh.bounds[0] == pytest.approx([9., 22., 30.8])
    assert mesh.bounds[1] == pytest.approx([11., 23., 31.])


def test_reading_compiler_keeps_unresolved_route_without_fabricating_geometry():
    reading = {'id': 'test-side', 'use': 0, 'views': [{'view': 0, 'usable': True, 'image': 'photo'}],
               'thisBuilding': True, 'wallTopM': 4.,
               'materials': [{'x0': 0., 'x1': 10., 'z0': 0., 'z1': 4., 'family': 'render', 'hex': '#aaaaaa'}],
               'banks': [], 'openings': [], 'bays': [],
               'porches': [{'x0': 1., 'x1': 4., 'topM': 3., 'floorM': 1., 'depthM': 1., 'steps': True}],
               'accessRoutes': [{'status': 'unresolved', 'serves': {'collection': 'porches', 'index': 0},
                                 'note': 'visible upper flight ends at an unmeasured terrace; continuation hidden'}]}
    piece = {'id': 'test-side', 'building': 'test-building', 'lengthM': 10., 'heightM': 4.,
             'views': [{'image': 'photo'}], 'normal': [0., 1.], 'frameU': [0., 10.], 'planeV': 0.}
    profile = to_profile(clean(reading, piece), piece)
    assert profile['accessRoutes'][0]['status'] == 'unresolved'
    assert profile['accessRoutes'][0]['serves'] == [{'collection': 'porches', 'index': 0}]
    assert profile['porches'][0]['sourceIndex'] == 0
    assert profile['accessRoutes'][0]['segments'] == []
    assert any(u.get('kind') == 'access_route' and 'unmeasured terrace' in u.get('reason', '')
               for u in profile['unresolved'])


def test_compiler_keeps_raw_opening_index_after_filtering_malformed_entries():
    reading = {'id': 'test-side', 'use': 0, 'views': [{'view': 0, 'usable': True, 'image': 'photo'}],
               'thisBuilding': True, 'wallTopM': 4.,
               'materials': [{'x0': 0., 'x1': 10., 'z0': 0., 'z1': 4., 'family': 'render', 'hex': '#aaaaaa'}],
               'banks': [], 'openings': [None,
                   {'x0': 1., 'x1': 2., 'z0': .4, 'z1': 2.2, 'kind': 'entry', 'steps': {'count': 6}}],
               'bays': [], 'accessRoutes': [{'status': 'unresolved',
                   'serves': {'collection': 'openings', 'index': 1}, 'note': 'entry route hidden'}]}
    piece = {'id': 'test-side', 'building': 'test-building', 'lengthM': 10., 'heightM': 4.,
             'views': [{'image': 'photo'}], 'normal': [0., 1.], 'frameU': [0., 10.], 'planeV': 0.}
    profile = to_profile(clean(reading, piece), piece)
    assert profile['openings'][0]['sourceIndex'] == 1
    assert profile['accessRoutes'][0]['serves'] == [{'collection': 'openings', 'index': 1}]


def test_route_ownership_is_explicit_and_suppresses_only_its_target():
    routes = [{'serves': [{'collection': 'porches', 'index': 3},
                          {'collection': 'openings', 'index': 1}]}]
    assert route_serves(routes, 'porches', 3)
    assert route_serves(routes, 'openings', 1)
    assert not route_serves(routes, 'porches', 1)
    assert not route_serves(routes, 'openings', 3)
