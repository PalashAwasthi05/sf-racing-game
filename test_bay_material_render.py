import pytest

from pipeline import bays


def _point(x, z, depth):
    return (x, depth, z)


def _draw(shape, openings, zones):
    emitted = []

    def put(mesh, key, material):
        if key == 'wall' and mesh.bounds[1, 2] - mesh.bounds[0, 2] > .15:
            area = (float(mesh.area) if not mesh.is_watertight
                    else abs(float(mesh.volume)) / bays.FACE_M)
            emitted.append((material, area))

    bay = {'x0': 1., 'x1': 5., 'z0': 0., 'z1': 6., 'p': .8,
           'shape': shape, 'cap': 'none'}
    bays.draw(_point, put, 'wall', 'detail', bay, openings, 'base', 'trim', zones)
    return emitted


@pytest.mark.parametrize('shape', ['square', 'angled', 'curved'])
def test_bay_face_material_zones_cover_opening_subtracted_surface_once(shape):
    zones = [
        {'rect': [1.5, 1., 4.5, 5.5], 'cladding': 'earlier'},
        {'rect': [2.5, .5, 3.5, 5.5], 'cladding': 'later'},
    ]
    opening = {'rect': [2.2, 2., 3.8, 4.], 'style': {'glassMaterial': 'blue_glass'}}

    with_opening = _draw(shape, [opening], zones)
    without_opening = _draw(shape, [], zones)
    plain_opening = _draw(shape, [opening], [])
    plain_without_opening = _draw(shape, [], [])

    assert {mat for mat, _ in with_opening} >= {'base', 'earlier', 'later'}
    # Zone subdivision preserves the same total single-layer area as the
    # unzoned face; the opening removes area in both, so it remains unfilled.
    assert sum(area for _, area in with_opening) == pytest.approx(
        sum(area for _, area in plain_opening), abs=1e-5)
    assert sum(area for _, area in without_opening) == pytest.approx(
        sum(area for _, area in plain_without_opening), abs=1e-5)
    assert sum(area for _, area in with_opening) < sum(area for _, area in without_opening)


def test_square_return_at_zone_boundary_inherits_zone_material():
    # A square bay's left cheek is at x=1. The inclusive boundary assigns it
    # the zone material, while the right cheek at x=5 remains base cladding.
    emitted = _draw('square', [], [{'rect': [1., 1., 2., 5.], 'cladding': 'edge_zone'}])
    assert any(material == 'edge_zone' for material, _ in emitted)
    assert any(material == 'base' for material, _ in emitted)


def test_later_zone_wins_where_it_overlaps_an_earlier_zone():
    emitted = _draw('square', [], [
        {'rect': [1.5, 1., 4.5, 5.5], 'cladding': 'earlier'},
        {'rect': [2.5, .5, 3.5, 5.5], 'cladding': 'later'},
    ])
    later_area = sum(area for material, area in emitted if material == 'later')
    assert later_area == pytest.approx(5.)
