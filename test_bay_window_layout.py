import math
import pytest

from pipeline.bays import faces
from regen.specs import _style, lay_bay_windows
from pipeline.bays import draw


def reading(shape, **overrides):
    bay = {
        'x0': 0., 'x1': 3., 'z0': 1., 'z1': 5., 'projectionM': .7,
        'shape': shape, 'sideAngleDeg': 45., 'rows': [[2., 3.]],
        'front': {'count': 2, 'widthM': 1.2},
        'side': {'widthM': 2.},
    }
    bay.update(overrides)
    return {'bays': [bay], 'openings': []}


@pytest.mark.parametrize('shape, front_bounds', [
    ('angled', (.7, 2.3)),
    ('square', (0., 3.)),
    ('curved', (0., 3.)),
])
def test_bay_windows_stay_inside_front_and_keep_count_with_impossible_width(shape, front_bounds):
    out = lay_bay_windows(reading(shape))
    panes = sorted((o for o in out['openings'] if o.get('bayWindow')), key=lambda o: o['x0'])
    # Angled bay also has two side panes, whose x intervals remain on their cheek faces.
    front = [o for o in panes if o['x0'] >= front_bounds[0] - 1e-9 and o['x1'] <= front_bounds[1] + 1e-9]
    assert len(front) == 2
    assert all(front_bounds[0] <= o['x0'] < o['x1'] <= front_bounds[1] for o in front)
    assert front[0]['x1'] < front[1]['x0']
    assert front[0]['x0'] == pytest.approx(front_bounds[0] + (front[1]['x0'] - front[0]['x1']))
    if shape == 'angled':
        cheeks = [o for o in panes if o not in front]
        assert len(cheeks) == 2
        assert all(0. <= o['x0'] < o['x1'] <= .7 for o in cheeks[:1])
        assert all(2.3 <= o['x0'] < o['x1'] <= 3. for o in cheeks[1:])


def test_normal_front_width_preserves_requested_count_and_width_with_positive_gaps():
    out = lay_bay_windows(reading('angled', front={'count': 3, 'widthM': .25}, side=None))
    panes = sorted((o for o in out['openings'] if o.get('bayWindow')), key=lambda o: o['x0'])
    assert len(panes) == 3
    assert all(o['x1'] - o['x0'] == pytest.approx(.25) for o in panes)
    assert all(a['x1'] < b['x0'] for a, b in zip(panes, panes[1:]))


def test_curved_bay_panes_survive_actual_facet_footprint_mapping():
    bay = {
        'x0': 5.85, 'x1': 8.4, 'z0': 3.8, 'z1': 10., 'projectionM': .95,
        'shape': 'curved', 'rows': [[4.55, 6.3], [7.85, 9.7]],
        'front': {'count': 3, 'widthM': .7}, 'side': None,
    }
    out = lay_bay_windows({'bays': [bay], 'openings': []})
    panes = sorted((o for o in out['openings'] if o.get('bayWindow')), key=lambda o: (o['z0'], o['x0']))
    assert len(panes) == 6
    for row in ([o for o in panes if o['z0'] == 4.55], [o for o in panes if o['z0'] == 7.85]):
        assert len(row) == 3
        assert all(o['x1'] - o['x0'] == pytest.approx(.7) for o in row)
        assert all(a['x1'] < b['x0'] for a, b in zip(row, row[1:]))

    # The adaptive curved shell has many small facets. Each opening remains
    # visible across their union instead of needing to fit one facet alone.
    _, bay_faces = faces(bay['x0'], bay['x1'], bay['projectionM'], 'curved')
    for op in panes:
        visible_width = 0.
        ox0, _, ox1, _ = op['x0'], op['z0'], op['x1'], op['z1']
        for xa, da, xb, db in bay_faces:
            length = math.hypot(xb - xa, db - da)
            fx = (xb - xa) / length
            a, b = max(ox0, min(xa, xb)), min(ox1, max(xa, xb))
            if abs(fx) < 1e-6 or b - a <= 0: continue
            s1, s2 = sorted(((a - xa) / fx, (b - xa) / fx))
            visible_width += max(0., s2 - s1)
        assert visible_width >= .3, op


def test_bay_layout_preserves_only_explicit_pane_divisions():
    measured = reading(
        'square', front={'count': 1, 'widthM': 1.1}, side=None,
        sash='double-hung', lites=[2, 1], meetingRail=.62,
        mullions=1, transoms=2,
    )
    panes = [o for o in lay_bay_windows(measured)['openings'] if o.get('bayWindow')]
    assert len(panes) == 1
    pane = panes[0]
    assert pane['lites'] == [2, 1]
    assert pane['mullions'] == 1 and pane['transoms'] == 2
    style = _style(pane)
    assert style['lites'] == [2, 1]
    assert style['mullionFractions'] == [.5]
    assert style['transomFractions'] == [.3333, .6667]

    ordinary = reading('square', front={'count': 1, 'widthM': 1.1}, side=None,
                       sash='double-hung', lites=[1, 1])
    ordinary_pane = next(o for o in lay_bay_windows(ordinary)['openings'] if o.get('bayWindow'))
    ordinary_style = _style(ordinary_pane)
    assert ordinary_style['lites'] == [1, 1]
    assert ordinary_style['mullionFractions'] == []
    assert ordinary_style['transomFractions'] == []


def test_front_and_side_pane_styles_override_shared_bay_style_independently():
    measured = reading(
        'angled',
        front={'count': 1, 'widthM': 1.1, 'lites': [2, 1], 'mullions': 0},
        side={'widthM': .5, 'lites': [1, 1], 'mullions': 0},
        sash='double-hung', lites=[1, 1], meetingRail=.5,
    )
    panes = [o for o in lay_bay_windows(measured)['openings'] if o.get('bayWindow')]
    fronts = [o for o in panes if o['bayFace'] == 'front']
    sides = [o for o in panes if o['bayFace'] == 'side']
    assert len(fronts) == 1 and len(sides) == 2
    assert _style(fronts[0])['lites'] == [2, 1]
    assert all(_style(p)['lites'] == [1, 1] for p in sides)
    assert all(p['sash'] == 'double-hung' for p in panes)


def test_bay_window_head_shape_is_preserved_and_front_can_override_it():
    measured = reading(
        'angled', head='pointed', springRatio=.62,
        front={'count': 1, 'widthM': 1.1, 'head': 'arch', 'springRatio': .8},
        side={'widthM': .5},
    )
    panes = [o for o in lay_bay_windows(measured)['openings'] if o.get('bayWindow')]
    front = next(p for p in panes if p['bayFace'] == 'front')
    side = next(p for p in panes if p['bayFace'] == 'side')
    assert _style(front)['head'] == 'arch'
    assert _style(front)['springRatio'] == pytest.approx(.8)
    assert _style(side)['head'] == 'pointed'
    assert _style(side)['springRatio'] == pytest.approx(.62)


def test_segmental_rise_is_preserved_from_bay_through_compiled_style():
    measured = reading('angled', head='segmental', riseRatio=.31,
                       front={'count': 1, 'widthM': 1.1}, side=None)
    pane = next(o for o in lay_bay_windows(measured)['openings'] if o.get('bayWindow'))
    assert pane['riseRatio'] == pytest.approx(.31)
    style = _style(pane)
    assert style['head'] == 'segmental'
    assert style['riseRatio'] == pytest.approx(.31)


def _draw_curved_openings(openings):
    meshes = []
    count = draw(
        point=lambda x, z, d: (x, d, z),
        put=lambda mesh, key, material: meshes.append((mesh, key, material)),
        layer='wall', detail='detail',
        bay={'x0': 0., 'x1': 3., 'z0': 0., 'z1': 5., 'p': .7, 'shape': 'curved', 'cap': 'none'},
        openings=openings, cladding='brick', trim='frame',
    )
    return count, meshes


def test_curved_wide_window_has_only_its_true_outer_jambs():
    count, meshes = _draw_curved_openings([
        {'rect': (.38, 1., 2.62, 4.), 'style': {'frameMaterial': 'frame', 'glassMaterial': 'glass'}}
    ])
    assert count == 1
    # The window crosses several curved facets. Test actual triangles spanning
    # its mid-height: mesh bounds alone include the separated top/bottom rails.
    jamb_x = []
    for mesh, _, material in meshes:
        if material != 'frame':
            continue
        z = mesh.triangles[:, :, 2]
        crossing = mesh.triangles[(z.min(axis=1) < 2.5) & (z.max(axis=1) > 2.5)]
        if len(crossing):
            jamb_x.extend(crossing[:, :, 0].mean(axis=1))
    assert jamb_x
    assert all(min(abs(x - .38), abs(x - 2.62)) < .12 for x in jamb_x)


def test_three_curved_windows_are_counted_as_openings_not_facet_fragments():
    openings = [
        {'rect': (x, 1., x + .7, 4.), 'style': {'frameMaterial': 'frame', 'glassMaterial': 'glass'}}
        for x in (.15, 1.15, 2.15)
    ]
    count, meshes = _draw_curved_openings(openings)
    assert count == 3
    assert sum(material == 'glass' for _, _, material in meshes) == 3
