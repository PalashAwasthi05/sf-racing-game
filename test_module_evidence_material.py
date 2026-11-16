import json

from shapely.geometry import box
from shapely.ops import unary_union

from pipeline.facade_grammar import _module_evidence_panels
from pipeline.material_recipes import parse_observed_key


def _profile(width, height, zone, pitch, color, panels=()):
    return {
        'material': f'timber_dark#{color[1:]}',
        'materialEvidence': {'family': 'timber', 'colorRgb': [int(color[i:i+2], 16) for i in (1, 3, 5)]},
        'panels': list(panels),
        'moduleEvidence': {
            'status': 'paired-semantic-review-accepted',
            'pattern': 'horizontal_periodic_boards',
            'axis': 'horizontal_courses',
            'pitchM': pitch,
            'sourceZoneM': zone,
            'material': {'family': 'timber', 'hex': color},
        },
    }


def _metric_rect(rect, width, height):
    return box(rect[0] * width, rect[1] * height, rect[2] * width, rect[3] * height)


def test_module_evidence_is_metric_and_covers_only_unowned_zone_cells():
    # Fixture mirrors the merged r31 profile: a 7 m x 11.2 m main timber
    # zone, with a separately owned upper color band clipped out of it.
    width, height = 8.35, 10.75
    old_panel = {'rect': [.09581, .83721, .93413, .89302], 'material': 'timber_dark#5b6073'}
    profile = _profile(width, height,
                       {'x0': .8, 'x1': 7.8, 'z0': -1.8, 'z1': 9.4},
                       .19975, '#9ea3b8', [old_panel])
    added = _module_evidence_panels(profile, profile['panels'], width, height)

    zone = box(.8, -1.8, 7.8, 9.4)
    existing = _metric_rect(old_panel['rect'], width, height)
    expected = zone.difference(existing)
    actual = unary_union([_metric_rect(p['rect'], width, height) for p in added])
    assert abs(actual.area - expected.area) < 1e-6
    assert actual.symmetric_difference(expected).area < 1e-6
    assert actual.intersection(existing).area < 1e-8
    assert actual.difference(zone).area < 1e-8

    decoded = [parse_observed_key(p['material']) for p in added]
    assert all(d['family'] == 'timber' and d['pattern'] == 'horizontal_courses'
               and abs(d['moduleM']['z'] - .19975) < 1e-9 for d in decoded)


def test_module_evidence_rejects_unowned_material_and_does_not_expand_scope():
    profile = _profile(4.73, 12.16, {'x0': 0, 'x1': 4.7, 'z0': 0, 'z1': 9.9},
                       .21777, '#c9d3da')
    wrong_color = json.loads(json.dumps(profile))
    wrong_color['moduleEvidence']['material']['hex'] = '#ffffff'
    assert _module_evidence_panels(wrong_color, [], 4.73, 12.16) == []
    panels = _module_evidence_panels(profile, [], 4.73, 12.16)
    assert len(panels) == 1
    bounds = _metric_rect(panels[0]['rect'], 4.73, 12.16).bounds
    assert all(abs(a-b) < 1e-9 for a, b in zip(bounds, (0., 0., 4.7, 9.9)))
