from regen.reading_coverage import audit_reading, CATEGORIES
from regen.specs import clean, to_profile
from pipeline.material_recipes import parse_observed_key
import pytest


def codes(report):
    return {x['code'] for x in report['issues']}


def piece():
    return {'lengthM': 8, 'heightM': 10, 'building': 'test-part', 'id': 'test-side',
            'views': [{'image': 'test-photo'}], 'normal': [0, 1], 'frameU': [0, 8], 'planeV': 0}


def test_porch_does_not_fabricate_entry_but_missing_access_is_reported():
    raw = {'porches': [{'x0': 2, 'x1': 4, 'floorM': 1, 'topM': 4}], 'openings': []}
    assert 'porch_access_unread' in codes(audit_reading(raw))
    profile = to_profile(clean(raw, piece()), piece())
    assert profile['openings'] == []
    assert any(x.get('code') == 'porch_access_unread' for x in profile['unresolved'])


def test_unmodelled_description_survives_compilation():
    observation = {'what': 'scalloped wall shingles', 'where': 'upper wall', 'why': 'unsupported pattern'}
    profile = to_profile(clean({'unmodelled': [observation]}, piece()), piece())
    assert {'kind': 'source-unmodelled', 'path': 'unmodelled[0]', 'observation': observation} in profile['unresolved']
    assert profile['readingCoverage']['status'] == 'needs_reading'


def test_inventory_entry_cannot_point_to_garage():
    raw = {'openings': [{'kind': 'garage'}], 'inventory': [
        {'category': 'entrance', 'visibility': 'visible', 'sourceImages': ['photo'],
         'targets': [{'collection': 'openings', 'index': 0}]}]}
    assert 'target_semantics_mismatch' in codes(audit_reading(raw))


def test_occluded_entry_is_unknown_not_a_request_for_invented_door():
    raw = {'porches': [{}], 'inventory': [{'category': 'entrance', 'visibility': 'occluded',
            'sourceImages': ['photo'], 'reason': 'closed security screen blocks the back wall'}]}
    assert 'porch_access_unread' not in codes(audit_reading(raw))


def test_complete_internal_accounting_is_not_image_verification():
    raw = {'inventory': [{'category': c, 'visibility': 'absent', 'sourceImages': ['photo']} for c in CATEGORIES]}
    report = audit_reading(raw)
    assert report['status'] == 'accounted_not_image_verified'
    assert report['imageVerified'] is False


def test_cleaned_away_feature_does_not_pass_coverage():
    raw = {'inventory': [{'category': 'entrance', 'visibility': 'visible', 'sourceImages': ['photo'],
                         'targets': [{'collection': 'openings', 'index': 0}]}],
           'openings': [{'kind': 'entry', 'x0': 4, 'x1': 2, 'z0': 0, 'z1': 2.3, 'door': {'kind': 'panel'}}]}
    profile = to_profile(clean(raw, piece()), piece())
    assert 'target_missing' in codes(profile['readingCoverage'])


def test_material_color_and_entry_frame_do_not_count_as_complete_structure():
    raw = {'materials': [{'family': 'timber', 'hex': '#eeeeee'}],
           'openings': [{'kind': 'entry', 'frame': '#333333'}]}
    assert {'surface_structure_unread', 'door_structure_unread'} <= codes(audit_reading(raw))


def test_observed_bay_structure_and_unread_door_panels_survive_adapter():
    raw = {'bays': [{'x0': 4, 'x1': 7, 'z0': 2, 'z1': 5, 'family': 'timber',
                    'hex': '#eeeeee', 'pattern': 'vertical_boards'}],
           'openings': [{'x0': 1, 'x1': 2, 'z0': 1, 'z1': 3.1, 'kind': 'entry',
                         'door': {'kind': 'panel', 'panels': []}}]}
    profile = to_profile(clean(raw, piece()), piece())
    assert parse_observed_key(profile['bays'][0]['cladding'])['pattern'] == 'vertical_boards'
    assert next(o for o in profile['openings'] if o['style']['kind'] == 'entry')['style']['door']['panels'] == []


def test_invalid_door_panel_bounds_are_not_silently_clamped():
    raw = {'openings': [{'x0': 1, 'x1': 2, 'z0': 1, 'z1': 3.1, 'kind': 'entry',
                         'door': {'kind': 'panel', 'panels': [[-.3, 0, 1, 1]]}}]}
    with pytest.raises(ValueError, match='leaf-relative'):
        to_profile(clean(raw, piece()), piece())
