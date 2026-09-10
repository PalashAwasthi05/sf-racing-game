from general_world.evidence import inspect_record
from general_world.policy import Policy


def _images(grade):
    return {
        str(i): {
            'imageSha256': str(i) * 64,
            'captureDate': '2026-09-01',
            'camera': {
                'providerPoseConsistent': True,
                'worldRegistrationValidated': True,
                'registrationGrade': grade,
            },
            'viewGroup': str(i),
        }
        for i in range(2)
    }


def _record(label):
    return {
        'producer': {'mode': 'automatic', 'inputHash': 'fixture'},
        'sourceIds': ['0', '1'],
        'confidence': 1.0,
        'label': label,
        'support': 'wall',
    }


def test_coarse_registration_admits_a_surface_but_not_an_opening():
    """A 1 m alignment says which wall, not where the window is."""
    images = _images('coarse')
    assert inspect_record(_record('wall'), images, Policy(), '2026-09-09')['accepted']
    for label in ('window', 'door', 'void'):
        verdict = inspect_record(_record(label), images, Policy(), '2026-09-09')
        assert not verdict['accepted']
        assert 'opening-needs-fine-world-registration' in verdict['reasons']


def test_fine_registration_admits_an_opening():
    images = _images('fine')
    for label in ('window', 'door', 'void'):
        assert inspect_record(_record(label), images, Policy(), '2026-09-09')['accepted']


def test_a_source_without_a_grade_is_treated_as_coarse_not_fine():
    """Sources admitted under the old boolean must not inherit fine accuracy."""
    images = _images('coarse')
    for image in images.values():
        del image['camera']['registrationGrade']
    assert inspect_record(_record('wall'), images, Policy(), '2026-09-09')['accepted']
    verdict = inspect_record(_record('window'), images, Policy(), '2026-09-09')
    assert not verdict['accepted']
    assert 'opening-needs-fine-world-registration' in verdict['reasons']


def test_unregistered_sources_still_admit_nothing():
    images = _images('coarse')
    for image in images.values():
        image['camera']['worldRegistrationValidated'] = False
    assert not inspect_record(_record('wall'), images, Policy(), '2026-09-09')['accepted']
