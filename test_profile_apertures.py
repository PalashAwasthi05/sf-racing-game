from regen.profile_apertures import audit_profile


def _profile():
    return {
        'frameU': [10.0, 20.0], 'heightRefM': 12.0,
        'sourceIds': ['mapillary:test'],
        'readingProvenance': {'status': 'ok', 'accepted': True,
            'registeredCropSource': True,
            'sourceReading': {'sha256': 'a' * 64}},
        'openings': [
            {'sourceIndex': 41, 'rect': [0.1, 0.1, 0.2, 0.2], 'style': {'kind': 'window'}},
            {'sourceIndex': 42, 'rect': [0.3, 0.1, 0.4, 0.2], 'style': {'kind': 'window'}},
        ],
        'groups': [{
            'region': [0.1, 0.2, 0.9, 0.8], 'columns': 2, 'rows': 2,
            'xCenters': [0.25, 0.75], 'yCenters': [0.25, 0.75],
            'omit': [[1, 0]],
            'style': {'kind': 'window', 'openingRatio': 0.6,
                      'openingHeightRatio': 0.6},
        }],
    }


def test_measured_group_appends_after_explicit_openings_without_reindexing():
    result, meta = audit_profile(_profile(), source_hash_verified=True, fits=[])
    assert result['openings'][:2] == _profile()['openings']
    assert meta['explicitOpeningCount'] == 2
    assert len(meta['groupOpeningIndices'][0]) == 3
    assert meta['groupOpeningIndices'][0] == [2, 3, 4]
    assert [o['_auditCell'] for o in result['openings'][2:]] == [[0, 0], [0, 1], [1, 1]]
    assert all(o['_receiverPolicy'] == 'preserve-existing-only' for o in result['openings'][2:])


def test_unverified_or_uncertain_groups_cannot_create_openings():
    profile = _profile()
    result, meta = audit_profile(profile, source_hash_verified=False, fits=[])
    assert result['openings'] == profile['openings']
    assert meta['groupOpeningCount'] == 0

    for update in (
        {'xCenters': None},
        {'fitId': 'fitted-bank'},
        {'style': {'kind': 'window', 'ribbon': True}},
    ):
        candidate = _profile()
        candidate['groups'][0].update(update)
        result, meta = audit_profile(candidate, source_hash_verified=True, fits=[])
        assert result['openings'] == candidate['openings']
        assert meta['groupOpeningCount'] == 0
