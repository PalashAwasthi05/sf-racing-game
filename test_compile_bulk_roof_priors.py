import numpy as np

from regen.compile_bulk_roof_priors import fit_depth_hinge, fit_lidar_front_slope, source_semantic_gate


def profile_with_semantics():
    return {
        'sourceSemanticEvidence': [
            {
                'imageId': 'view-a', 'rectSha256': 'hash-a', 'status': 'accepted',
                'above': 'sloping_roof', 'below': 'wall', 'overlapAmbiguous': False,
                'confidence': .92, 'continuousFraction': .97, 'xRange': [.08, .92],
                'boundaryHeightM': 8.01,
            },
            {
                'imageId': 'view-b', 'rectSha256': 'hash-b', 'status': 'accepted',
                'above': 'sloping_roof', 'below': 'wall', 'overlapAmbiguous': False,
                'confidence': .88, 'continuousFraction': .94, 'xRange': [.09, .91],
                'boundaryHeightM': 8.08,
            },
        ]
    }


def edges():
    return {
        'view-a': {'sha256': 'hash-a', 'heightM': 8.0},
        'view-b': {'sha256': 'hash-b', 'heightM': 8.1},
    }


def test_admits_only_two_hash_matched_continuous_roof_over_wall_views():
    result = source_semantic_gate(profile_with_semantics(), edges(), [.1, .9], 8.05)
    assert result['status'] == 'admitted'
    assert result['verifiedImageIds'] == ['view-a', 'view-b']


def test_broad_roof_edge_without_semantic_surface_labels_stays_proposal():
    result = source_semantic_gate({'roofs': [{'kind': 'mansard'}]}, edges(), [.1, .9], 8.05)
    assert result == {'status': 'unresolved', 'reason': 'source-surface-semantics-unavailable'}


def test_rejects_roof_plane_break_and_balcony_or_bay_overlap_classes():
    for change, reason in [
        ({'above': 'sloping_roof', 'below': 'roof_front'}, 'source-does-not-show-roof-over-wall'),
        ({'overlapAmbiguous': True}, 'source-semantic-overlap-unresolved'),
        ({'continuousFraction': .72}, 'source-semantic-span-not-continuous'),
        ({'xRange': [.3, .7]}, 'source-semantic-span-does-not-cover-prior'),
    ]:
        profile = profile_with_semantics()
        profile['sourceSemanticEvidence'][0].update(change)
        result = source_semantic_gate(profile, edges(), [.1, .9], 8.05)
        assert result['status'] == 'unresolved'
        assert result['reason'] == reason


def test_rejects_stale_crop_hash_or_height_disagreement():
    profile = profile_with_semantics()
    profile['sourceSemanticEvidence'][0]['rectSha256'] = 'stale'
    result = source_semantic_gate(profile, edges(), [.1, .9], 8.05)
    assert result['reason'] == 'source-semantic-crop-hash-mismatch'

    profile = profile_with_semantics()
    profile['sourceSemanticEvidence'][0]['boundaryHeightM'] = 9.0
    result = source_semantic_gate(profile, edges(), [.1, .9], 8.05)
    assert result['reason'] == 'source-semantic-boundary-height-disagrees'


def test_stereo_hinge_requires_receding_roof_over_vertical_wall_and_intersection():
    tracks = []
    for x in np.linspace(0, 10, 13):
        for z in np.linspace(8.3, 10.5, 8):
            tracks.append([x, z, 2.0 - .35 * (z - 8.0)])
        for z in np.linspace(6.0, 7.7, 7):
            tracks.append([x, z, 2.0])
    result = fit_depth_hinge(tracks, [0, 10], 8.0)
    assert result['status'] == 'geometric-hinge-support'
    assert result['roofDepthSlope'] < -.3
    assert abs(result['wallDepthSlope']) < .01
    assert result['junctionGapM'] < .05
    assert result['semanticWallIdentityVerified'] is False


def test_stereo_rejects_parallel_surface_or_roof_break_without_wall_plane():
    assert fit_depth_hinge([], [0, 10], 8.0)['reason'] == 'stereo-support-too-sparse'
    parallel = [[x, z, 2.0] for x in np.linspace(0, 10, 13)
                for z in np.r_[np.linspace(8.3, 10.5, 8), np.linspace(6.0, 7.7, 7)]]
    result = fit_depth_hinge(parallel, [0, 10], 8.0)
    assert result['reason'] == 'above-surface-not-receding-roof-plane'

    two_slopes = []
    for x in np.linspace(0, 10, 13):
        for z in np.linspace(8.3, 10.5, 8):
            two_slopes.append([x, z, 2.0 - .35 * (z - 8.0)])
        for z in np.linspace(6.0, 7.7, 7):
            two_slopes.append([x, z, 3.0 - .2 * (z - 8.0)])
    result = fit_depth_hinge(two_slopes, [0, 10], 8.0)
    assert result['reason'] == 'below-surface-not-vertical-wall-plane'


def test_lidar_eave_support_requires_repeated_tangent_bins_and_candidate_intersection():
    supported = []
    for x in np.arange(.1, 9.9, .25):
        for depth in np.linspace(0, 2.5, 12):
            supported.append([x, depth, 12.0 + 1.1 * depth])
    result = fit_lidar_front_slope(supported, [0, 10], 12.0)
    assert result['status'] == 'supported'
    assert result['supportFraction'] >= .75
    assert all(b['candidateOffsetM'] < .01 for b in result['bins'] if b.get('supported'))

    wrong_height = fit_lidar_front_slope(supported, [0, 10], 13.0)
    assert wrong_height['status'] == 'unresolved'
    assert wrong_height['supportingBins'] == 0

    partial = [p for p in supported if p[0] < 4.0]
    result = fit_lidar_front_slope(partial, [0, 10], 12.0)
    assert result['status'] == 'unresolved'
