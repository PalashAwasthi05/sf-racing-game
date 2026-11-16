"""Perimeter evidence is local and never inherits a remote deck's height."""
import numpy as np
from shapely.geometry import LineString, box

from pipeline.roof_boundary_support import perimeter_priors, check_silhouettes


def _proof():
    return {'status': 'source-verified', 'sourceSha256': 'a' * 64,
            'metadataSha256': 'b' * 64, 'calibrationSha256': 'c' * 64,
            'cameraXYValidated': False}


def _edge(z=(5.8, 6.2), role='eave'):
    return {'owner': 'host', 'role': role, 'geometry': LineString([(0, 0), (10, 0)]),
            'heightIntervalZ': list(z), 'locationUncertaintyM': .05,
            'confidence': 'medium', 'proof': _proof()}


def _plane(support=None, coefficients=(0, 2, 6)):
    return {'owner': 'host', 'status': 'validated-direct-inliers',
            'directSupport': support if support is not None else box(.2, .2, 9.8, 1.5),
            'plane': list(coefficients), 'heightUncertaintyM': .1,
            'gradientUncertaintyMPerM': .05, 'confidence': 'high'}


def test_internal_raised_deck_never_establishes_remote_outer_eaves():
    xy = np.array([[5., 0], [10, 5], [5, 10], [0, 5], [5, 5]])
    old = np.full(5, 10.)
    deck = _plane(box(4, 4, 6, 6), (0, 0, 10))
    reports = perimeter_priors(xy, old, [1, 1, 1, 1, 0], [0, 0, 0, 0, 1],
                               box(0, 0, 10, 10), 'host', [deck], [_edge()])
    assert reports[0]['preferredZ'] == 6
    assert reports[0]['status'] == 'supported-uncertain-boundary-prior'
    assert reports[0]['exactHardConstraint'] is False
    assert reports[0]['cameraXYValidated'] is False
    assert all(r['preferredZ'] is None for r in reports[1:])
    assert old[4] == 10  # the helper does not mutate deck or any other height
    assert any('too remote' in r['reason'] for r in reports[0]['rejectedEvidence'])


def test_admitted_lower_eave_intervals_allow_watertight_closure_with_fixed_raised_deck():
    import trimesh
    outer = [[0, 0], [10, 0], [10, 10], [0, 10]]
    inner = [[4, 4], [6, 4], [6, 6], [4, 6]]
    vertices = np.array([[*p, 10.] for p in outer + inner] + [[*p, 0.] for p in outer])
    faces = [[4, 5, 6], [4, 6, 7], [8, 10, 9], [8, 11, 10]]
    for i in range(4):
        j = (i + 1) % 4
        faces += [[i, j, 4 + j], [i, 4 + j, 4 + i],
                  [i, 8 + j, j], [i, 8 + i, 8 + j]]
    before = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert before.is_watertight and before.volume > 0
    edges = [{**_edge(), 'geometry': LineString([outer[i], outer[(i + 1) % 4]])} for i in range(4)]
    reports = perimeter_priors(vertices[:, :2], vertices[:, 2], [1] * 4 + [0] * 8,
                               [0] * 4 + [1] * 8, box(0, 0, 10, 10), 'host',
                               [_plane(box(4, 4, 6, 6), (0, 0, 10))], edges)
    after_vertices = vertices.copy()
    for record in reports:
        # A caller can choose the nearest admissible prior value; the interval
        # remains explicit and the deck's fixed samples are never moved.
        interval = record['heightIntervalZ']
        assert record['exactHardConstraint'] is False
        after_vertices[record['vertex'], 2] = np.clip(record['currentZ'], *interval)
    result = trimesh.Trimesh(vertices=after_vertices, faces=faces, process=False)
    assert result.is_watertight and result.volume > 0
    np.testing.assert_array_equal(result.vertices[4:8, 2], [10] * 4)
    np.testing.assert_allclose(result.vertices[:4, 2], [6.2] * 4)
    np.testing.assert_array_equal(result.faces, before.faces)


def test_adjoining_slope_gets_only_bounded_quantified_continuation():
    xy = np.array([[5., 0], [0, 5]])
    reports = perimeter_priors(xy, [10, 10], [1, 1], [0, 0], box(0, 0, 10, 10),
                               'host', [_plane()])
    assert reports[0]['preferredZ'] == 6
    np.testing.assert_allclose(reports[0]['heightIntervalZ'], [5.89, 6.11])
    assert reports[1]['preferredZ'] is None
    plane = _plane(); del plane['gradientUncertaintyMPerM']
    report = perimeter_priors(xy[:1], [10], [1], [0], box(0, 0, 10, 10), 'host', [plane])[0]
    assert report['preferredZ'] is None
    assert 'slope uncertainty' in report['rejectedEvidence'][0]['reason']


def test_fixed_roof_anchor_survives_conflicting_source_lower_eave():
    report = perimeter_priors([[5, 0]], [10], [1], [1], box(0, 0, 10, 10),
                              'host', source_edges=[_edge()])[0]
    assert report['status'] == 'fixed-anchor-preserved'
    assert report['preferredZ'] == 10
    assert report['conflictingEvidenceCount'] == 1


def test_visible_roof_edge_cannot_become_hidden_wall_eave_or_lower_native_roof():
    report = perimeter_priors([[5, 0]], [10], [1], [0], box(0, 0, 10, 10),
                              'host', source_edges=[_edge(role='visible-roof-edge')])[0]
    assert report['preferredZ'] is None
    assert 'wall/roof junction' in report['rejectedEvidence'][0]['reason']


def test_disagreeing_local_planes_are_not_averaged_into_a_fake_roof_step():
    planes = [_plane(coefficients=(0, 0, 6)), _plane(coefficients=(0, 0, 10))]
    report = perimeter_priors([[5, 0]], [10], [1], [0], box(0, 0, 10, 10), 'host', planes)[0]
    assert report['status'] == 'conflicting-boundary-evidence'
    assert report['preferredZ'] is None and report['currentZ'] == 10


def test_wrong_owner_and_unverified_source_do_not_establish_boundary():
    plane = _plane(); plane['owner'] = 'neighbor'
    edge = _edge(); edge['proof']['sourceSha256'] = 'stale'
    report = perimeter_priors([[5, 0]], [10], [1], [0], box(0, 0, 10, 10), 'host', [plane], [edge])[0]
    assert not report['evidence'] and report['preferredZ'] is None


def _skyline():
    return {'cameraXYZ': [0, 0, 0], 'basisRightDownForward': np.eye(3).tolist(),
            'focalPx': 100., 'sizePx': [400, 300], 'pointsPx': [[100, 100], [300, 100]],
            'uncertaintyPx': 2., 'skyAboveConfirmed': True, 'proof': _proof()}


def test_actual_skyline_checks_reject_only_verified_sky_and_keep_occlusion_unknown():
    # Pixel y=80 is above the observed skyline; y=120 is below it.
    source = _skyline()
    points = [[.005, -.695, 1.], [.005, -.295, 1.]]
    reports = check_silhouettes(points, [source])
    assert reports[0]['status'] == 'conditional-skyline-conflict'
    assert reports[0]['movesMeasuredAnchors'] is False
    assert reports[1]['status'] == 'not-rejected-by-skyline'
    assert reports[1]['qualifiesRoofHeight'] is False
    source['occludedXIntervalsPx'] = [[195, 205]]
    reports = check_silhouettes(points, [source])
    assert all(r['status'] == 'unobserved' for r in reports)
    assert all('occluded' in r['reason'] for r in reports)


def test_unknown_source_or_out_of_crop_is_not_counted_as_visual_agreement():
    source = _skyline()
    assert check_silhouettes([[9, 0, 1]], [source])[0]['status'] == 'unobserved'
    source['skyAboveConfirmed'] = False
    assert check_silhouettes([[0, 0, 1]], [source])[0]['status'] == 'unresolved-source'
