import numpy as np
from shapely.geometry import Polygon

from pipeline.facade_ownership import (
    area_photo_observations,
    duplicate_window_observations,
    match_profile_pairs,
    match_projected_windows,
    project_profile_windows,
    suppress_pair_features,
    wall_opening_at,
)


def _profile(name, plane, *, recess=False, extra=False, shifted=False):
    shift = .25 if shifted else 0.
    openings = [
        {'rect': [.25 + shift, .20, .40 + shift, .42], 'style': {'kind': 'window', 'head': 'flat'}},
        {'rect': [.60 + shift, .58, .75 + shift, .80], 'style': {'kind': 'window', 'head': 'flat'}},
    ]
    if extra:
        openings.append({'rect': [.82, .20, .92, .42], 'style': {'kind': 'window', 'head': 'flat'}})
    profile = {
        'id': name,
        'normal': [0., -1.],
        'frameU': [-2., 2.],
        'planeV': plane,
        'heightRefM': 4.,
        'reading': {'confidence': 'high', 'wallTopM': 4.},
        'groups': [],
        'openings': openings,
        'recesses': [],
    }
    if recess:
        rect = [.1, .1, .9, .9] if recess is True else [.25, .19, .45, .43]
        # The explicitly glazed rear opening exposes a set-back window.
        back = [dict(opening) for opening in openings if
                rect[0] <= opening['rect'][0] and opening['rect'][2] <= rect[2] and
                rect[1] <= opening['rect'][1] and opening['rect'][3] <= rect[3]]
        for opening in back:
            opening['kind'] = 'window'
        profile['recesses'] = [{'rect': rect, 'depthM': .6, 'backOpenings': back}]
    return profile


def _observations(front, far, *, far_extra=False):
    camera = {
        'imageId': 'pano',
        'center': [0., -100., 2.],
        # Camera right=x, down=-z, forward=+y.
        'rotation': [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
    }
    return (project_profile_windows(front, 0., camera),
            project_profile_windows(far, 0., camera), camera)


def _proof(*, recess=False, far_extra=False):
    front, far = _profile('front', 0., recess=recess), _profile('far', -1., extra=far_extra)
    pa, pb, camera = _observations(front, far)
    footprint = Polygon([(-2., 0.), (2., 0.), (2., 1.), (-2., 1.)])
    result = duplicate_window_observations(front, far, pa, pb, 0., 0., camera,
                                           host_geometry=footprint)
    return front, far, result


def test_backed_glazing_blocks_and_only_matched_rear_windows_are_suppressed():
    front, far, proof = _proof(far_extra=True)
    assert proof['status'] == 'occluded-duplicate'
    assert len(proof['blockedMatches']) == 2
    assert proof['visibleMatches'] == []
    pair = {'a': front, 'b': far, 'result': proof}
    selected, removed, _ = suppress_pair_features(pair, far, selected_plane=-1.)
    assert len(removed) == 2
    assert len(selected['openings']) == 1
    assert selected['openings'][0]['rect'] == [.82, .20, .92, .42]


def test_open_recess_keeps_setback_windows_visible():
    front, far, proof = _proof(recess=True)
    assert proof['status'] == 'setback-visible'
    assert len(proof['visibleMatches']) == 2
    pair = {'a': front, 'b': far, 'result': proof}
    selected, removed, returned = suppress_pair_features(pair, far, selected_plane=-1.)
    assert selected is far
    assert removed == []
    assert returned['status'] == 'setback-visible'


def test_partial_occlusion_suppresses_only_the_blocked_correspondence():
    front, far, proof = _proof(recess='lower', far_extra=True)
    assert proof['status'] == 'setback-visible'
    assert len(proof['blockedMatches']) == 1
    assert len(proof['visibleMatches']) == 1
    pair = {'a': front, 'b': far, 'result': proof}
    selected, removed, returned = suppress_pair_features(pair, far, selected_plane=-1.)
    assert removed == []  # A single blocked light is insufficient evidence to deduplicate.
    assert returned['status'] == 'setback-visible'
    assert len(selected['openings']) == 3


def test_adjacent_repeated_windows_do_not_match_by_nearest_neighbor():
    front = _profile('front', 0.)
    neighbor = _profile('neighbor', 0., shifted=True)
    pa, pb, _ = _observations(front, neighbor)
    assert match_projected_windows(pa, pb) == []


def test_projection_uses_only_the_profile_reading_source_panorama():
    side = {
        'normal': [0., -1.], 'frameU': [-2., 2.], 'planeV': 0.,
        'groundZ': 0., 'views': [{'image': 'pano-a'}, {'image': 'pano-b'}],
    }
    camera = {'center': [0., -100., 2.],
              'rotation': [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]}
    catalog = {
        'sides': {'front': side, 'rear': {**side, 'planeV': -1.}},
        'cameras': {'pano-a': {**camera, 'imageId': 'pano-a'},
                    'pano-b': {**camera, 'imageId': 'pano-b'}},
    }
    front, rear = _profile('front', 0.), _profile('rear', -1.)
    front['sourceIds'] = ['mapillary:pano-a']
    rear['sourceIds'] = ['mapillary:pano-b']
    obs = {front['id']: area_photo_observations(front, catalog),
           rear['id']: area_photo_observations(rear, catalog)}
    assert {x['imageId'] for x in obs['front']['projections']} == {'pano-a'}
    assert {x['imageId'] for x in obs['rear']['projections']} == {'pano-b'}
    # Both side records list both views, but unrelated read sources are not
    # treated as a measured cross-view correspondence.
    assert match_profile_pairs([front, rear], obs) == []


def test_window_is_not_an_open_portal_but_entry_and_recess_are():
    profile = _profile('front', 0.)
    # Opening center in the lower window.
    assert not wall_opening_at(profile, [-.7, 0.], .62, 0.)
    profile['openings'].append({'rect': [.1, .2, .9, .6], 'style': {'kind': 'entry'}})
    assert wall_opening_at(profile, [0., 0.], 1.6, 0.)
