from shapely.geometry import Polygon, MultiPolygon

from pipeline import facades, facade_grammar
from pipeline.facades import profile_plane_binding, choose_profile_for_edge


def _profile(**updates):
    return {
        'normal': [0.0, -1.0],
        'planeV': 0.0,
        'frameU': [0.0, 10.0],
        'bays': [],
        **updates,
    }


def test_profile_choice_prefers_metric_host_plane_independent_of_input_order():
    far = _profile(id='far-source', planeV=1.8)
    near = _profile(id='near-source', planeV=0.0)
    for candidates in ([far, near], [near, far]):
        assert choose_profile_for_edge(candidates, (0, 0), (10, 0), (0, -1))['id'] == 'near-source'
    # On the other physical plane, the profile measured there wins instead.
    for candidates in ([far, near], [near, far]):
        assert choose_profile_for_edge(candidates, (0, -1.8), (10, -1.8), (0, -1))['id'] == 'far-source'


def test_profile_choice_applies_coverage_and_angular_guards_before_plane_priority():
    incidental = _profile(id='incidental-exact', planeV=0.0, frameU=[1.3, 10.0])
    supported = _profile(id='supported-offset', planeV=1.0, frameU=[0.0, 10.0])
    # The exact-plane candidate covers less than half of this 2 m split edge.
    assert choose_profile_for_edge([incidental, supported], (0, 0), (2, 0), (0, -1))['id'] == 'supported-offset'
    misaligned = _profile(id='misaligned', planeV=0.0, normal=[0.35, -0.936])
    assert choose_profile_for_edge([misaligned, supported], (0, 0), (10, 0), (0, -1))['id'] == 'supported-offset'


def test_unframed_profile_keeps_legacy_precedence():
    legacy = {'id': 'legacy-unframed', 'normal': [1.0, 0.0]}
    framed = _profile(id='framed-exact', planeV=0.0)
    assert choose_profile_for_edge([legacy, framed], (0, 0), (10, 0), (0, -1)) is legacy


def test_only_nearest_supported_parallel_plane_owns_reading():
    # Two full frontage planes: source windows belong to y=0, not the second
    # wall 1.8 m in front that used to pass the 2.5 m tolerance.
    outline = MultiPolygon([
        Polygon([(0, 0), (10, 0), (10, 1), (0, 1)]),
        Polygon([(0, -1.8), (10, -1.8), (10, -0.8), (0, -0.8)]),
    ])
    p = _profile()
    assert profile_plane_binding(p, outline, (0, 0), (10, 0)) == (True, 'host-plane')
    assert profile_plane_binding(p, outline, (0, -1.8), (10, -1.8)) == (False, 'competing-host-plane')


def test_split_segments_on_same_supported_host_plane_stay_owned():
    outline = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    p = _profile()
    assert profile_plane_binding(p, outline, (0, 0), (4, 0)) == (True, 'host-plane')
    assert profile_plane_binding(p, outline, (4, 0), (10, 0)) == (True, 'host-plane')


def test_measured_bay_front_is_owned_by_bay_volume_not_second_wall_reading():
    # The host frontage has a four-metre recess in its outline. The measured
    # bay volume owns that front edge; openings remain on the host and bay
    # assembly exactly once.
    outline = Polygon([(0, 0), (3, 0), (3, -0.7), (7, -0.7), (7, 0),
                       (10, 0), (10, 4), (0, 4)])
    p = _profile(bays=[{'rect': [0.3, 0.2, 0.7, 0.8], 'projectionM': 0.7}])
    assert profile_plane_binding(p, outline, (0, 0), (3, 0)) == (True, 'host-plane')
    assert profile_plane_binding(p, outline, (3, -0.7), (7, -0.7)) == (False, 'bay-volume-owned')


def test_tiny_incidental_edge_cannot_claim_a_full_profile():
    outline = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    ok, reason = profile_plane_binding(_profile(), outline, (0, 0), (2, 0))
    assert not ok
    assert reason == 'insufficient-host-support'


def test_near_equidistant_host_planes_are_unresolved():
    outline = MultiPolygon([
        Polygon([(0, -0.2), (10, -0.2), (10, -1.2), (0, -1.2)]),
        Polygon([(0, 0.2), (10, 0.2), (10, 1.2), (0, 1.2)]),
    ])
    ok, reason = profile_plane_binding(_profile(), outline, (0, -0.2), (10, -0.2))
    assert not ok
    assert reason == 'ambiguous-host-plane'


def test_compile_keeps_near_host_reading_and_does_not_replace_rejected_plane_with_grid(monkeypatch):
    near = Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])
    far = Polygon([(0, -1.8), (10, -1.8), (10, -0.8), (0, -0.8)])
    outline = MultiPolygon([near, far])
    profile = _profile(
        id='source-front',
        match={'featureId': 'part-a', 'partId': 'part-a'},
        sourceIds=['mapillary:source'],
        heightRefM=6.0,
        material='brick',
        groups=[], openings=[{'rect': [.2, .2, .35, .45], 'style': {'kind': 'window'}}],
    )
    calls = []
    monkeypatch.setattr(facades, 'EXPERIMENTAL_PROFILE_HOST_BINDING', True)

    def measured_wall(ctx, c, a, b, z0, height, selected, fits):
        calls.append(selected['id'])
        return {'record': {'buildingId': c['id'], 'status': 'visual-grammar',
                           'grammarId': selected['id'], 'midpoint': ((a + b) / 2).tolist()},
                'components': 1}

    monkeypatch.setattr(facade_grammar, 'compile_wall', measured_wall)
    cube_calls = []
    ctx = {
        'PROFILES': [profile], 'FITS': [], 'cube': lambda *args: cube_calls.append(args),
        'cylinder': lambda *args: None,
        'polys': lambda geom: list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom],
        'lines': lambda geom: [geom] if geom.geom_type == 'LineString' else list(geom.geoms),
        'boundary': Polygon([(-5, -5), (15, -5), (15, 5), (-5, 5)]),
        'all_buildings': Polygon(),
    }
    c = {'id': 'part-a', 'sourceId': 'part-a', 'g': outline, 'sourceG': outline,
         'osm': None, 'full_area': 40.0, 'massHeightM': 6.0}
    records, _ = facades.compile(ctx, c, 0.0, 6.0, 6.0, {}, [])

    assert calls == ['source-front']
    far_fronts = [r for r in records if r.get('normal') == [0.0, -1.0]
                  and abs(r.get('midpoint', [0, 0])[1] + 1.8) < 1e-6]
    assert far_fronts
    assert all(r['status'] == 'unresolved' and r['floors'] == 0 for r in far_fronts)
    assert all('host plane' in ' '.join(r['unresolved']) for r in far_fronts)
    assert cube_calls


def test_panorama_patch_does_not_transfer_to_parallel_tier_or_other_end():
    p = _profile(sourceWallMidpointEN=[5, 0], sourceWallLengthM=10,
                 reading={'observedZRangeM': [1, 14]})
    assert facades.source_patch_owns_edge(p, (2, 0), (7, 0))
    assert not facades.source_patch_owns_edge(p, (2, 3), (7, 3))
    assert not facades.source_patch_owns_edge(p, (11, 0), (20, 0))
    assert not facades.source_patch_owns_edge(p, (2, 0), (7, 0), 15)


def test_partial_photo_preserves_unseen_upper_storeys_as_inferred():
    p = _profile(heightRefM=30, openings=[{'rect':[.3,.1,.6,.2]}],
                 reading={'observedZRangeM':[0,12]})
    style = dict(floorHeightM=3, groundHeightM=3, bayWidthM=3, openingRatio=.6,
                 openingHeightRatio=.7, mullions=1)
    result = facades.complete_partial_profile(p, style, 30)
    inferred = result['openings'][1:]
    assert inferred and all(o['evidence']=='inferred' and o['rect'][1]*30>=12 for o in inferred)
    assert result['openings'][0] == p['openings'][0] and len(p['openings'])==1


def test_sparse_entrance_reading_retains_measured_opening():
    from pipeline.facade_lint import clean
    profile = _profile(heightRefM=30, sourceIds=['osm:node:example'],
                       reading={'coverageMode':'positive-only','observedZRangeM':[0,2.2]},
                       groups=[],openings=[{'rect':[.42,0,.52,.07],'style':{'kind':'entry'}}])
    cleaned, _ = clean(profile)
    assert not cleaned.get('materialOnly')
    completed = facades.complete_partial_profile(cleaned,dict(floorHeightM=3,groundHeightM=3,
             bayWidthM=3,openingRatio=.65,openingHeightRatio=.7,mullions=1),30)
    measured = completed['openings'][0]['rect']
    for opening in completed['openings'][1:]:
        x0,y0,x1,y1=opening['rect']
        a,b,c,d=measured
        assert min(x1,c)<=max(x0,a) or min(y1,d)<=max(y0,b)
