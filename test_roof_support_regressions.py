"""Geometry regressions for observed front support and provisional continuation."""
import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import Polygon, Point, box
from types import SimpleNamespace

from general_world.roof_planes import evaluate
from pipeline.roof_forms import build as build_form
from pipeline.roof_shell import build, _limit_mesh_slope
from pipeline.roof_assemblies import resolve


def test_collinear_return_strip_does_not_displace_a_valid_roof_plane():
    from general_world.roof_planes import _candidates, RoofPolicy
    x, y = np.meshgrid(np.arange(.5, 8), np.arange(.5, 8))
    roof = np.c_[x.ravel(), y.ravel(), 8 + .2*x.ravel() - .1*y.ravel()]
    strip = np.c_[np.arange(20.5, 40), np.full(20, .5), np.full(20, 16.)]
    planes = _candidates(np.vstack([roof, strip]), RoofPolicy())
    assert planes and all(p is not None and np.isfinite(p).all() for p in planes)
    assert any(np.max(abs(np.c_[roof[:, :2], np.ones(len(roof))] @ p - roof[:, 2])) < 1e-8
               for p in planes)
    assert _candidates(strip, RoofPolicy()) == []


def test_measured_cell_rejection_filters_structural_raw_points_on_the_same_grid(monkeypatch):
    import pipeline.roof_shell as shell
    xx, yy = np.meshgrid(np.arange(.25, 10, .5), np.arange(.25, 10, .5))
    points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 8.)]
    points = np.vstack([points, [[.25, .25, 20.]] * 4])
    captured = []
    original = shell._fit_structural_regions
    def inspect(footprint, raw, *args):
        captured.append(raw.copy())
        return original(footprint, raw, *args)
    monkeypatch.setattr(shell, '_fit_structural_regions', inspect)
    result = shell.build(box(0, 0, 10, 10), points, 7., cell_m=.5)
    assert len(captured) == 1 and captured[0][:, 2].max() == 8
    assert result['audit']['structuralFitUsesMeasuredCellKeepMask']
    assert result['audit']['structuralInputPointsRejectedByMeasuredCellFilter'] == 5


def test_reassigned_tiny_plane_island_does_not_reenter_fixed_roof_ownership():
    from pipeline.roof_shell import _retain_owned_components
    main = np.array([[x, y] for x in range(3) for y in range(3)])
    keys = np.vstack([main, [[8, 0], [9, 0]]])
    samples = np.c_[keys + .5, np.full(len(keys), 8.)]
    kept, reports = _retain_owned_components(np.ones(len(keys), bool), keys, samples, np.array([0, 0, 8.]))
    assert kept[:9].all() and not kept[9:].any()
    assert sorted(r['cells'] for r in reports) == [2, 9]
    assert any(r['status'] == 'unresolved-small-disconnected-island' for r in reports)


def test_source_verified_small_roof_feature_can_preserve_its_bounded_measured_island():
    from pipeline.roof_shell import _retain_owned_components
    keys = np.array([[0, 0], [1, 0]])
    samples = np.c_[keys + .5, [8., 8.]]
    feature = {'geometry': box(0, 0, 2, 1), 'heightRangeZ': [7.9, 8.1],
               'provenance': {'status': 'source-derived', 'sourceSha256': 'a' * 64, 'metadataSha256': 'b' * 64}}
    kept, report = _retain_owned_components(np.ones(2, bool), keys, samples, np.array([0, 0, 8.]), [feature])
    assert kept.all() and report[0]['status'] == 'retained-source-feature-support'
    feature['heightRangeZ'] = [10, 11]
    rejected, _ = _retain_owned_components(np.ones(2, bool), keys, samples, np.array([0, 0, 8.]), [feature])
    assert not rejected.any()


def test_connected_inlier_recovery_never_bridges_empty_or_outlier_cells():
    from pipeline.roof_shell import _connected_inlier_cells
    keys = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [8, 0], [9, 0]])
    seed = np.array([1, 0, 0, 0, 0, 0, 0], bool)
    eligible = np.array([1, 1, 1, 0, 1, 1, 1], bool)
    result = _connected_inlier_cells(seed, eligible, keys)
    np.testing.assert_array_equal(result, [1, 1, 1, 0, 0, 0, 0])


def test_plane_recovery_uses_real_boundary_inliers_outside_regularized_outline(monkeypatch):
    from pipeline.roof_shell import _fit_structural_regions
    import general_world.roof_planes as planes
    footprint = box(0, 0, 10, 10)
    xs, ys = np.meshgrid(np.arange(.25, 8., .5), np.arange(.25, 10., .5))
    points = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 8.)]
    # The validated patch morphology omits the last half-metre of real roof
    # cells. Recover the perimeter strip, retaining the omitted interior as
    # unresolved rather than growing a global plane across empty regions.
    patch = {'geometry': box(0, 0, 7.5, 10), 'model': {'planes': [[0, 0, 8.]]}}
    monkeypatch.setattr(planes, 'fit_patches', lambda *args: ([patch], {'patches': []}))
    regions, audit = _fit_structural_regions(footprint, points, .5)
    assert regions[0]['geometry'].covers(Point(7.75, .25))
    assert not regions[0]['geometry'].covers(Point(7.75, 5.25))
    assert not regions[0]['geometry'].covers(Point(8.25, .25))
    assert audit['supportOwnership'][0]['recoveredConnectedInlierCellsOutsideFitOutline'] > 0


def test_rounding_of_a_supported_boundary_does_not_remove_plane_ownership():
    from pipeline.roof_shell import _harmonic_structural_heights
    xy = np.array([[-4e-10, 0], [1., 0], [1, 1], [0, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    z, known, unknown, audit = _harmonic_structural_heights(xy, faces,
        [{'geometry': box(0, 0, 1, 1), 'plane': np.array([0, 0, 8.])}],
        np.zeros(4, bool), np.full(4, np.nan), np.zeros(4))
    assert known.all() and not unknown.any()
    np.testing.assert_array_equal(z, [8] * 4)


def test_mixed_high_edge_returns_do_not_erase_locally_supported_lower_roof():
    from pipeline.roof_boundary_support import continue_adjoining_planes as _continue_adjoining_planes
    xs, ys = np.meshgrid(np.arange(3.25, 7., .5), np.arange(.25, 2.5, .5))
    lower = np.c_[xs.ravel(), ys.ravel(), 6 + .3 * ys.ravel()]
    high = lower.copy(); high[:, 2] += 3
    xy = np.array([[5., 0], [4., 0], [5., 5]])
    region = {'geometry': box(3, 1, 7, 3), 'plane': np.array([0., .3, 6.])}
    result, audit = _continue_adjoining_planes(box(0, 0, 10, 10), np.vstack([lower, high]),
        xy, np.full(3, 10.), [region], np.ones(3, bool), np.array([0, 1, 0], bool), .5)
    assert result[0] == pytest.approx(6)
    assert result[1] == 10  # independently fixed roof/source anchor
    assert result[2] == 10  # remote interior deck is outside perimeter scope
    assert audit['exactMeasurementsAdded'] == 0
    assert audit['planes'][0]['maximumInlierHullExtrapolationM'] <= .5


def test_boundary_plane_prior_declines_remote_deck_and_conflicting_levels():
    from pipeline.roof_boundary_support import continue_adjoining_planes as _continue_adjoining_planes
    xs, ys = np.meshgrid(np.arange(4.25, 6., .5), np.arange(4.25, 6., .5))
    raw = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 10.)]
    region = {'geometry': box(4, 4, 6, 6), 'plane': np.array([0., 0., 10.])}
    result, audit = _continue_adjoining_planes(box(0, 0, 10, 10), raw, np.array([[5., 0]]),
        np.array([6.]), [region], np.ones(1, bool), np.zeros(1, bool), .5)
    assert result[0] == 6 and audit['continuedVertices'] == 0


def _opposing_cell_regions():
    from shapely.ops import unary_union
    first, second = np.array([1., 0., 6.15]), np.array([-1., 0., 13.85])
    clouds, geometries = [[], []], [[], []]
    for x in np.arange(.25, 8, .5):
        for y in np.arange(.25, 12, .5):
            if abs(x - 3.85) < .7 and int(y * 2) % 4 in (1, 2):
                continue
            owner = int(x > 3.85)
            clouds[owner].append([x, y])
            geometries[owner].append(box(x - .25, y - .25, x + .25, y + .25))
    return [{'geometry': unary_union(g), 'plane': p, 'inlierXY': np.asarray(q)}
            for g, p, q in zip(geometries, (first, second), clouds)]


def test_bounded_opposing_planes_fill_cell_gaps_without_moving_supported_anchors():
    from pipeline.roof_shell import _recover_supported_ridges
    from shapely.ops import unary_union
    footprint = box(0, 0, 8, 12)
    original = _opposing_cell_regions()
    regions, creases, audit = _recover_supported_ridges(footprint, original, .5)
    assert audit['status'] == 'bounded-supported-creases-recovered'
    assert audit['creases'][0]['addedInterpolationAreaM2'] > 0
    assert audit['creases'][0]['bothSideNearIntersectionCells'][0] >= 6
    direct = unary_union([r['geometry'] for r in original])
    for region in regions[len(original):]:
        assert region['geometry'].intersection(direct).area < 1e-8
        assert region['geometry'].difference(direct.buffer(np.sqrt(2) * .5)).area < 1e-8
    # One off-grid crease remains a real two-plane intersection, not a whole
    # footprint style: both bounds and the source plane coefficients persist.
    np.testing.assert_allclose(creases[0]['line'], [2, 0, -7.7])
    for old, kept in zip(original, regions):
        assert old['geometry'].equals(kept['geometry'])
        np.testing.assert_array_equal(old['plane'], kept['plane'])


def test_supported_crease_never_replaces_third_roof_or_bridges_courtyard():
    from pipeline.roof_shell import _recover_supported_ridges
    footprint = Polygon([(0, 0), (8, 0), (8, 12), (0, 12)],
                        holes=[[(3, 4), (5, 4), (5, 5), (3, 5)]])
    original = _opposing_cell_regions()
    third = {'geometry': box(3, 8, 5, 10), 'plane': np.array([0., 0., 12.])}
    regions, creases, audit = _recover_supported_ridges(footprint, [*original, third], .5)
    assert creases
    for region in regions[3:]:
        assert region['geometry'].intersection(Polygon(footprint.interiors[0])).area < 1e-8
        assert region['geometry'].intersection(third['geometry']).area < 1e-8
    assert regions[2]['geometry'].equals(third['geometry'])
    for crease in creases:
        assert crease['zone'].intersection(third['geometry']).area < 1e-8


def test_two_flat_measured_levels_are_not_recast_as_a_ridge():
    from pipeline.roof_shell import _recover_supported_ridges
    regions = _opposing_cell_regions()
    regions[0]['plane'] = np.array([0., 0., 8.])
    regions[1]['plane'] = np.array([0., 0., 11.2])
    result, creases, audit = _recover_supported_ridges(box(0, 0, 8, 12), regions, .5)
    assert not creases
    assert len(result) == 2
    np.testing.assert_array_equal(result[0]['plane'], [0, 0, 8])
    np.testing.assert_array_equal(result[1]['plane'], [0, 0, 11.2])


def test_bounded_crease_recovery_is_invariant_to_nonaxis_rotation_and_large_translation():
    from pipeline.roof_shell import _recover_supported_ridges
    from shapely.affinity import rotate, translate
    original = _opposing_cell_regions()
    _, _, reference = _recover_supported_ridges(box(0, 0, 8, 12), original, .5)
    angle = np.deg2rad(31)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array([1234., -4321.])
    moved = []
    for region in original:
        gradient = rotation @ region['plane'][:2]
        moved.append({'geometry': translate(rotate(region['geometry'], 31, origin=(0, 0)), *offset),
                      'plane': np.r_[gradient, region['plane'][2] - gradient @ offset],
                      'inlierXY': region['inlierXY'] @ rotation.T + offset})
    footprint = translate(rotate(box(0, 0, 8, 12), 31, origin=(0, 0)), *offset)
    result, creases, audit = _recover_supported_ridges(footprint, moved, .5)
    assert creases and len(result) > 2
    assert audit['creases'][0]['addedInterpolationAreaM2'] == pytest.approx(reference['creases'][0]['addedInterpolationAreaM2'], abs=1e-7)


def test_recovered_off_grid_ridge_is_straight_in_generated_mesh(monkeypatch):
    import pipeline.roof_shell as shell
    regions = _opposing_cell_regions()
    points = np.vstack([np.c_[r['inlierXY'], np.c_[r['inlierXY'], np.ones(len(r['inlierXY']))] @ r['plane']]
                        for r in regions])
    monkeypatch.setattr(shell, '_fit_structural_regions', lambda *args: (regions, {'fit': {}}))
    result = shell.build(box(0, 0, 8, 12), points, 6., cell_m=.5)
    assert result['audit']['structuralPlaneSurface']['boundedCreaseInterpolation']['creases']
    ridge = result['surface'].vertices[np.isclose(result['surface'].vertices[:, 0], 3.85, atol=1e-8)]
    ridge = ridge[(ridge[:, 1] > .25) & (ridge[:, 1] < 11.75)]
    assert len(ridge) > 15
    np.testing.assert_allclose(ridge[:, 2], 10., atol=1e-8)


def test_measured_aperture_clearance_accepts_plinth_contact_but_never_covers_opening():
    profile = {'id': 'regen-view', 'match': {'partId': 'part'}, 'normal': [1, 0],
               'frameU': [0, 10], 'planeV': 10, 'reading': {'confidence': 'high'}}
    base = {'kind': 'dormer', 'x0M': 4, 'x1M': 6, 'bottomM': 5, 'topM': 6.5,
            'roof': 'gable', 'windowCount': 1, 'confidence': 'medium'}
    aperture = {'coordinateSpace': 'normalized-front', 'apertures': [
        {'kind': 'louver', 'outline01': [[.2, .12], [.8, .12], [.8, .6], [.5, .9], [.2, .6]]}]}
    footprint = box(4, 0, 10, 10)
    def surface(z):
        return SimpleNamespace(vertices=np.array([[4, 0, z], [10, 0, z], [10, 10, z], [4, 10, z]]),
                               faces=np.array([[0, 1, 2], [0, 2, 3]]))
    generic, _ = resolve([{'side': 'view', 'features': [base]}], [profile], 'part', 'source', footprint, 0, surface(5.12))
    assert not generic  # the legacy 8 cm sill cannot authorize this overlap
    observed = {**base, 'frontEvidence': aperture}
    accepted, audit = resolve([{'side': 'view', 'features': [observed]}], [profile], 'part', 'source', footprint, 0, surface(5.12))
    assert len(accepted) == 1
    assert audit['accepted'][0]['supportClearance'] == 'source-aperture-clearance'
    assert audit['accepted'][0]['maximumSupportZM'] < 5.235
    covered, _ = resolve([{'side': 'view', 'features': [observed]}], [profile], 'part', 'source', footprint, 0, surface(5.3))
    assert not covered


def _roof_points(footprint, spec, step=.25):
    patches, _ = build_form(footprint, {**spec, 'overhangM': 0}, 8., 8.)
    x0, y0, x1, y1 = footprint.bounds
    xx, yy = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    xy = np.c_[xx.ravel(), yy.ravel()]
    xy = xy[contains_xy(footprint, xy[:, 0], xy[:, 1])]
    return np.c_[xy, evaluate(patches[0]['model'], xy)]


def test_offset_frontage_reaches_real_hip_ridge_without_midpoint_alignment():
    footprint = box(0, 0, 10, 16)
    points = _roof_points(footprint, {'kind': 'hip', 'riseM': 2.5})
    # A partial facade ends before the right corner, so its midpoint is x=4,
    # while the independently supported hip ridge is at x=5.
    result = build(footprint, points, 7., eave_constraints=[{'a': [0, 0], 'b': [8, 0], 'z': 8}],
                   host_spec={'kind': 'hip', 'eaveZ': 8., 'ridgeZ': 10.5, 'confidence': 'high'})
    assert result['audit']['observedHostForm']['status'] == 'applied-photo-host-form'
    assert abs(result['surface'].vertices[:, 2].max() - 10.5) < 1e-6


def test_narrow_mansard_uses_actual_boundary_returns_without_changing_interior_floor():
    footprint = box(0, 0, 8, 12)
    points = _roof_points(footprint, {'kind': 'mansard', 'riseM': 1.2}, step=.15)
    interior = points[contains_xy(footprint.buffer(-.65), points[:, 0], points[:, 1])]
    common = dict(eave_constraints=[{'a': [0, 0], 'b': [8, 0], 'z': 8.}],
                  host_spec={'kind': 'mansard', 'eaveZ': 8., 'ridgeZ': None, 'confidence': 'high'})
    without = build(footprint, interior, 8., **common)
    assert without['audit']['observedHostForm']['status'] != 'applied-photo-host-form'
    result = build(footprint, interior, 8., boundary_points=points, **common)
    assert result['audit']['additionalBoundaryEvidencePointCount'] > 0
    assert result['audit']['primaryInteriorPointCount'] == len(interior)
    assert result['audit']['observedHostForm']['status'] == 'applied-photo-host-form'
    assert result['audit']['analyticHostMesh']['status'] == 'exact-plane-envelope'
    assert result['surface'].vertices[:, 2].min() >= 8 - 1e-6
    assert abs(result['surface'].vertices[:, 2].max() - 9.2) < .05


def test_slope_limit_moves_only_provisional_vertices_preserving_measured_anchors():
    xy = np.array([[0., 0.], [1., 0.], [.01, .02]])
    source = np.array([8., 8., 12.])
    heights, _, _, adjusted, unknown_grade = _limit_mesh_slope(
        xy, np.array([[0, 1, 2]]), source,
        eligible_vertices=np.array([False, False, True]), maximum_grade=2.5, sweeps=64)
    np.testing.assert_array_equal(heights[:2], source[:2])
    assert adjusted > 0
    assert unknown_grade <= 2.5 + 1e-6


def test_boundary_cloud_cannot_import_neighbor_points_or_fill_courtyard():
    footprint = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]])
    xx, yy = np.meshgrid(np.arange(.25, 10, .5), np.arange(.25, 10, .5))
    xy = np.c_[xx.ravel(), yy.ravel()]
    xy = xy[contains_xy(footprint, xy[:, 0], xy[:, 1])]
    primary = np.c_[xy, np.full(len(xy), 8.)]
    extra = np.array([[20., 20., 40.], [5., 5., 40.]])
    result = build(footprint, primary, 7., boundary_points=extra)
    assert result['audit']['additionalBoundaryEvidencePointCount'] == 0
    assert result['audit']['footprintHoles'] == 1
    assert not any(Polygon(t[:, :2]).contains(Point(5, 5)) for t in result['surface'].triangles)
    np.testing.assert_allclose(result['surface'].vertices[:, 2], 8., atol=1e-6)


def test_compound_roof_recovers_only_heldout_supported_thin_front_plane():
    from pipeline.roof_shell import _recover_front_planes
    footprint = box(0, 0, 8, 12)
    xx, yy = np.meshgrid(np.arange(.05, 8, .1), np.arange(.05, 3, .1))
    xy = np.c_[xx.ravel(), yy.ravel()]
    z = np.where(xy[:, 1] < .65, 8 + 2 * xy[:, 1], 7.)
    points = np.c_[xy, z]
    old = [{'geometry': footprint, 'plane': np.array([0., 0., 7.])}]
    regions, audit = _recover_front_planes(footprint, points,
        {'kind': 'mansard', 'confidence': 'high'}, [{'a': [0, 0], 'b': [8, 0]}], old, .5)
    assert audit['status'] == 'independently-validated-local-front-planes'
    assert audit['acceptedPlanes']
    assert all(p['heldOutCells'] >= 6 and p['heldOutP95M'] <= .25 for p in audit['acceptedPlanes'])
    assert audit['replacedCoarseLowerOwnershipM2'] > 0
    # A different supported rear roof stays in control; no whole-footprint
    # mansard is fabricated from the observed frontage plane.
    rear = [r for r in regions if r['geometry'].covers(Point(4, 8))]
    assert len(rear) == 1
    np.testing.assert_allclose(rear[0]['plane'], [0, 0, 7])
    front = [r for r in regions if r['geometry'].covers(Point(4, .25))]
    assert len(front) == 1
    assert abs(front[0]['plane'] @ [4, .25, 1] - 8.5) < .05


def test_local_front_recovery_declines_unstructured_returns_and_keeps_higher_roof():
    from pipeline.roof_shell import _recover_front_planes
    footprint = box(0, 0, 8, 12)
    xx, yy = np.meshgrid(np.arange(.125, 8, .25), np.arange(.125, 3, .25))
    xy = np.c_[xx.ravel(), yy.ravel()]
    rng = np.random.default_rng(174)
    noisy = np.c_[xy, rng.uniform(5, 15, len(xy))]
    _, audit = _recover_front_planes(footprint, noisy,
        {'kind': 'hip', 'confidence': 'high'}, [{'a': [0, 0], 'b': [8, 0]}], None, .5)
    assert not audit['acceptedPlanes']
    measured = np.c_[xy, 8 + .5 * xy[:, 1]]
    higher = [{'geometry': footprint, 'plane': np.array([0., 0., 12.])}]
    regions, audit = _recover_front_planes(footprint, measured,
        {'kind': 'hip', 'confidence': 'high'}, [{'a': [0, 0], 'b': [8, 0]}], higher, .5)
    assert not audit['acceptedPlanes']
    assert len(regions) == 1
    np.testing.assert_allclose(regions[0]['plane'], [0, 0, 12])


def test_photo_observed_dormer_owns_intersecting_host_without_moving_apertures():
    from pipeline.roof_shell import component_cutters, clip_component_intersections
    from pipeline.roof_features import build as build_features
    import trimesh
    profile = {'id': 'regen-front', 'match': {'partId': 'part'}, 'normal': [1, 0],
               'frameU': [0, 10], 'planeV': 10, 'reading': {'confidence': 'high'}}
    evidence = {'coordinateSpace': 'normalized-front', 'apertures': [
        {'kind': 'window', 'outline01': [[.2, .12], [.8, .12], [.8, .6], [.5, .9], [.2, .6]]}],
        'provenance': {'status': 'accepted-approximate', 'sourceSha256': 'a' * 64, 'metadataSha256': 'b' * 64}}
    feature = {'kind': 'dormer', 'x0M': 4, 'x1M': 6, 'bottomM': 5, 'topM': 6.5,
               'roof': 'gable', 'windowCount': 1, 'confidence': 'medium', 'frontEvidence': evidence}
    footprint = box(4, 0, 10, 10)
    surface = trimesh.Trimesh(vertices=[[4, 0, 5.6], [10, 0, 5.6], [10, 10, 5.6], [4, 10, 5.6]],
                              faces=[[0, 1, 2], [0, 2, 3]], process=False)
    specs, audit = resolve([{'side': 'front', 'features': [feature]}], [profile], 'part', 'source', footprint, 0, surface)
    assert len(specs) == 1
    assert specs[0]['composeWithHost']
    assert specs[0]['frontEvidence'] == evidence
    assert audit['accepted'][0]['supportPolicy'] == 'source-front-body-intersection'
    cutters, records = component_cutters(specs, 0)
    assert cutters and records[0]['frontCoordinatesMoved'] is False
    cut = clip_component_intersections(surface, cutters)
    assert cut.area < surface.area
    assert not any(Polygon(t[:, :2]).covers(Point(9.5, 5)) for t in cut.triangles)
    assert any(Polygon(t[:, :2]).covers(Point(5, 1)) for t in cut.triangles)
    parts = build_features(specs, footprint, 0, 5.6, [])
    assert any(role == 'glass' for _, role in parts)
    assert all(m.metadata.get('frontEvidenceStatus') != 'unresolved' for m, _ in parts)
    # The opaque facade roof band is clipped by the same bounded body, so it
    # cannot reappear as backing behind the source glazing.
    wall = trimesh.Trimesh(vertices=[[10, 0, 0], [10, 10, 0], [10, 10, 5.6], [10, 0, 5.6]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    cut_wall = clip_component_intersections(wall, cutters)
    assert not any(Polygon(t[:, 1:]).covers(Point(5, 5.4)) for t in cut_wall.triangles)
    assert any(Polygon(t[:, 1:]).covers(Point(1, 5.4)) for t in cut_wall.triangles)
    contradiction = surface.copy(); contradiction.vertices[:, 2] = 6.6
    rejected, _ = resolve([{'side': 'front', 'features': [feature]}], [profile], 'part', 'source', footprint, 0, contradiction)
    assert not rejected  # actual host above the source cap remains a contradiction


def test_source_front_cap_shape_controls_body_instead_of_generic_shed_ears():
    from pipeline.roof_features import build as build_features, dormer_cut_solids
    feature = {'kind': 'dormer', 'xy': [5, 5], 'facing': [1, 0], 'widthM': 2, 'depthM': 1,
               'bottomM': 5, 'eaveM': 6.2, 'ridgeM': 6.5, 'roof': 'shed', 'windowCount': 0,
               'frontEvidence': {'coordinateSpace': 'normalized-front',
                   'outline01': [[0, 0], [1, 0], [1, .6], [.8, .8], [.5, 1], [.2, .8], [0, .6]],
                   'apertures': []}}
    parts = build_features([feature], box(0, 0, 10, 10), 0, 5, [])
    roofs = [m for m, role in parts if role == 'roof']
    assert roofs
    roof = np.concatenate([m.vertices for m in roofs])
    assert roof[:, 1].min() >= 4 - 1e-8 and roof[:, 1].max() <= 6 + 1e-8
    side = roof[np.isclose(abs(roof[:, 1] - 5), 1)]
    assert side[:, 2].max() <= 5.9 + 1e-8
    assert abs(roof[:, 2].max() - 6.5) < 1e-8
    assert all(s.is_volume for s in dormer_cut_solids(feature, 0))
