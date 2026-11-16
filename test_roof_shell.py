import numpy as np
from pathlib import Path
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points

from pipeline.roof_shell import build, _aggregate_cells, _register_frontage_line


def _plane_points(bounds, step=.3):
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    xx, yy = np.meshgrid(xs, ys)
    return np.c_[xx.ravel(), yy.ravel(), 8 + .2 * xx.ravel() - .1 * yy.ravel()]


def _projected_area(mesh):
    tri = mesh.vertices[mesh.faces, :2]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    return float(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2)


def _close_with_bottom(result, base_z):
    surface = result["surface"]
    bottom = surface.vertices.copy()
    bottom[:, 2] = base_z
    cap = type(surface)(vertices=bottom, faces=surface.faces[:, ::-1], process=False)
    mesh = __import__('trimesh').util.concatenate([surface, result["skirt"], cap])
    mesh.merge_vertices(digits_vertex=8)
    return mesh


def _close_analytic_with_bottom(result, footprint, model, base_z):
    import trimesh
    from general_world.roof_planes import roof_solid

    reference = roof_solid(footprint, model, base_z)
    bottom = np.max(np.abs(reference.vertices[reference.faces, 2] - base_z), axis=1) <= 1e-6
    cap = trimesh.Trimesh(vertices=reference.vertices.copy(), faces=reference.faces[bottom], process=False)
    mesh = trimesh.util.concatenate([result["surface"], result["skirt"], cap])
    mesh.merge_vertices(digits_vertex=8)
    return mesh


def test_l_footprint_is_clipped_and_slope_is_preserved():
    footprint = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    result = build(footprint, _plane_points(footprint.bounds), base_z=2, cell_m=.5)
    surface = result["surface"]
    assert len(surface.faces) > 0
    projected_area = _projected_area(surface)
    assert abs(projected_area - footprint.area) < .02
    assert all(footprint.buffer(1e-8).covers(Polygon(surface.vertices[f, :2]).centroid)
               for f in surface.faces)
    expected = 8 + .2 * surface.vertices[:, 0] - .1 * surface.vertices[:, 1]
    # Near the footprint edge, samples just outside the measured convex hull
    # use the documented bounded nearest continuation.
    assert np.max(np.abs(surface.vertices[:, 2] - expected)) < .12
    closed = _close_with_bottom(result, 2)
    assert closed.is_watertight
    assert closed.is_winding_consistent
    assert closed.volume > 0


def test_courtyard_remains_uncapped_and_boundary_samples_are_dense():
    footprint = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)],
                        holes=[[(2, 2), (2, 3), (3, 3), (3, 2)]])
    result = build(footprint, _plane_points(footprint.bounds), base_z=1, cell_m=.5)
    surface = result["surface"]
    projected_area = _projected_area(surface)
    assert abs(projected_area - footprint.area) < .03
    assert not any(Polygon(surface.vertices[f, :2]).centroid.within(Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]))
                   for f in surface.faces)
    assert result["samples"].shape[1] == 3
    assert len(result["samples"]) >= 4 * 5 / .5
    closed = _close_with_bottom(result, 1)
    assert closed.is_watertight
    assert closed.is_winding_consistent
    assert closed.volume > 0


def test_mixed_collection_preserves_all_roof_areas_and_courtyard_with_boundary_evidence():
    courtyard = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)],
                        holes=[[(2, 2), (2, 3), (3, 3), (3, 2)]])
    separate = Polygon([(8, 0), (10, 0), (10, 2), (8, 2)])
    area = MultiPolygon([courtyard, separate])
    footprint = GeometryCollection([courtyard, GeometryCollection([
        MultiPolygon([separate]), LineString([(5, 0), (8, 0)]), Point(11, 1)])])
    assert footprint.boundary is None  # the full-world clipping failure
    points = _plane_points(area.bounds)
    boundary = np.vstack([points, [[6.5, 0, 80], [11, 1, 90]]])
    result = build(footprint, points, base_z=1, cell_m=.5, boundary_points=boundary)
    assert abs(_projected_area(result['surface']) - area.area) < .03
    assert result['audit']['inputFootprintGeometryType'] == 'GeometryCollection'
    assert result['audit']['polygonalFootprintGeometryType'] == 'MultiPolygon'
    assert result['audit']['footprintHoles'] == 1
    assert result['surface'].vertices[:, 2].max() < 12
    assert all(area.buffer(1e-8).covers(Point(p[:2])) for p in result['surface'].vertices)
    closed = _close_with_bottom(result, 1)
    assert closed.is_watertight and closed.is_winding_consistent


def test_multipolygon_input_keeps_disconnected_area_components():
    footprint = MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                              Polygon([(5, 0), (7, 0), (7, 2), (5, 2)])])
    result = build(footprint, _plane_points(footprint.bounds), base_z=1,
                   boundary_points=_plane_points(footprint.bounds))
    assert abs(_projected_area(result['surface']) - 8) < .02
    assert result['audit']['polygonalFootprintGeometryType'] == 'MultiPolygon'
    closed = _close_with_bottom(result, 1)
    assert closed.is_watertight and closed.is_winding_consistent


def test_shared_grid_edges_have_no_internal_skirt_faces():
    footprint = Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
    result = build(footprint, _plane_points(footprint.bounds), base_z=0, cell_m=.5)
    # Every skirt triangle lies on the actual footprint boundary. Internal
    # cell edges are welded in the top mesh and receive no vertical faces.
    skirt = result["skirt"]
    centroids = skirt.triangles_center
    assert all(footprint.boundary.distance(Point(p[:2])) < 1e-7
               for p in centroids)
    xy = np.round(result["surface"].vertices[:, :2], 9)
    assert len(np.unique(xy, axis=0)) == len(xy)
    assert np.max(np.abs(result["samples"][:, 2] -
                         (8 + .2 * result["samples"][:, 0] - .1 * result["samples"][:, 1]))) < .12


def test_frontage_registration_follows_a_kinked_boundary_run():
    footprint = Polygon([(0, 0), (4, 0), (4, 1), (6, 1), (6, 0),
                         (10, 0), (10, 6), (0, 6)])
    points = _plane_points(footprint.bounds)
    result = build(footprint, points, base_z=2., cell_m=.5,
                   eave_constraints=[{"a": [.2, -.4], "b": [9.8, -.4], "z": 8.}])
    report = result["audit"]["eaveConstraintRegistration"][0]
    assert result["audit"]["eaveConstraintsAccepted"] == 1, report
    assert report["status"] == "accepted", report
    assert report["rawToBoundaryMaxM"] < 1.5
    assert report["registeredResidualM"] == 0.
    # Both facade runs and the recessed kink share the same registered eave.
    vertices = result["surface"].vertices
    for target in ([1., 0.], [4., 0.], [4., 1.], [5., 1.], [6., 1.], [6., 0.], [9., 0.]):
        distances = np.linalg.norm(vertices[:, :2] - target, axis=1)
        vertex = vertices[np.argmin(distances)]
        assert distances.min() < 1e-7
        assert abs(vertex[2] - 8.) < 1e-7


def test_frontage_registration_rejects_nearby_wrong_tangent():
    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    line, report = _register_frontage_line(footprint, LineString([(1., -.8), (2., .2)]))
    assert line is None
    assert report["status"] == "rejected"
    assert report["reason"] in ("frontage-tangent-mismatch", "frontage-too-far-from-boundary")


def test_measured_noisy_roof_rejects_isolated_cells_and_bounds_extrapolation():
    fixture = Path(__file__).parent / "tests" / "fixtures" / "shell-noisy-roof.fixture"
    data = np.load(fixture)
    points, footprint = data["points"], Polygon(data["footprint"])
    aggregated, keys, _, _ = _aggregate_cells(points, .5)
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    raw_jumps = []
    for i, (x, y) in enumerate(keys):
        for dx, dy in ((1, 0), (0, 1)):
            j = lookup.get((x + dx, y + dy))
            if j is not None:
                raw_jumps.append(abs(aggregated[i, 2] - aggregated[j, 2]))
    result = build(footprint, points, base_z=0, cell_m=.5)
    mesh = result["surface"]
    edges = mesh.edges_unique
    delta = np.abs(mesh.vertices[edges[:, 0], 2] - mesh.vertices[edges[:, 1], 2])
    length = np.linalg.norm(mesh.vertices[edges[:, 0], :2] - mesh.vertices[edges[:, 1], :2], axis=1)
    grade = delta / np.maximum(length, 1e-9)
    audit = result["audit"]
    structural = audit["structuralPlaneSurface"]
    assert structural["status"] == "validated-local-structural-regions"
    assert structural["unresolvedAreaFraction"] > .40
    accepted = [p for p in structural["fit"]["patches"] if "heldOutP95M" in p and "rejected" not in p]
    assert accepted and max(p["heldOutP95M"] for p in accepted) <= .30
    assert audit["analyticHostMesh"]["status"] == "shared-local-plane-arrangement"
    assert structural["supportedOwnedFaceCount"] > 0
    assert structural["unsupportedOwnedFaceCount"] > 0
    assert structural["maximumSampledContinuationDistanceM"] > structural["maximumAllowedContinuationDistanceM"]
    assert len(structural["sharedSeams"]) > 0
    unknown_low, unknown_high = structural["unknownHeightRangeM"]
    anchor_low, anchor_high = structural["dirichletHeightRangeM"]
    assert anchor_low - 1e-6 <= unknown_low <= unknown_high <= anchor_high + 1e-6
    assert structural["maximumUnknownEdgeGrade"] < 2.0
    assert delta.max() < 1.0  # no unsupported multi-metre ridge/fill jumps
    assert max(raw_jumps) > 8
    assert audit["rejectedIsolatedCells"] >= 10
    assert audit["unsupportedVertexCount"] > 0
    assert audit["boundaryExtrapolatedSampleCount"] > 0
    assert audit["maximumExtrapolatedMeshGrade"] <= audit["extrapolatedEdgeGradeCap"] + 1e-3
    assert delta.max() < max(raw_jumps)
    assert abs(audit["maximumMeshGrade"] - structural["maximumUnknownEdgeGrade"]) < 1e-6
    low, high = audit["extrapolationHeightEnvelopeM"]
    assert result["samples"][:, 2].min() >= low - .01
    assert result["samples"][:, 2].max() <= high + .1
    closed = _close_with_bottom(result, 0)
    assert closed.is_watertight
    assert closed.is_winding_consistent
    assert closed.volume > 0


def test_validated_plane_fallback_builds_one_shared_ridge_shell():
    from general_world.roof_planes import evaluate
    from pipeline.roof_forms import build as build_form

    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    patches, _ = build_form(footprint, {"kind": "gable", "riseM": 2., "overhangM": 0.}, 8., 8.)
    model = patches[0]["model"]
    xs = np.arange(.25, 10, .5); ys = np.arange(.25, 6, .5)
    xx, yy = np.meshgrid(xs, ys); xy = np.c_[xx.ravel(), yy.ravel()]
    points = np.c_[xy, evaluate(model, xy)]
    result = build(footprint, points, base_z=2., cell_m=.5)
    audit = result["audit"]
    assert audit["analyticHostMesh"]["status"] == "shared-local-plane-arrangement"
    assert audit["structuralPlaneSurface"]["unresolvedAreaFraction"] == 0.
    surface = result["surface"]
    assert surface.is_winding_consistent
    ridge = surface.vertices[np.isclose(surface.vertices[:, 1], 3., atol=1e-7)]
    assert len(ridge) >= 2 and np.allclose(ridge[:, 2], 10., atol=1e-6)
    # Each accepted facet is planar in its interior and the two facets share
    # identical vertices along their exact common ridge.
    assert audit["structuralPlaneSurface"]["maximumSharedPlaneHeightDifferenceM"] < 1e-6
    assert np.max(np.abs(ridge[:, 2] - 10.)) < 1e-6
    closed = _close_with_bottom(result, 2.)
    assert closed.is_watertight and closed.is_winding_consistent


def test_measured_plane_gap_uses_bounded_harmonic_fill():
    fixture = Path(__file__).parent / "tests" / "fixtures" / "shell-seam-conflict.npz"
    data = np.load(fixture)
    footprint = Polygon(data["footprint"])
    result = build(footprint, data["points"], float(data["base_z"][0]), cell_m=.5)
    audit = result["audit"]["structuralPlaneSurface"]
    assert audit["method"] == "harmonic-graph-interpolation-with-Dirichlet-plane-and-eave-anchors"
    # Direct cell ownership leaves the mixed seam unowned instead of letting
    # morphology claim it for either plane.
    assert audit["unknownVertexCount"] > 0
    assert sum(q.get("removedUnsupportedExpansionM2", 0.) for q in audit["supportOwnership"]) > 0.
    anchor_low, anchor_high = audit["dirichletHeightRangeM"]
    unknown_low, unknown_high = audit["unknownHeightRangeM"]
    assert anchor_low - 1e-6 <= unknown_low <= unknown_high <= anchor_high + 1e-6
    assert audit["maximumUnknownEdgeGrade"] <= 1.5 + 1e-6
    edges = result["surface"].edges_unique
    vertices = result["surface"].vertices
    jumps = np.abs(vertices[edges[:, 0], 2] - vertices[edges[:, 1], 2])
    assert jumps.max() < .9
    closed = _close_with_bottom(result, float(data["base_z"][0]))
    assert closed.is_watertight and closed.is_winding_consistent


def test_eave_constraints_own_boundary_height_and_prevent_downward_skirts():
    footprint = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])
    result = build(footprint, _plane_points(footprint.bounds), base_z=7,
                   cell_m=.5, eave_constraints=[{"a": [0, 0], "b": [4, 0], "z": 7.5}])
    on_eave = np.abs(result["samples"][:, 1]) < 1e-8
    assert on_eave.any()
    assert np.max(np.abs(result["samples"][on_eave, 2] - 7.5)) < 1e-8
    assert np.min(result["surface"].vertices[:, 2]) >= 7
    assert np.min(result["skirt"].vertices[:, 2]) >= 7
    assert result["audit"]["eaveConstraintsAccepted"] == 1
    assert result["audit"]["eaveConstraintVerticesApplied"] >= 8
    closed = _close_with_bottom(result, 7)
    assert closed.is_watertight
    assert closed.is_winding_consistent


def test_piecewise_frontage_roofline_is_a_local_boundary_condition():
    footprint = Polygon([(0, 0), (8, 0), (8, 5), (0, 5)])
    points = _plane_points(footprint.bounds)
    constraints = [
        {"a": [0, 0], "b": [4, 0], "z": 5., "z1": 8.1, "role": "gable-frontage-roofline"},
        {"a": [4, 0], "b": [8, 0], "z": 8.1, "z1": 5., "role": "gable-frontage-roofline"},
    ]
    result = build(footprint, points, base_z=4., cell_m=.5, eave_constraints=constraints)
    boundary = result["samples"]
    for x, expected in ((0., 5.), (4., 8.1), (8., 5.)):
        near = np.linalg.norm(boundary[:, :2] - [x, 0.], axis=1) < 1e-7
        assert near.any()
        assert np.max(np.abs(boundary[near, 2] - expected)) < 1e-6
    assert result["audit"]["boundaryConstraintRoles"]["gable-frontage-roofline"] == 2
    chain = next(q["points"] for q in result["boundaryChains"] if q["ring"] == "exterior")
    assert np.linalg.norm(chain[0, :2] - chain[-1, :2]) > 0
    assert _close_with_bottom(result, 4.).is_watertight


def test_measured_gable_frontage_roofline_on_real_filtered_fixture():
    import json

    root = Path(__file__).parent
    fixture = root / "areas" / "pacific-heights" / "roof-evidence-baseline" / "roof-evidence" / "sf_201006_0116178.npz"
    profiles_path = root / "areas" / "pacific-heights" / "regen" / "profiles-shell-review-v5.json"
    observation_path = root / "areas" / "review" / "roof-rethink" / "observations-v3.json"
    if not (fixture.exists() and profiles_path.exists() and observation_path.exists()):
        import pytest
        pytest.skip("measured roof-shell review fixture is unavailable")
    massing_path = root / "areas" / "pacific-heights" / "roof-shell-r11-raw" / "massing-dump.json"
    profiles = json.loads(profiles_path.read_text())["profiles"]
    profile = next(p for p in profiles if p["id"].endswith("__s2p0") and "0116178" in p["id"])
    observations = json.loads(observation_path.read_text())
    if isinstance(observations, dict):
        observations = observations.get("observations", observations.get("roofObservations", []))
    observation = next(o for o in observations if o.get("side") == profile["id"].removeprefix("regen-"))
    roofline = np.asarray(observation["frontageRoofline"]["pointsM"], dtype=float)
    n = np.asarray(profile["normal"], dtype=float); n /= np.linalg.norm(n)
    tangent = np.array([-n[1], n[0]])
    xy = n[None, :] * float(profile["planeV"]) + tangent[None, :] * (float(profile["frameU"][0]) + roofline[:, 0, None])
    massing = json.loads(massing_path.read_text())
    item = next(q for q in massing if q.get("id") == "sf_201006_0116178")
    footprint = Polygon(item["outline"][0])
    base_z = float(item["baseRoofZ"])
    z0 = float(item["z0"])
    lidar = np.load(fixture)["points"]
    constraints = [{"a": xy[i].tolist(), "b": xy[i + 1].tolist(),
                    "z": z0 + float(roofline[i, 1]), "z1": z0 + float(roofline[i + 1, 1]),
                    "role": "gable-frontage-roofline"}
                   for i in range(len(roofline) - 1)]
    result = build(footprint, lidar, base_z=base_z, cell_m=.5, eave_constraints=constraints)
    boundary = result["samples"]
    for source_xy, (_, relative_z) in zip(xy, roofline):
        projected = np.asarray(nearest_points(footprint.boundary, Point(source_xy))[0].coords[0])
        index = int(np.argmin(np.linalg.norm(boundary[:, :2] - projected, axis=1)))
        assert np.linalg.norm(boundary[index, :2] - projected) < .15
        assert abs(boundary[index, 2] - (z0 + relative_z)) < .03
    assert result["audit"]["boundaryConstraintRoles"]["gable-frontage-roofline"] == 2
    assert result["boundaryChains"]
    # A previously extrapolated plane vertex at this location stood about 3 m
    # above the nearest local LiDAR. It must now be an unknown harmonic vertex,
    # not a plane-owned anchor.
    sail_xy = np.array([-214.5, -406.5])
    local = lidar[np.linalg.norm(lidar[:, :2] - sail_xy, axis=1) <= .9]
    assert len(local) > 0
    vertices = result["surface"].vertices
    vertex_id = int(np.argmin(np.linalg.norm(vertices[:, :2] - sail_xy, axis=1)))
    assert np.linalg.norm(vertices[vertex_id, :2] - sail_xy) < 1e-7
    assert vertices[vertex_id, 2] <= local[:, 2].max() + .5
    ownership = result["audit"]["structuralPlaneSurface"]["supportOwnership"]
    assert sum(q.get("removedUnsupportedExpansionM2", 0.) for q in ownership) > 5.


def test_thin_hole_in_supported_plane_does_not_become_plane_ownership():
    from pipeline.roof_shell import _fit_structural_regions

    footprint = Polygon([(0, 0), (12, 0), (12, 12), (0, 12)])
    xs = np.arange(2.25, 10., .5)
    ys = np.arange(2.25, 8.25, .5)
    xx, yy = np.meshgrid(xs, ys)
    points = np.c_[xx.ravel(), yy.ravel(), 8. + .15 * xx.ravel() - .05 * yy.ravel()]
    missing = ((np.abs(points[:, 0] - 6.25) < .1) & (np.abs(points[:, 1] - 5.25) < .1))
    points = points[~missing]
    regions, audit = _fit_structural_regions(footprint, points, .5)
    assert len(regions) == 1
    ownership = audit["supportOwnership"][0]
    assert ownership["fitRegionAreaM2"] == 48.
    assert ownership["removedUnsupportedExpansionM2"] == .25
    assert not regions[0]["geometry"].covers(Point(6.25, 5.25))


def test_observed_host_model_controls_frontage_and_joins_same_surface():
    from general_world.roof_planes import evaluate
    from pipeline.roof_forms import build as build_form

    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    patches, _ = build_form(footprint, {"kind": "gable", "riseM": 2., "overhangM": 0.}, 8., 8.)
    model = patches[0]["model"]
    xs = np.arange(.25, 10, .5); ys = np.arange(.25, 6, .5)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.c_[xx.ravel(), yy.ravel()]
    points = np.c_[xy, evaluate(model, xy)]
    # Sparse positive rooftop fixtures should not veto a building-scale host.
    points = np.vstack([points, [[4.75, 1.75, 12.5], [5.25, 1.75, 12.4]]])
    # Facade reading is offset 0.9 m outside the mapped footprint.
    frontage = [{"a": [0., -.9], "b": [10., -.9], "z": 8.}]
    result = build(footprint, points, base_z=2., cell_m=.5, eave_constraints=frontage,
                   host_spec={"kind": "gable", "eaveZ": 8., "ridgeZ": 10., "confidence": "high"})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-photo-host-form", audit
    assert audit["zoneDepthSource"] == "inferred-first-ridge-crossing-from-matched-eave"
    front = result["surface"].vertices[np.isclose(result["surface"].vertices[:, 1], 0.)]
    assert len(front) >= 2
    assert np.max(np.abs(front[:, 2] - 8.)) < 1e-6
    closed = _close_analytic_with_bottom(result, footprint, model, 2.)
    assert closed.is_watertight and closed.is_winding_consistent
    assert result["surface"].is_winding_consistent
    for chain in result["boundaryChains"]:
        chain_points = chain["points"]
        assert all(footprint.boundary.distance(Point(point[:2])) < 1e-7 for point in chain_points)
        expected_chain_z = np.asarray(evaluate(model, chain_points[:, :2]), dtype=float).reshape(-1)
        assert np.max(np.abs(chain_points[:, 2] - expected_chain_z)) < 1e-5


def test_observed_host_form_rejects_broad_incompatible_structure():
    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    xs = np.arange(.25, 10, .5); ys = np.arange(.25, 3, .5)
    xx, yy = np.meshgrid(xs, ys)
    points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 15.)]
    result = build(footprint, points, base_z=2., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [10., 0.], "z": 8.}],
                   host_spec={"kind": "gable", "eaveZ": 8., "ridgeZ": 10., "confidence": "medium"})
    assert result["audit"]["observedHostForm"]["status"] == "host-form-contradicted-by-roof-returns"


def test_occluded_ridge_is_robustly_inferred_from_frontage_lidar():
    from general_world.roof_planes import evaluate
    from pipeline.roof_forms import build as build_form

    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    patches, _ = build_form(footprint, {"kind": "hip", "riseM": 2.6, "overhangM": 0.}, 8., 8.)
    model = patches[0]["model"]
    xs = np.arange(.25, 10, .5); ys = np.arange(.25, 6, .5)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.c_[xx.ravel(), yy.ravel()]
    points = np.c_[xy, evaluate(model, xy)]
    # Two tall returns stand in for sparse roof equipment and should not move the rise.
    points = np.vstack([points, [[4.75, 1.75, 13.], [5.25, 1.75, 12.8]]])
    result = build(footprint, points, base_z=2., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [10., 0.], "z": 8.}],
                   host_spec={"kind": "hip", "eaveZ": 8., "ridgeZ": None, "confidence": "high"})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-photo-host-form", audit
    assert audit["ridgeSource"] == "inferred-frontal-lidar"
    assert abs(audit["inferredRiseM"] - 2.6) < .25
    assert audit["application"] == "whole-regular-footprint"
    assert result["audit"]["analyticHostMesh"]["status"] == "exact-plane-envelope"
    planes = np.asarray(model["planes"], float)
    vertices = result["surface"].vertices
    face_values = np.c_[vertices[:, :2], np.ones(len(vertices))] @ planes.T
    face_residuals = np.abs(face_values[result["surface"].faces] - vertices[result["surface"].faces, 2, None])
    assert np.max(np.min(np.max(face_residuals, axis=1), axis=1)) < 1e-5
    peak = float(vertices[:, 2].max())
    ridge = vertices[np.isclose(vertices[:, 1], 3., atol=1e-6) & np.isclose(vertices[:, 2], peak, atol=1e-5)]
    assert len(ridge) >= 2
    assert np.ptp(ridge[:, 2]) < 1e-5
    patches, _ = build_form(footprint, {"kind": "hip", "riseM": audit["inferredRiseM"],
                                         "overhangM": 0.}, 8., 8.)
    closed = _close_analytic_with_bottom(result, footprint, patches[0]["model"], 2.)
    assert closed.is_watertight and closed.is_winding_consistent
    assert result["surface"].is_winding_consistent


def test_observed_component_strip_is_excluded_from_host_fit_only():
    from general_world.roof_planes import evaluate
    from pipeline.roof_forms import build as build_form

    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    patches, _ = build_form(footprint, {"kind": "hip", "riseM": 2.6, "overhangM": 0.}, 8., 8.)
    model = patches[0]["model"]
    xs = np.arange(.25, 10, .5); ys = np.arange(.25, 6, .5)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.c_[xx.ravel(), yy.ravel()]
    points = np.c_[xy, evaluate(model, xy)]
    # A broad raised frontage component contaminates most host-fit returns.
    component = ((xy[:, 0] >= 2.) & (xy[:, 0] <= 8.)
                 & (xy[:, 1] >= .75) & (xy[:, 1] <= 4.5))
    points[component, 2] += 3.
    result = build(footprint, points, base_z=2., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [10., 0.], "z": 8.}],
                   host_spec={"kind": "hip", "eaveZ": 8., "ridgeZ": None, "confidence": "high",
                              "excludeStrips": [{"normal": [0., -1.], "planeV": 0.,
                                                 "u0": 2., "u1": 8., "maxDepthM": 6.}]})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-photo-host-form", audit
    assert audit["excludedHostFitSampleCount"] > 100
    assert abs(audit["inferredRiseM"] - 2.6) < .25
    # The fit mask does not remove any source returns from shell input.
    assert result["audit"]["inputPointCount"] == len(points)


def test_analytic_roof_base_may_equal_the_measured_eave():
    """A zero-clearance eave is a valid closed roof shell, not an intersection."""
    import pytest
    from general_world.roof_planes import roof_solid
    from pipeline.roof_forms import build as build_form

    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    patches, _ = build_form(footprint, {"kind": "hip", "riseM": 2.6, "overhangM": 0.}, 8., 0.)
    model = patches[0]["model"]
    shell = roof_solid(footprint, model, 8.)
    assert shell.is_volume and shell.is_watertight
    assert shell.volume > 0
    with pytest.raises(ValueError, match="intersects"):
        roof_solid(footprint, model, 8.01)


def test_build_uses_exact_host_mesh_when_eave_equals_base_without_skirt():
    from general_world.roof_planes import evaluate
    from pipeline.roof_forms import build as build_form

    footprint = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    patches, _ = build_form(footprint, {"kind": "hip", "riseM": 2., "overhangM": 0.}, 8., 0.)
    model = patches[0]["model"]
    xs = np.arange(.25, 10, .5); ys = np.arange(.25, 6, .5)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.c_[xx.ravel(), yy.ravel()]
    points = np.c_[xy, evaluate(model, xy)]
    result = build(footprint, points, base_z=8., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [10., 0.], "z": 8.}],
                   host_spec={"kind": "hip", "eaveZ": 8., "ridgeZ": 10., "confidence": "high"})
    audit = result["audit"]["analyticHostMesh"]
    assert audit["status"] == "exact-plane-envelope", audit
    assert len(result["surface"].faces) > 0
    assert len(result["skirt"].faces) == 0


def test_real_roof_fixture_keeps_supported_frontage_when_rear_disagrees():
    """A genuine supported frontage may still receive a host form when the rear differs."""
    fixture = Path(__file__).parent / "tests" / "fixtures" / "shell-noisy-roof.fixture"
    data = np.load(fixture)
    footprint = Polygon(data["footprint"])
    points = data["points"].copy()

    # Pick a short side of the measured outline that tracks the actual boundary.
    # This fixture is a translated/sanitized roof sample set, with no case IDs.
    corners = np.asarray(footprint.minimum_rotated_rectangle.exterior.coords)[:4]
    candidates = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        line = LineString([a, b])
        distances = [footprint.boundary.distance(line.interpolate(t, normalized=True))
                     for t in np.linspace(0, 1, 5)]
        if line.length < 12 and max(distances) <= .8:
            candidates.append((line.length, a, b))
    assert candidates
    _, a, b = min(candidates, key=lambda q: q[0])
    tangent = (b - a) / np.linalg.norm(b - a)
    inward = np.array([-tangent[1], tangent[0]])
    midpoint = (a + b) / 2
    if not footprint.buffer(1e-5).covers(Point(*(midpoint + inward * .05))):
        inward = -inward

    # Keep the measured frontage intact and add a broad, supported height change
    # only behind it. This exercises the production fallback to the frontage zone.
    rear = (points[:, :2] - midpoint) @ inward > 6.
    assert rear.sum() >= 100
    points[rear, 2] += 5.
    result = build(footprint, points, base_z=0., cell_m=.5,
                   eave_constraints=[{"a": a.tolist(), "b": b.tolist(), "z": 10.7}],
                   host_spec={"kind": "hip", "eaveZ": 10.7, "ridgeZ": None,
                              "confidence": "medium"})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-photo-host-form", audit
    assert audit["ridgeSource"] == "inferred-frontal-lidar"
    assert audit["application"] == "observed-frontage-zone"
    assert audit["sampleCells"] >= 30


def test_front_facing_gable_fits_both_slopes_and_stops_at_rear_step():
    footprint = Polygon([(0, 0), (8, 0), (8, 12), (0, 12)])
    xs = np.arange(.25, 8., .5)
    ys = np.arange(.75, 12., .5)
    xx, yy = np.meshgrid(xs, ys)
    ridge_u = 5.25
    ridge = 10.6 + .08 * yy
    zz = ridge + np.where(xx < ridge_u, .5 * (xx - ridge_u), -.5 * (xx - ridge_u))
    zz += (yy >= 7.) * 1.0  # a separately supported raised rear part
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    result = build(footprint, points, base_z=7., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [8., 0.], "z": 8.}],
                   host_spec={"kind": "gable", "frontageKind": "gable-end", "eaveZ": 8.6,
                              "anchorUncertaintyM": .8, "ridgeZ": None, "confidence": "high"})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-observed-front-gable", audit
    assert audit["ridgeSource"] == "robust-two-slope-lidar-fit"
    assert audit["heldOutP95M"] <= .30
    assert min(audit["heldOutFractionWithin0_30MBySlope"]) >= .70
    assert abs(audit["leftSlope"] - .5) < .05
    assert abs(audit["rightSlope"] + .5) < .05
    assert abs(audit["ridgeUFromFrontageStartM"] - ridge_u) < .08
    # The two corners differ in height, and the uncertain edge reading is not
    # imposed as a flat hard boundary.
    assert abs(audit["fittedCornerEaveZ"][0] - 8.0) < .08
    assert abs(audit["fittedCornerEaveZ"][1] - 9.225) < .08
    assert 5.5 <= audit["zoneDepthM"] <= 7.
    ridge_break = audit["ridgeBreakline"]
    assert ridge_break["status"] == "inserted"
    assert ridge_break["insertedVertexCount"] > 0
    coeff = np.asarray(ridge_break["implicitPlaneDifference"], dtype=float)
    line_error = np.c_[result["surface"].vertices[:, :2], np.ones(len(result["surface"].vertices))] @ coeff
    ridge_vertices = np.abs(line_error) < 1e-7
    from general_world.roof_planes import evaluate
    ridge_xy = result["surface"].vertices[ridge_vertices, :2]
    ridge_z = result["surface"].vertices[ridge_vertices, 2]
    predicted_ridge = np.asarray(evaluate(audit["model"], ridge_xy), dtype=float).reshape(-1)
    in_front_zone = ridge_xy[:, 1] <= audit["zoneDepthM"] + 1e-6
    assert in_front_zone.any()
    assert np.max(np.abs(ridge_z[in_front_zone] - predicted_ridge[in_front_zone])) < 1e-7
    # No top triangle inside the supported host zone straddles the intersection.
    face_line = line_error[result["surface"].faces]
    face_centers = result["surface"].vertices[result["surface"].faces, :2].mean(axis=1)
    face_in_zone = face_centers[:, 1] <= audit["zoneDepthM"] + 1e-6
    assert not np.any((face_line[face_in_zone].min(axis=1) < -1e-8)
                      & (face_line[face_in_zone].max(axis=1) > 1e-8))
    assert _close_with_bottom(result, 7.).is_watertight
    front = result["surface"].vertices[np.isclose(result["surface"].vertices[:, 1], 0.)]
    assert np.max(front[:, 2]) > 10.4
    rear = result["surface"].vertices[np.isclose(result["surface"].vertices[:, 1], 10.)]
    assert np.median(rear[:, 2]) >= 10.7


def test_front_gable_stops_when_only_one_slope_changes_regime():
    footprint = Polygon([(0, 0), (8, 0), (8, 12), (0, 12)])
    xs = np.arange(.25, 8., .5)
    ys = np.arange(.75, 12., .5)
    xx, yy = np.meshgrid(xs, ys)
    ridge_u = 5.25
    ridge = 10.6 + .04 * yy
    zz = ridge + np.where(xx < ridge_u, .5 * (xx - ridge_u), -.5 * (xx - ridge_u))
    changed = (xx < ridge_u) & (yy >= 4.5)
    zz += changed * 1.0
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    result = build(footprint, points, base_z=7., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [8., 0.], "z": 8.}],
                   host_spec={"kind": "gable", "frontageKind": "gable-end", "eaveZ": 8.6,
                              "anchorUncertaintyM": .8, "ridgeZ": None, "confidence": "high"})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-observed-front-gable", audit
    assert audit["depthChangePoint"]["slope"] == "left", audit["depthChangePoint"]
    assert 3.5 <= audit["zoneDepthM"] <= 4.5
    # At the deeper mixed regime, the shell matches the same measured-data
    # completion with host fitting disabled: neither branch is overwritten.
    unmodeled = build(footprint, points, base_z=7., cell_m=.5,
                      eave_constraints=[{"a": [0., 0.], "b": [8., 0.], "z": 8.}])
    unmodeled_by_xy = {(round(float(p[0]), 8), round(float(p[1]), 8)): float(p[2])
                       for p in unmodeled["surface"].vertices}
    deep_points = [p for p in result["surface"].vertices if np.isclose(p[1], 8.)
                   and (round(float(p[0]), 8), round(float(p[1]), 8)) in unmodeled_by_xy]
    assert deep_points
    np.testing.assert_allclose([p[2] for p in deep_points],
                               [unmodeled_by_xy[(round(float(p[0]), 8), round(float(p[1]), 8))]
                                for p in deep_points], atol=1e-8)
    bins = {item["bin"]: item for item in audit["depthResidualAudit"]}
    for k in (audit["depthChangePoint"]["startBin"], audit["depthChangePoint"]["endBin"]):
        assert bins[k]["slopes"][0]["support"] >= 2
        assert bins[k]["slopes"][0]["inlierFraction"] < .65


def test_front_gable_rejects_long_side_orientation():
    footprint = Polygon([(0, 0), (8, 0), (8, 12), (0, 12)])
    points = _plane_points(footprint.bounds, step=.25)
    result = build(footprint, points, base_z=7., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [0., 12.], "z": 8.}],
                   host_spec={"kind": "gable", "frontageKind": "gable-end", "eaveZ": 8.,
                              "anchorUncertaintyM": .8, "ridgeZ": None, "confidence": "medium"})
    assert result["audit"]["observedHostForm"]["status"] != "applied-observed-front-gable"
    assert result["audit"]["observedHostForm"]["status"] != "frontage-not-gable-end-orientation"


def test_front_gable_can_face_long_footprint_edge():
    footprint = Polygon([(0, 0), (8, 0), (8, 12), (0, 12)])
    xs = np.arange(.25, 8., .5)  # depth runs inward from x=0
    ys = np.arange(.25, 12., .5)  # the gable slopes span the 12m frontage
    xx, yy = np.meshgrid(xs, ys)
    ridge_u = 7.1
    zz = 11. + .05 * xx + np.where(yy < ridge_u, .5 * (yy - ridge_u), -.5 * (yy - ridge_u))
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    result = build(footprint, points, base_z=7., cell_m=.5,
                   eave_constraints=[{"a": [0., 0.], "b": [0., 12.], "z": 8.45}],
                   host_spec={"kind": "gable", "frontageKind": "gable-end", "eaveZ": 8.45,
                              "anchorRole": "visible-roof-edge", "anchorUncertaintyM": .8,
                              "ridgeZ": None, "confidence": "high"})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-observed-front-gable", audit
    assert audit["frontageLengthM"] > 10.
    assert abs(audit["ridgeUFromFrontageStartM"] - ridge_u) < .1
    assert audit["cornerPriorUsedAsEave"] is False


def test_front_gable_local_fit_uses_measured_lidar_fixture():
    import json

    root = Path(__file__).parent
    fixture = root / "areas" / "pacific-heights" / "roof-evidence-baseline" / "roof-evidence" / "sf_201006_0116178.npz"
    profile_path = root / "areas" / "pacific-heights" / "regen" / "profiles-shell-review-v5.json"
    massing_path = root / "areas" / "pacific-heights" / "roof-shell-r11-raw" / "massing-dump.json"
    if not (fixture.exists() and profile_path.exists() and massing_path.exists()):
        import pytest
        pytest.skip("measured gable host fixture is unavailable")
    profile = next(p for p in json.loads(profile_path.read_text())["profiles"] if "0116178" in p["id"])
    item = next(q for q in json.loads(massing_path.read_text()) if q.get("id") == "sf_201006_0116178")
    n = np.asarray(profile["normal"], dtype=float); n /= np.linalg.norm(n)
    tangent = np.array([-n[1], n[0]])
    a = n * float(profile["planeV"]) + tangent * float(profile["frameU"][0])
    b = n * float(profile["planeV"]) + tangent * float(profile["frameU"][1])
    eave_z = float(item["baseRoofZ"])
    result = build(Polygon(item["outline"][0]), np.load(fixture)["points"], eave_z, cell_m=.5,
                   eave_constraints=[{"a": a.tolist(), "b": b.tolist(), "z": eave_z, "role": "eave"}],
                   host_spec={"kind": "gable", "frontageKind": "gable-end", "eaveZ": eave_z,
                              "anchorRole": "visible-roof-edge",
                              "anchorUncertaintyM": .8, "ridgeZ": None, "confidence": "medium"})
    audit = result["audit"]["observedHostForm"]
    assert audit["status"] == "applied-observed-front-gable", audit
    assert audit["heldOutP95M"] <= .30
    assert audit["seedDepthM"] < 3.1
    assert audit["cornerPriorUsedAsEave"] is False
    assert result["audit"]["boundaryConstraintVerticesApplied"] == 0
