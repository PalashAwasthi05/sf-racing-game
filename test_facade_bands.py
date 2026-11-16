"""Measured facade bands follow the bay support at their own elevation."""

import numpy as np

from pipeline import bays
from pipeline.facade_bands import (band_footprint, bay_cap_replaced_by_band,
                                   should_emit_default_plinth, support_depth_at,
                                   support_path, sweep_band, sweep_profile_band,
                                   sweep_section)


def _bay(shape="square", x0=3., x1=6., z0=0., z1=8., projection=1.):
    return {"x0": x0, "x1": x1, "z0": z0, "z1": z1,
            "p": projection, "shape": shape}


def _point(x, z, d):
    # A rotated/skewed affine frame also exercises winding correction.
    return np.array([x + d, 3. + z, 2. * d], dtype=float)


def test_support_path_tracks_square_angled_and_curved_bays():
    for shape in ("square", "angled", "curved"):
        path = support_path(0., 10., 5., [_bay(shape=shape)])
        assert path.is_simple
        assert np.isclose(support_depth_at(path, 2.), 0.)
        assert np.isclose(support_depth_at(path, 4.5), 1.)
        assert np.isclose(support_depth_at(path, 8.), 0.)


def test_support_is_flat_when_band_elevation_is_outside_bay_face():
    bay = _bay(z0=2., z1=4., projection=.9)
    path = support_path(0., 10., 4.2, [bay])
    assert np.isclose(path.length, 10.)
    assert np.isclose(support_depth_at(path, 4.5), 0.)


def test_polygon_support_query_returns_none_outside_its_x_extent():
    footprint = band_footprint(support_path(2., 6., 1.), .1, .2)
    assert support_depth_at(footprint, 1.) is None
    assert support_depth_at(footprint, 7.) is None


def test_overlapping_bays_union_to_one_outer_support_path():
    path = support_path(0., 10., 5., [
        _bay(shape="square", x0=2., x1=5., projection=1.),
        _bay(shape="angled", x0=4., x1=7., projection=.8),
    ])
    assert path.is_simple
    assert np.isclose(support_depth_at(path, 3.), 1.)
    assert np.isclose(support_depth_at(path, 5.5), .8)


def test_band_sweep_and_relief_support_share_bay_profile():
    bay = _bay(shape="curved", projection=.8)
    path = support_path(0., 10., 5., [bay])
    meshes = sweep_band(_point, 0., 10., 4.9, .2, .2, .2, [bay])
    assert len(meshes) == 1
    mesh = meshes[0]
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0.
    # Local d is mapped into world y by 2*d. The outer extent is the same
    # elevation-specific profile used by a relief anchor.
    # Reliefs can query the actual swept footprint, not a separate guessed
    # offset, and therefore share exactly the same support surface.
    footprint = band_footprint(path, .2, .2)
    assert np.isclose(support_depth_at(footprint, 4.5), mesh.bounds[1, 2] / 2., atol=1e-6)
    assert np.isclose(mesh.bounds[1, 0], 10. + .2 + .1, atol=1e-6)


def test_section_sweeps_each_element_on_its_own_active_support():
    bay = _bay(shape="angled", z0=2., z1=4., projection=.7)
    meshes = sweep_section(_point, 0., 10., 2.5,
                           [(0., .2, .3), (2., .2, .3)], [bay])
    assert len(meshes) == 2
    assert all(mesh.is_watertight and mesh.volume > 0. for mesh in meshes)
    # First course is on the bay; the upper course is above the bay and lies on
    # the flat facade. Projection and course thickness remain in both bounds.
    assert meshes[0].bounds[1, 2] > meshes[1].bounds[1, 2]
    # Section rise is the centerline elevation, matching the grammar's
    # z + rise - height/2 bottom-elevation convention.
    assert np.isclose(meshes[0].bounds[0, 1], 2.4 + 3.)
    assert np.isclose(meshes[1].bounds[0, 1], 4.4 + 3.)


def test_default_plinth_requires_an_unsourced_legacy_profile():
    assert should_emit_default_plinth({"sourceIds": []})
    assert not should_emit_default_plinth({"sourceIds": ["mapillary:photo"]})
    assert not should_emit_default_plinth({"sourceIds": ["survey:reading"]})


def test_source_profile_keeps_its_explicit_string_course():
    profile = {"sourceIds": ["mapillary:photo"], "bands": [{"zM": 1.62}]}
    assert not should_emit_default_plinth(profile)
    course = sweep_band(_point, 0., 10., 1.51, .22, .7, .3)
    assert len(course) == 1
    assert course[0].is_watertight and course[0].is_winding_consistent
    assert np.isclose(course[0].bounds[0, 1], 3. + 1.51)
    assert np.isclose(course[0].bounds[1, 1], 3. + 1.73)


def test_profile_band_wrap_uses_source_plane_once_across_segmented_bay():
    for shape in ("curved", "angled", "square"):
        bay = {**_bay(shape=shape, x0=3., x1=6., z0=0., z1=8., projection=1.),
               "index": 2, "cap": "none"}

        def point_at(plane):
            return lambda x, z, d: np.array([x, 3. + z, plane + d], dtype=float)

        emitted = set()
        identity = ("source", "profile", "course", 0.)
        # First compiled edge is a mapped segment to the right of the bay
        # center. It still owns the one complete source-plane bay course.
        first = sweep_profile_band(point_at(.76), point_at(0.), 0., 10.,
                                   5.5, 8., 8.2, .1, .18, .1, [bay],
                                   emitted, identity)
        assert first
        assert sum(mesh.bounds[1, 2] > .95 for mesh in first) == 1
        assert all(mesh.is_watertight and mesh.is_winding_consistent for mesh in first)

        # An adjacent host segment and an intersecting mapped bay segment do
        # not recreate the cap; their ordinary non-bay runs stay on their own
        # planes, while the bay itself contributes no flat duplicate.
        second = sweep_profile_band(point_at(.01), point_at(0.), 0., 10.,
                                    0., 5.5, 8.2, .1, .18, .1, [bay],
                                    emitted, identity)
        third = sweep_profile_band(point_at(.32), point_at(0.), 0., 10.,
                                   3., 5.5, 8.2, .1, .18, .1, [bay],
                                   emitted, identity)
        assert sum(mesh.bounds[1, 2] > .95 for mesh in second + third) == 0
        assert len(emitted) == 1
        assert any(np.allclose(mesh.bounds[:, 2], [.77, .76 + .1 + .09])
                   for mesh in first)
        assert any(np.allclose(mesh.bounds[:, 2], [.02, .01 + .1 + .09])
                   for mesh in second)


def test_profile_section_parts_each_wrap_once_but_keep_distinct_material_runs():
    bay = {**_bay(shape="curved", x0=3., x1=6., z0=0., z1=8., projection=.8),
           "index": 0}

    def point_at(plane):
        return lambda x, z, d: np.array([x, z, plane + d], dtype=float)

    emitted = set()
    for part in (0, 1):
        for _ in range(2):  # repeated wall partitions only emit the bay once
            meshes = sweep_profile_band(point_at(.3), point_at(0.), 0., 10.,
                                        5., 8., 8.2 + part * .07, .1, .12, .08,
                                        [bay], emitted, ("profile", "band", part))
            assert meshes
    assert len(emitted) == 2


def test_profile_band_equal_horizontal_bounds_sort_without_merging_bays():
    """Stacked supported bays can share x extents and both cross one course."""
    def point_at(plane):
        return lambda x, z, d: np.array([x, z, plane + d], dtype=float)

    bays_at_same_x = [
        {**_bay(shape="curved", x0=3., x1=6., z0=0., z1=5., projection=.55),
         "index": 2},
        {**_bay(shape="angled", x0=3., x1=6., z0=4., z1=9., projection=.9),
         "index": 7},
    ]

    def emit(order):
        emitted = set()
        meshes = sweep_profile_band(
            point_at(.2), point_at(0.), 0., 10., 4.4, 4.6, 4.6,
            .1, .12, .08, order, emitted, ("same-horizontal-bounds",))
        identity = ("same-horizontal-bounds",)
        assert emitted == {(identity, 2), (identity, 7)}
        wraps = [mesh for mesh in meshes if mesh.bounds[1, 2] > .5]
        assert len(wraps) == 2
        assert all(mesh.is_watertight and mesh.is_winding_consistent
                   for mesh in wraps)
        return sorted((tuple(np.round(mesh.bounds.ravel(), 6)),
                       round(float(mesh.volume), 6)) for mesh in wraps)

    assert emit(bays_at_same_x) == emit(list(reversed(bays_at_same_x)))


def test_measured_band_replaces_only_full_width_overlapping_bay_cornice():
    bay = {"x0": 3., "x1": 6., "z1": 8., "cap": "cornice"}
    overlapping = {"height": .825, "thicknessM": .2, "xRange": [.2, .7]}
    separate = {"height": .79, "thicknessM": .1, "xRange": [.2, .7]}
    partial = {"height": .825, "thicknessM": .2, "xRange": [.2, .55]}
    assert bay_cap_replaced_by_band(bay, [overlapping], 10., 10.)
    assert not bay_cap_replaced_by_band(bay, [separate], 10., 10.)
    assert not bay_cap_replaced_by_band(bay, [partial], 10., 10.)


def test_compiled_segmented_frontage_emits_one_source_anchored_bay_band():
    from shapely.geometry import Polygon
    from pipeline.facade_grammar import compile_wall

    meshes = []
    ctx = {"polys": lambda geom: [geom],
           "put": lambda mesh, key, material: meshes.append((mesh, key, material)),
           "cube": lambda *args: None}
    building = {"id": "segmented-bay-band", "g": Polygon([
        (0., 0.), (10., 0.), (10., 10.), (7., 10.), (7., 11.),
        (3., 11.), (3., 10.), (0., 10.)])}
    profile = {"id": "measured-course", "normal": [0., 1.],
               "frameU": [-10., 0.], "planeV": 10., "heightRefM": 5.,
               "sourceIds": ["photo"], "material": "host",
               "bays": [{"rect": [.4, 0., .6, .8], "projectionM": 1.,
                         "shape": "curved", "cap": "none"}],
               "bands": [{"height": .85, "thicknessM": .1, "depthM": .18,
                          "projectionM": .1, "material": "course"}]}

    # Deliberately compile an ordinary left segment first, away from the bay
    # center, then the mapped bay-front edge.
    compile_wall(ctx, building, np.array([10., 10.]), np.array([7., 10.]),
                 0., 5., profile, [])
    before = [(mesh, material) for mesh, _, material in meshes if material == "course"]
    assert len(before) == 2   # flat course + one complete bay wrap
    is_bay_wrap = lambda mesh: mesh.bounds[0, 0] < 4. and mesh.bounds[1, 0] > 6.
    bay_caps = [mesh for mesh, _ in before if is_bay_wrap(mesh)]
    assert len(bay_caps) == 1
    assert bay_caps[0].is_watertight and bay_caps[0].is_winding_consistent

    compile_wall(ctx, building, np.array([7., 11.]), np.array([3., 11.]),
                 0., 5., profile, [])
    after = [mesh for mesh, _, material in meshes if material == "course"]
    assert len(after) == len(before) + 2  # the uncovered returns remain on their own plane
    after_bay_caps = [mesh for mesh in after if is_bay_wrap(mesh)]
    assert len(after_bay_caps) == 1
    assert np.allclose(after_bay_caps[0].bounds, bay_caps[0].bounds)


def test_bay_slabs_use_affine_frame_and_keep_curved_facet_shell_thickness():
    meshes = []

    def put(mesh, key, material):
        meshes.append(mesh)

    bays.draw(_point, put, "wall", "detail",
              {"x0": 1., "x1": 4., "z0": 1., "z1": 6.,
               "p": .8, "shape": "curved", "cap": "cornice"},
              [], "brick", "stone")
    assert meshes
    assert all(mesh.is_watertight for mesh in meshes)
    assert all(mesh.is_winding_consistent for mesh in meshes)
    assert all(mesh.volume > 0. for mesh in meshes if mesh.is_watertight)
