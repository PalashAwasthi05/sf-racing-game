import numpy as np
import trimesh

from pipeline import bays
from pipeline.bay_replacement import clip_structural_mesh, replacement_record


def _replacement(shape="curved", x0=0., x1=4., projection=1.):
    _shape, support = bays.corners(x0, x1, projection, shape, 45.)
    return replacement_record(origin=[0., 0.], tangent=[1., 0.], normal=[0., 1.],
                              plane=0., zBase=0., xRange=(x0, x1),
                              zRange=(0., 3.), support=support,
                              sourceId="source", profileId="profile", bayIndex=0)


def _box(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def test_curved_skin_replaces_only_forward_wall_and_keeps_source_backing():
    bay = _replacement("curved")
    backing = _box([4., .3, 3.], [2., -.15, 1.5])
    forward = _box([4., .3, 3.], [2., 1.15, 1.5])

    kept = clip_structural_mesh(backing, [bay])
    removed = clip_structural_mesh(forward, [bay])

    assert kept is not None and kept.is_watertight
    assert np.isclose(kept.volume, backing.volume)
    assert removed is None or removed.volume < forward.volume * .15


def test_curved_shell_cuts_only_skin_intersections_not_whole_bay_backing():
    bay = _replacement("curved")
    # This wall lies behind the curved center but reaches through the bay skin
    # at the shallow endpoints. The center backing remains, and the cut is a
    # closed solid rather than a hole through the whole wall.
    mesh = _box([4., .3, 3.], [2., .5, 1.5])
    clipped = clip_structural_mesh(mesh, [bay])
    assert clipped is not None and clipped.is_watertight
    assert mesh.volume * .5 < clipped.volume < mesh.volume


def test_angled_return_and_unrelated_wall_are_left_by_plan_projection():
    bay = _replacement("angled")
    # The return lies at the measured bay's sloped side. A parallel wall that
    # starts beyond its measured x/z extent is unrelated and remains intact.
    side = _box([.7, .3, 3.], [.65, .82, 1.5])
    unrelated = _box([2., .3, 3.], [8., 1.15, 1.5])
    cut_side = clip_structural_mesh(side, [bay])
    kept_unrelated = clip_structural_mesh(unrelated, [bay])
    assert cut_side is None or cut_side.volume < side.volume
    assert kept_unrelated is not None and np.isclose(kept_unrelated.volume, unrelated.volume)


def test_missing_validated_replacement_never_cuts_a_wall():
    wall = _box([4., .3, 3.], [2., 1.15, 1.5])
    untouched = clip_structural_mesh(wall, [])
    assert untouched is wall
    assert np.isclose(untouched.volume, wall.volume)


def test_facade_compile_stages_bay_before_flushing_structural_wall_meshes():
    from shapely.geometry import Polygon
    from pipeline import facades

    emitted = []
    original_cube = lambda *args: None
    original_put = lambda mesh, key, material, **kwargs: emitted.append((mesh, key, material))
    footprint = Polygon([(0., 0.), (3., 0.), (3., -.7), (7., -.7),
                         (7., 0.), (10., 0.), (10., 10.), (0., 10.)])
    profile = {"id": "observed-bay", "match": {"featureId": "part"},
               "normal": [0., -1.], "planeV": 0., "frameU": [0., 10.],
               "heightRefM": 6., "sourceIds": ["source-photo"],
               "material": "brick_modular", "groups": [], "openings": [],
               "bays": [{"rect": [.4, .2, .6, .8], "projectionM": .7,
                         "shape": "curved", "cap": "cornice"}]}
    ctx = {"PROFILES": [profile], "FITS": [], "cube": original_cube,
           "put": original_put, "cylinder": lambda *args: None,
           "polys": lambda geom: [geom] if geom.geom_type == "Polygon" else list(geom.geoms),
           "lines": lambda geom: [geom] if geom.geom_type == "LineString" else list(geom.geoms),
           "boundary": Polygon([(-20., -20.), (20., -20.), (20., 20.), (-20., 20.)]),
           "all_buildings": Polygon()}
    building = {"id": "part", "sourceId": "part", "g": footprint,
                "sourceG": footprint, "osm": None, "full_area": 100.,
                "massHeightM": 6.}

    records, _ = facades.compile(ctx, building, 0., 6., 6., {}, [])

    assert any(record.get("grammarId") == "observed-bay" for record in records)
    assert ctx["put"] is original_put and ctx["cube"] is original_cube
    assert "_observedBayReplacements" not in ctx
    assert emitted
    # Deferral keeps bay geometry until the wall pass has finished; its curved
    # shell is therefore emitted after all staged structural host/return faces.
    is_curved_shell = lambda mesh, key: (
        key == "buildings__part__observed_observed-bay"
        and np.isclose(mesh.bounds[0, 2], 1.2) and np.isclose(mesh.bounds[1, 2], 4.8)
        and mesh.bounds[0, 1] < -.1)
    structural = [index for index, (_mesh, key, _) in enumerate(emitted)
                  if key == "buildings__part"]
    assert structural
    bay_shell = [index for index, (mesh, key, _) in enumerate(emitted)
                 if is_curved_shell(mesh, key)]
    assert bay_shell
    assert min(bay_shell) > max(structural)
