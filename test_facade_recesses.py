import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from pipeline import facade_recesses


def _ctx():
    meshes = []
    blocks = []
    return ({'W': 4., 'Hs': 10., 'mat': 'stucco',
             # Facade tangent cross up points toward the street (-Y).
             'point': lambda x, z, d=0.: np.array([x, -d, z], float),
             'put': lambda mesh, key, material: meshes.append((mesh.copy(), key, material)),
             'block': lambda *args: blocks.append(args)}, meshes, blocks)


def test_observed_recess_cuts_host_wall_and_builds_cavity_surfaces():
    compiled = facade_recesses.compile_reading(
        {'recesses': [{'x0': .5, 'x1': 2.5, 'z0': 2., 'z1': 8., 'depthM': 1.8,
                       'kind': 'loggia',
                       'frontRailing': {'kind': 'metal', 'heightM': .95, 'horizontalRails': 2, 'postPitchM': .8},
                       'backOpenings': [
                           {'x0': 1., 'x1': 2., 'z0': 3., 'z1': 6., 'kind': 'glazed_door', 'lites': [2, 1]}]}]},
        lambda x: x / 4., lambda z: z / 10.)
    regions = facade_recesses.profile_regions({'recesses': compiled}, 4., 10., 10., 0., 4., box(0, 0, 4, 10))
    wall = box(0, 0, 4, 10)
    carved = facade_recesses.remove_host_wall(wall, regions)
    assert not carved.covers(Point(1.5, 4.5))
    assert carved.covers(Point(.25, 4.5))
    assert carved.area == 28.

    ctx, emitted, blocks = _ctx()
    facade_recesses.draw(ctx, regions, 'facade')
    assert emitted
    assert len(blocks) >= 5  # measured front railing members
    all_vertices = np.concatenate([mesh.vertices for mesh, _, _ in emitted])
    # The observed back plane and complete soffit/floor depth spans are present.
    assert np.isclose(all_vertices[:, 1].max(), 1.8)
    assert {2., 8.}.issubset(set(np.round(all_vertices[:, 2], 6)))
    # The back wall is cut around the measured glazed door, then pane/frame
    # surfaces are placed at the cavity back rather than on the street plane.
    back_triangles = [mesh for mesh, _, _ in emitted if np.allclose(mesh.vertices[:, 1], 1.8)]
    assert back_triangles
    assert all(np.all(mesh.face_normals[:, 1] < -.99) for mesh in back_triangles)
    door = box(1., 3., 2., 6.)
    for mesh in back_triangles:
        triangle = Polygon(mesh.vertices[:, [0, 2]])
        assert triangle.intersection(door).area < 1e-8
    assert any(material == 'glass' for _, _, material in emitted)
    assert any(material == 'stucco' for _, _, material in emitted)

    # The two returns, soffit and floor face into the cavity. These signs are
    # checked in an explicit facade frame with the viewer on the street side.
    left = [m for m, _, _ in emitted if np.allclose(m.vertices[:, 0], .5) and np.ptp(m.vertices[:, 1]) > 1.]
    right = [m for m, _, _ in emitted if np.allclose(m.vertices[:, 0], 2.5) and np.ptp(m.vertices[:, 1]) > 1.]
    soffit = [m for m, _, _ in emitted if np.allclose(m.vertices[:, 2], 8.) and np.ptp(m.vertices[:, 1]) > 1.]
    floor = [m for m, _, _ in emitted if np.allclose(m.vertices[:, 2], 2.) and np.ptp(m.vertices[:, 1]) > 1.]
    assert len(left) == len(right) == len(soffit) == len(floor) == 1
    assert np.all(left[0].face_normals[:, 0] > .99)
    assert np.all(right[0].face_normals[:, 0] < -.99)
    assert np.all(soffit[0].face_normals[:, 2] < -.99)
    assert np.all(floor[0].face_normals[:, 2] > .99)


def test_ordinary_balcony_does_not_create_a_wall_cutout():
    regions = facade_recesses.profile_regions(
        {'balconies': [{'rect': [.1, .2, .6, .8], 'depthM': 1.}], 'recesses': []},
        4., 10., 10., 0., 4., box(0, 0, 4, 10))
    wall = box(0, 0, 4, 10)
    assert not regions
    assert facade_recesses.remove_host_wall(wall, regions).equals(wall)


def test_recess_is_skipped_if_it_crosses_a_separately_owned_bay():
    profile = {'recesses': [{'rect': [.1, .2, .7, .8], 'depthM': 1.5}]}
    regions = facade_recesses.profile_regions(
        profile, 4., 10., 10., 0., 4., box(0, 0, 4, 10),
        bays=[{'x0': 1.5, 'x1': 2.5, 'z0': 4., 'z1': 6., 'p': 1.}])
    assert not regions


def test_recess_spanning_wall_tiles_has_no_fake_partition_returns():
    profile = {'recesses': [{'rect': [.25, .2, .75, .8], 'depthM': 1.5}]}
    tile_a = facade_recesses.profile_regions(profile, 4., 10., 10., 0., 2., box(0, 0, 2, 10))
    tile_b = facade_recesses.profile_regions(profile, 4., 10., 10., 2., 4., box(2, 0, 4, 10))
    assert tile_a[0]['leftReturn'] and not tile_a[0]['rightReturn']
    assert not tile_b[0]['leftReturn'] and tile_b[0]['rightReturn']
    ctx, emitted, _ = _ctx()
    facade_recesses.draw(ctx, tile_a + tile_b, 'facade')
    back = [m for m, _, _ in emitted if np.allclose(m.vertices[:, 1], 1.5)]
    assert back
    wall_face = unary_union([Polygon(m.vertices[:, [0, 2]]) for m in back])
    assert wall_face.symmetric_difference(box(1., 2., 3., 8.)).area < 1e-8
    # Tile x=2 has no vertical return plane extending through the cavity depth.
    assert not any(np.allclose(m.vertices[:, 0], 2.) and np.ptp(m.vertices[:, 1]) > 1.
                   for m, _, _ in emitted)


def test_hillside_recess_keeps_negative_wall_frame_elevation():
    profile = {'recesses': [{'rect': [.2, -.1, .6, .2], 'depthM': 1.}]}
    regions = facade_recesses.profile_regions(profile, 4., 10., 10., 0., 4., box(0, -2, 4, 10))
    assert regions[0]['bounds'][1] == -1.
    assert regions[0]['bounds'][3] == 2.


def test_recess_inherits_local_finish_and_keeps_observed_surface_overrides():
    ctx, emitted, _ = _ctx()
    ctx['mat'] = 'brick_modular#783f31'
    ctx['wall_mat'] = lambda x, z: 'render_white#e8dfcf' if x < 3. else ctx['mat']
    regions = facade_recesses.profile_regions(
        {'recesses': [{'rect': [.125, .2, .625, .8], 'depthM': 1.5,
                       'floorMaterial': 'concrete'}]},
        4., 10., 10., 0., 4., box(0, 0, 4, 10))
    facade_recesses.draw(ctx, regions, 'facade')
    assert all(material == 'render_white#e8dfcf' for mesh, _, material in emitted
               if not np.allclose(mesh.vertices[:, 2], 2.))
    floors = [(mesh, material) for mesh, _, material in emitted
              if np.allclose(mesh.vertices[:, 2], 2.)]
    assert floors and all(material == 'concrete' for _, material in floors)
