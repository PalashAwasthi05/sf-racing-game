"""Adjacent facade cells share one solid instead of exposing AO-visible internal faces."""

import numpy as np
import trimesh
from shapely.geometry import box, Polygon

from pipeline.facade_grammar import compile_wall, facade_prism


def test_pier_leaves_grouped_windows_and_observed_arch_open():
    from pipeline.facade_grammar import openings, opening_outline
    emitted=[]
    ctx={'polys':lambda g:[g], 'cube':lambda *a,**kw:None,
         'put':lambda mesh,key,mat:emitted.append((mat,mesh)), 'cylinder':lambda *a,**kw:None}
    profile={'id':'pier-fixture','normal':[0.,-1.],'sourceIds':['measured'],
             'frameU':[0,10],'heightRefM':8.,'material':'stucco',
             'piers':[{'x':.5,'widthM':4.,'projectionM':.1,'material':'pier-finish','y':[0.,1.]}],
             'groups':[{'columns':1,'rows':2,'region':[.4,.42,.6,.9],
                        'style':{'kind':'window','ribbon':False,'openingRatio':.7,'openingHeightRatio':.6}}],
             'openings':[{'rect':[.36,.04,.64,.35],'style':{'kind':'window','head':'arch'}}]}
    compile_wall(ctx,{'g':box(0,0,10,5),'id':'fixture'},np.array([0.,0.]),np.array([10.,0.]),0.,8.,profile,[])
    pier=trimesh.util.concatenate([m for mat,m in emitted if mat=='pier-finish'])
    measured=sum(opening_outline(op,10.,8.).area for op in openings(profile,[],10.,8.))
    assert np.isclose(pier.volume,(4.*8.-measured)*.4)
    assert pier.is_watertight


def test_wall_winding_survives_point_touching_aperture_and_rotated_world_frame():
    # A material boundary/window corner can leave this valid point contact.
    # fix_normals cannot infer volume orientation on its shared four-face edge.
    elevation = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)],
                        holes=[[(0, 2), (1, 1), (1, 3)]])
    assert elevation.is_valid
    for angle in (0., .63, 2.4):
        normal = np.array([np.sin(angle), -np.cos(angle)])
        tangent = np.array([-normal[1], normal[0]])
        origin = np.array([-215., -406., -51.])
        wall = facade_prism(elevation, origin, tangent, normal, .3)
        center = wall.triangles_center
        plane = (center[:, :2] - origin[:2]) @ normal
        front = np.abs(plane) < 1e-7
        back = np.abs(plane + .3) < 1e-7
        assert front.any() and back.any()
        assert np.all(wall.face_normals[front, :2] @ normal > .999)
        assert np.all(wall.face_normals[back, :2] @ normal < -.999)
        assert abs(wall.area_faces[front].sum() - elevation.area) < 1e-7
        assert abs(wall.volume - elevation.area * .3) < 1e-7
        # A pinched edge is shared, but there are no unpaired open edges.
        counts = np.bincount(wall.edges_unique_inverse)
        assert not np.any(counts == 1)


def test_wall_cells_union_without_filling_opening_or_leaving_partition_faces():
    emitted = []

    def cube(center, size, key, material, angle=0.):
        mesh = trimesh.creation.box(extents=size)
        if angle:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0., 0., 1.]))
        mesh.apply_translation(center)
        emitted.append((key, material, mesh))

    def put(mesh, key, material):
        emitted.append((key, material, mesh.copy()))

    ctx = {
        'polys': lambda g: [g], 'cube': cube, 'put': put,
        'cylinder': lambda *args, **kwargs: None,
    }
    frontage = box(0., 0., 10., 8.)
    profile = {
        'id': 'wall-union', 'normal': [0., -1.], 'sourceIds': ['test'],
        'heightRefM': 8., 'material': 'stucco',
        'openings': [{'rect': [.2, .25, .4, .5], 'style': {'kind': 'window'}}],
    }

    result = compile_wall(ctx, {'g': frontage, 'id': 'wall-union'},
                          np.array([0., 0.]), np.array([10., 0.]), 0., 8., profile, [])
    assert result is not None
    wall_meshes = [mesh for key, material, mesh in emitted
                   if key.startswith('buildings__') and material == 'stucco']
    assert wall_meshes
    wall = trimesh.util.concatenate(wall_meshes)

    # The exact 2 m x 2 m rectangular aperture remains empty in the front skin.
    centers = wall.triangles_center
    normals = wall.face_normals
    front_faces = (np.abs(normals[:, 1]) > .999) & (np.abs(centers[:, 1]) < 1e-6)
    front_area = float(wall.area_faces[front_faces].sum())
    assert abs(front_area - (10. * 8. - 2. * 2.)) < 1e-6

    # x=5 was only a partition coordinate in the old slice grid; it must not
    # survive as an internal vertical face in the merged wall solid.
    internal_x_faces = (np.abs(normals[:, 0]) > .999) & (np.abs(centers[:, 0] - 5.) < 1e-6)
    assert not internal_x_faces.any()


def test_observed_upper_cladding_reaches_actual_wall_top_without_inventing_crop_height():
    emitted = []
    ctx = {'polys': lambda g: [g], 'cube': lambda *a, **kw: None,
           'put': lambda mesh, key, material: emitted.append((key, material, mesh.copy())),
           'cylinder': lambda *a, **kw: None}
    profile = {'id': 'upper-material', 'normal': [0., -1.], 'sourceIds': ['test'],
               'heightRefM': 8., 'material': 'stucco',
               'panels': [{'rect': [0, .5, 1, 1], 'material': 'timber_dark', 'toWallTop': True}]}
    compile_wall(ctx, {'g': box(0, 0, 10, 8), 'id': 'upper-material'},
                 np.array([0., 0.]), np.array([10., 0.]), 0., 10., profile, [])
    upper = trimesh.util.concatenate([m for key, mat, m in emitted if key.startswith('buildings__') and mat == 'timber_dark'])
    center = upper.triangles_center
    front = (np.abs(upper.face_normals[:, 1]) > .999) & (np.abs(center[:, 1]) < 1e-6)
    assert abs(float(upper.area_faces[front].sum()) - 60.) < 1e-6
    assert upper.bounds[1, 2] == 10.
    assert profile['panels'][0]['rect'][3] == 1  # source frame stays unchanged

