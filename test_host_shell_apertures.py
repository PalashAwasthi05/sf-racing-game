"""Synthetic checks for the external, source-camera aperture ownership prototype."""

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

from regen.host_shell_apertures import (
    SOURCE_TO_GLB,
    TriangleIndex,
    clip_shell_mesh,
    opening_host_conflicts,
    opening_outline_uv,
    profile_tangent,
    reveal_mesh,
    build_measured_receiver,
)


def _profile():
    return {"id": "regen-fixture", "match": {"featureId": "fixture"},
            "normal": [0., 1.], "frameU": [-2., 2.], "planeV": 0.,
            "heightRefM": 4., "openings": []}


def _wall_sheet(y=0.):
    vertices = np.array([[-2., y, 0.], [2., y, 0.],
                         [2., y, 4.], [-2., y, 4.]])
    return trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]],
                           process=False)


def _projected_area(mesh, profile):
    t = profile_tangent(profile)
    total = 0.
    for face in mesh.faces:
        src = mesh.vertices[face]
        tri = np.column_stack((src[:, :2] @ t, src[:, 2]))
        total += Polygon(tri).area
    return total


def test_clip_removes_only_measured_parallel_slab_and_preserves_backing():
    profile = _profile()
    opening = {"rect": [.25, .25, .75, .75], "style": {"kind": "window"}}
    outline = opening_outline_uv(profile, opening, 0.)
    front = _wall_sheet(0.)
    backing = _wall_sheet(-.5)

    clipped, stats = clip_shell_mesh(front, profile, outline, [-.02, .02], 0.)

    assert stats["affectedTriangles"] == 2
    assert np.isclose(stats["removedProjectedAreaM2"], outline.area, atol=1e-7)
    assert np.isclose(_projected_area(clipped, profile), 16. - outline.area,
                      atol=1e-7)
    assert np.isclose(_projected_area(backing, profile), 16.)
    # Every remaining front triangle avoids the observed opening; nearby wall
    # surface and the separate backing sheet remain present.
    t = profile_tangent(profile)
    for face in clipped.faces:
        src = clipped.vertices[face]
        projected = Polygon(np.column_stack((src[:, :2] @ t, src[:, 2])))
        assert projected.intersection(outline).area < 1e-8
    assert clipped.bounds[0, 1] == 0.
    assert len(backing.faces) == 2


def test_clip_preserves_authored_vertex_normals_on_split_textured_triangles():
    profile=_profile()
    opening={"rect":[.25,.25,.75,.75],"style":{"kind":"window"}}
    front=_wall_sheet(0.)
    smooth=np.asarray([.2,1.,.3]);smooth/=np.linalg.norm(smooth)
    front.vertex_normals=np.tile(smooth,(len(front.vertices),1))
    from trimesh.visual.texture import TextureVisuals
    front.visual=TextureVisuals(uv=np.asarray([[0.,0.],[1.,0.],[1.,1.],[0.,1.]]))
    clipped,stats=clip_shell_mesh(front,profile,
        opening_outline_uv(profile,opening,0.),[-.02,.02],0.)
    assert stats['affectedTriangles']==2
    assert clipped.visual.uv.shape==(len(clipped.vertices),2)
    assert np.allclose(clipped.vertex_normals,
                       np.tile(smooth,(len(clipped.vertices),1)),atol=1e-6)


def test_noop_when_opening_outline_does_not_intersect_shell_slab():
    profile = _profile()
    opening = {"rect": [.25, .25, .75, .75], "style": {"kind": "window"}}
    clipped, stats = clip_shell_mesh(_wall_sheet(0.), profile,
                                     opening_outline_uv(profile, opening, 0.),
                                     [2., 3.], 0.)
    assert stats == {"removedProjectedAreaM2": 0., "affectedTriangles": 0}
    assert len(clipped.faces) == 2


def test_arch_cut_contour_does_not_remove_unobserved_square_corners():
    profile = _profile()
    opening = {"rect": [.25, .25, .75, .75],
               "style": {"kind": "window", "head": "arch", "springRatio": .75}}
    outline = opening_outline_uv(profile, opening, 0.)
    assert outline.is_valid
    assert outline.area < 4.
    assert not outline.covers(Point(-1., 3.))
    assert outline.covers(Point(0., 2.95))


def test_reveal_bridge_spans_measured_shell_to_profile_plane():
    profile = _profile()
    opening = {"rect": [.25, .25, .75, .75], "style": {"kind": "window"}}
    outline = opening_outline_uv(profile, opening, 0.)
    reveal = reveal_mesh(profile, outline, [-.4, .1], 0.)
    assert len(reveal.faces) >= 8
    assert np.isclose(reveal.bounds[0, 1], 0.)
    assert np.isclose(reveal.bounds[1, 1], .1)
    assert reveal.is_winding_consistent


def test_conflict_requires_matching_feature_host_and_measured_domain_overlap():
    profile = _profile()
    profile["openings"] = [{"rect": [.25, .25, .75, .75],
                           "style": {"kind": "window"}}]
    wall = _wall_sheet(0.)
    scene = trimesh.Scene()
    wall_glb = wall.copy()
    wall_glb.apply_transform(SOURCE_TO_GLB)
    scene.add_geometry(wall_glb, node_name="buildings__fixture___shell",
                       geom_name="fixture-shell")
    job = {"from": [0., 5., 2.], "at": [0., 0., 2.], "vfovDeg": 50.,
           "width": 1000, "height": 1000}
    conflicts = opening_host_conflicts(scene, profile, job, 0.)
    assert len(conflicts) == 1
    assert conflicts[0]["firstHit"]["node"] == "buildings__fixture___shell"
    assert conflicts[0]["outlineUV"].buffer(.025).covers(
        Point(*conflicts[0]["openingHitUV"]))

    other = trimesh.Scene()
    other.add_geometry(wall_glb.copy(), node_name="buildings__neighbor___shell",
                       geom_name="neighbor-shell")
    assert opening_host_conflicts(other, profile, job, 0.) == []


def test_clipped_ray_reaches_backing_while_neighboring_host_stays_solid():
    profile = _profile()
    profile["openings"] = [{"rect": [.25, .25, .75, .75],
                           "style": {"kind": "window"}}]
    host = _wall_sheet(0.)
    backing = _wall_sheet(-.5)
    scene = trimesh.Scene()
    host_glb = host.copy();host_glb.apply_transform(SOURCE_TO_GLB)
    back_glb = backing.copy();back_glb.apply_transform(SOURCE_TO_GLB)
    scene.add_geometry(host_glb,node_name="buildings__fixture___shell",
                       geom_name="fixture-host")
    scene.add_geometry(back_glb,node_name="buildings__fixture___backing",
                       geom_name="fixture-backing")
    job = {"from": [0., 5., 2.], "at": [0., 0., 2.], "vfovDeg": 50.,
           "width": 1000, "height": 1000}
    conflict = opening_host_conflicts(scene,profile,job,0.)[0]
    slab = [-.02,.02]
    clipped,stats = clip_shell_mesh(host,profile,conflict["outlineUV"],slab,0.)
    assert stats["affectedTriangles"] == 2
    host_glb = clipped.copy();host_glb.apply_transform(SOURCE_TO_GLB)
    scene.geometry["fixture-host"] = host_glb

    index = TriangleIndex.from_scene(scene)
    through_opening = index.ray_hits([0.,5.,2.],[0.,-1.,2.])
    beside_opening = index.ray_hits([0.,5.,2.],[1.5,-1.,2.])
    assert through_opening and through_opening[0]["node"] == "buildings__fixture___backing"
    assert beside_opening and beside_opening[0]["node"] == "buildings__fixture___shell"


def test_reassembled_arch_receiver_covers_full_measured_head_from_camera():
    profile = _profile()
    opening = {"rect": [.25, .25, .75, .75],
               "style": {"kind": "window", "head": "arch", "springRatio": .75,
                         "recessM": .12, "frameWidthM": .06,
                         "frameMaterial": "#e4e0d4", "sash": "double-hung",
                         "lites": [1, 1], "meetingRail": .5}}
    profile["openings"] = [opening]
    parts = build_measured_receiver(profile, opening, 0., opening_index=0)
    roles = {part["role"] for part in parts}
    assert {"glazing", "window-frame", "meeting-rail"} <= roles
    glass = next(part["mesh"] for part in parts if part["role"] == "glazing")
    # The pane follows the actual arch and does not fill the unmeasured square corners.
    normal=np.asarray(profile["normal"],float);tangent=profile_tangent(profile)
    pane_vertices=glass.vertices
    front_depth=pane_vertices[:,:2]@normal
    front=pane_vertices[np.isclose(front_depth,front_depth.max(),atol=1e-6)]
    from shapely.geometry import MultiPoint
    front_outline=MultiPoint(np.column_stack((front[:,:2]@tangent,front[:,2]))).convex_hull
    assert front_outline.area < opening_outline_uv(profile, opening, 0.).area
    assert not front_outline.covers(Point(-1.,3.))
    assert front_outline.covers(Point(0.,2.8))

    scene = trimesh.Scene()
    for i,part in enumerate(parts):
        mesh=part["mesh"].copy();mesh.apply_transform(SOURCE_TO_GLB)
        scene.add_geometry(mesh,node_name=f"facades__fixture__aperture_receiver_0_{i}_{part['role']}",
                           geom_name=f"receiver-{i}")
    index=TriangleIndex.from_scene(scene)
    tangent=profile_tangent(profile);normal=np.asarray(profile["normal"],float)
    # Interior samples include the curved head; their source-camera rays must
    # end on the newly assembled glass receiver rather than reaching open sky.
    camera=[0.,5.,2.]
    for u,z in ((0.,2.8),(-.35,2.65),(.35,2.65),(0.,1.7)):
        if not opening_outline_uv(profile,opening,0.).covers(Point(u,z)):
            continue
        xy=tangent*u+normal*profile["planeV"];pt=np.array([xy[0],xy[1],z]);c=np.asarray(camera)
        d=pt-c;d/=np.linalg.norm(d);hits=index.ray_hits(c,c+d*30.)
        assert hits and "aperture_receiver_0_0_glazing" in hits[0]["node"]


def test_reassembled_double_door_uses_two_measured_opaque_leaves():
    profile = _profile()
    opening = {"rect": [.25, .25, .75, .75],
               "style": {"kind": "entry", "recessM": .3,
                         "frameMaterial": "#2a2622",
                         "door": {"kind": "double", "material": "#2a2420"}}}
    parts=build_measured_receiver(profile,opening,0.,opening_index=3)
    leaves=[part for part in parts if part["role"].startswith("door-leaf-")]
    assert len(leaves)==2
    assert all(part["mesh"].is_winding_consistent for part in leaves)
    assert not any(part["role"]=='door-glass' for part in parts)
