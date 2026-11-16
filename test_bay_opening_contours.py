import pytest
import math
import numpy as np
from shapely.geometry import Polygon, box

from pipeline import bays


def _render(shape, outline):
    meshes = []
    opening = {'rect': list(outline.bounds), '_outline': outline,
               'style': {'frameMaterial': 'frame', 'glassMaterial': 'glass'}}
    count = bays.draw(
        lambda x, z, depth: (x, depth, z),
        lambda mesh, key, material: meshes.append((mesh, key, material)),
        'wall', 'detail',
        {'x0': 0., 'x1': 4., 'z0': 0., 'z1': 6., 'p': 1., 'shape': shape, 'cap': 'none'},
        [opening], 'brick', 'trim',
    )
    return count, meshes


def test_arched_aperture_keeps_contour_and_glass_inside_arch():
    arch = Polygon([(1., 1.), (3., 1.), (3., 3.), (2.7, 3.65),
                    (2.35, 3.9), (2., 4.), (1.65, 3.9), (1.3, 3.65), (1., 3.)])
    count, meshes = _render('square', arch)

    assert count == 1
    # The wall face area retains the rectangular bounding-box corners outside
    # the true arch, while the aperture itself removes precisely its polygon.
    wall_area = sum(abs(mesh.volume) / bays.FACE_M for mesh, key, material in meshes
                    if key == 'wall' and material == 'brick'
                    and mesh.bounds[1, 2] - mesh.bounds[0, 2] > .15)
    expected = (4. + 2.) * 6. - arch.area
    assert wall_area == pytest.approx(expected, abs=1e-5)

    glass = [mesh for mesh, _, material in meshes if material == 'glass']
    assert glass
    assert max(mesh.bounds[1, 2] for mesh in glass) < arch.bounds[3]
    assert all(not (mesh.bounds[0, 0] <= 1. and mesh.bounds[1, 2] >= 3.9)
               for mesh in glass)


def test_invalid_nonsimple_opening_outline_is_rejected():
    bow_tie = Polygon([(1., 1.), (3., 3.), (1., 3.), (3., 1.)])
    assert not bow_tie.is_valid
    count, meshes = _render('square', bow_tie)

    assert count == 0
    assert not any(material == 'glass' or material == 'frame'
                   for _, _, material in meshes)


def test_one_outline_spanning_curved_facets_has_no_facet_seam_jambs():
    outline = box(.38, 1., 2.62, 4.)
    count, meshes = _render('curved', outline)

    assert count == 1
    jamb_points = []
    for mesh, _, material in meshes:
        if material != 'frame':
            continue
        z = mesh.triangles[:, :, 2]
        crossing = mesh.triangles[(z.min(axis=1) < 2.5) & (z.max(axis=1) > 2.5)]
        if len(crossing):
            jamb_points.extend(crossing[:, :, 0].mean(axis=1))
    assert jamb_points
    assert all(min(abs(x - .38), abs(x - 2.62)) < .12 for x in jamb_points)

    glass_mesh = [mesh for mesh, _, material in meshes if material == 'glass']
    assert len(glass_mesh) == 1
    # Pane thickness offsets the chord by 2 cm. Project onto the chord normal:
    # all pane vertices share one plane direction across the curved frontage.
    _, shell_faces = bays.faces(0., 4., 1., 'curved')
    def support(x):
        for xa, da, xb, db in shell_faces:
            if min(xa, xb) - 1e-8 <= x <= max(xa, xb) + 1e-8:
                return da + (x - xa) / (xb - xa) * (db - da)
    d0, d1 = support(.38), support(2.62)
    chord_length = math.hypot(2.62 - .38, d1 - d0)
    fx, fd = (2.62 - .38) / chord_length, (d1 - d0) / chord_length
    chord_normal_positions = [float(-fd * v[0] + fx * v[1])
                              for mesh in glass_mesh for v in mesh.vertices]
    assert max(chord_normal_positions) - min(chord_normal_positions) == pytest.approx(.02, abs=1e-5)

    reveals = [mesh for mesh, key, material in meshes
               if key == 'detail' and material == 'frame' and not mesh.is_watertight]
    assert reveals
    assert max(float(mesh.bounds[1, 1]) for mesh in reveals) > max(float(mesh.bounds[1, 1]) for mesh in glass_mesh) + .05


def test_curved_shell_adaptive_facets_stay_within_sag_tolerance_and_cap_matches():
    _, meshes = _render('curved', box(.5, 1., 3.5, 4.))
    shape, points = bays.corners(0., 4., 1., 'curved')
    assert shape == 'curved'
    c, p = 4., 1.
    radius = (c * c / 4. + p * p) / (2. * p)
    center = 2.
    for (x0, d0), (x1, d1) in zip(points, points[1:]):
        xm = (x0 + x1) / 2.
        true_depth = p - radius + math.sqrt(radius * radius - (xm - center) ** 2)
        facet_depth = (d0 + d1) / 2.
        assert true_depth - facet_depth <= .010001
    assert len(points) - 1 <= 96

    shell_bounds = bays.plan(0., 4., 1., 'curved').bounds
    cap_meshes = [mesh for mesh, key, material in meshes
                  if key == 'wall' and material == 'brick'
                  and abs((mesh.bounds[1, 2] - mesh.bounds[0, 2]) - .14) < 1e-6]
    assert cap_meshes
    cap = cap_meshes[0]
    assert cap.bounds[0, 0] == pytest.approx(shell_bounds[0])
    assert cap.bounds[1, 0] == pytest.approx(shell_bounds[2])
    assert cap.bounds[0, 1] == pytest.approx(shell_bounds[1])
    assert cap.bounds[1, 1] == pytest.approx(shell_bounds[3])


def test_three_curved_chord_panes_are_not_occluded_by_facet_cladding():
    emitted = []
    openings = [
        {'rect': [x, 1., x + .65, 4.],
         'style': {'frameMaterial': 'frame', 'glassMaterial': 'glass'}}
        for x in (.3, 1.5, 2.7)
    ]
    count = bays.draw(
        lambda x, z, depth: (x, depth, z),
        lambda mesh, key, material: emitted.append((mesh, key, material)),
        'wall', 'detail',
        {'x0': 0., 'x1': 4., 'z0': 0., 'z1': 6., 'p': 1.,
         'shape': 'curved', 'cap': 'none'},
        openings, 'brick', 'trim',
    )
    panes = [mesh for mesh, _, material in emitted if material == 'glass']
    cladding = [mesh for mesh, key, material in emitted
                if key == 'wall' and material == 'brick']
    assert count == 3
    assert len(panes) == 3

    def ray_hits(mesh, origin, direction):
        triangles = mesh.triangles
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        e1, e2 = b - a, c - a
        h = np.cross(np.broadcast_to(direction, e2.shape), e2)
        det = np.einsum('ij,ij->i', e1, h)
        good = np.abs(det) > 1e-10
        inv = np.zeros_like(det)
        inv[good] = 1. / det[good]
        s = origin[None, :] - a
        u = inv * np.einsum('ij,ij->i', s, h)
        q = np.cross(s, e1)
        v = inv * np.einsum('j,ij->i', direction, q)
        t = inv * np.einsum('ij,ij->i', e2, q)
        return t[good & (u >= 0.) & (v >= 0.) & (u + v <= 1.) & (t > 1e-7)]

    _, shell_faces = bays.faces(0., 4., 1., 'curved')
    for pane_index, pane in enumerate(panes):
        target = pane.vertices.mean(axis=0)
        # Chord normal in world (x, depth), derived from the pane's fitted
        # plane; its slight thickness does not change this test direction.
        xs = [o['rect'][0] + .325 for o in openings]
        cx = min(xs, key=lambda x: abs(x - target[0]))
        def support(x):
            for xa, da, xb, db in shell_faces:
                if min(xa, xb) - 1e-8 <= x <= max(xa, xb) + 1e-8:
                    return da + (x - xa) / (xb - xa) * (db - da)
        xlo, xhi = cx - .325, cx + .325
        d0, d1 = support(xlo), support(xhi)
        length = math.hypot(xhi - xlo, d1 - d0)
        normal = np.array([-(d1 - d0) / length, (xhi - xlo) / length, 0.])
        direction = -normal
        # Exercise both edges and the center of each pane, where a per-facet
        # aperture or an overlaid cladding slab would otherwise hide glass.
        tangent = np.array([normal[1], -normal[0], 0.])
        for fraction in (.1, .3, .5, .7, .9):
            sample = target + tangent * ((fraction - .5) * .5)
            origin = sample + normal * 2.
            wall_hits = [t for wall in cladding for t in ray_hits(wall, origin, direction)]
            glass_hits = ray_hits(pane, origin, direction)
            assert len(glass_hits) > 0, (pane_index, fraction)
            assert not any(t < 2. - .02 for t in wall_hits), (pane_index, fraction)


def test_curved_face_cladding_uv_uses_continuous_metric_arc_length():
    emitted = []
    bay = {'x0': 0., 'x1': 4., 'z0': 0., 'z1': 6., 'p': 1.,
           'shape': 'curved', 'cap': 'none'}
    bays.draw(lambda x, z, depth: (x, depth, z),
              lambda mesh, key, material: emitted.append((mesh, key, material)),
              'wall', 'detail', bay, [], 'brick', 'trim')
    skins = [mesh for mesh, key, material in emitted
             if key == 'wall' and material == 'brick' and 'uvMeters' in mesh.metadata]
    _, shell_faces = bays.faces(0., 4., 1., 'curved')
    assert len(skins) == len(shell_faces)
    expected_arc = 0.
    for mesh, (xa, da, xb, db) in zip(skins, shell_faces):
        uv = np.asarray(mesh.metadata['uvMeters'])
        segment = math.hypot(xb - xa, db - da)
        assert uv.shape == (len(mesh.vertices), 2)
        assert uv[:, 0].min() == pytest.approx(expected_arc, abs=1e-6)
        assert uv[:, 0].max() == pytest.approx(expected_arc + segment, abs=1e-6)
        assert uv[:, 1].min() == pytest.approx(0.)
        assert uv[:, 1].max() == pytest.approx(6.)
        expected_arc += segment


def test_curved_side_panes_recess_in_facade_depth_without_cladding_bleed():
    """The curved pane setback must not move its edges across the shell aperture."""
    emitted = []
    # This rotation and translation mirror the world-to-GLB wall frame. The
    # facade tangent and normal are oblique in global X/Z, as on the source bay.
    tangent = np.array([.986, 0., -.1665])
    tangent /= np.linalg.norm(tangent)
    normal = np.array([.1665, 0., .986])
    normal /= np.linalg.norm(normal)
    origin = np.array([-60., -43.7, -303.])

    def point(x, z, depth):
        return origin + tangent * x + np.array([0., z, 0.]) + normal * depth

    x0, x1, projection = 0., 2.57, .95
    openings = [
        {'rect': [x, 1.2, x + .70, 2.95],
         'style': {'frameMaterial': 'frame', 'glassMaterial': 'glass'}}
        for x in (.12, .935, 1.75)
    ]
    bays.draw(point, lambda mesh, key, mat: emitted.append((mesh, key, mat)),
              'wall', 'detail', {'x0': x0, 'x1': x1, 'z0': 0., 'z1': 6.2,
                                  'p': projection, 'shape': 'curved', 'cap': 'none'},
              openings, 'brick', 'trim')

    panes = [mesh for mesh, _, material in emitted if material == 'glass']
    skins = [mesh for mesh, key, material in emitted
             if key == 'wall' and material == 'brick']
    assert len(panes) == 3
    assert skins

    # Recesses do not enlarge or slide the measured pane in facade x. The
    # center and both angled side panes keep their source extents.
    for pane, opening in zip(panes, openings):
        pane_x = (pane.vertices - origin) @ tangent
        assert pane_x.min() >= opening['rect'][0] - 1e-6
        assert pane_x.max() <= opening['rect'][2] + 1e-6

    def ray_hits(mesh, start, direction):
        tri = mesh.triangles
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        e1, e2 = b - a, c - a
        h = np.cross(np.broadcast_to(direction, e2.shape), e2)
        det = np.einsum('ij,ij->i', e1, h)
        good = np.abs(det) > 1e-10
        inv = np.zeros_like(det)
        inv[good] = 1. / det[good]
        v0 = start[None, :] - a
        u = inv * np.einsum('ij,ij->i', v0, h)
        q = np.cross(v0, e1)
        v = inv * np.einsum('j,ij->i', direction, q)
        dist = inv * np.einsum('ij,ij->i', e2, q)
        mask = good & (u >= 0.) & (v >= 0.) & (u + v <= 1.) & (dist > 1e-7)
        return dist[mask]

    # Cast from a street-side camera toward face-interior samples. Genuine
    # jamb occlusion is allowed at the very edge; these samples are well inside
    # each measured opening and must not be crossed by shell facet strips.
    for pane_index, (pane, opening) in enumerate(zip(panes, openings)):
        front = pane.face_normals @ normal > .5
        samples = pane.triangles_center[front]
        assert len(samples) >= 1
        x_lo, _, x_hi, _ = opening['rect']
        interior = [q for q in samples
                    if x_lo + .12 < float((q - origin) @ tangent) < x_hi - .12]
        assert interior, (pane_index, samples)
        for target in interior:
            camera = target + normal * 6. - np.array([0., 1.5, 0.])
            ray = target - camera
            distance = float(np.linalg.norm(ray))
            direction = ray / distance
            cladding_hits = [d for skin in skins for d in ray_hits(skin, camera, direction)]
            assert not any(d < distance - .02 for d in cladding_hits), (
                pane_index, float((target - origin) @ tangent), cladding_hits)
