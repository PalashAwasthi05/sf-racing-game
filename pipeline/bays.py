"""Bay windows as projecting volumes. A photographed bay (a reading's `bays`: its extent along the wall and up it, and
how far it stands out) was drawn as its windows pushed out one by one, each with its own cheeks: a stack of window
boxes. Here a bay is one volume standing out from the wall, clad like the wall it projects from, its windows set in
its faces:
  angled  a front parallel to the wall and two sides at 45 degrees (the commonest bay on a city street);
  square  a front and two sides square to the wall, when the bay is too narrow for 45-degree sides;
  curved  a shallow arc of glazed facets (a round bay), when the reading names it.
A reading may name the shape, the angle of an angled bay's sides, a trim course at each storey and corbels under a
bay that starts above the ground.
A rectified photograph shows an angled bay's side windows foreshortened inside the side's extent along the wall, so a
window is set on the face its extent falls on (split across two faces if it spans both). A square bay's sides face
along the wall and show no window in a square-on photograph. The bay is capped with a cornice in its windows' trim and
closed underneath when it starts above the ground. Location-free: the shape follows the measured extent alone."""
import math
from math import ceil
import numpy as np
from shapely.geometry import Polygon, LineString, box
from shapely.affinity import affine_transform
from shapely.ops import unary_union

FACE_M, GLASS_IN_M, FRAME_M = .12, .08, .06


def _apply_frame(mesh, origin, x_axis, y_axis, z_axis):
    """Map local mesh axes into the facade frame without rewriting vertices.

    ``Trimesh.apply_transform`` accounts for reflected frames by correcting
    triangle winding. Direct vertex assignment followed by ``fix_normals`` can
    reorient point-touching/non-manifold bay pieces inconsistently.
    """
    transform = np.eye(4, dtype=float)
    transform[:3, 0] = np.asarray(x_axis, dtype=float)
    transform[:3, 1] = np.asarray(y_axis, dtype=float)
    transform[:3, 2] = np.asarray(z_axis, dtype=float)
    transform[:3, 3] = np.asarray(origin, dtype=float)
    mesh.apply_transform(transform)
    return mesh


def _reveal_solid(shell_a, shell_b, chord_a, chord_b, put, key, material):
    """A closed, thin ruled quad joining one contour edge to a chord plane."""
    import trimesh
    front = np.asarray([shell_a, shell_b, chord_b, chord_a], dtype=float)
    normal = np.cross(front[1] - front[0], front[2] - front[0])
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        return
    normal /= length
    half = .004
    vertices = np.vstack((front + normal * half, front - normal * half))
    # Opposing thin faces render from either side of the reveal without
    # adding a false jamb where adjacent facet strips meet.
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    put(mesh, key, material)


def corners(x0, x1, p, shape=None, angle=45.):
    """(shape, corner points) of a bay over [x0, x1] standing p out, left to right in (along the wall, out from it)."""
    run = p / math.tan(math.radians(float(np.clip(angle, 20., 80.))))
    if shape == 'curved':
        c = x1 - x0; R = (c * c / 4 + p * p) / (2 * p); xm = (x0 + x1) / 2; a = math.asin(min(1., c / 2 / R))
        half_step = math.acos(max(-1., min(1., 1. - .01 / R)))
        n = min(96, max(6, int(ceil((2. * a) / max(half_step, 1e-6)))))
        return 'curved', [(xm + R * math.sin(t), p - R + R * math.cos(t))
                          for t in np.linspace(-a, a, n + 1)]
    if shape == 'square' or (shape is None and x1 - x0 < 2 * run + .8) or x1 - x0 < 2 * run + .4:
        return 'square', [(x0, 0.), (x0, p), (x1, p), (x1, 0.)]
    return 'angled', [(x0, 0.), (x0 + run, p), (x1 - run, p), (x1, 0.)]


def faces(x0, x1, p, shape=None, angle=45.):
    """(shape, faces) of a bay over [x0, x1] standing p out: each face (xa, da, xb, db) in (along the wall, out from it),
    running left to right around the bay."""
    shape, cs = corners(x0, x1, p, shape, angle)
    return shape, [(a[0], a[1], b[0], b[1]) for a, b in zip(cs, cs[1:])]


def plan(x0, x1, p, shape, angle=45.):
    """The bay's outline in plan (along the wall, out from it), closed along the wall."""
    return Polygon(corners(x0, x1, p, shape, angle)[1])


def draw(point, put, layer, detail, bay, openings, cladding, trim, claddingZones=None):
    """Draw one bay. `point(x, z, depth)` maps the wall frame to the world (x along the wall, z up, depth out from the
    wall's face); `put(mesh, key, material)` adds a mesh. `bay` = {'x0','x1','z0','z1','p'} in metres of the wall frame;
    `openings` are the grammar's openings inside it (rect in metres, style). Returns the number of windows set."""
    import trimesh
    x0, x1, z0, z1, p = bay['x0'], bay['x1'], bay['z0'], bay['z1'], bay['p']
    angle = bay.get('sideAngleDeg') or 45.
    shape, fs = faces(x0, x1, p, bay.get('shape'), angle)
    shell_arc = [0.]
    for xa, da, xb, db in fs:
        shell_arc.append(shell_arc[-1] + math.hypot(xb - xa, db - da))

    def slab(poly, frame_of, o0, o1, key, material, facade_depth=False):
        """A polygon in a face's own (s, z) frame, from o0 to o1 inward of the face (negative: standing out of it)."""
        for q in ([poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))):
            if q.is_empty or q.area < 1e-4: continue
            mesh = trimesh.creation.extrude_polygon(q, height=o1 - o0, engine='earcut')
            uv_meters = None
            if (shape == 'curved' and key == layer and abs(o0) < 1e-8
                    and abs(o1 - FACE_M) < 1e-8):
                # Share one metric texture coordinate system across all the
                # tessellated curved facets. The automatic world-axis
                # projection changes tangent at every facet and creates a
                # visible texture phase jump at each seam.
                s_values = np.asarray(mesh.vertices[:, 0], dtype=float)
                uv_meters = np.column_stack((frame_of.arc_start + s_values,
                                             np.asarray(mesh.vertices[:, 1], dtype=float)))
            origin = np.asarray(frame_of(0., 0., o0), dtype=float)
            s_axis = np.asarray(frame_of(1., 0., o0), dtype=float) - origin
            z_axis = np.asarray(frame_of(0., 1., o0), dtype=float) - origin
            if facade_depth:
                # A curved shell is partitioned into many adjacent facets.
                # Extruding each facet along its own normal shifts its back
                # edge laterally; the resulting end caps can span an opening
                # that was cut in the measured facade x/z frame. Keep the
                # observed front surface and thickness, but extrude the skin
                # in facade depth so its x/z aperture remains unchanged.
                o_axis = np.asarray(point(0., 0., 1.), dtype=float) - np.asarray(point(0., 0., 0.), dtype=float)
            else:
                o_axis = np.asarray(frame_of(0., 0., o0 + 1.), dtype=float) - origin
            _apply_frame(mesh, origin, s_axis, z_axis, o_axis)
            if uv_meters is not None:
                mesh.metadata['uvMeters'] = uv_meters
            put(mesh, key, material)

    # Keep each opening in one continuous wall-frame contour. Facets only
    # clip that contour for drawing; they never rebuild a head or add a jamb.
    contours = []
    for op in openings:
        try:
            outline = op.get('_outline')
            if outline is None:
                rx1, rz1, rx2, rz2 = map(float, op['rect'])
                outline = box(rx1, rz1, rx2, rz2)
            if (outline.geom_type != 'Polygon' or outline.is_empty or
                    not outline.is_valid or not outline.exterior.is_simple or
                    outline.area < .09):
                continue
            ox1, oz1, ox2, oz2 = map(float, outline.bounds)
            if ox2 - ox1 < .3 or oz2 - oz1 < .3:
                continue
            contours.append((op, outline, (ox1, oz1, ox2, oz2)))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    # One count per source opening, even when its contour crosses many facets.
    windows = sum(1 for _, _, (a, b, c, d) in contours
                  if c > x0 + 1e-8 and a < x1 - 1e-8 and d > z0 and b < z1)

    for face_index, (xa, da, xb, db) in enumerate(fs):
        L = math.hypot(xb - xa, db - da)
        if L < .05: continue
        fx, fd = (xb - xa) / L, (db - da) / L; nx_, nd_ = -fd, fx           # along the face, and out of it
        def frame_of(s, z, o, xa=xa, da=da, fx=fx, fd=fd, nx_=nx_, nd_=nd_):
            return point(xa + fx * s - nx_ * o, z, da + fd * s - nd_ * o)
        frame_of.arc_start = shell_arc[face_index]
        holes = []
        lo, hi = sorted((xa, xb))
        if abs(fx) >= 1e-6:
            clip = box(lo, z0 + .05, hi, z1 - .05)
            localize = lambda geom: affine_transform(
                geom.intersection(clip), [1. / fx, 0., 0., 1., -xa / fx, 0.])
            for op, outline, (ox1, oz1, ox2, oz2) in contours:
                a, b = max(ox1, lo), min(ox2, hi)
                if b - a <= 1e-8:
                    continue
                aperture_global = outline.intersection(clip)
                if aperture_global.is_empty or aperture_global.area < .01:
                    continue
                aperture = localize(aperture_global)
                st = op.get('style', {})
                frame = st.get('frameMaterial', trim)
                if shape == 'curved':
                    # The curved shell is cut along its facets, but the
                    # complete frame and pane are emitted once on a chord.
                    holes.append((aperture, st, []))
                    continue
                inner_global = outline.buffer(-FRAME_M, join_style=2)
                if inner_global.is_empty:
                    inner_global = outline
                elements = [
                    (localize(inner_global), GLASS_IN_M, GLASS_IN_M + .02,
                     st.get('glassMaterial', 'glass')),
                    (localize(outline.difference(inner_global)), 0., GLASS_IN_M + .02, frame),
                ]

                def bar(geom, near=.04, far=GLASS_IN_M):
                    geom = geom.intersection(outline)
                    if not geom.is_empty:
                        elements.append((localize(geom), near, far, frame))

                width, height = ox2 - ox1, oz2 - oz1
                for fraction in st.get('mullionFractions', []):
                    x = ox1 + width * float(fraction)
                    bar(box(x - .02, oz1, x + .02, oz2))
                for fraction in st.get('transomFractions', []):
                    z = oz1 + height * float(fraction)
                    bar(box(ox1, z - .02, ox2, z + .02))
                from .window_panes import inner_bars
                for strip,near in inner_bars((ox1,oz1,ox2,oz2),st):
                    bar(strip,near,GLASS_IN_M)
                sill = box(ox1 - .03, oz1 - .06, ox2 + .03, oz1)
                if not sill.intersection(clip).is_empty:
                    elements.append((localize(sill), -.05, .1, frame))
                holes.append((aperture, st, elements))
        wall = box(0., z0, L, z1).difference(unary_union([h for h, _, _ in holes])) if holes else box(0., z0, L, z1)
        # Zones are measured in the whole wall's (x, z) frame. Convert each
        # zone's along-wall interval into this face's local s coordinate. Each
        # face segment has a linear x(s), including every curved facet. A
        # constant-x return face belongs to a zone that includes its boundary.
        face_zones = []
        for zone in claddingZones or []:
            rect = zone.get('rect')
            material = zone.get('cladding')
            if not rect or len(rect) != 4 or not material:
                continue
            zx0, zz0, zx1, zz1 = map(float, rect)
            if zx1 <= zx0 or zz1 <= zz0:
                continue
            if abs(fx) < 1e-6:
                if min(zx0, zx1) - 1e-7 <= xa <= max(zx0, zx1) + 1e-7:
                    s0, s1 = 0., L
                else:
                    continue
            else:
                a, b = sorted(((zx0 - xa) / fx, (zx1 - xa) / fx))
                s0, s1 = max(0., a), min(L, b)
                if s1 - s0 <= 1e-8:
                    continue
            zone_shape = box(s0, max(z0, zz0), s1, min(z1, zz1))
            if not zone_shape.is_empty and zone_shape.area > 1e-8:
                face_zones.append((zone_shape, material))

        if face_zones:
            later = Polygon()
            for zone_shape, material in reversed(face_zones):
                assigned = wall.intersection(zone_shape).difference(later)
                slab(assigned, frame_of, 0., FACE_M, layer, material,
                     facade_depth=(shape == 'curved'))
                later = unary_union([later, zone_shape])
            slab(wall.difference(later), frame_of, 0., FACE_M, layer, cladding,
                 facade_depth=(shape == 'curved'))
        else:
            slab(wall, frame_of, 0., FACE_M, layer, cladding,
                 facade_depth=(shape == 'curved'))
        for h, st, custom_elements in holes:
            for geom, o0, o1, material in custom_elements:
                if not geom.is_empty:
                    slab(geom, frame_of, o0, o1, detail, material)

    # An observed bay may rise above a shorter LiDAR roof return on its host
    # wall. Close only that measured extension at the original wall plane so
    # its upper panes do not become open-backed projections. The opaque backing
    # stays behind the glass: it supplies interior occlusion without covering
    # the street-facing pane.
    host_top = float(bay.get('hostTopM', z1))
    back_lo = max(z0, host_top) + 1e-4
    if back_lo < z1 - 1e-4:
        back_region = box(x0, back_lo, x1, z1)
        mesh = trimesh.creation.extrude_polygon(back_region, height=FACE_M, engine='earcut')
        origin = np.asarray(point(0., 0., 0.), dtype=float)
        x_axis = np.asarray(point(1., 0., 0.), dtype=float) - origin
        z_axis = np.asarray(point(0., 1., 0.), dtype=float) - origin
        inward = np.asarray(point(0., 0., -1.), dtype=float) - origin
        _apply_frame(mesh, origin, x_axis, z_axis, inward)
        put(mesh, layer, bay.get('hostMaterial', cladding))

    if shape == 'curved':
        # A continuous chord window sits between the opening's shell support
        # depths. The reveal strips bridge its perimeter back to each curved
        # facet, subdivided at shell vertices so adjacent strips meet exactly.
        shell_x = sorted(set(float(v) for face in fs for v in (face[0], face[2])))

        def shell_depth(x):
            for xa, da, xb, db in fs:
                if min(xa, xb) - 1e-8 <= x <= max(xa, xb) + 1e-8 and abs(xb - xa) > 1e-9:
                    t = (x - xa) / (xb - xa)
                    return da + t * (db - da)
            return 0.

        for op, opening, (ox1, oz1, ox2, oz2) in contours:
            cx0, cx1 = max(x0, ox1), min(x1, ox2)
            if cx1 - cx0 < .3:
                continue
            window = opening.intersection(box(cx0, z0 + .05, cx1, z1 - .05))
            if window.geom_type != 'Polygon' or window.is_empty or window.area < .01:
                continue
            d0, d1 = shell_depth(cx0), shell_depth(cx1)
            dx = cx1 - cx0
            chord_length = math.hypot(dx, d1 - d0)
            if chord_length < .05:
                continue
            fx, fd = dx / chord_length, (d1 - d0) / chord_length
            nx_, nd_ = -fd, fx

            def chord_frame(s, z, o):
                # Keep the measured x/z aperture fixed as the pane recedes.
                # Offsetting along the chord normal moves an angled side pane
                # sideways by fd*o; its edge then crosses the curved shell's
                # per-facet jamb and opaque cladding bleeds into the glass.
                # Recess along facade depth instead. The pane remains parallel
                # to its chord, with the same normal distance when the chord
                # has a meaningful depth component. Near a depth-parallel
                # chord, use the facade-depth convention directly rather than
                # amplifying a tiny normal-depth component without bound.
                normal_depth = fx
                depth_recess = o / normal_depth if normal_depth >= .25 else o
                return point(cx0 + fx * s, z, d0 + fd * s - depth_recess)

            localize_chord = lambda geom: affine_transform(
                geom, [1. / fx, 0., 0., 1., -cx0 / fx, 0.])
            st = op.get('style', {})
            frame = st.get('frameMaterial', trim)
            inner = window.buffer(-FRAME_M, join_style=2)
            if inner.is_empty:
                inner = window
            pane_elements = [
                (inner, GLASS_IN_M, GLASS_IN_M + .02, st.get('glassMaterial', 'glass')),
                (window.difference(inner), 0., GLASS_IN_M + .02, frame),
            ]

            def chord_bar(geom, near=.04, far=GLASS_IN_M):
                geom = geom.intersection(window)
                if not geom.is_empty:
                    pane_elements.append((geom, near, far, frame))

            wx0, wz0, wx1, wz1 = window.bounds
            ww, wh = wx1 - wx0, wz1 - wz0
            for fraction in st.get('mullionFractions', []):
                x = wx0 + ww * float(fraction)
                chord_bar(box(x - .02, wz0, x + .02, wz1))
            for fraction in st.get('transomFractions', []):
                z = wz0 + wh * float(fraction)
                chord_bar(box(wx0, z - .02, wx1, z + .02))
            from .window_panes import inner_bars
            for strip,near in inner_bars((wx0,wz0,wx1,wz1),st):
                chord_bar(strip,near,GLASS_IN_M)
            sill = box(wx0 - .03, wz0 - .06, wx1 + .03, wz0)
            pane_elements.append((sill, -.05, .1, frame))
            for geom, o0, o1, material in pane_elements:
                slab(localize_chord(geom), chord_frame, o0, o1, detail, material)

            rings = [window.exterior, *window.interiors]
            for ring in rings:
                perimeter = LineString(ring.coords)
                clipped_perimeter = perimeter.intersection(box(cx0, z0 + .05, cx1, z1 - .05))
                lines = [clipped_perimeter] if clipped_perimeter.geom_type == 'LineString' else list(getattr(clipped_perimeter, 'geoms', []))
                for line in lines:
                    if line.geom_type != 'LineString':
                        continue
                    coords = list(line.coords)
                    for (ax, az), (bx, bz) in zip(coords, coords[1:]):
                        cuts = [0., 1.]
                        if abs(bx - ax) > 1e-9:
                            cuts.extend((x - ax) / (bx - ax) for x in shell_x
                                        if min(ax, bx) + 1e-8 < x < max(ax, bx) - 1e-8)
                        cuts = sorted(set(cuts))
                        for ta, tb in zip(cuts, cuts[1:]):
                            xa, za = ax + (bx - ax) * ta, az + (bz - az) * ta
                            xb, zb = ax + (bx - ax) * tb, az + (bz - az) * tb
                            dsa, dsb = shell_depth(xa), shell_depth(xb)
                            dca = d0 + (xa - cx0) * (d1 - d0) / dx
                            dcb = d0 + (xb - cx0) * (d1 - d0) / dx
                            if max(abs(dsa - dca), abs(dsb - dcb)) < 1e-5:
                                continue
                            _reveal_solid(
                                point(xa, za, dsa), point(xb, zb, dsb),
                                point(xb, zb, dcb), point(xa, za, dca),
                                put, detail, frame,
                            )
    outline = plan(x0, x1, p, shape, angle)

    def lid(poly, za, zb, key, material):
        """A plan polygon (along, out) from height za to zb."""
        poly = poly.intersection(box(x0 - 2., 0., x1 + 2., p + 2.))      # never behind the wall's face
        for q in ([poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))):
            if q.is_empty or q.area < 1e-4: continue
            mesh = trimesh.creation.extrude_polygon(q, height=zb - za, engine='earcut')
            origin = np.asarray(point(0., za, 0.), dtype=float)
            x_axis = np.asarray(point(1., za, 0.), dtype=float) - origin
            d_axis = np.asarray(point(0., za, 1.), dtype=float) - origin
            z_axis = np.asarray(point(0., za + 1., 0.), dtype=float) - origin
            _apply_frame(mesh, origin, x_axis, d_axis, z_axis)
            put(mesh, key, material)

    lid(outline, z1, z1 + .14, layer, cladding)
    if bay.get('cap', 'cornice') == 'cornice':
        lid(outline.buffer(.06, join_style=2), z1 + .14, z1 + .22, detail, trim)                  # cornice: a frieze,
        lid(outline.buffer(.16, join_style=2), z1 + .22, z1 + .34, detail, trim)                  # and its cap
    if z0 > .5:
        lid(outline, z0 - .14, z0, layer, cladding)                                               # underside
        lid(outline.buffer(.05, join_style=2), z0 - .22, z0 - .1, detail, trim)                   # its moulding
    for z in bay.get('storeys', []):
        # a trim course round the bay at each floor line inside it
        if z0 + .4 < z < z1 - .4: lid(outline.buffer(.035, join_style=2), z - .08, z + .08, detail, trim)
    if bay.get('corbel') and z0 > 1.5:
        # brackets under a bay that starts above the ground, at its front corners
        _, cs = corners(x0, x1, p, shape, angle)
        for xa, da in cs[1:-1] if shape != 'curved' else cs[1:-1:2]:
            for k, (h0, h1, dd) in enumerate(((z0 - .7, z0 - .45, .25), (z0 - .45, z0 - .22, .5), (z0 - .22, z0 - .1, .8))):
                lid(box(xa - .08, 0., xa + .08, max(.05, dd * da)), h0, h1, detail, trim)   # deeper toward the top
    return windows
